"""Read through every copy: a loop that only moves values joins no two loops.

A loop that computes nothing — one read and one store of it, in another layout or at another dtype: the chunks of a
sequence, a transpose, a broadcast, a cast to a wider float — holds no work a kernel could share. Every loop that
reads one takes the read itself: the buffer behind the copy, at the cell the copy reads there. The copy is then read
by no loop, and the kernel whose only work it was is gone.

So a copy joins nothing. Loops share a region through a value that is computed (``_region.regions``): one kernel
computes it once for all of them. Two loops that read one copy share what two loops that read one input share, and
are two kernels as those are, whether or not the program wrote a transpose between the input and them. A kernel that
carries a state, which no region holds, reads its operands where they are stored by the same rule.

A copy the program itself holds, a graph output, is a buffer whoever reads it: it is stored once and its readers
read it there, like any stored value. A copy that rounds is arithmetic and is not read through, nor is a copy of a
buffer another kernel stores over, nor one of a packed buffer.

It runs after ``005_roll_recurrence``, which compares the steps of a recurrence as the tracer left them: a step's
slice of a copy is a literal coordinate that advances with the step, where the read behind it is an offset that the
first step does not have.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.expression import Literal, Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Body, Load, Loop, Statement, Write
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.fusion._region import plain_load, wrap_multi_output_fragment

PATTERN = [Pattern("reader", LoopOp)]


def copied(graph: Graph, name: str) -> tuple[tuple, Load] | None:
    """``(the cell the copy stores, its one read)`` where ``name`` is the buffer of a loop that computes nothing —
    loops around one read of a buffer as it is stored and one store of the value read — and holds every value it
    read: the same dtype, or f32 over a 16-bit float. A copy that rounds is arithmetic."""
    producer = graph.producer(name)
    if producer is None or not isinstance(producer.op, LoopOp) or len(producer.outputs) != 1:
        return None
    body = producer.op.body
    if len(body.loads) != 1 or len(body.writes) != 1 or any(not isinstance(stmt, (Loop, Load, Write)) for stmt in body.iter()):
        return None
    (load,), (write,) = body.loads, body.writes
    if not plain_load(load) or tuple(write.values) != tuple(load.names):
        return None
    source, tensor = graph.buffer(load.input), producer.outputs[0]
    if source is None or tensor.alias is not None or source.dtype.logical_elems != 1 or tensor.dtype.logical_elems != 1:
        return None
    if any(stored.alias == source.name for user in graph.buffer_users(source.name) for stored in graph.nodes[user].outputs):
        return None  # a kernel stores over the source: the copy is what holds the values from before that store
    exact = source.dtype == tensor.dtype or (tensor.dtype == F32 and source.dtype.name in ("f16", "bf16"))
    # The cell is one a substitution of its coordinates inverts: every entry a coordinate or a literal, none twice.
    names = [entry.name for entry in write.index if isinstance(entry, Var)]
    plain = all(isinstance(entry, (Var, Literal)) for entry in write.index) and len(set(names)) == len(names)
    return (tuple(write.index), load) if exact and plain else None


def read_in_place(graph: Graph, body: Body) -> Body:
    """``body`` with each read of a copy taken through it, and through the copy behind that one."""

    def through(stmt: Statement) -> Statement:
        while isinstance(stmt, Load) and plain_load(stmt) and (copy := copied(graph, stmt.input)) is not None:
            if stmt.input in graph.outputs:
                break  # the program holds this copy: it is stored whoever reads it, and a reader finds its values there
            cells, read = copy
            if len(cells) != len(stmt.index) or any(isinstance(c, Literal) and c != at for c, at in zip(cells, stmt.index, strict=True)):
                break
            sigma = Sigma({cell.name: at for cell, at in zip(cells, stmt.index, strict=True) if isinstance(cell, Var)})
            stmt = stmt.replace(input=read.input, index=tuple(sigma.apply(entry) for entry in read.index))
        return stmt

    return body.map(through)


def rewrite(match: Match, reader: Node) -> Graph:
    graph = match.graph
    body = read_in_place(graph, reader.op.body)
    if body == reader.op.body:
        raise RuleSkipped("the loop reads no copy")
    fragment, match.output = wrap_multi_output_fragment(graph, replace(reader.op, body=body, inputs={}), tuple(reader.buffer_names()))
    return fragment
