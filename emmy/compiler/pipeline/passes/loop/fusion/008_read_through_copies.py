"""Read through a copy that another kernel would have to store for this one.

A loop that computes nothing — one read and one store of it, in another layout or at another dtype: the chunks of a
sequence, a transpose, a broadcast, a cast to a wider float — is a buffer only where something has to read it as one.
Inside its own region it never is: the splice inlines the read (``_region.regions``). Across the region's boundary it
was. A kernel that carries a state is in no region, so every operand it reads through a copy was a buffer some kernel
filled for it alone; and a copy two regions read was filled by one of them for the other. Such a reader now takes the
read itself: the buffer behind the copy, at the cell the copy reads there.

The regions stay what they were. A reader in the copy's own region is left to the splice, so the copy still joins
the loops it joined into one kernel; what goes is the store of the copy, and the kernel whose only work it was.

It runs after ``005_roll_recurrence``, which compares the steps of a recurrence as the tracer left them: a step's
slice of a copy is a literal coordinate that advances with the step, where the read behind it is an offset that the
first step does not have. The copy stays for whoever still reads it as a buffer: a graph output, a packed buffer's
reader, a reader of a buffer another kernel stores over.
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
from emmy.compiler.pipeline.passes.loop.fusion._region import joins_a_region, regions, wrap_multi_output_fragment

PATTERN = [Pattern("reader", LoopOp)]


def copied(graph: Graph, name: str) -> tuple[tuple, Load] | None:
    """``(the cell the copy stores, its one read)`` where ``name`` is the buffer of a loop that computes nothing and
    holds every value it read: the same dtype, or f32 over a 16-bit float. A copy that rounds is arithmetic."""
    producer = graph.producer(name)
    if producer is None or not isinstance(producer.op, LoopOp) or len(producer.outputs) != 1:
        return None
    body = producer.op.body
    if len(body.loads) != 1 or len(body.writes) != 1 or any(not isinstance(stmt, (Loop, Load, Write)) for stmt in body.iter()):
        return None
    (load,), (write,) = body.loads, body.writes
    source, tensor = graph.buffer(load.input), producer.outputs[0]
    if source is None or tensor.alias is not None or source.dtype.logical_elems != 1 or tensor.dtype.logical_elems != 1:
        return None
    if any(stored.alias == source.name for user in graph.buffer_users(source.name) for stored in graph.nodes[user].outputs):
        return None  # a kernel stores over the source: the copy is what holds the values from before that store
    cells = [entry.name for entry in write.index if isinstance(entry, Var)]
    plain = all(isinstance(entry, (Var, Literal)) for entry in write.index) and len(set(cells)) == len(cells)
    exact = source.dtype == tensor.dtype or (tensor.dtype == F32 and source.dtype.name in ("f16", "bf16"))
    whole = plain and exact and tuple(write.values) == tuple(load.names) and load.dtype is None and not load.carried
    return (tuple(write.index), load) if whole else None


def read_in_place(graph: Graph, body: Body, region: frozenset[str]) -> Body:
    """``body`` with each read of a copy that ``region`` does not hold taken through it, and through the copy behind
    that one."""

    def through(stmt: Statement) -> Statement:
        while isinstance(stmt, Load) and stmt.dtype is None and not stmt.carried and (copy := copied(graph, stmt.input)) is not None:
            if graph.producer(stmt.input).id in region:
                break
            cells, read = copy
            if len(cells) != len(stmt.index) or any(isinstance(c, Literal) and c != at for c, at in zip(cells, stmt.index, strict=True)):
                break
            sigma = Sigma({cell.name: at for cell, at in zip(cells, stmt.index, strict=True) if isinstance(cell, Var)})
            stmt = stmt.replace(input=read.input, index=tuple(sigma.apply(entry) for entry in read.index))
        return stmt

    return body.map(through)


def rewrite(match: Match, reader: Node) -> Graph:
    graph = match.graph
    region: frozenset[str] = frozenset()
    if joins_a_region(reader.op):
        # A copy this loop alone reads is its own region's: a loop whose every reader sits in one region joins it.
        # The regions are an answer about the whole graph, asked only for a loop that reads a copy with another reader.
        region = frozenset(graph.producer(name).id for name in reader.inputs if graph.buffer_users(name) == {reader.id})
        if read_in_place(graph, reader.op.body, region) != reader.op.body:
            region = regions(graph).get(reader.id, frozenset())
    body = read_in_place(graph, reader.op.body, region)
    if body == reader.op.body:
        raise RuleSkipped("the loop reads no copy of another region")
    fragment, match.output = wrap_multi_output_fragment(graph, replace(reader.op, body=body, inputs={}), tuple(reader.buffer_names()))
    return fragment
