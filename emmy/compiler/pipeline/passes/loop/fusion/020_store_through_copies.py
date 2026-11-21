"""Store a buffer where its one reader would only have copied it.

A kernel no region holds — one that carries a state, an ordered prefix — stores what it computes in a buffer of its
own, and the loops that read that buffer are another kernel. Where that kernel computes nothing, it only moves every
cell to where the program wants it: the steps of a walk stacked into one tensor, its tokens put back in sequence
order, a transpose. The producer then stores each value at the cell the reader would have moved it to, and the reader
is gone with the buffer between them.

It runs after ``010_merge_loop_ops``, on the reader's whole region: a region that computes anything keeps its
kernel. The reader of a walk's steps reads each step back by its position and chooses among them by the stacked
coordinate; that choice is one read at that coordinate (:func:`moved`).
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.expression import BinaryExpression, Literal, Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Body, Load, Loop, Select, Statement, Write
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.fusion._region import wrap_multi_output_fragment

PATTERN = [Pattern("reader", LoopOp)]


def _chosen(body: Body, select: Select) -> tuple | None:
    """The one cell ``select`` reads: its branches are reads of one buffer at cells that differ in one position, a
    literal there, and each is chosen where a loop's coordinate equals that literal — the read at that coordinate."""
    reads = {load.names[0]: load.index for load in body.loads}
    cells = [reads.get(branch.value) for branch in select.branches]
    if any(cell is None for cell in cells) or len({len(cell) for cell in cells}) != 1 or len(cells) < 2:
        return None
    differing = [at for at, entries in enumerate(zip(*cells, strict=True)) if len(set(entries)) > 1]
    if len(differing) != 1 or not all(isinstance(cell[differing[0]], Literal) for cell in cells):
        return None
    (at,) = differing
    *tests, otherwise = (branch.select for branch in select.branches)
    if otherwise != Literal(True, "bool") or not all(isinstance(test, BinaryExpression) and test.op == "==" for test in tests):
        return None
    sides = [{type(side): side for side in (test.left, test.right)} for test in tests]
    if any(set(side) != {Literal, Var} for side in sides) or len({side[Var] for side in sides}) != 1:
        return None
    if any(side[Literal] != cell[at] for side, cell in zip(sides, cells, strict=False)):
        return None
    coordinate = sides[0][Var]
    extents = {loop.axis.name: loop.axis.extent for loop in body.iter_of_type(Loop)}
    positions = sorted(cell[at].value for cell in cells)
    if coordinate.name not in extents or extents[coordinate.name].as_static() != len(cells) or positions != list(range(len(cells))):
        return None
    return (*cells[0][:at], coordinate, *cells[0][at + 1 :])


def moved(op: LoopOp) -> tuple[str, tuple, Write] | None:
    """``(the buffer read, the cell read, the store)`` of a loop that computes nothing and reads every cell of its one
    input once."""
    body = op.body
    loads, selects, writes = body.loads, body.iter_of_type(Select), body.writes
    if len(writes) != 1 or len(selects) > 1 or any(not isinstance(stmt, (Loop, Load, Select, Write)) for stmt in body.iter()):
        return None
    if len({load.input for load in loads}) != 1 or any(load.dtype is not None or load.carried or len(load.names) != 1 for load in loads):
        return None
    (write,) = writes
    if selects:
        cell = _chosen(body, selects[0]) if tuple(write.values) == (selects[0].name,) else None
    else:
        cell = tuple(loads[0].index) if len(loads) == 1 and tuple(write.values) == tuple(loads[0].names) else None
    if cell is None:
        return None
    coordinates = [entry.name for entry in cell if isinstance(entry, Var)]
    axes = {loop.axis.name for loop in body.iter_of_type(Loop)}
    if (
        any(not isinstance(entry, (Var, Literal)) for entry in cell)
        or len(set(coordinates)) != len(coordinates)
        or set(coordinates) != axes
    ):
        return None
    return loads[0].input, cell, write


def rewrite(match: Match, reader: Node) -> Graph:
    graph = match.graph
    found = moved(reader.op) if len(reader.outputs) == 1 else None
    if found is None:
        raise RuleSkipped("the loop computes something, or reads its input more than once")
    source, cell, write = found
    stored, kept, producer = graph.buffer(source), reader.outputs[0], graph.producer(source)
    if producer is None or not isinstance(producer.op, LoopOp) or source in graph.outputs or graph.buffer_users(source) != {reader.id}:
        raise RuleSkipped("the buffer is read elsewhere, or no loop stores it")
    if source in producer.op.inputs:
        raise RuleSkipped("the producer reads the buffer back")
    if stored.dtype != kept.dtype or stored.dtype.logical_elems != 1 or stored.alias is not None or kept.alias is not None:
        raise RuleSkipped("the reader converts what it reads, or one of the two is stored over another buffer")
    extents = {loop.axis.name: loop.axis.extent for loop in reader.op.body.iter_of_type(Loop)}
    whole = all(
        extents[entry.name] == size if isinstance(entry, Var) else (entry.value == 0 and size.as_static() == 1)
        for entry, size in zip(cell, stored.shape, strict=True)
    )
    if not whole:
        raise RuleSkipped("the reader reads a part of the buffer")

    def placed(stmt: Statement) -> Statement:
        if not isinstance(stmt, Write) or stmt.output != source:
            return stmt
        sigma = Sigma({entry.name: at for entry, at in zip(cell, stmt.index, strict=True) if isinstance(entry, Var)})
        return stmt.replace(output=kept.name, index=tuple(sigma.apply(entry) for entry in write.index))

    body = producer.op.body.map(placed)
    live = tuple(kept.name if name == source else name for name in producer.buffer_names())
    fragment, match.output = wrap_multi_output_fragment(graph, replace(producer.op, body=body, inputs={}, outputs={}), live)
    match.consumed = {producer.id, reader.id}
    return fragment
