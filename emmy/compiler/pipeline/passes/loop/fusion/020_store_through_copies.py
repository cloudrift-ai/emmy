"""Store a buffer where its one reader would only have copied it.

It runs after ``010_merge_loop_ops`` and ``015_merge_into_step``, so every loop it sees is a kernel: a region
already merged, or a kernel no region holds — one that carries a state, an ordered prefix — whose readers are another
kernel where they did not join its step. Where such a reader computes nothing, it only moves every cell to where the
program wants it: the state a walk ends with in another layout, the steps of a scan put in sequence order, a
transpose. The producer then stores each value at the cell the
reader would have moved it to, and the reader is gone with the buffer between them. A reader may round what it moves
to the dtype the program holds it at, as the cast of a walk's f32 outputs to a 16-bit model's dtype does: the
producer then rounds the value it stores. A reader that computes anything else keeps its kernel.

The reader of a walk's steps reads each step back by its position and chooses among them by the stacked coordinate,
or by the digit of a joined coordinate that the step is; that choice is one read at that coordinate (:func:`moved`).
That the reader moves every cell once, and where, is the proof the splice holds a layout copy of its own region to
(``layout_strides``), and the producer's stores are moved the way the splice moves a region's (``stored_through``).
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.dtype import DataType
from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.expression import Var
from emmy.compiler.ir.loop import LoopOp, layout_strides, stored_through
from emmy.compiler.ir.stmt import Assign, Load, Loop, Select, Statement, Write
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.fusion._region import chosen_cell, plain_load, wrap_multi_output_fragment

PATTERN = [Pattern("reader", LoopOp)]


def moved(graph: Graph, op: LoopOp) -> tuple[str, tuple, Write, DataType | None] | None:
    """``(the buffer read, the cell read, the store, the dtype the value is converted to)`` of a loop that computes
    nothing: a trip reads one cell of its one input — one read, or a choice among reads of the steps — and stores
    it, as it is (no dtype) or converted once."""
    body = op.body
    loads, writes = body.loads, body.writes
    rest = [stmt for stmt in body.iter() if not isinstance(stmt, (Loop, Load, Write))]
    selects = [stmt for stmt in rest if isinstance(stmt, Select)]
    rounds = [stmt for stmt in rest if isinstance(stmt, Assign) and stmt.dtype is not None and isinstance(stmt.value, Var)]
    if len(writes) != 1 or len({load.input for load in loads}) != 1 or not all(map(plain_load, loads)):
        return None
    if len(selects) > 1 or len(rounds) > 1 or len(selects) + len(rounds) != len(rest):
        return None
    value = tuple(writes[0].values)
    if rounds:
        value = (rounds[0].value.name,) if value == (rounds[0].name,) else None
    if selects:
        cell = chosen_cell(body, selects[0], graph.buffer(loads[0].input).shape) if value == (selects[0].name,) else None
    else:
        cell = tuple(loads[0].index) if len(loads) == 1 and value == tuple(loads[0].names) else None
    return None if cell is None else (loads[0].input, cell, writes[0], rounds[0].dtype if rounds else None)


def rewrite(match: Match, reader: Node) -> Graph:
    graph = match.graph
    found = moved(graph, reader.op) if len(reader.outputs) == 1 else None
    if found is None:
        raise RuleSkipped("the loop computes something, or reads its input more than once")
    source, cell, write, rounded = found
    stored, kept, producer = graph.buffer(source), reader.outputs[0], graph.producer(source)
    if producer is None or not isinstance(producer.op, LoopOp) or source in graph.outputs or graph.buffer_users(source) != {reader.id}:
        raise RuleSkipped("the buffer is read elsewhere, or no loop stores it")
    if (rounded is None and stored.dtype != kept.dtype) or stored.dtype.logical_elems != 1:
        raise RuleSkipped("the reader's store converts what it reads, or the buffer is packed")
    if stored.alias is not None or kept.alias is not None:
        raise RuleSkipped("one of the two is stored over another buffer")
    strides = layout_strides(reader.op, cell, write.index, stored, kept)
    op = stored_through(producer.op, stored, [(strides, kept)]) if strides is not None else None
    if op is None:
        raise RuleSkipped("the reader moves a part of the buffer, or the producer reads the buffer back")
    taken = set(op.body.defined)

    def rounding(stmt: Statement) -> Statement | tuple[Statement, ...]:
        if rounded is None or not isinstance(stmt, Write) or stmt.output != kept.name:
            return stmt
        name = f"{kept.name}__st"
        while name in taken:
            name += "_"
        taken.add(name)
        return Assign(name, Var(stmt.value), dtype=rounded), stmt.replace(values=(name,))

    live = tuple(kept.name if name == source else name for name in producer.buffer_names())
    fragment, match.output = wrap_multi_output_fragment(graph, replace(op, body=op.body.map(rounding)), live)
    match.consumed = {producer.id, reader.id}
    return fragment
