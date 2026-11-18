"""A carried walk in blocks: per block of rows, one kernel over the rows of earlier blocks and one short walk.

A kernel that carries a state and writes one row of it a step, summing over the rows it has written, is a walk as
long as the state is tall, and each of its steps computes the next value of every cell. Cut into blocks of rows
(``ir/tile/blocked.py`` owns the reasoning and the reading of the kernel), a step's sum splits in two and so does
the state:

- the rows of earlier blocks are finished when the block starts, so that part of every step's sum is one kernel
  that carries nothing, over the block's steps at once, at the row each step writes;
- the block's own rows are the walk that is left. It carries those rows alone: no step of the block reads or
  writes another row, so no other row is its state.

A block's rows start from the walk's own seed, cut out by a kernel of its own, and the blocks' rows are put back
into one state at the end.

Wired into no pass: a rewrite of the graph a caller applies by hand.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expression import BinaryExpression, Literal, TernaryExpression, Var, apply
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Assign, Body, Carry, Load, Loop, Pre, Select, SelectBranch, Write
from emmy.compiler.ir.tile.blocked import Walk, survey


def _plus(expression, amount: int):
    expression = Var(expression) if isinstance(expression, str) else expression
    if not amount:
        return expression
    return BinaryExpression("+" if amount > 0 else "-", expression, Literal(abs(amount), "int"))


def _nest(stmts: tuple, axes: tuple) -> tuple:
    for axis in reversed(axes):
        stmts = (Loop(axis=axis, body=Body(stmts)),)
    return stmts


def block_walk(graph: Graph, node_id: str, width: int) -> Graph:
    """Replace the carried walk ``node_id`` of ``graph`` by its block form, blocks of ``width`` rows, in place.

    Raises ``ValueError`` for a walk the form does not hold: one that stores more than its last state, that starts
    from a literal, whose steps do not move one row at a time, that sums over its rows in more than one place, or
    that reads a row outside its block anywhere but in that sum over earlier rows."""
    node = graph.nodes[node_id]
    body: Body = node.op.body
    walk = Walk(body)
    (position,) = (index for index, stmt in enumerate(body) if isinstance(stmt, Loop) and stmt.carries)
    before, after = tuple(body[:position]), tuple(body[position + 1 :])
    loop, carry = walk.loop, walk.carry
    step, state, seed = loop.axis.name, carry.name, carry.seed
    writes = [stmt for stmt in walk.stmts if isinstance(stmt, Write)]
    if len(writes) != 1 or writes[0].values != (carry.value,) or tuple(writes[0].index) != tuple(carry.index):
        raise ValueError("the block form takes a walk that stores its state alone, each step over the last")
    if not isinstance(seed, str):
        raise ValueError("the block form takes a walk that starts from a buffer")
    rows = walk.rows
    if walk.moving is None or (rows < 0).any() or (rows != rows[0] + np.arange(len(rows))).any():
        raise ValueError("the block form takes a walk that writes one row a step, each the next")
    at, moving = walk.moving, walk.cells[walk.moving]
    sums = [
        stmt
        for stmt in walk.stmts
        if isinstance(stmt, Loop)
        and stmt.is_reduce
        and any(isinstance(read, Pre) and read.carrier == state and read.index[at] == Var(stmt.axis.name) for read in stmt.body)
    ]
    if len(sums) != 1:
        raise ValueError("the block form takes a walk with one sum over the rows of its state")
    (summed,) = sums
    over = summed.axis.name
    inside = {read.name for read in summed.body if isinstance(read, Pre)}
    sums_names = [stmt.name for stmt in summed.body if isinstance(stmt, Accum)]
    axes = {stmt.axis.name: stmt.axis for stmt in (loop, *walk.stmts) if isinstance(stmt, Loop)}
    cells = tuple(Var(name) for name in walk.cells)
    others = tuple(Var(name) for name in walk.cells if name != moving)
    other_extents = tuple(Dim(extent) for index, extent in enumerate(walk.shape) if index != at)
    height = walk.shape[at]
    out = node.output

    def rowed(index: tuple, row) -> tuple:
        return tuple(row if place == at else coordinate for place, coordinate in enumerate(index))

    spans = {block: (block * width, min((block + 1) * width, height)) for block in range(-(-height // width))}
    stepped = {block: walk.steps_of(width, block) for block in spans}
    kept = {block: f"{out.name}__block{block}" for block in spans if len(stepped[block])}

    def home(block: int) -> tuple[str, int]:
        """Where a block's rows are once it is done, and the row they start at there: its own buffer, or the seed
        for a block no step writes."""
        return (kept[block], spans[block][0]) if block in kept else (seed, 0)

    frag = Graph()
    for name in node.inputs:
        frag.add_node(op=InputOp(), inputs=[], output=graph.buffer(name), node_id=name)

    def add(name: str, stmts: Body, shape: tuple, made: tuple = ()) -> None:
        loaded = {load.input for load in stmts.loads} | {stmt.seed for stmt in stmts.iter() if isinstance(stmt, Carry)}
        inputs = [buffer for buffer in dict.fromkeys((*node.inputs, *made)) if buffer in loaded]
        frag.add_node(op=LoopOp(body=stmts), inputs=inputs, output=Tensor(name, shape, F32))

    def shaped(count: int) -> tuple:
        return tuple(Dim(count) if place == at else Dim(extent) for place, extent in enumerate(walk.shape))

    def over_cells(count: int) -> tuple:
        return tuple(replace(axes[name], extent=Dim(count)) if name == moving else axes[name] for name in walk.cells)

    for block in kept:
        low, high = spans[block]
        steps = stepped[block]
        first, count = int(steps[0]), len(steps)
        found = survey(body, width, block)
        if any(reading.live and not (reading.name in inside and reading.block == 0) for reading in found):
            raise ValueError("the walk reads what a block wrote outside its sum over the block's own rows")
        if any(reading.block != 0 for reading in found if reading.name not in inside) or any(reading.block == 1 for reading in found):
            raise ValueError("the walk reads a row outside its block that its sum over earlier rows does not cover")
        if any(not reading.lifted for reading in found if reading.block == -1):
            raise ValueError("the rows of earlier blocks do not lift out whole")
        early = any(reading.block == -1 for reading in found)
        local = _plus(step, -first)
        shift = Sigma({step: _plus(step, first)})
        slots = tuple((Literal(slot, "int"),) if len(sums_names) > 1 else () for slot in range(len(sums_names)))

        # The block's rows as the walk found them: its own state starts from these.
        started = f"{out.name}__seed{block}"
        stmts = (
            Load(names=("_row",), input=seed, index=rowed(cells, _plus(moving, low))),
            Write(output=started, index=cells, values=("_row",)),
        )
        add(started, Body(_nest(stmts, over_cells(high - low))), shaped(high - low), (seed,))

        early_name = f"{out.name}__early{block}"
        if early:
            # The rows of earlier blocks, finished when the block starts: this part of every step's sum is read from
            # where each earlier block left its rows, at the row the step writes. The coefficients are that row's
            # own cells, which nothing has written yet: the seed's.
            pieces: list = []
            for earlier in range(block):
                buffer, start = home(earlier)
                here = Sigma({over: _plus(over, spans[earlier][0])})
                inner = []
                for original in summed.body:
                    part = original.substitute(here)
                    if isinstance(original, Pre) and original.carrier == state:
                        if original.index[at] == Var(over):
                            part = Load(names=(part.name,), input=buffer, index=rowed(part.index, _plus(part.index[at], -start)))
                        else:
                            part = Load(names=(part.name,), input=seed, index=part.index)
                    elif isinstance(original, Accum):
                        part = part.replace(name=f"{part.name}__{earlier}")
                    inner.append(part)
                rows_here = spans[earlier][1] - spans[earlier][0]
                pieces.append(Loop(axis=replace(summed.axis, extent=Dim(rows_here)), body=Body(tuple(inner))))
            for name, slot in zip(sums_names, slots, strict=True):
                total = f"{name}__0"
                for earlier in range(1, block):
                    pieces.append(Assign(f"{name}__to{earlier}", apply("add", total, f"{name}__{earlier}")))
                    total = f"{name}__to{earlier}"
                pieces.append(Write(output=early_name, index=(local, *others, *slot), values=(total,)))

            def lifted(stmt, pieces=tuple(pieces)):
                if isinstance(stmt, Loop) and stmt.is_reduce and stmt.axis.name == over:
                    return pieces
                if isinstance(stmt, Pre) and stmt.carrier == state:
                    return Load(names=(stmt.name,), input=seed, index=stmt.index)
                return stmt

            at_row = Sigma({moving: _plus(step, int(rows[0]))})

            def opened(stmt, at_row=at_row):
                return (
                    tuple(inner.substitute(at_row) for inner in stmt.body) if isinstance(stmt, Loop) and stmt.axis.name == moving else stmt
                )

            only = loop.body.map(lambda stmt: None if isinstance(stmt, Carry | Write) else stmt).map(lifted).pruned(()).map(opened)
            nest = Loop(axis=replace(loop.axis, extent=Dim(count)), body=Body(tuple(stmt.substitute(shift) for stmt in only)))
            shape = (Dim(count), *other_extents, *((Dim(len(sums_names)),) if len(sums_names) > 1 else ()))
            add(early_name, Body((*before, nest, *after)).pruned(()), shape, (seed, *kept.values()))

        # The block's own rows: a walk over the block's steps that carries those rows alone.
        inward = Sigma({moving: _plus(moving, low)})

        def walked(stmt, early=early, early_name=early_name, low=low, high=high, local=local, slots=slots, inward=inward):
            if isinstance(stmt, Loop) and stmt.is_reduce and stmt.axis.name == over:
                own = Sigma({over: _plus(over, low)})
                inner = tuple(part.substitute(own) for part in stmt.body)
                if early:
                    inner = tuple(part.replace(name=f"{part.name}__own") if isinstance(part, Accum) else part for part in inner)
                cut = Loop(axis=replace(stmt.axis, extent=Dim(high - low)), body=Body(inner))
                if not early:
                    return cut
                joined = []
                for name, slot in zip(sums_names, slots, strict=True):
                    joined += [
                        Load(names=(f"{name}__early",), input=early_name, index=(local, *others, *slot)),
                        Assign(name, apply("add", f"{name}__own", f"{name}__early")),
                    ]
                return (cut, *joined)
            if isinstance(stmt, Loop) and stmt.axis.name == moving:
                return Loop(
                    axis=replace(stmt.axis, extent=Dim(high - low)), body=Body(tuple(part.substitute(inward) for part in stmt.body))
                )
            return stmt

        def homed(stmt, low=low, started=started, left=kept[block]):
            # The state is the block's rows: a cell's row counts from the block's first.
            if isinstance(stmt, Pre) and stmt.carrier == state:
                return stmt.replace(index=rowed(stmt.index, _plus(stmt.index[at], -low)))
            if isinstance(stmt, Carry):
                return stmt.replace(index=cells, seed=started)
            return stmt.replace(output=left, index=cells) if isinstance(stmt, Write) else stmt

        only = loop.body.map(walked).map(homed)
        nest = Loop(axis=replace(loop.axis, extent=Dim(count)), body=Body(tuple(stmt.substitute(shift) for stmt in only)))
        add(kept[block], Body((*before, nest, *after)), shaped(high - low), (started, early_name))

    # The state again: every block's rows where they belong.
    whole = f"{out.name}__blocked"
    stmts: list = []
    branches = []
    for block, (low, high) in spans.items():
        buffer, start = home(block)
        mine = BinaryExpression(
            "&&", BinaryExpression("<=", Literal(low, "int"), Var(moving)), BinaryExpression("<", Var(moving), Literal(high, "int"))
        )
        index = (
            cells if buffer == seed else tuple(TernaryExpression(mine, i, Literal(0, "int")) for i in rowed(cells, _plus(moving, -start)))
        )
        stmts.append(Load(names=(f"_rows{block}",), input=buffer, index=index))
        last = block == max(spans)
        branches.append(
            SelectBranch(f"_rows{block}", Literal(True, "bool") if last else BinaryExpression("<", Var(moving), Literal(high, "int")))
        )
    stmts += [Select(name="_row", branches=tuple(branches)), Write(output=whole, index=cells, values=("_row",))]
    add(whole, Body(_nest(tuple(stmts), over_cells(height))), shaped(height), (seed, *kept.values()))
    frag.outputs = [whole]
    graph.splice(frag, consumed=[node_id], output=node_id)
    return graph
