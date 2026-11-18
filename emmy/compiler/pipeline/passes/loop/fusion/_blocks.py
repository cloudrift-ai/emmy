"""A carried walk in blocks: per block of rows, one kernel over the rows of earlier blocks and one short walk.

A kernel that carries a state and writes one row of it a step, summing over the rows it has written, is a walk as
long as the state is tall. Cut into blocks of rows (``ir/tile/blocked.py`` owns the reasoning and the reading of the
kernel), the sum of a step splits in two. The rows of earlier blocks are finished when the block starts, so that part
of every step's sum is one kernel that carries nothing, over the block's steps at once. The block's own rows are the
walk that is left: as many steps as the block has rows, each summing over the block alone.

Wired into no pass: a rewrite of the graph a caller applies by hand.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expression import BinaryExpression, Literal, Var, apply
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Assign, Body, Carry, Load, Loop, Pre, Write
from emmy.compiler.ir.tile.blocked import Walk, survey


def _plus(name: str, amount: int):
    return BinaryExpression("+", Var(name), Literal(amount, "int")) if amount else Var(name)


def block_walk(graph: Graph, node_id: str, width: int) -> Graph:
    """Replace the carried walk ``node_id`` of ``graph`` by its block form, blocks of ``width`` rows, in place.

    Raises ``ValueError`` for a walk the form does not hold: one that stores more than its last state, whose steps
    do not move one row at a time, that sums over its rows in more than one place, or that reads what a block wrote
    anywhere but in that sum."""
    node = graph.nodes[node_id]
    body: Body = node.op.body
    walk = Walk(body)
    (position,) = (index for index, stmt in enumerate(body) if isinstance(stmt, Loop) and stmt.carries)
    before, after = tuple(body[:position]), tuple(body[position + 1 :])
    loop, carry = walk.loop, walk.carry
    step, state = loop.axis.name, carry.name
    writes = [stmt for stmt in walk.stmts if isinstance(stmt, Write)]
    if len(writes) != 1 or writes[0].values != (carry.value,) or tuple(writes[0].index) != tuple(carry.index):
        raise ValueError("the block form takes a walk that stores its state alone, each step over the last")
    rows = walk.rows
    if walk.moving is None or (rows < 0).any() or (rows != rows[0] + np.arange(len(rows))).any():
        raise ValueError("the block form takes a walk that writes one row a step, each the next")
    moving = walk.cells[walk.moving]
    sums = [
        stmt
        for stmt in walk.stmts
        if isinstance(stmt, Loop)
        and stmt.is_reduce
        and any(isinstance(read, Pre) and read.carrier == state and read.index[walk.moving] == Var(stmt.axis.name) for read in stmt.body)
    ]
    if len(sums) != 1:
        raise ValueError("the block form takes a walk with one sum over the rows of its state")
    (summed,) = sums
    over, extent = summed.axis.name, summed.axis.extent.as_static()
    inside = {read.name for read in summed.body if isinstance(read, Pre)}
    sums_names = [stmt.name for stmt in summed.body if isinstance(stmt, Accum)]
    others = tuple(Var(name) for name in walk.cells if name != moving)
    other_extents = tuple(Dim(walk.shape[index]) for index, name in enumerate(walk.cells) if name != moving)
    out = node.output

    frag = Graph()
    for name in node.inputs:
        frag.add_node(op=InputOp(), inputs=[], output=graph.buffer(name), node_id=name)

    def reads(stmts: Body, *made: str) -> list[str]:
        loaded = {load.input for load in stmts.loads} | {
            stmt.seed for stmt in stmts.iter() if isinstance(stmt, Carry) and isinstance(stmt.seed, str)
        }
        return [name for name in dict.fromkeys((*node.inputs, *made)) if name in loaded]

    blocks = sorted({int(row) // width for row in rows})
    came = carry.seed  # what the block starts from: the seed, then the state the block before left
    for block in blocks:
        steps = walk.steps_of(width, block)
        first, count, low = int(steps[0]), len(steps), block * width
        found = survey(body, width, block)
        if any(reading.live and not (reading.name in inside and reading.block == 0) for reading in found):
            raise ValueError("the walk reads what a block wrote outside its sum over the block's own rows")
        if any(not reading.lifted for reading in found if reading.name in inside and reading.block == -1):
            raise ValueError("the rows of earlier blocks do not lift out whole")
        early = any(reading.block == -1 for reading in found if reading.name in inside)
        top = extent if any(reading.block == 1 for reading in found if reading.name in inside) else min(low + width, extent)
        local = BinaryExpression("-", Var(step), Literal(first, "int"))
        shift = Sigma({step: _plus(step, first)})
        slots = tuple((Literal(slot, "int"),) if len(sums_names) > 1 else () for slot in range(len(sums_names)))

        early_name = f"{out.name}__early{block}"
        if early and not isinstance(came, str):
            raise ValueError("the first block the walk writes has earlier rows, and the walk starts from a literal")
        if early:
            # The rows of earlier blocks: finished when the block starts, so this part of every step's sum is read
            # from the state the block starts from, at the row the step writes.
            def lifted(stmt, early_name=early_name, low=low, local=local, came=came, slots=slots):
                if isinstance(stmt, Loop) and stmt.is_reduce and stmt.axis.name == over:
                    cut = Loop(axis=replace(stmt.axis, extent=Dim(low)), body=stmt.body)
                    stores = tuple(
                        Write(output=early_name, index=(local, *others, *slot), values=(name,))
                        for name, slot in zip(sums_names, slots, strict=True)
                    )
                    return (cut, *stores)
                if isinstance(stmt, Pre) and stmt.carrier == state:
                    return Load(names=(stmt.name,), input=came, index=stmt.index)
                return None if isinstance(stmt, Carry | Write) else stmt

            at_row = Sigma({moving: _plus(step, int(rows[0]))})

            def opened(stmt, at_row=at_row):
                return (
                    tuple(inner.substitute(at_row) for inner in stmt.body) if isinstance(stmt, Loop) and stmt.axis.name == moving else stmt
                )

            kept = loop.body.map(lifted).pruned(()).map(opened)
            nest = Loop(axis=replace(loop.axis, extent=Dim(count)), body=Body(tuple(stmt.substitute(shift) for stmt in kept)))
            stmts = Body((*before, nest, *after)).pruned(())
            shape = (Dim(count), *other_extents, *((Dim(len(sums_names)),) if len(sums_names) > 1 else ()))
            frag.add_node(
                op=LoopOp(body=stmts),
                inputs=reads(stmts, *([came] if isinstance(came, str) else [])),
                output=Tensor(early_name, shape, F32),
            )

        # The block's own rows: the walk over the block's steps, its sum over the block alone, plus what was lifted.
        left = f"{out.name}__block{block}"

        def walked(stmt, early=early, early_name=early_name, low=low, top=top, local=local, came=came, left=left, slots=slots):
            if isinstance(stmt, Loop) and stmt.is_reduce and stmt.axis.name == over:
                own = Sigma({over: _plus(over, low)})
                inner = tuple(part.substitute(own) for part in stmt.body)
                if early:
                    inner = tuple(part.replace(name=f"{part.name}__own") if isinstance(part, Accum) else part for part in inner)
                cut = Loop(axis=replace(stmt.axis, extent=Dim(top - low)), body=Body(inner))
                if not early:
                    return cut
                joined = []
                for name, slot in zip(sums_names, slots, strict=True):
                    joined += [
                        Load(names=(f"{name}__early",), input=early_name, index=(local, *others, *slot)),
                        Assign(name, apply("add", f"{name}__own", f"{name}__early")),
                    ]
                return (cut, *joined)
            if isinstance(stmt, Carry):
                return stmt.replace(seed=came)
            return stmt.replace(output=left) if isinstance(stmt, Write) else stmt

        kept = loop.body.map(walked)
        nest = Loop(axis=replace(loop.axis, extent=Dim(count)), body=Body(tuple(stmt.substitute(shift) for stmt in kept)))
        stmts = Body((*before, nest, *after))
        made = [*([came] if isinstance(came, str) else []), *([early_name] if early else [])]
        frag.add_node(op=LoopOp(body=stmts), inputs=reads(stmts, *made), output=replace(out, name=left))
        came = left
    frag.outputs = [came]
    graph.splice(frag, consumed=[node_id], output=node_id)
    return graph
