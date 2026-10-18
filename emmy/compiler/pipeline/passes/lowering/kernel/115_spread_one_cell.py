"""Spread a single-cell kernel's free loops over its CTA's threads.

A kernel whose tile has no grid axis is one cell: its launch is one CTA, and the scalar tier
runs the whole body on thread 0 while the rest of the block idles. A term it lowers over a
coordinate no grid binds — a store over every slot beside a count, a list built from a rank —
is a free loop (``Loop`` with no carrier), whose iterations are independent. Here each such
loop at the top of the body strides over the block (``threadIdx.x`` by ``blockDim.x``), every
thread runs the rest of the body — a scalar the loops read is computed by each, a scalar store
writes the same value from each — and a barrier precedes a statement reading a buffer an
earlier spread loop wrote, since another thread may have written it.

It decides nothing: the kernel computes the same values, and only the threads that share the
work change, so it answers to no knob. The kernel's identity is its Loop IR, untouched.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.kernel import KernelOp, Tile
from emmy.compiler.ir.kernel.ir import Sync
from emmy.compiler.ir.stmt import Body, Load, Loop, Stmt, Write
from emmy.compiler.ir.stmt.blocks import StridedLoop
from emmy.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", KernelOp)]


def rewrite(root: Node) -> KernelOp:
    op: KernelOp = root.op
    tiles = [stmt for stmt in op.body if isinstance(stmt, Tile)]
    if len(tiles) != 1 or tiles[0].axes or tiles[0].block_threads is not None or tiles[0].spread:
        raise RuleSkipped("not a single-cell scalar tile")
    (tile,) = tiles
    if not any(isinstance(stmt, Loop) and not stmt.is_reduce for stmt in tile.body):
        raise RuleSkipped("no free loop to spread")
    body: list[Stmt] = []
    written: set[str] = set()
    for stmt in tile.body:
        if written & _reads(stmt):
            body.append(Sync())
            written.clear()
        if isinstance(stmt, Loop) and not stmt.is_reduce:
            body.append(StridedLoop(axis=stmt.axis, start=Var("threadIdx.x"), step=Var("blockDim.x"), body=stmt.body, seed=stmt.seed))
            written |= _writes(stmt)
        else:
            body.append(stmt)
    spread = replace(tile, body=Body(tuple(body)), spread=True)
    return replace(op, body=Body(tuple(spread if stmt is tile else stmt for stmt in op.body)))


def _reads(stmt: Stmt) -> set[str]:
    return {s.input for s in _walk(stmt) if isinstance(s, Load)}


def _writes(stmt: Stmt) -> set[str]:
    return {s.output for s in _walk(stmt) if isinstance(s, Write)}


def _walk(stmt: Stmt):
    yield stmt
    for body in stmt.nested():
        for child in body:
            yield from _walk(child)
