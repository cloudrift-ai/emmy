"""Keep a cooperative row in registers: unroll short lane-strided loops, so each cell is loaded once.

A cooperative reduce with a full-row projection (RMSNorm, RoPE over a normed row, softmax) walks
its row twice per lane: the reduce loop reads ``x[lane + coop·k]``, and after the combine the
projection loop reads the same cells again — and a rotate-half partner reads a cell another trip
of the same lane already holds. Both loops are short (a 128-wide row at 32 lanes is four trips),
but nvcc cannot reuse the first loop's values in the second: the second loop stores to the
output between trips, and it cannot prove the output does not alias the row.

This pass unrolls every lane-strided loop of at most ``_MAX_TRIPS`` trips whose extent the step
divides, substituting each trip's coordinate, and then runs the Kernel IR optimizer over the body:
its simplification folds each trip's index against the lane ranges (a partner read
``(i < 64) ? i + 64 : i - 64`` at ``i = lane + 32k`` resolves to another trip's own read), and its
sharing computes a load an earlier load of the same scope already read once, since no buffer of a
kernel aliases another. A reduce loop's carriers are seeded ahead of the trips, as the loop would
have seeded them.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.expression import BinaryExpression, Interval, Literal, SimplifyCtx
from emmy.compiler.ir.kernel import KernelOp, optimize_kernel_body
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Assign, Body, Init, Let, Load, Select, Statement, StridedLoop, Write
from emmy.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", KernelOp)]

#: The longest lane loop unrolled: past it, the registers a row holds outgrow what reuse saves.
_MAX_TRIPS = 8

_FLAT = (Assign, Accum, Load, Select, Let, Write)


def rewrite(root: Node) -> KernelOp:
    op: KernelOp = root.op
    body = _walk(op.body, SimplifyCtx.empty())
    if body == op.body:
        raise RuleSkipped("no short lane loop to unroll")
    return replace(op, body=optimize_kernel_body(body))


def _walk(body: Body, ctx: SimplifyCtx) -> Body:
    stmts: list[Statement] = []
    for stmt in body:
        inner = ctx
        if isinstance(stmt, Tile):
            for axis in stmt.axes:
                if axis.extent.is_static:
                    inner = inner.extend(axis.name, Interval(0, axis.extent.as_static() - 1))
        nested = stmt.nested()
        if nested:
            stmt = stmt.with_bodies(tuple(_walk(b, inner) for b in nested))
        stmts.append(stmt)
    unrolled: list[Statement] = []
    loops = 0
    for stmt in stmts:
        trips = _trips(stmt, ctx)
        if trips is None:
            unrolled.append(stmt)
            continue
        # Sibling loops over independently spliced cones bind the same names, each in its own C scope.
        # Flattened into one, a repeat would be a second declaration: give such a loop's trips their own.
        trip_stmts = _unroll(stmt, trips, "")
        bound = {name for s in unrolled for name in s.defines()}
        if any(name in bound for s in trip_stmts if not isinstance(s, (Init, Accum)) for name in s.defines()):
            trip_stmts = _unroll(stmt, trips, f"_{loops}")
        unrolled.extend(trip_stmts)
        loops += 1
    return Body(tuple(unrolled))


def _trips(stmt: Statement, ctx: SimplifyCtx) -> int | None:
    """How many trips a short lane loop takes, or ``None`` when it is not one this pass unrolls."""
    if not isinstance(stmt, StridedLoop) or stmt.end is not None or not stmt.axis.extent.is_static:
        return None
    if not isinstance(stmt.step, Literal) or not isinstance(stmt.step.value, int) or stmt.step.value < 1:
        return None
    extent, step = stmt.axis.extent.as_static(), stmt.step.value
    if extent % step or not 1 < extent // step <= _MAX_TRIPS:
        return None
    if not all(isinstance(s, _FLAT) and not s.nested() for s in stmt.body):
        return None
    # Every unrolled trip stays inside the loop only when ``start`` is proven in ``[0, step)``: a lane index or zero.
    start = stmt.start.range(ctx)
    if start is None or start.lo < 0 or start.hi >= step:
        return None
    return extent // step


def _unroll(loop: StridedLoop, trips: int, tag: str) -> list[Statement]:
    carried = {s.name for s in loop.body if isinstance(s, Accum)}
    out: list[Statement] = []
    if loop.seed:
        seen: set[str] = set()
        for s in loop.body:
            if isinstance(s, Accum) and s.name not in seen:
                seen.add(s.name)
                out.append(Init(s.name, s.op.identity, dtype=s.dtype))
    defined = {name for s in loop.body for name in s.defines()} - carried
    for k in range(trips):
        coord = BinaryExpression("+", loop.start, Literal(k * loop.step.value, "int"))
        sigma = Sigma({loop.axis.name: coord})
        suffix = f"__u{k}{tag}"
        for s in loop.body:
            out.append(s.rewrite(lambda name, suffix=suffix: f"{name}{suffix}" if name in defined else name, sigma))
    return out
