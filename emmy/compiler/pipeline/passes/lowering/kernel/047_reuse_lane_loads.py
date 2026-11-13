"""Keep a cooperative row in registers: unroll short lane-strided loops, then load each cell once.

A cooperative reduce with a full-row projection (RMSNorm, RoPE over a normed row, softmax) walks
its row twice per lane: the reduce loop reads ``x[lane + coop·k]``, and after the combine the
projection loop reads the same cells again — and a rotate-half partner reads a cell another trip
of the same lane already holds. Both loops are short (a 128-wide row at 32 lanes is four trips),
but nvcc cannot reuse the first loop's values in the second: the second loop stores to the
output between trips, and it cannot prove the output does not alias the row.

This pass unrolls every lane-strided loop of at most ``_MAX_TRIPS`` trips whose extent the step
divides, substituting each trip's coordinate. A reduce loop's carriers are seeded ahead of the
trips, as the loop would have seeded them. The body's optimization does the rest: it folds each
index against the lane ranges, so a partner read ``(i < 64) ? i + 64 : i - 64`` at ``i = lane + 32k``
resolves to another trip's own read, and it keeps one load of a cell the kernel never writes,
on either side of the combine's barrier.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Accum, Assign, Let, Load, Select, Statement, StridedLoop, Write
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.kernel._unroll import unroll_where

PATTERN = [Pattern("root", KernelOp)]

#: The longest lane loop unrolled: past it, the registers a row holds outgrow what reuse saves.
_MAX_TRIPS = 8

_FLAT = (Assign, Accum, Load, Select, Let, Write)


def rewrite(root: Node) -> KernelOp:
    op: KernelOp = root.op
    body = unroll_where(op.body, _copies)
    if body == op.body:
        raise RuleSkipped("no short lane loop to unroll")
    return replace(op, body=body)


def _copies(stmt: Statement, trips: tuple[int, int] | None) -> int | None:
    """The trips of a short lane loop, or ``None`` when it is not one this pass unrolls."""
    if trips is None or not isinstance(stmt, StridedLoop) or not all(isinstance(s, _FLAT) and not s.nested() for s in stmt.body):
        return None
    return trips[0] if trips[0] == trips[1] and 1 < trips[0] <= _MAX_TRIPS else None
