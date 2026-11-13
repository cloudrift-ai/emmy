"""Unroll every loop a lowering site flagged: a loop in the body is a loop in the code.

A lowering site that wants a loop unrolled says so on the statement (``unroll``; the ``UNROLL`` budget caps which
loops a site flags). Here each flagged ``Loop`` / ``StridedLoop`` whose trips a static number bounds becomes copies of
its body, the axis bound to its value in each (``_unroll.unroll_where``), innermost first:

- A loop of at most ``_WHOLE`` trips becomes one copy per trip. A trip only some threads run (a lane loop over a row
  its lanes do not divide) keeps the loop's bound as the guard of its copy.
- A longer loop stays a loop that runs up to ``_PER_TRIP`` copies of its body per trip: the tensor-core K loop of a
  deep contraction, which its site flags at any length.

The two numbers are what nvcc did when the flag was a ``#pragma unroll`` it read: it unrolled a 64-trip scalar reduce
in full and gave a 128-trip tensor-core loop 32 copies per trip, so the machine code keeps its shape. The body's
optimization then folds each copy's index and computes once what two copies share. A flagged loop with no static
bound on its trips (a symbolic extent, an early stop, a start of unknown range) stays as it is. In every case the
flag is spent: no statement carries it past this pass, so whatever generates the kernel's code unrolls nothing and is
asked for nothing.

It runs after the passes that match loops and the statements in them (the vector passes, the load pairing, the
guards) and before ``120_explicit_conversions``, the one place that decides types: a copy reads its axis as a
literal, and a literal is typed by the operation that reads it. It decides nothing: the flag is the decision.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Body, Loop, Statement, StridedLoop
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.kernel._unroll import unroll_where

PATTERN = [Pattern("root", KernelOp)]


#: The longest loop unrolled in full, and the copies a longer one runs per trip (the module doc).
_WHOLE = 64
_PER_TRIP = 32


def rewrite(root: Node) -> KernelOp:
    op: KernelOp = root.op
    if not any(isinstance(stmt, (Loop, StridedLoop)) and stmt.unroll for stmt in op.body.iter()):
        raise RuleSkipped("no loop flagged for unrolling")
    return replace(op, body=_unflagged(unroll_where(op.body, _copies)), knobs=dict(op.knobs))


def _copies(loop: Statement, trips: tuple[int, int]) -> int | None:
    """The copies a flagged loop takes."""
    every, most = trips
    if not loop.unroll:
        return None
    if most <= _WHOLE:
        return most
    return max((n for n in range(2, _PER_TRIP + 1) if most % n == 0), default=None) if every == most else None


def _unflagged(body: Body) -> Body:
    """``body`` with the flag cleared on the loops that stay."""
    out: list[Statement] = []
    for stmt in body:
        nested = stmt.nested()
        if nested:
            stmt = stmt.with_bodies(tuple(_unflagged(inner) for inner in nested))
        out.append(stmt.replace(unroll=False) if isinstance(stmt, (Loop, StridedLoop)) and stmt.unroll else stmt)
    return Body(out)
