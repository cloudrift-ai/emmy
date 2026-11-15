"""Unroll the loops no lowering site flagged, by the work one trip does.

``117_unroll_loops`` spends the flags; this pass decides the copies of every other loop, so the code generator writes
each loop as it is and ptxas unrolls none of them (a loop left stepping is marked ``nounroll``). The rule and its
numbers (``emmy.compiler.target``) are calibrated on NVIDIA's compilers for ``sm_120``: cicc's PTX and ptxas's
machine code across trip counts and body sizes. A body's size is the work one trip does (:func:`_size`): one for each
memory access, fold and branch, and one for each operation whose value changes from trip to trip, a math function
counting its instructions (``MATH_INSTRUCTIONS``, in the form ``130_approximate_math`` will pick under the live
precision decisions). What does not change across trips is computed once ahead of the loop, so it counts nothing.

- **A serial ``Loop`` over a static extent** of ``N`` trips is written out in full when ``N·(size + 2) ≤ 480``. Else
  it steps ``C`` trips at a time, ``C`` the largest power of two that divides ``N`` with ``C·(size + 2) ≤ 199``, and
  eight where the rule gives two and eight divide ``N`` (the f32 register-tile matmuls' 128-trip ``_ki`` loop). It
  stays as it is when ``N`` is odd. A stepping loop left with at most four trips of at most 128 work each is written
  out in full: the sibling loops of a register tile then run side by side
  (``fused/norm-weight-linear-f16-computed-b``, eight loops of 128 trips).
- **A loop whose trips only the launch knows** (a symbolic extent, :func:`_unroll.runtime_bound`) takes ``C`` copies,
  the largest of 16, 8, 4 and 2 with ``C·(size + 7) ≤ 228``, none past a size of 89: a loop stepping ``C`` trips
  while that many are left, then the trips left in blocks of half as many copies down to one, four and a loop of one
  copy past 8 copies (:func:`_unroll._runtime`).
- **A strided loop whose start differs between threads**, over a static extent that leaves every thread the same
  ``N`` trips, steps ``C`` trips at a time from 24 trips on: ``C`` the largest power of two dividing ``N``, at most
  16, and none past a body size of 89. Below 24 trips, and where the threads' trips differ, it stays as it is.

Every loop the pass leaves stepping several trips at a time says it is ``unrolled``, and the PTX compiler keeps it a
loop (``.pragma "nounroll"``): unmarked, ptxas writes such compact loops out in full (``qwen3emb/sdpa-s32``: 11,344
machine instructions unmarked, 2,216 marked). A loop no unroll pass takes is left unmarked, and ptxas still decides
its copies.

Only a body of target-neutral statements is sized. A loop holding a barrier, an asynchronous copy, a shuffle or a
fragment statement stays as it is. A loop already ``unrolled`` is taken no further, so the pass is idempotent.

How many copies a loop takes is a decision that moves under the ``UNROLL`` knob, like the two limits of ``117``. It
runs after ``117``, so a written-out inner loop counts in its outer loop's size, and before ``119_vectorize_copies``,
which widens the runs of loads the copies line up.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.context import Context
from emmy.compiler.graph import Node
from emmy.compiler.ir.expression import (
    BinaryExpression,
    CastExpression,
    Expression,
    FunctionCallExpression,
    Literal,
    TernaryExpression,
    Var,
)
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Accum, Assign, Cond, Init, Let, Load, Loop, Select, Statement, StridedLoop, Write
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.kernel._approximate import approximate_forms
from emmy.compiler.pipeline.passes.lowering.kernel._unroll import unroll_where
from emmy.compiler.target import (
    MATH_INSTRUCTIONS,
    PTXAS_AGAIN,
    PTXAS_WHOLE_TRIPS,
    PTXAS_WHOLE_WORK,
    RUNTIME_COPIES,
    RUNTIME_LARGEST,
    RUNTIME_TRIP,
    RUNTIME_WORK,
    STATIC_PARTIAL,
    STATIC_TRIP,
    STATIC_WHOLE,
    THREAD_START_LEAST,
)

PATTERN = [Pattern("root", KernelOp)]

#: The statements a sized body holds: the target-neutral ones.
_NEUTRAL = (Load, Assign, Select, Accum, Write, Init, Let, Cond, Loop, StridedLoop)


def rewrite(root: Node, ctx: Context) -> KernelOp:
    op: KernelOp = root.op
    forms = approximate_forms(ctx)
    body = unroll_where(op.body, lambda loop, trips: _copies(loop, trips, forms))
    if body == op.body:
        raise RuleSkipped("no loop to unroll by its size")
    return replace(op, body=body, knobs=dict(op.knobs))


def _copies(loop: Statement, trips: tuple[int, int] | None, forms: dict[str, str]) -> int | None:
    """The copies ``loop`` takes (the module doc), or ``None``."""
    if loop.unrolled or not all(isinstance(stmt, _NEUTRAL) for stmt in loop.body.iter()):
        return None
    size = _size(loop, forms)
    if trips is None:
        fits = (n for n in _powers(RUNTIME_COPIES) if n * (size + RUNTIME_TRIP) <= RUNTIME_WORK)
        return next(fits, None) if size <= RUNTIME_LARGEST else None
    if isinstance(loop, StridedLoop):
        copies = min(RUNTIME_COPIES, trips[1] & -trips[1])
        uniform = trips[0] == trips[1] >= THREAD_START_LEAST and not isinstance(loop.start, Literal)
        return copies if uniform and copies > 1 and size <= RUNTIME_LARGEST else None
    if trips[1] < 1:
        return None
    if trips[1] * (size + STATIC_TRIP) <= STATIC_WHOLE:
        return trips[1]
    copies = next((n for n in _powers(trips[1] & -trips[1]) if n * (size + STATIC_TRIP) <= STATIC_PARTIAL), None)
    if copies == 2 and trips[1] % (PTXAS_AGAIN * copies) == 0:
        copies *= PTXAS_AGAIN
    if copies is not None and trips[1] <= PTXAS_WHOLE_TRIPS * copies and copies * size <= PTXAS_WHOLE_WORK:
        return trips[1]
    return copies


def _powers(most: int) -> list[int]:
    """The powers of two from ``most`` down to 2."""
    return [most >> k for k in range(most.bit_length() - 1)]


def _size(loop: Statement, forms: dict[str, str]) -> int:
    """The work one trip of ``loop`` does: one per memory access, fold, selection, branch and inner loop, and one for
    each operation whose value changes from trip to trip (it reads the axis or a name the body defines). An access
    whose address changes pays one more for it, and the operations of its index that are no sum of scaled terms."""
    changing = {loop.axis.name} | {name for stmt in loop.body.iter() for name in stmt.defines()}
    changing |= {stmt.axis.name for stmt in loop.body.iter() if isinstance(stmt, (Loop, StridedLoop))}

    def varies(expression: Expression) -> bool:
        return bool(expression.free_vars() & changing)

    def work(expression: Expression) -> int:
        return sum(_cost(term, forms) for term in expression.subterms() if varies(term))

    def beyond_affine(coord: Expression) -> int:
        if not varies(coord) or isinstance(coord, Var):
            return 0
        if isinstance(coord, BinaryExpression) and coord.op in ("+", "-"):
            return beyond_affine(coord.left) + beyond_affine(coord.right)
        if isinstance(coord, BinaryExpression) and coord.op == "*" and not (varies(coord.left) and varies(coord.right)):
            return beyond_affine(coord.left) + beyond_affine(coord.right)
        return work(coord)

    def access(index) -> int:
        return 1 + int(any(varies(coord) for coord in index)) + sum(beyond_affine(coord) for coord in index)

    size = 0
    for stmt in loop.body.iter():
        if isinstance(stmt, (Load, Write)):
            size += access(stmt.index)
        elif isinstance(stmt, Select):
            size += len(stmt.branches) - 1 + sum(work(branch.select) for branch in stmt.branches)
        elif isinstance(stmt, (Accum, Cond, Loop, StridedLoop)):
            size += 1 + sum(work(expression) for expression in stmt.expressions())
        else:
            size += sum(work(expression) for expression in stmt.expressions())
    return size


def _cost(term: Expression, forms: dict[str, str]) -> int:
    """The instructions ``term``'s own operation takes: a math function's from ``MATH_INSTRUCTIONS``, one for any other."""
    if isinstance(term, FunctionCallExpression):
        return MATH_INSTRUCTIONS.get(forms.get(term.name, term.name), 1)
    return 1 if isinstance(term, (BinaryExpression, CastExpression, TernaryExpression)) else 0
