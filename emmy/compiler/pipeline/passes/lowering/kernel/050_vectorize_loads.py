"""Widen runs of consecutive scalar ``Load`` Stmts into one vector ``Load``.

Until this pass, the body of every materialized kernel carries scalar
``Load`` Stmts. Some sequences of those Loads have a "vector" shape: N
consecutive Loads from the same source buffer whose last-dim indices differ
by 0, 1, ..., N-1. The CUDA backend can emit those as a single
``float<N>`` / ``__half2`` reinterpret-cast read followed by N ``.x/.y/.z/.w``
unpacks. Folding the run into one ``Load(names=(n0, n1, ...), input, index)``
makes the optimization visible in the IR (``--ir kernel`` shows one Load with
multiple LHS names) while keeping the renderer simple — ``Load``'s render
branches on the vector form.

## What the pass does

For each ``Body`` (every nested Tile / Loop / StridedLoop / Cond body,
post-order):

1. Take the scalar ``Load``s of each buffer and their complete flat addresses.
2. Try widths 16 then 8 then 4 then 2 (16 only for 1-byte dtypes): ``n`` loads whose addresses form ``anchor,
   anchor+1, ..., anchor+n-1`` with an aligned anchor for every free coordinate, which the scope's ordering
   constraints let stand together (``Siblings``: nothing that must run between two of them, such as a write of the
   buffer). Where they stand says nothing: an unrolled pointwise body interleaves each element's load with its
   arithmetic. If the target also supports
   ``vector_type(elem_dtype, n)`` for the source-buffer dtype, the loads become one widened ``Load``.

Before grouping, a Load index ``(Y + r) / d`` whose ``Y`` is a non-negative multiple of ``g`` (variable coefficients
times the strided-loop alignment), with ``g`` dividing ``d`` and ``0 < r < g``, reads ``Y / d``: every element of a
lane's run then reads a block-scaled weight's scale at the run's own index, which nvcc loads once. On the V100 the
width-1 fp8 down projection read 8 scales per 8-byte run and ran 181 us; read once per run it runs 127 us.

## Why this needs the source-buffer dtype

The decision needs the source-buffer dtype, read off the stamped
``Load.dtype`` (``030_stamp_types``).

## Observed impact

ptxas coalesces scalar ``ld.shared`` runs once alignment is known, so for
shared memory the two source forms compile alike. Global f16 loads are
different: ptxas cannot prove their alignment and keeps them 2 bytes wide,
so an unrolled pointwise kernel reads a quarter of its bandwidth per
instruction until this pass widens them. ``VECTORIZE_LOADS`` is still not a
search dimension — only ``True`` is enumerated. ``EMMY_VECTORIZE_LOADS=0`` is
a manual override.
"""

from __future__ import annotations

from dataclasses import replace
from math import gcd

from emmy.compiler.backend.cuda.render_target import CudaRenderTarget
from emmy.compiler.graph import Node
from emmy.compiler.ir.expression import BinaryExpression, Expression, Literal, SimplifyCtx, Var, affine_form
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Body, Load, Statement, StridedLoop
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.kernel._siblings import Siblings
from emmy.compiler.pipeline.passes.lowering.kernel._vector import Address, flat_address, fuse_runs, strided_alignment
from emmy.compiler.pipeline.search.space import VECTORIZE_LOADS

PATTERN = [Pattern("root", KernelOp)]

_TARGET = CudaRenderTarget()


def rewrite(root: Node) -> KernelOp | None:
    top: KernelOp = root.op
    # Idempotence: the policy is recorded as the VECTORIZE_LOADS knob, so a
    # re-scan of the rebound op skips here.
    if VECTORIZE_LOADS.name in top.knobs:
        raise RuleSkipped("VECTORIZE_LOADS already decided (idempotence via knob)")
    # Only ``True`` is enumerated, so the autotuner never forks on this knob;
    # ``EMMY_VECTORIZE_LOADS=0`` still pins ``False``.
    if not VECTORIZE_LOADS.narrow((True,))[0]:
        return replace(top, body=top.body, knobs={**top.knobs, VECTORIZE_LOADS.name: False})
    # Stamp the policy (True) even when no run is foldable — the realized config
    # records that vectorization was enabled, keeping a uniform knob set.
    new_body = _vectorize_body(top, top.body)
    return replace(top, body=new_body, knobs={**top.knobs, VECTORIZE_LOADS.name: True})


def _vectorize_body(top: KernelOp, body: Body, aligned: dict[str, int] | None = None) -> Body:
    """Post-order body transform: recurse into nested bodies first, then
    fuse this scope's Load runs. Threads ``top`` through so
    constant-input filtering can resolve against the surrounding op, and
    ``aligned`` — the alignment of each enclosing strided loop's variable."""
    aligned = aligned or {}
    descended: list[Statement] = []
    for s in body:
        nested = s.nested()
        if nested:
            inner = {**aligned, s.axis.name: strided_alignment(s.start, s.step, aligned)} if isinstance(s, StridedLoop) else aligned
            descended.append(s.with_bodies(tuple(_vectorize_body(top, b, inner) for b in nested)))
        else:
            descended.append(s)

    descended = [s.map_expressions(lambda e: _fold_run_offset(e, aligned)) if isinstance(s, Load) else s for s in descended]
    # No literal-constant loads (those render as embedded scalar floats), none without the stamped dtype that picks the
    # vector type (``030_stamp_types``), and no scalar-constant input: it is inlined at CUDA lowering, so the kernel
    # takes no such buffer to reinterpret.
    atoms: dict[Expression, Var] = {}
    runs: dict[tuple, dict[int, Address]] = {}
    for position, stmt in enumerate(descended):
        if not isinstance(stmt, Load) or not stmt.is_scalar or stmt.carried or stmt.input is None or stmt.dtype is None:
            continue
        tensor = top.inputs.get(stmt.input)
        if tensor is not None and tensor.constant and tensor.value is not None:
            continue
        if (address := flat_address(stmt.index, tensor, atoms)) is not None:
            runs.setdefault((stmt.input, stmt.dtype), {})[position] = address
    scope = Siblings(descended) if any(len(addresses) > 1 for addresses in runs.values()) else None
    changed = False
    for (name, dtype), addresses in runs.items():
        if len(addresses) > 1:
            widths = [n for n in (16, 8, 4, 2) if _TARGET.vector_type(dtype.name, n) is not None]
            changed |= fuse_runs(
                scope,
                addresses,
                widths,
                aligned,
                lambda members, name=name, dtype=dtype: Load(
                    names=tuple(scope[m].name for m in members), input=name, index=scope[members[0]].index, dtype=dtype
                ),
            )
    vectorized = Body(tuple(scope.ordered() if changed else descended))
    return body if vectorized == body else vectorized


def _fold_run_offset(expression: Expression, aligned: dict[str, int]) -> Expression:
    """``(Y + r) / d`` as ``Y / d`` where ``Y`` is a non-negative multiple of ``g`` (its variables' coefficients
    times their known alignment), ``g`` divides ``d`` and ``0 < r < g``: the elements of a lane's run then read
    a block scale at one index, the run's own."""
    if not isinstance(expression, BinaryExpression):
        return expression
    left, right = _fold_run_offset(expression.left, aligned), _fold_run_offset(expression.right, aligned)
    if expression.op == "/" and isinstance(right, Literal) and isinstance(right.value, int) and right.value > 0:
        form = affine_form(left, left.free_vars())
        anchor = form[0].simplify(SimplifyCtx.empty()) if form is not None else None
        if isinstance(anchor, Literal) and isinstance(anchor.value, int) and anchor.value >= 0 and form[1]:
            terms = [c * aligned.get(v, 1) for v, c in form[1].items()]
            if all(t > 0 for t in terms):
                offset = anchor.value % gcd(right.value, *terms)
                if offset:
                    left = _affine(form[1], anchor.value - offset)
    return expression if (left, right) == (expression.left, expression.right) else BinaryExpression(expression.op, left, right)


def _affine(coefficients: dict[str, int], constant: int) -> Expression:
    """``Σ c·v + constant`` spelled the way the run's first element spells it."""
    terms = [Var(v) if c == 1 else BinaryExpression("*", Var(v), Literal(c, "int")) for v, c in coefficients.items()]
    expression = terms[0]
    for term in terms[1:]:
        expression = BinaryExpression("+", expression, term)
    return BinaryExpression("+", expression, Literal(constant, "int")) if constant else expression
