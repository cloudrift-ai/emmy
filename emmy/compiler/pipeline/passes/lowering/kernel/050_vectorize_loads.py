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

from emmy.compiler.graph import Node
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.kernel._vector import vectorize_loads
from emmy.compiler.pipeline.search.space import VECTORIZE_LOADS

PATTERN = [Pattern("root", KernelOp)]


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
    new_body = vectorize_loads(top, top.body)
    return replace(top, body=new_body, knobs={**top.knobs, VECTORIZE_LOADS.name: True})
