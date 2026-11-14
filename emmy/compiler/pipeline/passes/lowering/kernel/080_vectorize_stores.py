"""Widen runs of consecutive scalar ``Write`` Stmts into one vector ``Write``.

Symmetric to ``050_vectorize_loads``. Until this pass, the body of every
materialized kernel carries scalar ``Write`` Stmts. Some sequences have a
"vector" shape: N consecutive Writes to the same output buffer whose last-dim
indices differ by 0, 1, ..., N-1. The CUDA backend can emit those as one
``make_<vec_type>(...)`` + one
``*reinterpret_cast<<vec_type>*>(&buf[base]) = packed;`` transaction.

## What the pass does

For each ``Body`` (every nested Tile / Loop / StridedLoop / Cond body,
post-order):

1. Take the scalar ``Write``s of each output buffer and their complete flat addresses, outer row strides and
   recomposed div/mod coordinates included.
2. Try widths 8 then 4 then 2: ``n`` writes whose addresses form ``anchor, anchor+1, ..., anchor+n-1`` with an
   aligned anchor, which the scope's ordering constraints let stand together (``Siblings``: no read of the buffer
   between two of them). If the target also supports ``vector_type(elem_dtype, n)`` for the destination-buffer
   dtype, the writes become one widened ``Write``.

The destination-buffer dtype comes from the op's ``outputs`` / ``inputs``
(matcher-populated graph Tensors).

NOTE: atomic reduce-writes must NOT vectorize (each lane needs its own
``atomicAdd``) — the atomic guard below skips them.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.kernel._vector import vectorize_stores
from emmy.compiler.pipeline.search.space import VECTORIZE_STORES

PATTERN = [Pattern("root", KernelOp)]


def rewrite(root: Node) -> KernelOp | None:
    top: KernelOp = root.op
    # Idempotence + override via the recorded VECTORIZE_STORES policy knob — symmetric to
    # 050_vectorize_loads: only ``True`` is enumerated, ``EMMY_VECTORIZE_STORES=0`` pins ``False``.
    if VECTORIZE_STORES.name in top.knobs:
        raise RuleSkipped("VECTORIZE_STORES already decided (idempotence via knob)")
    if not VECTORIZE_STORES.narrow((True,))[0]:
        return replace(top, body=top.body, knobs={**top.knobs, VECTORIZE_STORES.name: False})
    new_body = vectorize_stores(top, top.body)
    return replace(top, body=new_body, knobs={**top.knobs, VECTORIZE_STORES.name: True})
