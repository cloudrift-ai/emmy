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
from emmy.compiler.ir.expression import Expression, Var
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Body, Statement, Write
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.kernel._siblings import Siblings
from emmy.compiler.pipeline.passes.lowering.kernel._vector import Address, flat_address, fuse_runs
from emmy.compiler.pipeline.search.space import VECTORIZE_STORES
from emmy.compiler.target import vector_widths

PATTERN = [Pattern("root", KernelOp)]


def rewrite(root: Node) -> KernelOp | None:
    top: KernelOp = root.op
    # Idempotence + override via the recorded VECTORIZE_STORES policy knob — symmetric to
    # 050_vectorize_loads: only ``True`` is enumerated, ``EMMY_VECTORIZE_STORES=0`` pins ``False``.
    if VECTORIZE_STORES.name in top.knobs:
        raise RuleSkipped("VECTORIZE_STORES already decided (idempotence via knob)")
    if not VECTORIZE_STORES.narrow((True,))[0]:
        return replace(top, body=top.body, knobs={**top.knobs, VECTORIZE_STORES.name: False})
    new_body = _vectorize_body(top, top.body)
    return replace(top, body=new_body, knobs={**top.knobs, VECTORIZE_STORES.name: True})


def _buf_dtype(top: KernelOp, name: str) -> str:
    """Resolve a destination buffer's canonical dtype. Writes target either a
    kernel output or, in rare cases, a kernel input (when an optimization
    folds a copy). Falls back to f32 when not found."""
    t = top.outputs.get(name) or top.inputs.get(name)
    if t is not None:
        return t.dtype.name
    return "f32"


def _vectorize_body(top: KernelOp, body: Body) -> Body:
    """Post-order body transform: recurse into nested bodies first, then
    fuse this scope's Write runs. Threads ``top`` through so
    ``_buf_dtype`` can resolve per-buffer dtypes against the same op."""
    descended: list[Statement] = []
    for s in body:
        nested = s.nested()
        if nested:
            descended.append(s.with_bodies(tuple(_vectorize_body(top, b) for b in nested)))
        else:
            descended.append(s)

    # Already-widened Writes never re-merge, and atomic reduce-writes never vectorize (each contributing lane needs
    # its own ``atomicAdd``). A swizzled-slab run merges only under one mode (the alignment proof then keeps the
    # n-aligned run inside one 16-byte swizzle chunk, where the XOR passes bits 0..2 through).
    atoms: dict[Expression, Var] = {}
    runs: dict[tuple, dict[int, Address]] = {}
    for position, stmt in enumerate(descended):
        if isinstance(stmt, Write) and not stmt.is_vector and not stmt.atomic:
            tensor = top.outputs.get(stmt.output) or top.inputs.get(stmt.output)
            if (address := flat_address(stmt.index, tensor, atoms)) is not None:
                runs.setdefault((stmt.output, stmt.swizzle, stmt.value_dtype), {})[position] = address
    scope = Siblings(descended) if any(len(addresses) > 1 for addresses in runs.values()) else None
    changed = False
    for (output, swizzle, value_dtype), addresses in runs.items():
        if len(addresses) > 1:
            widths = [n for n in sorted(vector_widths(_buf_dtype(top, output)), reverse=True) if n <= 8]
            changed |= fuse_runs(
                scope,
                addresses,
                widths,
                {},
                lambda members, output=output, swizzle=swizzle, value_dtype=value_dtype: Write(
                    output=output,
                    index=scope[members[0]].index,
                    values=tuple(scope[m].value for m in members),
                    value_dtype=value_dtype,
                    swizzle=swizzle,
                ),
            )
    vectorized = Body(tuple(scope.ordered() if changed else descended))
    return body if vectorized == body else vectorized
