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

1. Walk the stmts. At each position, try widths 8 then 4 then 2.
2. If ``[body[i], ..., body[i+n-1]]`` are all scalar ``Write``s to the same
   output buffer, matching outer indices, and last-dim indices that affinely
   decompose to ``anchor, anchor+1, ..., anchor+n-1`` (same coefficients on
   free vars), AND the target supports ``vector_type(elem_dtype, n)`` for the
   destination-buffer dtype, replace the run with one widened ``Write``.
   Contiguity and alignment are proved on the complete flat address, including
   outer row strides and recomposed div/mod coordinates.
3. Otherwise advance one stmt.

The destination-buffer dtype comes from the op's ``outputs`` / ``inputs``
(matcher-populated graph Tensors).

NOTE: atomic reduce-writes must NOT vectorize (each lane needs its own
``atomicAdd``) — the atomic guard below skips them.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from emmy.compiler.backend.cuda.render_target import CudaRenderTarget
from emmy.compiler.graph import Node
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Body, Stmt, Write
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.kernel._vector import vector_run
from emmy.compiler.pipeline.search.space import VECTORIZE_STORES

PATTERN = [Pattern("root", KernelOp)]

_TARGET = CudaRenderTarget()


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
    scan this scope for consecutive-Write runs. Threads ``top`` through so
    ``_buf_dtype`` can resolve per-buffer dtypes against the same op."""
    descended: list[Stmt] = []
    for s in body:
        nested = s.nested()
        if nested:
            descended.append(s.with_bodies(tuple(_vectorize_body(top, b) for b in nested)))
        else:
            descended.append(s)

    # Multi-output strips interleave stores by cell. A write-only run may group
    # independent buffers while retaining the order of writes to each buffer.
    grouped: list[Stmt] = []
    pending: dict[str, list[Write]] = {}
    for s in descended:
        if isinstance(s, Write) and not s.atomic:
            pending.setdefault(s.output, []).append(s)
        else:
            grouped.extend(store for stores in pending.values() for store in stores)
            pending.clear()
            grouped.append(s)
    grouped.extend(store for stores in pending.values() for store in stores)
    descended = grouped

    out: list[Stmt] = []
    i = 0
    while i < len(descended):
        replaced = False
        for run_n in (8, 4, 2):
            vec = _try_vec_store(descended, i, run_n, top)
            if vec is not None:
                out.append(vec)
                i += run_n
                replaced = True
                break
        if not replaced:
            out.append(descended[i])
            i += 1
    vectorized = Body(tuple(out))
    return body if vectorized == body else vectorized


def _try_vec_store(stmts: Iterable[Stmt], start: int, n: int, top: KernelOp) -> Write | None:
    """If ``stmts[start:start+n]`` matches the consecutive-Write pattern
    and the target supports ``vector_type(elem_dtype, n)`` for the
    destination buffer's dtype, return the widened :class:`Write`.
    Otherwise return ``None``."""
    stmts_list = list(stmts)
    if start + n > len(stmts_list):
        return None
    writes = stmts_list[start : start + n]
    if not all(isinstance(s, Write) for s in writes):
        return None
    # Already-widened Writes in the run aren't safe to re-merge — bail. Atomic reduce-writes
    # never vectorize (each contributing lane needs its own ``atomicAdd``). A swizzled-slab
    # run may merge only with a UNIFORM mode (the alignment proof below then keeps the
    # n-aligned run inside one 16-byte swizzle chunk, where the XOR passes bits 0..2 through).
    if any(s.is_vector or s.atomic for s in writes) or len({s.swizzle for s in writes}) != 1:
        return None

    outputs = {s.output for s in writes}
    if len(outputs) != 1:
        return None
    (output_name,) = outputs

    dst_dt = _buf_dtype(top, output_name)
    if _TARGET.vector_type(dst_dt, n) is None:
        return None

    tensor = top.outputs.get(output_name) or top.inputs.get(output_name)
    if not vector_run([write.index for write in writes], tensor, n):
        return None

    return Write(
        output=output_name,
        index=writes[0].index,
        values=tuple(s.value for s in writes),
        value_dtype=writes[0].value_dtype,
        swizzle=writes[0].swizzle,
    )
