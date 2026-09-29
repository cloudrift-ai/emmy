"""Store an mma GEMM's output tile through shared memory.

Each lane of an ``mma.sync`` warp owns two adjacent columns of eight rows per fragment, so the
plain epilogue is a burst of 4-byte global stores: 64 per thread for a 64×64 warp tile, issued by
every warp at once when the K-loop ends. On the A100 that burst throttles the load/store queue and
took a quarter of a one-wave 256×128 GEMM (39.1 µs against cuBLAS's 31.5 µs, which stores through
shared memory). Here the tile's fragments land in shared memory instead, over the operand slabs the
finished K-loop no longer reads, and the CTA then writes the tile out in 16-byte rows
(:class:`SmemTileStore`).

What is rewritten, judged structurally on the kernel: a 2-D contraction ``Tile`` whose body ends
in unguarded, non-atomic ``m16n8k16`` :class:`RegStore`\\ s of one 2-byte output with a contiguous
row, a tile at least 64 columns wide (the 128-byte swizzle that keeps both the fragment writes and
the row reads free of bank conflicts), an operand slab big enough to hold it, a cp.async or
synchronous fill (a TMA ring's tail copies land on an mbarrier this pass does not wait for), and no
fused epilogue reading shared memory. Anything else keeps its direct stores.

Ordering: every cp.async copy drains and the CTA crosses a barrier before the first fragment write
(the ring's tail copies still target the slabs, and other warps may still read the last chunk), and
a second barrier separates the fragment writes from the row reads.

A perf transform: the stored values, their rounding and their fused epilogue are the ``RegStore``\\ s'
own, so the output is bit-identical. ``EMMY_STORE_THROUGH_SMEM=0`` keeps the direct stores.
"""

from __future__ import annotations

import itertools
from dataclasses import replace

from emmy.compiler.backend.cuda.dtype import cuda_name
from emmy.compiler.graph import Node
from emmy.compiler.ir.expr import BinaryExpr, Literal, SimplifyCtx
from emmy.compiler.ir.kernel import KernelOp, Tile
from emmy.compiler.ir.kernel.ir import CpAsyncCopy, CpAsyncWait, MbarrierWait, RegStore, Smem, SmemTileStore, Sync, TmaLoad
from emmy.compiler.ir.stmt import Body
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.search.space import STORE_THROUGH_SMEM

PATTERN = [Pattern("root", KernelOp)]

#: The staged tile's name — one per kernel.
_TILE = "_c_smem"
_ROWS, _COLS = 16, 8  # an m16n8 C fragment


def rewrite(root: Node) -> KernelOp | None:
    op: KernelOp = root.op
    if STORE_THROUGH_SMEM.name in op.knobs:
        raise RuleSkipped("STORE_THROUGH_SMEM already decided (idempotence via knob)")
    if not STORE_THROUGH_SMEM.narrow((True,))[0]:
        return replace(op, knobs={**op.knobs, STORE_THROUGH_SMEM.name: False})
    body = Body(tuple(_through_smem(op, s) if isinstance(s, Tile) else s for s in op.body))
    return replace(op, body=body, knobs={**op.knobs, STORE_THROUGH_SMEM.name: True})


def _simplify(e):
    return e.simplify(SimplifyCtx.empty())


def _through_smem(op: KernelOp, tile: Tile) -> Tile:
    """``tile`` with its trailing fragment stores routed through shared memory, or ``tile`` itself
    when the structural conditions above do not hold."""
    if tile.raster_axes is None or tile.block_threads is None or tile.aux_threads:
        return tile
    stmts = list(tile.body)
    k = len(stmts)
    while k and isinstance(stmts[k - 1], RegStore):
        k -= 1
    stores: list[RegStore] = stmts[k:]
    if not stores or any(isinstance(s, (TmaLoad, MbarrierWait)) for s in tile.body.iter()):
        return tile
    dst = stores[0].dst_buffer
    out = op.outputs.get(dst)
    smem = op.smem_buffers
    ldm = _static(out.shape[-1]) if out is not None and len(out.shape) == 2 else None
    if ldm is None or out.dtype.nbytes != 2:
        return tile
    for s in stores:
        if (
            s.dst_buffer != dst
            or s.atomic
            or s.m_guard is not None
            or s.n_guard is not None
            or s.swizzle != "NONE"
            or s.fragment_layout != "m16n8k16"
            or s.volta_interleaved
            or len(s.dst_index) != 2
            or s.ldm not in (0, ldm)
            or s.ldn not in (0, 1)
            or any(name in smem for name in s.external_reads())
        ):
            return tile
    # Each store's (row, col) splits into the CTA's block origin and the fragment's place in the
    # tile: the local part is the index at block origin zero.
    blocks = dict.fromkeys(tile.raster_axes, Literal(0, "int"))
    local = [tuple(_simplify(e.substitute(blocks)) for e in s.dst_index) for s in stores]
    threads = {a.name: Literal(0, "int") for a in tile.axes if a.name not in blocks}
    at_zero = [_simplify(lo.substitute(threads)) for lo in local[0]]
    if not all(isinstance(z, Literal) for z in at_zero):
        return tile
    base = tuple(
        _simplify(BinaryExpr("-", e.substitute(threads), z)) for e, z in zip(stores[0].dst_index, at_zero, strict=True)
    )
    if any(e.free_vars() - set(blocks) for e in base):
        return tile
    # The split must hold for every store at any coordinate: index == base + local.
    probe = {a.name: 3 for a in tile.axes}
    for s, loc in zip(stores, local, strict=True):
        for e, lo, b in zip(s.dst_index, loc, base, strict=True):
            if e.eval(probe) != b.eval(probe) + lo.eval(probe):
                return tile
    # The tile's extent: every fragment origin over every warp coordinate.
    extents = {a.name: _static(a.extent) for a in tile.axes}
    free = sorted({v for loc in local for e in loc for v in e.free_vars()})
    if any(extents.get(v) is None for v in free):
        return tile
    ranges = [range(extents[v]) for v in free]
    cells = set()
    for point in itertools.product(*ranges):
        env = dict(zip(free, point, strict=True))
        for r, c in local:
            cells.add((r.eval(env), c.eval(env)))
    rows = max(r for r, _ in cells) + _ROWS
    cols = max(c for _, c in cells) + _COLS
    if min(r for r, _ in cells) or min(c for _, c in cells) or len(cells) * _ROWS * _COLS != rows * cols:
        return tile
    if cols % 64 or cols & (cols - 1) or ldm % 8 or base[1].eval(probe) % 8:
        return tile
    need = rows * cols * out.dtype.nbytes
    over = max((s for s in smem.values() if _nbytes(s) >= need), key=_nbytes, default=None)
    if over is None:
        return tile
    swizzle = "B128" if cols == 64 else f"B128@{cols.bit_length() - 1}"
    drain = [CpAsyncWait(group=0)] if any(isinstance(s, CpAsyncCopy) for s in tile.body.iter()) else []
    staged = [
        replace(s, dst_buffer=_TILE, dst_index=loc, ldm=0, ldn=0, row_dim=0, col_dim=1, swizzle=swizzle)
        for s, loc in zip(stores, local, strict=True)
    ]
    body = [
        *stmts[:k],
        *drain,
        Sync(),
        Smem(name=_TILE, extents=(rows, cols), dtype=cuda_name(out.dtype), over=over.name),
        *staged,
        Sync(),
        SmemTileStore(src=_TILE, dst=dst, base=base, rows=rows, cols=cols, ldm=ldm, threads=tile.block_threads, swizzle=swizzle),
    ]
    return replace(tile, body=Body(tuple(body)))


def _static(d) -> int | None:
    """``d`` as a static int, or ``None`` for a symbolic extent."""
    if isinstance(d, int):
        return d
    return d.as_static() if d.is_static else None


def _nbytes(s: Smem) -> int:
    from math import prod  # noqa: PLC0415

    from emmy.compiler.backend.cuda.dtype import nbytes_of  # noqa: PLC0415

    return prod(int(e) for e in s.extents) * nbytes_of(s.dtype)
