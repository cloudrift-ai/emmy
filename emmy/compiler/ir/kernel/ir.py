"""Kernel IR — the fully-scheduled kernel form, just above CUDA source.

Kernel IR sits between Tile IR (schedule decisions as structural Stmts)
and CUDA source (text). Its body contains the explicit hardware
machinery: ``Tile`` (thread/block coord bindings), ``Smem`` (``__shared__``
arrays), ``Sync`` (barriers), the transports into shared memory
(``CpAsync*``, ``Tma*`` + ``Mbarrier*``), the cross-thread combines
(``WarpShuffle``, ``TreeHalve``) and the tensor-core register fragments
(``RegFragment`` … ``RegStore``, ``Wgmma*``).

Pipeline shape::

    Tile IR ──materialize_tile──▶ Kernel IR
                    ──render_kernelop──▶ CUDA source

**Leaf compute reuses the shared statements directly**. ``Load`` / ``Assign`` /
``Let`` / ``Select`` / ``Write`` / ``Accum`` / ``Cond`` / ``Loop`` come straight
from ``ir.stmt`` — buffer names are strings, so they render as they are.

Kernel IR deliberately contains no scheduling decisions — those live in
Tile IR and are materialized away before reaching this layer. A
``KernelOp`` is what the CUDA backend turns into a ``RawKernel`` launch.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import cached_property

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32, INTEGER_NAMES, DataType
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.body_op import BodyOp
from emmy.compiler.ir.elementwise import _REDUCE_SPELLING, ElementwiseImpl
from emmy.compiler.ir.expr import (
    BinaryExpr,
    Builtin,
    CastExpr,
    Expr,
    FuncCallExpr,
    Literal,
    TernaryExpr,
    Var,
)
from emmy.compiler.ir.render import (
    RenderCtx,
    _extent_c,
    _pad,
    op_to_expr,
    render_address,
    render_body,
    render_index,
    render_merge_program,
    render_stmt,
)
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import (
    Accum,
    Assign,
    Body,
    Cond,
    Declared,
    Load,
    Loop,
    Select,
    SelectBranch,
    Stmt,
    StridedLoop,
    Write,
    pretty_body,
    role,
)

# The widest iteration space a 32-bit flat thread id can address — past this the
# ``Tile`` decode must widen to 64-bit (see ``Tile``'s render).
_INT32_MAX = 2**31 - 1


def _stride_c(axes, ctx: RenderCtx, *, wide: bool = False) -> str:
    """Build a C expression for the product of axis extents — used as the
    stride divisor in the grid-axis decode. ``1`` if ``axes`` is empty.
    ``wide`` prefixes the FIRST factor with a ``(long long)`` cast so the
    whole product evaluates in 64-bit — a symbolic-grid divisor like
    ``seq_len * 6144 * 256`` overflows int32 while being *computed*, before
    any promotion against a 64-bit dividend can help."""
    if not axes:
        return "1"
    factors = [_extent_c(a, ctx) for a in axes]
    if wide:
        factors[0] = f"(long long){factors[0]}"
    return " * ".join(factors)


def _render_grid_axis_decode(axes: tuple[Axis, ...], idx_expr: str, ctx: RenderCtx, *, wide: bool = False) -> list[str]:
    """Decode ``idx_expr`` (``blockIdx.x`` or ``threadIdx.x``) into per-axis ints.
    ``wide`` keeps every stride divisor in 64-bit (see :func:`_stride_c`); the decoded
    axis vars stay ``int`` — each is bounded by its extent."""
    pad = _pad(ctx.indent)
    if not axes:
        return []
    if len(axes) == 1:
        return [f"{pad}int {axes[0].name} = {idx_expr};"]
    decoded: list[str] = []
    stride_axes: list[Axis] = []  # accumulated INNER axes (per-step product is their extents)
    for ax in reversed(axes):
        extent = _extent_c(ax, ctx)
        if not stride_axes:
            decoded.append(f"int {ax.name} = {idx_expr} % {extent};")
        else:
            decoded.append(f"int {ax.name} = ({idx_expr} / ({_stride_c(stride_axes, ctx, wide=wide)})) % {extent};")
        stride_axes.append(ax)
    outer = axes[0]
    outer_stride = _stride_c(list(axes[1:]), ctx, wide=wide)
    decoded[-1] = f"int {outer.name} = {idx_expr} / ({outer_stride});"
    return [pad + line for line in reversed(decoded)]


# ---------------------------------------------------------------------------
# Hardware primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Smem(Declared):
    """Declare a per-block ``__shared__`` array.

    Renders to ``__shared__ [__align__(N)] <dtype> <name>[<prod(extents)>];``.
    ``extents`` is the multi-dim shape used to flatten ``Load`` / ``Write``
    indices against this buffer. smem_bytes for ``CudaOp`` is computed by
    walking the KernelOp body and summing ``prod(extents) * sizeof(dtype)``
    across distinct ``Smem`` declarations.

    ``align`` is an optional explicit byte alignment. TMA destinations
    (``cp.async.bulk.tensor``) require 16-byte aligned smem; the
    materializer sets ``align=16`` on TMA-target slabs. Default 0 means
    "no explicit alignment" — falls back to the dtype's natural alignment.
    """

    name: str = role("local_buffer")
    extents: tuple[int, ...]
    dtype: str = "float"
    align: int = 0

    def pretty(self, indent: str = "") -> list[str]:
        ext = ", ".join(str(e) for e in self.extents) or "-"
        ali = f" align={self.align}" if self.align else ""
        return [f"{indent}Smem {self.dtype} {self.name}[{ext}]{ali}"]


@render_stmt.register
def _(stmt: Smem, ctx: RenderCtx) -> list[str]:
    """``__shared__ <dtype> <name>[<prod(extents)>];`` and register the
    buffer's shape so subsequent ``Load``/``Write`` flatten correctly.

    When ``ctx.smem_dynamic_offsets`` carries this buffer's offset,
    emit a pointer alias into the shared ``_smem_pool`` (extern
    dynamic smem) instead of a static ``__shared__`` array — the
    renderer falls back to the dynamic path when total per-CTA smem
    would exceed the 48 KB static cap.
    """
    from emmy.compiler.backend.cuda.dtype import canonical_from_cuda_name  # noqa: PLC0415

    total = 1
    for e in stmt.extents:
        total *= int(e)
    ctx.shapes[stmt.name] = tuple(int(e) for e in stmt.extents)
    # Register the smem buffer's canonical dtype so Load/Write
    # against this name pick the right local C type. Mbarrier slabs
    # (``"unsigned long long"``) map to None and stay out of the
    # dtype-aware path.
    smem_canonical = canonical_from_cuda_name(stmt.dtype)
    if smem_canonical is not None:
        ctx.buffer_dtypes[stmt.name] = smem_canonical
    if stmt.name in ctx.smem_dynamic_offsets:
        offset = ctx.smem_dynamic_offsets[stmt.name]
        return [f"{_pad(ctx.indent)}{stmt.dtype}* {stmt.name} = reinterpret_cast<{stmt.dtype}*>(_smem_pool + {offset});"]
    ali = f"__align__({stmt.align}) " if stmt.align else ""
    return [f"{_pad(ctx.indent)}__shared__ {ali}{stmt.dtype} {stmt.name}[{total}];"]


@dataclass(frozen=True)
class Sync(Declared):
    """Thread-group barrier.

    ``barrier_id == 0`` (default): ``__syncthreads();`` — CTA-wide.

    ``warp=True``: ``__syncwarp();`` — warp-scope only. For ordering a warp's own
    smem writes before its own reads of a WARP-PRIVATE buffer (the flash C→A
    handoff slab): post-Volta independent thread scheduling means even
    intra-warp write→read needs the sync, but a CTA-wide barrier there convoys
    the block's warps for no ordering benefit.

    ``barrier_id > 0`` and ``count`` set: ``bar.sync <id>, <count>;`` —
    named-barrier synchronizing exactly ``count`` threads on barrier id
    ``<id>`` (one of 1..15). Used inside warp-specialized branches where
    only a subset of CTA threads execute the sync — `__syncthreads()`
    would be CUDA UB because the enclosing ``if (warp < P)`` condition
    diverges across warps. The count must equal the number of threads
    that actually arrive at this barrier in flight."""

    barrier_id: int = 0
    count: int | None = None
    warp: bool = False  # warp-scope __syncwarp() (barrier_id must stay 0)

    def pretty(self, indent: str = "") -> list[str]:
        if self.warp:
            return [f"{indent}Sync(warp)"]
        if self.barrier_id == 0:
            return [f"{indent}Sync"]
        return [f"{indent}Sync(bar={self.barrier_id}, count={self.count})"]


@render_stmt.register
def _(stmt: Sync, ctx: RenderCtx) -> list[str]:
    pad = _pad(ctx.indent)
    if stmt.warp:
        if stmt.barrier_id != 0:
            raise ValueError("Sync(warp=True) cannot carry a named barrier_id")
        return [f"{pad}__syncwarp();"]
    if stmt.barrier_id == 0:
        return [f"{pad}__syncthreads();"]
    if stmt.count is None:
        raise ValueError(f"Sync(barrier_id={stmt.barrier_id}) requires count")
    return [f'{pad}asm volatile("bar.sync {stmt.barrier_id}, {stmt.count};\\n" ::: "memory");']


@dataclass(frozen=True)
class Tile(Declared):
    """Tile coordinate binding — one GPU thread per output element.

    The kernel-IR realization of a :class:`~emmy.compiler.ir.tile.TileOp`'s
    thread schedule (geometry only — the combine, if any, lives in the wrapped
    ``body``): it maps the kernel's iteration space (the product of its ``axes``
    extents — the tile's ``grid_axes``) onto a 1-D linear thread grid. Each
    thread derives a global linear id
    ``_gid = blockIdx.x * blockDim.x + threadIdx.x``, guards ``_gid < N``
    (``N`` = ∏ extents), decodes its per-axis indices from ``_gid``, and
    runs the ``body`` once. Index motion (broadcast, transpose, gather) is
    already encoded in the body's ``Load`` index Exprs, so the decode only
    has to bind the axis induction variables.

    Renders to::

        int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (_gid < N) {
            int a0 = _gid / s0;          // outermost (s0 = ∏ inner extents)
            int a1 = _gid / s1 % e1;
            ...
            <body>
        }

    The cuda lowering reads ``N`` to size the launch grid
    (``ceil(N / blockDim)`` CTAs of ``blockDim`` threads). Static extents only
    for now — a symbolic axis raises (the runtime-arg / masked-tile forms land
    as the skeleton grows).

    ``block_threads`` is the per-CTA thread count for a **cooperative** tile —
    the materializer sets it to ``coop · ∏block-cells`` so the cuda lowering
    derives ``blockDim = block_threads`` and ``gridDim = N / block_threads``
    (the linear ``_gid`` decode then groups ``block_threads`` consecutive cells
    per CTA, the innermost being the cooperative lanes). ``None`` is the scalar
    tier (one thread per cell, ``blockDim = _BLOCK_SIZE``).

    ``aux_threads`` is the warp-specialized producer band appended PAST the
    cooperative cells (``blockDim = block_threads + aux_threads``, the grid
    still ``N / block_threads``). The decode becomes ``_gid = blockIdx.x ·
    block_threads + threadIdx.x % block_threads``: a compute thread decodes its
    own cell exactly as before, while an aux thread wraps onto the CTA's first
    cells — its BLOCK-axis vars (functions of ``blockIdx`` alone) are correct
    (the producer fill needs the CTA tile origin), its unit/lane vars alias a
    compute cell and are never read (the producer branch guards on the raw
    ``threadIdx.x``, and the store side is guarded to compute threads).
    """

    axes: tuple[Axis, ...] = role("axis")
    body: Body = role("body")
    block_threads: int | None = None
    aux_threads: int = 0
    # The ``(m, n)`` BLOCK axis names of a 2-D-tiled contraction output — the structural
    # eligibility for CTA rasterization, stamped by ``grid_tile`` (only there is the m/n
    # identity known). ``None`` = not a 2-D contraction grid (reduce tier, 1-D output).
    raster_axes: tuple[str, str] | None = role("axis_name", default=None)
    # The resolved ``RASTER`` codec (grid_tile, off the TileOp's stamped knobs): the flat CTA
    # order iterates ``raster_group`` block-tiles of the ``raster_orient`` axis fastest within
    # each launch stripe, so consecutive CTAs share the same streamed operand slab and its
    # repeat reads hit L2 instead of DRAM (``m`` groups M-tiles — B streamed; ``n`` the
    # transpose — A streamed). ``None`` = the plain N-fastest row-major order.
    raster_group: int | None = None
    raster_orient: str = "m"
    # A single cell whose free loops stride over the CTA's threads (``lowering/kernel``'s spread): every thread
    # runs the body, so the one-cell guard goes.
    spread: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.body, Body):
            object.__setattr__(self, "body", Body(self.body))
        if self.aux_threads and self.block_threads is None:
            raise ValueError("Tile.aux_threads requires a cooperative block_threads")

    def pretty(self, indent: str = "") -> list[str]:
        names = ", ".join(a.name for a in self.axes)
        # A symbolic grid (dynamic seq_len) prints its Dim product — ``n_elements`` would raise,
        # and ``pretty`` feeds ``structural_key`` / the op-inventory persist, so it must stay total.
        n = tile_n_elements(self) if tile_is_static(self) else tile_n_dim(self)
        return [f"{indent}Tile[{names}] (N={n})", *pretty_body(self.body, indent + "    ")]


def tile_extents(tile: Tile) -> tuple[int, ...]:
    """Static extent of each axis (raises on a symbolic extent)."""
    out: list[int] = []
    for a in tile.axes:
        if not a.extent.is_static:
            raise NotImplementedError(f"Tile: symbolic axis {a.name!r} not supported yet")
        out.append(a.extent.as_static())
    return tuple(out)


def tile_n_elements(tile: Tile) -> int:
    """Total iteration-space size — one thread per element. Static axes only (raises on a
    symbolic axis; the dynamic-grid path uses :func:`tile_n_dim` instead)."""
    from math import prod  # noqa: PLC0415

    return prod(tile_extents(tile)) if tile.axes else 1


def tile_cells(tile: Tile) -> int:
    """The per-CTA cooperative cell count — ``block_threads`` when set, else the scalar tier's
    ``_BLOCK_SIZE``. This is ``blockDim`` minus any aux band, i.e. the divisor the cuda lowering
    uses for ``gridDim = ceil(n_elements / cells)``. So ``n_elements % cells == 0`` (and no aux
    band) means the static grid covers the iteration space exactly — no partial last block, so
    the ``_gid < N`` tail guard is provably always-true and the render elides it."""
    if tile.block_threads is not None:
        return tile.block_threads
    from emmy.compiler.ir.kernel.render import _BLOCK_SIZE  # noqa: PLC0415

    return _BLOCK_SIZE


def tile_is_static(tile: Tile) -> bool:
    """True iff every grid axis has a static extent (the static launch path)."""
    return all(a.extent.is_static for a in tile.axes)


def tile_n_dim(tile: Tile):
    """Total iteration-space size as a :class:`Dim` — the ∏ of the axis extents, holding a
    symbolic factor (a dynamic ``seq_len``) when any axis is symbolic. The cuda lowering
    sizes the launch from this (``ceil(n_dim / blockDim)`` CTAs, ``seq_len`` resolved at
    launch); the renderer guards ``_gid < n_dim`` so the partial last block is masked."""
    from emmy.compiler.dim import Dim  # noqa: PLC0415

    n = Dim(1)
    for a in tile.axes:
        n = n * a.extent
    return n


def _raster_pair(tile: Tile) -> tuple[int, int] | None:
    """``(grouped, other)`` axis positions of an active grouped raster: the ``(m, n)`` block axes adjacent in
    the decode, ``m`` before ``n``; ``None`` when the order is the flat one."""
    if not (tile.raster_group and tile.raster_group > 1 and tile.raster_axes):
        return None
    names = [a.name for a in tile.axes]
    if tile.raster_axes[0] not in names or tile.raster_axes[1] not in names:
        return None
    mi, ni = names.index(tile.raster_axes[0]), names.index(tile.raster_axes[1])
    if ni != mi + 1:
        return None
    return (mi, ni) if tile.raster_orient == "m" else (ni, mi)


def _render_symbolic_decode(tile: Tile, ctx: RenderCtx) -> list[str]:
    """The flat id decoded into the axis vars of a symbolic grid, with the grouped raster when one is active:
    the static path's swizzle with the extents rendered as C expressions, so a dynamic ``seq_len`` grid takes
    the same L2-friendly launch order. The ragged last group always takes the clamp — its size is a runtime
    value."""
    decoded = _render_grid_axis_decode(tile.axes, "_gid", ctx, wide=True)
    pair = _raster_pair(tile)
    if pair is None:
        return decoded
    gi, oi = pair
    mi, ni = min(gi, oi), max(gi, oi)
    pad, g = _pad(ctx.indent), tile.raster_group
    ea, eb = _extent_c(tile.axes[gi], ctx), _extent_c(tile.axes[oi], ctx)
    inner = list(tile.axes[ni + 1 :])
    sub = f"_gid / ({_stride_c(inner, ctx, wide=True)})" if inner else "_gid"
    if mi != 0:
        sub = f"({sub}) % ((long long){ea} * {eb})"
    stripe = f"((long long){g} * {eb})"
    grouped = f"_rgrp * {g} + (_rsub % {stripe}) % _rsz"
    other = f"(_rsub % {stripe}) / _rsz"
    swizzle = {
        tile.axes[gi].name: grouped,
        tile.axes[oi].name: other,
    }
    lines = [
        f"{pad}long long _rsub = {sub};",
        f"{pad}long long _rgrp = _rsub / {stripe};",
        f"{pad}long long _rsz = min((long long){g}, (long long){ea} - _rgrp * {g});",
    ]
    for line in decoded:
        name = line.strip().removeprefix("int ").split(" = ", 1)[0]
        lines.append(f"{pad}int {name} = {swizzle[name]};" if name in swizzle else line)
    return lines


@render_stmt.register
def _(node: Tile, ctx: RenderCtx) -> list[str]:
    pad = _pad(ctx.indent)
    # Symbolic grid (a dynamic free axis): size the guard + decode from the runtime extents
    # via the symbolic-aware helpers (the ``Dim`` name renders to its ``int`` arg). Kept off
    # the static path so static codegen stays byte-identical.
    if not tile_is_static(node):
        inner = ctx.child()
        # A symbolic grid's total is unboundable at render time, so the flat id and every
        # stride divisor are ALWAYS 64-bit — a batched coop-reduce grid overflows int32 at
        # runtime-legal extents (batch·seq·N·coop = 4·512·6144·256 ≈ 3.2e9 on the demoted
        # fused gate/up kernel → cudaErrorIllegalAddress; the static branch below reaches
        # the same overflow class and widens exactly when its known total needs it). The
        # ``(long long)blockIdx.x`` lowers to one mul.wide; the decoded axis vars stay
        # ``int`` (each bounded by its extent).
        n = _stride_c(list(node.axes), inner, wide=True)
        # Same aux-band decode as the static path below: blockDim includes the producer band,
        # so the cell id must be derived from ``block_threads``, not ``blockDim``.
        sgid = (
            f"(long long)blockIdx.x * {node.block_threads} + threadIdx.x % {node.block_threads}"
            if node.aux_threads
            else "(long long)blockIdx.x * blockDim.x + threadIdx.x"
        )
        out = [
            f"{pad}long long _gid = {sgid};",
            f"{pad}if (_gid < {n}) {{",
        ]
        out.extend(_render_symbolic_decode(node, inner))
        out.extend(render_body(node.body, inner))
        out.append(f"{pad}}}")
        return out
    extents = tile_extents(node)
    # Inner-stride of each axis = product of the extents to its right.
    strides: list[int] = [1] * len(extents)
    acc = 1
    for i in range(len(extents) - 1, -1, -1):
        strides[i] = acc
        acc *= extents[i]
    n = tile_n_elements(node)
    # A flat id past INT32_MAX wraps negative in 32-bit arithmetic and every decoded index
    # goes wild (a per-cell coop-reduce grid reaches it: M·N·coop = 4096·3840·256 ≈ 4.0e9 on
    # the gemma-4 down-proj demote → cudaErrorIllegalAddress). Widen the id to 64-bit exactly
    # when the static iteration space needs it — ``(long long)blockIdx.x`` lowers to one
    # mul.wide, and the (slower) 64-bit axis divisions only land on kernels that would
    # otherwise fault. The symbolic branch above widens ALWAYS: its runtime extent is
    # unboundable at render time (batch·seq·N·coop crossed int32 at seq 512 on the merged
    # gate/up demote).
    wide = n > _INT32_MAX
    bidx = "(long long)blockIdx.x" if wide else "blockIdx.x"
    # A warp-specialized producer band (``aux_threads``) launches PAST the cooperative cells:
    # decode off ``blockIdx·block_threads + threadIdx % block_threads`` so an aux thread wraps
    # onto this CTA's cells (correct block coords; its unit/lane aliases are never read).
    gid = (
        f"{bidx} * {node.block_threads} + threadIdx.x % {node.block_threads}" if node.aux_threads else f"{bidx} * blockDim.x + threadIdx.x"
    )
    # ``gridDim = ceil(n / cells)`` CTAs cover the cells. When ``cells`` divides ``n`` exactly and
    # there is no warp-specialized aux band, every launched thread maps to a real cell — the tail
    # guard ``_gid < n`` is always-true (and nvcc can't prove it: ``gridDim`` is a launch arg), so
    # emit the decode + body flat. Otherwise the partial last block (or the aux threads that launch
    # past the cells) needs masking.
    guard = not node.spread and (node.aux_threads != 0 or n % tile_cells(node) != 0)
    out = [f"{pad}{'long long' if wide else 'int'} _gid = {gid};"]
    inner = ctx.child()
    ipad = _pad(inner.indent)
    if guard:
        out.append(f"{pad}if (_gid < {n}) {{")
    # Grouped CTA rasterization (the resolved ``RASTER`` codec): remap the flat ``(m, n)``
    # block sub-id so ``G`` block-tiles of the grouped axis iterate fastest within each
    # launch stripe. Consecutive CTAs then share the same streamed operand slab (L2 reuse)
    # instead of re-streaming it from DRAM once per row of the other axis (the flat
    # N-fastest order's failure mode). With the grouped axis ``a`` (extent Ea) and the
    # other axis ``b`` (extent Eb): ``a = grp·G + (sub % (G·Eb)) % gsz``,
    # ``b = (sub % (G·Eb)) / gsz`` — the standard threadblock swizzle, with
    # ``gsz = min(G, Ea − grp·G)`` handling the ragged last group.
    raster = None  # (grouped_idx, other_idx, G) when active
    if node.raster_group and node.raster_axes:
        names = [a.name for a in node.axes]
        if node.raster_axes[0] in names and node.raster_axes[1] in names:
            mi, ni = names.index(node.raster_axes[0]), names.index(node.raster_axes[1])
            if ni == mi + 1:
                gi, oi = (mi, ni) if node.raster_orient == "m" else (ni, mi)
                g_eff = min(node.raster_group, extents[gi])
                if extents[gi] > 1 and g_eff > 1:
                    raster = (gi, oi, g_eff)
    if raster is not None:
        gi, oi, g = raster
        ea, eb = extents[gi], extents[oi]
        mi, ni = min(gi, oi), max(gi, oi)
        sn = strides[ni]
        sub = "_gid" if sn == 1 else f"_gid / {sn}"
        # ``% (Em·En)`` drops when (m, n) are the outermost axes (the sub-id is bounded
        # by the ``_gid < N`` guard / exact grid).
        if mi != 0:
            sub = f"({sub}) % {ea * eb}"
        decls = [f"{ipad}int _rsub = {sub};"]
        if ea % g == 0:
            a_expr = f"(_rsub / {g * eb}) * {g} + _rsub % {g}"
            b_expr = f"(_rsub % {g * eb}) / {g}"
        else:
            decls.append(f"{ipad}int _rgrp = _rsub / {g * eb};")
            decls.append(f"{ipad}int _rsz = min({g}, {ea} - _rgrp * {g});")
            a_expr = f"_rgrp * {g} + (_rsub % {g * eb}) % _rsz"
            b_expr = f"(_rsub % {g * eb}) / _rsz"
    for i, a in enumerate(node.axes):
        s, e = strides[i], extents[i]
        if raster is not None and i == mi:
            out.extend(decls)
            out.append(f"{ipad}int {a.name} = {a_expr if i == gi else b_expr};")
            continue
        if raster is not None and i == ni:
            out.append(f"{ipad}int {a.name} = {a_expr if i == gi else b_expr};")
            continue
        term = "_gid" if s == 1 else f"_gid / {s}"
        # The outermost axis needs no modulo (``_gid / s0 < e0`` since _gid < N).
        expr = term if i == 0 else f"({term}) % {e}"
        out.append(f"{ipad}int {a.name} = {expr};")
    out.extend(render_body(node.body, inner))
    if guard:
        out.append(f"{pad}}}")
    return out


@dataclass(frozen=True)
class CpAsyncCopy(Declared):
    """Issue one ``cp.async.{ca,cg}.shared.global`` instruction.

    Replaces the per-thread ``Load(reg) + Write(smem)`` pair in cooperative
    loads on sm_80+. The hardware copies ``nbytes`` (one contiguous vector of
    ``nbytes / sizeof(dtype)`` elements) directly from global to shared without
    a register staging slot, freeing the thread register and removing the
    LDG → STS dependency.

    ``nbytes`` ∈ {4, 8, 16} (the cp.async copy sizes). 16-byte copies use the
    ``.cg`` (cache-global, bypass-L1) qualifier — the streaming form CUTLASS /
    cuBLAS use, requiring 16-byte-aligned smem + global addresses — and 4/8-byte
    copies use ``.ca``. ``smem_index`` / ``src_index`` address the *start* of
    the contiguous chunk; ``_stage_expand`` picks the widest legal ``nbytes`` so
    e.g. an fp16 tile stages 8 halves per ``cp.async``.

    Renders to inline PTX. The asm reads the smem address via
    ``cvta.to.shared.u32`` and the global pointer as a 64-bit value;
    indices flatten via ``render_index`` against the buffer's declared
    shape (same as ``Load`` / ``Write``)."""

    smem: str  # destination smem buffer name
    smem_index: tuple[Expr, ...] = role("expr")
    src: str = role("buffer_read")  # source global buffer name
    src_index: tuple[Expr, ...] = role("expr")
    nbytes: int = 4  # bytes per cp.async (4/8/16); 16 ⇒ .cg, else .ca
    # Software smem swizzle mode ("NONE"/"B32"/"B64"/"B128"): XOR the flattened destination
    # element index via ``emmy_swizzle_<mode>`` — the same address-based permutation the
    # ``LdmatrixLoad`` drain applies, so a swizzled slab round-trips (``LDMATRIX_SWIZZLE_XOR``).
    swizzle: str = "NONE"
    # The lane's own ``(row, col)`` within a CTA-wide copy stripe of ``lane_rows`` rows, when the
    # fill is unrolled per stripe: ``smem_index`` is then the stripe's base and the render adds the
    # lane through the swizzle split (:func:`swizzled_slab_index`) — the swizzle of the lane's part
    # computed once, the stripe base an immediate — where a loop over the flat chunk index
    # re-swizzled every copy's whole index. ``None`` addresses the whole chunk in ``smem_index``.
    lane_index: tuple[Expr, ...] | None = role("expr", default=None)
    lane_rows: int = 0
    # Whether the chunk lies inside the source, for a tile row past a masked edge: the copy then
    # writes zeros and reads nothing. The source index stays clamped in bounds. Without it the
    # clamped rows all read the edge's last row, and on the A100 a 96-row tile over 512 rows spent
    # 23 us on a 16 us GEMM contending for that one row.
    valid: Expr | None = role("expr", default=None)

    def pretty(self, indent: str = "") -> list[str]:
        smem_idx = ", ".join(str(e) for e in self.smem_index)
        src_idx = ", ".join(str(e) for e in self.src_index)
        swz = f" swz={self.swizzle}" if swizzle_xor(self.swizzle) else ""
        lane = f" + lane({', '.join(str(e) for e in self.lane_index)})" if self.lane_index is not None else ""
        return [f"{indent}cp.async[{self.nbytes}B] {self.smem}[{smem_idx}]{lane}{swz} <- {self.src}[{src_idx}]"]


@render_stmt.register
def _(node: CpAsyncCopy, ctx: RenderCtx) -> list[str]:
    smem_flat = render_index(node.smem, node.smem_index, ctx)
    src_flat = render_index(node.src, node.src_index, ctx)
    if node.lane_index is not None:
        cols = ctx.shapes[node.smem][-1]
        lane = f"({node.lane_index[0].render(ctx)}) * {cols} + ({node.lane_index[1].render(ctx)})"
        split = swizzled_slab_index(
            node.swizzle, cols, lane, node.smem_index[0], node.smem_index[1], ctx, lane_rows=node.lane_rows, lane_col_mod=cols
        )
        smem_flat = split if split is not None else f"{smem_flat} + {lane}"
        if split is None and swizzle_xor(node.swizzle):
            smem_flat = f"{swizzle_fn(node.swizzle)}({smem_flat})"
    elif swizzle_xor(node.swizzle):
        smem_flat = f"{swizzle_fn(node.swizzle)}({smem_flat})"
    pad = _pad(ctx.indent)
    # ``emmy_cp_async_{cg,ca}`` (the cp.async prelude) does the ``cvta`` internally, so this is a
    # single call — no ``_smem_addr`` local and no wrapping ``{ }`` block. .cg is 16-byte-only.
    dst, src = f"&{node.smem}[{smem_flat}]", f"&{node.src}[{src_flat}]"
    if node.valid is not None and node.nbytes == 16:
        call = f"emmy_cp_async_cg_z({dst}, {src}, {node.valid.render(ctx)})"
    else:
        call = f"emmy_cp_async_cg({dst}, {src})" if node.nbytes == 16 else f"emmy_cp_async_ca<{node.nbytes}>({dst}, {src})"
    return [f"{pad}{call};"]


@dataclass(frozen=True)
class CpAsyncCommit(Declared):
    """``cp.async.commit_group;`` — finalize the preceding cp.async copies
    issued by this thread into a commit group. Pairs with
    ``CpAsyncWait`` to wait for that group to drain."""

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}cp.async.commit_group"]


@render_stmt.register
def _(stmt: CpAsyncCommit, ctx: RenderCtx) -> list[str]:
    return [f"{_pad(ctx.indent)}emmy_cp_async_commit();"]


@dataclass(frozen=True)
class CpAsyncWait(Declared):
    """``cp.async.wait_group N;`` — block this thread until ≤ N cp.async
    groups remain in flight. ``group=0`` waits for everything (synchronous
    style); larger values stagger waits for software pipelining."""

    group: int = 0

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}cp.async.wait_group({self.group})"]


@render_stmt.register
def _(stmt: CpAsyncWait, ctx: RenderCtx) -> list[str]:
    return [f"{_pad(ctx.indent)}emmy_cp_async_wait<{stmt.group}>();"]


@dataclass(frozen=True)
class TmaDescriptor(Declared):
    """Host-built ``CUtensorMap`` descriptor for a TMA box copy.

    Declarative — renders nothing inside the kernel. The CUDA backend
    walks the ``KernelOp`` body, calls ``cuTensorMapEncodeTiled`` once
    per distinct ``TmaDescriptor`` at compile time, and binds the
    resulting 128-byte descriptor as a ``__grid_constant__`` kernel
    arg. ``name`` matches the parameter name referenced by sibling
    ``TmaLoad`` stmts. ``src_buf`` names the global buffer the descriptor
    addresses (used by the kernel-arg pipeline to add ``src_buf`` to the
    parameter list).
    """

    name: str
    src_buf: str = role("buffer_read")
    src_shape: tuple[int, ...]
    box_extents: tuple[int, ...]
    swizzle: str = "NONE"
    dtype: str = "float"

    def pretty(self, indent: str = "") -> list[str]:
        shape = ", ".join(str(e) for e in self.src_shape)
        box = ", ".join(str(e) for e in self.box_extents)
        return [f"{indent}TmaDescriptor {self.name} <- {self.src_buf}[{shape}] box=[{box}] swizzle={self.swizzle}"]


@render_stmt.register
def _(stmt: TmaDescriptor, ctx: RenderCtx) -> list[str]:
    # Descriptor is a kernel parameter built host-side; no in-kernel render.
    return []


@dataclass(frozen=True)
class TmaLoad(Declared):
    """``cp.async.bulk.tensor.<rank>d`` — single-thread box copy.

    Issued by exactly one thread of the CTA (the materializer wraps this
    in ``Cond(tid==0, [...])``). Source coords come from the staging
    Stmt's ``origin`` (CTA-uniform); destination is the dynamic-phase
    smem slab. Pairs with ``MbarrierArriveExpectTx`` (issued immediately
    before) and ``MbarrierWait`` (issued by every consumer thread)."""

    smem: str
    smem_index: tuple[Expr, ...] = role("expr")
    desc: str
    coords: tuple[Expr, ...] = role("expr")
    mbar: str
    mbar_slot: Expr | None = role("expr", default=None)

    def pretty(self, indent: str = "") -> list[str]:
        smem_idx = ", ".join(str(e) for e in self.smem_index)
        coords = ", ".join(str(e) for e in self.coords)
        s = "" if self.mbar_slot is None else f"[{self.mbar_slot}]"
        return [f"{indent}TmaLoad {self.smem}[{smem_idx}] <- {self.desc}({coords}) mbar={self.mbar}{s}"]


@render_stmt.register
def _(node: TmaLoad, ctx: RenderCtx) -> list[str]:
    smem_flat = render_index(node.smem, node.smem_index, ctx)
    rank = len(node.coords)
    # PTX expects coords in driver order (innermost-first), which is
    # the reverse of our C-order ``coords`` (== ``stage.origin``).
    coord_args = ", ".join(c.render(ctx) for c in reversed(node.coords))
    mbar_addr = "&" + node.mbar if node.mbar_slot is None else f"&{node.mbar}[{node.mbar_slot.render(ctx)}]"
    pad = _pad(ctx.indent)
    # The ``cp_async_bulk_tensor_<rank>d`` helpers are defined in the
    # kernel prelude — single-CTA ``.shared::cta`` qualifier; cluster
    # launches would need a separate helper variant.
    return [
        f"{pad}cp_async_bulk_tensor_{rank}d(&{node.smem}[{smem_flat}], {node.desc}, {coord_args}, {mbar_addr});",
    ]


@dataclass(frozen=True)
class MbarrierInit(Declared):
    """``mbarrier.init.shared.b64 [&mbar[slot]], count;`` — one-shot init.

    With ``slot=None``: initializes a scalar mbarrier (single-slot path).
    With ``slot=Expr``: initializes one element of a per-slot mbarrier
    array. The materializer emits one ``MbarrierInit`` per literal slot
    index in the buffer count, hoisted to the kernel prologue inside
    ``Cond(threadIdx.x == 0, [...]) + Sync()``."""

    mbar: str
    count: int = 1
    slot: Expr | None = role("expr", default=None)

    def pretty(self, indent: str = "") -> list[str]:
        s = "" if self.slot is None else f"[{self.slot}]"
        return [f"{indent}MbarrierInit({self.mbar}{s}, count={self.count})"]


@render_stmt.register
def _(stmt: MbarrierInit, ctx: RenderCtx) -> list[str]:
    pad = _pad(ctx.indent)
    addr = "&" + stmt.mbar if stmt.slot is None else f"&{stmt.mbar}[{stmt.slot.render(ctx)}]"
    return [f"{pad}mbarrier_init({addr}, {stmt.count});"]


@dataclass(frozen=True)
class MbarrierArriveExpectTx(Declared):
    """``mbarrier.arrive.expect_tx.shared.b64`` — declare the expected
    transaction byte count for the in-flight TMA copy on this barrier.
    With per-slot mbarrier arrays, ``slot`` selects the ring buffer slot."""

    mbar: str
    bytes_: int
    slot: Expr | None = role("expr", default=None)

    def pretty(self, indent: str = "") -> list[str]:
        s = "" if self.slot is None else f"[{self.slot}]"
        return [f"{indent}MbarrierArriveExpectTx({self.mbar}{s}, bytes={self.bytes_})"]


@render_stmt.register
def _(stmt: MbarrierArriveExpectTx, ctx: RenderCtx) -> list[str]:
    pad = _pad(ctx.indent)
    addr = "&" + stmt.mbar if stmt.slot is None else f"&{stmt.mbar}[{stmt.slot.render(ctx)}]"
    return [f"{pad}mbarrier_arrive_expect_tx({addr}, {stmt.bytes_});"]


@dataclass(frozen=True)
class MbarrierArrive(Declared):
    """``mbarrier.arrive.shared.b64`` — simple arrive (no transaction-byte
    count). Used by warp-specialized consumer warps to signal "slot empty"
    so the producer can refill it on the next ring iteration. The producer
    side uses ``MbarrierArriveExpectTx`` instead (which declares the
    incoming TMA byte count)."""

    mbar: str
    slot: Expr | None = role("expr", default=None)

    def pretty(self, indent: str = "") -> list[str]:
        s = "" if self.slot is None else f"[{self.slot}]"
        return [f"{indent}MbarrierArrive({self.mbar}{s})"]


@render_stmt.register
def _(stmt: MbarrierArrive, ctx: RenderCtx) -> list[str]:
    pad = _pad(ctx.indent)
    addr = "&" + stmt.mbar if stmt.slot is None else f"&{stmt.mbar}[{stmt.slot.render(ctx)}]"
    return [f"{pad}mbarrier_arrive({addr});"]


@dataclass(frozen=True)
class SetMaxNReg(Declared):
    """``setmaxnreg.{inc,dec}.sync.aligned.u32 N;`` — Hopper+ register-budget
    redistribution. ``direction="dec"`` shrinks the calling warp's max
    register count to ``count`` (returning registers to the pool);
    ``direction="inc"`` claims up to ``count`` registers. Used by warp-
    specialized kernels so producer warps drop registers and consumer
    warps claim them, decoupling occupancy from the consumer's pressure.

    Requires sm_90+. On older targets NVCC rejects the instruction at
    compile time — only a warp-specialized materialization emits this Stmt,
    and that path is already TMA-gated to sm_90+ anyway."""

    count: int
    direction: str  # "inc" or "dec"

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}SetMaxNReg.{self.direction}({self.count})"]


@render_stmt.register
def _(stmt: SetMaxNReg, ctx: RenderCtx) -> list[str]:
    pad = _pad(ctx.indent)
    return [f'{pad}asm volatile("setmaxnreg.{stmt.direction}.sync.aligned.u32 {stmt.count};\\n");']


@dataclass(frozen=True)
class MbarrierWait(Declared):
    """``mbarrier.try_wait.parity.shared.b64`` — block until the barrier's
    parity flips for ``phase``. Replaces ``CpAsyncWait + Sync`` for TMA
    transports — mbarrier arrival already provides CTA-wide visibility.
    ``slot`` selects the ring buffer slot when the mbarrier is an array."""

    mbar: str
    phase: Expr = role("expr")
    slot: Expr | None = role("expr", default=None)

    def pretty(self, indent: str = "") -> list[str]:
        s = "" if self.slot is None else f"[{self.slot}]"
        return [f"{indent}MbarrierWait({self.mbar}{s}, phase={self.phase})"]


@render_stmt.register
def _(stmt: MbarrierWait, ctx: RenderCtx) -> list[str]:
    pad = _pad(ctx.indent)
    addr = "&" + stmt.mbar if stmt.slot is None else f"&{stmt.mbar}[{stmt.slot.render(ctx)}]"
    phase_expr = stmt.phase.render(ctx)
    return [f"{pad}mbarrier_wait_parity({addr}, {phase_expr});"]


@dataclass(frozen=True)
class TreeHalve(Declared):
    """Cross-thread combine of a (possibly multi-component) monoid state over a
    PRE-POPULATED power-of-two smem tree — the > warp_size fallback for
    :class:`WarpShuffle`.

    Each state component has its own smem slab ``bufs[i][0..length)``, populated by
    the caller (``emit_combine`` writes the per-thread partials, or the
    hierarchical path writes one per-warp partial per slot). This stmt tree-reduces
    each slab into ``bufs[i][0]`` applying the carrier's ``combine_states`` at every
    halving step (rendered at ``dtype``), then broadcast-reassigns every
    ``state[i]`` from ``bufs[i][0]`` **in place** (no ``_b`` rename — every thread
    ends holding the full reduction in the carried SSA names). ``length`` is a power
    of two ≤ ``blockDim.x``. A scalar reduce is the 1-component case
    (``state=("acc",)``, ``combine_states=(Assign("acc", op, ("acc","acc__o")),)``).

    Named-barrier support: ``barrier_id > 0`` routes the per-iter sync to
    ``bar.sync <id>, <count>`` instead of ``__syncthreads()`` — required inside a
    warp-specialized consumer branch (a __syncthreads on a warp-divergent cond is
    UB).
    """

    bufs: tuple[str, ...]
    state: tuple[str, ...] = role("reads")
    state_b: tuple[str, ...] = role("local")
    combine_states: tuple[Assign, ...] = role("stmts")
    length: int
    tid_var: str
    dtype: DataType = F32
    barrier_id: int = 0
    barrier_count: int | None = None
    # Segment-indexed halving (the transposed ``coop-t`` combine): each slab holds ``length``
    # SEGMENTS of ``inner[1]`` slots, and the tree halves the segment index at a fixed inner
    # slot — ``buf[t·scale + inner_var]`` vs ``buf[(t+s)·scale + inner_var]``, broadcast from
    # ``buf[inner_var]``. ``None`` keeps the flat ``buf[t]`` layout.
    inner: tuple[str, int] | None = None

    def pretty(self, indent: str = "") -> list[str]:
        bar = "" if self.barrier_id == 0 else f", bar={self.barrier_id}/{self.barrier_count}"
        return [f"{indent}TreeHalve({', '.join(self.state)}, length={self.length}, tid={self.tid_var}{bar})"]


@render_stmt.register
def _(stmt: TreeHalve, ctx: RenderCtx) -> list[str]:
    pad = _pad(ctx.indent)
    in1 = _pad(ctx.indent + 1)
    in2 = _pad(ctx.indent + 2)
    t = stmt.tid_var
    ty = ctx.type_name(stmt.dtype.name)
    if stmt.barrier_id == 0:
        sync_line = f"{in1}__syncthreads();"
    else:
        if stmt.barrier_count is None:
            raise ValueError(f"TreeHalve(barrier_id={stmt.barrier_id}) requires barrier_count")
        sync_line = f'{in1}asm volatile("bar.sync {stmt.barrier_id}, {stmt.barrier_count};\\n" ::: "memory");'
    half = int(stmt.length) // 2
    if stmt.inner is not None:
        iv, scale = stmt.inner
        slot, slot_s, root = f"{t} * {scale} + {iv}", f"({t} + s) * {scale} + {iv}", str(iv)
    else:
        slot, slot_s, root = t, f"{t} + s", "0"
    # The carried names' dtypes as declared by the enclosing seed — a selecting fp16 carrier
    # (``__half`` max) keeps its declaration while the tree folds float shadows of the same
    # names. Captured before those shadows overwrite the flat ssa map, restored after the
    # broadcast so the epilogue's conversions read the live declaration, not the dead shadow.
    outer_dtypes = {st: ctx.ssa_dtypes.get(st) for st in stmt.state}
    if t == "warp" and stmt.inner is None and stmt.barrier_id == 0 and stmt.length <= 32:
        # The hierarchical cross-warp slab holds one partial per warp, so warp 0 alone folds it with a
        # register butterfly: one barrier before the broadcast instead of one per halving step.
        out = [f"{pad}if (warp == 0) {{"]
        for buf, st in zip(stmt.bufs, stmt.state, strict=True):
            out.append(f"{in1}{ty} {st} = {buf}[lane & {stmt.length - 1}];")
            ctx.ssa_dtypes[st] = stmt.dtype.name
        butterfly = WarpShuffle(
            state=stmt.state, state_b=stmt.state_b, combine_states=stmt.combine_states, length=stmt.length, dtype=stmt.dtype
        )
        out.extend(render_stmt(butterfly, ctx.child()))
        out.append(f"{in1}if (lane == 0) {{")
        out.extend(f"{in2}{buf}[0] = {st};" for buf, st in zip(stmt.bufs, stmt.state, strict=True))
        out += [f"{in1}}}", f"{pad}}}", f"{pad}__syncthreads();"]
        for buf, st in zip(stmt.bufs, stmt.state, strict=True):
            out.append(f"{pad}{st} = {buf}[0];")
            ctx.ssa_dtypes[st] = outer_dtypes[st] or stmt.dtype.name
        return out
    out: list[str] = [f"{pad}for (int s = {half}; s > 0; s >>= 1) {{", f"{in1}if ({t} < s) {{"]
    # Shadow temps named after the carried state so ``combine_states`` (which
    # reassigns ``state``) folds ``buf[t+s]`` into ``buf[t]`` per component.
    for buf, st, sb in zip(stmt.bufs, stmt.state, stmt.state_b, strict=True):
        out.append(f"{in2}{ty} {st} = {buf}[{slot}];")
        out.append(f"{in2}{ty} {sb} = {buf}[{slot_s}];")
        ctx.ssa_dtypes[st] = stmt.dtype.name
        ctx.ssa_dtypes[sb] = stmt.dtype.name
    out.extend(render_merge_program(stmt.combine_states, stmt.state, ctx, pad=in2, dtype=stmt.dtype))
    for buf, st in zip(stmt.bufs, stmt.state, strict=True):
        out.append(f"{in2}{buf}[{slot}] = {st};")
    out.append(f"{in1}}}")
    out.append(sync_line)
    out.append(f"{pad}}}")
    # Broadcast the reduced state back into the carried SSA names (in place) — from the
    # segment root (this lane's slot 0) when segment-indexed. The assignment reuses the
    # ENCLOSING declaration (implicitly narrowing for a selecting fp16 carrier), so the
    # dtype record must say what that declaration says.
    for buf, st in zip(stmt.bufs, stmt.state, strict=True):
        out.append(f"{pad}{st} = {buf}[{root}];")
        ctx.ssa_dtypes[st] = outer_dtypes[st] or stmt.dtype.name
    return out


@dataclass(frozen=True)
class WarpShuffle(Declared):
    """Cross-thread combine of a (possibly multi-component) monoid state over
    ``length`` lanes via ``__shfl_xor_sync`` — a register-only butterfly (no smem,
    no syncthreads).

    Each lane holds a full per-thread partial ``state`` tuple (a scalar reduce is
    the 1-component case ``state=("acc",)``). Each butterfly step shuffles **every**
    component down into ``state_b`` and applies the carrier's ``combine_states``
    (the state-merges-state monoid op, rendered at ``dtype``) to fold the shuffled
    neighbor into this lane's state, reassigning ``state`` **in place**. After the
    last step every lane holds the full reduction in the SAME SSA names — so the
    post-reduce epilogue reads them with no rename (kills the old scalar ``<name>_b``
    broadcast).

    ``length`` must be a power of two ≤ ``warp_size``; the XOR butterfly never
    crosses an aligned ``length``-lane group, so this is also the SEGMENTED per-row
    combine for strided-cooperative rows. ``commutative`` (required — the butterfly
    reorders) is checked at the carrier.

    The shuffle mask is ``__activemask()``, NOT a hard-coded ``0xffffffff``: a
    whole-CTA cooperative reduce with ``BR < warp_size`` launches a **partial warp**
    (e.g. ``block=8``), and naming the absent lanes ``8..31`` in the mask is undefined
    behavior — on the e2e captured-graph path it manifested as a multi-millisecond
    stall (a per-token×head RMSNorm went 28 ms instead of µs). The combine is reached
    with the warp converged (the per-CTA reduce has no data-dependent divergence and
    the only enclosing guard is warp-uniform), so ``__activemask()`` returns exactly
    the participating lanes — correct for a partial warp, a full warp, AND a
    non-warp-multiple block's partial final warp (where no single static literal mask
    is correct per warp).
    """

    state: tuple[str, ...] = role("reads")
    state_b: tuple[str, ...] = role("local")
    combine_states: tuple[Assign, ...] = role("stmts")
    length: int
    dtype: DataType = F32
    prefix: bool = False

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}WarpShuffle({', '.join(self.state)}, length={self.length}, prefix={self.prefix})"]


@render_stmt.register
def _(stmt: WarpShuffle, ctx: RenderCtx) -> list[str]:
    pad = _pad(ctx.indent)
    inner = _pad(ctx.indent + 1)
    ty = ctx.type_name(stmt.dtype.name)
    out: list[str] = []
    s = 1 if stmt.prefix else int(stmt.length) // 2
    while 0 < s < stmt.length:
        # Block-scope each step so the shuffled state_b + merge temps redeclare
        # cleanly per round (the carried state is declared by an enclosing Init).
        out.append(f"{pad}{{")
        for st, sb in zip(stmt.state, stmt.state_b, strict=True):
            exchange = (
                f"__shfl_up_sync(__activemask(), {st}, {s}, {stmt.length})"
                if stmt.prefix
                else f"__shfl_xor_sync(__activemask(), {st}, {s})"
            )
            out.append(f"{inner}{ty} {sb} = {exchange};")
            ctx.ssa_dtypes[sb] = stmt.dtype.name
        if stmt.prefix:
            out.append(f"{inner}if (threadIdx.x % {stmt.length} >= {s}) {{")
        merge_pad = _pad(ctx.indent + 2) if stmt.prefix else inner
        out.extend(render_merge_program(stmt.combine_states, stmt.state, ctx, pad=merge_pad, dtype=stmt.dtype))
        if stmt.prefix:
            out.append(f"{inner}}}")
        out.append(f"{pad}}}")
        s = s << 1 if stmt.prefix else s >> 1
    return out


@dataclass(frozen=True)
class WarpBroadcast(Declared):
    """Broadcast the last lane's state within each aligned lane group, in place."""

    state: tuple[str, ...] = role("reads")
    length: int

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}WarpBroadcast({', '.join(self.state)}, length={self.length})"]


@render_stmt.register
def _(stmt: WarpBroadcast, ctx: RenderCtx) -> list[str]:
    pad = _pad(ctx.indent)
    return [f"{pad}{name} = __shfl_sync(__activemask(), {name}, {stmt.length - 1}, {stmt.length});" for name in stmt.state]


#: The per-arg distribution kinds of a :class:`FragmentApply` operand.
FRAG = "frag"  # a C-fragment operand — indexed per element (``arg[i]``)
ROW = "row"  # a per-row scalar — broadcast by row (suffix ``0`` for rows g, ``1`` for rows g+8)
UNIFORM = "uniform"  # a cell-uniform scalar / literal — the same value for all 4 elements
COORD = "coord"  # a predicate over the element's absolute coordinates (``FRAG_ROW`` / ``FRAG_COL``)
GMEM = "gmem"  # a ``(buffer, index)`` load template over the element's absolute coordinates


@dataclass(frozen=True)
class FragLayout:
    """The **per-atom** mma C-fragment register layout — the one place the fragment-tier nodes read
    their geometry instead of hard-coding m16n8 magic numbers. Lifting it into a descriptor is the
    "geometry generality" seam: a second atom plugs in by adding a :func:`frag_layout` entry, not by
    editing each node.

    - ``n_elems`` — f32 registers per lane in the C-fragment.
    - ``elem_row`` — the row index (``0 .. rows_per_lane-1``) each element belongs to; ``rows_per_lane``
      derives from it.
    - ``reduce_xors`` — the row-reduce ``__shfl_xor`` butterfly lane masks.
    - ``lane_decl`` — the per-kernel lane preamble the coordinate ``Expr``s below reference.
    - ``lane_names`` — the names declared by that preamble.
    - ``row_off`` / ``col_off`` — the per-element coordinate **offsets as ``Expr``s** (over the
      ``lane_decl`` locals): ``row_off[r]`` the in-tile row of row-index ``r``, ``col_off[i]`` the
      in-N-atom column of element ``i``. A :class:`FragmentApply` COORD / GMEM operand adds the tile
      origin and substitutes these into its template (so a mask is a generic ``Expr``, not hard-coded CUDA).

    ``frag_layout`` raises for an unmodeled atom, so one fails loudly rather than miscompiling."""

    n_elems: int
    elem_row: tuple[int, ...]
    reduce_xors: tuple[int, ...]
    lane_decl: str
    lane_names: tuple[str, ...]
    row_off: tuple[Expr, ...]
    col_off: tuple[Expr, ...]

    @property
    def rows_per_lane(self) -> int:
        return max(self.elem_row) + 1


def _m16n8_col(b: int) -> Expr:
    """The in-N-atom column of an m16n8 C-fragment element with col-bit ``b``: ``_t * 2 + b``."""
    return BinaryExpr("+", BinaryExpr("*", Var("_t"), Literal(2, "int")), Literal(b, "int"))


#: The standard ``mma.sync.m16n8`` D/C-fragment layout — 4 f32 / lane: elements ``[0,1]`` are row
#: ``g = lane/4`` (cols ``(lane%4)*2 + {0,1}``), ``[2,3]`` are row ``g+8``; a row's 8 cols span the
#: 4 lanes differing in ``lane%4`` (the ``reduce_group = 4`` butterfly).
M16N8 = FragLayout(
    n_elems=4,
    elem_row=(0, 0, 1, 1),
    reduce_xors=(2, 1),
    lane_decl="const int _g = (threadIdx.x & 31) >> 2, _t = (threadIdx.x & 31) & 3;",
    lane_names=("_g", "_t"),
    row_off=(Var("_g"), BinaryExpr("+", Var("_g"), Literal(8, "int"))),
    col_off=(_m16n8_col(0), _m16n8_col(1), _m16n8_col(0), _m16n8_col(1)),
)


def _m8n8k4_col(offset: int) -> Expr:
    return Var("_vc") if offset == 0 else BinaryExpr("+", Var("_vc"), Literal(offset, "int"))


#: Volta's four m8n8k4 computation groups arranged as one logical 16x16 C fragment. A lane owns
#: two rows and four columns from each row; lanes of one row differ in bits 1 and 2.
M8N8K4 = FragLayout(
    n_elems=8,
    elem_row=(0, 0, 1, 1, 0, 0, 1, 1),
    reduce_xors=(4, 2),
    lane_decl=(
        "const int _vl = threadIdx.x & 31, _vq = (_vl & 15) >> 2, "
        "_vr = (_vq >> 1) * 8 + (_vl >> 4) * 4 + (_vl & 1), _vc = (_vq & 1) * 8 + (_vl & 2);"
    ),
    lane_names=("_vl", "_vq", "_vr", "_vc"),
    row_off=(Var("_vr"), BinaryExpr("+", Var("_vr"), Literal(2, "int"))),
    col_off=tuple(_m8n8k4_col(offset) for offset in (0, 1, 0, 1, 4, 5, 4, 5)),
)


def frag_layout(name: str) -> FragLayout:
    """The C-fragment layout named by an atom descriptor. A ``wgmma`` accumulator is, per warp, the
    ``m16n8`` C fragment repeated along N (:class:`WgmmaMma`), so it reads through that layout."""
    try:
        return {"m16n8k16": M16N8, "wgmma": M16N8, "m8n8k4": M8N8K4}[name]
    except KeyError as exc:
        raise ValueError(f"unmodeled C-fragment layout {name!r}") from exc


def _lane_preamble(ctx: RenderCtx, pad: str, decl: str, names: tuple[str, ...] = ("_g", "_t")) -> list[str]:
    """Emit one fragment-layout lane preamble per C scope."""
    if set(names) <= ctx.scope_decls:
        return []
    ctx.scope_decls.update(names)
    return [f"{pad}{decl}"]


@dataclass(frozen=True)
class FragmentArg:
    """One operand of a :class:`FragmentApply`. What it holds follows its ``kind``:

    - ``FRAG`` — ``names`` is the one C fragment;
    - ``ROW`` — ``names`` are the two per-row scalars, stored explicitly so an SSA rename keeps them consistent
      with their definitions;
    - ``UNIFORM`` — ``names`` is the one cell-uniform scalar, or ``exprs`` its one literal;
    - ``COORD`` — ``exprs`` is the one predicate;
    - ``GMEM`` — ``buffer`` is the buffer read and ``exprs`` its index template."""

    kind: str
    names: tuple[str, ...] = role("reads", default=())
    exprs: tuple[Expr, ...] = role("expr", default=())
    buffer: str | None = role("buffer_read", default=None)


def fragment_args(args: tuple, kinds: tuple[str, ...]) -> tuple[FragmentArg, ...]:
    """The operands of a :class:`FragmentApply` from the positional spelling the lowering builds — per kind a name
    (FRAG, UNIFORM), an expression (COORD, a UNIFORM literal), a pair of names (ROW) or ``(buffer, index)`` (GMEM)."""

    def one(arg, kind: str) -> FragmentArg:  # noqa: ANN001 — see the docstring
        if kind == GMEM:
            return FragmentArg(kind, exprs=tuple(arg[1]), buffer=arg[0])
        if kind == ROW:
            return FragmentArg(kind, names=tuple(arg))
        return FragmentArg(kind, exprs=(arg,)) if isinstance(arg, Expr) else FragmentArg(kind, names=(arg,))

    return tuple(one(arg, kind) for arg, kind in zip(args, kinds, strict=True))


@dataclass(frozen=True)
class FragmentApply(Declared):
    """The one per-element pointwise node over an ``mma.sync`` C fragment — the fragment-tier
    sibling of the scalar ``Assign``.

    Writes ``out`` (a layout-sized f32 C fragment) ``= op(args…)`` per element ``i`` via the same
    ``op_to_expr`` translation the scalar ``Assign`` uses — so ANY elementwise op reaches the
    tensor-core tier, not just softmax's ``exp`` / scale. Each arg (:class:`FragmentArg`) is one of five kinds:

    - ``FRAG`` — a C-fragment, indexed ``arg[i]``;
    - ``ROW`` — a per-row scalar pair, broadcast by the layout's element-to-row map;
    - ``UNIFORM`` — a cell-uniform scalar name or ``Literal``, the same value for every element;
    - ``COORD`` — a predicate ``Expr`` over the element's ABSOLUTE coordinates, written with the
      reserved :data:`FRAG_ROW` / :data:`FRAG_COL` vars (causal ``__fcol > __frow``, a symbolic
      boundary ``__fcol >= seq_len``); the render substitutes tile origin + layout offset;
    - ``GMEM`` — a ``(buffer, index)`` load template over the same reserved vars, read at each
      element's absolute coordinates and converted to f32 (SDPA's additive ``attn_mask`` bias;
      the gathered score tile of the softmax@V carrier).

    ``row_base`` / ``col_base`` are the tile origin the ``COORD`` / ``GMEM`` kinds add to the
    layout's per-element offsets; ``row_base`` is required iff a template references ``__frow``.

    Realizations: ``exp(s − m)`` = a ``subtract`` (FRAG, ROW) then an ``exp`` (FRAG); ``O *= α`` =
    an in-place ``multiply`` (FRAG, ROW); ``S *= scale`` = an in-place ``multiply`` (FRAG, UNIFORM);
    a coordinate mask = an in-place ``where`` (COORD, UNIFORM fill, FRAG) — the masked branch takes
    the carrier's finite identity instead of producing ``-inf - -inf`` in an all-masked chunk;
    a gmem bias = an in-place ``add`` (FRAG, GMEM). ``in_place`` reassigns ``out`` (no ``float
    out[4]`` decl — ``out`` must be a FRAG arg).

    ``layout`` names the atom's C-fragment geometry (:func:`frag_layout`)."""

    out: str = role("defines")
    op: ElementwiseImpl
    args: tuple[FragmentArg, ...] = role("record")
    in_place: bool = False
    layout: str = "m16n8k16"
    # Extra UNARY ops composed onto the result, outermost last — so ``exp(s − m)`` is one node
    # (``op=subtract``, ``post=(exp,)``) rendering ``out[_e] = expf(s[_e] − m)`` rather than a
    # subtract stmt feeding an exp stmt. Populated only by ``100_loopify``'s pin-gated chain fusion;
    # empty by default (the base op alone).
    post: tuple[ElementwiseImpl, ...] = ()
    row_base: Expr | None = role("expr", default=None)
    col_base: Expr | None = role("expr", default=None)

    def pretty(self, indent: str = "") -> list[str]:
        def _show(arg: FragmentArg) -> str:
            if arg.kind == FRAG:
                return f"{arg.names[0]}[]"
            if arg.kind == ROW:
                return f"{list(arg.names)}"
            if arg.kind == GMEM:
                return f"{arg.buffer}[{', '.join(str(e) for e in arg.exprs)}]"
            return str(arg.exprs[0]) if arg.exprs else arg.names[0]

        shown = ", ".join(_show(arg) for arg in self.args)
        chain = "".join(f" |> {p.name}" for p in self.post)
        return [f"{indent}FragmentApply({self.out} {'*=' if self.in_place else '<-'} {self.op.name}({shown}){chain})"]


def _row_expr(stmt: FragmentApply, names: tuple[str, ...]) -> Expr:
    """The per-row scalar for the loop element ``_e`` — a ternary over the layout's row split
    (m16n8: ``(_e < 2 ? row0 : row1)``). ``elem_row`` must be contiguous blocks (it is for every
    modeled atom), so the boundaries are ``block = n_elems / rows_per_lane`` apart."""
    lay = frag_layout(stmt.layout)
    rows, block = lay.rows_per_lane, lay.n_elems // lay.rows_per_lane
    contiguous = [r for r in range(rows) for _ in range(block)]
    if list(lay.elem_row) == contiguous:
        expr: Expr = Var(names[rows - 1])
        for r in range(rows - 2, -1, -1):
            expr = TernaryExpr(BinaryExpr("<", Var("_e"), Literal((r + 1) * block, "int")), Var(names[r]), expr)
        return expr
    expr = Var(names[lay.elem_row[-1]])
    for i in range(lay.n_elems - 2, -1, -1):
        expr = TernaryExpr(BinaryExpr("==", Var("_e"), Literal(i, "int")), Var(names[lay.elem_row[i]]), expr)
    return expr


def _frag_coords(stmt: FragmentApply, i: int) -> dict[str, Expr]:
    """Element ``i``'s absolute coordinates for the reserved vars: tile origin + layout offset."""
    lay = frag_layout(stmt.layout)
    sub: dict[str, Expr] = {}
    if stmt.col_base is not None:
        sub[FRAG_COL] = BinaryExpr("+", stmt.col_base, lay.col_off[i])
    if stmt.row_base is not None:
        sub[FRAG_ROW] = BinaryExpr("+", stmt.row_base, lay.row_off[lay.elem_row[i]])
    return sub


def _frag_operand(stmt: FragmentApply, arg: FragmentArg, ctx: RenderCtx, i: int | None) -> Expr:
    """One operand's expression for element ``i`` — or for the loop var ``_e`` when ``None``."""
    if arg.kind == FRAG:
        return Var(f"{arg.names[0]}[{'_e' if i is None else i}]")
    if arg.kind == ROW:
        return _row_expr(stmt, arg.names) if i is None else Var(arg.names[frag_layout(stmt.layout).elem_row[i]])
    if arg.kind == COORD:
        return arg.exprs[0].substitute(_frag_coords(stmt, i))
    if arg.kind == GMEM:
        sub = _frag_coords(stmt, i)
        text = f"{arg.buffer}[{render_index(arg.buffer, tuple(e.substitute(sub) for e in arg.exprs), ctx)}]"
        dt = ctx.buffer_dtypes.get(arg.buffer, "f32")
        return Var(text if dt == "f32" else ctx.target.convert(text, dt, "f32"))
    # UNIFORM — the fragment algebra is f32; a narrower scalar (e.g. an ``__half`` constant
    # load) converts through the target intrinsic, like the scalar ``Assign``'s promote rule.
    if arg.exprs:
        return arg.exprs[0]
    nm = arg.names[0]
    if ctx.literal_ssa and nm in ctx.literal_ssa:
        return Literal(float(ctx.literal_ssa[nm]))
    dt = ctx.ssa_dtypes.get(nm) if ctx.ssa_dtypes else None
    if dt and dt != "f32":
        from emmy.compiler.ir.expr import FuncCallExpr  # noqa: PLC0415

        converted = ctx.target.convert(nm, dt, "f32")
        paren = converted.index("(") if "(" in converted else -1
        if paren > 0 and converted.endswith(")"):
            return FuncCallExpr(converted[:paren], [Var(nm)])
    return Var(nm)


def _frag_element(stmt: FragmentApply, ctx: RenderCtx, i: int | None) -> str:
    argvars = [_frag_operand(stmt, arg, ctx, i) for arg in stmt.args]
    expr = op_to_expr(stmt.op.name, argvars)
    for post_op in stmt.post:  # compose the fused unary tail (outermost last)
        expr = op_to_expr(post_op.name, [expr])
    return expr.render(ctx)


@render_stmt.register
def _(stmt: FragmentApply, ctx: RenderCtx) -> list[str]:
    """One ``#pragma unroll`` loop over the fragment's ``n_elems`` — ``out[_e] = post…(op(args_e))``
    — instead of ``n_elems`` straight-line assignments. The ROW operand renders as the row-split
    ternary, so the m16n8 2-rows/lane structure stays legible (``_e < 2 ? row0 : row1``); identical
    SASS (nvcc unrolls the pragma). A COORD / GMEM operand differs per element in its coordinate
    expression rather than in an index, so it renders unrolled, one guarded line per element."""
    pad = _pad(ctx.indent)
    lay = frag_layout(stmt.layout)
    n = lay.n_elems
    lines = [] if stmt.in_place else [f"{pad}float {stmt.out}[{n}];"]
    if any(arg.kind in (COORD, GMEM) for arg in stmt.args):
        lines += _lane_preamble(ctx, pad, lay.lane_decl, lay.lane_names)
        return lines + [f"{pad}{stmt.out}[{i}] = {_frag_element(stmt, ctx, i)};" for i in range(n)]
    inner = ctx.child()
    lines.append(f"{pad}#pragma unroll")
    lines.append(f"{pad}for (int _e = 0; _e < {n}; _e++) {{")
    lines.append(f"{_pad(inner.indent)}{stmt.out}[_e] = {_frag_element(stmt, inner, None)};")
    lines.append(f"{pad}}}")
    return lines


@dataclass(frozen=True)
class FragmentRowReduce(Declared):
    """Per-ROW reduction over one warp's ``mma.sync`` C fragments — the chunk pivot and the
    per-row channel partials the twisted carrier's atom tier folds (``_atom._FlashOps``).

    A chunk-wide score tile is ``len(frags)`` C fragments side by side. Reducing over the chunk
    combines every element assigned to each of the lane's two rows, then applies the layout's
    ``__shfl_xor`` butterfly masks across the lanes that hold columns of the same row. Afterwards
    every lane in the group holds the full per-row value, so ``top`` and ``bot`` are the
    :data:`ROW` operands :class:`FragmentApply` broadcasts.

    Distinct from :class:`WarpShuffle`, which folds a whole per-thread monoid state over a
    cooperative-K lane set; this folds WITHIN one warp's fragments along the atom's N direction,
    keyed on the PTX C layout. It is why the tier requires the chunk to sit inside one warp column
    (``n_units == 1`` on the score): a reduction that crossed warps would need smem."""

    top: str = role("defines")  # the first per-lane row value, broadcast across its column group
    bot: str = role("defines")  # the second per-lane row value, broadcast across its column group
    frags: tuple[str, ...] = role("reads")  # the layout-sized C fragments spanning the chunk
    op: ElementwiseImpl  # the fold — ``maximum`` for the pivot, ``add`` for a summed channel
    layout: str = "m16n8k16"
    dtype: DataType = F32

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}FragmentRowReduce({self.top}, {self.bot} <- {', '.join(self.frags)}, op={self.op.name})"]


@render_stmt.register
def _(stmt: FragmentRowReduce, ctx: RenderCtx) -> list[str]:
    pad = _pad(ctx.indent)
    f32 = ctx.type_name("f32")

    def combine(parts: list[str]) -> str:
        e = parts[0]
        for p in parts[1:]:
            e = _binary_combine_expr(stmt.op, e, p, ctx.target, "f32")
        return e

    lay = frag_layout(stmt.layout)
    top_parts = [f"{f}[{i}]" for f in stmt.frags for i, row in enumerate(lay.elem_row) if row == 0]
    bot_parts = [f"{f}[{i}]" for f in stmt.frags for i, row in enumerate(lay.elem_row) if row == 1]
    out = [f"{pad}{f32} {stmt.top} = {combine(top_parts)};", f"{pad}{f32} {stmt.bot} = {combine(bot_parts)};"]
    ctx.ssa_dtypes[stmt.top] = ctx.ssa_dtypes[stmt.bot] = "f32"
    for s in lay.reduce_xors:
        for nm in (stmt.top, stmt.bot):
            shfl = f"__shfl_xor_sync(0xffffffff, {nm}, {s})"
            out.append(f"{pad}{nm} = {_binary_combine_expr(stmt.op, nm, shfl, ctx.target, 'f32')};")
    return out


#: The reserved coordinate Vars a :class:`FragmentApply` COORD / GMEM template is written over — the element's
#: ABSOLUTE query row / key column; the render substitutes each element's coords (tile origin +
#: layout offset) for these.
FRAG_ROW = "__frow"
FRAG_COL = "__fcol"
#: The reserved offset Vars a :class:`RegStore` epilogue is written over — the element's row / col
#: OFFSET within its fragment, added to the cell base the expression already carries.
ELEM_ROW = "__M__"
ELEM_COL = "__N__"


# ---------------------------------------------------------------------------
# Warp-level MMA: ``mma.sync.aligned`` + ``ldmatrix`` (the ``s16816`` path
# cuBLAS / CUTLASS use) — the sole tensor-core Stmt family. Operands are
# *explicit per-thread register arrays* with a PTX-fixed lane→element layout,
# referenced positionally inside inline PTX (``RegFragment`` / ``LdmatrixLoad``
# / ``MmaSyncPtx`` / ``RegStore``), rendered via the ``_MMA_SYNC_PRELUDE``
# helper wrappers (pure PTX — NVRTC-clean, no ``<mma.h>``). Emitted by
# the kernel atom lowering (``lowering/kernel/_atom.py``) from the tile's ``Atom`` spec.
# (The opaque ``nvcuda::wmma`` node family was removed — the swizzled mma.sync
# slab beat it.)
# ---------------------------------------------------------------------------


def _mma_sync_nregs(role: str, shape: tuple[int, int, int], dtype: DataType | None = None) -> int:
    """Per-lane register count for an mma.sync operand of cell ``shape``.

    ``a`` (M×K) and ``b`` (K×N) pack two 16-bit or four 8-bit elements per 32-bit register
    (``dtype`` is the operand element type; ``None`` keeps the 16-bit default); ``c`` holds one
    float per register (f32 accumulate) or two packed halfs (the f16-accumulate atom). For
    ``m16n8k16`` f16: ``a[4]`` (256 halfs / 32 lanes / 2), ``b[2]`` (128 / 32 / 2), ``c[4]`` f32
    (128 f32 / 32) / ``c[2]`` f16; for ``m16n8k32`` fp8: ``a[4]`` (512 bytes / 32 / 4), ``b[2]``."""
    m, n, k = shape
    if role in ("a", "b"):
        per_reg = 4 if dtype is not None and dtype.nbytes == 1 else 2
        return (m * k if role == "a" else k * n) // (32 * per_reg)
    if role == "c":
        return (m * n) // 64 if dtype is not None and dtype.nbytes == 2 else (m * n) // 32
    raise ValueError(f"mma.sync: unsupported role {role!r}; expected 'a', 'b', or 'c'")


@dataclass(frozen=True)
class RegFragment(Declared):
    """Declare an mma.sync per-thread register array.

    mma.sync multiplicands are explicit per-lane register arrays —
    ``unsigned a[4]`` / ``unsigned b[2]`` (f16, two halfs per 32-bit
    reg) — and the accumulator is ``float c[4]`` (f32) or, on the
    f16-accumulate atom, packed ``unsigned c[2]`` (two halfs per reg —
    the same element map, pair-packed, on m16n8k16). Volta uses explicit
    counts and distinct f16/f32 accumulator lane maps. ``shape`` is the cell
    ``(M, N, K)``; the count derives from ``shape`` + ``role`` (+ the C
    dtype) via :func:`_mma_sync_nregs`. The ``c`` array is
    zero-initialised at declaration, so the mma.sync path needs no
    separate fill node.

    ``count`` (default 1) arrays a whole fragment FAMILY into one declaration —
    ``float O_i_f[count][n_regs]`` — so a run of ``count`` sibling fragments
    (``O_i_f0 … O_i_f{count-1}``) collapses to a single arrayed decl indexed
    ``O_i_f[t]``. Only ``100_loopify`` sets ``count > 1`` (the pin-gated
    re-roll); ``count == 1`` renders the flat ``float name[n_regs]`` form,
    byte-identical. A family of SCALAR registers (``nregs == 0`` — the fp4
    block-scale fragments) arrays to one dimension, ``unsigned name[count]``,
    because its member is the register itself and not an array of them."""

    name: str = role("defines", "local_buffer")
    role: str  # "a" / "b" / "c"
    shape: tuple[int, int, int]
    dtype: DataType
    count: int = 1  # >1 arrays the fragment family: ``<ty> name[count][n_regs]`` (loopify only)
    nregs: int | None = None  # explicit for layouts whose one instruction realizes several PTX cells; 0 = one scalar register

    def pretty(self, indent: str = "") -> list[str]:
        m, n, k = self.shape
        return [f"{indent}RegFragment {self.role}:{self.dtype.name} {self.name}{_frag_dims(self)} ({m}x{n}x{k})"]


def _frag_nregs(stmt: RegFragment) -> int:
    return stmt.nregs if stmt.nregs is not None else _mma_sync_nregs(stmt.role, stmt.shape, stmt.dtype)


def _frag_dims(stmt: RegFragment) -> str:
    """The declaration's subscripts — the family count when arrayed, then the per-member
    register count, which a scalar register does not have. Spelled once so the listing and the
    emitted decl cannot disagree about a family's shape."""
    n_regs = _frag_nregs(stmt)
    extents = ((stmt.count,) if stmt.count > 1 else ()) + ((n_regs,) if n_regs else ())
    return "".join(f"[{e}]" for e in extents)


@render_stmt.register
def _(stmt: RegFragment, ctx: RenderCtx) -> list[str]:
    n_regs = _frag_nregs(stmt)
    ctx.ssa_dtypes[stmt.name] = stmt.dtype.name
    dims = _frag_dims(stmt)
    if stmt.role == "c" and stmt.dtype.nbytes != 2:
        # Brace-init zeros the whole (arrayed) accumulator; ``= {0.0f, ...}`` stays the flat form.
        init = " = {}" if stmt.count > 1 else f" = {{{', '.join(['0.0f'] * n_regs)}}}"
        return [f"{_pad(ctx.indent)}float {stmt.name}{dims}{init};"]
    if stmt.role == "c":
        # f16-accumulate C: packed half pairs, zero bits = 0.0h pairs (rezeroed by FragmentPromote).
        init = " = {}" if stmt.count > 1 else f" = {{{', '.join(['0u'] * n_regs)}}}"
        return [f"{_pad(ctx.indent)}unsigned {stmt.name}{dims}{init};"]
    return [f"{_pad(ctx.indent)}unsigned {stmt.name}{dims};"]


def frag_dtype(ctx: RenderCtx, frag: str) -> str | None:
    """Canonical dtype token of a fragment reference (``None`` when no :class:`RegFragment`
    declares it).

    A ``count``-arrayed family declares the STEM (``_b``) and its members are referenced
    subscripted (``_b[_r0]`` inside a re-rolled loop, ``_b[3]`` outside one), so the dtype hangs
    off the stem and a bare ``ctx.ssa_dtypes`` lookup on the reference misses. A miss is silent —
    every caller has a fallback — so it does not raise, it picks the wrong loader: a packed-pair
    (NVFP4) byte slab read as if its bytes were fp8 elements, one stored byte standing for two K
    values."""
    return ctx.ssa_dtypes.get(frag.partition("[")[0])


# Per-lane fp16 element XOR matching the TMA hardware smem swizzle (the
# ldmatrix consumer side). The producer (cuTensorMapEncodeTiled with
# SWIZZLE_{128,64,32}B) permutes 16-byte (8-fp16) chunks within each swizzle
# row by the row-index-mod-2^B; the ldmatrix consumer must apply the same
# permutation to its per-lane element offset. All three modes are CUTLASS
# ``Swizzle<B,4,3>`` (M=4 → 16-byte base, S=3 → 8-row tile), differing ONLY in
# B (3/2/1 XORed bits). In bytes the swizzle is
# ``off ^= ((off >> (M+S)) & (2^B-1)) << M`` = ``((off >> 7) & mask) << 4``;
# halving to fp16 elements turns ``>> 7`` into ``>> 6`` and ``<< 4`` into
# ``<< 3`` — so the element shift is 6 for EVERY mode (a per-mode shift of
# 6/5/4 silently corrupts B64/B32 while leaving B128 correct, which is easy to
# miss since B128 dominates). Only the row mask changes: 0x7 / 0x3 / 0x1. The
# 8-row × atom period divides the slab + slot strides, so the XOR is correct
# measured from the buffer base. Maps mode → (element shift, row mask); the
# chunk delta is always ``<< 3``.
#
# The XOR is purely address-based, so it is also the SOFTWARE swizzle: a
# ``cp.async``-filled slab applies the same helper to its fill DESTINATION
# index (:class:`CpAsyncCopy` ``swizzle``) and the ldmatrix drain reads it
# back through the identical XOR — fill and drain agree by construction, no
# hardware copy engine involved. Intra-chunk bits (0..2) pass through, so
# narrower 4/8-byte fills keep their offset inside the relocated 16-byte
# chunk and ``cp.async`` address alignment is preserved.
LDMATRIX_SWIZZLE_XOR: dict[str, tuple[int, int]] = {
    "B128": (6, 0x7),
    "B64": (6, 0x3),
    "B32": (6, 0x1),
}

#: The two software-written Volta operand layouts. Unlike the modern ``B*`` XOR modes, these
#: permute complete logical ``(row, col)`` coordinates so an SM70 warp can drain the slab through
#: ordinary 128-bit shared loads. They are internal lowering choices, not schedule knobs.
VOLTA_CROSSWISE = "V70A"
VOLTA_B_CONGRUOUS = "V70B"
VOLTA_SMEM_LAYOUTS = frozenset((VOLTA_CROSSWISE, VOLTA_B_CONGRUOUS))

#: A SOFTWARE-swizzled mode may override the element shift, spelled ``<mode>@<shift>``. The shift
#: above is ``log2(atom elems)``, which reads the row index only while a slab row IS one swizzle
#: atom. A WIDER row (the flash slabs: a 128-elem fp16 head-dim row is two 128 B atoms) leaves the
#: atom bit inside the shifted field, so consecutive rows collapse onto a quarter of the chunk
#: positions and an ``ldmatrix`` over 16 rows drains multi-way bank-conflicted (measured 7.8-way at
#: ``D = 128``, none at ``D = 64``). Taking the shift from the slab's OWN row stride restores
#: row-mod-8, hence the full chunk spread. Hardware (TMA) slabs keep the plain spelling: there the
#: copy engine fixes the permutation, and the descriptor already splits its box down to the atom.
_SWIZZLE_SHIFT_SEP = "@"


def swizzle_xor(mode: str) -> tuple[int, int] | None:
    """``(element shift, row mask)`` for a swizzle mode spelling, or ``None`` when the mode carries
    no XOR (``"NONE"`` / an unknown spelling). Accepts the plain mode and the software
    ``<mode>@<shift>`` override."""
    base, _, shift = mode.partition(_SWIZZLE_SHIFT_SEP)
    xor = LDMATRIX_SWIZZLE_XOR.get(base)
    if xor is None:
        return None
    return (int(shift), xor[1]) if shift else xor


def swizzle_base(mode: str) -> str:
    """The hardware mode name behind a (possibly shift-overridden) spelling — what the swizzle's
    slab alignment and TMA descriptor are keyed on."""
    return mode.partition(_SWIZZLE_SHIFT_SEP)[0]


def swizzle_fn(mode: str) -> str:
    """The ``emmy_swizzle_*`` helper name for a mode spelling — one helper per distinct
    ``(mask, shift)`` pair, so two slabs sharing a spelling share the emitted function."""
    base = swizzle_base(mode)
    return f"emmy_swizzle_{base.lower()}" + ("" if mode == base else f"_s{swizzle_xor(mode)[0]}")


def _multiple_of(expr: Expr, d: int, aligned: Mapping[str, int] | None = None) -> bool:
    """Whether ``expr`` is provably a multiple of ``d``: a literal that is, a loop counter whose
    stride is (``aligned``, a render context's), a sum or difference of two that are, or a product
    one of whose literal factors supplies what the other need not."""
    if d == 1:
        return True
    if isinstance(expr, Literal):
        return isinstance(expr.value, int) and expr.value % d == 0
    if isinstance(expr, Var):
        return aligned is not None and expr.name in aligned and aligned[expr.name] % d == 0
    if isinstance(expr, BinaryExpr) and expr.op in ("+", "-"):
        return _multiple_of(expr.left, d, aligned) and _multiple_of(expr.right, d, aligned)
    if isinstance(expr, BinaryExpr) and expr.op == "*":
        for lit, other in ((expr.left, expr.right), (expr.right, expr.left)):
            if (
                isinstance(lit, Literal)
                and isinstance(lit.value, int)
                and lit.value
                and _multiple_of(other, d // math.gcd(lit.value, d), aligned)
            ):
                return True
    return False


def swizzled_slab_index(
    mode: str, ldm: int, lane: str, row: Expr, col: Expr, ctx: RenderCtx, *, lane_rows: int, lane_col_mod: int
) -> str | None:
    """The swizzled slab element index of ``(row, col)`` plus a lane's own ``lane`` offset, split so
    the swizzle is applied to the lane's part ONCE — ``None`` when the split does not apply.

    The XOR swizzle ``e ^ (((e >> s) & m) << 3)`` is linear over bit-disjoint parts:
    ``swz(a | b) = swz(a) ^ swz(b)``. A lane's offset spans ``lane_rows`` rows and the column
    positions ``lane_col_mod`` apart, so a tile base whose row is a multiple of ``lane_rows`` and
    whose column is a multiple of ``lane_col_mod`` (within the row) shares no bit with it, and
    once the base row also clears the XOR's own field (a multiple of eight rows on every mode), its
    row lands above every swizzled bit and adds: ``(swz(lane) ^ swz(col)) + row·ldm``. ``swz(lane)``
    is one per-lane value a loop hoists, ``swz(col)`` a constant nvcc folds, and the row an
    immediate offset of the load — where re-applying the swizzle to every load's full index
    recomputed the XOR from ``threadIdx`` each time and was a fifth of an attention chunk's
    instructions."""
    xor = swizzle_xor(mode)
    if xor is None or ldm < 8 or ldm & (ldm - 1) or lane_rows & (lane_rows - 1) or lane_col_mod & (lane_col_mod - 1):
        return None
    shift, mask = xor
    field_mod = 1 << max(0, shift + (mask + 1).bit_length() - ldm.bit_length())
    if not (_multiple_of(row, max(lane_rows, field_mod), ctx.aligned) and _multiple_of(col, lane_col_mod, ctx.aligned)):
        return None
    fn = swizzle_fn(mode)
    return f"({fn}({lane}) ^ {fn}({col.render(ctx)})) + ({row.render(ctx)}) * {ldm}"


def smem_layout_index(mode: str, flat: str, buffer: str, ctx: RenderCtx, indices: tuple[Expr, ...] = ()) -> str:
    """Render ``flat`` through the slab layout selected by ``mode``.

    Modern layouts are the existing one-argument XOR. Volta A needs both logical dimensions:
    the crosswise physical stride is the slab's row count, while Volta B's congruous layout uses
    the ordinary logical row stride. Their writer already has those dimensions, so pass them to
    the layout directly instead of flattening and dividing the result apart again in every K-loop
    deposit. ``NONE`` and unknown spellings stay row-major.
    """
    if mode == VOLTA_CROSSWISE:
        assert len(indices) == 2, "the Volta A layout needs a logical (row, column) index"
        rows, _ = ctx.shapes[buffer]
        row, col = (e.render(ctx) for e in indices)
        return f"emmy_volta_crosswise({row}, {col}, {rows})"
    if mode == VOLTA_B_CONGRUOUS:
        assert len(indices) == 2, "the Volta B layout needs a logical (row, column) index"
        cols = ctx.shapes[buffer][-1]
        row, col = (e.render(ctx) for e in indices)
        return f"emmy_volta_b_congruous({row}, {col}, {cols})"
    return flat if not swizzle_xor(mode) else f"{swizzle_fn(mode)}({flat})"


@dataclass(frozen=True)
class BlockScaleLoad(Declared):
    """Load one lane's b32 of ue4m3 block scales for the block-scaled fp4 mma.

    The instruction takes its per-16-element scales as two ordinary register operands beside the
    multiplicands, one per side, and applies them itself. A supplying lane holds four K blocks as
    the four bytes of one b32 — exactly the ``k = 64`` an atom step spans — and the lane→row map
    differs per side and is NOT the data fragment's: SFA's supplying lanes are picked by
    ``(lane & 3)`` against a thread-id selector, SFB's by equality with one. Both maps were found
    by probing the hardware, and the render's prelude records them beside the PTX (the PTX ISA's
    documented thread-id range is wrong for ``scale_vec::4X`` — ptxas takes 0..1, not 0..3).

    ``src_buffer`` is the scale slab, ``src_index`` the fragment's ``(row, k-block)`` base and
    ``ldm`` the slab's row stride in bytes. Non-supplying lanes read nothing and the instruction
    ignores their register, so there is no guard here.

    Distinct from :class:`LdmatrixLoad`'s ``scale_buffer``, which is the OTHER way to spend a
    block scale: there the k16 drain folds it into a 16-bit fragment and no scale reaches the
    instruction. These two never appear on one operand."""

    frag: str = role("defines")
    src_buffer: str = role("buffer_read")
    src_index: tuple[Expr, ...] = role("expr")
    role: str  # "a" / "b" — which side's lane map supplies the rows
    ldm: int = 0

    def pretty(self, indent: str = "") -> list[str]:
        idx = ", ".join(str(e) for e in self.src_index)
        return [f"{indent}BlockScaleLoad {self.frag} <- {self.src_buffer}[{idx}] (sf{self.role}, ldm={self.ldm})"]


@render_stmt.register
def _(node: BlockScaleLoad, ctx: RenderCtx) -> list[str]:
    flat = render_index(node.src_buffer, node.src_index, ctx)
    # The slab declares the scales at their storage dtype (``__nv_fp8_e4m3``); the loader moves
    # them as the raw ue4m3 bytes the instruction reads, so the pointer casts rather than the
    # slab changing type — a byte-typed slab would lose that the bytes ARE e4m3 to every other
    # reader of ``ctx.buffer_dtypes``.
    src = f"reinterpret_cast<const unsigned char*>(&{node.src_buffer}[{flat}])"
    # An ASSIGNMENT: the scale fragment is declared once beside the operand fragments
    # (``RegFragment`` at zero registers), so a double-buffered k-loop reloads it per step.
    return [f"{_pad(ctx.indent)}{node.frag} = emmy_mma_load_sf{node.role}_f4({src}, {node.ldm});"]


@dataclass(frozen=True)
class LdmatrixLoad(Declared):
    """``ldmatrix.sync.aligned.m8n8.x{2,4}[.trans].b16`` — load one operand
    fragment from smem into a per-thread register array.

    ``frag`` is the destination :class:`RegFragment`; ``src_buffer`` /
    ``src_index`` are the cell's smem tile base (the same ``(buffer,
    base-offset)`` ``_mma_src_index`` computes); ``ldm``
    is the smem row stride in elements. Each lane derives its own row
    address from its warp lane id (``threadIdx.x & 31``): the 16×K ``a``
    tile uses ``x4`` (``row = lane%16``, K-col block ``(lane/16)*8``); the
    K×8 ``b`` tile uses ``x2.trans`` (``row = lane%16``) so a row-major
    smem slab feeds the mma's col-major B operand.

    ``staged`` (default ``True``) selects the transport. The modern fragment layout's ``ldmatrix`` is **smem
    only**, while the Volta layout uses its cooperative gather against either address space. When the
    operand was NOT staged into shared memory
    (``staged=False``, ``src_buffer`` is the gmem operand) the render emits the
    ``emmy_mma_load_{a,b}_gmem`` helper instead — a gmem-direct fragment load that
    replicates the same lane→element map without ldmatrix. Slower (no smem reuse)
    but correct; the atom lowering picks per operand based on whether an
    enclosing ``StageBundle`` staged it.

    ``gmem_guard`` (gmem-direct only) carries a masked-tile boundary as
    ``(base Expr, bound Expr)`` on the operand's lane-varying output axis
    (A's M rows, B's N cols): the render switches to the ``_mclamp`` /
    ``_nclamp`` helper, which clamps the lane coordinate to the ``bound -
    base`` in-range elements — the gmem-direct analogue of the staged
    slab-fill clamp, so an unstaged masked cell still lowers instead of
    reading past the runtime-sized buffer. Clamped lanes read duplicates;
    their stores are masked by the RegStore guards."""

    frag: str = role("updates")
    src_buffer: str = role("buffer_read")
    src_index: tuple[Expr, ...] = role("expr")
    role: str  # "a" → x4; "b" → x2.trans
    ldm: int = 0
    swizzle: str = "NONE"  # TMA smem swizzle mode the slab was written with
    staged: bool = True  # False → gmem-direct fragment load (operand not in smem)
    gmem_guard: tuple[Expr, Expr] | None = role("expr", default=None)  # masked-axis (base, bound); gmem-direct only
    k_zero: tuple[Expr, Expr] | None = role("expr", default=None)  # masked-K reduce axis (base, bound); gmem-direct only
    b_trans: bool = False  # role "b" only: B stored N×K (Q@K^T native col-major) → gmem-direct trans helper / staged plain x2
    # A second B fragment filled by the SAME ldmatrix (the ``096_pair_ldmatrix_loads`` peephole):
    # the paired x4 loads two slab-adjacent B fragments in one instruction — ``frag`` gets the
    # matrices addressed by lanes 0-15, ``pair_frag`` those of lanes 16-31. Staged role-"b"
    # NONE-swizzle only; ``src_index`` stays the FIRST (low) fragment's tile base.
    pair_frag: str | None = role("updates", default=None)
    # A staged 1-BYTE (fp8) operand slab: ldmatrix is b16-only below sm_100a, so the drain is the
    # cooperative per-lane byte gather instead — the SAME lane→element map the gmem-direct
    # fragment loaders spell, pointed at the smem slab. A 16-bit destination fragment (W8A16)
    # converts per element via the loader's F template arg; an fp8 fragment (the k32 atoms)
    # repacks raw bytes via the ``_b8`` family. NONE-swizzle by construction; the padded row
    # stride rides ``ldm``. Staged only.
    byte_slab: bool = False
    # The PACKED-PAIR byte slab's companion block-scale slab. A packed byte holds two logical K
    # elements, so its drain decodes both halves through the 16-entry e2m1 value table and scales
    # them by the weight's per-k-block scale — one drain reading TWO shared buffers. The bits are
    # ``src_buffer`` (``src_index`` is the fragment's ``(row, byte column)`` base, ``ldm`` its byte
    # row stride); the scales are here, ``scale_index`` the fragment's ``(row, k-block)`` base and
    # ``scale_ldm`` that slab's row stride in elements. ``None`` on every other drain — an fp8 byte
    # slab carries a k-INVARIANT scale, which commutes out of the fold into the epilogue instead.
    scale_buffer: str | None = role("buffer_read", default=None)
    scale_index: tuple[Expr, ...] = role("expr", default=())
    scale_ldm: int = 0
    fragment_layout: str = "m16n8k16"
    # Position inside this warp operand's register tile. Volta pairs even/odd logical fragments
    # after slot offsets have made structural coordinate-delta matching needlessly fragile.
    fragment_index: int = 0

    def pretty(self, indent: str = "") -> list[str]:
        idx = ", ".join(str(e) for e in self.src_index)
        if self.pair_frag is not None:
            variant = "x4 pair" if self.b_trans else "x4.trans pair"
            pair = f"({self.frag}, {self.pair_frag})"
            return [f"{indent}LdmatrixLoad {pair} <- {self.src_buffer}[{idx}] ({variant}, ldm={self.ldm or 'auto'})"]
        if self.scale_buffer is not None:
            variant = f"byte decode x {self.scale_buffer}"
        elif self.byte_slab:
            variant = "byte gather"
        elif self.fragment_layout == "m8n8k4" and self.staged:
            variant = "shared gather"
        else:
            variant = ("x4" if self.role == "a" else ("x2" if self.b_trans else "x2.trans")) if self.staged else "gmem-direct"
        guard = "" if self.gmem_guard is None else f" guard<{self.gmem_guard[1]}"
        return [f"{indent}LdmatrixLoad {self.frag} <- {self.src_buffer}[{idx}] ({variant}{guard}, ldm={self.ldm or 'auto'})"]


def _swizzled_addr(node: LdmatrixLoad, flat: str, lane: str, ctx: RenderCtx, lane_rows: int, lane_col_mod: int) -> str:
    """The lane's slab address: the tile base ``flat`` plus the lane's own ``lane`` offset,
    which spans ``lane_rows`` rows and columns ``lane_col_mod`` apart, through the slab's
    swizzle — split per :func:`swizzled_slab_index` wherever the tile base allows, so the
    swizzle of the lane's part is computed once for the whole drain."""
    if not swizzle_xor(node.swizzle):
        return f"&{node.src_buffer}[{flat} + {lane}]"
    split = None
    if len(node.src_index) == 2 and ctx.shapes.get(node.src_buffer, (0, 0))[-1] == node.ldm:
        split = swizzled_slab_index(
            node.swizzle, node.ldm, lane, node.src_index[0], node.src_index[1], ctx, lane_rows=lane_rows, lane_col_mod=lane_col_mod
        )
    # ``emmy_swizzle_<mode>`` (preamble, built from ``LDMATRIX_SWIZZLE_XOR``) applies
    # ``e ^ (((e >> shift) & mask) << 3)`` — the helper spells the (often long) element
    # index once instead of inlining it twice around the XOR.
    return f"&{node.src_buffer}[{split if split is not None else f'{swizzle_fn(node.swizzle)}({flat} + {lane})'}]"


@render_stmt.register
def _(node: LdmatrixLoad, ctx: RenderCtx) -> list[str]:

    flat = render_index(node.src_buffer, node.src_index, ctx)
    ldm = node.ldm if node.ldm else _resolve_ldm(node.src_buffer, ctx)
    if not node.staged:
        # Operand not staged in smem — ldmatrix can't reach gmem, so read the
        # fragment straight from gmem (each lane adds its own (row,col) inside
        # the helper). No swizzle: that's a TMA-smem-layout concern only.
        # A source buffer traced at a dtype other than the fragment's (the gemma layer's
        # f32 V intermediate) converts per element: the helper's fragment element type F
        # is passed explicitly, so the pack writes true 16-bit halves instead of raw f32
        # bit patterns (matched dtypes keep the bare call — byte-identical output).
        src_dt = ctx.buffer_dtypes.get(node.src_buffer, "f16")
        frag_dt = frag_dtype(ctx, node.frag) or src_dt
        targs = "" if src_dt == frag_dt else f"<{ctx.type_name(src_dt)}, {ctx.type_name(frag_dt)}>"
        b8 = frag_dt in ("f8e4m3", "f8e5m2")
        addr = render_address(node.src_buffer, node.src_index, ctx)
        if node.fragment_layout == "m8n8k4":
            shape = tuple(Dim(d) for d in ctx.shapes.get(node.src_buffer, ()))
            aligned = len(shape) == len(node.src_index) and all(d.is_static for d in shape)
            if aligned:
                aligned = all(
                    _multiple_of(index * Literal(math.prod(s.as_static() for s in shape[d + 1 :]), "int"), 4)
                    for d, index in enumerate(node.src_index)
                )
            vector = aligned and ldm % 4 == 0 and node.k_zero is None and src_dt in ("f16", "f32") and frag_dt == "f16"
            if vector and (node.role == "a" or node.b_trans):
                left = "16"
                if node.gmem_guard is not None:
                    base, bound = node.gmem_guard
                    left = f"({bound.render(ctx)}) - ({base.render(ctx)})"
                args = f"<{ctx.type_name(src_dt)}, {'true' if node.role == 'a' else 'false'}>"
                return [f"{_pad(ctx.indent)}emmy_mma884_load_gmem4{args}({node.frag}, {addr}, {ldm}, {left});"]
            if node.k_zero is not None:
                kbase, kbound = node.k_zero[0].render(ctx), node.k_zero[1].render(ctx)
                k_left = f"({kbound}) - ({kbase})"
                if node.gmem_guard is not None:
                    base, bound = node.gmem_guard[0].render(ctx), node.gmem_guard[1].render(ctx)
                    mn_left = f"({bound}) - ({base})"
                    helper = (
                        "emmy_mma884_load_a_gmem_mclamp_kzero"
                        if node.role == "a"
                        else ("emmy_mma884_load_b_gmem_trans_nclamp_kzero" if node.b_trans else "emmy_mma884_load_b_gmem_nclamp_kzero")
                    )
                    return [f"{_pad(ctx.indent)}{helper}{targs}({node.frag}, {addr}, {ldm}, {mn_left}, {k_left});"]
                helper = (
                    "emmy_mma884_load_a_gmem_kzero"
                    if node.role == "a"
                    else ("emmy_mma884_load_b_gmem_trans_kzero" if node.b_trans else "emmy_mma884_load_b_gmem_kzero")
                )
                return [f"{_pad(ctx.indent)}{helper}{targs}({node.frag}, {addr}, {ldm}, {k_left});"]
            if node.gmem_guard is not None:
                base, bound = node.gmem_guard[0].render(ctx), node.gmem_guard[1].render(ctx)
                helper = (
                    "emmy_mma884_load_a_gmem_mclamp"
                    if node.role == "a"
                    else ("emmy_mma884_load_b_gmem_trans_nclamp" if node.b_trans else "emmy_mma884_load_b_gmem_nclamp")
                )
                return [f"{_pad(ctx.indent)}{helper}{targs}({node.frag}, {addr}, {ldm}, ({bound}) - ({base}));"]
            helper = (
                "emmy_mma884_load_a_gmem"
                if node.role == "a"
                else ("emmy_mma884_load_b_gmem_trans" if node.b_trans else "emmy_mma884_load_b_gmem")
            )
            return [f"{_pad(ctx.indent)}{helper}{targs}({node.frag}, {addr}, {ldm});"]
        if b8:
            # fp8 fragments gather RAW bytes with the k32 fragment map (the ``_b8`` helper
            # family) — there is no per-element convert (the mma consumes storage bits), so
            # the source buffer must already carry the fragment dtype, and no masked-K byte
            # gather exists (the schedule offers the fp8 atoms only on a static K).
            if src_dt != frag_dt:
                raise ValueError(f"fp8 mma fragment {node.frag}: source buffer {node.src_buffer!r} is {src_dt}, not {frag_dt}")
            if node.k_zero is not None:
                raise ValueError("fp8 mma fragments have no masked-K byte gather — the fp8 atoms are static-K only")
        sfx = "_b8" if b8 else ""
        if node.k_zero is not None:
            # Masked-K (symbolic reduce): zero-fill (not clamp) the K halves
            # past the runtime extent so the mma reduction stays correct — K is
            # summed, so a duplicate corrupts it. ``k_left`` = in-range K elements
            # from the tile base; may co-occur with an M/N clamp. Transposed-B (K
            # contiguous) has its own (n, k)-swapped zero-fill helper.
            kbase, kbound = node.k_zero[0].render(ctx), node.k_zero[1].render(ctx)
            k_left = f"({kbound}) - ({kbase})"
            if node.gmem_guard is not None:
                base, bound = node.gmem_guard[0].render(ctx), node.gmem_guard[1].render(ctx)
                mn_left = f"({bound}) - ({base})"
                if node.role == "a":
                    helper = "emmy_mma_load_a_gmem_mclamp_kzero"
                else:
                    helper = "emmy_mma_load_b_gmem_trans_nclamp_kzero" if node.b_trans else "emmy_mma_load_b_gmem_nclamp_kzero"
                return [f"{_pad(ctx.indent)}{helper}{targs}({node.frag}, {addr}, {ldm}, {mn_left}, {k_left});"]
            if node.role == "a":
                helper = "emmy_mma_load_a_gmem_kzero"
            else:
                helper = "emmy_mma_load_b_gmem_trans_kzero" if node.b_trans else "emmy_mma_load_b_gmem_kzero"
            return [f"{_pad(ctx.indent)}{helper}{targs}({node.frag}, {addr}, {ldm}, {k_left});"]
        if node.gmem_guard is not None:
            # Masked axis: clamp the lane coordinate to the in-range
            # elements left from the tile base (>= 1 — the boundary Cond
            # admitted the tile).
            base, bound = node.gmem_guard[0].render(ctx), node.gmem_guard[1].render(ctx)
            if node.role == "a":
                helper = "emmy_mma_load_a_gmem_mclamp"
            else:
                helper = "emmy_mma_load_b_gmem_trans_nclamp" if node.b_trans else "emmy_mma_load_b_gmem_nclamp"
            return [f"{_pad(ctx.indent)}{helper}{sfx}{targs}({node.frag}, {addr}, {ldm}, ({bound}) - ({base}));"]
        if node.role == "a":
            helper = "emmy_mma_load_a_gmem"
        else:
            helper = "emmy_mma_load_b_gmem_trans" if node.b_trans else "emmy_mma_load_b_gmem"
        return [f"{_pad(ctx.indent)}{helper}{sfx}{targs}({node.frag}, {addr}, {ldm});"]
    if node.fragment_layout == "m8n8k4":
        # SM70 has no ldmatrix. The Volta fragment's cooperative lane map is the same for
        # global and shared addresses, so point its inlined gather at the staged slab; ptxas
        # resolves the generic pointer to ordinary LDS instructions. Fill-side clamps make
        # the slab exact, and the first implementation intentionally keeps it unswizzled.
        assert not node.byte_slab, "the Volta staged drain is an unpacked f16 gather"
        slab_dt = ctx.buffer_dtypes.get(node.src_buffer, "f16")
        frag_dt = frag_dtype(ctx, node.frag) or slab_dt
        targs = "" if slab_dt == frag_dt else f"<{ctx.type_name(slab_dt)}, {ctx.type_name(frag_dt)}>"
        if node.swizzle == VOLTA_CROSSWISE:
            # A, or a TRANSPOSED B — whose slab is K-contiguous like A's and so takes the
            # same storage; the reader differs only in which lane bit selects the half.
            assert node.pair_frag is not None and slab_dt == frag_dt == "f16"
            assert node.role == "a" or node.b_trans, "a canonical B reads the congruous layout, not crosswise"
            row, col = (e.render(ctx) for e in node.src_index)
            rows = ctx.shapes[node.src_buffer][0]
            helper = "emmy_mma884_load_a_crosswise_pair" if node.role == "a" else "emmy_mma884_load_b_crosswise_pair"
            return [f"{_pad(ctx.indent)}{helper}({node.frag}, {node.pair_frag}, {node.src_buffer}, {row}, {col}, {rows});"]
        if node.swizzle == VOLTA_B_CONGRUOUS:
            assert node.role == "b" and node.pair_frag is not None and slab_dt == frag_dt == "f16"
            row, col = (e.render(ctx) for e in node.src_index)
            return [
                f"{_pad(ctx.indent)}emmy_mma884_load_b_congruous_pair({node.frag}, {node.pair_frag}, "
                f"{node.src_buffer}, {row}, {col}, {ldm});"
            ]
        assert node.pair_frag is None and node.swizzle == "NONE"
        helper = (
            "emmy_mma884_load_a_smem"
            if node.role == "a"
            else ("emmy_mma884_load_b_smem_trans" if node.b_trans else "emmy_mma884_load_b_smem")
        )
        return [f"{_pad(ctx.indent)}{helper}{targs}({node.frag}, &{node.src_buffer}[{flat}], {ldm});"]
    if node.byte_slab:
        # Staged 1-byte (fp8) slab — the cooperative byte-gather drain: the gmem fragment
        # loaders' lane→element map pointed at the smem slab (generic-address loads; ptxas
        # resolves them to LDS after inlining). A 16-bit fragment (W8A16) converts per
        # element via the loader's F template arg (the same ``<__nv_fp8_e4m3, __half>``
        # spelling as the gmem-direct convert); an fp8 fragment (the k32 atoms) repacks raw
        # bytes via the ``_b8`` family. ``flat`` already strides the PADDED slab decl and
        # ``ldm`` carries the same padded row stride, so the two agree by construction.
        slab_dt = ctx.buffer_dtypes.get(node.src_buffer, "f8e4m3")
        frag_dt = frag_dtype(ctx, node.frag) or "f16"
        if node.scale_buffer is not None:
            scale_flat = render_index(node.scale_buffer, node.scale_index, ctx)
            args = f"&{node.src_buffer}[{flat}], {ldm}, &{node.scale_buffer}[{scale_flat}], {node.scale_ldm}"
            if slab_dt in ("f8e4m3", "f8e5m2"):
                # Block-scaled fp8 B: each byte pair converts to two exact 16-bit values, both
                # multiply the k block's f32 scale, and the product rounds once to the fragment.
                x2 = {"f8e4m3": "__nv_fp8x2_e4m3", "f8e5m2": "__nv_fp8x2_e5m2"}[slab_dt]
                call = f"emmy_mma_load_b_smem_trans_f8s_{frag_dt}<{ctx.type_name(slab_dt)}, {x2}>"
                return [f"{_pad(ctx.indent)}{call}({node.frag}, {args});"]
            # Packed-pair (NVFP4) B: each byte decodes to two values through the e2m1 value
            # table and both take the k block's scale, read from the companion slab. The scale
            # slab carries the fragment's own dtype, so it names which drain form to call.
            scale_dt = ctx.buffer_dtypes.get(node.scale_buffer, "f16")
            return [f"{_pad(ctx.indent)}emmy_mma_load_b_smem_trans_f4s_{scale_dt}({node.frag}, {args});"]
        k_contig = node.role == "a" or node.b_trans  # each lane's K run is slab-contiguous
        if frag_dt in ("f8e4m3", "f8e5m2", "f4e2m1x2"):
            # Raw-byte repack, no per-element convert: the mma consumes storage bits. The k64
            # 4-bit fragment map is byte-for-byte the k32 8-bit one — a lane owns 4 bytes per
            # register either way, and a packed row is K/2 bytes where an fp8 row is K — so
            # the same ``_b8v`` helpers serve both, with ``ldm`` carrying the byte stride.
            # Contiguous-K lanes load one u32; a canonical K-major B strides its 4 bytes
            # across rows — the scalar gather.
            if k_contig:
                trans = "_trans" if node.b_trans else ""
                call = f"emmy_mma_load_{node.role}_smem{trans}_b8v<{ctx.type_name(slab_dt)}>"
            else:
                call = "emmy_mma_load_b_gmem_b8"
        elif node.b_trans and frag_dt == "f16":
            # W8A16 pair convert: one fp8x2 load + one hardware cvt.rn.f16x2 per k-half.
            x2 = {"f8e4m3": "__nv_fp8x2_e4m3", "f8e5m2": "__nv_fp8x2_e5m2"}[slab_dt]
            call = f"emmy_mma_load_b_smem_trans_f8_f16<{ctx.type_name(slab_dt)}, {x2}>"
        else:
            # Canonical K-major B (K-strided bytes) or a bf16 fragment: the scalar
            # per-element convert via the gmem loader template, pointed at the slab.
            trans = "_trans" if node.b_trans else ""
            call = f"emmy_mma_load_{node.role}_gmem{trans}<{ctx.type_name(slab_dt)}, {ctx.type_name(frag_dt)}>"
        return [f"{_pad(ctx.indent)}{call}({node.frag}, &{node.src_buffer}[{flat}], {ldm});"]
    lane = "(threadIdx.x & 31)"
    if node.pair_frag is not None:
        # Paired x4 — two slab-adjacent B fragments in one ldmatrix (096_pair_ldmatrix_loads).
        # Transposed-B (N-major slab): the pair is N-adjacent — lanes 0-15 address ``frag``'s 8
        # N rows × two k8 col-halves, lanes 16-31 the same at N+8 (plain x4). Canonical-B
        # (K-major slab): the pair is N(col)-adjacent — lanes 0-15 address the 16 K rows at
        # ``frag``'s col, lanes 16-31 at col+8 (x4.trans).
        assert node.role == "b" and node.staged, "paired ldmatrix is a staged B-operand fusion"
        if node.b_trans:
            elem = f"(({lane} % 8) + ({lane} / 16) * 8) * {ldm} + (({lane} / 8) % 2) * 8"
            helper = "emmy_ldmatrix_x4_pair"
        else:
            elem = f"({lane} % 16) * {ldm} + ({lane} / 16) * 8"
            helper = "emmy_ldmatrix_x4_trans_pair"
        # The helper loads both fragments' registers directly (no ``_p4`` staging temp / block).
        return [f"{_pad(ctx.indent)}{helper}({node.frag}, {node.pair_frag}, {_swizzled_addr(node, flat, elem, ctx, 16, 16)});"]
    if node.role == "a":
        # 16×16 A: x4 — lane addresses M-row (lane%16), K-col block (lane/16)*8.
        addr = _swizzled_addr(node, flat, f"({lane} % 16) * {ldm} + ({lane} / 16) * 8", ctx, 16, 16)
        return [f"{_pad(ctx.indent)}emmy_ldmatrix_x4({node.frag}, {addr});"]
    # Transposed-B (Q@K^T): the slab keeps the operand's native N-major layout
    # (N rows × K cols), which IS the mma's col-major B — a plain x2 (no
    # .trans) with N-row addresses: lanes 0-7 address the 8 N rows of the k16
    # half 0, lanes 8-15 the same rows at K col 8 (each 8x8 matrix's rows land
    # as the fragment's (k, k+1) pairs, cf. ``emmy_mma_load_b_gmem_trans``).
    if node.b_trans:
        addr = _swizzled_addr(node, flat, f"({lane} % 8) * {ldm} + (({lane} / 8) % 2) * 8", ctx, 8, 16)
        return [f"{_pad(ctx.indent)}emmy_ldmatrix_x2({node.frag}, {addr});"]
    # 16×8 B: x2.trans — lane addresses K-row (lane%16); .trans yields col-major.
    addr = _swizzled_addr(node, flat, f"({lane} % 16) * {ldm}", ctx, 16, 8)
    return [f"{_pad(ctx.indent)}emmy_ldmatrix_x2_trans({node.frag}, {addr});"]


@dataclass(frozen=True)
class MmaSyncPtx(Declared):
    """``mma.sync.aligned.m{M}n{N}k{K}.row.col.f32.{ab}.{ab}.f32`` — one
    tensor-core MMA via inline PTX (the ``s16816`` instruction).

    ``c_frag`` is the accumulator array (both input and output —
    ``d = a·b + c``); ``a_frag`` / ``b_frag`` are the multiplicand
    arrays filled by :class:`LdmatrixLoad`. ``shape`` spells the M/N/K.
    ``ab_dtype`` (``"f16"`` / ``"bf16"`` / ``"e4m3"`` / ``"e5m2"`` — the PTX
    dtype field, :attr:`AtomKind.ab_dtype`) tags the operand element type —
    same-width dtypes share a fragment layout and differ only in the PTX
    instruction's dtype field, so the render just picks the
    matching ``emmy_mma_…`` wrapper. ``c_dtype`` is the accumulator element
    type: ``"f32"`` (the default — ``float c[4]``) or ``"f16"`` (the
    f16-accumulate atom — packed ``unsigned c[2]``, full HMMA rate on the
    consumer dies; the lowering pairs it with a periodic
    :class:`FragmentPromote` into an f32 shadow fragment)."""

    c_frag: str = role("updates")
    a_frag: str = role("reads")
    b_frag: str = role("reads")
    shape: tuple[int, int, int]
    ab_dtype: str = "f16"
    c_dtype: str = "f32"
    # The Volta canonical-B fast path uses the row-major B fragment mapping. Every other atom and
    # the Volta N-major B path keep the established column-major B mapping.
    b_row_major: bool = False
    # The BLOCK-SCALED form's two extra register operands (one packed b32 of ue4m3 block scales
    # per side), set together or not at all. Present only for the fp4 atom, whose instruction
    # applies the per-16-element scales itself; every other mma leaves them ``None``.
    sfa_frag: str | None = role("reads", default=None)
    sfb_frag: str | None = role("reads", default=None)

    def pretty(self, indent: str = "") -> list[str]:
        m, n, k = self.shape
        acc = "" if self.c_dtype == "f32" else f" acc:{self.c_dtype}"
        scales = f" scales:{self.sfa_frag},{self.sfb_frag}" if self.sfa_frag is not None else ""
        return [f"{indent}MmaSyncPtx {self.c_frag} += {self.a_frag} @ {self.b_frag} (m{m}n{n}k{k} {self.ab_dtype}{acc}{scales})"]


@render_stmt.register
def _(stmt: MmaSyncPtx, ctx: RenderCtx) -> list[str]:
    m, n, k = stmt.shape
    # Wrapper names follow the atom convention's <ab>_<acc> dtype pair (emmy_mma_m16n8k16_f16_f32).
    # ``c`` is passed for both the ``d`` (out) and ``c`` (in) operands.
    wrapper = f"emmy_mma_m{m}n{n}k{k}_{stmt.ab_dtype}_{stmt.c_dtype}"
    if stmt.b_row_major:
        assert (m, n, k) == (8, 8, 4), "only the Volta atom has a row-major B wrapper"
        wrapper += "_brow"
    if stmt.sfa_frag is not None:
        # The block-scaled wrapper accumulates in place, so it takes the scale registers
        # where the others take ``c`` a second time.
        args = f"{stmt.c_frag}, {stmt.a_frag}, {stmt.b_frag}, {stmt.sfa_frag}, {stmt.sfb_frag}"
        return [f"{_pad(ctx.indent)}{wrapper}({args});"]
    return [f"{_pad(ctx.indent)}{wrapper}({stmt.c_frag}, {stmt.a_frag}, {stmt.b_frag}, {stmt.c_frag});"]


@dataclass(frozen=True)
class FragmentPromote(Declared):
    """Fold an f16-accumulate mma **C fragment** into its f32 **shadow fragment** and rezero it —
    the chunked-accumulation promote of the ``_f16acc`` atom family: the mma chain accumulates at
    the full f16-accumulate HMMA rate, and every K chunk this promote-adds the packed f16 partials
    into the f32 shadow (``cvt.f32.f16`` + add per element) and re-zeros the f16 fragment, so the
    accumulation error stays bounded by one chunk's length. ``dst`` is the f32 shadow the store /
    epilogue reads; ``src`` is the packed f16 mma accumulator. The fragment layout selects
    their register counts and, on Volta, the warp shuffles that align their different lane maps. Both
    are defined here (``src`` is rezeroed), so reorderings keep the promote pinned between the
    mma chain and the store."""

    dst: str = role("updates")  # f32 shadow accumulator fragment — the store-side view
    src: str = role("updates")  # packed f16 mma accumulator fragment — rezeroed after the fold
    fragment_layout: str = "m16n8k16"

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}FragmentPromote {self.dst} += {self.src} (f16acc chunk fold, {self.src} rezeroed)"]


@render_stmt.register
def _(stmt: FragmentPromote, ctx: RenderCtx) -> list[str]:
    suffix = "_m8n8k4" if stmt.fragment_layout == "m8n8k4" else ""
    return [f"{_pad(ctx.indent)}emmy_mma_promote_f16acc{suffix}({stmt.dst}, {stmt.src});"]


@dataclass(frozen=True)
class FragmentRepack(Declared):
    """Convert mma **C fragments** into one 16-bit operand fragment in registers.

    The m16n8k16 layout takes two k-adjacent C fragments whose lanes already align with A. The
    Volta m8n8k4 layout takes one logical 16×16 C fragment and selects a four-column A slice
    or four-row B slice with warp shuffles. The emitter gates on the atom's C→A/C→B repack
    capabilities at schedule time.
    The m16n8k16 f16 B layout takes one C fragment and exchanges its packed column pairs
    between lanes. Every lane of the warp must participate in that exchange."""

    frag: str = role("defines")
    srcs: tuple[str, ...] = role("reads")
    ab_dtype: str = "f16"
    fragment_layout: str = "m16n8k16"
    part: int = 0
    role: str = "a"

    def pretty(self, indent: str = "") -> list[str]:
        role = ", role=b" if self.role == "b" else ""
        return [f"{indent}FragmentRepack {self.frag} <- {self.srcs} ({self.ab_dtype}, {self.fragment_layout}, part={self.part}{role})"]


@render_stmt.register
def _(stmt: FragmentRepack, ctx: RenderCtx) -> list[str]:
    assert stmt.role in ("a", "b")
    if stmt.fragment_layout == "m8n8k4":
        assert len(stmt.srcs) == 1 and stmt.ab_dtype == "f16"
        return [f"{_pad(ctx.indent)}emmy_c_to_{stmt.role}_f16_m8n8k4<{stmt.part}>({stmt.frag}, {stmt.srcs[0]});"]
    if stmt.role == "b":
        assert stmt.fragment_layout == "m16n8k16" and stmt.ab_dtype == "f16" and len(stmt.srcs) == 1
        return [f"{_pad(ctx.indent)}emmy_c_to_b_f16({stmt.frag}, {stmt.srcs[0]});"]
    assert len(stmt.srcs) == 2
    return [f"{_pad(ctx.indent)}emmy_c_to_a_{stmt.ab_dtype}({stmt.frag}, {stmt.srcs[0]}, {stmt.srcs[1]});"]


# ---------------------------------------------------------------------------
# Hopper warp-group MMA (``wgmma``): the asynchronous instruction discipline
# ---------------------------------------------------------------------------

#: PTX ``wgmma`` matrix-descriptor layout type (descriptor bits 62-63) per shared-memory swizzle
#: mode — the same modes the TMA fill and the ldmatrix drain are keyed on.
WGMMA_LAYOUT_TYPE: dict[str, int] = {"NONE": 0, "B128": 1, "B64": 2, "B32": 3}


def wgmma_descriptor_bits(addr: int, lbo: int, sbo: int, mode: int) -> int:
    """The 64-bit ``wgmma`` shared-memory matrix descriptor (PTX ISA "Matrix Descriptor Format"):
    bits 0-13 the start address >> 4, 16-29 the leading byte offset >> 4, 32-45 the stride byte
    offset >> 4, 49-51 the base offset (0 — every slab is aligned to its swizzle period), 62-63 the
    layout type (:data:`WGMMA_LAYOUT_TYPE`).

    The Python twin of ``emmy_wgmma_desc`` in the render prelude, which is what the kernel runs;
    the two spell one formula so a test can check the bits without a Hopper card."""
    return ((addr >> 4) & 0x3FFF) | (((lbo >> 4) & 0x3FFF) << 16) | (((sbo >> 4) & 0x3FFF) << 32) | (mode << 62)


@dataclass(frozen=True)
class WgmmaDescriptor(Declared):
    """``unsigned long long name = emmy_wgmma_desc(...)`` — the shared-memory matrix descriptor a
    ``wgmma.mma_async`` operand is read through (:func:`wgmma_descriptor_bits`). Replaces
    :class:`LdmatrixLoad` on a descriptor-fed operand: the tensor core reads the slab itself, so no
    per-lane registers are filled. ``smem_index`` (elements into ``smem``) selects the ring slot and
    the k step; ``lbo_bytes`` / ``sbo_bytes`` are the leading / stride byte offsets between core
    matrices, which the lowering derives from the slab's layout."""

    name: str = role("defines")
    smem: str = role("buffer_read")
    smem_index: Expr | None = role("expr")
    swizzle: str
    lbo_bytes: int
    sbo_bytes: int

    def pretty(self, indent: str = "") -> list[str]:
        idx = "" if self.smem_index is None else f"[{self.smem_index}]"
        return [f"{indent}WgmmaDescriptor {self.name} <- {self.smem}{idx} ({self.swizzle}, lbo={self.lbo_bytes}, sbo={self.sbo_bytes})"]


@render_stmt.register
def _(stmt: WgmmaDescriptor, ctx: RenderCtx) -> list[str]:
    mode = WGMMA_LAYOUT_TYPE[swizzle_base(stmt.swizzle)]
    src = stmt.smem if stmt.smem_index is None else f"&{stmt.smem}[{stmt.smem_index.render(ctx)}]"
    args = f"emmy_smem_u32({src}), {stmt.lbo_bytes}, {stmt.sbo_bytes}, {mode}"
    return [f"{_pad(ctx.indent)}unsigned long long {stmt.name} = emmy_wgmma_desc({args});"]


@dataclass(frozen=True)
class WgmmaMma(Declared):
    """``wgmma.mma_async.sync.aligned.m64n{N}k16.f32.{ab}.{ab}`` — one warp-group tensor-core MMA,
    issued asynchronously by the four warps of a group together.

    ``c_frags`` is the accumulator: per warp, the ``m64nN`` instruction's ``N / 2`` f32 registers
    are the ``N / 8`` ordinary ``m16n8`` C fragments (:class:`RegFragment`, ``float c[4]``) side by
    side — instruction register ``d[4j + k]`` is fragment ``j``, register ``k`` — so the epilogue,
    the softmax rescale and the P→A repack keep their lane maps. Exactly one of ``a_desc`` (the
    shared-memory form, ``_ss``) and ``a_frag`` (the register form, ``_rs`` — the ``m16n8k16`` A
    fragment, K-major only) names the A operand; ``b_desc`` is always a :class:`WgmmaDescriptor`.
    ``scale_d = 0`` discards the old accumulator (``d = a·b``, the first step of a fresh
    accumulator); ``trans_a`` / ``trans_b`` are the descriptor operands' MN-major transpose bits.

    The discipline around the cells (:class:`WgmmaFence`, :class:`WgmmaCommit`,
    :class:`WgmmaWait`) is the lowering's: nothing here orders the accumulator read."""

    c_frags: tuple[str, ...] = role("updates")
    b_desc: str = role("reads")
    shape: tuple[int, int, int]
    ab_dtype: str = "f16"
    a_desc: str | None = role("reads", default=None)
    a_frag: str | None = role("reads", default=None)
    scale_d: int = 1
    trans_a: int = 0
    trans_b: int = 0

    def __post_init__(self) -> None:
        m, n, k = self.shape
        if (m, k) != (64, 16) or n % 8 or not 8 <= n <= 256:
            raise ValueError(f"wgmma: the shape is m64n<8..256, a multiple of 8>k16, got {self.shape}")
        if len(self.c_frags) != n // 8:
            raise ValueError(f"wgmma m64n{n}: the accumulator is {n // 8} m16n8 C fragments, got {len(self.c_frags)}")
        if (self.a_desc is None) == (self.a_frag is None):
            raise ValueError("wgmma: exactly one of a_desc / a_frag names the A operand")
        if self.a_frag is not None and self.trans_a:
            raise ValueError("wgmma: the register-form A operand is K-major (no transpose bit)")

    def pretty(self, indent: str = "") -> list[str]:
        m, n, k = self.shape
        flags = ("" if self.scale_d else " scale_d=0") + (" trans_a" if self.trans_a else "") + (" trans_b" if self.trans_b else "")
        acc = ", ".join(self.c_frags)
        return [f"{indent}WgmmaMma {{{acc}}} += {_wgmma_a(self)} @ {self.b_desc} (m{m}n{n}k{k} {self.ab_dtype} {wgmma_form(self)}{flags})"]


def wgmma_form(stmt: WgmmaMma) -> str:
    """``"ss"`` (both operands through descriptors) or ``"rs"`` (A in registers)."""
    return "ss" if stmt.a_desc is not None else "rs"


def wgmma_wrapper(stmt: WgmmaMma) -> str:
    """The prelude wrapper this cell calls — one per ``(N, ab_dtype, form)`` a kernel uses."""
    return f"emmy_wgmma_m64n{stmt.shape[1]}k16_{stmt.ab_dtype}_f32_{wgmma_form(stmt)}"


def _wgmma_a(stmt: WgmmaMma) -> str:
    return stmt.a_desc if stmt.a_desc is not None else stmt.a_frag


@render_stmt.register
def _(stmt: WgmmaMma, ctx: RenderCtx) -> list[str]:
    # The transpose bits are PTX immediates, which nvcc accepts only as integral constant
    # expressions — template arguments of the wrapper, not function parameters.
    tnsp = f"<{stmt.trans_a}, {stmt.trans_b}>" if wgmma_form(stmt) == "ss" else f"<{stmt.trans_b}>"
    return [f"{_pad(ctx.indent)}{wgmma_wrapper(stmt)}{tnsp}({', '.join(stmt.c_frags)}, {_wgmma_a(stmt)}, {stmt.b_desc}, {stmt.scale_d});"]


@dataclass(frozen=True)
class WgmmaFence(Declared):
    """``wgmma.fence.sync.aligned`` — orders the group's earlier register and shared-memory accesses
    before the ``wgmma.mma_async`` that follows. The lowering issues it before the first cell after
    anything else touched the accumulator (the epilogue, the softmax rescale)."""

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}WgmmaFence"]


@render_stmt.register
def _(stmt: WgmmaFence, ctx: RenderCtx) -> list[str]:
    return [f'{_pad(ctx.indent)}asm volatile("wgmma.fence.sync.aligned;\\n" ::: "memory");']


@dataclass(frozen=True)
class WgmmaCommit(Declared):
    """``wgmma.commit_group.sync.aligned`` — closes the ``wgmma.mma_async`` cells issued since the
    last commit into one group for :class:`WgmmaWait` to count. The lowering commits after a
    chunk's cells."""

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}WgmmaCommit"]


@render_stmt.register
def _(stmt: WgmmaCommit, ctx: RenderCtx) -> list[str]:
    return [f'{_pad(ctx.indent)}asm volatile("wgmma.commit_group.sync.aligned;\\n" ::: "memory");']


@dataclass(frozen=True)
class WgmmaWait(Declared):
    """``wgmma.wait_group.sync.aligned N`` — block until at most ``group`` committed groups are still
    in flight. Carries no deps: the accumulator it completes is the ``c_frags`` of the cells before
    it, and the lowering places the wait before any read of those fragments and before the slot's
    release on the "empty" mbarrier. The redundant-sync and ldmatrix-pairing passes treat it as a
    barrier."""

    group: int = 0

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}WgmmaWait({self.group})"]


@render_stmt.register
def _(stmt: WgmmaWait, ctx: RenderCtx) -> list[str]:
    return [f'{_pad(ctx.indent)}asm volatile("wgmma.wait_group.sync.aligned {stmt.group};\\n" ::: "memory");']


@dataclass(frozen=True)
class RegStore(Declared):
    """Store an mma.sync f32 accumulator array to the output buffer with a
    per-lane epilogue downconvert.

    mma.sync has no ``store_matrix_sync`` — each lane owns 4 elements of
    the M×N=16×8 C tile (rows ``g`` / ``g+8`` with ``g = lane/4``, cols
    ``(lane%4)*2 + {0,1}``) and writes them directly. ``frag`` is the f32
    ``float c[4]``; when the destination buffer is narrower (``__half*``)
    each value is converted via ``__float2half``. ``ldm`` is the output row stride
    (N) — ``0`` auto-resolves from the buffer's inner extent.

    ``epilogue`` optionally carries a fused pointwise chain — residual adds, bias / scale
    broadcasts, activations, a coordinate ``Select`` — as a pure term over the ordinary ``Load`` /
    ``Assign`` / ``Select`` stmts of the projection tail, stored as its body: ``epilogue_params``
    bind, in order, the accumulator names to ``frag`` and ``extra_frags`` (a multi-fold node's
    additional C fragments, so the chain combines the channels per cell — SwiGLU et al.), every other
    name the body reads is a coordinate, and ``epilogue_result`` names the stored value. The lowering
    forms the body through ``Lambda.closing``, which fixes its statement order. Evaluated per fragment
    element right before the downconvert (the
    CUTLASS epilogue-visitor pattern): floating loads widen to f32, while integer loads retain
    their type for arithmetic on packed codes. Each param substitutes to that element, and the reserved
    :data:`ELEM_ROW` / :data:`ELEM_COL` vars in a ``Load`` index or ``Select`` predicate substitute
    to the element's row / col offset within the fragment (the cell base is already in the expression).

    ``m_guard`` / ``n_guard`` carry a masked-tile boundary as ``(base Expr,
    bound Expr)`` — the tile's row / col coordinate of fragment element (0,0)
    and the axis's real extent (a ``Literal`` for a static overhang axis, the
    symbolic ``Var`` for a runtime-sized one). The enclosing boundary ``Cond``
    only gates on the tile base (the atom-lane offsets ``_g`` / ``_t`` are
    render-local, invisible to σ), so an atom tile *straddling* the bound
    passes the Cond while its trailing rows / cols are out of range. A guard
    predicates each fragment element's store (and its epilogue gmem reads) at
    its own coordinate: ``base + _g (+8) < bound`` per row, ``base + _t*2
    (+1) < bound`` per col. ``None`` (the default) renders the unguarded
    fast path unchanged. ``m_guard`` alone keeps the vectorized row-pair
    stores (a pair shares a row); an ``n_guard`` forces per-element scalar
    stores (the pair straddles the column bound).

    ``row_dim`` / ``col_dim`` identify the output-index dimensions carrying the mma fragment's
    algebraic M/N axes. Their row-major strides may be either physical orientation: a contraction
    whose shared operand varies along the output's trailing axis stores a transposed fragment with
    scalar strided writes, while the ordinary contiguous-N case keeps packed pairs.

    A staged Volta tile using paired operand layouts also derives ``volta_interleaved`` and its
    register-cell ``fragment_index``. Together they select the coupled 32×32 accumulator map;
    neither is a schedule choice or a wire-codec field.

    ``atomic`` renders each store as an ``atomicAdd`` accumulate instead of a
    plain assign — ``030_cut``'s atomic finalize on the mma tier: every
    split partition's C fragment adds into the (per-launch zero-init'd) output.
    f16 / bf16 keep the vectorized row pair (native packed-pair ``atomicAdd``);
    an f32 destination has no packed atomic below sm_90, so it drops to
    per-element scalar ``atomicAdd``\\ s."""

    dst_buffer: str = role("buffer_write")
    dst_index: tuple[Expr, ...] = role("expr")
    frag: str = role("reads")
    shape: tuple[int, int, int]
    ldm: int = 0
    ldn: int = 0
    epilogue: Body | None = role("term", default=None)
    epilogue_params: tuple[str, ...] = ()
    epilogue_result: str | None = None
    extra_frags: tuple[str, ...] = role("reads", default=())
    m_guard: tuple[Expr, Expr] | None = role("expr", default=None)
    n_guard: tuple[Expr, Expr] | None = role("expr", default=None)
    atomic: bool = False
    fragment_layout: str = "m16n8k16"
    # The paired Volta operand layouts require CUTLASS's interleaved 32x32 accumulator map. This
    # is a derived SM70 lowering choice, not another atom fragment-layout or schedule spelling.
    volta_interleaved: bool = False
    # Cell position inside the warp register tile. The paired Volta layout groups adjacent cells
    # into CUTLASS's interleaved 32x32 accumulator map; other fragment layouts ignore it.
    fragment_index: tuple[int, int] = (0, 0)
    # Software smem slab swizzle mode ("NONE"/"B32"/"B64"/"B128") — the fragment fill's producer
    # side, when this store's destination is a swizzled smem SLAB rather than the kernel's output
    # (the fused flash weight tile: the score's C fragments land in the A slab the ``ldmatrix``
    # drain reads). The flattened store address passes through ``emmy_swizzle_<mode>`` exactly like
    # a :class:`CpAsyncCopy` fill, so the drain's identical XOR reads the tile back. The lane's two
    # columns are contiguous and even-based, so the XOR (which passes intra-chunk bits 0..2
    # through) keeps a vectorized pair inside its relocated 16-byte chunk. "NONE" is inert.
    swizzle: str = "NONE"
    # The ``dst_index`` position carrying the M (row) coordinate, set by the store's creator. The
    # auto ``ldm`` is the row stride — the product of the buffer extents AFTER this dim — which
    # equals the inner extent only while N is exactly the last dim; a re-fused split store
    # (``[…, m, n/Q, n%Q]``) spans N over the trailing dims, so ``shape[-1]`` under-strides the
    # fragment row offset. ``None`` keeps the legacy inner-extent resolution.
    row_dim: int | None = None
    col_dim: int | None = None
    # The next three cells along N (8 columns apart each), stored WITH this one (the
    # ``097_widen_fragment_stores`` peephole): the quad's four lanes trade their column pairs so
    # each lane writes one cell's eight columns of a row as a single 16-byte store — a quarter of
    # the stores, and every one of them a full sector. Each member keeps its own fragment and
    # epilogue; only the store is shared.
    run: tuple[RegStore, ...] = role("stmts", default=())

    def pretty(self, indent: str = "") -> list[str]:
        idx = ", ".join(str(e) for e in self.dst_index)
        epi = ""
        if self.epilogue is not None:
            chain = ", ".join(st.op.name for st in self.epilogue if isinstance(st, Assign))
            bufs = ", ".join(self.external_reads())
            epi = f" epilogue[{chain}]({bufs or 'no loads'})"
        guards = ""
        if self.m_guard is not None:
            guards += f" m<{self.m_guard[1]}"
        if self.n_guard is not None:
            guards += f" n<{self.n_guard[1]}"
        acc = " (atomic)" if self.atomic else ""
        run = "" if not self.run else f" +{','.join(cell.frag for cell in self.run)} (16-byte rows)"
        return [f"{indent}RegStore {self.dst_buffer}[{idx}] <- {self.frag}{run}{epi}{guards}{acc} (ldm={self.ldm or 'auto'})"]


def _store_swz(node: RegStore, addr: str) -> str:
    """The store address, XOR-permuted through the slab's swizzle helper when this store
    targets a swizzled smem slab — inert at the default ``"NONE"``."""
    return addr if not swizzle_xor(node.swizzle) else f"{swizzle_fn(node.swizzle)}({addr})"


def _pair_store(node: RegStore, addr: str, v0: str, v1: str, vec2: str, packer: str) -> str:
    """One vectorized row-pair store line — a plain assign, or (``atomic``) the packed-pair
    ``atomicAdd`` (native f16x2 / bf16x2 red; the caller keeps f32 off this path pre-sm90)."""
    tgt = f"reinterpret_cast<{vec2}*>(&{node.dst_buffer}[{_store_swz(node, addr)}])"
    if node.atomic:
        return f"atomicAdd({tgt}, {packer}({v0}, {v1}));"
    return f"*{tgt} = {packer}({v0}, {v1});"


def _elem_store(node: RegStore, addr: str, val: str, dst_dt: str) -> str:
    """One per-element store line — plain assign (the implicit f32→dst conversion), or
    (``atomic``) a scalar ``atomicAdd`` with the conversion made explicit (no float overload
    on a ``__half*`` / ``__nv_bfloat16*`` destination)."""
    if node.atomic:
        conv = {"f16": "__float2half({})", "bf16": "__float2bfloat16({})"}.get(dst_dt, "{}")
        return f"atomicAdd(&{node.dst_buffer}[{addr}], {conv.format(val)});"
    return f"{node.dst_buffer}[{_store_swz(node, addr)}] = {val};"


def _element_coords(node: RegStore) -> list[tuple[str, str, Expr, Expr]]:
    """Per-lane ``(row C text, col C text, row Expr, col Expr)`` for this fragment layout."""
    from emmy.compiler.ir.expr import BinaryExpr, Var  # noqa: PLC0415

    if node.volta_interleaved:
        mi, ni = (x & 1 for x in node.fragment_index)
        row_adjust, col_adjust = -12 * mi, -12 * ni

        def coord(base: str, offset: int) -> tuple[str, Expr]:
            if offset == 0:
                return base, Var(base)
            op = "+" if offset > 0 else "-"
            amount = abs(offset)
            return f"({base} {op} {amount})", BinaryExpr(op, Var(base), Literal(amount, "int"))

        out = []
        for i in range(8):
            row, row_expr = coord("_vr", row_adjust + (i & 2))
            col, col_expr = coord("_vc", col_adjust + (16 if i & 4 else 0) + (i & 1))
            out.append((row, col, row_expr, col_expr))
        return out
    if node.fragment_layout == "m8n8k4":
        out = []
        for i in range(8):
            ro, co = i & 2, (i & 4) + (i & 1)
            row = Var("_vr") if ro == 0 else BinaryExpr("+", Var("_vr"), Literal(ro, "int"))
            col = Var("_vc") if co == 0 else BinaryExpr("+", Var("_vc"), Literal(co, "int"))
            out.append(("_vr" if ro == 0 else f"(_vr + {ro})", "_vc" if co == 0 else f"(_vc + {co})", row, col))
        return out
    return [
        (
            "_g" if i < 2 else "(_g + 8)",
            f"(_t * 2 + {i & 1})",
            Var("_g") if i < 2 else BinaryExpr("+", Var("_g"), Literal(8, "int")),
            BinaryExpr("+", BinaryExpr("*", Var("_t"), Literal(2, "int")), Literal(i & 1, "int")),
        )
        for i in range(4)
    ]


def _element_values(node: RegStore, ctx: RenderCtx) -> tuple[list[list[str]], list[str]]:
    """``(per_element_preamble, values)``: the four per-lane store values,
    plus, per element, the leaf-load / chain-op declaration lines that
    element needs (so a guarded render can scope element ``i``'s gmem
    reads inside element ``i``'s boundary check). Without
    an epilogue the values are the bare ``frag[i]`` and the preambles are
    empty. With one, each element ``i`` (row ``_g``/``_g+8``, col
    ``2_t+{0,1}``) declares its leaf loads at the element's own coordinates
    (floating values widened to f32, integer values kept as integers) and the
    chain ops (via the scalar ``Assign`` renderer), all scoped inside the
    store's ``{ }`` block. Leaf loads are scalar; lanes ``_t = 0..3`` cover
    8 contiguous columns, so the warp's
    accesses coalesce regardless."""
    coords = _element_coords(node)
    if node.epilogue is None:
        return [[] for _ in coords], [f"{node.frag}[{i}]" for i in range(len(coords))]
    conv = {"f16": "__half2float({})", "bf16": "__bfloat162float({})"}
    frags = (node.frag, *node.extra_frags)
    per_elem: list[list[str]] = []
    vals: list[str] = []
    for i, (_row, _col, row_off, col_off) in enumerate(coords):
        lines: list[str] = []
        env = {acc: f"{fr}[{i}]" for acc, fr in zip(node.epilogue_params, frags, strict=True)}
        for value in env.values():
            ctx.ssa_dtypes[value] = "f32"
        # The reserved ``ELEM_ROW`` / ``ELEM_COL`` vars substitute to this element's row / col offset
        # within the fragment; a Load index and a Select predicate already carry the cell base.
        coord = {ELEM_ROW: row_off, ELEM_COL: col_off}
        for st in node.epilogue:
            if isinstance(st, Load):
                temp = f"{st.name}_e{i}"
                storage_dt = ctx.buffer_dtypes.get(st.input, "f32")
                value_dt = st.dtype.name if st.dtype is not None else storage_dt
                integer = value_dt in INTEGER_NAMES
                if st.input in ctx.literal_constants:
                    if integer:
                        lines.append(f"const {ctx.type_name(value_dt)} {temp} = {int(ctx.literal_constants[st.input])};")
                    else:
                        lines.append(f"const float {temp} = {float(ctx.literal_constants[st.input])!r}f;")
                else:
                    flat = render_index(st.input, tuple(e.substitute(coord) for e in st.index), ctx)
                    if integer:
                        lines.append(f"const {ctx.type_name(value_dt)} {temp} = {st.input}[{flat}];")
                    else:
                        lines.append(f"const float {temp} = {conv.get(storage_dt, '{}').format(f'{st.input}[{flat}]')};")
                env[st.name] = temp
                ctx.ssa_dtypes[temp] = value_dt if integer else "f32"
            elif isinstance(st, Select):
                # The select is declared f32, and a chain op keeps the tail's own dtype, so a
                # branch value narrowed by an earlier op converts back here — a ternary over a
                # ``__half`` and a ``float`` does not compile.
                def widened(value, ctx=ctx, env=env):
                    rendered = env[value]
                    return conv.get(ctx.ssa_dtypes.get(rendered, "f32"), "{}").format(rendered)

                expr = widened(st.branches[-1].value)
                for br in reversed(st.branches[:-1]):
                    rc = br.select.substitute(coord).render(ctx)
                    expr = f"(({rc}) ? {widened(br.value)} : {expr})"
                lines.append(f"const float {st.name}_e{i} = {expr};")
                env[st.name] = f"{st.name}_e{i}"
                ctx.ssa_dtypes[env[st.name]] = "f32"
            else:
                local = st.rewrite(
                    lambda name, st=st, i=i, env=env: f"{name}_e{i}" if name == st.name else env.get(name, name), Sigma(coord)
                )
                rendered = render_stmt(local, replace(ctx, indent=0))[0]
                lines.append(f"const {rendered}")
                env[st.name] = f"{st.name}_e{i}"
        per_elem.append(lines)
        vals.append(env[node.epilogue_result])
    return per_elem, vals


def _scaled_offset(term: str, stride: int | str) -> str:
    return term if stride == 1 else f"({term}) * {stride}"


def _store_addr(node: RegStore, flat: str, row: str, col: str, row_stride: int | str, col_stride: int | str) -> str:
    """Flat address of one fragment element under its independently derived M/N strides."""
    return f"{flat} + {_scaled_offset(row, row_stride)} + {_scaled_offset(col, col_stride)}"


def _render_run(node: RegStore, ctx: RenderCtx, *, flat: str, ldm, dst_dt: str) -> list[str] | None:
    """The four cells of :attr:`run` as one 16-byte store per lane and row, or ``None`` for a
    destination that is not 16-bit or whose row stride breaks the 16-byte alignment.

    Lane ``t`` of a quad holds columns ``2t, 2t+1`` of each cell. Three ``shfl.xor`` rounds
    transpose the quad's 4×4 grid of packed pairs: in round ``s`` a lane sends the pair of
    cell ``t^s`` and receives, from lane ``t^s``, that lane's pair of cell ``t``. Lane ``t``
    then holds all eight columns of cell ``t`` and writes them at ``8t`` past the first
    cell. The quad runs whole, as the fragment stores it replaces already assume."""
    vec2 = {"f16": "__half2", "bf16": "__nv_bfloat162"}.get(dst_dt)
    if vec2 is None or not isinstance(ldm, int) or ldm % 8:
        return None
    packer = {"f16": "__floats2half2_rn", "bf16": "__floats2bfloat162_rn"}[dst_dt]
    pad = _pad(ctx.indent)
    cells = (node, *node.run)
    names = [[f"_wp{r}_{k}" for k in range(len(cells))] for r in (0, 1)]
    lines = [
        f"{pad}{{ const int _g = (threadIdx.x & 31) >> 2; const int _t = (threadIdx.x & 31) & 3;",
        f"{pad}  unsigned {', '.join(n for row in names for n in row)};",
    ]
    for k, cell in enumerate(cells):
        pre, vals = _element_values(cell, ctx)
        lines.append(f"{pad}  {{")
        lines += [f"{pad}    {ln}" for group in pre for ln in group]
        for r in (0, 1):
            pair = f"{packer}({vals[2 * r]}, {vals[2 * r + 1]})"
            lines.append(f"{pad}    {{ {vec2} _h = {pair}; {names[r][k]} = *reinterpret_cast<unsigned*>(&_h); }}")
        lines.append(f"{pad}  }}")

    def pick(*by_lane: str) -> str:
        return f"(_t == 0 ? {by_lane[0]} : _t == 1 ? {by_lane[1]} : _t == 2 ? {by_lane[2]} : {by_lane[3]})"

    for r, row in ((0, "_g"), (1, "(_g + 8)")):
        p0, p1, p2, p3 = names[r]
        q1, q2, q3 = (f"_wq{r}_{s}" for s in (1, 2, 3))
        lines += [
            f"{pad}  const unsigned {q1} = __shfl_xor_sync(0xffffffffu, {pick(p1, p0, p3, p2)}, 1);",
            f"{pad}  const unsigned {q2} = __shfl_xor_sync(0xffffffffu, {pick(p2, p3, p0, p1)}, 2);",
            f"{pad}  const unsigned {q3} = __shfl_xor_sync(0xffffffffu, {pick(p3, p2, p1, p0)}, 3);",
            f"{pad}  *reinterpret_cast<uint4*>(&{node.dst_buffer}[{flat} + {row} * {ldm} + _t * 8]) = make_uint4("
            f"{pick(p0, q1, q2, q3)}, {pick(q1, p1, q3, q2)}, {pick(q2, q3, p2, q1)}, {pick(q3, q2, q1, p3)});",
        ]
    lines.append(f"{pad}}}")
    return lines


def _render_m8n8k4(node: RegStore, ctx: RenderCtx, *, flat: str, ldm, ldn, dst_dt: str, pre: list[list[str]], vals: list[str]) -> list[str]:
    """Store an m8n8k4 accumulator under the selected Volta warp-tile arrangement."""
    pad = _pad(ctx.indent)
    lane = "(threadIdx.x & 31)"
    # Each four-lane group and its +16 partner own one 8x8 computation. _vr/_vc are the lane's
    # base coordinates under either the ordinary 16x16 cell or paired 32x32 warp-tile map.
    if node.volta_interleaved:
        lines = [
            f"{pad}{{ const int _vl = {lane}; const int _vq = _vl >> 2;",
            f"{pad}  const int _vr = (((_vq & 4) >> 1) + (_vq & 1)) * 8 + (_vl & 1);",
            f"{pad}  const int _vc = ((_vq >> 1) & 1) * 8 + (_vl & 2);",
        ]
    else:
        lines = [
            f"{pad}{{ const int _vl = {lane}; const int _vq = (_vl & 15) >> 2;",
            f"{pad}  const int _vr = (_vq >> 1) * 8 + (_vl >> 4) * 4 + (_vl & 1);",
            f"{pad}  const int _vc = (_vq & 1) * 8 + (_vl & 2);",
        ]
    mbase = mbound = nbase = nbound = None
    if node.m_guard is not None:
        mbase, mbound = (e.render(ctx) for e in node.m_guard)
    if node.n_guard is not None:
        nbase, nbound = (e.render(ctx) for e in node.n_guard)
    vec2 = {"f16": "__half2", "bf16": "__nv_bfloat162", "f32": "float2"}.get(dst_dt)
    if nbound is None and ldn == 1 and vec2 is not None and not (node.atomic and dst_dt == "f32"):
        packer = {"f16": "__floats2half2_rn", "bf16": "__floats2bfloat162_rn", "f32": "make_float2"}[dst_dt]
        coords = _element_coords(node)
        for pair in range(0, len(coords), 2):
            row, col, _row_expr, _col_expr = coords[pair]
            indent = f"{pad}  "
            if mbound is not None:
                lines.append(f"{indent}if (({mbase}) + {row} < ({mbound})) {{")
                indent += "  "
            lines.extend(f"{indent}{ln}" for ln in (*pre[pair], *pre[pair + 1]))
            addr = _store_addr(node, flat, row, col, ldm, ldn)
            lines.append(f"{indent}{_pair_store(node, addr, vals[pair], vals[pair + 1], vec2, packer)}")
            if mbound is not None:
                lines.append(f"{pad}  }}")
        lines.append(f"{pad}}}")
        return lines
    for i, (row, col, _row_expr, _col_expr) in enumerate(_element_coords(node)):
        preds = []
        if mbound is not None:
            preds.append(f"({mbase}) + {row} < ({mbound})")
        if nbound is not None:
            preds.append(f"({nbase}) + {col} < ({nbound})")
        indent = f"{pad}  "
        if preds:
            lines.append(f"{indent}if ({' && '.join(preds)}) {{")
            indent += "  "
        lines.extend(f"{indent}{ln}" for ln in pre[i])
        addr = _store_addr(node, flat, row, col, ldm, ldn)
        lines.append(f"{indent}{_elem_store(node, addr, vals[i], dst_dt)}")
        if preds:
            lines.append(f"{pad}  }}")
    lines.append(f"{pad}}}")
    return lines


def _render_guarded(
    node: RegStore, ctx: RenderCtx, *, flat: str, ldm, ldn, dst_dt: str, pre: list[list[str]], vals: list[str]
) -> list[str]:
    """Masked-tile store: each fragment element's store (and its epilogue
    gmem reads, which index the same possibly-out-of-range coordinates)
    runs under that element's own boundary check. With only an ``m_guard``
    the two columns of a row share the row predicate, so the vectorized
    row-pair store survives; an ``n_guard`` splits the pair (its columns
    straddle the bound) into per-element scalar stores."""
    pad = _pad(ctx.indent)
    lane = "(threadIdx.x & 31)"
    m_pred: list[str | None] = [None] * 4
    n_pred: list[str | None] = [None] * 4
    if node.m_guard is not None:
        base, bound = node.m_guard[0].render(ctx), node.m_guard[1].render(ctx)
        m_pred = [f"({base}) + _g < ({bound})"] * 2 + [f"({base}) + _g + 8 < ({bound})"] * 2
    if node.n_guard is not None:
        base, bound = node.n_guard[0].render(ctx), node.n_guard[1].render(ctx)
        n_pred = [
            f"({base}) + _t * 2 < ({bound})",
            f"({base}) + _t * 2 + 1 < ({bound})",
            f"({base}) + _t * 2 < ({bound})",
            f"({base}) + _t * 2 + 1 < ({bound})",
        ]
    lines = [f"{pad}{{ const int _g = {lane} >> 2; const int _t = {lane} & 3;"]
    vec2 = {"f16": "__half2", "bf16": "__nv_bfloat162", "f32": "float2"}.get(dst_dt)
    if node.n_guard is None and ldn == 1 and vec2 is not None and not (node.atomic and dst_dt == "f32"):
        # Row-guarded vectorized pairs: elements {0,1} share row _g,
        # {2,3} share row _g+8; each pair's preamble + store sit inside
        # the pair's row check.
        packer = {"f16": "__floats2half2_rn", "bf16": "__floats2bfloat162_rn", "f32": "make_float2"}[dst_dt]
        for pair, addr_row in ((0, "_g"), (2, "(_g + 8)")):
            addr = f"{flat} + {addr_row} * {ldm} + _t * 2"
            lines.append(f"{pad}  if ({m_pred[pair]}) {{")
            lines += [f"{pad}    {ln}" for ln in (*pre[pair], *pre[pair + 1])]
            lines.append(f"{pad}    {_pair_store(node, addr, vals[pair], vals[pair + 1], vec2, packer)}")
            lines.append(f"{pad}  }}")
        lines.append(f"{pad}}}")
        return lines
    # Per-element scalar stores under the conjunction of the live guards.
    for i in range(4):
        row = "_g" if i < 2 else "(_g + 8)"
        preds = " && ".join(p for p in (m_pred[i], n_pred[i]) if p is not None)
        addr = _store_addr(node, flat, row, f"_t * 2 + {i & 1}", ldm, ldn)
        lines.append(f"{pad}  if ({preds}) {{")
        lines += [f"{pad}    {ln}" for ln in pre[i]]
        lines.append(f"{pad}    {_elem_store(node, addr, vals[i], dst_dt)}")
        lines.append(f"{pad}  }}")
    lines.append(f"{pad}}}")
    return lines


@render_stmt.register
def _(node: RegStore, ctx: RenderCtx) -> list[str]:
    flat = render_index(node.dst_buffer, node.dst_index, ctx)
    ldm = node.ldm if node.ldm else _resolve_ldm(node.dst_buffer, ctx, node.row_dim)
    ldn = node.ldn or (_dim_stride(node.dst_buffer, node.col_dim, ctx) if node.col_dim is not None else 1)
    dst_dt = ctx.buffer_dtypes.get(node.dst_buffer, "f32")
    pad = _pad(ctx.indent)
    lane = "(threadIdx.x & 31)"
    pre, vals = _element_values(node, ctx)
    if node.fragment_layout == "m8n8k4":
        return _render_m8n8k4(node, ctx, flat=flat, ldm=ldm, ldn=ldn, dst_dt=dst_dt, pre=pre, vals=vals)
    # C is 16×8: lane owns (row g, cols 2t,2t+1) and (row g+8, cols 2t,2t+1)
    # with g = lane/4, t = lane%4. The two cols per row are CONTIGUOUS, so
    # each row's pair is one vectorized 4-byte store (``__half2`` / ``float2``)
    # rather than two scalar stores — halves the epilogue store count and
    # the address arithmetic. ldm strides over N (the output row width). The
    # base ``flat`` is tile-aligned and ``2t`` is even, so the pair is
    # 4-/8-byte aligned. The ``{ }`` block scopes _g/_t (and the per-element
    # epilogue temps) per RegStore.
    if node.run and (wide := _render_run(node, ctx, flat=flat, ldm=ldm, dst_dt=dst_dt)) is not None:
        return wide
    if node.run:  # a destination the 16-byte row cannot serve stores each cell on its own
        return [ln for cell in (node.replace(run=()), *node.run) for ln in render_stmt(cell, ctx)]
    if node.m_guard is not None or node.n_guard is not None:
        return _render_guarded(node, ctx, flat=flat, ldm=ldm, ldn=ldn, dst_dt=dst_dt, pre=pre, vals=vals)
    lane_stmt = f"const int _g = {lane} >> 2; const int _t = {lane} & 3;"
    if node.epilogue is None:
        # Flash output store: no per-element epilogue temps, so declare _g/_t once per scope
        # (reused by sibling stores) and emit the stores flat — no wrapping { } block, so a
        # store re-rolled into a loop renders ``for (…) { … }`` not ``for (…) { { … } }``.
        head, base, close = _lane_preamble(ctx, pad, lane_stmt), pad, ""
    else:
        # Fused epilogue: the { } scopes the per-element temps (they collide across sibling
        # register-tile cells, which reuse the same names), so it stays.
        head, base, close = [f"{pad}{{ {lane_stmt}"], f"{pad}  ", " }"
    body = [f"{base}{ln}" for group in pre for ln in group]
    vec2 = {"f16": "__half2", "bf16": "__nv_bfloat162", "f32": "float2"}.get(dst_dt)
    if ldn == 1 and vec2 is not None and not (node.atomic and dst_dt == "f32"):  # no packed f32 atomicAdd pre-sm90
        packer = {"f16": "__floats2half2_rn", "bf16": "__floats2bfloat162_rn", "f32": "make_float2"}[dst_dt]
        lo = f"{flat} + _g * {ldm} + _t * 2"
        hi = f"{flat} + (_g + 8) * {ldm} + _t * 2"
        body += [
            f"{base}{_pair_store(node, lo, vals[0], vals[1], vec2, packer)}",
            f"{base}{_pair_store(node, hi, vals[2], vals[3], vec2, packer)}",
        ]
    else:  # per-element scalar stores (dtypes without a 2-vector packer; atomic f32)
        coords = (("_g", "_t * 2 + 0"), ("_g", "_t * 2 + 1"), ("(_g + 8)", "_t * 2 + 0"), ("(_g + 8)", "_t * 2 + 1"))
        body += [
            f"{base}{_elem_store(node, _store_addr(node, flat, row, col, ldm, ldn), vals[i], dst_dt)}"
            for i, (row, col) in enumerate(coords)
        ]
    if close:
        body[-1] += close
    return head + body


def _ext_to_c(ext, ctx: RenderCtx) -> int | str:
    """One buffer-shape extent as a C term: static dims yield their ``int``,
    a symbolic ``Dim`` renders its Expr against the runtime kernel arg
    (parenthesized — the result is interpolated into ``* {…}`` address
    strings)."""
    if hasattr(ext, "is_static"):
        if ext.is_static:
            return ext.as_static()
        return f"({ext.expr.render(ctx)})"
    return int(ext)


def _dim_stride(buffer: str, dim: int, ctx: RenderCtx) -> int | str:
    """Row-major element stride of ``buffer``'s dim ``dim`` — the product of
    the trailing extents. Used by the fragment-epilogue render to apply a
    fragment element's row / col motion on an operand dim at that operand's
    own layout (a transposed residual's "n" dim strides by its row width,
    not by 1). A symbolic trailing extent makes the stride a C expression
    string over the runtime kernel arg instead of an int."""
    shape = ctx.shapes.get(buffer)
    if shape is None:
        raise ValueError(f"RegStore epilogue: buffer {buffer!r} not in ctx.shapes (no shape registered)")
    static = 1
    sym: list[str] = []
    for ext in shape[dim + 1 :]:
        term = _ext_to_c(ext, ctx)
        if isinstance(term, int):
            static *= term
        else:
            sym.append(term)
    if not sym:
        return static
    expr = " * ".join(sym if static == 1 else [str(static), *sym])
    return f"({expr})"


def _resolve_ldm(buffer: str, ctx: RenderCtx, row_dim: int | None = None) -> int | str:
    """Look up the row-major row stride for ``buffer`` from the kernel render context. Used by
    :class:`LdmatrixLoad` / :class:`RegStore` when ``ldm == 0`` (auto).
    With ``row_dim`` (the index position carrying the M coordinate — ``RegStore.row_dim``) the
    stride is the product of every extent after that dim, which stays correct when the N
    coordinate spans several trailing dims (the re-fused split store); without it, the legacy
    inner-extent read. Accepts both raw int extents and :class:`Dim` extents (Tensor.shape) —
    a symbolic inner extent (e.g. QK^T's ``(seq, seq)`` output) resolves to
    a C expression over the runtime kernel arg (the M9 runtime-ldm path);
    every consumer interpolates ``ldm`` into an address string, so the
    int/str split is transparent."""
    if row_dim is not None:
        return _dim_stride(buffer, row_dim, ctx)
    shape = ctx.shapes.get(buffer)
    if shape is None:
        raise ValueError(f"_resolve_ldm: buffer {buffer!r} not in ctx.shapes (no shape registered)")
    return _ext_to_c(shape[-1], ctx)


def _binary_combine_expr(op: ElementwiseImpl, a: str, b: str, target=None, dt: str = "f32") -> str:
    """Render a 2-arg combine for ``ElementwiseImpl`` reduce ops at ``dt``.

    ``target`` (optional) provides the dtype-specific intrinsic spelling.
    ``None`` keeps the legacy f32 spellings for callers that haven't
    plumbed a target through yet.
    """
    spelling = _REDUCE_SPELLING.get(op.reduce_canon)
    if spelling is None:
        raise ValueError(f"TreeHalve: unsupported op {op.name!r}")
    if spelling.infix is not None:
        return f"{a} {spelling.infix} {b}"
    fn = target.intrinsic(spelling.intrinsic, dt) if target is not None else f"{spelling.intrinsic}f"
    return f"{fn}({a}, {b})"


# ``StridedLoop`` is shared infrastructure — defined in ``ir/stmt.py``
# and re-exported here. Used at Tile IR for cooperative iteration and
# at Kernel IR for cooperative smem loads.


# ---------------------------------------------------------------------------
# Top-level: KernelOp
# ---------------------------------------------------------------------------


# Hard ceiling on the launch grid for ``KernelOp.validate``. The CUDA driver
# allows ~2^31 CTAs per dim; the cap here is sized to allow any realistic
# matmul launch (including K-split fan-in, which multiplies CTA count by
# the split factor — e.g. (M=32, K=18944, N=3584) with BM=BN=16 and
# auto-splitK=37 produces 2 × 224 × 37 ≈ 17 k CTAs of heavy per-CTA work),
# while still rejecting truly degenerate launches that would saturate
# the GPU command processor with light per-CTA work. The autotune-side
# guard against pathological tiny-CTA × huge-grid variants is the
# enumeration gate (``_enumeration._matmul_thread_gate``) + the online
# prior, not this cap.
_MAX_CTAS = 65536


def pack_smem(smems) -> tuple[dict[str, int], int]:  # noqa: ANN001 — smems: Iterable[Smem]
    """Pack ``Smem`` buffers into one contiguous pool, padding each to its
    alignment. Walks the buffers in order, aligns the running cursor to
    ``max(nbytes_of(dtype), s.align)`` before placing each buffer, and returns
    ``({name: byte_offset}, total_padded_bytes)``.

    Single source of truth for the per-CTA smem footprint: ``KernelOp.smem_bytes``
    and ``render._compute_dynamic_smem_offsets`` both call this so the
    static-vs-dynamic gate and the launch-time pool size agree (an unpadded sum
    would under-report when a buffer's alignment exceeds its natural stride —
    e.g. a 512/1024 B swizzle-atom-aligned operand)."""
    from math import prod  # noqa: PLC0415

    from emmy.compiler.backend.cuda.dtype import nbytes_of  # noqa: PLC0415

    offsets: dict[str, int] = {}
    cursor = 0
    for s in smems:
        elements = prod(int(e) for e in s.extents) if s.extents else 1
        align = max(nbytes_of(s.dtype), int(s.align) if s.align else 0)
        if align:
            cursor = (cursor + align - 1) // align * align
        offsets[s.name] = cursor
        cursor += elements * nbytes_of(s.dtype)
    return offsets, cursor


@dataclass(frozen=True)
class KernelOp(BodyOp):
    """One ``__global__`` GPU kernel as a Kernel IR program.

    :class:`BodyOp` subclass parallel to ``TileOp`` / ``LoopOp``: lives as
    a graph node, carries a body of Kernel IR stmts plus a kernel name.

    Buffer shapes are *not* baked in — the surrounding graph supplies
    them at render time, same as ``TileOp``. Kernel signature is derived
    from the body: distinct ``Load.input`` names become kernel input
    params, distinct ``Write.output`` names become writeable output
    params, ordered by first appearance. ``Smem`` buffers are excluded.
    ``CpAsyncCopy.src`` and ``TmaDescriptor.src_buf`` also name kernel
    input parameters (the descriptor parameter itself is host-built and
    appended by the CUDA backend's argument pipeline, not a graph
    buffer).

    ``zero_delegated`` lists atomic-accumulator buffers whose per-launch zero-init a
    dataflow-PREDECESSOR kernel carries as a ``ZeroPrologue`` stmt
    (``lowering/cuda/005_delegate_zero_init``) — ``010_lower_kernelop`` drops them from the
    ``CudaOp.zero_outputs`` memset list."""

    zero_delegated: tuple[str, ...] = ()
    #: The placement's serial axes (:attr:`Placement.serial`): one launch per coordinate, the
    #: coordinate a runtime ``int`` param. Carried to the CUDA op, never a loop in the body.
    serial: tuple[Axis, ...] = ()

    @cached_property
    def smem_buffers(self) -> dict[str, Smem]:
        """Every ``Smem`` decl in the body, keyed by name. Cached — KernelOp
        is treated as immutable post-construction (passes return new
        instances), so the deep walk runs at most once per op. The deep
        walk matters because ``Smem`` stmts sit inside ``Tile`` / ``Loop``
        bodies post-lowering; a top-level-only iterator would miss them
        and let oversize kernels slip past :meth:`validate`."""
        return {s.name: s for s in self if isinstance(s, Smem)}

    def smem_bytes(self) -> int:
        """Total static + dynamic ``__shared__`` bytes declared in the body —
        the padding-aware pool footprint from :func:`pack_smem` (each buffer
        aligned to ``max(sizeof(dtype), Smem.align)``), so this matches the
        renderer's pool packer and the launch-time pool size."""
        return pack_smem(self.smem_buffers.values())[1]

    def validate(self, ctx) -> bool:
        """Drop kernels whose launch wouldn't fit the hardware. Runs
        after every tile-level rewrite has settled (the engine calls
        ``validate`` on each op rebind, but only at the ``KernelOp``
        stage are the THREAD axes guaranteed to reflect the final
        per-CTA launch geometry — earlier ``TileOp`` rewrites may still
        split or coalesce them). Three checks:

        - threads ≤ ``ctx.max_threads_per_cta`` (driver-side launch cap),
        - smem ≤ ``ctx.max_dynamic_smem`` (per-block ``cudaFuncSetAttribute``
          opt-in cap; smaller for older / consumer cards),
        - CTAs ≤ ``_MAX_CTAS`` (a hard cap on the launch grid — empirically
          variants with tens of thousands of light CTAs slot into the
          GPU command processor so slowly that the per-launch
          ``_KERNEL_TIMEOUT_MS`` watchdog stops being a useful escape
          hatch; better to drop them at the rule level before benching).

        NOTE: the per-CTA thread / grid-CTA checks walked the now-demolished
        tile-flavor wrappers (``GridTile`` / ``ThreadTile``); they are pending
        rebuild. Only the smem-footprint check survives."""
        if self.smem_bytes() > ctx.max_dynamic_smem:
            return False
        return True


# ---------------------------------------------------------------------------
# Tree walk — shared with Loop IR (drives off ``Stmt.nested``)
# ---------------------------------------------------------------------------


__all__ = [
    # Shared expressions (re-exported)
    "Var",
    "Literal",
    "BinaryExpr",
    "Builtin",
    "FuncCallExpr",
    "TernaryExpr",
    "CastExpr",
    "Expr",
    # Loop-IR leaves + control flow (reused)
    "Load",
    "Assign",
    "Select",
    "SelectBranch",
    "Write",
    "Accum",
    "Cond",
    "Loop",
    # Kernel-IR statements
    "Tile",
    "Smem",
    "Sync",
    "TreeHalve",
    "WarpShuffle",
    "WarpBroadcast",
    "CpAsyncCopy",
    "CpAsyncCommit",
    "CpAsyncWait",
    "TmaDescriptor",
    "TmaLoad",
    "MbarrierInit",
    "MbarrierArriveExpectTx",
    "MbarrierArrive",
    "MbarrierWait",
    "SetMaxNReg",
    "WgmmaDescriptor",
    "WgmmaMma",
    "WgmmaFence",
    "WgmmaCommit",
    "WgmmaWait",
    "StridedLoop",
    "Stmt",
    # Top-level
    "KernelOp",
    # Re-exports
    "Axis",
    "ElementwiseImpl",
]
