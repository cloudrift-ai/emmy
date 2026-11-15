"""Kernel IR — the fully-scheduled kernel form, the input of the PTX compiler.

Kernel IR sits between Tile IR (schedule decisions as structural Stmts)
and PTX (text). Its body contains the explicit hardware
machinery: ``Tile`` (thread/block coord bindings), ``Smem`` (shared
arrays), ``Sync`` (barriers), the transports into shared memory
(``CpAsync*``, ``Tma*`` + ``Mbarrier*``), the cross-thread combines
(``WarpShuffle``, ``TreeHalve``) and the tensor-core register fragments
(``RegFragment`` … ``RegStore``, ``Wgmma*``).

Pipeline shape::

    Tile IR ──materialize_tile──▶ Kernel IR
                    ──ptx.compile_kernel──▶ PTX

**Leaf compute reuses the shared statements directly**. ``Load`` / ``Assign`` /
``Let`` / ``Select`` / ``Write`` / ``Accum`` / ``Cond`` / ``Loop`` come straight
from ``ir.stmt`` — buffer names are strings, so they compile as they are.

Kernel IR deliberately contains no scheduling decisions — those live in
Tile IR and are materialized away before reaching this layer. A
``KernelOp`` is what the CUDA backend turns into a ``RawKernel`` launch.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import I32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.body_op import BodyOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expression import (
    BinaryExpression,
    Builtin,
    CastExpression,
    Expression,
    FunctionCallExpression,
    Literal,
    SimplifyCtx,
    TernaryExpression,
    Var,
)
from emmy.compiler.ir.stmt import (
    Accum,
    Assign,
    Body,
    Cond,
    Let,
    Load,
    Loop,
    Select,
    SelectBranch,
    Statement,
    StridedLoop,
    Write,
    free_names,
)
from emmy.emmy_ir import (  # the Kernel IR statement types, declared with the shared ones in the emmy-ir crate
    BlockScaleLoad,
    CpAsyncCommit,
    CpAsyncCopy,
    CpAsyncWait,
    FragmentApply,
    FragmentArg,
    FragmentLoad,
    FragmentPromote,
    FragmentRepack,
    FragmentRowReduce,
    Grid,
    Interval,
    LdmatrixLoad,
    MbarrierArrive,
    MbarrierArriveExpectTx,
    MbarrierInit,
    MbarrierWait,
    MmaSyncPtx,
    RegFragment,
    RegStore,
    SetMaxNReg,
    Smem,
    Sync,
    Tile,
    TmaDescriptor,
    TmaLoad,
    TreeHalve,
    WarpBroadcast,
    WarpShuffle,
    WgmmaCommit,
    WgmmaDescriptor,
    WgmmaFence,
    WgmmaMma,
    WgmmaWait,
    fragment_table,
)

# ---------------------------------------------------------------------------
# Hardware primitives
# ---------------------------------------------------------------------------


#: Threads per block of the scalar tier: one thread per cell over this fixed block.
BLOCK_SIZE = 256


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
    ``BLOCK_SIZE``. This is ``blockDim`` minus any aux band, i.e. the divisor the cuda lowering
    uses for ``gridDim = ceil(n_elements / cells)``. So ``n_elements % cells == 0`` (and no aux
    band) means the static grid covers the iteration space exactly — no partial last block, so
    the ``_gid < N`` tail guard is provably always-true and the grid elides it."""
    if tile.block_threads is not None:
        return tile.block_threads
    return BLOCK_SIZE


def tile_is_static(tile: Tile) -> bool:
    """True iff every grid axis has a static extent (the static launch path)."""
    return all(a.extent.is_static for a in tile.axes)


def tile_n_dim(tile: Tile):
    """Total iteration-space size as a :class:`Dim` — the ∏ of the axis extents, holding a
    symbolic factor (a dynamic ``seq_len``) when any axis is symbolic. The cuda lowering
    sizes the launch from this (``ceil(n_dim / blockDim)`` CTAs, ``seq_len`` resolved at
    launch); the grid guards ``_gid < n_dim`` so the partial last block is masked."""

    n = Dim(1)
    for a in tile.axes:
        n = n * a.extent
    return n


# The widest value a decode computes in 32 bits; past it (a grid past 2^31 - 1 threads, or one of a symbolic size)
# its arithmetic is 64-bit.
_INT32_MAX = 2**31 - 1
_THREAD, _BLOCK = Builtin("thread_idx.x"), Builtin("block_idx.x")

#: A thread's lane in its warp and its warp in the CTA, by the name a tile's body reads them under. The decode binds
#: each one the body reads, so no emitter declares it.
THREAD_COORDS: dict[str, Expression] = {
    "lane": BinaryExpression("%", _THREAD, Literal(32, "int")),
    "warp": BinaryExpression("//", _THREAD, Literal(32, "int")),
}


def launch_ranges(cells: Expression, block_threads: int, aux_threads: int = 0) -> SimplifyCtx:
    """The ranges of the built-ins under the launch a grid of ``cells`` takes: ``block_threads + aux_threads`` threads
    a CTA, ``ceil(cells / block_threads)`` CTAs. A symbolic count leaves the CTA index what the hardware allows."""
    ctx = SimplifyCtx.empty().extend(_THREAD.name, Interval(0, block_threads + aux_threads - 1), None)
    if isinstance(cells, Literal):
        ctx = ctx.extend(_BLOCK.name, Interval(0, max(-(-int(cells.value) // block_threads), 1) - 1), None)
    return ctx


def decode_tile(tile: Tile) -> Grid:
    """The ``Grid`` a ``Tile`` launches as, with its decode spelled out.

    The flat thread id is ``block_idx.x * cells + thread_idx.x`` (``thread_idx.x % cells`` past a warp-specialized
    producer band, which wraps onto the CTA's first cells): each axis decodes to its digit, the outermost one to the
    quotient, and the grouped raster remaps the ``(m, n)`` block axes it names (the standard threadblock swizzle, so
    ``G`` block-tiles of the grouped axis share an operand slab in L2). Every decode is simplified under the launch's
    ranges, so a block axis reads ``block_idx.x`` alone and a thread axis ``thread_idx.x`` alone, and one whose
    arithmetic can pass 2^31 - 1 is computed in 64 bits. A ``Cond`` masks the ids past the cell count where the launch
    does not cover it exactly (never on a spread single cell, whose every thread runs the body). A thread coordinate
    the body reads (``THREAD_COORDS``) is bound ahead of it, a value like any other."""
    cells = tile_cells(tile)
    count = tile_n_dim(tile).expression
    launch = launch_ranges(count, cells, tile.aux_threads)
    gid = _BLOCK * Literal(cells, "int") + (_mod(_THREAD, Literal(cells, "int")) if tile.aux_threads else _THREAD)
    extents = [a.extent.expression for a in tile.axes]
    strides = [_product(extents[i + 1 :]) for i in range(len(extents))]
    decode = [gid if strides[i] == _ONE else _div(gid, strides[i]) for i in range(len(extents))]
    decode[1:] = [_mod(value, extent) for value, extent in zip(decode[1:], extents[1:], strict=True)]
    if (pair := _raster_pair(tile)) is not None:
        for i, value in _raster(tile, pair, gid, extents, strides, launch).items():
            decode[i] = value
    body = tile.body
    guard = BinaryExpression("<", gid, count).simplify(launch)
    if not tile.spread and not (isinstance(guard, Literal) and guard.value):
        body = Body((Cond(cond=_settled(guard, launch), body=body),))
    read = free_names(tile)
    coords = tuple(Let(name, value, dtype=I32) for name, value in THREAD_COORDS.items() if name in read)
    return Grid(
        axes=tile.axes,
        decode=tuple(_settled(value, launch) for value in decode),
        body=Body((*coords, *body)),
        block_threads=cells,
        aux_threads=tile.aux_threads,
    )


def _raster(tile: Tile, pair: tuple[int, int], gid: Expression, extents, strides, launch) -> dict[int, Expression]:
    """The grouped raster's decode of its two block axes, by position: ``a = grp·G + (sub % (G·Eb)) % gsz`` and ``b =
    (sub % (G·Eb)) / gsz``, ``sub`` the flat ``(m, n)`` block id, ``grp`` its group and ``gsz = min(G, Ea - grp·G)``
    the ragged last group's size (``G`` where the groups divide ``Ea``)."""
    gi, oi = pair
    mi, ni = min(gi, oi), max(gi, oi)
    ea, eb = extents[gi], extents[oi]
    g = tile.raster_group
    if isinstance(ea, Literal):
        g = min(g, int(ea.value))
    group = Literal(g, "int")
    sub = gid if strides[ni] == _ONE else _div(gid, strides[ni])
    if mi != 0:
        sub = _mod(sub, ea * eb)
    sub = sub.simplify(launch)
    stripe = group * eb
    if isinstance(ea, Literal) and int(ea.value) % g == 0:
        a, b = _div(sub, stripe) * group + _mod(sub, group), _div(_mod(sub, stripe), group)
    else:
        grp = _div(sub, stripe)
        size = FunctionCallExpression("minimum", (group, ea - grp * group))
        a, b = grp * group + _mod(_mod(sub, stripe), size), _div(_mod(sub, stripe), size)
    return {gi: a, oi: b}


_ONE = Literal(1, "int")


def _div(a: Expression, b: Expression) -> Expression:
    return BinaryExpression("//", a, b)


def _mod(a: Expression, b: Expression) -> Expression:
    return BinaryExpression("%", a, b)


def _product(factors: list[Expression]) -> Expression:
    out: Expression = _ONE
    for factor in factors:
        out = factor if out == _ONE else out * factor
    return out


def _settled(value: Expression, launch: SimplifyCtx) -> Expression:
    """``value`` simplified under ``launch``, its arithmetic 64-bit where a part of it can pass 2^31 - 1."""
    value = value.simplify(launch)
    return _widened(value) if _passes_int32(value, launch) else value


def _passes_int32(value: Expression, ctx: SimplifyCtx) -> bool:
    """True when a part of ``value`` can pass 2^31 - 1 under ``ctx``: a name with no range (a symbolic size), or a
    part whose range does."""
    if isinstance(value, Var) and value.name not in ctx.ranges:
        return True
    span = value.range(ctx)
    return (span is not None and span.hi > _INT32_MAX) or any(_passes_int32(c, ctx) for c in value.children())


def _widened(value: Expression) -> Expression:
    """``value`` with every built-in and name it reads cast to 64 bits, so C computes all of it in 64 bits; a call's
    literal argument too, since ``min(int, long long)`` names no one overload."""
    if isinstance(value, (Builtin, Var)) or (isinstance(value, Literal) and value.dtype == "int"):
        return CastExpression("long long", value)
    children = value.children()
    if isinstance(value, FunctionCallExpression):
        return value.with_children(tuple(_widened(c) for c in children))
    return value.with_children(tuple(c if isinstance(c, Literal) else _widened(c) for c in children)) if children else value


def _raster_pair(tile: Tile) -> tuple[int, int] | None:
    """``(grouped, other)`` axis positions of an active grouped raster: the ``(m, n)`` block axes adjacent in
    the decode, ``m`` before ``n``; ``None`` when the order is the flat one or the grouped axis takes one group."""
    if not (tile.raster_group and tile.raster_group > 1 and tile.raster_axes):
        return None
    names = [a.name for a in tile.axes]
    if tile.raster_axes[0] not in names or tile.raster_axes[1] not in names:
        return None
    mi, ni = names.index(tile.raster_axes[0]), names.index(tile.raster_axes[1])
    if ni != mi + 1:
        return None
    gi, oi = (mi, ni) if tile.raster_orient == "m" else (ni, mi)
    extent = tile.axes[gi].extent
    if extent.is_static and min(tile.raster_group, extent.as_static()) <= 1:
        return None
    return gi, oi


#: The per-arg distribution kinds of a :class:`FragmentApply` operand.
FRAG = "frag"  # a C-fragment operand — indexed per element (``arg[i]``)
ROW = "row"  # a per-row scalar — broadcast by row (suffix ``0`` for rows g, ``1`` for rows g+8)
UNIFORM = "uniform"  # a cell-uniform scalar / literal — the same value for all 4 elements
COORD = "coord"  # a predicate over the element's absolute coordinates (``FRAG_ROW`` / ``FRAG_COL``)
GMEM = "gmem"  # a ``(buffer, index)`` load template over the element's absolute coordinates


def fragment_args(args: tuple, kinds: tuple[str, ...]) -> tuple[FragmentArg, ...]:
    """The operands of a :class:`FragmentApply` from the positional spelling the lowering builds — per kind a name
    (FRAG, UNIFORM), an expression (COORD, a UNIFORM literal), a pair of names (ROW) or ``(buffer, index)`` (GMEM)."""

    def one(arg, kind: str) -> FragmentArg:  # noqa: ANN001 — see the docstring
        if kind == GMEM:
            return FragmentArg(kind, expressions=tuple(arg[1]), buffer=arg[0])
        if kind == ROW:
            return FragmentArg(kind, names=tuple(arg))
        return FragmentArg(kind, expressions=(arg,)) if isinstance(arg, Expression) else FragmentArg(kind, names=(arg,))

    return tuple(one(arg, kind) for arg, kind in zip(args, kinds, strict=True))


#: The reserved coordinate Vars a :class:`FragmentApply` COORD / GMEM template is written over — the element's
#: ABSOLUTE query row / key column; the PTX compiler substitutes each element's coords (tile origin +
#: layout offset) for these.
FRAG_ROW = "__frow"
FRAG_COL = "__fcol"
#: The reserved offset Vars a :class:`RegStore` epilogue is written over — the element's row / col
#: OFFSET within its fragment, added to the cell base the expression already carries.
ELEM_ROW = "__M__"
ELEM_COL = "__N__"


# ---------------------------------------------------------------------------
# Warp-level MMA: ``mma.sync.aligned`` + ``ldmatrix`` (the ``s16816`` path
# cuBLAS / CUTLASS use) — the sole tensor-core Statement family. Operands are
# *explicit per-thread register arrays* with a PTX-fixed lane→element layout
# (``RegFragment`` / ``LdmatrixLoad`` / ``FragmentLoad`` / ``MmaSyncPtx`` / ``RegStore``), which the PTX
# compiler expands per lane from the lane-to-element table. Emitted by
# the kernel atom lowering (``lowering/kernel/_atom.py``) from the tile's ``Atom`` spec.
# ---------------------------------------------------------------------------


def fragment_registers(stmt: RegFragment) -> int:
    """How many registers a lane holds of the fragment ``stmt`` declares: its row of the lane-to-element table says."""
    return len(fragment_table(stmt.fragment_layout, stmt.role, stmt.dtype.name).registers)


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
#: The slab layouts that are a function of a store's logical row and column: a ``Write`` under one keeps both.
TWO_DIMENSIONAL_LAYOUTS = (VOLTA_CROSSWISE, VOLTA_B_CONGRUOUS)

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

    The PTX compiler writes the same formula into the kernel; this twin lets a test check the bits without a
    Hopper card."""
    return ((addr >> 4) & 0x3FFF) | (((lbo >> 4) & 0x3FFF) << 16) | (((sbo >> 4) & 0x3FFF) << 32) | (mode << 62)


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
    ``max(dtype.nbytes, s.align)`` before placing each buffer, and returns
    ``({name: byte_offset}, total_padded_bytes)``.

    Single source of truth for the per-CTA smem footprint: ``KernelOp.smem_bytes``
    and ``interface.shared_pool`` both call this so the
    static-vs-dynamic gate and the launch-time pool size agree (an unpadded sum
    would under-report when a buffer's alignment exceeds its natural stride —
    e.g. a 512/1024 B swizzle-atom-aligned operand)."""
    from math import prod  # noqa: PLC0415

    offsets: dict[str, int] = {}
    cursor = 0
    for s in smems:
        elements = prod(int(e) for e in s.extents) if s.extents else 1
        align = max(s.dtype.nbytes, int(s.align) if s.align else 0)
        if align:
            cursor = (cursor + align - 1) // align * align
        offsets[s.name] = cursor
        cursor += elements * s.dtype.nbytes
    return offsets, cursor


@dataclass(frozen=True)
class KernelOp(BodyOp):
    """One ``__global__`` GPU kernel as a Kernel IR program.

    :class:`BodyOp` subclass parallel to ``TileOp`` / ``LoopOp``: lives as
    a graph node, carries a body of Kernel IR stmts.

    Buffer shapes are *not* baked in — the surrounding graph supplies
    them where the kernel is compiled, same as ``TileOp``. Kernel signature is derived
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
        """Total static + dynamic shared bytes declared in the body —
        the padding-aware pool footprint from :func:`pack_smem` (each buffer
        aligned to ``max(dtype.nbytes, Smem.align)``), so this matches the
        launch-time pool size."""
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
# Tree walk — shared with Loop IR (drives off ``Statement.nested``)
# ---------------------------------------------------------------------------


__all__ = [
    # Shared expressions (re-exported)
    "Var",
    "Literal",
    "BinaryExpression",
    "Builtin",
    "FunctionCallExpression",
    "TernaryExpression",
    "CastExpression",
    "Expression",
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
    "Grid",
    "decode_tile",
    "launch_ranges",
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
    "BlockScaleLoad",
    "FragmentApply",
    "FragmentLoad",
    "FragmentPromote",
    "FragmentRepack",
    "FragmentRowReduce",
    "LdmatrixLoad",
    "MmaSyncPtx",
    "RegStore",
    "StridedLoop",
    "Statement",
    # Top-level
    "KernelOp",
    # Re-exports
    "Axis",
    "ElementwiseImpl",
]
