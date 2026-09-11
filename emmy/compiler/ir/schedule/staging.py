r"""Operand-stage RESOLUTION — the sizing arithmetic behind the schedule's ``STAGE`` family.

Exactly three resolvers live here — :func:`resolve_warp_stage`, :func:`resolve_scalar_stage`,
:func:`resolve_fill_stage` — plus the compute fill's own node refusals. The classic scheduler
offers a staged transport only when it RESOLVES here, and the row carries the
resolved spelling, so the fork, the stamped knobs and the kernel agree. A resolver is not a
predicate and this is not a legality layer: for the shared-memory budget the legal answer is a
SIZE — the resolved ``bk_elems`` slab chunk and the deepest ring the budget affords — so each
returns a sized :class:`Stage` (or ``None`` for "this transport does not engage here"), and
handing back the largest legal stage is what keeps an over-budget row out of the fork instead of
failing at materialization.

The ``str | None`` functions beside the fill resolver (:func:`computed_operand_cover`,
:func:`computed_operand_copy_dtype`, with :func:`converting_a` naming their converting-A case)
are the NODE-dependent geometry of the move :func:`resolve_fill_stage` realizes — they sit here
because the fill is the move they filter, one statement each so the unpinned enumeration's drop
and a pin's raise share it. What deliberately does NOT live here: the transport/target rule
(MOVE×target — ``Stage.available_on``, filtered in the ``stage_moves`` catalog; the scheduler's
pin path reads its message through :func:`stage_target`) and the fragment-seam relation (including
the paired register bound), which lives in :class:`ClassicScheduleContext`. Nothing here ranks or
narrows for speed."""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.ir.address import BYTE_SLAB_PAD
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.schedule import ResolvedStage, Stage, Tile
from emmy.compiler.ir.schedule.packing import block_scaled_atom, packed_readings
from emmy.compiler.ir.schedule.views import cone_seam

# TMA hardware: every box dim must fall in 1..256, and the swizzle-split box caps the operand rank
# at 4 so it stays within the 5-dim limit.
_TMA_MAX_BOX = 256
_TMA_ALIGN = 16  # the NONE-swizzle box-copy rule: 16 B-aligned inner dim and inner global stride

# cp.async needs a >= 4 B contiguous chunk, so a 2 B/elem slab's inner dim must be even.
_CP_ASYNC_MIN_ELEMS = 2


def _decline(why: list[str] | None, reason: str) -> None:
    """Record a resolver's decline reason for a caller that will report it (a pinned candidate)."""
    if why is not None:
        why.append(reason)


def stage_target(stage: Stage, ctx) -> str | None:
    """Why ``stage`` names a copy instruction family ``ctx`` lacks (``None`` when it has it) — the
    PIN path's message for the rule :meth:`Stage.available_on` states once (the catalog is already
    filtered through it, so only a pin can reach a transport the target lacks)."""
    if stage.available_on(ctx):
        return None
    need = "cp.async requires sm_80 or newer" if stage.transport == "smem-async" else "TMA requires sm_90 or newer"
    return f"STAGE {stage.spell()}: {need}"


#: The deepest ring a SPLIT blocking copy can run. Its in-flight chunk rides REGISTERS — issued at
#: the top of the loop body and landed at the bottom of the same body — so exactly one chunk is ever
#: in flight, however many slots the codec asks for. A third slot would hold no extra chunk, only
#: idle smem, so the ring caps here the way it caps at the smem budget. Both resolvers apply it; the
#: staged K-loop skeleton asserts it (a split fill primes exactly one chunk, and two primes would
#: redeclare the same staging registers).
SPLIT_COPY_DEPTH = 2


def _clamp_depth(depth: int, slot_bytes: int, budget: int) -> int:
    """The deepest ring the smem ``budget`` affords at ``slot_bytes`` per ringed slot, never deeper
    than asked. Shared by the warp copy ring and the fill's B-slab ring; the scalar resolver
    instead honors the requested depth by SHRINKING its slab chunk (its budget rule is the chunk
    ladder's), so it does not end here."""
    return min(depth, budget // slot_bytes)


def _tma_operand_rank(index: tuple, tile_name: str, k_name: str) -> bool:
    """Whether TMA's box can encode this operand's gmem index. The data plane is the TRAILING 2
    dims; extra LEADING dims ride as extent-1 box dims whose origin is evaluated once per fill, so
    those exprs must not move with the tile or the K loop."""
    if not 2 <= len(index) <= 4:
        return False
    return all(not ({tile_name, k_name} & e.free_vars()) for e in index[:-2])


def _warp_vector_copy(k_axis: Axis, tile_n: int, bk_elems: int, mask_n: bool, b_trans: bool, *, ragged: bool = False) -> bool:
    """Vector-copy staging: a STATIC, tile-divisible K, an unmasked N, and an even inner slab dim.

    ``ragged`` drops the K demand for a caller whose drain folds an overhanging chunk to nothing
    (:func:`_chunk_warp_stage` states the one such reading)."""
    if mask_n:
        return False
    if not ragged and (not k_axis.extent.is_static or k_axis.extent.as_static() % bk_elems):
        return False
    return bk_elems % _CP_ASYNC_MIN_ELEMS == 0 and (b_trans or tile_n % _CP_ASYNC_MIN_ELEMS == 0)


def _warp_tma(
    k_axis: Axis, n_axis: Axis, tile_n: int, bk_elems: int, a_bytes: int, b_bytes: int, mask_n: bool, b_trans: bool, *, ragged: bool = False
) -> bool:
    """TMA staging: STATIC tile-divisible K and N, and 16 B-aligned inner dims — each operand's
    box inner dim and gmem inner stride at its OWN element width (a byte-staged fp8 B sizes at
    1 B). A transposed B boxes N-major, so N drops out of the alignment gate.

    ``ragged`` is :func:`_warp_vector_copy`'s flag, and drops the same K demand — with it the
    spans K measures drop out of the alignment gate too, which is sound only where K is no inner
    dim of any staged operand (again :func:`_chunk_warp_stage`'s K-major reading)."""
    if mask_n or not n_axis.extent.is_static:
        return False
    if not ragged and (not k_axis.extent.is_static or k_axis.extent.as_static() % bk_elems):
        return False
    n = n_axis.extent.as_static()
    inner = [bk_elems * a_bytes, *((bk_elems * b_bytes,) if b_trans else (tile_n * b_bytes, n * b_bytes))]
    if not ragged:
        k = k_axis.extent.as_static()
        inner += [k * a_bytes, *((k * b_bytes,) if b_trans else ())]
    return all(x % _TMA_ALIGN == 0 for x in inner)


# The packed byte-slab stage's fixed geometry. The drain decodes an N-major slab through the k16
# f16 B fragment map and reads one scale per 16 K elements, so the format's block, the atom's K
# step and this constant are all the same 16; a different block or atom keeps the generic reading.
# A one-value byte (fp8) reads its scale once per atom step instead, so its block only has to hold
# whole atom steps and tile the chunk (or be tiled by it).
_PACKED_BLOCK = 16

#: The fragment dtypes the packed drain is spelled for. Both hold every e2m1 value exactly, so the
#: decode is a constant-table read in either; a wider or narrower operand has no such table.
_PACKED_FRAGMENT_DTYPES = ("f16", "bf16")


def chunk_key_stage(tile: Tile, stage: Stage, inputs, producer, producer_k, k_axis: Axis) -> tuple | None:
    """The chunk tier's SECOND streamed slab — its score's KEY — as ``(load, span)``, or ``None``
    when the streamed value slabs alone.

    A chunk carrier reads TWO operands along its key axis: the score's key and the value it
    streams. Both are B-shaped slabs of ``bk_elems`` key rows, so this asks the value's own
    questions of the key at the KEY's spans — its rows the chunk's keys, its columns the score's
    WHOLE contraction span, which the chunk loop covers in one pass. That span must therefore be
    static, and it must be the key's gmem-contiguous dim so the fill's copy chunks run along it.

    ``None`` where the carrier has no nested score at all — the gathered ``softmax@V`` shape, whose
    A is a stored probability tile read at the fragment lane map — or where the key declines any of
    those questions; the value then stages alone. The SCORE fragments themselves never leave
    registers either way (the chunk's own C fragments repack into the expectation's A), which is
    why this stages the key rather than the score.

    One statement, two readers: :func:`_chunk_warp_stage` sizes the ring with it and the chunk
    tier's emission builds the slab off it, so the fork and the kernel agree on which operands are
    resident."""
    if producer is None or producer_k is None or not producer_k.extent.is_static:
        return None
    key = next((edge for edge in producer.operands if k_axis.name in edge.free_axes), None)
    slab = key.as_slab() if key is not None else None
    if slab is None or producer_k.name not in slab.load.index[-1].free_vars():
        return None  # the score's contraction span must be the key's gmem inner dim (the fill's chunk)
    stored = inputs.get(slab.load.input) if inputs else None
    if stored is not None and stored.dtype != tile.atom.operand_dtype("b"):
        return None
    span, nbytes = producer_k.extent.as_static(), tile.atom.operand_dtype("b").nbytes
    bk_elems = tile.bk * tile.atom.atom_k
    # The key rows its slab by key and runs its copy chunks along the score's span, so a SYMBOLIC
    # key extent enters neither the chunk width nor the gmem row stride — the value's own
    # ragged-tail reading, at this operand's spans.
    ragged = not k_axis.extent.is_static
    copies = (
        _warp_vector_copy(k_axis, span, bk_elems, False, False, ragged=ragged)
        if stage.transport == "smem-async"
        else (
            stage.transport == "smem-tma"
            and _tma_operand_rank(slab.load.index, producer_k.name, k_axis.name)
            and max(span, bk_elems) <= _TMA_MAX_BOX
            and _warp_tma(k_axis, producer_k, span, bk_elems, nbytes, nbytes, False, False, ragged=ragged)
        )
    )
    return (slab.load, span) if copies else None


def _chunk_warp_stage(
    c: Fold, tile: Tile, stage: Stage, budget: int, inputs, k_axis: Axis, producer=None, producer_k=None
) -> ResolvedStage | None:
    """Resolve a ``Stage`` for a CHUNKED carrier — attention's flash tier.

    The ring holds the carrier's two STREAMED operands: the value it folds against, and the score's
    key when :func:`chunk_key_stage` says that one slabs too. Both stream along the carrier's own
    key axis at the chunk width ``Tile.bk`` already spells — which is why admitting a transport
    here adds no second spelling of that width. The score's own fragments are never copied: they
    repack in registers out of the chunk's C fragments, or, in the gathered shape, come off a
    stored tile at the fragment lane map.

    The eligibility gates are the generic arm's B half, asked of each operand that has a slab. An
    operand whose stored dtype is not the atom's declines rather than staging a converting slab:
    the chunk drain reads 16-bit fragments, and the gmem-direct fragment load converts per element
    correctly today.

    The synchronous ``smem`` transport declines: it is the Volta atom's blocking vector copy, and
    attention's chunk tier has no sm_70 kernel to serve. A SYMBOLIC key extent does NOT decline —
    see the ragged-tail reading below, which is what lets a serving-shaped attention kernel stage
    at all, though it keeps the single-buffer ring. On a STATIC extent depth is the ordinary budget
    clamp: the chunk loop carries the whole softmax between its fill and its drain, so a deeper ring
    prefetches the next chunk's key and value ACROSS that softmax, which is the longest overlap this
    tier has to offer.
    """
    atom, view, n = tile.atom, c.as_contraction(), tile.n
    slab = c.operands[1].as_slab()
    if view is None or slab is None:
        return None
    b_nbytes = atom.operand_dtype("b").nbytes
    stored = inputs.get(slab.load.input) if inputs else None
    if stored is not None and stored.dtype != atom.operand_dtype("b"):
        return None
    bk_elems = tile.bk * atom.atom_k
    # A SYMBOLIC key extent stages when the slab's gmem geometry never reads it. A K-MAJOR value —
    # the ordinary attention layout — rows the slab by key and runs its copy chunks along the head
    # dim, so the extent enters neither the chunk width nor the gmem row stride, and the last chunk
    # simply overhangs. Both ends of that tail are already disciplined: the fill clamps the
    # overhanging key row to the last valid one (a TMA box zero-fills instead), and the drain's
    # boundary ``FragmentMask`` has put those keys at the pivot identity, so they weigh exactly
    # zero and the duplicate value rows fold to nothing. A TRANSPOSED value strides its gmem rows
    # BY the key extent, so it keeps the static, chunk-divisible demand every staged operand makes.
    ragged = not k_axis.extent.is_static and not view.b_trans
    vector_copy_ok = _warp_vector_copy(k_axis, n.tile, bk_elems, n.mask, view.b_trans, ragged=ragged)
    tma_ok = (
        stage.transport == "smem-tma"
        and _tma_operand_rank(slab.load.index, n.axis.name, k_axis.name)
        and max(n.tile, bk_elems) <= _TMA_MAX_BOX
        and _warp_tma(k_axis, n.axis, n.tile, bk_elems, b_nbytes, b_nbytes, n.mask, view.b_trans, ragged=ragged)
    )
    cp_ok = stage.transport == "smem-async" and vector_copy_ok
    if not (tma_ok or cp_ok):
        return None
    b_rows, b_cols = (n.tile, bk_elems) if view.b_trans else (bk_elems, n.tile)
    slot_bytes = b_rows * b_cols * b_nbytes
    key = chunk_key_stage(tile, stage, inputs, producer, producer_k, k_axis)
    if key is not None:
        slot_bytes += bk_elems * key[1] * b_nbytes
    if slot_bytes > budget:
        return None
    # A RAGGED stream keeps the single-buffer ring the tail discipline above was written for.
    # Measured: with a prefetch slot on top of it the kernel HANGS and poisons the CUDA context
    # (``test_masked_symbolic_accuracy[demoted_pv-*]`` and ``[computed_a_symbolic_k_warp-16]``, which
    # run this tier at a symbolic key length); at one slot it is correct. What the runtime chunk
    # count does to the prefetch's clamp is not diagnosed, so the depth is refused here rather than
    # offered and left to fail at the card.
    depth = 1 if ragged else _clamp_depth(stage.depth, slot_bytes, budget)
    choice = replace(stage, depth=depth, reg_depth=min(stage.reg_depth, tile.bk))
    return ResolvedStage(choice, bk_elems=bk_elems)


def _packed_warp_stage(c: Fold, tile: Tile, stage: Stage, budget: int, packed, inputs, k_axis: Axis) -> ResolvedStage | None:
    """Resolve the PACKED byte-slab stage for a byte-coded k-block B — the NVFP4 weight cone, or a
    block-scaled fp8 weight.

    The scoped shape, which is what the fragment drain is written for: a copy transport (cp.async or
    TMA), an N-major byte-coded weight under an f16 or bf16 atom with a K step of 16, and an A
    already carrying the atom's dtype. A packed pair's block is that same 16; a one-value byte's
    block holds whole atom steps and tiles the chunk or is tiled by it, so every drain step reads
    one scale. Everything outside it declines and stays on the generic computed-B reading, which
    computes the same values through the sync compute-fill.

    The sizing is the fp8 byte slab's rule restated in the weight's own units. One stored byte is
    ``per_byte`` K elements, so the bits row is ``bk_elems / per_byte`` BYTES plus the cp.async
    row pad, and it must be 16-divisible for the same reason the fp8 one is: the fill copies 16 B
    chunks and a chunk never straddles a row. The gmem rows those chunks stride are ``K /
    per_byte`` bytes, so that span is 16-divisible too. On top of the ring the budget carries ONE
    scale slab, ``tile_n`` rows of one scale per block the chunk touches — single-buffer, because
    it is compute-filled and ringing a compute fill buys no overlap. A packed pair's scale slab
    holds the atom's element width; a one-value byte's holds f32, the dtype its scale multiplies
    the decoded value in before the round to the fragment.
    """
    atom = tile.atom
    if stage.transport not in ("smem-async", "smem-tma"):
        return None  # the sync compute fill has nothing to copy under, and split cuts a group this fold has one of
    if atom.atom_k != _PACKED_BLOCK or atom.fragment_layout != "m16n8k16":
        return None
    bk_elems = tile.bk * atom.atom_k
    if packed.per_byte == 2 and packed.block != _PACKED_BLOCK:
        return None
    if packed.block % atom.atom_k or (packed.block % bk_elems and bk_elems % packed.block):
        return None
    a_dtype, b_dtype = atom.operand_dtype("a"), atom.operand_dtype("b")
    if a_dtype != b_dtype or a_dtype.name not in _PACKED_FRAGMENT_DTYPES:
        return None  # the drain has one value table and one scale multiply, both at the operand dtype
    bits = inputs.get(packed.bits.input)
    if bits is None:
        return None
    if c.operands[0].as_slab() is not None:
        a_tensor = inputs.get(c.operands[0].as_slab().load.input)
        if a_tensor is None or a_tensor.dtype != a_dtype:
            return None
    # A COMPUTED A has no gmem tensor to match: it evaluates into its slab at the atom's operand
    # dtype, converting on the store, which is the compute fill's own contract.

    per_byte = packed.per_byte
    if bits.dtype.nbytes != 1 or bits.dtype.logical_elems != per_byte or len(bits.shape) != 2 or len(packed.bits.index) != 2:
        return None
    if k_axis.name not in packed.bits.index[-1].free_vars():
        return None  # a K-strided packed weight is not the N-major layout the drain reads
    if not k_axis.extent.is_static:
        return None
    k = k_axis.extent.as_static()
    if tile.n.mask or k % bk_elems or (bk_elems // per_byte) % 16 or (k // per_byte) % 16:
        return None
    # A TMA box deposits DENSE, so the byte rows carry no pad — the same split the fp8 byte slab
    # makes. Its extra demands are the hardware's: every box dim within the 256 limit, and a
    # 16 B-aligned inner span and gmem row stride per operand at its OWN width. The byte side is
    # already 16-divisible by the rule above; A's is ``bk_elems`` and ``k`` at two bytes each.
    pad = 0 if stage.transport == "smem-tma" else BYTE_SLAB_PAD
    if stage.transport == "smem-tma":
        if max(tile.m.tile, tile.n.tile, bk_elems, bk_elems // per_byte) > _TMA_MAX_BOX:
            return None
        if (bk_elems * a_dtype.nbytes) % _TMA_ALIGN or (k * a_dtype.nbytes) % _TMA_ALIGN:
            return None
    slot_bytes = tile.m.tile * bk_elems * a_dtype.nbytes + tile.n.tile * (bk_elems // per_byte + pad)
    scale_bytes = tile.n.tile * packed.scale_cols(bk_elems) * (b_dtype.nbytes if per_byte == 2 else 4)
    if scale_bytes + slot_bytes > budget:
        return None
    depth = _clamp_depth(stage.depth, slot_bytes, budget - scale_bytes)
    choice = replace(stage, depth=depth, reg_depth=min(stage.reg_depth, tile.bk))
    return ResolvedStage(choice, bk_elems=bk_elems)


def _row_major_k_inner(tensor, load, k_name: str) -> bool:
    """Whether a staged operand is ROW-MAJOR with the contraction axis innermost — the layout the
    byte gathers walk, asked without pinning a rank.

    A layer program and a weight constant carry the same operand at different ranks: a weight is
    ``[N, K/2]``, while an activation keeps its batch axis and its block axis as degenerate dims
    (``[1, M, K/2]``, ``[1, M, K/16, 1]``). Neither affects the address — a unit extent contributes
    no stride — so both are dropped before asking the one question that matters."""
    dims, idx = list(tensor.shape), list(load.index or ())
    if len(dims) != len(idx):
        return False
    while len(dims) > 2 and dims[-1].is_static and dims[-1].as_static() == 1 and not idx[-1].free_vars():
        dims.pop()
        idx.pop()
    while len(dims) > 2 and dims[0].is_static and dims[0].as_static() == 1:
        dims.pop(0)
        idx.pop(0)
    return len(dims) == 2 and all(d.is_static for d in dims) and k_name in idx[-1].free_vars()


def _block_scaled_warp_stage(c: Fold, tile: Tile, stage: Stage, budget: int, pair, inputs, k_axis: Axis) -> ResolvedStage | None:
    """Resolve the FOUR-SLAB stage of a block-scaled packed pair — the native fp4 cell.

    The simplest staging in the tier, because no SCALE is computed: the packed byte-slab stage next
    door compute-fills its scale slab, while here the instruction takes the stored e4m3 byte itself,
    so that fill has nothing left to evaluate. Both sides' scales and every stored side's codes are
    therefore verbatim copies. The one slab still filled is an activation whose codes this very
    matmul computes — its quantize fused in, leaving no buffer to copy from. The weight side is
    always stored, which is what the ``pair.b`` check below requires of every channel.

    The scoped shape: cp.async (the four-descriptor TMA box copy is not built — a missing-code
    fact, stated where the code would live), a k64 cell over 16-element blocks, both code
    operands canonically laid out with k innermost, and a static k the tile divides.

    Sizing restates the byte-slab rule in the format's units, twice per side. A codes row is
    ``bk_elems / 2`` bytes and a scales row ``bk_elems / block``; the fill copies 16 B chunks and
    a chunk never straddles a row, so both spans — and the gmem rows they stride, ``k / 2`` and
    ``k / block`` — must be 16-divisible. That is what bounds the tile from below: at block 16 a
    scales row needs ``bk_elems`` to be a multiple of 256, so the narrow-k tiles decline here and
    keep the generic reading.
    """
    atom = tile.atom
    if stage.transport != "smem-async":
        return None
    if atom.atom_k != 64 or pair.block != _PACKED_BLOCK or atom.operand_dtype("a") != atom.operand_dtype("b"):
        return None
    if not k_axis.extent.is_static or tile.n.mask:
        return None  # an N tile the copy would clamp element-by-element along the contiguous span
    if any(op.bits is None for op in pair.b):
        return None  # only the ACTIVATION side's codes are ever computed here; a weight is stored
    k, bk_elems, block = k_axis.extent.as_static(), tile.bk * atom.atom_k, pair.block
    # Every channel's weight rides the SAME N tile and the same column geometry, so each one is
    # sized on its own terms and any refusal declines the whole node.
    for side, tile_side, atom_dim in ((pair.a, tile.m, 0), *((op, tile.n, 1) for op in pair.b)):
        scale = inputs.get(side.scale.input)
        if scale is None or not _row_major_k_inner(scale, side.scale, k_axis.name):
            return None
        if side.bits is not None:
            # STORED codes copy verbatim, so their gmem layout has to be the one the byte gathers
            # walk. COMPUTED codes have no gmem tensor at all — the fill writes the slab — so
            # there is nothing to lay out and the 16 B chunk rule below applies to the copied
            # slabs only.
            bits = inputs.get(side.bits.input)
            if bits is None or bits.dtype != atom.operand_dtype("a") or not _row_major_k_inner(bits, side.bits, k_axis.name):
                return None
            if (k // 2) % 16:
                return None
        if tile_side.tile % atom.shape[atom_dim]:
            return None
    if k % bk_elems or (bk_elems // 2) % 16 or (bk_elems // block) % 16 or (k // block) % 16:
        return None
    rows = (tile.m.tile, tile.n.tile)
    slot_bytes = sum(r * (bk_elems // 2 + BYTE_SLAB_PAD) + r * (bk_elems // block + BYTE_SLAB_PAD) for r in rows)
    if slot_bytes > budget:
        return None
    choice = replace(stage, depth=_clamp_depth(stage.depth, slot_bytes, budget), reg_depth=min(stage.reg_depth, tile.bk))
    return ResolvedStage(choice, bk_elems=bk_elems)


def resolve_warp_stage(
    c: Fold,
    tile: Tile,
    stage: Stage,
    budget: int,
    inputs=None,
    *,
    readings: tuple | None = None,
    k_axis: Axis,
    producer=None,
    producer_k=None,
) -> ResolvedStage | None:
    """Resolve an operand ``Stage`` against the warp (mma) contraction ``c`` — synchronous copy,
    cp.async, TMA, or gmem-direct (``None``). The resolved stage carries ``bk_elems``, ``depth``
    clamped so the ring's slots fit ``budget``, and ``reg_depth`` clamped to ``bk``. A tile whose
    single depth-1 slot already exceeds ``budget`` DECLINES — unlike the scalar resolver it cannot
    shrink the slab.

    ``inputs`` (the per-buffer tensors) gates operand dtypes: the copy transports byte-copy into
    slabs sized at each operand's element width, so a slab is byte-copied verbatim and the drain
    reads exactly the bytes the fill deposited. Two dtype forms resolve: an operand traced AT the
    atom's operand dtype (the 16-bit family — ldmatrix drain), and a 1-byte (fp8-stored) operand —
    a B under a 16-bit atom stages as a RAW BYTE slab drained by the cooperative convert gather
    (W8A16), and the fp8 (k32) atoms stage both operands as byte slabs drained by the byte repack.
    Any other mismatch DECLINES and keeps the warp tier gmem-direct, whose fragment load converts
    per element. A byte slab's fill runs 16 B chunks and its cp.async row pad is 16 B
    (``ir.address.BYTE_SLAB_PAD``), so its inner span — and, canonical-B, the gmem row stride N — must
    be 16-divisible.

    A PACKED-PAIR B (an NVFP4 weight's decode cone) is the one COMPUTED edge that resolves here,
    through :func:`_packed_warp_stage`: its bits copy verbatim like any byte slab, and the block
    scales the cone would otherwise recompute per element ride a small compute-filled slab beside
    them. Every other computed edge declines — a copy transport cannot evaluate a producer cone —
    and takes :func:`resolve_fill_stage` instead."""
    # Which cell is being resolved decides which reading applies, so the pair question is asked
    # only for the atom that consumes a pair. Both operands packed under a 16-BIT atom is still
    # the single-sided shape: that drain decodes each operand into 16-bit fragments, which is
    # correct — just not what the native cell does.
    # ``readings`` is the caller's memo of the two packed questions, both pure functions of the
    # node. Recomputing them here costs a backward cone per side PER CANDIDATE, and a warp site
    # has hundreds; the prescan asks once and hands the answer down (``_SiteFacts.packed``).
    single, pair = readings if readings is not None else packed_readings((c,), inputs)[id(c)]
    if pair is not None and block_scaled_atom(tile.atom):
        return _block_scaled_warp_stage(c, tile, stage, budget, pair, inputs, k_axis)
    if single is not None:
        return _packed_warp_stage(c, tile, stage, budget, single, inputs, k_axis)
    if c.chunked():
        return _chunk_warp_stage(c, tile, stage, budget, inputs, k_axis, producer, producer_k)
    atom = tile.atom
    sync_copy = stage.transport == "smem" and atom.sync_copy_staging
    bk_elems = tile.bk * atom.atom_k
    m, n = tile.m, tile.n
    a_nbytes, b_nbytes = atom.operand_dtype("a").nbytes, atom.operand_dtype("b").nbytes
    if inputs:
        for edge, role in ((c.operands[0], "a"), (c.operands[1], "b")):
            t = inputs.get(edge.as_slab().load.input) if edge.as_slab() is not None else None
            if t is None or t.dtype == atom.operand_dtype(role):
                continue
            if sync_copy:
                return None  # the Volta shared gather consumes f16 slabs; synchronous copies do not convert
            if role == "b" and t.dtype.nbytes == 1 and t.dtype.logical_elems == 1 and b_nbytes == 2:
                b_nbytes = 1  # fp8-B under a 16-bit atom: byte slab, convert at the drain
                continue
            # A packed-pair byte (f4e2m1x2) is NOT an fp8 byte: one stored element is two
            # logical K elements, so the fp8 slab geometry would halve K. No slab takes it.
            return None
    for eb, inner, row_axis in (
        (a_nbytes, bk_elems, None),
        (b_nbytes, bk_elems if c.as_contraction().b_trans else n.tile, None if c.as_contraction().b_trans else n.axis),
    ):
        if eb != 1:
            continue
        if inner % 16:
            return None  # byte slab: 16 B chunks + the 16 B row pad need a 16-divisible inner span
        if row_axis is not None and (not row_axis.extent.is_static or row_axis.extent.as_static() % 16):
            return None  # canonical byte B: the 16 B gmem chunks stride rows of N bytes
    rank_ok = (
        c.operands[0].as_slab() is not None
        and c.operands[1].as_slab() is not None  # a descriptor needs a gmem address on BOTH edges
        and _tma_operand_rank(c.operands[0].as_slab().load.index, m.axis.name, k_axis.name)
        and _tma_operand_rank(c.operands[1].as_slab().load.index, n.axis.name, k_axis.name)
    )
    box_ok = max(m.tile, n.tile, bk_elems) <= _TMA_MAX_BOX
    tma_ok = (
        stage.transport == "smem-tma"
        and rank_ok
        and box_ok
        and _warp_tma(k_axis, n.axis, n.tile, bk_elems, a_nbytes, b_nbytes, n.mask, c.as_contraction().b_trans)
    )
    vector_copy_ok = _warp_vector_copy(k_axis, n.tile, bk_elems, n.mask, c.as_contraction().b_trans)
    cp_ok = stage.transport == "smem-async" and vector_copy_ok
    sync_ok = sync_copy and vector_copy_ok
    if not (tma_ok or cp_ok or sync_ok):
        return None
    pad_a, pad_b = (BYTE_SLAB_PAD if eb == 1 and cp_ok else 0 for eb in (a_nbytes, b_nbytes))
    b_rows, b_cols = (n.tile, bk_elems + pad_b) if c.as_contraction().b_trans else (bk_elems, n.tile + pad_b)
    slot_bytes = m.tile * (bk_elems + pad_a) * a_nbytes + b_rows * b_cols * b_nbytes
    if slot_bytes > budget:
        return None
    depth = _clamp_depth(stage.depth, slot_bytes, budget)
    if sync_ok:
        depth = min(depth, SPLIT_COPY_DEPTH)
    choice = replace(stage, depth=depth, reg_depth=min(stage.reg_depth, tile.bk))
    return ResolvedStage(choice, bk_elems=bk_elems)


def resolve_scalar_stage(c: Fold, tile: Tile, stage: Stage, inputs, budget: int, k_axis: Axis) -> ResolvedStage | None:
    """Resolve an operand ``Stage`` against the scalar register-tile contraction ``c``, or ``None``
    (gmem-direct). The slab K-chunk ``bk_elems`` is DERIVED to fit ``depth`` operand slots in the
    smem ``budget`` (the largest offered chunk dividing K) — not codec-spelled, so no schema change;
    when no chunk fits at the requested depth the depth steps down, single-buffer last."""
    if stage.transport not in ("smem", "smem-tma", "smem-async") or not k_axis.extent.is_static:
        return None
    # A masked-N B-slab fill would clamp a chunk-start column into a row-crossing gmem address and
    # hang on the misaligned copy; a transposed B has no scalar drain variant (the warp tier stages
    # it into an N-major slab).
    if tile.n.mask or c.as_contraction().b_trans:
        return None
    if not inputs or c.operands[0].as_slab() is None or c.operands[1].as_slab() is None or c.operands[0].as_slab().load.input not in inputs:
        return None
    # 1-byte (fp8) elements decline: the fill's chunk-width and alignment math below is written
    # for the 2/4-byte dtypes and is unaudited at nbytes == 1 — refusing keeps the tier
    # gmem-direct (correct, converts per element) instead of risking a mis-sized slab.
    if any(
        t is not None and t.dtype.nbytes < 2
        for t in (inputs.get(c.operands[0].as_slab().load.input), inputs.get(c.operands[1].as_slab().load.input))
    ):
        return None
    if stage.transport == "smem-tma" and not (
        _tma_operand_rank(c.operands[0].as_slab().load.index, tile.m.axis.name, k_axis.name)
        and _tma_operand_rank(c.operands[1].as_slab().load.index, tile.n.axis.name, k_axis.name)
    ):
        return None
    # Staging needs the CTA to BE one (tile_m x tile_n) output tile (the cooperative fill / drain
    # contract). A register-only tile launches the scalar default block over unrelated cells.
    if tile.launch_threads is None:
        return None
    if stage.transport == "smem-tma" and max(tile.m.tile, tile.n.tile) > _TMA_MAX_BOX:
        return None
    k = k_axis.extent.as_static()
    elem_bytes = inputs[c.operands[0].as_slab().load.input].dtype.nbytes
    # Every staged transport needs 16 B-aligned inner global strides — A's is K, B's is N.
    n_ext = tile.n.axis.extent
    if not n_ext.is_static or (k * elem_bytes) % _TMA_ALIGN or (n_ext.as_static() * elem_bytes) % _TMA_ALIGN:
        return None
    b_bytes = inputs[c.operands[1].as_slab().load.input].dtype.nbytes if c.operands[1].as_slab().load.input in inputs else elem_bytes
    # A scalar tile always copies with the blocking load/store, so its ``smem`` ring splits too.
    requested = min(stage.depth, SPLIT_COPY_DEPTH) if stage.transport == "smem" else stage.depth
    depth, bk_elems = max(1, requested), 0
    while depth >= 1:
        cap = budget // (depth * max(1, tile.m.tile * elem_bytes + tile.n.tile * b_bytes))
        bk_elems = next((v for v in (128, 64, 32, 16, 8, 4) if v <= cap and k % v == 0), 0)
        if bk_elems >= 4:
            break
        depth -= 1
    if bk_elems < 4:
        return None
    return ResolvedStage(replace(stage, depth=depth, reg_depth=1), bk_elems=bk_elems)


# ---- the smem compute fill --------------------------------------------------------------------- #


def converting_a(node: Fold, atom, inputs) -> bool:
    """Whether the ``a`` edge is a MATERIALIZED load whose dtype the atom cannot bind directly —
    the CONVERTING smem compute fill's case (an erased ``.float()`` cast ahead of an f16
    projection): the synchronous fill evaluates the load per slab cell and the typed slab store
    performs the conversion. A byte transport moves raw bits and cannot, so such an edge takes the
    fill or nothing. ``False`` for computed edges (the fill's native case), matching dtypes, and
    1-byte loads (the fp8 tiers move raw bits by design)."""
    if node.operands[0].as_slab() is None or not inputs:
        return False
    if atom.operand_dtype("a").nbytes < 2:
        return False
    t = inputs.get(node.operands[0].as_slab().load.input)
    return t is not None and t.dtype.nbytes >= 2 and t.dtype != atom.operand_dtype("a")


def computed_operand_cover(c: Fold, tile: Tile, *, converting: bool = False, k_axis: Axis) -> str | None:
    """Geometry required by a smem compute-filled contraction operand.

    A computed A leaves B on the async-copy path, whose contiguous N-vector copy cannot clamp a
    partial inner row element-by-element, so N must be exact. A computed B leaves materialized A
    as the async operand; M is its *outer* slab row and can be safely clamped as a whole. Computed
    B's own per-cell fill clamps N before evaluating the generic producer cone.

    A SYMBOLIC K rides the fill's own K mask: the cone's reads clamp in-bounds and every slab lane
    whose k index reaches past the runtime extent stores the fold identity 0 (the bilinear reading
    pins ⊕ = add, so a zero A contributes nothing and the drain may read the whole chunk
    unconditionally). What that mask cannot cover is a BYTE-COPIED peer whose slab row is K-MAJOR —
    a materialized A, and a transposed B, both stage K as the slab's contiguous inner dim, so their
    cp.async chunk runs ALONG K and a clamped chunk START still copies past the extent. Those keep
    the refusal; a converting materialized A does not, since it rides the fill as a cone.

    ``k_axis`` overrides the stored axis for a derived unit-marker contraction whose enclosing
    Fold owns the actual K sweep."""
    if not k_axis.extent.is_static:
        if c.operands[0].as_slab() is not None and not converting:
            return (
                "a materialized A stages K-major (K is the slab's contiguous row), so its cp.async "
                "chunk runs along K and cannot clamp a symbolic K's partial tail; the masked fill "
                "covers a COMPUTED (or converting) A only"
            )
        if c.as_contraction().b_trans:
            return (
                "a transposed B stages N-major (K contiguous), so its cp.async chunk runs along K "
                "and cannot clamp a symbolic K's partial tail; pin a canonical B layout"
            )
    materialized_b = [edge.as_slab() is not None for edge in c.operands[1:]]
    if any(materialized_b) and not all(materialized_b):
        return "the smem compute fill requires homogeneous B channels; mixed computed/materialized B layouts stay on the demoted reading"
    if tile.n.mask and any(edge.as_slab() is not None for edge in c.operands[1:]):
        return (
            f"a smem compute fill with a materialized B needs a TILE whose N width exactly covers "
            f"the static output columns (N={tile.n.axis.extent}; copied inner-row chunks cannot "
            f"clamp individual N cells); pick a dividing tile."
        )
    return None


def computed_operand_copy_dtype(c: Fold, tile: Tile, inputs, *, converting: bool = False) -> str | None:
    """Every BYTE-COPIED edge of a compute-filled contraction must already have the atom dtype.

    The ``smem`` stage evaluates computed (and converting) operands into their typed shared-memory
    slabs, but it *copies* every materialized peer byte-for-byte.  A copied f32 edge therefore
    cannot feed an f16 ``ldmatrix`` fragment merely because another edge is filled. Filled edges
    are exempt because their slab store performs the normal typed conversion — ``converting``
    marks a materialized ``a`` that rides the converting fill rather than the copy."""
    for edge, role in ((c.operands[0], "a"), *((edge, "b") for edge in c.operands[1:])):
        if edge.as_slab() is None or (role == "a" and converting):
            continue
        tensor = inputs.get(edge.as_slab().load.input) if inputs else None
        # Structural scheduler fixtures intentionally do not carry Tensor metadata. Absence is not
        # evidence of an unsafe byte copy; the concrete lowering path always supplies inputs and
        # is where a known mismatch must be rejected.
        if tensor is None:
            continue
        want = tile.atom.operand_dtype(role)
        if tensor.dtype == want:
            continue
        return (
            f"smem compute fill: materialized {role.upper()} edge {edge.as_slab().load.input!r} is {tensor.dtype}, but "
            f"atom {tile.atom.name} copies it into a {want} slab without conversion; only the "
            "``a`` role has a converting fill"
        )
    return None


def resolve_fill_stage(
    c: Fold,
    tile: Tile,
    budget: int,
    want_depth: int = 1,
    *,
    inputs=None,
    why: list[str] | None = None,
    seam: tuple | None = None,
    k_axis: Axis,
    producer: Fold | None = None,
    producer_k: Axis | None = None,
    axes: tuple = (),
) -> ResolvedStage | None:
    """The ``smem`` compute-fill :class:`Stage` for a computed-operand warp contraction under
    ``tile`` — MANDATORY for this form (the gmem-direct mma leaf refuses a computed A, and the
    byte-copy / cp.async / TMA transports move bytes and cannot evaluate a producer cone), so it
    has no gmem-direct ``""`` sibling and a ``STAGE`` pin can only choose its DEPTH. ``None`` when
    the slabs exceed ``budget``: one A slab, one B slab per channel, and one fp32 row per bridged
    statistic (:func:`~emmy.compiler.ir.pure.fold.cone_seam`'s ``stats`` and its per-chunk ``chunk`` stats — the same
    node boundary the materializer fills through).

    ``want_depth >= 2`` is the asymmetric B-only prefetch ring: only the B cp.async slabs ring
    (their copies for chunk ``i+d-1`` fly under chunk ``i``'s compute fill and drain), while the
    compute-filled A slab and the stat rows stay single-buffer — ringing a compute fill buys no
    overlap, it runs on the drain's own threads. Both depths are fork siblings, measured per shape.

    ``k_axis`` is the contraction's K with its extent (the enclosing Fold's when the contraction
    is a derived singleton marker), ``producer_k`` the nested producer's; ``axes`` is the kernel's
    axis table for the seam's lowering. ``why`` collects the decline reason when the tier refuses,
    so a PINNED caller reports the gate it actually hit."""
    atom = tile.atom
    if atom.operand_dtype("a").nbytes < 2:
        # fp8 atoms: the compute fill's slab store + ldmatrix drain are 16-bit-only
        _decline(why, f"the smem compute fill is 16-bit-only, but this atom's a operand is {atom.operand_dtype('a').nbytes}-byte")
        return None
    bk_elems = tile.bk * atom.atom_k
    if k_axis.extent.is_static and k_axis.extent.as_static() % bk_elems:
        # the staged driver unrolls WHOLE K chunks — the same rule the copy transports state on their own
        _decline(
            why,
            f"the smem compute fill unrolls whole K chunks, but its {bk_elems}-element chunk "
            f"does not divide the contraction K={k_axis.extent.as_static()}",
        )
        return None
    a_nbytes = atom.operand_dtype("a").nbytes
    b_nbytes = atom.operand_dtype("b").nbytes
    _, _, stats, chunk = (
        seam if seam is not None else cone_seam(c.operands[0], k_axis.name, axes) if c.operands[0].as_slab() is None else ((), (), (), ())
    )
    if chunk and chunk[2] % bk_elems:
        # The chunk statistic is evaluated once per staged chunk, which is only its value when the
        # chunk sits inside one K group.
        _decline(why, f"the fill's per-chunk statistic spans {chunk[2]}-element K groups, which a {bk_elems}-element chunk does not tile")
        return None
    a_bytes = tile.m.tile * bk_elems * a_nbytes
    stat_bytes = (len(stats) + (len(chunk[1]) if chunk else 0)) * tile.m.tile * 4
    sync_bytes = stat_bytes
    async_bytes = 0
    # A materialized A whose dtype the atom cannot bind rides the CONVERTING synchronous fill —
    # per-cell load + typed slab store — never the byte copy (which cannot convert).
    a_converts = converting_a(c, atom, inputs)
    if c.operands[0].as_slab() is not None and not a_converts:
        async_bytes += a_bytes
    else:
        sync_bytes += a_bytes
    for ch in c.operands[1:]:
        if ch.as_slab() is not None:
            async_bytes += tile.n.tile * bk_elems * b_nbytes
        else:
            sync_bytes += tile.n.tile * bk_elems * b_nbytes
    # A scheduled contraction producer contributes its own streamed and invariant operand slabs.
    # They do not ring: the streamed slab dies inside the block and the invariant slab does not
    # advance. Reserve both from the producer interface supplied by the scheduler.
    producer_extent = (
        producer_k.extent.as_static() if producer is not None and producer_k is not None and producer_k.extent.is_static else 0
    )
    producer_bytes = producer_extent * (bk_elems * b_nbytes + tile.m.tile * a_nbytes)
    if sync_bytes + async_bytes + producer_bytes > budget:
        _decline(why, f"the smem compute fill's slabs need {sync_bytes + async_bytes + producer_bytes} B, over the {budget} B smem budget")
        return None
    fixed = sync_bytes + producer_bytes
    # Only the asynchronous peer slabs ring (the compute-filled slab and stat rows stay
    # single-buffer), so the clamp budgets the ringed slot against what the fixed slabs leave.
    depth = _clamp_depth(want_depth, async_bytes, budget - fixed) if async_bytes else 1
    computed = [c.operands[0].exposes[-1]] if a_converts or c.operands[0].as_slab() is None else []
    computed.extend(edge.exposes[-1] for edge in c.operands[1:] if edge.as_slab() is None)
    return ResolvedStage(Stage(depth=depth, transport="smem"), smem=tuple(computed), bk_elems=bk_elems)


__all__ = [
    "chunk_key_stage",
    "computed_operand_copy_dtype",
    "computed_operand_cover",
    "converting_a",
    "resolve_fill_stage",
    "resolve_scalar_stage",
    "resolve_warp_stage",
    "stage_target",
]
