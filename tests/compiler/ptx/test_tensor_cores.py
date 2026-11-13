"""Tensor-core kernels compiled to PTX by the ``emmy-ptx`` crate and launched on a card: ``mma.sync`` fed through
``ldmatrix`` from shared memory, the 8-bit and the block-scaled 4-bit products, and the statements that stay in
registers between two products. A test body writes the fragment layouts by hand, as the PTX reference documents
them, since the statements that expand a fragment into loads and stores are not compiled yet. Operands are small
integers wherever a product must be exact, so each result is compared with NumPy bit for bit."""

from __future__ import annotations

import textwrap

import numpy as np
import pytest

from emmy.compiler import dtype as dtypes
from tests.compiler.helpers import requires_cuda, requires_sm
from tests.compiler.ptx import launch
from tests.compiler.ptx.launch import same_bits, under

pytestmark = [requires_cuda, pytest.mark.xdist_group("cuda"), pytest.mark.parametrize("level", launch.LEVELS, ids=["default", "O0"])]

BLOCKS = 5

GRID = f"Grid(axes=(r in 0..{BLOCKS}, t in 0..32), decode=(block_idx.x, thread_idx.x), block_threads=32):"


def _warp(*parts: str) -> str:
    """A kernel of one block a tile ``r`` and one warp a block, holding ``parts`` in order. ``g`` is the lane's group
    of four and ``q`` its place in the group: the two coordinates every fragment layout is written in."""
    lanes = "let i32 g = t / 4\nlet i32 q = t % 4"
    return under(GRID, "\n".join([lanes, *(textwrap.dedent(part).strip("\n") for part in parts)]))


#: The accumulator of an m16n8 tile into ``o``: a lane's register ``i`` is row ``g + 8 * (i / 2)``, column
#: ``2 * q + i % 2``.
STORE_C = """
f32 c0 = load fc[0]
f32 c1 = load fc[1]
f32 c2 = load fc[2]
f32 c3 = load fc[3]
f32 o[r * 128 + g * 8 + q * 2] = c0
f32 o[r * 128 + g * 8 + q * 2 + 1] = c1
f32 o[r * 128 + (g + 8) * 8 + q * 2] = c2
f32 o[r * 128 + (g + 8) * 8 + q * 2 + 1] = c3
"""


def _integers(shape: tuple[int, ...], seed: int, bound: int = 8) -> np.ndarray:
    return np.random.default_rng(seed).integers(-bound, bound + 1, size=shape).astype(np.float32)


def _bf16(values: np.ndarray) -> np.ndarray:
    """The 16 bits of each value as a bf16: the top half of its f32, exact for the small integers the tests use."""
    return (values.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


@requires_sm(8, 0)
@pytest.mark.parametrize("ab", ["f16", "bf16"])
def test_a_tile_is_multiplied_from_shared_slabs_through_ldmatrix(level, ab):
    """A 16 x 8 tile of ``A @ B`` over K = 48, three ``mma.sync`` steps of 16. Every thread stages a column of A
    and a row of B into shared slabs with plain loads and stores; after the barrier each step loads A's fragment
    with ``ldmatrix`` and B's with the transposed form, and the accumulator's registers are stored by the
    documented layout."""
    k = 48
    a, b = _integers((BLOCKS, 16, k), 1), _integers((BLOCKS, k, 8), 2)
    half = np.float16 if ab == "f16" else np.uint16
    encode = (lambda x: x.astype(np.float16)) if ab == "f16" else _bf16
    body = _warp(
        f"""
        Smem(name=sa, extents=(16, {k}), dtype={ab}, align=16)
        Smem(name=sb, extents=({k}, 8), dtype={ab}, align=16)
        for i in 0..16:
            for j in 0..{k}:
                if j % 32 == t:
                    {ab} x = load a[r * {16 * k} + i * {k} + j]
                    {ab} sa[i * {k} + j] = x
        for i in 0..{k}:
            for j in 0..8:
                if (i * 8 + j) % 32 == t:
                    {ab} y = load b[r * {8 * k} + i * 8 + j]
                    {ab} sb[i * 8 + j] = y
        Sync
        RegFragment(name=fa, role=a, dtype={ab})
        RegFragment(name=fb, role=b, dtype={ab})
        RegFragment(name=fc, role=c, dtype=f32)
        for s in 0..{k // 16}:
            LdmatrixLoad(frag=fa, src_buffer=sa, src_index=(0, s * 16), role=a, ldm={k})
            LdmatrixLoad(frag=fb, src_buffer=sb, src_index=(s * 16, 0), role=b, ldm=8)
            MmaSyncPtx(c_frag=fc, a_frag=fa, b_frag=fb, shape=(16, 8, 16), ab_dtype={ab})
        """,
        STORE_C,
    )
    inputs = {"a": encode(a).astype(half).ravel(), "b": encode(b).astype(half).ravel()}
    types = {"a": dtypes.BF16, "b": dtypes.BF16} if ab == "bf16" else None
    out = launch.run(body, inputs, {"o": (np.float32, BLOCKS * 128)}, blocks=BLOCKS, threads=32, level=level, types=types)["o"]
    same_bits(out.reshape(BLOCKS, 16, 8), a @ b)


def _pairs(values: np.ndarray, ab: str = "f16") -> np.ndarray:
    """Rows of 16-bit elements as rows of 32-bit words, two elements a word, the first the low half: what one
    register of an operand fragment holds where its two elements stand side by side."""
    halves = values.astype(np.float16).view(np.uint16) if ab == "f16" else _bf16(values)
    return np.ascontiguousarray(halves).view(np.uint32)


#: An m16n8k16 B fragment by hand from ``bt``, B stored with K along its rows as pairs: register ``i`` of a lane is
#: column ``g``, rows ``2 * q + 8 * i`` and the next.
FILL_B = """
u32 b0 = load bt[r * 64 + g * 8 + q]
u32 b1 = load bt[r * 64 + g * 8 + q + 4]
u32 fb[0] = b0
u32 fb[1] = b1
"""

#: An m16n8k16 A fragment by hand from ``a`` stored as pairs: register ``i`` is row ``g + 8 * (i % 2)``, columns
#: ``2 * q + 8 * (i / 2)`` and the next.
FILL_A = """
u32 a0 = load a[r * 128 + g * 8 + q]
u32 a1 = load a[r * 128 + (g + 8) * 8 + q]
u32 a2 = load a[r * 128 + g * 8 + q + 4]
u32 a3 = load a[r * 128 + (g + 8) * 8 + q + 4]
u32 fa[0] = a0
u32 fa[1] = a1
u32 fa[2] = a2
u32 fa[3] = a3
"""


def _fill_c(name: str, source: str, row_elements: int, column: int = 0) -> str:
    """An m16n8 accumulator by hand from the f32 buffer ``source``, whose rows hold ``row_elements``: the tile
    that starts at ``column``."""
    lines = []
    for i in range(4):
        row, col = f"(g + {8 * (i // 2)})", f"q * 2 + {column + i % 2}"
        lines.append(f"f32 {name}_{i} = load {source}[r * {16 * row_elements} + {row} * {row_elements} + {col}]")
        lines.append(f"f32 {name}[{i}] = {name}_{i}")
    return "\n".join(lines)


@requires_sm(8, 0)
def test_an_accumulator_of_halves_is_promoted_into_one_of_floats(level):
    """The f16-accumulate product over K = 48 with a promote after each step: the packed halves are widened and
    added into the accumulator of floats, then zeroed, so no step is counted twice."""
    k = 48
    a, b = _integers((BLOCKS, 16, k), 3), _integers((BLOCKS, k, 8), 4)
    body = _warp(
        f"""
        Smem(name=sa, extents=(16, {k}), dtype=f16, align=16)
        Smem(name=sb, extents=({k}, 8), dtype=f16, align=16)
        for i in 0..16:
            for j in 0..{k}:
                if j % 32 == t:
                    f16 x = load a[r * {16 * k} + i * {k} + j]
                    f16 sa[i * {k} + j] = x
        for i in 0..{k}:
            for j in 0..8:
                if (i * 8 + j) % 32 == t:
                    f16 y = load b[r * {8 * k} + i * 8 + j]
                    f16 sb[i * 8 + j] = y
        Sync
        RegFragment(name=fa, role=a, dtype=f16)
        RegFragment(name=fb, role=b, dtype=f16)
        RegFragment(name=fh, role=c, dtype=f16)
        RegFragment(name=fc, role=c, dtype=f32)
        for s in 0..{k // 16}:
            LdmatrixLoad(frag=fa, src_buffer=sa, src_index=(0, s * 16), role=a, ldm={k})
            LdmatrixLoad(frag=fb, src_buffer=sb, src_index=(s * 16, 0), role=b, ldm=8)
            MmaSyncPtx(c_frag=fh, a_frag=fa, b_frag=fb, shape=(16, 8, 16), c_dtype=f16)
            FragmentPromote(dst=fc, src=fh)
        """,
        STORE_C,
    )
    inputs = {"a": a.astype(np.float16).ravel(), "b": b.astype(np.float16).ravel()}
    out = launch.run(body, inputs, {"o": (np.float32, BLOCKS * 128)}, blocks=BLOCKS, threads=32, level=level)["o"]
    same_bits(out.reshape(BLOCKS, 16, 8), a @ b)


@requires_sm(8, 0)
@pytest.mark.parametrize("ab", ["f16", "bf16"])
def test_two_accumulators_are_repacked_into_an_a_fragment(level, ab):
    """Two accumulators side by side along K, filled by hand from a 16 x 16 matrix ``p``, become the A fragment of
    the next product by one packed conversion a register: the tile is ``p @ B``."""
    p, b = _integers((BLOCKS, 16, 16), 5), _integers((BLOCKS, 16, 8), 6)
    body = _warp(
        f"""
        RegFragment(name=c0, role=c, dtype=f32)
        RegFragment(name=c1, role=c, dtype=f32)
        RegFragment(name=fb, role=b, dtype={ab})
        RegFragment(name=fc, role=c, dtype=f32)
        """,
        _fill_c("c0", "p", 16),
        _fill_c("c1", "p", 16, column=8),
        FILL_B,
        f"""
        FragmentRepack(frag=fa, srcs=(c0, c1), ab_dtype={ab})
        MmaSyncPtx(c_frag=fc, a_frag=fa, b_frag=fb, shape=(16, 8, 16), ab_dtype={ab})
        """,
        STORE_C,
    )
    inputs = {"p": p.ravel(), "bt": _pairs(np.swapaxes(b, 1, 2), ab).ravel()}
    out = launch.run(body, inputs, {"o": (np.float32, BLOCKS * 128)}, blocks=BLOCKS, threads=32, level=level)["o"]
    same_bits(out.reshape(BLOCKS, 16, 8), p @ b)


@requires_sm(8, 0)
def test_an_accumulator_is_repacked_into_a_b_fragment_by_an_exchange_between_lanes(level):
    """One accumulator, a 16 x 8 tile filled by hand from ``p``, becomes a B fragment: each lane takes the packed
    pairs of the two lanes that hold its column and keeps its own half of each. The tile is ``A @ p``."""
    a, p = _integers((BLOCKS, 16, 16), 7), _integers((BLOCKS, 16, 8), 8)
    body = _warp(
        """
        RegFragment(name=c0, role=c, dtype=f32)
        RegFragment(name=fa, role=a, dtype=f16)
        RegFragment(name=fb, role=b, dtype=f16)
        RegFragment(name=fc, role=c, dtype=f32)
        """,
        _fill_c("c0", "p", 8),
        FILL_A,
        """
        FragmentRepack(frag=fb, srcs=(c0), role=b)
        MmaSyncPtx(c_frag=fc, a_frag=fa, b_frag=fb, shape=(16, 8, 16))
        """,
        STORE_C,
    )
    inputs = {"p": p.ravel(), "a": _pairs(a).ravel()}
    out = launch.run(body, inputs, {"o": (np.float32, BLOCKS * 128)}, blocks=BLOCKS, threads=32, level=level)["o"]
    same_bits(out.reshape(BLOCKS, 16, 8), a @ p)


@requires_sm(8, 9)
@pytest.mark.parametrize("fmt", ["e4m3", "e5m2"])
def test_a_tile_of_8_bit_floats_is_multiplied(level, fmt):
    """The m16n8k32 product over K = 64, two steps. An operand register holds four bytes that stand side by side:
    A's register ``i`` is row ``g + 8 * (i % 2)``, columns ``4 * q + 16 * (i / 2)`` on; B's is column ``g``, rows
    ``4 * q + 16 * i`` on. B is stored with K along its rows, so each register is one 32-bit load."""
    k = 64
    a, b = _integers((BLOCKS, 16, k), 9, 4), _integers((BLOCKS, k, 8), 10, 4)
    words = k // 4
    step = "\n".join(
        [
            *(f"u32 a{i} = load a[r * {16 * words} + (g + {8 * (i % 2)}) * {words} + s * 8 + q + {4 * (i // 2)}]" for i in range(4)),
            *(f"u32 fa[{i}] = a{i}" for i in range(4)),
            *(f"u32 b{i} = load bt[r * {8 * words} + g * {words} + s * 8 + q + {4 * i}]" for i in range(2)),
            *(f"u32 fb[{i}] = b{i}" for i in range(2)),
            f"MmaSyncPtx(c_frag=fc, a_frag=fa, b_frag=fb, shape=(16, 8, 32), ab_dtype={fmt})",
        ]
    )
    body = _warp(
        f"""
        RegFragment(name=fa, role=a, dtype=f8{fmt})
        RegFragment(name=fb, role=b, dtype=f8{fmt})
        RegFragment(name=fc, role=c, dtype=f32)
        """,
        f"for s in 0..{k // 32}:\n" + textwrap.indent(step, "    "),
        STORE_C,
    )
    encode = lambda x: np.ascontiguousarray(dtypes.encode_f8(x, f"f8{fmt}")).view(np.uint32)  # noqa: E731
    inputs = {"a": encode(a).ravel(), "bt": encode(np.swapaxes(b, 1, 2)).ravel()}
    out = launch.run(body, inputs, {"o": (np.float32, BLOCKS * 128)}, blocks=BLOCKS, threads=32, level=level)["o"]
    same_bits(out.reshape(BLOCKS, 16, 8), a @ b)


def _ue4m3(bits: np.ndarray) -> np.ndarray:
    """The unsigned e4m3 a block scale is: the sign bit of the signed format is unused."""
    exponent, mantissa = (bits >> 3) & 0xF, bits & 7
    subnormal = mantissa / 8 * 2.0**-6
    normal = (1 + mantissa / 8) * 2.0 ** (exponent.astype(np.int32) - 7)
    return np.where(exponent == 0, subnormal, normal).astype(np.float32)


@requires_sm(12, 0)
def test_the_block_scaled_product_of_4_bit_floats_matches_the_decoded_oracle(level):
    """The m16n8k64 product whose every 16 elements carry a scale a side, against the oracle of
    ``test_nvfp4_atom``: each block's sum times its two scales. The operands are random codes and the scales are
    held to exponents that keep every product exact, so the comparison is bit for bit. Only the lanes the
    instruction's selector names supply scales; ``BlockScaleLoad`` leaves every other at zero."""
    rng = np.random.default_rng(23)
    a_bits = rng.integers(0, 256, size=(BLOCKS, 16, 32), dtype=np.uint8)
    b_bits = rng.integers(0, 256, size=(BLOCKS, 8, 32), dtype=np.uint8)
    sa_bits = (0x30 + rng.integers(0, 0x19, size=(BLOCKS, 16, 4))).astype(np.uint8)
    sb_bits = (0x30 + rng.integers(0, 0x19, size=(BLOCKS, 8, 4))).astype(np.uint8)
    fill = "\n".join(
        [
            *(f"u32 a{i} = load a[r * 128 + (g + {8 * (i % 2)}) * 8 + q + {4 * (i // 2)}]" for i in range(4)),
            *(f"u32 fa[{i}] = a{i}" for i in range(4)),
            *(f"u32 b{i} = load bt[r * 64 + g * 8 + q + {4 * i}]" for i in range(2)),
            *(f"u32 fb[{i}] = b{i}" for i in range(2)),
        ]
    )
    body = _warp(
        """
        RegFragment(name=fa, role=a, dtype=f4e2m1x2)
        RegFragment(name=fb, role=b, dtype=f4e2m1x2)
        RegFragment(name=fc, role=c, dtype=f32)
        RegFragment(name=sfa, role=sfa, dtype=u8)
        RegFragment(name=sfb, role=sfb, dtype=u8)
        """,
        fill,
        """
        BlockScaleLoad(frag=sfa, src_buffer=sa, src_index=(r * 64), role=a, ldm=4)
        BlockScaleLoad(frag=sfb, src_buffer=sb, src_index=(r * 32), role=b, ldm=4)
        MmaSyncPtx(c_frag=fc, a_frag=fa, b_frag=fb, shape=(16, 8, 64), ab_dtype=e2m1, sfa_frag=sfa, sfb_frag=sfb)
        """,
        STORE_C,
    )
    inputs = {"a": a_bits.view(np.uint32).ravel(), "bt": b_bits.view(np.uint32).ravel(), "sa": sa_bits.ravel(), "sb": sb_bits.ravel()}
    out = launch.run(body, inputs, {"o": (np.float32, BLOCKS * 128)}, blocks=BLOCKS, threads=32, level=level, arch_specific=True)["o"]
    a = dtypes.decode_f4x2(a_bits).astype(np.float32)
    b = dtypes.decode_f4x2(b_bits).astype(np.float32)
    per_block = (a.reshape(BLOCKS, 16, 1, 4, 16) * b.reshape(BLOCKS, 1, 8, 4, 16)).sum(-1)
    want = (per_block * _ue4m3(sa_bits)[:, :, None, :] * _ue4m3(sb_bits)[:, None, :, :]).sum(-1)
    same_bits(out.reshape(BLOCKS, 16, 8), want.astype(np.float32))


#: Volta's m8n8k4 layouts, by hand. One instruction is four 8 x 8 products, one a group of lanes: ``arow`` is the
#: row of A a lane holds and ``bcol`` the column of B; an operand register is two elements along K. The
#: accumulator of floats is one 16 x 16 tile: a lane's register ``i`` is row ``vr + 2 * (i / 2 % 2)``, column
#: ``vc + i % 2 + 4 * (i / 4)``. A is stored as pairs, and B with K along its rows as pairs.
VOLTA_LANES = """
let i32 comp = t % 16 / 4
let i32 arow = comp / 2 * 8 + t % 4 + t / 16 * 4
let i32 bcol = comp % 2 * 8 + t % 4 + t / 16 * 4
let i32 vr = comp / 2 * 8 + t / 16 * 4 + t % 2
let i32 vc = comp % 2 * 8 + t / 2 % 2 * 2
"""
VOLTA_STORE = "\n".join(
    line
    for i in range(8)
    for line in (f"f32 c{i} = load fc[{i}]", f"f32 o[r * 256 + (vr + {2 * (i // 2 % 2)}) * 16 + vc + {i % 2 + 4 * (i // 4)}] = c{i}")
)


def _volta_fill(words: int, step: str) -> str:
    """The A and B fragments of one m8n8k4 step from ``a`` and ``bt``, rows of ``words`` pairs."""
    return "\n".join(
        [
            *(f"u32 a{i} = load a[r * {16 * words} + arow * {words} + {step} * 2 + {i}]" for i in range(2)),
            *(f"u32 fa[{i}] = a{i}" for i in range(2)),
            *(f"u32 b{i} = load bt[r * {16 * words} + bcol * {words} + {step} * 2 + {i}]" for i in range(2)),
            *(f"u32 fb[{i}] = b{i}" for i in range(2)),
        ]
    )


@pytest.mark.parametrize("acc", ["f32", "f16"])
def test_volta_s_product_fills_a_16_by_16_tile(level, acc):
    """The m8n8k4 product over K = 12, three steps, into an accumulator of floats, and into one of halves that is
    promoted after each step: there a lane's halves are one row of the tile and its floats two, so the promote
    takes each pair from the lane that holds it."""
    k = 12
    a, b = _integers((BLOCKS, 16, k), 11), _integers((BLOCKS, k, 16), 12)
    if acc == "f32":
        product = "MmaSyncPtx(c_frag=fc, a_frag=fa, b_frag=fb, shape=(8, 8, 4))"
    else:
        product = "MmaSyncPtx(c_frag=fh, a_frag=fa, b_frag=fb, shape=(8, 8, 4), c_dtype=f16)\n"
        product += "FragmentPromote(dst=fc, src=fh, fragment_layout=m8n8k4)"
    body = _warp(
        VOLTA_LANES,
        """
        RegFragment(name=fa, role=a, dtype=f16, fragment_layout=m8n8k4)
        RegFragment(name=fb, role=b, dtype=f16, fragment_layout=m8n8k4)
        RegFragment(name=fh, role=c, dtype=f16, fragment_layout=m8n8k4)
        RegFragment(name=fc, role=c, dtype=f32, fragment_layout=m8n8k4)
        """,
        f"for s in 0..{k // 4}:\n" + textwrap.indent(_volta_fill(k // 2, "s") + "\n" + product, "    "),
        VOLTA_STORE,
    )
    inputs = {"a": _pairs(a).ravel(), "bt": _pairs(np.swapaxes(b, 1, 2)).ravel()}
    out = launch.run(body, inputs, {"o": (np.float32, BLOCKS * 256)}, blocks=BLOCKS, threads=32, level=level)["o"]
    same_bits(out.reshape(BLOCKS, 16, 16), a @ b)


def test_volta_s_product_reads_b_by_rows(level):
    """The second m8n8k4 form takes B with N along a register: a lane holds row ``t % 4`` of the step, and its two
    registers are four columns from ``bcol - t % 4`` on."""
    k = 8
    a, b = _integers((BLOCKS, 16, k), 13), _integers((BLOCKS, k, 16), 14)
    step = "\n".join(
        [
            *(f"u32 a{i} = load a[r * {16 * k // 2} + arow * {k // 2} + s * 2 + {i}]" for i in range(2)),
            *(f"u32 fa[{i}] = a{i}" for i in range(2)),
            *(f"u32 b{i} = load b[r * {k * 8} + (s * 4 + t % 4) * 8 + (bcol - t % 4) / 2 + {i}]" for i in range(2)),
            *(f"u32 fb[{i}] = b{i}" for i in range(2)),
            "MmaSyncPtx(c_frag=fc, a_frag=fa, b_frag=fb, shape=(8, 8, 4), b_row_major=true)",
        ]
    )
    body = _warp(
        VOLTA_LANES,
        """
        RegFragment(name=fa, role=a, dtype=f16, fragment_layout=m8n8k4)
        RegFragment(name=fb, role=b, dtype=f16, fragment_layout=m8n8k4)
        RegFragment(name=fc, role=c, dtype=f32, fragment_layout=m8n8k4)
        """,
        f"for s in 0..{k // 4}:\n" + textwrap.indent(step, "    "),
        VOLTA_STORE,
    )
    inputs = {"a": _pairs(a).ravel(), "b": _pairs(b).ravel()}
    out = launch.run(body, inputs, {"o": (np.float32, BLOCKS * 256)}, blocks=BLOCKS, threads=32, level=level)["o"]
    same_bits(out.reshape(BLOCKS, 16, 16), a @ b)


VOLTA_FILL_C = "\n".join(
    line
    for i in range(8)
    for line in (f"f32 p{i} = load p[r * 256 + (vr + {2 * (i // 2 % 2)}) * 16 + vc + {i % 2 + 4 * (i // 4)}]", f"f32 c0[{i}] = p{i}")
)


@pytest.mark.parametrize("part", range(4))
@pytest.mark.parametrize("role", ["a", "b"])
def test_volta_s_accumulator_is_repacked_into_an_operand(level, role, part):
    """One m8n8k4 accumulator, a 16 x 16 tile filled by hand from ``p``, gives four of its columns as an A
    fragment or four of its rows as a B fragment, ``part`` saying which four. The lanes exchange packed pairs and
    each keeps what its own place in the operand's layout holds."""
    p, other = _integers((BLOCKS, 16, 16), 15), _integers((BLOCKS, 16, 4), 16)
    if role == "a":
        fill = "\n".join([*(f"u32 b{i} = load x[r * 32 + bcol * 2 + {i}]" for i in range(2)), *(f"u32 fb[{i}] = b{i}" for i in range(2))])
        want = p[:, :, 4 * part : 4 * part + 4] @ np.swapaxes(other, 1, 2)
    else:
        fill = "\n".join([*(f"u32 a{i} = load x[r * 32 + arow * 2 + {i}]" for i in range(2)), *(f"u32 fa[{i}] = a{i}" for i in range(2))])
        want = other @ p[:, 4 * part : 4 * part + 4, :]
    kept = "b" if role == "a" else "a"
    body = _warp(
        VOLTA_LANES,
        f"""
        RegFragment(name=c0, role=c, dtype=f32, fragment_layout=m8n8k4)
        RegFragment(name=f{kept}, role={kept}, dtype=f16, fragment_layout=m8n8k4)
        RegFragment(name=fc, role=c, dtype=f32, fragment_layout=m8n8k4)
        """,
        VOLTA_FILL_C,
        fill,
        f"""
        FragmentRepack(frag=f{role}, srcs=(c0), fragment_layout=m8n8k4, part={part}, role={role})
        MmaSyncPtx(c_frag=fc, a_frag=fa, b_frag=fb, shape=(8, 8, 4))
        """,
        VOLTA_STORE,
    )
    inputs = {"p": p.ravel(), "x": _pairs(other).ravel()}
    out = launch.run(body, inputs, {"o": (np.float32, BLOCKS * 256)}, blocks=BLOCKS, threads=32, level=level)["o"]
    same_bits(out.reshape(BLOCKS, 16, 16), want)


@requires_sm(8, 0)
@pytest.mark.parametrize("mode", ["B128", "B64", "B32"])
def test_ldmatrix_reads_a_swizzled_slab(level, mode):
    """The slabs are filled with each 16-byte chunk moved the way a swizzle mode moves it, the index written out in
    the test body, and ``ldmatrix`` reads them back through the statement's own swizzle. A's tile starts on a row
    and a column a lane's offset shares no bit with, so its lane part is swizzled alone; B's rows are too short
    for that and each load swizzles its whole index."""
    k, mask = 64, {"B128": 7, "B64": 3, "B32": 1}[mode]
    a, b = _integers((BLOCKS, 16, k), 17), _integers((BLOCKS, k, 8), 18)

    def moved(index: str) -> str:
        return f"({index}) ^ (((({index}) >> 6) & {mask}) << 3)"

    body = _warp(
        f"""
        Smem(name=sa, extents=(16, {k}), dtype=f16, align=1024)
        Smem(name=sb, extents=({k}, 8), dtype=f16, align=1024)
        for i in 0..16:
            for j in 0..{k}:
                if j % 32 == t:
                    f16 x = load a[r * {16 * k} + i * {k} + j]
                    f16 sa[{moved(f"i * {k} + j")}] = x
        for i in 0..{k}:
            for j in 0..8:
                if (i * 8 + j) % 32 == t:
                    f16 y = load b[r * {8 * k} + i * 8 + j]
                    f16 sb[{moved("i * 8 + j")}] = y
        Sync
        RegFragment(name=fa, role=a, dtype=f16)
        RegFragment(name=fb, role=b, dtype=f16)
        RegFragment(name=fc, role=c, dtype=f32)
        for s in 0..{k // 16}:
            LdmatrixLoad(frag=fa, src_buffer=sa, src_index=(0, s * 16), role=a, ldm={k}, swizzle={mode})
            LdmatrixLoad(frag=fb, src_buffer=sb, src_index=(s * 16, 0), role=b, ldm=8, swizzle={mode})
            MmaSyncPtx(c_frag=fc, a_frag=fa, b_frag=fb, shape=(16, 8, 16))
        """,
        STORE_C,
    )
    inputs = {"a": a.astype(np.float16).ravel(), "b": b.astype(np.float16).ravel()}
    out = launch.run(body, inputs, {"o": (np.float32, BLOCKS * 128)}, blocks=BLOCKS, threads=32, level=level)["o"]
    same_bits(out.reshape(BLOCKS, 16, 8), a @ b)
