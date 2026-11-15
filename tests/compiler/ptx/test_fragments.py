"""The fragment statements compiled to PTX by the ``emmy-ptx`` crate and launched on a card: operand fragments
gathered element by element from global memory (with a clamp of the operand's axis and a zero-fill of K), the
accumulator stored by the lane-to-element table (under guards, through an epilogue), and the operations per element
and the fold along rows that stay in registers. Operands are small integers, so every product is exact and each
result is compared with NumPy bit for bit."""

from __future__ import annotations

import textwrap

import numpy as np
import pytest

from tests.compiler.helpers import requires_cuda, requires_sm
from tests.compiler.ptx import launch
from tests.compiler.ptx.launch import same_bits, under

pytestmark = [
    requires_cuda,
    requires_sm(8, 0),
    pytest.mark.xdist_group("cuda"),
    pytest.mark.parametrize("level", launch.LEVELS, ids=["default", "O0"]),
]

BLOCKS = 3

GRID = f"Grid(axes=(r in 0..{BLOCKS}, t in 0..32), decode=(block_idx.x, thread_idx.x), block_threads=32):"


def _warp(*parts: str) -> str:
    """A kernel of one block a tile ``r`` and one warp a block, holding ``parts`` in order."""
    return under(GRID, "\n".join(textwrap.dedent(part).strip("\n") for part in parts))


def _integers(shape: tuple[int, ...], seed: int, bound: int = 8) -> np.ndarray:
    return np.random.default_rng(seed).integers(-bound, bound + 1, size=shape).astype(np.float32)


FRAGMENTS = """
RegFragment(name=fa, role=a, dtype=f16)
RegFragment(name=fb, role=b, dtype=f16)
RegFragment(name=fc, role=c, dtype=f32)
"""


@pytest.mark.parametrize("source", ["f16", "f32"])
@pytest.mark.parametrize("b_trans", [False, True], ids=["b", "b_trans"])
def test_operands_are_gathered_from_global_memory_and_the_accumulator_stored(level, b_trans, source):
    """A 16 x 8 tile of ``A @ B`` over K = 32 in two steps: each lane gathers the elements its fragments hold from
    global memory, converting an f32 source to the fragment's halves, and stores the four floats it holds. B is read
    with K along its rows, or transposed with K along a row."""
    k = 32
    a, b = _integers((BLOCKS, 16, k), 1), _integers((BLOCKS, k, 8), 2)
    stored = np.float16 if source == "f16" else np.float32
    if b_trans:
        b_load = f"src_index=(r * {8 * k} + s * 16), ldm={k}, b_trans=true"
        b_data = np.ascontiguousarray(b.transpose(0, 2, 1))
    else:
        b_load = f"src_index=(r * {8 * k} + s * 128), ldm=8"
        b_data = b
    body = _warp(
        FRAGMENTS,
        f"""
        for s in 0..2:
            FragmentLoad(frag=fa, src_buffer=a, src_index=(r * {16 * k} + s * 16), role=a, ldm={k}, src_dtype={source}, dtype=f16)
            FragmentLoad(frag=fb, src_buffer=b, {b_load}, role=b, src_dtype={source}, dtype=f16)
            MmaSyncPtx(c_frag=fc, a_frag=fa, b_frag=fb, shape=(16, 8, 16))
        RegStore(dst_buffer=o, dst_index=(r * 128), frag=fc, shape=(16, 8, 16), ldm=8)
        """,
    )
    inputs = {"a": a.astype(stored).ravel(), "b": b_data.astype(stored).ravel()}
    out = launch.run(body, inputs, {"o": (np.float32, BLOCKS * 128)}, blocks=BLOCKS, threads=32, level=level)["o"]
    same_bits(out.reshape(BLOCKS, 16, 8), a @ b)


@requires_sm(8, 9)
def test_an_e4m3_operand_is_gathered_into_a_bfloat16_fragment(level):
    """A 16 x 8 tile of ``A @ B`` over K = 32, A of bfloat16 and B transposed of e4m3: each register of B is two
    e4m3 elements side by side, read as one pair and converted into the fragment's bfloat16."""
    import torch

    from emmy.compiler import dtype as dtypes

    k = 32
    a, b = _integers((BLOCKS, 16, k), 5), _integers((BLOCKS, k, 8), 6)
    a_bits = (a.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    b_bits = torch.from_numpy(np.ascontiguousarray(b.transpose(0, 2, 1))).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    b_load = f"src_index=(r * {8 * k} + s * 16), ldm={k}, b_trans=true"
    body = _warp(
        """
        RegFragment(name=fa, role=a, dtype=bf16)
        RegFragment(name=fb, role=b, dtype=bf16)
        RegFragment(name=fc, role=c, dtype=f32)
        """,
        f"""
        for s in 0..2:
            FragmentLoad(frag=fa, src_buffer=a, src_index=(r * {16 * k} + s * 16), role=a, ldm={k}, src_dtype=bf16, dtype=bf16)
            FragmentLoad(frag=fb, src_buffer=b, {b_load}, role=b, src_dtype=f8e4m3, dtype=bf16)
            MmaSyncPtx(c_frag=fc, a_frag=fa, b_frag=fb, shape=(16, 8, 16), ab_dtype=bf16)
        RegStore(dst_buffer=o, dst_index=(r * 128), frag=fc, shape=(16, 8, 16), ldm=8)
        """,
    )
    inputs = {"a": a_bits.ravel(), "b": b_bits.ravel()}
    types = {"a": dtypes.BF16, "b": dtypes.F8E4M3}
    out = launch.run(body, inputs, {"o": (np.float32, BLOCKS * 128)}, blocks=BLOCKS, threads=32, level=level, types=types)["o"]
    same_bits(out.reshape(BLOCKS, 16, 8), a @ b)


def test_a_tile_past_its_bounds_is_clamped_zero_filled_and_guarded(level):
    """A tile of 16 x 8 over buffers of 5 rows, 27 along K and 6 columns. The operand's own axis is clamped, so no
    lane reads past the fifth row of A or the sixth column of B; K past 27 is zero and not read, which a row stride
    of 27 would otherwise wrap into the next row; and only the 5 x 6 elements in range are stored."""
    m, k, n = 5, 27, 6
    a, b = _integers((BLOCKS, m, k), 3), _integers((BLOCKS, k, n), 4)
    halves, k_left = "src_dtype=f16, dtype=f16", f"k_zero=(s * 16, {k})"
    body = _warp(
        FRAGMENTS,
        f"""
        for s in 0..2:
            FragmentLoad(frag=fa, src_buffer=a, src_index=(r * {m * k} + s * 16), role=a, ldm={k}, clamp=(0, {m}), {k_left}, {halves})
            FragmentLoad(frag=fb, src_buffer=b, src_index=(r * {k * n} + s * {16 * n}), role=b, ldm={n}, clamp=(0, {n}), {k_left}, {halves})
            MmaSyncPtx(c_frag=fc, a_frag=fa, b_frag=fb, shape=(16, 8, 16))
        RegStore(dst_buffer=o, dst_index=(r * {m * n}), frag=fc, shape=(16, 8, 16), ldm={n}, m_guard=(0, {m}), n_guard=(0, {n}))
        """,
    )
    inputs = {"a": a.astype(np.float16).ravel(), "b": b.astype(np.float16).ravel()}
    out = launch.run(body, inputs, {"o": (np.float32, BLOCKS * m * n)}, blocks=BLOCKS, threads=32, level=level)["o"]
    same_bits(out.reshape(BLOCKS, m, n), a @ b)


def test_a_store_epilogue_computes_each_element_at_its_coordinates(level):
    """The epilogue of a store runs once an element, with the element of the accumulator bound and the element's
    column in the fragment: here a bias of the column is added, and the sum is stored as a half."""
    k = 16
    a, b, bias = _integers((BLOCKS, 16, k), 5), _integers((BLOCKS, k, 8), 6), _integers((8,), 7)
    body = _warp(
        FRAGMENTS,
        f"""
        FragmentLoad(frag=fa, src_buffer=a, src_index=(r * {16 * k}), role=a, ldm={k}, src_dtype=f16, dtype=f16)
        FragmentLoad(frag=fb, src_buffer=b, src_index=(r * {8 * k}), role=b, ldm=8, src_dtype=f16, dtype=f16)
        MmaSyncPtx(c_frag=fc, a_frag=fa, b_frag=fb, shape=(16, 8, 16))
        RegStore(dst_buffer=o, dst_index=(r * 128), frag=fc, shape=(16, 8, 16), ldm=8, epilogue_params=(acc), epilogue_result=v):
            epilogue:
                f32 bias_n = load bias[__N__]
                f32 v = acc + bias_n
        """,
    )
    inputs = {"a": a.astype(np.float16).ravel(), "b": b.astype(np.float16).ravel(), "bias": bias}
    out = launch.run(body, inputs, {"o": (np.float16, BLOCKS * 128)}, blocks=BLOCKS, threads=32, level=level)["o"]
    same_bits(out.reshape(BLOCKS, 16, 8), (a @ b + bias).astype(np.float16))


#: The condition of an element above the diagonal, by its row and column, and the value a masked element takes.
ABOVE = "FragmentArg(kind=coord, expressions=(__frow < __fcol))"
FILL = "FragmentArg(kind=uniform, expressions=(-1e+30))"


def test_an_operation_per_element_and_a_fold_along_rows(level):
    """In registers: the accumulator scaled by a constant, masked above its diagonal by a condition over each
    element's row and column, folded to each row's maximum over the lanes that share the row, and the maximum of its
    row subtracted from each element. The fold and the difference are stored."""
    k = 16
    a, b = _integers((BLOCKS, 16, k), 8), _integers((BLOCKS, k, 8), 9)
    body = _warp(
        FRAGMENTS,
        f"""
        FragmentLoad(frag=fa, src_buffer=a, src_index=(r * {16 * k}), role=a, ldm={k}, src_dtype=f16, dtype=f16)
        FragmentLoad(frag=fb, src_buffer=b, src_index=(r * {8 * k}), role=b, ldm=8, src_dtype=f16, dtype=f16)
        MmaSyncPtx(c_frag=fc, a_frag=fa, b_frag=fb, shape=(16, 8, 16))
        FragmentApply(out=fh, op=multiply, args=(FragmentArg(kind=uniform, expressions=(0.5)), FragmentArg(kind=frag, names=(fc))))
        FragmentApply(out=fh, op=where, args=({ABOVE}, {FILL}, FragmentArg(kind=frag, names=(fh))), in_place=true, row_base=0, col_base=0)
        FragmentRowReduce(top=top, bot=bot, frags=(fh), op=maximum)
        FragmentApply(out=fd, op=subtract, args=(FragmentArg(kind=frag, names=(fh)), FragmentArg(kind=row, names=(top, bot))))
        RegStore(dst_buffer=o, dst_index=(r * 128), frag=fd, shape=(16, 8, 16), ldm=8)
        let i32 g = t / 4
        if t % 4 == 0:
            f32 peak[r * 16 + g] = top
            f32 peak[r * 16 + g + 8] = bot
        """,
    )
    inputs = {"a": a.astype(np.float16).ravel(), "b": b.astype(np.float16).ravel()}
    outputs = {"o": (np.float32, BLOCKS * 128), "peak": (np.float32, BLOCKS * 16)}
    out = launch.run(body, inputs, outputs, blocks=BLOCKS, threads=32, level=level)
    scaled = (a @ b) * np.float32(0.5)
    rows, columns = np.indices((16, 8))
    masked = np.where(rows < columns, np.float32(-1e30), scaled).astype(np.float32)
    peak = masked.max(axis=2)
    same_bits(out["peak"].reshape(BLOCKS, 16), peak)
    same_bits(out["o"].reshape(BLOCKS, 16, 8), (masked - peak[..., None]).astype(np.float32))
