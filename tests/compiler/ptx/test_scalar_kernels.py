"""Scalar kernels compiled to PTX by the ``emmy-ptx`` crate, assembled by ptxas, launched through the runtime and
compared with NumPy: bit for bit wherever no fused multiply-add is involved."""

from __future__ import annotations

import numpy as np
import pytest

from tests.compiler.helpers import requires_cuda
from tests.compiler.ptx import launch
from tests.compiler.ptx.launch import DTYPES as _DTYPES
from tests.compiler.ptx.launch import floats as _floats
from tests.compiler.ptx.launch import same_bits as _same_bits

pytestmark = [requires_cuda, pytest.mark.xdist_group("cuda"), pytest.mark.parametrize("level", launch.LEVELS, ids=["default", "O0"])]

#: Threads per block of every kernel here.
BLOCK = 256

#: One thread per element ``i`` of ``{n}``, with the launch's tail guarded: the statements of a test go under it.
ELEMENTWISE = """
Grid(axes=(i in 0..{n}), decode=(block_idx.x * 256 + thread_idx.x), block_threads=256):
    if block_idx.x * 256 + thread_idx.x < {n}:
"""


def _run(body: str, inputs: dict[str, np.ndarray], outputs: dict[str, np.dtype], cells: int, *, level: str | None, fuse: bool = False):
    """Launch ``body`` as ``ceil(cells / BLOCK)`` blocks. Every output holds one element per launched thread and
    starts at zero, so a thread past the cells that stores anything shows."""
    blocks = -(-cells // BLOCK)
    sized = {name: (dtype, blocks * BLOCK) for name, dtype in outputs.items()}
    return launch.run(body, inputs, sized, blocks=blocks, level=level, fuse=fuse)


def _elementwise(statements: str, n: int) -> str:
    """``statements`` as the body of one thread per element ``i`` of ``n``."""
    return launch.under(ELEMENTWISE.format(n=n), statements, depth=2)


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_pointwise_add_and_multiply_add_match_numpy_bit_for_bit(dtype, level):
    """``a + b`` and, with the fusion decision off, ``a * b + c`` round each operation as NumPy does, in f32 and
    natively in f16. The 1,000 cells leave a tail of 24 threads, which store nothing."""
    n, ty = 1000, _DTYPES[np.dtype(dtype)].name
    a, b, c = (_floats(n, dtype, seed) for seed in range(3))
    body = _elementwise(
        f"""
        {ty} x = load a[i]
        {ty} y = load b[i]
        {ty} z = load c[i]
        {ty} s = x + y
        {ty} p = x * y
        {ty} m = p + z
        {ty} sum[i] = s
        {ty} mad[i] = m
        """,
        n,
    )
    out = _run(body, {"a": a, "b": b, "c": c}, {"sum": dtype, "mad": dtype}, n, level=level)
    _same_bits(out["sum"][:n], a + b)
    _same_bits(out["mad"][:n], a * b + c)
    assert not out["sum"][n:].any() and not out["mad"][n:].any()


def test_a_fused_multiply_add_rounds_once(level):
    """With the fusion decision on, ``a * b + c`` is one ``fma``: the product is not rounded before the sum. Every
    result is the float64 value rounded to f32, and some differ from the two-rounding result."""
    n = 4096
    a, b, c = (_floats(n, np.float32, seed) for seed in range(3))
    body = _elementwise(
        """
        f32 x = load a[i]
        f32 y = load b[i]
        f32 z = load c[i]
        f32 p = x * y
        f32 m = p + z
        f32 mad[i] = m
        """,
        n,
    )
    out = _run(body, {"a": a, "b": b, "c": c}, {"mad": np.float32}, n, level=level, fuse=True)["mad"]
    # The product of two f32 values is exact in float64, so this is the fused value up to one more rounding.
    fused = (a.astype(np.float64) * b + c).astype(np.float32)
    np.testing.assert_allclose(out, fused, rtol=2e-7, atol=0)
    assert (out != a * b + c).any()


def test_a_loop_sums_its_trips_in_order(level):
    """A reduce loop steps its accumulator once per trip, in order: the f32 sum and sum of squares of eight elements
    per cell, each equal to NumPy's sum taken in the same order."""
    n, trips = 2000, 8
    a = _floats(n * trips, np.float32)
    body = _elementwise(
        f"""
        for j in 0..{trips}:
            f32 x = load a[i * {trips} + j]
            f32 q = x * x
            f32 s <- add(s, x) over j
            f32 t <- add(t, q) over j
            f32 m <- maximum(m, x) over j
        f32 sum[i] = s
        f32 squares[i] = t
        f32 largest[i] = m
        """,
        n,
    )
    out = _run(body, {"a": a}, {"sum": np.float32, "squares": np.float32, "largest": np.float32}, n, level=level)
    rows = a.reshape(n, trips)
    total, squares = np.zeros(n, np.float32), np.zeros(n, np.float32)
    for j in range(trips):
        total = total + rows[:, j]
        squares = squares + rows[:, j] * rows[:, j]
    _same_bits(out["sum"][:n], total)
    _same_bits(out["squares"][:n], squares)
    _same_bits(out["largest"][:n], rows.max(axis=1))


def test_a_select_takes_its_first_branch_that_holds(level):
    n = 1500
    a, b = _floats(n, np.float32, 0), _floats(n, np.float32, 1)
    body = _elementwise(
        """
        f32 x = load a[i]
        f32 y = load b[i]
        f32 z = -x
        v = select:
            x when i % 3 == 0
            y when x > y
            z when i >= 0
        f32 o[i] = v
        """,
        n,
    )
    out = _run(body, {"a": a, "b": b}, {"o": np.float32}, n, level=level)["o"]
    i = np.arange(n)
    _same_bits(out[:n], np.where(i % 3 == 0, a, np.where(a > b, b, -a)))


def test_an_integer_index_divides_and_takes_remainders_as_python_does(level):
    """A gather through quotients and remainders: by a power of two, by another constant over a non-negative index
    and over a value that goes negative (the multiply-high forms), and by a divisor of either sign read from a
    buffer (the divide instruction). Every quotient and remainder floors."""
    n = 3000
    a = _floats(4096, np.float32)
    d = np.random.default_rng(1).integers(1, 12, n).astype(np.int32) * np.where(np.arange(n) % 2, 1, -1).astype(np.int32)
    body = _elementwise(
        """
        i32 q = i / 7
        i32 r = i % 5
        i32 back = (i - 1000) / 9
        i32 wrap = (i - 1000) % 9
        i32 low = (i - 1000) % 8
        i32 half = (i - 1000) / 2
        i32 by = load d[i]
        i32 any = (i - 1000) / by
        i32 rest = (i - 1000) % by
        f32 x = load a[q * 3 + r + i / 4]
        i32 code = back * 16 + wrap + low + half
        i32 other = any * 16 + rest
        f32 o[i] = x
        i32 codes[i] = code
        i32 others[i] = other
        """,
        n,
    )
    out = _run(body, {"a": a, "d": d}, {"o": np.float32, "codes": np.int32, "others": np.int32}, n, level=level)
    i = np.arange(n, dtype=np.int64)
    _same_bits(out["o"][:n], a[(i // 7) * 3 + i % 5 + i // 4])
    shifted = i - 1000
    np.testing.assert_array_equal(out["codes"][:n], (shifted // 9) * 16 + shifted % 9 + shifted % 8 + shifted // 2)
    np.testing.assert_array_equal(out["others"][:n], (shifted // d) * 16 + shifted % d)


def test_a_grid_of_two_axes_guards_its_tail(level):
    """Rows of eight columns decoded from the flat thread index, 125 rows over four blocks: the last 24 threads are
    past the cells and store nothing."""
    rows, columns = 125, 8
    a = _floats(rows * columns, np.float32)
    body = f"""
Grid(axes=(r in 0..{rows}, c in 0..{columns}), decode=((block_idx.x * 256 + thread_idx.x) / 8, thread_idx.x % 8), block_threads=256):
    if block_idx.x * 256 + thread_idx.x < {rows * columns}:
        f32 x = load a[r * {columns} + c]
        f32 scale = (f32)(r + 1)
        f32 y = x * scale
        f32 v = y + (f32)c
        f32 o[r * {columns} + c] = v
"""
    out = _run(body, {"a": a}, {"o": np.float32}, rows * columns, level=level)["o"]
    r, c = np.divmod(np.arange(rows * columns), columns)
    _same_bits(out[: rows * columns], a * (r + 1).astype(np.float32) + c.astype(np.float32))
    assert not out[rows * columns :].any()


def test_half_values_convert_through_f32_and_back(level):
    """A half loaded, widened to f32, computed on and narrowed back rounds as NumPy's conversions do, and its
    truncation to an integer is C's."""
    n = 1000
    a = (_floats(n, np.float32) * 100).astype(np.float16)
    body = _elementwise(
        """
        f16 h = load a[i]
        f32 x = (f32)h
        f32 y = x * 1.5
        f32 z = sqrt(abs(y))
        f16 narrow = (f16)z
        i32 cut = (i32)y
        f16 o[i] = narrow
        i32 whole[i] = cut
        """,
        n,
    )
    out = _run(body, {"a": a}, {"o": np.float16, "whole": np.int32}, n, level=level)
    y = a.astype(np.float32) * np.float32(1.5)
    _same_bits(out["o"][:n], np.sqrt(np.abs(y)).astype(np.float16))
    np.testing.assert_array_equal(out["whole"][:n], np.trunc(y).astype(np.int32))
