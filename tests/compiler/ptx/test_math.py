"""The math functions the ``emmy-ptx`` crate writes as sequences, measured on a card against a float64 reference
computed on the host.

Each function has the bound NVIDIA's CUDA math documentation states for it (single-precision functions, maximum ulp
error). The default tests sample: every special value, the neighbourhoods of the points where a sequence changes
form, and two million random bit patterns. The tests marked ``sweep`` evaluate every f32 bit pattern and hold the
measured maximum, the one ``crates/emmy-ptx/ARCHITECTURE.md`` states; run them with ``pytest -m sweep``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from tests.compiler.helpers import requires_cuda
from tests.compiler.ptx import launch

pytestmark = [requires_cuda, pytest.mark.xdist_group("cuda")]

#: Threads per block of every kernel here.
BLOCK = 256

#: One thread per element, the function of each: the grid covers its launch, so no thread needs a guard.
KERNEL = """
Grid(axes=(i in 0..{n}), decode=(block_idx.x * 256 + thread_idx.x), block_threads=256):
    f32 x = load a[i]
    f32 v = {function}(x)
    f32 o[i] = v
"""

#: The same for a function of two operands.
KERNEL_OF_TWO = """
Grid(axes=(i in 0..{n}), decode=(block_idx.x * 256 + thread_idx.x), block_threads=256):
    f32 x = load a[i]
    f32 y = load b[i]
    f32 v = {function}(x, y)
    f32 o[i] = v
"""

#: The elements one launch of a sweep takes, and the elements a host thread compares at a time.
CHUNK = 1 << 24
HOST_BLOCK = 1 << 20

FLT_MAX = float(np.finfo(np.float32).max)
FLT_MIN = float(np.finfo(np.float32).tiny)

#: The values every sample holds, each with both signs: the zeros, the subnormals' ends, the normals' ends, one,
#: the infinities and a NaN.
SPECIAL = [0.0, 1e-45, 1.1754942e-38, FLT_MIN, 1.0, FLT_MAX, math.inf, math.nan]

Reference = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Function:
    """One function: its float64 reference, the bound the CUDA documentation states for it in ulps, the maximum
    measured over every f32 on an RTX 5090 (a ceiling a sweep holds), and the points its sequence changes form
    at, or was measured worst at."""

    name: str
    reference: Reference
    bound: float
    measured: float
    points: tuple[float, ...]


def _erf(x: np.ndarray) -> np.ndarray:
    return torch.erf(torch.from_numpy(x)).numpy()


FUNCTIONS = [
    Function("exp", np.exp, 2, 1.94, (math.log(2), 88.72284, 87.33655, 103.2789, 103.97208, 104.0, 13.899315)),
    Function("log", np.log, 1, 0.94, (2 / 3, 4 / 3, 0.5, 2.0, 2.6688592)),
    Function("tanh", np.tanh, 2, 1.37, (0.5, 9.0109, 1.0126345)),
    Function("erf", _erf, 2, 1.17, (3.92, 4.0, 1.0339911)),
    Function("rsqrt", lambda x: 1 / np.sqrt(x), 2, 1.52, (2.0, 4.0, 1.19399e-38)),
    Function("sin", np.sin, 2, 1.50, (math.pi / 4, math.pi / 2, 105615.0, 1e30, 3.4e38)),
    Function("cos", np.cos, 2, 1.52, (math.pi / 4, math.pi / 2, 105615.0, 1e30, 3.4e38)),
]
BY_NAME = {function.name: function for function in FUNCTIONS}


def ulp_error(got: np.ndarray, exact: np.ndarray) -> np.ndarray:
    """The distance of each f32 of ``got`` from the float64 ``exact``, in steps of the f32 grid at the exact value.
    An infinity counts as the next power of two past the largest float, so a result that overflows a step early
    is a step off. A NaN where the exact value is none, or none where it is one, is infinitely far."""
    limit = 2.0**128
    wanted = np.clip(exact, -limit, limit)
    held = np.clip(got.astype(np.float64), -limit, limit)
    with np.errstate(divide="ignore", invalid="ignore"):
        exponent = np.floor(np.log2(np.clip(np.abs(wanted), FLT_MIN, FLT_MAX)))
        error = np.abs(held - wanted) / np.exp2(exponent - 23)
    error[np.isnan(exact) & np.isnan(got)] = 0.0
    error[np.isnan(exact) != np.isnan(got)] = np.inf
    return error


def steps_from_rounded(got: np.ndarray, exact: np.ndarray) -> np.ndarray:
    """How many f32 values ``got`` is from the float64 ``exact`` rounded to f32, which is how the CUDA documentation
    counts an error in ulps. An infinity is the value after the largest float."""

    def ordinal(values: np.ndarray) -> np.ndarray:
        bits = values.view(np.uint32).astype(np.int64)
        return np.where(bits >> 31, -(bits & 0x7FFFFFFF), bits)

    with np.errstate(over="ignore"):
        return np.abs(ordinal(got) - ordinal(exact.astype(np.float32)))


def _sample(points: tuple[float, ...], seed: int = 0) -> np.ndarray:
    """The special values, 4,096 neighbours of every point on each side of zero, and two million random bit
    patterns, padded with ones to whole blocks."""
    around = np.arange(-2048, 2048, dtype=np.int64)
    centres = np.array([*SPECIAL, *points], np.float32).view(np.uint32).astype(np.int64)
    near = (centres[:, None] + around[None, :]).ravel()
    near = np.concatenate([near, near ^ (1 << 31)]) % (1 << 32)
    random = np.random.default_rng(seed).integers(0, 1 << 32, 1 << 21, dtype=np.uint32)
    bits = np.concatenate([near.astype(np.uint32), random])
    bits = np.concatenate([bits, np.full(-bits.size % BLOCK, 0x3F800000, np.uint32)])
    return bits.view(np.float32)


def _worst(function: Function, x: np.ndarray, got: np.ndarray) -> tuple[float, float, int]:
    """The largest error of ``got`` over ``x``, the input it is at, and how many inputs are off by over one ulp."""
    with np.errstate(all="ignore"):
        error = ulp_error(got, function.reference(x.astype(np.float64)))
    at = int(np.argmax(error))
    return float(error[at]), float(x[at]), int(np.count_nonzero(error > 1.0))


def _apply(function: str, *operands: np.ndarray, level: str | None = None, ftz: bool = False) -> np.ndarray:
    """``function`` of each element of its operands, whole blocks of them."""
    n = operands[0].size
    body = (KERNEL, KERNEL_OF_TWO)[len(operands) - 1].format(n=n, function=function)
    inputs = dict(zip("ab", operands, strict=False))
    return launch.run(body, inputs, {"o": (np.float32, n)}, blocks=n // BLOCK, level=level, ftz=ftz)["o"]


def _every_f32(function: str, reference: Reference, second: float | None = None) -> tuple[float, float, int]:
    """Every f32 bit pattern through ``function``, 2^24 a launch, each launch compared on host threads with
    ``reference`` of the float64 operands. ``second`` is a second operand, the same for every element. Returns the
    largest error, the operand it is at, and how many operands are off by over one ulp."""
    x0 = np.zeros(CHUNK, np.float32)
    inputs = {"a": x0} if second is None else {"a": x0, "b": np.full(CHUNK, second, np.float32)}
    body = (KERNEL if second is None else KERNEL_OF_TWO).format(n=CHUNK, function=function)

    def compare(base: int, got: np.ndarray) -> tuple[float, float, int]:
        worst = (0.0, 0.0, 0)
        for start in range(0, CHUNK, HOST_BLOCK):
            x = np.arange(base + start, base + start + HOST_BLOCK, dtype=np.uint32).view(np.float32)
            with np.errstate(all="ignore"):
                error = ulp_error(got[start : start + HOST_BLOCK], reference(x.astype(np.float64)))
            at = int(np.argmax(error))
            if error[at] > worst[0]:
                worst = (float(error[at]), float(x[at]), worst[2])
            worst = (*worst[:2], worst[2] + int(np.count_nonzero(error > 1.0)))
        return worst

    with launch.built(body, inputs, {"o": (np.float32, CHUNK)}, blocks=CHUNK // BLOCK) as program, ThreadPoolExecutor(8) as pool:
        pending = []
        for base in range(0, 1 << 32, CHUNK):
            program.rebind({"a": np.arange(base, base + CHUNK, dtype=np.uint32).view(np.float32)})
            program.run_once()
            pending.append(pool.submit(compare, base, program.outputs()["o"]))
            # At most sixteen results wait for a host thread, so they never pile up in memory.
            if len(pending) >= 16:
                pending[-16].result()
        results = [future.result() for future in pending]
    worst, at, _ = max(results)
    return worst, at, sum(result[2] for result in results)


@pytest.mark.parametrize("level", launch.LEVELS, ids=["default", "O0"])
@pytest.mark.parametrize("function", FUNCTIONS, ids=lambda function: function.name)
def test_a_function_is_within_its_documented_bound_on_a_sample(function, level):
    """Over the sample no result is further from the exact value than the documented bound, at ptxas's default
    level and at ``-O0``."""
    x = _sample(function.points)
    worst, at, _ = _worst(function, x, _apply(function.name, x, level=level))
    assert worst <= function.bound, f"{function.name}: {worst:.3f} ulp at {at!r}"


#: What each function gives at the values that have no rounding to them, as (operand, result) pairs.
INF, NAN = math.inf, math.nan
SPECIAL_VALUES = {
    "exp": [(0.0, 1.0), (-0.0, 1.0), (INF, INF), (-INF, 0.0), (1e-45, 1.0), (-1e-45, 1.0), (89.0, INF), (-104.0, 0.0), (-1e30, 0.0)],
    "log": [(0.0, -INF), (-0.0, -INF), (1.0, 0.0), (INF, INF), (-1.0, NAN), (-INF, NAN), (-1e-45, NAN)],
    "tanh": [(0.0, 0.0), (-0.0, -0.0), (INF, 1.0), (-INF, -1.0), (20.0, 1.0), (-20.0, -1.0), (1e-45, 1e-45), (-1e-45, -1e-45)],
    "erf": [(0.0, 0.0), (-0.0, -0.0), (INF, 1.0), (-INF, -1.0), (5.0, 1.0), (-5.0, -1.0)],
    "rsqrt": [(0.0, INF), (-0.0, -INF), (INF, 0.0), (1.0, 1.0), (4.0, 0.5), (-1.0, NAN), (-INF, NAN)],
    "sin": [(0.0, 0.0), (-0.0, -0.0), (INF, NAN), (-INF, NAN), (1e-45, 1e-45), (-1e-45, -1e-45)],
    "cos": [(0.0, 1.0), (-0.0, 1.0), (INF, NAN), (-INF, NAN), (1e-45, 1.0)],
}


@pytest.mark.parametrize("function", SPECIAL_VALUES)
def test_a_function_gives_the_special_values_its_cuda_function_gives(function):
    """Zeros with their signs, infinities and the operands past a function's range give what the CUDA function
    gives today, bit for bit; a NaN operand gives a NaN."""
    operands, expected = zip(*SPECIAL_VALUES[function], (NAN, NAN), strict=True)
    x = np.array(operands, np.float32)
    padded = np.concatenate([x, np.ones(-x.size % BLOCK, np.float32)])
    got = _apply(function, padded)[: x.size]
    want = np.array(expected, np.float32)
    nan = np.isnan(want)
    assert np.isnan(got[nan]).all(), got
    launch.same_bits(got[~nan], want[~nan])


def test_exp_keeps_subnormal_results_unless_the_kernel_flushes():
    """Below the smallest normal float ``exp`` goes through the subnormals to zero. With the flush-to-zero decision
    on, a subnormal result is zero."""
    x = np.linspace(-104.0, -87.0, 4096, dtype=np.float32)
    exact = np.exp(x.astype(np.float64))
    kept = _apply("exp", x)
    assert ulp_error(kept, exact).max() <= 2
    assert np.count_nonzero((kept > 0) & (kept < FLT_MIN)) > 3000
    flushed = _apply("exp", x, ftz=True)
    assert not ((flushed > 0) & (flushed < FLT_MIN)).any()
    normal = exact >= FLT_MIN * (1 + 1e-6)
    assert ulp_error(flushed[normal], exact[normal]).max() <= 2


@pytest.mark.parametrize("ftz", [False, True], ids=["subnormals", "flushing"])
def test_the_fast_exponential_is_within_the_bound_of_the_cuda_intrinsic(ftz):
    """``exp_fast`` is the hardware's power of two of ``x * log2(e)``, so its error grows with the operand: the
    CUDA documentation states ``2 + floor(|1.173 * x|)`` ulp for ``__expf``, counted from the correctly rounded
    result. With the flush-to-zero decision off it keeps a subnormal result; with it on, a result below the
    normals is zero."""
    x = _sample(BY_NAME["exp"].points)
    x = x[~np.isnan(x)]
    x = x[: x.size - x.size % BLOCK]
    got = _apply("exp_fast", x, ftz=ftz)
    with np.errstate(all="ignore"):
        exact = np.exp(x.astype(np.float64))
    steps = steps_from_rounded(got, exact)
    bound = 2 + np.floor(np.abs(1.173 * x.astype(np.float64)))
    if ftz:
        assert not ((got > 0) & (got < FLT_MIN)).any()
        normal = exact >= FLT_MIN * (1 + 1e-5)
        steps, bound, x = steps[normal], bound[normal], x[normal]
    else:
        assert ((got > 0) & (got < FLT_MIN)).any()
    assert (steps <= bound).all(), float(x[np.argmax(steps - bound)])


@pytest.mark.parametrize("function", ["sigmoid", "sigmoid_fast"])
def test_a_composite_function_computes_its_definition(function):
    """A composite that reaches the compiler as it was is compiled as its definition in primitive operations.
    ``sigmoid`` is ``1 / (1 + exp(-x))`` over the accurate exponential and a rounded division: the exponential's 2
    ulp and two roundings. ``sigmoid_fast`` is the hardware's reciprocal of one plus the hardware's power of two,
    whose error grows with the operand like ``exp_fast``'s."""
    x = np.linspace(-30, 30, 64 * BLOCK, dtype=np.float32)
    got = _apply(function, x)
    exact = 1 / (1 + np.exp(-x.astype(np.float64)))
    bound = 4 if function == "sigmoid" else 5 + np.floor(np.abs(1.173 * x.astype(np.float64)))
    assert (ulp_error(got, exact) <= bound).all()


#: What bounds the power: 4 ulp in the CUDA documentation, and the largest error found on an RTX 5090 over 1.3
#: billion sampled pairs.
POW_BOUND = 4
POW_MEASURED = 2.62


def _pow_reference(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.power(x.astype(np.float64), y.astype(np.float64))


def _pow_sample(seed: int = 0, each: int = 1 << 20) -> tuple[np.ndarray, np.ndarray]:
    """Pairs of the kinds the power's steps are tried by, ``each`` of every kind: a mantissa at the fold of the
    logarithm's range under a product near the ends of the exponent's range, where its error is largest; bases of
    every magnitude under products from one end to the other; random bit patterns; bases next to one under large
    exponents; and signed bases under small integer exponents."""
    rng = np.random.default_rng(seed)
    sign = lambda: np.where(rng.random(each) < 0.5, 1.0, -1.0)  # noqa: E731

    def exponent_for(x: np.ndarray, product: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            return (product / np.log2(x.astype(np.float64))).astype(np.float32)

    fold = np.where(rng.random(each) < 0.5, math.sqrt(2), math.sqrt(0.5)) * (1 + (rng.random(each) - 0.5) * 0.06)
    fold = (fold * np.exp2(rng.integers(-3, 4, each))).astype(np.float32)
    wide = np.exp2((rng.random(each) - 0.5) * 252).astype(np.float32)
    bits = rng.integers(0, 0x7F800000, each, dtype=np.uint32).view(np.float32)
    near = (1 + (rng.random(each) - 0.5) * np.exp2(-rng.integers(1, 23, each).astype(np.float64))).astype(np.float32)
    signed = (rng.standard_normal(each) * np.exp2(rng.integers(-20, 20, each))).astype(np.float32)
    pairs = [
        (fold, exponent_for(fold, (60 + rng.random(each) * 68.5) * sign())),
        (wide, exponent_for(wide, (rng.random(each) - 0.5) * 300)),
        (bits, (rng.standard_normal(each) * np.exp2(rng.integers(-8, 8, each))).astype(np.float32)),
        (near, exponent_for(near, (rng.random(each) - 0.5) * 280)),
        (signed, rng.integers(-40, 41, each).astype(np.float32)),
    ]
    return np.concatenate([x for x, _ in pairs]), np.concatenate([y for _, y in pairs])


@pytest.mark.parametrize("level", launch.LEVELS, ids=["default", "O0"])
def test_pow_is_within_its_documented_bound_on_a_sample(level):
    """Five million pairs, a million of each kind: none is further from the exact power than the documented bound,
    nor than the largest error found when the sequence was written."""
    x, y = _pow_sample()
    error = ulp_error(_apply("pow", x, y, level=level), _pow_reference(x, y))
    at = int(np.argmax(error))
    assert error[at] <= POW_MEASURED <= POW_BOUND, f"pow({x[at]!r}, {y[at]!r}): {error[at]:.3f} ulp"


def test_pow_gives_the_special_values_its_cuda_function_gives():
    """Every pair of a grid of zeros, ones, subnormals, integers of both parities, halves, the ends of the float
    range, infinities and a NaN, with both signs, gives what C's ``pow`` gives, which is what CUDA's ``powf`` gives:
    a NaN, an infinity or a zero where it does, with its sign, and a finite value within the bound of it. So a zero
    exponent and a base of one give one against a NaN, a negative base under a fractional exponent gives a NaN and
    under an odd one a negative power."""
    one_sided = [0.0, 1e-45, 1e-40, FLT_MIN, 1e-20, 0.3, 0.5, 0.99999994, 1.0, 1.0000001, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]
    one_sided += [100.0, 127.0, 128.0, 149.0, 150.0, 1000.0, 8388607.5, 8388608.0, 16777215.0, 16777216.0, 2147483648.0, 1e10, 1e30]
    one_sided += [FLT_MAX, math.inf]
    values = np.array([*one_sided, *(-value for value in one_sided), math.nan], np.float32)
    x, y = (grid.ravel() for grid in np.meshgrid(values, values, indexing="ij"))
    pad = -x.size % BLOCK
    got = _apply("pow", np.concatenate([x, np.ones(pad, np.float32)]), np.concatenate([y, np.ones(pad, np.float32)]))[: x.size]
    exact = _pow_reference(x, y)
    with np.errstate(over="ignore"):
        want = exact.astype(np.float32)

    def kind(values: np.ndarray) -> np.ndarray:
        """A NaN, or else the sign with which of zero, finite and infinite the value is."""
        magnitude = np.where(np.isinf(values), 2, np.where(values == 0, 0, 1))
        return np.where(np.isnan(values), 9, np.where(np.signbit(values), -1 - magnitude, magnitude))

    # A power that rounds to zero or to infinity may stand a step inside the floats, which the error below bounds.
    settled = np.isnan(want) | (np.abs(exact) > FLT_MAX * 1.0001) | (np.abs(exact) < 1e-46) | np.isfinite(want) & (want != 0)
    mismatched = np.nonzero((kind(got) != kind(want)) & settled)[0]
    assert mismatched.size == 0, [(float(x[i]), float(y[i]), float(got[i]), float(want[i])) for i in mismatched[:8]]
    assert ulp_error(got, exact).max() <= POW_BOUND
    assert (got[(y == 0) | (x == 1)] == 1).all()


@pytest.mark.sweep
@pytest.mark.parametrize("function", FUNCTIONS, ids=lambda function: function.name)
def test_a_function_is_within_its_measured_maximum_over_every_f32(function):
    """Every f32 bit pattern through the kernel. The maximum is within the documented bound and no larger than the
    one measured when the sequence was written."""
    worst, at, over = _every_f32(function.name, function.reference)
    print(f"\n{function.name}: max {worst:.4f} ulp at {at!r}; {over} of 2^32 inputs off by more than 1 ulp")
    assert worst <= function.bound, f"{function.name}: {worst:.3f} ulp at {at!r}"
    assert worst <= function.measured, f"{function.name}: {worst:.4f} ulp at {at!r}, measured {function.measured}"


@pytest.mark.sweep
@pytest.mark.parametrize("exponent", [0.5, 2.0, 3.0, -1.0, 1 / 3, 2.5, 100.0, -227.0])
def test_pow_is_within_its_bound_over_every_base(exponent):
    """Every f32 bit pattern as the base of one exponent: the negative bases too, which give a NaN under a
    fractional exponent and a signed power under an integer."""
    held = float(np.float32(exponent))
    # The exponent as an array: under a scalar one NumPy takes a square root or a square, which is no power at a
    # negative infinity or a negative zero.
    worst, at, over = _every_f32("pow", lambda x: np.power(x, np.full_like(x, held)), second=held)
    print(f"\npow(x, {held!r}): max {worst:.4f} ulp at {at!r}; {over} of 2^32 bases off by more than 1 ulp")
    assert worst <= POW_MEASURED <= POW_BOUND, f"pow({at!r}, {held!r}): {worst:.3f} ulp"
