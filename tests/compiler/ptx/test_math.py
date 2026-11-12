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


def _apply(function: str, x: np.ndarray, *, level: str | None = None, ftz: bool = False) -> np.ndarray:
    body = KERNEL.format(n=x.size, function=function)
    return launch.run(body, {"a": x}, {"o": (np.float32, x.size)}, blocks=x.size // BLOCK, level=level, ftz=ftz)["o"]


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


@pytest.mark.sweep
@pytest.mark.parametrize("function", FUNCTIONS, ids=lambda function: function.name)
def test_a_function_is_within_its_measured_maximum_over_every_f32(function):
    """Every f32 bit pattern through the kernel, 2^24 a launch, each launch compared on host threads. The maximum
    is within the documented bound and no larger than the one measured when the sequence was written."""
    x0 = np.zeros(CHUNK, np.float32)
    body = KERNEL.format(n=CHUNK, function=function.name)

    def compare(base: int, got: np.ndarray) -> tuple[float, float, int]:
        worst = (0.0, 0.0, 0)
        for start in range(0, CHUNK, HOST_BLOCK):
            x = np.arange(base + start, base + start + HOST_BLOCK, dtype=np.uint32).view(np.float32)
            error, at, over = _worst(function, x, got[start : start + HOST_BLOCK])
            worst = (max(worst[0], error), at if error > worst[0] else worst[1], worst[2] + over)
        return worst

    with launch.built(body, {"a": x0}, {"o": (np.float32, CHUNK)}, blocks=CHUNK // BLOCK) as program, ThreadPoolExecutor(8) as pool:
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
    over = sum(result[2] for result in results)
    print(f"\n{function.name}: max {worst:.4f} ulp at {at!r}; {over} of 2^32 inputs off by more than 1 ulp")
    assert worst <= function.bound, f"{function.name}: {worst:.3f} ulp at {at!r}"
    assert worst <= function.measured, f"{function.name}: {worst:.4f} ulp at {at!r}, measured {function.measured}"
