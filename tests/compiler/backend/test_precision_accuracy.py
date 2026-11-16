"""Each instruction a precision knob picks, run on the device against the host reference.

One knob per instruction class: a class takes its approximate instruction exactly when its own knob is on (by its pin,
or by the ``FAST_MATH`` umbrella it follows), and that instruction stays inside the bound the CUDA Programming Guide
documents for it. Every bound is in units in the last place (ulp) of the f32 result, against a float64 host value.
Each approximate instruction is an operation the kernel body names (``lowering/kernel/130_approximate_math``), so
these tests are what pins that a knob picks the instruction it names. The approximate exponential is ``__expf``, which
keeps subnormal results unless flush to zero is on. The functions no knob names (logarithm, sine, cosine, hyperbolic
tangent, power) take their approximate instruction under the umbrella with the exponential.
"""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler import dtype as dt
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import requires_cuda

pytestmark = requires_cuda

_N = 1 << 16
#: The inputs the class table draws at import; each test of its own draws from its own generator, in any order.
_RNG = np.random.default_rng(0)


def _run(steps: list[tuple[str, list[str]]], feed: dict[str, np.ndarray], pins: dict) -> np.ndarray:
    """``steps`` (one elementwise op each, over the inputs and earlier steps) compiled under ``pins`` and run on
    ``feed``: the last step's f32 output."""
    from emmy.compiler.backend.cuda.backend import CudaBackend

    graph = Graph()
    for name in feed:
        graph.add_node(InputOp(), [], Tensor(name, (_N,), dt.F32), node_id=name)
    for index, (op, operands) in enumerate(steps):
        graph.add_node(ElementwiseOp(op), operands, Tensor(f"s{index}", (_N,), dt.F32), node_id=f"s{index}")
    graph.inputs, graph.outputs = list(feed), [f"s{len(steps) - 1}"]
    backend = CudaBackend()
    with pinned_knobs(pins):
        result, _ = backend.run(backend.compile(graph), input_data=dict(feed))
    return result.outputs[graph.outputs[0]].astype(np.float64)


def _ulp(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    """The error of ``got`` in ulp of the f32 rounding of ``want``; a subnormal's ulp is the smallest subnormal."""
    return np.abs(got - want) / np.spacing(want.astype(np.float32)).astype(np.float64)


def _magnitudes(low: int, high: int) -> np.ndarray:
    return (_RNG.uniform(1, 2, _N) * 2.0 ** _RNG.integers(low, high, _N)).astype(np.float32)


# --------------------------------------------------------------------------- #
# The four classes with an approximate instruction, each against its own knob
# --------------------------------------------------------------------------- #

# Each class: its knob, the op, its inputs, the host reference, the bound of its approximate instruction and the bound
# of the instruction its knob off picks (0.5 ulp is a correctly rounded result).
_EXP_X = _RNG.uniform(-87, 88, _N).astype(np.float32)
_CLASSES = {
    # __expf: 2 + floor(|1.173 x|) ulp; the accurate expf: 2 ulp. The range keeps every result normal.
    "exp": ("FAST_EXP", [("exp", ["x"])], {"x": _EXP_X}, lambda x: np.exp(x), 2 + np.floor(np.abs(1.173 * _EXP_X)), 2.0),
    # Approximate division: 2 ulp for a divisor in [2^-126, 2^126]; IEEE division otherwise.
    "divide": (
        "FAST_DIV",
        [("divide", ["x", "y"])],
        {"x": _magnitudes(-60, 60) * _RNG.choice(np.float32([-1, 1]), _N), "y": _magnitudes(-60, 60)},
        lambda x, y: x / y,
        2.0,
        0.5,
    ),
    # Approximate reciprocal: 1 ulp.
    "reciprocal": ("FAST_DIV", [("reciprocal", ["x"])], {"x": _magnitudes(-120, 120)}, lambda x: 1 / x, 1.0, 0.5),
    # Approximate square root: 1 ulp.
    "sqrt": ("FAST_SQRT", [("sqrt", ["x"])], {"x": _magnitudes(-120, 120)}, lambda x: np.sqrt(x), 1.0, 0.5),
}


def _regimes(knob: str) -> list:
    other = "FAST_SQRT" if knob != "FAST_SQRT" else "FAST_EXP"
    return [
        pytest.param({"FAST_MATH": False}, False, id="standard"),
        pytest.param({"FAST_MATH": True}, True, id="umbrella"),
        pytest.param({"FAST_MATH": False, knob: True}, True, id="own-knob"),
        pytest.param({"FAST_MATH": True, knob: False}, False, id="umbrella-less-own-knob"),
        pytest.param({"FAST_MATH": False, other: True}, False, id="another-knob"),
    ]


@pytest.mark.parametrize(
    ("name", "pins", "approximate"),
    [pytest.param(name, *regime.values, id=f"{name}-{regime.id}") for name, spec in _CLASSES.items() for regime in _regimes(spec[0])],
)
def test_a_class_takes_its_approximate_instruction_under_its_own_knob(name, pins, approximate) -> None:
    _knob, steps, feed, reference, approximate_bound, exact_bound = _CLASSES[name]
    ulp = _ulp(_run(steps, feed, pins), reference(*(value.astype(np.float64) for value in feed.values())))
    if approximate:
        assert np.all(ulp <= approximate_bound), f"{name}: {ulp.max():.2f} ulp past the approximate instruction's bound"
        assert ulp.max() > exact_bound, f"{name}: the approximate instruction was not taken"
    else:
        assert ulp.max() <= exact_bound, f"{name}: {ulp.max():.2f} ulp past the accurate instruction's bound"


# --------------------------------------------------------------------------- #
# The functions no knob names: approximate under the umbrella with the exponential
# --------------------------------------------------------------------------- #

_LOG_X = _magnitudes(-60, 60)
_ANGLE = _RNG.uniform(-np.pi, np.pi, _N).astype(np.float32)
_TANH_X = _RNG.uniform(-12, 12, _N).astype(np.float32)
_POW_X, _POW_Y = _RNG.uniform(0.5, 2, _N).astype(np.float32), _RNG.uniform(-8, 8, _N).astype(np.float32)


def _pow_bound(want: np.ndarray) -> np.ndarray:
    """``__powf`` is ``exp2f(y * __log2f(x))``: the logarithm is within 2^-22 for a base in [0.5, 2], the product
    rounds once and the exponential is within 2 ulp. As a relative error of the result."""
    exponent = (_POW_Y.astype(np.float64) * np.log2(_POW_X.astype(np.float64))).astype(np.float32)
    return np.log(2) * (np.abs(_POW_Y) * 2.0**-22 + np.spacing(np.abs(exponent)) / 2) + 2.0**-22 + 0 * want


# Each function: the op, its inputs, the host reference, the absolute error its approximate instruction stays inside
# for these inputs (from the CUDA Programming Guide's table of intrinsics and the PTX reference) and the ulp bound of
# the accurate function.
_LIBRARY = {
    # __logf: 2^-21.41 for an input in [0.5, 2], 3 ulp otherwise.
    "log": (
        [("log", ["x"])],
        {"x": _LOG_X},
        np.log,
        lambda want: np.where((_LOG_X >= 0.5) & (_LOG_X <= 2), 2.0**-21.41, 3 * np.spacing(np.abs(want).astype(np.float32))),
        1.0,
    ),
    # __sinf: 2^-21.41 for an input in [-pi, pi]; __cosf: 2^-21.19.
    "sin": ([("sin", ["x"])], {"x": _ANGLE}, np.sin, lambda want: 2.0**-21.41 + 0 * want, 2.0),
    "cos": ([("cos", ["x"])], {"x": _ANGLE}, np.cos, lambda want: 2.0**-21.19 + 0 * want, 2.0),
    # tanh.approx.f32: a relative error of 2^-11 over the whole range.
    "tanh": ([("tanh", ["x"])], {"x": _TANH_X}, np.tanh, lambda want: 2.0**-11 * np.abs(want), 2.0),
    "pow": ([("pow", ["x", "y"])], {"x": _POW_X, "y": _POW_Y}, np.power, lambda want: _pow_bound(want) * np.abs(want), 4.0),
}


@pytest.mark.parametrize("name", list(_LIBRARY))
@pytest.mark.parametrize(
    ("pins", "approximate"),
    [
        ({"FAST_MATH": False}, False),
        ({"FAST_MATH": True}, True),
        ({"FAST_MATH": True, "FAST_EXP": False}, False),
        ({"FAST_MATH": False, "FAST_EXP": True}, False),
    ],
    ids=["standard", "umbrella", "umbrella-less-exp", "exp-alone"],
)
def test_a_function_no_knob_names_follows_the_umbrella_with_the_exponential(name, pins, approximate) -> None:
    steps, feed, reference, approximate_bound, exact_bound = _LIBRARY[name]
    want = reference(*(value.astype(np.float64) for value in feed.values()))
    got = _run(steps, feed, pins)
    ulp = _ulp(got, want)
    if approximate:
        assert np.all(np.abs(got - want) <= approximate_bound(want)), f"{name}: past the approximate instruction's bound"
        assert ulp.max() > exact_bound, f"{name}: the approximate instruction was not taken"
    else:
        assert ulp.max() <= exact_bound, f"{name}: {ulp.max():.2f} ulp past the accurate function's bound"


# --------------------------------------------------------------------------- #
# Flush to zero
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("pins", "flush"),
    [
        ({"FAST_MATH": False}, False),
        ({"FAST_MATH": True}, True),
        ({"FAST_MATH": False, "FLUSH_TO_ZERO": True}, True),
        ({"FAST_MATH": True, "FLUSH_TO_ZERO": False}, False),
    ],
    ids=["standard", "umbrella", "own-knob", "umbrella-less-own-knob"],
)
def test_flush_to_zero_flushes_exactly_the_subnormal_results(pins, flush) -> None:
    """A product is exact on the device but where it is subnormal: flushed to zero under ``FLUSH_TO_ZERO``, kept
    exact without it."""
    rng = np.random.default_rng(1)
    x = (rng.uniform(1, 2, _N) * 2.0 ** rng.integers(-130, -100, _N)).astype(np.float32)
    scale = np.full(_N, 2.0**-20, np.float32)
    got = _run([("multiply", ["x", "scale"])], {"x": x, "scale": scale}, pins)
    want = x.astype(np.float64) * scale
    subnormal = want < 2.0**-126
    assert subnormal.any() and (~subnormal).any()
    np.testing.assert_array_equal(got[~subnormal], want[~subnormal])
    np.testing.assert_array_equal(got[subnormal], 0.0 if flush else want[subnormal].astype(np.float32))


# --------------------------------------------------------------------------- #
# FMA contraction
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("pins", "fused"),
    [({"FAST_MATH": False}, True), ({"FAST_MATH": True}, True), ({"FAST_MATH": True, "CONTRACT_FMA": False}, False)],
    ids=["standard", "umbrella", "own-pin-off"],
)
def test_contraction_rounds_a_multiply_add_once(pins, fused) -> None:
    """``(1 + 2^-12)^2 - (1 + 2^-11)`` is ``2^-24`` rounded once, as a fused multiply-add computes it, and ``0`` when
    the product rounds on its own first. Contraction is on in both regimes; only its own pin turns it off."""
    a = np.full(_N, 1 + 2.0**-12, np.float32)
    c = np.full(_N, -(1 + 2.0**-11), np.float32)
    got = _run([("multiply", ["a", "b"]), ("add", ["s0", "c"])], {"a": a, "b": a, "c": c}, pins)
    np.testing.assert_array_equal(got, 2.0**-24 if fused else 0.0)


# --------------------------------------------------------------------------- #
# The fast exponential on subnormal results
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("pins", "flush"),
    [({"FAST_MATH": True}, True), ({"FAST_MATH": True, "FLUSH_TO_ZERO": False}, False), ({"FAST_MATH": False, "FAST_EXP": True}, False)],
    ids=["umbrella", "umbrella-less-flush", "own-knob"],
)
def test_the_fast_exponential_keeps_subnormal_results_unless_they_flush(pins, flush) -> None:
    """``__expf`` without flush to zero keeps a subnormal result: ptxas lowers its ``ex2.approx.f32`` to one hardware
    exponential on a halved argument and a squaring, so the arm of the exponential knob that keeps subnormals is
    ``FAST_EXP`` with ``FLUSH_TO_ZERO`` off. It stays inside the bound of ``__expf``, a subnormal's ulp being the
    smallest subnormal; with flush to zero every subnormal result is zero."""
    x = np.random.default_rng(2).uniform(-103, -80, _N).astype(np.float32)
    want = np.exp(x.astype(np.float64))
    got = _run([("exp", ["x"])], {"x": x}, pins)
    subnormal = want < 2.0**-126
    assert subnormal.any() and (~subnormal).any()
    ulp = _ulp(got, want)
    within = ulp <= 2 + np.floor(np.abs(1.173 * x))
    # Under flush to zero, a result within the instruction's error of the smallest normal may flush or not.
    clear = np.abs(want / 2.0**-126 - 1) > 1e-4 if flush else np.ones(_N, bool)
    normal = ~subnormal & clear
    assert within[normal].all() and ulp[normal].max() > 2, "the approximate exponential was not taken"
    assert (got[subnormal & clear] == 0).all() if flush else within[subnormal].all()
