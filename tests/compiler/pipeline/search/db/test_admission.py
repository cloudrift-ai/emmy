"""The admission rule: which measured rows a dataset's pools read (``db/admission.py``). No GPU."""

from __future__ import annotations

import dataclasses

from emmy.compiler.pipeline.search.dataset import REGIME_PINS, regime_of
from emmy.compiler.pipeline.search.db.admission import admission_reason
from tests.compiler.pipeline.search.helpers import F16_MATMUL_STAMPS, SQUARE_512_STAMPS, impossible_staged_row
from tests.compiler.pipeline.search.helpers import perf_row as _row

# A small fp32 matmul's stamps — what ``features.stamps`` computes from such a kernel — plausible at a few hundred µs.
_STAMPS = {
    "S_ext_free_prod": 4096.0,
    "S_ext_free_max": 64.0,
    "S_ext_reduce_max": 64.0,
    "S_ext_n_free_axis": 2.0,
    "S_ext_n_reduce_axis": 1.0,
    "S_loop_depth": 3.0,
    "S_dtype_f32": 3.0,
}

#: Its schedule row.
_ROW = {"TILE": "f2x2", "WORK": "t16x16"}


def test_reason_keeps_a_row_of_either_precision_regime() -> None:
    """A row's precision decisions alone decide its regime, spelled once and not read off this shell."""
    assert admission_reason(_row("k", us=500.0, knobs=_ROW), _STAMPS) is None
    assert admission_reason(_row("k", us=500.0, knobs=_ROW, precision={"FAST_MATH": True}), _STAMPS) is None
    following = (
        "FAST_MATH",
        "FAST_EXP",
        "F16_MMA_F32_ACC",
        "FP8_MMA",
        "TF32_MMA",
        "F32_AS_BF16_MMA",
        "FAST_DIV",
        "FAST_SQRT",
        "FLUSH_TO_ZERO",
    )
    # Contraction is on and the split TF32 atom off in both regimes: the umbrella moves neither.
    own = {"CONTRACT_FMA": True, "TF32_SPLIT_MMA": False, "COLD_CACHE": False}
    assert REGIME_PINS[""] == {**dict.fromkeys(following, False), **own}
    assert REGIME_PINS["FAST_MATH"] == {**dict.fromkeys(following, True), **own}
    assert regime_of(False, {"FAST_MATH": True}) == "FAST_MATH" and regime_of() == ""
    # Contraction is on in both regimes, so only turning it off is an independent override.
    assert regime_of(False, {"FAST_MATH": False, "CONTRACT_FMA": True}) == ""
    assert regime_of(False, {"FAST_MATH": True, "CONTRACT_FMA": False}) == "FAST_MATH CONTRACT_FMA=0"


def test_reason_drops_a_failed_bench() -> None:
    # A fail's median is the watchdog sentinel, not a measurement; the tune DB keeps the failure.
    assert admission_reason(_row("k", us=9.17, knobs=_ROW, status="bench_fail"), _STAMPS) == "bench_fail: not a measurement"


def test_reason_drops_a_card_the_registry_does_not_know() -> None:
    # Its H_* features cannot be derived, so no reader could featurize it.
    row = dataclasses.replace(_row("k", us=500.0, knobs=_ROW), gpu="Mystery GPU")
    assert admission_reason(row, _STAMPS).startswith("unknown card")


def test_reason_drops_a_row_whose_kernel_no_longer_lowers() -> None:
    # A row's shape is computed from its kernel's definition; a definition the compiler stopped taking back has none.
    assert admission_reason(_row("k", us=500.0, knobs=_ROW), None) == "stale kernel: its definition no longer lowers"


def test_reason_drops_implausible_value() -> None:
    # The shared f16 mlp_down extents at 9.17 µs imply ~6500 TFLOP/s at the default hint of its
    # symbolic axis — and an honest 13 TFLOP/s when the row says it was benched at one token.
    assert "implausible value" in admission_reason(_row("k", us=9.17), F16_MATMUL_STAMPS)
    assert admission_reason(_row("k", us=9.17, bindings={"m": 1}), F16_MATMUL_STAMPS) is None


def test_reason_drops_impossible_kernel() -> None:
    # The square.512 residue: over-cap cp.async slab -> legal-looking latency, invalid kernel.
    assert "impossible kernel" in admission_reason(_row("k", us=2.02, knobs=impossible_staged_row()), SQUARE_512_STAMPS)
