"""``spelled_arm`` — the one reading of a measured row at a kernel-set fork, shared by the deploy's
evidence pick and the golden replay."""

from __future__ import annotations

from emmy.compiler.pipeline.fork import DeferredFork
from emmy.compiler.pipeline.search.pins import spelled_arm, unreproducible_pin_flag


def test_recorded_precision_pins_preserve_the_default_and_overrides(monkeypatch):
    from types import SimpleNamespace

    from emmy.compiler.pipeline.search.pins import measured_precision_pins, regime_live

    for name in ("FAST_MATH", "FAST_EXP", "F16_MMA_F32_ACC", "FP8_MMA"):
        monkeypatch.delenv(f"EMMY_{name}", raising=False)
    assert measured_precision_pins() == {"FAST_MATH": True}
    assert regime_live(SimpleNamespace(pin_map={"FAST_MATH": True}))
    assert not regime_live(SimpleNamespace(pin_map={"FAST_MATH": False}))
    monkeypatch.setenv("EMMY_FAST_MATH", "0")
    monkeypatch.setenv("EMMY_F16_MMA_F32_ACC", "1")
    recorded = measured_precision_pins()
    assert recorded == {"FAST_MATH": False, "F16_MMA_F32_ACC": True}
    assert regime_live(SimpleNamespace(pin_map=recorded))
    assert not regime_live(SimpleNamespace(pin_map={"FAST_MATH": False}))


def _arm(knobs: dict, *, structural: bool = False) -> DeferredFork:
    return DeferredFork(materialize=lambda: None, knobs=knobs, structural=structural)


def test_a_placement_fork_reads_cut_fuse_and_stale_rows() -> None:
    fuse, cut_a, cut_b = (
        _arm({"PLACE": "fuse"}),
        _arm({"PLACE@map.1/map": "cut"}, structural=True),
        _arm({"PLACE@map.2/map": "cut"}, structural=True),
    )
    options = [fuse, cut_a, cut_b]

    assert spelled_arm(options, {"PLACE@map.2/map": "cut", "WORK": "t8"}) == (cut_b, {"PLACE@map.2/map": "cut"})
    assert spelled_arm(options, {"PLACE": "cut"}) == (cut_a, {"PLACE@map.1/map": "cut"}), "a bare cut is the root-most offered seam"
    assert spelled_arm(options, {"WORK": "t8", "TILE": "f2"}) == (fuse, {"PLACE": "fuse"}), "a schedule row says the kernel ran fused"
    assert spelled_arm(options, {"PLACE@map.1/map": "fuse"}) == (fuse, {"PLACE": "fuse"})
    assert spelled_arm(options, {}) == (fuse, {"PLACE": "fuse"}), "an empty receipt row says the same"
    assert spelled_arm(options, {"PLACE@map.9/twist": "cut"}) is None, "a cut this kernel does not offer decides nothing"


def test_a_split_fork_reads_the_cross_cta_half_alone() -> None:
    unsplit, g2k, g2a = (
        _arm({"REDUCE@inner": ""}),
        _arm({"REDUCE@inner": "g2k"}, structural=True),
        _arm({"REDUCE@inner": "g2a"}, structural=True),
    )
    options = [unsplit, g2k, g2a]

    assert spelled_arm(options, {"REDUCE": "g2a/coop", "WORK": "t32"}) == (g2a, {"REDUCE@inner": "g2a"})
    assert spelled_arm(options, {"REDUCE@inner": "g2k"}) == (g2k, {"REDUCE@inner": "g2k"})
    assert spelled_arm(options, {"REDUCE": "coop"}) == (unsplit, {"REDUCE@inner": ""}), "no cross-CTA half: the kernel ran whole"
    assert spelled_arm(options, {"WORK": "t32", "TILE": "f2"}) == (unsplit, {"REDUCE@inner": ""}), (
        "a schedule row measured the kernel whole"
    )
    assert spelled_arm(options, {"REDUCE": "g8k"}) is None, "a split this kernel does not offer decides nothing"


def test_a_row_marking_several_offered_seams_spells_the_composed_arm() -> None:
    """A pinned compile consumes every scoped PLACE pin that resolves on a kernel as ONE composed
    decision, and ``run --record-greedy`` writes it as one row naming every seam: where the fork
    offers that composed arm beside its single seams, the row spells the composed arm — never the
    first single seam it happens to mark. A single-seam row still spells its single arm, a seam
    another kernel offers is free, and a fork offering no composed arm reads as before."""
    fuse, cut_a, cut_b, cut_ab = (
        _arm({"PLACE": "fuse"}),
        _arm({"PLACE@map.1/map": "cut"}, structural=True),
        _arm({"PLACE@map.2/map": "cut"}, structural=True),
        _arm({"PLACE@map.1/map": "cut", "PLACE@map.2/map": "cut"}, structural=True),
    )
    both = {"PLACE@map.1/map": "cut", "PLACE@map.2/map": "cut"}
    options = [fuse, cut_a, cut_b, cut_ab]

    assert spelled_arm(options, both) == (cut_ab, both)
    assert spelled_arm(options, {**both, "WORK": "t8"}) == (cut_ab, both)
    assert spelled_arm(options, {**both, "PLACE@map.9/twist": "cut"}) == (cut_ab, both), "a seam this kernel does not offer is free"
    assert spelled_arm(options, {"PLACE@map.1/map": "cut"}) == (cut_a, {"PLACE@map.1/map": "cut"}), "one seam: its single arm"
    assert spelled_arm([fuse, cut_a, cut_b], both) == (cut_a, {"PLACE@map.1/map": "cut"}), "no composed arm offered: as before"


def test_unpinned_decisions_withdraws_live_decision_pins_and_restores_them(monkeypatch) -> None:
    """An evidence replay runs with every kernel-decision pin withdrawn — the family vars, their
    scoped forms and their ``EMMY_KNOBS`` entries — while precision and emission pins stay, and the
    environment comes back exactly as it was."""
    import os

    from emmy import config
    from emmy.compiler.pipeline.knob import family_pins
    from emmy.compiler.pipeline.search.pins import unpinned_decisions

    scoped = config.knob_var("TILE@map.1/twist")
    monkeypatch.setenv(scoped, "mma_m16n8k16_f16_f32/f1x8/k4")
    monkeypatch.setenv(config.knob_var("WORK"), "w4x1")
    monkeypatch.setenv(config.knob_var("PLACE"), "cut")
    monkeypatch.setenv(config.knob_var("FAST_MATH"), "1")
    monkeypatch.setenv(config.KNOBS, "STAGE@map.1/twist=d2/smem-tma,FAST_MATH=1,LOOPIFY=1")
    assert family_pins("TILE") and family_pins("WORK")
    with unpinned_decisions():
        assert scoped not in os.environ and config.knob_var("WORK") not in os.environ and config.knob_var("PLACE") not in os.environ
        assert not family_pins("TILE") and not family_pins("WORK") and not family_pins("STAGE")
        assert os.environ[config.knob_var("FAST_MATH")] == "1"
        assert os.environ[config.KNOBS] == "FAST_MATH=1,LOOPIFY=1"
    assert os.environ[scoped] == "mma_m16n8k16_f16_f32/f1x8/k4"
    assert os.environ[config.knob_var("WORK")] == "w4x1"
    assert os.environ[config.knob_var("PLACE")] == "cut"
    assert os.environ[config.KNOBS] == "STAGE@map.1/twist=d2/smem-tma,FAST_MATH=1,LOOPIFY=1"


def test_a_family_pinned_off_is_realized_by_a_kernel_that_never_stamps_it() -> None:
    """A recorded row spells a family it declined as ``''``, and a per-cell kernel stamps no ``TILE`` or ``STAGE``
    at all. That is the same schedule, not a miss: replaying such a row of the Gemma 4 inventory (an RMS norm, a
    QK norm) reported ``TILE= realized (unset)`` and left the row unbenched. A kernel that DECIDED the family
    still contradicts the OFF pin."""
    per_cell = [{"WORK": "", "REDUCE": "coop", "LOOPIFY": "0"}]
    assert unreproducible_pin_flag({"TILE": "", "STAGE": "", "REDUCE": "coop"}, per_cell) is None

    tiled = [{"WORK": "w2x2", "TILE": "mma_m16n8k16_f16_f32/f2x2/k2", "STAGE": "d2/smem-tma"}]
    flag = unreproducible_pin_flag({"TILE": ""}, tiled)
    assert flag is not None and "f2x2" in flag

    # A non-OFF pin the kernel never stamps is still a miss.
    assert unreproducible_pin_flag({"TILE": "f4"}, per_cell) is not None
