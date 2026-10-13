"""The autotuner's proposals: the prior's best rows first, then rows near what measured fast, never a row twice."""

import math

import pytest

from emmy.compiler.pipeline.search.autotune import START, Autotuner


def _space():
    rows = [
        {"WORK": f"w{w}x{h}", "TILE": f"mma_m16n8k16_f16_f32/f{f}x2/k{k}", "STAGE": s}
        for w in (1, 2, 4)
        for h in (1, 2, 4)
        for f in (1, 2, 4, 8)
        for k in (2, 4, 8)
        for s in ("d1/smem", "d2/smem-tma")
    ]

    # A smooth latency with its optimum at w4x2 / f4 / k4 / d2, far from where the scores point.
    def us(r):
        w, h = (int(x) for x in r["WORK"][1:].split("x"))
        f, k = int(r["TILE"].split("/f")[1].split("x")[0]), int(r["TILE"].split("/k")[1])
        return (
            10
            + (math.log2(w) - 2) ** 2
            + (math.log2(h) - 1) ** 2
            + (math.log2(f) - 2) ** 2
            + (math.log2(k) - 2) ** 2
            + (r["STAGE"] != "d2/smem-tma")
        )

    return rows, us, [float(i) for i in range(len(rows))]


def test_prior_best_rows_come_first():
    rows, _, scores = _space()
    tuner = Autotuner(rows, scores[::-1])
    assert tuner.propose(4) == [len(rows) - 1 - i for i in range(4)]


def test_finds_the_optimum_from_a_bad_start_without_repeats():
    rows, us, scores = _space()
    tuner = Autotuner(rows, scores)
    proposed: list[int] = []
    while len(tuner.us) < 60:
        batch = tuner.propose(8)
        assert batch and not set(batch) & set(proposed)
        proposed += batch
        tuner.observe((i, us(rows[i])) for i in batch)
    assert us(rows[tuner.best()]) == min(us(r) for r in rows)


def test_failed_rows_are_never_the_best():
    rows, _, scores = _space()
    tuner = Autotuner(rows, scores)
    first = tuner.propose(START)
    tuner.observe([(first[0], math.inf), *((i, 20.0 + i) for i in first[1:])])
    assert tuner.best() == first[1]
    assert first[0] not in tuner.propose(8)


@pytest.mark.parametrize("latency", [1.0, math.inf])
def test_a_schedule_with_only_off_values_is_measured_once(latency):
    """The scalar/default row is still a candidate when it has no numeric or categorical knob parts."""
    row = {"WORK": "", "RASTER": ""}
    tuner = Autotuner([row], [0.0])
    assert tuner.propose(12) == [0]
    assert tuner.rows[0] == row
    tuner.observe([(0, latency)])
    assert tuner.propose(12) == []
    assert tuner.best() == (0 if math.isfinite(latency) else None)


def test_featureless_candidates_survive_the_bayesian_phase_without_repeats():
    """Zero-dimensional points have equal distance; observation still distinguishes measured candidates."""
    tuner = Autotuner([{"WORK": ""} for _ in range(START + 2)], list(range(START + 2)))
    assert tuner.propose(START) == list(range(START))
    tuner.observe((index, 1.0 + index) for index in range(START))
    assert tuner.propose(8) == [START, START + 1]
    tuner.observe([(START, math.inf), (START + 1, 0.5)])
    assert tuner.best() == START + 1
    assert tuner.propose(8) == []
