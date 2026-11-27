"""The autotuner's proposals: the prior's best rows first, then rows near what measured fast, never a row twice; and
its candidates: every schedule row crossed with the arms of the forks the kernel lowering opens afterwards."""

import math

import numpy as np
import pytest

from emmy.compiler.context import Context
from emmy.compiler.pipeline.search.autotune import START, Autotuner, _encode, schedule_space
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.passes.test_shared_carry import _graph as _carried_state_graph


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


def test_candidates_spell_the_carried_states_storage_on_every_row_that_can_hold_it():
    """A kernel that carries a state has a ``STAGE`` site for it. Every schedule row offers the state in global
    memory; the rows a one-dimensional cooperative thread tile runs offer it in shared memory too, and no other."""
    with pinned_knobs({"FAST_MATH": True}):
        _, rows = schedule_space(_carried_state_graph(), Context.from_target((7, 0)))
    arms: dict[tuple, set] = {}
    for row in rows:
        arms.setdefault(tuple(sorted((k, v) for k, v in row.items() if k != "STAGE")), set()).add(row["STAGE"])
    assert arms and all("WORK" in dict(schedule) for schedule in arms)
    shared = {dict(schedule)["WORK"] for schedule, offered in arms.items() if offered == {"", "d1/smem"}}
    assert shared and all(work.startswith("t") and "x" not in work for work in shared)
    assert all(offered == {""} for schedule, offered in arms.items() if dict(schedule)["WORK"] not in shared)


def test_a_two_valued_knob_keeps_its_arms_apart_and_the_other_arm_is_proposed_next():
    """Global storage is not shared storage to the process: the two arms are distinct points, and once one measures
    the other is a one-part neighbour of the best row, so the next proposal reaches it."""
    rows = [{"WORK": f"t{t}", "STAGE": s} for t in (128, 256, 512) for s in ("", "d1/smem")]
    _, points = _encode(rows)
    assert not np.allclose(points[0], points[1])
    tuner = Autotuner(rows, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    tuner.observe([(0, 100.0), (2, 110.0), (4, 120.0)])
    assert 1 in tuner.propose(1)


def test_a_programs_tune_prices_an_arm_the_compile_did_not_take_from_its_kernels_alone(tmp_path):
    """What a tune of a whole program stands on, with no card: each kernel of the kernel set the compile takes is
    a program of its own, the same kernel with no fork left open; an arm the compile did not take is taken on the
    fork's kernel alone and leaves the kernels the whole program would; and once those kernels hold rows that beat
    the first set's, with the decision stored as a routing row, the next compile takes that arm."""
    from emmy.compiler.pipeline.search.autotune import kernel_set
    from emmy.compiler.pipeline.search.bench_record import kernel_row
    from emmy.compiler.pipeline.search.db import SearchDB
    from emmy.compiler.pipeline.search.inventory import record_measured_routing
    from emmy.compiler.pipeline.search.pins import unpinned_decisions
    from emmy.compiler.wire import kernel_bindings, kernel_identity, kernel_tile
    from tests.compiler.pipeline.search.helpers import CARDS
    from tests.compiler.realization import helpers as corpus

    case = corpus.load_case(corpus.CASES_DIR / "fused/linear-add-place-cut-sm70.json")
    ctx = Context.from_target(case.compute_cap, gpu_name=CARDS[case.compute_cap], precision={})
    db = SearchDB(tmp_path / "tune.db")

    def identities(found) -> list:
        return sorted(kernel_identity(op) for op, _ in found.kernels)

    def measure(found, us: float) -> None:
        for op, _ in found.kernels:
            db.record_kernel(kernel_row(kernel_tile(op)))
            db.record_perf(ctx, kernel_identity(op), bindings=kernel_bindings(kernel_tile(op)), knobs={}, status="ok", us=us)
        record_measured_routing(db, found.decisions, set(identities(found)))

    with pinned_knobs(case.regime), unpinned_decisions():
        first = kernel_set(case.program(), ctx, db)
        for op, alone in first.kernels:
            again = kernel_set(alone.copy(), ctx, db)
            assert identities(again) == [kernel_identity(op)]
            assert all(len(arms) == 1 for _, _, arms, _ in again.forks), "a kernel alone has no fork left to decide"
        _kernel, alone, arms, taken = first.forks[0]
        assert len(arms) > 1 and taken == arms[0], "with nothing measured the compile takes the placement prior's first arm"
        other = kernel_set(alone.copy(), ctx, db, take=arms[1])
        assert other.forks[0][3] == arms[1] and identities(other) != identities(first)

        measure(first, 100.0)
        assert identities(kernel_set(case.program(), ctx, db)) == identities(first)
        measure(other, 1.0)
        assert identities(kernel_set(case.program(), ctx, db)) == identities(other), "the measured arm must be taken"


def test_a_kernel_alone_keeps_the_paging_of_the_buffers_it_reads():
    """A paged buffer takes fewer schedules (no TMA), so a kernel tuned alone must stay paged as its program
    declares, or its rows are schedules the program's compile never offers."""
    from emmy.compiler.pipeline.search.autotune import kernel_set
    from emmy.compiler.pipeline.search.pins import unpinned_decisions
    from tests.compiler.pipeline.search.helpers import CARDS
    from tests.compiler.realization import helpers as corpus

    case = corpus.load_case(corpus.CASES_DIR / "fused/linear-add-place-cut-sm70.json")
    ctx = Context.from_target(case.compute_cap, gpu_name=CARDS[case.compute_cap], precision={})
    program = case.program()
    name = program.inputs[0]
    program.hints.set("cuda.paged_buffers", ((name, 0, 1), ("absent", 0, 1)))
    with pinned_knobs(case.regime), unpinned_decisions():
        found = kernel_set(program, ctx, None)
    reading = [alone for _, alone in found.kernels if alone.buffer(name) is not None]
    assert reading, "some kernel of the set reads the paged input"
    assert all(alone.hints.get("cuda.paged_buffers", ()) == ((name, 0, 1),) for alone in reading)
