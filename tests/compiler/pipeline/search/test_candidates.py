"""Candidate files: a few measured schedules that are not the best known one, for the groups a hardware golden
holds — how many a group is filled to, what a build takes from a freeze and keeps, and the repository's files held to
their goldens and to the fresh lowering of their own programs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from emmy.compiler.pipeline.search import candidates
from emmy.compiler.pipeline.search.candidates import (
    FEWEST,
    LARGE,
    MOST,
    SMALL,
    _groups,
    _schedule,
    build,
    candidate_paths,
    golden_groups,
    target,
)
from emmy.compiler.pipeline.search.golden import GoldenFile, Measurements, Row, restamp
from emmy.compiler.pipeline.search.golden.repository import _RECORDS_DIR, document_of


def test_a_group_is_filled_by_the_size_of_its_schedule_space() -> None:
    """The fewest for a small space, the most for a large one, never fewer for a larger space."""
    assert target(0) == target(SMALL) == FEWEST
    assert target(LARGE) == target(LARGE * 1000) == MOST
    sizes = [4, 100, 1287, 297_900, 27_580_500]
    assert [target(size) for size in sizes] == sorted(target(size) for size in sizes)
    assert FEWEST < target(1287) < MOST


def _measured(golden: GoldenFile, head: Row, times: list[float]) -> GoldenFile:
    """A freeze of ``golden``'s card: ``head``'s kernel measured at one made-up schedule per time."""
    rows = [
        replace(head, name=f"freeze.{n}", knobs={**head.knobs, "WORK": f"w{n + 100}x1"}, measurements=Measurements(emmy_us=us))
        for n, us in enumerate(times)
    ]
    return GoldenFile(gpu_name=golden.gpu_name, compute_cap=golden.compute_cap, kernels=list(golden.kernels), rows=rows)


def test_a_build_takes_a_spread_of_the_freeze_and_keeps_what_the_file_holds(monkeypatch) -> None:
    """A group takes its target of the freeze's rows across their latency, never a schedule its golden holds; a second
    build keeps those rows and adds none; a row whose group the golden dropped goes."""
    golden = GoldenFile.load(_RECORDS_DIR / "rtx4080_sm89.json")
    monkeypatch.setattr(candidates, "space_size", lambda pool: 1287)
    wanted = target(1287)
    head = next(row for row in golden.rows if row.measured and row.knobs)
    freeze = _measured(golden, head, [float(us) for us in range(10, 40)])
    freeze.rows.append(replace(head, name="the golden's own schedule", measurements=Measurements(emmy_us=1.0)))

    document, report = build(golden, [freeze])
    assert (document.programs, document.kernels, document.routing) == (golden.programs, golden.kernels, golden.routing)
    taken = [row.measurements.emmy_us for row in document.rows]
    assert len(taken) == wanted and taken[0] == 10.0 and taken[-1] == 39.0 and taken == sorted(taken)
    assert all(row.kernel == head.kernel and row.pins == head.pins for row in document.rows)
    assert len({row.name for row in document.rows}) == wanted
    assert report.rows == wanted and report.groups == len(report.short) + 1
    document.check()

    again, _ = build(golden, [_measured(golden, head, [5.0, 6.0])], document)
    assert again.rows == document.rows, "a full group takes nothing more"

    other = next(row for row in golden.rows if row.kernel != head.kernel and row.measured)
    without = replace(golden, rows=[row for row in golden.rows if row.kernel != head.kernel])
    left, _ = build(without, [_measured(golden, other, [7.0])], document)
    assert [row.measurements.emmy_us for row in left.rows] == [7.0]


def test_a_candidate_faster_than_its_golden_is_kept_and_reported(monkeypatch) -> None:
    golden = GoldenFile.load(_RECORDS_DIR / "rtx4080_sm89.json")
    monkeypatch.setattr(candidates, "space_size", lambda pool: 4)
    head = next(row for row in golden.rows if row.measured and row.knobs)
    best = min(row.measurements.emmy_us for row in golden.rows if row.measured and row.kernel == head.kernel)
    document, report = build(golden, [_measured(golden, head, [best / 2])])
    assert [row.measurements.emmy_us for row in document.rows] == [best / 2]
    assert [(name, us) for name, us, _ in report.faster] == [(document.rows[0].name, best / 2)]


def _golden_of(path: Path) -> GoldenFile:
    return document_of(_RECORDS_DIR / path.name)


@pytest.mark.parametrize("path", [pytest.param(path, id=path.name) for path in candidate_paths()])
def test_a_candidate_file_holds_candidates_of_its_hardware_goldens_groups(path: Path) -> None:
    """A candidate file carries its hardware golden's programs, kernels and decisions; every row is a measured or
    proposed schedule of one of the golden's groups that the golden does not hold; no group holds more than the
    most."""
    document, golden = GoldenFile.load(path), _golden_of(path)
    assert (document.gpu_name, document.compute_cap) == (golden.gpu_name, golden.compute_cap)
    assert (document.programs, document.kernels, document.routing) == (golden.programs, golden.kernels, golden.routing)
    groups = golden_groups(golden)
    for key, rows in _groups(document).items():
        key = key if key in groups else (key[0], "{}", key[2])  # a golden row that records no sizes counts at every size
        assert key in groups, f"{rows[0].name}: no group of the golden"
        assert len(rows) <= MOST, f"{rows[0].name}: {len(rows)} candidates"
        schedules = [_schedule(row) for row in rows]
        assert len(set(schedules)) == len(schedules), f"{rows[0].name}: one schedule twice"
        assert not {_schedule(row) for row in groups[key]} & set(schedules), f"{rows[0].name}: a schedule the golden holds"
    assert sum(len(rows) for rows in _groups(document).values()) == len(document.rows), "every row spells a schedule on a live kernel"


def _program_parameters():
    parameters = []
    for path in candidate_paths():
        document = document_of(path)
        # A piece has no program of its own: its rows are held to the program of the target it was cut from.
        roots = {path_to[0].parent if (path_to := document.path_to(row.kernel)) else row.kernel for row in document.rows}
        for index in sorted({document.kernel(root).traced for root in roots} - {None}):
            parameters.append(pytest.param(path, index, id=f"{path.name}/program-{index}"))
    return parameters


@pytest.mark.parametrize(("path", "traced"), _program_parameters())
def test_a_candidate_file_is_the_fresh_lowering(path: Path, traced: int) -> None:
    """A restamp of a candidate file onto the fresh lowering of one traced program changes nothing, as for a golden
    (``test_golden.py``): one node per program a candidate row's own kernel was traced from — the others are the
    golden's, lowered once in its own test."""
    document = document_of(path)
    fresh, report = restamp(document, traced=traced)
    assert fresh == document and not report.changed, "\n".join(report.lines())
