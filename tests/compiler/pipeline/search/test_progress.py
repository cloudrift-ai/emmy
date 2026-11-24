"""``search/progress`` — the progress lines and per-phase timing of the refit commands: a line per interval with
the count done, the rate and the time left; a phase per timed step in the open report; and a map whose results
keep the tasks' order on any worker count, with each task's seconds and the slowest tasks described."""

from __future__ import annotations

import logging
import math
from multiprocessing import get_context

import pytest

from emmy.compiler.pipeline.search import progress


def test_a_progress_line_counts_done_of_total_with_rate_and_eta(caplog):
    """A line is logged once the interval passed, never before it; it carries the count, the share and the ETA."""
    line = progress.Progress("pools", 4, "pools", every=3600)
    with caplog.at_level(logging.INFO, logger="emmy.progress"):
        line.advance()
        assert not caplog.records  # the interval has not passed
        line.every = 0
        line.advance()
    [record] = caplog.records
    assert record.getMessage().startswith("[pools] 2/4 pools (50.0%)") and "eta" in record.getMessage()


def test_a_phase_records_into_the_open_report_only():
    """A phase outside a recording is timed and dropped; inside one it lands in the report, in order."""
    with progress.phase("dropped"):
        pass
    with progress.recording("cmd") as report:
        with progress.phase("first"):
            pass
        with progress.phase("second", workers=3) as entry:
            entry.items = 7
    assert [p.name for p in report.phases] == ["first", "second"]
    assert report.to_json()["phases"][1] == {"name": "second", "seconds": pytest.approx(0, abs=1), "workers": 3, "items": 7}


@pytest.mark.parametrize("jobs", [1, 2])
def test_timed_map_keeps_the_order_and_records_the_slowest_tasks(jobs, caplog):
    """The results are the function's, in the tasks' order, at any worker count; the phase holds the count by card,
    the summed task seconds and the slowest tasks as the caller describes them; a closing line counts every task."""
    tasks = [4.0, 9.0, 16.0]
    with caplog.at_level(logging.INFO, logger="emmy.progress"), progress.recording("cmd") as report:
        results = progress.timed_map(
            math.sqrt,
            tasks,
            name="roots",
            jobs=jobs,
            mp_context=get_context("spawn"),
            describe=lambda task, root: {"task": task, "root": root},
            card=lambda t: f"c{t > 5}",
        )
    assert results == [2.0, 3.0, 4.0]
    [entry] = report.phases
    assert (entry.name, entry.items, entry.workers, entry.by_card) == ("roots", 3, jobs, {"cFalse": 1, "cTrue": 2})
    assert entry.task_seconds is not None and sorted(e["task"] for e in entry.slowest) == tasks
    assert all(e["root"] == math.sqrt(e["task"]) and e["seconds"] >= 0 for e in entry.slowest)
    assert caplog.records[-1].getMessage().startswith("[roots] 3/3 pools (100.0%)")


def test_timed_map_raises_a_workers_exception():
    with pytest.raises(ValueError):
        progress.timed_map(math.sqrt, [1.0, -1.0], name="roots", jobs=2, mp_context=get_context("spawn"))


def test_a_report_logs_its_phases_slowest_tasks_and_counts(caplog):
    with progress.recording("cmd") as report:
        progress.timed_map(math.sqrt, [1.0], name="roots", describe=lambda task, _root: {"kernel": "k", "card": "c"})
        report.count("golden pools", {"c": 1})
    with caplog.at_level(logging.INFO, logger="emmy.progress"):
        report.log()
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "timing: cmd" in text and "roots" in text and "kernel=k card=c" in text and "golden pools: c=1" in text
    assert set(report.to_json()) == {"command", "seconds", "phases", "counts"}
