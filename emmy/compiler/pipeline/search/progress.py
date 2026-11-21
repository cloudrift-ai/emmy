"""Progress lines and per-phase timing for the commands that refit the priors (``emmy db import``, ``emmy db export``,
``emmy fit``, ``emmy eval prior``).

Three pieces, all mechanical: timings and counts, never a reading of them.

- :class:`Progress` logs one line about every :data:`EVERY` seconds while a phase runs: the items done of the total,
  the rate, the elapsed time and an estimate of the time left.
- :func:`phase` times one step of a command into the :class:`Report` a :func:`recording` opened, if one is open.
- :func:`timed_map` runs one function over tasks, in this process or on worker processes, and returns the results in
  the tasks' order, as ``ProcessPoolExecutor.map`` would. Each worker times its own task and sends the seconds back
  with the result, so the timing adds no synchronization. The phase keeps the slowest tasks, each described by the
  caller (its kernel, card, sizes, how many rows it enumerated).

A command opens one :func:`recording`, logs :meth:`Report.log` at the end and may write :meth:`Report.to_json`
beside its output, so a past run's bottleneck can be read afterwards. The timings stay out of every deterministic
file (a dataset's manifest, a fit's metrics): those carry no clock.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import partial

# Outside the ``emmy.compiler`` tree, which ``eval prior`` mutes while its reproduction compiles.
logger = logging.getLogger("emmy.progress")

#: Seconds between two progress lines of one phase.
EVERY = 30.0
#: How many of a phase's slowest tasks its timing keeps.
SLOWEST = 10
#: The file a command writes its timing to, beside its output (a dataset directory, a fit run dir).
TIMING = "timing.json"


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    return f"{hours}h{rest // 60:02d}m{rest % 60:02d}s" if hours else f"{rest // 60}m{rest % 60:02d}s"


class Progress:
    """The progress of one phase over ``total`` items: :meth:`advance` counts them done and logs a line when
    :data:`EVERY` seconds passed since the last one (or since the start)."""

    def __init__(self, label: str, total: int, unit: str = "items", every: float = EVERY) -> None:
        self.label, self.total, self.unit, self.every = label, total, unit, every
        self.done = 0
        self.start = self.last = time.monotonic()

    def advance(self, n: int = 1) -> None:
        self.done += n
        self.tick()

    def tick(self) -> None:
        """Log a line if :attr:`every` seconds passed since the last one."""
        if time.monotonic() - self.last >= self.every:
            self.line()

    def line(self) -> None:
        now = time.monotonic()
        self.last = now
        elapsed = now - self.start
        rate = self.done / elapsed if elapsed > 0 else 0.0
        eta = _clock((self.total - self.done) / rate) if rate > 0 else "?"
        share = f" ({100 * self.done / self.total:.1f}%)" if self.total else ""
        logger.info(
            "[%s] %d/%d %s%s, %.2f/s, elapsed %s, eta %s", self.label, self.done, self.total, self.unit, share, rate, _clock(elapsed), eta
        )


@dataclass
class Phase:
    """One timed step of a command: its wall time, the items it handled (by card where the caller says which card an
    item belongs to), the worker processes it ran on, the summed per-task seconds, and its slowest tasks."""

    name: str
    seconds: float = 0.0
    items: int | None = None
    workers: int = 1
    by_card: dict[str, int] = field(default_factory=dict)
    task_seconds: float | None = None
    slowest: list[dict] = field(default_factory=list)

    def to_json(self) -> dict:
        out = {"name": self.name, "seconds": round(self.seconds, 3), "workers": self.workers}
        if self.items is not None:
            out["items"] = self.items
        if self.by_card:
            out["by_card"] = dict(sorted(self.by_card.items()))
        if self.task_seconds is not None:
            out["task_seconds"] = round(self.task_seconds, 3)
        if self.slowest:
            out["slowest"] = self.slowest
        return out


@dataclass
class Report:
    """The phases one command ran, in order, and the counts it reports beside them (``counts``: a name, such as
    ``golden pools``, to a count per card)."""

    command: str
    phases: list[Phase] = field(default_factory=list)
    counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def count(self, name: str, by_card: Counter | dict) -> None:
        self.counts[name] = dict(sorted(by_card.items()))

    def to_json(self) -> dict:
        return {
            "command": self.command,
            "seconds": round(sum(p.seconds for p in self.phases), 3),
            "phases": [p.to_json() for p in self.phases],
            "counts": self.counts,
        }

    def log(self) -> None:
        logger.info("timing: %s, %s over %d phases", self.command, _clock(sum(p.seconds for p in self.phases)), len(self.phases))
        for p in self.phases:
            items = "" if p.items is None else f", {p.items} items"
            busy = "" if p.task_seconds is None else f", {p.task_seconds:.1f} s in tasks"
            logger.info("  %-28s %9.1f s, %d workers%s%s", p.name, p.seconds, p.workers, items, busy)
            if p.by_card:
                logger.info("    by card: %s", ", ".join(f"{card}={n}" for card, n in sorted(p.by_card.items())))
            for entry in p.slowest:
                logger.info("    %8.1f s  %s", entry["seconds"], " ".join(f"{k}={v}" for k, v in entry.items() if k != "seconds"))
        for name, by_card in self.counts.items():
            logger.info("  %s: %s", name, ", ".join(f"{card}={n}" for card, n in by_card.items()))


_REPORT: ContextVar[Report | None] = ContextVar("emmy_progress_report", default=None)


@contextmanager
def recording(command: str) -> Iterator[Report]:
    """Open the :class:`Report` the phases run inside this block record into."""
    report = Report(command)
    token = _REPORT.set(report)
    try:
        yield report
    finally:
        _REPORT.reset(token)


def current() -> Report | None:
    """The :class:`Report` a :func:`recording` opened, or ``None``."""
    return _REPORT.get()


@contextmanager
def phase(name: str, *, workers: int = 1) -> Iterator[Phase]:
    """Time the block as one :class:`Phase` of the open report (none open: the phase is timed and dropped). The
    block may fill the phase's counts."""
    entry = Phase(name, workers=workers)
    start = time.monotonic()
    try:
        yield entry
    finally:
        entry.seconds = time.monotonic() - start
        if (report := current()) is not None:
            report.phases.append(entry)


def _timed(fn: Callable, task):
    start = time.perf_counter()
    result = fn(task)
    return result, time.perf_counter() - start


def timed_map(
    fn: Callable,
    tasks: Sequence,
    *,
    name: str,
    jobs: int = 1,
    unit: str = "pools",
    mp_context=None,
    initializer: Callable | None = None,
    initargs: tuple = (),
    describe: Callable | None = None,
    card: Callable | None = None,
) -> list:
    """``fn`` over ``tasks`` as a :func:`phase` called ``name``, in the tasks' order: in this process when ``jobs`` is
    1, else one task at a time per worker process (``mp_context``, ``initializer`` and ``initargs`` go to the
    executor). A :class:`Progress` line is logged about every :data:`EVERY` seconds, also while no task finishes.
    ``describe(task, result)`` spells one of the :data:`SLOWEST` slowest tasks for the phase's timing, and
    ``card(task)`` names the card a task's item belongs to for the phase's per-card count. A worker's exception
    is raised here, the tasks not yet started cancelled."""
    progress = Progress(name, len(tasks), unit)
    with phase(name, workers=jobs if jobs > 1 else 1) as entry:
        if jobs == 1:
            timed = []
            for task in tasks:
                timed.append(_timed(fn, task))
                progress.advance()
        else:
            with ProcessPoolExecutor(jobs, mp_context=mp_context, initializer=initializer, initargs=initargs) as workers:
                futures = [workers.submit(partial(_timed, fn), task) for task in tasks]
                pending = set(futures)
                while pending:
                    done, pending = wait(
                        pending, timeout=max(0.0, progress.every - (time.monotonic() - progress.last)), return_when=FIRST_COMPLETED
                    )
                    for future in done:
                        if future.exception() is not None:
                            for other in pending:
                                other.cancel()
                            raise future.exception()
                    progress.done += len(done)
                    progress.tick()
                timed = [future.result() for future in futures]
        results = [result for result, _seconds in timed]
        entry.items = len(tasks)
        entry.task_seconds = sum(seconds for _result, seconds in timed)
        if card is not None:
            entry.by_card = dict(Counter(card(task) for task in tasks))
        slowest = sorted(range(len(tasks)), key=lambda i: -timed[i][1])[:SLOWEST]
        entry.slowest = [
            {"seconds": round(timed[i][1], 3), **(describe(tasks[i], results[i]) if describe is not None else {"task": i})} for i in slowest
        ]
    if tasks:
        progress.line()
    return results
