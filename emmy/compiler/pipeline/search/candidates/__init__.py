"""Candidate files: measured schedules that are NOT the best known one, for the groups the hardware goldens hold.

A hardware golden records the best known schedule of each group — one kernel, at one set of sizes, in one input
regime. A candidate file holds a few other schedules measured on the same group: 3 for a kernel with a small schedule
space, up to 10 for the largest (:func:`target`), spread across their latency so the slow ones are there beside the
near misses. It is a file in the golden file format (``golden.GoldenFile``) and is never a golden: nothing in it is
the best of anything, no compile reads it as evidence, and the reproduction gate does not see it.

One file per hardware golden, under the same file name, in ``records/`` here. It copies the golden's traced programs,
kernels and decisions, so it restamps onto a fresh lowering exactly as its golden does (``emmy candidates restamp``),
and its rows are the candidates alone: the best row stays in the golden, and the two join on the kernel's exact
identity, the sizes and the regime.

:func:`build` fills a file from a measurement freeze of the same card, which is where a tune's measured rows end up;
a group the freeze does not cover is listed short, for a tune on the card to fill.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from emmy.compiler.pipeline.knob import family_of, is_off_value
from emmy.compiler.pipeline.search.dataset.pool import GoldenPool, GoldenRow, regime_of
from emmy.compiler.pipeline.search.db import knobs_json
from emmy.compiler.pipeline.search.golden import GoldenFile, Row
from emmy.compiler.pipeline.search.space import COLD_CACHE

logger = logging.getLogger(__name__)

#: The candidate files kept with the repository, one per hardware golden. Left out of the wheel: training data.
RECORDS = Path(__file__).parent / "records"

#: Candidates a group is filled to: ``FEWEST`` where the kernel's schedule space holds ``SMALL`` schedules or fewer,
#: ``MOST`` where it holds ``LARGE`` or more, by the logarithm of the size between.
FEWEST, MOST = 3, 10
SMALL, LARGE = 10, 1_000_000


def candidate_paths() -> list[Path]:
    """The repository's candidate files."""
    return sorted(RECORDS.glob("*.json"))


def target(size: int) -> int:
    """How many candidates a group whose kernel's schedule space holds ``size`` schedules is filled to."""
    scale = math.log(max(size, SMALL) / SMALL) / math.log(LARGE / SMALL)
    return min(MOST, FEWEST + round((MOST - FEWEST) * scale))


def space_size(pool: GoldenPool) -> int:
    """An upper bound on the schedules ``pool``'s kernel can run with: the bound its schedule fork declares
    (``Fork.pool_bound``), read off the lowering of the kernel's definition without walking the space. ``0`` for a
    kernel whose definition does not lower on its own, or that opens no schedule fork."""
    from emmy.compiler.pipeline import LOWERING_PASSES, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.pipeline import NO_OPTION, Run  # noqa: PLC0415
    from emmy.compiler.pipeline.search.pins import pinned_knobs, unpinned_decisions  # noqa: PLC0415
    from emmy.compiler.pipeline.search.ranking import pool_context  # noqa: PLC0415

    size = 0

    def decide(fork):
        nonlocal size
        size = max([size, *(bound for option in fork.options if (bound := getattr(option, "pool_bound", None)) is not None)])
        return next(fork.leaves(), NO_OPTION)

    try:
        with pinned_knobs(pool.pins), unpinned_decisions():
            Run(pipeline=Pipeline.build(LOWERING_PASSES[:-1]), ctx=pool_context(pool)).resolve(pool.kernel.program(pool.bindings), decide)
    except Exception as exc:  # noqa: BLE001 — whatever the lowering raises, the group has no space to size
        logger.debug("%s does not lower on its own (%s: %s)", pool.name, type(exc).__name__, exc)
        return 0
    return size


def _groups(document: GoldenFile) -> dict[tuple, list[Row]]:
    """``document``'s rows that spell a schedule, by group: the kernel's exact identity, the sizes and the regime."""
    identity = document.identities()
    out: dict[tuple, list[Row]] = {}
    for row in document.rows:
        if row.knobs is not None and row.kernel in identity:
            regime = regime_of(bool(row.pins.get(COLD_CACHE.name)), row.pins)
            out.setdefault((identity[row.kernel], knobs_json(row.bindings), regime), []).append(row)
    return out


def _of_group(groups: dict[tuple, list[Row]], key: tuple) -> list[Row]:
    """The rows of ``groups`` on the golden group ``key``. A golden row that records no sizes counts at every size of
    its kernel (``GoldenFile.target_rows``), so its group takes the kernel's rows in its regime at any sizes."""
    identity, sizes, regime = key
    if sizes != "{}":
        return groups.get(key, [])
    return [row for (other, _sizes, lane), rows in groups.items() if (other, lane) == (identity, regime) for row in rows]


def _schedule(row: Row) -> tuple:
    """A row's schedule as a key: its knobs by value, the ones at their OFF value left out."""
    return tuple(sorted((key, str(value)) for key, value in (row.knobs or {}).items() if not is_off_value(family_of(key), value)))


def _spread(rows: list[Row], count: int) -> list[Row]:
    """``count`` of ``rows`` at evenly spaced ranks of their latency, the fastest and the slowest among them."""
    rows = sorted(rows, key=lambda row: (row.measurements.emmy_us, _schedule(row)))
    if count >= len(rows):
        return rows
    if count == 1:
        return rows[:1]
    return [rows[round(index * (len(rows) - 1) / (count - 1))] for index in range(count)]


@dataclass
class Report:
    """What a build left: the groups and the candidate rows, the groups under their target as ``(name, held,
    target)``, and the candidates faster than every golden row of their group as ``(name, candidate µs, golden µs)``."""

    groups: int = 0
    rows: int = 0
    short: list[tuple[str, int, int]] = field(default_factory=list)
    faster: list[tuple[str, float, float]] = field(default_factory=list)


def build(golden: GoldenFile, freezes: Sequence[GoldenFile] = (), existing: GoldenFile | None = None) -> tuple[GoldenFile, Report]:
    """The candidate file of ``golden``: its programs, kernels and decisions, and for each of its measured groups the
    candidate rows ``existing`` holds, topped up to the group's :func:`target` from the measured rows ``freezes`` (the
    card's freeze files, a later file's row first at one schedule) hold on the same group. A schedule a golden row of
    the group spells is never a candidate; a row of ``existing`` whose group the golden no longer holds is dropped."""
    document = GoldenFile(
        gpu_name=golden.gpu_name,
        compute_cap=golden.compute_cap,
        programs=list(golden.programs),
        kernels=list(golden.kernels),
        routing=list(golden.routing),
    )
    report = Report()
    held = _groups(existing) if existing is not None else {}
    measured: dict[tuple, list[Row]] = {}
    for freeze in reversed(freezes):
        for key, rows in _groups(freeze).items():
            measured.setdefault(key, []).extend(rows)
    for key, rows in _groups(golden).items():
        best = [row for row in rows if row.measured]
        if not best:
            continue
        _identity, _sizes, regime = key
        head = min(best, key=lambda row: row.measurements.emmy_us)
        timed = tuple(GoldenRow({str(k): str(v) for k, v in row.knobs.items()}, row.measurements.emmy_us, "") for row in best)
        pool = GoldenPool(golden.gpu_name or "", tuple(golden.compute_cap), regime, golden.kernel(head.kernel), dict(head.bindings), timed)
        wanted = target(space_size(pool))
        taken = {_schedule(row) for row in rows}
        kept = [row for row in _of_group(held, key) if _schedule(row) not in taken]
        taken.update(_schedule(row) for row in kept)
        fresh = {}  # one row per schedule the group does not hold yet
        for row in _of_group(measured, key):
            if row.measured and _schedule(row) not in taken:
                fresh.setdefault(_schedule(row), row)
        names = {row.name for row in kept}
        numbers = (f"{head.name}.c{n}" for n in range(len(kept) + len(fresh) + 1))
        added = [
            Row(
                name=next(name for name in numbers if name not in names),
                kernel=head.kernel,
                bindings=dict(head.bindings or row.bindings),
                pins=dict(head.pins),
                knobs=dict(row.knobs),
                measurements=row.measurements,
            )
            for row in _spread(list(fresh.values()), max(0, wanted - len(kept)))
        ]
        # A restamp of the golden can re-key a kernel; a kept row names it as the golden's file does now.
        group = [replace(row, kernel=head.kernel) for row in kept] + added
        document.rows.extend(group)
        report.groups += 1
        report.rows += len(group)
        if len(group) < wanted:
            report.short.append((head.name, len(group), wanted))
        report.faster.extend(
            (row.name, row.measurements.emmy_us, head.measurements.emmy_us)
            for row in group
            if row.measured and row.measurements.emmy_us < head.measurements.emmy_us
        )
    return document, report
