"""Every row of a golden file that carries a schedule, compiled under it with the file as the compile's golden
evidence — the deploy contract asked of each row. It is the one check a measurement freeze admits: a freeze stores no
traced program, so a restamp lowers nothing afresh and leaves the file as it is while the compiler moves under it. A
row the compiler no longer realizes — the kernel it built is not the row's, or the schedule it stamped on it is not
the stored one — is evidence no compile can take; ``emmy golden realize`` reports those rows."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path

from emmy import config
from emmy.compiler.pipeline.search.golden.format import GoldenFile, Row

logger = logging.getLogger(__name__)


@dataclass
class Realization:
    """One file's report: how many rows carry a schedule, and ``(row name, kernel ref, reason)`` for each the compile
    does not realize."""

    file: str
    gpu_name: str
    rows: int
    failed: list[tuple[str, str, str]] = field(default_factory=list)

    def wire(self) -> dict:
        failed = [{"row": name, "kernel": kernel, "reason": reason} for name, kernel, reason in self.failed]
        return {"file": self.file, "gpu_name": self.gpu_name, "rows": self.rows, "failed": failed}


def realized(document: GoldenFile, row: Row) -> str | None:
    """Why the compile does not realize ``row`` — ``None`` when it does. The row's kernel is the whole program (its
    formed body, or its tile re-derived through the file's recorded decisions), its regime and schedule are pinned
    and the file is the golden evidence: the lowering ``compile --golden PATH --kernel K --realization ROW`` runs,
    at the file's card."""
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415
    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.knob import schedule_row_key  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden.repository import evidence_scope  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden.restamp import isolated_program  # noqa: PLC0415
    from emmy.compiler.pipeline.search.pins import golden_regime, pinned_knobs  # noqa: PLC0415
    from emmy.compiler.wire import kernel_identity  # noqa: PLC0415

    kernel = document.kernel(row.kernel)
    try:
        graph = isolated_program(document, kernel, row.bindings)
        ctx = Context.from_target(tuple(document.compute_cap), gpu_name=document.gpu_name or None)
        with pinned_knobs({**golden_regime(document, [row]), "PLACE": "fuse", **(row.knobs or {})}), evidence_scope([document]):
            result = Pipeline.build(CUDA_PASSES).run(graph, ctx=ctx, db=None)
    except Exception as exc:  # noqa: BLE001 — the reason is the product
        return " ".join(f"{type(exc).__name__}: {exc}".split())[:300]
    built = [node.op for node in result.nodes.values() if isinstance(node.op, CudaOp)]
    mine = [op for op in built if kernel_identity(op) == kernel.exact_identity]
    if not mine:
        return "the compile builds " + (", ".join(op.kernel_name for op in built) if built else "no CUDA kernel") + " instead"
    stored = dict(schedule_row_key(dict(row.knobs or {})))
    stamped = dict(schedule_row_key(dict(mine[0].knobs or {})))
    if not stamped:
        return "the compile stamps no schedule on the kernel"
    if stored != stamped:
        changed = sorted(key for key in set(stored) | set(stamped) if stored.get(key) != stamped.get(key))
        return "schedule differs: " + ", ".join(f"{key} {stored.get(key)!r} -> {stamped.get(key)!r}" for key in changed)
    return None


_DOCUMENTS: dict[str, GoldenFile] = {}  # the files this process has loaded, by path


def _realize_task(task: tuple[str, int]) -> str | None:
    path, index = task
    if path not in _DOCUMENTS:
        _DOCUMENTS[path] = GoldenFile.load(path)
    document = _DOCUMENTS[path]
    return realized(document, document.rows[index])


def realize(paths: Iterable[Path], *, jobs: int | None = None) -> list[Realization]:
    """The report of every file in ``paths``: each row with a schedule compiled under it, ``jobs`` rows side by side
    on spawned worker processes (``EMMY_WORKERS`` by default; ``1`` runs them in this process), each process loading
    a file once."""
    jobs = config.workers() if jobs is None else jobs
    files: list[tuple[Realization, GoldenFile, list[int]]] = []
    tasks: list[tuple[str, int]] = []
    for path in paths:
        document = _DOCUMENTS[str(path)] = GoldenFile.load(path)
        indices = [index for index, row in enumerate(document.rows) if row.knobs is not None]
        files.append((Realization(str(path), document.gpu_name or "", len(indices)), document, indices))
        tasks.extend((str(path), index) for index in indices)
    logger.info("Realizing %d rows of %d file(s), %d at a time ...", len(tasks), len(files), jobs)
    if jobs == 1 or not tasks:
        return list(_collect(files, map(_realize_task, tasks)))
    context = get_context("spawn")
    with ProcessPoolExecutor(min(jobs, len(tasks)), mp_context=context, initializer=config.set_workers, initargs=(1,)) as workers:
        return list(_collect(files, workers.map(_realize_task, tasks, chunksize=4)))


def _collect(files: list[tuple[Realization, GoldenFile, list[int]]], reasons: Iterator[str | None]) -> Iterator[Realization]:
    for report, document, indices in files:
        for index, reason in zip(indices, reasons, strict=False):
            if reason is not None:
                row = document.rows[index]
                report.failed.append((row.name, row.kernel, reason))
        logger.info("%s: %d of %d rows realize", report.file, report.rows - len(report.failed), report.rows)
        yield report
