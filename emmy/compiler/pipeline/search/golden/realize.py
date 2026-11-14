"""Every row of a golden file that carries a schedule, compiled under it with the file as the compile's golden
evidence — the deploy contract asked of each row. It is the one check a measurement freeze admits: a freeze stores no
traced program, so a restamp lowers nothing afresh and leaves the file as it is while the compiler moves under it. A
row the compiler no longer realizes — the kernel it built is not the row's, or a knob the row spells is not the one
the kernel carries (``pins.unreproducible_pin_flag``, the gate a bench holds a pinned row to) — is evidence no compile
can take; ``emmy golden realize`` reports those rows, and with ``--drop`` writes the file back without them.

A piece is rebuilt through the route a repository golden of the same card records for the same kernel
(:func:`golden_routes`), and through the file's own only when none does. A golden's decisions are held to the fresh
lowering; a freeze's are frozen with it and go stale, and a tune of one piece of a golden freezes the piece's rows
with no decision at all — the golden keeps the ones that cut it."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
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


def realized(document: GoldenFile, row: Row, route: tuple[GoldenFile, str] | None = None) -> str | None:
    """Why the compile does not realize ``row`` — ``None`` when it does. The row's kernel is the whole program (its
    formed body, or its tile re-derived through the file's recorded decisions), its regime and schedule are pinned
    and the file is the golden evidence: the lowering ``compile --golden PATH --kernel K --realization ROW`` runs,
    at the file's card. ``route`` is the golden, and the kernel's ``ref`` in it, whose recorded decisions re-derive a
    piece in place of the file's own."""
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415
    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.knob import tuning_knob_items  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden.repository import evidence_scope  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden.restamp import isolated_program  # noqa: PLC0415
    from emmy.compiler.pipeline.search.pins import golden_regime, pinned_knobs, unreproducible_pin_flag  # noqa: PLC0415
    from emmy.compiler.wire import kernel_identity  # noqa: PLC0415

    kernel = document.kernel(row.kernel)
    source, stored = (document, kernel) if route is None else (route[0], route[0].kernel(route[1]))
    try:
        graph = isolated_program(source, stored, row.bindings)
        ctx = Context.from_target(tuple(document.compute_cap), gpu_name=document.gpu_name or None)
        with pinned_knobs({**golden_regime(document, [row]), "PLACE": "fuse", **(row.knobs or {})}), evidence_scope([document]):
            result = Pipeline.build(CUDA_PASSES).run(graph, ctx=ctx, db=None)
    except Exception as exc:  # noqa: BLE001 — the reason is the product
        return " ".join(f"{type(exc).__name__}: {exc}".split())[:300]
    built = [node.op for node in result.nodes.values() if isinstance(node.op, CudaOp)]
    mine = [op for op in built if kernel_identity(op) == kernel.exact_identity]
    if not mine:
        return "the compile builds " + (", ".join(op.kernel_name for op in built) if built else "no CUDA kernel") + " instead"
    # The gate a bench holds a pinned row to before it times it: every knob the row spells is the one the kernel carries.
    return unreproducible_pin_flag(dict(tuning_knob_items(row.knobs or {})), [dict(op.knobs or {}) for op in mine])


def golden_routes(document: GoldenFile) -> dict[str, tuple[Path, str]]:
    """For each measured kernel of ``document`` that only a parent's program reaches, by its ``ref``: the repository
    golden of the same card that stores the same kernel under a route, and the kernel's ``ref`` there. Empty, at no
    cost, for a file that measured no such kernel."""
    from emmy import gpu  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden.repository import document_of, repository_golden_paths  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden.restamp import _identity  # noqa: PLC0415

    measured = {row.kernel for row in document.rows if row.knobs is not None}
    derived = {_identity(kernel): kernel.ref for kernel in document.kernels if kernel.ref in measured and not kernel.formed}
    derived.pop(None, None)
    routes: dict[str, tuple[Path, str]] = {}
    if not derived:
        return routes
    card = (gpu.canonical_name(document.gpu_name or ""), tuple(document.compute_cap))
    with repository_golden_paths() as paths:
        for path in paths:
            golden = document_of(path)
            if (gpu.canonical_name(golden.gpu_name or ""), tuple(golden.compute_cap)) != card:
                continue
            pieces = {child for route in golden.routing for child in route.children}
            for kernel in golden.kernels:
                if kernel.ref in pieces and (ref := derived.get(_identity(kernel))) is not None:
                    routes.setdefault(ref, (path, kernel.ref))
    return routes


_DOCUMENTS: dict[str, GoldenFile] = {}  # the files this process has loaded, by path


def _realize_task(task: tuple[str, int, tuple[Path, str] | None]) -> str | None:
    from emmy.compiler.pipeline.search.golden.repository import document_of  # noqa: PLC0415

    path, index, route = task
    if path not in _DOCUMENTS:
        _DOCUMENTS[path] = GoldenFile.load(path)
    document = _DOCUMENTS[path]
    return realized(document, document.rows[index], route and (document_of(route[0]), route[1]))


def realize(paths: Iterable[Path], *, jobs: int | None = None, drop: bool = False) -> list[Realization]:
    """The report of every file in ``paths``: each row with a schedule compiled under it, ``jobs`` rows side by side
    on spawned worker processes (``EMMY_WORKERS`` by default; ``1`` runs them in this process), each process loading
    a file once. With ``drop``, a file is written back without the rows the compile does not realize; its kernels and
    decisions stay, and a freeze sheds the ones no row is left under at its next fold."""
    jobs = config.workers() if jobs is None else jobs
    files: list[tuple[Realization, GoldenFile, list[int]]] = []
    tasks: list[tuple[str, int, tuple[Path, str] | None]] = []
    for path in paths:
        document = _DOCUMENTS[str(path)] = GoldenFile.load(path)
        indices = [index for index, row in enumerate(document.rows) if row.knobs is not None]
        files.append((Realization(str(path), document.gpu_name or "", len(indices)), document, indices))
        routes = golden_routes(document)
        tasks.extend((str(path), index, routes.get(document.rows[index].kernel)) for index in indices)
    logger.info("Realizing %d rows of %d file(s), %d at a time ...", len(tasks), len(files), jobs)
    if jobs == 1 or not tasks:
        return list(_collect(files, map(_realize_task, tasks), drop))
    context = get_context("spawn")
    with ProcessPoolExecutor(min(jobs, len(tasks)), mp_context=context, initializer=config.set_workers, initargs=(1,)) as workers:
        return list(_collect(files, workers.map(_realize_task, tasks, chunksize=4), drop))


def _collect(files: list[tuple[Realization, GoldenFile, list[int]]], reasons: Iterator[str | None], drop: bool) -> Iterator[Realization]:
    for report, document, indices in files:
        unrealized = {index: reason for index, reason in zip(indices, reasons, strict=False) if reason is not None}
        report.failed.extend(
            (document.rows[index].name, document.kernel(document.rows[index].kernel).name, reason) for index, reason in unrealized.items()
        )
        logger.info("%s: %d of %d rows realize", report.file, report.rows - len(report.failed), report.rows)
        if drop and unrealized:
            kept = [row for index, row in enumerate(document.rows) if index not in unrealized]
            replace(document, rows=kept).dump(report.file, overwrite=True)
        yield report
