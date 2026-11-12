"""Every row of a golden file that carries a schedule, compiled under it with the file as the compile's golden
evidence — the deploy contract asked of each row. It is the one check a measurement freeze admits: a freeze stores no
traced program, so a restamp lowers nothing afresh and leaves the file as it is while the compiler moves under it. A
row the compiler no longer realizes — the kernel it built is not the row's, or a knob the row spells is not the one
the kernel carries (``pins.unreproducible_pin_flag``, the gate a bench holds a pinned row to) — is evidence no compile
can take; ``emmy golden realize`` reports those rows, and with ``--drop`` writes the file back without them.

Before a row is compiled, its kernel has to be one a deploy can meet, and a deploy compiles the repository goldens'
programs: a freeze is those programs' candidate pools in full — every arm and every schedule a tune measured where a
golden keeps the one it took — and a row of any other kernel is evidence nothing reads. :func:`reach_tasks` walks the
goldens of a card as the placement dataset does (:func:`~..ranking.walk_placement`): from each target at the sizes
the golden's rows reach it, along the golden's own decisions one seam at a time, taking note of every kernel every
arm of every fork on the way leaves. A kernel one decision off a golden's path is reached; a kernel the lowering no
longer produces, or that only a program no repository golden holds produced, is not, and its rows are dropped first.
The full tree of arms is not walked: a remainder re-offers every seam it still holds, so the tree holds every subset
of a kernel's seams.

A piece whose stored body only a parent's program reaches is rebuilt through the route a repository golden of the
same card records for the same kernel, and through the file's own only when none does. A golden's decisions are held
to the fresh lowering; a freeze's are frozen with it and go stale, and a tune of one piece of a golden freezes the
piece's rows with no decision at all — the golden keeps the ones that cut it."""

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

UNREACHED = "no repository golden of the card reaches the kernel"


@dataclass
class Realization:
    """One file's report: how many rows carry a schedule, and ``(row name, kernel ref, reason)`` for each the compile
    does not realize — :data:`UNREACHED` for a row whose kernel no repository golden's program reaches."""

    file: str
    gpu_name: str
    rows: int
    failed: list[tuple[str, str, str]] = field(default_factory=list)

    def wire(self) -> dict:
        failed = [{"row": name, "kernel": kernel, "reason": reason} for name, kernel, reason in self.failed]
        return {"file": self.file, "gpu_name": self.gpu_name, "rows": self.rows, "failed": failed}


Card = tuple[str, tuple[int, ...]]  # a card as a golden file names it: canonical GPU name and compute capability


@dataclass
class Reach:
    """What the repository goldens of one card reach: every kernel's exact identity, and for a stored piece that only
    a parent's program reaches, the golden and the ``ref`` whose recorded route rebuilds it."""

    kernels: set[str] = field(default_factory=set)
    routes: dict[str, tuple[str, str]] = field(default_factory=dict)


def card_of(document: GoldenFile) -> Card:
    from emmy import gpu  # noqa: PLC0415

    return gpu.canonical_name(document.gpu_name or ""), tuple(document.compute_cap)


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
        identity = kernel.exact_identity
        graph = isolated_program(source, stored, row.bindings)
        ctx = Context.from_target(tuple(document.compute_cap), gpu_name=document.gpu_name or None)
        with pinned_knobs({**golden_regime(document, [row]), "PLACE": "fuse", **(row.knobs or {})}), evidence_scope([document]):
            result = Pipeline.build(CUDA_PASSES).run(graph, ctx=ctx, db=None)
    except Exception as exc:  # noqa: BLE001 — the reason is the product
        return " ".join(f"{type(exc).__name__}: {exc}".split())[:300]
    built = [node.op for node in result.nodes.values() if isinstance(node.op, CudaOp)]
    mine = [op for op in built if kernel_identity(op) == identity]
    if not mine:
        return "the compile builds " + (", ".join(op.kernel_name for op in built) if built else "no CUDA kernel") + " instead"
    # The gate a bench holds a pinned row to before it times it: every knob the row spells is the one the kernel carries.
    return unreproducible_pin_flag(dict(tuning_knob_items(row.knobs or {})), [dict(op.knobs or {}) for op in mine])


def reach_tasks(card: Card) -> list[tuple]:
    """The walks that make up the card's reach, one per repository golden of the card (its stored kernels) and one
    per target of it at each set of sizes and in each regime the golden's rows reach it at (every arm of every fork
    on the golden's path from there); a target no row reaches walks once, at the sizes it was recorded at."""
    from emmy.compiler.pipeline.search.dataset.pool import COLD_CACHE, regime_of  # noqa: PLC0415
    from emmy.compiler.pipeline.search.db import knobs_json  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden.evidence import reached_targets  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden.repository import document_of, repository_golden_paths  # noqa: PLC0415
    from emmy.compiler.pipeline.search.space import precision_pins  # noqa: PLC0415

    tasks: list[tuple] = []
    with repository_golden_paths() as paths:
        for path in paths:
            golden = document_of(path)
            if card_of(golden) != card:
                continue
            tasks.append((card, str(path)))
            targets = {kernel.ref: kernel for kernel in golden.targets()}
            walks: dict[tuple[str, str, str], tuple] = {}
            for ref, sizes, row in reached_targets(golden, golden.rows, targets):
                bindings = sizes or dict(targets[ref].bindings)  # a row that names no size counts at the target's own
                pins = row.pins | precision_pins(row.pins)
                regime = regime_of(bool(pins.get(COLD_CACHE.name)), pins)
                walks.setdefault((ref, knobs_json(bindings), regime), (card, str(path), ref, bindings, regime))
            for ref, kernel in targets.items():
                walks.setdefault((ref, knobs_json(kernel.bindings), regime_of()), (card, str(path), ref, kernel.bindings, regime_of()))
            tasks.extend(walks.values())
    return tasks


_DECISIONS: dict[str, dict[str, dict]] = {}  # per golden this process walked: the recorded kernel-set arm by kernel identity


def _reach_task(task: tuple) -> tuple[Card, Reach]:
    """One walk of :func:`reach_tasks`: a golden's stored kernels and the routes to its stored pieces, or the kernels
    every arm of every fork leaves on the golden's path from one target. A target whose walk the placement export
    does not take either (:func:`~..ranking._walk_packed`) reaches nothing beyond the golden's stored kernels."""
    from emmy.compiler.ir.base import ConstantOp  # noqa: PLC0415
    from emmy.compiler.pipeline.search.dataset import GoldenPool  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden.repository import document_of  # noqa: PLC0415
    from emmy.compiler.pipeline.search.ranking import pool_context, walk_placement  # noqa: PLC0415

    card, path, *walk = task
    golden = document_of(Path(path))
    identity = golden.identities()
    out = Reach()
    if not walk:
        out.kernels.update(identity.values())
        pieces = {child for route in golden.routing for child in route.children}
        out.routes.update((identity[ref], (path, ref)) for ref in identity if ref in pieces)
        return card, out
    if path not in _DECISIONS:
        # Of several decisions a golden took on one kernel, the one over more seams, as the placement export reads them.
        _DECISIONS[path] = {
            identity[route.parent]: {str(key): str(value) for key, value in route.arm.items()}
            for route in sorted(golden.routing, key=lambda route: len(route.arm))
            if route.parent in identity
        }
    ref, bindings, regime = walk
    target = golden.kernel(ref)
    try:
        graph = golden.executable(target, bindings)
        constants = {node_id: node.op.to_wire() for node_id, node in graph.nodes.items() if isinstance(node.op, ConstantOp)}
        pool = GoldenPool(golden.gpu_name or "", tuple(golden.compute_cap), regime, target, bindings, (), placement_constants=constants)
        forks, unmatched = walk_placement(pool, pool_context(pool), _DECISIONS[path])
    except Exception as exc:  # noqa: BLE001 — a target that does not walk leaves its rows in place; the log says which
        logger.warning(
            "%s: %s at %s does not walk (%s: %s); the golden's stored kernels alone are reached",
            Path(path).name,
            ref,
            bindings,
            type(exc).__name__,
            exc,
        )
        return card, out
    if unmatched:
        logger.info("%s: %s at %s: recorded cut not reached on %s", Path(path).name, ref, bindings, ", ".join(unmatched))
    out.kernels.update(key for fork in forks for left in fork.left for key in left)
    return card, out


_DOCUMENTS: dict[str, GoldenFile] = {}  # the files this process has loaded, by path
_REACH: dict[Card, Reach] = {}  # what the repository goldens reach, by card, handed to each worker


def _set_reach(reach: dict[Card, Reach]) -> None:
    _REACH.update(reach)


def _realize_task(task: tuple[str, int, Card]) -> str | None:
    from emmy.compiler.pipeline.search.golden.repository import document_of  # noqa: PLC0415

    path, index, card = task
    if path not in _DOCUMENTS:
        _DOCUMENTS[path] = GoldenFile.load(path)
    document = _DOCUMENTS[path]
    row, kernel = document.rows[index], document.kernel(document.rows[index].kernel)
    try:
        identity = kernel.exact_identity
    except Exception as exc:  # noqa: BLE001 — the reason is the product
        return " ".join(f"{type(exc).__name__}: {exc}".split())[:300]
    reach = _REACH[card]
    if identity not in reach.kernels:
        return UNREACHED
    route = None if kernel.formed else reach.routes.get(identity)
    return realized(document, row, route and (document_of(Path(route[0])), route[1]))


def realize(paths: Iterable[Path], *, jobs: int | None = None, drop: bool = False) -> list[Realization]:
    """The report of every file in ``paths``: the reach of each file's card walked (:func:`reach_tasks`), then each
    row with a schedule compiled under it, ``jobs`` walks or rows side by side on spawned worker processes
    (``EMMY_WORKERS`` by default; ``1`` runs them in this process), each process loading a file once. With ``drop``, a
    file is written back without the rows no golden reaches and the rows the compile does not realize; its kernels
    and decisions stay, and a freeze sheds the ones no row is left under at its next fold."""
    jobs = config.workers() if jobs is None else jobs
    files: list[tuple[Realization, GoldenFile, list[int]]] = []
    tasks: list[tuple[str, int, Card]] = []
    for path in paths:
        document = _DOCUMENTS[str(path)] = GoldenFile.load(path)
        indices = [index for index, row in enumerate(document.rows) if row.knobs is not None]
        files.append((Realization(str(path), document.gpu_name or "", len(indices)), document, indices))
        tasks.extend((str(path), index, card_of(document)) for index in indices)
    reach: dict[Card, Reach] = {card: Reach() for _, _, card in tasks}
    walks = [task for card in reach for task in reach_tasks(card)]
    logger.info("Walking %d golden target(s) of %d card(s), %d at a time ...", len(walks), len(reach), jobs)
    context = get_context("spawn")
    if jobs == 1 or not walks:
        walked = list(map(_reach_task, walks))
    else:
        with ProcessPoolExecutor(min(jobs, len(walks)), mp_context=context, initializer=config.set_workers, initargs=(1,)) as workers:
            walked = list(workers.map(_reach_task, walks, chunksize=1))
    for card, found in walked:
        reach[card].kernels |= found.kernels
        reach[card].routes.update(found.routes)
    logger.info("Realizing %d rows of %d file(s), %d at a time ...", len(tasks), len(files), jobs)
    if jobs == 1 or not tasks:
        _set_reach(reach)
        return list(_collect(files, map(_realize_task, tasks), drop))
    with ProcessPoolExecutor(min(jobs, len(tasks)), mp_context=context, initializer=_init_worker, initargs=(reach,)) as workers:
        return list(_collect(files, workers.map(_realize_task, tasks, chunksize=4), drop))


def _init_worker(reach: dict[Card, Reach]) -> None:
    config.set_workers(1)
    _set_reach(reach)


def _collect(files: list[tuple[Realization, GoldenFile, list[int]]], reasons: Iterator[str | None], drop: bool) -> Iterator[Realization]:
    for report, document, indices in files:
        unrealized = {index: reason for index, reason in zip(indices, reasons, strict=False) if reason is not None}
        report.failed.extend((document.rows[index].name, document.rows[index].kernel, reason) for index, reason in unrealized.items())
        unreached = sum(reason == UNREACHED for reason in unrealized.values())
        realized_rows = report.rows - len(report.failed)
        logger.info("%s: %d of %d rows realize (%d reached by no golden)", report.file, realized_rows, report.rows, unreached)
        if drop and unrealized:
            kept = [row for index, row in enumerate(document.rows) if index not in unrealized]
            replace(document, rows=kept).dump(report.file, overwrite=True)
        yield report
