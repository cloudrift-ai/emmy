"""Golden rows as tune DB rows — what a compile imports before it picks.

A golden file is the DB's shape, so the import goes row for row: every kernel a ``kernel`` row, every decision a
``routing`` row — and a ``taken`` row under the card, regime and sizes of each row of the file below it — every
measured row a ``perf`` row under the card and regime it was measured in. The one thing the import computes is
the key: a file names a kernel by a local ``ref``, the DB by the exact identity of its Loop IR
(``GoldenFile.identities``). The greedy then
has one read (``policy.greedy``): a kernel's measured schedule rows, and the kernel-set decisions stored on it
priced from the pieces' own rows.

:func:`evidence_db` is the compile's seam: the golden files in scope (``repository.documents_for_card``) are imported
once per scope digest into the tune DB the compile reads — a re-recorded file changes the digest, and the file's
earlier rows are let go first — or into a fresh in-memory instance when the compile has no DB. Only the rows of the
live precision regime are evidence there (:func:`regime_live`). :func:`rows_digests` is the part of that evidence one
compile's picks can depend on, which keys its serving pack entry. :func:`import_file` is ``emmy db import``'s: every
row under its own regime's context, and the constant inputs each target's program binds (:func:`import_constants`),
which the placement export reads back without the file.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

from emmy import config
from emmy.compiler.context import Context
from emmy.compiler.pipeline.knob import KnobType, family_of, registry
from emmy.compiler.pipeline.search.bench_record import point_stats
from emmy.compiler.pipeline.search.db import RoutingRow, SearchDB, knobs_json
from emmy.compiler.pipeline.search.db.freeze import is_lfs_pointer
from emmy.compiler.wire import symbolic_vars

from .format import GoldenFile, Row
from .repository import documents_for_card, scope_digest, scope_explicit

logger = logging.getLogger("emmy.compiler.pipeline")


def regime_live(pins: dict) -> bool:
    """Whether a row's input-pin regime IS the live one — exact per pin: the regime knobs (``space.REGIME_KNOBS``) as
    the row's pins spell them against the live regime (``space.regime_pins``, both sides — a knob the row omits reads as
    the recorded regime says), any other BOOL pin against its live pin (off when unset), anything else against the raw
    env string. Strict both ways: a row measured under ``FAST_MATH`` is no evidence for a standard deploy, and a
    standard row none under a live precision pin."""
    from emmy.compiler.pipeline.search.space import REGIME_KNOBS, regime_pins  # noqa: PLC0415

    if regime_pins(pins) != regime_pins():
        return False
    regime = {knob.name for knob in REGIME_KNOBS}
    knobs = registry()
    for name, value in pins.items():
        if str(name) in regime or family_of(str(name)) == "PLACE":
            continue
        kn = knobs.get(str(name))
        raw = kn.raw() if kn is not None else config.knob_raw(str(name))
        if kn is not None and kn.type is KnobType.BOOL:
            if bool(value) != (kn.parse(raw) if raw is not None else False):
                return False
        elif (raw or "") != str(value):
            return False
    return True


def regime_context(document: GoldenFile, pins: dict) -> Context:
    """The context a row measured under ``pins`` on ``document``'s card is filed under: the card, and the regime the
    pins spell."""
    return Context.from_target(tuple(document.compute_cap), gpu_name=document.gpu_name or None, precision=pins)


def import_rows(db: SearchDB, ctx: Context, document: GoldenFile, rows: Iterable[Row], *, source: str) -> int:
    """File ``document``'s kernels and decisions, and ``rows`` of it as ``perf`` rows under ``ctx``, ``source`` on
    every row. Returns the perf rows written: an unmeasured row is a proposal, not evidence. Measured or not, a
    row says its kernel is in the file under ``ctx`` at its sizes, so every decision on the way down to it
    (:meth:`GoldenFile.path_to`) is one the file took there."""
    identity = document.identities()
    for kernel in document.kernels:
        if kernel.ref in identity:
            db.record_kernel(kernel)
    for route in document.routing:
        if all(ref in identity for ref in (route.parent, *route.children)):
            db.record_routing(RoutingRow(identity[route.parent], route.arm, tuple(identity[child] for child in route.children)))
    written = 0
    for row in rows:
        if row.kernel not in identity:
            continue
        for route in document.path_to(row.kernel):
            sizes = symbolic_vars(document.kernel(route.parent).loop_ir)
            if route.parent in identity and sizes <= row.bindings.keys():
                bindings = {var: row.bindings[var] for var in sizes}
                db.record_taken(ctx, identity[route.parent], bindings=bindings, arm=route.arm, source=source)
        if row.measurements is None or row.knobs is None:
            continue
        db.record_perf(
            ctx,
            identity[row.kernel],
            bindings=row.bindings,
            knobs=row.knobs,
            status="ok",
            stats=point_stats(row.measurements.emmy_us),
            captured=True,
            source=source,
        )
        written += 1
    return written


def import_constants(db: SearchDB, document: GoldenFile, rows: Iterable[Row], *, source: str) -> int:
    """File under ``source`` the constant inputs each target kernel's program binds (``GoldenFile.executable``: a
    weight the kernel reads is the checkpoint's, as a deploy sees it), at every set of sizes ``rows`` of ``document``
    reach the target at — a row of the target at its own sizes, a piece's row at the target's symbolic dims out of
    them, the sizes :func:`import_rows` files the row and its decisions under. The placement export reads them
    back by source, kernel and sizes (``export.placement_pools``). Returns the constants written."""
    from emmy.compiler.ir.base import ConstantOp  # noqa: PLC0415

    identity = document.identities()
    targets = {kernel.ref: kernel for kernel in document.targets() if kernel.ref in identity}
    reached: dict[tuple[str, str], dict] = {}
    for row in rows:
        if row.kernel not in identity:
            continue
        path = document.path_to(row.kernel)
        root = path[0].parent if path else row.kernel
        if root not in targets:
            continue
        sizes = symbolic_vars(targets[root].loop_ir)
        if path and not sizes <= row.bindings.keys():
            continue
        bindings = {var: row.bindings[var] for var in sizes} if path else dict(row.bindings)
        reached.setdefault((root, knobs_json(bindings)), bindings)
    written = 0
    for (ref, _sizes), bindings in reached.items():
        inputs = targets[ref].program(bindings).inputs
        # Two files can store one kernel under other input names; the export walks the copy the DB keeps.
        stored = db.kernel(identity[ref])
        names = dict(zip(inputs, stored.program(bindings).inputs if stored is not None else inputs, strict=True))
        graph = document.executable(targets[ref], bindings)
        constants = {
            names[name]: node.op.to_wire() for name, node in graph.nodes.items() if name in names and isinstance(node.op, ConstantOp)
        }
        db.record_constants(source, identity[ref], bindings=bindings, constants=constants)
        written += len(constants)
    return written


def file_source(kind: str, path: Path | str) -> str:
    """The ``source`` an import files a file's rows under: what kind of file it is (``freeze`` or ``golden``) and the
    digest of its own bytes — a re-recorded file is another source, and a dataset holds a file once."""
    return f"{kind}:{hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]}"


def import_file(db: SearchDB, path: Path, source: str) -> Counter:
    """Import one golden file into ``db``: every row under the context of its own regime, and the constant inputs its
    targets' programs bind (:func:`import_constants`), filed under ``source`` (:func:`file_source`), which the
    instance then holds (``SearchDB.record_source``) whatever became of the rows. A source the instance already
    holds is skipped; a git-LFS pointer in the data's place is refused by name. Returns what was written, by kind."""
    if is_lfs_pointer(path):
        raise ValueError(f"{path} is a git-LFS pointer, not the data: run `git lfs install && git lfs pull` (in CI, check out with lfs)")
    if source in db.sources():
        logger.info("%s is already held (%s)", path.name, source)
        return Counter()
    document = GoldenFile.load(path)
    counts: Counter[str] = Counter(kernels=len(document.kernels), routing=len(document.routing))
    regimes = {tuple(sorted(row.pins.items())) for row in document.rows}
    for regime in sorted(regimes, key=repr):
        rows = [row for row in document.rows if tuple(sorted(row.pins.items())) == regime]
        counts["perf rows"] += import_rows(db, regime_context(document, dict(regime)), document, rows, source=source)
    counts["constants"] = import_constants(db, document, document.rows, source=source)
    db.record_source(source)
    logger.info("imported %s as %s: %s", path.name, source, ", ".join(f"{n} {what}" for what, n in sorted(counts.items())))
    return counts


def evidence_db(db: SearchDB | None, ctx: Context) -> SearchDB:
    """The DB a compile under ``ctx`` picks from, holding the golden rows in scope: ``db`` itself, the scope imported
    into it once per scope digest (a scope the file does not hold yet lets the earlier golden rows of this card and
    regime go first); with no ``db``, a fresh in-memory instance holding the scope."""
    gpu_name = getattr(ctx, "gpu_name", None) or ""
    documents = documents_for_card(gpu_name, tuple(ctx.compute_capability)) if gpu_name or scope_explicit() else []
    if not documents:
        return db if db is not None else SearchDB()
    source = f"golden:{scope_digest(gpu_name)[:12]}"
    if db is None:
        fresh = SearchDB()
        _import(fresh, ctx, documents, source)
        return fresh
    # Every worker of a parallel boot imports into one file at its first compile: the check, the forget and the
    # import are one step, or two workers collide on the rows' unique keys and a third reads a half-written scope.
    with db.exclusive():
        if source not in db.perf_sources(ctx):
            db.forget_perf(ctx, "golden:")
            _import(db, ctx, documents, source)
    return db


def rows_digests(documents: Iterable[GoldenFile], reads: Mapping[str, Iterable[str]]) -> dict[str, str]:
    """For each compile of ``reads`` (a name, and the exact identities of the kernels whose evidence it read:
    ``policy.greedy.evidence_reads``), a digest of the golden evidence in ``documents`` it picks from: the live
    regime's measured rows and the routing rows of those kernels and of every piece the routing rows reach from them,
    and whether any measured row of the live regime is in scope at all, since a compile with none prices no
    kernel-set arm from its pieces and so looks none of them up. Nothing else of the files can move its picks."""
    routes: dict[str, list] = {}
    rows: dict[str, list] = {}
    for document in documents:
        identity = document.identities()
        for route in document.routing:
            if all(ref in identity for ref in (route.parent, *route.children)):
                routes.setdefault(identity[route.parent], []).append([knobs_json(route.arm), [identity[c] for c in route.children]])
        for row in document.rows:
            if row.measured and row.knobs is not None and row.kernel in identity and regime_live(row.pins):
                rows.setdefault(identity[row.kernel], []).append(
                    [knobs_json(row.bindings), knobs_json(row.knobs), row.measurements.emmy_us]
                )

    def digest(kernels: Iterable[str]) -> str:
        reached, frontier = set(kernels), list(kernels)
        while frontier:  # the pieces of the routing rows on the kernels reached, and theirs
            for _, children in routes.get(frontier.pop(), ()):
                frontier.extend(child for child in children if child not in reached)
                reached.update(children)
        read = {kernel: [sorted(routes.get(kernel, ())), sorted(rows.get(kernel, ()))] for kernel in sorted(reached)}
        wire = [bool(rows), {kernel: entry for kernel, entry in read.items() if entry != [[], []]}]
        return hashlib.sha256(json.dumps(wire).encode()).hexdigest()[:16]

    return {name: digest(list(kernels)) for name, kernels in reads.items()}


def _import(db: SearchDB, ctx: Context, documents: list[GoldenFile], source: str) -> None:
    measured = wrong_regime = written = 0
    for document in documents:
        live = [row for row in document.rows if row.measured and regime_live(row.pins)]
        measured += sum(row.measured for row in document.rows)
        wrong_regime += sum(row.measured for row in document.rows) - len(live)
        written += import_rows(db, ctx, document, live, source=source)
    logger.info("golden evidence: %d perf row(s) imported into %s as %s", written, getattr(db, "_path", None) or "memory", source)
    if measured and not written:
        # The deploy would fall through to the prior with the files apparently loaded: say so.
        regime = f", {wrong_regime} recorded in another measurement regime" if wrong_regime else ""
        logger.warning(
            "golden scope holds %d measured row(s) but none is evidence on this card%s — the greedy will price every fork "
            "from the prior. Check the precision and cache input pins against the rows' recorded pins.",
            measured,
            regime,
        )
