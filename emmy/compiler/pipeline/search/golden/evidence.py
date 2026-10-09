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
live precision regime are evidence there (:func:`regime_live`). :func:`import_file` is ``emmy db import``'s: every
row under its own regime's context.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from emmy import config
from emmy.compiler.context import FAST_MATH_FLAG, Context
from emmy.compiler.pipeline.knob import KnobType, family_of, registry
from emmy.compiler.pipeline.search.bench_record import point_stats
from emmy.compiler.pipeline.search.db import RoutingRow, SearchDB
from emmy.compiler.pipeline.search.db.freeze import is_lfs_pointer
from emmy.compiler.wire import symbolic_vars

from .format import GoldenFile, Row
from .repository import documents_for_card, scope_digest, scope_explicit

logger = logging.getLogger("emmy.compiler.pipeline")


def regime_live(pins: dict) -> bool:
    """Whether a row's input-pin regime IS the live one — exact per pin: a BOOL pin against its effective precision
    policy (other BOOLs default off), anything else against the raw env string. Strict both ways: a row measured
    under ``FAST_MATH`` is no evidence for a standard deploy, and a standard row none under a live precision pin —
    the precision knobs are compared even where the row omits them (omitted = measured OFF)."""
    from emmy.compiler.pipeline.search.space import PRECISION_KNOBS, precision_pin  # noqa: PLC0415

    precision = {knob.name for knob in PRECISION_KNOBS}
    knobs = registry()
    for name, value in {"COLD_CACHE": False, **pins}.items():
        if family_of(str(name)) == "PLACE":
            continue
        kn = knobs.get(str(name))
        raw = kn.raw() if kn is not None else config.knob_raw(str(name))
        if kn is not None and kn.type is KnobType.BOOL:
            live = precision_pin(kn) if name in precision else kn.parse(raw) if raw is not None else False
            if bool(value) != live:
                return False
        elif (raw or "") != str(value):
            return False
    umbrella = bool(pins.get("FAST_MATH", False))
    for name in precision:
        kn = knobs.get(name)
        if bool(pins.get(name, umbrella)) != (bool(precision_pin(kn)) if kn is not None else False):
            return False
    return True


def regime_context(document: GoldenFile, pins: dict) -> Context:
    """The context a row measured under ``pins`` on ``document``'s card is filed under: the card, and the one
    arithmetic flags and cache mode."""
    return Context.from_target(
        tuple(document.compute_cap),
        gpu_name=document.gpu_name or None,
        compile_flags=FAST_MATH_FLAG if pins.get("FAST_MATH") else "",
        cold_cache=bool(pins.get("COLD_CACHE", False)),
    )


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
            backend="cuda",
            status="ok",
            stats=point_stats(row.measurements.emmy_us),
            captured=True,
            source=source,
        )
        written += 1
    return written


def file_source(kind: str, path: Path | str) -> str:
    """The ``source`` an import files a file's rows under: what kind of file it is (``freeze`` or ``golden``) and the
    digest of its own bytes — a re-recorded file is another source, and a dataset holds a file once."""
    return f"{kind}:{hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]}"


def import_file(db: SearchDB, path: Path, source: str) -> Counter:
    """Import one golden file into ``db``: every row under the context of its own regime, filed under ``source``
    (:func:`file_source`), which the instance then holds (``SearchDB.record_source``) whatever became of the rows. A
    source the instance already holds is skipped; a git-LFS pointer in the data's place is refused by name.
    Returns what was written, by kind."""
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


def _import(db: SearchDB, ctx: Context, documents: list[GoldenFile], source: str) -> None:
    measured = wrong_regime = written = 0
    for document in documents:
        live = [
            row
            for row in document.rows
            if row.measured and bool(row.pins.get("COLD_CACHE", False)) == ctx.cold_cache and regime_live(row.pins)
        ]
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
