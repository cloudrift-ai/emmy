"""Golden records as tune DB rows — what a compile imports before it picks.

A golden file is the durable record of a measurement: the program, the pins it was measured under, the knob
row and the microseconds. The tune DB is the cache a compile reads its evidence from, and a golden's rows
reach it the way a tune's do — through the writers the tuner uses — so the greedy pick has one read
(``policy.greedy``): a kernel's measured schedule rows (``perf``) and the kernel-set decisions stored on it
(``routing``), priced from the pieces' own rows.

:func:`import_goldens` lowers each set of entries — one target's entries in one input regime, the ones that
walk one kernel set together (``golden.siblings_of``) — once, every entry deciding the forks of the kernel it
names the way the deploy reads a row (``pins.spelled_arm`` at a kernel-set fork, its schedule row at a schedule
fork) and the set's leading entry every other, and files each measured entry's row under the kernel it
decorates: a plain entry under the target's one kernel, a child-identity receipt under the kernel its stored
identity names, as the schedule row it recorded. A routing entry is its decision, which the lowering's splice
records as a routing row. An entry whose row spells a cross-CTA split over a kernel set it timed as a whole is
a routing row and nothing more: the DB holds measurements of kernels, and the set's time is not a piece's —
the split arm is priced once the pieces are benched (``run --golden PATH --bench --record-greedy``).

:func:`evidence_db` is the compile's seam: the golden rows in scope (``golden.records_for_card``) are imported
once per golden digest into the tune DB the compile reads — a re-recorded file changes the digest, and the
file's earlier rows are let go first — or into a fresh in-memory instance when the compile has no DB.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

from emmy.compiler.context import FAST_MATH_FLAG, Context
from emmy.compiler.ir.cuda.ir import CudaOp
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import CUDA_PASSES, LOWERING_PASSES, Pipeline
from emmy.compiler.pipeline.fork import iter_leaves, leaf_for
from emmy.compiler.pipeline.knob import family_of
from emmy.compiler.pipeline.pipeline import Run, _is_structural_option
from emmy.compiler.pipeline.search.data.freeze import freeze_source, is_lfs_pointer
from emmy.compiler.pipeline.search.db import SearchDB, is_placement_knob
from emmy.compiler.pipeline.search.pins import composed_routes, pinned_knobs, regime_live, spelled_arm, unpinned_decisions
from emmy.compiler.pipeline.search.policy.terminal_bench import persist_kernel_perf, point_stats
from emmy.compiler.wire import kernel_tile

from .decode import _set_key, piece_row
from .format import GoldenFile
from .record import kernel_set_pins, regime_pins
from .repository import records_for_card, scope_digest, scope_explicit

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from emmy.compiler.context import Context
    from emmy.compiler.pipeline.search.golden import GoldenRecord

logger = logging.getLogger("emmy.compiler.pipeline")


def import_goldens(
    db: SearchDB, ctx: Context, records: Sequence[GoldenRecord], *, source: str, passes: Sequence[str] | None = None
) -> Counter:
    """Write ``records``' measurements into ``db`` under ``ctx``'s card and regime, ``source`` on every perf
    row. Only a measured entry in the live input regime (``golden.regime_live``) is evidence. Returns what
    became of the entries, by kind. A set the current compiler cannot lower is skipped: the strict decode is
    where that is loud. ``passes`` is the pipeline a target enters: the whole of it for a golden's traced
    slice (the default), the lowering passes alone for a freeze's kernel body, which the Loop passes would
    normalize into another kernel."""
    # the strategy package imports this module's evidence_db: a real cycle, so the import stays local
    from emmy.compiler.pipeline.search.strategy.two_level import KernelInventory, record_routing  # noqa: PLC0415

    counts: Counter[str] = Counter()
    consumed: set[str] = set()  # the deploy identities of the kernels a decision replaced: they ran as no kernel

    def on_routing(parent, arm, pieces, _ids) -> None:
        record_routing(db, parent, arm, pieces)
        consumed.add(parent.identity_key(with_io=True))
        counts["routing rows"] += 1

    pipeline = Pipeline.build(list(passes) if passes is not None else CUDA_PASSES).with_strategies(KernelInventory(on_routing=on_routing))
    sets: dict[tuple, list[GoldenRecord]] = {}
    for record in records:
        sets.setdefault(_set_key(record), []).append(record)
    for entries in sets.values():
        measured = [entry for entry in entries if entry.measurements is not None and entry.emmy_us > 0]
        if not measured or not regime_live(entries[0]):
            counts["unmeasured or in another regime"] += len(entries)
            continue
        try:
            graph = _lower(pipeline, ctx, entries)
        except Exception:  # noqa: BLE001 — a set the current compiler cannot lower is no evidence
            logger.debug("golden import: %s did not lower", entries[0].name, exc_info=True)
            counts["did not lower"] += len(entries)
            continue
        ops = [op for nid in graph.topological_order() if isinstance(op := graph.nodes[nid].op, CudaOp)]
        kernels = [(op, kernel_tile(op)) for op in ops]
        for entry in measured:
            if entry.is_routing:
                continue
            row = entry.schedule_row
            if any(is_placement_knob(key, value) for key, value in row.items()):
                counts["kernel sets timed as a whole"] += 1
                continue
            if entry.identity is not None:
                op = next((op for op, tile in kernels if tile is not None and tile.identity_key(with_io=True) == entry.identity), None)
                what = "kernels a decision replaced" if entry.identity in consumed else "identities no kernel carries"
            else:
                op = kernels[0][0] if len(kernels) == 1 else None
                what = "multi-kernel targets without receipts"
            if op is None:
                counts[what] += 1
                continue
            stats = point_stats(entry.emmy_us)
            if persist_kernel_perf(db, ctx, "cuda", op, stats=stats, status="ok", captured=True, knobs=row, source=source):
                counts["perf rows"] += 1
    return counts


def _lower(pipeline, ctx: Context, entries: list[GoldenRecord]):
    """The set's target through ``pipeline`` under ``ctx``: an entry naming a kernel by identity decides that
    kernel's forks — of two naming one kernel, the one spelling a route decides its cut — and the leading
    entry every other; a kernel-set fork takes the arm the decider's route spells, a schedule fork the leaf
    its row vouches for, either the first leaf when it spells none. The live decision pins are withdrawn: the
    rows filed hold for every pinned compile. The seams an entry marks cut together are one composed
    decision, offered to the cut pass as the deploy offers them."""

    lead = entries[0]
    spelling = {
        id(entry): {**kernel_set_pins(entry, entries), **entry.route, **{str(k): str(v) for k, v in entry.knobs.items()}}
        for entry in entries
    }
    named = {entry.identity: entry for entry in sorted(entries, key=lambda entry: bool(entry.route)) if entry.identity is not None}
    composed: list[tuple[None, tuple[str, ...]]] = []
    for row in spelling.values():
        keys = tuple(sorted(key for key, value in row.items() if family_of(key) == "PLACE" and value == "cut"))
        if len(keys) > 1 and (None, keys) not in composed:
            composed.append((None, keys))

    def decide(fp):
        identity = fp.root_op.identity_key(with_io=True) if isinstance(fp.root_op, TileOp) else None
        decider = named.get(identity, lead)
        row = spelling[id(decider)]
        if fp.structural:
            arm = spelled_arm(fp.options, row)
            if arm is not None:
                option, knobs = arm
                if _is_structural_option(option):
                    # A decision consumes the keys that spelled it (a bare ``PLACE=cut`` its one root-most
                    # cut), so the pieces are read against what the entry has left to say.
                    for key in (*(set(knobs) & set(row)), *(("PLACE",) if row.get("PLACE") == "cut" else ())):
                        row.pop(key, None)
                return option
        elif (asked := piece_row(decider.schedule_row)) and (hit := leaf_for(fp.options, asked)) is not None:
            return hit[0]
        return next(iter_leaves(fp.options))

    with unpinned_decisions(), composed_routes(composed):
        graph, _trace = Run(pipeline=pipeline, ctx=ctx).resolve(lead.target_program.copy(), decide)
    return graph


def import_file(db: SearchDB, path: Path) -> Counter:
    """Import one golden-shaped file — a freeze's card file, or a golden file — into ``db``: every kernel
    re-lowered from its definition through the lowering passes alone (a stored kernel body must not meet the
    Loop passes, which would normalize it into another kernel), once per regime the file's rows record, its rows
    sourced by the file's digest (``freeze.freeze_source``). The rows were measured at the deployable opt level
    under their regime's flags, whatever this machine compiles at. A file the instance already holds is skipped;
    a git-LFS pointer in the data's place is refused by name. Returns what became of the entries, by kind."""

    if is_lfs_pointer(path):
        raise ValueError(f"{path} is a git-LFS pointer, not the data: run `git lfs install && git lfs pull` (in CI, check out with lfs)")
    source = freeze_source(path)
    if source in db.perf_sources():
        logger.info("%s is already held (%s)", path.name, source)
        return Counter()
    document = GoldenFile.load(path)
    records = document.records()
    cap, gpu_name = tuple(document.compute_cap), document.gpu_name
    counts: Counter = Counter()
    for regime in sorted({tuple(sorted(regime_pins(record).items())) for record in records}):
        with pinned_knobs(dict(regime)):
            ctx = Context.from_target(cap, gpu_name=gpu_name, compile_flags=FAST_MATH_FLAG if dict(regime).get("FAST_MATH") else "")
            in_regime = [record for record in records if tuple(sorted(regime_pins(record).items())) == regime]
            counts += import_goldens(db, ctx, in_regime, source=source, passes=LOWERING_PASSES)
    logger.info("imported %s as %s: %s", path.name, source, ", ".join(f"{n} {what}" for what, n in sorted(counts.items())) or "nothing")
    return counts


def evidence_db(db: SearchDB | None, ctx: Context) -> SearchDB:
    """The DB a compile under ``ctx`` picks from, holding the golden rows in scope: ``db`` itself, the scope
    imported into it once per golden digest (a scope the file does not hold yet lets the earlier golden rows of
    this card and regime go first); with no ``db``, a fresh in-memory instance holding the scope."""

    gpu_name = getattr(ctx, "gpu_name", None) or ""
    records = records_for_card(gpu_name, tuple(ctx.compute_capability)) if gpu_name or scope_explicit() else []
    if not records:
        return db if db is not None else SearchDB()
    source = f"golden:{scope_digest(gpu_name)[:12]}"
    if db is None:
        fresh = SearchDB()
        _import(fresh, ctx, records, source)
        return fresh
    if source not in db.perf_sources(ctx):
        db.forget_perf(ctx, "golden:")
        _import(db, ctx, records, source)
    return db


def _import(db: SearchDB, ctx: Context, records: Sequence[GoldenRecord], source: str) -> None:

    counts = import_goldens(db, ctx, records, source=source)
    logger.info(
        "golden evidence: %d record(s) imported into %s as %s: %s",
        len(records),
        getattr(db, "_path", None) or "memory",
        source,
        ", ".join(f"{n} {what}" for what, n in sorted(counts.items())) or "nothing",
    )
    measured = [record for record in records if record.measurements is not None and record.emmy_us > 0]
    wrong_regime = sum(not regime_live(record) for record in measured)
    _warn_unused_evidence(len(measured), wrong_regime, counts["perf rows"] + counts["routing rows"])


def _warn_unused_evidence(measured: int, wrong_regime: int, kept: int) -> None:
    """Say so when measured rows are in scope and NONE of them became a row.

    A row can be in scope and still price nothing: recorded in another precision regime
    (``golden.regime_live`` — ``FAST_MATH`` has been enabled by default since #868, so a row recorded
    at ``FAST_MATH: false`` is evidence only under ``EMMY_FAST_MATH=0``), or carrying an identity
    the lowering no longer mints. Either way the deploy falls through to the prior with the file
    apparently loaded, which is the failure this warning exists to make visible: it cost a
    golden-bench corpus its whole recorded schedule set without a single line of output."""
    if kept or not measured:
        return
    regime = f", {wrong_regime} recorded in another precision regime" if wrong_regime else ""
    logger.warning(
        "golden scope holds %d measured row(s) but none became a row on this card%s — the greedy will price every "
        "fork from the prior. Check EMMY_FAST_MATH against the rows' recorded pins, then re-record what stays unused.",
        measured,
        regime,
    )
