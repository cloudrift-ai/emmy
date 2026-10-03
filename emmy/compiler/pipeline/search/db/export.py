"""``emmy db export`` — a DB instance's rows as a :class:`~..dataset.Dataset`: its golden pools, each enumerated from
the kernel's definition and packed into a training group (``ranking.build_golden_groups``), its measured pools
labelled with their microseconds, and the provenance a reader needs to know what it holds.

This is the one place the DB's rows become the dataset's groups, and the one direction the two packages meet: the
export reads ``db`` and builds ``dataset`` values, and nothing under ``dataset/`` reads a DB.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import replace

from emmy.compiler.pipeline.search.dataset import Dataset, GoldenPool, GoldenRow, MeasuredGroup, measured_features, regime_of, repo_commit
from emmy.compiler.pipeline.search.db import PerfRow, SearchDB, knobs_json
from emmy.compiler.pipeline.search.db.freeze import freeze_reason, schedule_row
from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION, knob_features
from emmy.compiler.pipeline.search.ranking import build_golden_groups, build_placement_groups
from emmy.compiler.structural import digest

logger = logging.getLogger(__name__)


def golden_pools(db: SearchDB) -> tuple[list[GoldenPool], dict[str, int]]:
    """The golden rows of ``db`` as pools — one per card, regime, kernel and sizes that holds at least one row a
    golden file sourced (``golden:`` — a repository golden the import filed, or the golden scope a compile imported
    into a tune DB) and the freeze admits (:func:`~.freeze.freeze_reason`, the one admission rule every measured-pool
    reader applies) — beside a count of the golden rows dropped, by reason, as :func:`measured_groups` returns its
    own. The kernel is the exact one the rows were measured on: a golden pool has to be enumerated from a
    definition, which is why it does not key on the stamp signature the measured pools share across bodies. In
    content order, so a report reads the same on every machine."""
    kernels = {k.exact_identity: k for k in db.iter_kernels()}
    dropped: dict[str, int] = defaultdict(int)
    buckets: dict[tuple, list[PerfRow]] = defaultdict(list)
    for row in db.iter_perf_rows(backend="cuda"):
        if not row.source.startswith("golden:"):
            continue
        reason = freeze_reason(row)
        if reason is not None:
            dropped[reason.split(":")[0]] += 1
        else:
            buckets[(row.gpu, divmod(row.cc, 10), regime_of(row.flags), row.kernel, knobs_json(row.bindings))].append(row)
    pools = []
    for (gpu, cap, regime, identity, _bindings), rows in sorted(buckets.items()):
        rows.sort(key=lambda r: knobs_json(r.knobs))
        golden = tuple(GoldenRow(schedule_row(row), row.stats.median, row.source) for row in rows)
        pools.append(GoldenPool(gpu, cap, regime, kernels[identity], dict(rows[0].bindings), golden))
    return pools, dict(dropped)


def placement_pools(db: SearchDB, pools: list[GoldenPool]) -> list[GoldenPool]:
    """The kernels whose placement forks the placement space ranks, each a pool whose one row is the ``PLACE``
    decision a golden took on it, or no row where the golden kept it fused: every kernel a golden cut, on the card
    and at the sizes it cut it (``taken``; of two cuts of one kernel there, the one over more seams), then every
    kernel a golden split and every golden pool (a piece a cut minted included — it is a kernel with forks of its
    own, walked from its own definition), which it kept one kernel. Nothing is priced: what the golden did is the
    label."""
    kernels = {k.exact_identity: k for k in db.iter_kernels()}
    out: dict[tuple, GoldenPool] = {}

    def add(pool: GoldenPool) -> None:
        out.setdefault((pool.kernel.exact_identity, pool.gpu, pool.regime, knobs_json(pool.bindings)), pool)

    taken = sorted((decision for decision in db.iter_taken() if decision[-1].startswith("golden:")), key=lambda decision: -len(decision[5]))
    for cut in (True, False):
        for gpu, cc, flags, kernel, bindings, arm, source in taken:
            if all(key.startswith("PLACE") for key in arm) == cut:
                row = GoldenRow({k: str(v) for k, v in arm.items()}, math.nan, source)  # a mark: a decision has no time
                add(GoldenPool(gpu, divmod(cc, 10), regime_of(flags), kernels[kernel], bindings, (row,) if cut else ()))
    for pool in pools:
        add(replace(pool, rows=()))
    return list(out.values())


def kernel_sig(feats: dict) -> str:
    """The op signature of the kernel a row measured, digested from its own ``S_*`` stamps — exactly what
    :meth:`~...passes.identity.Identity.op_sig` computes for an op, applied to the row's recorded stamps."""
    return digest(*sorted((k, float(v)) for k, v in feats.items() if k.startswith("S_")))


def measured_groups(rows) -> tuple[list[MeasuredGroup], dict[str, int]]:
    """Measured ``perf`` rows (:class:`~..db.PerfRow`) as groups labelled with measured µs, keyed
    ``(gpu, kernel_sig, opt, flags)`` — one group per set of configs that genuinely competed, plus a count
    of what was dropped and why.

    Each part of the key is load-bearing, and each has a plausible wrong answer:

    - **The kernel's own structural signature** (:func:`kernel_sig`). Two kernels of the same structure
      on the same card are ONE tuning problem whatever produced them — which is already how the deploy
      path joins evidence (``policy/greedy._db_measured_pick`` indexes on
      the ``S_*`` signature), so this makes the candidate pools agree with the tier that consumes them.
      Keying on where a decision was OFFERED instead gets it wrong in both directions: a site realized
      as several kernels files a piece beside the whole (the RTX 5090 freeze once paired a 5.9 µs norm
      kernel with a 131 ms whole-op row), and one kernel reached from two sites is tuned twice.
    - **The opt level and the precision regime.** The regimes must not pool — ``-O1`` and ``-O3`` invert
      often enough that a merged group measures neither, and fast math changes the code a kernel runs as.
      No other compiler flag is a regime.
    - **``gpu``.** Cards never pool.

    Admission is :func:`~.freeze.freeze_reason`, the rule a freeze is written under, failures excluded
    among the rest: the watchdog sentinel is a huge positive that any model gets right for free and that
    inflates every correlation over the pool.

    The counts are returned rather than logged because a report has to publish them: "Spearman 0.6 over
    340 groups" means something different when 143 rows were dropped than when none were. Reasons are
    keyed by their leading clause, the same normalization ``write_freeze`` counts its own drops by, so
    the two publish one vocabulary."""
    dropped: dict[str, int] = defaultdict(int)
    buckets: dict[tuple, list] = defaultdict(list)
    for r in rows:
        reason = freeze_reason(r)
        if reason is not None:
            dropped[reason.split(":")[0]] += 1
        else:
            buckets[(r.gpu, kernel_sig(r.knobs), float(r.opt), regime_of(r.flags))].append(r)

    groups = []
    for (gpu, sig, h_opt, regime), grp in sorted(buckets.items()):
        grp.sort(key=lambda r: (r.kernel, knobs_json(r.knobs)))  # a pool's row order is its own, not the DB's
        feats = [knob_features(measured_features(r)) for r in grp]
        key = f"{gpu}/{sig}@O{h_opt:g}" + (f" {regime}" if regime else "")
        groups.append(MeasuredGroup.from_measured(key, gpu, sig, h_opt, [r.stats.median for r in grp], feats))
    return groups, dict(dropped)


def export_dataset(db: SearchDB, *, source: str, pool_sample: int, seed: int, space: str = "schedule", jobs: int = 1) -> Dataset:
    """Every row of ``db`` as a dataset of one ``space``. The schedule space: the golden pools enumerated under
    their own card's context and packed (``sample`` candidates drawn per pool during enumeration, 0 for every
    row), and the measured pools. The placement space: each kernel's placement forks, the arms featurized and
    the golden's marked (:func:`placement_pools`), and no measured pools. Both carry the provenance — ``source``
    names the instance, the rest is what the rows and this checkout say. ``jobs`` worker processes enumerate the
    schedule space's pools side by side — the export's whole cost — one by default."""
    pools, dropped_golden = golden_pools(db)
    if space == "placement":
        pools = placement_pools(db, pools)
        logger.info("Walking the placement forks of %d kernels (%d decisions) ...", len(pools), sum(bool(p.rows) for p in pools))
        golden, skipped = build_placement_groups(pools)
        measured, dropped = [], {"golden": dropped_golden, "measured": {}}
    else:
        logger.info("Building %d golden pools (each under its own card's context, %d at a time) ...", len(pools), jobs)
        golden, skipped = build_golden_groups(pools, "*", sample=pool_sample, seed=seed, jobs=jobs)
        measured, dropped_measured = measured_groups(db.iter_perf_rows(backend="cuda"))
        dropped = {"golden": dropped_golden, "measured": dropped_measured}
    provenance = {
        "source": source,
        "space": space,
        "sources": dict(sorted(db.perf_sources().items())),
        "pool_sample": pool_sample,
        "seed": seed,
        "feat_ver": FEATURIZER_VERSION,
        "compiler": repo_commit(),
    }
    return Dataset(golden, measured, skipped, dropped, provenance)
