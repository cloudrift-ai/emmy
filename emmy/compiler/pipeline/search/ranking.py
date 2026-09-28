"""Enumerate and rank candidates for the goldens: the candidate pools of a dataset DB's golden rows
(:func:`golden_pools`, :func:`build_golden_groups`), and one program-backed record's own enumeration
(:func:`evaluate_record`).

A golden pool is one kernel's schedule space on one card, in one precision regime, at one set of sizes,
together with the verified rows the golden files record in it. The dataset DB holds everything the pool needs
without lowering a golden's program: each kernel's re-lowerable definition (``kernel.loop_ir``, the loop body it
was formed from), its stamps, the card and regime (``context``), the sizes (``perf.bindings``) and the golden's
schedule row. The pool is enumerated from the definition through the tile
lowering passes, the way the tuner enumerates a kernel's slice, and each golden row is found in it by its
schedule row's structural signature (``features.tile_signature``).
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from emmy.compiler.context import Context
from emmy.compiler.pipeline.search import features
from emmy.compiler.pipeline.search.data.freeze import REGIME_PINS, freeze_reason, regime_of, schedule_row
from emmy.compiler.pipeline.search.data.group import DEFAULT_FEATURES, GoldenGroup, feature_view, pack_features
from emmy.compiler.pipeline.search.data.shape import ShapeKey
from emmy.compiler.pipeline.search.db import KernelRow, PerfRow, SearchDB, knobs_json
from emmy.compiler.pipeline.search.metrics import dual_rank
from emmy.compiler.pipeline.search.pool import Candidates, PoolSample

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Ranked:
    """Where one record's recorded config landed in its own candidate enumeration.

    ``rank`` / ``rank_optimistic`` are ``None`` when the recorded knobs are not in the enumeration
    at all — a pin or dtype mismatch, which is a real defect class and must stay distinguishable
    from "ranked last". ``pool`` is the size of the enumeration the rank is against, and travels
    with it because a rank alone says nothing."""

    best: dict
    rank: int | None
    pool: int
    rank_optimistic: int | None


@dataclass(frozen=True)
class GoldenPool:
    """One candidate pool the dataset DB records verified rows in: one kernel on one card, in one precision
    regime (``regime``: a key of :data:`~.data.freeze.REGIME_PINS`), at one set of sizes, and the golden rows
    measured on it. The pool is enumerated from the kernel's own definition (``kernel.loop_ir``); a kernel
    formed from no loop op (``kernel.formed`` false: a piece carved from a twisted tree, which only its parent's
    program reaches) has none, and its pool is skipped by name."""

    gpu: str
    cap: tuple[int, int]
    regime: str
    kernel: KernelRow
    bindings: dict
    rows: tuple[PerfRow, ...]

    @property
    def name(self) -> str:
        """The pool's label in a report: the kernel's C name and the head of its exact identity — the name a
        freeze gives its realizations — with the sizes when the kernel is symbolic."""
        sizes = " ".join(f"{var}={size}" for var, size in sorted(self.bindings.items()))
        return f"{self.kernel.name}.{self.kernel.exact_identity[:12]}" + (f" {sizes}" if sizes else "")

    @property
    def pins(self) -> dict:
        """The input pins the pool's rows were measured under."""
        return REGIME_PINS[self.regime]

    def schedule_rows(self) -> list[dict[str, str]]:
        """Each golden row's schedule row — its knobs without the stamps and the identity a read row carries."""
        return [schedule_row(row) for row in self.rows]

    @property
    def emmy_us(self) -> float:
        """The fastest golden time recorded in the pool."""
        return min(row.stats.median for row in self.rows)


def golden_pools(db: SearchDB, *, kernel: str | None = None) -> list[GoldenPool]:
    """The golden rows of ``db`` as pools: one per card, regime, kernel and sizes that holds at least one row a
    golden file sourced and the freeze admits (:func:`~.data.freeze.freeze_reason` — the one admission rule every
    measured-pool reader applies). The kernel is the exact one the rows were measured on: a golden pool has to be
    enumerated from a definition, which is why it does not key on the stamp signature the measured pools
    (``group_measured``) share across bodies. ``kernel`` keeps only pools whose kernel's C name contains it. In
    content order, so a report reads the same on every machine."""
    kernels = {k.exact_identity: k for k in db.iter_kernels()}
    buckets: dict[tuple, list[PerfRow]] = defaultdict(list)
    for row in db.iter_perf_rows(backend="cuda"):
        if row.source.startswith("golden:") and freeze_reason(row) is None:
            buckets[(row.gpu, divmod(row.cc, 10), regime_of(row.flags), row.kernel, knobs_json(row.bindings))].append(row)
    pools = []
    for (gpu, cap, regime, identity, _bindings), rows in sorted(buckets.items()):
        if kernel is None or kernel in kernels[identity].name:
            rows.sort(key=lambda r: knobs_json(r.knobs))
            pools.append(GoldenPool(gpu, cap, regime, kernels[identity], dict(rows[0].bindings), tuple(rows)))
    return pools


def kernel_program(kernel: KernelRow, bindings: dict):
    """The program a kernel's pool is enumerated from: its definition, the symbolic dims hinted at the sizes
    the rows were benched at (a binding would make them static — another kernel)."""
    from emmy.compiler.graph import Graph  # noqa: PLC0415
    from emmy.compiler.specialize import rehint_program  # noqa: PLC0415

    return rehint_program(Graph.from_wire(kernel.loop_ir), bindings)


def pool_context(pool: GoldenPool) -> Context:
    """The context ``pool``'s rows were measured under: the recording card's own facts and the regime's flags,
    whatever card and flags this process runs with."""
    return Context.from_target(pool.cap, gpu_name=pool.gpu, compile_flags=pool.regime)


def enumerate_graph(graph, ctx: Context, *, family: str = "", passes: Sequence[str] | None = None) -> Candidates:
    """The planner's candidate enumeration for any ``graph`` — the SAME rows the scheduler's fork
    tree offers a live compile, captured by resolving the graph through ``passes`` (the whole tile
    pipeline by default; the tile lowering alone for a kernel's definition, which the Loop passes would
    normalize into another kernel) with a decide that flattens each fork's leaves. Every leaf encodes one
    accepted classic ``Schedule`` with bare kernel keys and exact ``@n`` / ``@n.e`` sites, which is exactly
    what ``tile_signature`` joins a golden against. ``family`` keeps only rows carrying that knob
    family (``"TILE"`` for a contraction pool); ``""`` keeps every row with a per-node schedule
    knob (a reduce's ``REDUCE`` fork). The one live-fork capture the matmul
    offline fitter and record evaluator share.

    Returns :class:`~.pool.Candidates` — the rows beside the size of the pools they came from.
    Under ``ctx.pool_sample`` the rows are a DRAW and ``total`` is the exact size, and BOTH count
    the same population: distinct schedule-space stamps. Equal problems produce the same stamp,
    so their identical draw and total are collected once — a rank against ``total`` is a rank
    within the space the fit actually ranks. With no sample the rows are every kernel's fork rows
    and ``total`` is ``len(rows)``, so a caller that reports both prints today's numbers unchanged."""
    from emmy.compiler.pipeline import TILE_PASSES, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.fork import iter_leaves, leaf_knobs  # noqa: PLC0415
    from emmy.compiler.pipeline.knob import family_of  # noqa: PLC0415
    from emmy.compiler.pipeline.pipeline import Run  # noqa: PLC0415
    from emmy.compiler.pipeline.search.space import WORK  # noqa: PLC0415

    rows: list[dict] = []
    wanted = (family,) if family else ("TILE", "REDUCE", "STAGE")
    # The sample's size sink is PER CALL: the enumerator writes each pool's exact size there as it
    # goes (a fork tree has no channel to return it through), so it is cleared before the walk and
    # read straight after.
    sample = ctx.pool_sample if ctx is not None else None
    if sample is not None:
        sample.totals.clear()
    seen_pools: set[str] = set()

    def decide(fp):
        if sample is not None:
            # One contribution per schedule-space stamp, matching the totals sink's keyed dedupe:
            # an equal problem overwrites the same total, and appending its identical draw again
            # would make ``rows`` and ``total`` count different populations.
            opened = set(sample.totals) - seen_pools
            seen_pools.update(opened)
            if not opened:
                return _first(fp.options)
        for leaf in iter_leaves(fp.options):
            row = leaf_knobs(leaf)
            # A schedule row always spells the kernel-global ``WORK``; a structural arm's knob
            # delta (a cut, the cross-CTA split's g-half or the unsplit receipt) never does — the
            # stated row-identity marker (the classic scheduler's leaf boundary).
            if WORK.name not in row:
                continue
            if any(family_of(k) in wanted for k in row):
                rows.append(row)
        return _first(fp.options)

    def _first(options):
        # A pin may empty an early lazy branch while leaving a later sibling live. Walk to the
        # first complete leaf across the whole sibling set, matching the resolver's own traversal;
        # only an entirely empty fork means the schedule is not offered.
        option = next(iter_leaves(options), None)
        if option is None:
            from emmy.compiler.pipeline.pipeline import NO_OPTION  # noqa: PLC0415

            return NO_OPTION
        return option

    terminal, _ = Run(pipeline=Pipeline.build(list(passes) if passes is not None else TILE_PASSES), ctx=ctx).resolve(graph, decide)
    if sample is None:
        # A fully pinned classic problem can collapse without opening a policy-visible fork. Its
        # complete row still belongs to the enumeration: read it from the realized kernel set so
        # multi-kernel structural targets expose every independently scheduled problem.
        from emmy.compiler.ir.tile.ir import TileOp  # noqa: PLC0415
        from emmy.compiler.pipeline.knob import SCHEDULE_FAMILIES  # noqa: PLC0415

        for node in terminal.nodes.values():
            if not isinstance(node.op, TileOp) or WORK.name not in node.op.knobs:
                continue
            row = {key: value for key, value in node.op.knobs.items() if family_of(key) in SCHEDULE_FAMILIES}
            if any(family_of(key) in wanted for key in row) and row not in rows:
                rows.append(row)
    return Candidates(rows, sum(sample.totals.values()) if sample is not None else len(rows))


def enumerate_pool(pool: GoldenPool, ctx: Context) -> Candidates:
    """``pool``'s candidates: its kernel's definition at the pool's sizes through the tile lowering under the
    regime's pins alone — the wire is the kernel itself, so a live kernel-decision pin means nothing on it."""
    from emmy.compiler.pipeline import TILE_LOWERING  # noqa: PLC0415
    from emmy.compiler.pipeline.search.pins import pinned_knobs, unpinned_decisions  # noqa: PLC0415

    with pinned_knobs(pool.pins), unpinned_decisions():
        return enumerate_graph(kernel_program(pool.kernel, pool.bindings), ctx, passes=TILE_LOWERING)


def _shape_group(shape: ShapeKey) -> str:
    """A pool's cross-validation fold group: its ``ShapeKey`` with the two fields normalized away that
    separate pools sharing candidates.

    ``is_dyn`` because a symbolic kernel enumerates its static counterpart's candidates at the hint size — split
    them across folds and the fold model is scored on rows it trained on. ``is_warp`` because the fp16/fp32
    twins of one geometry are the same physical shape; their pools are disjoint, so this one is conservatism
    rather than necessity.

    ``kind`` and the extents stay: ``flash`` / ``softmax`` / ``rms_norm`` / ``fused`` are different kernels, not
    variants of one shape. The result is a string so it lands in the metrics file as-is."""
    return str(replace(shape, is_dyn=False, is_warp=False))


@dataclass
class _Packed:
    """One packed candidate pool and every golden found in it, before it becomes a :class:`GoldenGroup`."""

    pool: GoldenPool
    tier: str
    shape: str
    packed: tuple
    total: int
    goldens: list[int]


def _pool_identity(pool: GoldenPool, tier: str, shape: str, packed) -> tuple:
    """A candidate pool's identity: everything about it except which golden pinned which row.

    Two enumerations belong in one group when this matches — the featurized pool is then byte-identical, so
    one enumeration's row index addresses the same row the other's does, and the goldens behind them are
    several verified answers to one question. The identity fields ride along with the matrix digest because
    they decide things the matrix does not: the weight set (``dynamic``), the fold group (``shape``) and the
    report axes. Requiring them to agree can only hold two pools apart, never fuse two that differ."""
    names, matrix, dynamic = packed
    return (pool.gpu, tier, shape, dynamic, names, hashlib.blake2b(matrix, digest_size=16).digest())


def build_golden_groups(
    pools: Sequence[GoldenPool], features_spec: str = DEFAULT_FEATURES, *, sample: int = 0, seed: int = 0
) -> tuple[list[GoldenGroup], list[tuple[str, str, str]]]:
    """Enumerate each golden pool (:func:`golden_pools`), pin its golden rows, and featurize every candidate,
    as :class:`GoldenGroup` records (name, tier, card, pinned rows, per-row features filtered through the
    ``features_spec`` view; ``key`` is ``"<gpu>/<pool name>"``, suffixed ``#2``, ``#3``, … when one name
    opens several distinct pools). The second return is the golden rows that did NOT land in a group, as
    ``(gpu, name, reason)``, so metrics can count every recorded golden.

    **A group is a candidate pool, not a golden.** Several golden rows can land on one pool — a shape
    recorded under two names, or recorded twice — and they then share ONE group, each contributing a row to
    its golden set. Each pool is enumerated, featurized and packed once, then folded by :func:`_pool_identity`
    with any other pool that packed identically; every group is built knowing all of its goldens.

    A pool is enumerated under its OWN card's context and regime (:func:`pool_context`): the golden set spans
    cards that differ in compute capability AND in SM count at the same cap, so both the candidate enumeration
    (cp.async / TMA tiers gate on cap) and the ``H_*`` / ``D_*`` occupancy features must use the recording
    card's regime for the rank objective to match the deployed per-card featurization. The base features are
    the context's and the kernel's stamps as the DB holds them — nothing is lowered.

    ``sample`` draws that many candidates per pool DURING enumeration (0 enumerates every row). The draw is a
    reservoir over the schedule walk's leaf stream — a pure function of that stream and ``(sample, seed)`` —
    and every golden signature recorded on the pool's card and regime survives it whatever the draw picks, so
    a golden that misses its pool still means what it always meant: a pin or dtype mismatch.

    A narrowed ``pools`` (``golden_pools(db, kernel=…)``) is a VIEW, for iterating on one kernel without paying
    for the rest: the keep-set spans only the pools given, so a filtered run's retained rows, groups and positives
    are its own, and only an unfiltered run compares against a fit."""
    keep = feature_view(features_spec)
    groups: list[GoldenGroup] = []
    skipped: list[tuple[str, str, str]] = []
    key_counts: dict[str, int] = {}
    matched = 0
    # The keep-set spans every pool of a card and regime: the rows a sample may not drop are every golden
    # signature recorded there, so two pools that turn out to be one retain identical rows and merge.
    keeps: dict[tuple, set] = defaultdict(set)
    if sample > 0:
        for pool in pools:
            keeps[(pool.gpu, pool.regime)].update(features.tile_signature(row) for row in pool.schedule_rows())
    ctxs: dict[tuple, Context] = {}  # ONE Context per card and regime: the facts are identical across its pools
    packed_pools: dict[tuple, _Packed] = {}
    for pool in pools:
        if not pool.kernel.formed:
            skipped.extend((pool.gpu, pool.name, "kernel formed from no loop op") for _ in pool.rows)
            continue
        card = (pool.cap, pool.gpu, pool.regime)
        ctx = ctxs.get(card)
        if ctx is None:
            ctx = ctxs[card] = pool_context(pool)
        base = {**ctx.features(), **pool.kernel.stamps}
        # The sample rides a REPLACED Context; the pool stamp keys on the sample too, so a sampled
        # enumeration can never be mistaken for a live one.
        keep_set = frozenset(keeps.get((pool.gpu, pool.regime), ()))
        enum_ctx = ctx if sample <= 0 else replace(ctx, pool_sample=PoolSample(sample, seed, keep_set))
        try:
            candidates = enumerate_pool(pool, enum_ctx)
        except Exception as exc:  # noqa: BLE001 — a definition the enumeration cannot lower is no pool; the rows are counted
            logger.info("  !! %s: did not lower — %s: %s", pool.name, type(exc).__name__, exc)
            skipped.extend((pool.gpu, pool.name, f"did not lower: {type(exc).__name__}") for _ in pool.rows)
            continue
        rows = candidates.rows
        if not rows:
            logger.info("  !! %s: nothing enumerated — skipping", pool.name)
            skipped.extend((pool.gpu, pool.name, "nothing enumerated") for _ in pool.rows)
            continue
        # Each golden row locates itself in the pool by schema-agnostic structural signature (free-axis slots +
        # reduce decomp + atom): the candidate rows use the native ``MOVE@element`` keys while a golden may
        # record legacy GEMM-letter keys, so comparing key-value tuples directly never matches.
        goldens = []
        for row in pool.schedule_rows():
            want = features.tile_signature(row)
            gidx = next((i for i, r in enumerate(rows) if features.tile_signature(r) == want), None)
            if gidx is None:
                logger.info("  !! %s: golden not in %d candidates — skipping", pool.name, len(rows))
                skipped.append((pool.gpu, pool.name, f"golden not in {len(rows)} candidates"))
            else:
                goldens.append(gidx)
        if not goldens:
            continue
        matched += len(goldens)
        shape = ShapeKey.from_s_features(pool.kernel.stamps)
        tier = "dyn" if shape.is_dyn else (shape.kind or ("warp" if shape.is_warp else "thread"))
        # The feature view (default ``DEFAULT_FEATURES``: ``D_*`` geometry/occupancy plus ``MMA_tier`` — see
        # its rationale in ``search/data/group.py``) filters here, before the pool is packed, so the
        # trained-under view is exactly what the Group stores. ``feature_view`` keeps the routing features
        # whatever the spec says, so a narrower ``--features`` cannot silently misroute a symbolic-axis pool.
        feats = [{k: v for k, v in features.knob_features({**base, **r}).items() if keep(k)} for r in rows]
        packed = pack_features(feats)
        # Two pools can still pack identically — the same kernel recorded at two sizes it does not depend on.
        # Fold those together, so a pool is one group however many times it was recorded.
        identity = _pool_identity(pool, tier, _shape_group(shape), packed)
        found = packed_pools.get(identity)
        if found is None:
            packed_pools[identity] = _Packed(pool, tier, _shape_group(shape), packed, candidates.total, goldens)
        else:
            found.goldens.extend(goldens)

    # Every pool now knows every golden in it, so each becomes ONE group whose labels are final at
    # construction. The ``#N`` suffix keeps ``Group.key`` unique (``cv.run_folds`` keys its train accumulator
    # on it) and is spent per POOL in first-appearance order, so goldens that merged never claim one.
    for entry in packed_pools.values():
        key_str = f"{entry.pool.gpu}/{entry.pool.name}"
        key_counts[key_str] = n = key_counts.get(key_str, 0) + 1
        groups.append(
            GoldenGroup.over(
                key_str if n == 1 else f"{key_str}#{n}",
                entry.pool.name,
                entry.tier,
                entry.pool.gpu,
                entry.shape,
                entry.packed,
                entry.goldens,
                entry.total,
            )
        )
    logger.info(
        "  %d matched goldens over %d candidate pools (%d beyond one golden per pool)",
        matched,
        len(groups),
        matched - len(groups),
    )
    return groups, skipped


def evaluate_record(record, ctx: Context, scorer: Callable[[dict], float] | None = None) -> Ranked:
    """Rank a generic program-backed record in its current candidate enumeration."""
    from emmy.compiler.pipeline.search.pins import pinned_knobs  # noqa: PLC0415
    from emmy.compiler.pipeline.search.prior import OfflinePrior  # noqa: PLC0415

    with pinned_knobs(record.pin_map):
        candidates = enumerate_graph(record.target_program.copy(), ctx)
    rows = candidates.rows
    if not rows:
        return Ranked({}, None, 0, None)
    if scorer is None:
        prior = OfflinePrior()
        base = {**ctx.features(), **record.structural_features}

        def scorer(row):
            return -prior.mean_score({**base, **row})

    want = features.tile_signature(record.knobs) if record.knobs else None
    golden_index = next((i for i, row in enumerate(rows) if features.tile_signature(row) == want), None) if want else None
    scores = [scorer(row) for row in rows]
    best = max(range(len(rows)), key=scores.__getitem__)
    rank, rank_opt = dual_rank(scores, golden_index) if golden_index is not None else (None, None)
    return Ranked(rows[best], rank, candidates.total, rank_opt)
