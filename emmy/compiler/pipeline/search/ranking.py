"""Enumerate and rank candidates for the goldens: the DB's golden pools (``db/export.golden_pools``)
as training groups (:func:`build_golden_groups`), and one program-backed record's own enumeration


A golden pool is one kernel's schedule space on one card, in one precision regime, at one set of sizes,
together with the verified rows the golden files record in it. The dataset DB holds everything the pool needs
without lowering a golden's program: each kernel's re-lowerable definition (``kernel.loop_ir``, the loop body it
was formed from), the card and regime (``context``), the sizes (``perf.bindings``) and the golden's
schedule row. The pool is enumerated from the definition through the tile
lowering passes, the way the tuner enumerates a kernel's slice, and each golden row is found in it by its
schedule row's structural signature (``features.tile_signature``).
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from multiprocessing import get_context

from emmy.compiler.context import Context
from emmy.compiler.pipeline.search import features
from emmy.compiler.pipeline.search.dataset.group import GoldenGroup, feature_view, pack_features
from emmy.compiler.pipeline.search.dataset.pool import GoldenPool
from emmy.compiler.pipeline.search.dataset.shape import ShapeKey
from emmy.compiler.pipeline.search.pins import pinned_knobs
from emmy.compiler.pipeline.search.pool import Candidates, PoolSample

logger = logging.getLogger(__name__)


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

    Returns :class:`~.pool.Candidates` — the rows beside what they were ranked among. Under
    ``ctx.pool_sample`` the rows are a DRAW (seeded descents through each schedule tree, the golden rows
    kept — ``PoolSample.draw``) and ``total`` is the draw's size; BOTH count the same population: distinct
    schedule-space stamps. Equal problems produce the same stamp, so their identical draw and total are
    collected once. With no sample the rows are every kernel's fork rows and ``total`` is ``len(rows)``."""
    from emmy.compiler.pipeline import TILE_PASSES, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.fork import leaf_knobs  # noqa: PLC0415
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
                return _first(fp)
        for leaf in fp.leaves():
            row = leaf_knobs(leaf)
            # A schedule row always spells the kernel-global ``WORK``; a structural arm's knob
            # delta (a cut, the cross-CTA split's g-half or the unsplit receipt) never does — the
            # stated row-identity marker (the classic scheduler's leaf boundary).
            if WORK.name not in row:
                continue
            if any(family_of(k) in wanted for k in row):
                rows.append(row)
        return _first(fp)

    def _first(fp):
        # A pin may empty an early lazy branch while leaving a later sibling live. Walk to the
        # first complete leaf across the whole sibling set, matching the resolver's own traversal;
        # only an entirely empty fork means the schedule is not offered.
        option = next(fp.leaves(), None)
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
    from emmy.compiler.pipeline.search.pins import unpinned_decisions  # noqa: PLC0415

    with pinned_knobs(pool.pins), unpinned_decisions():
        return enumerate_graph(pool.kernel.program(pool.bindings), ctx, passes=TILE_LOWERING)


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
    pools: list[GoldenPool]


def _pool_identity(gpu: str, tier: str, shape: str, packed) -> tuple:
    """A candidate pool's identity: everything about it except which golden pinned which row.

    Two enumerations belong in one group when this matches — the featurized pool is then byte-identical, so
    one enumeration's row index addresses the same row the other's does, and the goldens behind them are
    several verified answers to one question. The identity fields ride along with the matrix digest because
    they decide things the matrix does not: the regime (``dynamic``), the fold group (``shape``) and the
    report axes. Requiring them to agree can only hold two pools apart, never fuse two that differ."""
    names, matrix, dynamic = packed
    return (gpu, tier, shape, dynamic, names, hashlib.blake2b(matrix, digest_size=16).digest())


def _enumerate_packed(task: tuple) -> tuple[_Packed | None, list[tuple[str, str, str]], str | None]:
    """One pool's share of :func:`build_golden_groups` — the work one worker process does: enumerate the pool
    under its own card's context, locate its golden rows among the candidates, featurize and pack them. ``task``
    is the pool, the rows its draw may not drop, the draw's size and seed, and the feature-view spec. Returns the
    packed pool (``None`` when it opened no group), the golden rows that landed in no group as ``(gpu, name,
    reason)``, and the line to log for a pool that was skipped — logged by the caller, so the lines keep the
    pools' order whichever process did the work."""
    pool, keep_rows, sample, seed, features_spec = task
    keep = feature_view(features_spec)
    ctx = pool_context(pool)
    featurizer = features.Featurizer.of(ctx)
    # The sample rides a REPLACED Context; the pool stamp keys on the sample too, so a sampled
    # enumeration can never be mistaken for a live one.
    enum_ctx = ctx if sample <= 0 else replace(ctx, pool_sample=PoolSample(sample, seed, keep_rows))
    try:
        candidates = enumerate_pool(pool, enum_ctx)
    except ValueError as exc:
        # A definition the lowering does not take back — the reduce piece of a cross-CTA split re-offers the
        # split and mints the buffer it already holds — or sizes it cannot bind. The rows are counted, loudly.
        return None, [(pool.gpu, pool.name, "did not lower") for _ in pool.rows], f"did not lower — {exc}"
    rows = candidates.rows
    if not rows:
        return None, [(pool.gpu, pool.name, "nothing enumerated") for _ in pool.rows], "nothing enumerated"
    # Each golden row locates itself in the pool by schema-agnostic structural signature (free-axis slots +
    # reduce decomp + atom): the candidate rows use the native ``MOVE@element`` keys while a golden may
    # record legacy GEMM-letter keys, so comparing key-value tuples directly never matches.
    goldens, skipped = [], []
    for row in pool.schedule_rows():
        want = features.tile_signature(row)
        gidx = next((i for i, r in enumerate(rows) if features.tile_signature(r) == want), None)
        if gidx is None:
            skipped.append((pool.gpu, pool.name, f"golden not in {len(rows)} candidates"))
        else:
            goldens.append(gidx)
    if not goldens:
        return None, skipped, f"golden not in {len(rows)} candidates"
    kernel = pool.kernel.op(pool.bindings)
    shape = ShapeKey.from_s_features(features.stamps(kernel))
    tier = "dyn" if shape.is_dyn else (shape.kind or ("warp" if shape.is_warp else "thread"))
    # The feature view (default every feature) filters here, before the pool is packed, so the view is
    # exactly what the Group stores. ``feature_view`` keeps the routing features
    # whatever the spec says, so a narrower ``--features`` cannot silently misroute a symbolic-axis pool.
    # Featurized under the pool's regime, as it was enumerated: what a schedule space offers depends on it.
    with pinned_knobs(pool.pins):
        feats = [{k: v for k, v in featurizer.features(kernel, r).items() if keep(k)} for r in rows]
    return _Packed(pool, tier, _shape_group(shape), pack_features(feats), candidates.total, goldens, [pool]), skipped, None


def build_golden_groups(
    pools: Sequence[GoldenPool],
    features_spec: str = "*",
    *,
    sample: int = 0,
    seed: int = 0,
    kernel: str | None = None,
    jobs: int = 1,
) -> tuple[list[GoldenGroup], list[tuple[str, str, str]]]:
    """Enumerate each golden pool (``db/export.golden_pools``), pin its golden rows, and featurize every
    candidate, as :class:`GoldenGroup` records (name, tier, card, pinned rows, per-row features filtered through
    the ``features_spec`` view; ``key`` is ``"<gpu>/<pool name>"``, suffixed ``#2``, ``#3``, … when one name
    opens several distinct pools). The second return is the golden rows that did NOT land in a group, as
    ``(gpu, name, reason)``, so metrics can count every golden row the pools hold.

    **A group is a candidate pool, not a golden.** Several golden rows can land on one pool — a shape
    recorded under two names, or recorded twice — and they then share ONE group, each contributing a row to
    its golden set. Each pool is enumerated, featurized and packed once, then folded by :func:`_pool_identity`
    with any other pool that packed identically; every group is built knowing all of its goldens.

    A pool is enumerated under its OWN card's context and regime (:func:`pool_context`): the golden set spans
    cards that differ in compute capability AND in SM count at the same cap, so both the candidate enumeration
    (cp.async / TMA tiers gate on cap) and the ``H_*`` / ``D_*`` occupancy features must use the recording
    card's regime for the rank objective to match the deployed per-card featurization. The base features are
    the context's and the kernel's stamps, computed from its definition.

    ``sample`` draws that many complete rows per pool DURING enumeration (0 enumerates every row) by seeded
    descents through the pool's schedule tree — a pure function of the tree and ``(sample, seed)`` — and every
    golden row recorded on the pool's card and regime is reached by its own descent whatever the draw picks,
    so a golden that misses its pool still means what it always meant: a pin or dtype mismatch.

    ``kernel`` keeps only pools whose kernel's C name contains it — a narrowing VIEW, for iterating on one kernel
    without paying for the rest. Each retained pool's rank is unchanged by it: the keep-set is computed over every
    pool given, so a pool retains the same rows under the same draw; what changes is the group and positive counts,
    so only an unfiltered run compares against a fit.

    ``jobs`` enumerates that many pools side by side, one pool per worker process: a pool's draw is a pure function
    of its tree and the seed, so any process draws the same rows, and the results are folded in the pools' order,
    so the groups and their ``#N`` suffixes are the same at any count. One process by default — the suite runs
    its own workers — and the CLI asks for every core."""
    groups: list[GoldenGroup] = []
    skipped: list[tuple[str, str, str]] = []
    key_counts: dict[str, int] = {}
    matched = 0
    # The keep-set spans every pool of a card and regime: the rows a sample may not drop are every golden
    # row recorded there, so two pools that turn out to be one retain identical rows and merge.
    keeps: dict[tuple, set] = defaultdict(set)
    if sample > 0:
        for pool in pools:
            keeps[(pool.gpu, pool.regime)].update(tuple(sorted(row.items())) for row in pool.schedule_rows())
    tasks = []
    for pool in pools:
        if kernel is not None and kernel not in pool.kernel.name:
            continue
        if not pool.kernel.formed:
            skipped.extend((pool.gpu, pool.name, "kernel formed from no loop op") for _ in pool.rows)
            continue
        tasks.append((pool, tuple(sorted(keeps.get((pool.gpu, pool.regime), ()))), sample, seed, features_spec))
    if jobs == 1:
        results = map(_enumerate_packed, tasks)
    else:
        # Spawned: the one start method every platform has, and a fresh interpreter per worker, so what a worker
        # imports is the whole contract (the kernel definition imports the IR it decodes with).
        with ProcessPoolExecutor(jobs, mp_context=get_context("spawn")) as workers:
            results = list(workers.map(_enumerate_packed, tasks, chunksize=1))
    packed_pools: dict[tuple, _Packed] = {}
    for (pool, *_), (entry, missed, note) in zip(tasks, results, strict=True):
        skipped.extend(missed)
        if note is not None:
            logger.warning("  !! %s: %s", pool.name, note)
        if entry is None:
            continue
        matched += len(entry.goldens)
        # Two pools can still pack identically — the same kernel recorded at two sizes it does not depend on.
        # Fold those together, so a pool is one group however many times it was recorded.
        identity = _pool_identity(pool.gpu, entry.tier, entry.shape, entry.packed)
        found = packed_pools.get(identity)
        if found is None:
            packed_pools[identity] = entry
        else:
            found.goldens.extend(entry.goldens)
            found.pools.append(pool)

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
                pools=entry.pools,
            )
        )
    logger.info(
        "  %d matched goldens over %d candidate pools (%d beyond one golden per pool)",
        matched,
        len(groups),
        matched - len(groups),
    )
    return groups, skipped


PLACEMENT_PASSES = ("tile/lift", "tile/cut")


@dataclass(frozen=True)
class PlacementFork:
    """One placement fork as the walk saw it: each arm's feature row and its label (``fuse``, or the seams it
    cuts), the arms the golden took (``positives``), and the arm the walk itself took (``pick``) — ``None`` when it
    took the composed route, which is no arm of the pool."""

    feats: list[dict]
    labels: list[str]
    positives: list[int]
    pick: int | None


class _FirstFork(Exception):
    """Ends a walk that needs only the kernel's own fork."""


def walk_placement(
    pool: GoldenPool, ctx: Context, decisions: dict[str, dict], prior=None, *, first: bool = False
) -> tuple[list[PlacementFork], list[str]]:
    """One pool's placement forks, in walk order, and the kernels whose recorded decision the fork did not offer
    (a stale spelling). Without ``prior`` the walk follows the golden (``decisions``: a kernel's exact identity to
    the ``PLACE`` arm recorded on it; fused where none is); with a placement ``prior`` it takes the arm the prior
    ranks first, which is what a deploy would do at that fork. ``first`` stops
    the walk at the kernel's own fork, which is all a reproduction verdict reads."""
    from emmy.compiler.pipeline import Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.fork import leaf_knobs  # noqa: PLC0415
    from emmy.compiler.pipeline.knob import family_of  # noqa: PLC0415
    from emmy.compiler.pipeline.pipeline import NO_OPTION, Run  # noqa: PLC0415
    from emmy.compiler.pipeline.search.pins import composed_routes, unpinned_decisions  # noqa: PLC0415

    featurizer = features.Featurizer.of(ctx)
    forks: list[PlacementFork] = []
    unmatched: list[str] = []

    def decide(fp):
        leaves = list(fp.leaves())
        if not leaves:
            return NO_OPTION
        rows = [leaf_knobs(leaf) for leaf in leaves]
        place = [i for i, row in enumerate(rows) if any(family_of(k) == "PLACE" for k in row)]
        if not place:
            return leaves[0]
        root = fp.root_op.with_io(fp.match.graph, fp.match.root)
        identity = root.identity_key(structural=False, with_io=True)
        taken = decisions.get(identity)
        fused = next(i for i in place if "fuse" in rows[i].values())
        # An arm spells every occurrence of each seam; the seams themselves are the keys its aliases map onto.
        aliases = {alias: key for i in place for alias, key in (getattr(leaves[i], "aliases", None) or {}).items()}
        seams = {i: {aliases.get(k, k) for k in rows[i]} for i in place}
        chosen, steer, positives = fused, None, [fused]
        if taken is not None:
            keys = {aliases.get(k, k) for k in taken}
            if matching := [i for i in place if seams[i] == keys]:
                chosen = matching[-1]
                # The composed arm is offered last, and only because the walk registered the route.
                steer = chosen if len(keys) > 1 else None
                positives = [i for i in place if i != steer and i != fused and seams[i] <= keys]
            else:
                unmatched.append(identity[:12])
        arms = [i for i in place if i != steer]
        feats = [featurizer.features(root, rows[i], pieces=leaves[i].expand()[0]) for i in arms]
        if prior is not None:
            scores = prior.mean_scores_features(feats)
            chosen = arms[min(range(len(arms)), key=scores.__getitem__)]
        labels = ["fuse" if i == fused else " ".join(sorted(k.removeprefix("PLACE@") for k in seams[i])) for i in arms]
        forks.append(PlacementFork(feats, labels, [arms.index(i) for i in positives], arms.index(chosen) if chosen in arms else None))
        if first:
            raise _FirstFork
        return leaves[chosen]

    routes = [(None, tuple(arm)) for arm in decisions.values() if len(arm) > 1]
    with pinned_knobs(pool.pins), unpinned_decisions(), composed_routes(routes):
        try:
            Run(pipeline=Pipeline.build(list(PLACEMENT_PASSES)), ctx=ctx).resolve(pool.kernel.program(pool.bindings), decide)
        except _FirstFork:
            pass
    return forks, unmatched


def placement_decisions(pools: Sequence[GoldenPool], like: GoldenPool) -> dict[str, dict]:
    """Recorded ``PLACE`` arms in ``like``'s card, precision regime and sizes, by kernel identity."""
    return {
        pool.kernel.exact_identity: pool.rows[0].knobs
        for pool in pools
        if pool.rows
        and (pool.gpu, pool.regime) == (like.gpu, like.regime)
        and all(like.bindings.get(name) == size for name, size in pool.bindings.items())
    }


def build_placement_groups(pools: Sequence[GoldenPool]) -> tuple[list[GoldenGroup], list[tuple[str, str, str]]]:
    """Enumerate each placement pool's forks and pack them as :class:`GoldenGroup` records, one per fork: the
    arms the cut pass offers unpinned (keep fused, one seam each, the full-projection cut), each featurized from
    the kernels it leaves (:meth:`~.features.Featurizer.features`), with the cheapest measured arms marked. The second return
    is the pools that produced no group, as ``(gpu, name, reason)``.

    A placement pool's one row is the ``PLACE`` routing decision recorded on its kernel (none: it stayed fused).
    The walk runs the tile lift and the cut pass only (``PLACEMENT_PASSES``) and follows the golden: at a kernel
    with a decision it takes that arm — registered as a composed route, so a several-seam decision is one arm
    whose pieces are the routing row's own children and a nested decision is found by identity — and at a
    kernel without one it keeps the kernel fused. The composed arm steers the walk and is not a row: unpinned,
    the pass offers single seams, and those are what the prior ranks. A single seam the decision names is a
    positive, as is the full-projection arm when it is exactly the decision; fused is the positive where
    nothing was recorded."""
    groups: list[GoldenGroup] = []
    skipped: list[tuple[str, str, str]] = []
    ctxs: dict[tuple, Context] = {}
    for pool in pools:
        if not pool.kernel.formed:
            skipped.append((pool.gpu, pool.name, "kernel formed from no loop op"))
            continue
        card = (pool.cap, pool.gpu, pool.regime)
        ctx = ctxs.get(card)
        if ctx is None:
            ctx = ctxs[card] = pool_context(pool)
        try:
            forks, unmatched = walk_placement(pool, ctx, placement_decisions(pools, pool))
        except ValueError as exc:
            # The same definition the schedule enumeration does not take back (``build_golden_groups``): the
            # reduce piece of a cross-CTA split re-offers the split and mints the buffer it already holds.
            logger.warning("  !! %s: did not lower — %s", pool.name, exc)
            skipped.append((pool.gpu, pool.name, "did not lower"))
            continue
        if unmatched:
            skipped.append((pool.gpu, pool.name, f"decision not offered on {', '.join(unmatched)}"))
        if not forks:
            skipped.append((pool.gpu, pool.name, "no placement fork"))
            continue
        shape = _shape_group(ShapeKey.from_s_features(features.stamps(pool.kernel.op(pool.bindings))))
        for n, fork in enumerate(forks, 1):
            key = f"{pool.gpu}/{pool.name}" + (f"@{n}" if n > 1 else "")
            packed = pack_features(fork.feats)
            # A group's tier agrees with the routing stamp its rows carry (``GoldenGroup.over``).
            tier = "dyn" if packed[2] else "place"
            groups.append(GoldenGroup.over(key, pool.name, tier, pool.gpu, shape, packed, fork.positives, len(fork.feats), pools=(pool,)))
    logger.info("  %d placement forks over %d pools (%d skipped)", len(groups), len(pools), len(skipped))
    return groups, skipped
