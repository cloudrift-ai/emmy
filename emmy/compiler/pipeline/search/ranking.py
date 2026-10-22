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
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from itertools import product
from math import prod
from multiprocessing import get_context

from emmy.compiler.context import Context
from emmy.compiler.pipeline.search import features
from emmy.compiler.pipeline.search.dataset.group import GoldenGroup, pack_features
from emmy.compiler.pipeline.search.dataset.pool import GoldenPool
from emmy.compiler.pipeline.search.dataset.shape import ShapeKey
from emmy.compiler.pipeline.search.pins import pinned_knobs
from emmy.compiler.pipeline.search.pool import Candidates, PoolSample

logger = logging.getLogger(__name__)


def pool_context(pool: GoldenPool) -> Context:
    """The context ``pool``'s rows were measured under: the recording card's own facts and the regime's pins,
    whatever card and pins this process runs with."""
    return Context.from_target(pool.cap, gpu_name=pool.gpu, precision=pool.pins)


class LaterForks:
    """The arms of the forks the kernel lowering opens on a scheduled kernel after its schedule row is decided (the
    carried-state storage, ``SHARED_CARRY``), read at the one schedule a walk takes and crossed with every row of
    that kernel: the lowering runs once per kernel, so what it offers at one schedule it offers at all. The dataset's
    pools are crossed this way, with the arms the lowering offers; the greedy and the tune cross with every registered
    value instead (:func:`later_arms`), and a prior fit on the offered arms scores those rows as it scores the pool's
    (a value the lowering never offers scores as a row nothing was measured on)."""

    def __init__(self) -> None:
        self.arms: dict[str, list[list[dict]]] = {}

    def observe(self, fp) -> None:
        """Record the arms of ``fp``, a later fork on a scheduled kernel: the knobs its options disagree on."""
        from emmy.compiler.pipeline.fork import iter_leaves, leaf_knobs  # noqa: PLC0415

        rows = [leaf_knobs(leaf) for leaf in iter_leaves(fp.options)]
        decided = {k for row in rows for k in row if len({str(r.get(k)) for r in rows}) > 1}
        self.arms.setdefault(fp.node_id, []).append([{k: row[k] for k in sorted(decided)} for row in rows])

    def cross(self, node_id: str, rows: list[dict]) -> list[dict]:
        """``rows`` of the kernel at ``node_id`` crossed with every arm of each later fork observed on it."""
        for arms in self.arms.get(node_id, ()):
            rows = [{**row, **arm} for row in rows for arm in arms]
        return rows

    def factor(self, node_id: str) -> int:
        """How many rows one schedule row of the kernel at ``node_id`` becomes."""
        return prod(len(arms) for arms in self.arms.get(node_id, ()))


def later_knob(key: str) -> bool:
    """Whether ``key`` is a knob a later fork decides on a scheduled kernel: registered, offered in more than one
    value, and no kernel decision (a cut, a layout, the schedule row itself)."""
    from emmy.compiler.pipeline.knob import KERNEL_DECISION_FAMILIES, family_of, get  # noqa: PLC0415

    return (knob := get(key)) is not None and family_of(key) not in KERNEL_DECISION_FAMILIES and len(knob.hints) > 1


def later_arms() -> list[dict]:
    """Every combination of the hinted values of the knobs the kernel lowering decides after the schedule (the
    carried-state storage, ``SHARED_CARRY``), read from the knob registry, as rows a schedule row is crossed with. The
    one crossing the dataset's pools, ``run --tune`` and the greedy's schedule fork share, so a prior fit on the pools
    scores every knob a compile decides on a kernel. A combination a kernel or a schedule cannot realize is a row the
    lowering never offers: a negative in the pools, never the pick at the fork that decides it."""
    from emmy.compiler.pipeline.knob import registry  # noqa: PLC0415

    knobs = [knob for name, knob in sorted(registry().items()) if later_knob(name)]
    return [dict(zip((knob.name for knob in knobs), values, strict=True)) for values in product(*(knob.hints for knob in knobs))]


def enumerate_graph(
    graph, ctx: Context, *, family: str = "", passes: Sequence[str] | None = None, prefer: Mapping | None = None
) -> Candidates:
    """The planner's candidate enumeration for any ``graph`` — the SAME rows the scheduler's fork
    tree offers a live compile, captured by resolving the graph through ``passes`` (the whole pipeline
    to the kernel lowering by default; from the tile lowering for a kernel's definition, which the Loop passes
    would normalize into another kernel) with a decide that flattens each fork's leaves. Every leaf encodes one
    accepted classic ``Schedule`` with bare kernel keys and exact ``@n`` / ``@n.e`` sites, which is exactly
    what ``tile_signature`` joins a golden against, crossed with the arms of every fork the kernel lowering
    opens on the scheduled kernel afterwards (:class:`LaterForks`), so a row spells every knob a compile
    decides on the kernel. ``family`` keeps only rows carrying that knob
    family (``"TILE"`` for a contraction pool); ``""`` keeps every row with a per-node schedule
    knob (a reduce's ``REDUCE`` fork). ``prefer`` is the schedule row the walk continues through where a fork
    offers it (a golden row's), so the later forks are read at a schedule that opens them; the first leaf
    otherwise. The one live-fork capture the matmul offline fitter and record evaluator share.

    Returns :class:`~.pool.Candidates` — the rows beside what they were ranked among. Under
    ``ctx.pool_sample`` the rows are a DRAW (seeded descents through each schedule tree, the golden rows
    kept — ``PoolSample.draw``) and ``total`` is the draw's size; BOTH count the same population: distinct
    schedule-space stamps. Equal problems produce the same stamp, so their identical draw and total are
    collected once. With no sample the rows are every kernel's fork rows and ``total`` is ``len(rows)``."""
    from emmy.compiler.pipeline import KERNEL_PASSES, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.fork import leaf_for, leaf_knobs  # noqa: PLC0415
    from emmy.compiler.pipeline.knob import family_of  # noqa: PLC0415
    from emmy.compiler.pipeline.pipeline import Run  # noqa: PLC0415
    from emmy.compiler.pipeline.search.space import WORK  # noqa: PLC0415

    rows_of: dict[str, list[dict]] = {}  # a scheduled kernel's node -> its schedule rows
    later = LaterForks()
    node_of: dict[str, str] = {}  # a drawn pool's stamp -> the node it was drawn at
    wanted = (family,) if family else ("TILE", "REDUCE", "STAGE")
    # The sample's size sink is PER CALL: the enumerator writes each pool's exact size there as it
    # goes (a fork tree has no channel to return it through), so it is cleared before the walk and
    # read straight after.
    sample = ctx.pool_sample if ctx is not None else None
    if sample is not None:
        sample.totals.clear()
    seen_pools: set[str] = set()

    def decide(fp):
        node = fp.node_id if fp.match is not None else ""
        if rows_of.get(node):
            # A later fork on a kernel whose schedule rows this walk already holds.
            later.observe(fp)
            return _first(fp)
        if sample is not None:
            # One contribution per schedule-space stamp, matching the totals sink's keyed dedupe:
            # an equal problem overwrites the same total, and appending its identical draw again
            # would make ``rows`` and ``total`` count different populations.
            opened = set(sample.totals) - seen_pools
            seen_pools.update(opened)
            node_of.update(dict.fromkeys(opened, node))
            if not opened:
                return _first(fp)
        for leaf in fp.leaves():
            row = leaf_knobs(leaf)
            # A schedule row always spells the kernel-global ``WORK``; a structural arm's knob
            # delta (a cut, the cross-CTA split's g-half or the unsplit receipt) never does — the
            # stated row-identity marker (the classic scheduler's leaf boundary).
            if WORK.name in row and any(family_of(k) in wanted for k in row):
                rows_of.setdefault(node, []).append(row)
        preferred = leaf_for(fp.options, _schedule_only(dict(prefer))) if prefer is not None else None
        return preferred[0] if preferred is not None else _first(fp)

    def _first(fp):
        # A pin may empty an early lazy branch while leaving a later sibling live. Walk to the
        # first complete leaf across the whole sibling set, matching the resolver's own traversal;
        # only an entirely empty fork means the schedule is not offered.
        option = next(fp.leaves(), None)
        if option is None:
            from emmy.compiler.pipeline.pipeline import NO_OPTION  # noqa: PLC0415

            return NO_OPTION
        return option

    terminal, _ = Run(pipeline=Pipeline.build(list(passes) if passes is not None else KERNEL_PASSES), ctx=ctx).resolve(graph, decide)
    rows = [row for node_id, node_rows in rows_of.items() for row in later.cross(node_id, node_rows)]
    if sample is None:
        # A fully pinned classic problem can collapse without opening a policy-visible fork. Its
        # complete row still belongs to the enumeration: read it from the realized kernel set so
        # multi-kernel structural targets expose every independently scheduled problem.
        from emmy.compiler.pipeline.knob import SCHEDULE_FAMILIES  # noqa: PLC0415

        for node in terminal.nodes.values():
            knobs = getattr(node.op, "knobs", None) or {}
            if WORK.name not in knobs:
                continue
            row = {key: value for key, value in knobs.items() if family_of(key) in SCHEDULE_FAMILIES or later_knob(key)}
            # A later knob decided with one arm opened no fork, so the rows spell it by its absence.
            if any(family_of(key) in wanted for key in row) and row not in rows and _schedule_only(row) not in rows:
                rows.append(row)
        return Candidates(rows, len(rows))
    return Candidates(rows, sum(total * later.factor(node_of.get(stamp, "")) for stamp, total in sample.totals.items()))


def enumerate_pool(pool: GoldenPool, ctx: Context) -> Candidates:
    """``pool``'s candidates: its kernel's definition at the pool's sizes through the tile and kernel lowering under
    the regime's pins alone — the wire is the kernel itself, so a live kernel-decision pin means nothing on it — the
    later forks read at the first golden row's schedule, which opens the arm that golden took."""
    from emmy.compiler.pipeline import LOWERING_PASSES  # noqa: PLC0415
    from emmy.compiler.pipeline.search.pins import unpinned_decisions  # noqa: PLC0415

    prefer = pool.rows[0].knobs if pool.rows else None
    with pinned_knobs(pool.pins), unpinned_decisions():
        return enumerate_graph(pool.kernel.program(pool.bindings), ctx, passes=LOWERING_PASSES[:-1], prefer=prefer)


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


#: How much slower than the fastest verified row of its pool a verified row may measure and still train as a positive:
#: the spread two records of one row show across days on one card. A verified row beyond it is measured slower, and
#: is neither a positive nor drawn as a negative (:class:`~.prior.fit.catboost.CatBoostTrainer`).
POSITIVE_WITHIN = 1.1


def _schedule_only(row: dict) -> dict:
    """``row`` without the knobs a later fork decides: the schedule row a kept draw descends to, before the
    crossing adds the arms back."""
    return {k: v for k, v in row.items() if not later_knob(k)}


def _later_match(golden: dict, candidate: dict) -> bool:
    """Whether ``golden`` spells every later-fork knob of ``candidate`` with the same value, an absent one as off."""
    from emmy.compiler.pipeline.knob import get  # noqa: PLC0415

    return all(str(golden.get(k, get(k).off)) == str(v) for k, v in candidate.items() if later_knob(k))


def _enumerate_packed(task: tuple) -> tuple[_Packed | None, list[tuple[str, str, str]], str | None]:
    """One pool's share of :func:`build_golden_groups` — the work one worker process does: enumerate the pool
    under its own card's context, locate its golden rows among the candidates, featurize and pack them. ``task``
    is the pool, the rows its draw may not drop, and the draw's size and seed. Returns the packed pool (``None``
    when it opened no group), the golden rows that landed in no group as ``(gpu, name, reason)``, and the line to
    log for a pool that was skipped — logged by the caller, so the lines keep the pools' order whichever process
    did the work."""
    pool, keep_rows, sample, seed = task
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
    # record legacy GEMM-letter keys, so comparing key-value tuples directly never matches. The knobs a later
    # fork decided on the kernel join by value; a golden recorded before a knob existed was measured at its off
    # value.
    # A pool's positives are its fastest verified rows: a verified row another verified row of the same pool beats by
    # more than measurement noise is a measured slower row, and labelling it good would teach the fit to deploy it.
    goldens, skipped = [], []
    for golden in pool.rows:
        want = features.tile_signature(golden.knobs)
        gidx = next((i for i, r in enumerate(rows) if features.tile_signature(r) == want and _later_match(golden.knobs, r)), None)
        if gidx is None:
            skipped.append((pool.gpu, pool.name, f"golden not in {len(rows)} candidates"))
        elif golden.us <= POSITIVE_WITHIN * pool.emmy_us:
            goldens.append(gidx)
    if not goldens:
        return None, skipped, f"golden not in {len(rows)} candidates"
    kernel = pool.kernel.op(pool.bindings)
    shape = ShapeKey.from_s_features(features.stamps(kernel))
    tier = "dyn" if shape.is_dyn else (shape.kind or ("warp" if shape.is_warp else "thread"))
    # Every feature the featurizer computes is packed: the featurizer is the one definition both priors train on.
    # Featurized under the pool's regime, as it was enumerated: what a schedule space offers depends on it.
    with pinned_knobs(pool.pins):
        feats = [featurizer.features(kernel, r) for r in rows]
    return _Packed(pool, tier, _shape_group(shape), pack_features(feats), candidates.total, goldens, [pool]), skipped, None


def build_golden_groups(
    pools: Sequence[GoldenPool],
    *,
    sample: int = 0,
    seed: int = 0,
    kernel: str | None = None,
    jobs: int = 1,
) -> tuple[list[GoldenGroup], list[tuple[str, str, str]]]:
    """Enumerate each golden pool (``db/export.golden_pools``), pin its golden rows, and featurize every
    candidate, as :class:`GoldenGroup` records (name, tier, card, pinned rows, every feature the featurizer
    computes per row; ``key`` is ``"<gpu>/<pool name>"``, suffixed ``#2``, ``#3``, … when one name
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
            keeps[(pool.gpu, pool.regime)].update(tuple(sorted(_schedule_only(row).items())) for row in pool.schedule_rows())
    tasks = []
    for pool in pools:
        if kernel is not None and kernel not in pool.kernel.name:
            continue
        if not pool.kernel.formed:
            skipped.extend((pool.gpu, pool.name, "kernel formed from no loop op") for _ in pool.rows)
            continue
        tasks.append((pool, tuple(sorted(keeps.get((pool.gpu, pool.regime), ()))), sample, seed))
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
        identity = (pool.regime, _pool_identity(pool.gpu, entry.tier, entry.shape, entry.packed))
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
    """One kernel-set fork as the walk saw it: its domain's tier, each arm's feature row and its label (``fuse`` or
    the seams a cut cuts, ``unsplit`` or a split's width, a layout), the arms on the golden's path (``positives``)
    and the arm the walk itself took (``pick``)."""

    tier: str
    feats: list[dict]
    labels: list[str]
    positives: list[int]
    pick: int


def _place_ballot(
    leaves: list, rows: list[dict], identity: str, name: str, pending: Counter | None, continued: frozenset[str]
) -> tuple[list[int], list[str], list]:
    """A placement fork's ballot as ``(positives, labels, carved)``: the arms on the golden's path, every arm's
    label, and per arm the pieces it carves (every kernel it leaves but the remainder, the one keeping the fork
    kernel's ``name``), by exact identity with multiplicity. ``pending`` is what the golden's kernel set still holds
    below this kernel: ``None`` where the golden cut nothing here, which keeps the kernel whole, as does a kernel
    that is itself a pending piece. Otherwise an arm is positive when every piece it carves is pending or is a
    kernel the golden cut in its turn (``continued``), whose own fork carries the walk on; an arm leaving exactly
    the pending kernels is positive instead of those (a complete natural cut over its single seams). No positive:
    the golden's kernel set is not reached one seam at a time from here. Equivalent cuts may spell different
    seams, which is why a piece counts by identity and never by spelling."""
    fused = next(i for i, row in enumerate(rows) if "fuse" in row.values())
    # An arm spells every occurrence of each seam; the seams themselves are the keys its aliases map onto.
    aliases = {alias: key for leaf in leaves for alias, key in (getattr(leaf, "aliases", None) or {}).items()}
    seams = [{aliases.get(k, k) for k in row} for row in rows]
    labels = ["fuse" if i == fused else " ".join(sorted(k.removeprefix("PLACE@") for k in seams[i])) for i in range(len(rows))]
    carved: list = [None] * len(rows)
    if pending is None or identity in pending:
        return [fused], labels, carved
    whole: dict[int, Counter] = {}
    for i in range(len(rows)):
        if i == fused:
            continue
        pieces = [(op.name, op.identity_key(structural=False, with_io=True)) for op, _ in features.kernel_pieces(leaves[i].expand()[0])]
        whole[i] = Counter(key for _, key in pieces if key)
        carved[i] = Counter(key for piece, key in pieces if key and piece != name)
    complete = [i for i in whole if whole[i] == pending]
    positives = complete or [
        i
        for i in whole
        if carved[i]
        and Counter({key: n for key, n in carved[i].items() if key not in continued}) <= pending
        and all(key in pending or key in continued for key in carved[i])
    ]
    # Deepest seam first: a cone is carved before the cone containing it, so the outer piece reads the inner
    # workspace — the kernel a composed cut of both leaves — and a seam's spelling never moves under the walk.
    positives.sort(key=lambda i: -max(key.count("/") for key in seams[i]))
    return positives, labels, carved


def _resolve_placement(pool: GoldenPool, ctx: Context, routes: list, decide):
    """The pool's program (its constant inputs restored) resolved through the lift and the cut pass under ``decide``,
    with the pool's regime pinned, every kernel-decision pin withdrawn and ``routes`` registered as composed."""
    from emmy.compiler.ir.base import ConstantOp  # noqa: PLC0415
    from emmy.compiler.pipeline import Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.pipeline import Run  # noqa: PLC0415
    from emmy.compiler.pipeline.search.pins import composed_routes, unpinned_decisions  # noqa: PLC0415

    with pinned_knobs(pool.pins), unpinned_decisions(), composed_routes(routes):
        program = pool.kernel.program(pool.bindings)
        for name, wire in pool.placement_constants.items():
            program.nodes[name].op = ConstantOp.from_wire(wire)
            program.inputs.remove(name)
        terminal, _trace = Run(pipeline=Pipeline.build(list(PLACEMENT_PASSES)), ctx=ctx).resolve(program, decide)
    return terminal


def _recorded_kernel_set(
    pool: GoldenPool, ctx: Context, decisions: dict[str, dict], cuts: dict[str, tuple[str, ...]], unmatched: list[str]
) -> tuple[Counter, frozenset[str]]:
    """The kernel set the golden left below the pool's kernel, by exact identity with multiplicity, and the kernels
    it cut on the way there. The walk that follows the golden ends in it: at a placement fork the recorded arm —
    ``cuts``, the seams each decision cuts, registers a several-seam decision as one composed route under its
    kernel's identity, so the one decision a pinned compile consumed those seams as is one arm, and a decision on
    a piece is found by identity; a kernel with no decision stays fused, and every kernel stays unsplit and
    folded, since a cut is decided before either. A kernel whose recorded cut is not on offer (a stale spelling)
    is named in ``unmatched`` and kept whole."""
    from emmy.compiler.pipeline.pipeline import NO_OPTION, _structural_domain  # noqa: PLC0415
    from emmy.compiler.pipeline.search.pins import KERNEL_SET_DOMAINS, spelled_arm  # noqa: PLC0415

    cut: set[str] = set()

    def decide(fp):
        leaves = list(fp.leaves())
        if not leaves:
            return NO_OPTION
        if KERNEL_SET_DOMAINS.get(_structural_domain(fp.options)) != "place" or len(leaves) < 2:
            return leaves[0]
        root = fp.root_op.with_io(fp.match.graph, fp.match.root)
        identity = root.identity_key(structural=False, with_io=True)
        arm = spelled_arm(leaves, decisions.get(identity, {}))
        if arm is None:
            unmatched.append(identity[:12])
            return leaves[0]
        if "cut" in arm[1].values():
            cut.add(identity)
        return arm[0]

    routes = [(identity, keys) for identity, keys in cuts.items() if len(keys) > 1]
    terminal = _resolve_placement(pool, ctx, routes, decide)
    left = Counter(key for op, _ in features.kernel_pieces(terminal) if (key := op.identity_key(structural=False, with_io=True)))
    return left, frozenset(cut)


def walk_placement(
    pool: GoldenPool, ctx: Context, decisions: dict[str, dict], prior=None, *, own: bool = False
) -> tuple[list[PlacementFork], list[str]]:
    """One pool's kernel-set forks (``pins.KERNEL_SET_DOMAINS``: placement cuts, cross-CTA and carry splits,
    layouts), in walk order, and the kernels whose recorded decision the walk could not take (a stale spelling, or
    a kernel set no single-seam walk reaches). The walk cuts one seam at a time, as a deploy with no recorded route
    does: where the golden cut the pool's kernel, an arm is positive when every piece it carves belongs to the
    kernel set the golden left (:func:`_recorded_kernel_set`), and the remainder re-enters the walk owing what is
    still to carve, until it is itself a piece of that set and stays whole; a split or layout fork's positive is
    the arm the golden recorded on that kernel (``decisions``: a kernel's exact identity to the kernel-set arm
    recorded on it; unsplit and folded where none is). An arm carving a kernel the golden cut on the way to its
    set is positive too: that kernel's own fork carries the walk on. The walk takes the first positive at every
    fork, so its forks are the dataset's groups; a placement ``prior`` records at each of them the arm it ranks
    first, which a deploy would take there. ``own`` keeps only the forks of the pool's own kernel and the placement
    forks of the kernels its cut leaves to cut further, which is all a reproduction verdict reads."""
    from emmy.compiler.ir.tile import TileOp  # noqa: PLC0415
    from emmy.compiler.pipeline.fork import leaf_knobs  # noqa: PLC0415
    from emmy.compiler.pipeline.knob import family_of  # noqa: PLC0415
    from emmy.compiler.pipeline.pipeline import NO_OPTION, _structural_domain  # noqa: PLC0415
    from emmy.compiler.pipeline.search.pins import KERNEL_SET_DOMAINS, spelled_arm  # noqa: PLC0415

    featurizer = features.Featurizer.of(ctx)
    forks: list[PlacementFork] = []
    unmatched: list[str] = []
    # The seams each recorded decision cuts, by its kernel's identity.
    cuts = {
        identity: tuple(sorted(k for k, v in arm.items() if family_of(k) == "PLACE" and v == "cut")) for identity, arm in decisions.items()
    }
    cuts = {identity: keys for identity, keys in cuts.items() if keys}
    recorded, continued = (
        _recorded_kernel_set(pool, ctx, decisions, cuts, unmatched) if pool.kernel.exact_identity in cuts else (None, frozenset())
    )
    # What the golden's kernel set still holds below a kernel the walk cuts further, by its graph node id.
    pending: dict[str, Counter] = {}

    def decide(fp):
        leaves = list(fp.leaves())
        if not leaves:
            return NO_OPTION
        tier = KERNEL_SET_DOMAINS.get(_structural_domain(fp.options))
        if tier is None or len(leaves) < 2:
            return leaves[0]
        root = fp.root_op.with_io(fp.match.graph, fp.match.root)
        identity = root.identity_key(structural=False, with_io=True)
        owed = recorded if identity == pool.kernel.exact_identity else pending.get(fp.match.root_node_id)
        if own and identity != pool.kernel.exact_identity and (tier != "place" or owed is None):
            return leaves[0]
        rows = [leaf_knobs(leaf) for leaf in leaves]
        carved: list = []
        if tier == "place":
            positives, labels, carved = _place_ballot(leaves, rows, identity, root.name, owed, continued)
            if not positives:
                unmatched.append(identity[:12])
                return leaves[0]
        else:
            # A split or layout fork offers one kernel-set decision per arm; the first arm keeps the kernel as it is.
            arm = spelled_arm(leaves, decisions.get(identity, {}))
            chosen = next((i for i, leaf in enumerate(leaves) if arm is not None and leaf is arm[0]), None)
            if chosen is None:
                unmatched.append(identity[:12])
                chosen = 0
            positives = [chosen]
            labels = [" ".join(str(v) for v in row.values() if v) or "unsplit" for row in rows]
        # Featurized as a deploy featurizes them (``policy/greedy._kernel_set_pick``): an arm that keeps the kernel
        # leaves the kernel itself.
        feats = [
            featurizer.features(root, row, pieces=leaf.expand()[0] if leaf.structural else root)
            for leaf, row in zip(leaves, rows, strict=True)
        ]
        followed = positives[0]
        pick = followed if prior is None else min(range(len(leaves)), key=prior.mean_scores_features(feats).__getitem__)
        forks.append(PlacementFork(tier, feats, labels, positives, pick))
        if tier == "place" and carved[followed] is not None:
            left = owed - carved[followed]
            fragment = leaves[followed].expand()[0]
            for node_id, node in fragment.nodes.items():
                if isinstance(node.op, TileOp) and (
                    node.op.name == root.name or node.op.with_io(fragment, node).identity_key(structural=False, with_io=True) in continued
                ):
                    # The remainder and a piece the golden cut in its turn are cut further; the splice promotes
                    # a piece onto the buffer it replaces.
                    pending[node_id] = pending[node_id.removesuffix("__placed")] = left
        return leaves[followed]

    _resolve_placement(pool, ctx, [], decide)
    return forks, unmatched


def placement_decisions(pools: Sequence[GoldenPool], like: GoldenPool) -> dict[str, dict]:
    """Recorded kernel-set arms (a ``PLACE`` cut, a ``REDUCE`` split) in ``like``'s card, precision regime and sizes, by kernel identity."""
    return {
        pool.kernel.exact_identity: pool.rows[0].knobs
        for pool in pools
        if pool.rows
        and (pool.gpu, pool.regime) == (like.gpu, like.regime)
        and all(like.bindings.get(name) == size for name, size in pool.bindings.items())
    }


def _walk_packed(task: tuple) -> tuple[list[GoldenGroup], list[tuple[str, str, str]], str | None]:
    """One pool's share of :func:`build_placement_groups` — the work one worker process does: walk the pool under
    its own card's context and pack each fork as a group. ``task`` is the pool and the recorded kernel-set
    decisions in its card, regime and sizes. Returns the groups, the pool's skips as ``(gpu, name, reason)``, and
    the line to log for a pool that did not lower — logged by the caller, so the lines keep the pools' order
    whichever process did the work."""
    pool, decisions = task
    try:
        forks, unmatched = walk_placement(pool, pool_context(pool), decisions)
    except ValueError as exc:
        # The same definition the schedule enumeration does not take back (``build_golden_groups``): the
        # reduce piece of a cross-CTA split re-offers the split and mints the buffer it already holds.
        return [], [(pool.gpu, pool.name, "did not lower")], f"did not lower — {exc}"
    skipped = [(pool.gpu, pool.name, f"recorded cut not reached on {', '.join(unmatched)}")] if unmatched else []
    if not forks:
        return [], [*skipped, (pool.gpu, pool.name, "no kernel-set fork")], None
    shape_key = ShapeKey.from_s_features(features.stamps(pool.kernel.op(pool.bindings)))
    shape = _shape_group(shape_key)
    groups = []
    for n, fork in enumerate(forks, 1):
        key = f"{pool.gpu}/{pool.name}" + (f"@{n}" if n > 1 else "")
        tier = "dyn" if shape_key.is_dyn else fork.tier
        groups.append(
            GoldenGroup.over(
                key, pool.name, tier, pool.gpu, shape, pack_features(fork.feats), fork.positives, len(fork.feats), pools=(pool,)
            )
        )
    return groups, skipped, None


def build_placement_groups(pools: Sequence[GoldenPool], *, jobs: int = 1) -> tuple[list[GoldenGroup], list[tuple[str, str, str]]]:
    """Enumerate each placement pool's kernel-set forks and pack them as :class:`GoldenGroup` records, one per
    fork, in its domain's tier (``pins.KERNEL_SET_DOMAINS``; ``dyn`` for a symbolic kernel): the arms the cut pass
    offers unpinned (keep fused, one seam each, the full-projection cut; unsplit and each split width; each
    layout), each featurized from the kernels it leaves (:meth:`~.features.Featurizer.features`), with the
    golden's arms marked. The second return is the pools that produced no group, as ``(gpu, name, reason)``.

    A placement pool's one row is the kernel-set decision recorded on its kernel — a ``PLACE`` cut or a
    ``REDUCE`` split; none: it stayed one kernel. The walk runs the tile lift and the cut pass only
    (``PLACEMENT_PASSES``) and cuts one seam at a time, as a deploy with no recorded route does
    (:func:`walk_placement`): a several-seam decision is the kernel set it left, and every fork on the way to that
    set is a group — the kernel's, then each remainder's, until the remainder is itself a piece of the set — with
    the arms carving pieces of it marked. A piece's own fork, where it stays whole, is a group too, as is a split
    or layout fork with the arm the decision names marked, and the first arm marked where nothing was recorded. A
    kernel set no single-seam walk reaches is counted out by name: a deploy cannot take it either. ``jobs`` walks that
    many pools side by side, one pool per worker process, as :func:`build_golden_groups` enumerates."""
    groups: list[GoldenGroup] = []
    skipped: list[tuple[str, str, str]] = []
    tasks = []
    for pool in pools:
        if not pool.kernel.formed:
            skipped.append((pool.gpu, pool.name, "kernel formed from no loop op"))
            continue
        tasks.append((pool, placement_decisions(pools, pool)))
    if jobs == 1:
        results = map(_walk_packed, tasks)
    else:
        with ProcessPoolExecutor(jobs, mp_context=get_context("spawn")) as workers:
            results = list(workers.map(_walk_packed, tasks, chunksize=1))
    for (pool, _), (found, missed, note) in zip(tasks, results, strict=True):
        if note is not None:
            logger.warning("  !! %s: %s", pool.name, note)
        groups.extend(found)
        skipped.extend(missed)
    logger.info("  %d kernel-set forks over %d pools (%d skipped)", len(groups), len(pools), len(skipped))
    return groups, skipped
