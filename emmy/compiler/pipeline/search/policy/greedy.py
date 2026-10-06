"""The greedy compile pick — :func:`greedy_decide`, a ``Run.resolve`` decide
factory choosing one **complete** leaf via direct evidence or the prior, else option-0.

This is the deterministic pick for ``compile`` / ``run`` and the assembled-graph
lowering. It is NOT a search and not
a ``Search`` policy: there is no frontier to rank, no tree, no benching — a
deterministic resolution is a fold over the pipeline (at each fork, a pure
function of ``(options, op, prior)``, argmin, continue), so its process state
is :meth:`Run.resolve`'s returned trace, never accumulated policy attributes.

**Evaluate complete rows.** A branch carries only a partial schedule, so a prior cannot score it as
though it were a complete row. Measured rows descend to their exact spelling;
otherwise greedy scores the complete offered rows and chooses the argmin — streamed off the
lazy walk in bounded chunks (:func:`_stream_tiers`), so the scan is O(chunk) memory however large
the pool, and bounded in descent work by the cold-pool budget except when one complete path itself
has a larger declared bound: that pool gets exactly one descent attempt. A pool whose minted size
bound exceeds :data:`_POOL_BUDGET` is ranked over a deterministic drawn subset of its complete rows
instead of walked at full length. Sampling complete rows is not branch substitution — no branch is
ever scored as a stand-in for the schedules it contains — and the argmin is global again the moment
evidence exists, because measured rows descend directly whatever the pool size.

**Greedy is ranked by evidence and by nothing else.** Measured rows first — the tune DB's rows,
the golden rows in scope imported among them (``golden.evidence``), compared on µs alone — then
the fitted prior; every measured row is a
recording of something that ran. There is no hand-written step: no leaf is
promoted, demoted, withheld or given a head start here, and no fallback
default is chosen for being safe. Where nothing measured and no prior speaks
the pick degenerates to the enumeration's first leaf, which carries no meaning
and can be arbitrarily slow. That is the accepted cost of the rule, not a
defect to patch: a bad unmeasured pick is fixed by measuring (a tune, a benched
golden row) or by fitting the prior better, never by teaching this module a
preference — or refused outright under strict evidence
(:func:`_require_evidence`), which raises :class:`EvidenceError` for a fork
no measurement decides.

**Kernel-set forks follow the same rule, and are decided before any schedule.** Every measured
row of the kernel spells one offered arm (:func:`~emmy.compiler.pipeline.search.pins.spelled_arm`):
a schedule row the fused / unsplit arm — the kernel it decorates ran that way — and every
kernel-set decision the tune DB stores on the exact kernel (a routing row) its cut or split,
priced from the pieces' own rows (``SearchDB.priced_arms``). :func:`_route_candidates` turns them
into candidates priced at those µs, nothing installed on the kernel, and the fastest wins. With no
measured arm the placement prior ranks the arms by the kernels each leaves
(:func:`_kernel_set_pick`). No arm is scheduled to decide the fork: the pieces an arm mints are
brand-new kernels, decided at their own forks from their own rows.

**A measurement can also DISQUALIFY.** The measured sources above all RANK, and a
ranking needs a latency — which a ``bench_fail`` row does not have, only the
watchdog's timeout sentinel. Those rows are still a recording of something that
ran (or failed to), so they are read, but as an elimination rather than a score:
where every measured variant of one kernel failed, an arm that leaves that kernel
is off the kernel-set ballot while another arm remains. Still evidence, still no
preference — the alternative is that an all-failed kernel has no ``ok`` row,
therefore no evidence at all, and falls through to the prior as though nothing
were known about it. That is how DeepSeek-V4's post block kept a fused arm whose
every benched variant hung.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from functools import lru_cache
from typing import TYPE_CHECKING, NamedTuple

from emmy.compiler.graph import Graph
from emmy.compiler.pipeline.fork import iter_leaves, leaf_for, leaf_knobs, parallel_descent_rows
from emmy.compiler.pipeline.knob import schedule_pin_fingerprint
from emmy.compiler.pipeline.search.features import Featurizer
from emmy.compiler.wire import kernel_identity

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline.pipeline import ForkPoint


def tile_identity(knobs: dict) -> frozenset:
    """The blocklist key for a pick — its canonical tuning-knob view
    (:func:`~emmy.compiler.pipeline.knob.tuning_knob_items`: marker booleans dropped,
    values stringified) as a hashable set.
    Computed identically for a greedy leaf's fork knobs (or a splice's decision knobs) and
    for the trace entry of the pick a retry blocklists — the same dict, read at the fork and
    recorded by the resolve — so :func:`greedy_decide` can skip a leaf that already failed
    ``validate(ctx)`` downstream (the smem / thread-budget gate). The terminal node's own knob
    row is NOT that key: a kernel-stage pass can stamp it with a policy knob no leaf spells."""
    from emmy.compiler.pipeline.knob import tuning_knob_items  # noqa: PLC0415

    return frozenset(tuning_knob_items(knobs))


def _tile_blocked(fork_knobs: dict, blocked: set[frozenset]) -> bool:
    """True if a leaf's complete knob row matches a blocklisted tile. Only a leaf
    fork carries every identity knob, so a partial (branch) fork — whose identity
    is a strict subset — never equals a full-row entry and is never skipped."""
    return tile_identity(fork_knobs) in blocked


# ---------------------------------------------------------------------------
# ``greedy_decide`` — the greedy pick as a ``Run.resolve`` decide callback.
# ``Pipeline.run`` routes through this.
# ---------------------------------------------------------------------------

# Sentinel distinguishing "load the global prior lazily on the first fork"
# (the ``Pipeline.run`` default) from an explicitly injected prior — which may
# legitimately be ``None`` (= no prior, option-0 emission order).
_LOAD_PRIOR = object()


@lru_cache(maxsize=1)
def _load_prior_cached(path_str: str):  # noqa: ARG001 — the arg is the cache key
    """The prior for one weights artifact path — the process-wide memo behind
    :func:`_load_prior_safe`. ``maxsize=1`` evicts on any key change. The deploy path only
    *reads* this prior (``mean_scores_features``), so one shared instance is safe across the
    ~96 program compiles of a serve boot."""
    from emmy.compiler.pipeline.search.prior import load_prior  # noqa: PLC0415

    return load_prior()


def _load_prior_safe():
    """Load the one prior, memoized per process on the offline weights path. Best-effort: any
    load failure → ``None`` → emission order (option-0) — a bad/missing prior must never break
    compile."""
    try:
        from emmy import config  # noqa: PLC0415

        return _load_prior_cached(str(config.offline_path() or ""))
    except Exception:  # noqa: BLE001
        return None


@lru_cache(maxsize=1)
def _load_placement_prior():
    """The placement prior the shipped ``weights/placement.json`` names, memoized per process; ``None``
    when the file is absent or does not load, and then every kernel-set fork no measured arm decides takes its first
    arm."""
    from emmy.compiler.pipeline.search.prior import OfflinePrior  # noqa: PLC0415
    from emmy.compiler.pipeline.search.prior.offline import default_file  # noqa: PLC0415

    try:
        prior = OfflinePrior(path=str(default_file("placement")))
    except Exception:  # noqa: BLE001
        return None
    return prior if prior.space == "placement" else None


def _kernel_set_pick(fp: ForkPoint, prior, failed: dict) -> object:
    """The placement prior's argmin over a kernel-set fork's arms — keep the kernel whole and every cut, split or
    layout the pass offers — each featurized from the kernels it leaves (``Featurizer.features``), exactly as the
    arms of the placement dataset the prior was fit on (``ranking.walk_placement``). An arm that leaves a kernel
    whose every measured variant failed (``failed``, :class:`_Measured`) is off the ballot while another arm
    remains. With no ``prior`` the first arm left wins, and the first of equally scored arms wins."""
    from emmy.compiler.pipeline.pipeline import _is_structural_option  # noqa: PLC0415
    from emmy.compiler.pipeline.search.features import kernel_pieces  # noqa: PLC0415

    leaves = fp.flat()
    if len(leaves) == 1:
        return leaves[0]  # a pin, or legality, left one arm: nothing to rank
    root = fp.root_op.with_io(fp.match.graph, fp.match.root)
    pieces = [_leaf_graph(o) if _is_structural_option(o) else root for o in leaves]
    live = [i for i, left in enumerate(pieces) if not any(kernel_identity(op) in failed for op, _ in kernel_pieces(left))]
    live = live or list(range(len(leaves)))
    if prior is None:
        return leaves[live[0]]
    featurizer = Featurizer.of(fp.ctx)
    scores = prior.mean_scores_features([featurizer.features(root, leaf_knobs(leaves[i]), pieces=pieces[i]) for i in live])
    return leaves[live[min(range(len(live)), key=scores.__getitem__)]]


def _find_decided_leaf(fp, want: dict) -> object | None:
    """The leaf carrying exactly the memoized row ``want`` — the row-directed descent
    (``ForkPoint.find``: the schedule root re-sourced to the row, so the walk is one path), held
    to an exact match at the leaf. ``None`` when no leaf matches — emission drift between two
    offers of one key — and the caller re-decides."""
    hit = fp.find(want)
    return hit[0] if hit is not None and hit[1] == want else None


def _leaf_graph(leaf: object) -> Graph:
    """The ``Graph`` behind a raw, concrete, or deferred structural leaf."""
    if isinstance(leaf, Graph):
        return leaf
    option = getattr(leaf, "option", None)
    return option if option is not None else leaf.expand()[0]


def _decision_key(fp: ForkPoint, blocked: dict | None) -> tuple | None:
    """The decision memo's key for one schedule fork, or ``None`` where the memo does not apply.

    GREEDY-ONLY and scoped to one factory call (one compile attempt), because a decision is a
    CONCLUSION over evidence — MCTS must explore, and evidence may move between attempts. Within
    one attempt the pick is deterministic, so N same-shape kernels — 28 identical per-layer
    matmuls — decide once and the rest replay by tree descent instead of a flatten-and-score.

    ``TileOp``-rooted forks only, keyed on the enumeration's MINTED pool identity
    (:attr:`~emmy.compiler.pipeline.fork.Fork.pool_id` — the enumeration's minted stamp: the
    variant key + hints + pins + the sample identity). One minting site, one spelling; the memo
    fails safe on anything the stamp cannot see, because a replayed row that no longer decodes
    (``_find_decided_leaf`` → ``None``) simply re-decides. A fork carrying no stamp
    (offered outside the schedule enumeration) falls back to the kernel's variant key
    (``identity_key`` with io + knobs) + pins. The rule identity separates two
    forks offered on one op, and the node's blocklist CONTENT keys the validate-retry path — a
    retry with a blocked tile is a different decision."""
    from emmy.compiler.ir.tile.ir import TileOp  # noqa: PLC0415

    if not isinstance(fp.root_op, TileOp):
        return None
    pid = next((p for o in fp.variants if (p := getattr(o, "pool_id", None)) is not None), None)
    rule = fp.match.rule
    node_blocked = blocked.get(fp.node_id) if blocked else None
    return (
        getattr(getattr(rule, "pass_", None), "name", None),
        getattr(rule, "name", None),
        pid if pid is not None else (fp.root_op.identity_key(with_io=True, with_knobs=True), schedule_pin_fingerprint()),
        frozenset(node_blocked) if node_blocked else frozenset(),
    )


# Process-wide memo for the built DB index, keyed on (db path, mtime, context key, card).
# The index depends only on the DB file, the card and cc+nvcc-flags (NOT the
# op shape — ``structural_key`` folds neither), so for a serve boot it is identical
# across all ~96 program compiles; without this the 527 MB perf scan reran each time.
# Bounded to the current key (cleared on miss), like ``_load_prior_cached``.
_DB_INDEX_CACHE: dict = {}


def _db_measured_index(db, ctx) -> _Measured:
    """Caching wrapper over :func:`_db_measured_index_build` — memoizes the built index per
    process on ``(db path, mtime, context key, card)``, invalidated when the DB file's mtime
    changes (a golden import into it changes it). An in-memory DB (no ``_path``) or an unstatable
    file bypasses the cache and rebuilds. Best-effort throughout: a failed key computation just
    rebuilds."""
    path = getattr(db, "_path", None)
    if db is not None and path is None:
        return _db_measured_index_build(db, ctx)
    try:
        # Stat the main file AND its ``-wal`` sidecar: in WAL mode a ``record_perf``
        # commit can land in the WAL without bumping the main file's mtime, so a
        # main-mtime-only key could serve a stale index to a same-process
        # write-then-read (the tune lane). ``os.stat`` on a missing WAL → skip it.
        if path is not None:
            wal = path.with_name(path.name + "-wal")
            mtime = (path.stat().st_mtime_ns, wal.stat().st_mtime_ns if wal.exists() else 0)
        else:
            mtime = None
        key = (str(path), mtime, ctx.structural_key(), ctx.hardware_id())
    except Exception:  # noqa: BLE001 — any key-build failure → just rebuild uncached
        return _db_measured_index_build(db, ctx)
    hit = _DB_INDEX_CACHE.get(key)
    if hit is not None:
        return hit
    index = _db_measured_index_build(db, ctx)
    _DB_INDEX_CACHE.clear()  # keep only the current (path, mtime, keys)
    _DB_INDEX_CACHE[key] = index
    return index


class _Measured(NamedTuple):
    """One evidence scan's answers, because the scan is expensive and both come from the same rows.

    ``ok`` RANKS — the measured schedule rows a pick argmins over, by the kernel's exact identity.
    ``failed`` DISQUALIFIES — the kernels whose every measured variant failed, a different kind of answer
    that cannot be expressed as a latency. A kernel-set decision is no row here: it is a routing
    row on the exact kernel, priced per kernel from the pieces' own measurements
    (:meth:`SearchDB.priced_arms`) when the fork is decided."""

    ok: dict[str, list[tuple[dict, float]]]
    failed: dict[str, list[float]]


_EMPTY_MEASURED = _Measured({}, {})


def _db_measured_index_build(db, ctx) -> _Measured:
    """Every measured row this compile may deploy, split into what ranks and what disqualifies: the
    DB's CUDA ``perf`` rows for this compile's regime — the tune's own and the golden rows in scope,
    imported among them before the compile picks (``evidence.evidence_db``); ``db`` may be
    ``None`` on a probe that reads no evidence.

    Rows are indexed by the exact identity of the kernel they measured — the one evidence join: the same
    identity is the same kernel, and nothing else about a kernel takes part (knob values are stringified
    because perf knobs round-trip JSON). One context key is sufficient: tune measures in the deployable regime, and
    ``Context.structural_key`` gives that regime one key however its flags are spelled. Rows from a
    deliberately non-deployable compile key elsewhere and are not consulted, and neither are rows
    another card measured (``SearchDB.iter_perf`` reads this card's rows in this regime).

    A non-``ok`` row is evidence too — the bench watchdog measured that variant not finishing — but
    it is evidence a ranker cannot use, since its sentinel latency is a timeout constant rather
    than a speed. It lands in ``failed`` instead, and only where NO variant of that kernel was
    measured ``ok``: one surviving row means the kernel is realizable and merely has bad rows.

    Best-effort: any failure returns an empty index so deploy falls back to the prior.
    """
    index: dict[str, list[tuple[dict, float]]] = {}
    survived: set[str] = set()
    failures: dict[str, list[float]] = {}
    try:
        for row in db.iter_perf(ctx, backend="cuda") if db is not None else ():
            if row.status != "ok":
                failures.setdefault(row.kernel, []).append(float(getattr(row.stats, "median", 0.0) or 0.0))
                continue
            survived.add(row.kernel)
            if row.stats.median <= 0:
                continue
            index.setdefault(row.kernel, []).append(({k: str(v) for k, v in row.knobs.items()}, float(row.stats.median)))
    except Exception:  # noqa: BLE001 — an evidence consult failure must never break compile
        logger.debug("measured-evidence index build failed", exc_info=True)
        return _EMPTY_MEASURED
    return _Measured(index, {kernel: us for kernel, us in failures.items() if kernel not in survived})


def _db_measured_pick(
    measured: list[tuple[dict, float]],
    rows: list[dict],
    *,
    exact_families: frozenset[str] = frozenset(),
) -> tuple[int, float] | None:
    """Measured-evidence argmin over candidate knob rows against ``measured``, the rows the index holds
    for the kernel the candidates schedule — the prefix-consistency contract of
    :func:`~emmy.compiler.pipeline.knob.evidence_row_vouches` (every knob the candidate specifies must
    match the measured row; undecided knobs are free). Every indexed row was measured in this compile's
    regime, so the argmin over matching rows is the answer. This keeps a config tune measured fastest
    from losing deploy to an unmeasured model extrapolation.
    """
    from emmy.compiler.pipeline.knob import canonical_row_key, evidence_row_vouches  # noqa: PLC0415

    row_key: dict[int, tuple] = {}  # i → canonical_row_key(rows[i]), computed at most once

    def key_of(i: int) -> tuple:
        if i not in row_key:
            row_key[i] = canonical_row_key(rows[i])
        return row_key[i]

    def better(us: float, i: int, cur: tuple[int, float] | None) -> bool:
        # Tie on µs (one measured row matching several candidates) breaks by the
        # candidates' canonical content, never their enumeration order.
        return cur is None or us < cur[1] or (us == cur[1] and key_of(i) < key_of(cur[0]))

    best: tuple[int, float] | None = None
    for i, cand in enumerate(rows):
        cand_tun = {k: str(v) for k, v in cand.items()}
        for row_tun, us in measured:
            # A row counts as evidence when it matches every knob the candidate
            # has decided; undecided knobs are free (``evidence_row_vouches``).
            if not evidence_row_vouches(cand_tun, row_tun, exact_families=exact_families):
                continue
            if better(us, i, best):
                best = (i, us)
    return best


def _warn_disjoint_evidence(measured: list[tuple[dict, float]], node_id: str, n_rows: int) -> None:
    """Warn when a fork's candidate set is DISJOINT from its measured evidence:
    the DB holds rows for this kernel, yet :func:`_db_measured_pick` matched none of them
    against any offered candidate. That condition is exactly "the tune measured a schedule tier
    the deploy did not offer" — the model then extrapolates over an
    evidence-free candidate set, which shipped gemma o_proj on a scalar tile
    16x its own measured mma rows (the stale-placeholder offer gap). A cold
    compile (no rows for the kernel at all) stays silent — extrapolation
    is expected there."""
    if measured:
        logger.warning(
            "deploy: node %r has %d measured DB row(s) for its kernel, but none matches any of the "
            "%d offered candidates — the tune measured a schedule tier this compile did not offer; falling back to "
            "the model prediction. Investigate the enumeration (offer gates) for this kernel.",
            node_id,
            len(measured),
            n_rows,
        )


def _schedule_fork(fp: ForkPoint) -> bool:
    """Whether ``fp`` decides one kernel's schedule (its options carry the enumeration's ``pool_id``
    stamp) rather than which kernels exist (the cut pass's placement / split fork)."""
    return any(getattr(o, "pool_id", None) is not None for o in fp.options)


class EvidenceError(RuntimeError):
    """Raised under strict evidence (``config.strict_evidence``) when a fork must be decided and
    no measured row — tune DB or golden — vouches for any of its candidates."""


def _require_evidence(fp: ForkPoint, why: str) -> None:
    """Under strict evidence, refuse to decide ``fp`` by anything but a measurement."""
    from emmy import config  # noqa: PLC0415

    if not config.strict_evidence():
        return
    name = getattr(fp.root_op, "name", None) or fp.node_id
    raise EvidenceError(
        f"strict evidence: kernel {name!r} (node {fp.node_id!r}) has no measured evidence for its {fp.match.rule.name} fork "
        f"({why}) — record it with `emmy run --golden PATH --bench`, tune it, or load a golden that covers it"
    )


def _route_candidates(fp: ForkPoint, index: _Measured, db) -> list[tuple[object, float]]:
    """The measured arms at this kernel-set fork: one ``(option, µs)`` per measured row of
    the kernel that spells an arm on the ballot
    (:func:`~emmy.compiler.pipeline.search.pins.spelled_arm`) — a schedule row the fused /
    unsplit arm, since the kernel it decorates ran that way — and one per kernel-set decision the
    DB stores on this exact kernel that its pieces' rows price at the fork's bindings
    (:meth:`SearchDB.priced_arms`). The option is the cut pass's own offer; the pieces it mints
    are brand-new kernels whose own forks consult their own rows. A schedule fork has none."""
    from emmy.compiler.ir.tile import TileOp  # noqa: PLC0415
    from emmy.compiler.pipeline.knob import family_of  # noqa: PLC0415
    from emmy.compiler.pipeline.pipeline import _structural_domain  # noqa: PLC0415
    from emmy.compiler.pipeline.search.pins import spelled_arm  # noqa: PLC0415
    from emmy.compiler.wire import kernel_bindings  # noqa: PLC0415

    root = fp.root_op
    if not isinstance(root, TileOp) or root.op is None or _schedule_fork(fp):
        return []
    if _structural_domain(fp.options) not in (("PLACE",), ("REDUCE",)):
        return []
    kernel = root.identity_key(structural=False, with_io=True)
    measured = list(index.ok.get(kernel, ()))
    if db is not None and kernel is not None:
        measured.extend(
            (arm, us)
            for arm, us in db.priced_arms(fp.ctx, kernel, bindings=kernel_bindings(root))
            if not any(family_of(key) == "LAYOUT" for key in arm)
        )
    out: list[tuple[object, float]] = []
    for row, us in measured:
        arm = spelled_arm(fp.options, row)
        if arm is not None:
            out.append((arm[0], us))
    if db is not None:
        out.extend((splice, us) for splice in fp.splices if (us := _pieces_price(splice, fp.ctx, db)) is not None)
    return out


def _layout_candidates(fp: ForkPoint, index: _Measured, db) -> list[tuple[object, float]]:
    """Price each storage-layout arm from its measured kernel or a measured route from it."""
    from emmy.compiler.ir.tile import TileOp  # noqa: PLC0415
    from emmy.compiler.pipeline.knob import family_of  # noqa: PLC0415
    from emmy.compiler.pipeline.pipeline import _is_structural_option, _structural_domain  # noqa: PLC0415
    from emmy.compiler.pipeline.search.slice import single_node_graph  # noqa: PLC0415
    from emmy.compiler.wire import kernel_bindings  # noqa: PLC0415

    if _structural_domain(fp.options) != ("LAYOUT",):
        return []
    out = []
    for option in fp.options:
        graph = _leaf_graph(option) if _is_structural_option(option) else single_node_graph(fp.match.graph, fp.node_id)
        nodes = [node for node in graph.nodes.values() if isinstance(node.op, TileOp)]
        if len(nodes) != 1:
            continue
        tile = nodes[0].op.with_io(graph, nodes[0])
        kernel = tile.identity_key(structural=False, with_io=True)
        if kernel is None:
            continue
        times = [us for _, us in index.ok.get(kernel, ())]
        if db is not None:
            times.extend(
                us
                for arm, us in db.priced_arms(fp.ctx, kernel, bindings=kernel_bindings(tile))
                if not any(family_of(key) == "LAYOUT" for key in arm)
            )
        if times:
            out.append((option, min(times)))
    return out


def _pieces_price(splice: object, ctx: Context, db) -> float | None:
    """A splice's price as the sum of its pieces' best rows (:meth:`SearchDB.best_per_op_time`), each piece
    named by the exact identity it carries once spliced, all-or-nothing — the price a routing row gets, for
    an arm whose pieces were benched with no routing row recorded (a sweep's split)."""
    from emmy.compiler.wire import kernel_bindings  # noqa: PLC0415

    fragment = _leaf_graph(splice)
    total = 0.0
    for node in fragment.nodes.values():
        if node.op.identity_key(with_io=True, with_knobs=True) is None:
            continue
        piece = node.op.with_io(fragment, node)
        kernel = piece.identity_key(structural=False, with_io=True)
        us = db.best_per_op_time(ctx, kernel, bindings=kernel_bindings(piece)) if kernel is not None else None
        if us is None:
            return None
        total += us
    return total


def _direct_measured_pick(fp: ForkPoint, blocked, db_index: dict) -> tuple[object, dict, float] | None:
    """Descend directly to the fastest offered tune-DB row.

    Evidence rows already spell complete schedules, so scoring branch representatives would be
    both slower and less exact. Expansions are memoized across records; each tree branch is opened
    at most once during the lookup.
    """
    from emmy.compiler.pipeline.knob import canonical_row_key  # noqa: PLC0415

    node_blocked = blocked.get(fp.node_id) if blocked else None

    def offered(records):
        # ``leaf_for`` descends only the branches that admit the row (``Fork.admits``): a family
        # the row leaves undecided is free here as everywhere, so a partial row still reaches its
        # leaf whatever the pool size — the one path that does not depend on the cold-pool budget.
        ordered = sorted(records, key=lambda item: (item[1], canonical_row_key(item[0])))
        skip = (lambda knobs: _tile_blocked(knobs, node_blocked)) if node_blocked is not None else None
        for record, price in ordered:
            if (hit := fp.find(record, skip=skip)) is not None:
                return hit[0], hit[1], float(price)
        return None

    records = db_index.get(kernel_identity(fp.root_op)) if db_index else None
    return offered(records) if records else None


#: Leaves scored per batch in the streamed scan: large enough to amortize CatBoost's per-``predict``
#: overhead (its batched surface exists because per-row calls pay it N times), small enough that the
#: transient row dicts stay bounded — the flat 486k-row pools this replaces held ~GBs of them at once.
_CHUNK = 4096

#: The cold-pool budget: a pool whose minted size bound (``Fork.pool_bound`` — Π of the per-node
#: option tuples, legality only shrinks it) exceeds this is not walked at full length on a cold
#: deploy. The research-class fused terms enumerate millions of legal schedules, and a model
#: argmin over all of them buys nothing a bounded sample doesn't: the cold pick only needs a
#: REASONABLE kernel — the optimal one comes from evidence (a tune, a benched golden row), which
#: the measured descent deploys directly regardless of pool size.
_POOL_BUDGET = 65_536


def _descent_sample(options, pool_id: str, node_blocked) -> list[dict]:
    """The knob rows of up to ``EMMY_POOL_DRAW`` complete leaves of a cold pool, drawn by
    :func:`~emmy.compiler.pipeline.fork.parallel_descent_rows` seeded on the pool identity on
    ``EMMY_WORKERS`` processes, blocklisted rows retried. Duplicates are kept (a repeat costs a scoring
    slot, never a wrong pick). Structural options never appear here — the caller samples only the variant side."""
    from emmy import config  # noqa: PLC0415

    skip = None if node_blocked is None else (lambda leaf: _tile_blocked(leaf_knobs(leaf), node_blocked))
    return parallel_descent_rows(options, draw=config.pool_draw(), seed=pool_id, skip=skip, workers=config.workers())


def _argmin(scores: list[float], rows: list[dict]) -> tuple[int, float]:
    """The lowest score's index and score, ties broken by :func:`~emmy.compiler.pipeline.knob.canonical_row_key`
    (candidate content, never enumeration order — an order-broken tie flips the deployed kernel per boot). Two
    stages: the key is a canonicalizing sort over the whole row, so it is spelled only for the tied rows."""
    from emmy.compiler.pipeline.knob import canonical_row_key  # noqa: PLC0415

    lo = min(scores)
    ties = [j for j, score in enumerate(scores) if score == lo]
    return (ties[0] if len(ties) == 1 else min(ties, key=lambda j: canonical_row_key(rows[j]))), lo


def _stream_tiers(fp: ForkPoint, the_prior, node_blocked, db_idx: dict) -> tuple[object, dict | None, float | None, str | None]:
    """The deploy evidence hierarchy over a schedule pool, in ONE streamed walk.

    The lazy walk is not free — each branch expansion re-spells its schedule step, and on the
    research-class pools (a 486k-row explicit-mask softmax term) the walk itself costs minutes —
    so this scan walks exactly once, like the flatten it replaces, and evaluates every source
    chunk-wise as the leaves go by: the evidence index's measured best (tune DB rows and golden
    rows) and the model score, each folded into its own running best. The PRIORITY is applied
    after the stream ends (index > model); the one behavioral trade is that the model's scoring runs even
    when a later chunk turns up evidence — acceptable because measured forks are normally decided
    upstream by the direct measured descent, never here. The pick is EXACTLY the flattened argmin: every source
    breaks ties by candidate content (``canonical_row_key``), never enumeration order, so
    per-chunk winners folded through a running ``(price, key)`` min are chunk-invariant.

    A pool whose minted size bound exceeds :data:`_POOL_BUDGET` is not walked: the scan ranks a
    deterministic drawn subset instead (:func:`_descent_sample` — seeded uniform descents, legal
    complete rows only). Above the budget the pick is the argmin over the draw, not the pool —
    the accepted cold-deploy trade: a reasonable kernel now, the optimal one from evidence (the
    measured descent reaches it directly whatever the pool size, and a bad cold pick is fixed by
    measuring, as ever).

    ``(leaf, None, None, None)`` is the degenerate plain return (≤1 leaf, or every leaf
    blocklisted — no score, no decision memo); ``(leaf, knobs, price, tier)`` is the ranked pick,
    ``tier`` being ``"evidence"`` (a measured row decided) or ``"model"``."""
    from emmy.compiler.pipeline.knob import canonical_row_key  # noqa: PLC0415
    from emmy.compiler.pipeline.pipeline import NO_OPTION  # noqa: PLC0415

    featurizer = Featurizer.of(fp.ctx)
    measured = db_idx.get(kernel_identity(fp.root_op), []) if db_idx else []
    # Per-source running bests: (price, canonical_row_key, leaf, knobs).
    best_db: tuple | None = None
    best_model: tuple | None = None

    def fold(best: tuple | None, chunk: list, got: tuple[int, float] | None) -> tuple | None:
        if got is None:
            return best
        i, price = got
        key = (price, canonical_row_key(chunk[i][1]))
        if best is None or key < (best[0], best[1]):
            return (price, key[1], chunk[i][0], chunk[i][1])
        return best

    def scan(chunk: list) -> None:
        nonlocal best_db, best_model
        rows = [knobs for _, knobs in chunk]
        if measured:
            best_db = fold(best_db, chunk, _db_measured_pick(measured, rows))
        scores = the_prior.mean_scores_features([featurizer.features(fp.root_op, knobs) for knobs in rows])
        best_model = fold(best_model, chunk, _argmin(scores, rows))

    opts = fp.options
    # The cold-pool budget: a pool whose minted bound exceeds _POOL_BUDGET is sampled by seeded
    # descents instead of walked — the sources below then rank the drawn complete rows exactly as
    # they would the full pool. An empty draw fails explicitly: walking the full oversized pool
    # would silently discard the bound. Report the empty subtree to the resolver so it can keep
    # walking the current rule batch; returning a partial branch would violate the prior's
    # complete-row contract.
    bound = next((b for o in opts if (b := getattr(o, "pool_bound", None)) is not None), None)
    drawn = None
    if bound is not None and bound > _POOL_BUDGET:
        pid = next((p for o in opts if (p := getattr(o, "pool_id", None)) is not None), "")
        drawn = _descent_sample(opts, pid, node_blocked)
        if not drawn:
            return NO_OPTION, None, None, None
    n_leaves = n_live = 0
    first: tuple | None = None
    chunk: list = []
    # A drawn row comes back without its leaf (``parallel_descent_rows``): only the one picked is built.
    entries = ((None, row) for row in drawn) if drawn is not None else ((leaf, leaf_knobs(leaf)) for leaf in iter_leaves(opts))
    for leaf, knobs in entries:
        n_leaves += 1
        if first is None:
            first = (leaf, knobs)
        if node_blocked is not None and _tile_blocked(knobs, node_blocked):
            continue
        n_live += 1
        chunk.append((leaf, knobs))
        if len(chunk) >= _CHUNK:
            scan(chunk)
            chunk = []
    if n_leaves == 0:
        return NO_OPTION, None, None, None

    def built(leaf, knobs):
        if leaf is not None:
            return leaf
        hit = leaf_for(opts, knobs)
        if hit is None or hit[1] != knobs:
            raise RuntimeError(f"drawn row {knobs} does not build back to its own leaf at {fp.node_id}")
        return hit[0]

    if n_leaves == 1 or n_live == 0:
        return built(*first), None, None, None
    if chunk:
        scan(chunk)
    if best_db is not None:
        return built(best_db[2], best_db[3]), best_db[3], best_db[0], "evidence"
    _warn_disjoint_evidence(measured, fp.node_id, n_live)
    return built(best_model[2], best_model[3]), best_model[3], best_model[0], "model"


def greedy_decide(
    blocked: dict[str, set[frozenset]] | None = None,
    *,
    prior: object = _LOAD_PRIOR,
    placement_prior: object = _LOAD_PRIOR,
    db: object | None = None,
) -> Callable[[ForkPoint], object]:
    """The greedy compile pick as a :meth:`Run.resolve` ``decide`` callback, in two kinds of decision that never
    price one through the other.

    A **kernel-set fork** — a placement cut, a split, a storage layout (``pins.KERNEL_SET_DOMAINS``) — is decided
    from what its arms are: the fastest measured arm (:func:`_route_candidates`, :func:`_layout_candidates`),
    else the ``placement_prior`` — the shipped placement weights, loaded lazily — over every arm
    (:func:`_kernel_set_pick`), else the first arm. No arm is scheduled to decide it: the pieces an arm mints are
    brand-new kernels, decided at their own forks.

    A **schedule fork** descends directly to exact evidence when available, otherwise streams the complete rows
    in bounded chunks (:func:`_stream_tiers`), skips ``blocked`` tile identities, and takes the prior's global
    argmin. The prior is the ``OfflinePrior`` ``load_prior`` builds. With no prior at all (a failed load, or the
    explicit ``prior=None`` emission-order resolve) every fork falls to emission order (option-0, first leaf).
    Stamps the pick's measured or predicted µs on ``fp.score``, so the resolve trace carries the per-fork price.

    ``blocked`` (``{node_id: {tile_identity, ...}}``) lists the picks a previous compile
    attempt made at a node that then failed to lower — a leaf's knob row, or a splice's
    decision knobs — and ``Pipeline.run`` retries the deterministic resolution with them
    blocklisted so the next best non-blocked leaf is picked, or the fork decides again without
    the withdrawn splice (the analogue of how ``tune`` benches-and-skips an unviable tile;
    greedy benches nothing, so the validity signal must come from the retry)."""
    from emmy.compiler.pipeline.pipeline import _is_structural_option  # noqa: PLC0415
    from emmy.compiler.pipeline.search.pins import KERNEL_SET_DOMAINS  # noqa: PLC0415

    #: The DECISION memo (:func:`_decision_key` → the winning row + its price), one compile attempt. A repeat
    #: offer replays by :func:`~emmy.compiler.pipeline.fork.find_leaf` descent; only genuinely new (key,
    #: blocklist) states pay the stream-and-score.
    decisions: dict = {}
    loaded = prior is not _LOAD_PRIOR
    the_prior = prior if loaded else None
    placement = placement_prior if placement_prior is not _LOAD_PRIOR else _load_placement_prior()
    # Lazily-built per-compile measured-evidence index (needs a fork point's ctx for the
    # context keys): the tune DB's rows and the golden rows in scope, one index.
    # ``None`` sentinel = not built yet.
    db_state: list = [None]

    def db_index() -> dict:
        return (db_state[0].ok if db_state[0] is not None else None) or {}

    def decide(fp: ForkPoint) -> object:
        # A retired cut: a splice whose decision identity the strategy blocklisted at this node
        # (the same identity its trace entry carries) is withdrawn, and the fork decides again over
        # what remains — the one structural pick retired, every other arm still on the ballot.
        node_blocked = blocked.get(fp.node_id) if blocked else None
        if not node_blocked or not fp.splices:
            return pick(fp)
        from emmy.compiler.pipeline.pipeline import _choice_knobs  # noqa: PLC0415

        live = [o for o in fp.options if not (_is_structural_option(o) and tile_identity(_choice_knobs(o, o, fp.root_op)) in node_blocked)]
        narrowed = replace(fp, options=live)
        chosen = pick(narrowed)
        fp.score = narrowed.score
        return chosen

    def kernel_set(fp: ForkPoint, index: _Measured, domain: tuple) -> object:
        """A kernel-set fork: every measured row of this kernel spells one offered arm, and the fastest measured
        arm wins (a tie breaks by the arm's content, never by emission order); otherwise the arms are ranked by
        what they are, which strict evidence refuses."""
        from emmy.compiler.pipeline.knob import canonical_row_key  # noqa: PLC0415

        measured = _layout_candidates(fp, index, db) if domain == ("LAYOUT",) else _route_candidates(fp, index, db)
        if measured:
            chosen, us = min(measured, key=lambda candidate: (candidate[1], canonical_row_key(leaf_knobs(candidate[0]))))
            fp.score = us
            return chosen
        if len(fp.options) > 1:
            _require_evidence(fp, "no measured row spells a kernel-set arm")
        # No schedule prior on this resolve (a failed load, or the emission-order re-resolve) ranks no arm either.
        return _kernel_set_pick(fp, placement if the_prior is not None else None, index.failed)

    def pick(fp: ForkPoint) -> object:
        nonlocal loaded, the_prior
        from emmy.compiler.pipeline.pipeline import _structural_domain  # noqa: PLC0415

        if db_state[0] is None:
            db_state[0] = _db_measured_index(db, fp.ctx)
        index: _Measured = db_state[0]
        if not loaded:
            loaded = True
            the_prior = _load_prior_safe()
        if not _schedule_fork(fp) and (domain := _structural_domain(fp.options)) in KERNEL_SET_DOMAINS:
            return kernel_set(fp, index, domain)
        dkey = _decision_key(fp, blocked)
        if dkey is not None and dkey in decisions:
            want, price = decisions[dkey]
            found = _find_decided_leaf(fp, want)
            if found is not None:
                fp.score = price
                return found
        if dkey is not None and _schedule_fork(fp):
            picked = _direct_measured_pick(fp, blocked, db_index())
            if picked is not None:
                leaf, row, price = picked
                fp.score = price
                decisions[dkey] = (dict(row), price)
                return leaf
        if the_prior is None:
            # No prior on this resolve — a failed ``load_prior`` (corrupt/unreadable
            # checkpoint) or ``Pipeline.run``'s explicit emission-order fallback
            # (``prior=None``): emission order (option-0, first leaf).
            leaves = fp.leaves()
            first = next(leaves)
            if next(leaves, None) is not None:
                _require_evidence(fp, "no prior loaded; emission order would decide")
            return first
        # Greedy benches nothing, so it must pick the globally best COMPLETE
        # tile, not a partial branch (the prior is blind at a partial ``BM/BN``
        # branch: ``knob_features`` can't compute the tile's area / occupancy
        # until ``FM/FN`` exist). ``_stream_tiers`` scores those complete rows
        # off the lazy walk in bounded chunks — the pick equals the flattened
        # scoring's argmin exactly (content-keyed tie rules make the running
        # min chunk-invariant), without ever retaining the O(pool) leaf and
        # row lists that made a 486k-row cold pool an OOM.
        node_blocked = blocked.get(fp.node_id) if blocked else None
        leaf, row, price, tier = _stream_tiers(fp, the_prior, node_blocked, db_index())
        if row is None:
            return leaf  # degenerate pool (≤1 leaf / all blocklisted): plain, unscored return
        if tier == "model":
            _require_evidence(fp, "no measured row vouches for any offered candidate")
            fp.score = price
        if dkey is not None:
            decisions[dkey] = (dict(row), price)
        return leaf

    return decide
