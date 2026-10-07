"""Focused tests for greedy schedule-space traversal."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline.fork import DeferredFork, Fork, iter_leaves, leaf_knobs
from emmy.compiler.pipeline.knob import canonical_row_key
from emmy.compiler.pipeline.pipeline import NO_OPTION, ForkPoint
from emmy.compiler.pipeline.search.features import Featurizer
from emmy.compiler.pipeline.search.policy import greedy
from emmy.compiler.pipeline.search.policy.greedy import (
    EvidenceError,
    _db_measured_index_build,
    _direct_measured_pick,
    _require_evidence,
    _route_candidates,
    _stream_tiers,
    tile_identity,
)
from tests.compiler.pipeline.search.helpers import StubKernel
from tests.compiler.terms import projection


def _kernel(identity: str | None = "k") -> StubKernel:
    """A stand-in for the kernel a synthetic fork decides: no knobs, one stamp, and the exact identity its
    measured rows are filed under (:func:`_identities` reads it back)."""
    kernel = StubKernel({"S_shape": 128.0})
    kernel.knobs, kernel.identity = {}, identity
    kernel.identity_key = lambda **_kw: identity
    return kernel


@pytest.fixture(autouse=True)
def _identities(monkeypatch) -> None:
    """The pick reads a kernel's identity off its tile (``wire.kernel_identity``); a stub names its own."""
    from emmy.compiler.wire import kernel_identity

    monkeypatch.setattr(greedy, "kernel_identity", lambda op: op.identity if isinstance(op, StubKernel) else kernel_identity(op))


def test_db_measured_index_collects_kernels_whose_every_measured_variant_failed() -> None:
    """A ``bench_fail`` row is evidence too — the watchdog measured that variant not finishing.
    When EVERY measured variant of one structural shape failed, the shape itself is disqualified;
    one surviving ``ok`` variant means only some rows are bad and the shape stays rankable.

    Failures are collected before the placement-route filter the ``ok`` tier applies: a route's
    LATENCY is unattributable without a child-schedule receipt, but a kernel that hung is
    attributable to the kernel whatever route produced it."""

    def row(kernel: str, status: str, us: float, **knobs) -> SimpleNamespace:
        return SimpleNamespace(kernel=kernel, status=status, stats=SimpleNamespace(median=us), knobs=knobs)

    rows = [
        row("doomed", "bench_fail", 2_000_000.0, WORK="t32"),
        row("doomed", "bench_fail", 2_000_000.0, WORK="t64"),
        row("mixed", "bench_fail", 2_000_000.0, WORK="t32"),
        row("mixed", "ok", 7.0, WORK="t64"),
    ]
    db = SimpleNamespace(iter_perf=lambda *_args, **_kwargs: rows)
    ctx = SimpleNamespace(structural_key=lambda: "ctx", gpu_name=None, compute_capability=(8, 9), features=lambda: {"H_opt": 3.0})

    measured = _db_measured_index_build(db, ctx)
    assert "doomed" in measured.failed, "every measured variant of this kernel hit the watchdog"
    assert "mixed" not in measured.failed, "a kernel with one ok variant is not disqualified"
    assert measured.ok == {"mixed": [({"WORK": "t64"}, 7.0)]}


@dataclass(frozen=True)
class _Branch(Fork):
    """A synthetic branch: the knobs it pins and the options below it."""

    knobs: dict
    children: tuple

    def expand(self):
        return list(self.children)


def _tree(rows, materialize) -> _Branch:
    """A two-level tree over ``rows``: one branch per ``TILE`` value, one deferred leaf per row below it."""
    by_tile: dict[str, list[dict]] = {}
    for row in rows:
        by_tile.setdefault(row["TILE"], []).append(row)
    return _Branch(
        {},
        tuple(
            _Branch({"TILE": tile}, tuple(DeferredFork(lambda row=row: materialize(row), dict(row)) for row in group))
            for tile, group in by_tile.items()
        ),
    )


def test_schedule_pick_descends_directly_to_complete_measured_row() -> None:
    materialized = []
    rows = [{"TILE": str(tile), "STAGE": str(stage)} for tile in range(100) for stage in range(100)]
    tree = _tree(rows, lambda row: materialized.append(row))

    point = ForkPoint(
        match=SimpleNamespace(root_node_id="node", graph=None),
        options=[tree],
        root_op=_kernel(),
        ctx=SimpleNamespace(features=lambda: {"H_opt": 3.0}),
    )
    index = {"k": [({"TILE": "42", "STAGE": "73"}, 1.25)], "another": [({"TILE": "1", "STAGE": "1"}, 0.5)]}
    leaf, knobs, price = _direct_measured_pick(point, None, index)

    assert knobs == {"TILE": "42", "STAGE": "73"}
    assert leaf.knobs == knobs
    assert price == 1.25
    assert materialized == []


def test_measured_schedule_survives_missing_prior(monkeypatch) -> None:
    point = _point([{"TILE": "0", "STAGE": "0"}, {"TILE": "1", "STAGE": "0"}])
    monkeypatch.setattr(greedy, "_schedule_fork", lambda _fp: True)
    monkeypatch.setattr(greedy, "_decision_key", lambda _fp, _blocked: ("schedule",))
    monkeypatch.setattr(
        greedy,
        "_db_measured_index",
        lambda _db, _ctx: SimpleNamespace(ok={"k": [({"TILE": "1", "STAGE": "0"}, 1.0)]}, failed=set()),
    )
    chosen = greedy.greedy_decide(prior=None, placement_prior=_BarePrior(), db=object())(point)
    assert leaf_knobs(chosen) == {"TILE": "1", "STAGE": "0"}
    assert point.score == 1.0


def test_strict_evidence_refuses_lazy_schedule_without_prior(monkeypatch) -> None:
    point = _point([{"TILE": "0", "STAGE": "0"}, {"TILE": "1", "STAGE": "0"}])
    point.match.rule = SimpleNamespace(name="040_schedule")
    monkeypatch.setenv("EMMY_STRICT_EVIDENCE", "1")
    monkeypatch.setattr(greedy, "_schedule_fork", lambda _fp: True)
    monkeypatch.setattr(greedy, "_decision_key", lambda _fp, _blocked: ("schedule",))
    monkeypatch.setattr(greedy, "_db_measured_index", lambda _db, _ctx: SimpleNamespace(ok={}, failed=set()))
    with pytest.raises(EvidenceError, match="no prior loaded"):
        greedy.greedy_decide(prior=None, placement_prior=_BarePrior(), db=object())(point)


def test_measured_rows_do_not_cross_exact_kernel_identities() -> None:
    """Two kernels of one structure are two kernels: each one's rows price its own candidates, and a kernel
    nothing measured has none."""
    rows = [
        SimpleNamespace(kernel="flat", status="ok", stats=SimpleNamespace(median=17.0), knobs={"WORK": "t32"}),
        SimpleNamespace(kernel="strided", status="ok", stats=SimpleNamespace(median=27.0), knobs={"WORK": "t512"}),
        SimpleNamespace(kernel="flat", status="ok", stats=SimpleNamespace(median=1.0), knobs={"WORK": "t64"}),
    ]
    db = SimpleNamespace(iter_perf=lambda *_args, **_kwargs: rows)
    ctx = SimpleNamespace(gpu_name="card", compute_capability=(7, 0))
    index = _db_measured_index_build(db, ctx)
    candidates = [{"WORK": work} for work in ("t32", "t64", "t512")]

    assert greedy._db_measured_pick(index.ok["strided"], candidates) == (2, 27.0)
    assert greedy._db_measured_pick(index.ok["flat"], candidates) == (1, 1.0)
    assert "unmeasured" not in index.ok


def test_strict_evidence_refuses_a_fork_no_measurement_decides(monkeypatch) -> None:
    point = SimpleNamespace(
        node_id="node", root_op=SimpleNamespace(name="k_linear"), match=SimpleNamespace(rule=SimpleNamespace(name="040_schedule"))
    )
    monkeypatch.delenv("EMMY_STRICT_EVIDENCE", raising=False)
    _require_evidence(point, "nothing measured")  # permissive by default
    monkeypatch.setenv("EMMY_STRICT_EVIDENCE", "1")
    with pytest.raises(EvidenceError, match="k_linear"):
        _require_evidence(point, "nothing measured")


def test_strict_evidence_lets_a_hand_pin_decide_a_kernel_set_fork(monkeypatch) -> None:
    """A pinned route leaves the placement fork one arm. Nothing is predicted or compared there, so
    strict evidence must not refuse it: recording a kernel set under a hand pin with
    ``--strict-evidence`` is how its pieces are proven measured before the routing row exists."""
    from emmy.compiler.pipeline.pipeline import ForkPoint
    from emmy.compiler.pipeline.search.policy.greedy import greedy_decide

    monkeypatch.setenv("EMMY_STRICT_EVIDENCE", "1")
    cut = DeferredFork(materialize=lambda: None, knobs={"PLACE@map.1/map": "cut"}, structural=True)
    fuse = DeferredFork(materialize=lambda: None, knobs={"PLACE": "fuse"})
    ctx = SimpleNamespace(structural_key=lambda: "ctx", gpu_name="", compute_capability=(8, 9), features=lambda: {"H_opt": 3.0})
    match = SimpleNamespace(root_node_id="node", rule=SimpleNamespace(name="030_cut"), graph=None)

    def point(options):
        return ForkPoint(match=match, options=options, root_op=TileOp(op=projection()), ctx=ctx)

    assert greedy_decide(prior=_BarePrior())(point([cut])) is cut
    with pytest.raises(EvidenceError, match="kernel-set arm"):
        greedy_decide(prior=_BarePrior())(point([fuse, cut]))


# ---------------------------------------------------------------------------
# _stream_tiers — the streamed scan must equal the flattened scoring exactly.
# ---------------------------------------------------------------------------


def _score(row: dict) -> float:
    # Deliberately tie-heavy so the content tiebreak (canonical_row_key) decides across chunks.
    return float(int(sum(abs(v) for k, v in row.items() if k.startswith("D_"))) % 4)


class _BarePrior:
    """A prior over feature rows, tie-heavy on the ``D_*`` features the codec rows encode to."""

    def mean_scores_features(self, rows):
        return [_score(r) for r in rows]


def _point(rows):
    tree = _tree(rows, lambda row: (_ for _ in ()).throw(AssertionError("no leaf may materialize during ranking")))
    return ForkPoint(
        match=SimpleNamespace(root_node_id="node", graph=None),
        options=[tree],
        root_op=_kernel(),
        ctx=SimpleNamespace(features=lambda: {"H_opt": 3.0}),
    )


def _rows(n_tile=18, n_stage=7):
    return [{"TILE": str(t), "STAGE": str(s)} for t in range(n_tile) for s in range(n_stage)]


def test_streamed_model_pick_equals_flattened_argmin(monkeypatch) -> None:
    monkeypatch.setattr(greedy, "_CHUNK", 10)  # force many uneven chunks over the 126-leaf pool
    point = _point(_rows())
    got = _stream_tiers(point, _BarePrior(), None, {})
    assert got is not None
    leaf, knobs, price, _tier = got

    flat = [(o, leaf_knobs(o)) for o in list(iter_leaves(point.options))]
    rows = [k for _, k in flat]
    scores = _BarePrior().mean_scores_features([Featurizer({"H_opt": 3.0}).features(point.root_op, k) for _, k in flat])
    best_i = min(range(len(rows)), key=lambda i: (scores[i], canonical_row_key(rows[i])))
    assert knobs == flat[best_i][1]
    # The lazy walk mints fresh (content-equal) leaf objects per expansion, so identity is by row.
    assert leaf_knobs(leaf) == flat[best_i][1]
    assert price == scores[best_i]


def test_streamed_db_tier_outranks_the_model(monkeypatch) -> None:
    monkeypatch.setattr(greedy, "_CHUNK", 10)
    # The measured DB row must win the deploy even though the model scores other rows better
    # (every row with _score == 0.0 beats the measured row's model score).
    db_idx = {"k": [({"TILE": "7", "STAGE": "3"}, 2.0)]}
    got = _stream_tiers(_point(_rows()), _BarePrior(), None, db_idx)
    assert got is not None
    leaf, knobs, price, _tier = got
    assert knobs == {"TILE": "7", "STAGE": "3"}
    assert price == 2.0


def test_streamed_degenerate_pools() -> None:
    # Single-leaf pool: plain (unscored) return of that leaf.
    point = _point([{"TILE": "0", "STAGE": "0"}])
    leaf, knobs, price, _tier = _stream_tiers(point, _BarePrior(), None, {})
    assert knobs is None and price is None
    assert leaf_knobs(leaf) == {"TILE": "0", "STAGE": "0"}
    # Every leaf blocklisted: plain return of the first leaf.
    rows = _rows(3, 2)
    point = _point(rows)
    blocked = {tile_identity(dict(r)) for r in rows}
    leaf, knobs, price, _tier = _stream_tiers(point, _BarePrior(), blocked, {})
    assert knobs is None and price is None
    assert leaf_knobs(leaf) == rows[0]


def test_budgeted_pool_ranks_a_deterministic_drawn_subset(monkeypatch) -> None:
    """Above the cold-pool budget the scan ranks seeded descents instead of walking: the pick is
    a legal complete row, identical across calls (the RNG seeds from the pool identity), and the
    model scores at most the draw, never the pool."""
    from dataclasses import dataclass, field

    from emmy.compiler.pipeline.fork import Fork

    @dataclass(frozen=True)
    class _BoundedFork(Fork):
        inner: Fork = None
        knobs: dict = field(default_factory=dict)
        expansions: list = field(default_factory=list, compare=False)
        pool_bound = 10**9
        pool_id = "test-pool"
        is_leaf = False

        def expand(self):
            self.expansions.append(None)
            return self.inner.expand()

    class _CountingPrior(_BarePrior):
        def __init__(self):
            self.scored = 0

        def mean_scores_features(self, rows):
            self.scored += len(rows)
            return super().mean_scores_features(rows)

    monkeypatch.setenv("EMMY_POOL_DRAW", "64")
    rows = _rows(30, 20)  # 600 leaves ≫ the draw
    all_rows = {(r["TILE"], r["STAGE"]) for r in rows}
    point = _point(rows)
    point.options = [_BoundedFork(inner=point.options[0])]
    prior = _CountingPrior()
    got = _stream_tiers(point, prior, None, {})
    assert got is not None
    leaf, knobs, price, _tier = got
    assert (knobs["TILE"], knobs["STAGE"]) in all_rows  # a legal complete row off the real tree
    assert leaf_knobs(leaf) == knobs  # the drawn row built back to its own leaf
    assert prior.scored <= 64  # the draw, never the pool
    prior2 = _CountingPrior()
    again = _stream_tiers(point, prior2, None, {})
    assert again[1] == knobs and again[2] == price  # seeded off the pool identity → reproducible

    blocked_point = _point(rows)
    wrapper = _BoundedFork(inner=blocked_point.options[0])
    blocked_point.options = [wrapper]
    blocked = {tile_identity(dict(row)) for row in rows}
    assert _stream_tiers(blocked_point, _CountingPrior(), blocked, {}) == (NO_OPTION, None, None, None)
    assert len(wrapper.expansions) == 4 * 64  # four attempts per drawn row, and no exhaustive fallback


def _cut_fork(db_ctx):
    """A placement fork on a real tile kernel: the fuse arm and one cut, the shape the cut pass offers."""
    from tests.compiler.helpers import case_target_tile

    tile = case_target_tile("fused/norm-linear-f16-scalar-reduce.json")
    fuse = DeferredFork(materialize=lambda: None, knobs={"PLACE": "fuse"})
    cut = DeferredFork(materialize=lambda: None, knobs={"PLACE@map.1/map": "cut"}, structural=True)
    return SimpleNamespace(options=[fuse, cut], splices=(), node_id="node", root_op=tile, ctx=db_ctx), tile, fuse, cut


def test_a_stored_cut_is_priced_from_its_pieces_at_the_fork() -> None:
    """The tune DB holds no row spelling a cut: the decision is a routing row, and at the fork it is
    priced as the sum of its pieces' fastest rows on this card — every piece, or the arm is off the
    ballot. The parent is matched by exact identity, so another kernel's cut never prices this one."""
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline.search.db import PerfStats, RoutingRow, SearchDB
    from tests.compiler.pipeline.search.helpers import GPU_5090, kernel_row

    ctx = Context.from_target((12, 0), gpu_name=GPU_5090)
    point, tile, _fuse, cut = _cut_fork(ctx)
    parent = tile.identity_key(structural=False, with_io=True)
    db = SearchDB()
    for row in (kernel_row(parent), kernel_row("c1"), kernel_row("c2")):
        db.record_kernel(row)
    db.record_routing(RoutingRow(parent=parent, arm={"PLACE@map.1/map": "cut"}, children=("c1", "c2")))

    def measured(kernel: str, us: float, work: str = "w1x8") -> None:
        # Through the live context, so the row lands in the regime this compile reads (the suite's flags included).
        stats = PerfStats(median=us, min=us, max=us, mean=us, variance=0.0, n_samples=30)
        db.record_perf(ctx, kernel, bindings={}, knobs={"WORK": work}, backend="cuda", status="ok", stats=stats)

    measured("c1", 30.0)
    assert _route_candidates(point, greedy._EMPTY_MEASURED, db) == [], "a piece without a row leaves the cut unpriced"
    measured("c2", 50.0)
    measured("c2", 70.0, work="t8")
    assert _route_candidates(point, greedy._EMPTY_MEASURED, db) == [(cut, 80.0)]
    assert _route_candidates(point, greedy._EMPTY_MEASURED, None) == []
    other = Context.from_target((12, 0), gpu_name="NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition")
    assert _route_candidates(SimpleNamespace(**{**vars(point), "ctx": other}), greedy._EMPTY_MEASURED, db) == []


def test_a_measured_split_is_priced_from_its_pieces_with_no_routing_row() -> None:
    """A sweep benches a split's pieces but writes no routing row: the deploy reads only perf rows. The
    split arm is then priced as the sum of its pieces' fastest rows, so a 21 + 1.4 us split beats a
    52 us unsplit GEMV, and a split whose pieces are slower than the GEMV loses to it."""
    from emmy.compiler.context import Context
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.ir.frontend.ir import MatmulOp
    from emmy.compiler.pipeline import LOOP_PASSES, Pipeline
    from emmy.compiler.pipeline.pipeline import Run
    from emmy.compiler.pipeline.search.bench_record import kernel_row
    from emmy.compiler.pipeline.search.db import PerfStats, SearchDB
    from tests.compiler.pipeline.search.helpers import GPU_5090

    ctx = Context.from_target((12, 0), gpu_name=GPU_5090)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, 1024), "f16"), node_id="x")
    g.add_node(InputOp(), [], Tensor("w", (1024, 16), "f16"), node_id="w")
    g.add_node(MatmulOp(), ["x", "w"], Tensor("y", (1, 16), "f16"), node_id="y")
    g.inputs, g.outputs = ["x", "w"], ["y"]

    def kernels(decide) -> dict:
        """The kernels the cut pass leaves, by node id: what a sweep benches, before any is scheduled."""
        lowered = Pipeline.build(LOOP_PASSES).run(g, ctx=ctx)
        terminal, _trace = Run(pipeline=Pipeline.build(["tile/lift", "tile/cut"]), ctx=ctx).resolve(lowered, decide)
        return {nid: node.op for nid, node in terminal.nodes.items() if isinstance(node.op, TileOp)}

    def arm(reduce: str):
        return lambda fp: next(o for o in fp.options if leaf_knobs(o).get("REDUCE", "") == reduce)

    def measured(db, op, us: float) -> None:
        stats = PerfStats(median=us, min=us, max=us, mean=us, variance=0.0, n_samples=30)
        db.record_kernel(kernel_row(op, op.name))
        db.record_perf(
            ctx,
            op.identity_key(structural=False, with_io=True),
            bindings={},
            knobs={"WORK": "w1x8"},
            backend="cuda",
            status="ok",
            stats=stats,
        )

    [fused] = kernels(arm("")).values()
    pieces = kernels(arm("g8k"))
    assert set(pieces) == {"y__partial", "y"}, "the g8k arm runs as a partial and a finalize"

    def deployed(partial_us: float) -> set[str]:
        db = SearchDB()
        measured(db, fused, 52.0)
        for nid, op in pieces.items():
            measured(db, op, partial_us if nid == "y__partial" else 1.4)
        return set(kernels(greedy.greedy_decide(prior=None, db=db)))

    assert deployed(21.0) == {"y__partial", "y"}, "a measured split faster than the unsplit kernel must be taken"
    assert deployed(60.0) == {"y"}, "a measured split slower than the unsplit kernel must lose to it"


def test_stored_placement_cuts_register_composed_and_child_routes() -> None:
    """Measured cut routes register composed offers and later piece continuations."""
    from emmy.compiler.pipeline.search.db import RoutingRow, SearchDB
    from emmy.compiler.pipeline.search.strategy.greedy import _measured_composed_routes
    from tests.compiler.pipeline.search.helpers import kernel_row

    db = SearchDB()
    for row in (kernel_row("p"), kernel_row("c1"), kernel_row("c2"), kernel_row("c3")):
        db.record_kernel(row)
    db.record_routing(RoutingRow(parent="p", arm={"PLACE@a": "cut", "PLACE@b": "cut"}, children=("c1", "c2", "c3")))
    db.record_routing(RoutingRow(parent="p", arm={"PLACE@a": "cut"}, children=("c1", "c2")))

    assert _measured_composed_routes(db) == [("p", ("PLACE@a", "PLACE@b")), ("p", ("PLACE@a",))]
    assert _measured_composed_routes(SearchDB()) == []


# ---------------------------------------------------------------------------
# A kernel-set fork no measurement decides is ranked by what its arms are.
# ---------------------------------------------------------------------------


def _kernel_sets(decide, case: str = "fused/linear-add-place-cut-sm70.json"):
    """The kernels the cut pass leaves on a corpus case's program under ``decide``."""
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline import Pipeline
    from emmy.compiler.pipeline.pipeline import Run
    from emmy.compiler.pipeline.search.pins import pinned_knobs, unpinned_decisions
    from tests.compiler.pipeline.search.helpers import CARDS
    from tests.compiler.realization import helpers as corpus

    case = corpus.load_case(corpus.CASES_DIR / case)
    ctx = Context.from_target(case.compute_cap, gpu_name=CARDS[case.compute_cap], compile_flags="")
    with pinned_knobs(case.regime), unpinned_decisions():
        terminal, _trace = Run(pipeline=Pipeline.build(["tile/lift", "tile/cut"]), ctx=ctx).resolve(case.program(), decide)
    return [node.op for node in terminal.nodes.values() if isinstance(node.op, TileOp)]


class _NoSchedule:
    """A schedule prior a kernel-set decision must never consult: no arm is scheduled to decide one."""

    def mean_scores_features(self, rows):
        raise AssertionError("a kernel-set fork must not score a schedule row")


def _pieces_prior(weight: float) -> SimpleNamespace:
    """A placement prior over the arms' ``P_*`` rows; lower is better, so a positive weight rewards pieces."""
    return SimpleNamespace(mean_scores_features=lambda rows: [-weight * row.get("P_n_pieces", 0.0) for row in rows])


@pytest.mark.parametrize(("weight", "kernels"), [(1.0, 3), (-1.0, 1)])
def test_the_placement_prior_decides_every_unmeasured_kernel_set_fork(weight: float, kernels: int) -> None:
    """A kernel-set fork with no measured arm goes to the placement prior, which ranks the arms the cut pass
    offers by their ``P_*`` rows: a prior rewarding pieces cuts the corpus case's kernel and splits a piece, one
    penalizing them keeps it one kernel — and no schedule row is scored for either answer."""
    assert len(_kernel_sets(greedy.greedy_decide(prior=_NoSchedule(), placement_prior=_pieces_prior(weight)))) == kernels


def test_with_no_placement_prior_every_kernel_set_fork_takes_its_first_arm() -> None:
    """With nothing to rank the arms with, the kernel stays fused and unsplit."""
    assert len(_kernel_sets(greedy.greedy_decide(prior=_NoSchedule(), placement_prior=None))) == 1


def test_an_arm_leaving_a_kernel_that_always_failed_is_off_the_ballot(monkeypatch) -> None:
    """A measurement can disqualify: where every measured variant of a kernel failed, an arm that leaves that kernel
    loses to any arm that does not, whatever the prior says. The join is the kernel's exact identity, so a failure
    recorded on another kernel condemns nothing — that is how DeepSeek-V4's post block kept a fused arm whose every
    benched variant hung, until the failures were read."""
    from emmy.compiler.wire import kernel_identity

    cut = _kernel_sets(greedy.greedy_decide(prior=_NoSchedule(), placement_prior=_pieces_prior(1.0)))
    assert len(cut) > 1

    def kernels_with_failed(failed: set[str]) -> list[str]:
        monkeypatch.setattr(greedy, "_db_measured_index", lambda *_: greedy._Measured({}, {kernel: [2e6] for kernel in failed}))
        return [kernel_identity(k) for k in _kernel_sets(greedy.greedy_decide(prior=_NoSchedule(), placement_prior=_pieces_prior(1.0)))]

    failed = kernel_identity(cut[0])
    assert failed not in kernels_with_failed({failed}), "an arm leaving a failed kernel must lose"
    assert kernels_with_failed({"another"}) == [kernel_identity(k) for k in cut], "a failure on another kernel condemns nothing"
