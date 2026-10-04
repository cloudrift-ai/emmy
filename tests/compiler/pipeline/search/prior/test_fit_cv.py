"""The ``emmy fit`` cross-validation harness (``search/prior/fit/cv``) and run harness
(``search/prior/fit/run``): shape-grouped fold partitioning, dual-rank tie semantics, several positives per candidate pool, metrics-file
determinism, the golden group builder's pool merge, and the trainer-callable seam
— all on synthetic groups, no tracing, no GPU."""

import argparse
import json
from dataclasses import dataclass, replace

import numpy as np
import pytest

from emmy.commands.fit import register_fit_command
from emmy.compiler.context import FAST_MATH_FLAG
from emmy.compiler.pipeline.search import features, ranking
from emmy.compiler.pipeline.search.dataset import Dataset, GoldenPool, GoldenRow
from emmy.compiler.pipeline.search.dataset.group import DEFAULT_FEATURES, GoldenGroup, feature_view
from emmy.compiler.pipeline.search.dataset.kernel import KernelDef
from emmy.compiler.pipeline.search.dataset.shape import ShapeKey
from emmy.compiler.pipeline.search.pool import Candidates
from emmy.compiler.pipeline.search.prior.fit import cv as fit_cv
from emmy.compiler.pipeline.search.prior.fit.run import run_fit
from emmy.compiler.pipeline.search.ranking import build_golden_groups
from tests.compiler.pipeline.search.helpers import kernel_row

# --- feature view ------------------------------------------------------------------


def test_default_feature_view_keeps_the_geometry_and_atom_features():
    """The default spec keeps the ``D_``-prefixed geometry features, the two atom features that vary within a
    candidate pool — ``MMA_tier`` and ``MMA_acc_bits``, the f16-vs-f32 accumulate discriminator — and ``H_cc``,
    which a tree combines with them to rank per architecture. The other shape/hardware pass-throughs are left to
    an explicit ``--features``."""
    keep = feature_view(DEFAULT_FEATURES)
    sample = {"D_waves": 1, "D_": 2, "MMA_tier": 3, "MMA_acc_bits": 4, "MMA_atom_m": 5, "S_ext_free_prod": 6, "H_cc": 7, "H_opt": 8}
    assert {k for k in sample if keep(k)} == {"D_waves", "D_", "MMA_tier", "MMA_acc_bits", "H_cc"}


def test_feature_view_globs_and_names():
    keep = feature_view("MMA_*, D_waves")
    assert keep("MMA_tier") and keep("MMA_acc_bits") and keep("D_waves")
    assert not keep("D_bk_gap") and not keep("S_ext_free_prod") and not keep("MMA")


def test_a_merged_case_reports_its_positive_count():
    """A group is a candidate pool, so a merged one is one row in ``per_golden`` where two goldens used to be
    two. ``positives`` is what makes that visible: without it a metrics file with fewer groups looks like lost
    data rather than a pool that gained a second verified answer."""
    groups = [_case("m.512", "warp", "gpuA", pinned=1), _case("m.1024", "warp", "gpuA", pinned=(0, 2))]
    model = _StubModel(-1.0)
    rows = fit_cv.evaluate_full_train(groups, model)["per_golden"]
    assert rows["gpuA/m.512"]["positives"] == 1
    assert rows["gpuA/m.1024"]["positives"] == 2
    # The merged group scores on its BEST positive: D_a descends, so row 0 wins and the pool ranks top-1.
    assert rows["gpuA/m.1024"]["rank"] == 0 and rows["gpuA/m.512"]["rank"] == 1
    out = fit_cv.run_folds(groups, trainer=FOLD_TRAINER, k=2)
    assert out["holdout_per_golden"]["gpuA/m.1024"]["positives"] == 2
    assert out["train_per_golden"]["gpuA/m.1024"]["positives"] == 2


# --- the golden group builder's pool merge ------------------------------------------


def _pool(name, rows, goldens, *, regime="", kernel="k"):
    """A :class:`GoldenPool` with its candidate pool handed in directly instead of enumerated — it rides on the
    kernel row's wire, where the stub enumerator below reads it. ``rows`` are single-token dicts the stub
    signature reads; ``goldens`` the tokens the pool's golden rows recorded. ``kernel`` is the kernel row's
    identity, so two pools can be two kernels; the pool is labelled ``<name>.<kernel>``."""
    row = replace(kernel_row(kernel, name=name), loop_ir={"rows": rows}).keyed(kernel)
    rows_ = tuple(GoldenRow({"TILE": tag}, 1.0, "golden:test") for tag in goldens)
    return GoldenPool("gpuA", (12, 0), regime, row, {}, rows_)


@dataclass(frozen=True)
class _StubContext:
    """A dataclass because the builder ``replace``s it to attach a sample - the same shape the real
    Context has, so the plumbing under test is the real plumbing."""

    pool_sample: object = None

    def features(self):
        return {}


def _build(pools, monkeypatch, **kwargs):
    """``build_golden_groups`` over stub pools: a row is ``{"TILE": tag}``, and its one feature is the tag's
    position in the alphabet — enough for two pools to featurize differently whenever their rows differ. The
    stub enumerator honours ``ctx.pool_sample`` the way the tree draw does: the kept rows first, then a draw —
    here the pool's prefix."""

    def enumerate_stub(pool, ctx):
        rows = pool.kernel.loop_ir["rows"]
        if rows == "broken":
            raise ValueError("a definition the lowering cannot take back")
        sample = ctx.pool_sample
        if sample is None:
            return Candidates(list(rows), len(rows))
        kept = [row for row in rows if tuple(sorted(row.items())) in sample.keep]
        drawn = kept + [row for row in rows[: sample.rows] if row not in kept]
        return Candidates(drawn, len(drawn))

    monkeypatch.setattr(ranking, "pool_context", lambda pool: _StubContext())  # noqa: ARG005
    monkeypatch.setattr(ranking, "enumerate_pool", enumerate_stub)
    monkeypatch.setattr(ranking.ShapeKey, "from_s_features", classmethod(lambda cls, s: ShapeKey(512, 512, True)))  # noqa: ARG005
    monkeypatch.setattr(ranking.features, "tile_signature", lambda knobs: knobs["TILE"])
    # The stub wire carries candidate rows, not a body, so there is no kernel to lower and no stamps to compute.
    monkeypatch.setattr(KernelDef, "op", lambda self, bindings=None: None)  # noqa: ARG005
    monkeypatch.setattr(ranking.features, "stamps", lambda op: {})  # noqa: ARG005
    monkeypatch.setattr(ranking.features.Featurizer, "features", lambda self, kernel, row: {"D_a": float(ord(row["TILE"][0]))})  # noqa: ARG005
    return build_golden_groups(pools, **kwargs)


def test_builder_pins_every_golden_row_of_a_pool_and_keys_regimes_apart(monkeypatch):
    """A pool is one kernel on one card in one regime, and every golden row measured on it pins a row of ONE
    group. The kernel's fast-math rows are another pool — the fast-math enumeration offers an atom the standard
    one never emits, so pinning row indices across the two would name rows that do not exist — and a row whose
    signature no candidate carries is skipped on its own; the pool still stands for the rows that hit."""
    std = [{"TILE": "a"}, {"TILE": "b"}, {"TILE": "c"}]
    fastmath = [*std, {"TILE": "f"}]  # the extra atom row the standard enumeration never emits
    groups, skipped = _build(
        [_pool("m.512", std, ["b", "c", "zz"]), _pool("m.512", fastmath, ["f"], regime=FAST_MATH_FLAG)],
        monkeypatch,
    )
    assert [(c.key, c.golden_ids, len(c.feats)) for c in groups] == [("gpuA/m.512.k", (1, 2), 3), ("gpuA/m.512.k#2", (3,), 4)]
    assert skipped == [("gpuA", "m.512.k", "golden not in 3 candidates")]
    # The ``#N`` suffix is spent only where a key would otherwise collide, so the first group keeps the plain
    # key that ``cv.run_folds`` accumulates train ranks under.
    assert len({c.key for c in groups}) == len(groups)


def test_two_goldens_on_one_pool_still_merge_under_sampling(monkeypatch):
    """Sampling must not break the merge, and the merge is what makes two verified answers to one
    question one training group instead of two.

    The two are one group before the draw happens — they are rows of one pool, so the builder enumerates it
    once. What sampling must not break is the PINS: both recorded rows have to survive the draw, which they do
    because it is a pure function of the tree and ``(sample, seed)`` and every golden row of the pool's card
    and regime is in the keep-set. The last row is one a 4-of-26 prefix draw never reaches."""
    pool = [{"TILE": chr(ord("a") + i)} for i in range(26)]
    groups, skipped = _build([_pool("m.512", pool, ["a", "z"])], monkeypatch, sample=4)  # the FIRST and the LAST row
    assert skipped == []
    assert len(groups) == 1, "one pool, one group - the draw must not fracture it into two"
    (group,) = groups
    assert len(group.golden_ids) == 2, "both recorded rows survive the draw and land in the group"
    assert group.total == len(group.feats) < 26, "the draw's size travels beside the sample"
    kept = {chr(int(v)) for v in group.feats[:, 0]}
    assert {"a", "z"} <= kept


def test_builder_folds_away_a_golden_recorded_twice_at_one_config(monkeypatch):
    """Two recordings of the same config over the same pool are ONE fact. They verify the same row, so the
    label set does not grow — where the ``#2`` group they used to become counted that one fact twice in every
    metric."""
    std = [{"TILE": "a"}, {"TILE": "b"}]
    groups, _ = _build([_pool("m.512", std, ["b", "b"])], monkeypatch)
    assert [(c.key, c.golden_ids) for c in groups] == [("gpuA/m.512.k", (1,))]


def test_builder_folds_two_pools_that_pack_identically(monkeypatch):
    """The reason the packed pool decides membership and the DB key only groups the work: two kernels the
    featurizer cannot tell apart enumerate twice and produce identical pools, and the second stage folds them,
    so a pool is one group however many times it was recorded."""
    std = [{"TILE": "a"}, {"TILE": "b"}, {"TILE": "c"}]
    groups, _ = _build([_pool("m.512", std, ["b"]), _pool("m.512", std, ["c"], kernel="other")], monkeypatch)
    assert [(c.key, c.golden_ids) for c in groups] == [("gpuA/m.512.k", (1, 2))]


def test_a_pool_none_of_whose_goldens_is_found_is_no_group(monkeypatch):
    """A pool whose every golden row misses its candidates is skipped whole — a group with no positive would
    put a rank in ``metrics.json`` under a name the same run reports as skipped — and so is one whose definition
    the enumeration cannot lower: its rows are counted under its name, and the run goes on."""
    std = [{"TILE": "a"}, {"TILE": "b"}]
    pools = [_pool("m.512.absent", std, ["zz"]), _pool("m.512.broken", "broken", ["b"]), _pool("m.512", std, ["b"])]
    groups, skipped = _build(pools, monkeypatch)
    assert [(c.key, c.name) for c in groups] == [("gpuA/m.512.k", "m.512.k")]
    assert skipped == [("gpuA", "m.512.absent.k", "golden not in 2 candidates"), ("gpuA", "m.512.broken.k", "did not lower")]


# --- synthetic groups ---------------------------------------------------------------


def _case(name, tier, gpu, pinned=1, n_rows=6, key=None, shape=None):
    """A tiny group whose rows carry a monotone D_a, so a ranker has signal. EVERY group carries the
    routing stamp on every row, as the featurizer writes it (``passes/identity._extents`` emits the key
    unconditionally, 0.0 when no axis is symbolic) — that stamp's VALUE, not the tier label, is what
    marks the group dynamic.

    ``pinned`` is the pool's positive row, or several of them (a pool the builder matched more than
    one golden into). ``shape`` is the fold group; it defaults to the name with any ``.dynM`` suffix
    stripped, which mirrors what the builder does for real: a dynamic golden enumerates its static
    twin's pool, so the twins share a group."""
    stamp = {"S_ext_n_symbolic_axis": 1.0 if tier == "dyn" else 0.0}
    feats = [{"D_a": float(i), "D_b": float((i * 7) % 3), **stamp} for i in range(n_rows)]
    shape = shape or name.removesuffix(".dynM")
    return GoldenGroup.from_dicts(key or f"{gpu}/{name}", name, tier, gpu, shape, pinned, feats)


def _cases():
    return [
        _case("matmul.square.512", "thread", "gpuA"),
        _case("matmul.square.1024", "thread", "gpuA", pinned=2),
        _case("matmul.qkv.h4096", "warp", "gpuA", pinned=0),
        _case("matmul.qkv.h4096.dynM", "dyn", "gpuA", pinned=3),
        _case("matmul.square.512", "thread", "gpuB"),
        _case("matmul.square.512.dynM", "dyn", "gpuB", pinned=2),
        _case("matmul.qkv.h4096", "warp", "gpuB", pinned=4),
        _case("reduce.k2048", "reduce", "gpuB", pinned=0),
    ]


# --- routing stamp + absent features ------------------------------------------------


def test_routing_stamp_is_packed_and_agrees_with_the_tier():
    """``S_ext_n_symbolic_axis`` is PACKED like any other column, carrying the real stamp — the model splits
    the two regimes on it. A NaN column here rather than 1.0 would be a train/serve skew: live candidates
    always carry the value."""
    static, dyn = _case("m.512", "warp", "gpuA"), _case("m.512.dynM", "dyn", "gpuA")
    assert (static.dynamic, dyn.dynamic) == (False, True)
    assert (dyn.matrix(["S_ext_n_symbolic_axis"]) == 1.0).all()
    # The tier decides nothing, but it still has to AGREE: it and the stamp are the same fact arriving
    # by two routes, so a disagreement means one of them is wrong.
    with pytest.raises(ValueError, match="disagree about the regime"):
        GoldenGroup.from_dicts("gpuA/x", "x", "warp", "gpuA", "x", 0, [{"D_a": 1.0, "S_ext_n_symbolic_axis": 1.0}])
    with pytest.raises(ValueError, match="disagree about the regime"):
        GoldenGroup.from_dicts("gpuA/x", "x", "dyn", "gpuA", "x", 0, [{"D_a": 1.0}])


def test_matrix_projects_absent_features_to_nan():
    """``GoldenGroup`` stores absent features as ``NaN``, the model's missing bucket. Both kinds of absence are
    covered: a name the pool never stamped, and a key missing from one row whose siblings carry it."""
    g = GoldenGroup.from_dicts("gpuA/x", "x", "warp", "gpuA", "x", 0, [{"D_a": 1.0, "D_b": 2.0}, {"D_a": 3.0}])
    nans = g.matrix(["D_a", "D_b", "D_never"])
    assert nans[0].tolist()[:2] == [1.0, 2.0] and np.isnan(nans[0][2])
    assert nans[1][0] == 3.0 and np.isnan(nans[1][1]) and np.isnan(nans[1][2])
    # A genuine 0.0 is never confused for an absent value.
    z = GoldenGroup.from_dicts("gpuA/z", "z", "warp", "gpuA", "z", 0, [{"D_a": 0.0}])
    assert z.matrix(["D_a"]).tolist() == [[0.0]]


def test_matrix_is_memoized_and_read_only():
    """Building the projection is a strided copy of the whole pool, and cross-validation asks for the
    same one over and over — measured, those copies were 85% of a fit's wall time. So it is built once
    and shared, which is only safe if it cannot be mutated underneath another holder: a caller that
    needs to write copies inside itself.

    One entry is enough — a fit uses one column list for its whole run — so a different request evicts
    rather than accumulating."""
    g = GoldenGroup.from_dicts("gpuA/x", "x", "warp", "gpuA", "x", 0, [{"D_a": 1.0, "D_b": 2.0}, {"D_a": 3.0}])
    first = g.matrix(["D_a", "D_b"])
    assert g.matrix(["D_a", "D_b"]) is first, "a repeat request must not rebuild the projection"
    assert not first.flags.writeable
    with pytest.raises(ValueError):
        first[0, 0] = 99.0
    # ... and copying is what a mutating caller does.
    mine = g.matrix(["D_a", "D_b"]).copy()
    mine[0, 0] = 99.0
    assert g.matrix(["D_a", "D_b"])[0, 0] == 1.0, "the shared projection survived a caller's private copy"
    # A different column list evicts, and the evicted request rebuilds correctly rather than
    # returning the wrong shape.
    other = g.matrix(["D_a"])
    assert other.shape == (2, 1) and g.matrix(["D_a", "D_b"]).shape == (2, 2)


def test_no_feature_view_can_drop_the_routing_stamp():
    """Routing is not a view choice. A spec that names neither the stamp nor a prefix covering it still
    keeps it — otherwise the model could not tell a symbolic-axis pool from a static one."""
    for spec in (DEFAULT_FEATURES, "D_waves"):
        assert feature_view(spec)("S_ext_n_symbolic_axis"), spec
    assert not feature_view("D_waves")("S_ext_free_prod")  # only the routing features are exempt


def test_feature_view_exclusions_apply_to_names_and_globs():
    keep = feature_view("D_*,MMA_tier,-D_near_*,-MMA_tier")
    assert keep("D_threads") and not keep("D_near_area") and not keep("MMA_tier")


def test_featurizer_preserves_the_routing_stamp():
    """The stamp survives ``knob_features`` — the one step between a golden's structural features and
    the row the fit packs. Without it the model could not tell the two regimes apart."""
    assert features.knob_features({"S_ext_n_symbolic_axis": 1.0, "S_ext_free_prod": 4096.0})["S_ext_n_symbolic_axis"] == 1.0
    assert "S_ext_n_symbolic_axis" not in features.knob_features({"S_ext_free_prod": 4096.0})


NAMES = ["D_a", "D_b"]


class _StubModel:
    """A minimal fitted model — enough to prove the harness needs nothing from a model beyond ``score_rows``."""

    def __init__(self, w):
        self.w = w

    def score_rows(self, group):
        return group.matrix(NAMES) @ np.array([self.w, 0.0])


class _StubTrainer:
    """The trainer seam: one object with ``fit(groups) -> model``, reused across every fold."""

    def __init__(self, w):
        self.w = w

    def fit(self, groups):  # noqa: ARG002 — a stub ignores the data
        return _StubModel(self.w)


FOLD_TRAINER = _StubTrainer(-1.0)


# --- fold partitioning + pooling ---------------------------------------------------


def test_folds_hold_out_every_golden_exactly_once():
    out = fit_cv.run_folds(_cases(), trainer=FOLD_TRAINER, k=3)
    assert set(out["holdout_per_golden"]) == {c.key for c in _cases()}
    for row in out["holdout_per_golden"].values():
        assert isinstance(row["rank"], int) and isinstance(row["rank_optimistic"], int)
        assert row["rank"] >= row["rank_optimistic"]  # pessimistic can only add ties
    # Train side: same keys (every group was in the other folds' training slices).
    assert set(out["train_per_golden"]) == {c.key for c in _cases()}
    # Aggregates are per card only — a REPORT axis, independent of what the folds group by — and the
    # gap is their arithmetic difference.
    for gpu in ("gpuA", "gpuB"):
        summaries = {(c["axes"]["cv_split"], c["axes"]["gpu"]): c for c in out["summaries"]}
        assert ("holdout", gpu) in summaries and ("train", gpu) in summaries
        assert out["gap"][gpu] == round(
            summaries[("holdout", gpu)]["metrics"]["rank"]["median"] - summaries[("train", gpu)]["metrics"]["rank"]["median"], 2
        )


def test_a_shape_group_is_never_split_across_folds():
    """THE leakage guard, and the reason this fitter folds by shape at all.

    Goldens sharing an extent identity enumerate the same candidate pool. Hold one out while training
    on another and the fold model has already been shown the answer, so its "holdout" rank is not a
    holdout rank. The retired ``op_family`` axis keyed on the golden's NAME and missed exactly this:
    on the real corpus, 178 shape groups spanned more than one family, covering 695 of 1385 goldens.

    The groups below are that situation in miniature — one shape under three unrelated names, on two
    different cards, which must still land in ONE fold."""
    groups = [
        _case("rms_norm.k2048", "reduce", "gpuA", shape="S(free=2048,red=2048)"),
        _case("gemma4_12b.rms_norm", "reduce", "gpuA", shape="S(free=2048,red=2048)"),
        _case("olmoe_1b7b.rms_norm.k2048.m1", "reduce", "gpuB", shape="S(free=2048,red=2048)"),
        _case("matmul.square.512", "thread", "gpuA", shape="S(free=512,red=512)"),
        _case("matmul.square.1024", "thread", "gpuB", shape="S(free=1024,red=1024)"),
    ]
    by_shape = fit_cv.assign_folds(groups, 3)
    folds = {c.key: by_shape[c.shape] for c in groups}
    shared = [k for k in folds if "rms_norm" in k]
    assert len({folds[k] for k in shared}) == 1, "one shape, one fold — on every card"

    out = fit_cv.run_folds(groups, trainer=FOLD_TRAINER, k=3)
    held = out["holdout_per_golden"]
    assert len({held[k]["fold"] for k in shared}) == 1


def test_folds_are_balanced_and_deterministic():
    """Groups are very uneven on the real corpus (largest 122 groups, 162 singletons), so assignment is
    largest-first onto the currently-smallest fold; a thin fold's median would be noise. Assignment is
    a pure function of the group list, so a re-run reproduces it."""
    groups = [_case(f"m.{i}", "thread", "gpuA", shape=f"s{i // 2}") for i in range(12)]  # 6 groups of 2
    a = fit_cv.assign_folds(groups, 3)
    assert a == fit_cv.assign_folds(list(reversed(groups)), 3)
    counts = {f: sum(1 for c in groups if a[c.shape] == f) for f in set(a.values())}
    assert set(counts) == {0, 1, 2} and max(counts.values()) - min(counts.values()) <= 1


def test_more_folds_than_groups_leaves_no_empty_fold_scored():
    """k above the group count is not an error — the surplus folds simply hold nothing and are
    skipped, rather than being scored as empty holdouts."""
    groups = [_case("m.1", "thread", "gpuA", shape="s1"), _case("m.2", "thread", "gpuA", shape="s2")]
    out = fit_cv.run_folds(groups, trainer=FOLD_TRAINER, k=5)
    assert set(out["holdout_per_golden"]) == {c.key for c in groups}
    assert all(d["n"] > 0 for d in out["fold_detail"]["holdout_medians"].values())


# --- metrics assembly: determinism + skip accounting -------------------------------


def _metrics():
    groups = _cases()
    model = _StubModel(-1.0)
    cv = fit_cv.run_folds(groups, trainer=FOLD_TRAINER, k=3)
    skipped = [
        ("gpuA", "attention.hd128", fit_cv.OUT_OF_SCOPE),
        ("gpuA", "matmul.o_proj.h4096", "golden not in 12 candidates"),
        ("gpuC", "softmax.k2048", fit_cv.OUT_OF_SCOPE),  # a card with no ranked group at all
    ]
    header = {"data": "golden", "seed": 0}
    return fit_cv.build_metrics(header, groups, skipped, model, cv)


def test_metrics_json_is_deterministic():
    a = json.dumps(_metrics(), indent=2, sort_keys=True)
    b = json.dumps(_metrics(), indent=2, sort_keys=True)
    assert a == b


def test_metrics_counts_every_skipped_golden():
    m = _metrics()
    assert all(c["axes"]["cv_split"] == "full_train" for c in m["full_train"]["summaries"])
    # Skipped goldens sit BESIDE the summaries: they have no pool, so they are a fact about the corpus rather
    # than about a scored card — and a fit summary stays the same shape an eval report emits.
    assert set(m["full_train"]["summaries"][0]) == {"axes", "groups", "metrics"}
    assert m["full_train"]["skipped"]["gpuA"] == {"unranked": 1, "out_of_scope": 1}
    assert m["full_train"]["skipped"]["gpuB"] == {"unranked": 0, "out_of_scope": 0}
    # A card whose every golden was skipped still appears — as a skipped entry with no summary.
    assert m["full_train"]["skipped"]["gpuC"]["out_of_scope"] == 1
    assert "gpuC" not in {c["axes"]["gpu"] for c in m["full_train"]["summaries"]}
    # full_train per-golden rows carry the pool size; ranks respect the tie ordering.
    row = m["full_train"]["per_golden"]["gpuA/matmul.square.512"]
    assert row["pool"] == 6 and row["rank"] >= row["rank_optimistic"]


# --- run harness: the trainer-callable seam ----------------------------------------


def _run_stub_fit():
    return run_fit(
        _cases(),
        [("gpuC", "softmax.k2048", fit_cv.OUT_OF_SCOPE)],
        trainer=_StubTrainer(-1.0),
        folds=3,
        header={"trainer": "stub", "seed": 0},
    )


def test_run_fit_stub_trainer_deterministic():
    """``run_fit`` is a pure function of its inputs and knows nothing about the model: a stub trainer whose
    models implement only ``score_rows`` yields the full metrics shape, identically on every call. It returns
    the FIT, not an artifact — assembling one is the caller's business."""
    metrics, fit = _run_stub_fit()
    assert set(metrics) == {"header", "full_train", "cv"}
    assert metrics["header"] == {"trainer": "stub", "seed": 0}
    assert metrics["full_train"]["skipped"]["gpuC"]["out_of_scope"] == 1
    assert set(metrics["cv"]["holdout_per_golden"]) == {c.key for c in _cases()}
    assert fit.w == -1.0
    a, b = _run_stub_fit(), _run_stub_fit()
    assert json.dumps(a[0], sort_keys=True) == json.dumps(b[0], sort_keys=True)


# --- CLI surface -------------------------------------------------------------------


def test_fit_command_defaults():
    parser = argparse.ArgumentParser()
    register_fit_command(parser.add_subparsers())
    args = parser.parse_args(["fit", "_data/dataset", "_tune/offline.json"])
    assert (args.dataset, args.weights, args.seed, args.folds) == ("_data/dataset", "_tune/offline.json", 0, 5)
    # Both paths are explicit: a fit never writes anywhere it was not told to, the shipped weights included.
    with pytest.raises(SystemExit):
        parser.parse_args(["fit", "_data/dataset"])
    # --features defaults to the space's view, resolved in the handler rather than by argparse.
    assert args.features is None


def test_handle_fit_writes_metrics_and_a_loadable_artifact(tmp_path):
    """The command layer end to end on a synthetic dataset — trainer wiring, artifact assembly and
    both output files — without enumerating a single pool. The weights path is the test's own, so the run never
    touches the shipped weights."""
    provenance = {"source": "stub", "sources": {}, "pool_sample": 0, "seed": 0, "feat_ver": features.FEATURIZER_VERSION, "compiler": "test"}
    Dataset(_cases(), [], [], {"golden": {}, "measured": {}}, provenance).dump(tmp_path / "dataset")
    parser = argparse.ArgumentParser()
    register_fit_command(parser.add_subparsers())
    args = parser.parse_args(
        [
            "fit",
            str(tmp_path / "dataset"),
            str(tmp_path / "weights.json"),
            "--folds",
            "3",
            "--iterations",
            "20",
            "--out",
            str(tmp_path / "run"),
        ]
    )
    args.func(args)

    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert metrics["header"]["trainer_params"]["objective"] == "QuerySoftMax"
    assert metrics["cv"]["summaries"]
    # Cases are candidate pools now, so the header carries how many verified rows they hold between them —
    # otherwise a group count that fell because two goldens shared a pool reads as lost data.
    assert metrics["header"]["groups"] == {"total": 8, "positives": 8, "merged": 0}

    artifact = json.loads((tmp_path / "weights.json").read_text())
    assert artifact["cols"] and "oblivious_trees" in artifact["model"]
    assert artifact["provenance"]["groups"] == {"static": 6, "dynamic": 2} and artifact["provenance"]["positives"] == 8
    assert "top1=" in artifact["provenance"]["notes"]
