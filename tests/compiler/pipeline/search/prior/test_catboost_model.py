"""The CatBoost offline model: its artifact round-trip, its absent-feature semantics, and the trainer that
produces it. No GPU — CatBoost fits on CPU in a fraction of a second at these sizes.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest

from emmy.compiler.pipeline.search.dataset.group import PLACEMENT_FEATURES, GoldenGroup, feature_view
from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION
from emmy.compiler.pipeline.search.prior import OfflinePrior
from emmy.compiler.pipeline.search.prior.catboost_model import ABSENT, CatBoostModel
from emmy.compiler.pipeline.search.prior.fit.catboost import CatBoostTrainer

FEATURES = ("D_a", "D_b")


def _groups(n_pools: int = 8, n_rows: int = 30, dynamic: bool = False) -> list[GoldenGroup]:
    """Pools whose golden is the row maximizing ``D_a`` — a signal any working ranker recovers, and one
    that is invisible to ``D_b`` (pure noise), so a fitted model must have learned the right column."""
    rng = np.random.default_rng(0)
    stamp = {"S_ext_n_symbolic_axis": 1.0 if dynamic else 0.0}
    out = []
    for gi in range(n_pools):
        rows = [{"D_a": float(i), "D_b": float(rng.integers(0, 5)), **stamp} for i in range(n_rows)]
        out.append(GoldenGroup.from_dicts(f"gpuA/p{gi}", f"p{gi}", "dyn" if dynamic else "warp", "gpuA", f"shape{gi}", n_rows - 1, rows))
    return out


def _fit(groups=None, *, feature_names=FEATURES, **kw) -> CatBoostModel:
    trainer = CatBoostTrainer(feature_names=feature_names, iterations=40, negatives=12, **kw)
    return trainer.fit(groups if groups is not None else _groups()).model


# --- the trainer -------------------------------------------------------------------


def test_fit_recovers_the_planted_signal():
    """Every golden ranks first in its own full pool — the fit objective reaching its floor on a dataset
    where the answer is a single feature."""
    fit = CatBoostTrainer(feature_names=FEATURES, iterations=40, negatives=12).fit(_groups())
    assert fit.ranks == [0] * 8
    assert "not byte-reproducible" in fit.notes


def test_the_tree_prices_the_two_regimes_from_the_routing_column():
    """The capability the packed column restores. Static and dynamic pools here rank OPPOSITELY on ``D_a``,
    which one model can only fit by splitting on the regime. While the column arrived all-NaN every row sat
    in one bucket, the split was unavailable, and the fit had to compromise across the two regimes.

    Guards the fix at the level it operates: not the column's name, its contents."""
    rng = np.random.default_rng(0)

    def pools(dynamic, offset):
        out = []
        for gi in range(8):
            rows = [
                {"D_a": float(i), "D_b": float(rng.integers(0, 5)), "S_ext_n_symbolic_axis": 1.0 if dynamic else 0.0} for i in range(30)
            ]
            pinned = 0 if dynamic else 29  # dynamic pools want the LOW D_a row, static the high one
            out.append(
                GoldenGroup.from_dicts(
                    f"gpuA/p{gi + offset}", f"p{gi + offset}", "dyn" if dynamic else "warp", "gpuA", f"shape{gi + offset}", pinned, rows
                )
            )
        return out

    groups = pools(False, 0) + pools(True, 8)
    fit = CatBoostTrainer(feature_names=(*FEATURES, "S_ext_n_symbolic_axis"), iterations=60, negatives=12).fit(groups)
    assert fit.ranks == [0] * 16
    importance = dict(zip(fit.model.cols, fit.model.booster.get_feature_importance(type="PredictionValuesChange"), strict=True))
    assert importance["S_ext_n_symbolic_axis"] > 0.0


def test_routing_stamp_is_an_ordinary_column():
    """The tree prices both regimes in ONE model: the routing stamp is a plain packed column it reads like
    any other, so there is no second weight set and no unfittable-dynamic-set fold to exclude.

    The load-bearing assertion is that the column arrives with the STAMP in it. It used to arrive all-NaN:
    the dataset withheld the name, the trainer appended it to ``cols`` anyway, and ``GoldenGroup.matrix`` filled
    a column it could not find with ``ABSENT``. Every training row landed in one bucket while every live
    candidate carried a real 0.0/1.0 — a train/serve skew that ``cols`` alone could never reveal."""
    routing = ("S_ext_n_symbolic_axis",)
    groups = _groups() + _groups(dynamic=True)
    model = _fit(groups, feature_names=(*FEATURES, *routing))
    assert model.cols == (*FEATURES, *routing)
    static, dynamic = groups[0], groups[-1]
    assert (dynamic.matrix(list(routing)) == 1.0).all()
    assert (static.matrix(list(routing)) == 0.0).all()


def test_placement_ranking_can_differ_between_same_capability_cards():
    """A tree needs the card fact even when it is constant within each placement fork."""
    groups = []
    for memory, golden in ((16, 0), (32, 1)):
        for index in range(8):
            rows = [{"P_n_pieces": float(pieces), "H_cc": 70.0, "H_total_mem": float(memory)} for pieces in (1, 2)]
            groups.append(GoldenGroup.from_dicts(f"gpu{memory}/p{index}", f"p{index}", "place", f"gpu{memory}", "s", golden, rows))
    names = tuple(name for name in groups[0].feat_names if feature_view(PLACEMENT_FEATURES)(name))
    fit = CatBoostTrainer(feature_names=names, iterations=40, negatives=2).fit(groups)
    assert fit.ranks == [0] * len(groups)


def test_score_rows_covers_the_full_pool_not_the_sample():
    """Training sampled 12 negatives per pool; the metric still ranks the golden among all 30 rows, so the
    reported rank stays exactly the deployed quantity."""
    groups = _groups()
    fit = CatBoostTrainer(feature_names=FEATURES, iterations=40, negatives=12).fit(groups)
    assert fit.rows < sum(len(g.feats) for g in groups)
    assert fit.score_rows(groups[0]).shape == (30,)


def test_score_rows_projects_onto_the_models_own_columns():
    """A pool need not carry the columns the model was trained on, nor in its order — projecting it is the
    MODEL's job now, and every other pool in this file stamps every column, so nothing else reaches this.

    Column ORDER is the catchable half, and this pins it: the same pool read in the wrong order scores
    differently. The FILL is deliberately not asserted through a score — CatBoost's ``nan_mode="Min"`` puts
    NaN on the same side of every split as the training minimum, and ``0.0`` IS that minimum here, so a
    mis-filled column would score identically. That is why the fill is pinned where it is decided
    (:func:`test_absent_features_are_nan_not_zero`) rather than by its effect."""
    model = _fit()  # trained on ("D_a", "D_b")
    group = GoldenGroup.from_dicts("gpuA/p", "p", "warp", "gpuA", "s", 0, [{"D_a": float(i)} for i in range(6)])
    assert group.feat_names == ("D_a",)  # the pool is missing a column the model wants

    scores = model.score_rows(group)
    assert np.array_equal(scores, model.quality_rows(group.matrix(list(model.cols))))
    assert not np.array_equal(replace(model, cols=("D_b", "D_a")).score_rows(group), scores)


def test_hard_negative_mining_grows_the_training_set():
    """Round 0 draws negatives uniformly; each further round adds the rows the current model ranks near the
    golden. More rounds therefore train on strictly more rows — the mechanism, not merely the flag."""
    one = CatBoostTrainer(feature_names=FEATURES, iterations=20, negatives=5, rounds=1).fit(_groups())
    two = CatBoostTrainer(feature_names=FEATURES, iterations=20, negatives=5, rounds=2).fit(_groups())
    assert two.rows > one.rows


# --- several positives per pool ----------------------------------------------------


def test_a_verified_row_is_never_drawn_as_a_negative():
    """A pool the builder matched several goldens into holds several verified-good configs, and none of them
    may be handed to the loss at label 0.0 — teaching the model that a config we measured as good is bad is
    exactly the contamination ``DEFAULT_ROUNDS`` warns about, only self-inflicted.

    Checked on both draws: the small-pool branch, which contributes every row it has rather than sampling,
    and the sampling branch. And on the mined draw, whose window has to widen by the pin count or a pool
    whose pins all rank high would come back short."""
    trainer = CatBoostTrainer(feature_names=FEATURES, negatives=12)
    rng = np.random.default_rng(0)
    pins = (2, 5)
    small = trainer._uniform(8, pins, rng)  # 6 unpinned rows <= 12: the every-row branch
    assert sorted(small) == [0, 1, 3, 4, 6, 7]
    big = trainer._uniform(200, pins, rng)
    assert len(big) == 12 and not set(pins) & set(big.tolist())
    scores = np.arange(50, dtype=float)[::-1]  # row 0 scores highest, so the pins sit inside the window
    mined = trainer._hard(scores, (0, 1))
    assert len(mined) == 12 and not {0, 1} & set(mined.tolist())


def test_row_accounting_covers_every_positive():
    """``rows`` is what the fit reports it trained on, so a pool with two pins must count two — otherwise the
    number silently understates the training set as the corpus grows siblings."""
    trainer = CatBoostTrainer(feature_names=FEATURES, negatives=4)
    rows = [{"D_a": float(i), "D_b": 0.0, "S_ext_n_symbolic_axis": 0.0} for i in range(30)]
    extra = GoldenGroup.from_dicts("gpuA/p1", "p1", "warp", "gpuA", "s1", (0, 1, 29), rows)
    groups = [_groups(n_pools=1)[0], extra]
    assert (len(groups[0].golden_ids), len(extra.golden_ids)) == (1, 3)
    assert trainer._n_rows([np.arange(4), np.arange(4)], groups) == (4 + 1) + (4 + 3)


def test_fit_treats_every_pin_as_a_positive():
    """``QuerySoftMax`` takes several positives per group as they are, so a pool with two verified configs
    needs no reshaping: both pins ride into the training set at label 1.0 and both end up ahead of every
    unpinned row. A run that had drawn one of them as a negative would have pushed it back down."""
    rows = [{"D_a": float(i), "D_b": 0.0} for i in range(20)]
    groups = [GoldenGroup.from_dicts(f"gpuA/p{i}", f"p{i}", "warp", "gpuA", f"shape{i}", (18, 19), rows) for i in range(6)]
    fit = CatBoostTrainer(feature_names=FEATURES, iterations=40, learning_rate=0.05, negatives=8).fit(groups)
    assert fit.rows == 6 * (8 + 2)
    assert set(np.argsort(-fit.score_rows(groups[0]), kind="stable")[:2].tolist()) == {18, 19}
    # Was [1] * 6 while the trainer appended a routing name the matrix could not fill: the resulting
    # all-NaN column collapsed rows 18 and 19 onto one leaf, so the two pins tied and a tie counts against
    # the golden. Without that junk column the tree separates them and each golden ranks first.
    assert fit.ranks == [0] * 6


# --- the model surface -------------------------------------------------------------


def test_batched_and_single_row_scoring_agree():
    model = _fit()
    rows = [{"D_a": 3.0, "D_b": 1.0}, {"D_a": 29.0, "D_b": 4.0}]
    assert model.mean_scores_features(rows) == [model.mean_score_features(r) for r in rows]
    assert model.mean_scores_features([]) == []


def test_score_polarity_is_lower_is_better():
    """``mean_score`` is a latency proxy, so the greedy argmin means "fastest": the row the ranker prefers must
    score LOWER."""
    model = _fit()
    assert model.mean_score_features({"D_a": 29.0, "D_b": 1.0}) < model.mean_score_features({"D_a": 0.0, "D_b": 1.0})


def test_absent_features_are_nan_not_zero():
    """The absent/decided-zero distinction, which is the whole reason a tree gets NaN: a row that never
    stamped ``D_b`` must not be read as a row that stamped it as 0.0."""
    model = _fit()
    packed = model.matrix([{"D_a": 5.0}])
    assert math.isnan(packed[0][1]) and packed[0][0] == 5.0
    assert math.isnan(ABSENT)


# --- the artifact ------------------------------------------------------------------


def _write(model: CatBoostModel, path) -> None:
    from emmy import storage

    storage.write_json(path, model.to_artifact(provenance={"fitted": "2026-10-01"}), indent=1)


def test_artifact_round_trip_preserves_predictions(tmp_path):
    """The booster rides in the JSON as CatBoost's own JSON model: a reloaded artifact scores identically, so an
    A/B against a written artifact measures the fit rather than the serialization."""
    model = _fit()
    _write(model, tmp_path / "weights.json")
    art = json.loads((tmp_path / "weights.json").read_text())
    assert art["feat_ver"] == FEATURIZER_VERSION and "oblivious_trees" in art["model"]
    reloaded = CatBoostModel.from_artifact(art)
    assert reloaded.cols == model.cols and reloaded.scale == model.scale
    rows = [{"D_a": float(i), "D_b": 1.0 if i % 3 else math.nan} for i in range(10)]
    # CatBoost's JSON writes leaf values in decimal, so a reload agrees to the last bit, not bit for bit.
    assert reloaded.mean_scores_features(rows) == pytest.approx(model.mean_scores_features(rows), rel=1e-12)


def test_two_fits_on_one_input_write_the_same_model_info(tmp_path):
    """CatBoost stamps a GUID and a finish time per training run; the artifact drops them, so they cannot be the
    only difference two refits show."""
    model = _fit()
    info = model.to_artifact(provenance={})["model"]["model_info"]
    assert "model_guid" not in info and "train_finish_time" not in info


def test_offline_prior_loads_an_artifact(tmp_path, monkeypatch):
    """The deploy path end to end: the prior loads the file the override names and ranks through it."""
    path = tmp_path / "tree.json"
    _write(_fit(), path)
    monkeypatch.setenv("EMMY_OFFLINE_FILE", str(path))
    prior = OfflinePrior()
    assert isinstance(prior.model, CatBoostModel)
    fast, slow = {"D_a": 29.0, "D_b": 1.0}, {"D_a": 0.0, "D_b": 1.0}
    assert prior.mean_score_features(fast) < prior.mean_score_features(slow)
    assert max(prior.mean_scores_features([fast, slow])) > 0


def test_artifact_missing_a_key_is_a_hard_error(tmp_path, monkeypatch):
    """An artifact without its column order, params or model must not load — and a retired linear artifact has
    no ``model``."""
    from emmy import storage

    art = _fit().to_artifact(provenance={})
    del art["cols"]
    path = tmp_path / "partial.json"
    storage.write_json(path, art)
    monkeypatch.setenv("EMMY_OFFLINE_FILE", str(path))
    with pytest.raises(RuntimeError, match="cols"):
        OfflinePrior()


def test_shipped_artifacts_load():
    """Both checked-in priors load, each naming its own space."""
    from emmy.compiler.pipeline.search.prior.offline import default_file

    for space in ("schedule", "placement"):
        assert OfflinePrior(path=str(default_file(space))).space == space
