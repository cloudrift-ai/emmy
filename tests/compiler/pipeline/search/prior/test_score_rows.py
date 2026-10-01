"""``Prior.score_rows`` — the pool-shaped scoring surface every prior answers.

The dict surfaces (``mean_scores`` / ``mean_scores_features``) cannot be used on the pools a ranking question
is actually asked over: a matmul enumeration runs to ~10^5 rows and the per-row dict representation is what
made the fit OOM. This one takes the packed pool. What each implementation must get right is its OWN column
list and its OWN absent-value fill — which is why it cannot be one shared function — and one polarity, since
the fitted model classes compute quality (higher = faster).
"""

from __future__ import annotations

from emmy.compiler.pipeline.search.dataset.group import GoldenGroup
from emmy.compiler.pipeline.search.prior.linear_model import LinearModel
from emmy.compiler.pipeline.search.prior.offline import OfflinePrior


def _group(feats: list[dict], *, tier: str = "thread") -> GoldenGroup:
    return GoldenGroup.from_dicts("card/k", "k", tier, "card", "k", 0, feats)


def _linear(**weights) -> OfflinePrior:
    return OfflinePrior(
        model=LinearModel(
            weights=dict(weights), weights_dynamic=dict(weights), scale=1.0, atomic_free_weight=0.0, atomic_free_split_threshold=4.0
        )
    )


def _stamp(dynamic: float = 0.0) -> dict:
    """The stamps the featurizer writes on every row: the routing one — 0.0 when no axis is symbolic, never
    absent — and the regime, which ``ctx.features()`` always supplies and the prior's dataset requires."""
    return {"S_ext_n_symbolic_axis": dynamic, "H_opt": 3.0}


def test_the_linear_half_scores_a_whole_pool_by_its_own_weights():
    """Higher quality = predicted faster, and the model reads only the columns it has weights for — extra
    columns in the pool are ignored rather than being an error."""
    prior = _linear(D_a=2.0)
    scores = prior.score_rows(_group([{**_stamp(), "D_a": 1.0, "D_unused": 99.0}, {**_stamp(), "D_a": 3.0}]))

    assert scores.tolist() == [2.0, 6.0]


def test_a_narrower_feature_view_does_not_move_the_linear_halfs_ranks():
    """Why ``eval prior --pools golden`` may build its pools over the FULL featurization while ``emmy fit``
    trains under a narrow ``D_*`` view, and still report the same rank: the model projects the pool onto its
    own weight names, so every column outside them is inert."""
    prior = _linear(D_a=2.0)
    narrow = [{**_stamp(), "D_a": 1.0}, {**_stamp(), "D_a": 3.0}]
    wide = [{**row, "S_ext_free_prod": 4096.0, "H_opt": 3.0, "MMA_a_bits": 16.0} for row in narrow]

    assert prior.score_rows(_group(narrow)).tolist() == prior.score_rows(_group(wide)).tolist()


def test_the_linear_half_declines_a_pool_it_fitted_no_weight_set_for():
    """``None``, not an exception and not a silent zero: an unfittable cross-validation fold and an
    all-static fit both hit this, and a report has to count the pools rather than average them in."""
    static_only = LinearModel(
        weights={"D_a": 1.0}, weights_dynamic=None, scale=1.0, atomic_free_weight=0.0, atomic_free_split_threshold=4.0
    )
    prior = OfflinePrior(model=static_only)

    assert prior.score_rows(_group([{**_stamp(), "D_a": 1.0}])) is not None
    assert prior.score_rows(_group([{**_stamp(1.0), "D_a": 1.0}], tier="dyn")) is None
