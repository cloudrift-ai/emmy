"""Offline prior — a stateless, fit-offline :class:`Prior` over ``features.Featurizer`` rows.

The ONE ranking model a compile consults where nothing measured decides a fork, fit by ``emmy fit`` from the
dataset's golden groups and shipped with the repo.

``mean_scores_features`` returns a positive latency *proxy* (``exp(-scale · quality)``), **lower is better**. The proxy is
not calibrated µs; only its ordering matters (the greedy argmin).

The scoring itself lives in :class:`~.catboost_model.CatBoostModel`; this class is the adapter that loads it and
satisfies the ``Prior`` contract around it.

The model lives in the repo-checked artifact ``weights/schedule.json`` beside this module (override with
``EMMY_OFFLINE_FILE`` / ``emmy eval … --offline-file`` to A/B a candidate fit), written by ``emmy fit DATASET
WEIGHTS`` jointly over EVERY kernel regime, so one model ranks them all. The artifact is version-gated on
``feat_ver`` — its columns are that featurizer version's, so a cross-version file is meaningless and loading it is
a hard error (refit, don't guess).
"""

from __future__ import annotations

import functools
from pathlib import Path

from emmy import config, storage
from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION
from emmy.compiler.pipeline.search.prior.base import Prior
from emmy.compiler.pipeline.search.prior.catboost_model import CatBoostModel

_DEFAULT_FILE = Path(__file__).parent / "weights" / "schedule.json"
_PLACEMENT_FILE = Path(__file__).parent / "weights" / "placement.json"


def default_file(space: str) -> Path:
    """The checked-in weights artifact of ``space``: the schedule prior, or the placement prior."""
    return _PLACEMENT_FILE if space == "placement" else _DEFAULT_FILE


# The keys every artifact carries — what ``CatBoostModel.from_artifact`` reads.
_KEYS = ("cols", "params", "model")


@functools.lru_cache(maxsize=8)
def _load_artifact(path_str: str) -> dict:
    """Load and validate a weights artifact — hard error on a missing/corrupt file, a ``feat_ver``
    mismatch, or a missing key. No silent fallback: an A/B
    that quietly reverts to other weights measures nothing, and both shipped artifacts are
    schema-tested."""
    obj = storage.read_json(Path(path_str))
    if not isinstance(obj, dict):
        raise RuntimeError(
            f"offline prior weights artifact missing or unreadable: {path_str} "
            f"(set EMMY_OFFLINE_FILE to a fitted artifact or regenerate the default "
            f"with 'emmy fit DATASET WEIGHTS' — README, 'Fit the offline prior')"
        )
    found = obj.get("feat_ver")
    if not isinstance(found, int) or found != FEATURIZER_VERSION:
        raise RuntimeError(
            f"offline prior weights artifact {path_str} has feat_ver={found!r}, "
            f"expected {FEATURIZER_VERSION} — its features are spelled in a different "
            f"featurizer vocabulary. Refit it: emmy fit DATASET WEIGHTS (README, 'Fit the offline prior')"
        )
    missing = [k for k in _KEYS if k not in obj]
    if missing:
        raise RuntimeError(f"offline prior weights artifact {path_str} lacks {missing} — refit it with 'emmy fit DATASET WEIGHTS'")
    return obj


class OfflinePrior(Prior):
    """Fixed ranker over ``Featurizer`` rows — the cold-start prior.

    An adapter, not a model: the scoring is a :class:`CatBoostModel`, and this class adds what ``Prior`` needs
    around it: the artifact load and its space. Pass a ready ``model``, or let it resolve from the weights artifact
    (``path`` → ``config.offline_path()`` override → the repo-checked default)."""

    def __init__(self, *, model: CatBoostModel | None = None, path: str | None = None) -> None:
        #: The space the weights rank — ``schedule`` for a ready model, else the artifact's.
        self.space = "schedule"
        if model is None:
            path = str(path or config.offline_path() or _DEFAULT_FILE)
            art = _load_artifact(path)
            self.space = art.get("space", "schedule")
            model = CatBoostModel.from_artifact(art)
        self._model = model

    @property
    def model(self):
        """The scoring function this prior ranks with."""
        return self._model

    @property
    def fitted(self) -> bool:
        return True

    def mean_score_features(self, feats: dict) -> float:
        """Latency proxy (``exp(-scale · quality)``) of one feature row, lower is better. An absent key lands in
        the tree's ``NaN`` missing bucket."""
        return self._model.mean_score_features(feats)

    def mean_scores_features(self, feats_list: list[dict]) -> list[float]:
        """Batched :meth:`mean_score_features` — one scoring pass over the whole set."""
        return self._model.mean_scores_features(feats_list)

    def score_rows(self, group):
        """The whole packed pool's ranking quality — straight through to the model, which is the object that
        knows its own columns. This adapter adds nothing here precisely because there is no knob dict to
        featurize: the pool arrives already packed, which is the same form ``emmy fit`` trains and scores it
        in, so a golden's rank under a fitted artifact and under this deployed prior are the same number by
        construction rather than by two paths agreeing."""
        return self._model.score_rows(group)


def load_prior() -> OfflinePrior:
    """The one prior a compile ranks with — the offline model the weights artifact names."""
    return OfflinePrior()
