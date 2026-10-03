"""The fitted CatBoost ranker as a value object — the one model class
:class:`~emmy.compiler.pipeline.search.prior.offline.OfflinePrior` ranks with.

Two access shapes over one arithmetic: a feature-dict path (how a live candidate is scored) and a packed-matrix path
(what ``emmy fit`` trains and evaluates on, entered for a whole candidate pool through
:meth:`CatBoostModel.score_rows`).

A tree needs no hand-built terms: it splits on the symbolic-axis routing stamp to price both regimes inside one
model, and forms interactions such as ``D_finalize_kernel × D_splitk`` from the two columns.

**Absent features are ``NaN``**, CatBoost's own missing-value bucket (``nan_mode="Min"``). "This knob is not decided
/ not stamped on this row" is a different fact from a knob that is present and legitimately zero (``WM=0``,
``STAGE="00"`` → popcount 0.0), and a tree can branch on the difference. The training rows carry the same semantics —
see ``Group.matrix``.

``quality`` (the raw ranker output) is HIGHER = predicted faster; the deployed score is the monotone
``exp(-scale · quality)`` of it, lower is better.

**The artifact is one JSON file.** The booster rides in it as CatBoost's own JSON model format (``format="json"``):
text a reader can open, loaded back by CatBoost with predictions identical to the binary form. The envelope around it
carries the column order, the scalar params and the provenance.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION
from emmy.compiler.pipeline.search.prior.base import latency_proxy

if TYPE_CHECKING:
    # Annotation only: importing ``search.dataset`` for real would pull it (and, through the golden format,
    # subprocess) onto the deploy path, which loads none of it today.
    from emmy.compiler.pipeline.search.dataset.group import Group

# Raw QuerySoftMax outputs sit around O(1), so the exp wrapper needs no shrinking to stay in a sane range.
DEFAULT_SCALE = 1.0

# Absent-feature fill. Spelled once: the deploy-side dict path and the fit-side matrix path must agree, or the
# model is asked with semantics it was never trained under.
ABSENT = np.nan

# ``model_info`` fields CatBoost stamps per training run. They carry no part of the model, and dropping them keeps
# two fits on identical inputs from differing in a GUID and a timestamp.
_RUN_STAMPS = ("model_guid", "train_finish_time")


def to_json(booster) -> dict:
    """A fitted booster as CatBoost's JSON model, minus the per-run stamps. CatBoost writes the format only to a
    file, so it round-trips through a tempfile — ONE spelling of that round-trip."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    try:
        booster.save_model(tmp, format="json")
        with open(tmp) as fh:
            obj = json.load(fh)
    finally:
        os.unlink(tmp)
    for key in _RUN_STAMPS:
        obj.get("model_info", {}).pop(key, None)
    return obj


def from_json(obj: dict):
    """A ranker loaded from CatBoost's JSON model — the read half of :func:`to_json`."""
    booster = new_ranker()
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(obj, f)
        tmp = f.name
    try:
        booster.load_model(tmp, format="json")
    finally:
        os.unlink(tmp)
    return booster


def new_ranker(**params):
    """A ``CatBoostRanker`` carrying the settings every fit and load here shares. Imported lazily so a ``compile``
    that never scores a candidate does not pay the catboost import."""
    from catboost import CatBoostRanker  # noqa: PLC0415

    return CatBoostRanker(verbose=False, allow_writing_files=False, **params)


@dataclass(frozen=True)
class CatBoostModel:
    """A fitted CatBoost ranker over ``features.Featurizer`` rows, plus the column order it reads and the scalar
    scoring param. Immutable, so a fit result is a value the caller can swap and serialize.

    ``cols`` is the model's feature vocabulary IN ORDER — the booster indexes by position, so this tuple ships in
    the artifact beside the booster."""

    booster: Any
    cols: tuple[str, ...]
    scale: float = DEFAULT_SCALE

    # --- the model surface (Prior's own names) ----------------------------------------------------------

    def mean_score_features(self, feats: dict) -> float:
        """Latency proxy (``exp(-scale · quality)``), lower is better."""
        return self.mean_scores_features([feats])[0]

    def mean_scores_features(self, feats_list: list[dict]) -> list[float]:
        """Batched :meth:`mean_score_features` — ONE ``predict`` over the whole ``N × cols`` matrix.

        This is the path that matters, not an optimization of the single-row one: a greedy deploy flattens a
        kernel's entire candidate set into one scoring call, and per-row prediction over a ~78k-row pool would pay
        CatBoost's per-call overhead 78k times."""
        if not feats_list:
            return []
        return [latency_proxy(q, self.scale) for q in self.quality_rows(self.matrix(feats_list))]

    # --- the packed-pool path ---------------------------------------------------------------------------

    def matrix(self, feats_list: list[dict]) -> np.ndarray:
        """Feature dicts packed into this model's own column order, absent key = :data:`ABSENT`."""
        return np.array([[f.get(c, ABSENT) for c in self.cols] for f in feats_list], dtype=float)

    def score_rows(self, group: Group) -> np.ndarray:
        """:meth:`quality_rows` over a whole packed pool — the ONE place the model's column choice is made:
        :attr:`cols` in its own order, absent filling to :data:`ABSENT`, which is what the booster was trained
        against."""
        return self.quality_rows(group.matrix(list(self.cols)))

    def quality_rows(self, mat: np.ndarray) -> np.ndarray:
        """Per-row ranking quality (higher = predicted faster) over a matrix already in :attr:`cols` order — the
        raw ranker output, before the monotone ``exp(-scale·)`` wrapper. This is the quantity ``emmy fit`` ranks
        goldens by and the quantity the deployed score is a transform of, so the fitted objective IS the deployed
        ranking rather than a proxy for it."""
        return np.asarray(self.booster.predict(mat), dtype=float)

    # --- artifact round-trip -----------------------------------------------------------------------------

    @classmethod
    def from_artifact(cls, obj: dict) -> CatBoostModel:
        """Construct from an artifact dict: column order, params, and the booster from its JSON model.

        Deliberately does NOT version-gate: the strict ``feat_ver`` check is deploy policy and lives in
        ``offline._load_artifact``."""
        return cls(booster=from_json(obj["model"]), cols=tuple(obj["cols"]), scale=float(obj["params"]["scale"]))

    def to_artifact(self, *, provenance: dict, space: str = "schedule") -> dict:
        """This model as a weights artifact dict. ``provenance`` is caller-supplied whole so the assembly stays
        pure."""
        return {
            "feat_ver": FEATURIZER_VERSION,
            "space": space,
            "cols": list(self.cols),
            "params": {"scale": float(self.scale)},
            "provenance": provenance,
            "model": to_json(self.booster),
        }
