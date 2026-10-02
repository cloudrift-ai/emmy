"""The :class:`Prior` — the ranking model a compile consults where nothing measured decides a fork.

A prior scores a complete knob row (a kernel's schedule and tuning knobs plus its ``S_*`` / ``H_*``
stamps) with a latency-like number, **lower is better**. Two scoring surfaces: :meth:`mean_score`
/ :meth:`mean_scores` over knob dicts (the greedy argmin), and :meth:`score_rows` over a packed
candidate pool (what ``emmy fit`` and ``emmy eval prior`` rank with). The one concrete prior is the
fit-offline :class:`~.offline.OfflinePrior`; :func:`~.load_prior` builds it.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from emmy.compiler.pipeline.search.dataset.group import Group


def latency_proxy(quality: float, scale: float) -> float:
    """``exp(-scale · quality)`` — the transform that turns a ranking quality into a deployed score. Lower is
    better, so one greedy argmin and one kernel-set sum consume it like a latency."""
    return math.exp(-scale * quality)


class Prior(ABC):
    """Abstract ranking model over complete knob rows."""

    @property
    @abstractmethod
    def fitted(self) -> bool: ...

    @abstractmethod
    def mean_score(self, knobs: dict) -> float:
        """Prediction for ranking a candidate — the greedy argmin. Lower is better."""

    def mean_scores(self, knobs_list: list[dict]) -> list[float]:
        """Batched :meth:`mean_score` — the greedy driver flattens a kernel's whole candidate set into one
        scoring pass, so a model with a vectorized predict overrides this to score the lot in a single
        call. The default maps element-wise."""
        return [self.mean_score(k) for k in knobs_list]

    # --- scoring already-featurized rows (the model classes' own seam) ------------------------

    def mean_score_features(self, feats: dict) -> float:
        """:meth:`mean_score` on an ALREADY-featurized row (``features.knob_features``
        output). The seam each model class implements: :meth:`mean_score` featurizes and
        delegates here, so a model never has to know how a knob dict becomes features. A pool is
        scored through :meth:`score_rows` instead — the dict form does not survive that size.
        Contract:
        ``mean_score_features(knob_features(knobs)) == mean_score(knobs)``, and a
        DELETED key lands in the model's ``NaN`` missing bucket."""
        raise NotImplementedError

    def mean_scores_features(self, feats_list: list[dict]) -> list[float]:
        """Batched :meth:`mean_score_features`; override for a vectorized predict."""
        return [self.mean_score_features(f) for f in feats_list]

    @abstractmethod
    def score_rows(self, group: Group) -> np.ndarray:
        """This model's ranking QUALITY for every row of a packed candidate pool — higher = predicted faster.

        The pool-shaped scoring surface, and the third one this class has after :meth:`mean_scores` (knob dicts)
        and :meth:`mean_scores_features` (feature dicts). It exists because the dict surfaces cannot be used on
        the pools these questions are actually asked over: a matmul enumeration runs to ~10^5 rows, and the
        per-row dict of ~63 floats that made the fit OOM would make an evaluation OOM for the same reason. The
        packed matrix is that representation done once, and the model projects it onto its OWN column list (see
        :meth:`Group.matrix`).

        Polarity is deliberately the opposite of :meth:`mean_score`'s. This returns quality because that is what
        the fitted model classes compute and what the rank metrics take; the deployed score is the monotone
        ``exp(-scale·quality)`` of it, and a caller wanting the cost family (:func:`~..metrics.topk_regret`,
        :func:`~..metrics.spearman`) negates. Both orders are read as ORDER only, so the negation is exact.

        The pool must carry every column the model reads: a column the group lacks fills absent, and a model that
        reads ``S_*`` / ``H_*`` columns would then be asked about a kernel with no shape. Which columns a group
        carries is the BUILDER's decision, so an evaluation builds its pools over the full featurization."""

    def pick(self, rows: list[dict]) -> tuple[int, float]:
        """The :meth:`mean_scores` argmin over ``rows``. Score ties break by
        :func:`~emmy.compiler.pipeline.knob.canonical_row_key` (candidate content, never enumeration
        order — same-featurized siblings score identically, and an order-broken tie flips the deployed
        kernel per boot). Returns ``(index, predicted µs)``."""
        from emmy.compiler.pipeline.knob import canonical_row_key  # noqa: PLC0415

        scores = self.mean_scores(rows)
        best_i = min(range(len(scores)), key=lambda i: (scores[i], canonical_row_key(rows[i])))
        return best_i, scores[best_i]
