"""The :class:`Prior` — the ranking model a compile consults where nothing measured decides a fork.

A prior scores a complete knob row (a kernel's schedule and tuning knobs plus its ``S_*`` / ``H_*``
stamps) with a latency-like number, **lower is better**. Two scoring surfaces: :meth:`mean_score`
/ :meth:`mean_scores` over knob dicts (the greedy argmin), and :meth:`score_rows` over a packed
candidate pool (what ``emmy fit`` and ``emmy eval prior`` rank with). The one concrete prior is the
fit-offline :class:`~.offline.OfflinePrior`; :func:`~.load_prior` builds it.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from emmy.compiler.pipeline.search.dataset.group import Group

logger = logging.getLogger(__name__)


# The exponent bound for :func:`latency_proxy`. ``exp`` overflows a float64 just past 709.8, so this is a
# float-representation boundary rather than a modelling choice, and it is one the shipped artifact cannot
# approach: over all 1377 recorded goldens its quality spans 0 to 277, an exponent of at most 27.7.
#
# It is reachable only by a weight vector nothing bounds. The rank objective is indifferent to that vector's
# MAGNITUDE — scaling it preserves every ordering — so the raw-space L2 is the only thing that pins it, which
# is why ``--l2``'s help calls it a tie-breaker rather than a shrinkage term. ``--l2 0`` removes it.
PROXY_CLIP = 700.0

# Warn once per process. If this fires at all the run is already producing garbage rankings, and a pool holds
# ~78k rows — a line each would bury the finding under itself.
_clip_warned = False


def latency_proxy(quality: float, scale: float) -> float:
    """``exp(-scale · quality)`` — the latency proxy BOTH model classes return from ``mean_score_features``,
    so the two cannot drift on the transform that turns a ranking quality into a deployed score. Lower is
    better, matching the online prior's predicted µs, which is what lets one greedy argmin and one policy
    normalization consume either model.

    Clipping is the last resort and it is LOUD, because a clipped exponent silently destroys a ranking: every
    row past the bound lands on the same float, and a greedy argmin over a plateau of equal scores falls
    through to enumeration order. That is not hypothetical — it is the 2026-07 incident, caused by a ±80 clip
    on the QUALITY, which sat inside the live range and collapsed the whole good region onto one ``exp(-8)``
    value. Moving the bound onto the exponent put it two orders of magnitude outside anything reachable; the
    warning is what makes a return to that regime visible instead of silent. A consumer needing a
    BOUNDED value clamps on its side."""
    global _clip_warned
    arg = -scale * quality
    if not -PROXY_CLIP <= arg <= PROXY_CLIP:
        if not _clip_warned:
            _clip_warned = True
            logger.warning(
                "[prior] latency proxy exponent %.1f is outside +/-%.0f and was clipped — every row past the "
                "bound now scores identically, so this ranking is decided by enumeration order, not by the "
                "model. The shipped artifact peaks near 28; a weight vector this large means a fit with no "
                "effective L2 (--l2 0) or a hand-edited artifact.",
                arg,
                PROXY_CLIP,
            )
        arg = max(min(arg, PROXY_CLIP), -PROXY_CLIP)
    return math.exp(arg)


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
        DELETED key carries each model's own absent-feature semantics (``0.0`` term
        for the linear offline prior, ``NaN`` routing for CatBoost)."""
        raise NotImplementedError

    def mean_scores_features(self, feats_list: list[dict]) -> list[float]:
        """Batched :meth:`mean_score_features`; override for a vectorized predict."""
        return [self.mean_score_features(f) for f in feats_list]

    @abstractmethod
    def score_rows(self, group: Group) -> np.ndarray | None:
        """This model's ranking QUALITY for every row of a packed candidate pool — higher = predicted faster.
        ``None`` when this model cannot score the pool at all (the linear model asked for a dynamic weight set
        it never fit); a model with nothing fitted yet still answers, with a constant.

        The pool-shaped scoring surface, and the third one this class has after :meth:`mean_scores` (knob dicts)
        and :meth:`mean_scores_features` (feature dicts). It exists because the dict surfaces cannot be used on
        the pools these questions are actually asked over: a matmul enumeration runs to ~10^5 rows, and the
        per-row dict of ~63 floats that made the fit OOM would make an evaluation OOM for the same reason. The
        packed matrix is that representation done once, and each implementation here projects it onto its OWN
        column list with its OWN absent-value fill (see :meth:`Group.matrix`) — which is the whole reason this
        cannot be one shared function over ``feat_names``.

        Polarity is deliberately the opposite of :meth:`mean_score`'s. This returns quality because that is what
        the fitted model classes compute and what the rank metrics take; the deployed score is the monotone
        ``exp(-scale·quality)`` of it, and a caller wanting the cost family (:func:`~..metrics.topk_regret`,
        :func:`~..metrics.spearman`) negates. Both orders are read as ORDER only, so the negation is exact.

        The pool must carry every column the model reads. A group packed under a narrow feature view — the fit's
        ``D_*`` view, say — answers the linear model correctly, because its weight names are inside that view,
        while the online model's ``S_*`` / ``H_*`` columns would all fill absent and its predictions would be
        about a kernel with no shape. Which columns a group carries is the BUILDER's decision, so an evaluation
        that scores both halves builds its pools over the full featurization."""

    def pick(self, rows: list[dict]) -> tuple[int, float]:
        """The :meth:`mean_scores` argmin over ``rows``. Score ties break by
        :func:`~emmy.compiler.pipeline.knob.canonical_row_key` (candidate content, never enumeration
        order — same-featurized siblings score identically, and an order-broken tie flips the deployed
        kernel per boot). Returns ``(index, predicted µs)``."""
        from emmy.compiler.pipeline.knob import canonical_row_key  # noqa: PLC0415

        scores = self.mean_scores(rows)
        best_i = min(range(len(scores)), key=lambda i: (scores[i], canonical_row_key(rows[i])))
        return best_i, scores[best_i]
