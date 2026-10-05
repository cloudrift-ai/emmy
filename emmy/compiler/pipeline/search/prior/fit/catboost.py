"""The CatBoost trainer — the offline learning-to-rank fit of a :class:`CatBoostModel`.

:class:`CatBoostTrainer` holds the hyperparameters and :meth:`~CatBoostTrainer.fit` turns a
:class:`GoldenGroup` list into a :class:`CatBoostFit`. The trainer is immutable and one instance serves every
cross-validation fold. **A fit is not byte-reproducible**: CatBoost's histogram build is threaded, so two runs on
identical inputs give models that differ in the last bits. Deliberate: the alternative is
``thread_count=1``, and the metrics file plus the rank tables are what a fit is compared by, not a checksum.

This module owns what is specific to the model class; everything a different model class would also need lives
elsewhere (the scoring function in :mod:`..catboost_model`, the dataset in ``search/dataset/group.py``, the rank
metrics in :mod:`~..metrics`, the fold harness in :mod:`.cv`).

**The objective.** ``QuerySoftMax`` over one group per pool: every pinned row is a positive (label 1.0) and the
sampled negatives are 0.0. That is the loss shape the golden dataset actually has — *verified* rows only, no
graded labels — and it is what the deployed ranking is: an argmin over one fork's candidates, the same order
the softmax ranks by. ``QuerySoftMax`` takes several positives per group as they are, so
a pool with more than one verified config needs no reshaping — and those siblings are then never drawn as
negatives against each other, which is what labelling a verified-good config 0.0 used to do.

**Why negatives are sampled.** The pools are the training set, and all of them at once is ~38 M rows / ~18 GB before
CatBoost's own quantized copy. So each round draws ``negatives`` rows per pool, uniformly, from the rows that are NOT pinned.
The full pool is still what :meth:`CatBoostFit.score_rows` ranks the golden within, so the *metric* never sees
the sampling.

A further round instead mines **hard negatives** — the rows the current model ranks nearest the golden. That is
implemented and reachable (``--rounds 2``) but OFF by default, because the one measurement of it says it hurts;
:data:`DEFAULT_ROUNDS` carries the numbers and the likely reason.

The routing feature (``S_ext_n_symbolic_axis``) is an ordinary packed column, read like any other: one model
prices both regimes by splitting on it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from emmy.compiler.pipeline.search.dataset.group import GoldenGroup
from emmy.compiler.pipeline.search.metrics import best_rank
from emmy.compiler.pipeline.search.prior.catboost_model import DEFAULT_SCALE, CatBoostModel, new_ranker
from emmy.compiler.pipeline.search.prior.fit.tables import topk_table

logger = logging.getLogger(__name__)

# Sampled negatives per pool per round. ~500 against a pool's handful of positives puts the softmax denominator in
# the range the loss was designed for, and 490 golden pools × 500 rows is a dataset CatBoost fits in seconds.
DEFAULT_NEGATIVES = 500

# Trees for the placement space, whose forks are few (a handful of arms each) and whose rare labels — a split taken
# under fast math only — need the trees the schedule space's default stops short of. Cross-validated over the
# repository goldens (2026-10-05, 2414 forks), 500 lifted held-out top-1 on recorded cuts from 87.5% to 89.7% and on
# recorded splits from 76.0% to 78.7% against 200, with no fork kind lower; 1000 bought nothing more.
PLACEMENT_ITERATIONS = 500
# Mining rounds. Round 0 is the uniform draw; each further round adds the rows the current model ranks near the
# golden. ONE by default — mining is implemented and reachable (``--rounds 2``), but it is not on, because the
# only measurement of it says it hurts: over the 1278-group golden dataset, round 1 moved top-1 from 545 to 517
# and the mean log2 rank from 2.08 to 2.35, i.e. worse on the fit's own objective, scored over full pools.
#
# The likely reason is that our negatives are UNLABELED, not known-bad. Mining selects the rows the model ranks
# closest to the golden, which in a ~78k-row pool are exactly the rows most likely to be genuinely faster than
# it; labelling those 0.0 teaches the model to bury good configs. That is the contamination the debiased
# contrastive-learning bound predicts — it scales with the expected NUMBER of better-but-unlabeled rows in the
# contrast set, and mining maximizes that number by construction. Standard hard-negative mining assumes the
# negatives are known bad; here only the pinned rows are verified.
#
# The evidence is in-sample (no cross-validation run yet), so this is a default, not a verdict: re-measure
# ``--rounds 2`` held-out before concluding mining is useless.
DEFAULT_ROUNDS = 1


@dataclass(frozen=True)
class CatBoostTrainer:
    """The tree trainer: hyperparameters in, a fitted :class:`CatBoostFit` out.

    Immutable, and :meth:`fit` never touches ``self``, so ONE instance serves every cross-validation fold and a
    fit is a function of ``(groups, hyperparameters)`` alone — modulo CatBoost's threading, which is why this
    trainer promises reproducible *metrics* rather than a reproducible artifact.

    No warm start: every fit starts from nothing, so a fold model cannot inherit anything from a model trained on
    the held-out golden."""

    feature_names: tuple[str, ...]
    # A cross-validated sweep over the repository goldens (trees 100-400, depth 3-8, rate 0.05-0.5) found held-out
    # top-1 still rising with size; 200 trees of depth 6 at 0.3 sit within noise of the best at half its file size.
    iterations: int = 200
    depth: int = 6
    learning_rate: float = 0.3
    negatives: int = DEFAULT_NEGATIVES
    rounds: int = DEFAULT_ROUNDS
    random_state: int = 0
    scale: float = DEFAULT_SCALE

    def fit(self, groups: list[GoldenGroup]) -> CatBoostFit:
        """Fit over ``groups``, mining hard negatives across :attr:`rounds`.

        Each round rebuilds the training set (golden + the negatives sampled so far) and fits a FRESH model
        rather than continuing the previous one: continuing would weight the early uniform rows by however many
        rounds they have survived, and the point of mining is to shift the distribution, not to accumulate
        emphasis on the first draw."""
        rng = np.random.default_rng(self.random_state)
        cols = list(self.feature_names)
        pools = [g.matrix(cols) for g in groups]
        # Sampled negative indices per pool, grown each round. The pinned rows are added at assembly time, so
        # one can never be sampled in as a negative — against itself or against a verified sibling.
        sampled = [self._uniform(len(m), g.golden_ids, rng) for m, g in zip(pools, groups, strict=True)]
        inert = sum(1 for s, m in zip(sampled, pools, strict=True) if len(s) >= len(m) - 1)
        if inert:
            logger.warning(
                "--negatives %d is inert on %d of %d pools: they are no larger than the draw, so every row is a "
                "negative and the setting selects nothing there",
                self.negatives,
                inert,
                len(pools),
            )

        model = self._fit_once(pools, groups, sampled)
        ranks = self._ranks(model, pools, groups)
        logger.info("  round 0 uniform (%d rows): %s", self._n_rows(sampled, groups), topk_table(ranks))
        for r in range(1, max(1, self.rounds)):
            sampled = [
                np.union1d(s, self._hard(model.quality_rows(m), g.golden_ids)) for s, m, g in zip(sampled, pools, groups, strict=True)
            ]
            model = self._fit_once(pools, groups, sampled)
            ranks = self._ranks(model, pools, groups)
            logger.info("  round %d mined (%d rows): %s", r, self._n_rows(sampled, groups), topk_table(ranks))

        return CatBoostFit(model, ranks, rows=self._n_rows(sampled, groups))

    @staticmethod
    def _n_rows(sampled, groups: list[GoldenGroup]) -> int:
        return sum(len(s) + len(g.golden_ids) for s, g in zip(sampled, groups, strict=True))  # each pool's pins ride along

    @staticmethod
    def _ranks(model: CatBoostModel, pools, groups: list[GoldenGroup]) -> list[int]:
        """Every pool's best golden rank in its FULL pool under ``model`` — the fit-objective tie convention
        (:func:`~..metrics.best_rank`)."""
        return [best_rank(model.quality_rows(m), g.golden_ids) for m, g in zip(pools, groups, strict=True)]

    def _uniform(self, n: int, pinned: Sequence[int], rng) -> np.ndarray:
        """A uniform draw of negative row indices from one pool, every pinned row excluded. Small pools
        contribute every row they have rather than being padded with duplicates.

        This is a SECOND-STAGE draw: the pool it is handed may itself already be a uniform sample of a
        larger one (``emmy fit --pool-sample``), and a uniform draw from a uniform draw is a uniform draw
        from the original - so the two nest by construction and neither needs to know about the other.
        What that also means is that ``--negatives`` goes inert once it reaches the size of the pool it
        is given, which the caller reports rather than leaving to be inferred from a flat rank table."""
        others = np.delete(np.arange(n), list(pinned))
        if len(others) <= self.negatives:
            return others
        return rng.choice(others, size=self.negatives, replace=False)

    def _hard(self, scores: np.ndarray, pinned: Sequence[int]) -> np.ndarray:
        """The current model's top-ranked unpinned rows — the ones it would deploy INSTEAD of a verified
        config, so exactly the rows the loss needs to push down. A uniform draw over a 78k-row pool almost never
        lands here, which is why one uniform round is not enough. The window widens by the pin count so a pool
        whose pins all rank high still yields ``negatives`` rows."""
        pins = set(pinned)
        order = np.argsort(-scores, kind="stable")
        return np.array([i for i in order[: self.negatives + len(pins)] if i not in pins][: self.negatives])

    def _fit_once(self, pools, groups: list[GoldenGroup], sampled) -> CatBoostModel:
        """One CatBoost fit over the assembled groups — the pinned rows first in each, then its sampled
        negatives."""
        from catboost import Pool  # noqa: PLC0415

        x, y, gid = [], [], []
        for i, (mat, g, neg) in enumerate(zip(pools, groups, sampled, strict=True)):
            pins = np.asarray(g.golden_ids, dtype=int)
            rows = np.concatenate((pins, np.asarray(neg, dtype=int)))
            x.append(mat[rows])
            y.append(np.concatenate((np.ones(len(pins)), np.zeros(len(rows) - len(pins)))))
            gid.append(np.full(len(rows), i))
        booster = new_ranker(
            loss_function="QuerySoftMax",
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            random_seed=self.random_state,
            thread_count=-1,
            nan_mode="Min",  # absent (NaN) features get their own split-off bucket — see catboost_model
        )
        booster.fit(Pool(np.concatenate(x), np.concatenate(y), group_id=np.concatenate(gid)))
        return CatBoostModel(booster=booster, cols=self.feature_names, scale=self.scale)


@dataclass(frozen=True)
class CatBoostFit:
    """One :meth:`CatBoostTrainer.fit` result: the fitted model plus the golden ranks it reached.

    One model ranks every pool, so there is one rank list."""

    model: CatBoostModel
    ranks: list[int]
    rows: int

    def score_rows(self, group: GoldenGroup) -> np.ndarray:
        """The trainer protocol's scoring entry point, answered by the fitted model itself — the column
        choice is the model's, not a copy of it kept here. Over the group's FULL pool: training sampled
        negatives, the metric never does."""
        return self.model.score_rows(group)

    @property
    def notes(self) -> str:
        """The one-line provenance summary the artifact records."""
        return f"{topk_table(self.ranks)}; trained on {self.rows} sampled rows; not byte-reproducible (threaded fit)"
