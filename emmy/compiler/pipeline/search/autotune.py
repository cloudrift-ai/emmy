"""Autotuning one kernel: which complete schedule rows to measure next, given the ones already measured.

``emmy run --tune N`` drives it: the program's one scheduled kernel is enumerated under the greedy kernel set
(:func:`schedule_space`), the prior's ten best rows are measured first, then each batch is the rows with the highest
expected improvement under a Gaussian process fit to the log latencies measured so far (Bayesian optimization). The
caller measures — this module never touches the GPU.

A knob value is split into its parts before the process sees it (``TILE`` ``mma_m16n8k16_f16_f32/f2x4/k2`` → the
atom, a ``2×4`` fragment count and a ``k2`` step), so two rows that differ in one tile dimension are near each other.
On the 5090, against a local search over the same parts, the process matched it on three kernels and found a 1.5×
faster attention kernel that local search could not reach from a local optimum.
"""

from __future__ import annotations

import math
import re

import numpy as np

from emmy.compiler.pipeline.search.features import Featurizer

#: Rows the prior picks before the process has anything to fit.
START = 10
#: Random rows scored per proposal, beside the best row's one-part neighbours.
_SCORED = 8000
_PART = re.compile(r"^([a-z_]*?)(\d+)(?:x(\d+))?$")


def schedule_space(graph, ctx, db=None) -> tuple[object, list[dict]]:
    """The one scheduled kernel of ``graph`` and every complete schedule row its fork offers, under the kernel set
    the greedy compile takes. A program with several scheduled kernels is refused: a bare pin reaches them all."""
    from emmy.compiler.pipeline import TILE_PASSES, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.fork import iter_leaves, leaf_knobs  # noqa: PLC0415
    from emmy.compiler.pipeline.pipeline import Run  # noqa: PLC0415
    from emmy.compiler.pipeline.search.policy.greedy import _schedule_fork, greedy_decide  # noqa: PLC0415
    from emmy.compiler.pipeline.search.space import WORK  # noqa: PLC0415

    greedy = greedy_decide(db=db)
    pools: dict[str, tuple[object, list[dict]]] = {}

    def decide(fp):
        if _schedule_fork(fp) and fp.node_id not in pools:
            pools[fp.node_id] = (fp.root_op, [row for leaf in iter_leaves(fp.options) if WORK.name in (row := leaf_knobs(leaf))])
        return greedy(fp)

    Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx, db=db).resolve(graph, decide)
    if len(pools) != 1:
        raise ValueError(f"--tune tunes a program with one scheduled kernel; this one has {len(pools)}")
    return next(iter(pools.values()))


def _parts(row: dict) -> dict:
    """A row's knob values split into named parts: numbers where the value spells them, else the token."""
    out: dict = {}
    for knob, value in row.items():
        for i, part in enumerate(str(value).split("/") if value else []):
            m = _PART.match(part)
            if m and not part.startswith("mma"):
                out[f"{knob}.{m.group(1) or i}"] = int(m.group(2))
                if m.group(3):
                    out[f"{knob}.{m.group(1) or i}.1"] = int(m.group(3))
            else:
                out[f"{knob}.{i}"] = part
    return out


def _encode(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Each row as level indices per part (for neighbours) and as a point the process measures distance in: a
    number by its log2 scaled to [0, 1], a token one-hot, an absent part as its own level."""
    parts = [_parts(r) for r in rows]
    names = sorted({k for p in parts for k in p})
    levels, points = [], []
    for name in names:
        values = [p.get(name) for p in parts]
        present = sorted({v for v in values if v is not None}, key=str)
        index = {v: i + 1 for i, v in enumerate(present)}
        col = np.array([index.get(v, 0) for v in values])
        levels.append(col)
        if all(isinstance(v, int) for v in present):
            logs = np.array([0.0] + [math.log2(max(v, 1)) for v in present])
            span = max(np.ptp(logs[1:]), 1.0) if present else 1.0
            points += [((logs - (logs[1:].min() if present else 0.0)) / span)[col][:, None], (col == 0).astype(float)[:, None]]
        else:
            points.append(np.eye(len(present) + 1)[col] / math.sqrt(2))
    return np.stack(levels, 1), np.hstack(points)


def _erf(x: np.ndarray) -> np.ndarray:
    t = 1 / (1 + 0.3275911 * np.abs(x))
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return np.sign(x) * y


def prior_scores(op, rows: list[dict], ctx) -> list[float]:
    """The schedule prior's score of every row, lower is better."""
    from emmy.compiler.pipeline.search.prior import load_prior  # noqa: PLC0415

    prior, featurizer = load_prior(), Featurizer.of(ctx)
    scores: list[float] = []
    for i in range(0, len(rows), 4096):
        scores += prior.mean_scores_features([featurizer.features(op, r) for r in rows[i : i + 4096]])
    return scores


class Autotuner:
    """The proposal side of one kernel's tune over ``rows`` ranked by ``scores`` (lower is better). ``propose(k)``
    returns up to ``k`` unmeasured row indices; ``observe`` takes their microseconds (``math.inf`` for a row that
    failed to compile, bench or match the reference)."""

    def __init__(self, rows: list[dict], scores: list[float], *, seed: int = 0) -> None:
        self.rows = rows
        self.levels, self.points = _encode(rows)
        self.ranked = [int(i) for i in np.argsort(scores, kind="stable")]
        self.us: dict[int, float] = {}
        self.rng = np.random.default_rng(seed)

    def observe(self, results) -> None:
        """``results``: ``(row index, µs)`` pairs."""
        self.us.update(results)

    def best(self) -> int | None:
        ok = {i: us for i, us in self.us.items() if math.isfinite(us)}
        return min(ok, key=ok.get) if ok else None

    def propose(self, k: int) -> list[int]:
        """Up to ``k`` unmeasured row indices: the prior's best first, then expected improvement."""
        fresh = [i for i in self.ranked[:START] if i not in self.us]
        if fresh or self.best() is None:
            return (fresh or [i for i in self.ranked if i not in self.us])[:k]
        seen = list(self.us)
        y = np.log([self.us[i] for i in seen if math.isfinite(self.us[i])])
        # A row that failed is a bad row, not a missing one: scored just past the slowest that ran.
        y = np.concatenate([y, np.full(len(seen) - len(y), y.max() + 0.5)])
        seen = [i for i in seen if math.isfinite(self.us[i])] + [i for i in seen if not math.isfinite(self.us[i])]
        y = (y - y.mean()) / (y.std() or 1.0)
        x = self.points[seen]
        d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
        fits = []
        for scale in (0.25, 0.5, 1.0, 2.0):
            try:
                lc = np.linalg.cholesky(np.exp(-d2 / (2 * scale * scale)) + 1e-3 * np.eye(len(seen)))
            except np.linalg.LinAlgError:
                continue
            alpha = np.linalg.solve(lc.T, np.linalg.solve(lc, y))
            fits.append((-0.5 * y @ alpha - np.log(np.diag(lc)).sum(), scale, lc, alpha))
        _, scale, lc, alpha = max(fits, key=lambda f: f[0])
        best = self.best()
        near = np.flatnonzero((self.levels != self.levels[best]).sum(1) == 1)
        pool = np.unique(np.concatenate([self.rng.choice(len(self.rows), min(len(self.rows), _SCORED), replace=False), near]))
        pool = pool[~np.isin(pool, seen)]
        if not len(pool):
            return []
        xc = self.points[pool]
        kc = np.exp(-np.maximum((xc * xc).sum(1)[:, None] + (x * x).sum(1)[None, :] - 2 * xc @ x.T, 0) / (2 * scale * scale))
        mean = kc @ alpha
        sd = np.sqrt(np.maximum(1 - (np.linalg.solve(lc, kc.T) ** 2).sum(0), 1e-12))
        z = (y.min() - mean) / sd
        ei = (y.min() - mean) * 0.5 * (1 + _erf(z / math.sqrt(2))) + sd * np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        return [int(i) for i in pool[np.argsort(-ei, kind="stable")[:k]]]
