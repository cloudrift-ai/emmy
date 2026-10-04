"""Golden reproduction — the deploy-faithful gate on a prior.

With no measurement in scope, does the greedy pick reproduce the decision each golden pool records? A rank over
an exported pool is a screen; this asks the question the way a deploy asks it. Two spaces, one verdict shape:

- **schedule** — the kernel's definition at the pool's sizes through the tile lowering under the regime's pins
  alone, the resolved knobs against the pool's *closest* golden row (most knobs reproduced). Exact when every
  golden knob is reproduced — what ``emmy eval prior`` prints. The gate asks the approximate form instead
  (:func:`schedule_ranks`): where the golden row sits in the pool as the prior orders it, reproduced when it is
  within the top ``SCHEDULE_TOP`` of the pool. A pool holds tens of thousands of rows within noise of each other,
  so the exact form is a bar no ranker clears, while the rank is a baseline a better prior tightens.
- **placement** — the kernel walked through the lift and the cut pass with the placement prior deciding every
  placement fork (``ranking.walk_placement``), the arm it takes at the kernel's own fork — the first, where the
  golden's decision on this kernel lives; a nested decision is a pool of its own — against the arms the golden
  took. Exact when the pick is one of them.

Both run GPU-free, under the pool's own card. ``emmy eval prior`` prints the verdicts; the reproduction test holds
every repository golden to one tolerance and names what fell short. Report failures in the PR; routine refits
belong to nightly refresh (README, "Fit the priors").
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from emmy.compiler.pipeline.search.dataset import GoldenPool

logger = logging.getLogger(__name__)

#: The fraction of a schedule pool the golden row must sit within, as the schedule prior orders it, to count as
#: reproduced — the baseline the gate holds the shipped weights to, tightened as the prior improves. Measured on
#: the hardware goldens' 191 pools (2026-09-30): the golden is in the better half of 93% of them, in the top 10% of
#: 78%, and the median rank is 3.8% of the pool — so the half is where the gate's tolerance holds today.
SCHEDULE_TOP = 0.5


@dataclass(frozen=True)
class Verdict:
    """One pool's reproduction: what the greedy found against what the golden recorded, and how much of the
    golden it reproduced (``matched`` of ``total``) — or the error that stopped the pick."""

    pool: GoldenPool
    found: dict | str = field(default_factory=dict)
    golden: dict | str = field(default_factory=dict)
    matched: int = 0
    total: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.matched == self.total


def knob_eq(key: str, golden_value, found: dict) -> bool:
    """Whether the picked knob dict ``found`` reproduces the golden's ``key=golden_value`` — value equality
    through the registry-canonical :func:`~emmy.compiler.pipeline.knob.values_equal`, so a legacy spelling
    in the golden corpus keeps matching the site-form pick."""
    from emmy.compiler.pipeline.knob import values_equal  # noqa: PLC0415

    return key in found and values_equal(key, golden_value, found[key])


def bare_families(knobs: dict) -> dict:
    """Exact-site tuning knobs aggregated by family — one resolved choice per family rather than schedule
    identities; the first key wins on a family collision."""
    from emmy.compiler.pipeline.knob import family_of  # noqa: PLC0415

    out: dict = {}
    for key, value in knobs.items():
        out.setdefault(family_of(key), value)
    return out


def _schedule_pick(pool: GoldenPool) -> dict:
    from emmy.compiler.pipeline import TILE_LOWERING, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden.repository import evidence_scope  # noqa: PLC0415
    from emmy.compiler.pipeline.search.pins import pinned_knobs, unpinned_decisions  # noqa: PLC0415
    from emmy.compiler.pipeline.search.ranking import pool_context  # noqa: PLC0415

    with pinned_knobs(pool.pins), unpinned_decisions(), evidence_scope([]):
        compiled = Pipeline.build(TILE_LOWERING).run(pool.kernel.program(pool.bindings), ctx=pool_context(pool))
    knobs: dict = {}
    for node in compiled.nodes.values():
        knobs.update(getattr(node.op, "knobs", None) or {})
    return bare_families(knobs)


def reproduce_schedule(pools: Sequence[GoldenPool], *, kernel: str | None = None) -> list[Verdict]:
    """The schedule verdicts of every matmul pool (the kernels whose schedule the greedy ranks), one per pool."""
    from emmy.compiler.pipeline.search.dataset import is_matmul  # noqa: PLC0415
    from emmy.compiler.pipeline.search.features import stamps  # noqa: PLC0415

    out: list[Verdict] = []
    for pool in pools:
        if not pool.kernel.formed or (kernel and kernel not in pool.kernel.name):
            continue
        try:
            if not is_matmul(stamps(pool.kernel.op(pool.bindings))):
                continue
            found = _schedule_pick(pool)
        except Exception as exc:  # noqa: BLE001 — one pool's error must not abort the gate
            out.append(Verdict(pool, error=" ".join(f"{type(exc).__name__}: {exc}".split())[:100]))
            continue
        # The closest golden: most knobs reproduced, tie-broken by match fraction.
        scored = [(sum(knob_eq(k, row[k], found) for k in row), row) for row in pool.schedule_rows()]
        matched, golden = max(scored, key=lambda t: (t[0], t[0] / len(t[1]) if t[1] else 1.0))
        out.append(Verdict(pool, found, golden, matched, len(golden)))
    return out


def reproduce_placement(pools: Sequence[GoldenPool], prior, *, kernel: str | None = None) -> list[Verdict]:
    """The placement verdicts of every pool with a placement fork, one per pool: its kernel's own fork under
    the placement ``prior``."""
    from emmy.compiler.pipeline.search.ranking import placement_decisions, pool_context, walk_placement  # noqa: PLC0415

    out: list[Verdict] = []
    for pool in pools:
        if not pool.kernel.formed or (kernel and kernel not in pool.kernel.name):
            continue
        try:
            forks, _unmatched = walk_placement(pool, pool_context(pool), placement_decisions(pools, pool), prior, first=True)
        except Exception as exc:  # noqa: BLE001
            out.append(Verdict(pool, error=" ".join(f"{type(exc).__name__}: {exc}".split())[:100]))
            continue
        if forks:
            fork = forks[0]
            golden = " | ".join(fork.labels[i] for i in fork.positives)
            out.append(Verdict(pool, fork.labels[fork.pick], golden, int(fork.pick in fork.positives), 1))
    return out


def schedule_ranks(pools: Sequence[GoldenPool], prior, *, sample: int, kernel: str | None = None) -> list[Verdict]:
    """The schedule verdicts of every golden pool: the golden row's rank in the pool as ``prior`` orders it — the
    pool enumerated as the dataset builds it (``ranking.build_golden_groups``, ``sample`` rows drawn, the golden
    kept), tie-pessimistic like a greedy argmin — reproduced when within the top :data:`SCHEDULE_TOP` of the rows."""
    from emmy.compiler.pipeline.search.metrics import best_dual_rank  # noqa: PLC0415
    from emmy.compiler.pipeline.search.ranking import build_golden_groups  # noqa: PLC0415

    groups, _skipped = build_golden_groups(pools, "*", sample=sample, seed=0, kernel=kernel)
    out: list[Verdict] = []
    for group in groups:
        quality = prior.score_rows(group)
        rank, _optimistic = best_dual_rank(quality, group.golden_ids)
        rows = len(group.feats)
        out.append(Verdict(group.pools[0], f"rank {rank} of {rows}", f"top {SCHEDULE_TOP:.0%}", int(rank <= SCHEDULE_TOP * rows), 1))
    return out


def reproduction_rate(verdicts: Sequence[Verdict]) -> float:
    """The fraction of the verdicts that reached a pick that reproduced the golden exactly; 1.0 over none. A pool
    whose definition the walk could not take back (the error the export skips by name) is no verdict either way."""
    judged = [v for v in verdicts if v.error is None]
    return sum(v.ok for v in judged) / len(judged) if judged else 1.0
