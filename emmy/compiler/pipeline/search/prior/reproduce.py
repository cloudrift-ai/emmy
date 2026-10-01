"""Golden reproduction — the deploy-faithful gate on a prior.

With no measurement in scope, does the greedy pick reproduce the decision each golden pool records? A rank over
an exported pool is a screen; this asks the question the way a deploy asks it. Two spaces, one verdict shape:

- **schedule** — the kernel's definition at the pool's sizes through the tile lowering under the regime's pins
  alone, the resolved knobs against the pool's *closest* golden row (most knobs reproduced). Exact when every
  golden knob is reproduced.
- **placement** — the kernel walked through the lift and the cut pass with the placement prior deciding every
  placement fork (``ranking.walk_placement``), the arm it takes at the kernel's own fork — the first, where the
  golden's decision on this kernel lives; a nested decision is a pool of its own — against the arms the golden
  took. Exact when the pick is one of them.

Both run GPU-free, under the pool's own card. ``emmy eval prior`` prints the verdicts; the reproduction test holds
every repository golden to one tolerance and names what fell short, which is the signal to refit on the repository
goldens (README, "Fit the priors").
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from emmy.compiler.pipeline.search.dataset import GoldenPool

logger = logging.getLogger(__name__)


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
    from emmy.compiler.pipeline.knob import METADATA_PREFIXES  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden.repository import records_override  # noqa: PLC0415
    from emmy.compiler.pipeline.search.pins import pinned_knobs, unpinned_decisions  # noqa: PLC0415
    from emmy.compiler.pipeline.search.ranking import pool_context  # noqa: PLC0415

    with pinned_knobs(pool.pins), unpinned_decisions(), records_override(()):
        compiled = Pipeline.build(TILE_LOWERING).run(pool.kernel.program(pool.bindings), ctx=pool_context(pool))
    knobs: dict = {}
    for node in compiled.nodes.values():
        knobs.update(getattr(node.op, "knobs", None) or {})
    return bare_families({k: v for k, v in knobs.items() if not k.startswith(METADATA_PREFIXES)})


def reproduce_schedule(pools: Sequence[GoldenPool], *, kernel: str | None = None) -> list[Verdict]:
    """The schedule verdicts of every matmul pool (the kernels whose schedule the greedy ranks), one per pool."""
    from emmy.compiler.pipeline.search.dataset import is_matmul  # noqa: PLC0415

    out: list[Verdict] = []
    for pool in pools:
        if not pool.kernel.formed or not is_matmul(pool.kernel.stamps) or (kernel and kernel not in pool.kernel.name):
            continue
        try:
            found = _schedule_pick(pool)
        except Exception as exc:  # noqa: BLE001 — one pool's error must not abort the gate
            out.append(Verdict(pool, error=" ".join(f"{type(exc).__name__}: {exc}".split())[:100]))
            continue
        # The closest golden: most knobs reproduced, tie-broken by match fraction.
        scored = [(sum(knob_eq(k, row[k], found) for k in row), row) for row in pool.schedule_rows()]
        matched, golden = max(scored, key=lambda t: (t[0], t[0] / len(t[1]) if t[1] else 1.0))
        out.append(Verdict(pool, found, golden, matched, len(golden)))
    return out


def reproduce_placement(pools: Sequence[GoldenPool], scorer: Callable, *, kernel: str | None = None) -> list[Verdict]:
    """The placement verdicts of every pool with a placement fork, one per pool: its kernel's own fork under
    ``scorer`` (the placement prior's ``mean_scores_features``)."""
    from emmy.compiler.pipeline.search.ranking import placement_decisions, pool_context, walk_placement  # noqa: PLC0415

    decisions = placement_decisions(pools)
    out: list[Verdict] = []
    for pool in pools:
        if not pool.kernel.formed or (kernel and kernel not in pool.kernel.name):
            continue
        try:
            forks, _unmatched = walk_placement(pool, pool_context(pool), decisions, scorer=scorer)
        except Exception as exc:  # noqa: BLE001
            out.append(Verdict(pool, error=" ".join(f"{type(exc).__name__}: {exc}".split())[:100]))
            continue
        if forks:
            fork = forks[0]
            golden = " | ".join(fork.labels[i] for i in fork.positives)
            out.append(Verdict(pool, fork.labels[fork.pick], golden, int(fork.pick in fork.positives), 1))
    return out


def reproduction_rate(verdicts: Sequence[Verdict]) -> float:
    """The fraction of the verdicts that reached a pick that reproduced the golden exactly; 1.0 over none. A pool
    whose definition the walk could not take back (the error the export skips by name) is no verdict either way."""
    judged = [v for v in verdicts if v.error is None]
    return sum(v.ok for v in judged) / len(judged) if judged else 1.0
