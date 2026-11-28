"""The reproduction gate on the shipped priors: with no measurement in scope, the priors must reproduce what the
hardware goldens record — the set they are fit on — at one tolerance, in both spaces: a placement fork's arm
exactly, a schedule row within the top ``SCHEDULE_TOP`` of its pool as the prior orders it. Every hardware golden
runs in ``make test``, its pools in slices of ``SLICE`` so the work spreads over
the xdist workers: one node is one slice of one golden in one space, and holds the tolerance over that slice. A slice
takes every ``n``-th pool, ``n`` the golden's slice count, not a contiguous run: the export orders the pools heaviest
first (the decisions with the longest arms), and a placement pool costs what realizing every arm of every fork on its
root costs, minutes for the fused prefill kernels of a 1,000-kernel golden, so contiguous slices put a whole hour on
one node. The
schedule half draws as many rows per pool as a greedy compile in this lane does (``EMMY_POOL_DRAW``): the fit keeps
its measured 2000, and a rank fraction with the golden row kept reads the same on a smaller draw, only coarser per
pool.

A red node names the rows the prior cannot reproduce. A change to a hardware golden refits the priors in the same PR
(README, "Fit the priors"); a recipe's golden is neither trained on nor gated, and its rows reach the priors once
promoted into the card's hardware golden. Never lower the tolerance. A hardware golden's node the
user has accepted as a known miss is ``xfail`` in ``KNOWN_MISSES``, strict: the refit that reproduces it turns the
node into an unexpected pass, which is the signal to drop the entry.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pytest

from emmy import config
from emmy.compiler.pipeline.search.golden.repository import hardware_golden_paths
from tests.compiler.helpers import golden_lane

#: The fraction of a slice's pools whose recorded decision the shipped prior must re-decide, in either space. The
#: CatBoost priors re-decide every one: each slice of the 2026-10-01 refit reproduces all of its pools.
TOLERANCE = 1.0
#: Pools per node: enough for the tolerance to mean something, few enough that a schedule node — a draw per pool,
#: seconds each — stays a few minutes on a CI runner, which is several times slower than a dev box.
SLICE = 16


@functools.cache
def _imported(path: Path):
    """A DB holding exactly the golden's rows, each kernel re-lowered by this compiler, with the document and the
    source they were filed under. The constants the targets' programs bind are not filed here: collection only counts
    pools, and filing them lowers a twin per target (:func:`_placement_pools` files them when a placement node runs)."""
    from emmy.compiler.pipeline.search.db import SearchDB
    from emmy.compiler.pipeline.search.golden import GoldenFile
    from emmy.compiler.pipeline.search.golden.evidence import file_source, import_rows, regime_context

    db = SearchDB()
    document = GoldenFile.load(path)
    source = file_source("golden", path)
    for regime in sorted({tuple(sorted(row.pins.items())) for row in document.rows}, key=repr):
        rows = [row for row in document.rows if tuple(sorted(row.pins.items())) == regime]
        import_rows(db, regime_context(document, dict(regime)), document, rows, source=source)
    return db, document, source


@functools.cache
def _pools(path: Path):
    """The golden's pools — the schedule pools and the placement pools, which are the subset with a placement fork."""
    from emmy.compiler.pipeline.search.db.export import golden_pools, placement_pools

    db, _document, _source = _imported(path)
    pools, _dropped = golden_pools(db)
    return {"schedule": pools, "placement": placement_pools(db, pools)}


@functools.cache
def _placement_pools(path: Path):
    """The placement pools with the constant inputs their programs bind, filed once per golden on first use."""
    from emmy.compiler.pipeline.search.db.export import placement_pools
    from emmy.compiler.pipeline.search.golden.evidence import import_constants

    db, document, source = _imported(path)
    import_constants(db, document, document.rows, source=source)
    return placement_pools(db, _pools(path)["schedule"])


_PLACEMENT = "the placement prior misses a cut or split fork it trained on"
_SCHEDULE = "the schedule prior misses a pool it trained on"
KNOWN_MISSES: dict[tuple[str, str, int], str] = {
    ("a100_sm80.json", "schedule", 24): _SCHEDULE,
    ("a100_sm80.json", "schedule", 31): _SCHEDULE,
    ("h100_sm90.json", "placement", 13): _PLACEMENT,
    ("h100_sm90.json", "schedule", 20): _SCHEDULE,
    ("h100_sm90.json", "schedule", 46): _SCHEDULE,
    ("h100_sm90.json", "schedule", 49): _SCHEDULE,
    ("h100_sm90.json", "schedule", 87): _SCHEDULE,
    ("rtx4090_sm89.json", "placement", 0): _PLACEMENT,
    ("rtx4090_sm89.json", "placement", 6): _PLACEMENT,
    ("rtx4090_sm89.json", "placement", 8): _PLACEMENT,
    ("rtx4090_sm89.json", "placement", 58): _PLACEMENT,
    ("rtx5090_sm120.json", "placement", 1): _PLACEMENT,
    ("rtx5090_sm120.json", "placement", 21): _PLACEMENT,
    ("rtxpro6000_se_sm120.json", "placement", 4): _PLACEMENT,
    ("rtxpro6000_se_sm120.json", "placement", 14): _PLACEMENT,
    ("rtxpro6000_se_sm120.json", "placement", 34): _PLACEMENT,
    ("rtxpro6000_se_sm120.json", "placement", 62): _PLACEMENT,
    ("rtxpro6000_se_sm120.json", "placement", 63): _PLACEMENT,
    ("rtxpro6000_se_sm120.json", "placement", 88): _PLACEMENT,
    ("rtxpro6000_se_sm120.json", "placement", 108): _PLACEMENT,
    ("rtxpro6000_se_sm120.json", "placement", 119): _PLACEMENT,
    ("rtxpro6000_se_sm120.json", "placement", 122): _PLACEMENT,
    ("rtxpro6000_se_sm120.json", "placement", 127): _PLACEMENT,
    ("rtxpro6000_se_sm120.json", "schedule", 4): _SCHEDULE,
    ("rtxpro6000_se_sm120.json", "schedule", 13): _SCHEDULE,
    ("rtxpro6000_se_sm120.json", "schedule", 84): _SCHEDULE,
    ("v100_sxm2_sm70.json", "placement", 10): _PLACEMENT,
    ("v100_sxm2_sm70.json", "placement", 20): _PLACEMENT,
    ("v100_sxm2_sm70.json", "schedule", 77): _SCHEDULE,
}


def _marks(path: Path, space: str, k: int) -> tuple:
    reason = KNOWN_MISSES.get((path.name, space, k))
    return golden_lane(path) + ((pytest.mark.xfail(reason=reason, strict=True),) if reason else ())


def _parameters():
    """One node per slice of each hardware golden's pools, per space, in a stable order."""
    return [
        pytest.param(path, space, k, id=f"{path.name}/{space}/{k}", marks=_marks(path, space, k))
        for path in hardware_golden_paths()
        for space, pools in _pools(path).items()
        for k in range(_slices(pools))
    ]


def _slices(pools) -> int:
    return -(-len(pools) // SLICE)


@pytest.mark.parametrize(("path", "space", "k"), _parameters())
def test_the_shipped_priors_reproduce_the_goldens(path: Path, space: str, k: int) -> None:
    from emmy.compiler.pipeline.search.prior import OfflinePrior
    from emmy.compiler.pipeline.search.prior.offline import default_file
    from emmy.compiler.pipeline.search.prior.reproduce import reproduce_placement, reproduction_rate, schedule_ranks

    prior = OfflinePrior(path=str(default_file(space)))
    every = _placement_pools(path) if space == "placement" else _pools(path)["schedule"]
    pools = every[k :: _slices(every)]
    if space == "placement":
        # The decisions in scope are the whole golden's: a cut recorded on a piece outside the slice is on the path
        # of a kernel inside it.
        verdicts = reproduce_placement(pools, prior, among=every)
    else:
        verdicts = schedule_ranks(pools, prior, sample=config.pool_draw())
    judged = [v for v in verdicts if v.error is None]
    if not judged:
        pytest.skip("no pool of this slice opens a fork in this space")
    rate = reproduction_rate(verdicts)
    missed = [f"{v.pool.name}: {v.found} -> {v.golden}" for v in judged if not v.ok]
    assert rate >= TOLERANCE, (
        f"{path.name} {space}, slice {k}: {sum(v.ok for v in judged)}/{len(judged)} reproduced, "
        f"below {TOLERANCE:.2f}; report this node in the PR and leave routine refits to nightly refresh:\n  " + "\n  ".join(missed)
    )
