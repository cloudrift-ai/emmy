"""The reproduction gate on the shipped priors: with no measurement in scope, the priors must reproduce what the
repository goldens record — the set they are fit on — at one tolerance, in both spaces: a placement fork's arm
exactly, a schedule row within the top ``SCHEDULE_TOP`` of its pool as the prior orders it. Every repository golden
runs in ``make test``, its pools in slices of ``SLICE`` so the work spreads over the xdist workers: one node is one
slice of one golden in one space, and holds the tolerance over that slice.

A red node names the rows the prior cannot reproduce. The fix is a refit on the repository goldens (README, "Fit the
priors"), or a better prior — never a lower tolerance.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pytest

from emmy.compiler.pipeline.search.golden.repository import _RECORDS_DIR, repository_golden_paths

#: The fraction of a slice's pools whose recorded decision the shipped prior must re-decide, in either space.
TOLERANCE = 0.9
#: Pools per node: enough for the tolerance to mean something, few enough that a schedule node — a 2000-row draw per
#: pool, seconds each — stays a few minutes on a CI runner.
SLICE = 32


def _golden_id(path: Path) -> str:
    return path.name if path.parent == _RECORDS_DIR else f"{path.parent.parent.name}/{path.name}"


@functools.cache
def _pools(path: Path):
    """The golden's pools, from a DB holding exactly its rows, each kernel re-lowered by this compiler — the
    schedule pools and the placement pools, which are the subset with a placement fork."""
    from emmy.compiler.pipeline.search.db import SearchDB
    from emmy.compiler.pipeline.search.db.export import golden_pools, placement_pools
    from emmy.compiler.pipeline.search.golden.evidence import file_source, import_file

    db = SearchDB()
    import_file(db, path, file_source("golden", path))
    pools, _dropped = golden_pools(db)
    return {"schedule": pools, "placement": placement_pools(db, pools)[0]}


def _parameters():
    """One node per slice of each repository golden's pools, per space, in a stable order."""
    with repository_golden_paths() as paths:
        ordered = sorted(paths, key=_golden_id)
    return [
        pytest.param(path, space, start, id=f"{_golden_id(path)}/{space}/{start // SLICE}")
        for path in ordered
        for space, pools in _pools(path).items()
        for start in range(0, len(pools), SLICE)
    ]


@pytest.mark.parametrize(("path", "space", "start"), _parameters())
def test_the_shipped_priors_reproduce_the_goldens(path: Path, space: str, start: int) -> None:
    from emmy.compiler.pipeline.search.pool import DEFAULT_SAMPLE
    from emmy.compiler.pipeline.search.prior import OfflinePrior
    from emmy.compiler.pipeline.search.prior.offline import default_file
    from emmy.compiler.pipeline.search.prior.reproduce import reproduce_placement, reproduction_rate, schedule_ranks

    prior = OfflinePrior(path=str(default_file(space)))
    pools = _pools(path)[space][start : start + SLICE]
    if space == "placement":
        verdicts = reproduce_placement(pools, prior.mean_scores_features)
    else:
        verdicts = schedule_ranks(pools, prior, sample=DEFAULT_SAMPLE)
    judged = [v for v in verdicts if v.error is None]
    if not judged:
        pytest.skip("no pool of this slice opens a fork in this space")
    rate = reproduction_rate(verdicts)
    missed = [f"{v.pool.name}: {v.found} -> {v.golden}" for v in judged if not v.ok]
    assert rate >= TOLERANCE, (
        f"{_golden_id(path)} {space}, slice {start // SLICE}: {sum(v.ok for v in judged)}/{len(judged)} reproduced, "
        f"below {TOLERANCE:.2f}; refit the {space} prior on the repository goldens (README, 'Fit the priors'):\n  " + "\n  ".join(missed)
    )
