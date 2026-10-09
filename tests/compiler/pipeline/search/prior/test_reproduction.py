"""The reproduction gate on the shipped priors: with no measurement in scope, the priors must reproduce what the
repository goldens record — the set they are fit on — at one tolerance, in both spaces: a placement fork's arm
exactly, a schedule row within the top ``SCHEDULE_TOP`` of its pool as the prior orders it. Every hardware golden
and every maintained recipe's golden runs in ``make test``, its pools in slices of ``SLICE`` so the work spreads over
the xdist workers: one node is one slice of one golden in one space, and holds the tolerance over that slice. The
schedule half draws as many rows per pool as a greedy compile in this lane does (``EMMY_POOL_DRAW``): the fit keeps
its measured 2000, and a rank fraction with the golden row kept reads the same on a smaller draw, only coarser per
pool.

A red node names the rows the prior cannot reproduce. A change to a hardware golden refits the priors in the same PR;
a recipe golden the shipped priors do not reproduce is either refit for or tagged ``prior-pending``, which skips its
nodes until a refit reproduces it (README, "Fit the priors"). Never lower the tolerance.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pytest
import yaml

from emmy import config
from emmy.compiler.pipeline.search.golden.repository import _RECORDS_DIR, repository_golden_paths
from emmy.recipe.lifecycle import PRIOR_PENDING_TAG, validate_recipe_tags

#: The fraction of a slice's pools whose recorded decision the shipped prior must re-decide, in either space. The
#: CatBoost priors re-decide every one: each slice of the 2026-10-01 refit reproduces all of its pools.
TOLERANCE = 1.0
#: Pools per node: enough for the tolerance to mean something, few enough that a schedule node — a draw per pool,
#: seconds each — stays a few minutes on a CI runner, which is several times slower than a dev box.
SLICE = 16


def _golden_id(path: Path) -> str:
    return path.name if path.parent == _RECORDS_DIR else f"{path.parent.parent.name}/{path.name}"


def _prior_pending(path: Path) -> bool:
    """Whether ``path`` is the golden of a recipe tagged ``prior-pending``: hardware goldens never are."""
    if path.parent == _RECORDS_DIR:
        return False
    recipe = yaml.safe_load((path.parent.parent / "recipe.yaml").read_text()) or {}
    return PRIOR_PENDING_TAG in validate_recipe_tags(recipe.get("tags"))


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
    return {"schedule": pools, "placement": placement_pools(db, pools)}


def _parameters():
    """One node per slice of each repository golden's pools, per space, in a stable order."""
    with repository_golden_paths() as paths:
        ordered = sorted(paths, key=_golden_id)
    pending = pytest.mark.skip(reason=f"the recipe is tagged {PRIOR_PENDING_TAG}; the refit that reproduces it drops the tag")
    return [
        *(
            pytest.param(path, None, 0, id=f"{_golden_id(path)}/{PRIOR_PENDING_TAG}", marks=pending)
            for path in ordered
            if _prior_pending(path)
        ),
        *(
            pytest.param(path, space, start, id=f"{_golden_id(path)}/{space}/{start // SLICE}")
            for path in ordered
            if not _prior_pending(path)
            for space, pools in _pools(path).items()
            for start in range(0, len(pools), SLICE)
        ),
    ]


@pytest.mark.parametrize(("path", "space", "start"), _parameters())
def test_the_shipped_priors_reproduce_the_goldens(path: Path, space: str, start: int) -> None:
    from emmy.compiler.pipeline.search.prior import OfflinePrior
    from emmy.compiler.pipeline.search.prior.offline import default_file
    from emmy.compiler.pipeline.search.prior.reproduce import reproduce_placement, reproduction_rate, schedule_ranks

    prior = OfflinePrior(path=str(default_file(space)))
    pools = _pools(path)[space][start : start + SLICE]
    if space == "placement":
        verdicts = reproduce_placement(pools, prior)
    else:
        verdicts = schedule_ranks(pools, prior, sample=config.pool_draw())
    judged = [v for v in verdicts if v.error is None]
    if not judged:
        pytest.skip("no pool of this slice opens a fork in this space")
    rate = reproduction_rate(verdicts)
    missed = [f"{v.pool.name}: {v.found} -> {v.golden}" for v in judged if not v.ok]
    assert rate >= TOLERANCE, (
        f"{_golden_id(path)} {space}, slice {start // SLICE}: {sum(v.ok for v in judged)}/{len(judged)} reproduced, "
        f"below {TOLERANCE:.2f}; report this node in the PR and leave routine refits to nightly refresh:\n  " + "\n  ".join(missed)
    )
