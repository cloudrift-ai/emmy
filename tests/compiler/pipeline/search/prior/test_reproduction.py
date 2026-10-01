"""The reproduction gate on the shipped priors: with no measurement in scope, the greedy must reproduce what the
repository goldens record — the set the priors are fit on — at one tolerance, per golden file and space. The hardware
goldens' placement forks run in ``make test``; a model golden's walk and the schedule half take minutes per file and
run under ``make test-priors``.

A red node names the rows the prior cannot reproduce. The fix is a refit on the repository goldens (README, "Fit the
priors"), or a better prior — never a lower tolerance.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pytest

from emmy.compiler.pipeline.search.golden.repository import _RECORDS_DIR, repository_golden_paths

#: The fraction of a golden file's pools whose recorded decision the shipped prior must re-decide, in either space.
TOLERANCE = 0.9


def _golden_id(path: Path) -> str:
    return path.name if path.parent == _RECORDS_DIR else f"{path.parent.parent.name}/{path.name}"


def _parameters():
    """One node per corpus and space: the hardware goldens' placement forks on the default lane; the whole
    repository corpus, and the schedule half, under the off-lane ``priors`` marker (``make test-priors``)."""
    return [
        pytest.param("hardware", "placement", id="hardware/placement"),
        pytest.param("repository", "placement", id="repository/placement", marks=pytest.mark.priors),
        pytest.param("hardware", "schedule", id="hardware/schedule", marks=pytest.mark.priors),
        pytest.param("repository", "schedule", id="repository/schedule", marks=pytest.mark.priors),
    ]


def _paths(corpus: str) -> list[Path]:
    with repository_golden_paths() as paths:
        return sorted((p for p in paths if corpus == "repository" or p.parent == _RECORDS_DIR), key=_golden_id)


@functools.cache
def _pools(path: Path):
    """The golden's pools, from a DB holding exactly its rows, each kernel re-lowered by this compiler."""
    from emmy.compiler.pipeline.search.db import SearchDB
    from emmy.compiler.pipeline.search.db.export import golden_pools, placement_pools
    from emmy.compiler.pipeline.search.golden.evidence import file_source, import_file

    db = SearchDB()
    import_file(db, path, file_source("golden", path))
    pools, _dropped = golden_pools(db)
    return pools, placement_pools(db, pools)[0]


@pytest.mark.parametrize(("corpus", "space"), _parameters())
def test_the_shipped_priors_reproduce_the_goldens(corpus: str, space: str) -> None:
    from emmy.compiler.pipeline.search.prior import OfflinePrior
    from emmy.compiler.pipeline.search.prior.offline import default_file
    from emmy.compiler.pipeline.search.prior.reproduce import reproduce_placement, reproduce_schedule, reproduction_rate

    scorer = OfflinePrior(path=str(default_file("placement"))).mean_scores_features if space == "placement" else None
    verdicts = []
    for path in _paths(corpus):
        pools, placement = _pools(path)
        found = reproduce_placement(placement, scorer) if space == "placement" else reproduce_schedule(pools)
        verdicts.extend((path, v) for v in found)
    rate = reproduction_rate([v for _, v in verdicts])
    judged = [v for _, v in verdicts if v.error is None]
    missed = [f"{_golden_id(path)} {v.pool.name}: {v.found} -> {v.golden}" for path, v in verdicts if v.error is None and not v.ok]
    assert rate >= TOLERANCE, (
        f"{corpus} goldens, {space}: {sum(v.ok for v in judged)}/{len(judged)} reproduced, below {TOLERANCE:.2f}; "
        f"refit the {space} prior on the repository goldens (README, 'Fit the priors'):\n  " + "\n  ".join(missed)
    )
