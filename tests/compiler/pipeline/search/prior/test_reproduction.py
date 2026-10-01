"""The reproduction gate on the shipped priors: with no measurement in scope, the greedy must reproduce what the
repository goldens record — exactly on the hardware goldens' placement forks, which the placement prior is fit on
and which ``make test`` runs, and at least at the recorded rate elsewhere (``FLOORS``: a ratchet, raised when a refit
improves it), under ``make test-priors``.

A red node names the rows the prior cannot reproduce. The fix is a refit (README, "Fit the priors"), and where the
rows are a model golden's, extending the hardware golden with them first (``emmy golden extract``) — never a
lower floor to make the node green without naming the change that did it.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pytest

from emmy.compiler.pipeline.search.golden.repository import _RECORDS_DIR, repository_golden_paths

# ``<golden id>: {space: floor}`` — the fraction of a file's verdicts the shipped priors reproduce exactly. The
# hardware goldens' placement floor is 1.0 by construction: the placement prior is fit on them.
FLOORS: dict[str, dict[str, float]] = {
    "h100_sm90.json": {"placement": 1.0, "schedule": 0.0},
    "rtx4080_sm89.json": {"placement": 1.0, "schedule": 0.0},
    "rtx4090_sm89.json": {"placement": 1.0, "schedule": 0.0},
    "rtx5090_sm120.json": {"placement": 1.0, "schedule": 0.0},
    "rtxpro6000_sm120.json": {"placement": 1.0, "schedule": 0.0},
    "v100_sm70.json": {"placement": 1.0, "schedule": 0.0},
    "DeepSeek-V4-Flash-0731/v100_sm70.json": {"placement": 0.82, "schedule": 0.03},
    "OLMoE-1B-7B-0125-Instruct/rtx5090_sm120.json": {"placement": 1.0, "schedule": 0.0},
    "Qwen3.5-122B-A10B/v100_sm70.json": {"placement": 1.0, "schedule": 0.0},
    "Qwen3.8-27B-AWQ-INT4/v100_sm70.json": {"placement": 0.6, "schedule": 0.25},
    "Qwen3.8-27B-EXL3/v100_sm70.json": {"placement": 0.5, "schedule": 0.33},
    "Qwen3.8-27B-FP8/v100_sm70.json": {"placement": 0.5, "schedule": 0.6},
    "Qwen3.8-27B-GPTQ-Int4/v100_sm70.json": {"placement": 0.6, "schedule": 0.25},
    "gemma-4-12B-it/rtx4090_sm89.json": {"placement": 1.0, "schedule": 0.0},
    "gemma-4-12B-it/rtx5090_sm120.json": {"placement": 0.5, "schedule": 0.0},
}


def _golden_id(path: Path) -> str:
    return path.name if path.parent == _RECORDS_DIR else f"{path.parent.parent.name}/{path.name}"


def _parameters():
    with repository_golden_paths() as paths:
        files = sorted(paths, key=_golden_id)

    # The default lane holds the exact gate alone — the hardware goldens' placement forks, seconds per file. The
    # schedule half re-walks every kernel's pool through the greedy, minutes per file, and a model golden's
    # placement walk is minutes too: both carry the off-lane ``priors`` marker (``make test-priors`` runs them).
    def marks(path: Path, space: str) -> tuple:
        return () if space == "placement" and path.parent == _RECORDS_DIR else (pytest.mark.priors,)

    spaces = ("placement", "schedule")
    return [pytest.param(path, space, id=f"{_golden_id(path)}/{space}", marks=marks(path, space)) for path in files for space in spaces]


@functools.lru_cache(maxsize=2)
def _pools(path: Path):
    """The golden's pools, from a DB holding exactly its rows, each kernel re-lowered by this compiler."""
    from emmy.compiler.pipeline.search.db import SearchDB
    from emmy.compiler.pipeline.search.db.export import golden_pools, placement_pools
    from emmy.compiler.pipeline.search.golden.evidence import file_source, import_file

    db = SearchDB()
    import_file(db, path, file_source("golden", path))
    pools, _dropped = golden_pools(db)
    return pools, placement_pools(db, pools)[0]


@pytest.mark.parametrize(("path", "space"), _parameters())
def test_the_shipped_priors_reproduce_the_golden(path: Path, space: str) -> None:
    from emmy.compiler.pipeline.search.prior import OfflinePrior
    from emmy.compiler.pipeline.search.prior.offline import default_file
    from emmy.compiler.pipeline.search.prior.reproduce import reproduce_placement, reproduce_schedule, reproduction_rate

    floors = FLOORS.get(_golden_id(path))
    assert floors is not None, f"{_golden_id(path)} has no reproduction floor: measure it (`emmy eval prior`) and record it in FLOORS"
    pools, placement = _pools(path)
    if space == "placement":
        verdicts = reproduce_placement(placement, OfflinePrior(path=str(default_file("placement"))).mean_scores_features)
    else:
        verdicts = reproduce_schedule(pools)
    rate = reproduction_rate(verdicts)
    missed = [f"{v.pool.name}: {v.error or f'{v.found} -> {v.golden}'}" for v in verdicts if not v.ok]
    assert rate >= floors[space], (
        f"{_golden_id(path)} {space}: {sum(v.ok for v in verdicts)}/{len(verdicts)} reproduced, floor {floors[space]:.2f}; "
        f"refit the {space} prior, or extend the hardware golden first with "
        f"`emmy golden extract {path} {_RECORDS_DIR / path.name} --kernel NAME`:\n  " + "\n  ".join(missed)
    )
