"""The nightly prior update requires a material golden-rank gain without a regression."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[2] / ".github" / "scripts" / "compare_prior_eval.py"
SPEC = importlib.util.spec_from_file_location("compare_prior_eval", MODULE_PATH)
compare_prior_eval = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(compare_prior_eval)


def _report(*medians: float, unscored: int = 0) -> dict:
    return {
        "header": {"dataset": "golden", "source": "same dataset"},
        "summaries": [
            {
                "axes": {"half": "offline", "gpu": f"gpu{i}", "tier": "warp", "pool": "<1k"},
                "groups": 10,
                "unscored": unscored,
                "metrics": {"rank": {"median": median}},
            }
            for i, median in enumerate(medians)
        ],
    }


def test_promotes_at_five_percent_with_no_regression():
    result = compare_prior_eval.compare(_report(20, 4), _report(19, 4))
    assert result["updated"]
    assert (result["compared"], result["improved"], result["regressed"]) == (2, 1, 0)


def test_keeps_current_below_threshold_or_if_any_cell_regresses():
    assert not compare_prior_eval.compare(_report(20), _report(19.1))["updated"]
    result = compare_prior_eval.compare(_report(20, 4), _report(18, 4.1))
    assert not result["updated"]
    assert result["regressed"] == 1


def test_refuses_incomparable_reports():
    with pytest.raises(ValueError, match="different datasets"):
        compare_prior_eval.compare(_report(20), {**_report(18), "header": {"dataset": "golden", "source": "other"}})
    with pytest.raises(ValueError, match="coverage changed"):
        compare_prior_eval.compare(_report(20), _report(18, unscored=1))
    with pytest.raises(ValueError, match="different GPU"):
        compare_prior_eval.compare(_report(20), _report(18, 4))
