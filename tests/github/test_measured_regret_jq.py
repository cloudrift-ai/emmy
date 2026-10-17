"""The nightly refresh's one-line summary of an `emmy eval prior --pools measured --json` report."""

import json
import subprocess
from pathlib import Path

FILTER = Path(__file__).parents[2] / ".github" / "workflows" / "scripts" / "measured_regret.jq"


def _line(report: dict) -> str:
    return subprocess.run(["jq", "-r", "-f", FILTER], input=json.dumps(report), capture_output=True, text=True, check=True).stdout.strip()


V100 = "NVIDIA Tesla V100 SXM2 16GB"
_NONE = {"median": None, "worst": None, "groups": 0}


def _summary(gpu: str, regret1: dict, regret10: dict) -> dict:
    return {"axes": {"half": "current", "gpu": gpu, "H_opt": 3}, "groups": 4, "metrics": {"regret1": regret1, "regret10": regret10}}


def test_each_card_with_a_regret_is_one_clause_without_the_metrics_it_lacks_and_dropped_rows_are_counted():
    report = {
        "header": {"rows": 320, "groups": 12, "dropped": {"implausible value": 2}},
        "summaries": [
            _summary(V100, {"median": 9.4, "worst": 56.6, "groups": 5}, {"median": 1.0, "worst": 2.4, "groups": 2}),
            _summary("NVIDIA GeForce RTX 5090", _NONE, _NONE),
            _summary("NVIDIA H100 80GB", {"median": 1.137, "worst": 813.442, "groups": 41}, _NONE),
        ],
    }
    assert _line(report) == (
        "Measured pools: 12 over 320 rows. NVIDIA Tesla V100 SXM2 16GB: regret@1 9.4x median, 56.6x worst (5); "
        "regret@10 1x median, 2.4x worst (2). NVIDIA H100 80GB: regret@1 1.14x median, 813.44x worst (41). "
        "Dropped before grouping: 2 implausible value"
    )


def test_a_report_with_no_scorable_pool_says_so():
    report = {"header": {"rows": 3, "groups": 3, "dropped": {}}, "summaries": []}
    assert _line(report) == "Measured pools: 3 over 3 rows; no pool holds two rows"
