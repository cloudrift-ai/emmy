"""Decide whether a nightly prior fit improves the golden rank report enough to ship."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MIN_IMPROVEMENT = 0.05


def _cells(report: dict) -> dict[tuple[tuple[str, str], ...], dict]:
    if report["header"]["dataset"] != "golden":
        raise ValueError("prior promotion needs golden-pool evaluations")
    cells = {}
    for summary in report["summaries"]:
        key = tuple(sorted(summary["axes"].items()))
        if key in cells:
            raise ValueError(f"duplicate evaluation cell: {key}")
        cells[key] = summary
    if not cells:
        raise ValueError("prior evaluation has no golden pools")
    return cells


def compare(current: dict, candidate: dict) -> dict:
    """Require a 5% lower median in one cell, with no regression or coverage change in any cell."""
    if current["header"] != candidate["header"]:
        raise ValueError("prior evaluations used different datasets or filters")
    old, new = _cells(current), _cells(candidate)
    if old.keys() != new.keys():
        raise ValueError("prior evaluations have different GPU, tier, or pool-size cells")

    improved = []
    regressed = []
    compared = 0
    for key in sorted(old):
        before, after = old[key], new[key]
        if (before["groups"], before["unscored"]) != (after["groups"], after["unscored"]):
            raise ValueError(f"prior evaluation coverage changed for {key}")
        baseline = before["metrics"]["rank"]["median"]
        fitted = after["metrics"]["rank"]["median"]
        if baseline is None or fitted is None:
            if baseline != fitted:
                raise ValueError(f"prior evaluation rank coverage changed for {key}")
            continue
        compared += 1
        if fitted > baseline:
            regressed.append(key)
        elif baseline > 0 and fitted <= baseline * (1 - MIN_IMPROVEMENT):
            improved.append((key, baseline, fitted))

    if not compared:
        raise ValueError("prior evaluation has no scored golden pools")
    updated = bool(improved) and not regressed
    cells = "cell" if compared == 1 else "cells"
    if regressed:
        message = f"kept current weights: median rank rose in {len(regressed)} of {compared} {cells}"
    elif not improved:
        message = f"kept current weights: no median rank fell by at least 5% across {compared} {cells}"
    else:
        key, baseline, fitted = max(improved, key=lambda row: (row[1] - row[2]) / row[1])
        axes = dict(key)
        message = (
            f"updated weights: {len(improved)} of {compared} {cells} improved by at least 5%, none regressed; "
            f"largest change {baseline:g} to {fitted:g} on {axes['gpu']}, {axes['tier']}, {axes['pool']}"
        )
    return {"updated": updated, "message": message, "compared": compared, "improved": len(improved), "regressed": len(regressed)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("current", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.current.read_text()), json.loads(args.candidate.read_text()))
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    logger.info("%s", result["message"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
