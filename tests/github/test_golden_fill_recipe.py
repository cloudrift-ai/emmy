"""The nightly golden fill recipe lists every hardware golden once, on the card its file names, so the job can fill
any file it picks with ``--filter golden=PATH``."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from emmy.compiler.pipeline.search.golden import GoldenFile
from emmy.compiler.pipeline.search.golden.repository import _RECORDS_DIR

ROOT = Path(__file__).parents[2]
RECIPE = ROOT / ".github" / "golden-fill" / "recipe.yaml"
FILL = ROOT / ".github" / "golden-fill" / "fill.sh"


def test_fill_recipe_matrix_is_every_hardware_golden_on_its_card():
    expected = {(str(path.relative_to(ROOT)), GoldenFile.load(path).gpu_name) for path in _RECORDS_DIR.glob("*.json")}
    rows = yaml.safe_load(RECIPE.read_text())["matrices"]
    assert sorted((row["golden"], row["deploy.gpu"]) for row in rows) == sorted(expected)


# A stand-in for `emmy`: `golden list --missing --json PATH` writes the rows it is given, `run` logs its arguments and
# the knobs it was pinned under, and fails for the one row named FAIL.
_STUB = """#!/bin/bash
if [ "$1" = golden ]; then
  out="${@: -1}"
  case "$out" in *still_missing.json) echo "[]" > "$out" ;; *) cp "$STUB_MISSING" "$out" ;; esac
  exit 0
fi
printf '%s|%s|%s\\n' "${EMMY_KNOBS-}" "${EMMY_FAST_MATH-}" "$*" >> "$STUB_LOG"
case "$*" in *FAIL*) exit 1 ;; esac
"""


@pytest.mark.skipif(shutil.which("jq") is None, reason="the fill script reads its rows with jq")
def test_the_fill_records_every_scheduled_row_by_its_kernel_before_any_whole_target(tmp_path):
    """A target whose rows were all demoted has nothing a strict whole-program record can price, so the fill records
    the rows with a schedule first, each as an isolated record of its own kernel: a piece under its knobs, a target
    under `PLACE=fuse` and its knobs. Only then the whole targets: a seed with no schedule, and a target missing its
    torch.compile time. A failed row is listed with its phase."""

    def entry(row, kernel, knobs, *, lane=None, missing="emmy", piece=False):
        pins = {} if lane is None else {"FAST_MATH": lane}
        return {"row": row, "kernel": kernel, "knobs": knobs, "pins": pins, "missing": missing, "piece": piece}

    missing = [
        entry("seed", "k_a", {}, lane=False),
        entry("piece.1", "k_a__place_1", {"WORK": "t128"}, piece=True),
        entry("fused", "k_b", {"WORK": "w2x2", "TILE": "mma"}, lane=True),
        entry("timed", "k_c", {"WORK": "t64"}, missing="tcompile"),
        entry("piece.FAIL", "k_a__place_2", {"WORK": "t32"}, piece=True),
    ]
    stub = tmp_path / "emmy"
    stub.write_text(_STUB)
    stub.chmod(0o755)
    (tmp_path / "missing.fixture.json").write_text(json.dumps(missing))
    golden = tmp_path / "golden.json"
    golden.write_text("{}")
    task = tmp_path / "task"
    task.mkdir()
    fixture, log = tmp_path / "missing.fixture.json", tmp_path / "log"
    env = {"PATH": "/usr/bin:/bin", "EMMY": str(stub), "STUB_MISSING": str(fixture), "STUB_LOG": str(log)}
    subprocess.run(["bash", str(FILL), str(golden), str(task), "600"], env=env, check=True)

    calls = [line.split("|") for line in log.read_text().splitlines()]
    working = str(task / "working.json")
    assert calls == [
        ["WORK=t128", "", f"run --kernel k_a__place_1 --record-greedy --strict --golden {working} --realization piece.1 --bench"],
        ["PLACE=fuse,WORK=w2x2,TILE=mma", "1", f"run --kernel k_b --record-greedy --strict --golden {working} --realization fused --bench"],
        ["WORK=t32", "", f"run --kernel k_a__place_2 --record-greedy --strict --golden {working} --realization piece.FAIL --bench"],
        ["", "0", f"run --record --golden {working} --realization seed --bench"],
        ["", "", f"run --record --golden {working} --realization timed --bench"],
    ]
    assert (task / "failed.txt").read_text().splitlines() == ["piece piece.FAIL"]
    assert json.loads((task / "still_missing.json").read_text()) == []
