"""The nightly golden fill recipe lists every hardware golden once, on the card its file names, so the job can fill
any file it picks with ``--filter golden=PATH``."""

from pathlib import Path

import yaml

from emmy.compiler.pipeline.search.golden import GoldenFile
from emmy.compiler.pipeline.search.golden.repository import _RECORDS_DIR

ROOT = Path(__file__).parents[2]
RECIPE = ROOT / ".github" / "golden-fill" / "recipe.yaml"


def test_fill_recipe_matrix_is_every_hardware_golden_on_its_card():
    expected = {(str(path.relative_to(ROOT)), GoldenFile.load(path).gpu_name) for path in _RECORDS_DIR.glob("*.json")}
    rows = yaml.safe_load(RECIPE.read_text())["matrices"]
    assert sorted((row["golden"], row["deploy.gpu"]) for row in rows) == sorted(expected)
