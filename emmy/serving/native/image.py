"""What a recipe's native lane bakes into its serving image: the pack's preparation command and the image tag.

``docker/emmy-native/bake.sh`` evaluates this module's output as shell assignments; nothing here prepares, builds or
pushes anything."""

from __future__ import annotations

import shlex
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from emmy.publish import model_slug
from emmy.recipe import resolve_for_hardware


def native_lane(recipe_dir: str | Path, gpu: str):
    """The recipe's native lane on ``gpu``: the matrix entry for that card whose engine is the native server."""
    return resolve_for_hardware(str(recipe_dir), gpu, 1, engine="native")


def preparation(recipe, recipe_dir: str | Path, emmy: str = "emmy") -> tuple[dict[str, str], list[str]]:
    """The environment and ``emmy serve`` argv that prepare ``recipe``'s pack from its golden alone, under strict
    evidence; the caller appends ``--export DIR``."""
    llm = recipe.engine.llm
    argv = [emmy, "serve", recipe.model_name, "--runner", "generate", "--native"]
    if recipe.model.revision:
        argv += ["--revision", recipe.model.revision]
    if llm.context_length is not None:
        argv += ["--max-model-len", str(llm.context_length)]
    argv += ["--golden", str(Path(recipe_dir) / llm.native.golden), "--strict-evidence", *shlex.split(llm.extra_args)]
    return dict(llm.extra_env), argv


def image_tag(recipe) -> str:
    """The tag the bake builds: the canonical ``cloudriftai/emmy-native-<slug>:<emmy version>-<source sha>``."""
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    return f"cloudriftai/emmy-native-{model_slug(recipe.model_name)}:{version('emmy-ml')}-{sha}"


def main(argv: list[str]) -> int:
    recipe_dir, gpu, emmy = argv
    recipe = native_lane(recipe_dir, gpu)
    env, command = preparation(recipe, recipe_dir, emmy)
    print(f"IMAGE={shlex.quote(image_tag(recipe))}")
    print(f"MODEL={shlex.quote(recipe.model_name)}")
    print(f"VERSION={shlex.quote(version('emmy-ml'))}")
    print(f"PREPARE=(env {shlex.join(f'{key}={value}' for key, value in env.items())} {shlex.join(command)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
