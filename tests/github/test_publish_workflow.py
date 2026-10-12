"""Safety contract for the lightweight PyPI publication gate."""

from pathlib import Path

import yaml

GITHUB = Path(__file__).parents[2] / ".github"
WORKFLOW = GITHUB / "workflows" / "publish.yml"
TESTS_WORKFLOW = GITHUB / "workflows" / "tests.yml"
SMOKE_ACTION = GITHUB / "actions" / "smoke-wheel" / "action.yml"


def test_publish_builds_and_smokes_main_source_without_requiring_a_pull_request():
    document = yaml.safe_load(WORKFLOW.read_text())
    jobs = document["jobs"]

    assert set(jobs) == {"build", "publish", "github-release"}
    steps = jobs["build"]["steps"]
    build_script = "\n".join(step.get("run", "") for step in steps)
    assert 'git merge-base --is-ancestor "$GITHUB_SHA" origin/main' in build_script
    assert "gh api" not in build_script
    assert "make pypi-dist" in build_script
    assert "./.github/actions/smoke-wheel" in [step.get("uses") for step in steps]
    assert '"$BIN/emmy" --version' in build_script
    assert "make test" not in build_script
    assert "make setup-ci" not in build_script


def test_the_wheel_smoke_test_is_one_action_both_gates_run():
    """The release gate and the pull-request dry run install the wheel into a clean environment and check the same
    things, so a wheel that cannot find its goldens fails the pull request, not the release (#1148)."""
    action = yaml.safe_load(SMOKE_ACTION.read_text())
    script = "\n".join(step.get("run", "") for step in action["runs"]["steps"])
    assert '"$SMOKE_ENV/bin/pip" install' in script
    assert "import emmy.emmy_runtime" in script
    assert '"$BIN/emmy" recipe list --json' in script
    assert "repository_golden_paths" in script
    package = yaml.safe_load(TESTS_WORKFLOW.read_text())["jobs"]["package"]
    assert "./.github/actions/smoke-wheel" in [step.get("uses") for step in package["steps"]]
