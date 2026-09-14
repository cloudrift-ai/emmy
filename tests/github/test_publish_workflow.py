"""Safety contract for the lightweight PyPI publication gate."""

from pathlib import Path

import yaml

WORKFLOW = Path(__file__).parents[2] / ".github" / "workflows" / "publish.yml"


def test_publish_builds_and_smokes_merged_source_without_repeating_the_suite():
    document = yaml.safe_load(WORKFLOW.read_text())
    jobs = document["jobs"]

    assert set(jobs) == {"build", "publish", "github-release"}
    build_script = "\n".join(step.get("run", "") for step in jobs["build"]["steps"])
    assert 'git merge-base --is-ancestor "$GITHUB_SHA" origin/main' in build_script
    assert "commits/$GITHUB_SHA/pulls" in build_script
    assert "actions/workflows/tests.yml/runs?head_sha=$PR_HEAD" in build_script
    assert "make pypi-dist" in build_script
    assert '"$SMOKE_ENV/bin/pip" install' in build_script
    assert '"$SMOKE_ENV/bin/emmy" --version' in build_script
    assert '"$SMOKE_ENV/bin/emmy" recipe list --json' in build_script
    assert "make test" not in build_script
    assert "make setup-ci" not in build_script
