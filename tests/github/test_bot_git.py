import os
import subprocess
from pathlib import Path

import pytest

HELPERS = Path(__file__).parents[2] / ".github" / "workflows" / "scripts" / "bot_git.sh"
COMMIT_IDENTITY = ["-c", "user.name=setup", "-c", "user.email=setup@example.com"]


def _git(repo, *args, check=True):
    return subprocess.run(["git", *COMMIT_IDENTITY, *args], cwd=repo, capture_output=True, text=True, check=check)


def _commit(repo, path, text, message):
    (repo / path).parent.mkdir(parents=True, exist_ok=True)
    (repo / path).write_text(text)
    _git(repo, "add", "--", path)
    _git(repo, "commit", "-q", "-m", message)


@pytest.fixture
def remote(tmp_path):
    """A bare origin with one commit on main, a nightly checkout of it, and a second clone standing in for the other job."""
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", "--initial-branch=main", str(origin))
    seed = tmp_path / "seed"
    _git(tmp_path, "clone", "-q", str(origin), str(seed))
    _commit(seed, "tests/durations.json", "{}\n", "seed")
    _commit(seed, "weights/schedule.json", "{}\n", "seed weights")
    _git(seed, "push", "-q", "origin", "HEAD:main")
    nightly = tmp_path / "nightly"
    _git(tmp_path, "clone", "-q", str(origin), str(nightly))
    other = tmp_path / "other"
    _git(tmp_path, "clone", "-q", str(origin), str(other))
    return origin, nightly, other


def _push_to_main(repo, tmp_path, *, tolerated=""):
    # gh only has to accept `gh auth setup-git`; a stub on PATH stands in for it.
    stub = tmp_path / "bin"
    stub.mkdir(exist_ok=True)
    (stub / "gh").write_text("#!/bin/sh\nexit 0\n")
    (stub / "gh").chmod(0o755)
    environment = {**os.environ, "PATH": f"{stub}{os.pathsep}{os.environ['PATH']}", "TOLERATED_PATHS": tolerated}
    return subprocess.run(
        ["bash", "-c", f'source "{HELPERS}" && push_to_main "tests: refresh durations" tests/durations.json'],
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_push_to_main_rebases_over_the_other_jobs_file(remote, tmp_path):
    origin, nightly, other = remote
    _commit(other, "weights/schedule.json", '{"refit": true}\n', "compiler: refresh schedule prior")
    _commit(other, "recipes/new-model/recipe.yaml", "tags: [onboarding]\n", "recipes: refresh model lifecycle")
    _git(other, "push", "-q", "origin", "HEAD:main")
    (nightly / "tests/durations.json").write_text('{"measured": true}\n')

    # A tolerated path is a git pathspec, so the recipe glob reaches git unexpanded.
    result = _push_to_main(nightly, tmp_path, tolerated="weights/schedule.json weights/placement.json recipes/*/recipe.yaml")

    assert result.returncode == 0, result.stderr
    assert _git(nightly, "log", "--format=%s", "origin/main").stdout.splitlines() == [
        "tests: refresh durations",
        "recipes: refresh model lifecycle",
        "compiler: refresh schedule prior",
        "seed weights",
        "seed",
    ]
    assert _git(nightly, "show", "origin/main:tests/durations.json").stdout == '{"measured": true}\n'
    assert _git(nightly, "log", "-1", "--format=%an <%ae>", "origin/main").stdout.strip() == (
        "emmy-onboarding-bot <emmy-onboarding-bot[bot]@users.noreply.github.com>"
    )


def test_push_to_main_refuses_when_main_moved_elsewhere(remote, tmp_path):
    origin, nightly, other = remote
    _commit(other, "emmy/compiler.py", "pass\n", "Change the compiler")
    _git(other, "push", "-q", "origin", "HEAD:main")
    before = _git(other, "rev-parse", "origin/main").stdout.strip()
    (nightly / "tests/durations.json").write_text('{"measured": true}\n')

    result = _push_to_main(nightly, tmp_path, tolerated="weights/schedule.json")

    assert result.returncode == 1
    assert "main changed beyond weights/schedule.json" in result.stderr
    _git(other, "fetch", "-q", "origin")
    assert _git(other, "rev-parse", "origin/main").stdout.strip() == before
    # The measurement stays as a local commit, so a rerun of the step can decide again.
    assert _git(nightly, "log", "-1", "--format=%s").stdout.strip() == "tests: refresh durations"
