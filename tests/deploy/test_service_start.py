"""run_deploy's service start: a detached start, short /health polls, and a fast failure when a service exits."""

import pytest

from emmy.deploy import orchestrate
from emmy.recipe.types import Recipe


def _recipe() -> Recipe:
    return Recipe.from_dict(
        {
            "model": {"huggingface": "test-org/test-model"},
            "engine": {"llm": {"context_length": 8192, "vllm": {"image": "vllm/vllm-openai:v0.17.0"}}},
            "deploy": {"gpu": "NVIDIA GeForce RTX 5090", "gpu_count": 1},
        }
    )


def _host(health: list[bool], exited: str = ""):
    """A fake SSH runner: every step succeeds except /health, which answers from ``health`` in order."""
    calls: list[str] = []

    async def run_cmd(command, stream=True, timeout=600, log_output=False):
        calls.append(command)
        if command.startswith("curl -sf"):
            return (0 if health.pop(0) else 7), "", ""
        if command.startswith("docker compose ps"):
            return 0, exited, ""
        if command.startswith("docker compose config --images"):
            return 0, "vllm/vllm-openai:v0.17.0", ""
        if "/v1/chat/completions" in command:
            return 0, '{"id": "chatcmpl-1"}', ""
        return 0, "", ""

    async def write_file(path, content):
        return None

    return calls, run_cmd, write_file


@pytest.fixture(autouse=True)
def _no_poll_delay(monkeypatch):
    async def instant(_seconds):
        return None

    monkeypatch.setattr(orchestrate.asyncio, "sleep", instant)


async def test_a_dropped_probe_during_load_costs_one_probe_not_the_deploy():
    # A probe whose SSH call was reset reads exactly like a server that is not up yet.
    calls, run_cmd, write_file = _host(health=[False, False, True])

    ok = await orchestrate.run_deploy(run_cmd, write_file, _recipe(), "/hf", "", "host", check_smoke_output=False)

    assert ok
    assert "docker compose up -d" in calls
    assert not any("--wait" in command for command in calls)
    assert sum(command.startswith("curl -sf") for command in calls) == 3


async def test_a_service_that_exits_while_loading_fails_at_once():
    calls, run_cmd, write_file = _host(health=[False], exited="3f2a1b\n")

    ok = await orchestrate.run_deploy(run_cmd, write_file, _recipe(), "/hf", "", "host", check_smoke_output=False)

    assert ok is False
    assert sum(command.startswith("curl -sf") for command in calls) == 1
    assert "docker compose logs --tail=100" in calls
