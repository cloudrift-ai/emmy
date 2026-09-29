"""run_deploy's service start: detached starts in the order the GPUs allow, short /health polls, and a fast
failure when a service exits."""

import pytest

from emmy.deploy import Service, orchestrate
from emmy.recipe.types import Recipe


def _recipe(gpu_count: int = 1, tensor_parallel_size: int = 1, model: str = "test-org/test-model") -> Recipe:
    return Recipe.from_dict(
        {
            "model": {"huggingface": model},
            "engine": {
                "llm": {
                    "tensor_parallel_size": tensor_parallel_size,
                    "context_length": 8192,
                    "vllm": {"image": "vllm/vllm-openai:v0.17.0"},
                }
            },
            "deploy": {"gpu": "NVIDIA GeForce RTX 5090", "gpu_count": gpu_count},
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
        if "/v1/" in command:
            return 0, '{"id": "chatcmpl-1"}', ""
        return 0, "", ""

    async def write_file(path, content):
        return None

    return calls, run_cmd, write_file


def _starts_and_probes(calls: list[str]) -> list[str]:
    return [command for command in calls if command.startswith(("docker compose up -d", "curl -sf"))]


@pytest.fixture(autouse=True)
def _no_poll_delay(monkeypatch):
    async def instant(_seconds):
        return None

    monkeypatch.setattr(orchestrate.asyncio, "sleep", instant)


async def test_a_dropped_probe_during_load_costs_one_probe_not_the_deploy():
    # A probe whose SSH call was reset reads exactly like a server that is not up yet.
    calls, run_cmd, write_file = _host(health=[False, False, True])

    ok = await orchestrate.run_deploy(run_cmd, write_file, [Service(_recipe())], "/hf", "", "host", check_smoke_output=False)

    assert ok
    assert "docker compose up -d vllm_0" in calls
    assert not any("--wait" in command for command in calls)
    assert sum(command.startswith("curl -sf") for command in calls) == 3


async def test_a_service_that_exits_while_loading_fails_at_once():
    calls, run_cmd, write_file = _host(health=[False], exited="3f2a1b\n")

    ok = await orchestrate.run_deploy(run_cmd, write_file, [Service(_recipe())], "/hf", "", "host", check_smoke_output=False)

    assert ok is False
    assert sum(command.startswith("curl -sf") for command in calls) == 1
    assert "docker compose logs --tail=100" in calls


async def test_a_model_sharing_a_gpu_starts_only_after_the_earlier_one_is_healthy():
    """vLLM checks free memory against its whole fraction at start-up, so the second model on a
    device waits for the first; a model on another device starts at once."""
    services = [Service(_recipe(), [0], 8000), Service(_recipe(model="org/other"), [0], 8001), Service(_recipe(), [1], 8002)]
    calls, run_cmd, write_file = _host(health=[False, True, True, True])

    ok = await orchestrate.run_deploy(run_cmd, write_file, services, "/hf", "", "host", check_smoke_output=False)

    assert ok
    assert _starts_and_probes(calls) == [
        "docker compose up -d vllm_0",
        "curl -sf http://localhost:8000/health",
        "curl -sf http://localhost:8000/health",
        "docker compose up -d vllm_1",
        "docker compose up -d vllm_2",
        "curl -sf http://localhost:8001/health",
        "curl -sf http://localhost:8002/health",
    ]
    assert [command for command in calls if "/v1/" in command] == [
        f"curl --fail-with-body -s http://localhost:{port}/v1/chat/completions -H 'Content-Type: application/json'"
        f' -d \'{{"model": "{model}", "messages": [{{"role": "user", "content": "What is 2+2? Answer with just the number."}}], "max_tokens": 128}}\''
        for port, model in [(8000, "test-org/test-model"), (8001, "org/other"), (8002, "test-org/test-model")]
    ]


async def test_replicas_on_their_own_gpus_start_together_and_nginx_last():
    services = [Service(_recipe(4, 2), [0, 1], 8000), Service(_recipe(4, 2), [2, 3], 8001)]
    calls, run_cmd, write_file = _host(health=[True, True, True])

    ok = await orchestrate.run_deploy(run_cmd, write_file, services, "/hf", "", "host", load_balancer=True, check_smoke_output=False)

    assert ok
    assert _starts_and_probes(calls) == [
        "docker compose up -d vllm_0",
        "docker compose up -d vllm_1",
        "curl -sf http://localhost:8000/health",
        "curl -sf http://localhost:8001/health",
        "docker compose up -d nginx",
        "curl -sf http://localhost:8080/health",
    ]


async def test_a_failing_shared_gpu_service_is_named():
    services = [Service(_recipe(), [0], 8000), Service(_recipe(model="org/other"), [0], 8001)]
    calls, run_cmd, write_file = _host(health=[True, False], exited="3f2a1b\n")

    ok = await orchestrate.run_deploy(run_cmd, write_file, services, "/hf", "", "host", check_smoke_output=False)

    assert ok is False
    assert _starts_and_probes(calls)[-1] == "curl -sf http://localhost:8001/health"
