"""run_deploy's image pull: its budget is sized for a host that reaches the registry only through a
slow forward proxy, not for a direct Docker Hub connection."""

from emmy.deploy import Service, orchestrate
from emmy.recipe.types import Recipe


def _recipe() -> Recipe:
    return Recipe.from_dict(
        {
            "model": {"huggingface": "test-org/test-model"},
            "engine": {"llm": {"tensor_parallel_size": 1, "context_length": 8192, "vllm": {"image": "vllm/vllm-openai:v0.17.0"}}},
            "deploy": {"gpu": "NVIDIA GeForce RTX 5090", "gpu_count": 1},
        }
    )


async def test_image_pull_gets_the_two_hour_budget():
    timeouts: dict[str, int] = {}

    async def run_cmd(command, stream=True, timeout=600, log_output=False):
        timeouts[command] = timeout
        if command.startswith("docker compose config --images"):
            return 0, "vllm/vllm-openai:v0.17.0", ""
        if "/v1/" in command:
            return 0, '{"id": "chatcmpl-1"}', ""
        return 0, "", ""

    async def write_file(path, content):
        return None

    ok = await orchestrate.run_deploy(run_cmd, write_file, [Service(_recipe())], "/hf", "", "host", check_smoke_output=False)

    assert ok
    # Two vLLM images are 20-30 GB; through FCBK's Squid (~10 MB/s) the old 1800 s cut the pull off.
    assert timeouts["docker compose pull --ignore-pull-failures"] == orchestrate.IMAGE_PULL_TIMEOUT == 7200
