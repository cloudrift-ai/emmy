"""Session workloads preserve conversation prefixes and backend usage accounting."""

import asyncio
import copy
import json

import pytest

from emmy.benchmark.sessions import initial_length, measure


def test_session_length_is_centered():
    first = initial_length(32000, 12, 330, 512)
    lengths = [first + turn * 842 for turn in range(12)]
    assert sum(lengths) / len(lengths) == 32000
    with pytest.raises(ValueError):
        initial_length(10, 12, 330, 512)


async def test_sessions_append_generated_tokens_and_keep_worker_affinity(caplog):
    requests = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    class Client:
        async def post(self, url, json, headers=None):
            await asyncio.sleep(0)
            requests.append((copy.deepcopy(json), headers))
            return Response(
                {
                    "choices": [{"token_ids": [101, 102], "finish_reason": "length"}],
                    "usage": {"prompt_tokens": len(json["prompt"]), "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 7}},
                }
            )

    config = {
        "model": "test",
        "base_url": "http://test",
        "dp_size": 2,
        "benchmark": {
            "session_calls": 3,
            "num_prompts": 6,
            "random_input_len": 20,
            "random_output_len": 2,
            "session_new_tokens": 3,
            "max_concurrency": 2,
            "seed": 9,
            "session_affinity": True,
            "random_prefix_len": 2,
        },
    }
    with caplog.at_level("INFO"):
        await measure(config, Client(), list(range(1, 100)))
    assert len(requests) == 6
    for rank in ["0", "1"]:
        prompts = [body["prompt"] for body, headers in requests if headers["X-data-parallel-rank"] == rank]
        assert [len(p) for p in prompts] == [15, 20, 25]
        for previous, current in zip(prompts, prompts[1:]):
            assert current[: len(previous) + 2] == previous + [101, 102]
    assert requests[0][0]["prompt"] != requests[1][0]["prompt"]
    rows = [json.loads(record.message) for record in caplog.records]
    assert len([row for row in rows if "usage" in row]) == 6
    assert all(row["usage"]["prompt_tokens_details"]["cached_tokens"] == 7 for row in rows if "usage" in row)


async def test_recipe_session_options_reach_the_remote_client_with_unique_repeat_seeds():
    import shlex

    from emmy.benchmark.workload import run_benchmark_workload
    from emmy.recipe.types import Recipe

    recipe = Recipe.from_dict(
        {
            "model": {"huggingface": "test"},
            "benchmark": {
                "session_calls": 3,
                "session_new_tokens": 3,
                "session_concurrencies": [2, 4],
                "random_input_len": 20,
                "random_output_len": 2,
                "num_prompts": 12,
                "repeats": 2,
            },
        }
    )
    configs = []

    async def run(command, **kwargs):
        outer = shlex.split(command)
        inner = shlex.split(outer[-1])
        assert inner[:2] == ["python3", "-c"]
        compile(inner[2], "<remote-client>", "exec")
        configs.append(json.loads(inner[3]))
        return 0, "raw measurements", ""

    success, _, _, _ = await run_benchmark_workload(run, recipe)
    assert success
    assert [item["benchmark"]["seed"] for item in configs] == [0, 24]
    assert configs[0]["benchmark"]["session_concurrencies"] == [2, 4]
