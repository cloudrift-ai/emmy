"""Embedding-recipe bench command and smoke-response checks."""

import asyncio
import json

from emmy.benchmark.workload import build_bench_command
from emmy.deploy.orchestrate import (
    _check_chat_response,
    _check_completion_response,
    _check_embedding_response,
    _check_image_response,
    _image_request,
    _smoke_response_check,
    _smoke_test,
)
from emmy.deploy.params import Service
from emmy.recipe.types import Recipe


def _recipe(task: str) -> Recipe:
    return Recipe.from_dict(
        {
            "model": {"huggingface": "Qwen/Qwen3-Embedding-0.6B", "task": task},
            "engine": {"llm": {"context_length": 4096, "vllm": {}}},
            "benchmark": {"max_concurrency": 8, "num_prompts": 32, "random_input_len": 128, "random_output_len": 1},
        }
    )


def test_embed_bench_command_targets_embeddings_endpoint():
    cmd = build_bench_command(_recipe("embed"))
    assert "--backend openai-embeddings" in cmd
    assert "--endpoint /v1/embeddings" in cmd
    assert "--random-output-len" not in cmd
    assert "--random-input-len 128" in cmd


def test_generate_bench_command_unchanged():
    cmd = build_bench_command(_recipe("generate"))
    assert "--backend" not in cmd
    assert "--random-output-len 1" in cmd


def test_check_embedding_response():
    good = '{"data": [{"embedding": [0.6, 0.8], "index": 0}]}'
    assert _check_embedding_response(good)[0] == "pass"
    nan = '{"data": [{"embedding": [1.0, null], "index": 0}]}'
    assert _check_embedding_response(nan)[0] in ("fail", "retry")
    unnormalized = '{"data": [{"embedding": [3.0, 4.0], "index": 0}]}'
    verdict, detail = _check_embedding_response(unnormalized)
    assert verdict == "fail" and "norm" in detail
    not_ready = '{"error": "loading"}'
    assert _check_embedding_response(not_ready)[0] == "retry"
    assert _check_embedding_response("not json")[0] == "retry"


def test_check_chat_response():
    assert _check_chat_response('{"choices": [{"message": {"content": "The answer is 4."}}]}')[0] == "pass"
    assert _check_chat_response('{"choices": [{"message": {"content": "five"}}]}')[0] == "fail"
    assert _check_chat_response("oops")[0] == "retry"
    assert _check_chat_response('{"choices": [{"message": {}}]}')[0] == "retry"


def test_check_image_response():
    assert _check_image_response('{"choices": [{"message": {"content": "Red."}}]}')[0] == "pass"
    assert _check_image_response('{"choices": [{"message": {"content": "It is blue."}}]}')[0] == "fail"
    assert _check_image_response("oops")[0] == "retry"


def test_image_request_inlines_a_png_data_url():
    path, body = _image_request(_recipe("generate"))
    parts = body["messages"][0]["content"]
    assert path == "/v1/chat/completions"
    assert parts[0]["type"] == "text"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,iVBOR")


def test_smoke_test_sends_an_image_only_for_image_recipes():
    """An image recipe gets a second probe with the inline image; a text recipe and benchmark readiness do not."""
    commands = []

    async def run_cmd(cmd, **_):
        commands.append(cmd)
        answer = "Red" if "image_url" in cmd else "4"
        return 0, json.dumps({"choices": [{"message": {"content": answer}}]}), ""

    def probes(recipe, check_smoke_output):
        commands.clear()
        assert asyncio.run(_smoke_test(run_cmd, Service(recipe), "svc", check_smoke_output))
        return [("image" if "image_url" in cmd else "text") for cmd in commands]

    text = Recipe.from_dict({"model": {"huggingface": "org/chat"}, "engine": {"llm": {"vllm": {}}}})
    vision = Recipe.from_dict({"model": {"huggingface": "org/vl", "input_modalities": ["text", "image"]}, "engine": {"llm": {"vllm": {}}}})
    assert probes(text, True) == ["text"]
    assert probes(vision, True) == ["text", "image"]
    assert probes(vision, False) == ["text"]


def test_check_completion_response():
    assert _check_completion_response('{"choices": [{"text": " 4\\n"}]}')[0] == "pass"
    assert _check_completion_response('{"choices": [{"text": " five"}]}')[0] == "fail"
    assert _check_completion_response("oops")[0] == "retry"
    assert _check_completion_response('{"choices": [{}]}')[0] == "retry"


def test_bench_readiness_does_not_judge_model_output():
    cases = [
        (_recipe("generate"), '{"choices": [{"message": {"content": "five"}}]}'),
        (
            Recipe.from_dict(
                {
                    "model": {"huggingface": "org/base", "task": "generate", "smoke_test": "completion"},
                    "engine": {"llm": {"vllm": {}}},
                }
            ),
            '{"choices": [{"text": "five"}]}',
        ),
        (_recipe("embed"), '{"data": [{"embedding": [3.0, 4.0]}]}'),
    ]
    for recipe, response in cases:
        check = _smoke_response_check(recipe, check_smoke_output=False)
        assert check(response)[0] == "pass"
