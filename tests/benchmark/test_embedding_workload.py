"""Embedding-recipe bench command and smoke-response checks."""

import asyncio
import base64
import io
import json
import shlex
import wave

from emmy.benchmark.workload import build_bench_command
from emmy.deploy.orchestrate import (
    _audio_request,
    _check_audio_response,
    _check_chat_response,
    _check_completion_response,
    _check_embedding_response,
    _check_image_response,
    _image_request,
    _request,
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


def test_transcription_bench_command_sends_dataset_clips_to_the_transcription_endpoint():
    recipe = Recipe.from_dict(
        {
            "model": {"huggingface": "org/speech", "input_modalities": ["text", "audio"]},
            "engine": {"llm": {"vllm": {}}},
            "benchmark": {
                "max_concurrency": 4,
                "num_prompts": 64,
                "transcription_dataset": "openslr/librispeech_asr",
                "transcription_subset": "clean",
                "transcription_split": "test",
                "temperature": 0.0,
                "ignore_eos": True,
            },
        }
    )
    cmd = build_bench_command(recipe)
    for arg in (
        "--backend openai-audio",
        "--endpoint /v1/audio/transcriptions",
        "--dataset-name hf",
        "--dataset-path openslr/librispeech_asr",
        "--hf-subset clean",
        "--hf-split test",
        "--temperature 0.0",
    ):
        assert arg in cmd
    assert "--random-" not in cmd and "--ignore-eos" not in cmd


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


def test_audio_request_inlines_a_16khz_wav():
    _, body = _audio_request(_recipe("generate"))
    clip = body["messages"][0]["content"][0]["input_audio"]
    with wave.open(io.BytesIO(base64.b64decode(clip["data"]))) as wav:
        assert (clip["format"], wav.getframerate(), wav.getnchannels(), wav.getnframes()) == ("wav", 16000, 1, 16000)


def test_check_audio_response():
    assert _check_audio_response('{"choices": [{"message": {"content": "A steady beep."}}]}')[0] == "pass"
    assert _check_audio_response('{"choices": [{"message": {}}]}')[0] == "retry"


def test_smoke_test_sends_media_only_for_recipes_that_declare_it():
    """An image or audio recipe gets one more probe per declared input; a text recipe and benchmark readiness do not."""
    commands = []

    def kind(cmd):
        return "image" if "image_url" in cmd else "audio" if "input_audio" in cmd else "text"

    async def run_cmd(cmd, **_):
        commands.append(cmd)
        answer = {"image": "Red", "audio": "A tone", "text": "4"}[kind(cmd)]
        return 0, json.dumps({"choices": [{"message": {"content": answer}}]}), ""

    def probes(modalities, check_smoke_output):
        commands.clear()
        recipe = Recipe.from_dict({"model": {"huggingface": "org/mm", "input_modalities": modalities}, "engine": {"llm": {"vllm": {}}}})
        assert asyncio.run(_smoke_test(run_cmd, Service(recipe), "svc", check_smoke_output))
        return [kind(cmd) for cmd in commands]

    assert probes(["text"], True) == ["text"]
    assert probes(["text", "image"], True) == ["text", "image"]
    assert probes(["text", "audio"], True) == ["text", "audio"]
    assert probes(["text", "image", "audio"], True) == ["text", "image", "audio"]
    assert probes(["text", "image", "audio"], False) == ["text"]


def test_smoke_request_body_is_one_shell_word():
    """The curl body reaches the shell as a single quoted word, whatever the recipe values contain."""
    commands = []

    async def run_cmd(cmd, **_):
        commands.append(cmd)
        return 0, json.dumps({"choices": [{"message": {"content": "4"}}]}), ""

    recipe = Recipe.from_dict({"model": {"huggingface": "org/it's a `name`"}, "engine": {"llm": {"vllm": {}}}})
    assert asyncio.run(_smoke_test(run_cmd, Service(recipe), "svc", True))
    words = shlex.split(commands[0])
    assert json.loads(words[words.index("-d") + 1]) == _request(recipe)[1]


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
