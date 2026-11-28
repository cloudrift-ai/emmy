"""deploy boot-script: the JSON it prints, and what it refuses before rendering."""

import json
from argparse import Namespace
from unittest.mock import AsyncMock

import httpx
import pytest
import yaml

from emmy.commands.deploy import boot_script

H200 = "NVIDIA H200 141GB"
KEY = "test-key-abcdefghijkl"


def _args(tmp_path, engine=None):
    directory = tmp_path / "recipe"
    directory.mkdir()
    recipe = {
        "model": {"huggingface": "org/model"},
        "engine": {"llm": {"tensor_parallel_size": 1, **(engine or {"vllm": {"image": "example/vllm:1"}})}},
        "matrices": [{"deploy.gpu": H200, "deploy.gpu_count": 1}],
    }
    (directory / "recipe.yaml").write_text(yaml.safe_dump(recipe))
    return Namespace(recipe=str(directory), gpu=H200, gpu_count=1, instance_type="h200-8-generic")


def _exit_code(args):
    with pytest.raises(SystemExit) as exit_info:
        boot_script.handle_boot_script(args)
    return exit_info.value.code


def test_prints_image_port_and_script(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(boot_script.API_KEY_ENV, KEY)
    monkeypatch.setattr(boot_script, "gated_models", AsyncMock(return_value=[]))

    boot_script.handle_boot_script(_args(tmp_path))

    out = json.loads(capsys.readouterr().out)
    assert out["schema_version"] == 1
    assert out["port"] == 8000
    assert out["image_url"].endswith(".img")
    assert f"VLLM_API_KEY={KEY}" in out["script"]


def test_refuses_a_gated_model(tmp_path, monkeypatch):
    monkeypatch.setenv(boot_script.API_KEY_ENV, KEY)
    monkeypatch.setattr(boot_script, "gated_models", AsyncMock(side_effect=lambda names: list(names)))

    assert _exit_code(_args(tmp_path)) == 2


def test_an_unreachable_hugging_face_api_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv(boot_script.API_KEY_ENV, KEY)
    monkeypatch.setattr(boot_script, "gated_models", AsyncMock(side_effect=httpx.ConnectError("unreachable")))

    assert _exit_code(_args(tmp_path)) == 1


def test_a_recipe_the_library_refuses_is_refused_before_the_gate_check(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv(boot_script.API_KEY_ENV, KEY)
    gated_models = AsyncMock(return_value=[])
    monkeypatch.setattr(boot_script, "gated_models", gated_models)

    assert _exit_code(_args(tmp_path, engine={"sglang": {"image": "example/sglang:1"}})) == 2
    assert "VLLM_API_KEY" in caplog.text
    gated_models.assert_not_awaited()


def test_refuses_an_unsafe_api_key_naming_its_variable(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv(boot_script.API_KEY_ENV, "short")

    assert _exit_code(_args(tmp_path)) == 2
    assert boot_script.API_KEY_ENV in caplog.text
