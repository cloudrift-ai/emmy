"""A protected deployment: every model request must carry the serving API key, and no log shows the key."""

import argparse
import os

import pytest
import yaml

from emmy.deploy import Service, generate_compose, orchestrate
from emmy.deploy.serving_key import serving_api_key
from emmy.recipe import Recipe, load_recipe, validate_extra_args
from emmy.redact import redact_secrets

KEY = "sk-serving-key-0123456789"
NATIVE = {
    "model": {"huggingface": "org/m"},
    "engine": {"llm": {"context_length": 4096, "native": {"image": "cloudriftai/emmy-native-m:0.3.22-abcdef1", "golden": "g"}}},
}

# ── the engines ─────────────────────────────────────────────────────


def test_compose_without_a_key_requires_none(sample_config, sample_config_sglang):
    for config in (sample_config, sample_config_sglang):
        result = generate_compose([Service(Recipe.from_dict(config))], "/mnt/models", "test-token")
        assert "API_KEY" not in result
        assert "--api-key" not in result


def test_vllm_takes_the_key_from_its_environment(sample_config):
    """vLLM logs its command-line arguments at start-up, so the key never goes on the command."""
    result = generate_compose([Service(Recipe.from_dict(sample_config))], "/mnt/models", "test-token", api_key=KEY)

    service = yaml.safe_load(result)["services"]["vllm_0"]
    assert f"VLLM_API_KEY={KEY}" in service["environment"]
    assert KEY not in service["command"]


def test_sglang_takes_the_key_as_a_flag(sample_config_sglang):
    result = generate_compose([Service(Recipe.from_dict(sample_config_sglang))], "/mnt/models", "test-token", api_key=KEY)

    service = yaml.safe_load(result)["services"]["sglang_0"]
    assert f"--api-key {KEY}" in service["command"]
    assert not any(KEY in line for line in service["environment"])


def test_a_native_service_cannot_be_protected():
    with pytest.raises(ValueError, match="native engine has no API key option"):
        generate_compose([Service(Recipe.from_dict(NATIVE), None)], "/mnt/models", "tok", api_key=KEY)


def test_a_recipe_cannot_carry_a_key(tmp_path):
    """The key is a deploy input: neither engine's spelling may sit in a recipe."""
    for engine in ("vllm", "sglang"):
        with pytest.raises(ValueError, match="--api-key"):
            validate_extra_args("--api-key sk-123", engine=engine)
    config = {"model": {"huggingface": "org/m"}, "engine": {"llm": {"vllm": {"extra_env": {"VLLM_API_KEY": "sk-123"}}}}}
    (tmp_path / "recipe.yaml").write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="VLLM_API_KEY"):
        load_recipe(str(tmp_path))


# ── the smoke test ──────────────────────────────────────────────────


async def test_the_smoke_request_carries_the_key_and_the_health_poll_does_not(sample_config, monkeypatch):
    """Both engines leave /health open, so only the model request needs the key."""
    calls: list[str] = []

    async def run_cmd(command, stream=True, timeout=600, log_output=False):
        calls.append(command)
        if command.startswith("docker compose ps"):
            return 0, "running 0\n", ""
        if command.startswith("docker compose config --images"):
            return 0, "vllm/vllm-openai:v0.17.0", ""
        return 0, ('{"id": "chatcmpl-1"}' if "/v1/" in command else ""), ""

    async def write_file(path, content):
        return None

    service = Service(Recipe.from_dict(sample_config))
    assert await orchestrate.run_deploy(run_cmd, write_file, [service], "/hf", "", "host", check_smoke_output=False, api_key=KEY)

    assert [command for command in calls if command.startswith("curl -sf")] == ["curl -sf http://localhost:8000/health"]
    (smoke,) = [command for command in calls if "/v1/" in command]
    assert smoke.startswith(
        "curl --fail-with-body -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json'"
        f" -H 'Authorization: Bearer {KEY}' -d "
    )


# ── the key stays out of logs ───────────────────────────────────────


def test_a_key_is_redacted_once_parsed():
    key = "sk-another-serving-key-42"
    assert serving_api_key(key) == key
    assert redact_secrets(f"curl -H 'Authorization: Bearer {key}'") == "curl -H 'Authorization: Bearer ***'"


@pytest.mark.parametrize("bad", ["short", "has space in the key 123", "quote'in-the-key-0123456789"])
def test_a_short_or_unsafe_key_is_refused_without_repeating_it(bad):
    with pytest.raises(argparse.ArgumentTypeError) as error:
        serving_api_key(bad)
    assert bad not in str(error.value)


def _dry_run_ssh(run_cli, recipes_dir, *extra):
    return run_cli(
        "deploy",
        "ssh",
        "--recipe",
        os.path.join(recipes_dir, "Qwen3-Embedding-8B"),
        "--ssh",
        "user@1.2.3.4",
        "--gpu",
        "NVIDIA GeForce RTX 5090",
        "--gpu-count",
        "1",
        "--dry-run",
        *extra,
    )


def test_a_dry_run_never_prints_the_key(run_cli, recipes_dir, monkeypatch):
    monkeypatch.delenv("EMMY_SERVING_API_KEY", raising=False)
    rc, stdout, stderr = _dry_run_ssh(run_cli, recipes_dir, "--serving-api-key", KEY)

    assert rc == 0, stderr
    assert KEY not in stdout + stderr
    assert '-H "Authorization: Bearer $EMMY_SERVING_API_KEY"' in stdout


def test_the_environment_variable_is_the_fallback(run_cli, recipes_dir, monkeypatch):
    monkeypatch.setenv("EMMY_SERVING_API_KEY", KEY)
    rc, stdout, stderr = _dry_run_ssh(run_cli, recipes_dir)

    assert rc == 0, stderr
    assert KEY not in stdout + stderr
    assert "Authorization: Bearer $EMMY_SERVING_API_KEY" in stdout


def test_an_open_deployment_prints_no_authorization_header(run_cli, recipes_dir, monkeypatch):
    monkeypatch.delenv("EMMY_SERVING_API_KEY", raising=False)
    rc, stdout, stderr = _dry_run_ssh(run_cli, recipes_dir)

    assert rc == 0, stderr
    assert "Authorization" not in stdout


def test_the_cli_refuses_an_unsafe_key_without_repeating_it(run_cli, recipes_dir):
    rc, stdout, stderr = _dry_run_ssh(run_cli, recipes_dir, "--serving-api-key", "tooshort")

    assert rc == 2
    assert "tooshort" not in stdout
    assert "at least 16 characters" in stderr


def test_the_cli_refuses_a_protected_native_deployment(run_cli, tmp_path, monkeypatch):
    monkeypatch.delenv("EMMY_SERVING_API_KEY", raising=False)
    (tmp_path / "recipe.yaml").write_text(yaml.safe_dump({**NATIVE, "deploy": {"gpu": "NVIDIA GeForce RTX 5090", "gpu_count": 1}}))
    rc, stdout, stderr = run_cli(
        "deploy", "local", "--recipe", str(tmp_path), "--gpu", "NVIDIA GeForce RTX 5090", "--gpu-count", "1", "--dry-run",
        "--serving-api-key", KEY,
    )  # fmt: skip

    assert rc == 1
    assert "native engine has no API key option" in stdout + stderr
