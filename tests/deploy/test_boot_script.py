"""The first-boot script a self-serving VM runs: files, unit, token-free download, health wait."""

import os
import stat
import subprocess
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import yaml

from emmy.deploy import replica_services
from emmy.deploy.boot_script import (
    CRASH_RESTARTS,
    HF_MODEL_API,
    SERVE_DIR,
    STATUS_FILE,
    UNIT_NAME,
    gated_models,
    render_boot_script,
    serving_port,
)
from emmy.recipe import resolve_for_hardware

H200 = "NVIDIA H200 141GB"
AMD = "AMD Instinct MI350X"
KEY = "test-key-abcdefghijkl"
MARKER = "serve.sh <<'EMMY_BOOT_EOF'\n"


def _services(tmp_path, gpu=H200, gpu_count=1, tensor_parallel=1, engine=None, deploy=None):
    directory = tmp_path / "recipe"
    directory.mkdir()
    recipe = {
        "model": {"huggingface": "org/model"},
        "engine": {"llm": {"tensor_parallel_size": tensor_parallel, **(engine or {"vllm": {"image": "example/vllm:1"}})}},
        "matrices": [{"deploy.gpu": gpu, "deploy.gpu_count": gpu_count}],
    }
    if deploy:
        recipe["deploy"] = deploy
    (directory / "recipe.yaml").write_text(yaml.safe_dump(recipe))
    return replica_services(resolve_for_hardware(str(directory), gpu, gpu_count))


def _serve(script):
    start = script.index(MARKER) + len(MARKER)
    return script[start : script.index("\nEMMY_BOOT_EOF", start)]


def _unit(script):
    marker = f"{UNIT_NAME} <<'EMMY_BOOT_EOF'\n"
    start = script.index(marker) + len(marker)
    return script[start : script.index("\nEMMY_BOOT_EOF", start)].splitlines()


def _stub(directory, name, body):
    path = directory / name
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)


def _run_serve(tmp_path, serve, *, installed, restarts=0, healthy=True):
    """Run ``serve`` against stub docker / curl / sleep with SERVE_DIR moved under ``tmp_path``.

    Returns the final status line and every docker command line, in order. An AMD recipe keeps the run off the
    NVIDIA setup, which would probe the host's /proc.
    """
    serve_dir, stubs = tmp_path / "serve", tmp_path / "bin"
    serve_dir.mkdir()
    stubs.mkdir()
    if installed:
        (serve_dir / "installed").touch()
    log = tmp_path / "docker.log"
    _stub(stubs, "docker", f'echo "$*" >> {log}\n[ "$1" = inspect ] && echo {restarts}\nexit 0')
    _stub(stubs, "curl", f"exit {0 if healthy else 22}")
    _stub(stubs, "sleep", "exit 0")
    script = tmp_path / "serve.sh"
    script.write_text(serve.replace(SERVE_DIR, str(serve_dir)))
    subprocess.run(["bash", str(script)], env={**os.environ, "PATH": f"{stubs}:/usr/bin:/bin"}, capture_output=True, timeout=60)
    calls = log.read_text().splitlines() if log.exists() else []
    return (serve_dir / "status").read_text().strip(), calls


def test_script_serves_behind_the_api_key_without_a_token(tmp_path):
    services, load_balancer = _services(tmp_path)

    script = render_boot_script(services, load_balancer, KEY)

    assert f"VLLM_API_KEY={KEY}" in script
    assert "HUGGING_FACE_HUB_TOKEN= " in script and "HUGGING_FACE_HUB_TOKEN=\n" in script
    assert "hf download org/model" in script
    assert f"/etc/systemd/system/{UNIT_NAME}" in script
    assert f"systemctl start --no-block {UNIT_NAME}" in script
    assert f"echo pending > {STATUS_FILE}" in script
    assert "nvidia-ctk" in script
    assert serving_port(load_balancer, services) == 8000


def test_runs_in_a_subshell_that_plain_sh_accepts(tmp_path):
    """CloudRift's own commands and its VM-ready webhook follow in the same /bin/sh script."""
    services, load_balancer = _services(tmp_path)

    lines = render_boot_script(services, load_balancer, KEY).splitlines()

    assert lines[0] == "(" and lines[1] == "set -e" and lines[-1] == ")"
    assert subprocess.run(["sh", "-n"], input="\n".join(lines), text=True, capture_output=True).returncode == 0


def test_image_with_its_own_model_cache_fails_before_any_download(tmp_path):
    services, load_balancer = _services(tmp_path)

    serve = _serve(render_boot_script(services, load_balancer, KEY))

    refusal = serve.index("grep -qx HF_HUB_OFFLINE=1 && fail 'image ships its own model cache; not supported yet'")
    assert serve.index("docker compose pull") < refusal < serve.index("hf download")


def test_autoheal_starts_only_once_the_engine_is_healthy(tmp_path):
    services, load_balancer = _services(tmp_path, gpu_count=2, tensor_parallel=1)

    serve = _serve(render_boot_script(services, load_balancer, KEY))

    assert "docker compose up -d vllm_0 vllm_1 nginx ||" in serve
    assert serve.index("curl -sf -m 5 http://localhost:8080/health") < serve.index("docker compose up -d autoheal")
    assert serve.index("docker compose up -d autoheal") < serve.index("status ready")


def test_replicas_are_served_through_the_load_balancer(tmp_path):
    services, load_balancer = _services(tmp_path, gpu_count=2, tensor_parallel=1)

    script = render_boot_script(services, load_balancer, KEY)

    assert load_balancer and len(services) == 2
    assert "/opt/emmy-model/nginx.conf" in script
    assert serving_port(load_balancer, services) == 8080


def test_nvswitch_hosts_get_fabric_manager(tmp_path):
    services, load_balancer = _services(tmp_path)

    serve = _serve(render_boot_script(services, load_balancer, KEY))

    assert "/proc/driver/nvidia-nvswitch/devices/" in serve
    assert "nvidia-fabricmanager=$ver" in serve


def test_amd_hosts_skip_the_nvidia_setup(tmp_path):
    services, load_balancer = _services(tmp_path, gpu="AMD Instinct MI350X")

    script = render_boot_script(services, load_balancer, KEY)

    assert "nvidia-ctk" not in script
    assert "nvidia-fabricmanager" not in script


def test_unit_starts_after_docker(tmp_path):
    services, load_balancer = _services(tmp_path)

    unit = _unit(render_boot_script(services, load_balancer, KEY))

    assert "Wants=network-online.target docker.service" in unit
    assert "After=network-online.target docker.service" in unit


def test_image_pull_is_quiet(tmp_path):
    services, load_balancer = _services(tmp_path)

    assert "docker compose pull --quiet ||" in _serve(render_boot_script(services, load_balancer, KEY))


def test_single_service_is_served_through_the_keyed_proxy(tmp_path):
    services, load_balancer = _services(tmp_path)

    script = render_boot_script(services, load_balancer, KEY)

    assert not load_balancer
    assert f'"Bearer {KEY}" 1;' in script
    assert "docker compose up -d vllm_0 nginx ||" in _serve(script)
    assert "curl -sf -m 5 http://localhost:8000/health" in _serve(script)


def test_files_holding_the_key_are_readable_by_root_only(tmp_path):
    """The compose file and nginx.conf hold the key; the unit and serve.sh stay world-readable."""
    services, load_balancer = _services(tmp_path)
    script = render_boot_script(services, load_balancer, KEY)
    root = tmp_path / "vm"
    for old, new in [(SERVE_DIR, root / "serve"), ("/etc/systemd/system", root / "unit"), ("/hf_models", root / "models")]:
        script = script.replace(old, str(new))
    (root / "unit").mkdir(parents=True)
    stubs = tmp_path / "bin"
    stubs.mkdir()
    _stub(stubs, "systemctl", "exit 0")

    run = subprocess.run(["sh", "-c", f"umask 022\n{script}"], env={**os.environ, "PATH": f"{stubs}:/usr/bin:/bin"}, capture_output=True)

    assert run.returncode == 0, run.stderr

    def mode(path):
        return stat.S_IMODE(path.stat().st_mode)

    assert mode(root / "serve" / "docker-compose.yaml") == mode(root / "serve" / "nginx.conf") == 0o600
    assert KEY in (root / "serve" / "nginx.conf").read_text()
    assert mode(root / "serve" / "serve.sh") == 0o755
    assert mode(root / "unit" / UNIT_NAME) == 0o644
    assert mode(root / "serve" / "status") == 0o644


# ── serve.sh, run against stub docker ───────────────────────────────


def test_first_boot_installs_downloads_and_marks_the_install(tmp_path):
    services, load_balancer = _services(tmp_path, gpu=AMD)

    status, calls = _run_serve(tmp_path, _serve(render_boot_script(services, load_balancer, KEY)), installed=False)

    assert status == "ready"
    assert (tmp_path / "serve" / "installed").exists()
    assert calls.index("compose pull --quiet") < calls.index("compose up -d vllm_0 nginx") < calls.index("compose up -d autoheal")
    assert any(call.startswith("run --rm") and "hf download org/model" in call for call in calls)


def test_reboot_waits_for_the_cached_deployment_without_the_network(tmp_path):
    """Once installed, a reboot only waits: no pull and no download, so a VM without network still comes back."""
    services, load_balancer = _services(tmp_path, gpu=AMD)

    status, calls = _run_serve(tmp_path, _serve(render_boot_script(services, load_balancer, KEY)), installed=True)

    assert status == "ready"
    assert not any(call.startswith(("compose pull", "run ", "image ")) for call in calls)
    assert calls[0] == "compose up -d vllm_0 nginx" and calls[-1] == "compose up -d autoheal"


def test_a_crash_looping_engine_fails_instead_of_waiting_out_the_timeout(tmp_path):
    """`restart: unless-stopped` keeps a crashing engine `restarting`, never `exited`; its restart count tells."""
    services, load_balancer = _services(tmp_path, gpu=AMD, gpu_count=2, tensor_parallel=1)
    serve = _serve(render_boot_script(services, load_balancer, KEY))

    status, calls = _run_serve(tmp_path, serve, installed=True, restarts=CRASH_RESTARTS, healthy=False)

    assert status == "failed: engine keeps crashing"
    assert "inspect --format {{.RestartCount}} vllm_0 vllm_1" in calls
    assert "compose up -d autoheal" not in calls


def test_the_exited_check_ignores_autoheal_stopped_by_a_reboot(tmp_path):
    services, load_balancer = _services(tmp_path)

    serve = _serve(render_boot_script(services, load_balancer, KEY))

    assert 'docker compose ps -q --status exited --status dead vllm_0)" ] && fail "engine exited"' in serve


def test_serve_step_is_valid_bash(tmp_path):
    services, load_balancer = _services(tmp_path)
    serve = _serve(render_boot_script(services, load_balancer, KEY))

    assert subprocess.run(["bash", "-n"], input=serve, text=True, capture_output=True).returncode == 0


# ── refusals ────────────────────────────────────────────────────────


@pytest.mark.parametrize("key", ["", "short", "has space in it 123", "quote'injection-123456"])
def test_refuses_an_unsafe_api_key(tmp_path, key):
    services, load_balancer = _services(tmp_path)

    with pytest.raises(ValueError, match="API key"):
        render_boot_script(services, load_balancer, key)


def test_refuses_an_engine_that_does_not_read_the_api_key(tmp_path):
    """SGLang ignores VLLM_API_KEY, so its endpoint would be open."""
    services, load_balancer = _services(tmp_path, engine={"sglang": {"image": "example/sglang:1"}})

    with pytest.raises(ValueError, match="sglang does not read"):
        render_boot_script(services, load_balancer, KEY)


@pytest.mark.parametrize("pin", [{"driver_version": "580.95.05"}, {"cuda_version": "13.0"}])
def test_refuses_a_pinned_driver_or_cuda_version(tmp_path, pin):
    services, load_balancer = _services(tmp_path, deploy=pin)

    with pytest.raises(ValueError, match="pins an NVIDIA driver or CUDA version"):
        render_boot_script(services, load_balancer, KEY)


# ── gated_models ────────────────────────────────────────────────────


def _hub_client(gated_by_model):
    """An httpx.AsyncClient double answering the model API with each model's ``gated`` field; other models 404."""

    def respond(url):
        request = httpx.Request("GET", url)
        name = url.removeprefix(HF_MODEL_API.format(""))
        if name not in gated_by_model:
            return httpx.Response(404, request=request)
        return httpx.Response(200, json={"gated": gated_by_model[name]}, request=request)

    client = AsyncMock()
    client.get.side_effect = respond
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


async def test_gated_models_names_the_models_behind_an_access_agreement():
    client = _hub_client({"org/open": False, "org/manual": "manual", "org/auto": "auto"})

    with patch("emmy.deploy.boot_script.httpx.AsyncClient", return_value=client):
        gated = await gated_models(["org/open", "org/manual", "org/auto"])

    assert gated == ["org/manual", "org/auto"]


async def test_gated_models_raises_when_the_model_api_does_not_know_the_model():
    with patch("emmy.deploy.boot_script.httpx.AsyncClient", return_value=_hub_client({})):
        with pytest.raises(httpx.HTTPStatusError):
            await gated_models(["org/missing"])
