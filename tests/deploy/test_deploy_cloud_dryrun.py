"""Dry-run end-to-end tests for the deploy cloud command."""

import json
import os

import pytest
import yaml

# ── deploy cloud dry-run ─────────────────────────────────────────


def test_deploy_cloud_dry_run(run_cli, tmp_path):
    """Cloud deploy resolves matrix entry from --gpu and --gpu-count."""
    recipe = {
        "model": {"huggingface": "test-org/test-model"},
        "engine": {
            "llm": {
                "tensor_parallel_size": 1,
                "vllm": {"image": "vllm/vllm-openai:v0.17.0"},
            }
        },
        "matrices": [
            {
                "deploy.gpu": "NVIDIA GeForce RTX 5090",
                "deploy.gpu_count": 1,
            },
        ],
    }
    with open(tmp_path / "recipe.yaml", "w") as f:
        yaml.dump(recipe, f)

    rc, stdout, stderr = run_cli(
        "deploy",
        "cloud",
        "--recipe",
        str(tmp_path),
        "--gpu",
        "NVIDIA GeForce RTX 5090",
        "--gpu-count",
        "1",
        "--dry-run",
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
    assert "[dry-run]" in stdout


def test_deploy_cloud_dry_run_deploy_steps(run_cli, tmp_path):
    recipe = {
        "model": {"huggingface": "test-org/test-model"},
        "engine": {
            "llm": {
                "tensor_parallel_size": 1,
                "vllm": {"image": "vllm/vllm-openai:v0.17.0"},
            }
        },
        "matrices": [
            {
                "deploy.gpu": "NVIDIA GeForce RTX 5090",
                "deploy.gpu_count": 1,
            },
        ],
    }
    with open(tmp_path / "recipe.yaml", "w") as f:
        yaml.dump(recipe, f)

    rc, stdout, stderr = run_cli(
        "deploy",
        "cloud",
        "--recipe",
        str(tmp_path),
        "--gpu",
        "NVIDIA GeForce RTX 5090",
        "--gpu-count",
        "1",
        "--dry-run",
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
    # VM provisioning step
    assert "Creating CloudRift instance" in stdout
    # Deploy steps
    assert "docker compose pull" in stdout
    assert "docker compose up" in stdout
    assert "dry-run (not deployed)" in stdout


def test_deploy_cloud_provider_flag_override_dry_run(run_cli, tmp_path):
    """--provider gcp forces GCP dispatch for an H200 (CloudRift is default)."""
    recipe = {
        "model": {"huggingface": "test-org/test-model"},
        "engine": {
            "llm": {
                "tensor_parallel_size": 1,
                "vllm": {"image": "vllm/vllm-openai:v0.17.0"},
            }
        },
        "matrices": [
            {
                "deploy.gpu": "NVIDIA H200 141GB",
                "deploy.gpu_count": 1,
            },
        ],
    }
    with open(tmp_path / "recipe.yaml", "w") as f:
        yaml.dump(recipe, f)

    rc, stdout, stderr = run_cli(
        "deploy",
        "cloud",
        "--recipe",
        str(tmp_path),
        "--gpu",
        "NVIDIA H200 141GB",
        "--gpu-count",
        "1",
        "--provider",
        "gcp",
        "--dry-run",
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
    # --provider gcp must restrict candidates to GCP only (no cloudrift entry).
    assert "Trying candidate: gcp a3-ultragpu-1g" in stdout
    assert "Trying candidate: cloudrift" not in stdout


def test_deploy_cloud_network_flag_dry_run(run_cli, tmp_path):
    """--network propagates into the CloudRift rent payload."""
    recipe = {
        "model": {"huggingface": "test-org/test-model"},
        "engine": {
            "llm": {
                "tensor_parallel_size": 1,
                "vllm": {"image": "vllm/vllm-openai:v0.17.0"},
            }
        },
        "matrices": [
            {
                "deploy.gpu": "NVIDIA GeForce RTX 5090",
                "deploy.gpu_count": 1,
            },
        ],
    }
    with open(tmp_path / "recipe.yaml", "w") as f:
        yaml.dump(recipe, f)

    rc, stdout, stderr = run_cli(
        "deploy",
        "cloud",
        "--recipe",
        str(tmp_path),
        "--gpu",
        "NVIDIA GeForce RTX 5090",
        "--gpu-count",
        "1",
        "--network",
        "public",
        "--dry-run",
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
    assert '"network": "public"' in stdout


def test_deploy_cloud_provider_default_is_cloudrift_for_h200(run_cli, tmp_path):
    """Without --provider, H200 defaults to CloudRift (first entry in hardware table)."""
    recipe = {
        "model": {"huggingface": "test-org/test-model"},
        "engine": {
            "llm": {
                "tensor_parallel_size": 1,
                "vllm": {"image": "vllm/vllm-openai:v0.17.0"},
            }
        },
        "matrices": [
            {
                "deploy.gpu": "NVIDIA H200 141GB",
                "deploy.gpu_count": 1,
            },
        ],
    }
    with open(tmp_path / "recipe.yaml", "w") as f:
        yaml.dump(recipe, f)

    rc, stdout, stderr = run_cli(
        "deploy",
        "cloud",
        "--recipe",
        str(tmp_path),
        "--gpu",
        "NVIDIA H200 141GB",
        "--gpu-count",
        "1",
        "--dry-run",
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
    # Without --provider, CloudRift is the first hardware-table entry for H200,
    # so the orchestrator must try it first.
    assert "Trying candidate: cloudrift h200-26-200-500-packed.1" in stdout


def test_deploy_cloud_missing_gpu_flag_fails(run_cli, recipes_dir):
    """Without --gpu and --gpu-count, cloud deploy fails (required args)."""
    rc, stdout, stderr = run_cli(
        "deploy",
        "cloud",
        "--recipe",
        os.path.join(recipes_dir, "Qwen3-Embedding-8B"),
        "--dry-run",
    )
    assert rc == 2
    assert "--gpu and --gpu-count are required" in stdout + stderr


# ── CLI help ─────────────────────────────────────────────────────


def test_deploy_cloud_help(run_cli):
    rc, stdout, _ = run_cli("deploy", "cloud", "--help")
    assert rc == 0
    assert "--recipe" in stdout
    assert "--ssh-key" in stdout
    assert "--dry-run" in stdout
    assert "--name" in stdout
    assert "--vm-active-timeout" in stdout
    assert "--vm-proxy" in stdout


def test_deploy_help_includes_cloud(run_cli):
    rc, stdout, _ = run_cli("deploy", "--help")
    assert rc == 0
    assert "cloud" in stdout


# ── plan mode ─────────────────────────────────────────────────────


def _plan_recipe(tmp_path, name, fractions):
    """A recipe directory with one H200 x1 matrix entry per fraction."""
    directory = tmp_path / name
    directory.mkdir()
    recipe = {
        "model": {"huggingface": f"org/{name}"},
        "engine": {"llm": {"tensor_parallel_size": 1, "vllm": {"image": "vllm/vllm-openai:v0.17.0"}}},
        "matrices": [
            {"deploy.gpu": "NVIDIA H200 141GB", "deploy.gpu_count": 1, "engine.llm.gpu_memory_utilization": fraction}
            for fraction in fractions
        ],
    }
    (directory / "recipe.yaml").write_text(yaml.safe_dump(recipe))
    return str(directory)


def _write_plan(tmp_path, models):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"schema_version": 1, "gpu": "NVIDIA H200 141GB", "gpu_count": 1, "models": models}))
    return str(path)


def test_deploy_cloud_plan_dry_run(run_cli, tmp_path):
    """Two models on one GPU: validated, rented with one port each, started in order, nothing written."""
    big = _plan_recipe(tmp_path, "big", [0.9, 0.55])
    small = _plan_recipe(tmp_path, "small", [0.35])
    plan = _write_plan(
        tmp_path,
        [
            {"recipe": big, "gpu_memory_utilization": 0.55, "gpu_device_ids": [0]},
            {"recipe": small, "gpu_memory_utilization": 0.35, "gpu_device_ids": [0]},
        ],
    )

    rc, stdout, stderr = run_cli(
        "deploy", "cloud", "--plan", plan, "--result-json", str(tmp_path / "out.json"),
        "--lease", str(tmp_path / "lease.json"), "--owner", "relay/deployment-1", "--dry-run",
    )  # fmt: skip

    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
    assert "[dry-run]" in stdout
    assert "Model 0: org/big on GPU 0 at gpu_memory_utilization=0.55, port 8000" in stdout
    assert "Model 1: org/small on GPU 0 at gpu_memory_utilization=0.35, port 8001" in stdout
    assert '"8001"' in stdout and '"8080"' not in stdout  # the rent payload opens one port per model
    assert stdout.index("docker compose up -d vllm_0") < stdout.index("docker compose up -d vllm_1")
    assert "nginx" not in stdout
    assert "Endpoint: http://dry-run-host:8001/v1" in stdout
    assert not (tmp_path / "out.json").exists()
    assert not (tmp_path / "lease.json").exists()


def test_deploy_cloud_invalid_plan_exits_before_renting(run_cli, tmp_path):
    recipe = _plan_recipe(tmp_path, "one", [0.6, 0.4])
    plan = _write_plan(
        tmp_path,
        [
            {"recipe": recipe, "gpu_memory_utilization": 0.6, "gpu_device_ids": [0]},
            {"recipe": recipe, "gpu_memory_utilization": 0.4, "gpu_device_ids": [0]},
        ],
    )

    rc, stdout, stderr = run_cli("deploy", "cloud", "--plan", plan, "--dry-run")

    assert rc == 1
    assert "fractions add up to 1, above 0.95" in stdout
    assert "Creating CloudRift instance" not in stdout


@pytest.mark.parametrize(
    ("extra", "expected_rc"),
    [(["--recipe", "recipes/Qwen3-Embedding-8B"], 2), (["--gpu", "NVIDIA H200 141GB"], 2), (["--lease", "lease.json"], 1)],
    ids=["recipe", "gpu", "lease-without-owner"],
)
def test_deploy_cloud_plan_rejects_conflicting_flags(run_cli, tmp_path, extra, expected_rc):
    recipe = _plan_recipe(tmp_path, "one", [0.9])
    plan = _write_plan(tmp_path, [{"recipe": recipe, "gpu_memory_utilization": 0.9, "gpu_device_ids": [0]}])
    rc, stdout, stderr = run_cli("deploy", "cloud", "--plan", plan, *extra, "--dry-run")
    assert rc == expected_rc, f"stderr: {stderr}\nstdout: {stdout}"
    assert "Creating CloudRift instance" not in stdout


def test_deploy_cloud_help_lists_plan_flags(run_cli):
    rc, stdout, _ = run_cli("deploy", "cloud", "--help")
    assert rc == 0
    for flag in ("--plan", "--result-json", "--lease", "--owner"):
        assert flag in stdout


# ── --vm-active-timeout ───────────────────────────────────────────


@pytest.mark.parametrize(
    ("timeout", "expected_rc"),
    [("3600", 0), ("0", 2), ("-5", 2), ("soon", 2)],
    ids=["positive", "zero", "negative", "not-a-number"],
)
def test_deploy_cloud_vm_active_timeout_must_be_positive(run_cli, tmp_path, timeout, expected_rc):
    recipe = _plan_recipe(tmp_path, "one", [0.9])
    plan = _write_plan(tmp_path, [{"recipe": recipe, "gpu_memory_utilization": 0.9, "gpu_device_ids": [0]}])
    rc, stdout, stderr = run_cli("deploy", "cloud", "--plan", plan, "--vm-active-timeout", timeout, "--dry-run")
    assert rc == expected_rc, f"stderr: {stderr}\nstdout: {stdout}"
    if expected_rc:
        assert "expected a positive integer" in stderr
        assert "Creating CloudRift instance" not in stdout


# ── --vm-proxy ────────────────────────────────────────────────────


def _proxy_dry_run(run_cli, tmp_path, url):
    recipe = _plan_recipe(tmp_path, "one", [0.9])
    plan = _write_plan(tmp_path, [{"recipe": recipe, "gpu_memory_utilization": 0.9, "gpu_device_ids": [0]}])
    return run_cli("deploy", "cloud", "--plan", plan, "--vm-proxy", url, "--dry-run")


def test_deploy_cloud_vm_proxy_configures_the_daemon_and_checks_it_before_the_pull(run_cli, tmp_path):
    rc, stdout, stderr = _proxy_dry_run(run_cli, tmp_path, "http://10.0.0.1:3128")
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
    configure = stdout.index("Configuring Docker daemon proxy http://10.0.0.1:3128 on ")
    dropin = stdout.index(
        'Environment="HTTP_PROXY=http://10.0.0.1:3128" "HTTPS_PROXY=http://10.0.0.1:3128" '
        '"NO_PROXY=localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"'
    )
    assert "> /etc/systemd/system/docker.service.d/http-proxy.conf && systemctl daemon-reload && systemctl restart docker" in stdout
    check = stdout.index(
        "curl -sS --connect-timeout 10 -x http://10.0.0.1:3128 -o /dev/null -w '%{http_code}' https://registry-1.docker.io/v2/"
    )
    assert configure < dropin < check < stdout.index("docker compose pull")
    # The weight download runs in its own container before compose up; it gets the proxy in both spellings.
    download = next(line for line in stdout.splitlines() if "hf download" in line)
    assert " -e HTTPS_PROXY=http://10.0.0.1:3128 " in download and " -e https_proxy=http://10.0.0.1:3128 " in download
    assert "must be able to resolve" not in stdout


def test_deploy_cloud_vm_proxy_named_by_hostname_warns(run_cli, tmp_path):
    rc, stdout, stderr = _proxy_dry_run(run_cli, tmp_path, "http://squid.corp:3128")
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
    assert "proxy http://squid.corp:3128 is named by hostname; the host itself must be able to resolve squid.corp" in stdout


def test_deploy_cloud_vm_proxy_credentials_never_reach_the_output(run_cli, tmp_path):
    rc, stdout, stderr = _proxy_dry_run(run_cli, tmp_path, "http://relay:hunter22@10.0.0.1:3128")
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
    assert "hunter22" not in stdout + stderr
    assert "Configuring Docker daemon proxy *** on " in stdout


@pytest.mark.parametrize(
    "url",
    ["10.0.0.1:3128", "socks5://10.0.0.1:1080", "http://10.0.0.1", "http://10.0.0.1:3128/path"],
    ids=["no-scheme", "socks", "no-port", "path"],
)
def test_deploy_cloud_vm_proxy_must_be_an_http_url_with_a_port(run_cli, tmp_path, url):
    rc, stdout, stderr = _proxy_dry_run(run_cli, tmp_path, url)
    assert rc == 2, f"stderr: {stderr}\nstdout: {stdout}"
    assert "expected http://HOST:PORT or https://HOST:PORT" in stderr
    assert "Creating CloudRift instance" not in stdout
