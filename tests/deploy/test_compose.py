"""Unit tests for compose and nginx generation."""

import yaml

from emmy.deploy import Service, generate_compose, generate_nginx_conf, replica_services
from emmy.provisioning.proxy import NO_PROXY, proxy_env
from emmy.recipe import Recipe


def _compose_replicas(recipe):
    """The compose file of a recipe fanned out over its deploy.gpu_count (replicas + nginx)."""
    services, load_balancer = replica_services(recipe)
    return generate_compose(services, "/mnt/models", "test-token", load_balancer=load_balancer)


# ── generate_compose ────────────────────────────────────────────────


def test_compose_single_instance(sample_config):
    recipe = Recipe.from_dict(sample_config)
    result = generate_compose([Service(recipe)], "/mnt/models", "test-token")

    assert "vllm_0:" in result
    assert "vllm_1:" not in result
    assert "nginx:" not in result
    assert "count: all" in result
    assert '"8000:8000"' in result
    assert "--tensor-parallel-size 1" in result
    assert "--pipeline-parallel-size 1" in result
    assert "--data-parallel-size 1" in result
    assert "--model test-org/test-model" in result
    assert "--served-model-name test-org/test-model" in result
    assert "--max-model-len 8192" in result
    assert "HUGGING_FACE_HUB_TOKEN=test-token" in result
    assert "/mnt/models:/mnt/models" in result


def test_compose_context_length_and_max_concurrent(sample_config):
    sample_config["engine"]["llm"]["max_concurrent_requests"] = 256
    recipe = Recipe.from_dict(sample_config)
    result = generate_compose([Service(recipe)], "/mnt/models", "token")
    assert "--max-model-len 8192" in result
    assert "--max-num-seqs 256" in result


def test_compose_uses_named_model_revision(sample_config):
    sample_config["model"]["revision"] = "0123456789abcdef"
    recipe = Recipe.from_dict(sample_config)
    result = generate_compose([Service(recipe)], "/mnt/models", "token")
    assert "--revision 0123456789abcdef" in result


def test_compose_uses_pinned_adapter_snapshot(sample_config):
    revision = "a" * 40
    sample_config["engine"]["llm"]["vllm"]["lora_adapter"] = {
        "name": "limo",
        "huggingface": "org/adapter",
        "revision": revision,
        "rank": 8,
    }
    recipe = Recipe.from_dict(sample_config)
    result = generate_compose([Service(recipe)], "/mnt/models", "token")

    assert "--enable-lora" in result
    assert "--max-lora-rank 8" in result
    assert f"--lora-modules limo=/mnt/models/hub/models--org--adapter/snapshots/{revision}" in result


def test_compose_omits_unset_named_fields():
    config = {
        "model": {"huggingface": "test-org/test-model"},
        "engine": {
            "llm": {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
                "vllm": {
                    "image": "vllm/vllm-openai:v0.17.0",
                },
            }
        },
    }
    recipe = Recipe.from_dict(config)
    result = generate_compose([Service(recipe)], "/mnt/models", "token")
    assert "--max-model-len" not in result
    assert "--max-num-seqs" not in result


def test_compose_multi_instance(sample_config_multi):
    recipe = Recipe.from_dict(sample_config_multi)
    result = _compose_replicas(recipe)

    # Two vLLM services
    assert "vllm_0:" in result
    assert "vllm_1:" in result
    assert "vllm_2:" not in result

    # Nginx load balancer
    assert "nginx:" in result
    assert "nginx_lb" in result
    assert '"8080:8080"' in result

    # GPU device IDs for multi-instance (not count: all)
    assert "device_ids:" in result
    assert "'0'" in result
    assert "'3'" in result

    # Ports
    assert '"8000:8000"' in result
    assert '"8001:8000"' in result


def test_compose_parses_as_valid_yaml(sample_config):
    recipe = Recipe.from_dict(sample_config)
    result = generate_compose([Service(recipe)], "/mnt/models", "test-token")
    parsed = yaml.safe_load(result)
    assert "services" in parsed
    assert "vllm_0" in parsed["services"]


def test_compose_multi_gpu_allocation(sample_config_multi):
    recipe = Recipe.from_dict(sample_config_multi)
    result = _compose_replicas(recipe)
    parsed = yaml.safe_load(result)

    # Instance 0 should get GPUs 0-3, instance 1 gets GPUs 4-7
    vllm_0 = parsed["services"]["vllm_0"]
    vllm_1 = parsed["services"]["vllm_1"]

    gpu_0 = vllm_0["deploy"]["resources"]["reservations"]["devices"][0]
    gpu_1 = vllm_1["deploy"]["resources"]["reservations"]["devices"][0]

    assert gpu_0["device_ids"] == ["0", "1", "2", "3"]
    assert gpu_1["device_ids"] == ["4", "5", "6", "7"]


def test_compose_image_from_config(sample_config):
    sample_config["engine"]["llm"]["vllm"]["image"] = "custom/image:v2"
    recipe = Recipe.from_dict(sample_config)
    result = generate_compose([Service(recipe)], "/mnt/models", "token")
    assert "custom/image:v2" in result


def test_compose_gpu_device_ids_override(sample_config):
    """gpu_device_ids restricts GPU visibility in single-instance mode."""
    recipe = Recipe.from_dict(sample_config)
    result = generate_compose([Service(recipe, [0, 1])], "/mnt/models", "token")
    assert "device_ids:" in result
    assert "'0'" in result
    assert "'1'" in result
    assert "count: all" not in result


def test_compose_no_gpu_device_ids_uses_count_all(sample_config):
    """Without gpu_device_ids, single-instance uses count: all."""
    recipe = Recipe.from_dict(sample_config)
    result = generate_compose([Service(recipe)], "/mnt/models", "token")
    assert "count: all" in result
    assert "device_ids:" not in result


def test_compose_proxy_reaches_every_engine_service_but_not_nginx(sample_config_multi):
    url = "http://10.0.0.1:3128"
    services, load_balancer = replica_services(Recipe.from_dict(sample_config_multi))
    result = generate_compose(services, "/mnt/models", "token", load_balancer=load_balancer, proxy=url)

    proxy_block = (
        f"      - HTTP_PROXY={url}\n"
        f"      - HTTPS_PROXY={url}\n"
        f"      - NO_PROXY={NO_PROXY}\n"
        f"      - http_proxy={url}\n"
        f"      - https_proxy={url}\n"
        f"      - no_proxy={NO_PROXY}\n"
        "    ports:\n"
    )
    assert result.count(proxy_block) == len(services) == 2
    parsed = yaml.safe_load(result)
    for name in ("vllm_0", "vllm_1"):
        assert parsed["services"][name]["environment"][-6:] == [f"{k}={v}" for k, v in proxy_env(url).items()]
    assert "environment" not in parsed["services"]["nginx"]


def test_compose_without_proxy_is_unchanged(sample_config):
    recipe = Recipe.from_dict(sample_config)
    result = generate_compose([Service(recipe)], "/mnt/models", "token")
    assert result == generate_compose([Service(recipe)], "/mnt/models", "token", proxy=None)
    assert "proxy" not in result.lower()


# ── generate_nginx_conf ─────────────────────────────────────────────


def test_nginx_basic_structure():
    result = generate_nginx_conf(["vllm_0", "vllm_1"])
    assert "least_conn" in result
    assert "vllm_0:8000" in result
    assert "vllm_1:8000" in result
    assert "proxy_buffering off" in result
    assert "listen 8080" in result
    assert "llm_backend" in result


def test_nginx_server_count():
    result = generate_nginx_conf(["vllm_0", "vllm_1", "vllm_2", "vllm_3"])
    assert "vllm_0:8000" in result
    assert "vllm_1:8000" in result
    assert "vllm_2:8000" in result
    assert "vllm_3:8000" in result
    assert "vllm_4:8000" not in result


def test_nginx_proxy_timeouts():
    result = generate_nginx_conf(["vllm_0", "vllm_1"])
    assert "proxy_connect_timeout 600s" in result
    assert "proxy_send_timeout 600s" in result
    assert "proxy_read_timeout 600s" in result


def test_nginx_sglang_engine():
    result = generate_nginx_conf(["sglang_0", "sglang_1"])
    assert "sglang_0:8000" in result
    assert "sglang_1:8000" in result
    assert "vllm_0" not in result
    assert "llm_backend" in result


# ── SGLang compose ─────────────────────────────────────────────────


def test_compose_sglang_single_instance(sample_config_sglang):
    recipe = Recipe.from_dict(sample_config_sglang)
    result = generate_compose([Service(recipe)], "/mnt/models", "test-token")

    assert "sglang_0:" in result
    assert "vllm_0:" not in result
    assert "lmsysorg/sglang:v0.5.9" in result
    assert "entrypoint: python3 -m sglang.launch_server" in result
    assert "--model-path test-org/test-model" in result
    assert "--model test-org/test-model" not in result
    assert "--tp 1" in result
    assert "--pp-size 1" in result
    assert "--dp 1" in result
    assert "--mem-fraction-static 0.9" in result
    assert "--context-length 8192" in result


def test_compose_vllm_no_entrypoint(sample_config):
    """vLLM compose should not have an entrypoint override."""
    recipe = Recipe.from_dict(sample_config)
    result = generate_compose([Service(recipe)], "/mnt/models", "test-token")
    assert "entrypoint:" not in result


def test_compose_vllm_explicit_entrypoint_for_same_image_control(sample_config):
    sample_config["engine"]["llm"]["vllm"]["entrypoint"] = "python3 -m vllm.entrypoints.openai.api_server"
    recipe = Recipe.from_dict(sample_config)

    result = generate_compose([Service(recipe)], "/mnt/models", "test-token")

    assert "entrypoint: python3 -m vllm.entrypoints.openai.api_server" in result


def test_compose_sglang_parses_as_valid_yaml(sample_config_sglang):
    recipe = Recipe.from_dict(sample_config_sglang)
    result = generate_compose([Service(recipe)], "/mnt/models", "test-token")
    parsed = yaml.safe_load(result)
    assert "services" in parsed
    assert "sglang_0" in parsed["services"]


# ── extra_env compose ──────────────────────────────────────────────


def test_compose_extra_env_appears_in_environment(sample_config):
    sample_config["engine"]["llm"]["vllm"]["extra_env"] = {
        "VLLM_ATTENTION_BACKEND": "FLASHINFER",
        "CUDA_LAUNCH_BLOCKING": "1",
    }
    recipe = Recipe.from_dict(sample_config)
    result = generate_compose([Service(recipe)], "/mnt/models", "token")
    assert "VLLM_ATTENTION_BACKEND=FLASHINFER" in result
    assert "CUDA_LAUNCH_BLOCKING=1" in result


def test_compose_empty_extra_env_produces_no_extra_lines(sample_config):
    recipe = Recipe.from_dict(sample_config)
    result = generate_compose([Service(recipe)], "/mnt/models", "token")
    parsed = yaml.safe_load(result)
    env = parsed["services"]["vllm_0"]["environment"]
    assert len(env) == 2  # HUGGING_FACE_HUB_TOKEN and HF_HOME only


# ── prebuilt-image HF cache ───────────────────────────────────────


def test_compose_defers_to_baked_hf_home(sample_config):
    """A prebuilt per-model image bakes the snapshot under its own HF_HOME and runs
    offline; overriding HF_HOME would hide it, so the compose override is dropped."""
    recipe = Recipe.from_dict(sample_config)
    result = generate_compose([Service(recipe)], "/mnt/models", "token", baked_images={"vllm/vllm-openai:v0.17.0"})
    parsed = yaml.safe_load(result)
    env = parsed["services"]["vllm_0"]["environment"]
    assert not any(e.startswith("HF_HOME=") for e in env)
    assert "HUGGING_FACE_HUB_TOKEN=token" in env


def test_compose_baked_hf_home_yields_to_speculative_drafter(sample_config):
    """A speculative-config lane names a model the baked cache does not hold, and the
    image pins offline mode — the drafter can resolve only from the host cache, so the
    HF_HOME override returns for exactly this case (the emmy+MTP lanes booted with
    'Invalid repository ID' for the drafter otherwise, 2026-08-06)."""
    sample_config["engine"]["llm"]["vllm"]["extra_args"] = (
        '--speculative-config \'{"method":"mtp","model":"google/gemma-4-12B-it-assistant","num_speculative_tokens":2}\''
    )
    recipe = Recipe.from_dict(sample_config)
    result = generate_compose([Service(recipe)], "/mnt/models", "token", baked_images={"vllm/vllm-openai:v0.17.0"})
    env = yaml.safe_load(result)["services"]["vllm_0"]["environment"]
    assert "HF_HOME=/mnt/models" in env


def test_compose_baked_hf_home_keeps_extra_env(sample_config):
    """Dropping the HF_HOME override must not disturb the recipe's own env lines."""
    sample_config["engine"]["llm"]["vllm"]["extra_env"] = {"EMMY_FAST_MATH": "1"}
    recipe = Recipe.from_dict(sample_config)
    result = generate_compose([Service(recipe)], "/mnt/models", "token", baked_images={"vllm/vllm-openai:v0.17.0"})
    env = yaml.safe_load(result)["services"]["vllm_0"]["environment"]
    assert "EMMY_FAST_MATH=1" in env
    assert len(env) == 2  # token + the recipe's own var, no HF_HOME


def test_compose_with_extra_env_parses_as_valid_yaml(sample_config):
    sample_config["engine"]["llm"]["vllm"]["extra_env"] = {"MY_VAR": "hello"}
    recipe = Recipe.from_dict(sample_config)
    result = generate_compose([Service(recipe)], "/mnt/models", "token")
    parsed = yaml.safe_load(result)
    env = parsed["services"]["vllm_0"]["environment"]
    assert "MY_VAR=hello" in env


# ── docker_options compose ────────────────────────────────────────


def test_compose_docker_options_renders_in_service(sample_config):
    sample_config["engine"]["llm"]["docker_options"] = {
        "security_opt": ["seccomp=unconfined"],
        "cap_add": ["SYS_PTRACE"],
    }
    recipe = Recipe.from_dict(sample_config)
    result = generate_compose([Service(recipe)], "/mnt/models", "token")
    parsed = yaml.safe_load(result)
    svc = parsed["services"]["vllm_0"]
    assert svc["security_opt"] == ["seccomp=unconfined"]
    assert svc["cap_add"] == ["SYS_PTRACE"]


def test_compose_docker_options_valid_yaml(sample_config):
    sample_config["engine"]["llm"]["docker_options"] = {
        "security_opt": ["seccomp=unconfined"],
        "privileged": True,
    }
    recipe = Recipe.from_dict(sample_config)
    result = generate_compose([Service(recipe)], "/mnt/models", "token")
    parsed = yaml.safe_load(result)
    assert "services" in parsed
    assert "vllm_0" in parsed["services"]


def test_compose_empty_docker_options_no_change(sample_config):
    recipe = Recipe.from_dict(sample_config)
    result = generate_compose([Service(recipe)], "/mnt/models", "token")
    assert "security_opt" not in result
    assert "cap_add" not in result


def test_compose_docker_options_nested_values(sample_config):
    sample_config["engine"]["llm"]["docker_options"] = {
        "ulimits": {"memlock": {"soft": -1, "hard": -1}},
    }
    recipe = Recipe.from_dict(sample_config)
    result = generate_compose([Service(recipe)], "/mnt/models", "token")
    parsed = yaml.safe_load(result)
    svc = parsed["services"]["vllm_0"]
    assert svc["ulimits"] == {"memlock": {"soft": -1, "hard": -1}}


# ── restart policy ────────────────────────────────────────────────


def test_compose_restart_policy_on_engine_service(sample_config):
    recipe = Recipe.from_dict(sample_config)
    result = generate_compose([Service(recipe)], "/mnt/models", "token")
    parsed = yaml.safe_load(result)
    assert parsed["services"]["vllm_0"]["restart"] == "unless-stopped"


def test_compose_restart_policy_on_nginx_service(sample_config_multi):
    recipe = Recipe.from_dict(sample_config_multi)
    result = _compose_replicas(recipe)
    parsed = yaml.safe_load(result)
    assert parsed["services"]["nginx"]["restart"] == "unless-stopped"


def test_compose_autoheal_restarts_only_labelled_engine_services(sample_config):
    """An engine whose core died can keep its container up; autoheal restarts it once unhealthy,
    and only containers carrying its label, never another container on the host."""
    recipe = Recipe.from_dict(sample_config)
    parsed = yaml.safe_load(generate_compose([Service(recipe)], "/mnt/models", "token"))
    assert parsed["services"]["vllm_0"]["labels"] == ["autoheal=true"]
    autoheal = parsed["services"]["autoheal"]
    assert "AUTOHEAL_CONTAINER_LABEL=autoheal" in autoheal["environment"]
    assert "/var/run/docker.sock:/var/run/docker.sock" in autoheal["volumes"]


# ── several models on one host ─────────────────────────────────────


def test_compose_services_pin_their_devices_and_ports(sample_config, sample_config_sglang):
    """Two models sharing device 0: each service keeps its own engine, fraction and host port, no nginx."""
    sample_config["engine"]["llm"]["gpu_memory_utilization"] = 0.55
    sample_config_sglang["engine"]["llm"]["gpu_memory_utilization"] = 0.35
    services = [Service(Recipe.from_dict(sample_config), [0], 8000), Service(Recipe.from_dict(sample_config_sglang), [0], 8001)]

    parsed = yaml.safe_load(generate_compose(services, "/mnt/models", "token"))

    assert list(parsed["services"]) == ["vllm_0", "sglang_1", "autoheal"]
    for name in ["vllm_0", "sglang_1"]:
        assert parsed["services"][name]["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"] == ["0"]
        assert "depends_on" not in parsed["services"][name]
    assert parsed["services"]["vllm_0"]["ports"] == ["8000:8000"]
    assert parsed["services"]["sglang_1"]["ports"] == ["8001:8000"]
    assert "--gpu-memory-utilization=0.55" in parsed["services"]["vllm_0"]["command"]
    assert "--mem-fraction-static 0.35" in parsed["services"]["sglang_1"]["command"]


def test_compose_load_balancer_has_no_depends_on(sample_config_multi):
    """The orchestrator orders the start; nginx no longer waits on the engines inside compose."""
    parsed = yaml.safe_load(_compose_replicas(Recipe.from_dict(sample_config_multi)))
    assert "depends_on" not in parsed["services"]["nginx"]
    assert "depends_on" not in yaml.safe_load(_compose_replicas(Recipe.from_dict(sample_config_multi)))["services"]["vllm_1"]


def test_replica_services_single_service_sees_the_given_devices(sample_config):
    recipe = Recipe.from_dict(sample_config)
    assert replica_services(recipe) == ([Service(recipe, None, 8000)], False)
    assert replica_services(recipe, [2, 3]) == ([Service(recipe, [2, 3], 8000)], False)
