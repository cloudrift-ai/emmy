"""`emmy deploy ssh --plan`: a plan's services deploy to an existing host of exactly the plan's shape."""

import json
from argparse import Namespace
from unittest.mock import AsyncMock

import pytest
import yaml

from emmy.commands.deploy import ssh

GPU = "NVIDIA H100 80GB"


def _recipe_dir(tmp_path, name, fraction):
    directory = tmp_path / name
    directory.mkdir()
    (directory / "recipe.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {"huggingface": f"org/{name}"},
                "engine": {"llm": {"gpu_memory_utilization": fraction, "vllm": {"image": "example/vllm"}}},
                "matrices": [{"deploy.gpu": GPU, "deploy.gpu_count": 1}],
            }
        )
    )
    return str(directory)


def _plan(tmp_path, gpu_count=1):
    big = _recipe_dir(tmp_path, "big", 0.62)
    small = _recipe_dir(tmp_path, "small", 0.3)
    plan = tmp_path / "plan.json"
    models = [
        {"recipe": big, "gpu_memory_utilization": 0.62, "gpu_device_ids": [0]},
        {"recipe": small, "gpu_memory_utilization": 0.3, "gpu_device_ids": [0]},
    ]
    plan.write_text(json.dumps({"schema_version": 1, "gpu": GPU, "gpu_count": gpu_count, "models": models}))
    return plan


def _args(plan, **overrides):
    args = Namespace(
        plan=str(plan),
        recipe=None,
        ssh="rift@203.0.113.10:2222",
        server=None,
        ssh_port=None,
        ssh_key="~/.ssh/id_ed25519",
        gpu=None,
        gpu_count=None,
        hf_token="",
        model_dir="/mnt/models",
        teardown=False,
        dry_run=False,
        scale_out_strategy="data-parallelism",
    )
    vars(args).update(overrides)
    return args


@pytest.mark.asyncio
async def test_plan_deploys_its_services_to_the_detected_host(tmp_path, monkeypatch):
    monkeypatch.setattr(ssh, "detect_remote_gpus", AsyncMock(return_value=(GPU, 1)))
    monkeypatch.setattr(ssh, "provision_remote", AsyncMock())
    monkeypatch.setattr(ssh, "deploy_entry", AsyncMock(return_value=True))

    await ssh._handle_ssh(_args(_plan(tmp_path)))

    params = ssh.deploy_entry.call_args.args[0]
    assert (params.server, params.ssh_port) == ("rift@203.0.113.10", 2222)
    assert [(service.gpu_device_ids, service.port) for service in params.services] == [([0], 8000), ([0], 8001)]
    assert [service.recipe.engine.llm.gpu_memory_utilization for service in params.services] == [0.62, 0.3]
    assert params.load_balancer is False


@pytest.mark.asyncio
async def test_plan_for_another_shape_is_refused_before_provisioning(tmp_path, monkeypatch):
    monkeypatch.setattr(ssh, "detect_remote_gpus", AsyncMock(return_value=(GPU, 1)))
    monkeypatch.setattr(ssh, "provision_remote", AsyncMock())
    monkeypatch.setattr(ssh, "deploy_entry", AsyncMock(return_value=True))

    with pytest.raises(SystemExit, match="1"):
        await ssh._handle_ssh(_args(_plan(tmp_path, gpu_count=2)))

    ssh.provision_remote.assert_not_called()
    ssh.deploy_entry.assert_not_called()
