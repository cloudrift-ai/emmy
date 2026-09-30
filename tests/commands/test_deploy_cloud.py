"""deploy cloud with a plan: the rental request, the lease, and the result file it writes."""

import json
from argparse import Namespace
from unittest.mock import AsyncMock

import yaml

from emmy.commands.deploy import cloud
from emmy.provisioning.lease import VmLeaseObserver
from emmy.provisioning.types import VMConnectionInfo

GPU = "NVIDIA H200 141GB"


def _recipe_dir(tmp_path, name, fraction):
    directory = tmp_path / name
    directory.mkdir()
    recipe = {
        "model": {"huggingface": f"org/{name}"},
        "engine": {"llm": {"tensor_parallel_size": 1, "vllm": {"image": "example/vllm"}}},
        "matrices": [{"deploy.gpu": GPU, "deploy.gpu_count": 1, "engine.llm.gpu_memory_utilization": fraction}],
    }
    (directory / "recipe.yaml").write_text(yaml.safe_dump(recipe))
    return str(directory)


async def test_plan_deploy_rents_the_exact_shape_and_writes_the_result(tmp_path, monkeypatch):
    big = _recipe_dir(tmp_path, "big", 0.55)
    small = _recipe_dir(tmp_path, "small", 0.35)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "gpu": GPU,
                "gpu_count": 1,
                "models": [
                    {"recipe": big, "gpu_memory_utilization": 0.55, "gpu_device_ids": [0]},
                    {"recipe": small, "gpu_memory_utilization": 0.35, "gpu_device_ids": [0]},
                ],
            }
        )
    )
    provision = AsyncMock(
        return_value=VMConnectionInfo(
            host="203.0.113.10",
            username="riftuser",
            ssh_port=2222,
            port_mappings=[(22, 2222), (8000, 40000), (8001, 40001)],
            delete_info=("cloudrift", "inst-1"),
        )
    )
    monkeypatch.setattr(cloud, "provision_cloud_vm", provision)
    monkeypatch.setattr(cloud, "provision_remote", AsyncMock())
    monkeypatch.setattr(cloud, "deploy_entry", AsyncMock(return_value=True))
    args = Namespace(
        plan=str(plan),
        recipe=None,
        gpu=None,
        gpu_count=None,
        name="cloud-deploy",
        ssh_key=str(tmp_path / "id_ed25519"),
        authorized_key=None,
        hf_token="",
        model_dir="/hf_models",
        dry_run=False,
        billing_exempt=False,
        network=None,
        provider=None,
        lease=tmp_path / "lease.json",
        owner="relay/deployment-42",
        result_json=str(tmp_path / "out.json"),
    )

    await cloud._handle_cloud(args)

    kwargs = provision.call_args.kwargs
    assert (kwargs["gpu_name"], kwargs["gpu_count"], kwargs["ports"], kwargs["exact_gpu_count"]) == (GPU, 1, [22, 8000, 8001], True)
    assert kwargs["allocation_observer"] == VmLeaseObserver(tmp_path / "lease.json", "relay/deployment-42", GPU, 1)
    params = cloud.deploy_entry.call_args.args[0]
    assert [(service.gpu_device_ids, service.port) for service in params.services] == [([0], 8000), ([0], 8001)]
    assert params.load_balancer is False
    assert json.loads((tmp_path / "out.json").read_text()) == {
        "schema_version": 1,
        "cloud_instance_id": "inst-1",
        "models": [
            {"index": 0, "recipe": big, "endpoint": "http://203.0.113.10:40000/v1"},
            {"index": 1, "recipe": small, "endpoint": "http://203.0.113.10:40001/v1"},
        ],
    }
