"""Local command experiments retain the normal records without SSH or provisioning."""

import pytest

from emmy.benchmark.execution import run_execution_group
from emmy.planner import BenchmarkTask, ExecutionGroup
from emmy.planner.variant import Variant
from emmy.provisioning.types import VMConnectionInfo
from emmy.recipe.types import CommandConfig, Recipe


async def test_local_command_group_skips_provisioning(tmp_path, monkeypatch):
    async def forbidden(*args, **kwargs):
        pytest.fail("local commands must not provision or use SSH")

    async def system_info(run_cmd):
        assert await run_cmd("printf local", stream=False) == (0, "local", "")
        return None

    monkeypatch.setattr("emmy.benchmark.execution.provision_remote", forbidden)
    monkeypatch.setattr("emmy.benchmark.execution.SystemInformation.retrieve", system_info)
    monkeypatch.setattr("emmy.provisioning.ssh_transport.scp_from_remote", forbidden)
    gpu = "NVIDIA GeForce RTX 4080"
    recipe = Recipe(command=CommandConfig(run="echo raw > $task_dir/result.txt", result_files=["result.txt"]))
    task = BenchmarkTask(
        recipe_dir=str(tmp_path), variant=Variant(params={"deploy.gpu": gpu, "deploy.gpu_count": 1}),
        recipe=recipe, run_dir=tmp_path,
    )
    results = await run_execution_group(
        ExecutionGroup(gpu_name=gpu, gpu_count=1, tasks=[task]), {"benchmark": {}}, "",
        preallocated_conn=VMConnectionInfo(host="127.0.0.1", username="test", is_local=True),
    )
    assert results[0][1]
    assert task.record.status == "succeeded"
    assert task.record.execution.infrastructure.state == "external"
    assert task.record_path().exists()
    assert (tmp_path / f"{task.file_stem}_result.txt").read_text() == "raw\n"
