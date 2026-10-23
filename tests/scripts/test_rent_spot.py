"""Spot retries stop on success or terminal errors and keep the exact owned lease."""

from argparse import Namespace
from unittest.mock import AsyncMock

import pytest

from emmy.provisioning.errors import TerminalProvisionError
from scripts import rent_spot


@pytest.mark.parametrize("terminal", [False, True])
async def test_rental_retries_capacity_but_not_terminal_errors(tmp_path, monkeypatch, terminal):
    args = Namespace(
        lease=tmp_path / "lease.json",
        owner="test",
        gpu="NVIDIA H200 141GB",
        gpu_count=8,
        zones=["valid-a", "absent-b"],
        config="config.yaml",
        ssh_key=tmp_path / "key",
        name="test",
        retry_seconds=300,
    )
    monkeypatch.setattr(rent_spot, "run_shell_cmd", AsyncMock(return_value=(0, '[{"zone":"zones/valid-a"}]', "")))
    monkeypatch.setattr(rent_spot, "load_config", lambda _: {"providers": {}})
    monkeypatch.setattr(rent_spot, "GPU_GCP_ZONES", {})
    provision = AsyncMock(side_effect=[TerminalProvisionError("auth")] if terminal else [None, object()])
    monkeypatch.setattr(rent_spot, "provision_cloud_vm", provision)
    sleep = AsyncMock()
    monkeypatch.setattr(rent_spot.asyncio, "sleep", sleep)
    monkeypatch.setattr(rent_spot, "load_owned_lease", lambda *args: {"vm": {"ssh_target": "user@host"}})
    if terminal:
        with pytest.raises(TerminalProvisionError):
            await rent_spot.rent(args)
        sleep.assert_not_awaited()
    else:
        assert await rent_spot.rent(args) == "user@host"
        assert provision.await_count == 2
        sleep.assert_awaited_once_with(300)
    assert rent_spot.GPU_GCP_ZONES[args.gpu] == ["valid-a"]
    assert provision.call_args.kwargs["exact_gpu_count"] is True
    assert provision.call_args.kwargs["provisioning_model"] == "SPOT"


async def test_failed_benchmark_releases_only_its_owned_vm(tmp_path, monkeypatch):
    args = Namespace(lease=tmp_path / "lease.json", owner="test-owner", recipes=["recipe"], ssh_key=tmp_path / "key", config="config.yaml")
    monkeypatch.setattr(rent_spot, "rent", AsyncMock(return_value="user@host"))
    process = Namespace(returncode=1, wait=AsyncMock(return_value=1))
    monkeypatch.setattr(rent_spot.asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    cleanup = AsyncMock()
    monkeypatch.setattr(rent_spot, "delete_owned_vm", cleanup)
    assert await rent_spot.run(args) == 1
    cleanup.assert_awaited_once_with(args.lease, args.owner)
