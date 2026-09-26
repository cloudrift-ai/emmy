"""Local command deadlines retire the complete process group."""

import asyncio
import os
from pathlib import Path

import pytest

from emmy.provisioning.ssh_transport import make_run_cmd


@pytest.mark.parametrize("mode", ["timeout", "cancel", "exit"])
async def test_local_command_kills_descendants(tmp_path, mode):
    pid_file = tmp_path / "child.pid"
    run = make_run_cmd(None, None, None, local=True)
    command = f"sleep 60 >/dev/null 2>&1 & echo $! > {pid_file}"
    if mode != "exit":
        command += "; wait"
    job = asyncio.create_task(run(command, stream=False, timeout=0.2))
    while not pid_file.exists():
        await asyncio.sleep(0.005)
    child = int(pid_file.read_text())
    if mode == "cancel":
        job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job
    else:
        assert (await job)[0] == (1 if mode == "timeout" else 0)
    await _wait_until_gone(child)


async def test_local_command_cancelled_during_spawn_kills_descendants(tmp_path, monkeypatch):
    """A cancel that lands while the shell is still being spawned must reach its descendants too:
    asyncio then closes the transport, which kills the shell alone, so the group kill has to wait
    for the spawn to finish."""
    pid_file = tmp_path / "child.pid"
    spawned, release = asyncio.Event(), asyncio.Event()
    real_spawn = asyncio.create_subprocess_exec

    async def spawn_slowly(*argv, **kwargs):
        proc = await real_spawn(*argv, **kwargs)
        spawned.set()
        await release.wait()
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn_slowly)
    run = make_run_cmd(None, None, None, local=True)
    job = asyncio.create_task(run(f"sleep 60 >/dev/null 2>&1 & echo $! > {pid_file}; wait", stream=False))
    await spawned.wait()
    while not pid_file.exists():
        await asyncio.sleep(0.005)
    job.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await job
    await _wait_until_gone(int(pid_file.read_text()))


async def _wait_until_gone(child):
    # The signal reaches the group before the kernel schedules the descendant out, so a loaded
    # host can still report it running for a moment; an orphan then remains a zombie until the
    # host's init reaps it. Both are transient, and only a survivor is a failure. The process can
    # also disappear mid-check, which procfs reports as either a missing entry or a dead process.
    procfs = Path("/proc")
    for _ in range(500):
        try:
            os.kill(child, 0)
            if procfs.is_dir() and (procfs / str(child) / "stat").read_text().split()[2] == "Z":
                return
        except (ProcessLookupError, FileNotFoundError):
            return
        await asyncio.sleep(0.01)
    pytest.fail(f"descendant {child} outlived the run")
