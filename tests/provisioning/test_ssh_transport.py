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
    # The signal reaches the group before the kernel schedules the descendant out, so a loaded
    # host can still report it running for a moment; an orphan then remains a zombie until the
    # host's init reaps it. Both are transient, and only a survivor is a failure.
    for _ in range(500):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            return
        stat = Path(f"/proc/{child}/stat")
        if not stat.exists() or stat.read_text().split()[2] == "Z":
            return
        await asyncio.sleep(0.01)
    pytest.fail(f"descendant {child} outlived the run")
