"""Local command deadlines retire the complete process group."""

import asyncio
import os
from pathlib import Path

import pytest

from emmy.provisioning.ssh_transport import make_run_cmd


@pytest.mark.parametrize("cancel", [False, True])
async def test_local_timeout_or_cancellation_kills_descendants(tmp_path, cancel):
    pid_file = tmp_path / "child.pid"
    run = make_run_cmd(None, None, None, local=True)
    job = asyncio.create_task(run(f"sleep 60 & echo $! > {pid_file}; wait", stream=False, timeout=0.2))
    while not pid_file.exists():
        await asyncio.sleep(0.005)
    child = int(pid_file.read_text())
    if cancel:
        job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job
    else:
        assert (await job)[0] == 1
    try:
        os.kill(child, 0)
    except ProcessLookupError:
        pass
    else:
        # An orphan can remain briefly as a zombie until the host's init reaps it.
        assert Path(f"/proc/{child}/stat").read_text().split()[2] == "Z"
