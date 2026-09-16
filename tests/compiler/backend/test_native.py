"""Native worker failures share the existing supervisor's hard cleanup contract."""

import asyncio
import json
import sys

import pytest

from emmy.compiler.backend.native import NativeWorker


class StubWorker(NativeWorker):
    def __init__(self, source):
        super().__init__(executable=sys.executable)
        self.source = source

    def _command(self):
        return [self.executable, "-c", self.source]


async def test_native_eof_is_not_retried(tmp_path):
    marker = tmp_path / "starts"
    worker = StubWorker(f"from pathlib import Path; p=Path({str(marker)!r}); p.write_text(p.read_text()+'x' if p.exists() else 'x')")
    with pytest.raises(RuntimeError, match="EOF"):
        await worker.run_job({"op": "release"}, wall_timeout_s=5)
    assert marker.read_text() == "x"
    assert worker._proc is None


async def test_native_timeout_and_cancellation_reap_process():
    worker = StubWorker("import time; time.sleep(60)")
    with pytest.raises(RuntimeError, match="wall budget"):
        await worker.run_job({"op": "release"}, wall_timeout_s=0.1)
    assert worker._proc is None
    task = asyncio.create_task(worker.run_job({"op": "release"}, wall_timeout_s=60))
    while worker._proc is None:
        await asyncio.sleep(0.001)
    proc = worker._proc
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert proc.returncode is not None
    assert worker._proc is None


async def test_native_bad_response_retires_process():
    worker = StubWorker("import sys,time; sys.stdin.buffer.read(8); sys.stdout.buffer.write((1).to_bytes(8,'little')+b'?'); sys.stdout.flush(); time.sleep(60)")
    with pytest.raises(json.JSONDecodeError):
        await worker.run_job({"op": "release"}, wall_timeout_s=5)
    assert worker._proc is None


def test_native_wire_version_and_error_translation():
    assert json.loads(NativeWorker._encode({"op": "release"})) == {"version": 1, "command": {"op": "release"}}
    assert NativeWorker._decode(b'{"version":1,"result":{"released":true}}')["released"]
    assert NativeWorker._decode(b'{"version":1,"error":"bad"}')["_retire_worker"]
    with pytest.raises(ValueError, match="version"):
        NativeWorker._decode(b'{"version":2}')
