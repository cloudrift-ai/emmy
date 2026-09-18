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
    worker = StubWorker(
        "import sys,time; sys.stdin.buffer.read(8); sys.stdout.buffer.write((1).to_bytes(8,'little')+b'?'); "
        "sys.stdout.flush(); time.sleep(60)"
    )
    with pytest.raises(json.JSONDecodeError):
        await worker.run_job({"op": "release"}, wall_timeout_s=5)
    assert worker._proc is None


def test_native_wire_version_and_error_translation():
    assert json.loads(NativeWorker._encode({"op": "release"})) == {"version": 1, "command": {"op": "release"}}
    assert NativeWorker._decode(b'{"version":1,"result":{"released":true}}')["released"]
    assert NativeWorker._decode(b'{"version":1,"error":"bad"}')["_retire_worker"]
    with pytest.raises(ValueError, match="version"):
        NativeWorker._decode(b'{"version":2}')


async def test_pack_comparison_matches_lifetimes_and_closes_workers(tmp_path, monkeypatch):
    from pathlib import Path

    from emmy.compiler.backend import native
    from emmy.compiler.backend.plan import ExecutionPlan, plan_to_dict

    plan = ExecutionPlan("cuda", [], [], [], {}, {}, [], {})
    (tmp_path / "plan.json").write_text(json.dumps(plan_to_dict(plan)))
    (tmp_path / "manifest.json").write_text(json.dumps({"programs": {"test": "plan.json"}}))
    workers = []

    class Worker:
        def __init__(self, **kwargs):
            self.loads = self.runs = self.closes = 0
            workers.append(self)

        async def run_job(self, request, **kwargs):
            if request["op"] == "load":
                self.loads += 1
                assert Path(request["root"]) == tmp_path
                return {"loaded": True}
            self.runs += 1
            assert request["iterations"] == 7 and request["warmup"] == 2
            return {"time_ms": 0.1}

        async def aclose(self):
            self.closes += 1

    monkeypatch.setattr(native, "NativeWorker", Worker)
    monkeypatch.setattr(native, "PythonPackWorker", Worker)
    result = await native.benchmark_pack(tmp_path, warmup=2, iterations=7)
    assert len(result["rows"]) == 24
    assert len(workers) == 8
    assert [w.loads for w in workers] == [2, 3] * 4
    assert len(result["reloads"]) == 4
    assert all(w.runs == 3 and w.closes >= 1 for w in workers)
