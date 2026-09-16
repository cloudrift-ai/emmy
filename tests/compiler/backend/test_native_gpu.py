"""Independent cubin execution, stable graph bindings, and zero-before-launch semantics."""

import asyncio
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from emmy.compiler.backend.cuda.program import CompiledProgram
from emmy.compiler.backend.gpu_lock import gpu_lock
from emmy.compiler.backend.native import NativeWorker
from emmy.compiler.backend.pack import save_executable
from emmy.compiler.backend.plan import BufferSpec, ExecutionPlan, KernelSpec, LaunchSpec
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from tests.compiler.helpers import requires_cuda

pytestmark = [requires_cuda, pytest.mark.xdist_group("cuda")]


def _plan():
    return ExecutionPlan(
        "cuda", ["x"], ["y"],
        [BufferSpec("x", (Dim(32),), F32, "input"), BufferSpec("y", (Dim(32),), F32, "output")],
        {}, {},
        [LaunchSpec("y", "add", ("y", "x"), ((1,), (1,), (1,)), ((32,), (1,), (1,)), 0, ("y",))],
        {"add": KernelSpec(source='extern "C" __global__ void add(float* y, const float* x) { int i=threadIdx.x; y[i] += x[i]*2.0f; }')},
    )


def test_native_pack_parity_rebind_graph_and_retirement(tmp_path, monkeypatch):
    executable = shutil.which("emmy-runtime-worker")
    if not executable:
        pytest.skip("build emmy-runtime-worker and add it to PATH")
    monkeypatch.setenv("EMMY_CUBIN_CACHE", str(tmp_path / "cache"))
    x = np.arange(32, dtype=np.float32)
    with gpu_lock():
        plan = _plan()
        reference = CompiledProgram.build_from_plan(plan, {"x": x})
        reference.run_once()
        expected = reference.outputs()["y"]
        root = save_executable(tmp_path / "bundle", {"test": plan}, bindings={"test": {"x": x.tobytes()}}, key={})
        del reference
        shutil.rmtree(tmp_path / "cache")
        # The native child cannot find Python, nvcc, or the removed cubin cache.
        monkeypatch.setenv("PATH", "/nonexistent")

        async def check():
            worker = NativeWorker(executable=executable)
            output = tmp_path / "out.bin"
            try:
                await worker.run_job({"op": "load", "root": str(root), "program": "test"}, wall_timeout_s=30)
                pid = worker._proc.pid
                for capture in (False, True):
                    result = await worker.run_job(
                        {"op": "run", "warmup": 2, "iterations": 3, "capture": capture, "outputs": {"y": str(output)}},
                        wall_timeout_s=30,
                    )
                    assert result["time_ms"] > 0
                    np.testing.assert_array_equal(np.fromfile(output, np.float32), expected)
                    assert worker._proc.pid == pid
                changed = tmp_path / "changed.bin"
                changed.write_bytes((x + 10).tobytes())
                await worker.run_job({"op": "bind", "inputs": {"x": str(changed)}}, wall_timeout_s=30)
                await worker.run_job(
                    {"op": "run", "warmup": 0, "iterations": 2, "capture": True, "outputs": {"y": str(output)}}, wall_timeout_s=30,
                )
                np.testing.assert_array_equal(np.fromfile(output, np.float32), (x + 10) * 2)
                with pytest.raises(RuntimeError, match="unknown program output"):
                    await worker.run_job(
                        {"op": "run", "warmup": 0, "iterations": 1, "capture": False, "outputs": {"missing": str(output)}},
                        wall_timeout_s=30,
                    )
                assert worker._proc is None
                await worker.run_job({"op": "load", "root": str(root), "program": "test"}, wall_timeout_s=30)
                assert worker._proc.pid != pid
                await worker.run_job({"op": "release"}, wall_timeout_s=30)
            finally:
                await worker.aclose()

        asyncio.run(check())
