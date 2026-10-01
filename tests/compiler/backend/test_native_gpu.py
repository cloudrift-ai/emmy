"""Independent cubin execution, stable graph bindings, and zero-before-launch semantics."""

import asyncio
import shutil

import numpy as np
import pytest

from emmy.compiler.backend.cuda.program import CompiledProgram, kernel_attributes
from emmy.compiler.backend.gpu_lock import gpu_lock
from emmy.compiler.backend.native import NativeWorker
from emmy.compiler.backend.pack import save_executable
from emmy.compiler.backend.plan import BufferSpec, ExecutionPlan, KernelSpec, LaunchSpec
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.ir.cuda import TmaDescMeta
from tests.compiler.helpers import requires_cuda, requires_sm

pytestmark = [requires_cuda, pytest.mark.xdist_group("cuda")]


def _plan():
    return ExecutionPlan(
        "cuda",
        ["x"],
        ["y"],
        [
            BufferSpec("x", (Dim(32),), F32, "input"),
            BufferSpec("w", (Dim(1),), F32, "constant"),
            BufferSpec("s", (Dim(32),), F32, "scratch"),
            BufferSpec("y", (Dim(32),), F32, "output"),
        ],
        {"w": 2.0},
        {},
        [
            LaunchSpec(out, "add", (out, inp, "w"), ((1,), (1,), (1,)), ((32,), (1,), (1,)), 0, (out,))
            for inp, out in (("x", "s"), ("s", "y"))
        ],
        {
            "add": KernelSpec(
                source='extern "C" __global__ void add(float* y, const float* x, const float* w) { int i=threadIdx.x; y[i] += x[i]*w[0]; }'
            )
        },
    )


def _bindings(x):
    return {"x": x.tobytes(), "w": np.float32(2).tobytes()}


@requires_sm(9, 0)
def test_first_program_graph_capture_prepares_tma_descriptors():
    source = r"""struct __align__(64) CUtensorMap { unsigned long long opaque[16]; };
extern "C" __global__ void tma_copy(float* y, const CUtensorMap* x_desc) {
    __shared__ __align__(128) float tile[16][16];
    __shared__ unsigned long long bar;
    if (threadIdx.x == 0) {
        unsigned int tile_addr = __cvta_generic_to_shared(tile);
        unsigned int bar_addr = __cvta_generic_to_shared(&bar);
        unsigned long long state;
        asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(bar_addr) : "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 %0, [%1], 1024;"
                     : "=l"(state) : "r"(bar_addr) : "memory");
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
                     "[%0], [%1, {%2, %3}], [%4];"
                     :: "r"(tile_addr), "l"(x_desc), "r"(0), "r"(0), "r"(bar_addr) : "memory");
        asm volatile("{.reg .pred p; wait: mbarrier.try_wait.parity.shared.b64 p, [%0], 0; @!p bra wait;}"
                     :: "r"(bar_addr) : "memory");
    }
    __syncthreads();
    y[threadIdx.x] = tile[threadIdx.x / 16][threadIdx.x % 16] + 1.0f;
}
"""
    plan = ExecutionPlan(
        "cuda",
        ["x"],
        ["y"],
        [BufferSpec("x", (Dim(16), Dim(16)), F32, "input"), BufferSpec("y", (Dim(16), Dim(16)), F32, "output")],
        {},
        {},
        [
            LaunchSpec(
                "y",
                "tma_copy",
                ("y", "x_desc"),
                ((1,), (1,), (1,)),
                ((256,), (1,), (1,)),
                0,
                (),
                tma_descriptors=(TmaDescMeta("x_desc", "x", (16, 16)),),
            )
        ],
        {"tma_copy": KernelSpec(source=source, arch_specific=True)},
    )
    x = np.arange(256, dtype=np.float32).reshape(16, 16)
    with gpu_lock():
        program = CompiledProgram.build_from_plan(plan, {"x": x})
        # Capture before run_once, timing, or any other descriptor-preparing launch.
        program.capture_program_graph()
        for values in (x, x + 10):
            program.upload_prefix({"x": values})
            program.replay_program_graph()
            np.testing.assert_array_equal(program.outputs()["y"], values + 1)


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
        np.testing.assert_array_equal(expected, x * 4)
        root = save_executable(tmp_path / "bundle", {"test": plan}, bindings={"test": _bindings(x)}, key={})
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
                    {"op": "run", "warmup": 0, "iterations": 2, "capture": True, "outputs": {"y": str(output)}},
                    wall_timeout_s=30,
                )
                np.testing.assert_array_equal(np.fromfile(output, np.float32), (x + 10) * 4)
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


@pytest.mark.parametrize("fault", ["deadline", "cuda_error"])
def test_native_gpu_failure_restarts_cleanly(tmp_path, monkeypatch, fault):
    executable = shutil.which("emmy-runtime-worker")
    if not executable:
        pytest.skip("build emmy-runtime-worker and add it to PATH")
    monkeypatch.setenv("EMMY_CUBIN_CACHE", str(tmp_path / "cache"))
    good = _plan()
    bad = _plan()
    body = (
        "unsigned long long start=clock64(); while(clock64()-start < 1000000000ULL) {} y[threadIdx.x]=x[threadIdx.x];"
        if fault == "deadline"
        else "*(volatile float*)0 = 1.0f;"
    )
    bad.kernels["add"] = KernelSpec(source='extern "C" __global__ void add(float* y, const float* x, const float* w) {' + body + "}")
    with gpu_lock():
        x = np.arange(32, dtype=np.float32)
        root = save_executable(
            tmp_path / "bundle",
            {"good": good, "bad": bad},
            bindings={name: _bindings(x) for name in ("good", "bad")},
            key={},
        )

        async def check():
            worker = NativeWorker(executable=executable)
            try:
                await worker.run_job({"op": "load", "root": str(root), "program": "bad"}, wall_timeout_s=30)
                old_pid = worker._proc.pid
                with pytest.raises(RuntimeError):
                    await worker.run_job(
                        {"op": "run", "warmup": 0, "iterations": 1, "capture": False, "outputs": {}},
                        wall_timeout_s=0.02 if fault == "deadline" else 30,
                    )
                assert worker._proc is None
                await worker.run_job({"op": "load", "root": str(root), "program": "good"}, wall_timeout_s=30)
                assert worker._proc.pid != old_pid
                path = tmp_path / "out.bin"
                await worker.run_job(
                    {"op": "run", "warmup": 0, "iterations": 1, "capture": False, "outputs": {"y": str(path)}},
                    wall_timeout_s=30,
                )
                np.testing.assert_array_equal(np.fromfile(path, np.float32), x * 4)
            finally:
                await worker.aclose()

        asyncio.run(check())


@pytest.mark.parametrize("static_count,dynamic_count", [(8192, 0), (0, 4096), (8192, 4096)], ids=["static", "dynamic", "mixed"])
def test_shared_memory_uses_cubin_static_and_plan_dynamic_storage(tmp_path, static_count, dynamic_count):
    executable = shutil.which("emmy-runtime-worker")
    if not executable:
        pytest.skip("build emmy-runtime-worker and add it to PATH")
    declarations, writes, reads = [], [], []
    for name, count, declaration in (
        ("fixed", static_count, f"__shared__ float fixed[{static_count}];"),
        ("extra", dynamic_count, "extern __shared__ float extra[];"),
    ):
        if count:
            declarations.append(declaration)
            writes.append(f"for (int i=threadIdx.x; i<{count}; i+=blockDim.x) {name}[i]=x[i%128];")
            reads.append(f"{name}[{count}-1-threadIdx.x]")
    source = 'extern "C" __global__ void shared(const float* x, float* y) {'
    source += "".join(declarations + writes) + "__syncthreads(); y[threadIdx.x]=" + "+".join(reads) + ";}"
    plan = ExecutionPlan(
        "cuda",
        ["x"],
        ["y"],
        [BufferSpec("x", (Dim(128),), F32, "input"), BufferSpec("y", (Dim(128),), F32, "output")],
        {},
        {},
        [LaunchSpec("shared", "shared", ("x", "y"), ((1,), (1,), (1,)), ((128,), (1,), (1,)), (static_count + dynamic_count) * 4, ())],
        {"shared": KernelSpec(source=source)},
    )
    values = np.arange(128, dtype=np.float32)
    expected = values[::-1] * len(reads)
    with gpu_lock():
        program = CompiledProgram.build_from_plan(plan, {"x": values})
        assert kernel_attributes("shared", plan.kernels["shared"])["shared_size_bytes"] == static_count * 4
        program.run_once()
        np.testing.assert_array_equal(program.outputs()["y"], expected)
        program.capture_program_graph()
        program.replay_program_graph()
        np.testing.assert_array_equal(program.outputs()["y"], expected)
        root = save_executable(tmp_path / "pack", {"shared": plan}, bindings={"shared": {"x": values.tobytes()}}, key={})

        async def check():
            worker = NativeWorker(executable=executable)
            try:
                await worker.run_job({"op": "load", "root": str(root), "program": "shared"}, wall_timeout_s=30)
                for capture in (False, True):
                    await worker.run_job(
                        {"op": "run", "warmup": 0, "iterations": 1, "capture": capture, "outputs": {"y": str(tmp_path / "out")}},
                        wall_timeout_s=30,
                    )
                    np.testing.assert_array_equal(np.fromfile(tmp_path / "out", np.float32), expected)
            finally:
                await worker.aclose()

        asyncio.run(check())


def test_a_lent_operand_swaps_without_waiting_for_queued_launches():
    """``alias_buffer(..., wait=False)`` leaves launches already queued on the old memory and points
    later ones at the new, with no host wait in between: three swaps and three runs are all queued
    before one synchronize, and each run reads the weights it was launched with."""
    import torch

    kernel = 'extern "C" __global__ void mul(float* y, const float* x, const float* w) { int i=threadIdx.x; y[i] = x[i]*w[i]; }'
    plan = ExecutionPlan(
        "cuda",
        ["x", "w"],
        ["y"],
        [BufferSpec("x", (Dim(32),), F32, "input"), BufferSpec("w", (Dim(32),), F32, "input"), BufferSpec("y", (Dim(32),), F32, "output")],
        {},
        {},
        [LaunchSpec("mul", "mul", ("y", "x", "w"), ((1,), (1,), (1,)), ((32,), (1,), (1,)), 0, ())],
        {"mul": KernelSpec(source=kernel)},
    )
    x = np.arange(32, dtype=np.float32)
    weights = [torch.full((32,), float(k + 2), device="cuda") for k in range(3)]
    with gpu_lock():
        program = CompiledProgram.build_from_plan(plan, {"x": x, "w": np.ones(32, dtype=np.float32)})
        torch.cuda.synchronize()
        outputs = []
        with program.on_stream(torch.cuda.current_stream()):
            for w in weights:
                program.alias_buffer("w", w, wait=False)
                program.run_once()
                outputs.append(program.output_prefix_device()["y"].clone())
        torch.cuda.synchronize()
    for w, y in zip(weights, outputs, strict=True):
        np.testing.assert_array_equal(y.cpu().numpy(), x * w.cpu().numpy())
