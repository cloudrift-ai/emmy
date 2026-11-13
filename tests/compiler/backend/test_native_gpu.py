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
from tests.compiler.helpers import device_compute_capability, kernel_ptx, requires_cuda, requires_sm

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
                source=kernel_ptx(
                    "add",
                    """
                    Grid(axes=(i in 0..32), decode=(thread_idx.x), block_threads=32):
                        f32 before = load y[i]
                        f32 a = load x[i]
                        f32 scale = load w[0]
                        f32 product = a * scale
                        f32 after = before + product
                        f32 y[i] = after
                    """,
                    (("y", "output", "f32"), ("x", "input", "f32"), ("w", "input", "f32")),
                    threads=32,
                )
            )
        },
    )


def _module(name: str, params: tuple[str, ...], code: str, declarations: str = "") -> str:
    """A hand-written PTX kernel ``name`` for the live card: its pointer ``params`` loaded into ``%<param>`` as global
    addresses, the thread index in ``%t`` and its byte offset in ``%t4``, then ``code``."""
    major, minor = device_compute_capability()
    loads = "".join(f"\tld.param.u64 %{p}, [{p}_param];\n\tcvta.to.global.u64 %{p}, %{p};\n" for p in params)
    return (
        f".version {'8.7' if major >= 10 else '7.8'}\n.target sm_{major}{minor}\n.address_size 64\n{declarations}"
        f".visible .entry {name}({', '.join(f'.param .u64 {p}_param' for p in params)})\n{{\n"
        "\t.reg .pred %p;\n\t.reg .b32 %t, %i, %s;\n\t.reg .f32 %a, %b, %c;\n\t.reg .b64 %t4, %at, %clock, %start;\n"
        f"\t.reg .b64 {', '.join(f'%{p}' for p in params)};\n{loads}"
        "\tmov.u32 %t, %tid.x;\n\tmul.wide.u32 %t4, %t, 4;\n"
        f"{code}\tret;\n}}\n"
    )


def _bindings(x):
    return {"x": x.tobytes(), "w": np.float32(2).tobytes()}


@requires_sm(9, 0)
def test_first_program_graph_capture_prepares_tma_descriptors():
    # Thread 0 copies the 16 x 16 tile through the tensor map into a slab; every thread waits for the copy.
    source = kernel_ptx(
        "tma_copy",
        """
        Grid(axes=(t in 0..256), decode=(thread_idx.x), block_threads=256):
            Smem(name=tile, extents=(1, 16, 16), dtype=f32, align=128)
            Smem(name=bar, extents=(1), dtype=u64)
            TmaDescriptor(name=x_desc, src_buf=x, src_shape=(16, 16), box_extents=(16, 16))
            if t == 0:
                MbarrierInit(mbar=bar, count=1, slot=0)
            Sync
            if t == 0:
                MbarrierArriveExpectTx(mbar=bar, bytes_=1024, slot=0)
                TmaLoad(smem=tile, smem_index=(0, 0, 0), desc=x_desc, coords=(0, 0), mbar=bar, mbar_slot=0)
            MbarrierWait(mbar=bar, phase=0, slot=0)
            f32 v = load tile[t]
            f32 u = v + 1.0
            f32 y[t] = u
        """,
        (("y", "output", "f32"), ("x_desc", "descriptor", None)),
        threads=256,
        arch_specific=True,
    )
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
        root = save_executable(tmp_path / "bundle", {"test": plan}, bindings={"test": _bindings(x)}, key={}, reads={"test": frozenset()})
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
    # A kernel that spins for a second of clocks, or one that stores through a null address.
    code = (
        "\tmov.u64 %start, %clock64;\nWAIT:\n\tmov.u64 %clock, %clock64;\n\tsub.u64 %clock, %clock, %start;\n"
        "\tsetp.lt.u64 %p, %clock, 1000000000;\n\t@%p bra WAIT;\n"
        "\tadd.u64 %at, %x, %t4;\n\tld.global.f32 %a, [%at];\n\tadd.u64 %at, %y, %t4;\n\tst.global.f32 [%at], %a;\n"
        if fault == "deadline"
        else "\tmov.u64 %at, 0;\n\tst.volatile.global.f32 [%at], 0f3F800000;\n"
    )
    bad.kernels["add"] = KernelSpec(source=_module("add", ("y", "x", "w"), code))
    with gpu_lock():
        x = np.arange(32, dtype=np.float32)
        root = save_executable(
            tmp_path / "bundle",
            {"good": good, "bad": bad},
            bindings={name: _bindings(x) for name in ("good", "bad")},
            key={},
            reads={"good": frozenset(), "bad": frozenset()},
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
    # Each array filled from ``x`` by a strided loop, then every thread sums its element of each, counted from the end.
    declarations, code, reads = "", "", []
    for name, count, declaration in (
        ("fixed", static_count, f".shared .align 4 .f32 fixed[{static_count}];\n"),
        ("extra", dynamic_count, ".extern .shared .align 4 .f32 extra[];\n"),
    ):
        if count:
            declarations += declaration
            code += (
                f"\tmov.u32 %i, %t;\nFILL_{name}:\n\tand.b32 %s, %i, 127;\n\tmul.wide.u32 %at, %s, 4;\n"
                f"\tadd.u64 %at, %x, %at;\n\tld.global.f32 %a, [%at];\n\tmov.u32 %s, {name};\n\tmad.lo.u32 %s, %i, 4, %s;\n"
                f"\tst.shared.f32 [%s], %a;\n\tadd.u32 %i, %i, 128;\n\tsetp.lt.u32 %p, %i, {count};\n\t@%p bra FILL_{name};\n"
            )
            reads.append((name, count))
    code += "\tbar.sync 0;\n\tmov.f32 %c, 0f00000000;\n"
    for name, count in reads:
        code += (
            f"\tsub.u32 %i, {count - 1}, %t;\n\tmov.u32 %s, {name};\n\tmad.lo.u32 %s, %i, 4, %s;\n"
            "\tld.shared.f32 %b, [%s];\n\tadd.f32 %c, %c, %b;\n"
        )
    code += "\tadd.u64 %at, %y, %t4;\n\tst.global.f32 [%at], %c;\n"
    source = _module("shared", ("x", "y"), code, declarations)
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
        root = save_executable(
            tmp_path / "pack", {"shared": plan}, bindings={"shared": {"x": values.tobytes()}}, key={}, reads={"shared": frozenset()}
        )

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

    kernel = _module(
        "mul",
        ("y", "x", "w"),
        "\tadd.u64 %at, %x, %t4;\n\tld.global.f32 %a, [%at];\n\tadd.u64 %at, %w, %t4;\n\tld.global.f32 %b, [%at];\n"
        "\tmul.rn.f32 %c, %a, %b;\n\tadd.u64 %at, %y, %t4;\n\tst.global.f32 [%at], %c;\n",
    )
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
