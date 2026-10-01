"""The existing generation wrapper preserves logical BF16 across device I/O."""

import numpy as np
import pytest

from emmy.compiler.backend.gpu_lock import gpu_lock
from emmy.compiler.backend.plan import BufferSpec, ExecutionPlan, KernelSpec, LaunchSpec
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import BF16, encode_bf16
from emmy.serving.gen_runner import _Program
from tests.compiler.helpers import requires_cuda

pytestmark = [requires_cuda, pytest.mark.xdist_group("cuda")]


def test_generation_device_wrapper_preserves_bf16_input_and_output():
    import torch

    from emmy.compiler.backend.cuda.program import CompiledProgram

    source = """
    #include <cuda_bf16.h>
    extern "C" __global__ void copy_bf16(__nv_bfloat16* y, const __nv_bfloat16* x) {
        if (threadIdx.x < 4) y[threadIdx.x] = x[threadIdx.x];
    }
    """
    plan = ExecutionPlan(
        "cuda",
        ["x"],
        ["y"],
        [BufferSpec("x", (Dim(4),), BF16, "input"), BufferSpec("y", (Dim(4),), BF16, "output")],
        {},
        {},
        [LaunchSpec("y", "copy_bf16", ("y", "x"), ((1,), (1,), (1,)), ((32,), (1,), (1,)), 0, ())],
        {"copy_bf16": KernelSpec(source=source, arch_specific=True)},
    )
    values = np.array([1.0, -2.0, 3.140625, -0.5], dtype=np.float32)
    with gpu_lock():
        compiled = CompiledProgram.build_from_plan(plan, {"x": encode_bf16(values)})
    runner = _Program(compiled, ["x"], ["y"])
    x = torch.tensor(values, device="cuda", dtype=torch.bfloat16)
    for expected in (x, x + 1):
        [got] = runner.run_device([expected])
        assert got.dtype == torch.bfloat16
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
