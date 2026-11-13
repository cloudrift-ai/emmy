"""A cooperative matrix-vector product reads each lane's run of fp8 weight bytes as one vector load.

At one row a product streams its weight once, so the weight load is the kernel. On a ``coop/v<n>`` band each lane
reads ``n`` adjacent bytes of a K-contiguous weight row: the run reads as one word, each byte decodes straight to
f16, every element of the run reads its block scale at the run's own index, and the run carries no bounds mask
when ``n`` divides K (a mask spells every index as a modulo, which no vector load can cover).
"""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.backend.cuda.backend import CudaBackend
from emmy.compiler.context import Context
from emmy.compiler.dtype import F16, decode_f8
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expression import CastExpression
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Assign, Load, parse_body
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import compile_with_kernels, device_compute_capability, requires_cuda, statements, text

# K divides into 16-byte runs but not into the 128-lane band's 2048-element stride, so the band ends mid-row.
N, K = 256, 3072


def _graph() -> Graph:
    """``out[n] = Σ_k x[k] · f16(scale[n/128, k/128] · decode(w_bits[n, k]))``: a block-scaled fp8 weight's decode
    cone fused into the product, as a width-1 projection reads it."""
    body = parse_body(
        f"""
        for n in 0..{N}:
            for k in 0..{K}:
                wb = load w_bits[n, k]
                f16 wd = from_f8e4m3(wb)
                ws = load w_scale[n / 128, k / 128]
                f16 w = multiply(ws, wd)
                xv = load x[k]
                f32 p = multiply(xv, w)
                sum <- add(sum, p) over k
            out[n] = sum
        """
    )
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("w_bits", (N, K), "f8e4m3"), node_id="w_bits")
    graph.add_node(InputOp(), [], Tensor("w_scale", (N // 128, K // 128), "f32"), node_id="w_scale")
    graph.add_node(InputOp(), [], Tensor("x", (K,), "f16"), node_id="x")
    graph.add_node(
        LoopOp(body=body, name="k_fp8_matvec"),
        ["w_bits", "w_scale", "x"],
        Tensor("out", (N,), "f32"),
        node_id="out",
    )
    graph.inputs, graph.outputs = ["w_bits", "w_scale", "x"], ["out"]
    return graph


def _compiled(run: int, cc=(7, 0)):
    with pinned_knobs({"PLACE": "fuse", "WORK": "t128", "REDUCE": f"coop/v{run}", "TILE": "", "STAGE": ""}):
        return Pipeline.build(CUDA_PASSES).run(_graph(), ctx=Context.from_target(cc))


def _kernel(run: int):
    with pinned_knobs({"PLACE": "fuse", "WORK": "t128", "REDUCE": f"coop/v{run}", "TILE": "", "STAGE": ""}):
        _, kernels = compile_with_kernels(_graph(), ctx=Context.from_target((7, 0)))
    (kernel,) = kernels.values()
    return kernel


@pytest.mark.parametrize("run", [4, 8, 16])
def test_a_lane_reads_its_run_of_weight_bytes_as_one_word(run):
    kernel = _kernel(run)
    weights = [load for load in statements(kernel, Load) if load.input == "w_bits"]
    assert weights and all(load.width == run for load in weights), "one vector load per run, no scalar byte load left"
    assert f"% {K}" not in text(kernel), "a run that divides K needs no bounds mask"
    scales = {str(load.index) for load in statements(kernel, Load) if load.input == "w_scale"}
    assert len(scales) == len(weights), "a run inside one 128-wide block reads its scale at one index"


def test_the_decode_converts_each_byte_straight_to_f16():
    kernel = _kernel(16)
    weights = {name for load in statements(kernel, Load) if load.input == "w_bits" for name in load.names}
    decodes = [stmt for stmt in statements(kernel, Assign) if set(stmt.deps()) & weights]
    assert len(decodes) == len(weights) and all(stmt.dtype == F16 and isinstance(stmt.value, CastExpression) for stmt in decodes)


@requires_cuda
@pytest.mark.parametrize("run", [4, 16])
def test_the_run_matches_an_independent_reference(run):
    rng = np.random.default_rng(5)
    bits = rng.integers(0, 256, size=(N, K), dtype=np.uint8)
    bits[(bits & 0x7F) == 0x7F] = 0x38  # no NaN codes: the reference sum would be NaN either way
    scale = rng.uniform(0.5, 2.0, size=(N // 128, K // 128)).astype(np.float32)
    x = rng.standard_normal(K).astype(np.float16)
    weight = (np.repeat(np.repeat(scale, 128, 0), 128, 1) * decode_f8(bits, "f8e4m3").astype(np.float32)).astype(np.float16)
    expected = weight.astype(np.float32) @ x.astype(np.float32)
    result, _ = CudaBackend().run(_compiled(run, device_compute_capability()), input_data={"w_bits": bits, "w_scale": scale, "x": x})
    np.testing.assert_allclose(result.outputs["out"], expected, rtol=1e-3, atol=1e-2)
