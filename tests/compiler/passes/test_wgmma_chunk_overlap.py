"""Independent carried row folds may run while the value product is in flight."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from emmy.commands.trace import graph_from_code
from emmy.compiler.backend.cuda.backend import CudaBackend
from emmy.compiler.context import Context
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import requires_cuda, requires_sm90


def _compile(keys, stage, ctx, heads=2):
    query = f"torch.randn(1, {heads}, 128, 128, dtype=torch.float16)"
    streamed = f"torch.randn(1, {heads}, {keys}, 128, dtype=torch.float16)"
    graph = graph_from_code(f"F.scaled_dot_product_attention({query}, {streamed}, {streamed}, is_causal=True)")[0]
    pins = {
        "PLACE": "fuse",
        "FAST_MATH": False,
        "WORK": "w4x1",
        "TILE@map.1/twist": "wgmma_m64n128k16_f16_f32/f1x16/k4",
        "TILE@map.1/twist.1/inner": "wgmma_m64n64k16_f16_f32/f1x8/k4",
        "REDUCE@map.1/twist": "",
        "REDUCE@map.1/twist.1/inner": "",
        "STAGE@map.1/twist": stage,
        "STAGE@map.1/twist.1/inner": stage,
    }
    with pinned_knobs(pins):
        compiled = Pipeline.build(CUDA_PASSES).run(graph, ctx=ctx)
    return graph, compiled


@pytest.mark.parametrize("keys", [64, 128, 192])
@pytest.mark.parametrize("stage", ["d1/smem-async", "d2/smem-async"])
def test_row_folds_finish_before_the_value_slot_is_released(keys, stage):
    _, compiled = _compile(keys, stage, Context.from_target((9, 0)))
    (source,) = [node.op.kernel_source for node in compiled.nodes.values() if isinstance(node.op, CudaOp)]
    loop = source[source.index("for (int a2__ck") :]
    pending = loop[loop.rindex("wgmma.commit_group") : loop.rindex("wgmma.wait_group")]
    assert "__shfl_xor_sync" in pending, "the independent row combine runs under the value product"
    assert "_c0_" not in pending and "_a0_" not in pending, "neither pending accumulator nor probability operand is touched"
    release = loop[loop.rindex("wgmma.wait_group") :]
    assert release.index("__syncthreads()") < release.index("_c0_"), "the value product settles before slab release and stores"


@pytest.mark.parametrize(("heads", "bounded"), [(2, False), (72, True)])
def test_the_row_overlap_preserves_existing_causal_bounds(heads, bounded):
    ctx = replace(Context.from_target((9, 0)), sm_count=132)
    _, compiled = _compile(192, "d2/smem-async", ctx, heads=heads)
    (source,) = [node.op.kernel_source for node in compiled.nodes.values() if isinstance(node.op, CudaOp)]
    assert ("a2__ck_end" in source) == bounded


@requires_sm90
@requires_cuda
@pytest.mark.parametrize(("stage", "heads"), [("d1/smem-async", 2), ("d2/smem-async", 2), ("d2/smem-async", 72), ("d2/smem-tma", 2)])
def test_overlapped_row_folds_preserve_three_chunk_causal_attention(stage, heads):
    graph, compiled = _compile(192, stage, Context.probe(), heads=heads)
    backend = CudaBackend()
    for seed in range(3):
        rng = np.random.default_rng(seed)
        shapes = [(1, heads, 128, 128), (1, heads, 192, 128), (1, heads, 192, 128)]
        feed = {name: rng.standard_normal(shape).astype(np.float16) for name, shape in zip(graph.inputs, shapes, strict=True)}
        operands = [torch.as_tensor(feed[name], device="cuda") for name in graph.inputs]
        expected = torch.nn.functional.scaled_dot_product_attention(*operands, is_causal=True).cpu().numpy()
        actual = backend.run(compiled, input_data=feed)[0].outputs[graph.outputs[0]]
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
