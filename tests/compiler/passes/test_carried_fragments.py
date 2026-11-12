"""A cp.async GEMM ring at ``/p2`` carries its fragments across chunks: each chunk's first atom-K step is
loaded before the previous chunk's last mma, behind the chunk's one barrier."""

from __future__ import annotations

import re

import numpy as np
import pytest

from emmy.commands.trace import graph_from_code
from emmy.compiler.backend.cuda.backend import CudaBackend
from emmy.compiler.context import Context
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import requires_cuda

_TILE = {"WORK": "w2x2", "TILE": "mma_m16n8k16_f16_f32/f4x4/k4", "RASTER": "gm8", "REDUCE": ""}


def _code(k: int) -> str:
    return f"torch.matmul(torch.randn((256, {k}), dtype=torch.float16), torch.randn(({k}, 256), dtype=torch.float16))"


def _source(k: int, stage: str, ctx: Context) -> str:
    graph = graph_from_code(_code(k))[0]
    with pinned_knobs({**_TILE, "STAGE": stage}):
        lowered = Pipeline.build(CUDA_PASSES).run(graph, ctx=ctx)
    (source,) = [node.op.kernel_source for node in lowered.nodes.values() if isinstance(node.op, CudaOp)]
    return source


def _chunk_loop(source: str) -> int:
    """Where the K chunk loop starts: the one loop over the whole K extent of 1024 in steps of a 64-wide chunk."""
    return re.search(r"for \(int (\w+) = 0; \1 < 1024; \1 \+= 64\)", source).start()


def test_a_carried_ring_loads_the_next_chunk_behind_its_one_barrier() -> None:
    """One barrier per chunk, and the first fragment load of the loop body reads the slot set the
    loop's head (or the previous iteration's last step) did not: the next step's buffer."""
    loop = _source(1024, "d3/smem-async/p2", Context.from_target((8, 0)))
    head, body = loop[: _chunk_loop(loop)], loop[_chunk_loop(loop) :]
    first_step = set(re.findall(r"emmy_ldmatrix_x4\((\w+),", head))
    assert first_step, "the head loads the first chunk's first step"
    assert body.count("__syncthreads()") == 1, "one barrier per chunk"
    after = body[body.index("__syncthreads()") :]
    (fragment,) = re.findall(r"emmy_ldmatrix_x4\((\w+),", after)[:1]
    assert fragment in first_step, "past the barrier the loop loads the slot the head loaded: the next chunk's first step"
    assert after.index("emmy_ldmatrix_x4(") < after.index("emmy_mma_m16n8k16_f16_f32("), "the next chunk loads before this step's mma"


def test_a_two_slot_ring_keeps_the_ordinary_schedule() -> None:
    """At two slots the prefetch would target the slot about to be read, so there is nothing to carry."""
    loop = _source(1024, "d2/smem-async/p2", Context.from_target((8, 0)))
    assert loop[_chunk_loop(loop) :].count("__syncthreads()") == 2


@requires_cuda
@pytest.mark.parametrize("k", [192, 1024, 3072])  # three chunks (the whole ring), 16 (it wraps mid-cycle), 48
def test_a_carried_ring_computes_the_right_answer(k: int) -> None:
    if not Context.probe().has_cp_async:
        pytest.skip("a cp.async ring needs sm_80 or newer")
    graph = graph_from_code(_code(k))[0]
    rng = np.random.default_rng(0)
    a = rng.standard_normal((256, k)).astype(np.float16)
    b = rng.standard_normal((k, 256)).astype(np.float16)
    with pinned_knobs({**_TILE, "STAGE": "d3/smem-async/p2"}):
        compiled = Pipeline.build(CUDA_PASSES).run(graph, ctx=Context.probe())
    result = CudaBackend().run(compiled, input_data=dict(zip(graph.inputs, (a, b), strict=True)))[0]
    (out,) = graph.outputs
    expected = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(result.outputs[out].astype(np.float32), expected, rtol=2e-2, atol=5e-1)


def test_a_scalar_staged_gemv_compiles_on_a_cp_async_ring() -> None:
    """The scalar drain carries no fragments: a staged GEMV on a cp.async ring keeps its own schedule."""
    graph = graph_from_code("torch.matmul(torch.randn((1, 1024), dtype=torch.float16), torch.randn((1024, 1024), dtype=torch.float16))")[0]
    with pinned_knobs({"WORK": "t256", "TILE": "f1", "STAGE": "d3/smem-async"}):
        lowered = Pipeline.build(CUDA_PASSES).run(graph, ctx=Context.from_target((8, 0)))
    assert [node for node in lowered.nodes.values() if isinstance(node.op, CudaOp)]
