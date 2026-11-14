"""A cp.async GEMM ring at ``/p2`` carries its fragments across chunks: each chunk's first atom-K step is
loaded before the previous chunk's last mma, behind the chunk's one barrier."""

from __future__ import annotations

import numpy as np
import pytest

from emmy.commands.trace import graph_from_code
from emmy.compiler.backend.cuda.backend import CudaBackend
from emmy.compiler.context import Context
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.kernel.ir import LdmatrixLoad, MmaSyncPtx, Sync
from emmy.compiler.ir.stmt import Body, StridedLoop
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import compile_with_kernels, requires_cuda

_TILE = {"WORK": "w2x2", "TILE": "mma_m16n8k16_f16_f32/f4x4/k4", "RASTER": "gm8", "REDUCE": ""}


def _code(k: int) -> str:
    return f"torch.matmul(torch.randn((256, {k}), dtype=torch.float16), torch.randn(({k}, 256), dtype=torch.float16))"


def _split_at_chunk_loop(k: int, stage: str, ctx: Context) -> tuple[list, list]:
    """The kernel's statements ahead of the K chunk loop and from it on, nested ones included, in body order."""
    graph = graph_from_code(_code(k))[0]
    with pinned_knobs({**_TILE, "STAGE": stage}):
        _, kernels = compile_with_kernels(graph, ctx=ctx)
    ((grid,),) = [kernel.body for kernel in kernels.values()]
    at = next(i for i, stmt in enumerate(grid.body) if isinstance(stmt, StridedLoop) and stmt.axis.name == "_ks")
    return list(Body(grid.body[:at]).iter()), list(Body(grid.body[at:]).iter())


def _first_step_load(stmt) -> bool:
    return isinstance(stmt, LdmatrixLoad) and stmt.frag == "_a0_s0"


def test_a_carried_ring_loads_the_next_chunk_behind_its_one_barrier() -> None:
    """One barrier per chunk, and the first fragment load of the loop body reads the slot set the
    loop's head (or the previous iteration's last step) did not: the next step's buffer."""
    head, body = _split_at_chunk_loop(1024, "d3/smem-async/p2", Context.from_target((8, 0)))
    assert any(_first_step_load(stmt) for stmt in head), "the head loads the first chunk's first step"
    assert sum(isinstance(stmt, Sync) for stmt in body) == 1, "one barrier per chunk"
    after = body[next(i for i, stmt in enumerate(body) if isinstance(stmt, Sync)) :]
    load = next(i for i, stmt in enumerate(after) if _first_step_load(stmt))
    assert load < next(i for i, stmt in enumerate(after) if isinstance(stmt, MmaSyncPtx)), "the next chunk loads before this step's mma"


def test_a_two_slot_ring_keeps_the_ordinary_schedule() -> None:
    """At two slots the prefetch would target the slot about to be read, so there is nothing to carry."""
    _, body = _split_at_chunk_loop(1024, "d2/smem-async/p2", Context.from_target((8, 0)))
    assert sum(isinstance(stmt, Sync) for stmt in body) == 2


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
