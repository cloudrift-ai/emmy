"""The Hopper warp-group GEMM's K loop and store.

Under ``STAGE`` ``/p2`` a ``wgmma`` drain leaves each chunk's group running while the next one is
issued: it waits for all but one group, and the ring slot the running group reads is refilled one
chunk later, past the barrier after the next chunk's issue. ``/p1`` waits every chunk out. At the
end, four adjacent accumulator cells are stored as one 16-byte row per lane, the quad trading its
column pairs through shuffles. Source shape on any machine; the answer on an H100.
"""

from __future__ import annotations

import numpy as np
import pytest

from emmy.commands.trace import graph_from_code
from emmy.compiler.backend.cuda.backend import CudaBackend
from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.expression import Literal
from emmy.compiler.ir.kernel.ir import Cond, CpAsyncCopy, CpAsyncWait, MbarrierArrive, RegStore, Sync, WgmmaWait
from emmy.compiler.ir.stmt import Body, StridedLoop
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import compile_with_kernels, device_compute_capability, requires_cuda, statements

_N128 = "wgmma_m64n128k16_f16_f32/f1x16/k4"
_HOPPER = Context.from_target((9, 0))

requires_hopper = pytest.mark.skipif(device_compute_capability() != (9, 0), reason="wgmma runs on sm_90 only")


def _code(n: int, k: int, epilogue: str = "") -> str:
    mm = f"torch.matmul(torch.randn((256, {k}), dtype=torch.float16), torch.randn(({k}, {n}), dtype=torch.float16))"
    return f"{epilogue}({mm})" if epilogue else mm


def _compile(code: str, pins: dict, ctx: Context):
    graph = graph_from_code(code)[0]
    with pinned_knobs({"RASTER": "gm8", "REDUCE": "", **pins}):
        return graph, Pipeline.build(CUDA_PASSES).run(graph, ctx=ctx)


def _kernel(code: str, pins: dict):
    graph = graph_from_code(code)[0]
    with pinned_knobs({"RASTER": "gm8", "REDUCE": "", **pins}):
        compiled, kernels = compile_with_kernels(graph, ctx=_HOPPER)
    (kernel,) = kernels.values()
    return compiled, kernel


def _chunk_loops(kernel) -> list[StridedLoop]:
    """The K chunk loops: 1024 in steps of a 64-wide chunk (the consumer's, and a producer band's own)."""
    return [loop for loop in statements(kernel, StridedLoop) if loop.axis.extent == Dim(1024) and loop.step == Literal(64, "int")]


def _waits(stmts) -> list[int]:
    """The ``wgmma`` group waits among ``stmts``, nested ones included, by how many groups each leaves running."""
    return [wait.group for wait in Body(tuple(stmts)).iter() if isinstance(wait, WgmmaWait)]


def test_p2_keeps_one_group_in_flight_and_refills_past_the_barrier() -> None:
    _, kernel = _kernel(_code(256, 1024), {"WORK": "w4x1", "TILE": _N128, "STAGE": "d4/smem-async/p2"})
    (chunk,) = _chunk_loops(kernel)
    loop = list(chunk.body)
    (drain,) = [i for i, stmt in enumerate(loop) if isinstance(stmt, Cond) and _waits([stmt])]
    guard = loop[drain]
    assert str(guard.cond) == f"{chunk.axis.name} + 64 < 1024", "the last chunk waits its group out inside the loop"
    assert (_waits(guard.body), _waits(guard.else_body)) == ([1], [0])
    barrier = next(i for i in range(drain, len(loop)) if isinstance(loop[i], Sync))
    refill = next(i for i, stmt in enumerate(loop) if isinstance(stmt, CpAsyncCopy))
    assert drain < barrier < refill, "the refill waits for the barrier"
    assert [wait.group for wait in loop if isinstance(wait, CpAsyncWait)] == [2], "one fewer chunk in flight at the wait (four slots)"
    assert len(statements(kernel, WgmmaWait)) == 2, "no wait after the loop"


def test_p1_waits_every_chunk_out() -> None:
    _, kernel = _kernel(_code(256, 1024), {"WORK": "w4x1", "TILE": _N128, "STAGE": "d4/smem-async"})
    (chunk,) = _chunk_loops(kernel)
    loop = list(chunk.body)
    assert 1 not in _waits(loop)
    refill = next(i for i, stmt in enumerate(loop) if isinstance(stmt, CpAsyncCopy))
    assert refill < next(i for i, stmt in enumerate(loop) if _waits([stmt]) == [0]), "the refill leads the chunk"


def test_the_producer_band_releases_the_previous_slot_under_p2() -> None:
    compiled, kernel = _kernel(_code(256, 1024), {"WORK": "w4x1+p1", "TILE": _N128, "STAGE": "d4/smem-tma/p2"})
    assert 1 in _waits(statements(kernel))
    # The release waits for a previous slot: from the second chunk on (``1 <= chunk / 64`` in the normal form).
    (release,) = [
        guard
        for guard in statements(kernel, Cond)
        if any(isinstance(stmt, MbarrierArrive) and stmt.mbar == "_mbar_empty" for stmt in guard.body)
    ]
    (arrive,) = release.body
    (consumer,) = [loop for loop in _chunk_loops(kernel) if release in loop.body]
    chunk = consumer.axis.name
    assert f"1 <= {chunk} / 64" in str(release.cond) and str(arrive.slot) == f"({chunk} / 64 + 3) % 4", (release.cond, arrive.slot)
    # CUDA spelling: moves to the PTX compiler's tests (the slot index is a shift and a mask)
    (source,) = [node.op.kernel_source for node in compiled.nodes.values() if isinstance(node.op, CudaOp)]
    assert "+ 3 & 3]" in source[source.index("mbarrier_arrive(&_mbar_empty") :].split("\n")[0]


def test_four_cells_store_one_16_byte_row_each_lane() -> None:
    compiled, kernel = _kernel(_code(256, 256, "torch.relu"), {"WORK": "w4x1", "TILE": _N128, "STAGE": "d2/smem-tma"})
    (grid,) = kernel.body
    stores = [stmt for stmt in grid.body if isinstance(stmt, RegStore)]
    assert [len(store.run) for store in stores] == [3] * (16 // 4), "the sixteen cells store as runs of four"
    # CUDA spelling: moves to the PTX compiler's tests (a run of four cells as one 16-byte row per lane)
    (source,) = [node.op.kernel_source for node in compiled.nodes.values() if isinstance(node.op, CudaOp)]
    rows = source[source.index("wgmma.wait_group.sync.aligned 0") :]
    assert rows.count("*reinterpret_cast<uint4*>") == 2 * 16 // 4, "two rows per group of four cells"
    assert rows.count("__shfl_xor_sync") == 3 * 2 * 16 // 4
    assert "__half2*>(&" not in rows


@requires_cuda
@requires_hopper
@pytest.mark.parametrize(
    ("n", "k", "pins", "epilogue"),
    [
        (256, 1024, {"WORK": "w4x1", "TILE": _N128, "STAGE": "d4/smem-async/p2"}, ""),
        (256, 1024, {"WORK": "w4x1", "TILE": _N128, "STAGE": "d8/smem-tma/p2"}, ""),
        (256, 192, {"WORK": "w4x1+p1", "TILE": _N128, "STAGE": "d2/smem-tma/p2"}, ""),
        (256, 1024, {"WORK": "w8x1", "TILE": _N128, "STAGE": "d4/smem-tma/p2"}, "torch.relu"),
        (256, 1024, {"WORK": "w4x1+p1", "TILE": _N128, "STAGE": "d4/smem-tma/p2"}, ""),
    ],
)
def test_the_warp_group_gemm_computes_the_right_answer(n: int, k: int, pins: dict, epilogue: str) -> None:
    graph, compiled = _compile(_code(n, k, epilogue), pins, Context.probe())
    rng = np.random.default_rng(0)
    a = rng.standard_normal((256, k)).astype(np.float16)
    b = rng.standard_normal((k, n)).astype(np.float16)
    result = CudaBackend().run(compiled, input_data=dict(zip(graph.inputs, (a, b), strict=True)))[0]
    (out,) = graph.outputs
    expected = a.astype(np.float32) @ b.astype(np.float32)
    if epilogue:
        expected = np.maximum(expected, 0)
    np.testing.assert_allclose(result.outputs[out].astype(np.float32), expected, rtol=2e-2, atol=5e-1)
