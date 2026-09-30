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
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import device_compute_capability, requires_cuda

_N128 = "wgmma_m64n128k16_f16_f32/f1x16/k4"
_HOPPER = Context.from_target((9, 0))

requires_hopper = pytest.mark.skipif(device_compute_capability() != (9, 0), reason="wgmma runs on sm_90 only")


def _code(n: int, k: int, epilogue: str = "") -> str:
    mm = f"torch.matmul(torch.randn((256, {k}), dtype=torch.float16), torch.randn(({k}, {n}), dtype=torch.float16))"
    return f"{epilogue}({mm})" if epilogue else mm


def _compile(code: str, pins: dict, ctx: Context):
    graph = graph_from_code(code)[0]
    with pinned_knobs({"RASTER": "gm8", **pins}):
        return graph, Pipeline.build(CUDA_PASSES).run(graph, ctx=ctx)


def _source(code: str, pins: dict) -> str:
    _, lowered = _compile(code, pins, _HOPPER)
    (source,) = [node.op.kernel_source for node in lowered.nodes.values() if isinstance(node.op, CudaOp)]
    return source


def test_p2_keeps_one_group_in_flight_and_refills_past_the_barrier() -> None:
    src = _source(_code(256, 1024), {"WORK": "w4x1", "TILE": _N128, "STAGE": "d4/smem-async/p2"})
    loop = src[src.index("for (int _ks") : src.index("uint4")]
    one, last = loop.index("wgmma.wait_group.sync.aligned 1"), loop.index("wgmma.wait_group.sync.aligned 0")
    assert "_ks + 64 < 1024" in loop[:one] and one < last, "the last chunk waits its group out inside the loop"
    assert last < loop.index("__syncthreads()", last) < loop.index("emmy_cp_async_cg"), "the refill waits for the barrier"
    assert "emmy_cp_async_wait<2>" in loop, "one fewer chunk in flight at the wait (four slots)"
    assert loop.count("wgmma.wait_group") == 2, "no wait after the loop"


def test_p1_waits_every_chunk_out() -> None:
    src = _source(_code(256, 1024), {"WORK": "w4x1", "TILE": _N128, "STAGE": "d4/smem-async"})
    loop = src[src.index("for (int _ks") :]
    assert "wgmma.wait_group.sync.aligned 1" not in loop
    assert loop.index("emmy_cp_async_cg") < loop.index("wgmma.wait_group.sync.aligned 0"), "the refill leads the chunk"


def test_the_producer_band_releases_the_previous_slot_under_p2() -> None:
    src = _source(_code(256, 1024), {"WORK": "w4x1+p1", "TILE": _N128, "STAGE": "d4/smem-tma/p2"})
    assert "wgmma.wait_group.sync.aligned 1" in src
    release = src[src.index("mbarrier_arrive(&_mbar_empty") - 200 : src.index("mbarrier_arrive(&_mbar_empty")]
    assert ">= 1" in release and "+ 3) % 4" in src[src.index("mbarrier_arrive(&_mbar_empty") :].split("\n")[0]


def test_four_cells_store_one_16_byte_row_each_lane() -> None:
    src = _source(_code(256, 256, "torch.relu"), {"WORK": "w4x1", "TILE": _N128, "STAGE": "d2/smem-tma"})
    stores = src[src.index("wgmma.wait_group.sync.aligned 0") :]
    assert stores.count("*reinterpret_cast<uint4*>") == 2 * 16 // 4, "two rows per group of four cells"
    assert stores.count("__shfl_xor_sync") == 3 * 2 * 16 // 4
    assert "__half2*>(&" not in stores


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
