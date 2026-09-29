"""An mma GEMM's output tile is stored through shared memory, then to global memory in 16-byte rows.

The plain epilogue is a burst of 4-byte stores per lane that throttled the A100's load/store queue at
the end of the K-loop; the staged store lands the fragments over the dead operand slabs and writes
whole rows. It is a perf transform: the stored values are the fragment stores' own."""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.context import Context
from emmy.compiler.dtype import F16
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import LinearOp
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from tests.compiler.helpers import requires_cuda, requires_sm

K16 = "mma_m16n8k16_f16_f32"


def _graph(m: int, n: int, k: int) -> Graph:
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("a", (m, k), dtype=F16), node_id="a")
    graph.add_node(InputOp(), [], Tensor("b", (n, k), dtype=F16), node_id="b")
    graph.add_node(LinearOp(), ["a", "b"], Tensor("c", (m, n), dtype=F16), node_id="c")
    graph.inputs, graph.outputs = ["a", "b"], ["c"]
    return graph


def _pin(monkeypatch, work: str, tile: str, *, out: bool = True) -> None:
    monkeypatch.setenv("EMMY_WORK", work)
    monkeypatch.setenv("EMMY_TILE", f"{K16}/{tile}")
    monkeypatch.setenv("EMMY_STAGE", "d2/smem-async/out" if out else "d2/smem-async")
    monkeypatch.setenv("EMMY_REDUCE", "")
    monkeypatch.setenv("EMMY_RASTER", "")


def _source(graph: Graph) -> str:
    lowered = Pipeline.build(CUDA_PASSES).run(graph, ctx=Context(compute_capability=(8, 0)))
    (src,) = [n.op.kernel_source for n in lowered.nodes.values() if getattr(n.op, "kernel_source", None)]
    return src


def test_a_wide_tile_stores_through_the_operand_slabs(monkeypatch) -> None:
    """A 128×64 tile: the fragments write the staged tile laid over an operand slab, and the CTA
    stores it out as 16-byte rows once the K-loop's copies have drained."""
    _pin(monkeypatch, "w2x2", "f4x4/k2")
    src = _source(_graph(256, 256, 128))
    assert "_c_smem = reinterpret_cast<__half*>" in src
    assert "emmy_cp_async_wait<0>();" in src
    assert "*reinterpret_cast<uint4*>(&c[" in src
    assert "&c[" not in src.replace("*reinterpret_cast<uint4*>(&c[", "")


def test_a_stage_without_out_keeps_the_direct_stores(monkeypatch) -> None:
    _pin(monkeypatch, "w2x2", "f4x4/k2", out=False)
    src = _source(_graph(256, 256, 128))
    assert "_c_smem" not in src and "uint4" not in src


def test_the_stage_codec_spells_out_last() -> None:
    from emmy.compiler.ir.schedule import Stage  # noqa: PLC0415

    assert Stage.parse("d4/smem-async/p2/out").spell() == "d4/smem-async/p2/out"
    assert Stage.parse("d4/smem-async/p2/out").out and not Stage.parse("d4/smem-async/p2").out
    with pytest.raises(ValueError):
        Stage.parse("d1/reg/out")


def test_a_tile_narrower_than_the_swizzle_row_stores_directly(monkeypatch) -> None:
    """A 32-column tile has no conflict-free 128-byte swizzle row, so it keeps its direct stores."""
    _pin(monkeypatch, "w2x2", "f2x2/k2")
    src = _source(_graph(256, 256, 128))
    assert "_c_smem" not in src


@requires_cuda
@requires_sm(8)
@pytest.mark.parametrize(("work", "tile"), [("w2x2", "f4x4/k2"), ("w2x2", "f2x4/k2"), ("w4x2", "f2x8/k2")])
def test_the_staged_store_is_bit_identical(monkeypatch, work, tile) -> None:
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    rng = np.random.default_rng(0)
    feed = {"a": (rng.standard_normal((256, 256)) * 0.1).astype(np.float16), "b": (rng.standard_normal((512, 256)) * 0.1).astype(np.float16)}
    outs = {}
    for out in (False, True):
        _pin(monkeypatch, work, tile, out=out)
        be = CudaBackend()
        compiled = be.compile(_graph(256, 512, 256))
        src = "\n".join(n.op.kernel_source for n in compiled.nodes.values() if getattr(n.op, "kernel_source", None))
        assert ("_c_smem" in src) == out
        outs[out] = np.asarray(be.run(compiled, input_data=feed)[0].outputs["c"])
    np.testing.assert_array_equal(outs[True], outs[False])
