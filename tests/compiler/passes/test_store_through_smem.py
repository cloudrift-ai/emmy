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


def test_a_masked_m_edge_bounds_the_row_copy(monkeypatch) -> None:
    """A 96-row tile over 256 rows overhangs the output: the fragment stores keep their row guard and
    the row copy stops at the output's last row."""
    _pin(monkeypatch, "w2x2", "f3x8/k2")
    src = _source(_graph(256, 256, 128))
    assert "_c_smem" in src
    assert "_r < 256) *reinterpret_cast<uint4*>(&c[" in src
    # The A rows past the edge are zero-filled, not re-read from the edge's last row.
    assert "emmy_cp_async_cg_z(&_a_smem[" in src and "emmy_cp_async_cg_z(&_b_smem[" not in src


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


def test_a_tile_narrower_than_the_swizzle_row_refuses_out(monkeypatch) -> None:
    """A 32-column tile has no conflict-free 128-byte swizzle row: ``out`` is not offered there, so a
    pin asking for it is refused rather than silently building the direct stores."""
    _pin(monkeypatch, "w2x2", "f2x2/k2")
    with pytest.raises(ValueError, match="STAGE pin 'd2/smem-async/out' does not resolve"):
        _source(_graph(256, 256, 128))


@requires_cuda
@requires_sm(8)
@pytest.mark.parametrize(("work", "tile"), [("w2x2", "f4x4/k2"), ("w2x2", "f2x4/k2"), ("w4x2", "f2x8/k2"), ("w2x2", "f3x8/k2")])
def test_the_staged_store_is_bit_identical(monkeypatch, work, tile) -> None:
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    rng = np.random.default_rng(0)
    feed = {
        "a": (rng.standard_normal((256, 256)) * 0.1).astype(np.float16),
        "b": (rng.standard_normal((512, 256)) * 0.1).astype(np.float16),
    }
    outs = {}
    for out in (False, True):
        _pin(monkeypatch, work, tile, out=out)
        be = CudaBackend()
        compiled = be.compile(_graph(256, 512, 256))
        src = "\n".join(n.op.kernel_source for n in compiled.nodes.values() if getattr(n.op, "kernel_source", None))
        assert ("_c_smem" in src) == out
        outs[out] = np.asarray(be.run(compiled, input_data=feed)[0].outputs["c"])
    np.testing.assert_array_equal(outs[True], outs[False])


@pytest.mark.parametrize(
    ("work", "tile", "fits"),
    [("w2x2", f"{K16}/f4x4/k2", True), ("w2x2", f"{K16}/f2x2/k2", False), ("w2x2", f"{K16}/f2x3/k2", False)],
)
def test_out_is_offered_only_where_the_staged_store_applies(work, tile, fits) -> None:
    """A 32-column or non-power-of-two tile keeps its direct stores, so ``out`` there would only
    build the plain kernel again under a second schedule row."""
    from emmy.compiler.ir.schedule import Tile, Work  # noqa: PLC0415
    from emmy.compiler.ir.schedule.classic.refusals import _staged_store_fits  # noqa: PLC0415

    assert _staged_store_fits(Tile.parse(tile, Work.parse(work))) is fits


def test_a_rewrite_keeps_the_zero_fill_predicate_and_the_staged_store_geometry() -> None:
    """Coordinate substitution reaches a copy's ``valid`` predicate and a staged store's base and
    bound, and keeps everything else."""
    from emmy.compiler.ir.expr import BinaryExpr, Literal, Var  # noqa: PLC0415
    from emmy.compiler.ir.kernel.ir import CpAsyncCopy, SmemTileStore  # noqa: PLC0415
    from emmy.compiler.ir.sigma import Sigma  # noqa: PLC0415
    from emmy.compiler.ir.stmt.passes import rewrite  # noqa: PLC0415

    sigma = Sigma({"r": Literal(5, "int")})
    copy = CpAsyncCopy(
        smem="s", smem_index=(Var("r"),), src="a", src_index=(Var("r"),), nbytes=16, valid=BinaryExpr("<", Var("r"), Literal(8, "int"))
    )
    assert rewrite(copy, lambda n: n, sigma).valid.pretty() == BinaryExpr("<", Literal(5, "int"), Literal(8, "int")).pretty()
    store = SmemTileStore(src="t", dst="c", base=(Var("r"), Literal(0, "int")), rows=96, cols=128, ldm=256, threads=128, bound=Var("r"))
    out = rewrite(store, lambda n: n, sigma)
    assert out.base[0] == Literal(5, "int") and out.bound == Literal(5, "int") and out.rows == 96 and out.dst == "c"
