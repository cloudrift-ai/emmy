"""A tile that overhangs its operand's M edge zero-fills the rows past it.

The cp.async fill clamps its source row in bounds; without a predicate every clamped row re-read the
edge's last row, and on the A100 a 96-row tile over 512 rows spent 23 µs on a 16 µs GEMM contending
for that one row. The predicated copy writes zeros and reads nothing."""

from __future__ import annotations

import numpy as np

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


def _pin(monkeypatch, work: str, tile: str) -> None:
    monkeypatch.setenv("EMMY_WORK", work)
    monkeypatch.setenv("EMMY_TILE", f"{K16}/{tile}")
    monkeypatch.setenv("EMMY_STAGE", "d2/smem-async")
    monkeypatch.setenv("EMMY_REDUCE", "")
    monkeypatch.setenv("EMMY_RASTER", "")


def test_a_masked_m_edge_zero_fills_the_rows_past_it(monkeypatch) -> None:
    """A 96-row tile over 256 rows overhangs A: its rows past the edge are zero-filled, not re-read
    from the edge's last row; B has no masked edge and copies unpredicated."""
    _pin(monkeypatch, "w2x2", "f3x8/k2")
    lowered = Pipeline.build(CUDA_PASSES).run(_graph(256, 256, 128), ctx=Context(compute_capability=(8, 0)))
    (src,) = [n.op.kernel_source for n in lowered.nodes.values() if getattr(n.op, "kernel_source", None)]
    assert "emmy_cp_async_cg_z(&_a_smem[" in src and "emmy_cp_async_cg_z(&_b_smem[" not in src


def test_a_rewrite_keeps_the_zero_fill_predicate() -> None:
    """Coordinate substitution reaches a copy's ``valid`` predicate."""
    from emmy.compiler.ir.expr import BinaryExpr, Literal, Var  # noqa: PLC0415
    from emmy.compiler.ir.kernel.ir import CpAsyncCopy  # noqa: PLC0415
    from emmy.compiler.ir.sigma import Sigma  # noqa: PLC0415
    from emmy.compiler.ir.stmt.passes import rewrite  # noqa: PLC0415

    sigma = Sigma({"r": Literal(5, "int")})
    copy = CpAsyncCopy(
        smem="s", smem_index=(Var("r"),), src="a", src_index=(Var("r"),), nbytes=16, valid=BinaryExpr("<", Var("r"), Literal(8, "int"))
    )
    assert rewrite(copy, lambda n: n, sigma).valid.pretty() == BinaryExpr("<", Literal(5, "int"), Literal(8, "int")).pretty()


@requires_cuda
@requires_sm(8)
def test_a_masked_m_edge_computes_the_right_answer(monkeypatch) -> None:
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    _pin(monkeypatch, "w2x2", "f3x8/k2")
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((256, 256)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((512, 256)) * 0.1).astype(np.float16)
    be = CudaBackend()
    got = np.asarray(be.run(be.compile(_graph(256, 512, 256)), input_data={"a": a, "b": b})[0].outputs["c"], dtype=np.float32)
    want = a.astype(np.float32) @ b.astype(np.float32).T
    np.testing.assert_allclose(got, want, atol=2e-2, rtol=2e-2)
