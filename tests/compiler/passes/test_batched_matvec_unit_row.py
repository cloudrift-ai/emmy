"""A per-batch matrix-vector product can use a unit MMA row."""

import numpy as np
import pytest

from emmy.compiler.backend.cuda.backend import CudaBackend
from emmy.compiler.context import Context
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import requires_cuda


def _graph(dtype: str) -> Graph:
    b, m, n, k = (Var(name) for name in ("b", "m", "n", "k"))
    cell = (
        Loop(
            Axis("k", 64),
            (
                Load("av", "a", (b, m, k)),
                Load("bv", "w", (b, m, k, n)),
                Assign("product", "multiply", ("av", "bv")),
                Accum("sum", "product", axes=("k",)),
            ),
        ),
        Write("out", (b, m, n), "sum"),
    )
    body = Body((Loop(Axis("b", 2), (Loop(Axis("m", 4), (Loop(Axis("n", 32), cell),)),)),))
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("a", (2, 4, 64), dtype), node_id="a")
    graph.add_node(InputOp(), [], Tensor("w", (2, 4, 64, 32), dtype), node_id="w")
    graph.add_node(LoopOp(body=body, name="k_batched_matvec"), ["a", "w"], Tensor("out", (2, 4, 32), "f32"), node_id="out")
    graph.inputs, graph.outputs = ["a", "w"], ["out"]
    return graph


def _compiled(dtype: str):
    with pinned_knobs({"PLACE": "fuse", "WORK": "w1x1", "TILE": f"mma_m16n8k16_{dtype}_f32/f1x1", "REDUCE": "", "STAGE": ""}):
        return Pipeline.build(CUDA_PASSES).run(_graph(dtype), ctx=Context.from_target((12, 0)))


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_batched_matvec_emits_mma(dtype: str) -> None:
    compiled = _compiled(dtype)
    sources = [node.op.kernel_source for node in compiled.nodes.values() if isinstance(node.op, CudaOp)]
    assert len(sources) == 1
    assert f"emmy_mma_m16n8k16_{dtype}_f32(" in sources[0]


@requires_cuda
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_batched_matvec_matches_independent_reference(dtype: str) -> None:
    import torch

    rng = np.random.default_rng(19)
    a = rng.standard_normal((2, 4, 64)).astype(np.float32)
    w = rng.standard_normal((2, 4, 64, 32)).astype(np.float32)
    if dtype == "bf16":
        at, wt = torch.from_numpy(a).to(torch.bfloat16), torch.from_numpy(w).to(torch.bfloat16)
        a, w = at.view(torch.uint16).numpy(), wt.view(torch.uint16).numpy()
        a_ref, w_ref = at.float().numpy(), wt.float().numpy()
    else:
        a, w = a.astype(np.float16), w.astype(np.float16)
        a_ref, w_ref = a.astype(np.float32), w.astype(np.float32)
    result, _ = CudaBackend().run(_compiled(dtype), input_data={"a": a, "w": w})
    expected = np.einsum("bmk,bmkn->bmn", a_ref, w_ref)
    np.testing.assert_allclose(result.outputs["out"].reshape(expected.shape), expected, rtol=1e-3, atol=1e-3)
