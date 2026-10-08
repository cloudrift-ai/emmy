"""A per-batch matrix-vector product can use a unit MMA row."""

import numpy as np
import pytest

from emmy.compiler.backend.cuda.backend import CudaBackend
from emmy.compiler.context import Context
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import requires_cuda, requires_sm


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


def _compiled(dtype: str, reduce: str = ""):
    with pinned_knobs({"PLACE": "fuse", "WORK": "w1x1", "TILE": f"mma_m16n8k16_{dtype}_f32/f1x1", "REDUCE": reduce, "STAGE": ""}):
        return Pipeline.build(CUDA_PASSES).run(_graph(dtype), ctx=Context.from_target((12, 0)))


def _rank_one_graph(dtype: str, k_size: int = 64, n_size: int = 32, *, reshape_output: bool = False) -> Graph:
    n, k = Var("n"), Var("k")
    index = (BinaryExpr("/", n, Literal(128, "int")), BinaryExpr("%", n, Literal(128, "int"))) if reshape_output else (n,)
    output_shape = (n_size // 128, 128) if reshape_output else (n_size,)
    cell = (
        Loop(
            Axis("k", k_size),
            (
                Load("av", "a", (k,)),
                Load("bv", "w", (k, n)),
                Assign("product", "multiply", ("av", "bv")),
                Accum("sum", "product", axes=("k",)),
            ),
        ),
        Write("out", index, "sum"),
    )
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("a", (k_size,), dtype), node_id="a")
    graph.add_node(InputOp(), [], Tensor("w", (k_size, n_size), dtype), node_id="w")
    graph.add_node(
        LoopOp(body=Body((Loop(Axis("n", n_size), cell),)), name="k_rank_one_projection"),
        ["a", "w"],
        Tensor("out", output_shape, "f32"),
        node_id="out",
    )
    graph.inputs, graph.outputs = ["a", "w"], ["out"]
    return graph


def _rank_one_compiled(dtype: str, k_size: int = 64, n_size: int = 32, *, reshape_output: bool = False):
    with pinned_knobs({"PLACE": "fuse", "WORK": "w1x1", "TILE": f"mma_m16n8k16_{dtype}_f32/f1x1", "REDUCE": "", "STAGE": ""}):
        return Pipeline.build(CUDA_PASSES).run(
            _rank_one_graph(dtype, k_size, n_size, reshape_output=reshape_output), ctx=Context.from_target((12, 0))
        )


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize(("k_size", "n_size"), [(64, 32), (5120, 2048), (5120, 6144), (5120, 10240)])
def test_rank_one_projection_emits_mma(dtype: str, k_size: int, n_size: int) -> None:
    compiled = _rank_one_compiled(dtype, k_size, n_size)
    sources = [node.op.kernel_source for node in compiled.nodes.values() if isinstance(node.op, CudaOp)]
    assert len(sources) == 1
    assert f"emmy_mma_m16n8k16_{dtype}_f32(" in sources[0]


@pytest.mark.parametrize("n_size", [256, 2048, 6144])
def test_rank_one_reshaped_projection_emits_mma(n_size: int) -> None:
    compiled = _rank_one_compiled("bf16", 5120, n_size, reshape_output=True)
    sources = [node.op.kernel_source for node in compiled.nodes.values() if isinstance(node.op, CudaOp)]
    assert len(sources) == 1
    assert "emmy_mma_m16n8k16_bf16_f32(" in sources[0]


@requires_cuda
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize(("n_size", "reshape_output"), [(32, False), (256, True)])
def test_rank_one_projection_matches_independent_reference(dtype: str, n_size: int, reshape_output: bool) -> None:
    import torch

    rng = np.random.default_rng(47)
    a = rng.standard_normal((64,)).astype(np.float32)
    w = rng.standard_normal((64, n_size)).astype(np.float32)
    if dtype == "bf16":
        at, wt = torch.from_numpy(a).to(torch.bfloat16), torch.from_numpy(w).to(torch.bfloat16)
        a, w = at.view(torch.uint16).numpy(), wt.view(torch.uint16).numpy()
        a_ref, w_ref = at.float().numpy(), wt.float().numpy()
    else:
        a, w = a.astype(np.float16), w.astype(np.float16)
        a_ref, w_ref = a.astype(np.float32), w.astype(np.float32)
    result, _ = CudaBackend().run(_rank_one_compiled(dtype, n_size=n_size, reshape_output=reshape_output), input_data={"a": a, "w": w})
    expected = (a_ref @ w_ref).reshape((n_size // 128, 128) if reshape_output else (n_size,))
    np.testing.assert_allclose(result.outputs["out"], expected, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_batched_matvec_emits_mma(dtype: str) -> None:
    compiled = _compiled(dtype)
    sources = [node.op.kernel_source for node in compiled.nodes.values() if isinstance(node.op, CudaOp)]
    assert len(sources) == 1
    assert f"emmy_mma_m16n8k16_{dtype}_f32(" in sources[0]


@requires_cuda
@requires_sm(8, 0)  # the m16n8k16 atom these programs are pinned to
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize("reduce", ["", "g2k"])
def test_batched_matvec_matches_independent_reference(dtype: str, reduce: str) -> None:
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
    result, _ = CudaBackend().run(_compiled(dtype, reduce), input_data={"a": a, "w": w})
    expected = np.einsum("bmk,bmkn->bmn", a_ref, w_ref)
    np.testing.assert_allclose(result.outputs["out"].reshape(expected.shape), expected, rtol=1e-3, atol=1e-3)


def _row_scaled_fp8_graph() -> Graph:
    m, n, k = (Var(name) for name in ("m", "n", "k"))
    cell = (
        Loop(
            Axis("k", 32),
            (
                Load("scale_value", "scale", (m, k)),
                Load("bits_value", "bits", (k, n)),
                Assign("weight", "from_f8e4m3", ("bits_value",)),
                Assign("scaled", "multiply", ("weight", "scale_value")),
                Load("activation", "a", (m, k)),
                Assign("product", "multiply", ("activation", "scaled")),
                Accum("sum", "product", axes=("k",)),
            ),
        ),
        Write("out", (m, n), "sum"),
    )
    body = Body((Loop(Axis("m", 2), (Loop(Axis("n", 16), cell),)),))
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("a", (2, 32), "f16"), node_id="a")
    graph.add_node(InputOp(), [], Tensor("scale", (2, 32), "f16"), node_id="scale")
    graph.add_node(InputOp(), [], Tensor("bits", (32, 16), "f8e4m3"), node_id="bits")
    graph.add_node(LoopOp(body=body), ["a", "scale", "bits"], Tensor("out", (2, 16), "f32"), node_id="out")
    graph.inputs, graph.outputs = ["a", "scale", "bits"], ["out"]
    return graph


@requires_cuda
def test_row_scaled_fp8_b_matches_independent_reference() -> None:
    """A scale indexed by m and k remains in B for each independent m batch."""
    from emmy.compiler.dtype import decode_f8

    rng = np.random.default_rng(41)
    a = rng.standard_normal((2, 32)).astype(np.float16)
    scale = (rng.standard_normal((2, 32)) * 0.05).astype(np.float16)
    bits = rng.integers(0, 256, (32, 16), dtype=np.uint8)
    bits[(bits == 0x7F) | (bits == 0xFF)] = 0
    with pinned_knobs({"PLACE": "fuse", "WORK": "w1x1", "TILE": "mma_m16n8k16_f16_f32/f1x1", "REDUCE": "", "STAGE": ""}):
        compiled = Pipeline.build(CUDA_PASSES).run(_row_scaled_fp8_graph(), ctx=Context.from_target((12, 0)))
    sources = [node.op.kernel_source for node in compiled.nodes.values() if isinstance(node.op, CudaOp)]
    assert len(sources) == 1
    result, _ = CudaBackend().run(compiled, input_data={"a": a, "scale": scale, "bits": bits})
    expected = np.einsum("mk,mk,kn->mn", a.astype(np.float32), scale.astype(np.float32), decode_f8(bits, "f8e4m3"))
    np.testing.assert_allclose(result.outputs["out"].reshape(expected.shape), expected, rtol=2e-3, atol=2e-3)


def _grouped_weight_graph() -> Graph:
    from emmy.compiler.ir.expr import BinaryExpr, Literal

    h, m, n, k = (Var(name) for name in ("h", "m", "n", "k"))
    group = BinaryExpr("//", h, Literal(3, "int"))
    row = BinaryExpr("*", m, Literal(6 * 16, "int"))
    flat = BinaryExpr("+", BinaryExpr("+", row, BinaryExpr("*", group, Literal(16, "int"))), n)
    cell = (
        Loop(
            Axis("k", 32),
            (
                Load("activation", "a", (h, m, k)),
                Load("weight", "w", (k, flat)),
                Assign("product", "multiply", ("activation", "weight")),
                Accum("sum", "product", axes=("k",)),
            ),
        ),
        Write("out", (h, m, n), "sum"),
    )
    body = Body((Loop(Axis("h", 6), (Loop(Axis("m", 2), (Loop(Axis("n", 16), cell),)),)),))
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("a", (6, 2, 32), "f16"), node_id="a")
    graph.add_node(InputOp(), [], Tensor("w", (32, 2 * 6 * 16), "f16"), node_id="w")
    graph.add_node(LoopOp(body=body), ["a", "w"], Tensor("out", (6, 2, 16), "f32"), node_id="out")
    graph.inputs, graph.outputs = ["a", "w"], ["out"]
    return graph


@requires_cuda
@requires_sm(8, 0)  # the m16n8k16 atom this program is pinned to
def test_grouped_weight_row_batch_matches_independent_reference() -> None:
    """The grouped weight address retains both h and m outside the unit MMA row."""
    rng = np.random.default_rng(43)
    a = rng.standard_normal((6, 2, 32)).astype(np.float16)
    w = rng.standard_normal((32, 2 * 6 * 16)).astype(np.float16)
    with pinned_knobs({"PLACE": "fuse", "WORK": "w1x1", "TILE": "mma_m16n8k16_f16_f32/f1x1", "REDUCE": "", "STAGE": ""}):
        compiled = Pipeline.build(CUDA_PASSES).run(_grouped_weight_graph(), ctx=Context.from_target((12, 0)))
    sources = [node.op.kernel_source for node in compiled.nodes.values() if isinstance(node.op, CudaOp)]
    assert len(sources) == 1 and "mma.sync" in sources[0]
    result, _ = CudaBackend().run(compiled, input_data={"a": a, "w": w})
    expected = np.empty((6, 2, 16), dtype=np.float32)
    for head in range(6):
        for row in range(2):
            start = row * 6 * 16 + (head // 3) * 16
            expected[head, row] = a[head, row].astype(np.float32) @ w[:, start : start + 16].astype(np.float32)
    np.testing.assert_allclose(result.outputs["out"].reshape(expected.shape), expected, rtol=2e-3, atol=2e-3)
