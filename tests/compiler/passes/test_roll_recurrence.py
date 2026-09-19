"""An unrolled recurrence rolls into one kernel that carries its state, before fusion inlines it.

A chunked delta rule traces as a Python loop, so fused whole, chunk ``j``'s consumer re-derives every
earlier state. The fusion stage reads the chain of states instead: one step between two of them, the
steps one body at a stride, and replaces them with one kernel that carries the state.
The numerics are checked against eager PyTorch, and the negative case — steps that are not one body
at a stride — must be left alone, because a chain rolled wrongly is a wrong answer.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from emmy.commands.trace import graph_from_code
from emmy.compiler.context import Context
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.loop.runner import execute_loop_op_cpp
from emmy.compiler.ir.stmt import Loop
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, Pipeline
from tests.compiler.helpers import requires_cuda

_DELTA = """
import torch, torch.nn as nn
class Delta(nn.Module):
    # the inter-chunk half of a gated delta rule: S carried across chunks of C tokens
    def forward(self, q, k, v, g):
        B, T, D = q.shape
        C = {chunk}
        S = torch.zeros(B, D, D, dtype=q.dtype)
        outs = []
        for c in range(T // C):
            qc, kc, vc = q[:, c*C:(c+1)*C], k[:, c*C:(c+1)*C], v[:, c*C:(c+1)*C]
            gc = g[:, c*C:(c+1)*C].sum(1)[:, None, None]
            outs.append(qc @ S)
            S = S * torch.exp(gc) + kc.transpose(1, 2) @ vc
        return torch.cat(outs{kept}, 1)
m = Delta()
m(torch.randn({b}, {t}, {d}), torch.randn({b}, {t}, {d}), torch.randn({b}, {t}, {d}), torch.randn({b}, {t}))
"""


def _delta(b: int = 2, t: int = 16, d: int = 8, chunk: int = 4, kept: str = "") -> str:
    return _DELTA.format(b=b, t=t, d=d, chunk=chunk, kept=kept)


def _carriers(graph) -> list[LoopOp]:
    return [node.op for node in graph.nodes.values() if isinstance(node.op, LoopOp) and node.op.body.carries]


def test_an_unrolled_delta_rule_rolls_into_one_kernel_that_carries_its_state() -> None:
    graph, _, _ = graph_from_code(_delta())
    graph = Pipeline.build(LOOP_PASSES).run(graph)

    (rolled,) = _carriers(graph)
    (steps,) = (stmt for stmt in rolled.body if isinstance(stmt, Loop))
    # Four chunks: the last chunk's state is never read, so three steps are carried.
    assert steps.axis.extent.as_static() == 3 and list(steps.carries) == [carry.name for carry in rolled.body.carries]
    # Every input is read at a step-relative offset, and the state is stored once per step.
    assert all(steps.axis.name in load.index[1].free_vars() for load in rolled.body.loads)
    (store,) = rolled.outputs
    # Each consumer reads ONE stored step: nothing re-derives an earlier state.
    readers = [node.op for node in graph.nodes.values() if isinstance(node.op, LoopOp) and store in node.inputs]
    reads = sorted(load.index[0].value for op in readers for load in op.body.loads if load.input == store)
    assert reads == [0, 1, 2] and all(len(op.body.accums) == 1 for op in readers)


def _run(graph) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    arrays: dict[str, np.ndarray] = {}
    for node in graph.nodes.values():
        if isinstance(node.op, InputOp | ConstantOp):
            tensor = node.outputs[0]
            shape = tuple(dim.as_static() for dim in tensor.shape)
            fill = rng.standard_normal(shape) * 0.5 if isinstance(node.op, InputOp) else np.full(shape, node.op.value)
            arrays[tensor.name] = fill.astype(np.float32)
    for node_id in graph.topological_order():
        node = graph.nodes[node_id]
        if isinstance(node.op, TileOp):
            # The serial axis is the outermost loop of the closed program, so a read one launch
            # back sees every cell the previous step stored.
            loop = LoopOp(body=node.op.loop_body)
            shapes = {tensor.name: tuple(dim.as_static() for dim in tensor.shape) for tensor in node.outputs}
            result = execute_loop_op_cpp(loop, arrays, shapes)
            arrays.update(zip(loop.outputs, result if isinstance(result, tuple) else (result,), strict=True))
    return arrays


def _chunks(arrays: dict[str, np.ndarray]) -> np.ndarray:
    names = sorted((name for name in arrays if name.startswith("matmul")), key=lambda name: int(name.split("_")[1]) if "_" in name else 0)
    return np.concatenate([arrays[name] for name in names], axis=1)


@pytest.mark.parametrize(("b", "t", "d", "chunk"), [(1, 6, 4, 2), (2, 16, 8, 4)])
def test_the_rolled_kernel_matches_eager(b: int, t: int, d: int, chunk: int) -> None:
    graph, _, (module, _, _) = graph_from_code(_delta(b, t, d, chunk))
    graph = Pipeline.build(["lowering/tile"], select=["lift"]).run(Pipeline.build(LOOP_PASSES).run(graph))
    assert sum(bool(node.op.place.serial) for node in graph.nodes.values() if isinstance(node.op, TileOp)) == 1

    arrays = _run(graph)

    reference = module(*(torch.from_numpy(arrays[name]) for name in ("q", "k", "v", "g"))).numpy()
    np.testing.assert_allclose(_chunks(arrays), reference, rtol=1e-4, atol=1e-5)


def test_steps_that_are_not_one_body_at_a_stride_do_not_roll() -> None:
    """The second chunk decays by a different tensor, so its step is not the first step read one
    stride later. Rolling it would fold ``h`` where the program said ``g``."""
    code = (
        _delta()
        .replace("def forward(self, q, k, v, g):", "def forward(self, q, k, v, g, h):")
        .replace("gc = g[:, c*C:(c+1)*C].sum(1)[:, None, None]", "gc = (g if c % 2 == 0 else h)[:, c*C:(c+1)*C].sum(1)[:, None, None]")
        .replace("torch.randn(2, 16))", "torch.randn(2, 16), torch.randn(2, 16))")
    )
    graph, _, _ = graph_from_code(code)

    assert not _carriers(Pipeline.build(LOOP_PASSES).run(graph))


_GATED_DELTA_RULE = """
import torch, torch.nn as nn
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
class GatedDeltaRule(nn.Module):
    def forward(self, q, k, v, g, beta):
        out, _ = torch_chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=4, use_qk_l2norm_in_kernel=True)
        return out
m = GatedDeltaRule()
m(torch.randn(1, 16, 2, 8), torch.randn(1, 16, 2, 8), torch.randn(1, 16, 2, 8), -torch.rand(1, 16, 2), torch.rand(1, 16, 2))
"""


def test_the_chunked_gated_delta_rule_of_a_real_model_rolls_and_matches_eager() -> None:
    """Qwen3.5's own chunk rule. Its step contracts over the previous state (``k_cumdecay @ S``),
    its zero state enters the first step through a broadcast that belongs to the step, and it keeps
    a second buffer per step (``v_new``) beside the state. Run as Loop IR, where a carrier executes
    as it is spelled."""
    graph, _, (module, _, _) = graph_from_code(_GATED_DELTA_RULE)
    graph = Pipeline.build(LOOP_PASSES).run(graph)

    (rolled,) = _carriers(graph)
    assert len(rolled.outputs) == 2 and any(isinstance(stmt, Loop) and stmt.carries for stmt in rolled.body)

    rng = np.random.default_rng(0)
    arrays: dict[str, np.ndarray] = {}
    for node_id in graph.topological_order():
        node = graph.nodes[node_id]
        shapes = {tensor.name: tuple(dim.as_static() for dim in tensor.shape) for tensor in node.outputs}
        if isinstance(node.op, InputOp | ConstantOp):
            (shape,) = shapes.values()
            fill = rng.random(shape) if isinstance(node.op, InputOp) else np.full(shape, node.op.value)
            arrays[node.outputs[0].name] = fill.astype(np.float32)
        elif isinstance(node.op, LoopOp):
            result = execute_loop_op_cpp(node.op, arrays, shapes)
            arrays.update(zip(node.op.outputs, result if isinstance(result, tuple) else (result,), strict=True))
    reference = module(*(torch.from_numpy(arrays[name]) for name in ("q", "k", "v", "g", "beta"))).numpy()
    (out,) = graph.outputs
    np.testing.assert_allclose(arrays[out].reshape(reference.shape), reference, rtol=1e-4, atol=1e-5)


def test_the_rolled_kernel_lowers_to_one_launch_per_step() -> None:
    """The serial axis reaches the CUDA op as a runtime ``int`` and a launch count; the body holds
    no loop over it and reads the previous step behind the guard."""
    graph, _, _ = graph_from_code(_delta())
    lowered = Pipeline.build(CUDA_PASSES).run(graph, ctx=Context.from_target((9, 0)))

    (step,) = (node.op for node in lowered.nodes.values() if isinstance(node.op, CudaOp) and node.op.serial)
    ((axis, launches),) = step.serial
    assert launches == 3 and axis in step.runtime_args
    assert f"int {axis}" in step.kernel_source and f"for (int {axis}" not in step.kernel_source
    assert f"({axis} > 0) ? ({axis} - 1) : (0)" in step.kernel_source


@requires_cuda
@pytest.mark.xdist_group("cuda")
def test_the_rolled_kernel_matches_eager_on_the_gpu() -> None:
    from emmy.compiler.backend.cuda.program import run_program  # noqa: PLC0415

    # The last two chunks only: a cat of more than two tensors has no lowering of its own yet.
    graph, _, (module, _, _) = graph_from_code(_delta(kept="[-2:]"))
    rng = np.random.default_rng(0)
    arrays = {name: (rng.standard_normal((2, 16) if name == "g" else (2, 16, 8)) * 0.5).astype(np.float32) for name in ("q", "k", "v", "g")}

    result, _ = run_program(Pipeline.build(CUDA_PASSES).run(graph), arrays)

    reference = module(*(torch.from_numpy(arrays[name]) for name in ("q", "k", "v", "g"))).numpy()
    (out,) = result.outputs.values()
    np.testing.assert_allclose(np.asarray(out).reshape(reference.shape), reference, rtol=1e-4, atol=1e-5)
