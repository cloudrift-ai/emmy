"""A carried walk in blocks of steps (``tile/lift/005_block``), on the in-place solve the compiler traces and rolls.

The pass rewrites the kernel into two levels of steps and a second state; it is still one kernel. The cut pass then
runs the outer level as kernels in order.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from emmy.commands.trace import graph_from_code
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.loop.runner import execute_loop_op_cpp
from emmy.compiler.ir.stmt import Loop
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, Pipeline
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import requires_cuda
from tests.compiler.passes.test_roll_recurrence import _INVERSE, _delta, _run

in_blocks = importlib.import_module("emmy.compiler.pipeline.passes.tile.lift.005_block").in_blocks


def _walk(code: str):
    graph, _, (module, _, _) = graph_from_code(code)
    rolled = Pipeline.build(LOOP_PASSES).run(graph)
    (walk,) = (node for node in rolled.nodes.values() if isinstance(node.op, LoopOp) and node.op.body.carries)
    return rolled, walk, module


def _levels(body) -> list[Loop]:
    return [stmt for stmt in body.iter() if isinstance(stmt, Loop) and stmt.carries]


@pytest.mark.parametrize(("n", "width"), [(19, 4), (16, 4), (20, 6), (35, 16)], ids=["even", "ragged", "ragged-6", "two-blocks"])
def test_the_walk_in_blocks_is_one_kernel_and_matches_eager(n: int, width: int) -> None:
    """The roll leaves the first and the last row to its neighbours, so the walk has ``n - 3`` steps. In blocks it
    is two levels of steps: the blocks, which seed the state, and the walk's own steps, which run on."""
    rolled, walk, module = _walk(_INVERSE.format(n=n))
    body = LoopOp(body=in_blocks(walk.op.body, width)).body
    blocks, steps = _levels(body)
    assert blocks.seed and not steps.seed and len(blocks.carries) == 2 and len(steps.carries) == 1
    assert blocks.axis.extent.as_static() == -(-(n - 3) // width) and steps.axis.extent.as_static() == width

    a = (np.random.default_rng(0).standard_normal((2, n, n)) * 0.1).astype(np.float32)
    values = {"a": a}
    for node_id in rolled.topological_order():
        node = rolled.nodes[node_id]
        if isinstance(node.op, LoopOp):
            op = LoopOp(body=body) if node is walk else node.op
            shape = tuple(dim.as_static() for dim in node.output.shape)
            values[node_id] = execute_loop_op_cpp(op, {name: values[name] for name in node.inputs}, {node.output.name: shape})
    want = module(torch.from_numpy(a)).numpy()
    np.testing.assert_allclose(values[rolled.outputs[0]].reshape(want.shape), want, rtol=1e-4, atol=1e-5)


def test_a_walk_the_block_form_does_not_hold_is_refused() -> None:
    """One block that holds every step gains nothing. The delta rule's chunk loop sums over no rows of its state:
    it is all cycle, and the form that breaks it is not this one."""
    _, solve, _ = _walk(_INVERSE.format(n=16))
    with pytest.raises(ValueError, match="one block holds the walk"):
        in_blocks(solve.op.body, 64)
    _, delta, _ = _walk(_delta(b=2, t=16, d=8, chunk=4))
    with pytest.raises(ValueError):
        in_blocks(delta.op.body, 4)


def _compiled(n: int, width: int | None, passes: list[str]):
    graph, _, (module, _, _) = graph_from_code(_INVERSE.format(n=n))
    with pinned_knobs({} if width is None else {"BLOCK": width}):
        return Pipeline.build(passes).run(graph), module


@pytest.mark.parametrize(("n", "width"), [(16, 4), (19, 8)], ids=["ragged", "even"])
def test_a_pinned_width_runs_the_blocks_as_kernels_in_order(n: int, width: int) -> None:
    """Under the pin the walk is lifted in two levels and the cut runs the outer one as kernels: a walk a block,
    and between two blocks the kernel that sums over the rows before the next. The walks' steps are the walk's own,
    none more: a last block the width does not fill has only the steps that write."""
    passes = CUDA_PASSES[: CUDA_PASSES.index("tile/cut") + 1]
    plain, _ = _compiled(n, None, passes)
    cut, module = _compiled(n, width, passes)
    walks = [node.op for node in cut.nodes.values() if isinstance(node.op, TileOp) and node.op.carries]
    blocks = -(-(n - 3) // width)
    assert len(walks) == blocks and len(cut.nodes) == len(plain.nodes) + 2 * (blocks - 1)
    assert sum(level.axis.extent.as_static() for op in walks for level in _levels(op.loop_body)) == n - 3

    arrays = _run(cut)
    want = module(torch.from_numpy(arrays["a"])).numpy()
    np.testing.assert_allclose(arrays[cut.outputs[0]].reshape(want.shape), want, rtol=1e-4, atol=1e-5)
    scheduled = Pipeline.build(["tile/schedule", "lowering/kernel"]).run(cut)
    assert not any(isinstance(node.op, TileOp) for node in scheduled.nodes.values())


@requires_cuda
@pytest.mark.xdist_group("cuda")
def test_the_walk_in_blocks_matches_eager_on_the_gpu() -> None:
    from emmy.compiler.backend.cuda.program import run_program  # noqa: PLC0415

    graph, module = _compiled(32, 8, CUDA_PASSES)
    a = (np.random.default_rng(0).standard_normal((2, 32, 32)) * 0.1).astype(np.float32)
    result, _ = run_program(graph, {"a": a})
    want = module(torch.from_numpy(a)).numpy()
    np.testing.assert_allclose(np.asarray(result.outputs[graph.outputs[0]]).reshape(want.shape), want, rtol=1e-4, atol=1e-5)
