"""A carried walk in blocks: the rewrite that emits the block form from the kernel itself.

The in-place triangular solve traces as one row substitution a step and rolls into one kernel that carries its
state. Cut into blocks of rows, the rows of earlier blocks become one kernel a block that carries nothing, and what
is left is a walk as long as the block. The numerics are checked against eager PyTorch through the Loop IR runner.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from emmy.commands.trace import graph_from_code
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Loop
from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, Pipeline
from emmy.compiler.pipeline.passes.loop.fusion._blocks import block_walk
from tests.compiler.helpers import requires_cuda
from tests.compiler.passes.test_roll_recurrence import _INVERSE, _delta, _run


def _blocked(n: int, width: int):
    graph, _, (module, _, _) = graph_from_code(_INVERSE.format(n=n))
    rolled = Pipeline.build(LOOP_PASSES).run(graph)
    (walk,) = (node for node in rolled.nodes.values() if isinstance(node.op, LoopOp) and node.op.body.carries)
    return block_walk(rolled, walk.id, width), module


def _steps(op: LoopOp) -> int:
    (loop,) = (stmt for stmt in op.body if isinstance(stmt, Loop) and stmt.carries)
    return loop.axis.extent.as_static()


@pytest.mark.parametrize(("n", "width"), [(16, 4), (20, 6), (32, 16), (16, 64)], ids=["even", "ragged", "two-blocks", "one-block"])
def test_the_solve_in_blocks_matches_eager(n: int, width: int) -> None:
    graph, module = _blocked(n, width)
    kernels = [node.op for node in graph.nodes.values() if isinstance(node.op, LoopOp)]
    walks = [op for op in kernels if op.body.carries]
    # The roll leaves the first and the last row to its neighbours: rows 2 to n - 2 are the walk's.
    blocks = len({row // width for row in range(2, n - 1)})
    assert len(walks) == blocks and sum(_steps(op) for op in walks) == n - 3 and max(_steps(op) for op in walks) <= width
    # Every block after the first has one kernel over the rows of the blocks before it, and it carries nothing.
    early = [node for node in graph.nodes.values() if "__early" in node.id]
    assert len(early) == blocks - 1 and not any(node.op.body.carries for node in early)

    lifted = Pipeline.build(["tile/lift"], select=["lift"]).run(graph)
    arrays = _run(lifted)
    want = module(torch.from_numpy(arrays["a"])).numpy()
    (out,) = lifted.outputs
    np.testing.assert_allclose(arrays[out].reshape(want.shape), want, rtol=1e-4, atol=1e-5)


def test_a_walk_the_block_form_does_not_hold_is_refused() -> None:
    """The delta rule's chunk loop stores every chunk's state and moves along no row of it."""
    graph, _, _ = graph_from_code(_delta(b=2, t=16, d=8, chunk=4))
    rolled = Pipeline.build(LOOP_PASSES).run(graph)
    (walk,) = (node for node in rolled.nodes.values() if isinstance(node.op, LoopOp) and node.op.body.carries)
    with pytest.raises(ValueError):
        block_walk(rolled, walk.id, 2)


@requires_cuda
@pytest.mark.xdist_group("cuda")
def test_the_solve_in_blocks_matches_eager_on_the_gpu() -> None:
    from emmy.compiler.backend.cuda.program import run_program  # noqa: PLC0415

    graph, module = _blocked(32, 16)
    graph = Pipeline.build(CUDA_PASSES).run(graph)
    a = (np.random.default_rng(0).standard_normal((2, 32, 32)) * 0.1).astype(np.float32)
    result, _ = run_program(graph, {"a": a})
    (out,) = graph.outputs
    want = module(torch.from_numpy(a)).numpy()
    np.testing.assert_allclose(np.asarray(result.outputs[out]).reshape(want.shape), want, rtol=1e-4, atol=1e-5)
