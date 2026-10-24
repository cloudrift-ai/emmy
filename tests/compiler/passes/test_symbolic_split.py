"""A cross-CTA split of a SYMBOLIC reduce stream: each partition's slice spans ``⌈K/w⌉`` positions, the last
overhanging the stream's end, whose loads read the stream's last position and whose lift folds the monoid's seed
past it. Run on a card at lengths no width divides, the split contraction is the unsplit one."""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler import dtype as dt
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import MatmulOp
from emmy.compiler.ir.pure.fold import binding_extents
from emmy.compiler.ir.stmt import Select
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.ops import head
from emmy.compiler.pipeline import LOOP_PASSES, Match, Pipeline
from emmy.compiler.pipeline.passes.tile._split import realize_split, split_forks
from tests.compiler.helpers import requires_cuda

KEYS = Dim("keys")


def _matvec(n: int = 96) -> Graph:
    """``a[1, keys] @ b[keys, n]``: a contraction over a stream whose length is known only at run time."""
    graph = Graph()
    for name, shape in (("a", (1, KEYS)), ("b", (KEYS, n))):
        graph.add_node(InputOp(), [], Tensor(name, shape, dt.F32), node_id=name)
    graph.add_node(MatmulOp(), ["a", "b"], Tensor("o", (1, n), dt.F32), node_id="o")
    graph.inputs, graph.outputs = ["a", "b"], ["o"]
    return graph


def test_a_symbolic_stream_splits_into_overhanging_masked_slices() -> None:
    lifted = Pipeline.build([*LOOP_PASSES, "tile/lift"]).run(_matvec())
    (node,) = (n for n in lifted.nodes.values() if isinstance(n.op, TileOp))
    match = Match(graph=lifted, root_node_id=node.id, rule=None)
    offered = {value for fork in split_forks(match, node) for value in fork.knobs.values()}
    assert {"", "g4k", "g32k"} <= offered
    with binding_extents({"keys"}):  # the launch binds the stream length, as the pipeline run does
        split = realize_split(match, node, 4, "kernel")
    (partial,) = (n.op for n in split.nodes.values() if isinstance(n.op, TileOp) and n.op.name.endswith("__partial"))
    fold = head(partial.op)
    # ⌈keys / 4⌉ positions a slice, and a lift that keeps a position only inside the stream.
    extent = partial.axis_of(fold.axis).extent
    assert not extent.is_static and extent.expr.eval({"keys": 37}) == 10
    assert any(isinstance(stmt, Select) for stmt in fold.lift.body)


@requires_cuda
@pytest.mark.xdist_group("cuda")
@pytest.mark.parametrize("length", [1, 37, 128, 130])
def test_a_split_symbolic_contraction_is_the_unsplit_one_on_the_gpu(length: int) -> None:
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415
    from emmy.compiler.backend.cuda.program import CompiledProgram  # noqa: PLC0415
    from emmy.compiler.backend.gpu_lock import gpu_lock  # noqa: PLC0415
    from emmy.compiler.pipeline.search.pins import pinned_knobs  # noqa: PLC0415

    with pinned_knobs({"REDUCE": "g4k"}):
        compiled = CudaBackend().compile(_matvec())
    rng = np.random.default_rng(length)
    a, b = rng.normal(size=(1, length)).astype(np.float32), rng.normal(size=(length, 96)).astype(np.float32)
    with gpu_lock():
        program = CompiledProgram.build(compiled, {"a": a, "b": b})
        assert any("__partial" in launch.kernel_name for launch in program.plan.launches)
        program.run_once()
        np.testing.assert_allclose(program.outputs()["o"], a @ b, rtol=1e-4, atol=1e-4)
