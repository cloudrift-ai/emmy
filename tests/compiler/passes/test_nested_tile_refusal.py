"""A scalar outer fold must not silently discard its operand's selected contraction tile."""

import pytest

from emmy.compiler.context import Context
from emmy.compiler.dtype import F16, F32
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.cuda.ir import CudaOp
from emmy.compiler.ir.frontend.ir import MatmulOp, ReshapeOp
from emmy.compiler.ir.tensor.ir import ReduceOp
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import CUDA_PASSES, TILE_PASSES, Pipeline
from emmy.compiler.pipeline.pipeline import LoweringError
from emmy.compiler.pipeline.search.pins import pinned_knobs


def test_block_maximum_refuses_an_ignored_nested_matmul_tile() -> None:
    graph = Graph()
    for name in ("a", "b"):
        graph.add_node(InputOp(), [], Tensor(name, (128, 128), F16), node_id=name)
    graph.add_node(MatmulOp(), ["a", "b"], Tensor("mm", (128, 128), F32), node_id="mm")
    graph.add_node(ReshapeOp((128, 8, 16)), ["mm"], Tensor("blocks", (128, 8, 16), F32), node_id="blocks")
    graph.add_node(ReduceOp("maximum", -1), ["blocks"], Tensor("out", (128, 8, 1), F32), node_id="out")
    graph.inputs, graph.outputs = ["a", "b"], ["out"]
    ctx = Context.from_target((12, 0))
    pins = {"TILE": "mma_m16n8k16_f16_f32/f1x2/k2", "WORK": "w1x1", "STAGE": "", "REDUCE": "", "PLACE": "fuse"}

    with pinned_knobs(pins):
        scheduled = Pipeline.build(TILE_PASSES).run(graph, ctx=ctx, db=None)
        tiles = [node.op for node in scheduled.nodes.values() if isinstance(node.op, TileOp)]
        assert len(tiles) == 1 and tiles[0].knobs["TILE"] == pins["TILE"]
        with pytest.raises(LoweringError, match="serial lowering of reduce cannot realize TILE at reduce.1/inner"):
            Pipeline.build(CUDA_PASSES).run(graph, ctx=ctx, db=None)

    with pinned_knobs({"PLACE": "fuse"}):
        compiled = Pipeline.build(CUDA_PASSES).run(graph, ctx=ctx, db=None)
    kernels = [node.op for node in compiled.nodes.values() if isinstance(node.op, CudaOp)]
    assert len(kernels) == 1 and all(not value for key, value in kernels[0].knobs.items() if key.partition("@")[0] == "TILE")
