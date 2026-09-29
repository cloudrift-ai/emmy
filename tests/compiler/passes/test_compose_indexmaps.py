"""Index-map composition preserves every externally observed value."""

import numpy as np

from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import placeholder
from emmy.compiler.ir.tensor.ir import IndexMapOp, IndexSource
from emmy.compiler.pipeline import Pipeline


def test_composing_a_returned_intermediate_preserves_both_outputs(run_graph):
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (2, 3)), node_id="x")
    graph.add_node(
        IndexMapOp(out_shape=(3, 2), sources=(IndexSource(input_idx=0, coord_map=(placeholder(1), placeholder(0))),)),
        ["x"], Tensor("transposed", (3, 2)), node_id="transposed",
    )
    graph.add_node(
        IndexMapOp(out_shape=(6,), sources=(IndexSource(input_idx=0, coord_map=(placeholder(0) / 2, placeholder(0) % 2)),)),
        ["transposed"], Tensor("flat", (6,)), node_id="flat",
    )
    graph.inputs, graph.outputs = ["x"], ["transposed", "flat"]
    optimized = Pipeline.build(["frontend/optimization"]).run(graph)
    assert optimized.outputs == ["transposed", "flat"]
    assert optimized.nodes["flat"].inputs == ["x"]
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    actual = run_graph(optimized, {"x": values})
    np.testing.assert_array_equal(actual["transposed"], values.T)
    np.testing.assert_array_equal(actual["flat"], values.T.reshape(-1))
