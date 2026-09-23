"""A fused region's body does not depend on the order its kernels merged in.

``h = x + r; n = rms_norm(h, w)`` with both ``h`` and ``n`` live is the case: merged all at once,
the add's store and the norm's store share one output loop and the add is computed once for
them. Merged as "the norm cone first, then the add", the splicer lands the add's store in a
loop named after its own axis, beside the norm's output loop of the same extent, and demands
the add at both scopes. The sibling free-loop merge in the normalizer folds those two loops
back into one, after which load deduplication leaves one add; both orders then give one body.
"""

from __future__ import annotations

from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import RmsNormOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Assign, Load, Loop, Write
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.pipeline import Pipeline
from emmy.compiler.pipeline.passes.loop.fusion._region import build_merged_region, live_outputs_of


def _lifted_residual_norm() -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (8, 256)), node_id="x")
    g.add_node(InputOp(), [], Tensor("r", (8, 256)), node_id="r")
    g.add_node(InputOp(), [], Tensor("w", (256,)), node_id="w")
    g.add_node(ElementwiseOp("add"), ["x", "r"], Tensor("add", (8, 256)), node_id="add")
    g.add_node(RmsNormOp(eps=1e-6), ["add", "w"], Tensor("rms_norm", (8, 256)), node_id="rms_norm")
    g.inputs, g.outputs = ["x", "r", "w"], ["rms_norm", "add"]
    return Pipeline.build(["frontend/decomposition", "frontend/optimization", "loop/lifting"]).run(g)


def _merge_all_at_once(lifted: Graph) -> LoopOp:
    region = {nid for nid, n in lifted.nodes.items() if isinstance(n.op, LoopOp)}
    merged = build_merged_region(lifted, region, live_outputs_of(lifted, region))
    assert merged is not None
    return merged


def _merge_norm_cone_then_add(lifted: Graph) -> LoopOp:
    cone = {nid for nid, n in lifted.nodes.items() if isinstance(n.op, LoopOp) and nid != "add"}
    live = live_outputs_of(lifted, cone)
    cone_op = build_merged_region(lifted, cone, live)
    assert cone_op is not None
    g = Graph()
    for nid, n in lifted.nodes.items():
        if not isinstance(n.op, LoopOp):
            g.add_node(n.op, list(n.inputs), outputs=n.outputs, node_id=nid)
    add = lifted.nodes["add"]
    g.add_node(add.op, list(add.inputs), outputs=add.outputs, node_id="add")
    (norm,) = live
    g.add_node(cone_op, list(cone_op.inputs), outputs=[lifted.buffer(norm)], node_id=norm)
    g.outputs = [norm, "add"]
    region = {"add", norm}
    merged = build_merged_region(g, region, live_outputs_of(g, region))
    assert merged is not None
    return merged


def _adds_of_loads(sweep: Loop) -> list[Assign]:
    """The adds in ``sweep``'s own body whose operands are both loads there: the residual add."""
    loaded = {name for s in sweep.body if isinstance(s, Load) for name in s.defines()}
    return [s for s in sweep.body if isinstance(s, Assign) and s.op.name == "add" and set(s.args) <= loaded]


def test_both_merge_orders_give_one_body() -> None:
    lifted = _lifted_residual_norm()
    assert _merge_norm_cone_then_add(lifted).body == _merge_all_at_once(lifted).body


def test_the_add_is_computed_once_per_output_loop_and_both_stores_share_that_loop() -> None:
    merged = _merge_norm_cone_then_add(_lifted_residual_norm())
    (outer,) = [s for s in merged.body if isinstance(s, Loop)]
    output_loops = [s for s in outer.body if isinstance(s, Loop) and not s.is_reduce]
    assert len(output_loops) == 1, [loop.axis.name for loop in output_loops]
    (sweep,) = output_loops
    assert sorted(w.output for w in sweep.body if isinstance(w, Write)) == ["add", "rms_norm"]
    assert len(_adds_of_loads(sweep)) == 1
