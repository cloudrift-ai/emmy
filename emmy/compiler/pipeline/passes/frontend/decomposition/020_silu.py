"""Decompose silu(x) into x / (1 + exp(-x)) with one final output rounding."""

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.pipeline import Match, Pattern
from emmy.compiler.pipeline.passes.frontend.decomposition._helpers import const_bc, open_fragment, opmath_dtype

PATTERN = [Pattern("root", ElementwiseOp, {"fn": "silu"})]


def rewrite(match: Match, inp_x: Node, out: Tensor) -> Graph | None:
    graph = match.graph
    frag = open_fragment(graph, [inp_x])
    opmath = opmath_dtype(inp_x.output.dtype)

    work_x = inp_x
    if inp_x.output.dtype != opmath:
        work_id = frag.add_node(
            op=ElementwiseOp(op="copy"),
            inputs=[inp_x],
            output=Tensor(f"{out.name}_opmath", out.shape, opmath),
        )
        work_x = frag.nodes[work_id]

    neg_id = frag.add_node(op=ElementwiseOp(op="negative"), inputs=[work_x], output=Tensor(f"{out.name}_neg", out.shape, opmath))
    exp_id = frag.add_node(op=ElementwiseOp(op="exp"), inputs=[neg_id], output=Tensor(f"{out.name}_exp", out.shape, opmath))
    one_bc = const_bc(frag, name=f"{out.name}_one", value=1.0, target_shape=out.shape, dtype=opmath)
    add_id = frag.add_node(
        op=ElementwiseOp(op="add"),
        inputs=[one_bc, exp_id],
        output=Tensor(f"{out.name}_denom", out.shape, opmath),
    )
    result_id = frag.add_node(
        op=ElementwiseOp(op="divide"),
        inputs=[work_x, add_id],
        output=Tensor(out.name, out.shape, out.dtype),
    )

    frag.outputs = [result_id]
    return frag
