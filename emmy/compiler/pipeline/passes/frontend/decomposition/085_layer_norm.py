"""Decompose layer_norm(x [, w [, b]]) into (x - mean(x)) * rsqrt(mean((x - mean(x))^2) + eps) [* w] [+ b].

A 16-bit layer norm computes in f32 and rounds its output once, as torch's does: its mean and variance in the input
format would round every row's statistics, and the normalized values with them."""

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.frontend.ir import LayerNormOp, MeanOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.pipeline import Match, Pattern
from emmy.compiler.pipeline.passes.frontend.decomposition._helpers import (
    broadcast_to,
    const_bc,
    open_fragment,
    reduction_shape,
)

PATTERN = [Pattern("root", LayerNormOp)]


def rewrite(match: Match, root: Node, inp_x: Node, inp_w: Node | None, inp_b: Node | None, out: Tensor) -> Graph | None:
    graph = match.graph
    red_shape = reduction_shape(out.shape, -1) if out.shape else (1,)
    widen = out.dtype.name in ("f16", "bf16")
    dtype = "f32" if widen else out.dtype

    frag = open_fragment(graph, [n for n in (inp_x, inp_w, inp_b) if n is not None])

    def computed(node, name):
        """``node`` in the computing format: an f32 copy of a 16-bit operand."""
        if not widen:
            return node
        shape = frag.nodes[node].output.shape if isinstance(node, str) else node.output.shape
        return frag.add_node(op=ElementwiseOp(op="copy"), inputs=[node], output=Tensor(f"{out.name}_{name}_f32", shape, dtype))

    inp_x = computed(inp_x, "x")

    mean_id = frag.add_node(
        op=MeanOp(axis=-1),
        inputs=[inp_x],
        output=Tensor(f"{out.name}_mean", red_shape, dtype),
    )
    mean_bc = broadcast_to(frag, mean_id, out.shape)
    xc_id = frag.add_node(
        op=ElementwiseOp(op="subtract"),
        inputs=[inp_x, mean_bc],
        output=Tensor(f"{out.name}_centered", out.shape, dtype),
    )
    sq_id = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[xc_id, xc_id],
        output=Tensor(f"{out.name}_sq", out.shape, dtype),
    )
    var_id = frag.add_node(
        op=MeanOp(axis=-1),
        inputs=[sq_id],
        output=Tensor(f"{out.name}_var", red_shape, dtype),
    )
    eps_bc = const_bc(frag, name=f"{out.name}_eps", value=root.op.eps, target_shape=red_shape, dtype=dtype)
    add_id = frag.add_node(
        op=ElementwiseOp(op="add"),
        inputs=[var_id, eps_bc],
        output=Tensor(f"{out.name}_add_eps", red_shape, dtype),
    )
    rsq_id = frag.add_node(
        op=ElementwiseOp(op="rsqrt"),
        inputs=[add_id],
        output=Tensor(f"{out.name}_rsq", red_shape, dtype),
    )
    rsq_bc = broadcast_to(frag, rsq_id, out.shape)
    last = f"{out.name}_f32" if widen else out.name
    cur_id = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[xc_id, rsq_bc],
        output=Tensor(last if inp_w is None and inp_b is None else f"{out.name}_norm", out.shape, dtype),
    )
    if inp_w is not None:
        w_bc = broadcast_to(frag, computed(inp_w, "w"), out.shape)
        cur_id = frag.add_node(
            op=ElementwiseOp(op="multiply"),
            inputs=[cur_id, w_bc],
            output=Tensor(f"{out.name}_scaled" if inp_b is not None else last, out.shape, dtype),
        )
    if inp_b is not None:
        b_bc = broadcast_to(frag, computed(inp_b, "b"), out.shape)
        cur_id = frag.add_node(
            op=ElementwiseOp(op="add"),
            inputs=[cur_id, b_bc],
            output=Tensor(last, out.shape, dtype),
        )
    if widen:
        cur_id = frag.add_node(op=ElementwiseOp(op="copy"), inputs=[cur_id], output=Tensor(out.name, out.shape, out.dtype))

    frag.outputs = [cur_id]
    return frag
