"""Decompose a dense Conv2dOp through im2col, the two-axis form of the dense ``conv1d`` rule.

One map builds the ``(N, KH * KW * C_in, H_out * W_out)`` window matrix. Its stacked row
``ck`` splits into ``tap = ck / C_in`` (then ``kh = tap / KW``, ``kw = tap % KW``) and
``ci = ck % C_in``, and its column into ``oh = l / W_out`` and ``ow = l % W_out``. The input
is read at ``(oh * stride + kh * dilation - padding, ow * stride + kw * dilation - padding)``
and a zero source fills what padding puts outside it. One map reorders the weight into the
same tap-major ``(C_out, KH * KW * C_in)`` layout, one ``MatmulOp`` contracts the two, and a
reshape restores the two output axes.
"""

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import ConstantOp
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, placeholder
from emmy.compiler.ir.frontend.ir import Conv2dOp, MatmulOp, ReshapeOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp, IndexMapOp, IndexSource
from emmy.compiler.pipeline import Match, Pattern
from emmy.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment, single_indexmap, static_extent

PATTERN = [Pattern("root", Conv2dOp)]


def rewrite(match: Match, root: Node, inp_x: Node, inp_w: Node, inp_bias: Node | None, out: Tensor) -> Graph | None:
    graph = match.graph
    op: Conv2dOp = root.op
    frag = open_fragment(graph, [inp_x, inp_w] + ([inp_bias] if inp_bias else []))

    out_channels, channels, kernel_h, kernel_w = (static_extent(d) for d in inp_w.output.shape)
    in_h, in_w = (static_extent(d) for d in inp_x.output.shape[2:])
    out_h, out_w = (static_extent(d) for d in out.shape[2:])
    batch = out.shape[0]
    taps = kernel_h * kernel_w

    def lit(value: int) -> Literal:
        return Literal(value, "int")

    def div(a, b: int):
        return BinaryExpr("/", a, lit(b))

    def mod(a, b: int):
        return BinaryExpr("%", a, lit(b))

    stacked, position = placeholder(1), placeholder(2)
    tap = div(stacked, channels)
    coords, conds = [], []
    for axis, (out_pos, kernel_pos, extent) in enumerate(
        ((div(position, out_w), div(tap, kernel_w), in_h), (mod(position, out_w), mod(tap, kernel_w), in_w))
    ):
        coord = BinaryExpr("+", BinaryExpr("*", out_pos, lit(op.stride[axis])), BinaryExpr("*", kernel_pos, lit(op.dilation[axis])))
        if op.padding[axis]:
            coord = BinaryExpr("-", coord, lit(op.padding[axis]))
            inside = BinaryExpr("&&", BinaryExpr(">=", coord, lit(0)), BinaryExpr("<", coord, lit(extent)))
            conds.append(inside)
            # Clamp the off-domain coord so the post-fusion unconditional Load stays in range.
            coord = TernaryExpr(cond=inside, if_true=coord, if_false=lit(0))
        coords.append(coord)
    col_shape = (batch, taps * channels, out_h * out_w)
    coord_map = (placeholder(0), mod(stacked, channels), *coords)
    if conds:
        in_bounds = conds[0] if len(conds) == 1 else BinaryExpr("&&", *conds)
        zero = frag.add_node(
            op=ConstantOp(name=f"{out.name}_zero", value=0.0), inputs=[], output=Tensor(f"{out.name}_zero", (1,), inp_x.output.dtype)
        )
        col = frag.add_node(
            op=IndexMapOp(
                out_shape=col_shape,
                sources=(IndexSource(input_idx=0, coord_map=coord_map, select=in_bounds), IndexSource(input_idx=1, coord_map=(lit(0),))),
            ),
            inputs=[inp_x, zero],
            output=Tensor(f"{out.name}_im2col", col_shape, inp_x.output.dtype),
        )
    else:
        col = single_indexmap(frag, inp_x, out_shape=col_shape, coord_map=coord_map, name=f"{out.name}_im2col")

    column = placeholder(1)
    flat_w = single_indexmap(
        frag,
        inp_w,
        out_shape=(out_channels, taps * channels),
        coord_map=[placeholder(0), mod(column, channels), div(div(column, channels), kernel_w), mod(div(column, channels), kernel_w)],
        name=f"{out.name}_w_flat",
    )
    flat_shape = (batch, out_channels, out_h * out_w)
    product = frag.add_node(op=MatmulOp(), inputs=[flat_w, col], output=Tensor(f"{out.name}_mm", flat_shape, out.dtype))
    acc = frag.add_node(
        op=ReshapeOp(shape=tuple(out.shape)),
        inputs=[product],
        output=Tensor(f"{out.name}_spatial" if inp_bias else out.name, tuple(out.shape), out.dtype),
    )
    if inp_bias:
        shaped = single_indexmap(frag, inp_bias, out_shape=tuple(out.shape), coord_map=[placeholder(1)], name=f"{out.name}_bias")
        acc = frag.add_node(op=ElementwiseOp(op="add"), inputs=[acc, shaped], output=Tensor(out.name, tuple(out.shape), out.dtype))

    frag.outputs = [acc]
    return frag
