"""Decompose ConvTranspose1dOp in polyphase form: one stride-1 im2col GEMM, then an interleave.

Input position ``i`` scatters tap ``k`` to output position ``o = i * stride + k`` (before
padding crops it). Write ``o = q * stride + r`` and ``k = j * stride + r``: tap ``k`` lands on
phase ``r`` from input ``i = q - j``. So every phase is an ordinary stride-1 convolution over
the ``K / stride`` taps ``j``, and all phases share the same window reads:

* one map builds the im2col matrix ``(N, (K / stride) * C_in, Q)`` of ``x[n, ci, q - j]``,
  zero where ``q - j`` leaves the input, exactly as the dense ``conv1d`` builds its own;
* one map reorders the weight into ``(C_out * stride, (K / stride) * C_in)``, row ``co * stride
  + r`` holding tap ``j * stride + r`` of output channel ``co``;
* one ``MatmulOp`` contracts them, and a final map reads output ``o`` from row
  ``co * stride + (o + padding) % stride``, column ``(o + padding) / stride``.

The GEMM does exactly the convolution's multiply-adds. Inserting ``stride - 1`` zeros
between input positions and running a dense ``conv1d`` instead would multiply ``stride``
times as many, most of them by zero.
"""

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.expr import BinaryExpr, Literal, placeholder
from emmy.compiler.ir.frontend.ir import ConvTranspose1dOp, MatmulOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.pipeline import Match, Pattern
from emmy.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment, padded_read, single_indexmap, static_extent

PATTERN = [Pattern("root", ConvTranspose1dOp)]


def rewrite(match: Match, root: Node, inp_x: Node, inp_w: Node, inp_bias: Node | None, out: Tensor) -> Graph | None:
    graph = match.graph
    op: ConvTranspose1dOp = root.op
    frag = open_fragment(graph, [inp_x, inp_w] + ([inp_bias] if inp_bias else []))

    channels, out_channels, taps = (static_extent(d) for d in inp_w.output.shape)
    length = static_extent(inp_x.output.shape[-1])
    stride = op.stride
    if taps % stride:
        raise NotImplementedError(f"aten.conv_transpose1d needs a kernel that is a multiple of its stride, got {taps} and {stride}")
    phase_taps = taps // stride
    batch = out.shape[0]
    # Columns past the input's end still collect the last taps of the final positions.
    positions = length + phase_taps

    def lit(value: int) -> Literal:
        return Literal(value, "int")

    stacked = placeholder(1)
    col = padded_read(
        frag,
        inp_x,
        out_shape=(batch, phase_taps * channels, positions),
        channel=BinaryExpr("%", stacked, lit(channels)),
        length=BinaryExpr("-", placeholder(2), BinaryExpr("/", stacked, lit(channels))),
        in_length=length,
        name=f"{out.name}_im2col",
    )
    row, column = placeholder(0), placeholder(1)
    phase_w = single_indexmap(
        frag,
        inp_w,
        out_shape=(out_channels * stride, phase_taps * channels),
        coord_map=[
            BinaryExpr("%", column, lit(channels)),
            BinaryExpr("/", row, lit(stride)),
            BinaryExpr("+", BinaryExpr("*", BinaryExpr("/", column, lit(channels)), lit(stride)), BinaryExpr("%", row, lit(stride))),
        ],
        name=f"{out.name}_w_phase",
    )
    phases = frag.add_node(
        op=MatmulOp(),
        inputs=[phase_w, col],
        output=Tensor(f"{out.name}_phases", (batch, out_channels * stride, positions), out.dtype),
    )
    shifted = BinaryExpr("+", placeholder(2), lit(op.padding)) if op.padding else placeholder(2)
    acc = single_indexmap(
        frag,
        phases,
        out_shape=tuple(out.shape),
        coord_map=[
            placeholder(0),
            BinaryExpr("+", BinaryExpr("*", placeholder(1), lit(stride)), BinaryExpr("%", shifted, lit(stride))),
            BinaryExpr("/", shifted, lit(stride)),
        ],
        name=f"{out.name}_interleave" if inp_bias else out.name,
    )
    if inp_bias:
        shaped = single_indexmap(frag, inp_bias, out_shape=tuple(out.shape), coord_map=[placeholder(1)], name=f"{out.name}_bias")
        acc = frag.add_node(
            op=ElementwiseOp(op="add"),
            inputs=[acc, shaped],
            output=Tensor(out.name, tuple(out.shape), out.dtype),
        )

    frag.outputs = [acc.id if isinstance(acc, Node) else acc]
    return frag
