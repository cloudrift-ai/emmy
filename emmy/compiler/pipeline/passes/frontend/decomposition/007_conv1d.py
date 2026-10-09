"""Decompose Conv1dOp into its two honest forms: im2col for dense, shifted taps for depthwise.

Every read of the input is one ``IndexMapOp`` whose length coordinate is
``l * stride + tap * dilation - padding``. Stride, dilation and padding are therefore all
paid for in the same expression, and nothing else in the rule has to know about them.
Padding specifically is a second source rather than a materialized zero-padded tensor: the
map selects the input where that coordinate is in range and a zero constant where it is
not, which is the same shape of trick ``150_cat`` uses and costs no extra buffer.

The two forms differ in what they do with those reads:

* **Dense** (``groups == 1``) builds the im2col matrix ``(N, C_in * K, L_out)`` in a single
  map — the stacked channel ``ck`` splits into ``tap = ck / C_in`` and ``ci = ck % C_in`` —
  and contracts it against the flattened weight with one ``MatmulOp``. A convolution then
  reaches exactly the GEMM path every other projection takes.
* **Depthwise** (``groups == C_in``) has no reduction across channels at all. im2col would
  build a ``(C_out, C_in * K)`` weight that is zero except for one band per channel, making
  the GEMM do ``C_in`` times the necessary work. It instead scales each of the ``K`` window
  reads by that tap's per-channel weight and sums them — pure elementwise, which also lets
  the chain fuse into its neighbours.

An f16 or bf16 convolution computes at f32 and converts to its output dtype once, at the last node,
the way ``torch.nn.functional.conv1d`` does. In the depthwise form the tap products and partial sums
are f32; in the dense form the ``MatmulOp`` result is f32 when a bias follows it. This is the rule
``matmul_decompose`` applies to a half-precision dot product. Rounding each partial sum to 16 bits
instead adds one rounding error per tap.
"""

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import BF16, F16, F32
from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.expr import BinaryExpr, Literal, placeholder
from emmy.compiler.ir.frontend.ir import Conv1dOp, MatmulOp, ReshapeOp, TransposeOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.pipeline import Match, Pattern
from emmy.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment, padded_read, single_indexmap, static_extent

PATTERN = [Pattern("root", Conv1dOp)]


def _length_at(tap: object, *, stride: int, dilation: int, padding: int) -> object:
    """``l * stride + tap * dilation - padding`` for a constant or computed ``tap``."""
    expr = placeholder(2)
    if stride != 1:
        expr = BinaryExpr("*", expr, Literal(stride, "int"))
    shift = tap if isinstance(tap, int) else None
    if shift is not None:
        offset = shift * dilation - padding
        return expr if offset == 0 else BinaryExpr("+", expr, Literal(offset, "int"))
    scaled = tap if dilation == 1 else BinaryExpr("*", tap, Literal(dilation, "int"))
    expr = BinaryExpr("+", expr, scaled)
    return expr if padding == 0 else BinaryExpr("-", expr, Literal(padding, "int"))


def rewrite(match: Match, root: Node, inp_x: Node, inp_w: Node, inp_bias: Node | None, out: Tensor) -> Graph | None:
    graph = match.graph
    op: Conv1dOp = root.op
    frag = open_fragment(graph, [inp_x, inp_w] + ([inp_bias] if inp_bias else []))

    taps = static_extent(inp_w.output.shape[-1])
    channels = static_extent(inp_x.output.shape[-2])
    # The input length is only needed to bound a padded read. Without padding every
    # coordinate is in range by construction of L_out, so a symbolic length is fine there.
    in_length = 0
    if op.padding:
        extent = inp_x.output.shape[-1]
        if isinstance(extent, Dim) and not extent.is_static:
            raise NotImplementedError(f"aten.conv1d with padding needs a static input length to bound the pad, got {extent}")
        in_length = static_extent(extent)
    out_shape = tuple(out.shape)
    acc_dtype = F32 if out.dtype in (F16, BF16) else out.dtype
    geometry = {"stride": op.stride, "dilation": op.dilation, "padding": op.padding}

    if op.groups == 1:
        # One map builds the whole im2col matrix: ck = tap * C_in + ci, tap-major.
        stacked = channels * taps
        stacked_coord = placeholder(1)
        tap_expr = BinaryExpr("/", stacked_coord, Literal(channels, "int"))
        col = padded_read(
            frag,
            inp_x,
            out_shape=(out_shape[0], stacked, out_shape[-1]),
            channel=BinaryExpr("%", stacked_coord, Literal(channels, "int")),
            length=_length_at(tap_expr, **geometry),
            in_length=in_length,
            name=f"{out.name}_im2col",
        )
        w_shape = tuple(static_extent(d) for d in inp_w.output.shape)
        w_t = frag.add_node(
            op=TransposeOp(axes=(0, 2, 1)),
            inputs=[inp_w],
            output=Tensor(f"{out.name}_w_t", (w_shape[0], w_shape[2], w_shape[1]), inp_w.output.dtype),
        )
        flat_w = frag.add_node(
            op=ReshapeOp(shape=(w_shape[0], stacked)),
            inputs=[w_t],
            output=Tensor(f"{out.name}_w_flat", (w_shape[0], stacked), inp_w.output.dtype),
        )
        acc: Node | str = frag.add_node(
            op=MatmulOp(),
            inputs=[flat_w, col],
            output=Tensor(f"{out.name}_mm", out_shape, acc_dtype) if inp_bias else Tensor(out.name, out_shape, out.dtype),
        )
    else:
        acc = None
        for tap in range(taps):
            window = padded_read(
                frag,
                inp_x,
                out_shape=out_shape,
                channel=placeholder(1),
                length=_length_at(tap, **geometry),
                in_length=in_length,
                name=f"{out.name}_win{tap}",
            )
            # out[n, c, l] scales by weight[c, 0, tap] — channel-indexed, constant in n and l.
            tap_w = single_indexmap(
                frag,
                inp_w,
                out_shape=out_shape,
                coord_map=[placeholder(1), Literal(0, "int"), Literal(tap, "int")],
                name=f"{out.name}_w{tap}",
            )
            scaled = frag.add_node(
                op=ElementwiseOp(op="multiply"),
                inputs=[window, tap_w],
                output=Tensor(f"{out.name}_scaled{tap}", out_shape, out.dtype if taps == 1 and not inp_bias else acc_dtype),
            )
            if acc is None:
                acc = scaled
                continue
            last = tap == taps - 1 and not inp_bias
            acc = frag.add_node(
                op=ElementwiseOp(op="add"),
                inputs=[acc, scaled],
                output=Tensor(out.name, out_shape, out.dtype) if last else Tensor(f"{out.name}_acc{tap}", out_shape, acc_dtype),
            )

    if inp_bias:
        shaped = single_indexmap(frag, inp_bias, out_shape=out_shape, coord_map=[placeholder(1)], name=f"{out.name}_bias")
        acc = frag.add_node(
            op=ElementwiseOp(op="add"),
            inputs=[acc, shaped],
            output=Tensor(out.name, out_shape, out.dtype),
        )

    frag.outputs = [acc.id if isinstance(acc, Node) else acc]
    return frag
