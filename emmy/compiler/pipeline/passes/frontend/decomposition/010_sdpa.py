"""Decompose scaled_dot_product_attention into QK^T → scale [→ mask] → softmax → @V. A causal or a sliding-window
attention folds each query's keys over a range of the key axis; an explicit mask operand is added to the scores.

For GQA (Grouped Query Attention) where Q has more heads than K/V, an
explicit IndexMapOp is inserted on K and V to broadcast the head dim
via integer-divide indexing: ``K[b, q_head // group_size, s, d]``.
"""

import math

from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.expression import Literal, apply, placeholder
from emmy.compiler.ir.frontend.ir import SdpaOp, TransposeOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.pipeline import Match, Pattern
from emmy.compiler.pipeline.passes.frontend.decomposition._helpers import (
    broadcast_to,
    const_bc,
    gqa_broadcast,
    matmul_decompose,
    open_fragment,
    softmax_decompose,
)

PATTERN = [Pattern("root", SdpaOp)]


def _maybe_gqa(frag: Graph, src: Node | str, q_batch: tuple, src_batch: tuple, target_last_dims: tuple, *, name: str) -> Node | str:
    """Broadcast src's head axis to match q's head count via integer-divide indexing.

    Returns ``src`` unchanged when there is no GQA mismatch. Head axis is the last
    batch dim on each side; ranks may differ (V's prefix is preserved).
    """
    if not (q_batch and src_batch):
        return src
    q_heads = q_batch[-1].as_static() if q_batch[-1].is_static else None
    s_heads = src_batch[-1].as_static() if src_batch[-1].is_static else None
    if not (q_heads and s_heads and q_heads > s_heads and q_heads % s_heads == 0):
        return src
    head_axis = len(src_batch) - 1
    target_shape = tuple(src_batch[:head_axis]) + (q_heads,) + tuple(target_last_dims)
    return gqa_broadcast(
        frag,
        src,
        target_shape=target_shape,
        head_axis=head_axis,
        group_size=q_heads // s_heads,
        name=name,
    )


def rewrite(match: Match, root: Node, inp_q: Node, inp_k: Node, inp_v: Node, inp_mask: Node | None, out: Tensor) -> Graph | None:
    graph = match.graph
    q_shape = inp_q.output.shape
    k_shape = inp_k.output.shape
    v_shape = inp_v.output.shape
    dtype, name = out.dtype, out.name

    head_dim = q_shape[-1] if len(q_shape) >= 2 else 64
    seq_len = q_shape[-2] if len(q_shape) >= 3 else q_shape[-1]
    # The key length is K's own, not the query's: a decode step (one query over a populated
    # cache) scores every key, and a kt/scores shape read off the query would score only key 0.
    kv_len = k_shape[-2] if len(k_shape) >= 3 else k_shape[-1]
    q_batch = q_shape[:-2] if len(q_shape) > 2 else ()
    k_batch = k_shape[:-2] if len(k_shape) > 2 else ()
    v_batch = v_shape[:-2] if len(v_shape) > 2 else ()
    scores_shape = q_batch + (seq_len, kv_len)

    exts = [inp_q, inp_k, inp_v] + ([inp_mask] if inp_mask is not None else [])
    frag = open_fragment(graph, exts)

    # K^T then GQA broadcast.
    kt_shape = k_batch + (head_dim, kv_len) if head_dim.is_static else k_shape
    kt_id = frag.add_node(
        op=TransposeOp(axes=(-2, -1)),
        inputs=[inp_k],
        output=Tensor(f"{name}_kt", kt_shape, inp_k.output.dtype),
    )
    kt = _maybe_gqa(frag, kt_id, q_batch, k_batch, (head_dim, kv_len), name=f"{name}_kt_gqa")

    # QK^T matmul.
    qk = matmul_decompose(frag, inp_q, kt, name=f"{name}_qk")

    # Scale: the op's captured ``scale=`` kwarg, else torch's 1/sqrt(head_dim) default.
    op_scale = getattr(root.op, "scale", None)  # pre-scale-field IR dumps lack the attr
    if op_scale is not None:
        scale_value = op_scale
    else:
        scale_value = 1.0 / math.sqrt(head_dim.as_static()) if head_dim.is_static else None
    # Scores and their softmax are f32 whatever the operand dtype, and so is the scale: torch applies
    # it at f32 (the flash kernels' float ``softmax_scale``, the math path's upcast). Stored at f16,
    # 1/sqrt(128) lost 1e-4 of every logit and moved a sixth of the attention outputs one f16 step
    # off eager. Only P.V's result takes the output dtype.
    scale_bc = const_bc(frag, name=f"{name}_scale", value=scale_value, target_shape=scores_shape, dtype=F32)
    scaled_id = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[qk, scale_bc],
        output=Tensor(f"{name}_scaled", scores_shape, F32),
    )

    # Explicit additive mask: scores += attn_mask (broadcast to scores_shape).
    # HF passes its causal mask as a (1,1,S,S) float bias (0 / -inf) rather
    # than via is_causal, so this is the path the whole-model trace takes.
    if inp_mask is not None:
        mask_bc = broadcast_to(frag, inp_mask.id, scores_shape)
        scaled_id = frag.add_node(
            op=ElementwiseOp(op="add"),
            inputs=[scaled_id, mask_bc],
            output=Tensor(f"{name}_masked", scores_shape, F32),
        )

    # The keys a query folds, as the range of the key axis over the query's row: causal attention folds the keys
    # at or before the row, a sliding window the last ``window`` of them. The softmax's two reductions and the
    # product with V walk that range, so a key outside it is no part of the row's distribution. A stamped SDPA
    # states the range beside an explicit bias (the bias may mask more, e.g. padding).
    window = root.op.sliding_window

    def keys(row):
        """The range of the key axis for the query at ``row``, or ``None`` when every key is folded."""
        start, end = Literal(0, "int"), kv_len.expression if not kv_len.is_static else Literal(kv_len.as_static(), "int")
        if root.op.is_causal:
            # A query past the last key (more queries than keys) still ends at the last key.
            end = row + Literal(1, "int") if seq_len == kv_len else apply("minimum", row + Literal(1, "int"), end)
        # A window no shorter than a static sequence leaves every key in.
        if window is not None and not (seq_len.is_static and window >= seq_len.as_static()):
            start = apply("maximum", start, row - Literal(window - 1, "int"))
        return (start, end) if root.op.is_causal or start != Literal(0, "int") else None

    ndim_scores = len(scores_shape)
    softmax = softmax_decompose(frag, scaled_id, -1, name=f"{name}_softmax", range=keys(placeholder(ndim_scores - 2)))

    # Softmax @ V (with GQA on V).
    v_last = v_shape[-2:] if len(v_shape) >= 2 else v_shape
    v_eff = _maybe_gqa(frag, inp_v, q_batch, v_batch, v_last, name=f"{name}_v_gqa")
    sv = matmul_decompose(frag, softmax, v_eff, name=name, dtype=dtype, range=keys)

    frag.outputs = [sv.id]
    return frag
