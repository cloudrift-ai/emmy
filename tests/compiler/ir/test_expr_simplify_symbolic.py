"""The Python side of the symbolic simplifier: ``extend_simplify_ctx`` registers an axis's sign and bound, and a
``Sigma`` composes with the simplifier. The folds themselves are tested in the crate."""

from __future__ import annotations

from emmy.compiler.ir.axis import Axis, extend_simplify_ctx
from emmy.compiler.ir.expression import BinaryExpression, Literal, SimplifyCtx, Var


def test_symbolic_axis_gets_nonneg_range():
    """A symbolic-extent axis must register ``lo == 0`` (the non-negativity the
    ``(i*c + …)//c → i`` div fold relies on) plus its extent as the bound."""
    ctx = extend_simplify_ctx(SimplifyCtx.empty(), Axis("a0", "seq_len"))
    assert ctx.ranges["a0"].lo == 0
    assert ctx.bounds["a0"] == Var("seq_len")


def test_segmented_k_collapses_oproj_delinearized_read():
    """The segmented-K mechanism: a C-aligned split of the flat matmul K plus
    range-aware simplify collapses the o_proj attn-out delinearized read to a
    clean ``[head, seq, within]`` access — no div/mod, no producer needed.

    Real o_proj A-operand index (flat K = a2, over physical [16, seq, 128]):
        head = (a0*2048 + a2)/128 % 16 ,  seq = (a0*2048 + a2)/2048 % seq_len ,
        dim  = (a0*2048 + a2) % 128
    Split a2 = a2_o*128 + a2_i (a2_o = head 0..15, a2_i = within 0..127); with
    axis ranges/bounds the three coords fold to a2_o / a0 / a2_i."""
    from emmy.compiler.ir.sigma import Sigma

    a0, a2 = Var("a0"), Var("a2")
    flat = BinaryExpression("+", BinaryExpression("*", a0, Literal(2048, "int")), a2)
    head = BinaryExpression("%", BinaryExpression("//", flat, Literal(128, "int")), Literal(16, "int"))
    seq = BinaryExpression("%", BinaryExpression("//", flat, Literal(2048, "int")), Var("seq_len"))
    dim = BinaryExpression("%", flat, Literal(128, "int"))

    seg = Sigma({"a2": BinaryExpression("+", BinaryExpression("*", Var("a2_o"), Literal(128, "int")), Var("a2_i"))})
    ctx = SimplifyCtx.empty()
    ctx = extend_simplify_ctx(ctx, Axis("a0", "seq_len"))
    ctx = extend_simplify_ctx(ctx, Axis("a2_o", 16))
    ctx = extend_simplify_ctx(ctx, Axis("a2_i", 128))

    assert seg.apply(head).simplify(ctx) == Var("a2_o")  # head, no div/mod
    assert seg.apply(seq).simplify(ctx) == Var("a0")  # seq, no % seq_len
    assert seg.apply(dim).simplify(ctx) == Var("a2_i")  # contiguous inner K
