"""``FragmentApply``'s coordinate and global-memory operand kinds — the fragment-tier mask and bias."""

from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import COORD, FRAG, FRAG_COL, FRAG_ROW, GMEM, UNIFORM, FragmentApply
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import RenderCtx


def _causal_mask() -> FragmentApply:
    return FragmentApply(
        out="s",
        op=ElementwiseImpl("where"),
        args=(BinaryExpr(">", Var(FRAG_COL), Var(FRAG_ROW)), Literal(-1e30), "s"),
        kinds=(COORD, UNIFORM, FRAG),
        in_place=True,
        row_base=Var("q"),
        col_base=BinaryExpr("+", Var("kv"), Literal(8, "int")),
    )


def test_coordinate_mask_renders_one_guarded_line_per_element() -> None:
    lines = _causal_mask().render(RenderCtx())
    stores = [ln.strip() for ln in lines if ln.strip().startswith("s[")]
    assert len(stores) == 4, "every element of the m16n8 fragment is masked at its own coordinates"
    assert stores[0] == "s[0] = ((kv + 8 + (_t * 2 + 0) > q + _g) ? (-1e+30f) : (s[0]));"
    assert "_g + 8" in stores[2], "the second row of the lane's pair sits eight rows down"
    assert not any("#pragma unroll" in ln for ln in lines), "a coordinate operand differs per element, so no loop"


def test_gmem_operand_reads_the_buffer_at_absolute_coordinates_and_converts() -> None:
    bias = FragmentApply(
        out="s",
        op=ElementwiseImpl("add"),
        args=("s", ("mask", (Literal(0, "int"), Var(FRAG_ROW), Var(FRAG_COL)))),
        kinds=(FRAG, GMEM),
        in_place=True,
        row_base=Var("q"),
        col_base=Var("kv"),
    )
    ctx = RenderCtx(shapes={"mask": (1, 64, 64)}, buffer_dtypes={"mask": "f16"})
    lines = bias.render(ctx)
    assert bias.external_reads() == ("mask",)
    assert bias.rename_buffers({"mask": "m0"}).external_reads() == ("m0",)
    assert "s[3] = s[3] + __half2float(mask[(q + (_g + 8)) * 64 + (kv + (_t * 2 + 1))]);" in [ln.strip() for ln in lines]


def test_rewrite_substitutes_bases_and_templates_but_not_the_reserved_coordinates() -> None:
    sigma = Sigma({"q": Var("a1"), "kv": Var("a3")})
    rewritten = _causal_mask().rewrite(lambda n: {"s": "s9"}.get(n, n), sigma)
    assert rewritten.out == "s9" and rewritten.args[2] == "s9"
    assert rewritten.row_base == Var("a1")
    assert rewritten.args[0] == BinaryExpr(">", Var(FRAG_COL), Var(FRAG_ROW)), "the reserved vars are not local axes"
    assert rewritten.deps() == ("s9",), "a predicate and a literal contribute no SSA reads"
    assert rewritten.exprs()[0] == rewritten.args[0]


def test_plain_operands_keep_the_unrolled_loop_form() -> None:
    scale = FragmentApply(out="s", op=ElementwiseImpl("multiply"), args=("s", Literal(0.125)), kinds=(FRAG, UNIFORM), in_place=True)
    lines = scale.render(RenderCtx())
    assert any("#pragma unroll" in ln for ln in lines)
    assert any("s[_e] = s[_e] * 0.125f;" in ln for ln in lines)
