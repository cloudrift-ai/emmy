"""Register epilogues retain the Loop tail's per-Assign dtype semantics."""

from emmy.compiler.dtype import F16
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.kernel.ir import RegStore
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Assign, RenderCtx, Write
from emmy.compiler.pipeline.passes.lowering.kernel._atom import _warp_epilogue


def _render(*assigns) -> tuple[str, object]:
    tail = [*assigns, Write(output="out", index=(Var("m"), Var("n")), value=assigns[-1].name)]
    epilogue = _warp_epilogue(tail, "acc", "m", "n", Sigma.IDENTITY)
    assert epilogue is not None
    store = RegStore(
        dst_buffer="out",
        dst_index=(Var("m"), Var("n")),
        frag="_c",
        shape=(16, 8, 16),
        ldm=16,
        epilogue=epilogue,
    )
    ctx = RenderCtx(shapes={"out": (16, 16)}, buffer_dtypes={"out": "f16"})
    return "\n".join(store.render(ctx)), epilogue


def test_typed_copy_narrows_the_fragment_value_before_its_consumer() -> None:
    source, epilogue = _render(
        Assign(name="narrow", op=ElementwiseImpl("copy"), args=("acc",), dtype=F16),
        Assign(name="result", op=ElementwiseImpl("copy"), args=("narrow",)),
    )

    assert epilogue.ops[0][3] is F16
    assert "const __half narrow_e0 = __float2half(_c[0]);" in source
    assert "const __half result_e0 = narrow_e0;" in source


def test_untyped_copy_keeps_the_existing_f32_epilogue() -> None:
    source, epilogue = _render(Assign(name="result", op=ElementwiseImpl("copy"), args=("acc",)))

    assert epilogue.ops[0][3] is None
    assert "const float result_e0 = _c[0];" in source
    assert "__float2half(_c[0])" not in source


def test_transposed_fragment_store_uses_both_output_strides() -> None:
    store = RegStore(
        dst_buffer="out",
        dst_index=(Var("n"), Var("m")),
        frag="_c",
        shape=(16, 8, 16),
        row_dim=1,
        col_dim=0,
    )
    source = "\n".join(store.render(RenderCtx(shapes={"out": (16, 16)}, buffer_dtypes={"out": "f16"})))

    assert "reinterpret_cast<__half2*>" not in source
    assert "(_t * 2 + 1) * 16" in source
    assert "out[n * 16 + m + _g + (_t * 2 + 0) * 16]" in source


def test_a_mask_over_an_op_result_renders_after_that_op() -> None:
    """A causal mask may select a value the chain computes (a scaled score), not only a loaded one:
    the ternary renders once that op has run."""
    from emmy.compiler.ir.expr import BinaryExpr, Literal
    from emmy.compiler.ir.stmt import Load, Select, SelectBranch

    source, _ = _render(
        Load(name="ninf", input="neg_inf", index=(Literal(0, "int"),)),
        Assign(name="scaled", op=ElementwiseImpl("multiply"), args=("acc", "acc")),
        Select(
            name="masked",
            branches=(SelectBranch("scaled", BinaryExpr("<=", Var("n"), Var("m"))), SelectBranch("ninf", Literal(1, "int"))),
        ),
        Assign(name="result", op=ElementwiseImpl("copy"), args=("masked",)),
    )

    assert source.index("scaled_e0 =") < source.index("masked_e0 = ((") < source.index("result_e0 = masked_e0")


def test_a_mask_over_a_narrowed_value_widens_both_branches() -> None:
    """A chain op keeps the tail's dtype, so a mask over a narrowed value converts back: a ternary
    mixing ``__half`` and ``float`` does not compile."""
    from emmy.compiler.ir.expr import BinaryExpr, Literal
    from emmy.compiler.ir.stmt import Load, Select, SelectBranch

    source, _ = _render(
        Load(name="ninf", input="neg_inf", index=(Literal(0, "int"),)),
        Assign(name="narrow", op=ElementwiseImpl("copy"), args=("acc",), dtype=F16),
        Select(
            name="masked",
            branches=(SelectBranch("narrow", BinaryExpr("<=", Var("n"), Var("m"))), SelectBranch("ninf", Literal(1, "int"))),
        ),
        Assign(name="result", op=ElementwiseImpl("copy"), args=("masked",)),
    )

    assert "const __half narrow_e0 = __float2half(_c[0]);" in source
    assert "const float masked_e0 = ((n + (_t * 2 + 0) <= m + _g) ? __half2float(narrow_e0) : ninf_e0);" in source
