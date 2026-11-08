"""Generic integer elementwise typing and CUDA statement rendering."""

from __future__ import annotations

from importlib import import_module

import pytest

from emmy.compiler.dtype import U32
from emmy.compiler.graph import Node, Tensor
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.render import RenderCtx, render_stmt
from emmy.compiler.ir.stmt import Statement, parse_body


def _stmt(text: str) -> Statement:
    (stmt,) = parse_body(text)
    return stmt


def _stamped(result: str) -> list[str]:
    """The lines of ``out[0] = result`` over two ``u32`` buffers once the kernel's types are stamped."""
    body = parse_body(
        f"""
        a = load lhs[0]
        b = load rhs[0]
        {result}
        out[0] = result
        """
    )
    out = Tensor("out", (1,), U32)
    kernel = KernelOp(
        body=body,
        name="integer_elementwise",
        inputs={"lhs": Tensor("lhs", (1,), U32), "rhs": Tensor("rhs", (1,), U32)},
        outputs={"out": out},
    )
    root = Node(id="out", op=kernel, inputs=["lhs", "rhs"], outputs=(out,))
    stamped = import_module("emmy.compiler.pipeline.passes.lowering.kernel.030_stamp_types").rewrite(root)
    assert stamped is not None
    return str(stamped.body).splitlines()


# The infix each integer operation is spelled by in a body's text and in C.
_SYMBOL = {
    "add": "+",
    "subtract": "-",
    "multiply": "*",
    "floor_divide": "//",
    "remainder": "%",
    "left_shift": "<<",
    "right_shift": ">>",
    "bitwise_and": "&",
    "bitwise_or": "|",
    "bitwise_xor": "^",
}


@pytest.mark.parametrize(
    ("op", "symbol"),
    [
        ("floor_divide", "/"),
        ("remainder", "%"),
        ("left_shift", "<<"),
        ("right_shift", ">>"),
        ("bitwise_and", "&"),
        ("bitwise_or", "|"),
        ("bitwise_xor", "^"),
    ],
)
def test_integer_assign_renders_natively(op, symbol):
    ctx = RenderCtx(ssa_dtypes={"a": "u32", "b": "u32"})
    line = render_stmt(_stmt(f"result = {op}(a, b)"), ctx)[0]
    assert line == f"    unsigned int result = a {symbol} b;"
    assert ctx.ssa_dtypes["result"] == "u32"


def test_f32_bitwise_names_remain_logical_masks():
    ctx = RenderCtx(ssa_dtypes={"a": "f32", "b": "f32"})
    assert render_stmt(_stmt("both = bitwise_and(a, b)"), ctx) == ["    float both = a && b;"]
    assert render_stmt(_stmt("either = bitwise_or(a, b)"), ctx) == ["    float either = a || b;"]


@pytest.mark.parametrize(
    "op",
    ["add", "subtract", "multiply", "floor_divide", "remainder", "left_shift", "right_shift", "bitwise_and", "bitwise_or", "bitwise_xor"],
)
def test_stamp_types_preserves_integer_assign_and_write(op):
    lines = _stamped(f"result = {op}(a, b)")
    assert f"u32 result = a {_SYMBOL[op]} b" in lines
    assert "u32 out[0] = result" in lines


def test_stamp_types_repairs_a_stale_float_stamp_on_integer_algebra():
    assert "u32 result = a << b" in _stamped("f32 result = left_shift(a, b)")


def test_an_integer_constant_load_keeps_its_dtype_when_it_folds_to_a_literal():
    """A one-element constant buffer renders as a literal binding rather than a buffer read. An
    INTEGER one must keep its own dtype: the binding's SSA dtype decides the CONSUMER's arithmetic,
    and a shift amount folded to f32 sends its shift down the f32 path, which has no `<<`."""
    ctx = RenderCtx(literal_constants={"shift_c": 4.0})
    assert render_stmt(_stmt("u32 shift = load shift_c[0]"), ctx) == ["    unsigned int shift = 4;"]
    assert ctx.ssa_dtypes["shift"] == "u32"
    ctx.ssa_dtypes["odd"] = "u32"
    assert render_stmt(_stmt("hi = left_shift(odd, shift)"), ctx) == ["    unsigned int hi = odd << shift;"]


def test_a_float_constant_load_still_binds_as_a_float_literal():
    """The float case is untouched — every kernel reading a scalar float constant renders as before."""
    ctx = RenderCtx(literal_constants={"scale_c": 0.125})
    assert render_stmt(_stmt("f32 scale = load scale_c[0]"), ctx) == ["    float scale = 0.125f;"]
    assert ctx.ssa_dtypes["scale"] == "f32"


def test_the_literal_prefold_leaves_an_integer_constant_named():
    """`render_body`'s pre-pass drops a literal-constant Load and inlines its value at every use
    site — as a FLOAT literal, since that is what its map holds. An integer constant stays out of
    it and keeps its named local, or the bit operation reading it lands back in f32."""
    from emmy.compiler.ir.render import render_body

    ctx = RenderCtx(literal_constants={"shift_c": 4.0, "scale_c": 0.125})
    body = parse_body(
        """
        u32 shift = load shift_c[0]
        f32 scale = load scale_c[0]
        hi = left_shift(odd, shift)
        lo = multiply(odd, scale)
        """
    )
    ctx.ssa_dtypes["odd"] = "u32"
    lines = render_body(body, ctx)
    assert "    unsigned int shift = 4;" in lines, lines
    assert not any("scale = 0.125f" in ln for ln in lines), "a float constant still inlines at its use site"
    assert any("odd << shift" in ln for ln in lines), lines
