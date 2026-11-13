"""Generic integer elementwise typing and CUDA statement rendering."""

from __future__ import annotations

from importlib import import_module

import pytest

from emmy.compiler.dtype import U32
from emmy.compiler.graph import Node, Tensor
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.render import RenderCtx
from emmy.compiler.ir.stmt import Statement, parse_body
from tests.compiler.helpers import render_typed


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
    line = render_typed([_stmt(f"result = {op}(a, b)")], ctx)[0]
    assert line == f"    unsigned int result = a {symbol} b;"
    assert ctx.ssa_dtypes["result"] == "u32"


def test_a_float_literal_over_integer_operands_renders_a_float_operation():
    """The normal form spells a float constant inline where a typed binding stood (``let f32 half = 0.5`` read by
    ``multiply(half, code)``), so a float literal over integer operands keeps the operation a float one; an integer
    literal still contributes no dtype."""
    ctx = RenderCtx(ssa_dtypes={"a": "i32"})
    assert render_typed([_stmt("v = multiply(0.5, a)")], ctx) == ["    float a__f32 = a;", "    float v = 0.5f * a__f32;"]
    assert ctx.ssa_dtypes["v"] == "f32"
    ctx = RenderCtx(ssa_dtypes={"a": "i32"})
    assert render_typed([_stmt("v = maximum(0.0, a)")], ctx)[-1].startswith("    float v = ")
    ctx = RenderCtx(ssa_dtypes={"a": "i32"})
    assert render_typed([_stmt("v = multiply(2, a)")], ctx)[0].startswith("    int v = ")


def test_a_whole_number_float_literal_keeps_its_float_spelling_in_the_normal_form():
    """The normal form inlines a float constant and spells its sum in affine form; a whole-number coefficient or offset
    is an integer only when every literal behind it was one, so ``code - eight`` with ``f32 eight = 8.0`` stays
    ``in0 - 8.0`` — a float operation over the u8 — and never ``in0 - 8``, which the stamp would type at the code's
    width and compute as 251 where torch computes -5.0."""
    from emmy.compiler.ir.stmt import normalize_body

    shifted = normalize_body(parse_body("u8 code = load C[0]\nf32 eight = 8.0\nv = subtract(code, eight)\nO[0] = v"))
    assert "v0 = in0 - 8.0" in str(shifted)
    scaled = normalize_body(parse_body("u8 code = load C[0]\nv = multiply(2.0, code)\nO[0] = v"))
    assert "v0 = 2.0 * in0" in str(scaled)
    doubled = normalize_body(parse_body("u8 code = load C[0]\nv = add(code, code)\nO[0] = v"))
    assert "v0 = 2 * in0" in str(doubled)
    # A sum that would spell no literal keeps one: ``code * 1.0`` squared is a float, never ``u8 v = in0 * in0``.
    collapsed = normalize_body(parse_body("u8 code = load C[0]\nv = multiply(code, 1.0)\nw = multiply(v, v)\nO[0] = w"))
    assert "v0 = 1.0 * in0\n" in str(collapsed) and "v1 = v0 * v0" in str(collapsed)
    ctx = RenderCtx(ssa_dtypes={"in0": "u8"})
    assert render_typed([_stmt("v0 = in0 - 8.0")], ctx) == ["    float in0__f32 = in0;", "    float v0 = in0__f32 - 8.0f;"]
    assert ctx.ssa_dtypes["v0"] == "f32"


def test_f32_bitwise_names_remain_logical_masks():
    ctx = RenderCtx(ssa_dtypes={"a": "f32", "b": "f32"})
    assert render_typed([_stmt("both = bitwise_and(a, b)")], ctx) == ["    float both = a && b;"]
    assert render_typed([_stmt("either = bitwise_or(a, b)")], ctx) == ["    float either = a || b;"]


@pytest.mark.parametrize(
    "op",
    ["add", "subtract", "multiply", "floor_divide", "remainder", "left_shift", "right_shift", "bitwise_and", "bitwise_or", "bitwise_xor"],
)
def test_stamp_types_preserves_integer_assign_and_write(op):
    lines = _stamped(f"result = {op}(a, b)")
    assert f"u32 result = a {_SYMBOL[op]} b" in lines
    assert "u32 out[0] = result" in lines


def test_stamp_types_keeps_a_declared_integer_width_on_integer_algebra():
    """A narrower integer dtype declared on integer algebra is the result's width — the normal form spells a narrowing
    as the dtype of the definition it narrows, where a separate ``copy`` once did — and the stamp keeps it."""
    assert "u16 result = a + b" in _stamped("u16 result = add(a, b)")


def test_stamp_types_keeps_a_declared_float_dtype_on_integer_algebra():
    """A declared dtype is the result type: the shift computes at its operands' u32 and converts once, as torch
    converts the integer result — never ``(float)a << (float)b``, which has no float spelling, nor a re-stamp to u32
    that would run an all-integer consumer at u32 where torch's float path does not overflow."""
    assert "f32 result = a << b" in _stamped("f32 result = left_shift(a, b)")
    ctx = RenderCtx()
    ctx.ssa_dtypes.update(a="u32", b="u32")
    lines = render_typed([_stmt("f32 result = left_shift(a, b)")], ctx)
    assert lines == ["    unsigned int result__u32 = a << b;", "    float result = result__u32;"]
