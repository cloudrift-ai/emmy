"""Unit tests for ``op_to_expression`` — elementwise op-name → Expression translation."""

import numpy as np
import pytest

from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expression import BinaryExpression, FunctionCallExpression, Literal, TernaryExpression
from emmy.compiler.ir.render import op_to_expression
from emmy.compiler.ir.stmt import dtype_promote


def test_square_renders_to_self_multiply():
    """``square`` lowers to ``x * x``.

    ReLU²/squared-ReLU MLPs (e.g. AFM-4.5B) emit a ``square`` elementwise op; the CUDA
    renderer previously had no case for it and raised ``NotImplementedError``.
    """
    x = Literal(3.0, "float")
    e = op_to_expression("square", [x])
    assert isinstance(e, BinaryExpression)
    assert e.op == "*"
    assert e.left == x and e.right == x


def test_copy_is_identity_passthrough():
    """``copy`` returns its input unchanged (dropout lowers through this path)."""
    x = Literal(1.0, "float")
    assert op_to_expression("copy", [x]) is x


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_zero_width_pad_is_an_exact_typed_identity(dtype):
    """Stored unary ``pad`` nodes represent only the zero-width frontend case."""
    x = Literal(1.0, "float")
    assert op_to_expression("pad", [x]) is x

    values = np.array([-1.5, 0.0, 2.25], dtype=dtype)
    actual = ElementwiseImpl("pad")(values)
    assert actual.dtype == values.dtype
    np.testing.assert_array_equal(actual, values)


def test_sin_cos_render_as_intrinsics():
    """DiT timestep embedding keeps its runtime trigonometric ops."""
    x = Literal(1.0, "float")
    for name in ("sin", "cos"):
        expression = op_to_expression(name, [x])
        assert isinstance(expression, FunctionCallExpression)
        assert expression.name == name


def test_mask_ops_render():
    """The explicit-mask subgraph's ops (comparisons, bool combines, where)
    translate — the gemma-4 whole-model tune previously died rendering
    ``equal``. ``bitwise_*`` carry bool-mask semantics and spell as logical
    ops (a float ``|`` would not compile on the f32-promoted operands)."""
    a, b = Literal(1.0, "float"), Literal(2.0, "float")
    for name, c_op in [
        ("equal", "=="),
        ("not_equal", "!="),
        ("greater", ">"),
        ("less", "<"),
        ("greater_equal", ">="),
        ("less_equal", "<="),
        ("bitwise_or", "||"),
        ("bitwise_and", "&&"),
    ]:
        e = op_to_expression(name, [a, b])
        assert isinstance(e, BinaryExpression) and e.op == c_op

    w = op_to_expression("where", [a, b, Literal(3.0, "float")])
    assert isinstance(w, TernaryExpression) and w.cond == a


def test_boolean_bitwise_not_is_logical_zero_test_and_integer_fails_closed():
    a = Literal(1.0, "float")
    for dtype in (None, "f32", "bool"):
        expression = op_to_expression("bitwise_not", [a], dtype=dtype)
        assert isinstance(expression, BinaryExpression) and expression.op == "=="

    import pytest

    with pytest.raises(NotImplementedError, match="boolean masks only"):
        op_to_expression("bitwise_not", [a], dtype="u32")


def test_integer_ops_keep_integer_spelling_without_changing_mask_semantics():
    """Integer bit manipulation uses C integer operators; f32 masks stay logical."""
    a, b = Literal(7, "int"), Literal(3, "int")
    for name, c_op in [
        ("floor_divide", "//"),
        ("remainder", "%"),
        ("left_shift", "<<"),
        ("right_shift", ">>"),
        ("bitwise_or", "|"),
        ("bitwise_and", "&"),
        ("bitwise_xor", "^"),
    ]:
        expression = op_to_expression(name, [a, b], dtype="u32")
        assert isinstance(expression, BinaryExpression) and expression.op == c_op

    assert op_to_expression("bitwise_and", [a, b]).op == "&&"
    assert op_to_expression("bitwise_or", [a, b]).op == "||"


def test_integer_elementwise_promotion_is_operation_specific():
    for op in ("add", "multiply", "floor_divide", "remainder", "left_shift", "bitwise_xor", "bitwise_count"):
        assert dtype_promote(op, ["u32", "u32"]) == "u32"
    assert dtype_promote("bitwise_or", ["u64", "u64"]) == "u64"
    assert dtype_promote("left_shift", ["u16", "u32"]) == "u32"
    assert dtype_promote("right_shift", ["u64", "i32"]) == "u64"
    assert dtype_promote("bitwise_and", ["i64", "u32"]) == "i64"
    assert dtype_promote("bitwise_and", ["f32", "f32"]) == "f32"
    assert dtype_promote("equal", ["i32", "i32"]) == "f32"


def test_integer_bitwise_count_stays_an_abstract_intrinsic():
    value = Literal(7, "int")
    expression = op_to_expression("bitwise_count", [value], dtype="u32")
    assert isinstance(expression, FunctionCallExpression) and expression.name == "bitwise_count" and expression.args == (value,)
