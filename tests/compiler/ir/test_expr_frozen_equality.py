"""Lock in that ``Expression`` nodes are frozen + structurally hashable.

Required so ``Dim`` (which will carry an ``Expression``) keeps a stable
``__hash__`` across construction paths and survives use as a dict key /
set member in ``Axis`` and graph structural keys.
"""

from __future__ import annotations

import pytest

from emmy.compiler.ir.expression import (
    BinaryExpression,
    Builtin,
    CastExpression,
    FunctionCallExpression,
    Literal,
    TernaryExpression,
    Var,
)


def test_var_structural_equality():
    assert Var("x") == Var("x")
    assert Var("x") != Var("y")
    assert hash(Var("x")) == hash(Var("x"))


def test_literal_structural_equality():
    assert Literal(1, "int") == Literal(1, "int")
    assert Literal(1, "int") != Literal(1, "float")
    assert hash(Literal(1, "int")) == hash(Literal(1, "int"))


def test_binary_expression_structural_equality():
    e1 = BinaryExpression("+", Var("a"), Literal(2, "int"))
    e2 = BinaryExpression("+", Var("a"), Literal(2, "int"))
    assert e1 == e2
    assert hash(e1) == hash(e2)


def test_funccallexpr_structural_equality():
    f1 = FunctionCallExpression("exp", (Var("x"),))
    f2 = FunctionCallExpression("exp", (Var("x"),))
    assert f1 == f2
    assert hash(f1) == hash(f2)


def test_ternary_and_cast_hashable():
    t = TernaryExpression(Var("c"), Var("a"), Var("b"))
    assert hash(t) == hash(TernaryExpression(Var("c"), Var("a"), Var("b")))
    c = CastExpression("int", Var("x"))
    assert hash(c) == hash(CastExpression("int", Var("x")))


def test_builtin_hashable():
    assert Builtin("thread_idx.x") == Builtin("thread_idx.x")
    assert hash(Builtin("thread_idx.x")) == hash(Builtin("thread_idx.x"))


def test_var_is_frozen():
    v = Var("x")
    with pytest.raises(AttributeError):
        v.name = "y"  # type: ignore[misc]


def test_binary_expression_is_frozen():
    e = BinaryExpression("+", Var("a"), Literal(1, "int"))
    with pytest.raises(AttributeError):
        e.op = "-"  # type: ignore[misc]


def test_funccallexpr_args_is_tuple():
    f = FunctionCallExpression("exp", (Var("x"),))
    assert isinstance(f.args, tuple)


def test_expression_works_as_dict_key():
    """Used to key ``SimplifyCtx.ranges``, hash-cache equivalent subexpressions, etc."""
    d: dict[object, int] = {}
    d[BinaryExpression("+", Var("a"), Literal(1, "int"))] = 7
    assert d[BinaryExpression("+", Var("a"), Literal(1, "int"))] == 7
