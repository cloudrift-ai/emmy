"""The affine normal form: every spelling of one sum, scale or difference is one body with one identity."""

from __future__ import annotations

from emmy.compiler.dtype import F32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Affine, Assign, Body, Let, Load, Loop, Select, SelectBranch, Write


def _op(*stmts) -> LoopOp:
    i = Var("i")
    return LoopOp(
        body=Body((Loop(Axis("i", 8), (Load(name="x", input="X", index=(i,)), *stmts, Write(output="O", index=(i,), value="o"))),))
    )


def _literal(name: str, value: float) -> Let:
    """A constant as a body spells it: a ``Let`` binding the literal at the default precision."""
    return Let(name, Literal(value), F32)


def _affines(op: LoopOp) -> list[tuple[tuple[float, ...], float]]:
    return [(tuple(sorted(s.coefficients)), s.offset) for s in op.body.iter() if isinstance(s, Affine)]


def test_every_spelling_of_a_scale_is_one_body() -> None:
    twice = _op(Assign("o", "add", ("x", "x")))
    scaled = _op(_literal("c", 2.0), Assign("o", "multiply", ("x", "c")))
    halved = _op(_literal("c", 0.5), Assign("o", "divide", ("x", "c")))
    assert twice.body == scaled.body == halved.body, twice.pretty_body()
    assert _affines(twice) == [((2.0,), 0.0)]
    assert not any(isinstance(s, Let) for s in scaled.body.iter()), "the literal the fold took is dropped"
    exact = {"structural": False}
    assert twice.identity_key(**exact) == scaled.identity_key(**exact) == halved.identity_key(**exact)
    assert twice.identity_key(**exact) != _op(_literal("c", 3.0), Assign("o", "multiply", ("x", "c"))).identity_key(**exact)


def test_a_division_by_a_literal_is_the_product_by_its_reciprocal() -> None:
    divided = _op(_literal("c", 4.0), Assign("o", "divide", ("x", "c")))
    assert _affines(divided) == [((0.25,), 0.0)]
    assert divided.body == _op(_literal("c", 0.25), Assign("o", "multiply", ("x", "c"))).body
    assert _affines(_op(_literal("c", 1.0), Assign("o", "divide", ("x", "c")))) == [], "a division by one is a copy"


def test_like_terms_sum_and_cancel() -> None:
    op = _op(_literal("c", 3.0), Assign("t", "multiply", ("x", "c")), Assign("s", "add", ("t", "x")), Assign("o", "subtract", ("s", "x")))
    assert _affines(op) == [((3.0,), 0.0)]
    zero = _op(Assign("o", "subtract", ("x", "x")))
    assert _affines(zero) == []
    (constant,) = [s for s in zero.body.iter() if isinstance(s, Let)]
    assert constant.value == Literal(0.0) and constant.name == zero.body.writes[0].value, zero.pretty_body()


def test_a_select_whose_branches_bind_one_value_is_that_value() -> None:
    i = Var("i")
    op = _op(Select("o", (SelectBranch("x", i.lt(Var("i"))), SelectBranch("x", Var("i").lt(Var("i"))))))
    assert not any(isinstance(s, Select) for s in op.body.iter()), op.pretty_body()
    assert op.body.writes[0].value == op.body.loads[0].name
