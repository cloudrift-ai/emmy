"""Fast-math reciprocal hoisting preserves types and leaves structural normalization pure."""

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest

from emmy.compiler.dtype import BF16, F16, F32, F64, I32
from emmy.compiler.ir.expression import Literal, Var
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Assign, Load, Loop, normalize_body, parse_body
from emmy.compiler.pipeline import RuleSkipped

rule = import_module("emmy.compiler.pipeline.passes.lowering.kernel.040_split_invariant_divides")


def _names(stmt) -> tuple[str, ...]:
    """The names an ``Assign``'s value reads as direct operands (a bare-name value is its one operand)."""
    operands = (stmt.value,) if isinstance(stmt.value, Var) else stmt.value.operands
    return tuple(operand.name for operand in operands)


def _body(dtype=F32, *, varying=False, divide="divide"):
    return parse_body(
        f"""
        for i in 0..4:
            {dtype.name} recip_0 = load x[i]
            {dtype.name} denominator = load scale[{"i" if varying else ""}]
            {dtype.name} out = {divide}(recip_0, denominator)
            {dtype.name} y[i] = out
        """
    )


@pytest.mark.parametrize("dtype", [F16, BF16, F32, F64])
@pytest.mark.parametrize(("umbrella", "division"), [(None, None), ("1", None), ("0", None), ("1", "0"), ("0", "1")])
def test_reciprocal_hoisting_follows_the_division_knob(monkeypatch, dtype, umbrella, division):
    """The reciprocal is ``FAST_DIV``'s, which follows the ``FAST_MATH`` umbrella unless pinned on its own."""
    for name, value in (("EMMY_FAST_MATH", umbrella), ("EMMY_FAST_DIV", division)):
        monkeypatch.delenv(name, raising=False)
        if value is not None:
            monkeypatch.setenv(name, value)
    body = _body(dtype)
    root = SimpleNamespace(op=KernelOp(body=body))
    if (division or umbrella) == "0":
        with pytest.raises(RuleSkipped, match="FAST_DIV is off"):
            rule.rewrite(root)
    else:
        result = rule.rewrite(root)
        reciprocal = next(s for s in result.body if isinstance(s, Assign))
        assert reciprocal.value.operation.name == "reciprocal" and reciprocal.dtype == dtype
        (numerator,) = (s.name for s in result.body.iter() if isinstance(s, Load) and s.input == "x")
        multiply = next(s for s in result.body.iter() if isinstance(s, Assign) and s.value.operation.name == "multiply")
        assert _names(multiply) == (numerator, reciprocal.name) and multiply.dtype == dtype
        with pytest.raises(RuleSkipped, match="no invariant"):
            rule.rewrite(SimpleNamespace(op=result))
    assert [s.value.operation.name for s in normalize_body(body).iter() if isinstance(s, Assign)] == ["divide"]


@pytest.mark.parametrize(("dtype", "varying"), [(I32, False), (F32, True)])
def test_integer_or_varying_division_is_not_split(monkeypatch, dtype, varying):
    monkeypatch.setenv("EMMY_FAST_MATH", "1")
    with pytest.raises(RuleSkipped, match="no invariant"):
        rule.rewrite(SimpleNamespace(op=KernelOp(body=_body(dtype, varying=varying))))


def test_round_to_nearest_division_is_not_split(monkeypatch):
    monkeypatch.setenv("EMMY_FAST_MATH", "1")
    with pytest.raises(RuleSkipped, match="no invariant"):
        rule.rewrite(SimpleNamespace(op=KernelOp(body=_body(divide="divide_rn_f32"))))


def _hoist(monkeypatch, text: str):
    """The pass over ``text``: the statements left outside the loop, and the loop."""
    monkeypatch.setenv("EMMY_FAST_MATH", "1")
    result = rule.rewrite(SimpleNamespace(op=KernelOp(body=parse_body(text))))
    (loop,) = (s for s in result.body if isinstance(s, Loop))
    return [s for s in result.body if not isinstance(s, Loop)], loop


def test_a_staged_copy_stays_in_the_loop(monkeypatch):
    """The two halves of a staged copy cross the loop's back edge by their own field, whatever their index reads."""
    outside, loop = _hoist(
        monkeypatch,
        """
        f32 a = load K[0]
        for i in 0..4:
            f32 w = load x[i]
            f32 a = load K[0] carried=load
            f32 a = load K[0] carried=unpack
            f32 q = divide(w, a)
            f32 Y[i] = q
        """,
    )
    assert [s.carried for s in outside if isinstance(s, Load)] == [""]
    assert [s.carried for s in loop.body if isinstance(s, Load) and s.carried] == ["load", "unpack"]


@pytest.mark.parametrize("division", ["0", "1"])
def test_a_literal_divisor_is_the_division_knobs(monkeypatch, division):
    """A float literal divisor varies with no axis, so ``FAST_DIV`` takes it like any invariant one: a multiply by its
    reciprocal when on, the division itself when off. This is where a literal the normal form leaves a division lands.
    The kernel body's normal form folds the reciprocal of the literal to its value at the statement's dtype."""
    monkeypatch.setenv("EMMY_FAST_MATH", "1")
    monkeypatch.setenv("EMMY_FAST_DIV", division)
    body = parse_body(
        """
        for i in 0..4:
            f32 v = load x[i]
            f32 out = divide(v, 3.0)
            f32 y[i] = out
        """
    )
    if division == "0":
        with pytest.raises(RuleSkipped, match="FAST_DIV is off"):
            rule.rewrite(SimpleNamespace(op=KernelOp(body=body)))
        return
    result = rule.rewrite(SimpleNamespace(op=KernelOp(body=body)))
    (multiply,) = [s for s in result.body.iter() if isinstance(s, Assign)]
    assert multiply.value.operation.name == "multiply"
    assert float(np.float32(1 / 3)) in [operand.value for operand in multiply.value.operands if isinstance(operand, Literal)]
