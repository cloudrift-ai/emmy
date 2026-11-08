"""Fast-math reciprocal hoisting preserves types and leaves structural normalization pure."""

from importlib import import_module
from types import SimpleNamespace

import pytest

from emmy.compiler.dtype import BF16, F16, F32, F64, I32
from emmy.compiler.ir.expression import Var
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Assign, normalize_body, parse_body
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
@pytest.mark.parametrize("enabled", [None, "1", "0"])
def test_reciprocal_hoisting_follows_fast_math(monkeypatch, dtype, enabled):
    monkeypatch.delenv("EMMY_FAST_MATH", raising=False)
    if enabled is not None:
        monkeypatch.setenv("EMMY_FAST_MATH", enabled)
    body = _body(dtype)
    root = SimpleNamespace(op=KernelOp(body=body))
    if enabled == "0":
        with pytest.raises(RuleSkipped, match="disabled"):
            rule.rewrite(root)
    else:
        result = rule.rewrite(root)
        reciprocal = next(s for s in result.body if isinstance(s, Assign))
        assert reciprocal.value.operation.name == "reciprocal" and reciprocal.dtype == dtype
        assert reciprocal.name != "recip_0"
        multiply = next(s for s in result.body.iter() if isinstance(s, Assign) and s.value.operation.name == "multiply")
        assert _names(multiply) == ("recip_0", reciprocal.name) and multiply.dtype == dtype
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
