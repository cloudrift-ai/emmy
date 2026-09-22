"""Fast-math reciprocal hoisting preserves types and leaves structural normalization pure."""

from importlib import import_module
from types import SimpleNamespace

import pytest

from emmy.compiler.dtype import BF16, F16, F32, F64, I32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Assign, Body, Load, Loop, Write
from emmy.compiler.ir.stmt.normalize import normalize_body
from emmy.compiler.pipeline import RuleSkipped

rule = import_module("emmy.compiler.pipeline.passes.lowering.kernel.040_split_invariant_divides")


def _body(dtype=F32, *, varying=False):
    return Body((Loop(Axis("i", 4), Body((
        Load("recip_0", "x", (Var("i"),), dtype=dtype),
        Load("denominator", "scale", (Var("i"),) if varying else (), dtype=dtype),
        Assign("out", "divide", ("recip_0", "denominator"), dtype=dtype),
        Write("y", (Var("i"),), value="out", value_dtype=dtype),
    ))),))


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
        assert reciprocal.op.name == "reciprocal" and reciprocal.dtype == dtype
        assert reciprocal.name != "recip_0"
        multiply = next(s for s in result.body.iter() if isinstance(s, Assign) and s.op.name == "multiply")
        assert multiply.args == ("recip_0", reciprocal.name) and multiply.dtype == dtype
        with pytest.raises(RuleSkipped, match="no invariant"):
            rule.rewrite(SimpleNamespace(op=result))
    assert [s.op.name for s in normalize_body(body).iter() if isinstance(s, Assign)] == ["divide"]


@pytest.mark.parametrize(("dtype", "varying"), [(I32, False), (F32, True)])
def test_integer_or_varying_division_is_not_split(monkeypatch, dtype, varying):
    monkeypatch.setenv("EMMY_FAST_MATH", "1")
    with pytest.raises(RuleSkipped, match="no invariant"):
        rule.rewrite(SimpleNamespace(op=KernelOp(body=_body(dtype, varying=varying))))
