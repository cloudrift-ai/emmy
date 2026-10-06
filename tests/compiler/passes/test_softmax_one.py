"""The one-element softmax simplification is an exact measured kernel choice."""

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest

from emmy.compiler.dtype import F16, F32
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Assign, Body
from emmy.compiler.pipeline import RuleSkipped
from tests.compiler.helpers import requires_cuda

rule = import_module("emmy.compiler.pipeline.passes.lowering.kernel.086_softmax_one")


def _body(exp="exp") -> Body:
    return Body(
        (
            Assign("delta", "subtract", ("score", "score")),
            Assign("weight", exp, ("delta",)),
            Assign("out", "divide", ("weight", "weight")),
        )
    )


@pytest.mark.parametrize("exp", ["exp", "exp_fast"])
def test_one_element_softmax_is_a_kernel_choice(monkeypatch, exp):
    monkeypatch.delenv("EMMY_SOFTMAX_ONE", raising=False)
    original = KernelOp(body=_body(exp))
    off, on = rule.rewrite(SimpleNamespace(op=original))
    assert off.body == original.body and off.knobs["SOFTMAX_ONE"] == 0
    assert on.body[-1].op.name == "softmax_one" and on.body[-1].args == ("score",)
    assert on.knobs["SOFTMAX_ONE"] == 1
    with pytest.raises(RuleSkipped, match="already decided"):
        rule.rewrite(SimpleNamespace(op=on))
    monkeypatch.setenv("EMMY_SOFTMAX_ONE", "1")
    assert len(rule.rewrite(SimpleNamespace(op=original))) == 1


def test_simplification_preserves_nonfinite_values():
    x = np.array([0.0, -0.0, 2.0, np.inf, -np.inf, np.nan], dtype=np.float32)
    with np.errstate(invalid="ignore"):
        expected = np.exp(x - x) / np.exp(x - x)
        actual = ElementwiseImpl("softmax_one")(x)
    np.testing.assert_array_equal(actual, expected)


def test_unrelated_quotient_is_unchanged():
    body = Body((Assign("out", "divide", ("left", "right")),))
    with pytest.raises(RuleSkipped, match="no one-key"):
        rule.rewrite(SimpleNamespace(op=KernelOp(body=body)))


@requires_cuda
@pytest.mark.parametrize("dtype", [F16, F32])
def test_cuda_preserves_nonfinite_values(monkeypatch, dtype):
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.ir.tensor.ir import ElementwiseOp

    monkeypatch.setenv("EMMY_FAST_MATH", "0")
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (6,), dtype), node_id="x")
    graph.add_node(ElementwiseOp("softmax_one"), ["x"], Tensor("out", (6,), dtype), node_id="out")
    graph.inputs = ["x"]
    graph.outputs = ["out"]
    x = np.array([0.0, -0.0, 2.0, np.inf, -np.inf, np.nan], dtype=dtype.np)
    backend = CudaBackend()
    result, _ = backend.run(backend.compile(graph), input_data={"x": x})
    with np.errstate(invalid="ignore"):
        expected = ElementwiseImpl("softmax_one")(x)
    np.testing.assert_array_equal(result.outputs["out"], expected)
