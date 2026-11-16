"""``125_spell_composites`` writes each composite function of a typed kernel body out in primitive operations."""

from importlib import import_module
from types import SimpleNamespace

from emmy.compiler.dtype import F16, F32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expression import FunctionCallExpression, Var
from emmy.compiler.ir.kernel.ir import KernelOp, Tile
from emmy.compiler.ir.stmt import Assign, Body, check_types, parse_body
from emmy.compiler.ir.types import explicit_conversions
from emmy.compiler.target import native_f16_ops
from emmy.compiler.tensor import Tensor

_RULE = import_module("emmy.compiler.pipeline.passes.lowering.kernel.125_spell_composites")
_BODY = """
    x = load X[a]
    v = sigmoid(x)
    r = relu(x)
    q = reciprocal(x)
    s = square(x)
    V[a] = v
    R[a] = r
    Q[a] = q
    S[a] = s
"""


def _typed(body: str, dtype) -> tuple[KernelOp, dict[str, Tensor]]:
    """``body`` over one grid axis ``a``, its buffers of ``dtype``, typed in full as ``120_explicit_conversions`` does."""
    tiled = KernelOp(body=Body((Tile(axes=(Axis("a", 4),), body=parse_body(body)),)))
    tensors = {name: Tensor(name, (4,), dtype) for name in (*tiled.inputs, *tiled.outputs)}
    typed = explicit_conversions(tiled.body, {name: dtype for name in tensors}, native_f16_ops())
    return KernelOp(body=typed), tensors


def _composites(op: KernelOp) -> set[str]:
    values = [stmt.value for stmt in op.body.iter() if isinstance(stmt, Assign)]
    return {node.operation.name for value in values for node in value.subterms() if node.definition is not None}


def test_every_composite_is_primitives_at_the_dtype_it_computed_at() -> None:
    typed, tensors = _typed(_BODY, F16)
    assert _composites(typed) == {"sigmoid", "relu", "reciprocal", "square"}
    spelled = _RULE.rewrite(SimpleNamespace(op=typed))
    assert not _composites(spelled)
    check_types(spelled.body, {name: "f16" for name in tensors})
    lines = [line.strip() for line in str(spelled.body).splitlines()]
    # An f16 sigmoid computes in f32 and converts once.
    assert "f32 v__f32__p1 = exp(v__f32__p0)" in lines
    assert "f32 v__f32 = 1.0 / v__f32__p2" in lines and "f16 v = (f16)v__f32" in lines
    # A composite the target has at f16 stays there, and the f32 square of an f16 value stays in f32.
    assert "f16 r = maximum(0.0, x)" in lines
    assert "f16 q = 1.0 / x" in lines
    assert "f32 s__f32 = x__f32 * x__f32" in lines


def test_two_composites_share_the_primitive_they_both_compute() -> None:
    """A sigmoid and a value divided by the same denominator compute one exponential."""
    typed, _tensors = _typed("x = load X[a]\nv = sigmoid(x)\nw = x / (1.0 + exp(-x))\nV[a] = v\nW[a] = w", F32)
    assert str(_RULE.rewrite(SimpleNamespace(op=typed)).body).count("exp(") == 1


def test_the_host_text_of_a_composite_is_its_definition() -> None:
    """The Loop IR runner prints a composite through the same table."""
    from emmy.compiler.ir.loop.render import RenderCtx, op_to_expression, render_expression

    spelled = op_to_expression("sigmoid", [FunctionCallExpression("fmax", (Var("x"), Var("y")))])
    assert render_expression(spelled, RenderCtx()) == "1.0f / (1.0f + exp(0.0f - fmax(x, y)))"
    assert render_expression(op_to_expression("relu", [Var("x")]), RenderCtx()) == "fmax(0.0f, x)"
