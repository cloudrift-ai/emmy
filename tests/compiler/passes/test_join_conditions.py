"""``121_join_conditions`` joins the conditions side by side that test the same thing."""

from importlib import import_module
from types import SimpleNamespace

import pytest

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.kernel.ir import KernelOp, Tile
from emmy.compiler.ir.stmt import Body, Cond, parse_body
from emmy.compiler.pipeline import RuleSkipped

_RULE = import_module("emmy.compiler.pipeline.passes.lowering.kernel.121_join_conditions")


def _joined(body: str) -> list:
    op = KernelOp(body=Body((Tile(axes=(Axis("i", 4),), body=parse_body(body)),)), name="k")
    return list(_RULE.rewrite(SimpleNamespace(op=op), None).body[0].body)


def test_a_run_of_one_test_is_one_condition_with_the_bodies_in_order() -> None:
    """Four stores of a row under the same guard: one condition holding the four, in order; a different test after
    them starts a condition of its own."""
    stores = "\n".join(f"if i < n:\n    f32 x{k} = load A[i, {k}]\n    f32 O[i, {k}] = x{k}" for k in range(4))
    out = _joined(stores + "\nif i + 1 < n:\n    f32 y = load A[i, 4]\n    f32 O[i, 4] = y")
    assert [type(stmt) for stmt in out] == [Cond, Cond]
    assert [str(stmt.index[-1]) for stmt in out[0].body if hasattr(stmt, "index")] == ["0", "0", "1", "1", "2", "2", "3", "3"]


@pytest.mark.parametrize(
    "body",
    [
        "if i < n:\n    f32 a = load A[i, 0]\n    f32 O[i, 0] = a\nelse:\n    f32 b = load A[i, 1]\n    f32 O[i, 1] = b\n"
        "if i < n:\n    f32 c = load A[i, 2]\n    f32 O[i, 2] = c",
        "if i < n:\n    f32 a = load A[i, 0]\n    f32 O[i, 0] = a\nf32 b = load A[i, 1]\nf32 O[i, 1] = b\n"
        "if i < n:\n    f32 c = load A[i, 2]\n    f32 O[i, 2] = c",
    ],
    ids=["other-side", "apart"],
)
def test_conditions_with_another_side_or_apart_stay(body: str) -> None:
    with pytest.raises(RuleSkipped):
        _joined(body)
