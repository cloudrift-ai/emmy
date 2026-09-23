"""A ``Select`` branch the constants decide is dropped, not carried.

Fusing a scatter at a literal coordinate folds one branch's predicate to a constant. The branch
left standing keeps its whole producer cone alive through ``Select.deps``, and a reduce in a cone
nothing can select does not read the output sweep — which is what makes ``promoted_sweep`` refuse
the grid and leave the kernel sweeping every cell in one block.
"""

from importlib import import_module

import pytest

from emmy.compiler.dtype import F16, F32, U32
from emmy.compiler.graph import Node, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import RenderCtx
from emmy.compiler.ir.stmt.blocks import Loop
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.leaves import Assign, Load, Select, SelectBranch, Write
from emmy.compiler.ir.stmt.normalize import simplify_body

ELSE = Literal(True, "bool")


def _selecting(predicate: Expr) -> Body:
    """One scatter read: the near value under ``predicate``, the base buffer otherwise."""
    return Body(
        (
            Loop(
                axis=Axis("a", 4),
                body=(
                    Load(name="near", input="scattered", index=(Var("a"),)),
                    Load(name="far", input="base", index=(Var("a"),)),
                    Assign(name="cone", op="multiply", args=("near", "near")),
                    Select(name="v", branches=(SelectBranch("cone", predicate), SelectBranch("far", ELSE))),
                    Write(output="out", index=(Var("a"),), value="v"),
                ),
            ),
        )
    )


def test_a_branch_that_can_never_hold_leaves_and_releases_its_cone() -> None:
    (loop,) = simplify_body(_selecting(BinaryExpr("<", Var("a"), Literal(0, "int"))))
    (select,) = [stmt for stmt in loop.body if isinstance(stmt, Select)]
    assert select.branches == (SelectBranch("far", ELSE),)
    assert "cone" not in Body(loop.body).ssa_uses


def test_a_branch_that_always_holds_becomes_the_else() -> None:
    (loop,) = simplify_body(_selecting(BinaryExpr("<", Var("a"), Literal(4, "int"))))
    (select,) = [stmt for stmt in loop.body if isinstance(stmt, Select)]
    assert [branch.value for branch in select.branches] == ["cone"]
    assert "far" not in Body(loop.body).ssa_uses


def test_an_undecided_branch_is_kept_whole() -> None:
    (loop,) = simplify_body(_selecting(BinaryExpr("<", Var("a"), Literal(2, "int"))))
    (select,) = [stmt for stmt in loop.body if isinstance(stmt, Select)]
    assert [branch.value for branch in select.branches] == ["cone", "far"]


@pytest.mark.parametrize(
    ("left", "right", "result", "ctype"), [(F16, F16, F16, "__half"), (F16, F32, F32, "float"), (U32, U32, U32, "unsigned int")]
)
def test_select_preserves_the_common_type_in_its_consumers(left, right, result, ctype):
    """A half concatenation must not promote a following product and drop its half rounding."""
    index = (Literal(0, "int"),)
    selection = Select("chosen", (SelectBranch("a", Var("condition")), SelectBranch("b", ELSE)))
    output = Tensor("out", (1,), result)
    op = KernelOp(
        name="selected_product",
        inputs={"lhs": Tensor("lhs", (1,), left), "rhs": Tensor("rhs", (1,), right)},
        outputs={"out": output},
        body=Body(
            (
                Load(name="a", input="lhs", index=index),
                Load(name="b", input="rhs", index=index),
                selection,
                Assign(name="product", op="multiply", args=("chosen", "a")),
                Write(output="out", index=index, value="product"),
            )
        ),
    )
    stamped = import_module("emmy.compiler.pipeline.passes.lowering.kernel.030_stamp_types").rewrite(
        Node(id="out", op=op, inputs=["lhs", "rhs"], outputs=(output,))
    )
    assert next(s for s in stamped.body if isinstance(s, Assign)).dtype == result
    assert next(s for s in stamped.body if isinstance(s, Write)).value_dtype == result
    ctx = RenderCtx(ssa_dtypes={"a": left.name, "b": right.name})
    assert selection.render(ctx)[0].startswith(f"    {ctype} chosen = ")
    assert ctx.ssa_dtypes["chosen"] == result.name
