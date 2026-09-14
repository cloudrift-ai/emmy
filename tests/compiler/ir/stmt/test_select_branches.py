"""A ``Select`` branch the constants decide is dropped, not carried.

Fusing a scatter at a literal coordinate folds one branch's predicate to a constant. The branch
left standing keeps its whole producer cone alive through ``Select.deps``, and a reduce in a cone
nothing can select does not read the output sweep — which is what makes ``promoted_sweep`` refuse
the grid and leave the kernel sweeping every cell in one block.
"""

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
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
