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
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.render import RenderCtx, render_stmt
from emmy.compiler.ir.stmt import parse_body, pretty_body, simplify_body


def _branches(predicate: str) -> list[str]:
    """The branches simplification leaves of one scatter read: the near value under ``predicate``, the base buffer
    otherwise. Nothing but the select reads ``cone`` or ``far``, so a branch that left released its value."""
    body = parse_body(
        f"""
        for a in 0..4:
            near = load scattered[a]
            far = load base[a]
            cone = multiply(near, near)
            v = select:
                cone when {predicate}
                far when true
            out[a] = v
        """
    )
    return [line.strip() for line in pretty_body(simplify_body(body)) if " when " in line]


def test_a_branch_that_can_never_hold_leaves_and_releases_its_cone() -> None:
    assert _branches("a < 0") == ["far when true"]


def test_a_branch_that_always_holds_becomes_the_else() -> None:
    assert [branch.split(" when ")[0] for branch in _branches("a < 4")] == ["cone"]


def test_an_undecided_branch_is_kept_whole() -> None:
    assert _branches("a < 2") == ["cone when a < 2", "far when true"]


@pytest.mark.parametrize(
    ("left", "right", "result", "ctype"), [(F16, F16, F16, "__half"), (F16, F32, F32, "float"), (U32, U32, U32, "unsigned int")]
)
def test_select_preserves_the_common_type_in_its_consumers(left, right, result, ctype):
    """A half concatenation must not promote a following product and drop its half rounding."""
    body = parse_body(
        """
        a = load lhs[0]
        b = load rhs[0]
        chosen = select:
            a when condition
            b when true
        product = multiply(chosen, a)
        out[0] = product
        """
    )
    selection = body[2]
    output = Tensor("out", (1,), result)
    op = KernelOp(
        name="selected_product",
        inputs={"lhs": Tensor("lhs", (1,), left), "rhs": Tensor("rhs", (1,), right)},
        outputs={"out": output},
        body=body,
    )
    stamped = import_module("emmy.compiler.pipeline.passes.lowering.kernel.030_stamp_types").rewrite(
        Node(id="out", op=op, inputs=["lhs", "rhs"], outputs=(output,))
    )
    lines = str(stamped.body).splitlines()
    assert f"{result.name} product = multiply(chosen, a)" in lines
    assert f"{result.name} out[0] = product" in lines
    ctx = RenderCtx(ssa_dtypes={"a": left.name, "b": right.name})
    assert render_stmt(selection, ctx)[0].startswith(f"    {ctype} chosen = ")
    assert ctx.ssa_dtypes["chosen"] == result.name
