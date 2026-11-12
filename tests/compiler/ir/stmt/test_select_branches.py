"""A ``Select`` keeps the common type of its branches for its consumers; which branches simplification keeps is
tested in the crate."""

from importlib import import_module

import pytest

from emmy.compiler.dtype import F16, F32, U32
from emmy.compiler.graph import Node, Tensor
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.render import RenderCtx
from emmy.compiler.ir.stmt import parse_body
from tests.compiler.helpers import render_typed


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
    assert f"{result.name} product = a * chosen" in lines
    assert f"{result.name} out[0] = product" in lines
    ctx = RenderCtx(ssa_dtypes={"a": left.name, "b": right.name})
    assert render_typed([selection], ctx)[-1].startswith(f"    {ctype} chosen = ")
    assert ctx.ssa_dtypes["chosen"] == result.name
