"""A coordinate select's two private load chains merge into one without changing memory order."""

from collections import Counter
from importlib import import_module
from textwrap import indent

import numpy as np
import pytest

from emmy.compiler.ir.expression import Literal, Var
from emmy.compiler.ir.kernel import KernelOp, Sync
from emmy.compiler.ir.loop.runner import execute_loop_op_cpp
from emmy.compiler.ir.stmt import Body, Load, parse_body

merge = import_module("emmy.compiler.pipeline.passes.lowering.kernel.045_merge_select_loads")


def _body():
    return parse_body(
        """
        f32 v0 = load x[0, i % 4]
        f32 v1 = load x[1, i % 4]
        selected = select:
            v0 when i < 4
            v1 when true
        f32 out[i] = selected
        """
    )


def _rewrite(body):
    return merge._walk(body, Counter(name for s in body.iter() for name in merge._reads(s)))


def test_two_selected_loads_merge_into_one():
    original = _body()
    changed = _rewrite(original)
    assert len([s for s in changed if isinstance(s, Load)]) == 1
    x = np.arange(8, dtype=np.float32).reshape(2, 4)
    for body in (original, changed):
        op = KernelOp(body=parse_body("for i in 0..12:\n" + indent(str(body), "    ")))
        actual = execute_loop_op_cpp(op, {"x": x}, {"out": (12,)})
        np.testing.assert_array_equal(actual, np.concatenate((x.flatten(), x[-1])))


@pytest.mark.parametrize("unsafe", ["write", "sync", "late_predicate", "late_index", "carried"])
def test_selected_loads_keep_memory_effects_and_late_coordinates(unsafe):
    stmts = list(_body())
    if unsafe == "write":
        stmts[1:1] = parse_body("x[1, 0] = v0")
    elif unsafe == "sync":
        stmts.insert(1, Sync())
    elif unsafe == "late_predicate":
        select = stmts[-2]
        stmts[-2] = select.replace(branches=(select.branches[0].replace(select=Var("later").lt(2)), *select.branches[1:]))
        stmts[-2:-2] = parse_body("let later = 0")
    elif unsafe == "late_index":
        stmts[1] = stmts[1].replace(index=(Literal(1, "int"), Var("later")))
        stmts[1:1] = parse_body("let later = 0")
    else:
        stmts[0] = stmts[0].replace(carried="load")
    body = Body(stmts)
    assert _rewrite(body) == body
