"""A coordinate select's two private load chains merge into one without changing memory order."""

from collections import Counter
from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest

from emmy.compiler.dtype import F32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.kernel import KernelOp, Sync
from emmy.compiler.ir.loop.runner import execute_loop_op_cpp
from emmy.compiler.ir.stmt import Body, Let, Load, Loop, Select, SelectBranch, Write

merge = import_module("emmy.compiler.pipeline.passes.lowering.kernel.045_merge_select_loads")


def _body():
    i = Var("i")
    loads = tuple(Load(f"v{j}", "x", (Literal(j, "int"), i % Literal(4, "int")), dtype=F32) for j in range(2))
    select = Select("selected", (SelectBranch("v0", i.lt(4)), SelectBranch("v1", Literal(True, "bool"))))
    return Body((*loads, select, Write("out", (i,), "selected", value_dtype=F32)))


def _rewrite(body):
    return merge._walk(body, Counter(name for s in body.iter() for name in merge._reads(s)))


def test_two_selected_loads_merge_into_one():
    original = _body()
    changed = _rewrite(original)
    assert len([s for s in changed if isinstance(s, Load)]) == 1
    x = np.arange(8, dtype=np.float32).reshape(2, 4)
    for body in (original, changed):
        op = KernelOp(body=Body((Loop(Axis("i", 12), body),)))
        actual = execute_loop_op_cpp(op, {"x": x}, {"out": (12,)})
        np.testing.assert_array_equal(actual, np.concatenate((x.flatten(), x[-1])))


@pytest.mark.parametrize("unsafe", ["write", "sync", "late_predicate", "late_index", "carried"])
def test_selected_loads_keep_memory_effects_and_late_coordinates(unsafe):
    stmts = list(_body())
    if unsafe == "write":
        stmts.insert(1, Write("x", (Literal(1, "int"), Literal(0, "int")), "v0"))
    elif unsafe == "sync":
        stmts.insert(1, Sync())
    elif unsafe == "late_predicate":
        select = stmts[-2]
        stmts[-2] = replace(select, branches=(replace(select.branches[0], select=Var("later").lt(2)), *select.branches[1:]))
        stmts.insert(-2, Let("later", Literal(0, "int")))
    elif unsafe == "late_index":
        stmts[1] = replace(stmts[1], index=(Literal(1, "int"), Var("later")))
        stmts.insert(1, Let("later", Literal(0, "int")))
    else:
        stmts[0] = replace(stmts[0], carried="load")
    body = Body(stmts)
    assert _rewrite(body) == body
