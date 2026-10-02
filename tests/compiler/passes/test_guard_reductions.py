"""Coordinate selects can discard a scalar reduction without executing its loop."""

from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest

from emmy.compiler.dtype import F32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.kernel import KernelOp, Sync
from emmy.compiler.ir.loop.runner import execute_loop_op_cpp
from emmy.compiler.ir.stmt import Accum, Assign, Body, Cond, Load, Loop, Select, SelectBranch, StridedLoop, Write

guard = import_module("emmy.compiler.pipeline.passes.lowering.kernel.090_guard_reductions")


def _body(*, extra_use=False, overlapping=False, strided=False) -> Body:
    reduction = Loop(
        Axis("k", 7),
        (Load(name="value", input="x", index=(Var("i"), Var("k"))), Accum(name="sum", op="add", value="value")),
    )
    if strided:
        reduction = StridedLoop(reduction.axis, Literal(1, "int"), Literal(2, "int"), reduction.body, end=Literal(6, "int"))
    select = Select(
        "selected",
        (
            SelectBranch("neg", Var("i").lt(2)),
            SelectBranch("other" if overlapping else "neg", Var("i").lt(4)),
            SelectBranch("other", Literal(False, "bool")),
        ),
    )
    body = (
        reduction,
        Load(name="other", input="fallback", index=(Var("i"),)),
        Assign(name="neg", op="negative", args=("sum",)),
        select,
        Write(output="out", index=(Var("i"),), value="selected"),
    )
    if extra_use:
        body += (Write(output="unmasked", index=(Var("i"),), value="sum"),)
    return Body((Loop(Axis("i", 6), body),))


def _run(body: Body, *, extra_use=False) -> tuple:
    outputs = {"out": (6,)}
    if extra_use:
        outputs["unmasked"] = (6,)
    x = np.arange(42, dtype=np.float32).reshape(6, 7)
    x[4:] = np.nan  # An ignored nonfinite reduction must not contaminate the selected value.
    result = execute_loop_op_cpp(KernelOp(body=body), {"x": x, "fallback": np.arange(6, dtype=np.float32)}, outputs)
    return result if isinstance(result, tuple) else (result,)


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("overlapping", [False, True])
@pytest.mark.parametrize("extra_use", [False, True])
def test_coordinate_demand_preserves_values_and_skips_only_unused_iterations(strided, overlapping, extra_use):
    body = _body(strided=strided, overlapping=overlapping, extra_use=extra_use)
    changed = guard._guard(body)
    reduction = changed[0].body[0]
    if extra_use:
        assert reduction == body[0].body[0], "one unmasked reader requires every original iteration"
    else:
        assert isinstance(reduction, StridedLoop)
        needed_rows = 2 if overlapping else 4
        for i in range(6):
            assert reduction.end.eval({"i": i}) == ((6 if strided else 7) if i < needed_rows else 0)
    for actual, expected in zip(_run(changed, extra_use=extra_use), _run(body, extra_use=extra_use), strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("unsafe", ["store", "sync", "carried", "later_predicate"])
def test_guard_does_not_skip_effects_or_use_an_unavailable_predicate(unsafe):
    body = _body()
    outer = body[0]
    reduction, *tail = outer.body
    if unsafe == "store":
        reduction = replace(reduction, body=(*reduction.body, Write(output="side", index=(Var("k"),), value="value")))
    elif unsafe == "sync":
        reduction = replace(reduction, body=(*reduction.body, Sync()))
    elif unsafe == "carried":
        reduction = replace(reduction, seed=False)
    else:
        tail = [
            replace(s, branches=tuple(replace(b, select=b.select.substitute({"i": Var("later")})) for b in s.branches))
            if isinstance(s, Select)
            else s
            for s in tail
        ]
    candidate = Body((replace(outer, body=(reduction, *tail)),))
    assert guard._guard(candidate)[0].body[0] == reduction


def _selected_stores() -> Body:
    return Body(
        (
            Load(name="a", input="x", index=(Var("i"),), dtype=F32),
            Assign(name="neg", op="negative", args=("a",), dtype=F32),
            Load(name="b", input="fallback", index=(Var("i"),), dtype=F32),
            Select("chosen", (SelectBranch("neg", Var("i").lt(3)), SelectBranch("b", Literal(True, "bool")))),
            Write(output="out", index=(Var("i"),), value="chosen", value_dtype=F32),
            Write(output="other", index=(Var("i"),), value="chosen", value_dtype=F32),
        )
    )


def test_selected_stores_evaluate_only_the_chosen_private_cone():
    body = _selected_stores()
    changed = guard._guard_stores(body)
    (branch,) = changed
    assert isinstance(branch, Cond)
    assert [s.input for s in branch.body if isinstance(s, Load)] == ["x"]
    assert [s.input for s in branch.else_body if isinstance(s, Load)] == ["fallback"]
    inputs = {"x": np.arange(6, dtype=np.float32), "fallback": np.arange(10, 16, dtype=np.float32)}
    outputs = {"out": (6,), "other": (6,)}
    for candidate in (body, changed):
        op = KernelOp(body=Body((Loop(Axis("i", 6), candidate),)))
        actual = execute_loop_op_cpp(op, inputs, outputs)
        for result in actual:
            np.testing.assert_array_equal(result, np.array([0, -1, -2, 13, 14, 15], dtype=np.float32))


@pytest.mark.parametrize("unsafe", ["reload", "sync", "atomic", "extra_reader", "index_reader"])
def test_selected_stores_keep_memory_effects_and_other_readers(unsafe):
    stmts = list(_selected_stores())
    if unsafe == "reload":
        stmts.insert(2, Write(output="x", index=(Var("i"),), value="replacement"))
    elif unsafe == "sync":
        stmts.insert(2, Sync())
    elif unsafe == "atomic":
        stmts[-1] = replace(stmts[-1], atomic=True)
    elif unsafe == "extra_reader":
        stmts.append(Assign("used", "abs", ("chosen",), dtype=F32))
    else:
        stmts[-1] = replace(stmts[-1], index=(Var("chosen"),))
    body = Body(stmts)
    assert guard._guard_stores(body) == body
