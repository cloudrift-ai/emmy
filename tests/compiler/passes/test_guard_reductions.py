"""Coordinate selects can discard a scalar reduction without executing its loop."""

from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.kernel import KernelOp, Sync
from emmy.compiler.ir.loop.runner import execute_loop_op_cpp
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Select, SelectBranch, StridedLoop, Write

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
