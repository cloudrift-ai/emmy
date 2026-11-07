"""Coordinate selects can discard a scalar reduction without executing its loop."""

from importlib import import_module

import numpy as np
import pytest

from emmy.compiler.ir.expression import Var
from emmy.compiler.ir.kernel import KernelOp, Sync
from emmy.compiler.ir.loop.runner import execute_loop_op_cpp
from emmy.compiler.ir.stmt import Body, Select, StridedLoop, parse_body

guard = import_module("emmy.compiler.pipeline.passes.lowering.kernel.090_guard_reductions")


def _body(*, extra_use=False, overlapping=False, strided=False) -> Body:
    return parse_body(
        f"""
        for i in 0..6:
            for k in {"1..7 step 2 until 6" if strided else "0..7"}:
                value = load x[i, k]
                sum <- add(sum, value)
            other = load fallback[i]
            neg = negative(sum)
            selected = select:
                neg when i < 2
                {"other" if overlapping else "neg"} when i < 4
                other when false
            out[i] = selected
            {"unmasked[i] = sum" if extra_use else ""}
        """
    )


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
        reduction = reduction.replace(body=(*reduction.body, *parse_body("side[k] = value")))
    elif unsafe == "sync":
        reduction = reduction.replace(body=(*reduction.body, Sync()))
    elif unsafe == "carried":
        reduction = reduction.replace(seed=False)
    else:
        tail = [
            s.replace(branches=tuple(b.replace(select=b.select.substitute({"i": Var("later")})) for b in s.branches))
            if isinstance(s, Select)
            else s
            for s in tail
        ]
    candidate = Body((outer.replace(body=(reduction, *tail)),))
    assert guard._guard(candidate)[0].body[0] == reduction
