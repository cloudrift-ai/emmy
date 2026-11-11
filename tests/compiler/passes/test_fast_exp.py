"""``085_fast_exp`` lowers every ``exp`` a kernel computes, the ones a statement holds as a field included."""

import re

import pytest

from emmy.compiler.context import Context
from emmy.compiler.ir.cuda.ir import CudaOp
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.kernel.ir import TreeHalve, WarpShuffle
from emmy.compiler.pipeline import FINAL_LOWERING_PASS, KERNEL_PASSES, Pipeline
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.realization.helpers import CASES_DIR, load_case, pin_of


def _lowered(name: str, pins: dict) -> tuple[KernelOp, str]:
    """A cooperative softmax corpus case lowered under its own schedule row and ``pins``: its kernel and CUDA source."""
    case = load_case(CASES_DIR / f"reduce/{name}.json")
    ctx = Context.from_target(case.compute_cap)
    with pinned_knobs({**pin_of(case, case.row), **pins}):
        lowered = Pipeline.build(KERNEL_PASSES).run(case.program(), ctx=ctx, db=None)
        (kernel,) = [node.op for node in lowered.nodes.values() if isinstance(node.op, KernelOp)]
        compiled = Pipeline.build([FINAL_LOWERING_PASS]).run(lowered, ctx=ctx)
    (source,) = [node.op.kernel_source for node in compiled.nodes.values() if isinstance(node.op, CudaOp)]
    return kernel, source


def _exp_ops(stmt) -> set[str]:
    return {node.operation.name for e in stmt.expressions() for node in e.subterms() if node.operation} & {"exp", "exp_fast"}


@pytest.mark.parametrize(("case", "combine"), [("combine-softmax-coop-warp", WarpShuffle), ("combine-softmax-coop-hier", TreeHalve)])
def test_a_cooperative_merge_takes_the_fast_exponential(case, combine) -> None:
    """The online-softmax merge of a cooperative reduce is a statement field of its combine (``combine_states``), not
    a body: under ``FAST_EXP`` with the umbrella off it takes ``__expf`` as the per-lane fold does, and no ``expf``."""
    kernel, source = _lowered(case, {"FAST_MATH": False, "FAST_EXP": True})
    merges = [stmt for stmt in kernel.body.iter() if isinstance(stmt, combine)]
    assert merges, f"{case} must reach a {combine.__name__} merge"
    assert all(_exp_ops(merge) == {"exp_fast"} for merge in merges)
    assert "__expf(" in source and not re.search(r"(?<!_)expf\(", source)


def test_the_merge_keeps_the_accurate_exponential_under_its_own_pin() -> None:
    kernel, source = _lowered("combine-softmax-coop-warp", {"FAST_MATH": True, "FAST_EXP": False})
    merges = [stmt for stmt in kernel.body.iter() if isinstance(stmt, WarpShuffle)]
    assert merges and all(_exp_ops(merge) == {"exp"} for merge in merges)
    assert "__expf(" not in source


def test_an_intrinsic_an_index_calls_passes_through() -> None:
    """An index may call an intrinsic the target spells itself (a Volta swizzle); the walk reads a call's name, so it
    lowers the exponential beside it and leaves the intrinsic alone."""
    from importlib import import_module
    from types import SimpleNamespace

    from emmy.compiler.ir.expression import FunctionCallExpression, Var
    from emmy.compiler.ir.stmt import Assign, Body, parse_body

    rule = import_module("emmy.compiler.pipeline.passes.lowering.kernel.085_fast_exp")
    swizzled = Assign("x", FunctionCallExpression("emmy_volta_crosswise", (Var("i"),)))
    op = KernelOp(body=Body((swizzled, *parse_body("e = exp(x)\nO[i] = e"))))
    with pinned_knobs({"FAST_EXP": True}):
        lowered = rule.rewrite(SimpleNamespace(op=op)).body
    assert "emmy_volta_crosswise(i)" in str(lowered)
    assert "exp_fast(x)" in str(lowered), str(lowered)
