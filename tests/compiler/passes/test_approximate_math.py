"""``130_approximate_math`` picks the approximate form of each instruction class its knob turns on, in every place a
kernel holds an operation, and only where the operation computes in f32."""

import re
from importlib import import_module
from types import SimpleNamespace

import pytest

from emmy.compiler.context import Context
from emmy.compiler.ir.cuda.ir import CudaOp
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.kernel.ir import FragmentApply, RegStore, TreeHalve, WarpShuffle
from emmy.compiler.ir.stmt import Assign, Body, parse_body
from emmy.compiler.pipeline import FINAL_LOWERING_PASS, KERNEL_PASSES, Pipeline
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.realization.helpers import CASES_DIR, load_case, pin_of

_RULE = import_module("emmy.compiler.pipeline.passes.lowering.kernel.130_approximate_math")

#: A typed body with one definition per class, an integer quotient and two operations native in f16, each stored.
_BODY = """
f32 a = load A[i]
f32 b = load B[i]
i32 j = load J[i]
f16 g = load G[i]
f32 e = exp(a)
f32 q = a / b
f32 r = 1.0 / a
f32 s = sqrt(a)
f32 w = rsqrt(a)
f32 l = log(a)
f32 n = sin(a)
f32 c = cos(a)
f32 t = tanh(a)
f32 p = pow(a, b)
i32 k = j / 2
f16 h = sqrt(g)
f16 d = g / g
O[i, 0] = e
O[i, 1] = q
O[i, 2] = r
O[i, 3] = s
O[i, 4] = w
O[i, 5] = l
O[i, 6] = n
O[i, 7] = c
O[i, 8] = t
O[i, 9] = p
K[i] = k
H[i, 0] = h
H[i, 1] = d
"""
_EXACT = {"e": "exp", "q": "divide", "r": "divide", "s": "sqrt", "l": "log", "n": "sin", "c": "cos", "t": "tanh", "p": "pow"}
_EXACT |= {"w": "rsqrt", "k": "divide", "h": "sqrt", "d": "divide"}
_DIV = {"q": "divide_fast", "r": "reciprocal_fast"}
_LIBRARY = {"l": "log_fast", "n": "sin_fast", "c": "cos_fast", "t": "tanh_fast", "p": "pow_fast"}


def _picked(body: str, pins: dict, cap: tuple[int, int] = (12, 0)) -> KernelOp:
    with pinned_knobs(pins):
        return _RULE.rewrite(None, SimpleNamespace(op=KernelOp(body=Body(parse_body(body)))), Context.from_target(cap))


def _operations(kernel: KernelOp) -> dict[str, str]:
    return {stmt.name: stmt.value.operation.name for stmt in kernel.body.iter() if isinstance(stmt, Assign) and stmt.value.operation}


@pytest.mark.parametrize(
    ("pins", "approximate"),
    [
        ({"FAST_MATH": False}, {}),
        ({"FAST_MATH": True}, {"e": "exp_fast", "s": "sqrt_fast", **_DIV, **_LIBRARY}),
        ({"FAST_MATH": False, "FAST_EXP": True}, {"e": "exp_fast"}),
        ({"FAST_MATH": False, "FAST_DIV": True}, _DIV),
        ({"FAST_MATH": False, "FAST_SQRT": True}, {"s": "sqrt_fast"}),
        ({"FAST_MATH": True, "FAST_EXP": False}, {"s": "sqrt_fast", **_DIV}),
        ({"FAST_MATH": True, "FAST_DIV": False}, {"e": "exp_fast", "s": "sqrt_fast", **_LIBRARY}),
    ],
    ids=["standard", "umbrella", "exp", "div", "sqrt", "umbrella-less-exp", "umbrella-less-div"],
)
def test_each_class_follows_its_own_knob_where_it_computes_in_f32(pins, approximate) -> None:
    """The functions no knob names follow the umbrella with the exponential, as nvcc's fast library did. A division
    of one is the reciprocal instruction. An integer quotient, an operation native in f16 and the reciprocal square
    root, whose one instruction is approximate already, are never picked."""
    kernel = _picked(_BODY, pins)
    assert _operations(kernel) == {**_EXACT, **approximate}
    assert kernel.knobs["FAST_EXP"] is bool(approximate.get("e"))


def test_the_hyperbolic_tangent_is_picked_where_the_target_has_its_instruction() -> None:
    assert _operations(_picked(_BODY, {"FAST_MATH": True}, (7, 0)))["t"] == "tanh"
    assert _operations(_picked(_BODY, {"FAST_MATH": True}, (7, 5)))["t"] == "tanh_fast"


@pytest.mark.parametrize("function", ["exp", "tanh"])
def test_a_half_exponential_and_tangent_are_picked_and_compute_in_f32(function) -> None:
    """The exponential knob moves a half exponential to f32, as it always did, and the native half tangent is the
    f32 function on the converted value, which nvcc's fast library reached: both are picked at a half, and the
    type rules place them."""
    kernel = _picked(f"f16 g = load G[i]\nf16 x = {function}(g)\nH[i] = x", {"FAST_MATH": True})
    (fast,) = [stmt for stmt in kernel.body.iter() if isinstance(stmt, Assign) and stmt.value.operation]
    assert (fast.value.operation.name, fast.dtype.name) == (f"{function}_fast", "f32")
    assert "f16 x = (f16)" in str(kernel.body)


#: The epilogue's ``w`` and ``z`` declare no dtype: the type rules make both halves, and ``z`` a native square root.
_HELD = """
RegStore(dst_buffer=out, dst_index=(i, 0), frag=c0, shape=(16, 8, 16), epilogue_params=(acc), epilogue_result=y):
    epilogue:
        f16 v = acc
        in4 = load S[i]
        i32 n4 = load N[i]
        u = acc / in4
        w = maximum(v, 0.0)
        z = sqrt(w)
        i32 kk = n4 / 2
        let i32 column = n
        i32 kc = column / 2
        f16 y = sigmoid(u)
TreeHalve(bufs=(bb), state=(acc9), state_b=(acc9__o), length=8, tid_var=tt):
    combine_states:
        f = exp(acc9__o)
        acc9 = acc9 / f
FragmentApply(out=c1, op=reciprocal, args=(FragmentArg(kind=frag, names=(c0))))
"""


@pytest.mark.parametrize(
    ("pins", "exp", "divide", "reciprocal", "sigmoid"),
    [
        ({"FAST_MATH": False}, "exp", "divide", "reciprocal", "sigmoid"),
        ({"FAST_MATH": True}, "exp_fast", "divide_fast", "reciprocal_fast", "sigmoid_fast"),
        ({"FAST_MATH": False, "FAST_DIV": True}, "exp", "divide_fast", "reciprocal_fast", "sigmoid"),
    ],
    ids=["standard", "umbrella", "div"],
)
def test_an_operation_a_statement_holds_is_picked_too(pins, exp, divide, reciprocal, sigmoid) -> None:
    """A store's epilogue, a combine's merge program and a fragment operation hold operations no body walk enters.
    The epilogue is typed only when it is printed, so its definitions are typed here to read where each computes: the
    f32 division is picked, the half square root and the integer quotients are not. A composite stays whole in all
    three: the reciprocal has its own instruction, and a ``sigmoid`` takes the approximate exponential and division
    together or neither."""
    kernel = _picked(_HELD, pins)
    store, merge, fragment = (next(s for s in kernel.body.iter() if isinstance(s, kind)) for kind in (RegStore, TreeHalve, FragmentApply))
    epilogue = {stmt.name: stmt.value.operation.name for stmt in store.epilogue if isinstance(stmt, Assign) and stmt.value.operation}
    assert epilogue == {"u": divide, "w": "maximum", "z": "sqrt", "kk": "divide", "kc": "divide", "y": sigmoid}
    assert [stmt.value.operation.name for stmt in merge.combine_states] == [exp, divide]
    assert fragment.op.name == reciprocal


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


def test_the_helpers_in_inline_ptx_flush_with_the_kernel() -> None:
    """CUDA has no intrinsic for the approximate reciprocal: it is a helper in inline PTX, which nvcc's ``--ftz`` does
    not reach, so the helper says ``.ftz`` when the kernel's flush-to-zero decision is on."""
    _, flushed = _lowered("combine-softmax-coop-warp", {"FAST_MATH": True})
    _, kept = _lowered("combine-softmax-coop-warp", {"FAST_MATH": True, "FLUSH_TO_ZERO": False})
    assert '"rcp.approx.ftz.f32 %0, %1;"' in flushed and "emmy_rcp_approx(" in flushed
    assert '"rcp.approx.f32 %0, %1;"' in kept


def test_an_intrinsic_an_index_calls_passes_through() -> None:
    """An index may call an intrinsic the target spells itself (a Volta swizzle), which is no elementwise operation:
    the walk leaves it alone and picks the exponential beside it."""
    from emmy.compiler.ir.expression import FunctionCallExpression, Var

    swizzled = Assign("x", FunctionCallExpression("emmy_volta_crosswise", (Var("i"),)))
    op = KernelOp(body=Body((swizzled, *parse_body("e = exp(x)\nO[i] = e"))))
    with pinned_knobs({"FAST_EXP": True}):
        lowered = _RULE.rewrite(None, SimpleNamespace(op=op), Context.from_target((12, 0))).body
    assert "emmy_volta_crosswise(i)" in str(lowered)
    assert "exp_fast(x)" in str(lowered), str(lowered)
