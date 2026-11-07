"""A body's text: ``str`` and ``parse_body`` are exact inverses, over every statement kind and every stored kernel."""

import json
from textwrap import dedent

import pytest

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F8E4M3, F16, F32
from emmy.compiler.ir.axis import Axis, Window
from emmy.compiler.ir.expression import BinaryExpr, Literal, Var
from emmy.compiler.ir.stmt import (
    Accum,
    Affine,
    Assign,
    Body,
    Call,
    Carry,
    Cond,
    Init,
    Let,
    Load,
    Loop,
    Pre,
    Select,
    SelectBranch,
    Statement,
    StridedLoop,
    Subroutine,
    Write,
    ZeroPrologue,
    parse_body,
)
from tests.compiler.realization import helpers

A0, A1, ONE = Var("a0"), Var("a1"), Literal(1, "int")
K = Axis("k", 64)

STATEMENTS = [
    (Load(("in0",), "x", (A0, A1 + ONE)), "in0 = load x[a0, a1 + 1]"),
    (Load(names=("in0", "in1"), input="x", index=(A0,), dtype=F8E4M3), "f8e4m3 in0, in1 = load x[a0]"),
    (Load(("in0",), "x", (A0,), carried="unpack"), "in0 = load x[a0] carried=unpack"),
    (Load(("in0",), "model.layers.0.weight", (A0,)), "in0 = load `model.layers.0.weight`[a0]"),
    (Pre("p", "state", (A0, Literal(0, "int"))), "p = pre state[a0, 0]"),
    (Assign("v1", "multiply", ("in0", "in1")), "v1 = multiply(in0, in1)"),
    (Assign("v1", "exp", ("in0",), F16), "f16 v1 = exp(in0)"),
    (Assign("load", "copy", ("for",)), "`load` = copy(`for`)"),
    (Affine("v2", ("x",), (1,)), "v2 = x"),
    (Affine("v2", ("x", "y"), (-1, 0.5)), "v2 = -x + 0.5*y"),
    (Affine("v2", ("x", "y"), (2, -1), 1e-06, F32), "f32 v2 = 2*x - y + 1e-06"),
    (Affine("v2", ("x",), (3,), -4), "v2 = 3*x - 4"),
    (Let("v0", Literal(0.125)), "let v0 = 0.125"),
    (Let("i", A0 * Literal(4, "int") + A1, F32), "let f32 i = a0 * 4 + a1"),
    (Init("acc", float("-inf"), dtype=F32), "init f32 acc = -inf"),
    (Accum("acc0", "v1"), "acc0 <- add(acc0, v1)"),
    (Accum("acc0", "v1", "maximum", F32, ("a1", "a2")), "f32 acc0 <- maximum(acc0, v1) over a1, a2"),
    (Accum("acc0", "v1", base="scaled"), "acc0 <- add(scaled, v1)"),
    (Carry("state", "v", (A0, A1), 0.0), "state[a0, a1] <- v seed 0"),
    (Carry("state", "v", (A0,), "state_in", F32), "f32 state[a0] <- v seed state_in"),
    (Write("y", (A0, A1), ("v",)), "y[a0, a1] = v"),
    (Write("y", (A0,), values=("v0", "v1"), value_dtype=F16), "f16 y[a0] = (v0, v1)"),
    (Write("y", (A0,), ("v",), atomic=True), "y[a0] += v"),
    (Write("slab", (A0,), ("v",), swizzle="B32"), "slab[a0] = v swizzle=B32"),
    (ZeroPrologue("acc", 1024), "zero acc words=1024"),
    (
        Select("v3", (SelectBranch("x", A0.lt(ONE)), SelectBranch("y", BinaryExpr(">=", A0, ONE)))),
        """\
        v3 = select:
            x when a0 < 1
            y when a0 >= 1""",
    ),
    (
        Loop(Axis("a0", 4), (Write("y", (A0,), ("v",)),)),
        """\
        for a0 in 0..4:
            y[a0] = v""",
    ),
    (Loop(Axis("a0", Dim("seq", hint=128)), (), unroll=True, seed=False), "for a0 in 0..seq~128 unroll=true seed=false:"),
    (Loop(Axis("a0", Dim(Var("seq") * Literal(2, "int"), hint=64)), ()), "for a0 in 0..seq * 2~64:"),
    (
        Loop(Axis("kb", 16, window=Window(parent=Axis("ko", 32, window=Window(parent=K)), partition=True)), ()),
        "for kb in 0..16 window(parent=ko in 0..32 window(parent=k in 0..64), partition=true):",
    ),
    (Loop(Axis("kb", 16, window=Window(base=A0, bound=A0 + ONE)), ()), "for kb in 0..16 window(base=a0, bound=a0 + 1):"),
    (Loop(Axis("kb", 16, window=Window()), ()), "for kb in 0..16 window():"),
    (
        StridedLoop(K, Var("t"), Literal(32, "int"), (Write("y", (Var("k"),), ("v",)),)),
        """\
        for k in t..64 step 32:
            y[k] = v""",
    ),
    (
        StridedLoop(K, Literal(0, "int"), Literal(8, "int"), (), unroll=True, end=Var("k_end")),
        "for k in 0..64 step 8 until k_end unroll=true:",
    ),
    (
        Cond(A0.lt(ONE), (Write("y", (A0,), ("v",)),), (Write("y", (A0,), ("w",)),)),
        """\
        if a0 < 1:
            y[a0] = v
        else:
            y[a0] = w""",
    ),
    (Cond(A0.lt(ONE), ()), "if a0 < 1:"),
]


def _exact(body: Body) -> list:
    """Everything the statements hold, including what equality leaves out: an axis's window, a dim's hint."""
    axes = [stmt.axis for stmt in body.iter() if isinstance(stmt, (Loop, StridedLoop))]
    return [list(body.iter()), [(axis.window, axis.extent.hint) for axis in axes]]


@pytest.mark.parametrize(("stmt", "text"), STATEMENTS, ids=[type(stmt).__name__ for stmt, _ in STATEMENTS])
def test_a_statement_and_its_text_read_each_other_back(stmt, text):
    text = dedent(text)
    assert str(stmt) == text
    assert _exact(parse_body(text)) == _exact(Body((stmt,)))


def test_a_body_is_its_statements_one_under_another():
    text = dedent(
        """\
        let f32 v0 = 0.125
        for a0 in 0..4:
            for a1 in 0..8:
                in0 = load x[a0, a1]
                v1 = 2*in0 + v0
                acc0 <- add(acc0, v1) over a1
            y[a0] = acc0"""
    )
    body = parse_body(text)
    assert str(body) == text
    assert [type(stmt).__name__ for stmt in body.iter()] == ["Let", "Loop", "Loop", "Load", "Affine", "Accum", "Write"]
    assert parse_body("") == Body() and str(Body()) == ""


def test_the_reader_takes_any_consistent_indentation_and_blank_lines():
    assert parse_body("\n  for a0 in 0..4:\n\n      y[a0] = v\n") == parse_body("for a0 in 0..4:\n    y[a0] = v")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("y[a0] = v\n    z[a0] = v", "line 2: unexpected indentation"),
        ("for a0 in 0..4:\n        y[a0] = v\n    z[a0] = v", "line 3: unexpected indentation"),
        ("v = 1", "line 1: .*no term"),
        ("y[a0] = v\nv1 = multiply(a, b", "line 2: expected"),
        ("v = select:\n    x where a0 < 1", "line 2: expected 'when'"),
        ("for a0 in 2..4:", "line 1: .*names its step"),
        ("else:", "line 1"),
        ("a, b = multiply(x, y)", "line 1: only a load binds several names"),
        ("v = x + ", "line 1: unexpected end"),
        ("y[a0] = v extra", "line 1: unexpected text"),
    ],
)
def test_a_reading_error_names_its_line(text, message):
    with pytest.raises(ValueError, match=message):
        parse_body(text)


CALLS = """\
sub inner(r in 0..4) -> s:
    for k in 0..8:
        x = load a[r, k]
        s <- add(s, x) over k
sub outer(r in 0..4, c in 0..2 window(parent=cc in 0..4)) -> t:
    u = call inner[a](r)
    w = load b[c]
    t = multiply(u, w)
for i in 0..4:
    p = call outer[a, b](i, 0)
    q = call outer[a, b](i, 1)
    z = call outer[m, b](i, 1)
    v = p + q + z
    y[i] = v"""


def test_a_call_names_a_subroutine_defined_once_ahead_of_it():
    """Subroutines compare by object, so reading a text back keeps which calls share one, and the text itself."""
    body = parse_body(CALLS)
    assert str(body) == CALLS
    p, q, z = (stmt.target for stmt in body.iter_of_type(Call))
    assert p is q and p.name == z.name == "outer"
    assert (p.buffers, z.buffers, p.params, p.result) == (("a", "b"), ("m", "b"), ("r", "c"), "t")
    assert p.axes[1].window.parent.name == "cc"
    assert str(parse_body(str(body))) == CALLS


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("p = call missing[a](i)", "line 1: no subroutine 'missing' is defined above"),
        ("sub f(r in 0..4) -> s:\n    x = load a[r]\np = call f[a](0)", "line 1: .*must define its result"),
        ("sub f(r in 0..4) -> x:\n    x = load a[r]\np = call f[a](0, 1)", "line 3: .*do not match its parameters"),
    ],
)
def test_a_call_that_does_not_fit_its_subroutine_is_refused(text, message):
    with pytest.raises(ValueError, match=message):
        parse_body(text)


def test_two_subroutines_of_one_name_have_no_text():
    def sub(buffer: str) -> Subroutine:
        return Subroutine(
            "f", (Axis("r", 4),), (Load(("x",), buffer, (Var("r"),)), Assign("y", "exp" if buffer == "a" else "log", ("x",))), "y"
        )

    body = Body((Call("p", sub("a"), (A0,)), Call("q", sub("b"), (A0,))))
    with pytest.raises(ValueError, match="two subroutines are named 'f'"):
        str(body)


def _stored_bodies(value) -> list:
    if isinstance(value, dict):
        own = [value["attrs"]["body"]] if value.get("op") == "loop" else []
        return own + [body for item in value.values() for body in _stored_bodies(item)]
    return [body for item in value for body in _stored_bodies(item)] if isinstance(value, list) else []


def test_every_corpus_kernel_reads_back_from_its_text():
    stored = [Body.from_wire(wire) for path in helpers.case_files() for wire in _stored_bodies(json.loads(path.read_text()))]
    assert len(stored) > 200
    for body in stored:
        back = parse_body(str(body))
        assert _exact(back) == _exact(body)
        assert str(back) == str(body)


#: Six cases whose Kernel IR reaches every kind the corpus lowers to; ``SetMaxNReg`` is in no corpus kernel.
KERNEL_CASES = [
    "attention/sdpa-hd64-wgmma-value-sm90.json",
    "matmul/f16-mma-f16acc-tma.json",
    "matmul/f16-warp-band-p1-tma-d1-m128.json",
    "matmul/fp4-tma-w4a4-linear-ring.json",
    "reduce/computed-provider-coop-t-sm70.json",
    "scan/f32-cumsum-warp-prefix.json",
]


def _kernel_bodies(case_path: str) -> list[Body]:
    from emmy.compiler.ir.kernel.ir import KernelOp

    case = helpers.load_case(helpers.CASES_DIR / case_path)
    graph, _ = helpers.lowered(case, case.context())
    return [op.body for node in graph.nodes.values() for op in node.op.source_chain() if isinstance(op, KernelOp)]


@pytest.mark.parametrize("case_path", KERNEL_CASES)
def test_a_kernel_ir_body_reads_back_from_its_text(case_path):
    """A kind of the Kernel IR prints and reads through its declaration, so a lowered kernel's text is exact."""
    bodies = _kernel_bodies(case_path)
    assert bodies
    for body in bodies:
        back = parse_body(str(body))
        assert _exact(back) == _exact(body)
        assert str(back) == str(body)


def test_every_kernel_ir_kind_has_a_text():
    from emmy.compiler.ir.kernel import ir as kernel

    kinds = {
        kind
        for case_path in KERNEL_CASES
        for body in _kernel_bodies(case_path)
        for stmt in body.iter()
        if (kind := type(stmt)).__module__ == kernel.__name__
    }
    declared = {
        cls for cls in vars(kernel).values() if isinstance(cls, type) and issubclass(cls, Statement) and cls.__module__ == kernel.__name__
    }
    assert declared - kinds == {kernel.SetMaxNReg}
    assert str(Body((kernel.SetMaxNReg(count=232, direction="inc"),))) == "SetMaxNReg(count=232, direction=inc)"
