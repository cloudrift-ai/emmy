"""``118_unroll_as_nvcc`` writes the copies cicc gave a loop no lowering site flagged, by cicc's size rule, and
``119_vectorize_copies`` widens the runs of loads the copies line up."""

from dataclasses import replace
from importlib import import_module
from types import SimpleNamespace

import pytest

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.kernel.ir import KernelOp, Tile
from emmy.compiler.ir.stmt import Body, Cond, Load, Loop, StridedLoop, parse_body
from emmy.compiler.pipeline import RuleSkipped

_RULE = import_module("emmy.compiler.pipeline.passes.lowering.kernel.118_unroll_as_nvcc")
_WIDEN = import_module("emmy.compiler.pipeline.passes.lowering.kernel.119_vectorize_copies")
_CTX = SimpleNamespace(has_tanh_approx=True)


@pytest.fixture(autouse=True)
def _exact(monkeypatch) -> None:
    """The exact forms: the sizes below count the exponential as cicc's ``expf``."""
    monkeypatch.setenv("EMMY_FAST_MATH", "0")


def _kernel(body: str) -> KernelOp:
    return KernelOp(body=Body((Tile(axes=(Axis("i", 4),), body=parse_body(body)),)), name="k")


def _unrolled(body: str) -> KernelOp:
    return _RULE.rewrite(SimpleNamespace(op=_kernel(body)), _CTX)


def _loops(op: KernelOp) -> list:
    return [stmt for stmt in op.body.iter() if isinstance(stmt, (Loop, StridedLoop))]


def _visits(body: Body, env: dict) -> list:
    """The index of every load ``body`` runs, in order, with the launch's values in ``env``."""
    out = []
    for stmt in body:
        if isinstance(stmt, Load):
            out.append(stmt.index[-1].eval(env))
        elif isinstance(stmt, Cond):
            out.extend(_visits(stmt.body if stmt.cond.eval(env) else stmt.else_body, env))
        elif isinstance(stmt, (Loop, StridedLoop)):
            start, step = (stmt.start.eval(env), stmt.step.eval(env)) if isinstance(stmt, StridedLoop) else (0, 1)
            end = stmt.end if isinstance(stmt, StridedLoop) and stmt.end is not None else stmt.axis.extent.expression
            at = start
            while at < end.eval(env):
                out.extend(_visits(stmt.body, {**env, stmt.axis.name: at}))
                at += step
        elif stmt.nested():
            for inner in stmt.nested():
                out.extend(_visits(inner, env))
    return out


_SUM = "for j in 0..n:\n    f32 a = load A[i, j]\n    f32 b = a * a\n    f32 acc <- add(acc, b) over j\nf32 O[i] = acc"


def test_a_loop_whose_trips_the_launch_decides_takes_sixteen_copies_and_a_remainder() -> None:
    """A three-statement body: 16 copies while 16 trips are left, then blocks of 8 and 4, then a loop of one copy;
    the two loops say they are unrolled. Every trip runs once, in order, whatever the extent."""
    op = _unrolled(_SUM)
    stepping, rest = _loops(op)
    assert (str(stepping.step), stepping.unrolled, str(rest.step), rest.unrolled) == ("16", True, "1", True)
    blocks = [stmt for stmt in op.body.iter() if isinstance(stmt, Cond) and stmt.body and isinstance(stmt.body[0], Load)]
    assert [sum(isinstance(s, Load) for s in block.body) for block in blocks] == [8, 4]
    for n in range(40):
        assert _visits(op.body[0].body, {"i": 0, "n": n}) == list(range(n)), n


def test_a_strided_loop_whose_extent_the_launch_decides_keeps_its_start_and_step() -> None:
    """A lane's loop from its own start: the copies and the remainder visit what the loop did."""
    op = _unrolled("for j in i..n step 4:\n    f32 a = load A[0, j]\n    f32 acc <- add(acc, a) over j\nf32 O[i] = acc")
    for n in range(50):
        for lane in range(4):
            assert _visits(op.body[0].body, {"i": lane, "n": n}) == list(range(lane, n, 4)), (n, lane)


@pytest.mark.parametrize(
    ("body", "step"),
    [
        (_SUM.replace("0..n", "0..1024"), "32"),
        (_SUM.replace("0..n", "0..1024").replace("a * a", "exp(a)"), "8"),
        (_SUM.replace("0..n", "0..96"), "32"),
        (_SUM.replace("0..n", "0..128").replace("a * a", "exp(exp(exp(exp(a))))"), "8"),
    ],
)
def test_a_static_loop_steps_by_the_largest_power_of_two_that_fits_and_divides(body: str, step: str) -> None:
    """Sized 4 a trip (a load at a changing address, a product, a fold), 32 copies fit and divide 1024 and 96; the
    exponential costs what cicc writes for it, 14, so 8 copies fit. Four exponentials fit two copies, which ptxas
    copied four times again: eight."""
    [loop] = _loops(_unrolled(body))
    assert (str(loop.step), loop.unrolled) == (step, True)


def test_a_short_static_loop_is_written_out_in_full() -> None:
    op = _unrolled(_SUM.replace("0..n", "0..16"))
    assert not _loops(op) and sum(isinstance(stmt, Load) for stmt in op.body.iter()) == 16


@pytest.mark.parametrize(("extent", "step"), [(1024, "64"), (96, "32")])
def test_a_lane_s_loop_over_a_static_extent_steps_sixteen_trips(extent: int, step: str) -> None:
    """A loop from each lane's own start, 256 or 24 trips on every lane: 16 copies a trip where 16 divides the
    trips, else the largest power of two that does, as cicc copied a lane's loop it could bound. Every lane visits
    what the loop did."""
    op = _unrolled(f"for j in i..{extent} step 4:\n    f32 a = load A[0, j]\n    f32 acc <- add(acc, a) over j\nf32 O[i] = acc")
    [loop] = _loops(op)
    assert (str(loop.start), str(loop.step), loop.unrolled) == ("i", step, True)
    for lane in range(4):
        assert _visits(op.body[0].body, {"i": lane}) == list(range(lane, extent, 4)), lane


@pytest.mark.parametrize(
    "body",
    [
        _SUM.replace("0..n", "0..1023"),
        "for j in i..64 step 4:\n    f32 a = load A[0, j]\n    f32 acc <- add(acc, a) over j\nf32 O[i] = acc",
        "for j in i..1023 step 4:\n    f32 a = load A[0, j]\n    f32 acc <- add(acc, a) over j\nf32 O[i] = acc",
        "for j in 0..n:\n    f32 a = load A[i, j]\n    Sync()\n    f32 acc <- add(acc, a) over j\nf32 O[i] = acc",
    ],
    ids=["odd-trips", "lane-start-short", "lane-start-ragged", "barrier"],
)
def test_a_loop_cicc_left_to_ptxas_stays(body: str) -> None:
    with pytest.raises(RuleSkipped):
        _unrolled(body)


def test_the_rule_takes_no_loop_it_wrote() -> None:
    with pytest.raises(RuleSkipped):
        _RULE.rewrite(SimpleNamespace(op=_unrolled(_SUM)), _CTX)
    with pytest.raises(RuleSkipped):
        _RULE.rewrite(SimpleNamespace(op=_unrolled(_SUM.replace("0..n", "0..1024"))), _CTX)


def test_the_copies_of_a_load_along_the_axis_are_vector_loads() -> None:
    """Copies stepping 32 trips at a time read consecutive elements from a multiple of 32: eight 4-wide loads."""
    op = _unrolled(_SUM.replace("0..n", "0..1024").replace("A[i, j]", "A[1024 * i + j]"))
    op = _WIDEN.rewrite(SimpleNamespace(op=replace(op, knobs={"VECTORIZE_LOADS": True, "VECTORIZE_STORES": True})))
    [loop] = _loops(op)
    assert [len(stmt.names) for stmt in loop.body if isinstance(stmt, Load)] == [4] * 8
