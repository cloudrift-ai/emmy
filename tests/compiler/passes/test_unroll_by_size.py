"""``118_unroll_by_size`` writes the copies NVIDIA's compilers gave a loop no lowering site flagged, by its size, and
``119_vectorize_copies`` widens the runs of loads the copies line up."""

from dataclasses import replace
from importlib import import_module
from types import SimpleNamespace

import pytest

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.kernel.ir import KernelOp, Tile
from emmy.compiler.ir.stmt import Body, Cond, Load, Loop, parse_body
from emmy.compiler.pipeline import RuleSkipped

_RULE = import_module("emmy.compiler.pipeline.passes.lowering.kernel.118_unroll_by_size")
_WIDEN = import_module("emmy.compiler.pipeline.passes.lowering.kernel.119_vectorize_copies")
_CTX = SimpleNamespace(has_tanh_approx=True)


@pytest.fixture(autouse=True)
def _exact(monkeypatch) -> None:
    """The exact forms: the sizes below count the exponential as cicc's ``expf``."""
    monkeypatch.setenv("EMMY_FAST_MATH", "0")


def _kernel(body: str) -> KernelOp:
    return KernelOp(body=Body((Tile(axes=(Axis("i", 4),), body=parse_body(body)),)))


def _unrolled(body: str) -> KernelOp:
    return _RULE.rewrite(SimpleNamespace(op=_kernel(body)), _CTX)


def _loops(op: KernelOp) -> list:
    return [stmt for stmt in op.body.iter() if isinstance(stmt, Loop)]


def _visits(body: Body, env: dict) -> list:
    """The index of every load ``body`` runs, in order, with the launch's values in ``env``."""
    out = []
    for stmt in body:
        if isinstance(stmt, Load):
            out.append(stmt.index[-1].eval(env))
        elif isinstance(stmt, Cond):
            out.extend(_visits(stmt.body if stmt.cond.eval(env) else stmt.else_body, env))
        elif isinstance(stmt, Loop):
            start, end, step = stmt.walk()
            at = start.eval(env)
            while at < end.eval(env):
                out.extend(_visits(stmt.body, {**env, stmt.axis.name: at}))
                at += step.eval(env)
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
        (_SUM.replace("0..n", "0..192"), "32"),
        (_SUM.replace("0..n", "0..128").replace("a * a", "exp(exp(exp(exp(a))))"), "8"),
    ],
)
def test_a_static_loop_steps_by_the_largest_power_of_two_that_fits_and_divides(body: str, step: str) -> None:
    """Sized 4 a trip (a load at a changing address, a product, a fold), 32 copies fit and divide 1024 and 192; the
    exponential costs what cicc writes for it, 14, so 8 copies fit. Four exponentials fit two copies, which ptxas
    copied four times again: eight."""
    [loop] = _loops(_unrolled(body))
    assert (str(loop.step), loop.unrolled) == (step, True)


def test_a_short_static_loop_is_written_out_in_full() -> None:
    op = _unrolled(_SUM.replace("0..n", "0..16"))
    assert not _loops(op) and sum(isinstance(stmt, Load) for stmt in op.body.iter()) == 16


def test_a_stepping_loop_of_a_few_small_trips_is_written_out_in_full() -> None:
    """96 trips sized 4 would step 32 copies three times; ptxas wrote such a loop out, and so does the pass. A larger
    body steps 16 copies six times and stays a loop."""
    op = _unrolled(_SUM.replace("0..n", "0..96"))
    assert not _loops(op) and sum(isinstance(stmt, Load) for stmt in op.body.iter()) == 96
    [loop] = _loops(_unrolled(_SUM.replace("0..n", "0..96").replace("a * a", "a * a * a * a * a")))
    assert (str(loop.step), loop.unrolled) == ("16", True)


@pytest.mark.parametrize(("extent", "step"), [(1024, "64"), (96, "32")])
def test_a_lane_s_loop_over_a_static_extent_steps_sixteen_trips(extent: int, step: str) -> None:
    """A loop from each lane's own start, 256 or 24 trips on every lane: 16 copies a trip where 16 divides the
    trips, else the largest power of two that does, as cicc copied a lane's loop it could bound. Every lane visits
    what the loop did."""
    op = _unrolled(f"for j in i..{extent} step 4:\n    f32 a = load A[0, j]\n    f32 acc <- add(acc, a) over j\nf32 O[i] = acc")
    [loop] = _loops(op)
    assert (str(loop.walk()[0]), str(loop.step), loop.unrolled) == ("i", step, True)
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


def test_the_stores_of_a_running_sum_are_vector_stores() -> None:
    """A prefix sum written out stores its accumulator after each step: one name, another value each time. Each
    store takes a copy of the value it stores, so the four stores of a run stand together behind the fourth step
    and become one. A fold that stores its total once has no run and takes no copy."""
    from emmy.compiler.dtype import F32
    from emmy.compiler.graph import Tensor
    from emmy.compiler.ir.expression import Var
    from emmy.compiler.ir.stmt import Accum, Assign, Write

    running = """
        for j in 0..16:
            f32 x = load A[16 * i + j]
            f32 acc <- add(acc, x) over j
            f32 out[16 * i + j] = acc
    """
    op = replace(_kernel(running), inputs={"A": Tensor("A", (64,), F32)}, outputs={"out": Tensor("out", (64,), F32)})
    op = _RULE.rewrite(SimpleNamespace(op=op), _CTX)
    assert not _loops(op)
    op = _WIDEN.rewrite(SimpleNamespace(op=replace(op, knobs={"VECTORIZE_LOADS": True, "VECTORIZE_STORES": True})))
    stores = [stmt for stmt in op.body.iter() if isinstance(stmt, Write)]
    assert [len(stmt.values) for stmt in stores] == [4] * 4
    # In order: every value a store holds is a copy taken right after the step that made it, and the last store
    # holds the accumulator itself, which no later step changes.
    flat = list(op.body.iter())
    copies = {stmt.name: flat.index(stmt) for stmt in flat if isinstance(stmt, Assign) and stmt.value == Var("acc")}
    steps = [position for position, stmt in enumerate(flat) if isinstance(stmt, Accum)]
    assert [value for stmt in stores for value in stmt.values] == [*sorted(copies, key=copies.get), "acc"]
    assert all(any(step < at < later for at in copies.values()) for step, later in zip(steps, steps[1:], strict=False))


def test_the_copies_of_a_store_along_the_axis_are_vector_stores() -> None:
    """A loop stepping 32 trips at a time states its step, so the axis is a multiple of 32 in every copy: the stores
    of a copy loop are eight 4-wide stores a trip, proved aligned as its loads are. Without the loop's step the
    stores proved nothing and stayed scalar, two of every three memory operations of a kernel that only copies."""
    from emmy.compiler.dtype import F32
    from emmy.compiler.graph import Tensor
    from emmy.compiler.ir.stmt import Write

    copy = """
        for j in 0..1024:
            f32 x = load A[1024 * i + j]
            f32 out[1024 * i + j] = x
    """
    buffers = {"inputs": {"A": Tensor("A", (4096,), F32)}, "outputs": {"out": Tensor("out", (4096,), F32)}}
    op = _RULE.rewrite(SimpleNamespace(op=replace(_kernel(copy), **buffers)), _CTX)
    op = _WIDEN.rewrite(SimpleNamespace(op=replace(op, knobs={"VECTORIZE_LOADS": True, "VECTORIZE_STORES": True})))
    [loop] = _loops(op)
    assert [len(stmt.names) for stmt in loop.body if isinstance(stmt, Load)] == [4] * 8
    assert [len(stmt.values) for stmt in loop.body if isinstance(stmt, Write)] == [4] * 8
