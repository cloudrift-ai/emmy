"""Tests for Load deduplication during body normalization."""

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.stmt.blocks import Loop
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.leaves import Assign, Load, Write
from emmy.compiler.ir.stmt.normalize import dedup_loads, normalize_body


def test_dedup_loads_preserves_loads_under_a_rebound_coordinate() -> None:
    """A normalization sum must read every channel when its index shadows the output index."""
    import numpy as np

    from emmy.compiler.ir.loop import LoopOp
    from emmy.compiler.ir.loop.runner import execute_loop_op_cpp
    from emmy.compiler.ir.stmt import Accum

    body = Body((Loop(axis=Axis("k", 4), body=Body((
        Load(name="outer", input="x", index=(Var("k"),)),
        Loop(axis=Axis("k", 4), body=Body((
            Load(name="inner", input="x", index=(Var("k"),)),
            Accum(name="total", value="inner", op="add", axes=("k",)),
        ))),
        Assign(name="value", op="divide", args=("outer", "total")),
        Write(output="out", index=(Var("k"),), value="value"),
    ))),))
    values = np.array([1, 2, 4, 8], dtype=np.float32)
    actual = execute_loop_op_cpp(LoopOp(body=dedup_loads(body)), {"x": values}, {"out": (4,)})

    np.testing.assert_allclose(actual, values / values.sum(), rtol=1e-6)


def test_normalize_body_dedups_loads_and_rewires_gather_indices() -> None:
    body = Body(
        (
            Loop(
                axis=Axis("a", 4),
                body=(
                    Load(name="idx0", input="indices", index=(Var("a"),)),
                    Load(name="idx1", input="indices", index=(Var("a"),)),
                    Load(name="x0", input="values", index=(Var("idx0"),)),
                    Load(name="x1", input="values", index=(Var("idx1"),)),
                    Assign(name="sum", op="add", args=("x0", "x1")),
                    Write(output="out", index=(Var("a"),), value="sum"),
                ),
            ),
        )
    )

    (loop,) = normalize_body(body)

    assert [stmt for stmt in loop.body if isinstance(stmt, Load)] == [
        Load(name="in0", input="indices", index=(Var("a0"),)),
        Load(name="in1", input="values", index=(Var("in0"),)),
    ]
    assert loop.body[-2] == Assign(name="v0", op="add", args=("in1", "in1"))


ZERO = (Literal(0, "int"),)


def test_dedup_loads_does_not_capture_a_rebinding_inner_scope() -> None:
    """A nested scope re-binding a deduped name binds a DIFFERENT variable — the outer alias must
    stop there, or the loop is handed a redeclaration of the survivor and the wrong arithmetic."""
    inner = Body(
        (
            Load(name="in0", input="x", index=ZERO),
            Load(name="in1", input="y", index=ZERO),
            Assign(name="v", op="add", args=("in0", "in1")),
        )
    )
    body = Body(
        (
            Load(name="in0", input="const", index=ZERO),
            Load(name="in1", input="const", index=ZERO),  # duplicate -> dropped, alias in1 -> in0
            Loop(axis=Axis("a", 4), body=inner),
        )
    )

    out = dedup_loads(body)

    assert out[0] == Load(name="in0", input="const", index=ZERO)
    assert out[1].body == inner


def test_dedup_loads_still_rewires_an_inner_use_of_the_dropped_name() -> None:
    """The mirror case: the loop only *reads* the dropped name, so the alias must reach inside."""
    body = Body(
        (
            Load(name="in0", input="const", index=ZERO),
            Load(name="in1", input="const", index=ZERO),
            Loop(axis=Axis("a", 4), body=Body((Assign(name="v", op="add", args=("in0", "in1")),))),
        )
    )

    out = dedup_loads(body)

    assert [s.name for s in out if isinstance(s, Load)] == ["in0"]
    assert out[-1].body == Body((Assign(name="v", op="add", args=("in0", "in0")),))


def test_dedup_loads_rewires_every_vector_lane() -> None:
    """A duplicate vector load aliases each lane to the corresponding kept lane."""
    body = Body(
        (
            Load(names=("x0", "x1"), input="X", index=ZERO),
            Load(names=("y0", "y1"), input="X", index=ZERO),
            Assign(name="sum", op="add", args=("y0", "y1")),
        )
    )

    out = dedup_loads(body)

    assert out == Body(
        (
            Load(names=("x0", "x1"), input="X", index=ZERO),
            Assign(name="sum", op="add", args=("x0", "x1")),
        )
    )


def test_dedup_loads_invalidates_a_read_after_writing_its_buffer() -> None:
    body = Body(
        (
            Load(name="old", input="B", index=ZERO),
            Load(name="replacement", input="R", index=ZERO),
            Write(output="B", index=ZERO, value="replacement"),
            Load(name="new", input="B", index=ZERO),
            Write(output="O", index=ZERO, value="new"),
        )
    )

    out = dedup_loads(body)

    assert [stmt.name for stmt in out if isinstance(stmt, Load) and stmt.input == "B"] == ["old", "new"]


def test_dedup_loads_does_not_reuse_a_read_across_a_loop_that_writes_its_buffer() -> None:
    body = Body(
        (
            Load(name="before", input="B", index=ZERO),
            Loop(
                axis=Axis("a", 4),
                body=(
                    Load(name="current", input="B", index=ZERO),
                    Write(output="B", index=ZERO, value="current"),
                ),
            ),
        )
    )

    out = dedup_loads(body)

    assert out[1].body[0] == Load(name="current", input="B", index=ZERO)


def test_normalize_body_dedups_identical_accumulations() -> None:
    """Two consumers of one fused value carry two copies of its accumulation in one reduce loop
    after the splice; the second is the first under another name."""
    from emmy.compiler.ir.stmt.leaves import Accum

    body = Body(
        (
            Loop(
                axis=Axis("m", 4),
                body=(
                    Loop(
                        axis=Axis("k", 8),
                        body=(
                            Load(name="x0", input="x", index=(Var("m"), Var("k"))),
                            Load(name="w0", input="w", index=(Var("k"),)),
                            Assign(name="p0", op="multiply", args=("x0", "w0")),
                            Accum(name="acc0", value="p0", op="add", axes=("k",)),
                            Load(name="x1", input="x", index=(Var("m"), Var("k"))),
                            Load(name="w1", input="w", index=(Var("k"),)),
                            Assign(name="p1", op="multiply", args=("x1", "w1")),
                            Accum(name="acc1", value="p1", op="add", axes=("k",)),
                        ),
                    ),
                    Assign(name="gate", op="exp", args=("acc0",)),
                    Assign(name="up", op="exp", args=("acc1",)),
                    Assign(name="out", op="multiply", args=("gate", "up")),
                    Write(output="out", index=(Var("m"),), value="out"),
                ),
            ),
        )
    )
    (loop,) = normalize_body(body)
    (inner,) = [stmt for stmt in loop.body if isinstance(stmt, Loop)]
    assert [type(stmt).__name__ for stmt in inner.body] == ["Load", "Load", "Assign", "Accum"]
    assert [stmt for stmt in loop.body if isinstance(stmt, Assign)][-1].args == ("v1", "v1")
