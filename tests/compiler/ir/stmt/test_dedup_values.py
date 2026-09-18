"""Tests for the one-definition-per-value pass of body normalization."""

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.stmt.blocks import Loop
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Load, Write
from emmy.compiler.ir.stmt.normalize import dedup_values, normalize_body


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


def test_dedup_values_does_not_capture_a_rebinding_inner_scope() -> None:
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

    out = dedup_values(body)

    assert out[0] == Load(name="in0", input="const", index=ZERO)
    assert out[1].body == inner


def test_dedup_values_still_rewires_an_inner_use_of_the_dropped_name() -> None:
    """The mirror case: the loop only *reads* the dropped name, so the alias must reach inside."""
    body = Body(
        (
            Load(name="in0", input="const", index=ZERO),
            Load(name="in1", input="const", index=ZERO),
            Loop(axis=Axis("a", 4), body=Body((Assign(name="v", op="add", args=("in0", "in1")),))),
        )
    )

    out = dedup_values(body)

    assert [s.name for s in out if isinstance(s, Load)] == ["in0"]
    assert out[-1].body == Body((Assign(name="v", op="add", args=("in0", "in0")),))


def test_dedup_values_rewires_every_vector_lane() -> None:
    """A duplicate vector load aliases each lane to the corresponding kept lane."""
    body = Body(
        (
            Load(names=("x0", "x1"), input="X", index=ZERO),
            Load(names=("y0", "y1"), input="X", index=ZERO),
            Assign(name="sum", op="add", args=("y0", "y1")),
        )
    )

    out = dedup_values(body)

    assert out == Body(
        (
            Load(names=("x0", "x1"), input="X", index=ZERO),
            Assign(name="sum", op="add", args=("x0", "x1")),
        )
    )


def test_dedup_values_invalidates_a_read_after_writing_its_buffer() -> None:
    body = Body(
        (
            Load(name="old", input="B", index=ZERO),
            Load(name="replacement", input="R", index=ZERO),
            Write(output="B", index=ZERO, value="replacement"),
            Load(name="new", input="B", index=ZERO),
            Write(output="O", index=ZERO, value="new"),
        )
    )

    out = dedup_values(body)

    assert [stmt.name for stmt in out if isinstance(stmt, Load) and stmt.input == "B"] == ["old", "new"]


def test_dedup_values_does_not_reuse_a_read_across_a_loop_that_writes_its_buffer() -> None:
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

    out = dedup_values(body)

    assert out[1].body[0] == Load(name="current", input="B", index=ZERO)


def _reduce(*body) -> Loop:
    return Loop(axis=Axis("k", 4), body=Body((Load(name="x", input="X", index=(Var("k"),)), *body)))


def test_dedup_values_folds_a_copied_reduction_once() -> None:
    """Fusion inlines a producer once per reader: the same product, folded into two accumulators.
    One value, so one accumulator — and the alias follows it out of the loop to both readers."""
    body = Body(
        (
            _reduce(
                Assign(name="p0", op="multiply", args=("x", "x")),
                Accum(name="s0", value="p0"),
                Assign(name="p1", op="multiply", args=("x", "x")),
                Accum(name="s1", value="p1"),
            ),
            Assign(name="r0", op="exp", args=("s0",)),
            Assign(name="r1", op="exp", args=("s1",)),
            Write(output="O", index=ZERO, value="r0"),
            Write(output="P", index=ZERO, value="r1"),
        )
    )

    loop, *rest = dedup_values(body)

    assert loop.body[1:] == Body((Assign(name="p0", op="multiply", args=("x", "x")), Accum(name="s0", value="p0")))
    assert rest == [
        Assign(name="r0", op="exp", args=("s0",)),
        Write(output="O", index=ZERO, value="r0"),
        Write(output="P", index=ZERO, value="r0"),
    ]


def test_dedup_values_reads_a_commutative_operation_in_either_order() -> None:
    body = Body(
        (
            Load(name="a", input="A", index=ZERO),
            Load(name="b", input="B", index=ZERO),
            Assign(name="u", op="add", args=("a", "b")),
            Assign(name="v", op="add", args=("b", "a")),
            Assign(name="w", op="subtract", args=("b", "a")),
            Assign(name="z", op="subtract", args=("a", "b")),
        )
    )

    assert [s.name for s in dedup_values(body) if isinstance(s, Assign)] == ["u", "w", "z"]


def test_dedup_values_keeps_an_accumulator_two_statements_fold() -> None:
    """``s0`` sums both contributions; ``s1`` sums one. Equal first statements, different values."""
    body = Body((_reduce(Accum(name="s0", value="x"), Accum(name="s0", value="x"), Accum(name="s1", value="x")),))

    (loop,) = dedup_values(body)

    assert [s.name for s in loop.body if isinstance(s, Accum)] == ["s0", "s0", "s1"]


def test_dedup_values_keeps_accumulators_of_different_loops_apart() -> None:
    """Whether two loops sweep one iteration space is the sibling merge's question, not this pass's:
    an accumulator's key never leaves the body that folds it."""
    loops = tuple(Loop(axis=Axis("k", 4), body=Body((Accum(name=name, value="x"),))) for name in ("s0", "s1"))
    body = Body((Load(name="x", input="X", index=ZERO), *loops))

    out = dedup_values(body)

    assert [s.body[0].name for s in out if isinstance(s, Loop)] == ["s0", "s1"]


def test_dedup_values_forgets_a_read_of_an_accumulator_once_it_advances() -> None:
    """A running value read before and after its fold is two values, however alike they spell."""
    body = Body(
        (
            _reduce(
                Assign(name="before", op="exp", args=("s",)),
                Accum(name="s", value="x"),
                Assign(name="after", op="exp", args=("s",)),
                Write(output="O", index=(Var("k"),), value="before"),
                Write(output="P", index=(Var("k"),), value="after"),
            ),
        )
    )

    (loop,) = dedup_values(body)

    assert [s.name for s in loop.body if isinstance(s, Assign)] == ["before", "after"]
