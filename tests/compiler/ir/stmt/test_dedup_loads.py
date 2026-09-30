"""Tests for Load deduplication during body normalization."""

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.stmt.blocks import Cond, Loop
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Init, Let, Load, Select, SelectBranch, Write
from emmy.compiler.ir.stmt.normalize import dedup_loads, hoist_common_branches, normalize_body


def test_dedup_loads_preserves_loads_under_a_rebound_coordinate() -> None:
    """A normalization sum must read every channel when its index shadows the output index."""
    import numpy as np

    from emmy.compiler.ir.loop import LoopOp
    from emmy.compiler.ir.loop.runner import execute_loop_op_cpp
    from emmy.compiler.ir.stmt import Accum

    body = Body(
        (
            Loop(
                axis=Axis("k", 4),
                body=Body(
                    (
                        Load(name="outer", input="x", index=(Var("k"),)),
                        Loop(
                            axis=Axis("k", 4),
                            body=Body(
                                (
                                    Load(name="inner", input="x", index=(Var("k"),)),
                                    Accum(name="total", value="inner", op="add", axes=("k",)),
                                )
                            ),
                        ),
                        Assign(name="value", op="divide", args=("outer", "total")),
                        Write(output="out", index=(Var("k"),), value="value"),
                    )
                ),
            ),
        )
    )
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


def test_cse_does_not_use_expression_printing(monkeypatch) -> None:
    monkeypatch.setattr(BinaryExpr, "pretty", lambda self: "index")
    indices = [BinaryExpr("+", Var("i"), Literal(n, "int")) for n in (1, 2)]
    body = Body(Load(name=f"x{n}", input="x", index=(index,)) for n, index in enumerate(indices))
    assert dedup_loads(body) == body


def test_cse_shares_selections_and_their_downstream_cones() -> None:
    body = Body((
        Let(name="zero", value=Literal(0, "int")),
        Let(name="other_zero", value=Literal(0, "int")),
        Select(name="left", branches=(SelectBranch("x", Var("p")), SelectBranch("zero", Literal(1, "int")))),
        Select(name="right", branches=(SelectBranch("x", Var("p")), SelectBranch("other_zero", Literal(1, "int")))),
        Assign(name="a", op="exp", args=("left",)),
        Assign(name="b", op="exp", args=("right",)),
        Write(output="out", index=ZERO, value="b"),
    ))
    out = dedup_loads(body)
    assert [type(s) for s in out] == [Let, Select, Assign, Write]
    assert out[-1].value == "a"


def test_cse_selection_predicates_observe_rebound_coordinates() -> None:
    def selection(name):
        return Select(name=name, branches=(SelectBranch("x", Var("k")), SelectBranch("y", Literal(1, "int"))))

    body = Body((selection("a"), Loop(axis=Axis("k", 4), body=(selection("b"), Write(output="out", index=(Var("k"),), value="b")))))
    assert dedup_loads(body) == body


def test_cse_does_not_drop_repeated_accumulator_updates() -> None:
    update = Accum(name="sum", value="x", axes=("k",))
    body = Body((Loop(axis=Axis("k", 8), body=(update, update)),))
    assert dedup_loads(body) == body


def test_cse_does_not_alias_distinct_unseeded_accumulators() -> None:
    body = Body((Loop(axis=Axis("k", 8), seed=False, body=(
        Accum(name="left", value="x", axes=("k",)), Accum(name="right", value="x", axes=("k",)),
    )),))
    assert dedup_loads(body) == body


def test_cse_partial_state_changes_invalidate_dependent_values() -> None:
    body = Body((Loop(axis=Axis("k", 8), body=(
        Assign(name="before", op="exp", args=("sum",)),
        Accum(name="sum", value="x", axes=("k",)),
        Assign(name="after", op="exp", args=("sum",)),
        Write(output="out", index=(Var("k"),), value="after"),
    )),))
    assert dedup_loads(body) == body


def test_cse_does_not_reuse_a_staged_load_assignment() -> None:
    body = Body((Load(name="value", input="x", index=ZERO), Load(name="value", input="x", index=ZERO, carried="load")))
    assert dedup_loads(body) == body


def test_cse_does_not_merge_reductions_with_distinct_explicit_seeds() -> None:
    body = Body((Init("left", 0.0, dtype="f32"), Init("right", 1.0, dtype="f32"),
                 Loop(axis=Axis("k", 8), body=(Accum("left", "x"), Accum("right", "x")))))
    assert dedup_loads(body) == body


def test_normalization_closes_simplification_cse_and_invariant_motion() -> None:
    """An equal gather index exposes a constant predicate, then an invariant shared cone."""
    body = Body((
        Load(name="x", input="x", index=ZERO),
        Loop(axis=Axis("i", 4), body=(
            Load(name="a", input="indices", index=(Var("i"),)),
            Load(name="b", input="indices", index=(Var("i"),)),
            Select(name="s", branches=(
                SelectBranch("x", BinaryExpr("<", BinaryExpr("-", Var("a"), Var("b")), Literal(1, "int"))),
                SelectBranch("a", Literal(1, "int")),
            )),
            Assign(name="v", op="exp", args=("s",)),
            Write(output="out", index=(Var("i"),), value="v"),
        )),
    ))
    normalized = normalize_body(body)
    assert any(isinstance(s, Assign) and s.op.name == "exp" for s in normalized)
    assert normalize_body(Body(tuple(normalized))) == normalized


def test_cse_factors_common_branch_cones_without_speculation() -> None:
    def branch(prefix, extra):
        return Body((
            Load(name=f"{prefix}i", input="indices", index=ZERO),
            Load(name=f"{prefix}x", input="x", index=(Var(f"{prefix}i"),)),
            Assign(name=f"{prefix}v", op="exp", args=(f"{prefix}x",)),
            Load(name=f"{prefix}only", input=extra, index=ZERO),
            Assign(name=f"{prefix}out", op="add", args=(f"{prefix}v", f"{prefix}only")),
            Write(output="out", index=ZERO, value=f"{prefix}out"),
        ))
    body = Body((Cond(cond=Var("predicate"), body=branch("a", "left"), else_body=branch("b", "right")),))
    out = normalize_body(body)
    assert [type(stmt) for stmt in out] == [Load, Load, Assign, Cond]
    assert out[2].op.name == "exp"
    for inner in out[-1].nested():
        assert [type(stmt) for stmt in inner] == [Load, Assign, Write]
        assert out[2].name in inner[1].args
    assert normalize_body(Body(tuple(out))) == out


def test_common_branch_load_cannot_cross_a_write_on_either_path() -> None:
    read = Load(name="value", input="x", index=ZERO)
    write = Write(output="x", index=ZERO, value="replacement")
    result = Write(output="out", index=ZERO, value="value")
    for left, right in (((write, read, result), (read, result)), ((read, result), (write, read, result))):
        body = Body((Cond(cond=Var("predicate"), body=left, else_body=right),))
        assert hoist_common_branches(body) == body


def test_normalized_quotient_addresses_fuse_and_share_the_whole_reduction() -> None:
    def reduction(axis, name, nested):
        index = BinaryExpr("/", Var(axis), Literal(128 if nested else 256, "int"))
        if nested:
            index = BinaryExpr("/", index, Literal(2, "int"))
        return Loop(axis=Axis(axis, 1024), body=(
            Load(name=f"{name}x", input="x", index=(index,)),
            Assign(name=f"{name}v", op="exp", args=(f"{name}x",)),
            Accum(name=name, value=f"{name}v", axes=(axis,)),
        ))
    body = Body((reduction("i", "left", True), reduction("j", "right", False),
                 Write(output="out", index=ZERO, value="right"), Write(output="other", index=ZERO, value="left")))
    out = normalize_body(body)
    assert len(out.loads) == len(out.accums) == 1
    assert len([s for s in out.iter() if isinstance(s, Loop)]) == 1
    assert out[-1].value == out[-2].value
    assert normalize_body(Body(tuple(out))) == out


def test_dedup_loads_closes_commutative_chains_after_aliasing() -> None:
    """An alias that reverses sorted operands must not hide the rest of a duplicate chain."""
    body = Body(
        (
            Load(name="z", input="X", index=ZERO),
            Load(name="a", input="X", index=ZERO),
            Load(name="m", input="Y", index=ZERO),
            Assign(name="z1", op="multiply", args=("m", "z")),
            Assign(name="a1", op="multiply", args=("a", "m")),
            Assign(name="z2", op="add", args=("m", "z1")),
            Assign(name="a2", op="add", args=("a1", "m")),
            Assign(name="left", op="subtract", args=("m", "z1")),
            Assign(name="right", op="subtract", args=("a1", "m")),
            Write(output="O", index=ZERO, value="a2"),
            Write(output="L", index=ZERO, value="left"),
            Write(output="R", index=ZERO, value="right"),
        )
    )

    out = dedup_loads(body)

    assert [stmt.name for stmt in out if isinstance(stmt, Assign)] == ["z1", "z2", "left", "right"]
    assert out[-4] == Assign(name="right", op="subtract", args=("z1", "m"))
    assert out[-3] == Write(output="O", index=ZERO, value="z2")
    assert dedup_loads(out) == out


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


def test_cse_alias_cannot_be_captured_by_its_destination_name() -> None:
    body = Body((
        Load(name="a", input="x", index=ZERO),
        Load(name="b", input="x", index=ZERO),
        Cond(cond=Var("p"), body=(
            Load(name="a", input="y", index=ZERO),
            Assign(name="v", op="subtract", args=("a", "b")),
            Write(output="out", index=ZERO, value="v"),
        )),
    ))
    out = dedup_loads(body)
    assert len(out) == 2
    inner = out[1].body
    assert inner[0].name != out[0].name
    assert inner[1].args == (inner[0].name, out[0].name)


def test_cse_reuses_dominating_values_in_both_branches() -> None:
    outer = Load(name="a", input="x", index=ZERO)
    body = Body((outer, Cond(cond=Var("p"), body=(
        Load(name="b", input="x", index=ZERO), Write(output="left", index=ZERO, value="b"),
    ), else_body=(Load(name="c", input="x", index=ZERO), Write(output="right", index=ZERO, value="c")))))
    out = dedup_loads(body)
    assert len(out.loads) == 1
    assert all(child[0].value == "a" for child in out[1].nested())


def test_cse_reuses_a_dominating_read_until_the_write_on_that_branch() -> None:
    body = Body((Load("old", "x", ZERO), Cond(cond=Var("p"), body=(
        Load("before", "x", ZERO), Write("out", ZERO, "before"),
        Write("x", ZERO, "replacement"), Load("after", "x", ZERO), Write("new", ZERO, "after"),
    ))))
    out = dedup_loads(body)
    assert out[1].body[0] == Write("out", ZERO, "old")
    assert [load.name for load in out.loads] == ["old", "after"]


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
