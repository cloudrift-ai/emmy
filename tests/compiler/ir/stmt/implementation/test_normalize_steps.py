"""Inner steps of normalization called on their own: expression canonicalization, body preparation, renaming."""

from __future__ import annotations

from itertools import product

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.stmt import Accum, Assign, Body, Cond, Load, Loop, Select, SelectBranch, Write, normalize_body
from emmy.compiler.ir.stmt.normalize import (
    _canonicalize_exprs,
    _normalize_body,
    drop_unread_values,
    fold_values,
    prepare_body,
    rename_ssa_sequential,
)


def test_atomic_partition_axis_order_converges_independently_of_affine_spelling() -> None:
    """An axis absent from an atomic output must not trade places with a scaled output axis every round."""
    forms = set()
    for names, reverse_loops, reverse_sum in product((("row", "part"), ("z", "a")), (False, True), (False, True)):
        row, part = names
        terms = (Var(row) * 256, Var(part) * 64)
        index = terms[1] + terms[0] if reverse_sum else terms[0] + terms[1]
        body = Body((Load(name="x", input="X", index=(index,)), Write(output="O", index=(Var(row) * 2,), value="x", atomic=True)))
        for name in names if reverse_loops else reversed(names):
            body = Body((Loop(axis=Axis(name, 2), body=body),))
        prepared = rename_ssa_sequential(body)
        # Check a bounded number of rounds before calling the fixed-point driver: the regression used to cycle.
        for _ in range(4):
            prepared = prepare_body(prepared)
        assert prepare_body(prepared) == prepared
        normalized = normalize_body(body)
        assert normalize_body(Body.from_wire(normalized.to_wire())) == normalized
        forms.add(normalized)
    assert len(forms) == 1


def test_a_commutative_comparison_orders_its_axes_by_binding_not_spelling() -> None:
    """Normalization orders a commutative comparison's operands by where each axis is bound. Ordered by spelling, a
    body re-spelled by its own normalization — as a golden's stored kernel is when it is read back — normalized again
    to another body, and the kernel's identity moved."""

    def condition(outer: str, inner: str) -> BinaryExpr:
        body = Body(
            (
                Loop(
                    axis=Axis(outer, 4),
                    body=(
                        Loop(
                            axis=Axis(inner, 4),
                            body=(
                                Load(name="x", input="X", index=(Var(outer), Var(inner))),
                                Cond(
                                    BinaryExpr("==", Var(inner), Var(outer)),
                                    body=(Write(output="O", index=(Var(outer), Var(inner)), value="x"),),
                                ),
                            ),
                        ),
                    ),
                ),
            )
        )
        return _canonicalize_exprs(body)[0].body[0].body[1].cond

    assert condition("a10", "a5") == BinaryExpr("==", Var("a10"), Var("a5"))
    assert condition("a5", "a10") == BinaryExpr("==", Var("a5"), Var("a10"))


def test_normal_form_is_a_fixed_point_when_the_order_breaks_a_tie_by_spelling() -> None:
    """Two loop-invariant values of one shape tie in the canonical order, which breaks the tie by their spelling: the
    order runs again on its own output until it leaves it alone, so normalizing the normal form changes nothing."""
    tile = (Literal(0, "int"), Var("a0"), Var("a2") / Literal(64, "int"), Var("a2") % Literal(64, "int"))
    body = Body(
        (
            Loop(
                Axis("a0", 48),
                (
                    Loop(
                        Axis("a2", 512),
                        (
                            Assign("v36", "multiply", ("v35", "v3")),
                            Assign("v37", "pad", ("v36",)),
                            Write(output="reshape_9", index=tile, value="v37"),
                            Loop(
                                Axis("a7", 128),
                                (
                                    Assign("v146", "pad", ("v145",)),
                                    Write(output="reshape_5", index=(*tile, Var("a7")), value="v146"),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
    )
    once = _normalize_body(body)
    assert repr(_normalize_body(once)) == repr(once)


def _one_key_softmax() -> Body:
    """``v * (exp(s - s) / exp(s - s))``, which is ``v``."""
    i = Var("i")
    return Body(
        (
            Loop(
                Axis("i", 8),
                (
                    Load(name="v", input="x", index=(i,)),
                    Load(name="s", input="y", index=(i,)),
                    Assign("z", "subtract", ("s", "s")),
                    Assign("u", "exp", ("z",)),
                    Assign("w", "divide", ("u", "u")),
                    Assign("o", "multiply", ("v", "w")),
                    Write(output="O", index=(i,), value="o"),
                ),
            ),
        )
    )


def _equal_gather_index() -> Body:
    """Two reads of one index make a predicate constant, the select it decided a copy, and its reader invariant."""
    i = Var("i")
    decided = Select(
        name="s",
        branches=(
            SelectBranch("x", BinaryExpr("<", BinaryExpr("-", Var("a"), Var("b")), Literal(1, "int"))),
            SelectBranch("a", Literal(1, "int")),
        ),
    )
    loop = (
        Load(name="a", input="indices", index=(i,)),
        Load(name="b", input="indices", index=(i,)),
        decided,
        Assign(name="v", op="exp", args=("s",)),
        Write(output="out", index=(i,), value="v"),
    )
    return Body((Load(name="x", input="x", index=(Literal(0, "int"),)), Loop(axis=Axis("i", 4), body=loop)))


def test_unread_values_leave_with_everything_only_they_read() -> None:
    """A dropped value leaves its operands unread in turn: one call drops the whole chain."""
    i = Var("i")
    chain = (Load(name="y", input="Y", index=(i,)), Assign("a", "exp", ("y",)), Assign("b", "exp", ("a",)), Assign("c", "exp", ("b",)))
    kept = (Load(name="x", input="X", index=(i,)), Write(output="O", index=(i,), value="x"))
    assert drop_unread_values(Body((Loop(Axis("i", 8), (*chain, *kept)),))) == Body((Loop(Axis("i", 8), kept),))


def test_the_folds_run_to_their_own_fixed_point() -> None:
    """The folds feed each other: the difference is zero to the affine fold, its exponential and the quotient one to
    the constant fold, and the product by it a copy to the affine fold again. One call settles all of it."""
    folded = fold_values(prepare_body(rename_ssa_sequential(_one_key_softmax())))
    assert fold_values(folded) is folded
    assert [s.op.name for s in folded.iter() if isinstance(s, Assign)] == ["copy"]


def test_no_step_changes_a_prepared_body_the_folds_leave_alone() -> None:
    """The driver runs the steps again only after a fold changed the body, so a prepared body the folds leave alone
    has to be one every step leaves alone."""
    for body in (_one_key_softmax(), _equal_gather_index()):
        prepared = rename_ssa_sequential(body)
        for _ in range(8):
            prepared = prepare_body(prepared)
            folded = fold_values(prepared)
            if folded is prepared:
                break
            prepared = folded
        else:
            raise AssertionError("the folds did not settle")
        assert prepare_body(prepared) == prepared


def test_a_reduction_only_another_bindings_readers_kept_alive_is_dropped() -> None:
    """A cut can splice one cone twice under the same names: a statistic, then a contraction that reads it, twice.
    Placement keeps the first statistic for both readers and the merge folds the second into a contraction, where
    nothing reads it. It shares the first one's name, and the unread-value drop reads by name, so only the names the
    order gives tell the two apart: the normal form holds one statistic, and normalization leaves it alone."""
    i, j, k, m = Var("i"), Var("j"), Var("k"), Var("m")

    def cone(weights: str, out: str) -> tuple:
        statistic = Loop(Axis("k", 4), (Load(name="x", input="X", index=(i, k)), Assign("sq", "multiply", ("x", "x")), Accum("s", "sq")))
        contraction = Loop(
            Axis("m", 4),
            (
                Load(name="y", input="X", index=(i, m)),
                Assign("scaled", "multiply", ("y", "r")),
                Load(name=f"w_{out}", input=weights, index=(m, j)),
                Assign(f"p_{out}", "multiply", (f"w_{out}", "scaled")),
                Accum(out, f"p_{out}"),
            ),
        )
        return statistic, Assign("r", "rsqrt", ("s",)), contraction

    writes = (Write(output="O1", index=(i, j), value="o1"), Write(output="O2", index=(i, j), value="o2"))
    body = Body((Loop(Axis("i", 8), (Loop(Axis("j", 3), (*cone("W1", "o1"), *cone("W2", "o2"), *writes)),)),))
    normalized = normalize_body(body)
    assert len(normalized.accums) == 3
    assert _normalize_body(normalized) == normalized
