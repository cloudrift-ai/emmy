"""Inner steps of normalization called on their own: expression canonicalization, body preparation, renaming."""

from __future__ import annotations

from itertools import product

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.stmt import Assign, Body, Cond, Load, Loop, Write, normalize_body
from emmy.compiler.ir.stmt.normalize import _canonicalize_exprs, _normalize_body, prepare_body, rename_ssa_sequential


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
