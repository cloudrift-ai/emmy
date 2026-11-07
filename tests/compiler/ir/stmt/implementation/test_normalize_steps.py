"""Inner steps of normalization called on their own: expression canonicalization, body preparation, renaming."""

from __future__ import annotations

from itertools import product
from textwrap import indent

from emmy.emmy_ir import (
    canonicalize_expressions,
    drop_unread_values,
    fold_affine,
    fold_trivial_selects,
    normalize_uncached,
    prepare_body,
    rename_ssa_sequential,
)

from emmy.compiler.ir.stmt import Body, normalize_body, parse_body


def test_atomic_partition_axis_order_converges_independently_of_affine_spelling() -> None:
    """An axis absent from an atomic output must not trade places with a scaled output axis every round."""
    forms = set()
    for names, reverse_loops, reverse_sum in product((("row", "part"), ("z", "a")), (False, True), (False, True)):
        row, part = names
        terms = (f"{row} * 256", f"{part} * 64")
        index = f"{terms[1]} + {terms[0]}" if reverse_sum else f"{terms[0]} + {terms[1]}"
        text = f"x = load X[{index}]\nO[{row} * 2] += x"
        for name in names if reverse_loops else reversed(names):
            text = f"for {name} in 0..2:\n{indent(text, '    ')}"
        body = parse_body(text)
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

    def condition(outer: str, inner: str) -> str:
        body = parse_body(
            f"""
            for {outer} in 0..4:
                for {inner} in 0..4:
                    x = load X[{outer}, {inner}]
                    if {inner} == {outer}:
                        O[{outer}, {inner}] = x
            """
        )
        return str(canonicalize_expressions(body)[0].body[0].body[1].cond)

    assert condition("a10", "a5") == "a10 == a5"
    assert condition("a5", "a10") == "a5 == a10"


def test_normal_form_is_a_fixed_point_when_the_order_breaks_a_tie_by_spelling() -> None:
    """Two loop-invariant values of one shape tie in the canonical order, which breaks the tie by their spelling: the
    spelling reads nothing the order decides, so normalizing the normal form changes nothing."""
    body = parse_body(
        """
        for a0 in 0..48:
            for a2 in 0..512:
                v36 = multiply(v35, v3)
                v37 = pad(v36)
                reshape_9[0, a0, a2 / 64, a2 % 64] = v37
                for a7 in 0..128:
                    v146 = pad(v145)
                    reshape_5[0, a0, a2 / 64, a2 % 64, a7] = v146
        """
    )
    once = normalize_uncached(body)
    assert repr(normalize_uncached(once)) == repr(once)


def _one_key_softmax() -> Body:
    """``v * (exp(s - s) / exp(s - s))``, which is ``v``."""
    return parse_body(
        """
        for i in 0..8:
            v = load x[i]
            s = load y[i]
            z = subtract(s, s)
            u = exp(z)
            w = divide(u, u)
            o = multiply(v, w)
            O[i] = o
        """
    )


def _equal_gather_index() -> Body:
    """Two reads of one index make a predicate constant, the select it decided a copy, and its reader invariant."""
    return parse_body(
        """
        x = load x[0]
        for i in 0..4:
            a = load indices[i]
            b = load indices[i]
            s = select:
                x when a - b < 1
                a when 1
            v = exp(s)
            out[i] = v
        """
    )


def test_unread_values_leave_with_everything_only_they_read() -> None:
    """A dropped value leaves its operands unread in turn: one call drops the whole chain."""
    chain = ("y = load Y[i]", "a = exp(y)", "b = exp(a)", "c = exp(b)")
    kept = ("x = load X[i]", "O[i] = x")

    def loop(*lines: str) -> Body:
        return parse_body("\n".join(("for i in 0..8:", *(f"    {line}" for line in lines))))

    assert drop_unread_values(loop(*chain, *kept)) == loop(*kept)


def test_no_step_changes_a_prepared_body_the_folds_leave_alone() -> None:
    """The driver runs the steps again only after a fold changed the body, so a prepared body the folds leave alone
    has to be one every step leaves alone."""
    for body in (_one_key_softmax(), _equal_gather_index()):
        prepared = rename_ssa_sequential(body)
        for _ in range(8):
            prepared = prepare_body(prepared)
            folded = fold_affine(fold_trivial_selects(prepared))
            if folded == prepared:
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

    def cone(weights: str, out: str) -> str:
        return f"""
            for k in 0..4:
                x = load X[i, k]
                sq = multiply(x, x)
                s <- add(s, sq)
            r = rsqrt(s)
            for m in 0..4:
                y = load X[i, m]
                scaled = multiply(y, r)
                w_{out} = load {weights}[m, j]
                p_{out} = multiply(w_{out}, scaled)
                {out} <- add({out}, p_{out})"""

    writes = """
            O1[i, j] = o1
            O2[i, j] = o2"""
    body = parse_body(f"for i in 0..8:\n    for j in 0..3:{cone('W1', 'o1')}{cone('W2', 'o2')}{writes}")
    normalized = normalize_body(body)
    assert len(normalized.accums) == 3
    assert normalize_uncached(normalized) == normalized
