"""The affine normal form: every spelling of one sum, scale or difference is one body with one identity."""

from __future__ import annotations

from emmy.compiler.ir.stmt import Body, normalize_body, parse_body
from tests.compiler.ir.stmt import listing


def _body(*lines: str) -> Body:
    """A loop that reads ``x`` from ``X``, runs ``lines`` and stores ``o``."""
    return parse_body("\n".join(["for i in 0..8:", "    x = load X[i]", *(f"    {line}" for line in lines), "    O[i] = o"]))


def _normal(*lines: str) -> str:
    return str(normalize_body(_body(*lines)))


def _scaled(coefficient: str) -> str:
    """The normal form of ``O[i] = coefficient * X[i]``."""
    return listing(
        f"""
        for a0 in 0..8:
            in0 = load X[a0]
            v0 = {coefficient}*in0
            O[a0] = v0
        """
    )


COPY = listing(
    """
    for a0 in 0..8:
        in0 = load X[a0]
        O[a0] = in0
    """
)


def test_every_spelling_of_a_scale_is_one_body() -> None:
    twice = _body("o = add(x, x)")
    scaled = _body("let f32 c = 2.0", "o = multiply(x, c)")
    halved = _body("let f32 c = 0.5", "o = divide(x, c)")
    # One affine statement, and the literal the fold took is dropped.
    assert str(normalize_body(twice)) == str(normalize_body(scaled)) == str(normalize_body(halved)) == _scaled("2")
    exact = {"structural": False}
    assert twice.structural_key(**exact) == scaled.structural_key(**exact) == halved.structural_key(**exact)
    assert twice.structural_key(**exact) != _body("let f32 c = 3.0", "o = multiply(x, c)").structural_key(**exact)


def test_a_division_by_a_literal_is_the_product_by_its_reciprocal() -> None:
    assert _normal("let f32 c = 4.0", "o = divide(x, c)") == _scaled("0.25")
    assert _normal("let f32 c = 0.25", "o = multiply(x, c)") == _scaled("0.25")
    assert _normal("let f32 c = 1.0", "o = divide(x, c)") == COPY, "a division by one is a copy"


def test_like_terms_sum_and_cancel() -> None:
    assert _normal("let f32 c = 3.0", "t = multiply(x, c)", "s = add(t, x)", "o = subtract(s, x)") == _scaled("3")
    # Every term cancels: the value is the constant zero, which the store reads, and nothing loads ``X``.
    assert _normal("o = subtract(x, x)") == listing(
        """
        let f32 v0 = 0.0
        for a0 in 0..8:
            O[a0] = v0
        """
    )


def test_a_select_whose_branches_bind_one_value_is_that_value() -> None:
    assert _normal("o = select:", "    x when i < i", "    x when i < i") == COPY
