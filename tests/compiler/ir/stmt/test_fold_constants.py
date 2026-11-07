"""Constant folding inside normalization: a value literals alone determine is the literal it computes."""

from __future__ import annotations

from emmy.compiler.ir.stmt import normalize_body, parse_body
from tests.compiler.ir.stmt import listing


def _normal(*lines: str) -> str:
    """The normal form of a loop that reads ``v`` from ``x`` and ``s`` from ``y``, then runs ``lines``."""
    body = "\n".join(["for i in 0..8:", "    v = load x[i]", "    s = load y[i]", *(f"    {line}" for line in lines)])
    return str(normalize_body(parse_body(body)))


def test_one_key_softmax_weight_folds_to_nothing() -> None:
    """``v * (exp(s - s) / exp(s - s))`` is ``v``: the difference is zero, its exponential one, the quotient one."""
    assert _normal("z = subtract(s, s)", "u = exp(z)", "w = divide(u, u)", "o = multiply(v, w)", "O[i] = o") == listing(
        """
        for a0 in 0..8:
            in0 = load x[a0]
            O[a0] = in0
        """
    )


def test_a_constant_read_by_a_write_keeps_its_statement() -> None:
    assert _normal("z = subtract(s, s)", "u = exp(z)", "O[i] = u") == listing(
        """
        let f32 v0 = 1.0
        for a0 in 0..8:
            O[a0] = v0
        """
    )


def test_a_typed_constant_is_not_read_through() -> None:
    """A stamped dtype is a rounding the program performed: the literal keeps it, and nothing computes past it."""
    assert _normal("z = subtract(s, s)", "u = exp(z)", "f16 h = copy(u)", "o = multiply(v, h)", "O[i] = o") == listing(
        """
        let f16 v0 = 1.0
        for a0 in 0..8:
            in0 = load x[a0]
            v1 = multiply(in0, v0)
            O[a0] = v1
        """
    )


def test_a_score_nothing_reads_takes_its_dot_product_with_it() -> None:
    """A one-key softmax weight is ``1`` whatever the score, so the scaled score, the dot product that fed it
    and the loop that summed it reach no output: normalization drops them all, and keeps the stores."""
    body = parse_body(
        """
        for i in 0..8:
            for k in 0..4:
                q = load Q[i, k]
                kk = load K[i, k]
                qk = multiply(q, kk)
                acc <- add(acc, qk) over k
            let f32 scale = 0.125
            s = multiply(acc, scale)
            z = subtract(s, s)
            w = exp(z)
            v = load V[i]
            o = multiply(v, w)
            O[i] = o
        """
    )
    assert str(normalize_body(body)) == listing(
        """
        for a0 in 0..8:
            in0 = load V[a0]
            O[a0] = in0
        """
    )
