"""The split-index recomposition fold in ``BinaryExpression.simplify``: ``(x/c)*c + x%c → x``.

The free-axis re-fusion the tile lift applies spells a split store's coordinates as ``f/Q`` and
``f%Q`` in separate buffer dims; the row-major address flatten then produces exactly
``(f/Q)*(Q·s) + (f%Q)·s`` chains, and this fold collapses them back to the affine ``f·s`` so
the emitted address is byte-identical to the unsplit spelling. The identity holds for every
int under C truncated division AND Python floor division, so the fold is unconditional."""

from __future__ import annotations

import pytest

from emmy.compiler.ir.expression import BinaryExpression, Literal, SimplifyCtx, Var


def _simplify(e):
    return e.simplify(SimplifyCtx.empty())


def _lit(v: int) -> Literal:
    return Literal(v, "int")


def test_bare_pair_recomposes():
    f = Var("f")
    e = BinaryExpression("+", BinaryExpression("*", BinaryExpression("//", f, _lit(512)), _lit(512)), BinaryExpression("%", f, _lit(512)))
    assert _simplify(e) == f


@pytest.mark.parametrize("c,k", [(512, 1), (64, 2), (24, 512), (8, 8)])
def test_scaled_pair_preserves_value(c: int, k: int):
    """``(f/c)*(c·k) + (f%c)·k`` folds to ``f·k`` — numerically pinned over a range."""
    f = Var("f")
    quot = BinaryExpression("*", BinaryExpression("//", f, _lit(c)), _lit(c * k))
    rem = BinaryExpression("%", f, _lit(c)) if k == 1 else BinaryExpression("*", BinaryExpression("%", f, _lit(c)), _lit(k))
    e = BinaryExpression("+", quot, rem)
    s = _simplify(e)
    assert "%" not in str(s) and "//" not in str(s) and "/" not in str(s)
    for v in range(0, 4 * c + 3):
        assert e.eval({"f": v}) == s.eval({"f": v}), f"mismatch at f={v}: {s}"


def test_pair_inside_longer_chain():
    """Other addends survive around the recomposed pair — the flattened store address shape
    (``row·N + (f/Q)·Q + f%Q``)."""
    f, row = Var("f"), Var("row")
    e = BinaryExpression(
        "+",
        BinaryExpression(
            "+", BinaryExpression("*", row, _lit(12288)), BinaryExpression("*", BinaryExpression("//", f, _lit(512)), _lit(512))
        ),
        BinaryExpression("%", f, _lit(512)),
    )
    s = _simplify(e)
    assert "%" not in str(s)
    for fv, rv in ((0, 0), (511, 1), (513, 3), (12287, 7)):
        assert e.eval({"f": fv, "row": rv}) == s.eval({"f": fv, "row": rv})


def test_mismatched_divisors_do_not_fold():
    f = Var("f")
    e = BinaryExpression("+", BinaryExpression("*", BinaryExpression("//", f, _lit(512)), _lit(512)), BinaryExpression("%", f, _lit(256)))
    s = _simplify(e)
    for v in (0, 255, 256, 511, 513):
        assert e.eval({"f": v}) == s.eval({"f": v})
    assert "%" in str(s), "different divisors must not recompose"


def test_mismatched_scale_does_not_fold():
    """``(f/c)*a + f%c`` with ``a`` not a multiple of ``c`` stays as-is."""
    f = Var("f")
    e = BinaryExpression("+", BinaryExpression("*", BinaryExpression("//", f, _lit(512)), _lit(500)), BinaryExpression("%", f, _lit(512)))
    s = _simplify(e)
    for v in (0, 511, 512, 1025):
        assert e.eval({"f": v}) == s.eval({"f": v})
    assert "%" in str(s)


def test_different_dividends_do_not_fold():
    f, g = Var("f"), Var("g")
    e = BinaryExpression("+", BinaryExpression("*", BinaryExpression("//", f, _lit(64)), _lit(64)), BinaryExpression("%", g, _lit(64)))
    s = _simplify(e)
    assert "%" in str(s)
