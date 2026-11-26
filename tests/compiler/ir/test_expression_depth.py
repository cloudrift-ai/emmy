"""An expression too deep for the compiler's walks is refused where it is built, whatever built it."""

from __future__ import annotations

import pytest

from emmy.compiler.ir.expression import Var


def test_a_chain_of_a_thousand_additions_is_refused() -> None:
    expression = Var("a")
    with pytest.raises(RecursionError):
        for _ in range(1000):
            expression = expression + Var("b")


def test_a_chain_within_the_limit_is_accepted_whole() -> None:
    expression = Var("a")
    for _ in range(200):
        expression = expression + Var("b")
    assert str(expression).count("+") == 200
