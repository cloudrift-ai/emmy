"""The body's own topological order, as Python calls it; the constraints and the identity it serves are tested in the crate."""

from __future__ import annotations

from emmy.compiler.ir.stmt import parse_body


def test_topological_order_keeps_an_order_the_constraints_allow() -> None:
    body = parse_body(
        """
        right = exp(x)
        left = abs(x)
        """
    )
    incoming = (frozenset(), frozenset())
    assert body.topological_order(incoming) == body
