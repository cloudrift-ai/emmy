"""The statement-order step of normalization, called on its own."""

from __future__ import annotations

from itertools import permutations

from emmy.compiler.ir.stmt import Body, normalize_body, parse_body


def test_normalize_order_matches_all_small_dependency_valid_forms() -> None:
    statements = parse_body(
        """
        x = load X[]
        absolute = abs(x)
        y = load Y[]
        result = add(absolute, y)
        O[] = result
        """
    )
    normalized = {normalize_body(Body(order)) for order in permutations(statements)}
    assert len(normalized) == 1


def test_executable_order_ignores_the_iteration_order_of_a_statements_buffers(monkeypatch) -> None:
    """Two projections of one normalized input, symmetric but for their weight buffer: which one
    comes first must not follow the order a set of buffer names iterates in, which is the
    per-process hash seed's. A fused two-projection serving kernel re-spelled on every boot."""
    from emmy.compiler.ir.stmt import order

    def projection(weight: str) -> str:
        k = f"k_{weight}"
        return f"""
        for {k} in 0..8:
            x_{weight} = load X[i, {k}]
            n_{weight} = load N[{k}]
            w_{weight} = load {weight}[{k}, j]
            s_{weight} = multiply(scale, x_{weight})
            t_{weight} = multiply(n_{weight}, s_{weight})
            p_{weight} = multiply(t_{weight}, w_{weight})
            acc_{weight} <- add(acc_{weight}, p_{weight}) over {k}"""

    def make() -> Body:  # fresh each time: a body caches its normal form
        weights = ("W_up", "W_gate")
        inner = "".join(projection(w) for w in weights) + "".join(f"\n        O_{w}[i, j] = acc_{w}" for w in weights)
        return parse_body("for i in 0..4:\n    for j in 0..4:" + inner)

    resources = order._resources
    forms = set()
    for reverse in (False, True):
        monkeypatch.setattr(order, "_resources", lambda stmt, r=reverse: tuple(sorted(names, reverse=r) for names in resources(stmt)))
        forms.add(repr(normalize_body(make())))
    assert len(forms) == 1
