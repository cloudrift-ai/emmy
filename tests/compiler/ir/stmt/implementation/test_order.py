"""The statement-order step of normalization, called on its own."""

from __future__ import annotations

from itertools import permutations

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.stmt import Assign, Body, Load, Loop, Write, normalize_body


def test_normalize_order_matches_all_small_dependency_valid_forms() -> None:
    statements = (
        Load(name="x", input="X", index=()),
        Assign(name="absolute", op="abs", args=("x",)),
        Load(name="y", input="Y", index=()),
        Assign(name="result", op="add", args=("absolute", "y")),
        Write(output="O", index=(), value="result"),
    )
    normalized = {normalize_body(Body(order)) for order in permutations(statements)}
    assert len(normalized) == 1


def test_executable_order_ignores_the_iteration_order_of_a_statements_buffers(monkeypatch) -> None:
    """Two projections of one normalized input, symmetric but for their weight buffer: which one
    comes first must not follow the order a set of buffer names iterates in, which is the
    per-process hash seed's. A fused two-projection serving kernel re-spelled on every boot."""
    from emmy.compiler.ir.stmt import Accum, order

    def projection(weight: str) -> Loop:
        k = f"k_{weight}"
        return Loop(
            axis=Axis(k, 8),
            body=(
                Load(name=f"x_{weight}", input="X", index=(Var("i"), Var(k))),
                Load(name=f"n_{weight}", input="N", index=(Var(k),)),
                Load(name=f"w_{weight}", input=weight, index=(Var(k), Var("j"))),
                Assign(name=f"s_{weight}", op="multiply", args=("scale", f"x_{weight}")),
                Assign(name=f"t_{weight}", op="multiply", args=(f"n_{weight}", f"s_{weight}")),
                Assign(name=f"p_{weight}", op="multiply", args=(f"t_{weight}", f"w_{weight}")),
                Accum(name=f"acc_{weight}", op="add", value=f"p_{weight}", axes=(k,)),
            ),
        )

    def make() -> Body:  # fresh each time: a body caches its normal form
        weights = ("W_up", "W_gate")
        writes = tuple(Write(output=f"O_{w}", index=(Var("i"), Var("j")), value=f"acc_{w}") for w in weights)
        return Body((Loop(axis=Axis("i", 4), body=(Loop(axis=Axis("j", 4), body=(*map(projection, weights), *writes)),)),))

    resources = order._resources
    forms = set()
    for reverse in (False, True):
        monkeypatch.setattr(order, "_resources", lambda stmt, r=reverse: tuple(sorted(names, reverse=r) for names in resources(stmt)))
        forms.add(repr(normalize_body(make())))
    assert len(forms) == 1
