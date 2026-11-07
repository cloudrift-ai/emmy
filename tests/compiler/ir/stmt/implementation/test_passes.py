"""Statement rewriting."""

import pytest
from emmy.emmy_ir import place_values

from emmy.compiler.ir.expression import Var
from emmy.compiler.ir.stmt import Body, Cond, Statement, parse_body, rename_free
from tests.compiler.ir.stmt import listing


@pytest.mark.parametrize("depth", [2, 24])
@pytest.mark.parametrize("eliminate", [False, True], ids=["rename", "eliminate"])
def test_free_rename_visits_nested_statements_once(monkeypatch, depth, eliminate):
    text = listing(
        """
        if outer:
            shadow = exp(outer)
            left[] = shadow
        else:
            right[] = outer
        """
    )
    for level in range(depth):
        text = f"for i{level} in 0..4:\n" + "\n".join(f"    {line}" for line in text.splitlines())
    (stmt,) = parse_body(text)
    visits = 0
    original = Statement.rewrite

    def counted(*args):
        nonlocal visits
        visits += 1
        return original(*args)

    monkeypatch.setattr(Statement, "rewrite", counted)
    if eliminate:
        result = place_values(parse_body(f"outer = copy(renamed)\nshadow = copy(wrong)\n{text}"))[0]
    else:
        result = rename_free(stmt, {"outer": "renamed", "shadow": "wrong"})
    members = tuple(Body((result,)).iter())
    assert visits <= 3 * len(members), "linear in the statements, whatever the depth"
    branch = next(member for member in members if isinstance(member, Cond))
    assert branch.cond == Var("renamed")
    assert str(branch.body) == "shadow = exp(renamed)\nleft[] = shadow"
    assert str(branch.else_body) == "right[] = renamed"
