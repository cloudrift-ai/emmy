"""Statement rewriting and projection legality."""

import pytest

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.stmt import Body, Cond, Loop
from emmy.compiler.ir.stmt.leaves import Assign, Write
from emmy.compiler.ir.stmt.normalize import eliminate_copy_aliases
from emmy.compiler.ir.stmt.passes import projection_distributes, rename_free


def test_copy_alias_does_not_capture_an_inner_binding():
    body = Body(
        (
            Assign("b", "copy", ("a",)),
            Cond(
                cond=Var("p"),
                body=(
                    Assign("a", "exp", ("x",)),
                    Assign("v", "subtract", ("a", "b")),
                    Write("out", (), "v"),
                ),
            ),
        )
    )
    (branch,) = eliminate_copy_aliases(body)
    assert branch.body[0].name != "a"
    assert branch.body[1].args == (branch.body[0].name, "a")


@pytest.mark.parametrize("depth", [2, 24])
@pytest.mark.parametrize("eliminate", [False, True], ids=["rename", "eliminate"])
def test_free_rename_visits_nested_statements_once(monkeypatch, depth, eliminate):
    from emmy.compiler.ir.stmt import passes

    stmt = Cond(
        cond=Var("outer"),
        body=(Assign("shadow", "exp", ("outer",)), Write("left", (), "shadow")),
        else_body=(Write("right", (), "outer"),),
    )
    for level in range(depth):
        stmt = Loop(Axis(f"i{level}", 4), (stmt,))
    visits = 0
    original = passes._rewrite_kind

    def counted(*args):
        nonlocal visits
        visits += 1
        return original(*args)

    monkeypatch.setattr(passes, "_rewrite_kind", counted)
    if eliminate:
        result = eliminate_copy_aliases(Body((Assign("outer", "copy", ("renamed",)), Assign("shadow", "copy", ("wrong",)), stmt)))[0]
    else:
        result = rename_free(stmt, {"outer": "renamed", "shadow": "wrong"})
    members = tuple(Body((result,)).iter())
    assert visits <= 2 * len(members)
    branch = next(member for member in members if isinstance(member, Cond))
    assert branch.cond == Var("renamed")
    assert branch.body == Body((Assign("shadow", "exp", ("renamed",)), Write("left", (), "shadow")))
    assert branch.else_body == Body((Write("right", (), "renamed"),))


@pytest.mark.parametrize(
    ("operation", "args", "expected"),
    [
        ("divide", ("state", "count"), True),
        ("divide", ("count", "state"), False),
        ("divide", ("state", "state"), False),
        ("multiply", ("state", "count"), True),
        ("multiply", ("count", "state"), True),
        ("multiply", ("state", "state"), False),
        ("add", ("state", "count"), False),
    ],
)
def test_projection_distributes(operation, args, expected):
    body = [Assign("scaled", operation, args), Write("out", (), value="scaled")]
    assert projection_distributes(body, ("state",)) is expected
