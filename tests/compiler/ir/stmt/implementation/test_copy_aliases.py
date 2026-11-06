"""Copy aliases through value placement, one inner step of normalization."""

from emmy.compiler.dtype import F32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expression import Var
from emmy.compiler.ir.stmt import Assign, Body, Load, Loop, Write
from emmy.compiler.ir.stmt.normalize import place_values


def test_typed_copy_is_a_cast_not_an_alias():
    cast = Assign(name="wide", op="copy", args=("narrow",), dtype=F32)
    assert place_values(Body((cast,))) == Body((cast,))


def test_copy_aliases_do_not_leak_between_sibling_scopes():
    body = Body(
        (
            Loop(
                axis=Axis("i", 4),
                body=(
                    Load(name="x", input="A", index=(Var("i"),)),
                    Assign(name="y", op="copy", args=("x",)),
                    Write(output="OA", index=(Var("i"),), value="y"),
                ),
            ),
            Loop(
                axis=Axis("i", 4),
                body=(
                    Load(name="x", input="B", index=(Var("i"),)),
                    Assign(name="y", op="exp", args=("x",)),
                    Write(output="OB", index=(Var("i"),), value="y"),
                ),
            ),
        )
    )

    out = place_values(body)

    assert out[1].body[1] == Assign(name="y", op="exp", args=("x",))
