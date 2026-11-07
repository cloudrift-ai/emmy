"""Every error the crate raises reaches Python as the exception of the same name, with its place in the text."""

from __future__ import annotations

import pytest

from emmy.compiler.ir.expression import Expression, Var
from emmy.compiler.ir.kernel import ir as kernel_ir
from emmy.compiler.ir.stmt import Body, normalize_body, parse_body

_CALLS = """
sub f(r in 0..4) -> s:
    s = exp(r)
x = call f[](1)
O[] = x
"""


def _two_subroutines_of_one_name() -> str:
    first, second = parse_body(_CALLS), parse_body(_CALLS.replace("exp", "abs"))
    return str(Body((first[0], second[0], first[1])))


@pytest.mark.parametrize(
    ("raise_it", "message"),
    [
        (lambda: parse_body("x = load"), "line 1: "),
        (lambda: parse_body("sub f(r in 0..4) -> s:\n    s = exp(r)\nx = call f[a](1, 2)\nO[] = x"), "line 3: subroutine arguments"),
        (_two_subroutines_of_one_name, "subroutine"),
        (lambda: Expression.from_wire("a +"), "position 3"),
        (lambda: str(Var("a`b")), "no spelling"),
        (lambda: normalize_body(parse_body(_CALLS)), "no normal form"),
        (lambda: kernel_ir.WgmmaMma(c_frags=("c0",), b_desc="bD", shape=(16, 8, 16), a_desc="aD"), "m64n"),
        (
            lambda: kernel_ir.WgmmaMma(c_frags=tuple(f"c{i}" for i in range(8)), b_desc="bD", shape=(64, 64, 16), a_desc="aD", a_frag="a"),
            "exactly one of a_desc / a_frag",
        ),
    ],
    ids=[
        "reader",
        "call-arity",
        "two-subroutines",
        "expression-wire",
        "backtick-name",
        "call-normal-form",
        "wgmma-shape",
        "wgmma-operands",
    ],
)
def test_a_value_error_names_its_place(raise_it, message) -> None:
    with pytest.raises(ValueError, match=message):
        raise_it()


def test_a_wrong_type_is_a_type_error() -> None:
    with pytest.raises(TypeError):
        Body(("not a statement",))
    with pytest.raises(TypeError, match="Load.names"):
        kernel_ir.Load(names="x", input="a", index=(Var("i"),))
