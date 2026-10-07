"""``fold_unit_factors``: a factor that is one wherever it is not NaN folds into a subtraction, bit for bit."""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.stmt.blocks import Loop
from emmy.compiler.ir.stmt.leaves import Assign, Load, Write
from emmy.compiler.ir.stmt.normalize import normalize_body

SPECIAL = [0.0, -0.0, 2.5, -3.0, 1e-30, 6e4, np.inf, -np.inf, np.nan]


def _one_key_softmax_times_value() -> tuple:
    """``out[i] = v[i] * (exp(s[i] - s[i]) / exp(s[i] - s[i]))`` — a one-key attention weight."""
    return (
        Loop(
            axis=Axis("i", 8),
            body=(
                Load(name="s", input="S", index=(Var("i"),)),
                Load(name="v", input="V", index=(Var("i"),)),
                Assign(name="d", op="subtract", args=("s", "s")),
                Assign(name="e", op="exp", args=("d",)),
                Assign(name="w", op="divide", args=("e", "e")),
                Assign(name="o", op="multiply", args=("v", "w")),
                Write(output="out", index=(Var("i"),), value="o"),
            ),
        ),
    )


def _ops(body) -> list[str]:
    return [stmt.op.name for stmt in body.iter() if isinstance(stmt, Assign)]


def test_one_key_softmax_weight_folds_to_a_subtraction() -> None:
    assert _ops(normalize_body(_one_key_softmax_times_value())) == ["subtract", "subtract"]


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_fold_is_bit_exact_for_every_value(dtype) -> None:
    s, v = (np.array(x, dtype=dtype).reshape(-1) for x in np.meshgrid(SPECIAL, SPECIAL))
    with np.errstate(all="ignore"):
        d = s - s
        e = np.exp(d)
        before = v * (e / e)
        after = v - d
        quotient = v / e
    assert before.dtype == after.dtype == dtype
    bits = np.uint16 if dtype == np.float16 else np.uint32
    for result in (before, quotient):
        nan = np.isnan(result)
        assert np.array_equal(nan, np.isnan(after))
        assert np.array_equal(result[~nan].view(bits), after[~nan].view(bits))  # -0 stays -0

def test_a_factor_read_elsewhere_is_kept() -> None:
    loop = _one_key_softmax_times_value()[0]
    body = (Loop(axis=loop.axis, body=(*loop.body, Write(output="weight", index=(Var("i"),), value="w"))),)
    assert sorted(_ops(normalize_body(body))) == ["divide", "exp", "subtract", "subtract"]


def test_inexact_identities_are_left_alone() -> None:
    i = Var("i")
    body = (
        Loop(
            axis=Axis("i", 8),
            body=(
                Load(name="s", input="S", index=(i,)),
                Load(name="v", input="V", index=(i,)),
                Assign(name="d", op="subtract", args=("s", "s")),
                Assign(name="q", op="divide", args=("s", "s")),
                Assign(name="e", op="exp", args=("s",)),
                Assign(name="o", op="multiply", args=("v", "q")),
                Assign(name="p", op="multiply", args=("o", "e")),
                Assign(name="r", op="add", args=("p", "d")),
                Write(output="out", index=(i,), value="r"),
            ),
        ),
    )
    assert sorted(_ops(normalize_body(body))) == ["add", "divide", "exp", "multiply", "multiply", "subtract"]
