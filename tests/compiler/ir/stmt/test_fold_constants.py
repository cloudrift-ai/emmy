"""Constant folding inside normalization: a value literals alone determine is the literal it computes."""

from __future__ import annotations

from emmy.compiler.dtype import F16, F32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Accum, Assign, Body, Let, Load, Loop, Write


def _body(*stmts) -> Body:
    i = Var("i")
    return Body((Loop(Axis("i", 8), (Load(name="v", input="x", index=(i,)), Load(name="s", input="y", index=(i,)), *stmts)),))


def _ops(op: LoopOp) -> list[str]:
    return [type(s).__name__ if not isinstance(s, Assign) else s.op.name for s in op.body.iter() if not isinstance(s, (Loop, Load))]


def test_one_key_softmax_weight_folds_to_nothing() -> None:
    """``v * (exp(s - s) / exp(s - s))`` is ``v``: the difference is zero, its exponential one, the quotient one."""
    i = Var("i")
    op = LoopOp(
        body=_body(
            Assign("z", "subtract", ("s", "s")),
            Assign("u", "exp", ("z",)),
            Assign("w", "divide", ("u", "u")),
            Assign("o", "multiply", ("v", "w")),
            Write(output="O", index=(i,), value="o"),
        )
    )
    assert _ops(op) == ["Write"] and op.body.writes[0].value in {load.name for load in op.body.loads}, op.pretty_body()


def test_a_constant_read_by_a_write_keeps_its_statement() -> None:
    i = Var("i")
    op = LoopOp(
        body=_body(
            Assign("z", "subtract", ("s", "s")),
            Assign("u", "exp", ("z",)),
            Write(output="O", index=(i,), value="u"),
        )
    )
    (constant,) = [s for s in op.body.iter() if isinstance(s, Let)]
    assert (constant.value, constant.dtype) == (Literal(1.0), F32) and constant.name == op.body.writes[0].value, op.pretty_body()


def test_a_typed_constant_is_not_read_through() -> None:
    """A stamped dtype is a rounding the program performed: the literal keeps it, and nothing computes past it."""
    i = Var("i")
    op = LoopOp(
        body=_body(
            Assign("z", "subtract", ("s", "s")),
            Assign("u", "exp", ("z",)),
            Assign("h", "copy", ("u",), F16),
            Assign("o", "multiply", ("v", "h")),
            Write(output="O", index=(i,), value="o"),
        )
    )
    (constant,) = [s for s in op.body.iter() if isinstance(s, Let)]
    assert (constant.value, constant.dtype) == (Literal(1.0), F16) and "multiply" in _ops(op), op.pretty_body()


def test_a_score_nothing_reads_takes_its_dot_product_with_it() -> None:
    """A one-key softmax weight is ``1`` whatever the score, so the scaled score, the dot product that fed it
    and the loop that summed it reach no output: normalization drops them all, and keeps the stores."""
    i, k = Var("i"), Var("k")
    dot = Loop(
        Axis("k", 4),
        (
            Load(name="q", input="Q", index=(i, k)),
            Load(name="kk", input="K", index=(i, k)),
            Assign("qk", "multiply", ("q", "kk")),
            Accum(name="acc", value="qk", axes=("k",)),
        ),
    )
    op = LoopOp(
        body=Body(
            (
                Loop(
                    Axis("i", 8),
                    (
                        dot,
                        Let("scale", Literal(0.125), F32),
                        Assign("s", "multiply", ("acc", "scale")),
                        Assign("z", "subtract", ("s", "s")),
                        Assign("w", "exp", ("z",)),
                        Load(name="v", input="V", index=(i,)),
                        Assign("o", "multiply", ("v", "w")),
                        Write(output="O", index=(i,), value="o"),
                    ),
                ),
            )
        )
    )
    assert _ops(op) == ["Write"] and {load.input for load in op.body.loads} == {"V"}, op.pretty_body()
