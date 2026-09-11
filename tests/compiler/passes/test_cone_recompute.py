"""A computed operand cone that reads one traced fold through two edges lowers that fold once.

Attention's output is normalized by its own row sum, and the tree forms the softmax fold twice on
the way — once under the normalize, once under the reciprocal it derives — as two equal ``Fold``
nodes it could not share. The o_proj's compute fill replicates the cone per cell, so lowering both
declared every state of that fold twice in one scope (``acc3__c3 has already been declared``: the
prefill SDPA + o_proj + residual target on every card). The seam keeps the first lowering only.
No GPU."""

from __future__ import annotations

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.schedule.views import cone_seam
from emmy.compiler.ir.stmt import Body, Load, Loop
from emmy.compiler.ir.stmt.body import dedup_recomputes
from emmy.compiler.ir.stmt.leaves import Assign
from tests.compiler.terms import contraction, projection

S = Axis("s", Dim(8))


def _pv():
    """The P·V fold over the key axis, reading the contraction axis ``k`` of the o_proj it feeds — built
    afresh each time, the way the tree forms one traced value twice."""
    p = Load(name="p", input="p", index=(Var("m"), Var("s")))
    v = Load(name="v", input="v", index=(Var("s"), Var("k")))
    return contraction(S, p, (v, "acc"))


def _cone():
    """``out = acc · f(acc)``: the fold read directly and through a derived edge."""
    derived = projection(
        operands=(_pv(),), body=(Assign(name="inv", op=ElementwiseImpl("multiply"), args=("acc", "acc")),), results=("inv",)
    )
    return projection(
        operands=(_pv(), derived),
        body=(Assign(name="out", op=ElementwiseImpl("multiply"), args=("acc", "inv")),),
        results=("out",),
    )


def test_the_seam_lowers_a_twice_read_fold_once() -> None:
    _, cell, _, _ = cone_seam(_cone(), "k", axes=(S,))
    loops = [stmt for stmt in cell if isinstance(stmt, Loop)]
    assert len(loops) == 1
    defined = [name for stmt in cell for name in stmt.defines()]
    assert len(defined) == len(set(defined)), defined
    assert {"inv", "out"} <= Body(cell).ssa_defs


def test_dedup_keeps_a_repeated_effect_and_the_first_definition() -> None:
    fold = tuple(_pv().lower(axes=(S,)))
    twice = (*fold, *fold)
    assert dedup_recomputes(twice) == fold
    assert dedup_recomputes(fold) == fold
