"""A cut piece whose grid coordinate carries a fused (head, head-dim) pair splits it back into two.

Attention's output reaches the o_proj as one flat channel ``i``; a cut piece computing P.V inherits
that spelling, so its P operand reads ``i / d`` (the head) while the V operand reads the channel's low
part. Split, the piece is a batched matmul the warp tier tiles; fused, it is a per-cell reduce.
"""

from __future__ import annotations

import pytest

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.pipeline.passes.tile._cut import _fused_pair_factor
from tests.compiler.terms import contraction, slab

_AXES = (Axis("m", 8), Axis("i", 16))


def _pv(v_index) -> object:
    p = slab("p", "P", "m", Var("i") / 4, "k")
    return contraction(Axis("k", 32), p, (slab("v", "V", v_index, "k"), "acc"))


@pytest.mark.parametrize("v_index", [Var("i"), Var("i") % 4], ids=["workspace-at-the-fused-channel", "remainder"])
def test_a_head_read_beside_the_low_part_is_a_fused_pair(v_index) -> None:
    """V read at the plain channel is what a cut V projection leaves: its workspace is indexed by the
    fused name. It depends on the low part as much as ``i % d`` does."""
    assert _fused_pair_factor(_pv(v_index), _AXES) == ("i", 4)


def test_a_plain_coordinate_on_both_operands_is_no_pair() -> None:
    """Without a division on one side the coordinate is an ordinary axis; splitting it re-spells a
    kernel the tier already schedules."""
    term = contraction(Axis("k", 32), slab("p", "P", "m", "i", "k"), (slab("v", "V", "i", "k"), "acc"))
    assert _fused_pair_factor(term, _AXES) is None
