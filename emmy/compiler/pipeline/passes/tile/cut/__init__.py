"""Kernel-set decisions to a fixpoint: every stored Fold-edge cut, then every cross-CTA split."""

from __future__ import annotations

from emmy.compiler.ir.tile import TileOp


def takes_kernel_set_decision(op) -> bool:
    """Whether ``op`` is a kernel the cut pass still decides: a tile kernel with a tree, neither placed on a grid
    nor scheduled. The one gate of the pass's rule, and of every offer that is one of its decisions (the layout
    fork's eligibility, which serving reads to settle a shared weight's layout)."""
    return isinstance(op, TileOp) and op.op is not None and not op.place.is_mapped and op.schedule is None
