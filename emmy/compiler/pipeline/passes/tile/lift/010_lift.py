"""Lift a ``LoopOp`` completely into one unmapped Fold-tree ``TileOp``."""

from __future__ import annotations

from emmy.compiler.dtype import F32
from emmy.compiler.graph import Node, Tensor
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.pipeline import Match, Pattern
from emmy.compiler.pipeline.passes.tile._cut import _input_fragment
from emmy.compiler.pipeline.passes.tile._row import lift_kernel
from emmy.compiler.pipeline.passes.tile._split import add_output_piece

PATTERN = [Pattern("root", LoopOp)]


def rewrite(match: Match, root: Node, ctx=None):
    del ctx
    from dataclasses import replace  # noqa: PLC0415

    loop: LoopOp = root.op
    tile = lift_kernel(loop, name=loop.name)
    if not tile.carries:
        return replace(tile, outputs={root.output.name: root.output})
    # A carried state: the term carries it (``Fold.cells``), and the classic schedule realizes it
    # as a buffer the kernel owns over the step axis, the free axes outside the carrying loop and
    # the cells (the index ``states_as_buffers`` writes) — so the node gains that port here, a
    # splice where every other lift is a rebind. Register storage drops it again.
    outer = tuple(axis.name for axis in tile.place.free)
    states = tuple(
        Tensor(name=f"{root.id}__{state}", shape=tuple(tile.axis_of(axis).extent for axis in (node.axis, *outer, *node.cells)), dtype=F32)
        for node in (site.node for site in tile.sites if site.node.carries)
        for state in node.base.results
    )
    return add_output_piece(match, _input_fragment(match, root), root, tile, list(root.inputs), suffix="__lifted", states=states)
