"""Lift a ``LoopOp`` completely into one unmapped Fold-tree ``TileOp``."""

from __future__ import annotations

from emmy.compiler.graph import Node
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.pipeline import Match, Pattern
from emmy.compiler.pipeline.passes.tile._cut import _input_fragment
from emmy.compiler.pipeline.passes.tile._row import lift_kernel
from emmy.compiler.pipeline.passes.tile._split import add_output_piece, state_ports

PATTERN = [Pattern("root", LoopOp)]


def rewrite(match: Match, root: Node, ctx=None):
    del ctx
    from dataclasses import replace  # noqa: PLC0415

    loop: LoopOp = root.op
    tile = lift_kernel(loop, name=loop.name)
    if not tile.carries:
        return replace(tile, outputs={root.output.name: root.output})
    # A carried state: the term carries it (``Fold.cells``), and the classic schedule realizes it
    # as a buffer the kernel owns — so the node gains that port here, a splice where every other
    # lift is a rebind — which threads the loop op in as the tile's ``source``, so the splice does the
    # same. Register storage drops the port again.
    ports = tuple(port for port in state_ports(tile, root.id) if port.name not in root.buffer_names())
    lifted = replace(tile, source=loop)
    return add_output_piece(match, _input_fragment(match, root), root, lifted, list(root.inputs), suffix="__lifted", states=ports)
