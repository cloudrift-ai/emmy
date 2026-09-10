"""Lift a ``LoopOp`` completely into one unmapped Fold-tree ``TileOp``."""

from __future__ import annotations

from emmy.compiler.graph import Node
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import Match, Pattern
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import lift_loop_op
from emmy.compiler.pipeline.passes.lowering.tile._row import ROW_AXIS, binds_the_row, row_bound_body, row_candidates

PATTERN = [Pattern("root", LoopOp)]


def rewrite(match: Match, root: Node, ctx=None) -> TileOp:
    del match, ctx
    from dataclasses import replace  # noqa: PLC0415

    loop: LoopOp = root.op
    tile = lift_loop_op(loop, name=loop.name)
    # A contraction that owns no free axis has no row for any tier to tile. Its row is a size-one
    # output dimension Loop-IR normalization inlined; bound back, the term keeps every per-cell
    # choice it had and gains the fragment ones beside them.
    for position in row_candidates(loop, tile):
        bound = lift_loop_op(loop, name=loop.name, body=row_bound_body(loop, position, ROW_AXIS))
        if binds_the_row(bound, ROW_AXIS):
            tile = bound
            break
    return replace(tile, outputs={root.output.name: root.output})
