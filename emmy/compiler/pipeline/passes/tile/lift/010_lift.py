"""Lift a ``LoopOp`` completely into one unmapped Fold-tree ``TileOp``."""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.pipeline import Match, Pattern
from emmy.compiler.pipeline.passes.tile._row import lift_kernel

PATTERN = [Pattern("root", LoopOp)]


def rewrite(match: Match, root: Node, ctx=None):
    del match, ctx
    return replace(lift_kernel(root.op), outputs={root.output.name: root.output})
