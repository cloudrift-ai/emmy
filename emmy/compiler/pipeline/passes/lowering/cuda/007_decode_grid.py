"""Decode each kernel's thread grid into its body: the ``Tile`` becomes a ``Grid``.

A ``Tile`` binds its axes for its body, and its launch is implied: ``ceil(N / cells)`` CTAs over a flat thread id.
Here the decode is spelled as Kernel IR (``decode_tile``): each axis a ``Let`` over ``thread_idx.x`` and
``block_idx.x``, simplified under the launch's ranges, and the tail guard a ``Cond`` where the launch does not cover
the cells exactly. A thread's lane and warp, where the body reads them, are bound the same way. What follows sees
every value the kernel computes, the coordinates included, and an emitter has no decode of its own to spell and no
name of its own to declare. It runs after the rules that still read the ``Tile`` (the zero-init delegation, the
kernel cache), and it decides nothing: the kernel computes the same values.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.kernel import KernelOp, Tile
from emmy.compiler.ir.kernel.ir import decode_tile
from emmy.compiler.ir.stmt import Body
from emmy.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", KernelOp)]


def rewrite(root: Node) -> KernelOp:
    op: KernelOp = root.op
    if not any(isinstance(stmt, Tile) for stmt in op.body):
        raise RuleSkipped("no tile to decode")
    return replace(op, body=Body(tuple(decode_tile(stmt) if isinstance(stmt, Tile) else stmt for stmt in op.body)))
