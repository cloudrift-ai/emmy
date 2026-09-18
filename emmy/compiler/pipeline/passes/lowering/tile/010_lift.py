"""Lift a ``LoopOp`` completely into one unmapped Fold-tree ``TileOp``."""

from __future__ import annotations

from emmy.compiler.dtype import F32
from emmy.compiler.graph import Node, Tensor
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.pipeline import Match, Pattern
from emmy.compiler.pipeline.passes.lowering.tile._cut import _input_fragment
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import lift_loop_op, states_as_buffers
from emmy.compiler.pipeline.passes.lowering.tile._row import lift_kernel
from emmy.compiler.pipeline.passes.lowering.tile._split import add_output_piece

PATTERN = [Pattern("root", LoopOp)]


def rewrite(match: Match, root: Node, ctx=None):
    del ctx
    from dataclasses import replace  # noqa: PLC0415

    loop: LoopOp = root.op
    if loop.body.carries:
        # A carried state: it is a buffer the kernel owns and reads one step back, so
        # the node gains that port — a splice, where every other lift is a rebind.
        body, serial, shapes = states_as_buffers(loop.body, root.id)
        tile = lift_loop_op(loop, name=loop.name, body=body, serial=serial)
        states = tuple(
            Tensor(name=name, shape=tuple(1 if isinstance(d, int) else d.extent for d in shape), dtype=F32)
            for name, shape in shapes.items()
        )
        return add_output_piece(match, _input_fragment(match, root), root, tile, list(root.inputs), suffix="__lifted", states=states)
    return replace(lift_kernel(loop, name=loop.name), outputs={root.output.name: root.output})
