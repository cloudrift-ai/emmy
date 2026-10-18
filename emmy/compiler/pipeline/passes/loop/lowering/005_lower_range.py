"""Lower a symbolic ``RangeOp`` to a ``LoopOp`` that writes each element's index.

A static range folds to a constant before this stage; a symbolic one has no value until the launch binds its
length, so it is a loop over that length whose cell stores its own coordinate.
"""

from __future__ import annotations

from emmy.compiler.dtype import get as get_dtype
from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.loop import Axis, Loop, LoopOp, Write
from emmy.compiler.ir.stmt import Body, Let
from emmy.compiler.ir.tensor.ir import RangeOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [Pattern("root", RangeOp)]


def rewrite(match: Match, root: Node) -> Graph:
    if root.op.length is None:
        raise RuleSkipped("a static range folds to a constant")
    axis = Axis(name="a0", extent=root.op.length)
    body = (Let("v", Var(axis.name), dtype=get_dtype(root.op.dtype)), Write(output=f"loop_{root.id}", index=(Var(axis.name),), value="v"))
    kernel = LoopOp(body=Body((Loop(axis=axis, body=Body(body)),)))
    frag = Graph()
    out_id = frag.add_node(kernel, [], Tensor(root.output.name, root.output.shape, root.output.dtype), node_id=f"loop_{root.id}")
    frag.outputs = [out_id]
    return frag
