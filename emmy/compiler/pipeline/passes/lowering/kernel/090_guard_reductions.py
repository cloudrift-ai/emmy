"""Skip pure scalar reductions outside the coordinates that consume their results."""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt.normalize import guard_reductions
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.search.space import GUARD_REDUCTIONS

PATTERN = [Pattern("root", KernelOp)]


def rewrite(root: Node) -> KernelOp:
    op: KernelOp = root.op
    if GUARD_REDUCTIONS.name in op.knobs:
        raise RuleSkipped("GUARD_REDUCTIONS already decided")
    enabled = GUARD_REDUCTIONS.narrow((True,))[0]
    return replace(op, body=guard_reductions(op.body) if enabled else op.body, knobs={**op.knobs, GUARD_REDUCTIONS.name: enabled})
