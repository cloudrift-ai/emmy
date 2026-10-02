"""Skip pure scalar reductions outside the coordinates that consume their results."""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.kernel._guard import guard_reductions as _guard
from emmy.compiler.pipeline.search.space import GUARD_REDUCTIONS

PATTERN = [Pattern("root", KernelOp)]


def rewrite(root: Node) -> KernelOp:
    op: KernelOp = root.op
    if GUARD_REDUCTIONS.name in op.knobs:
        raise RuleSkipped("GUARD_REDUCTIONS already decided")
    enabled = GUARD_REDUCTIONS.narrow((True,))[0]
    return replace(op, body=_guard(op.body) if enabled else op.body, knobs={**op.knobs, GUARD_REDUCTIONS.name: enabled})
