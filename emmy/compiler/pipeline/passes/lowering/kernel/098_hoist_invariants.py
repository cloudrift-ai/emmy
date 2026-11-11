"""Move each value out of every loop whose trips compute it alike (``HOIST``).

A loop of a kernel body can compute a value every trip computes alike: a load or a product over the coordinates
around the loop, a reciprocal of a divisor the loop does not vary. Computed ahead of the loop it is computed once and
held in a register across every trip: fewer instructions against a longer live range, which can cost occupancy or a
spill. That is a trade, so ``HOIST`` decides it, never a rule (passes ``ARCHITECTURE.md``, "Performance is never a
pass decision"). The motion is the crate's placement (``hoist_invariants``), the one normalization places a body
with: a value leaves loops only, never a branch, a tile or a barrier's order, so nothing runs that the body did not.
It runs after the memory rules, which read the body as the lowering spelled it.

``HOIST`` offers only today's spelling (``0``) until a tune measures the motion on each card and a refit scores it;
``EMMY_HOIST=1`` pins it.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.kernel import KernelOp, hoist_invariants
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.search.space import HOIST

PATTERN = [Pattern("root", KernelOp)]


def rewrite(root: Node) -> KernelOp | list[KernelOp]:
    op: KernelOp = root.op
    if HOIST.name in op.knobs:
        raise RuleSkipped("HOIST already decided")
    values = HOIST.narrow(HOIST.hints)
    if values == (0,):
        raise RuleSkipped("every value stays where the lowering placed it")
    variants = [
        replace(op, body=hoist_invariants(op.body) if value else op.body, knobs={**op.knobs, HOIST.name: value}) for value in values
    ]
    return variants if len(variants) > 1 else variants[0]
