"""Roll every recurrence a region holds, before a loop reads through a copy and before a region is spliced.

An unrolled recurrence is not spliced as it stands: the copies of the step would be inlined into one another, every
state re-derived wherever the next step reads it. The copies are nodes of one region, what ``010_merge_loop_ops``
would splice into one kernel, so the regions are where they are looked for, and each recurrence found is rolled into
one kernel that carries its state (``_recurrence.roll``). A kernel that carries a state is in no region: the rules
after this one read through the copies and splice the regions of the graph the rolls leave.

The roll is a question about the graph, not about one loop, so it is asked once a scan, at the last of the rule's
matches, and again after every roll until no region holds a recurrence.
"""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.fusion._recurrence import roll
from emmy.compiler.pipeline.passes.loop.fusion._region import regions

PATTERN = [Pattern("loop", LoopOp)]
FIXPOINT = True


def rewrite(match: Match, loop: Node) -> Graph:
    if not match.is_last:
        raise RuleSkipped("the recurrences are looked for once a scan")
    rolled = roll(match, regions(match.graph))
    if rolled is None:
        raise RuleSkipped("no region holds a recurrence")
    return rolled
