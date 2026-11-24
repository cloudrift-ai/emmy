"""Fuse the Loop subgraph into its regions: roll the recurrence a region holds, and splice the region into one kernel.

A region is the set of ``LoopOp`` nodes one kernel computes, and the graph decides every region
before any is merged (:func:`_region.regions`): the partition is a function of the graph alone, so the
kernel set is the same whatever order the producers are visited in. Fusion has only correctness
boundaries. Neither tile lifting, nor scheduling, nor speed narrows it, and a region the splicer
cannot build is a compiler bug it raises, never a smaller region.

An unrolled recurrence is not spliced as it stands: the copies of its step would be inlined into one another, every
state re-derived wherever the next step reads it. A roll is a splice with more asked of its nodes
(``_recurrence.roll``), so the rule asks for it first: before it splices any region it rolls the recurrences the
regions hold, each into one kernel that carries its state. Such a kernel is in no region, and the loops around it
read what it stores through the slices the roll leaves. The rule then asks for the pass's rules to be scanned again
(``Match.rescan``): those loops read through their copies (``008_read_through_copies``) before a region is spliced.
"""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.fusion._recurrence import roll
from emmy.compiler.pipeline.passes.loop.fusion._region import (
    build_merged_region,
    live_outputs_of,
    read_in_place,
    regions,
    wrap_multi_output_fragment,
)

PATTERN = [Pattern("producer", LoopOp)]


def rewrite(match: Match, producer: Node) -> Graph:
    graph = match.graph
    partition = regions(graph)
    region = partition.get(producer.id)
    if region is None or len(region) < 2:
        raise RuleSkipped("the producer is a region of its own")
    if (rolled := roll(match, partition)) is not None:
        return rolled
    if any(isinstance(node.op, LoopOp) and read_in_place(graph, node.op.body) != node.op.body for node in graph.nodes.values()):
        # A roll of this scan left slices of its stores, and the regions around it are new: the loops read through
        # them before a region is spliced with a copy in it that a loop outside would otherwise have stored for it.
        match.rescan = True
        raise RuleSkipped("the loops around a rolled recurrence have not read through their copies yet")
    live_outputs = live_outputs_of(graph, region)
    if not live_outputs:
        raise RuleSkipped("nothing reads the region")
    merged = build_merged_region(graph, region, live_outputs)
    if merged is None:
        raise ValueError(f"fusion cannot splice the region of {producer.id!r}: {sorted(region)}")
    fragment, output_map = wrap_multi_output_fragment(graph, merged, live_outputs)
    match.consumed = set(region)
    match.output = live_outputs[0] if len(live_outputs) == 1 else output_map
    return fragment
