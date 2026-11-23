"""Fuse the Loop subgraph into its regions, one splice each, and roll the recurrence a region holds.

A region is the set of ``LoopOp`` nodes one kernel computes, and the graph decides every region
before any is merged (:func:`_region.regions`): the partition is a function of the graph alone, so the
kernel set is the same whatever order the producers are visited in. Fusion has only correctness
boundaries. Neither tile lifting, nor scheduling, nor speed narrows it, and a region the splicer
cannot build is a compiler bug it raises, never a smaller region.

An unrolled recurrence is not spliced as it stands: the copies of the step would be inlined into one another, every
state re-derived wherever the next step reads it. Before the rule splices a region it rolls every recurrence the
graph holds (``_recurrence.roll``), each into one kernel that carries its state, and the regions around those
kernels are what a later match of this rule splices.
"""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.fusion._recurrence import roll
from emmy.compiler.pipeline.passes.loop.fusion._region import (
    build_merged_region,
    joins_a_region,
    live_outputs_of,
    read_in_place,
    regions,
    wrap_multi_output_fragment,
)

PATTERN = [Pattern("producer", LoopOp)]


def _reads_settled(graph: Graph, partition: dict[str, frozenset[str]]) -> bool:
    """Whether no loop reads a copy that ``008_read_through_copies`` would take the read through: one held outside
    the loop's own region, or any at all for a loop no region holds."""
    for node in graph.nodes.values():
        if isinstance(node.op, LoopOp):
            own = partition.get(node.id, frozenset()) if joins_a_region(node.op) else frozenset()
            if read_in_place(graph, node.op.body, own) != node.op.body:
                return False
    return True


def rewrite(match: Match, producer: Node) -> Graph:
    graph = match.graph
    partition = regions(graph)
    region = partition.get(producer.id)
    if region is None or len(region) < 2:
        raise RuleSkipped("the producer is a region of its own")
    # A kernel that carries a state is in no region, so the regions to splice are those of the graph the rolls leave.
    if (rolled := roll(match)) is not None:
        return rolled
    # The rolls of this batch changed the regions after the rules before this one had read them: those rules read
    # the graph again before a region is spliced with a copy in it that a loop outside could read through.
    if not _reads_settled(graph, partition):
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
