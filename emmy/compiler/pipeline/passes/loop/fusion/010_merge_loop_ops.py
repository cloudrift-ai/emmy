"""Fuse the Loop subgraph into its regions, one splice each.

A region is the set of ``LoopOp`` nodes one kernel computes, and the graph decides every region
before any is merged (:func:`_region.regions`): the partition is a function of the graph alone, so the
kernel set is the same whatever order the producers are visited in. Fusion has only correctness
boundaries. Neither tile lifting, nor scheduling, nor speed narrows it, and a region the splicer
cannot build is a compiler bug it raises, never a smaller region.
"""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.fusion._region import build_merged_region, live_outputs_of, regions, wrap_multi_output_fragment

PATTERN = [Pattern("producer", LoopOp)]


def rewrite(match: Match, producer: Node) -> Graph:
    graph = match.graph
    region = regions(graph).get(producer.id)
    if region is None or len(region) < 2:
        raise RuleSkipped("the producer is a region of its own")
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
