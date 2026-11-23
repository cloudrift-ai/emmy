"""Read through a copy that another kernel would have to store for this one.

A loop that computes nothing — one read and one store of it, in another layout or at another dtype: the chunks of a
sequence, a transpose, a broadcast, a cast to a wider float — is a buffer only where something has to read it as one.
Inside its own region it never is: the splice inlines the read (``_region.regions``). Across the region's boundary it
was. A kernel that carries a state is in no region, so every operand it reads through a copy was a buffer some kernel
filled for it alone; and a copy two regions read was filled by one of them for the other. Such a reader now takes the
read itself: the buffer behind the copy, at the cell the copy reads there.

The regions stay what they were. A reader in the copy's own region is left to the splice, so the copy still joins
the loops it joined into one kernel; what goes is the store of the copy, and the kernel whose only work it was.

The copies of a step that a region still holds are left as the tracer wrote them, which is how the roll compares
them (``_recurrence``): a step's slice of a copy is a literal coordinate that advances with the step, where the read
behind it is an offset that the first step does not have. A roll changes the regions, so ``010_merge_loop_ops`` has
the rules scanned again once it has rolled every recurrence (``Match.rescan``), and this rule sees the kernels that
carry a state and the regions around them before any of those is spliced. The copy stays for whoever still reads it as a buffer: a
graph output, a packed buffer's reader, a reader of a buffer another kernel stores over.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.fusion._region import joins_a_region, read_in_place, regions, wrap_multi_output_fragment

PATTERN = [Pattern("reader", LoopOp)]


def rewrite(match: Match, reader: Node) -> Graph:
    graph = match.graph
    region: frozenset[str] = frozenset()
    if joins_a_region(reader.op):
        # A copy this loop alone reads is its own region's: a loop whose every reader sits in one region joins it.
        # The regions are an answer about the whole graph, asked only for a loop that reads a copy with another reader.
        region = frozenset(graph.producer(name).id for name in reader.inputs if graph.buffer_users(name) == {reader.id})
        if read_in_place(graph, reader.op.body, region) != reader.op.body:
            region = regions(graph).get(reader.id, frozenset())
    body = read_in_place(graph, reader.op.body, region)
    if body == reader.op.body:
        raise RuleSkipped("the loop reads no copy of another region")
    fragment, match.output = wrap_multi_output_fragment(graph, replace(reader.op, body=body, inputs={}), tuple(reader.buffer_names()))
    return fragment
