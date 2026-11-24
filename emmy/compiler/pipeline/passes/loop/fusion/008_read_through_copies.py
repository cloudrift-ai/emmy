"""Read through every copy: what its readers share is the buffer behind it.

A loop that computes nothing — one read and one store of it, in another layout or at another dtype: the chunks of a
sequence, a transpose, a broadcast, a cast to a wider float — holds no work a kernel could share. Every loop that
reads one takes the read itself: the buffer behind the copy, at the cell the copy reads there. The copy is then read
by no loop, and the kernel whose only work it was is gone.

What two readers of a copy share is then the buffer they both read, and loops that read one buffer share a region
(``_region.regions``): one kernel reads it once for all of them. Whether the program wrote a transpose between a
buffer and its readers decides no kernel boundary: they are one kernel either way. A kernel that carries a state,
which no region holds, reads its operands where they are stored by the same rule.

A copy the program itself holds, a graph output, is a buffer whoever reads it: it is stored once and its readers
read it there, like any stored value. A copy that rounds is arithmetic and is not read through, nor is a copy of a
buffer another kernel stores over, nor one of a packed buffer.

It runs before the merge rule rolls a recurrence and splices a region (``010_merge_loop_ops``), so the steps of a
recurrence read their buffers themselves, at offsets that advance with the step, and a first step that read through
a broadcast reads as the others do. A roll leaves slices of what its kernel stores; the merge rule then has the
pass's rules scanned again (``Match.rescan``), and this rule reads through those before a region is spliced.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.fusion._region import read_in_place, wrap_multi_output_fragment

PATTERN = [Pattern("reader", LoopOp)]


def rewrite(match: Match, reader: Node) -> Graph:
    graph = match.graph
    body = read_in_place(graph, reader.op.body)
    if body == reader.op.body:
        raise RuleSkipped("the loop reads no copy")
    fragment, match.output = wrap_multi_output_fragment(graph, replace(reader.op, body=body, inputs={}), tuple(reader.buffer_names()))
    return fragment
