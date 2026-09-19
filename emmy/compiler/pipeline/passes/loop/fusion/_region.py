"""What the fusion rules share: a region's live buffers and its spliced body."""

from __future__ import annotations

from emmy.compiler.graph import Graph
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.loop import LoopOp, splice_graph


def carries_state(op: LoopOp) -> bool:
    """Whether ``op`` holds a carried state — a recurrence rolled into one kernel."""
    return bool(op.body.carries)


def live_outputs_of(graph: Graph, region: set[str]) -> tuple[str, ...]:
    """The region's buffers read outside it or exported by the graph, in topological order."""
    graph_outputs = set(graph.outputs)
    live: list[str] = []
    for nid in graph.topological_order():
        if nid not in region:
            continue
        for buf in graph.nodes[nid].buffer_names():
            if buf in graph_outputs or graph.buffer_users(buf) - region:
                live.append(buf)
    return tuple(live)


def build_merged_region(graph: Graph, region: set[str], live_outputs: tuple[str, ...]) -> LoopOp | None:
    """Splice one maximal region, sharing equal upstream demands across roots."""
    order = [nid for nid in graph.topological_order() if nid in region]
    sub = Graph()
    external: list[str] = []
    for nid in order:
        for inp in graph.nodes[nid].inputs:
            producer = graph.producer(inp)
            assert producer is not None
            if producer.id not in region and inp not in external:
                external.append(inp)
    for ext_id in external:
        ext_t = graph.buffer(ext_id)
        assert ext_t is not None
        sub.add_node(InputOp(), [], ext_t, node_id=ext_id)
    for nid in order:
        node = graph.nodes[nid]
        sub.add_node(node.op, list(node.inputs), outputs=node.outputs, node_id=nid)
    sub.outputs = list(live_outputs)

    result = splice_graph(sub, surface_unfusable=True)
    return result[0] if result is not None else None
