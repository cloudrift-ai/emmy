"""What the fusion rules share: a region's live buffers and its spliced body."""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Graph, Tensor
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

    result = splice_graph(sub)
    return result[0] if result is not None else None


def wrap_multi_output_fragment(graph: Graph, merged: LoopOp, live_outputs: tuple[str, ...]) -> tuple[Graph, dict[str, str]]:
    """Wrap one merged LoopOp and map every old live buffer to its new port."""
    owner = graph.producer(live_outputs[0])
    assert owner is not None
    node_id = f"merged_{owner.id}"
    new_buffers = (node_id, *(f"{node_id}__out{i}" for i in range(1, len(live_outputs))))
    rename = dict(zip(live_outputs, new_buffers, strict=True))

    tensors: list[Tensor] = []
    for i, (old, new) in enumerate(zip(live_outputs, new_buffers, strict=True)):
        tensor = graph.buffer(old)
        assert tensor is not None
        tensors.append(Tensor(tensor.name if i == 0 else new, tensor.shape, tensor.dtype))
    merged = merged.rename_buffers(rename)
    # Root insertion may reorder sibling loop nests. Kernel ABI order follows
    # graph liveness, not incidental body order.
    merged = replace(merged, outputs=dict(zip(new_buffers, tensors, strict=True)))
    frag = Graph()
    for inp_id in merged.inputs:
        ext_t = graph.buffer(inp_id)
        assert ext_t is not None
        frag.add_node(InputOp(), [], ext_t, node_id=inp_id)
    frag.add_node(merged, list(merged.inputs), outputs=tensors, node_id=node_id)
    frag.outputs = list(new_buffers)
    return frag, rename
