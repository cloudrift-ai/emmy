"""Fuse the Loop subgraph into its regions, one splice each.

A region is the set of ``LoopOp`` nodes one kernel computes, and the graph decides every region
before any is merged (:func:`regions`): the partition is a function of the graph alone, so the
kernel set is the same whatever order the producers are visited in. Fusion has only correctness
boundaries. Neither tile lifting, nor scheduling, nor speed narrows it, and a region the splicer
cannot build is a compiler bug it raises, never a smaller region.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.loop import LoopOp, observes_running_accumulator
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.fusion._region import build_merged_region, carries_state, live_outputs_of

PATTERN = [Pattern("producer", LoopOp)]


def _packed(graph: Graph, node: Node) -> bool:
    """Whether ``node`` writes a PACKED buffer — the storage sense: one stored element carries
    several logical values (``dtype.logical_elems > 1``, two e2m1 codes to the byte), not a
    concatenated projection.

    Why a region stops there, when fusion is otherwise maximal. A packed dtype states a relation
    between a tensor's stored extent and its logical one: the stored last axis is half the logical
    one (``dtype.py``). Only a tensor carries that relation. The splice deletes the tensor. The
    codes then survive as an ``Assign`` at the packed dtype — a value with no extent. A consumer's
    index no longer names one of them; it names half a byte. The merged body answers that by
    carrying the graph's own pack arithmetic into the consumer, deriving the whole byte at every
    logical index and reading one half of it.

    The splice also goes ONE WAY, and that is what makes this a refusal rather than a merge
    evidence could cut back: no ``030_cut`` seam offers a packed workspace, so the merged form
    would be the only one left rather than the widest of several. ``passes/ARCHITECTURE.md`` works
    that half through, beside the seam dtypes it turns on. A packed CONSTANT — every quantized
    weight — is already stored, and a region reads it as an ordinary external input.
    """
    return any((tensor := graph.buffer(buf)) is not None and tensor.dtype.logical_elems > 1 for buf in node.buffer_names())


def regions(graph: Graph) -> dict[str, frozenset[str]]:
    """Every loop's region, by member — the partition of the Loop subgraph fusion merges.

    Some nodes no region holds: a kernel that carries a state (the splice inlines a store into its
    readers, and a state is stored once per step, not once), an ordered prefix output
    (:func:`observes_running_accumulator`) and any op that is not a loop. A region also stops at a
    packed buffer it computes (:func:`_packed`): the producer stays, its readers begin another.

    Two loops share a region when the same such nodes lie upstream of both — the boundaries — and a
    chain of loop edges joins them. Equal boundaries are what keep a region convex: a path between
    two members through a node outside would put that node, or a boundary past it, upstream of one
    member and not the other, so the merged kernel never depends on a node that depends on it.

    A cut materializes every buffer crossing it, so a loop whose every reader sits in one other region
    joins it: stored where it is, its value is read once, at whatever shape it happens to have — a
    reconstructed scale broadcast across its block where the raw per-block scales are what a consumer
    can index. The packed producer never leaves; its buffer is the cut.
    """
    order = graph.topological_order()
    nodes = graph.nodes
    fusable = {
        nid
        for nid in order
        if isinstance(nodes[nid].op, LoopOp) and not (carries_state(nodes[nid].op) or observes_running_accumulator(nodes[nid].op))
    }
    stops = {nid for nid in fusable if _packed(graph, nodes[nid])}
    boundaries: dict[str, frozenset[str]] = {}
    for nid in order:
        producers = {graph.producer(name).id for name in nodes[nid].inputs}
        boundaries[nid] = frozenset().union(
            *(boundaries[p] | ({p} if p in stops or (p not in fusable and nodes[p].inputs) else frozenset()) for p in producers)
        )
    member = {nid: nid for nid in fusable}

    def find(nid: str) -> str:
        while member[nid] != nid:
            member[nid] = member[member[nid]]
            nid = member[nid]
        return nid

    for nid in fusable:
        for user in graph.users(nid):
            if user in fusable and boundaries[user] == boundaries[nid]:
                member[find(user)] = find(nid)
    region_of = {nid: find(nid) for nid in fusable}
    # Readers are downstream, so in reverse topological order every reader's region is final.
    for nid in reversed(order):
        if nid in fusable and nid not in stops:
            readers = {region_of.get(user) for user in graph.users(nid)}
            if len(readers) == 1 and (target := next(iter(readers))) not in (None, region_of[nid]):
                region_of[nid] = target
    # A member released from between two others may leave them unconnected: a region is one chain.
    member = dict(region_of)
    for nid in fusable:
        member[nid] = nid
    for nid in fusable:
        for user in graph.users(nid):
            if user in fusable and region_of[user] == region_of[nid]:
                member[find(user)] = find(nid)
    grouped: dict[str, set[str]] = {}
    for nid in fusable:
        grouped.setdefault(find(nid), set()).add(nid)
    return {nid: frozenset(members) for members in grouped.values() for nid in members}


def _wrap_multi_output_fragment(
    graph: Graph,
    merged: LoopOp,
    live_outputs: tuple[str, ...],
) -> tuple[Graph, dict[str, str]]:
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
    fragment, output_map = _wrap_multi_output_fragment(graph, merged, live_outputs)
    match.consumed = set(region)
    match.output = live_outputs[0] if len(live_outputs) == 1 else output_map
    return fragment
