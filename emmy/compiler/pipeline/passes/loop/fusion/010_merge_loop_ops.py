"""Fuse the Loop subgraph into its regions, one splice each.

A region is the set of ``LoopOp`` nodes one kernel computes, and the graph decides every region
before any is merged (:func:`regions`): the partition is a function of the graph alone, so the
kernel set is the same whatever order the producers are visited in. Fusion has only correctness
boundaries. Neither tile lifting, nor scheduling, nor speed narrows it, and a region the splicer
cannot build is a compiler bug it raises, never a smaller region.
"""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.loop import LoopOp, observes_running_accumulator
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.fusion._region import (
    build_merged_region,
    carries_state,
    live_outputs_of,
    wrap_multi_output_fragment,
    writes_in_place,
)

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
    packed buffer it computes (:func:`_packed`): the producer stays, its readers begin another. A
    kernel that writes in place (:func:`writes_in_place`) is such a stop too, and so is the producer of
    the buffer it writes over: the output is that buffer's memory with some cells restored, a value no
    reader can inline, and the buffer has to exist for the store to land in — while the producers of
    what the kernel stores fuse into it like any other.

    Two loops share a region when the same such nodes lie upstream of both — the boundaries — and a
    chain of loop edges joins them. Equal boundaries are what keep a region convex: a path between
    two members through a node outside would put that node, or a boundary past it, upstream of one
    member and not the other, so the merged kernel never depends on a node that depends on it. Two regions a loop
    edge joins then merge while their union stays convex (:func:`_joined`).

    A cut materializes every buffer crossing it, so a loop whose every reader sits in one other region
    joins it: stored where it is, its value is read once, at whatever shape it happens to have — a
    reconstructed scale broadcast across its block where the raw per-block scales are what a consumer
    can index. The packed producer never leaves; its buffer is the cut.

    A kernel stored over a buffer never shares a region with a loop that computes from that buffer's old
    cells (:func:`_clashes`): one thread of the merged kernel could store a cell before another reads it.
    What it stores from the old cells — a recurrent state's next value — is computed by a kernel before it.
    """
    order = graph.topological_order()
    nodes = graph.nodes
    fusable = {
        nid
        for nid in order
        if isinstance(nodes[nid].op, LoopOp) and not (carries_state(nodes[nid].op) or observes_running_accumulator(nodes[nid].op))
    }
    stops = {nid for nid in fusable if _packed(graph, nodes[nid]) or writes_in_place(nodes[nid])}
    stops |= {
        producer.id
        for nid in fusable
        for tensor in nodes[nid].outputs
        if tensor.alias is not None and (producer := graph.producer(tensor.alias)) is not None and producer.id in fusable
    }
    boundaries: dict[str, frozenset[str]] = {}
    for nid in order:
        producers = {graph.producer(name).id for name in nodes[nid].inputs}
        boundaries[nid] = frozenset().union(
            *(boundaries[p] | ({p} if p in stops or (p not in fusable and nodes[p].inputs) else frozenset()) for p in producers)
        )
    downstream: dict[str, set[str]] = {}
    for nid in reversed(order):
        downstream[nid] = set().union(*({user} | downstream[user] for user in graph.users(nid)))
    clash = _clashes(graph, fusable, downstream)
    position = {nid: index for index, nid in enumerate(order)}
    member = {nid: nid for nid in fusable}
    held = {nid: {nid} for nid in fusable}

    def find(nid: str) -> str:
        while member[nid] != nid:
            member[nid] = member[member[nid]]
            nid = member[nid]
        return nid

    for nid in order:
        for user in sorted(graph.users(nid), key=position.get) if nid in fusable else ():
            first, second = find(nid), find(user) if user in fusable else None
            if second not in (None, first) and boundaries[user] == boundaries[nid] and not _apart(clash, held[first], held[second]):
                member[second] = first
                held[first] |= held.pop(second)
    region_of = {nid: find(nid) for nid in fusable}
    # Readers are downstream, so in reverse topological order every reader's region is final.
    for nid in reversed(order):
        if nid in fusable and nid not in stops:
            readers = {region_of.get(user) for user in graph.users(nid)}
            target = next(iter(readers)) if len(readers) == 1 else None
            if target not in (None, region_of[nid]) and not _apart(clash, {nid}, held[target]):
                held[region_of[nid]].discard(nid)
                held[target].add(nid)
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
    groups = list(grouped.values())
    return {nid: frozenset(members) for members in _joined(graph, order, fusable, stops, clash, downstream, groups) for nid in members}


def _clashes(graph: Graph, fusable: set[str], downstream: dict[str, set[str]]) -> dict[str, set[str]]:
    """Each loop's loops it may not share a kernel with: a loop stored over a buffer (``Tensor.alias``) and every
    loop computing from that buffer's old cells — a reader of the buffer or of a value computed from it, short of
    what reads the store's own result."""
    clash: dict[str, set[str]] = {nid: set() for nid in fusable}
    for nid in fusable:
        for tensor in graph.nodes[nid].outputs:
            if tensor.alias is None:
                continue
            readers = graph.buffer_users(tensor.alias)
            for other in (readers.union(*(downstream[reader] for reader in readers)) & fusable) - downstream[nid] - {nid}:
                clash[nid].add(other)
                clash[other].add(nid)
    return clash


def _apart(clash: dict[str, set[str]], first: set[str], second: set[str]) -> bool:
    """Whether one kernel may not hold both ``first`` and ``second`` (:func:`_clashes`)."""
    return any(clash[nid] & second for nid in first)


def _joined(
    graph: Graph,
    order: list[str],
    fusable: set[str],
    stops: set[str],
    clash: dict[str, set[str]],
    downstream: dict[str, set[str]],
    groups: list[set[str]],
) -> list[set[str]]:
    """``groups`` with every two joined by a loop edge merged while the union stays convex. Equal boundaries part two
    loops one of which reads past a boundary the other does not (a scan feeding only one), though nothing lies between
    them: the union is still one kernel when every node on a path between its members can join it, no member
    reads past a stop of its own, and no two members clash (:func:`_clashes`)."""
    upstream: dict[str, set[str]] = {nid: set() for nid in order}
    for nid, below in downstream.items():
        for other in below:
            upstream[other].add(nid)
    group_of = {nid: index for index, members in enumerate(groups) for nid in members}

    def union(first: int, second: int) -> set[int] | None:
        joined = {first, second}
        while True:
            members = set().union(*(groups[index] for index in joined))
            between = set().union(*(downstream[nid] for nid in members)) & set().union(*(upstream[nid] for nid in members))
            outside = between - members
            if not outside:
                break
            if not outside <= fusable:
                return None
            joined |= {group_of[nid] for nid in outside}
        if any(downstream[stop] & members for stop in stops & members) or _apart(clash, members, members):
            return None
        return joined

    merged = True
    while merged:
        merged = False
        for nid in order:
            for user in graph.users(nid) if nid in fusable else ():
                if user not in fusable or group_of[user] == group_of[nid]:
                    continue
                if (joined := union(group_of[nid], group_of[user])) is None:
                    continue
                keep = min(joined)
                for index in joined - {keep}:
                    groups[keep] |= groups[index]
                    groups[index] = set()
                group_of.update({member: keep for member in groups[keep]})
                merged = True
    return [members for members in groups if members]


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
