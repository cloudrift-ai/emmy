"""What the fusion rules share: the regions of the Loop subgraph, a region's live buffers and its spliced body."""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expression import BinaryExpression, Literal
from emmy.compiler.ir.loop import LoopOp, observes_running_accumulator, splice_graph
from emmy.compiler.ir.stmt import Body, Load, Select


def carries_state(op: LoopOp) -> bool:
    """Whether ``op`` holds a carried state — a recurrence rolled into one kernel."""
    return bool(op.body.carries)


def joins_a_region(op: LoopOp) -> bool:
    """Whether a region may hold ``op``. A kernel that carries a state is in none: the splice inlines a store into
    its readers, and a state is stored once per step, not once. It takes the kernels around it into its step once
    the regions are merged (``015_merge_into_step``). Neither is an ordered prefix output in a region
    (:func:`observes_running_accumulator`)."""
    return not (carries_state(op) or observes_running_accumulator(op))


def plain_load(load: Load) -> bool:
    """Whether ``load`` reads one value of a buffer as it is stored: no dtype of its own, no carried state."""
    return load.dtype is None and not load.carried and len(load.names) == 1


def chosen_cell(body: Body, select: Select, shape: tuple) -> tuple | None:
    """The one cell ``select`` reads of a buffer of ``shape``: its branches are reads at cells that differ in one
    position, a literal there, one for every cell of that dimension, and each is chosen where one expression of the
    coordinates equals its literal — the read at that expression."""
    reads = {load.names[0]: load.index for load in body.loads}
    cells = [reads.get(branch.value) for branch in select.branches]
    if any(cell is None for cell in cells) or len({len(cell) for cell in cells}) != 1 or len(cells) < 2:
        return None
    differing = [at for at, entries in enumerate(zip(*cells, strict=True)) if len(set(entries)) > 1]
    if len(differing) != 1 or not all(isinstance(cell[differing[0]], Literal) for cell in cells):
        return None
    (at,) = differing
    *tests, otherwise = (branch.select for branch in select.branches)
    if otherwise != Literal(True, "bool") or not all(isinstance(test, BinaryExpression) and test.op == "==" for test in tests):
        return None
    sides = [(test.right, test.left) if isinstance(test.left, Literal) else (test.left, test.right) for test in tests]
    if len({chosen for chosen, _ in sides}) != 1 or any(position != cell[at] for (_, position), cell in zip(sides, cells, strict=False)):
        return None
    if not shape[at].is_static or sorted(cell[at].value for cell in cells) != list(range(shape[at].as_static())):
        return None
    return (*cells[0][:at], sides[0][0], *cells[0][at + 1 :])


def writes_in_place(node: Node) -> bool:
    """Whether an output of ``node`` is stored in another buffer's memory (``Tensor.alias``): a scatter,
    whose unstored cells are the destination's and so have no spelling a reader could inline."""
    return any(tensor.alias is not None for tensor in node.outputs)


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


def stops_a_region(graph: Graph, node: Node) -> bool:
    """Whether the buffers ``node`` writes have to exist as it writes them (:func:`regions`): one is packed, one is
    stored in another buffer's memory, or a kernel stores over one."""
    over = any(
        stored.alias == name for name in node.buffer_names() for user in graph.buffer_users(name) for stored in graph.nodes[user].outputs
    )
    return over or _packed(graph, node) or writes_in_place(node)


def regions(graph: Graph) -> dict[str, frozenset[str]]:
    """Every loop's region, by member — the partition of the Loop subgraph fusion merges.

    Some nodes no region holds: a kernel that carries a state, an ordered prefix output
    (:func:`joins_a_region`) and any op that is not a loop. A region also stops at a
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
    fusable = {nid for nid in order if isinstance(nodes[nid].op, LoopOp) and joins_a_region(nodes[nid].op)}
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


def build_merged_region(
    graph: Graph, region: set[str], live_outputs: tuple[str, ...], ops: dict[str, LoopOp] | None = None
) -> LoopOp | None:
    """Splice one maximal region, sharing equal upstream demands across roots. ``ops`` gives a member the loop it is
    spliced as where that is not the node's own."""
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
        sub.add_node((ops or {}).get(nid, node.op), list(node.inputs), outputs=node.outputs, node_id=nid)
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
        tensors.append(Tensor(tensor.name if i == 0 else new, tensor.shape, tensor.dtype, alias=tensor.alias))
    merged = merged.rename_buffers(rename)
    # Root insertion may reorder sibling loop nests. Kernel ABI order follows
    # graph liveness, not incidental body order.
    merged = replace(merged, outputs=dict(zip(new_buffers, tensors, strict=True)))
    # A buffer an output is stored over is an edge the body never loads; it stays an input.
    inputs = [*merged.inputs, *(t.alias for t in tensors if t.alias is not None and t.alias not in merged.inputs)]
    frag = Graph()
    for inp_id in inputs:
        ext_t = graph.buffer(inp_id)
        assert ext_t is not None
        frag.add_node(InputOp(), [], ext_t, node_id=inp_id)
    frag.add_node(merged, inputs, outputs=tensors, node_id=node_id)
    frag.outputs = list(new_buffers)
    return frag, rename
