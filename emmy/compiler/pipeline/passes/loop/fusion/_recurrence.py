"""Roll an unrolled recurrence into ONE kernel that carries its state, where fusion would have inlined it.

``010_merge_loop_ops`` asks this before it splices a region (:func:`roll`): the copies of one step, each reading
the state the copy before it stored, are rolled first, and the regions of the graph that leaves are what it splices.

A roll is a splice with more asked of its nodes, in three parts. :func:`found` lists the ways to read part of a
region as a recurrence: which nodes are its states and which are each step's. :func:`_proved` holds one of them to
what a recurrence is: every step splices into one body, the bodies are one step once the integers of their
coordinates are left out and each integer moves by one amount a step (:func:`prove`), and each step reads the
state the step before it stored. :func:`_fragment` writes the kernel. The first reading that is proved is taken.

A Python loop over chunks leaves the tracer unrolled: step ``j``'s kernels read the state step
``j − 1`` stored, so fusing them whole makes every later state re-derive every earlier one, and a
chain of twenty reductions demands two to the twentieth bindings of its first step. Here the
structure is still plain. The copies are nodes of one region, the set fusion would splice into one kernel, so
the region is where they are looked for. The states are a chain of its nodes of one kind (:func:`kinds`: one
tensor type and one body once the integers of its coordinates are left out), each depending
on the last; what lies between two of them is one step; and the steps differ only in the integer
literals they read and mask at — an offset, a slice bound — each advancing affinely with the step.

Each step is spliced into one body by the fusion rule's own splicer, the steps must be the first step's body with
its integers moved ``j`` steps on (:func:`prove`), and only then is the chain replaced — by
one kernel that carries the state (``Carry``) from the tensor the loop started with, stores what
each step defined (the last state alone, where nothing but the next step reads a state), and by one
slice of those stores per replaced buffer. A chain whose states alternate two bodies (a Sinkhorn row
step then a column step) rolls as one step of two; a chain that is not one step advanced is left
alone: a recurrence rolled wrongly is a wrong answer. A chain whose steps read their state only at
the cell they write, and whose states nothing but the next step reads, is left alone too: fused
whole it re-derives no state, so it is straight-line code, and rolling it would only store every
state. A step at the edge of its matrix, whose mask the normal form shortened by the bound the extent
gives, is taken with the chain where it is the first step advanced (:func:`_chain`).
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, replace

from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.expression import Literal, Var
from emmy.compiler.ir.loop import LoopOp, UnfusableStmt
from emmy.compiler.ir.stmt import Accum, Assign, Body, Carry, Load, Loop, Statement, Write, prove_steps
from emmy.compiler.pipeline import Match, RuleSkipped
from emmy.compiler.pipeline.passes.loop.fusion._region import build_merged_region, live_outputs_of

logger = logging.getLogger(__name__)

_STATE, _STEP, _CONST = "carried_state", "carried_step", "carried_const"
#: The longest step tried, in states: one, or two for a chain whose states alternate two bodies.
_PERIODS = (1, 2)


def _ancestors(graph: Graph, node_id: str) -> set[str]:
    seen: set[str] = set()
    pending = [node_id]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        pending.extend(producer.id for name in graph.nodes[current].inputs if (producer := graph.producer(name)) is not None)
    return seen


def kinds(graph: Graph, partition: dict[str, frozenset[str]]) -> dict[str, tuple]:
    """Per loop of a region, what a state of a recurrence shares with the states after it: the region, the shape and
    the type of what it stores, and its body once the integers of its coordinates and static extents are left out
    (``Body.identity_apart``). The copies of a step are nodes of one region: what fusion would splice into one kernel."""
    found = {}
    for node_id, region in partition.items():
        node = graph.nodes[node_id]
        found[node_id] = (region, node.outputs[0].shape, node.outputs[0].dtype, node.op.body.identity_apart())
    return found


def _chain(graph: Graph, first: Node, kind: dict[str, tuple]) -> tuple[list[Node], Node | None]:
    """``first`` and every later state: what descends from it and is of its kind (:func:`kinds`). And the state
    after the last of them, if there is one of its region and tensor with another body: a step at the edge of its
    matrix. A step that writes row ``i`` masks its store by ``i <= row < i + 1``, and at the last row the normal form
    drops the bound the extent already gives, so the body is the first step's advanced and no longer spells like it.
    Whether it is that step is the roll's own test of every step. A first step at the matrix's other edge is not
    offered."""
    chain, reached, edge = [first], {first.id}, None
    for node_id in graph.topological_order():
        node = graph.nodes[node_id]
        if not any(producer.id in reached for name in node.inputs if (producer := graph.producer(name)) is not None):
            continue
        reached.add(node_id)
        if kind.get(node_id) == kind[first.id]:
            chain.append(node)
            edge = None
        elif edge is None and kind.get(node_id, ())[:3] == kind[first.id][:3]:
            edge = node
    return chain, edge


def _is_zero(graph: Graph, name: str) -> bool:
    producer = graph.producer(name)
    if producer is None:
        return False
    if isinstance(producer.op, ConstantOp):
        return producer.op.value == 0
    layout = isinstance(producer.op, LoopOp) and not Body(producer.op.body).iter_of_type(Assign, Accum)
    return layout and all(_is_zero(graph, source) for source in producer.inputs)


def _external(graph: Graph, region: set[str]) -> set[str]:
    return {name for node_id in region for name in graph.nodes[node_id].inputs if graph.producer(name).id not in region}


def _steps(graph: Graph, chain: list[Node]) -> tuple[list[set[str]], list[str], int]:
    """``(one region per step, the state each step reads, how many states the recurrence spans)``.

    The first step has no state node before it: its state is the seed, the one buffer of the state's
    shape it reads beyond what every step reads — the tensor the loop started from, or the zeros it
    was seeded with. It is the buffer whose ancestry, taken out, leaves a first step shaped like
    every later one: as many nodes, reading the same things beside its state."""
    regions = [_ancestors(graph, after.id) - _ancestors(graph, before.id) for before, after in zip(chain, chain[1:], strict=False)]
    states = [node.id for node in chain[:-1]]
    shared = _external(graph, regions[0]) - {states[0]}
    alike = (len(regions[0]), frozenset(shared))
    # The chain is every later node of the first state's body shape; a later division of the same
    # shape that closes the block is one too, with a region of everything before it. The recurrence
    # is the prefix whose steps are alike, and it ends where they stop being.
    steps = 0
    for region, state in zip(regions, states, strict=True):
        if (len(region), frozenset(_external(graph, region) - {state})) != alike:
            break
        steps += 1
    regions, states = regions[:steps], states[:steps]
    beyond: set[str] = set()
    for name in shared:
        beyond |= _ancestors(graph, graph.producer(name).id)
    first = _ancestors(graph, chain[0].id) - beyond
    tensor = chain[0].outputs[0]
    seeds = []
    for node_id in first:
        if node_id == chain[0].id or graph.nodes[node_id].outputs[0].shape != tensor.shape:
            continue
        seed = node_id
        region = first - _ancestors(graph, node_id)
        if (len(region), frozenset(_external(graph, region) - {seed})) == alike:
            seeds.append((seed, region))
    if len(seeds) != 1:
        raise RuleSkipped(f"{len(seeds)} buffers of the state's shape leave a first step like the others")
    seed, region = seeds[0]
    return [region, *regions], [seed, *states], steps + 1


def _stateful(graph: Graph, region: set[str], state: str) -> set[str]:
    """The nodes of ``region`` that depend on ``state``. The rest — a slice of an input the step
    reads — is no part of the recurrence: a reader outside keeps reading the node itself."""
    found: set[str] = set()
    pending = [user for user in graph.buffer_users(state) if user in region]
    while pending:
        node_id = pending.pop()
        if node_id not in found:
            found.add(node_id)
            pending.extend(user for user in graph.users(node_id) if user in region)
    return found


def _readers(graph: Graph, regions: list[set[str]], states: list[str], chain: list[Node]) -> list[set[str]]:
    """What each step's own readers add to its region: every node outside the steps that reads the state the step
    starts from, or a value the step computes from it, and nothing of another step, with what such a node reads
    that no other step's readers read.

    The rolled kernel computes such a reader in the step, from the state it holds. Left outside, the reader takes
    its state from a store of every step, which the kernel then has to write. A node that reads two steps (the
    stack of every step's output) reads those stores, and so does a reader of the state the chain ends with, which
    no step starts from."""
    step: dict[str, int | None] = {chain[-1].id: None}
    for index, (region, state) in enumerate(zip(regions, states, strict=True)):
        step.setdefault(state, index)
        step.update((node_id, index) for node_id in _stateful(graph, region, state) if node_id != chain[index].id)
    taken = set().union(*regions)
    readers: list[set[str]] = [set() for _ in regions]
    for node_id in graph.topological_order():
        if node_id in step or node_id in taken:
            continue
        producers = (graph.producer(name) for name in graph.nodes[node_id].inputs)
        read = {step[producer.id] for producer in producers if producer is not None and producer.id in step}
        if read:
            step[node_id] = next(iter(read)) if len(read) == 1 else None
            if step[node_id] is not None:
                readers[step[node_id]].add(node_id)
    # What the recurrence itself descends from is an input of the body or a step's own, and so is what the readers
    # of two steps both read: it is read at every step.
    recurrence = _ancestors(graph, chain[-1].id)
    reads = [set().union(*(_ancestors(graph, node_id) for node_id in found)) - recurrence for found in readers]
    seen: set[str] = set()
    shared: set[str] = set()
    for nodes in reads:
        shared |= nodes & seen
        seen |= nodes
    return [nodes - shared for nodes in reads]


def _spliced(graph: Graph, region: set[str], state: str, stored: str) -> tuple[Body, tuple[str, ...], tuple[tuple[str, object], ...]]:
    """The step as ONE body under step-independent buffer names, the buffers it keeps — the state
    it stores first, then whatever else of the recurrence is read outside the step — and the
    constants it reads, ``(name, value)`` in first-use order. The tracer gives every step its own
    copy of a scalar constant; the body names each by its position, so two steps reading equal
    constants spell the same body, and the rolled kernel reads the first step's copies."""
    stateful = _stateful(graph, region, state)
    live = (stored, *(name for name in live_outputs_of(graph, region) if name != stored and graph.producer(name).id in stateful))
    try:
        merged = build_merged_region(graph, region, live)
    except UnfusableStmt as doom:
        raise RuleSkipped(f"a step does not splice: {doom}") from doom
    if merged is None:
        raise RuleSkipped("a step does not splice")
    constants = [
        (load.input, graph.producer(load.input).op.value)
        for load in dict.fromkeys(merged.body.loads)
        if load.input in region and isinstance(graph.nodes[load.input].op, ConstantOp)
    ]
    constants = list(dict.fromkeys(constants))
    names = {state: _STATE, **{name: f"{_STEP}{index}" for index, name in enumerate(live)}}
    names.update((name, f"{_CONST}{index}") for index, (name, _) in enumerate(constants))
    return LoopOp(body=Body(merged.body).rename_buffers(names)).body, live, tuple(constants)


def _spelling(body: Body) -> tuple[str, tuple[str, ...]]:
    """What two steps must agree on to be one computation: the body's structure, blind to the names
    of values and axes and to the order of commutative operands, and the buffer that fills each of its roles.
    The order its loads stand in is no part of it: a product may spell its two reads either way round."""
    identity = body.identity(structural=False)
    return identity.key, tuple(identity.arguments)


class Unproven(RuleSkipped):
    """Programs that pass for the copies of one step — as many nodes each, reading the same buffers beside their
    state — and are not proved to be one: the difference :func:`prove` names."""


def prove(*programs: Body):
    """The proof that ``programs`` are one step taken again and again, and that step at any place in the walk
    (the proof's ``at``): each program is the first with every integer of its coordinate expressions and static
    extents moved as far as it moves from the first to the second, as many times as the program is steps later.

    The crate's proof (``prove_steps``) pairs the statements of two programs by what they compute and where they
    nest, never by where they stand, and holds for the programs that are the first one's statements with other
    integers. It stops at a program the normal form spells otherwise: the step at the edge of its matrix, whose
    mask lost the bound the extent already gives. Such a program is held to the step with its step number put in,
    in normal form. Raises :class:`Unproven` with the first difference."""
    reads = [frozenset(load.input for load in body.loads) for body in programs]
    for index, read in enumerate(reads):
        if read != reads[0]:
            # Steps that read other buffers are other computations: the layers of a model, each with its weights.
            raise RuleSkipped(f"step {index} reads {sorted(read - reads[0])} where step 0 reads {sorted(reads[0] - read)}")
    try:
        proved = prove_steps(list(programs))
    except ValueError as why:
        raise Unproven(f"step 1 is not step 0 a step later: {why}") from None
    for index in range(proved.steps, len(programs)):
        if _spelling(LoopOp(body=proved.at(index)).body) != _spelling(programs[index]):
            why = proved.stopped if index == proved.steps else "it is another body with its step number put in"
            raise Unproven(f"step {index} is not step 0 {index} steps later: {why}")
    return proved


def _rolled(step: Body, steps: int, seed: float | str, every: bool) -> Body:
    """The step under a loop that carries its state: the state read becomes a load of the state one step
    back, its store the state's definition, and every kept buffer gains the step as its leading axis. A state
    read at its last step alone (``every`` false) gains none: each step stores over the last one's,
    so the buffer holds the state the loop ended with."""

    def roll(stmt: Statement) -> Statement | tuple[Statement, ...]:
        if isinstance(stmt, Load) and stmt.input == _STATE:
            return stmt.replace(index=(Var("step") - Literal(1, "int"), *stmt.index))
        if not isinstance(stmt, Write):
            return stmt
        kept = stmt.replace(index=(Var("step"), *stmt.index))
        if stmt.output != f"{_STEP}0":
            return kept
        if len(stmt.values) != 1 or not all(isinstance(e, Var) or e == Literal(0, "int") for e in stmt.index):
            raise RuleSkipped("the state is not stored one value per cell")
        return (Carry(name=_STATE, value=stmt.values[0], index=kept.index, seed=seed), kept if every else stmt)

    return Body((Loop(axis=Axis("step", steps), body=step.map(roll)),))


def _slice(tensor: Tensor, source: str, step: int | None) -> LoopOp:
    """``tensor`` read from the rolled kernel's store: its ``step``, or the whole of a store that keeps one."""
    axes = tuple(Axis(f"s{dim}", extent) for dim, extent in enumerate(tensor.shape))
    cell = tuple(Var(axis.name) for axis in axes)
    at = cell if step is None else (Literal(step, "int"), *cell)
    body = Body((Load(names=("kept",), input=source, index=at), Write(output=tensor.name, index=cell, values=("kept",))))
    for axis in reversed(axes):
        body = Body((Loop(axis=axis, body=body),))
    return LoopOp(body=body)


def roll(match: Match, partition: dict[str, frozenset[str]]) -> Graph | None:
    """The fragment that rolls a recurrence one of the graph's regions holds, or ``None`` where none holds one: the
    first of the regions' loops, in topological order, that starts one (:func:`roll_from`)."""
    graph = match.graph
    kind = kinds(graph, partition)
    alike = Counter(kind.values())
    for node_id in graph.topological_order():
        # A chain is two states of one kind: most loops have no such second.
        if alike[kind.get(node_id)] < 2:
            continue
        try:
            return roll_from(match, graph.nodes[node_id], kind)
        except RuleSkipped as why:
            logger.debug("no recurrence rolls from %s: %s", node_id, why)
    return None


def roll_from(match: Match, first: Node, kind: dict[str, tuple]) -> Graph:
    """The fragment that rolls the recurrence whose first state ``first`` stores: the first way to read one there
    (:func:`found`) whose steps are proved to be one step (:func:`_proved`), written as one kernel
    (:func:`_fragment`)."""
    declined = RuleSkipped("no later state of this kind depends on this one")
    for recurrence in found(match.graph, first, kind):
        try:
            return _fragment(match, recurrence, *_proved(match.graph, recurrence))
        except RuleSkipped as why:
            declined = why
    raise declined


@dataclass(frozen=True)
class Recurrence:
    """One way to read part of a region as a recurrence: its states in order, the nodes of each step, and the state
    each step starts from, the first one's the seed."""

    chain: list[Node]
    steps: list[set[str]]
    states: list[str]


def found(graph: Graph, first: Node, kind: dict[str, tuple]) -> Iterator[Recurrence]:
    """The ways to read what follows ``first`` as a recurrence whose first state it stores, the one to take first:
    at a stride of one state then of two, with the step at the matrix's edge then without it, each step with its own
    readers then alone. Which of them is a recurrence is the proof's to say."""
    if first.outputs[0].dtype != F32 or first.id not in kind:
        return  # a carried state is an f32 tensor a loop of a region stores
    chain, edge = _chain(graph, first, kind)
    for period in _PERIODS:
        states = chain[::period]
        if len(states) < 2:
            break
        # The step at the matrix's edge follows the chain's last where the stride reaches it.
        last = [edge] if edge is not None and len(chain) % period == 0 else []
        try:
            steps, read, count = _steps(graph, [*states, *last])
        except RuleSkipped:
            continue
        if count == len(states) + 1:
            yield from _with_readers(graph, [*states, *last], steps, read)
            steps, read, count = steps[:-1], read[:-1], count - 1
        yield from _with_readers(graph, states[:count], steps, read)


def _with_readers(graph: Graph, chain: list[Node], steps: list[set[str]], states: list[str]) -> Iterator[Recurrence]:
    """Each step with its own readers, where those are one body at a stride as the steps are; then the steps alone,
    where every reader takes its state from a store. Steps that are one body hold as many nodes each, which is asked
    before any of them is spliced: two unrelated products of one matrix are a chain of two by their bodies, the
    first one's readers are then everything that reads the matrix, and a splice of that many nodes over an unrolled
    recurrence does not come back."""
    readers = _readers(graph, steps, states, chain)
    joined = [step | read for step, read in zip(steps, readers, strict=True)]
    if any(readers) and len({len(nodes) for nodes in joined}) == 1:
        yield Recurrence(chain, joined, states)
    yield Recurrence(chain, steps, states)


def _proved(graph: Graph, recurrence: Recurrence) -> tuple[Body, list[tuple[str, ...]], tuple[tuple[str, object], ...], bool]:
    """What makes ``recurrence`` one: each step splices into one body, the bodies keep as many buffers and read the
    same constants, and they are one step taken again and again (:func:`prove`). Raises ``RuleSkipped`` with what
    does not hold. Returns that step under the loop that carries the state, the buffers each step keeps (the state
    it stores first), the constants the step reads, and whether every state is stored or the last alone."""
    chain, steps, states = recurrence.chain, recurrence.steps, recurrence.states
    # A step is loops and the leaves they read — a constant, an input first read there — which the
    # splicer reads as external inputs of the merged body; anything else is not a step.
    if len(steps) < 2 or not all(isinstance(graph.nodes[node_id].op, (LoopOp, ConstantOp, InputOp)) for step in steps for node_id in step):
        raise RuleSkipped("the steps are not alike")

    spliced = [_spliced(graph, step, state, stored.id) for step, state, stored in zip(steps, states, chain, strict=True)]
    constants = spliced[0][2]
    # Fused whole, a chain re-derives a state only when a step reads it at another cell, or when a
    # state is read by anything but the next step. Without either it is straight-line code, which
    # fusion inlines once; rolling it would only store every state.
    # A state nothing but the next step reads is stored for no one: the store keeps the last state alone.
    every = any(
        node.outputs[0].name in graph.outputs or not set(graph.buffer_users(node.outputs[0].name)) <= steps[index + 1]
        for index, node in enumerate(chain[:-1])
    )
    if not every and all(
        len(body.writes) == 1 and all(load.index == body.writes[0].index for load in body.loads if load.input == _STATE)
        for body, _, _ in spliced
    ):
        raise RuleSkipped("no state is re-derived when the chain fuses whole")
    if len({len(live) for _, live, _ in spliced}) != 1:
        raise RuleSkipped("the steps keep different buffers")
    if any(tuple(value for _, value in read) != tuple(value for _, value in constants) for _, _, read in spliced):
        raise RuleSkipped("the steps read different constants")
    proved = prove(*(body for body, _, _ in spliced))
    try:
        step = proved.at("step", len(steps))
    except ValueError as why:
        raise RuleSkipped(str(why)) from None
    seed: float | str = 0.0 if _is_zero(graph, states[0]) else states[0]
    return _rolled(step, len(steps), seed, every), [live for _, live, _ in spliced], constants, every


def _fragment(match: Match, recurrence: Recurrence, body: Body, kept: list[tuple[str, ...]], constants: tuple, every: bool) -> Graph:
    """The fragment that replaces the chain: one kernel that carries the state over the steps (``body``, as
    :func:`_proved` returns it), and a slice of its stores for each buffer of a step that is read outside."""
    graph = match.graph
    chain, steps, states = recurrence.chain, recurrence.steps, recurrence.states
    stores = tuple(f"{chain[0].id}__steps{index}" for index in range(len(kept[0])))
    back = {f"{_STEP}{i}": name for i, name in enumerate(stores)}
    back.update((f"{_CONST}{index}", name) for index, (name, _) in enumerate(constants))
    rolled = LoopOp(body=body.rename_buffers(back))
    tensors = tuple(
        Tensor(name, (*((len(steps),) if every or index else ()), *graph.buffer(old).shape), graph.buffer(old).dtype)
        for index, (name, old) in enumerate(zip(stores, kept[0], strict=True))
    )
    fragment = Graph()
    for name in rolled.inputs:
        fragment.add_node(InputOp(), [], graph.buffer(name), node_id=name)
    fragment.add_node(
        replace(rolled, outputs=dict(zip(stores, tensors, strict=True))), list(rolled.inputs), outputs=tensors, node_id=stores[0]
    )
    renamed: dict[str, str] = {}
    for index, live in enumerate(kept):
        for store, old in zip(stores, live, strict=True):
            last = store == stores[0] and not every
            if last and index < len(steps) - 1:
                continue
            # The splice restores the replaced buffer's id on the slice; its tensor carries that name.
            renamed[old] = f"{old}__rolled"
            tensor = replace(graph.buffer(old), name=old)
            fragment.add_node(_slice(tensor, store, None if last else index), [store], tensor, node_id=renamed[old])
    fragment.outputs = list(renamed.values())
    match.consumed = set().union(*(_stateful(graph, step, state) for step, state in zip(steps, states, strict=True)))
    match.output = renamed
    return fragment
