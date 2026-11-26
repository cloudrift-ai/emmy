"""Roll an unrolled recurrence into ONE kernel that carries its state.

A Python loop over chunks leaves the tracer unrolled: step ``j``'s loops read the state step ``j − 1`` stored, so
splicing them whole makes every later state re-derive every earlier one, and a chain of twenty reductions demands
two to the twentieth bindings of its first step. The merge rule (``010_merge_loop_ops``) asks :func:`roll` before
it splices a region, and rolls such a chain where it would have inlined it.

A roll is a splice with more asked of its nodes, in three parts kept apart:

- :func:`found` lists the ways to read part of a region as a recurrence. The copies of a step are nodes of one
  region, the set the merge rule would splice into one kernel, so a region is where they are looked for. The states
  are a chain of its nodes of one kind (:func:`kinds`), each depending on the last; a step is what lies between two
  of them, and the first step starts from the seed. A reading may take every second state, where the states
  alternate two bodies (a Sinkhorn row step then a column step).
- :func:`_proved` holds a reading to what a recurrence is: every step splices into one body, the bodies are one step
  once the integers of their coordinates are left out and each integer moves by one amount a step (:func:`prove`),
  and each step reads the state the step before it stored. A chain that is not one step advanced is left alone: a
  recurrence rolled wrongly is a wrong answer. So is a chain whose steps read their state only at the cell they
  write and whose states nothing but the next step reads: fused whole it re-derives no state, so it is
  straight-line code, and rolling it would only store every state.
- :func:`_fragment` writes the kernel: the step under a loop that carries the state (``Carry``) from the seed and
  stores what each step defined (the last state alone, where nothing but the next step reads a state), and one
  slice of those stores per replaced buffer.

The readings are held to the proof largest first, by the nodes their steps hold, and the first that is proved is
written.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field, replace

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


@dataclass(frozen=True)
class Kinds:
    """What the roll reads of a graph that stands still, worked out once for every reading it is asked for.

    ``of`` is, per loop of a region of two or more, what a state of a recurrence shares with the states after it:
    the region, the shape and the type of what it stores, and its body once the integers of its coordinates and
    static extents are left out (``Body.identity_apart``). The copies of a step are nodes of one region: what
    fusion would splice into one kernel. A loop alone in its region starts no chain, and it is the kernel a
    region was spliced into, whose body is the costly one to key. ``alike`` is the loops of each kind and
    ``order`` the loops of ``of``, both in topological order.
    ``ancestors`` is, per node, itself and every node it descends from, and ``reads`` each buffer it reads with the
    node that stores it."""

    of: dict[str, tuple]
    alike: dict[tuple, list[str]]
    order: list[str]
    ancestors: dict[str, frozenset[str]]
    reads: dict[str, tuple[tuple[str, str], ...]]
    _between: dict[tuple[str, str], tuple[frozenset[str], frozenset[str]]] = field(default_factory=dict)

    def external(self, region: frozenset[str]) -> frozenset[str]:
        """The buffers the nodes of ``region`` read that none of them stores."""
        return frozenset(name for node_id in region for name, producer in self.reads[node_id] if producer not in region)

    def step(self, before: str, after: str) -> tuple[frozenset[str], frozenset[str]]:
        """The nodes between two states and the buffers they read from outside (:meth:`external`). The chains that
        start at the states of one kind share their pairs, so each pair is worked out once."""
        found = self._between.get((before, after))
        if found is None:
            region = self.ancestors[after] - self.ancestors[before]
            found = self._between[before, after] = (region, self.external(region))
        return found


def kinds(graph: Graph, partition: dict[str, frozenset[str]]) -> Kinds:
    of = {}
    for node_id, region in partition.items():
        if len(region) < 2:
            continue
        node = graph.nodes[node_id]
        of[node_id] = (region, node.outputs[0].shape, node.outputs[0].dtype, node.op.body.identity_apart())
    ancestors: dict[str, frozenset[str]] = {}
    reads: dict[str, tuple[tuple[str, str], ...]] = {}
    order = graph.topological_order()
    for node_id in order:
        producers = [(name, graph.producer(name)) for name in graph.nodes[node_id].inputs]
        producers = [(name, producer) for name, producer in producers if producer is not None]
        reads[node_id] = tuple((name, producer.id) for name, producer in producers)
        ancestors[node_id] = frozenset({node_id}).union(*(ancestors[producer.id] for _, producer in producers))
    order = [node_id for node_id in order if node_id in of]
    alike: dict[tuple, list[str]] = {}
    for node_id in order:
        alike.setdefault(of[node_id], []).append(node_id)
    return Kinds(of, alike, order, ancestors, reads)


def _chain(graph: Graph, first: Node, kind: Kinds) -> list[Node]:
    """``first`` and every later state: what descends from it and is of its kind (:class:`Kinds`)."""
    of_its_kind = kind.alike[kind.of[first.id]]
    later = of_its_kind[of_its_kind.index(first.id) + 1 :]
    return [first, *(graph.nodes[node_id] for node_id in later if first.id in kind.ancestors[node_id])]


def _is_zero(graph: Graph, name: str) -> bool:
    producer = graph.producer(name)
    if producer is None:
        return False
    if isinstance(producer.op, ConstantOp):
        return producer.op.value == 0
    layout = isinstance(producer.op, LoopOp) and not Body(producer.op.body).iter_of_type(Assign, Accum)
    return layout and all(_is_zero(graph, source) for source in producer.inputs)


def _steps(graph: Graph, chain: list[Node], kind: Kinds) -> tuple[list[frozenset[str]], list[str], int]:
    """``(one region per step, the state each step reads, how many states the recurrence spans)``.

    The first step has no state node before it: its state is the seed, the one buffer of the state's
    shape it reads beyond what every step reads — the tensor the loop started from, or the zeros it
    was seeded with. It is the buffer whose ancestry, taken out, leaves a first step shaped like
    every later one: as many nodes, reading the same things beside its state."""
    ancestors = kind.ancestors
    between = [kind.step(before.id, after.id) for before, after in zip(chain, chain[1:], strict=False)]
    states = [node.id for node in chain[:-1]]
    shared = between[0][1] - {states[0]}
    alike = (len(between[0][0]), shared)
    # The chain is every later node of the first state's body shape; a later division of the same
    # shape that closes the block is one too, with a region of everything before it. The recurrence
    # is the prefix whose steps are alike, and it ends where they stop being.
    steps = 0
    for (region, read), state in zip(between, states, strict=True):
        if (len(region), read - {state}) != alike:
            break
        steps += 1
    regions, states = [region for region, _ in between[:steps]], states[:steps]
    beyond: set[str] = set()
    for name in shared:
        beyond |= ancestors[graph.producer(name).id]
    first = ancestors[chain[0].id] - beyond
    tensor = chain[0].outputs[0]
    seeds = []
    for node_id in first:
        if node_id == chain[0].id or graph.nodes[node_id].outputs[0].shape != tensor.shape:
            continue
        seed = node_id
        region = first - ancestors[node_id]
        if (len(region), kind.external(region) - {seed}) == alike:
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


def prove(*programs: Body):
    """The proof that ``programs`` are one step taken again and again, and that step at any place in the walk
    (the proof's ``at``): each program is the first with every integer of its coordinate expressions and static
    extents moved as far as it moves from the first to the second, as many times as the program is steps later.

    The crate's proof (``prove_steps``) pairs the statements of two programs by what they compute and where they
    nest, never by where they stand, and holds for the programs that are the first one's statements with other
    integers. Raises ``RuleSkipped`` with the first difference."""
    reads = [frozenset(load.input for load in body.loads) for body in programs]
    for index, read in enumerate(reads):
        if read != reads[0]:
            # Steps that read other buffers are other computations: the layers of a model, each with its weights.
            raise RuleSkipped(f"step {index} reads {sorted(read - reads[0])} where step 0 reads {sorted(reads[0] - read)}")
    try:
        proved = prove_steps(list(programs))
    except ValueError as why:
        raise RuleSkipped(f"step 1 is not step 0 a step later: {why}") from None
    if proved.steps < len(programs):
        raise RuleSkipped(f"step {proved.steps} is not step 0 {proved.steps} steps later: {proved.stopped}")
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
    """The fragment that rolls the largest recurrence the graph's regions hold, or ``None`` where they hold none.

    Every loop that starts a chain gives its readings (:func:`found`). They are held to the proof largest first, by
    the nodes their steps hold, and in topological order where two hold as many; the first that is proved
    (:func:`_proved`) is written (:func:`_fragment`). A short recurrence that shares a state with a long one — a
    softmax's own division ahead of a Sinkhorn's row step — would otherwise be taken first and leave the long one a
    step short."""
    graph = match.graph
    kind = kinds(graph, partition)
    # A chain is two states of one kind: most loops have no such second.
    firsts = [graph.nodes[node_id] for node_id in kind.order if len(kind.alike[kind.of[node_id]]) > 1]
    readings = [recurrence for first in firsts for recurrence in found(graph, first, kind)]
    for recurrence in sorted(readings, key=lambda reading: -sum(map(len, reading.steps))):
        try:
            return _fragment(match, recurrence, *_proved(graph, recurrence))
        except RuleSkipped as why:
            logger.debug("no recurrence rolls from %s: %s", recurrence.chain[0].id, why)
    return None


@dataclass(frozen=True)
class Recurrence:
    """One way to read part of a region as a recurrence: its states in order, the nodes of each step, and the state
    each step starts from, the first one's the seed."""

    chain: list[Node]
    steps: list[frozenset[str]]
    states: list[str]


def found(graph: Graph, first: Node, kind: Kinds) -> Iterator[Recurrence]:
    """The ways to read what follows ``first`` as a recurrence whose first state it stores, the one to take first:
    at a stride of one state, then of two. Which of them is a recurrence is the proof's to say."""
    if first.outputs[0].dtype != F32 or first.id not in kind.of:
        return  # a carried state is an f32 tensor a loop of a region stores
    chain = _chain(graph, first, kind)
    for period in _PERIODS:
        states = chain[::period]
        if len(states) < 2:
            break
        try:
            steps, read, count = _steps(graph, states, kind)
        except RuleSkipped:
            continue
        yield Recurrence(states[:count], steps, read)


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
