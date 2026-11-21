"""Roll an unrolled recurrence into ONE kernel that carries its state, before fusion inlines it.

A Python loop over chunks leaves the tracer unrolled: step ``j``'s kernels read the state step
``j − 1`` stored, so fusing them whole makes every later state re-derive every earlier one, and a
chain of twenty reductions demands two to the twentieth bindings of its first step. Here the
structure is still plain. The states are a chain of same-shaped nodes of one body, each depending
on the last; what lies between two of them is one step; and the steps differ only in the integer
literals they read and mask at — an offset, a slice bound — each advancing affinely with the step.

Each step is spliced into one body by the fusion rule's own splicer, the first step's body with its
literals advanced by ``j`` must normalize to step ``j``'s, and only then is the chain replaced — by
one kernel that carries the state (``Carry``) from the tensor the loop started with, stores what
each step defined (the last state alone, where nothing but the next step reads a state), and by one
slice of those stores per replaced buffer. A chain whose states alternate two bodies (a Sinkhorn row
step then a column step) rolls as one step of two; a chain that is not one step advanced is left
alone: a recurrence rolled wrongly is a wrong answer. A chain whose steps read their state only at
the cell they write, and whose states nothing but the next step reads, is left alone too: fused
whole it re-derives no state, so it is straight-line code, and rolling it would only store every
state.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.expression import BinaryExpression, Expression, Literal, Var, affine_form
from emmy.compiler.ir.loop import LoopOp, UnfusableStmt
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Assign, Body, Carry, Load, Loop, Pre, Statement, Write
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.fusion._region import build_merged_region, live_outputs_of

PATTERN = [Pattern("first", LoopOp)]

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


@lru_cache(maxsize=256)
def _census(body: Body) -> tuple[tuple[str, int], ...]:
    """How many statements of each kind ``body`` holds, nested ones included — a linear-time
    necessary condition for two bodies to be one computation, to ask before an exact key."""
    counts: dict[str, int] = {}
    for stmt in body.iter():
        kind = type(stmt).__name__
        counts[kind] = counts.get(kind, 0) + 1
    return tuple(sorted(counts.items()))


@lru_cache(maxsize=256)
def _literals_abstracted(body: Body) -> tuple[Body, tuple[int, ...]]:
    """``body`` with every integer literal of its expressions replaced by a positional
    variable ``__lit<i>``, and the literals it held, in that order. The copies of one computation
    an unrolled loop leaves read at successive offsets and mask at successive bounds: they share
    the abstracted body and differ in the literals, affinely in the step."""
    literals: list[int] = []

    def held(value: int) -> Var:
        literals.append(value)
        return Var(f"__lit{len(literals) - 1}")

    def abstract(expression):
        if isinstance(expression, Literal) and expression.dtype == "int" and type(expression.value) is int:
            return held(expression.value)
        return expression

    def lift(expression):
        # An affine index carries its anchor whole, a zero one included: ``a`` and ``a + 4`` are
        # one shape read at two offsets, and normalization spells the first without the ``+ 0``.
        # A value's float offset (``acc + 1e-6``) is no integer literal and stays as spelled.
        if isinstance(expression, Literal):
            return abstract(expression)
        form = affine_form(expression, expression.free_vars())
        try:
            anchor = form[0].eval({}) if form is not None else None
        except (KeyError, TypeError, ValueError):
            anchor = None
        if type(anchor) is not int:
            return expression.rebuild(abstract)
        rest = None
        for name, coeff in sorted(form[1].items()):
            term = Var(name) if coeff == 1 else BinaryExpression("*", Var(name), Literal(coeff, "int"))
            rest = term if rest is None else BinaryExpression("+", rest, term)
        return held(anchor) if rest is None else BinaryExpression("+", rest, held(anchor))

    def abstracted(stmt):
        # A loop's static extent is a literal too: a step whose reduction covers one more
        # element than the last is still the same step.
        if isinstance(stmt, Loop) and isinstance(stmt.axis.extent.expression, Literal) and type(stmt.axis.extent.expression.value) is int:
            return stmt.replace(axis=replace(stmt.axis, extent=Dim(held(stmt.axis.extent.expression.value))))
        return stmt.map_expressions(lift)

    return body.map(abstracted), tuple(literals)


@lru_cache(maxsize=256)
def _literal_free_key(body: Body) -> str:
    """The exact identity of ``body`` blind to its integer literals — what the copies of one
    computation at successive offsets and bounds share (:func:`_literals_abstracted`)."""
    return _literals_abstracted(body)[0].structural_key(structural=False)


def _chain(graph: Graph, first: Node) -> list[Node]:
    """``first`` and every later state: what descends from it with the same shape and the same body
    once its integer literals are taken out."""
    body, tensor = first.op.body, first.outputs[0]
    chain, reached = [first], {first.id}
    for node_id in graph.topological_order():
        node = graph.nodes[node_id]
        if not any(producer.id in reached for name in node.inputs if (producer := graph.producer(name)) is not None):
            continue
        reached.add(node_id)
        alike = isinstance(node.op, LoopOp) and node.outputs[0].shape == tensor.shape and node.outputs[0].dtype == tensor.dtype
        # The exact key canonicalizes the body, which a fused kernel of thousands of statements pays
        # for in seconds: ask only of a body with the same statements to begin with.
        if alike and _census(node.op.body) == _census(body) and _literal_free_key(node.op.body) == _literal_free_key(body):
            chain.append(node)
    return chain


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


def _steps(graph: Graph, chain: list[Node]) -> tuple[list[set[str]], list[str], set[str], int]:
    """``(one region per step, the state each step reads, what every step reads beside it, how many
    states the recurrence spans)``.

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
    return [region, *regions], [seed, *states], shared, steps + 1


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
    beside = [
        set().union(*(_ancestors(graph, node_id) for node_id in found)) - _ancestors(graph, state) - taken
        for found, state in zip(readers, states, strict=True)
    ]
    # What the readers of two steps both read is read at every step: an input of the body, as the step's own are.
    seen: set[str] = set()
    shared: set[str] = set()
    for nodes in beside:
        shared |= nodes & seen
        seen |= nodes
    return [nodes - shared for nodes in beside]


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


def _advanced(abstract: Body, literals: tuple[int, ...], deltas: tuple[int, ...], by: int | Var, steps: int = 1) -> Body:
    """The step read ``by`` steps later: every literal of the first step advanced by its delta —
    exactly, for a step number, or as an expression of the step axis for the rolled body. A
    reduction whose extent grows with the step takes its widest extent in the rolled body and walks
    the range of the step it is in, no wider than a step the chain ran."""

    def value(literal: int, delta: int) -> Expression:
        if isinstance(by, int):
            return Literal(literal + delta * by, "int")
        return (
            BinaryExpression("+", Literal(literal, "int"), BinaryExpression("*", by, Literal(delta, "int")))
            if delta
            else Literal(literal, "int")
        )

    def fold(expression: Expression) -> Expression:
        # The abstraction lifts every affine anchor, so a zero one comes back as ``+ 0``.
        if isinstance(expression, BinaryExpression) and expression.op == "+" and expression.right == Literal(0, "int"):
            return expression.left
        return expression

    sigma = Sigma({f"__lit{index}": value(literal, delta) for index, (literal, delta) in enumerate(zip(literals, deltas, strict=True))})

    def folds(stmt: Statement) -> Statement:
        if isinstance(stmt, (Write, Carry)):
            raise RuleSkipped("a loop whose extent grows with the step stores")
        return stmt

    def advance(stmt: Statement) -> Statement:
        if isinstance(stmt, Loop):
            expression = stmt.axis.extent.expression
            if not (isinstance(expression, Var) and expression.name.startswith("__lit")):
                return stmt
            literal, delta = literals[int(expression.name[5:])], deltas[int(expression.name[5:])]
            if isinstance(by, int) or delta == 0:
                return stmt.replace(axis=replace(stmt.axis, extent=Dim(literal + delta * (by if isinstance(by, int) else 0))))
            widest = max(literal, literal + delta * (steps - 1))
            walked = (Literal(0, "int"), value(literal, delta))
            return stmt.replace(axis=replace(stmt.axis, extent=Dim(widest)), range=walked, body=stmt.body.map(folds))
        if stmt.nested():
            return stmt
        return stmt.rewrite(lambda name: name, sigma).map_expressions(lambda expression: expression.rebuild(fold))

    return abstract.map(advance)


def _spelling(body: Body) -> tuple[str, tuple[str, ...]]:
    """What two steps must agree on to be one computation: the body's structure, blind to the names
    of values and axes and to the order of commutative operands, and the buffer that fills each of its roles.
    The order its loads stand in is no part of it: a product may spell its two reads either way round."""
    identity = body.identity(structural=False)
    return identity.key, tuple(identity.arguments)


def _rolled(step: Body, steps: int, seed: float | str, every: bool) -> Body:
    """The step under a loop that carries its state: the state read becomes a ``Pre`` read, its
    store the state's definition, and every kept buffer gains the step as its leading axis. A state
    read at its last step alone (``every`` false) gains none: each step stores over the last one's,
    so the buffer holds the state the loop ended with."""

    def roll(stmt: Statement) -> Statement | tuple[Statement, ...]:
        if isinstance(stmt, Load) and stmt.input == _STATE:
            return Pre(name=stmt.names[0], carrier=_STATE, index=stmt.index)
        if not isinstance(stmt, Write):
            return stmt
        kept = stmt.replace(index=(Var("step"), *stmt.index))
        if stmt.output != f"{_STEP}0":
            return kept
        if len(stmt.values) != 1 or not all(isinstance(e, Var) or e == Literal(0, "int") for e in stmt.index):
            raise RuleSkipped("the state is not stored one value per cell")
        return (Carry(name=_STATE, value=stmt.values[0], index=stmt.index, seed=seed), kept if every else stmt)

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


def rewrite(match: Match, first: Node) -> Graph:
    graph = match.graph
    if first.outputs[0].dtype != F32:
        raise RuleSkipped("a carried state is f32")
    chain = _chain(graph, first)
    if len(chain) < 2:
        raise RuleSkipped("no later state of this shape depends on this one")
    declined: RuleSkipped | None = None
    for period in _PERIODS:
        if len(chain[::period]) < 2:
            break
        try:
            return _roll(match, chain[::period])
        except RuleSkipped as why:
            declined = why
    assert declined is not None
    raise declined


def _roll(match: Match, chain: list[Node]) -> Graph:
    regions, states, _, count = _steps(match.graph, chain)
    chain = chain[:count]
    readers = _readers(match.graph, regions, states, chain)
    if any(readers):
        # Each step with its own readers, where those are one body at a stride as the steps are; else the steps
        # alone, and every reader takes its state from a store.
        try:
            return _replaced(match, chain, [region | read for region, read in zip(regions, readers, strict=True)], states)
        except RuleSkipped:
            pass
    return _replaced(match, chain, regions, states)


def _replaced(match: Match, chain: list[Node], regions: list[set[str]], states: list[str]) -> Graph:
    """The fragment that replaces the chain: one kernel that carries the state over ``regions``, one a step, and a
    slice of its stores for each buffer of a step that is read outside."""
    graph = match.graph
    # A step is loops and the leaves they read — a constant, an input first read there — which the
    # splicer reads as external inputs of the merged body; anything else is not a step.
    if len(regions) < 2 or not all(
        isinstance(graph.nodes[node_id].op, (LoopOp, ConstantOp, InputOp)) for region in regions for node_id in region
    ):
        raise RuleSkipped("the steps are not alike")

    spliced = [_spliced(graph, region, state, stored.id) for region, state, stored in zip(regions, states, chain, strict=True)]
    step, kept, constants = spliced[0]
    # Fused whole, a chain re-derives a state only when a step reads it at another cell, or when a
    # state is read by anything but the next step. Without either it is straight-line code, which
    # fusion inlines once; rolling it would only store every state.
    # A state nothing but the next step reads is stored for no one: the store keeps the last state alone.
    every = any(
        node.outputs[0].name in graph.outputs or not set(graph.buffer_users(node.outputs[0].name)) <= regions[index + 1]
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
    abstract, literals = _literals_abstracted(step)
    second, later = _literals_abstracted(spliced[1][0])
    if _literal_free_key(second) != _literal_free_key(abstract) or len(later) != len(literals):
        raise RuleSkipped("the second step is not the first step's body")
    deltas = tuple(b - a for a, b in zip(literals, later, strict=True))
    for index, (body, _, _) in enumerate(spliced):
        if _spelling(LoopOp(body=_advanced(abstract, literals, deltas, index)).body) != _spelling(body):
            raise RuleSkipped(f"step {index} is not the first step advanced {index} times")

    seed: float | str = 0.0 if _is_zero(graph, states[0]) else states[0]
    stores = tuple(f"{chain[0].id}__steps{index}" for index in range(len(kept)))
    body = _rolled(_advanced(abstract, literals, deltas, Var("step"), len(regions)), len(regions), seed, every)
    back = {f"{_STEP}{i}": name for i, name in enumerate(stores)}
    back.update((f"{_CONST}{index}", name) for index, (name, _) in enumerate(constants))
    rolled = LoopOp(body=body.rename_buffers(back))
    tensors = tuple(
        Tensor(name, (*((len(regions),) if every or index else ()), *graph.buffer(old).shape), graph.buffer(old).dtype)
        for index, (name, old) in enumerate(zip(stores, kept, strict=True))
    )
    fragment = Graph()
    for name in rolled.inputs:
        fragment.add_node(InputOp(), [], graph.buffer(name), node_id=name)
    fragment.add_node(
        replace(rolled, outputs=dict(zip(stores, tensors, strict=True))), list(rolled.inputs), outputs=tensors, node_id=stores[0]
    )
    renamed: dict[str, str] = {}
    for index, (_, live, _) in enumerate(spliced):
        for store, old in zip(stores, live, strict=True):
            last = store == stores[0] and not every
            if last and index < len(spliced) - 1:
                continue
            # The splice restores the replaced buffer's id on the slice; its tensor carries that name.
            renamed[old] = f"{old}__rolled"
            tensor = replace(graph.buffer(old), name=old)
            fragment.add_node(_slice(tensor, store, None if last else index), [store], tensor, node_id=renamed[old])
    fragment.outputs = list(renamed.values())
    match.consumed = set().union(*(_stateful(graph, region, state) for region, state in zip(regions, states, strict=True)))
    match.output = renamed
    return fragment
