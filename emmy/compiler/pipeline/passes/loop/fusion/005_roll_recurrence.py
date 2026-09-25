"""Roll an unrolled recurrence into ONE kernel that carries its state, before fusion inlines it.

A Python loop over chunks leaves the tracer unrolled: step ``j``'s kernels read the state step
``j − 1`` stored, so fusing them whole makes every later state re-derive every earlier one, and a
chain of twenty reductions demands two to the twentieth bindings of its first step. Here the
structure is still plain. The states are a chain of same-shaped nodes of one body, each depending
on the last; what lies between two of them is one step; and the steps differ only in the integer
literals they read and mask at — an offset, a slice bound — each advancing affinely with the step.

Nothing about the step is assumed. Each step is spliced into one body by the fusion rule's own
splicer, the first step's body with its literals advanced by ``j`` must normalize to step ``j``'s,
and only then is the chain replaced — by one kernel that carries the state (``Carry``) from the
tensor the loop started with, stores what each step defined, and by one slice of those stores per
replaced buffer. A chain whose states alternate two bodies (a Sinkhorn row step then a column
step) rolls as one step of two; a chain that is not one step advanced is left alone: a recurrence
rolled wrongly is a wrong answer.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
from emmy.compiler.ir.loop import LoopOp, UnfusableStmt
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Assign, Body, Carry, Let, Load, Loop, Pre, Select, SelectBranch, Stmt, Write
from emmy.compiler.ir.stmt.passes import map_exprs
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
        if alike and node.op.body.census == body.census and node.op.body.literal_free_key == body.literal_free_key:
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
    reduction whose extent grows with the step runs at its widest in the rolled body, and past the
    step's own bound each accumulate folds its identity instead — the mask the tracer itself
    spells a partial reduction with, and no wider than a step the chain ran."""

    def value(literal: int, delta: int) -> Expr:
        if isinstance(by, int):
            return Literal(literal + delta * by, "int")
        return BinaryExpr("+", Literal(literal, "int"), BinaryExpr("*", by, Literal(delta, "int"))) if delta else Literal(literal, "int")

    def fold(expr: Expr) -> Expr:
        # The abstraction lifts every affine anchor, so a zero one comes back as ``+ 0``.
        if isinstance(expr, BinaryExpr) and expr.op == "+" and expr.right == Literal(0, "int"):
            return expr.left
        return expr

    sigma = Sigma({f"__lit{index}": value(literal, delta) for index, (literal, delta) in enumerate(zip(literals, deltas, strict=True))})
    masked = 0

    def mask(within: Expr):
        def past_the_bound(stmt: Stmt) -> Stmt | tuple[Stmt, ...]:
            nonlocal masked
            if not isinstance(stmt, Accum):
                if isinstance(stmt, (Write, Carry)):
                    raise RuleSkipped("a loop whose extent grows with the step stores")
                return stmt
            if stmt.op.identity is None:
                raise RuleSkipped(f"{stmt.op.name} has no identity to fold past a step's bound")
            idle, contributed = f"{stmt.name}__idle{masked}", f"{stmt.name}__in{masked}"
            masked += 1
            select = Select(name=contributed, branches=(SelectBranch(stmt.value, within), SelectBranch(idle, Literal(True, "bool"))))
            return (Let(name=idle, value=stmt.init), select, replace(stmt, value=contributed))

        return past_the_bound

    def advance(stmt: Stmt) -> Stmt:
        if isinstance(stmt, Loop):
            expr = stmt.axis.extent.expr
            if not (isinstance(expr, Var) and expr.name.startswith("__lit")):
                return stmt
            literal, delta = literals[int(expr.name[5:])], deltas[int(expr.name[5:])]
            if isinstance(by, int) or delta == 0:
                return replace(stmt, axis=replace(stmt.axis, extent=Dim(literal + delta * (by if isinstance(by, int) else 0))))
            widest = max(literal, literal + delta * (steps - 1))
            within = BinaryExpr("<", Var(stmt.axis.name), value(literal, delta))
            return replace(stmt, axis=replace(stmt.axis, extent=Dim(widest)), body=stmt.body.map(mask(within)))
        if stmt.nested():
            return stmt
        return map_exprs(stmt.rewrite(lambda name: name, sigma), lambda expr: expr.rebuild(fold))

    return abstract.map(advance)


def _spelling(body: Body) -> tuple[str, tuple[str, ...]]:
    """What two steps must agree on to be one computation: the body's structure, blind to the names
    of values and axes and to the order of commutative operands, and the buffers it reads, in order."""
    return body.structural_key(structural=False), tuple(load.input for load in body.loads)


def _rolled(step: Body, steps: int, seed: float | str) -> Body:
    """The step under a loop that carries its state: the state read becomes a ``Pre`` read, its
    store the state's definition, and every kept buffer gains the step as its leading axis."""

    def roll(stmt: Stmt) -> Stmt | tuple[Stmt, ...]:
        if isinstance(stmt, Load) and stmt.input == _STATE:
            return Pre(name=stmt.names[0], carrier=_STATE, index=stmt.index)
        if not isinstance(stmt, Write):
            return stmt
        kept = replace(stmt, index=(Var("step"), *stmt.index))
        if stmt.output != f"{_STEP}0":
            return kept
        if len(stmt.values) != 1 or not all(isinstance(e, Var) or e == Literal(0, "int") for e in stmt.index):
            raise RuleSkipped("the state is not stored one value per cell")
        return (Carry(name=_STATE, value=stmt.values[0], index=stmt.index, seed=seed), kept)

    return Body((Loop(axis=Axis("step", steps), body=step.map(roll)),))


def _slice(tensor: Tensor, source: str, step: int) -> LoopOp:
    axes = tuple(Axis(f"s{dim}", extent) for dim, extent in enumerate(tensor.shape))
    cell = tuple(Var(axis.name) for axis in axes)
    body = Body((Load(name="kept", input=source, index=(Literal(step, "int"), *cell)), Write(output=tensor.name, index=cell, value="kept")))
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
    graph = match.graph
    regions, states, shared, count = _steps(graph, chain)
    chain = chain[:count]
    # A step is loops and the leaves they read — a constant, an input first read there — which the
    # splicer reads as external inputs of the merged body; anything else is not a step.
    if len(regions) < 2 or not all(
        isinstance(graph.nodes[node_id].op, (LoopOp, ConstantOp, InputOp)) for region in regions for node_id in region
    ):
        raise RuleSkipped("the steps are not alike")

    spliced = [_spliced(graph, region, state, stored.id) for region, state, stored in zip(regions, states, chain, strict=True)]
    step, kept, constants = spliced[0]
    if len({len(live) for _, live, _ in spliced}) != 1:
        raise RuleSkipped("the steps keep different buffers")
    if any(tuple(value for _, value in read) != tuple(value for _, value in constants) for _, _, read in spliced):
        raise RuleSkipped("the steps read different constants")
    abstract, literals = step.literals_abstracted
    second, later = spliced[1][0].literals_abstracted
    if second.literal_free_key != abstract.literal_free_key or len(later) != len(literals):
        raise RuleSkipped("the second step is not the first step's body")
    deltas = tuple(b - a for a, b in zip(literals, later, strict=True))
    for index, (body, _, _) in enumerate(spliced):
        if _spelling(LoopOp(body=_advanced(abstract, literals, deltas, index)).body) != _spelling(body):
            raise RuleSkipped(f"step {index} is not the first step advanced {index} times")

    seed: float | str = 0.0 if _is_zero(graph, states[0]) else states[0]
    stores = tuple(f"{chain[0].id}__steps{index}" for index in range(len(kept)))
    body = _rolled(_advanced(abstract, literals, deltas, Var("step"), len(regions)), len(regions), seed)
    back = {f"{_STEP}{i}": name for i, name in enumerate(stores)}
    back.update((f"{_CONST}{index}", name) for index, (name, _) in enumerate(constants))
    rolled = LoopOp(body=body.rename_buffers(back))
    tensors = tuple(
        Tensor(name, (len(regions), *graph.buffer(old).shape), graph.buffer(old).dtype) for name, old in zip(stores, kept, strict=True)
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
            # The splice restores the replaced buffer's id on the slice; its tensor carries that name.
            renamed[old] = f"{old}__rolled"
            tensor = replace(graph.buffer(old), name=old)
            fragment.add_node(_slice(tensor, store, index), [store], tensor, node_id=renamed[old])
    fragment.outputs = list(renamed.values())
    match.consumed = set().union(*(_stateful(graph, region, state) for region, state in zip(regions, states, strict=True)))
    match.output = renamed
    return fragment
