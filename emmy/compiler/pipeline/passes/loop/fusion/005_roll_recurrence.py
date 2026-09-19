"""Roll an unrolled recurrence into ONE kernel that carries its state, before fusion inlines it.

A Python loop over chunks leaves the tracer unrolled: step ``j``'s kernels read the state step
``j − 1`` stored, so fusing them whole makes every later state re-derive every earlier one. Here the
structure is still plain. The states are a chain of same-shaped nodes, each depending on the last;
what lies between two of them is one step; and the steps differ only in where they read their
inputs.

Nothing about the step is assumed. Each step is spliced into one body by the fusion rule's own
splicer, the first step's body advanced by ``j`` strides must normalize to step ``j``'s, and only
then is the chain replaced — by one kernel that carries the state (``Carry``) and stores
what each step defined, and by one slice of those stores per replaced buffer. A chain that is not
one step at a stride is left alone: a recurrence rolled wrongly is a wrong answer.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, Var, split_anchor
from emmy.compiler.ir.loop import LoopOp, UnfusableStmt
from emmy.compiler.ir.stmt import Accum, Assign, Body, Carry, Load, Loop, Pre, Stmt, Write
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.fusion._region import build_merged_region, live_outputs_of

PATTERN = [Pattern("first", LoopOp)]

_STATE, _STEP = "carried_state", "carried_step"


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
    once load anchors are taken out."""
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
        if alike and node.op.body.census == body.census and node.op.body.unanchored_key == body.unanchored_key:
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


def _steps(graph: Graph, chain: list[Node]) -> tuple[list[set[str]], list[str], set[str]]:
    """``(one region per step, the state each step reads, what every step reads beside it)``."""
    regions = [_ancestors(graph, after.id) - _ancestors(graph, before.id) for before, after in zip(chain, chain[1:], strict=False)]
    states = [node.outputs[0].name for node in chain[:-1]]
    shared = _external(graph, regions[0]) - {states[0]}
    # The first step has no state node before it: it is what the first state needs beyond what
    # every step reads, less the zero state it starts from. That zero is a buffer of the state's
    # own shape; a broadcast of it further on is the step's, as it is in every later step.
    beyond: set[str] = set()
    for name in shared:
        beyond |= _ancestors(graph, graph.producer(name).id)
    first = _ancestors(graph, chain[0].id) - beyond
    shape = chain[0].outputs[0].shape
    seeds = {tensor.name for nid in first if (tensor := graph.nodes[nid].outputs[0]).shape == shape and _is_zero(graph, tensor.name)}
    if len(seeds) != 1:
        raise RuleSkipped("the chain does not start from a zero state")
    first -= _ancestors(graph, graph.producer(next(iter(seeds))).id)
    if _external(graph, first) - shared != seeds:
        raise RuleSkipped("the first step reads more than the zero state and what every step reads")
    return [first, *regions], [next(iter(seeds)), *states], shared


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


def _spliced(graph: Graph, region: set[str], state: str, stored: str) -> tuple[Body, tuple[str, ...]]:
    """The step as ONE body under step-independent buffer names, and the buffers it keeps — the
    state it stores first, then whatever else of the recurrence is read outside the step."""
    stateful = _stateful(graph, region, state)
    live = (stored, *(name for name in live_outputs_of(graph, region) if name != stored and graph.producer(name).id in stateful))
    try:
        merged = build_merged_region(graph, region, live)
    except UnfusableStmt as doom:
        raise RuleSkipped(f"a step does not splice: {doom}") from doom
    if merged is None:
        raise RuleSkipped("a step does not splice")
    names = {state: _STATE, **{name: f"{_STEP}{index}" for index, name in enumerate(live)}}
    return LoopOp(body=Body(merged.body).rename_buffers(names)).body, live


def _strides(first: Body, second: Body) -> dict[tuple[int, int], int]:
    """How far each load index moves from one step to the next, by load position and dimension."""
    strides: dict[tuple[int, int], int] = {}
    if len(first.loads) != len(second.loads):
        raise RuleSkipped("two steps read a different number of values")
    for position, (load, later) in enumerate(zip(first.loads, second.loads, strict=True)):
        for dim, (expr, moved) in enumerate(zip(load.index, later.index, strict=False)):
            here, there = split_anchor(expr), split_anchor(moved)
            if here is not None and there is not None and there[0] != here[0]:
                strides[position, dim] = there[0] - here[0]
    return strides


def _advanced(body: Body, strides: dict[tuple[int, int], int], by: Expr) -> Body:
    """``body`` read ``by`` steps later: every strided load index advanced."""
    positions = {id(load): position for position, load in enumerate(body.loads)}

    def advance(stmt: Stmt) -> Stmt:
        if not isinstance(stmt, Load):
            return stmt
        position = positions[id(stmt)]
        index = tuple(
            BinaryExpr("+", expr, BinaryExpr("*", by, Literal(strides[position, dim], "int"))) if (position, dim) in strides else expr
            for dim, expr in enumerate(stmt.index)
        )
        return replace(stmt, index=index)

    return body.map(advance)


def _rolled(step: Body, strides: dict[tuple[int, int], int], steps: int) -> Body:
    """The step under a loop that carries its state: the state read becomes a ``Pre`` read, its
    store the state's definition, and every kept buffer gains the step as its leading axis."""
    time = Var("step")

    def roll(stmt: Stmt) -> Stmt | tuple[Stmt, ...]:
        if isinstance(stmt, Load) and stmt.input == _STATE:
            return Pre(name=stmt.names[0], carrier=_STATE, index=stmt.index)
        if not isinstance(stmt, Write):
            return stmt
        kept = replace(stmt, index=(time, *stmt.index))
        if stmt.output != f"{_STEP}0":
            return kept
        if len(stmt.values) != 1 or not all(isinstance(e, Var) or e == Literal(0, "int") for e in stmt.index):
            raise RuleSkipped("the state is not stored one value per cell")
        return (Carry(name=_STATE, value=stmt.values[0], index=stmt.index, seed=0.0), kept)

    return Body((Loop(axis=Axis("step", steps), body=_advanced(step, strides, time).map(roll)),))


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
    regions, states, shared = _steps(graph, chain)
    sizes = {(len(region), frozenset(_external(graph, region) - {state})) for region, state in zip(regions, states, strict=True)}
    if len(sizes) != 1 or not all(isinstance(graph.nodes[node_id].op, LoopOp) for region in regions for node_id in region):
        raise RuleSkipped("the steps are not alike")

    spliced = [_spliced(graph, region, state, stored.outputs[0].name) for region, state, stored in zip(regions, states, chain, strict=True)]
    step, kept = spliced[0]
    if len({len(live) for _, live in spliced}) != 1:
        raise RuleSkipped("the steps keep different buffers")
    strides = _strides(step, spliced[1][0])
    for index, (body, _) in enumerate(spliced):
        if LoopOp(body=_advanced(step, strides, Literal(index, "int"))).body != body:
            raise RuleSkipped(f"step {index} is not the first step read {index} strides later")

    stores = tuple(f"{chain[0].id}__steps{index}" for index in range(len(kept)))
    rolled = LoopOp(body=_rolled(step, strides, len(regions)).rename_buffers({f"{_STEP}{i}": name for i, name in enumerate(stores)}))
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
    for index, (_, live) in enumerate(spliced):
        for store, old in zip(stores, live, strict=True):
            renamed[old] = f"{old}__rolled"
            tensor = replace(graph.buffer(old), name=renamed[old])
            fragment.add_node(_slice(tensor, store, index), [store], tensor, node_id=renamed[old])
    fragment.outputs = list(renamed.values())
    match.consumed = set().union(*(_stateful(graph, region, state) for region, state in zip(regions, states, strict=True)))
    match.output = renamed
    return fragment
