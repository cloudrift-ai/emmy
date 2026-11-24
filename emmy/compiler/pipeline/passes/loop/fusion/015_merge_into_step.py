"""Merge the kernels around a kernel that carries a state into its step.

``010_merge_loop_ops`` has merged every region of plain loops by now, with a kernel that carries a state between
them. Such a kernel is one step under the loop that carries the state, and a kernel beside it joins that step where
it can run there:

- a kernel only the step reads is computed where the step reads it, like any producer a splice inlines;
- a kernel that reads what the step stores joins where each of its trips reads one step, the same one for every read:
  its loop over the coordinate the step is a digit of becomes the step and the trips of that step
  (:func:`in_step`), and what it stores it then stores from inside the step.

What cannot run in the step stays a kernel: a reader of the state the loop ends with, a reduction over the steps,
another kernel that carries a state. The merge is the fusion rule's own splice; which of these kernels are worth
keeping apart is a cut's decision, taken on the merged kernel.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expression import BinaryExpression, Interval, Literal, SimplifyCtx, Var, affine_form
from emmy.compiler.ir.loop import LoopOp, splice_graph
from emmy.compiler.ir.stmt import Accum, Body, Carry, Let, Load, Loop, Select, SelectBranch, Statement, Write
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.fusion._region import (
    build_merged_region,
    chosen_cell,
    joins_a_region,
    live_outputs_of,
    plain_load,
    stops_a_region,
    wrap_multi_output_fragment,
)

PATTERN = [Pattern("carrying", LoopOp)]

#: Where a buffer holds the step: ``(position, width)`` — the cell at index ``i`` was stored at step
#: ``i[position] / width``.
_Stepped = dict[str, tuple[int, int]]


def _extents(body: Body) -> dict[str, int] | None:
    """The static extent of every loop of ``body`` by its axis, ``None`` where one is symbolic or a name is bound
    twice."""
    extents: dict[str, int] = {}
    for loop in body.iter_of_type(Loop):
        if not loop.axis.extent.is_static or loop.axis.name in extents:
            return None
        extents[loop.axis.name] = loop.axis.extent.as_static()
    return extents


def _width(entry, step: str, extents: dict[str, int]) -> int | None:
    """``width`` where ``entry`` is ``width · step`` plus a rest that stays inside ``0..width``: the coordinate
    whose digit the step is."""
    form = affine_form(entry, entry.free_vars())
    if form is None or form[1].get(step, 0) <= 0:
        return None
    try:
        low = high = form[0].eval({})
    except (KeyError, TypeError, ValueError):
        return None
    if type(low) is not int:
        return None
    for name, coefficient in form[1].items():
        if name == step:
            continue
        if name not in extents:
            return None
        reach = coefficient * (extents[name] - 1)
        low, high = low + min(reach, 0), high + max(reach, 0)
    return form[1][step] if low >= 0 and high < form[1][step] else None


def stepped_stores(body: Body) -> tuple[Loop, _Stepped] | None:
    """The loop that carries ``body``'s state, where nothing but literals stands beside it and its steps are
    counted, and each buffer the step stores at a cell that holds the step."""
    loops = [stmt for stmt in body if not isinstance(stmt, Let)]
    if len(loops) != 1 or not isinstance(loops[0], Loop) or not loops[0].carries or not loops[0].axis.extent.is_static:
        return None
    loop, extents = loops[0], _extents(body)
    if extents is None:
        return None
    found: dict[str, set[tuple[int, int]]] = {}
    for write in body.writes:
        widths = [(at, width) for at, entry in enumerate(write.index) if (width := _width(entry, loop.axis.name, extents))]
        found.setdefault(write.output, set()).add(widths[0] if len(widths) == 1 else (-1, 0))
    return loop, {name: next(iter(places)) for name, places in found.items() if len(places) == 1 and (-1, 0) not in places}


def _one_read(graph: Graph, body: Body, stepped: _Stepped) -> Body:
    """``body`` with each choice among reads of the steps of one buffer as the one read it is
    (:func:`chosen_cell`)."""
    reads = {load.names[0]: load for load in body.loads if plain_load(load)}
    chosen: dict[str, Load] = {}
    for select in body.iter_of_type(Select):
        loads = [reads.get(branch.value) for branch in select.branches]
        if any(load is None for load in loads) or len({load.input for load in loads}) != 1 or loads[0].input not in stepped:
            continue
        cell = chosen_cell(body, select, graph.buffer(loads[0].input).shape)
        if cell is not None:
            chosen[select.name] = Load(names=(select.name,), input=loads[0].input, index=cell)
    if not chosen:
        return body
    body = body.map(lambda stmt: chosen.get(stmt.name, stmt) if isinstance(stmt, Select) else stmt)
    used = body.ssa_uses
    return body.map(lambda stmt: () if isinstance(stmt, Load) and stmt.input in stepped and not set(stmt.names) & used else stmt)


def in_step(graph: Graph, reader: LoopOp, stepped: _Stepped, steps: int) -> tuple[LoopOp, _Stepped] | None:
    """``reader`` as it runs inside the step, and where each buffer it stores holds the step: its outermost loop is
    the loop over the ``steps``, and a trip of it does the trips of ``reader`` that read that step. ``None`` where
    some trip reads two steps, or a step that no coordinate of its own names."""
    body = _one_read(graph, reader.body, stepped)
    extents = _extents(body)
    if extents is None:
        return None
    ctx = SimplifyCtx.empty()
    for name, extent in extents.items():
        ctx = ctx.extend(name, Interval(0, extent - 1), Literal(extent, "int"))
    digits: set[tuple[str, int]] = set()
    for load in body.loads:
        if load.input not in stepped:
            continue
        at, width = stepped[load.input]
        if not plain_load(load) or at >= len(load.index):
            return None
        entry = load.index[at] if width == 1 else BinaryExpression("/", load.index[at], Literal(width, "int"))
        entry = entry.simplify(ctx)
        if isinstance(entry, Var):
            digits.add((entry.name, 1))
        elif isinstance(entry, BinaryExpression) and entry.op == "/" and isinstance(entry.left, Var) and isinstance(entry.right, Literal):
            digits.add((entry.left.name, entry.right.value))
        else:
            return None
    if len(digits) != 1:
        return None
    ((axis, width),) = digits
    # The coordinate is a loop of the reader's own, over every step, and no reduction folds it: a trip reads one step.
    if extents.get(axis) != width * steps or any(axis in accum.axes for accum in body.iter_of_type(Accum)):
        return None
    step = "step"
    while step in body.axis_names or step in body.defined:
        step += "_"
    moved = BinaryExpression("+", BinaryExpression("*", Var(step), Literal(width, "int")), Var(axis)) if width > 1 else Var(step)

    def cut(stmt: Statement) -> Statement | tuple[Statement, ...]:
        if not isinstance(stmt, Loop) or stmt.axis.name != axis:
            return stmt
        inner = Body(tuple(part.substitute({axis: moved}) for part in stmt.body))
        return tuple(inner) if width == 1 else stmt.replace(axis=replace(stmt.axis, extent=Dim(width)), body=inner)

    # Every store sits under the loop that became the step: another would be stored once a step.
    if sum(len(loop.body.writes) for loop in body.iter_of_type(Loop) if loop.axis.name == axis) != len(body.writes):
        return None
    try:
        op = LoopOp(body=Body((Loop(axis=Axis(step, steps), body=body.map(cut)),)))
    except ValueError:
        return None
    loops = [stmt for stmt in op.body if not isinstance(stmt, Let)]
    inner = _extents(op.body)
    if len(loops) != 1 or not isinstance(loops[0], Loop) or inner is None:
        return None
    places: _Stepped = {}
    for write in op.body.writes:
        widths = [(at, found) for at, entry in enumerate(write.index) if (found := _width(entry, loops[0].axis.name, inner))]
        if len(widths) != 1 or places.setdefault(write.output, widths[0]) != widths[0]:
            return None
    return op, places


def _neighbours(graph: Graph, carrying: Node) -> dict[str, LoopOp]:
    """``carrying`` and every kernel that joins its step, each as it is spliced: a kernel only the step reads as it
    is, a reader of the step's stores as it runs inside the step (:func:`in_step`)."""
    members: dict[str, LoopOp] = {carrying.id: carrying.op}
    frame = stepped_stores(carrying.op.body)
    stepped: _Stepped = dict(frame[1]) if frame is not None else {}
    order = graph.topological_order()
    downstream: dict[str, set[str]] = {}
    for nid in reversed(order):
        downstream[nid] = set().union(*({user} | downstream[user] for user in graph.users(nid)))

    # The state starts from a buffer: the kernel that fills it stays one. And a kernel that reads a buffer this one
    # stores over reads it before that store, so it is a kernel before this one.
    seeds = {carry.seed for carry in carrying.op.body.carries if isinstance(carry.seed, str)}
    over = {tensor.alias for tensor in carrying.outputs if tensor.alias is not None}

    def joins(node: Node) -> bool:
        if not isinstance(node.op, LoopOp) or not joins_a_region(node.op) or stops_a_region(graph, node):
            return False
        if seeds & set(node.buffer_names()) or over & set(node.inputs):
            return False
        # No kernel left outside lies between the members: the merged kernel never reads what reads it.
        below = set().union(*(downstream[member] for member in members)) - set(members)
        return not any(graph.producer(name).id in below for name in node.inputs)

    grown = True
    while grown:
        grown = False
        for nid in order:
            node = graph.nodes[nid]
            if nid in members or not joins(node):
                continue
            read = [name for name in node.inputs if graph.producer(name).id in members]
            if graph.users(nid) and graph.users(nid) <= set(members) and not set(node.buffer_names()) & set(graph.outputs):
                members[nid] = node.op
            elif frame is not None and read and all(name in stepped for name in read):
                moved = in_step(graph, node.op, {name: stepped[name] for name in read}, frame[0].axis.extent.as_static())
                if moved is None:
                    continue
                members[nid] = moved[0]
                stepped.update(moved[1])
            else:
                continue
            grown = True
    return members


def _without(index: tuple, at: int) -> tuple:
    return (*index[:at], *index[at + 1 :])


def one_loop(graph: Graph, walk: Node, inner: Node) -> tuple[LoopOp, tuple[str, ...]] | None:
    """PROTOTYPE. ``walk`` and ``inner``, two kernels that each carry a state, as ONE kernel under one loop.

    ``inner`` runs its whole recurrence once per trip of a plain loop of its own, and ``walk`` reads what it stores
    at the trip that is the walk's step. The one loop runs over both step numbers, ``time = steps of inner * step of
    walk + step of inner``: the inner state starts from its seed where its own step is the first, and the walk's
    state moves where it is the last. ``None`` where the two do not stand so."""
    outer, nested = stepped_stores(walk.op.body), stepped_stores(inner.op.body)
    if outer is None or nested is None or not outer[0].axis.extent.is_static:
        return None
    (walking, _), (rows, _) = outer, nested
    stores = {tensor.name for tensor in inner.outputs}
    if any(graph.buffer_users(name) != {walk.id} or name in graph.outputs for name in stores):
        return None
    steps, trips = walking.axis.extent.as_static(), rows.axis.extent.as_static()
    states = {carry.name: carry for carry in inner.op.body.carries}
    if any(isinstance(carry.seed, str) for carry in states.values()):
        return None
    # The loop of ``inner`` that the walk's step is: the one coordinate at which the walk reads its stores.
    writes = {write.output: write for write in inner.op.body.writes}
    trip: set[tuple[str, str, int]] = set()
    for load in walk.op.body.loads:
        if load.input not in stores:
            continue
        write = writes[load.input]
        where = [at for at, entry in enumerate(load.index) if entry == Var(walking.axis.name)]
        if len(where) != 1 or not isinstance(write.index[where[0]], Var):
            return None
        trip.add((write.index[where[0]].name, load.input, where[0]))
    axes = {axis for axis, _, _ in trip}
    if len(axes) != 1:
        return None
    (chunk,) = axes
    at_store = {name: at for _, name, at in trip}
    if set(at_store) != stores:
        return None
    extents = _extents(inner.op.body)
    if extents is None or extents.get(chunk) != steps:
        return None
    time = "time"
    while time in walk.op.body.axis_names | inner.op.body.axis_names or time in walk.op.body.defined | inner.op.body.defined:
        time += "_"
    now, before = Var(time), Var(time) - Literal(1, "int")
    digit = BinaryExpression("%", now, Literal(trips, "int"))
    first = BinaryExpression("==", digit, Literal(0, "int"))
    last = BinaryExpression("==", digit, Literal(trips - 1, "int"))
    at_state = {}
    for name, carry in states.items():
        where = [at for at, entry in enumerate(carry.index) if entry == Var(chunk)]
        if len(where) > 1:
            return None
        at_state[name] = where[0] if where else None

    def cell(name: str, index: tuple) -> tuple:
        return index[1:] if at_state[name] is None else _without(index, at_state[name])[1:]

    def nested_step(stmt: Statement) -> Statement | tuple[Statement, ...]:
        if isinstance(stmt, Loop) and stmt.axis.name == chunk:
            return tuple(stmt.body)
        if isinstance(stmt, Load) and stmt.input in states:
            held, seed = f"{stmt.names[0]}__held", f"{stmt.names[0]}__seed"
            return (
                stmt.replace(names=(held,), index=(before, *cell(stmt.input, stmt.index))),
                Let(name=seed, value=Literal(float(states[stmt.input].seed))),
                Select(name=stmt.names[0], branches=(SelectBranch(seed, first), SelectBranch(held, Literal(True, "bool")))),
            )
        if isinstance(stmt, Carry):
            return stmt.replace(index=(now, *cell(stmt.name, stmt.index)))
        if isinstance(stmt, Write):
            return stmt.replace(output=f"{stmt.output}__now", index=(now, *_without(stmt.index, at_store[stmt.output])))
        return stmt

    moved = {rows.axis.name: digit, chunk: BinaryExpression("/", now, Literal(trips, "int"))}
    inner_body = Body(part.substitute(moved) for part in rows.body.map(nested_step))
    walk_states = {carry.name for carry in walk.op.body.carries}

    def walk_step(stmt: Statement) -> Statement | tuple[Statement, ...]:
        if isinstance(stmt, Load) and stmt.input in walk_states:
            return stmt.replace(index=(before, *stmt.index[1:]))
        if isinstance(stmt, Load) and stmt.input in stores:
            return stmt.replace(input=f"{stmt.input}__now", index=(now, *_without(stmt.index, at_store[stmt.input])))
        if isinstance(stmt, Carry):
            held, kept = f"{stmt.value}__held", f"{stmt.value}__kept"
            return (
                Load(names=(held,), input=stmt.name, index=(before, *stmt.index[1:])),
                Select(name=kept, branches=(SelectBranch(stmt.value, last), SelectBranch(held, Literal(True, "bool")))),
                stmt.replace(value=kept, index=(now, *stmt.index[1:])),
            )
        return stmt

    walk_body = Body(part.substitute({walking.axis.name: moved[chunk]}) for part in walking.body.map(walk_step))
    lets = lambda body: tuple(stmt for stmt in body if isinstance(stmt, Let))  # noqa: E731
    try:
        nested_op = LoopOp(body=Body((*lets(inner.op.body), Loop(axis=Axis(time, steps * trips), body=inner_body))))
        walk_op = LoopOp(body=Body((*lets(walk.op.body), Loop(axis=Axis(time, steps * trips), body=walk_body))))
    except ValueError:
        return None
    live = live_outputs_of(graph, {walk.id, inner.id})
    sub = Graph()
    for name in dict.fromkeys(name for node in (inner, walk) for name in node.inputs if name not in stores):
        sub.add_node(InputOp(), [], graph.buffer(name), node_id=name)
    now_tensors = []
    for tensor in inner.outputs:
        shape = (steps * trips, *_without(tuple(tensor.shape), at_store[tensor.name]))
        now_tensors.append(Tensor(f"{tensor.name}__now", shape, tensor.dtype))
    sub.add_node(nested_op, list(nested_op.inputs), outputs=now_tensors, node_id=now_tensors[0].name)
    sub.add_node(walk_op, list(walk_op.inputs), outputs=walk.outputs, node_id=walk.id)
    sub.outputs = list(live)
    merged = splice_graph(sub)
    return None if merged is None else (merged[0], live)


def rewrite(match: Match, carrying: Node) -> Graph:
    graph = match.graph
    if not carrying.op.body.carries:
        raise RuleSkipped("the kernel carries no state")
    for name in carrying.inputs:
        inner = graph.producer(name)
        if inner is not None and isinstance(inner.op, LoopOp) and inner.op.body.carries and inner.id != carrying.id:
            together = one_loop(graph, carrying, inner)
            if together is not None:
                fragment, output_map = wrap_multi_output_fragment(graph, together[0], together[1])
                match.consumed = {carrying.id, inner.id}
                match.output = together[1][0] if len(together[1]) == 1 else output_map
                return fragment
    members = _neighbours(graph, carrying)
    if len(members) == 1:
        raise RuleSkipped("no kernel beside this one runs in its step")
    live = live_outputs_of(graph, set(members))
    merged = build_merged_region(graph, set(members), live, members)
    if merged is None:
        raise ValueError(f"fusion cannot splice the step of {carrying.id!r} with {sorted(set(members) - {carrying.id})}")
    fragment, output_map = wrap_multi_output_fragment(graph, merged, live)
    match.consumed = set(members)
    match.output = live[0] if len(live) == 1 else output_map
    return fragment
