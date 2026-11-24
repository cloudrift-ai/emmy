"""Merge the kernels around a kernel that carries a state into its step.

``010_merge_loop_ops`` has merged every region of plain loops by now, with a kernel that carries a state between
them. Such a kernel is one step under the loop that carries the state, and a kernel beside it joins that step where
it can run there:

- a kernel only the step reads is computed where the step reads it, like any producer a splice inlines. That holds
  for the kernel that fills the buffer a state starts from: the state reads the buffer on its first step
  (:func:`from_a_read`);
- a kernel that reads what the step stores joins where each of its trips reads one step, the same one for every read:
  its loop over the coordinate the step is a digit of becomes the step and the trips of that step
  (:func:`in_step`), and what it stores it then stores from inside the step;
- a kernel each store of which reads one step by a literal is stored by that step (:func:`at_their_steps`). The state
  the loop ends with is what the last step stores, so its reader is one of these;
- a kernel whose trip reads the state before its step reads the state itself, and where it has a trip after the last
  step the loop runs that trip too, with the states held.

What cannot run in the step stays a kernel: a reader whose one store reads several steps, a reduction over the
steps, a choice among the steps by a range of a coordinate. The merge is the fusion rule's own splice; which of these
kernels are worth keeping apart is a cut's decision, taken on the merged kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expression import BinaryExpression, Expression, Interval, Literal, SimplifyCtx, TernaryExpression, Var, affine_form
from emmy.compiler.ir.loop import LoopOp, splice_graph
from emmy.compiler.ir.stmt import Accum, Assign, Body, Carry, Cond, Let, Load, Loop, Select, SelectBranch, Statement, Write
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.fusion._region import (
    build_merged_region,
    chosen_cell,
    copied,
    joins_a_region,
    live_outputs_of,
    plain_load,
    stops_a_region,
    wrap_multi_output_fragment,
)
from emmy.compiler.pipeline.passes.tile._fromloop import seed_index

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


#: What may stand beside the loop that carries a state: values read or computed once, of no step.
_ONCE = (Let, Load, Assign, Select)


def stepped_stores(body: Body) -> tuple[Loop, _Stepped] | None:
    """The loop that carries ``body``'s state, where nothing but values of no step stands beside it and its steps
    are counted, and each buffer the step stores at a cell that holds the step."""
    loops = [stmt for stmt in body if not isinstance(stmt, _ONCE)]
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


def _one_read(graph: Graph, body: Body, stepped: _Stepped, before: dict[str, str] | None = None) -> Body:
    """``body`` with each choice among reads of the steps of one buffer as the one read it is
    (:func:`chosen_cell`, :func:`_chosen_step`). ``before`` names, for a buffer that holds a state after each step,
    the twin that holds it before each step."""
    reads = {load.names[0]: load for load in body.loads if plain_load(load)}
    chosen: dict[str, Statement | tuple[Statement, ...]] = {}
    for select in body.iter_of_type(Select):
        loads = [reads.get(branch.value) for branch in select.branches]
        whole = not any(load is None for load in loads) and len({load.input for load in loads}) == 1 and loads[0].input in stepped
        cell = chosen_cell(body, select, graph.buffer(loads[0].input).shape) if whole and graph.buffer(loads[0].input) is not None else None
        if cell is not None:
            chosen[select.name] = Load(names=(select.name,), input=loads[0].input, index=cell)
        elif (found := _chosen_step(body, select, stepped, before or {})) is not None:
            chosen[select.name] = found
    if not chosen:
        return body
    body = body.map(lambda stmt: chosen.get(stmt.name, stmt) if isinstance(stmt, Select) else stmt)
    used = body.ssa_uses
    return body.map(lambda stmt: () if isinstance(stmt, Load) and stmt.input in stepped and not set(stmt.names) & used else stmt)


def _chosen_step(body: Body, select: Select, stepped: _Stepped, before: dict[str, str]) -> tuple[Statement, ...] | None:
    """PROTOTYPE. ``select`` where some of its branches are reads of one buffer at the steps, each chosen where one
    coordinate equals its step, or its step plus one: those branches as ONE read, at the coordinate. A read one step
    back is a read of the state before the step, taken from the buffer's twin (``before``). The other branches stay
    — the trip before the first step, the trip after the last."""
    reads = {load.names[0]: load for load in body.loads if plain_load(load)}
    picked = [(at, reads[branch.value]) for at, branch in enumerate(select.branches) if branch.value in reads]
    picked = [(at, load) for at, load in picked if load.input in stepped]
    extents = _extents(body)
    if len(picked) < 2 or len({load.input for _, load in picked}) != 1 or extents is None:
        return None
    source = picked[0][1].input
    position, width = stepped[source]
    if width != 1 or not all(isinstance(load.index[position], Literal) for _, load in picked):
        return None
    # Every branch but the last is chosen where one coordinate equals a literal; the last takes the trip left over.
    where: list[int] = []
    along: set[Expression] = set()
    for branch in select.branches[:-1]:
        test = branch.select
        if not isinstance(test, BinaryExpression) or test.op != "==":
            return None
        literal, other = (test.left, test.right) if isinstance(test.left, Literal) else (test.right, test.left)
        if not isinstance(literal, Literal):
            return None
        where.append(literal.value)
        along.add(other)
    if select.branches[-1].select != _TRUE or len(along) != 1:
        return None
    (coordinate,) = along
    if not isinstance(coordinate, Var) or coordinate.name not in extents:
        return None
    left = set(range(extents[coordinate.name])) - set(where)
    if len(left) != 1:
        return None
    where.append(left.pop())
    offsets = {where[at] - load.index[position].value for at, load in picked}
    if len(offsets) != 1 or (offset := offsets.pop()) not in (0, 1) or (offset == 1 and source not in before):
        return None
    # The cell each branch reads where it is chosen is one cell, the step apart.
    ctx = SimplifyCtx.empty()
    cells = {
        tuple(entry.substitute({coordinate.name: Literal(where[at], "int")}).simplify(ctx) for entry in load.index)[:position]
        + tuple(entry.substitute({coordinate.name: Literal(where[at], "int")}).simplify(ctx) for entry in load.index)[position + 1 :]
        for at, load in picked
    }
    if len(cells) != 1:
        return None
    (cell,) = cells
    one = f"{select.name}__step"
    read = Load(names=(one,), input=before[source] if offset else source, index=(*cell[:position], coordinate, *cell[position:]))
    taken = {at for at, _ in picked}
    branches = tuple(SelectBranch(one, branch.select) if at in taken else branch for at, branch in enumerate(select.branches))
    return read, select.replace(branches=branches)


_TRUE = Literal(True, "bool")


@dataclass(frozen=True)
class Place:
    """PROTOTYPE. Where a kernel stands on the ONE loop it shares with another: an affine map from its own
    coordinates to that loop's ``time``,

        ``time = stride * over + under + shift``,

    with ``over`` and ``under`` loops of the kernel (either may be absent). ``fine`` trips of ``over`` pass in one
    time where the map runs the other way, ``over = fine * time + rest``. Everything a kernel does differently on
    the shared loop follows from the map (:func:`_schedule`)."""

    over: str | None = None
    stride: int = 1
    under: str | None = None
    shift: int = 0
    fine: int = 1


@dataclass(frozen=True)
class Alignment:
    """How a reader and the kernel it reads share one loop of ``extent`` trips: each one's :class:`Place` (``None``
    for a producer whose own steps are the loop), the stepped stores the reader takes at this time (``now``: the
    position that held the step), and the stores it takes as they stand at this time through a twin that holds
    every time (``twins``: the position that held the producer's pinned loop)."""

    extent: int
    producer: Place | None
    reader: Place
    now: dict[str, int]
    twins: dict[str, int]


def _fresh(name: str, *bodies: Body) -> str:
    while any(name in body.axis_names or name in body.defined for body in bodies):
        name += "_"
    return name


def _all(*tests: Expression | None) -> Expression | None:
    held = [test for test in tests if test is not None]
    return None if not held else held[0] if len(held) == 1 else BinaryExpression("&&", held[0], _all(*held[1:]))


def _schedule(place: Place, own: str | None, trips: int, time: str, extent: int):
    """What the map of ``place`` means for a kernel whose loop of steps is ``own`` (``trips`` long), on a loop over
    ``time`` of ``extent``: ``(step, clamp, active, restart, pins)``. ``step`` is the kernel's step at a time and
    ``clamp`` says it is one only where ``active`` holds; ``active`` is where the kernel has a step at all (``None``:
    everywhere); ``restart`` is where its states start again from their seeds; ``pins`` are the plain loops that
    follow the time, each with its value and the trips it keeps."""
    here = Var(time)
    if place.fine > 1:
        return None, False, None, None, {place.over: (here, place.fine)}
    lead = here - Literal(place.shift, "int") if place.shift else here
    stride = Literal(place.stride, "int")
    # Where the shift is less than a stride, the high part is the time's own, and it is a step at every time.
    high = BinaryExpression("/", here if 0 <= place.shift < place.stride else lead, stride) if place.stride > 1 else lead
    value = {place.over: high, place.under: BinaryExpression("%", lead, stride)}
    pins = {axis: (value[axis], 1) for axis in (place.over, place.under) if axis is not None and axis != own}
    if own is None:
        return None, False, None, None, pins
    if own == place.under:
        # The step is the low part of the time: it runs through, and starts again each time the high part moves.
        restart = BinaryExpression("==", value[own], Literal(0, "int")) if place.over is not None else None
        return value[own], False, None, restart, pins
    # The step is the high part: the kernel has a step once every ``stride`` times, from ``shift`` on.
    last = place.shift + place.stride * (trips - 1)
    active = _all(
        BinaryExpression("==", BinaryExpression("%", here, stride), Literal(place.shift % place.stride, "int"))
        if place.stride > 1
        else None,
        BinaryExpression("<=", Literal(place.shift, "int"), here) if place.shift >= place.stride else None,
        BinaryExpression("<", here, Literal(last + 1, "int")) if last + 1 < extent else None,
    )
    return value[own], not 0 <= place.shift < place.stride or last + 1 < extent, active, None, pins


def relate(producer: Body | None, reader: Body, stepped: _Stepped, steps: int, time: str) -> Alignment | None:
    """The alignment of ``reader`` with what it reads: the index equations of its reads, solved for a map of each
    kernel's coordinates to one time (:class:`Place`). ``None`` where they do not solve so.

    A read says where the reader's coordinates stand to the producer's: the index of the read equals the index of
    the store. ``stepped`` names the stores that hold the step of the producer's loop of ``steps``, and where.

    - A store that holds the producer's step, read at ``v + k`` for the reader's own step ``v``: the two steps are
      one position, ``k`` apart. ``time = producer's step = reader's step + k``.
    - Such a store read at ``v / c`` for a plain loop ``v`` of the reader: ``c`` trips of that loop are one step.
      ``time = producer's step = v / c``.
    - A store that holds no step, read at the reader's own step in the place of a plain loop of the producer: the
      producer's whole loop runs before each step of the reader.
      ``time = producer's steps * that loop + producer's step = producer's steps * reader's step + the last``."""
    extents = _extents(reader)
    if extents is None:
        return None
    carried = [stmt for stmt in reader if isinstance(stmt, Loop) and stmt.carries]
    own = carried[0].axis.name if len(carried) == 1 else None
    ctx = SimplifyCtx.empty()
    for name, extent in extents.items():
        ctx = ctx.extend(name, Interval(0, extent - 1), Literal(extent, "int"))
    writes = {} if producer is None else {write.output: write for write in producer.writes}
    # The equations. ``steps_at``: the producer's step equals ``v / c + k``. ``loops_at``: a plain loop of the
    # producer equals the reader's own step.
    steps_at: set[tuple[str, int, int]] = set()
    loops_at: set[str] = set()
    now: dict[str, int] = {}
    twins: dict[str, int] = {}
    for load in reader.loads:
        if load.input in stepped:
            at, width = stepped[load.input]
            if not plain_load(load) or at >= len(load.index):
                return None
            entry = (load.index[at] if width == 1 else BinaryExpression("/", load.index[at], Literal(width, "int"))).simplify(ctx)
            form = affine_form(entry, entry.free_vars())
            if isinstance(entry, BinaryExpression) and entry.op == "/" and isinstance(entry.left, Var) and isinstance(entry.right, Literal):
                steps_at.add((entry.left.name, entry.right.value, 0))
            elif form is not None and len(form[1]) == 1 and next(iter(form[1].values())) == 1:
                try:
                    steps_at.add((next(iter(form[1])), 1, form[0].eval({})))
                except (KeyError, TypeError, ValueError):
                    return None
            else:
                return None
            if width == 1:
                now[load.input] = at
        elif load.input in writes and own is not None:
            where = [at for at, entry in enumerate(load.index) if entry == Var(own)]
            if len(where) != 1 or not isinstance(writes[load.input].index[where[0]], Var):
                return None
            loops_at.add(writes[load.input].index[where[0]].name)
            twins[load.input] = where[0]
    if len(steps_at) == 1 and not loops_at:
        ((axis, fine, offset),) = steps_at
        if axis == own and fine == 1 and 0 <= offset and offset + extents[axis] <= steps:
            return Alignment(steps, None, Place(over=own, shift=offset), now, {})
        folded = any(axis in accum.axes for accum in reader.iter_of_type(Accum))
        under = sum(len(loop.body.writes) for loop in reader.iter_of_type(Loop) if loop.axis.name == axis)
        # A plain loop follows the time where a trip of it reads one step, no reduction folds it, and every store
        # sits under it: another would be stored once a step.
        if own is None and not offset and extents.get(axis) == fine * steps and not folded and under == len(reader.writes):
            return Alignment(steps, None, Place(over=axis, fine=fine), {} if fine > 1 else now, {})
        return None
    if steps_at or len(loops_at) != 1 or producer is None:
        return None
    rows = [stmt for stmt in producer if isinstance(stmt, Loop) and stmt.carries]
    inner = _extents(producer)
    (chunk,) = loops_at
    if len(rows) != 1 or inner is None or set(twins) != set(writes) or inner.get(chunk) != extents[own]:
        return None
    if any(isinstance(carry.seed, str) for carry in producer.carries):
        return None
    trips = rows[0].axis.extent.as_static()
    inside = Place(over=chunk, stride=trips, under=rows[0].axis.name)
    return Alignment(trips * extents[own], inside, Place(over=own, stride=trips, shift=trips - 1), {}, twins)


def placed(op: LoopOp, time: str, extent: int, place: Place, now: dict[str, int], twins: dict[str, int], stores: bool) -> LoopOp | None:
    """``op`` on the one loop over ``time``, at its :class:`Place`: its states read one time back and held where it
    has no step, its stores kept to its own steps, its pinned loops following the time. ``stores`` says the
    ``twins`` are its own stores (it writes the twin) and not stores it reads (it reads the twin)."""
    body = op.body
    carried = [stmt for stmt in body if isinstance(stmt, Loop) and stmt.carries]
    if len(carried) > 1:
        return None
    # What stands beside the loop of steps stays beside the one loop: it is of no step.
    lets = tuple(stmt for stmt in body if stmt is not carried[0]) if carried else tuple(stmt for stmt in body if isinstance(stmt, Let))
    inner = carried[0].body if carried else Body(stmt for stmt in body if not isinstance(stmt, Let))
    own = carried[0].axis.name if carried else None
    trips = carried[0].axis.extent.as_static() if carried else 0
    step, clamp, active, restart, pins = _schedule(place, own, trips, time, extent)
    here, before = Var(time), Var(time) - Literal(1, "int")
    states = {carry.name: carry for carry in body.carries}
    gone = {axis for axis, (_, kept) in pins.items() if kept == 1}

    def cell(name: str, index: tuple) -> tuple:
        held = states[name].index
        return tuple(entry for at, entry in enumerate(index) if at and not (isinstance(held[at], Var) and held[at].name in gone))

    def without(index: tuple, at: int) -> tuple:
        return (here, *index[:at], *index[at + 1 :])

    def move(stmt: Statement) -> Statement | tuple[Statement, ...]:
        if isinstance(stmt, Load) and stmt.input in states:
            read = stmt.replace(index=(before, *cell(stmt.input, stmt.index)))
            if restart is None:
                return read
            held, seed = f"{stmt.names[0]}__held", f"{stmt.names[0]}__seed"
            return (
                read.replace(names=(held,)),
                Let(name=seed, value=Literal(float(states[stmt.input].seed))),
                Select(name=stmt.names[0], branches=(SelectBranch(seed, restart), SelectBranch(held, _TRUE))),
            )
        if isinstance(stmt, Load) and stmt.input in now:
            at = now[stmt.input]
            return stmt.replace(index=(*stmt.index[:at], here, *stmt.index[at + 1 :]))
        if isinstance(stmt, Load) and stmt.input in twins and not stores:
            return stmt.replace(input=f"{stmt.input}__now", index=without(stmt.index, twins[stmt.input]))
        if isinstance(stmt, Carry):
            index = (here, *cell(stmt.name, stmt.index))
            if active is None:
                return stmt.replace(index=index)
            held, kept = f"{stmt.value}__held", f"{stmt.value}__kept"
            return (
                Load(names=(held,), input=stmt.name, index=(before, *index[1:])),
                Select(name=kept, branches=(SelectBranch(stmt.value, active), SelectBranch(held, _TRUE))),
                stmt.replace(value=kept, index=index),
            )
        if isinstance(stmt, Write) and stmt.output in twins and stores:
            return stmt.replace(output=f"{stmt.output}__now", index=without(stmt.index, twins[stmt.output]))
        if isinstance(stmt, Write) and active is not None:
            # Under the test the step is the kernel's own: the store names it as it is, with no clamp.
            return Cond(cond=active, body=Body((stmt.substitute({own: step}) if clamp else stmt,)))
        return stmt

    def follow(stmt: Statement) -> Statement | tuple[Statement, ...]:
        if not isinstance(stmt, Loop) or stmt.axis.name not in pins:
            return stmt
        value, kept = pins[stmt.axis.name]
        if kept == 1:
            return tuple(part.substitute({stmt.axis.name: value}) for part in stmt.body)
        moved = BinaryExpression("+", BinaryExpression("*", value, Literal(kept, "int")), Var(stmt.axis.name))
        inside = Body(tuple(part.substitute({stmt.axis.name: moved}) for part in stmt.body))
        return stmt.replace(axis=replace(stmt.axis, extent=Dim(kept)), body=inside)

    stepped_body = inner.map(move).map(follow)
    if step is not None:
        at_step = TernaryExpression(active, step, Literal(0, "int")) if clamp and active is not None else step
        stepped_body = Body(part.substitute({own: at_step}) for part in stepped_body)
    try:
        return LoopOp(body=Body((*lets, Loop(axis=Axis(time, extent), body=stepped_body))))
    except ValueError:
        return None


def _places(op: LoopOp) -> _Stepped | None:
    """Where each buffer ``op`` stores holds the step of its one outer loop, or ``None`` where one holds none."""
    loops = [stmt for stmt in op.body if not isinstance(stmt, _ONCE)]
    inner = _extents(op.body)
    if len(loops) != 1 or not isinstance(loops[0], Loop) or inner is None:
        return None
    places: _Stepped = {}
    for write in op.body.writes:
        widths = [(at, found) for at, entry in enumerate(write.index) if (found := _width(entry, loops[0].axis.name, inner))]
        if len(widths) != 1 or places.setdefault(write.output, widths[0]) != widths[0]:
            return None
    return places


def at_their_steps(body: Body, stepped: _Stepped, steps: int, time: str) -> LoopOp | None:
    """PROTOTYPE. A plain kernel each store of which reads the steps at one literal, on the loop over the steps: a
    store is made at the time it reads, from what the step holds then. ``None`` where a store reads two steps, or
    none, or a step that is a digit of a wider coordinate."""
    here = Var(time)

    def read_at(load: Load) -> int | None:
        at, width = stepped[load.input]
        entry = load.index[at] if plain_load(load) and at < len(load.index) else None
        return entry.value if width == 1 and isinstance(entry, Literal) else None

    when: dict[Write, int] = {}
    for write in body.writes:
        alone = body.map(lambda stmt, write=write: () if isinstance(stmt, Write) and stmt != write else stmt).pruned(write.values)
        times = {read_at(load) for load in alone.loads if load.input in stepped}
        if len(times) != 1 or None in times:
            return None
        when[write] = times.pop()

    def move(stmt: Statement) -> Statement:
        if isinstance(stmt, Load) and stmt.input in stepped:
            at = stepped[stmt.input][0]
            return stmt.replace(index=(*stmt.index[:at], here, *stmt.index[at + 1 :]))
        if isinstance(stmt, Write):
            return Cond(cond=BinaryExpression("==", here, Literal(when[stmt], "int")), body=Body((stmt,)))
        return stmt

    lets = tuple(stmt for stmt in body if isinstance(stmt, Let))
    rest = Body(stmt for stmt in body if not isinstance(stmt, Let)).map(move)
    try:
        return LoopOp(body=Body((*lets, Loop(axis=Axis(time, steps), body=rest))))
    except ValueError:
        return None


def in_step(graph: Graph, reader: LoopOp, stepped: _Stepped, steps: int) -> tuple[LoopOp, _Stepped] | None:
    """A plain ``reader`` as it runs inside the step, and where each buffer it stores holds the step
    (:func:`relate`, :func:`placed`): its trips follow the steps where each reads one, and where its stores each
    read one step by a literal they are made at that step (:func:`at_their_steps`). ``None`` where neither holds."""
    body = _one_read(graph, reader.body, stepped)
    time = _fresh("step", body)
    alignment = relate(None, body, stepped, steps, time)
    if alignment is None:
        once = at_their_steps(body, stepped, steps, time)
        return None if once is None else (once, {})
    op = placed(replace(reader, body=body, inputs={}), time, alignment.extent, alignment.reader, alignment.now, {}, False)
    places = None if op is None else _places(op)
    return None if places is None else (op, places)


def from_a_read(graph: Graph, op: LoopOp, seeds: set[str]) -> LoopOp:
    """PROTOTYPE. ``op`` with each state that starts from one of ``seeds`` reading that buffer on its first step, and
    starting from zero. The kernel that fills the buffer is then a kernel the step reads, like any other."""
    starts = {carry.name: carry for carry in op.body.carries if carry.seed in seeds}

    def move(stmt: Statement) -> Statement | tuple[Statement, ...]:
        if isinstance(stmt, Carry) and stmt.name in starts:
            return stmt.replace(seed=0.0)
        if not isinstance(stmt, Load) or stmt.input not in starts:
            return stmt
        carry = starts[stmt.input]
        held, first = f"{stmt.names[0]}__held", f"{stmt.names[0]}__first"
        at = seed_index(stmt.index[1:], graph.buffer(carry.seed).shape)
        on_first = BinaryExpression("==", carry.index[0], Literal(0, "int"))
        return (
            stmt.replace(names=(held,)),
            Load(names=(first,), input=carry.seed, index=at),
            Select(name=stmt.names[0], branches=(SelectBranch(first, on_first), SelectBranch(held, _TRUE))),
        )

    return replace(op, body=op.body.map(move), inputs={}) if starts else op


def _neighbours(graph: Graph, carrying: Node) -> tuple[dict[str, LoopOp], set[str], dict[str, tuple[Tensor, ...]]]:
    """``carrying`` and every kernel that joins its step, each as it is spliced: a kernel the step reads as it is, a
    reader of the step's stores as it runs inside the step (:func:`in_step`). And those of them that stay kernels
    of their own beside the step, for whoever else reads them. And the twins each member stores beside its own
    buffers."""
    members: dict[str, LoopOp] = {carrying.id: carrying.op}
    kept: set[str] = set()
    frame = stepped_stores(carrying.op.body)
    stepped: _Stepped = dict(frame[1]) if frame is not None else {}
    last = Literal(frame[0].axis.extent.as_static() - 1, "int") if frame is not None else None
    # PROTOTYPE. A buffer the step stores at a cell that holds no step holds, after the loop, what the last step
    # stored. Its twin holds every step's, so a reader of the buffer is a reader of the last step.
    standing = _standing(carrying, stepped) if frame is not None else {}
    stepped.update({twin.name: (0, 1) for twin in standing.values()})
    # PROTOTYPE. A buffer that holds a state after each step: its twin holds the state before each step, the seed
    # before the first and the state the loop ends with after the last. A kernel whose trip reads the step before
    # reads the twin at its own step, and where it has a trip after the last step the loop runs that trip too.
    states = _state_stores(carrying.op.body, stepped) if frame is not None else {}
    before = {name: f"{name}__before" for name in states}
    extent = frame[0].axis.extent.as_static() if frame is not None else 0
    for reader in sorted(set().union(*(graph.buffer_users(name) for name in states)) if states else ()):
        if isinstance(graph.nodes[reader].op, LoopOp):
            extent = max(extent, _reach(_one_read(graph, graph.nodes[reader].op.body, stepped, before), set(before.values())))
    twins = (
        *standing.values(),
        *(Tensor(before[name], (Dim(extent), *graph.buffer(name).shape[1:]), graph.buffer(name).dtype) for name in states),
    )
    stepped.update({name: (0, 1) for name in before.values()})
    order = graph.topological_order()
    downstream: dict[str, set[str]] = {}
    for nid in reversed(order):
        downstream[nid] = set().union(*({user} | downstream[user] for user in graph.users(nid)))
    # A kernel that carries a state and comes after this one takes it into its own loop first (:func:`together`):
    # what both of them read is then read by one kernel, and joins it whole.
    nested = any(isinstance(later := graph.nodes[nid].op, LoopOp) and later.body.carries for nid in downstream[carrying.id])

    # A state that starts from a buffer a plain kernel fills reads it on its first step (:func:`from_a_read`). And a
    # kernel that reads a buffer this one stores over reads it before that store, so it is a kernel before this one.
    seeds = {carry.seed for carry in carrying.op.body.carries if isinstance(carry.seed, str)}
    filled = {name for name in seeds if isinstance(graph.producer(name).op, LoopOp) and joins_a_region(graph.producer(name).op)}
    members[carrying.id] = _with_twins(from_a_read(graph, carrying.op, filled), standing)
    if frame is not None and extent > frame[0].axis.extent.as_static():
        window = _fresh("time", carrying.op.body)
        members[carrying.id] = placed(members[carrying.id], window, extent, Place(over=frame[0].axis.name), {}, {}, False)
    if states:
        members[carrying.id] = _with_states_before(members[carrying.id], {state: before[name] for name, state in states.items()})
    seeds -= filled
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
            elif not read and not nested and graph.users(nid) & set(members):
                # PROTOTYPE. A plain kernel the step reads beside other readers is computed in the step too, where
                # the step reads it, and stays a kernel for the others. Storing it once for all of them is memory
                # taken for recomputation: a cut's choice, on the merged kernel.
                members[nid] = node.op
                kept.add(nid)
            elif frame is not None and read and all(name in stepped or name in standing for name in read):
                if any(name in standing for name in read) and copied(graph, node.outputs[0].name) is not None:
                    continue  # a copy of what the last step stored: the step stores it there (``020``)
                body = node.op.body.map(
                    lambda stmt: (
                        stmt.replace(input=standing[stmt.input].name, index=(last, *stmt.index))
                        if isinstance(stmt, Load) and stmt.input in standing
                        else stmt
                    )
                )
                body = _one_read(graph, body, stepped, before)
                reads = {load.input: stepped[load.input] for load in body.loads if load.input in stepped}
                moved = in_step(graph, replace(node.op, body=body, inputs={}), reads, extent)
                if moved is None:
                    continue
                members[nid] = moved[0]
                stepped.update(moved[1])
            else:
                continue
            grown = True
    return members, kept, {carrying.id: twins if frame is not None else ()}


def _state_stores(body: Body, stepped: _Stepped) -> dict[str, str]:
    """Each buffer that holds a state after each step, to the state: the step stores the state's own value, at the
    state's own cell."""
    carried = {(carry.value, tuple(carry.index)): carry.name for carry in body.carries}
    return {
        write.output: carried[write.values[0], tuple(write.index)]
        for write in body.writes
        if stepped.get(write.output) == (0, 1) and len(write.values) == 1 and (write.values[0], tuple(write.index)) in carried
    }


def _reach(body: Body, twins: set[str]) -> int:
    """How many trips the loop a kernel reads the states before each step along runs: one past the last step where
    the kernel has a trip after it."""
    extents = _extents(body) or {}
    along = {load.index[0].name for load in body.loads if load.input in twins and isinstance(load.index[0], Var)}
    return max((extents.get(name, 0) for name in along), default=0)


def _with_states_before(op: LoopOp, twins: dict[str, str]) -> LoopOp:
    """``op`` storing, on every trip of its loop, each state as it stands before the trip's step."""

    def both(stmt: Statement) -> Statement | tuple[Statement, ...]:
        if not isinstance(stmt, Carry) or stmt.name not in twins:
            return stmt
        stood = f"{stmt.value}__stood"
        read = Load(names=(stood,), input=stmt.name, index=(stmt.index[0] - Literal(1, "int"), *stmt.index[1:]))
        return (read, Write(output=twins[stmt.name], index=stmt.index, values=(stood,)), stmt)

    return replace(op, body=op.body.map(both), inputs={})


def _standing(carrying: Node, stepped: _Stepped) -> dict[str, Tensor]:
    """The twin of each buffer ``carrying`` stores on every step at a cell that holds no step: the buffer with the
    step before its cell."""
    body = carrying.op.body
    steps = next(stmt for stmt in body if isinstance(stmt, Loop) and stmt.carries).axis.extent
    sometimes = {write.output for cond in body.iter_of_type(Cond) for write in cond.body.writes}
    return {
        tensor.name: Tensor(f"{tensor.name}__now", (steps, *tensor.shape), tensor.dtype)
        for tensor in carrying.outputs
        if tensor.name not in stepped and tensor.name not in sometimes and tensor.alias is None
    }


def _with_twins(op: LoopOp, standing: dict[str, Tensor]) -> LoopOp:
    """``op`` storing the twin of each standing buffer beside the buffer."""
    (step,) = (stmt.axis.name for stmt in op.body if isinstance(stmt, Loop) and stmt.carries)

    def both(stmt: Statement) -> Statement | tuple[Statement, ...]:
        if not isinstance(stmt, Write) or stmt.output not in standing:
            return stmt
        return (stmt, stmt.replace(output=standing[stmt.output].name, index=(Var(step), *stmt.index)))

    return replace(op, body=op.body.map(both), inputs={}) if standing else op


def together(graph: Graph, walk: Node, inner: Node) -> tuple[LoopOp, tuple[str, ...]] | None:
    """PROTOTYPE. ``walk`` and ``inner``, a kernel that carries a state and one it reads that carries one too, as ONE
    kernel under one loop (:func:`relate`): side by side where the walk reads the other's steps an offset apart, the
    other's whole recurrence inside each step of the walk where it reads its result at its own step."""
    frame = stepped_stores(inner.op.body)
    if frame is None or stepped_stores(walk.op.body) is None:
        return None
    stores = {tensor.name for tensor in inner.outputs}
    time = _fresh("time", walk.op.body, inner.op.body)
    stepped = {name: place for name, place in frame[1].items() if name in stores}
    alignment = relate(inner.op.body, walk.op.body, stepped, frame[0].axis.extent.as_static(), time)
    if alignment is None:
        return None
    if alignment.twins and any(graph.buffer_users(name) != {walk.id} or name in graph.outputs for name in stores):
        return None
    reader = placed(walk.op, time, alignment.extent, alignment.reader, alignment.now, alignment.twins, False)
    producer = inner.op
    if alignment.producer is not None:
        producer = placed(inner.op, time, alignment.extent, alignment.producer, {}, alignment.twins, True)
    if reader is None or producer is None:
        return None
    live = live_outputs_of(graph, {walk.id, inner.id})
    sub = Graph()
    for name in dict.fromkeys(name for node in (inner, walk) for name in node.inputs if name not in stores):
        sub.add_node(InputOp(), [], graph.buffer(name), node_id=name)
    outputs = []
    for tensor in inner.outputs:
        at = alignment.twins.get(tensor.name)
        shape = tensor.shape if at is None else (alignment.extent, *tensor.shape[:at], *tensor.shape[at + 1 :])
        outputs.append(tensor if at is None else Tensor(f"{tensor.name}__now", tuple(shape), tensor.dtype))
    sub.add_node(producer, list(producer.inputs), outputs=outputs, node_id=outputs[0].name if alignment.twins else inner.id)
    sub.add_node(reader, list(reader.inputs), outputs=walk.outputs, node_id=walk.id)
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
            merged = together(graph, carrying, inner)
            if merged is not None:
                fragment, output_map = wrap_multi_output_fragment(graph, merged[0], merged[1])
                match.consumed = {carrying.id, inner.id}
                match.output = merged[1][0] if len(merged[1]) == 1 else output_map
                return fragment
    members, kept, twins = _neighbours(graph, carrying)
    if len(members) == 1:
        raise RuleSkipped("no kernel beside this one runs in its step")
    live = tuple(name for name in live_outputs_of(graph, set(members)) if graph.producer(name).id not in kept)
    merged = build_merged_region(graph, set(members), live, members, twins)
    if merged is None:
        raise ValueError(f"fusion cannot splice the step of {carrying.id!r} with {sorted(set(members) - {carrying.id})}")
    fragment, output_map = wrap_multi_output_fragment(graph, merged, live)
    match.consumed = set(members) - kept
    match.output = live[0] if len(live) == 1 else output_map
    return fragment
