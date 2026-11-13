"""Unroll loops of a kernel body: one copy of a loop's body per trip, the axis bound to the trip's value.

The one unroll of the kernel lowering. ``047_reuse_lane_loads`` calls it for the short lane loops whose loads it wants
in registers, ``117_unroll_loops`` for every loop a lowering site flagged. The body's optimization does the rest: it
folds each copy's index and computes once what two copies share.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from itertools import chain, count

from emmy.compiler.ir.expression import BinaryExpression, Expression, Interval, Literal, SimplifyCtx, Var, apply_binop
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Body, Carry, Cond, Init, Loop, Statement, StridedLoop


def trip_counts(loop: Statement, ctx: SimplifyCtx) -> tuple[int, int] | None:
    """The trips of ``loop`` that every thread runs and the most any thread runs, or ``None`` when no static number
    bounds them.

    A ``Loop`` over a static extent runs that many on every thread. A ``StridedLoop`` over a static extent with a
    static step and no early stop runs the trips its start leaves: the two counts are those of its largest and its
    smallest start under the ranges of ``ctx``. They are equal for a start of zero and for a lane index under a step
    the extent divides, and differ where the lanes share a row their count does not divide.
    """
    if any(isinstance(stmt, Carry) for stmt in loop.body) or not loop.axis.extent.is_static:
        return None
    extent = loop.axis.extent.as_static()
    if isinstance(loop, Loop):
        return extent, extent
    step = loop.step.value if isinstance(loop.step, Literal) else None
    if loop.end is not None or not isinstance(step, int) or step < 1:
        return None
    start = loop.start.range(ctx)
    if start is None:
        return None
    every, most = (max(0, -(-(extent - at) // step)) for at in (start.hi, start.lo))
    return every, most


def runtime_bound(loop: Statement) -> bool:
    """Whether only the launch knows how many trips ``loop`` runs: a symbolic extent, no early stop, a constant step
    and nothing carried. Such a loop takes its copies with a remainder (:func:`_runtime`)."""
    if any(isinstance(stmt, Carry) for stmt in loop.body) or loop.axis.extent.is_static:
        return False
    if isinstance(loop, Loop):
        return True
    step = loop.step.value if isinstance(loop.step, Literal) else None
    return loop.end is None and isinstance(step, int) and step >= 1


#: What ``copies`` reads and answers: a loop and its :func:`trip_counts` (``None`` for a :func:`runtime_bound` loop),
#: and the copies the loop takes, if any.
Copies = Callable[[Statement, tuple[int, int] | None], int | None]


def unroll_where(body: Body, copies: Copies) -> Body:
    """``body`` with every loop ``copies`` gives a number unrolled into that many copies of its body, innermost
    first. ``copies`` reads a loop and its :func:`trip_counts`, or ``None`` for a :func:`runtime_bound` loop; a number
    below the loop's trips keeps the loop, each of its trips running that many copies, and a runtime-bound loop keeps
    it with a remainder after it (:func:`_runtime`)."""
    return _walk(body, copies, SimplifyCtx.empty(), frozenset())


def _walk(body: Body, copies: Copies, ctx: SimplifyCtx, outer: frozenset[str]) -> Body:
    outer = outer | {name for stmt in body for name in stmt.defines()}
    out: list[Statement] = []
    for stmt in body:
        looping = isinstance(stmt, (Loop, StridedLoop))
        inner = ctx
        if isinstance(stmt, Tile):
            for axis in stmt.axes:
                if axis.extent.is_static:
                    inner = inner.extend(axis.name, Interval(0, axis.extent.as_static() - 1))
        elif looping and stmt.axis.extent.is_static:
            start = stmt.start.range(ctx) if isinstance(stmt, StridedLoop) else Interval(0, 0)
            if start is not None:
                inner = inner.extend(stmt.axis.name, Interval(start.lo, stmt.axis.extent.as_static() - 1))
        nested = stmt.nested()
        if nested:
            walked = stmt.with_bodies(tuple(_walk(b, copies, inner, outer) for b in nested))
            if looping and stmt.seed and _folds(walked) != _folds(stmt):
                # A loop seeds the folds that stand in its body, and the copies of a loop unrolled into it brought
                # theirs: it seeds its own ahead of itself, as a statement, and the copies keep the seed they had.
                out.extend(_seeds(stmt))
                walked = walked.replace(seed=False)
            stmt = walked
        counts = trip_counts(stmt, ctx) if looping else None
        n = copies(stmt, counts) if counts is not None or (looping and runtime_bound(stmt)) else None
        if n is None or (counts is None and n < 2):
            out.append(stmt)
        elif counts is None:
            out.extend(_runtime(stmt, n, outer))
        elif n < counts[1]:
            out.append(_stepped(stmt, n, outer))
        else:
            out.extend(_unroll(stmt, n, ctx, outer, {name for s in out for name in s.defines()}))
    return Body(out)


def _folds(loop: Statement) -> dict[str, Accum]:
    """The folds that stand in ``loop``'s body, the first of each carrier: the ones the loop seeds when it seeds."""
    folds: dict[str, Accum] = {}
    for stmt in loop.body:
        if isinstance(stmt, Accum):
            folds.setdefault(stmt.name, stmt)
    return folds


def _carried(loop: Statement) -> set[str]:
    """The carriers ``loop`` folds from trip to trip: those of its folds that no statement of its body seeds again.
    A carrier seeded in the body is each trip's own, and so each copy's."""
    return set(_folds(loop)) - {stmt.name for stmt in loop.body if isinstance(stmt, Init)}


def _seeds(loop: Statement) -> list[Statement]:
    """The seed of each carrier ``loop`` seeds, as the statement that stands ahead of it."""
    return [Init(name, fold.op.identity, dtype=fold.dtype) for name, fold in _folds(loop).items()] if loop.seed else []


def _unroll(loop: Statement, trips: int, ctx: SimplifyCtx, outer: Collection[str], taken: Collection[str]) -> list[Statement]:
    """``loop`` as ``trips`` copies of its body, each with the axis bound to its trip's value.

    A carrier the loop seeds is seeded once ahead of the copies, as the loop would have seeded it. A trip some
    threads do not run keeps the loop's bound as its guard, and a guard the trip's value decides under the ranges of
    ``ctx`` is dropped: the copy holds the branch taken. A name the body defines is each copy's own
    (``<name>__u<trip>``, and a number after it where ``taken``, the names its scope already holds, has that one: two
    loops of one scope may define the same names, each in its own). The exception is a name that stands outside the
    loop: a carrier, or storage a statement of ``outer`` (the names defined around the loop) declared and the body
    updates in place.
    """
    out = _seeds(loop)
    own = _own(loop, outer) - {seed.name for seed in out}
    tags = chain(("",), (f"_{n}" for n in count(1)))
    tag = next(t for t in tags if not any(f"{name}__u{trip}{t}" in taken for name in own for trip in range(trips)))
    for trip in range(trips):
        at = Literal(trip, "int")
        if isinstance(loop, StridedLoop):
            at = BinaryExpression("+", loop.start, Literal(trip * loop.step.value, "int"))
        copy = _copy(loop, at, {name: f"{name}__u{trip}{tag}" for name in own})
        if isinstance(loop, StridedLoop):
            copy = Body((Cond(BinaryExpression("<", at, loop.axis.extent.expression), copy),))
        out.extend(_decided(copy, ctx))
    return out


def _stepped(loop: Statement, copies: int, outer: Collection[str]) -> StridedLoop:
    """``loop`` stepping ``copies`` trips at a time: its body holds that many copies of the original, the ``n``-th at
    the axis ``n`` steps on. Every thread runs a number of trips ``copies`` divides, so no copy needs a guard. It says
    it is ``unrolled``, so whatever generates its code keeps it a loop."""
    start, step = (loop.start, loop.step.value) if isinstance(loop, StridedLoop) else (Literal(0, "int"), 1)
    own = _own(loop, outer) - _carried(loop)
    axis = Var(loop.axis.name)
    body = [stmt for n in range(copies) for stmt in _copy(loop, axis + Literal(n * step, "int"), {name: f"{name}__u{n}" for name in own})]
    return StridedLoop(axis=loop.axis, start=start, step=Literal(step * copies, "int"), body=Body(body), seed=loop.seed, unrolled=True)


def _runtime(loop: Statement, copies: int, outer: Collection[str]) -> list[Statement]:
    """A :func:`runtime_bound` ``loop`` as a loop stepping ``copies`` trips at a time, then the trips left, in the
    shape cicc gives a loop it unrolls at run time:

    - The loop runs ``copies`` copies of the body a trip, the ``n``-th at the axis ``n`` steps on, while that many
      trips are left (its early stop). It seeds what the loop seeds.
    - The trips left run as blocks of half as many copies, a quarter as many, and so on down to one, each where its
      bit of the count left is set. Past 8 copies the blocks stop at four, and the last three trips at most are a
      loop of one copy.
    - The blocks stand under the condition that the loop runs at all, where the count of trips is not negative.

    Each copy defines names of its own (``<name>__u<n>``, ``n`` counting every copy). Both loops say they are
    ``unrolled``, as cicc marks the two loops it writes ``nounroll``.
    """
    start, step = (loop.start, loop.step.value) if isinstance(loop, StridedLoop) else (Literal(0, "int"), 1)
    bound = loop.axis.extent.expression
    own = _own(loop, outer) - _carried(loop)
    serial = count()

    def copy(at: Expression) -> Body:
        n = next(serial)
        return _copy(loop, at, {name: f"{name}__u{n}" for name in own})

    def block(at: Expression) -> Body:
        # A copy outside the loop folds over no axis: the loop's leaves its folds before the axis is bound.
        alone = loop.replace(
            body=loop.body.map(lambda s: s.replace(axes=tuple(a for a in s.axes if a != loop.axis.name)) if isinstance(s, Accum) else s)
        )
        n = next(serial)
        return _copy(alone, at, {name: f"{name}__u{n}" for name in own})

    def lit(value: int) -> Literal:
        return Literal(value, "int")

    def binary(op: str, left: Expression, right: Expression) -> Expression:
        return BinaryExpression(op, left, right)

    axis = Var(loop.axis.name)
    stepping = StridedLoop(
        axis=loop.axis,
        start=start,
        step=lit(step * copies),
        end=binary("-", bound, lit(step * (copies - 1))),
        body=Body(stmt for n in range(copies) for stmt in copy(binary("+", axis, lit(n * step)))),
        seed=loop.seed,
        unrolled=True,
    )
    span = binary("-", bound, start)
    trips = span if step == 1 else binary("/", binary("+", span, lit(step - 1)), lit(step))

    def at(trip: Expression) -> Expression:
        return binary("+", start, trip if step == 1 else binary("*", trip, lit(step)))

    blocks = [copies >> k for k in range(1, copies.bit_length())]
    looped = len(blocks) > 3
    left: list[Statement] = []
    for size in blocks[:2] if looped else blocks:
        done = binary("-", trips, binary("%", trips, lit(2 * size)))
        body = [stmt for n in range(size) for stmt in block(at(binary("+", done, lit(n))))]
        left.append(Cond(binary(">=", binary("%", trips, lit(2 * size)), lit(size)), Body(body)))
    if looped:
        done = binary("-", trips, binary("%", trips, lit(4)))
        left.append(StridedLoop(axis=loop.axis, start=at(done), step=lit(step), body=loop.body, seed=False, unrolled=True))
    return [stepping, Cond(binary("<", start, bound), Body(left))]


def _own(loop: Statement, outer: Collection[str]) -> set[str]:
    """The names ``loop``'s body defines that no statement around the loop does."""
    return {name for stmt in loop.body.iter() for name in stmt.defines()} - set(outer)


def _copy(loop: Statement, at: Expression, rename: dict[str, str]) -> Body:
    """``loop``'s body with its axis bound to ``at`` and its own names renamed.

    The axis is an integer, so an operation the binding leaves over two integer literals is the integer one, and the
    copy folds it so (``1 / 5`` is ``0``). The body's optimization would fold a quotient of integer literals in a value
    field as the float one, and a select's predicate over coordinates is a value field."""
    bound = Body(stmt.rewrite(lambda name: rename.get(name, name), Sigma({loop.axis.name: at})) for stmt in loop.body)
    return bound.map(lambda stmt: stmt.map_expressions(lambda expression: expression.rebuild(_integer_folded)))


def _integer_folded(expression: Expression) -> Expression:
    """``expression`` folded when it is an integer operation over two integer literals."""
    if not (isinstance(expression, BinaryExpression) and expression.op in ("+", "-", "*", "/", "%")):
        return expression
    left, right = expression.left, expression.right
    if not all(isinstance(side, Literal) and side.dtype == "int" for side in (left, right)):
        return expression
    if expression.op in ("/", "%") and right.value == 0:
        return expression
    return Literal(apply_binop(expression.op, left.value, right.value), "int")


def _decided(body: Body, ctx: SimplifyCtx) -> Body:
    """``body`` with each guard the ranges of ``ctx`` decide replaced by the branch it takes."""
    out: list[Statement] = []
    for stmt in body:
        nested = stmt.nested()
        if nested:
            stmt = stmt.with_bodies(tuple(_decided(inner, ctx) for inner in nested))
        cond = stmt.cond.simplify(ctx) if isinstance(stmt, Cond) else None
        if isinstance(cond, Literal):
            out.extend(stmt.body if cond.value else stmt.else_body)
        else:
            out.append(stmt)
    return Body(out)
