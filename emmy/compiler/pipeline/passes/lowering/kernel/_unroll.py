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


def unroll_where(body: Body, copies: Callable[[Statement, tuple[int, int]], int | None]) -> Body:
    """``body`` with every loop ``copies`` gives a number unrolled into that many copies of its body, innermost
    first. ``copies`` reads a loop and its :func:`trip_counts`; a number below the loop's trips keeps the loop, each
    of its trips running that many copies."""
    return _walk(body, copies, SimplifyCtx.empty(), frozenset())


def _walk(body: Body, copies: Callable[[Statement, tuple[int, int]], int | None], ctx: SimplifyCtx, outer: frozenset[str]) -> Body:
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
        n = copies(stmt, counts) if counts is not None else None
        if n is None:
            out.append(stmt)
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
    the axis ``n`` steps on. Every thread runs a number of trips ``copies`` divides, so no copy needs a guard."""
    start, step = (loop.start, loop.step.value) if isinstance(loop, StridedLoop) else (Literal(0, "int"), 1)
    own = _own(loop, outer) - set(_folds(loop))
    axis = Var(loop.axis.name)
    body = [stmt for n in range(copies) for stmt in _copy(loop, axis + Literal(n * step, "int"), {name: f"{name}__u{n}" for name in own})]
    return StridedLoop(axis=loop.axis, start=start, step=Literal(step * copies, "int"), body=Body(body), seed=loop.seed)


def _own(loop: Statement, outer: Collection[str]) -> set[str]:
    """The names ``loop``'s body defines that no statement around the loop does."""
    return loop.body.defined - set(outer)


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
