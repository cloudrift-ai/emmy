"""What the steps of a carried state read and write, each with its guard: the comparisons that hold where it counts.

A kernel that carries a state defines the next value of every cell at every step. Which cells a step really writes,
and which reads feed them, is said by its selects and by the conditionals inside its index expressions. This reading
turns those into comparisons, with nothing rewritten and no extent's size looked at:

- a cell is WRITTEN where the next state is not the cell's own previous value;
- a READ of the state counts where its value reaches a written cell, and its cell is its index with every
  conditional settled by the guard it is read under.

A dependence question is then a few comparisons for ``ir/constraints.py``: this read's cell is that write's cell,
at an earlier step.
"""

from __future__ import annotations

from dataclasses import dataclass

from emmy.compiler.ir.constraints import always, can_hold, negated
from emmy.compiler.ir.expression import BinaryExpression, Expression, Literal, TernaryExpression, Var
from emmy.compiler.ir.stmt import Accum, Assign, Body, Carry, Loop, Pre, Select

Guard = tuple[Expression, ...]


@dataclass(frozen=True)
class StateRead:
    """One read of the carried state where its value reaches a written cell."""

    name: str
    index: tuple[Expression, ...]  # the cell it reads, its conditionals settled
    guard: Guard


@dataclass(frozen=True)
class Carried:
    """A loop that carries one state: where its steps write, and what they read to do so."""

    loop: Loop
    carry: Carry
    bounds: Guard  # every loop variable inside its loop
    written: tuple[Guard, ...]  # alternatives: the comparisons under which a cell takes a new value
    reads: tuple[StateRead, ...]

    @property
    def step(self) -> str:
        return self.loop.axis.name

    @property
    def cells(self) -> tuple[str, ...]:
        return tuple(index.name for index in self.carry.index)

    def moving(self) -> tuple[int, int] | None:
        """``(position, offset)`` of the state's axis the steps move along, a step writing the one coordinate
        ``step + offset`` of it; ``None`` when the steps move along no axis, or along more than one."""
        found = []
        for position, cell in enumerate(self.cells):
            # A written cell's coordinate is pinned to the step by a pair of comparisons: collect what they say.
            candidates = set()
            for guard in self.written:
                for condition in guard:
                    for side in _comparisons(condition):
                        if {cell, self.step} <= side.free_vars():
                            for offset in range(-2, 3):
                                candidates |= {value + offset for value in _literals(side)}
            row = BinaryExpression("-", Var(cell), Var(self.step))
            for offset in sorted(candidates | {0}):
                same = BinaryExpression("==", row, Literal(offset, "int"))
                if all(always(same, *guard, *self.bounds) for guard in self.written):
                    found.append((position, offset))
                    break
        return found[0] if len(found) == 1 else None


def _comparisons(condition: Expression) -> list[Expression]:
    if isinstance(condition, BinaryExpression) and condition.op in ("&&", "||"):
        return [*_comparisons(condition.left), *_comparisons(condition.right)]
    return [condition]


def _literals(expression: Expression) -> set[int]:
    if isinstance(expression, Literal):
        return {int(expression.value)} if isinstance(expression.value, int) else set()
    return set().union(*(_literals(child) for child in expression.children())) if expression.children() else set()


def _settled(expression: Expression, guard: Guard) -> list[tuple[Expression, Guard]]:
    """``expression`` with each conditional inside it replaced by the branch ``guard`` picks. Where the guard does
    not pick, both: every result comes with the comparisons that were added to pick it."""
    if isinstance(expression, TernaryExpression):
        out = []
        for cond, extra in _settled(expression.cond, guard):
            known = (*guard, *extra)
            if always(cond, *known):
                choices = [(expression.if_true, extra)]
            elif not can_hold(cond, *known):
                choices = [(expression.if_false, extra)]
            else:
                choices = [(expression.if_true, (*extra, cond)), (expression.if_false, (*extra, negated(cond)))]
            for branch, more in choices:
                out += [(settled, (*more, *last)) for settled, last in _settled(branch, (*guard, *more))]
        return out
    children = expression.children()
    if not children:
        return [(expression, ())]
    results: list[tuple[tuple, Guard]] = [((), ())]
    for child in children:
        results = [((*done, settled), (*extra, *more)) for done, extra in results for settled, more in _settled(child, (*guard, *extra))]
    return [(expression.with_children(list(done)), extra) for done, extra in results]


def carried(body: Body) -> Carried:
    """Read the one loop of ``body`` that carries a state."""
    (loop,) = (stmt for stmt in body if isinstance(stmt, Loop) and stmt.carries)
    stmts = list(loop.body.iter())
    (carry,) = (stmt for stmt in stmts if isinstance(stmt, Carry))
    own = tuple(carry.index)
    bounds: list[Expression] = []
    for inner in (loop, *(stmt for stmt in stmts if isinstance(stmt, Loop))):
        bounds.append(BinaryExpression("<=", Literal(0, "int"), Var(inner.axis.name)))
        if inner.axis.extent.is_static:
            bounds.append(BinaryExpression("<", Var(inner.axis.name), Literal(inner.axis.extent.as_static(), "int")))
    defined = {stmt.name: stmt for stmt in stmts if isinstance(stmt, Assign | Select | Accum | Pre)}

    def paths(name: str, guard: Guard) -> list[tuple[str, Guard]]:
        """Where ``name``'s value comes from under ``guard``, through selects and copies: each source with the
        comparisons that pick it."""
        stmt = defined.get(name)
        if isinstance(stmt, Assign) and isinstance(stmt.value, Var):
            return paths(stmt.value.name, guard)
        if not isinstance(stmt, Select):
            return [(name, guard)]
        out, passed = [], [guard]
        for branch in stmt.branches:
            later = []
            for known in passed:
                for cond, extra in _settled(branch.select, (*known, *bounds)):
                    taken = (*known, *extra, cond)
                    if can_hold(*taken, *bounds):
                        out += paths(branch.value, taken)
                    refused = (*known, *extra, negated(cond))
                    if can_hold(*refused, *bounds):
                        later.append(refused)
            passed = later
        return out

    def own_value(name: str, guard: Guard) -> bool:
        stmt = defined.get(name)
        if not isinstance(stmt, Pre) or stmt.carrier != carry.name:
            return False
        same = [BinaryExpression("==", index, cell) for index, cell in zip(stmt.index, own, strict=True)]
        return all(
            always(settled, *guard, *extra, *bounds) for comparison in same for settled, extra in _settled(comparison, (*guard, *bounds))
        )

    written = tuple(guard for name, guard in paths(carry.value, ()) if not own_value(name, guard))
    reads: dict[tuple, StateRead] = {}

    def feeds(name: str, guard: Guard, seen: frozenset) -> None:
        """Every read of the state that ``name``'s value under ``guard`` is computed from."""
        for source, known in paths(name, guard):
            stmt = defined.get(source)
            if isinstance(stmt, Pre) and stmt.carrier == carry.name:
                results: list[tuple[tuple, Guard]] = [((), ())]
                for index in stmt.index:
                    results = [
                        ((*done, s), (*extra, *more)) for done, extra in results for s, more in _settled(index, (*known, *extra, *bounds))
                    ]
                for index, extra in results:
                    read = StateRead(source, index, (*known, *extra))
                    reads.setdefault((source, tuple(map(str, index)), tuple(sorted(map(str, read.guard)))), read)
            elif isinstance(stmt, Assign | Accum) and (source, known) not in seen:
                for used in stmt.value.free_vars():
                    if used in defined:
                        feeds(used, known, seen | {(source, known)})

    for guard in written:
        feeds(carry.value, guard, frozenset())
    return Carried(loop, carry, tuple(bounds), written, tuple(reads.values()))
