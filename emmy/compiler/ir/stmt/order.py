"""Dependency order for statement bodies.

A scope's ordering constraints (:func:`ordering_constraints`) are what every sibling order must respect: a
definition ahead of its reads, a buffer's write ahead of the reads and writes that follow it, a carried state's
updates in sequence, and an ordered execution protocol (a barrier) pinned relative to every sibling.
:func:`topological_sort` restores a dependency-valid order before normalization; the canonical order itself is
chosen by normalization from the value numbering.
"""

from __future__ import annotations

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.stmt.base import Stmt
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.leaves import Init

__all__ = ["bound_axes", "ordering_constraints", "topological_sort"]


def _ordered_sibling_defs(stmt: Stmt) -> tuple[str, ...]:
    """Names visible to siblings in structural body order."""
    children = stmt.nested()
    if not children:
        return stmt.defines()
    return tuple(dict.fromkeys(name for child in children for name in child.carried_names))


def _free_ssa(stmt: Stmt) -> frozenset[str]:
    """SSA names ``stmt`` reads from the scope around it.

    A nested scope binds only what it defines at its own level, in any order; a deeper scope's
    definition of the same spelling is a different binder and hides nothing read above it.
    """
    children = stmt.nested()
    if not children or stmt.deps_deep:
        return frozenset(stmt.deps())
    reads = set(stmt.deps())
    for child in children:
        reads.update(child.free_ssa)
    return frozenset(reads)


def bound_axes(stmt: Stmt) -> tuple[Axis, ...]:
    """The axes a statement binds, as :class:`Axis` values."""
    axis = getattr(stmt, "axis", None)
    if isinstance(axis, Axis):
        return (axis,)
    return tuple(axis for axis in getattr(stmt, "axes", ()) if isinstance(axis, Axis))


def _resources(stmt: Stmt) -> tuple[set[str], set[str], set[str]]:
    members = tuple(member for child in stmt.nested() for member in child.iter()) or (stmt,)
    reads = {name for member in members for name in member.external_reads()}
    writes = {name for member in members for name in member.external_writes()}
    state = {name for member in members for name in getattr(member, "carried_names", lambda: ())()}
    if isinstance(stmt, Init):
        state.update(stmt.defines())
    return reads, writes, state


def ordering_constraints(body: Body, *, effects: bool, redefinitions: bool = True) -> list[set[int]]:
    """Dependency and, when requested, effect predecessors for one lexical scope."""
    defs_uses = [(frozenset(_ordered_sibling_defs(stmt)), _free_ssa(stmt)) for stmt in body]
    definitions: dict[str, list[int]] = {}
    for index, (defines, _) in enumerate(defs_uses):
        for name in defines:
            definitions.setdefault(name, []).append(index)

    def defining_stmt(name: str, consumer: int) -> int | None:
        sites = definitions.get(name, ())
        if not redefinitions:
            return sites[0] if sites and sites[0] != consumer else None
        preceding = [site for site in sites if site < consumer]
        if preceding:
            return preceding[-1]
        return next((site for site in sites if site != consumer), None)

    incoming: list[set[int]] = []
    for index, (_, uses) in enumerate(defs_uses):
        incoming.append({source for name in uses if (source := defining_stmt(name, index)) is not None})
    if redefinitions:
        for reader, (_, uses) in enumerate(defs_uses):
            for name in uses:
                for later_definition in definitions.get(name, ()):
                    if later_definition > reader:
                        incoming[later_definition].add(reader)

    if effects:
        accesses = [_resources(stmt) for stmt in body]
        last_write: dict[str, int] = {}
        readers: dict[str, set[int]] = {}
        last_state: dict[str, int] = {}
        for index, (reads, writes, state) in enumerate(accesses):
            for name in reads:
                if (writer := last_write.get(name)) is not None:
                    incoming[index].add(writer)
                if name not in writes:
                    readers.setdefault(name, set()).add(index)
            for name in writes:
                if (writer := last_write.get(name)) is not None:
                    incoming[index].add(writer)
                incoming[index].update(readers.pop(name, ()))
                last_write[name] = index
            for name in state:
                if (previous := last_state.get(name)) is not None:
                    incoming[index].add(previous)
                last_state[name] = index

        # A no-dataflow, non-pure leaf is an ordered execution protocol (barriers, async
        # commit/wait, declarations, and future primitives of the same kind). Pin it relative to
        # every sibling. Resource and carried-state leaves are already ordered above; ordinary
        # computations expose defs/deps and remain freely topological.
        protocol = {
            index for index, stmt in enumerate(body) if not stmt.pure and not stmt.nested() and not stmt.defines() and not stmt.deps()
        }
        preceding: list[int] = []
        previous_protocol: int | None = None
        for index in range(len(body)):
            if index in protocol:
                incoming[index].update(preceding)
                if previous_protocol is not None:
                    incoming[index].add(previous_protocol)
                preceding.clear()
                previous_protocol = index
            else:
                if previous_protocol is not None:
                    incoming[index].add(previous_protocol)
                preceding.append(index)
    return incoming


def topological_sort(stmts: Body) -> Body:
    """Stable recursive dependency sort used before structural normalization.

    A scope's definitions bind its reads whatever order they were emitted in — the splicer lands
    consumers above producers — and shadow an enclosing scope's binding of the same spelling.
    """
    body = Body(
        stmt.with_bodies(tuple(topological_sort(child) for child in stmt.nested())) if stmt.nested() else stmt
        for stmt in Body.coerce(stmts)
    )
    return body.topological_order(ordering_constraints(body, effects=False, redefinitions=False))
