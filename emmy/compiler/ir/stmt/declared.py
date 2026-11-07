"""A statement kind declared as data: typed fields, and the role each field plays.

:func:`statement` declares one: it makes the class a frozen dataclass and writes every member of the statement protocol
from those roles, once, when the class is defined. So a kind writes none of them: the shared kinds and the Kernel IR
kinds are all declared. A role says what a field is to the rest of the compiler:

================  ==========================================================================================
``defines``       SSA names the statement defines
``reads``         SSA names it reads
``updates``       SSA names it reads and defines in place
``local``         names it binds for its own use: renamed with it, hidden from what it reads
``buffer_read``   buffers it reads
``buffer_write``  buffers it writes
``local_buffer``  buffers it declares kernel-local
``expression``          expressions
``axis``          axes it binds for its child bodies
``axis_name``     names of axes it binds, which follow those axes when they are renamed
``body``          a child body
``stmts``         statements it carries: rewritten with it, their free reads, buffers and expressions its own
``term``          the body of a pure term: its statements' names are private, the rest is the statement's
``record``        records whose own fields play roles
================  ==========================================================================================

A field holds one value, a tuple of them, or ``None``. A number where a name, a buffer or an expression is expected is
plain data: a stride that is a constant, a seed that is a literal. A field with no role is plain data: nothing renames
it and no member reports it.

An ``expression`` field that also ``reads`` is read through: the names its expressions mention are names the statement
reads, each listed once.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import MISSING, Field, dataclass, field, fields, replace
from functools import cache
from typing import dataclass_transform

from emmy.compiler.ir.expression import Expression
from emmy.compiler.ir.stmt.base import Statement, _identity_rename
from emmy.compiler.ir.stmt.body import Body

_DEFINES = frozenset({"defines", "updates"})
_READS = frozenset({"reads", "updates"})
_LOCAL = frozenset({"local"})
_NAMES = _DEFINES | _READS | _LOCAL
_BUFFER_READS = frozenset({"buffer_read"})
_BUFFER_WRITES = frozenset({"buffer_write"})
_LOCAL_BUFFERS = frozenset({"local_buffer"})
_EXPRS = frozenset({"expression"})
_AXES = frozenset({"axis"})
_CARRIED = frozenset({"stmts", "term"})
_ROLES = _NAMES | _BUFFER_READS | _BUFFER_WRITES | _LOCAL_BUFFERS | _EXPRS | _AXES | _CARRIED | {"axis_name", "body", "record"}


def role(*roles: str, default: object = MISSING) -> Field:
    """A dataclass field that plays ``roles`` — the one way a declared kind or record states what a field is."""
    if unknown := set(roles) - _ROLES:
        raise ValueError(f"unknown field role(s) {sorted(unknown)}; the roles are {sorted(_ROLES)}")
    return field(default=default, metadata={"roles": frozenset(roles)})


#: Every declared kind by name — what the text reader builds a kind of another layer from.
KINDS: dict[str, type] = {}


@dataclass_transform(frozen_default=True, field_specifiers=(role, field))
def statement(cls: type | None = None, /, **options: bool):  # noqa: ANN201 — the class, or the decorator that takes it
    """Declare a statement kind: ``cls`` becomes a frozen dataclass (``options`` go to ``dataclass``) and gets every
    protocol member its fields' roles give. A member the class states itself, because no role gives it, is kept."""

    def declare(cls: type) -> type:
        cls = dataclass(frozen=True, **options)(cls)
        KINDS[cls.__name__] = cls
        for name, member in _members(cls).items():
            stated = getattr(cls, name)
            if stated is getattr(Statement, name) or getattr(stated, "derived", False):
                member.derived = True
                setattr(cls, name, member)
        return cls

    return declare if cls is None else declare(cls)


@cache
def _plan(cls: type) -> tuple[tuple[str, frozenset[str]], ...]:
    """The fields of a declared class that play a role, in declaration order."""
    return tuple((f.name, f.metadata["roles"]) for f in fields(cls) if "roles" in f.metadata)


_OWN, _STMTS, _RECORDS = range(3)


@cache
def _route(cls: type, roles: frozenset[str], carried: bool) -> tuple[tuple[str, int], ...]:
    """The fields a gather over ``roles`` visits, and how: a field's own values, the statements it carries, or its
    records."""
    route = []
    for name, has in _plan(cls):
        if has & roles:
            route.append((name, _OWN))
        elif carried and has & _CARRIED:
            route.append((name, _STMTS))
        elif "record" in has:
            route.append((name, _RECORDS))
    return tuple(route)


def _items(value: object) -> tuple:
    if type(value) is tuple:
        return value
    return () if value is None or isinstance(value, (int, float)) else (value,)


def _each(value: object, fn: Callable) -> object:
    if type(value) is tuple:
        return tuple(fn(item) for item in value)
    return value if isinstance(value, (int, float)) else fn(value)


def _gather(obj: object, route: tuple[tuple[str, int], ...], roles: frozenset[str], member: str | None = None) -> Iterator:
    """What the fields of ``obj`` on ``route`` hold, in field order. With ``member``, a statement the object carries
    adds its own answer to that protocol member."""
    for name, how in route:
        value = getattr(obj, name)
        if value is None:
            continue
        if how == _OWN:
            yield from _items(value)
        elif how == _STMTS:
            for stmt in value:
                yield from getattr(stmt, member)()
        else:
            for record in _items(value):
                yield from _gather(record, _route(type(record), roles, member is not None), roles, member)


def _rebuilt(obj: object, change: Callable[[frozenset[str], object], object]) -> object:
    """``obj`` with the value of every role-playing field passed through ``change``, records field by field; ``obj``
    itself when nothing changes."""
    changes = {}
    for name, has in _plan(type(obj)):
        value = getattr(obj, name)
        if value is None:
            continue
        new = _each(value, lambda record: _rebuilt(record, change)) if "record" in has else change(has, value)
        # A body that changed is another object (:func:`_mapped`); comparing two bodies would walk both.
        if new is not value and (type(new) is Body or new != value):
            changes[name] = new
    return replace(obj, **changes) if changes else obj


def _mapped(body: Body, fn: Callable[[Statement], Statement]) -> Body:
    """``body`` with ``fn`` applied to each statement; ``body`` itself when every statement comes back as it was."""
    stmts = tuple(fn(stmt) for stmt in body)
    return body if all(new is old for new, old in zip(stmts, body, strict=True)) else Body(stmts)


_NO_AXES: frozenset[str] = frozenset()


def _members(cls: type) -> dict[str, Callable]:
    """The protocol members the roles of ``cls``'s fields give. A member the roles leave at the base's default is not
    listed: a kind with no child body lists none and has none to replace."""
    plan = _plan(cls)
    bodies = tuple(name for name, has in plan if "body" in has)
    members: dict[str, Callable | None] = {"deps": _deps(cls), "_rewrite": lambda self, rename, sigma, axis_fn: self}

    def reader(roles: frozenset[str], member: str | None = None, unique: bool = False) -> Callable | None:
        route = _route(cls, roles, member is not None)
        if not route:
            return None
        if len(route) == 1 and route[0][1] == _OWN:
            ((name, _),) = route
            if unique:
                return lambda self: tuple(dict.fromkeys(_items(getattr(self, name))))
            return lambda self: _items(getattr(self, name))
        if unique:
            return lambda self: tuple(dict.fromkeys(_gather(self, route, roles, member)))
        return lambda self: tuple(_gather(self, route, roles, member))

    members["defines"] = reader(_DEFINES)
    members["external_reads"] = reader(_BUFFER_READS, "external_reads", unique=True)
    members["external_writes"] = reader(_BUFFER_WRITES, "external_writes", unique=True)
    members["local_decls"] = reader(_LOCAL_BUFFERS)
    if (expressions := reader(_EXPRS, "expressions")) is not None:
        members.update(expressions=expressions, map_expressions=_map_expressions)
    if axes := _route(cls, _AXES, False):
        members["bound_axes"] = lambda self: tuple(_gather(self, axes, _AXES))
        members["binds_axes"] = lambda self: frozenset(axis.name for axis in _gather(self, axes, _AXES))
    if len(bodies) == 1:
        members["nested"] = lambda self: (getattr(self, bodies[0]),)
    elif bodies:
        members["nested"] = lambda self: tuple(getattr(self, name) for name in bodies)
    if bodies:
        members["with_bodies"] = lambda self, new: replace(self, **dict(zip(bodies, new, strict=True)))
    if any(has & (_BUFFER_READS | _BUFFER_WRITES | _CARRIED) or "record" in has for _, has in plan):
        members["rename_buffers"] = _rename_buffers
    if any(has - _BUFFER_READS - _BUFFER_WRITES - _LOCAL_BUFFERS for _, has in plan):
        members["_rewrite"] = rewrite_by_roles
    return {name: member for name, member in members.items() if member is not None}


def _deps(cls: type) -> Callable | None:
    route = _route(cls, _READS, False)
    scopes = tuple(name for name, has in _plan(cls) if "stmts" in has)
    locals_ = _route(cls, _LOCAL, False)
    if not route and not scopes:
        return None
    if len(route) == 1 and route[0][1] == _OWN and not scopes and "expression" not in dict(_plan(cls))[route[0][0]]:
        ((name, _),) = route
        return lambda self: _items(getattr(self, name))

    def deps(self: Statement) -> tuple[str, ...]:
        reads: list[str] = []
        for item in _gather(self, route, _READS):
            if isinstance(item, Expression):  # a field read through: the names its expressions mention, each listed once
                reads += [name for name in sorted(item.free_vars()) if name not in reads]
            else:
                reads.append(item)
        if scopes:
            hidden = {*reads, *_gather(self, locals_, _LOCAL)}
            # The carried statements are a scope of their own: one's definition hides the next one's read of it.
            for name in scopes:
                for stmt in getattr(self, name):
                    reads += [dep for dep in dict.fromkeys(stmt.deps()) if dep not in hidden]
                    hidden.update(reads, stmt.defines())
        return tuple(reads)

    return deps


def _map_expressions(self: Statement, fn: Callable[[Expression], Expression]) -> Statement:
    def change(has: frozenset[str], value):  # noqa: ANN001, ANN202 — a field's value
        if "expression" in has:
            return _each(value, fn)
        if "term" in has:
            return _mapped(value, lambda stmt: stmt.map_expressions(fn))
        if "stmts" in has:
            return tuple(stmt.map_expressions(fn) for stmt in value)
        return value

    return _rebuilt(self, change)


def _rename_buffers(self: Statement, rename) -> Statement:  # noqa: ANN001 — see ``Statement.rename_buffers``
    def change(has: frozenset[str], value):  # noqa: ANN001, ANN202 — a field's value
        if has & (_BUFFER_READS | _BUFFER_WRITES):
            return _each(value, lambda buffer: rename.get(buffer, buffer))
        if "term" in has:
            return _mapped(value, lambda stmt: stmt.rename_buffers(rename))
        if "stmts" in has:
            return tuple(stmt.rename_buffers(rename) for stmt in value)
        return value

    return _rebuilt(self, change)


def rewrite_by_roles(self: Statement, rename, sigma, axis_fn) -> Statement:  # noqa: ANN001 — see ``Statement.rewrite``
    """What a declared kind's ``_rewrite`` is: every name renamed, every expression substituted and renamed, every
    axis mapped, every child and carried statement rewritten. A kind that states its own ``_rewrite`` calls this for
    the fields the roles cover."""

    def change(has: frozenset[str], value):  # noqa: ANN001, ANN202 — a field's value
        if "expression" in has:
            return _each(value, lambda expression: sigma.apply(expression).rename(rename))
        if has & _NAMES:
            return _each(value, rename)
        if "axis" in has:
            return _each(value, axis_fn)
        if "axis_name" in has:
            axes = {axis.name: axis_fn(axis).name for axis in _gather(self, _route(type(self), _AXES, False), _AXES)}
            return _each(value, lambda name: axes.get(name, name))
        if "body" in has:
            return _mapped(value, lambda stmt: stmt.rewrite(rename, sigma, axis_fn))
        if "stmts" in has:
            return tuple(stmt.rewrite(rename, sigma, axis_fn) for stmt in value)
        if "term" in has:  # its statements' names are its own; a coordinate its expressions read is not
            defined = {name for stmt in value for name in stmt.defines()}
            moved = (stmt.rewrite(_identity_rename, sigma, axis_fn) for stmt in value)
            return Body(
                stmt.map_expressions(lambda expression: expression.rename(lambda name: name if name in defined else rename(name)))
                for stmt in moved
            )
        return value

    return _rebuilt(self, change)
