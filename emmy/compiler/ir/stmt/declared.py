"""A statement kind declared as data: typed fields, and the role each field plays.

:class:`Declared` computes every member of the statement protocol from those roles, so a kind that subclasses it
writes none of them: the shared kinds and the Kernel IR kinds are all declared. A role says what a field is to the
rest of the compiler:

================  ==========================================================================================
``defines``       SSA names the statement defines
``reads``         SSA names it reads
``updates``       SSA names it reads and defines in place
``local``         names it binds for its own use: renamed with it, hidden from what it reads
``buffer_read``   buffers it reads
``buffer_write``  buffers it writes
``local_buffer``  buffers it declares kernel-local
``expr``          expressions
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

An ``expr`` field that also ``reads`` is read through: the names its expressions mention are names the statement
reads, each listed once.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import MISSING, Field, field, fields, replace
from functools import cache
from itertools import islice
from typing import NamedTuple

from emmy.compiler.ir.expr import Expr
from emmy.compiler.ir.stmt.base import Stmt, _identity_rename
from emmy.compiler.ir.stmt.body import Body

_DEFINES = frozenset({"defines", "updates"})
_READS = frozenset({"reads", "updates"})
_LOCAL = frozenset({"local"})
_NAMES = _DEFINES | _READS | _LOCAL
_BUFFER_READS = frozenset({"buffer_read"})
_BUFFER_WRITES = frozenset({"buffer_write"})
_LOCAL_BUFFERS = frozenset({"local_buffer"})
_EXPRS = frozenset({"expr"})
_AXES = frozenset({"axis"})
_CARRIED = frozenset({"stmts", "term"})
_ROLES = _NAMES | _BUFFER_READS | _BUFFER_WRITES | _LOCAL_BUFFERS | _EXPRS | _AXES | _CARRIED | {"axis_name", "body", "record"}


def role(*roles: str, default: object = MISSING) -> Field:
    """A dataclass field that plays ``roles`` — the one way a declared kind or record states what a field is."""
    if unknown := set(roles) - _ROLES:
        raise ValueError(f"unknown field role(s) {sorted(unknown)}; the roles are {sorted(_ROLES)}")
    return field(default=default, metadata={"roles": frozenset(roles)})


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


class _Shape(NamedTuple):
    """What the members asked most often need of a class, worked out once: the field a member reads when it reads one
    field's own values and nothing else, and the fields that hold bodies, carried statements and axes."""

    reads: str | None
    defines: str | None
    exprs: str | None
    bodies: tuple[str, ...]
    scopes: tuple[str, ...]
    axes: bool


_SHAPES: dict[type, _Shape] = {}


def _shape(cls: type) -> _Shape:
    def only(roles: frozenset[str], carried: bool = False) -> str | None:
        route = _route(cls, roles, carried)
        return route[0][0] if len(route) == 1 and route[0][1] == _OWN else None

    def playing(role_name: str) -> tuple[str, ...]:
        return tuple(name for name, has in _plan(cls) if role_name in has)

    reads = only(_READS)
    shape = _SHAPES[cls] = _Shape(
        reads=None if reads in playing("expr") else reads,
        defines=only(_DEFINES),
        exprs=only(_EXPRS, True),
        bodies=playing("body"),
        scopes=playing("stmts"),
        axes=bool(_route(cls, _AXES, False)),
    )
    return shape


def _items(value: object) -> tuple:
    if type(value) is tuple:
        return value
    return () if value is None or isinstance(value, (int, float)) else (value,)


def _each(value: object, fn: Callable) -> object:
    if type(value) is tuple:
        return tuple(fn(item) for item in value)
    return value if isinstance(value, (int, float)) else fn(value)


def _gather(obj: object, roles: frozenset[str], member: str | None = None) -> Iterator:
    """What the fields of ``obj`` that play one of ``roles`` hold, in field order. With ``member``, a statement the
    object carries adds its own answer to that protocol member."""
    for name, how in _route(type(obj), roles, member is not None):
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
                yield from _gather(record, roles, member)


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


def _mapped(body: Body, fn: Callable[[Stmt], Stmt]) -> Body:
    """``body`` with ``fn`` applied to each statement; ``body`` itself when every statement comes back as it was."""
    stmts = tuple(fn(stmt) for stmt in body)
    return body if all(new is old for new, old in zip(stmts, body, strict=True)) else Body(stmts)


_NO_AXES: frozenset[str] = frozenset()


class Declared(Stmt):
    """A statement kind whose protocol members follow from its fields' roles (see the module docstring)."""

    def deps(self) -> tuple[str, ...]:
        shape = _SHAPES.get(type(self)) or _shape(type(self))
        if shape.reads is not None:
            return _items(getattr(self, shape.reads))
        reads: list[str] = []
        for item in _gather(self, _READS):
            if isinstance(item, Expr):
                reads += [name for name in sorted(item.free_vars()) if name not in reads]
            else:
                reads.append(item)
        if shape.scopes:
            hidden = {*reads, *_gather(self, _LOCAL)}
            # The carried statements are a scope of their own: one's definition hides the next one's read of it.
            for name in shape.scopes:
                for stmt in getattr(self, name):
                    reads += [dep for dep in dict.fromkeys(stmt.deps()) if dep not in hidden]
                    hidden.update(reads, stmt.defines())
        return tuple(reads)

    def defines(self) -> tuple[str, ...]:
        only = (_SHAPES.get(type(self)) or _shape(type(self))).defines
        return _items(getattr(self, only)) if only is not None else tuple(_gather(self, _DEFINES))

    def external_reads(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(_gather(self, _BUFFER_READS, "external_reads")))

    def external_writes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(_gather(self, _BUFFER_WRITES, "external_writes")))

    def local_decls(self) -> tuple[str, ...]:
        return tuple(_gather(self, _LOCAL_BUFFERS))

    def binds_axes(self) -> frozenset[str]:
        if not (_SHAPES.get(type(self)) or _shape(type(self))).axes:
            return _NO_AXES
        return frozenset(axis.name for axis in _gather(self, _AXES))

    def exprs(self) -> tuple[Expr, ...]:
        only = (_SHAPES.get(type(self)) or _shape(type(self))).exprs
        return _items(getattr(self, only)) if only is not None else tuple(_gather(self, _EXPRS, "exprs"))

    def with_exprs(self, exprs: tuple[Expr, ...]) -> Stmt:
        fresh = iter(exprs)

        def change(has: frozenset[str], value):  # noqa: ANN001, ANN202 — a field's value
            if "expr" in has:
                return _each(value, lambda _expr: next(fresh))
            if has & _CARRIED:
                carried = (stmt.with_exprs(tuple(islice(fresh, len(tuple(stmt.exprs()))))) for stmt in value)
                return Body(carried) if "term" in has else tuple(carried)
            return value

        rebuilt = _rebuilt(self, change)
        if next(fresh, None) is not None:
            raise ValueError(f"{type(self).__name__} holds fewer expressions than it was given")
        return rebuilt

    def nested(self) -> tuple[Body, ...]:
        bodies = (_SHAPES.get(type(self)) or _shape(type(self))).bodies
        return tuple(getattr(self, name) for name in bodies) if bodies else ()

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        names = (_SHAPES.get(type(self)) or _shape(type(self))).bodies
        return replace(self, **dict(zip(names, bodies, strict=True))) if names else super().with_bodies(bodies)

    def rename_buffers(self, rename) -> Stmt:  # noqa: ANN001 — see ``Stmt.rename_buffers``
        def change(has: frozenset[str], value):  # noqa: ANN001, ANN202 — a field's value
            if has & (_BUFFER_READS | _BUFFER_WRITES):
                return _each(value, lambda buffer: rename.get(buffer, buffer))
            if has & _CARRIED:
                carried = (stmt.rename_buffers(rename) for stmt in value)
                return Body(carried) if "term" in has else tuple(carried)
            return value

        return _rebuilt(self, change)

    def _rewrite(self, rename, sigma, axis_fn):  # noqa: ANN001, ANN202 — see ``Stmt.rewrite``
        def change(has: frozenset[str], value):  # noqa: ANN001, ANN202 — a field's value
            if "expr" in has:
                return _each(value, lambda expr: sigma.apply(expr).rename(rename))
            if has & _NAMES:
                return _each(value, rename)
            if "axis" in has:
                return _each(value, axis_fn)
            if "axis_name" in has:
                axes = {axis.name: axis_fn(axis).name for axis in _gather(self, _AXES)}
                return _each(value, lambda name: axes.get(name, name))
            if "body" in has:
                return _mapped(value, lambda stmt: stmt.rewrite(rename, sigma, axis_fn))
            if "stmts" in has:
                return tuple(stmt.rewrite(rename, sigma, axis_fn) for stmt in value)
            if "term" in has:  # its statements' names are its own; a coordinate its expressions read is not
                defined = {name for stmt in value for name in stmt.defines()}
                moved = (stmt.rewrite(_identity_rename, sigma, axis_fn) for stmt in value)
                return Body(
                    stmt.with_exprs(tuple(expr.rename(lambda name: name if name in defined else rename(name)) for expr in stmt.exprs()))
                    for stmt in moved
                )
            return value

        return _rebuilt(self, change)
