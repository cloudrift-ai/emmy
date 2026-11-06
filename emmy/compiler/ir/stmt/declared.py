"""A statement kind declared as data: typed fields, and the role each field plays.

:class:`Declared` computes every member of the statement protocol from those roles, so a kind that subclasses it
writes no member of its own but ``pretty``. A role says what a field is to the rest of the compiler:

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

A field holds one value, a tuple of them, or ``None``. A field with no role is plain data: nothing renames it and
no member reports it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import MISSING, Field, field, fields, replace
from functools import cache
from itertools import islice

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


def _items(value: object) -> tuple:
    return value if isinstance(value, tuple) else (value,)


def _each(value: object, fn: Callable) -> object:
    return tuple(fn(item) for item in value) if isinstance(value, tuple) else fn(value)


def _gather(obj: object, roles: frozenset[str], member: str | None = None) -> Iterator:
    """What the fields of ``obj`` that play one of ``roles`` hold, in field order. With ``member``, a statement the
    object carries adds its own answer to that protocol member."""
    for name, has in _plan(type(obj)):
        value = getattr(obj, name)
        if value is None:
            continue
        if has & roles:
            yield from _items(value)
        elif member is not None and has & _CARRIED:
            for stmt in value:
                yield from getattr(stmt, member)()
        elif "record" in has:
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
        if new != value:
            changes[name] = new
    return replace(obj, **changes) if changes else obj


class Declared(Stmt):
    """A statement kind whose protocol members follow from its fields' roles (see the module docstring)."""

    def deps(self) -> tuple[str, ...]:
        reads = list(_gather(self, _READS))
        hidden = {*reads, *_gather(self, _LOCAL)}
        for name, has in _plan(type(self)):
            if "stmts" in has:
                # The carried statements are a scope of their own: one's definition hides the next one's read of it.
                for stmt in getattr(self, name):
                    reads += [dep for dep in dict.fromkeys(stmt.deps()) if dep not in hidden]
                    hidden.update(reads, stmt.defines())
        return tuple(reads)

    def defines(self) -> tuple[str, ...]:
        return tuple(_gather(self, _DEFINES))

    def external_reads(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(_gather(self, _BUFFER_READS, "external_reads")))

    def external_writes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(_gather(self, _BUFFER_WRITES, "external_writes")))

    def local_decls(self) -> tuple[str, ...]:
        return tuple(_gather(self, _LOCAL_BUFFERS))

    def binds_axes(self) -> frozenset[str]:
        return frozenset(axis.name for axis in _gather(self, _AXES))

    def exprs(self) -> tuple[Expr, ...]:
        return tuple(_gather(self, _EXPRS, "exprs"))

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
        return tuple(getattr(self, name) for name, has in _plan(type(self)) if "body" in has)

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        names = [name for name, has in _plan(type(self)) if "body" in has]
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
        axes = {axis.name: axis_fn(axis).name for axis in _gather(self, _AXES)}

        def change(has: frozenset[str], value):  # noqa: ANN001, ANN202 — a field's value
            if has & _NAMES:
                return _each(value, rename)
            if "expr" in has:
                return _each(value, lambda expr: sigma.apply(expr).rename(rename))
            if "axis" in has:
                return _each(value, axis_fn)
            if "axis_name" in has:
                return _each(value, lambda name: axes.get(name, name))
            if "body" in has:
                return Body(stmt.rewrite(rename, sigma, axis_fn) for stmt in value)
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
