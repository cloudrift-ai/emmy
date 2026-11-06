"""Rewrites over one statement that the statement protocol does not carry: an expression map and a hygienic rename."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Expr
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt.base import Stmt, _axis_identity
from emmy.compiler.ir.stmt.body import Body

Rename = Callable[[str], str]
AxisFn = Callable[[Axis], Axis]


def map_exprs(stmt: Stmt, fn: Callable[[Expr], Expr]) -> Stmt:
    """``stmt`` with ``fn`` applied to each of its own expressions — an index, a predicate, a
    stride — and not to nested bodies, which :meth:`Body.map` reaches on its own."""
    old = tuple(stmt.exprs())
    new = tuple(fn(expr) for expr in old)
    return stmt if new == old else stmt.with_exprs(new)


def rename_free(stmt: Stmt, alias: Mapping[str, str]) -> Stmt:
    """:func:`rewrite` under an alias map, made **hygienic**: the rename stops at a nested scope
    that re-binds one of the aliased names.

    ``rewrite`` descends into every nested body and maps a stmt's OWN bindings as well as its
    reads. But ``Assign`` / ``Load`` / ``Select`` names bound inside a ``Loop`` / ``Cond`` body are
    scoped to that body (see :class:`~emmy.compiler.ir.stmt.blocks.Loop`), so an inner binding that
    merely shares a spelling with an aliased outer name is a DIFFERENT variable. Renaming it both
    redeclares the survivor inside the scope and rewires the inner arithmetic to the outer value.

    Use this — not a bare ``rewrite`` — whenever the alias comes from dropping a binding (load
    dedup, CSE) rather than from a whole-subtree renumbering.
    """
    if not alias:
        return stmt
    bodies = stmt.nested()
    shell = stmt.with_bodies(tuple(Body() for _ in bodies)) if bodies else stmt
    renamed = shell.rewrite(lambda nm: alias.get(nm, nm), Sigma.IDENTITY, _axis_identity)
    if not bodies:
        return renamed
    # Rewrite each child once, with only the aliases its scope does not rebind.
    inner = []
    for b in bodies:
        bound = b.local_defs
        pruned = {k: v for k, v in alias.items() if k not in bound and k not in stmt.binds_axes()}
        # The surviving value may itself be shadowed here. Rename the local binding before
        # substituting, so dropping outer ``b = a`` cannot capture the use of b as an inner a.
        captured = bound.intersection(pruned.values())
        if captured:
            used = set(b.ssa_defs | b.ssa_uses | b.axis_names) | set(alias) | set(alias.values())
            renames = {}
            for name in sorted(captured):
                fresh = name + "_local"
                while fresh in used:
                    fresh += "_local"
                renames[name] = fresh
                used.add(fresh)
            b = Body(member.rename(renames) for member in b)
        inner.append(Body(tuple(rename_free(c, pruned) for c in b)))
    return renamed.with_bodies(tuple(inner))
