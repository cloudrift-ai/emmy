"""Lexical loop scopes and definition analysis for statement reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.stmt.base import Stmt
from emmy.compiler.ir.stmt.blocks import Cond, Loop
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.leaves import Accum, Carry, Write


@dataclass(frozen=True)
class Scope:
    """Enclosing loop nest from outermost to innermost.

    A ``Scope`` identifies a location in a statement body: the sequence
    of ``Loop`` axes one descends to reach that point. Empty = body root.
    Used by analysis passes that need to know where a named SSA value was
    defined or where a new stmt should be emitted.
    """

    enclosing: tuple[Axis, ...] = ()

    def nest(self, axis: Axis) -> Scope:
        return Scope(enclosing=self.enclosing + (axis,))


@dataclass(frozen=True)
class BodyAnalysis:
    """Precomputed lookups over a statement body.

    - ``body``: the source statement body, including compact subroutine calls during fusion.
    - ``defs``: SSA name → defining ``Stmt`` (every defining leaf: ``Load`` / ``Assign`` / ``Select`` /
      ``Let`` / ``Accum`` / ``Call`` …). A ``Write`` has no SSA name and is not here.
    - ``scopes``: SSA name → binding ``Scope`` (where the value is live
      after its def). For plain stmts this is the enclosing axis chain;
      for ``Accum`` the reduce axis is excluded — the Accum binds *after*
      the reduce Loop completes.
    - ``reduce_axes``: ``Accum`` name → its reduce ``Axis`` (the tail of
      the raw enclosing chain, stripped from ``scopes``). Only present
      for ``Accum`` defs.
    - ``writes``: every ``Write`` stmt paired with the ``Scope`` it sits
      in — one entry per output, in body order.
    - ``live_axes``: SSA name → axis names transitively reachable through
      Expr subtrees (``Load.index``, ``SelectBranch.select``) while resolving
      the stmt's dep chain. For an ``Accum``, the reduce axis is excluded
      since it gets freshened at emission time.
    """

    body: Body
    defs: dict[str, Stmt]
    scopes: dict[str, Scope]
    reduce_axes: dict[str, Axis]
    writes: tuple[tuple[Write, Scope], ...]
    live_axes: dict[str, frozenset[str]]

    @classmethod
    def from_body(cls, body: Body, enclosing: tuple[Axis, ...] = ()) -> BodyAnalysis:
        """Analyze a loop body or a subroutine with its formal coordinates already bound."""

        defs: dict[str, Stmt] = {}
        scopes: dict[str, Scope] = {}
        reduce_axes: dict[str, Axis] = {}
        writes: list[tuple[Write, Scope]] = []

        def walk(stmts: Body, scope: Scope) -> None:
            for s in stmts:
                if isinstance(s, Loop):
                    walk(s.body, scope.nest(s.axis))
                elif isinstance(s, Cond):
                    walk(s.body, scope)
                    walk(s.else_body, scope)
                elif isinstance(s, Accum):
                    defs[s.name] = s
                    # Binding scope excludes the reduce axis (the Accum is live
                    # after its reduce Loop completes).
                    if scope.enclosing:
                        reduce_axes[s.name] = scope.enclosing[-1]
                        scopes[s.name] = Scope(enclosing=scope.enclosing[:-1])
                    else:
                        scopes[s.name] = scope
                elif isinstance(s, Write):
                    writes.append((s, scope))
                elif not isinstance(s, Carry):
                    # Every other defining leaf — a load, an assignment, a selection, a call, a ``Let`` binding a
                    # literal or an index — binds its names at this scope; one it does not define reads as external.
                    for name in s.defines():
                        defs[name] = s
                        scopes[name] = scope

        walk(body, Scope(enclosing))
        bound = body
        for axis in reversed(enclosing):
            bound = Body((Loop(axis, bound),))
        return BodyAnalysis(
            body=body,
            defs=defs,
            scopes=scopes,
            reduce_axes=reduce_axes,
            writes=tuple(writes),
            live_axes=bound.axis_dependencies,
        )
