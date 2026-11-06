"""Block-structured ``Stmt`` subclasses — ``Loop``, ``StridedLoop``, ``Cond``.

Each carries a child body (or two, for ``Cond``) and overrides
``Stmt.nested`` so a body walk can recurse uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Expr, Var
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.declared import Declared, role
from emmy.compiler.ir.stmt.leaves import Accum, Carry

# The loop-carried reduce accumulators — a Loop is a *reduce* loop iff its immediate
# body holds one of these (the predicate `is_reduce` keys off, see below).
_CARRIERS = (Accum,)


def carried_cells(body: Body) -> dict[str, tuple[int | Axis, ...]]:
    """The carried states a ``Loop`` over ``body`` carries, each with its cell shape — one entry
    per index position, the cell's ``Axis`` or ``1`` where normalization dropped a size-one axis.

    A state is carried by the nearest enclosing loop whose axis its ``Carry`` index does not read:
    the loops over its cells are skipped, so the loop over the steps carries it from OUTSIDE them.
    Read off the body, never annotated — like :attr:`Loop.is_reduce`, which this extends.
    """
    if not body.carries:
        return {}
    out: dict[str, tuple[int | Axis, ...]] = {}

    def walk(stmts: Body, inner: dict[str, Axis]) -> None:
        for s in stmts:
            if isinstance(s, Carry) and set(inner) == set(s.cells):
                out[s.name] = tuple(inner[e.name] if isinstance(e, Var) else 1 for e in s.index)
            for child in s.nested():
                walk(child, {**inner, s.axis.name: s.axis} if isinstance(s, Loop) else inner)

    walk(body, {})
    return out


@dataclass(frozen=True)
class Loop(Declared):
    """Explicit iteration block — one loop over an axis.

    ``body`` executes ``axis.extent`` times, once per axis value. Reduce-
    kind Loops fold any ``Accum`` statements in their body into the named
    accumulator (one sweep over the axis per accumulator). Free-kind
    Loops run in parallel with no folding.

    SSA scoping: ``Assign`` / ``Select`` names defined inside ``body`` are
    scoped to that body — invisible to statements outside the Loop. Only
    ``Accum`` targets cross the Loop boundary, carrying the finalized
    reduced value.

    Used by Loop IR for general iteration; reused by Kernel IR for
    serial (post-materialization) loops inside cooperative blocks.

    ``unroll=True`` annotates the loop for ``#pragma unroll`` at render
    time. Set by scheduling passes (``mark_unroll``); has no
    effect on the IR's iteration semantics.

    The loop carries NO algebra — the fold's ⊕ lives on the
    :class:`~emmy.compiler.ir.pure.fold.Fold` node's stored ``combine``, and the dissolved fold
    ``Accum``\\ s in the body are the loop-level spelling; a loop folds iff its body carries one.
    """

    axis: Axis = role("axis")
    body: Body = role("body")
    unroll: bool = False
    seed: bool = True  # emit the per-Accum ``<acc> = identity;`` seed before the loop; False when a
    # nested drain whose accumulators are pre-seeded once outside an enclosing loop (the staged
    # scalar contraction — re-seeding per outer-slab iteration would zero the running sum).

    def __post_init__(self) -> None:
        # Coerce so ``Loop(body=tuple_value)`` keeps working without
        # forcing every construction site to wrap explicitly.
        if not isinstance(self.body, Body):
            object.__setattr__(self, "body", Body(self.body))

    @cached_property
    def carries(self) -> dict[str, tuple[int | Axis, ...]]:
        """The carried states THIS loop carries (:func:`carried_cells`)."""
        return carried_cells(self.body)

    @property
    def is_reduce(self) -> bool:
        """A loop is a reduce-loop iff it carries something: its immediate body contains a carrier
        (``Accum``), or it carries a state (:attr:`carries`) — read off the body, never annotated.
        A loop over a state's CELLS carries nothing: it is a free loop the step runs under."""
        return any(isinstance(s, _CARRIERS) for s in self.body) or bool(self.carries)


@dataclass(frozen=True)
class StridedLoop(Declared):
    """Strided iteration: ``for (axis = start; axis < axis.extent; axis += step)``.

    Cooperative variant of ``Loop`` — used at Tile IR to express "threads
    of the CUDA block stride through this axis" (typical
    ``start = Var('t'), step = BLOCK_SIZE``). The body uses the original
    axis Var directly; the strided iteration shape is encoded by the
    loop construct itself rather than via affine indexing in the body.

    ``end`` overrides the ``<`` bound with an arbitrary ``Expr`` over in-scope names
    (default: the axis extent) — a data-independent early stop such as the causal
    flash stream's triangular kv bound. The axis extent stays the analysis-visible
    range; ``end`` must never exceed it.

    Reduction detection mirrors ``Loop``: a ``StridedLoop`` is a
    reduce-loop iff its body contains an ``Accum``."""

    axis: Axis = role("axis")
    start: Expr = role("expr")
    step: Expr = role("expr")
    body: Body = role("body")
    unroll: bool = False
    end: Expr | None = role("expr", default=None)
    seed: bool = True  # emit the per-``Accum`` identity seed before the loop; False when the
    # carrier this loop folds is declared once OUTSIDE it at a seed the term names and the ⊕'s
    # identity does not (a twisted carrier's pivot seeds at the term's own ``init``, not at
    # ``maximum``'s neutral element). The same flag ``Loop`` carries, for the same reason.

    def __post_init__(self) -> None:
        if not isinstance(self.body, Body):
            object.__setattr__(self, "body", Body(self.body))

    def binds_axes(self) -> frozenset[str]:
        # An ``end`` override is hoisted into the for-init as ``<var>_end``, which the body may read.
        return frozenset({self.axis.name, f"{self.axis.name}_end"} if self.end is not None else {self.axis.name})

    @property
    def is_reduce(self) -> bool:
        """A strided loop is a reduce-loop iff its immediate body contains a carrier (``Accum``) —
        read off the body, never annotated."""
        return any(isinstance(s, _CARRIERS) for s in self.body)


@dataclass(frozen=True)
class Cond(Declared):
    """Conditional block — ``if (cond) { body } [else { else_body }]``.

    ``cond`` is an ``Expr`` over axis Vars and previously-defined SSA
    names; ``body`` and ``else_body`` are stmt sequences executed when
    the predicate evaluates true / false respectively. ``else_body``
    empty means a bare ``if``.

    SSA scoping mirrors ``Loop``: names defined inside either body are
    scoped to that body, except ``Accum`` targets which cross the boundary
    with their finalized value (matching Loop semantics).

    ``deps`` are the SSA names referenced inside ``cond`` — the splicer /
    dataflow analyses need them to thread the predicate's reads through.
    Names referenced inside ``body`` / ``else_body`` are the body stmts'
    own deps; the recursive walker picks them up.
    """

    cond: Expr = role("expr", "reads")
    body: Body = role("body")
    else_body: Body = role("body", default=Body())

    def __post_init__(self) -> None:

        if not isinstance(self.body, Body):
            object.__setattr__(self, "body", Body(self.body))
        if not isinstance(self.else_body, Body):
            object.__setattr__(self, "else_body", Body(self.else_body))
