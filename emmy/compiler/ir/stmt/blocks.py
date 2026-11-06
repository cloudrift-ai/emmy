"""Block-structured ``Stmt`` subclasses — ``Loop``, ``StridedLoop``, ``Cond``.

Each carries a child body (or two, for ``Cond``) and overrides
``Stmt.nested`` so a body walk can recurse uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Expr, Var
from emmy.compiler.ir.stmt.base import INDENT, Stmt, pretty_body
from emmy.compiler.ir.stmt.body import Body
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


def _carries_note(carries: dict[str, tuple[int | Axis, ...]]) -> str:
    """``# carries <name>[<extent>, ...]`` for the first line of a carrying loop.

    Cell EXTENTS, not axis names: the cell axes are bound inside the loop, so their names are
    not in scope on the line this prints on.
    """

    def cell(shape: tuple[int | Axis, ...]) -> str:
        return ", ".join(str(d if isinstance(d, int) else d.extent) for d in shape)

    return "# carries " + ", ".join(f"{name}[{cell(shape)}]" for name, shape in carries.items())


def _source_suffix(axis: Axis) -> str:
    """Render ``" (of <source.name>)"`` when ``axis`` was carved out of a parent.

    Returns empty for top-level axes (``source_axis is None``) or self-referential
    sources (``source is axis``). Surfaces the partition-planner's split parentage
    in IR dumps without affecting structural keys.
    """
    src = axis.source_axis
    if src is None or src is axis or src.name == axis.name:
        return ""
    return f" (of {src.name})"


@dataclass(frozen=True)
class Loop(Stmt):
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

    axis: Axis
    body: Body
    unroll: bool = False
    seed: bool = True  # emit the per-Accum ``<acc> = identity;`` seed before the loop; False when a
    # nested drain whose accumulators are pre-seeded once outside an enclosing loop (the staged
    # scalar contraction — re-seeding per outer-slab iteration would zero the running sum).

    def __post_init__(self) -> None:
        # Coerce so ``Loop(body=tuple_value)`` keeps working without
        # forcing every construction site to wrap explicitly.
        if not isinstance(self.body, Body):
            object.__setattr__(self, "body", Body(self.body))

    def deps(self) -> tuple[str, ...]:
        return ()

    def nested(self) -> tuple[Body, ...]:
        return (self.body,)

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        (body,) = bodies
        return Loop(axis=self.axis, body=body, unroll=self.unroll, seed=self.seed)

    def binds_axes(self) -> frozenset[str]:
        return frozenset({self.axis.name})

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

    def pretty(self, indent: str = "") -> list[str]:
        head = f"{indent}for {self.axis.name} in 0..{self.axis.extent}{_source_suffix(self.axis)}"
        if not self.carries:
            return [head, *pretty_body(self.body, indent + INDENT)]
        # The bar marks the carrying loop as :func:`carried_cells` derives it: every step the
        # state lives across, closed by the ``# commit`` line.
        bar = indent + "|" + " " * (len(INDENT) - 1)
        return [
            f"{head}  {_carries_note(self.carries)}",
            *pretty_body(self.body, bar),
            *(f"{bar}# commit {name}" for name in self.carries),
        ]

    def _rewrite(self, rename, sigma, axis_fn):  # noqa: ANN001, ANN202 — see ``Stmt.rewrite``
        # Preserve the reduce ``role`` annotation through σ-offsets / axis-renames. The loop carries
        # no algebra — the fold's ⊕ lives on the ``Fold`` node, whose own rewrite handler renames the
        # stored combine in lockstep (``Lambda.rename``).
        return Loop(
            axis=axis_fn(self.axis),
            body=tuple(c.rewrite(rename, sigma, axis_fn) for c in self.body),
            unroll=self.unroll,
            seed=self.seed,
        )


@dataclass(frozen=True)
class StridedLoop(Stmt):
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

    axis: Axis
    start: Expr
    step: Expr
    body: Body
    unroll: bool = False
    end: Expr | None = None
    seed: bool = True  # emit the per-``Accum`` identity seed before the loop; False when the
    # carrier this loop folds is declared once OUTSIDE it at a seed the term names and the ⊕'s
    # identity does not (a twisted carrier's pivot seeds at the term's own ``init``, not at
    # ``maximum``'s neutral element). The same flag ``Loop`` carries, for the same reason.

    def __post_init__(self) -> None:
        if not isinstance(self.body, Body):
            object.__setattr__(self, "body", Body(self.body))

    def deps(self) -> tuple[str, ...]:
        return ()

    def nested(self) -> tuple[Body, ...]:
        return (self.body,)

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        (body,) = bodies
        return StridedLoop(
            axis=self.axis,
            start=self.start,
            step=self.step,
            body=body,
            unroll=self.unroll,
            end=self.end,
            seed=self.seed,
        )

    def binds_axes(self) -> frozenset[str]:
        # An ``end`` override is hoisted into the for-init as ``<var>_end``, which the body may read.
        return frozenset({self.axis.name, f"{self.axis.name}_end"} if self.end is not None else {self.axis.name})

    def exprs(self) -> tuple[Expr, ...]:
        out = (self.start, self.step) if isinstance(self.step, Expr) else (self.start,)
        return (*out, self.end) if self.end is not None else out

    def with_exprs(self, exprs: tuple[Expr, ...]) -> Stmt:
        start, *rest = exprs
        step = rest.pop(0) if isinstance(self.step, Expr) else self.step
        return self.replace(start=start, step=step, end=rest[0] if rest else None)

    @property
    def is_reduce(self) -> bool:
        """A strided loop is a reduce-loop iff its immediate body contains a carrier (``Accum``) —
        read off the body, never annotated."""
        return any(isinstance(s, _CARRIERS) for s in self.body)

    def pretty(self, indent: str = "") -> list[str]:
        start = str(self.start)
        step = str(self.step) if isinstance(self.step, Expr) else self.step
        bound = str(self.end) if self.end is not None else self.axis.extent
        head = f"{indent}for {self.axis.name} in {start}..{bound}:{step}{_source_suffix(self.axis)}"
        return [head, *pretty_body(self.body, indent + INDENT)]

    def _rewrite(self, rename, sigma, axis_fn):  # noqa: ANN001, ANN202 — see ``Stmt.rewrite``
        def _expr(expr: Expr) -> Expr:
            return sigma.apply(expr).rename(rename)

        step = _expr(self.step) if isinstance(self.step, Expr) else self.step
        return StridedLoop(
            axis=axis_fn(self.axis),
            start=_expr(self.start),
            step=step,
            body=tuple(c.rewrite(rename, sigma, axis_fn) for c in self.body),
            unroll=self.unroll,
            end=_expr(self.end) if self.end is not None else None,
            seed=self.seed,
        )


@dataclass(frozen=True)
class Cond(Stmt):
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

    cond: Expr
    body: Body
    else_body: Body = Body()

    def __post_init__(self) -> None:

        if not isinstance(self.body, Body):
            object.__setattr__(self, "body", Body(self.body))
        if not isinstance(self.else_body, Body):
            object.__setattr__(self, "else_body", Body(self.else_body))

    def deps(self) -> tuple[str, ...]:
        return tuple(self.cond.free_vars())

    def nested(self) -> tuple[Body, ...]:
        return (self.body, self.else_body)

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        body, else_body = bodies
        return Cond(cond=self.cond, body=body, else_body=else_body)

    def exprs(self) -> tuple[Expr, ...]:
        return (self.cond,)

    def with_exprs(self, exprs: tuple[Expr, ...]) -> Stmt:
        return self.replace(cond=exprs[0])

    def pretty(self, indent: str = "") -> list[str]:
        lines = [f"{indent}if ({self.cond}):", *pretty_body(self.body, indent + INDENT)]
        if self.else_body:
            lines.append(f"{indent}else:")
            lines.extend(pretty_body(self.else_body, indent + INDENT))
        return lines

    def _rewrite(self, rename, sigma, axis_fn):  # noqa: ANN001, ANN202 — see ``Stmt.rewrite``
        return Cond(
            cond=sigma.apply(self.cond).rename(rename),
            body=tuple(c.rewrite(rename, sigma, axis_fn) for c in self.body),
            else_body=tuple(c.rewrite(rename, sigma, axis_fn) for c in self.else_body),
        )
