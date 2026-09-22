"""Stmt rewrite + simplify, dispatched by type."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import singledispatch

from emmy.compiler.ir.axis import Axis, extend_simplify_ctx
from emmy.compiler.ir.expr import Expr, Literal, SimplifyCtx, Var
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt.base import Stmt, _axis_identity
from emmy.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.leaves import (
    Accum,
    Assign,
    Carry,
    Init,
    Let,
    Load,
    Pre,
    Select,
    SelectBranch,
    Write,
    ZeroPrologue,
)

Rename = Callable[[str], str]
AxisFn = Callable[[Axis], Axis]


def _rename_ssa_vars_in_expr(e: Expr, rename: Rename) -> Expr:
    """Apply ``rename`` to every free ``Var`` leaf inside ``e``.

    Used by ``Load`` / ``Write`` rewriters so that *indirect* indices
    (gather: ``x[a0, (int)in0]``, scatter: ``out[(int)idx_v] = ...``)
    have their SSA-name references rewritten when the enclosing body
    is replicated. Without this, the register-tile replicator in
    ``010_split_register_axes`` suffixes the defining Load's name
    (``in0`` → ``in0_1``) but leaves dependent indirect Loads pointing
    at the original ``in0`` — silently dropping the cross-replica data
    dependency.

    Axis-name Vars (``a0``, ``M_b``, …) are never in the rename map
    (it only carries SSA defines), so ``rename(name) == name`` for
    them and they pass through unchanged.
    """
    mapping = {n: Var(rename(n)) for n in e.free_vars() if rename(n) != n}
    return e.substitute(mapping) if mapping else e


# ---------------------------------------------------------------------------
# rewrite — sigma + axis_fn + SSA renaming
# ---------------------------------------------------------------------------


@singledispatch
def _rewrite_kind(stmt: Stmt, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    raise NotImplementedError(f"rewrite not registered for {type(stmt).__name__}")


def rewrite(stmt: Stmt, rename: Rename, sigma: Sigma = Sigma.IDENTITY, axis_fn: AxisFn = _axis_identity) -> Stmt:
    """Rename, σ-substitute and axis-map one subtree. Prefer :meth:`Stmt.rename` /
    :meth:`Stmt.substitute`, which say which of the two operations a caller means.

    σ is applied HYGIENICALLY: the mapping for a name this stmt binds is dropped before its
    subtree is rewritten, because that binder introduces a different variable. A RENAME is
    unaffected — it travels through ``rename``/``axis_fn`` and renames the binder itself, which
    is exactly why it may pass where a substitution may not.
    """
    shadowed = stmt.binds_axes() & sigma.mapping.keys() if sigma.mapping else ()
    if shadowed:
        sigma = Sigma({name: value for name, value in sigma.mapping.items() if name not in shadowed})
    return _rewrite_kind(stmt, rename, sigma, axis_fn)


@_rewrite_kind.register
def _(s: Load, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return Load(
        names=tuple(rename(n) for n in s.names),
        input=s.input,
        index=tuple(_rename_ssa_vars_in_expr(sigma.apply(e), rename) for e in s.index),
        dtype=s.dtype,
    )


@_rewrite_kind.register
def _(s: Assign, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return Assign(name=rename(s.name), op=s.op, args=tuple(rename(a) for a in s.args), dtype=s.dtype)


@_rewrite_kind.register
def _(s: Accum, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    new_axes = tuple(sorted({rename(n) for old in s.axes for n in _rewrite_axis_name(old, sigma)}))
    return Accum(
        name=rename(s.name),
        value=rename(s.value),
        op=s.op,
        dtype=s.dtype,
        axes=new_axes,
        base=rename(s.base) if s.base is not None else None,
    )


@_rewrite_kind.register
def _(s: Carry, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    index = tuple(_rename_ssa_vars_in_expr(sigma.apply(e), rename) for e in s.index)
    return Carry(name=rename(s.name), value=rename(s.value), index=index, seed=s.seed, dtype=s.dtype)


@_rewrite_kind.register
def _(s: Pre, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return Pre(
        name=rename(s.name), carrier=rename(s.carrier), index=tuple(_rename_ssa_vars_in_expr(sigma.apply(e), rename) for e in s.index)
    )


def _rewrite_axis_name(name: str, sigma: Sigma) -> tuple[str, ...]:
    """Apply ``sigma`` to an axis name and return the resulting axis
    name(s). Handles three cases:

    - ``sigma`` doesn't touch ``name``: returns ``(name,)``.
    - Pure rename (``Var(old) → Var(new)``): returns ``(new,)``.
    - σ-split (``Var(K) → Var(K_o)*N + Var(K_i)``, etc.): returns the
      free-var names of the substitution expression. An Accum that
      reduced over the original axis now reduces over the split sub-
      axes.
    """
    replacement = sigma.mapping.get(name)
    if replacement is None:
        return (name,)
    return tuple(sorted(replacement.free_vars()))


@_rewrite_kind.register
def _(s: Init, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    # ``identity`` is a constant scalar — only the name moves. Renamed in lockstep with
    # the fold's ``Accum`` (registered above) so the seed stays paired.
    return Init(name=rename(s.name), identity=s.identity, dtype=s.dtype)


@_rewrite_kind.register
def _(s: Let, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return Let(name=rename(s.name), value=_rename_ssa_vars_in_expr(sigma.apply(s.value), rename), dtype=s.dtype)


@_rewrite_kind.register
def _(s: ZeroPrologue, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    # Buffer name + word count only — like ``Write.output``, ``dst`` is a buffer, and buffers
    # are not SSA names this rewrite renames. Identity, so a body carrying a delegated
    # zero-init can normalize (``structural_key`` / ``cache_key`` on the carrying kernel).
    return s


@_rewrite_kind.register
def _(s: Write, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return Write(
        output=s.output,
        index=tuple(_rename_ssa_vars_in_expr(sigma.apply(e), rename) for e in s.index),
        values=tuple(rename(n) for n in s.values),
        value_dtype=s.value_dtype,
        atomic=s.atomic,
        swizzle=s.swizzle,
    )


@_rewrite_kind.register
def _(s: Select, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    # A branch condition reads coordinates exactly as a Load index does: σ first, then the rename
    # — an axis renumbering that reached the indices and not the conditions left the causal mask
    # comparing loop variables its kernel no longer bound.
    return Select(
        name=rename(s.name),
        branches=tuple(
            SelectBranch(value=rename(b.value), select=_rename_ssa_vars_in_expr(sigma.apply(b.select), rename)) for b in s.branches
        ),
    )


@_rewrite_kind.register
def _(s: Loop, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    # Preserve the reduce ``role`` annotation through σ-offsets / axis-renames. The loop carries
    # no algebra — the fold's ⊕ lives on the ``Fold`` node, whose own rewrite handler renames the
    # stored combine in lockstep (``Lambda.rename``).
    return Loop(
        axis=axis_fn(s.axis),
        body=tuple(rewrite(c, rename, sigma, axis_fn) for c in s.body),
        unroll=s.unroll,
        seed=s.seed,
    )


@_rewrite_kind.register
def _(s: StridedLoop, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    def _expr(expr: Expr) -> Expr:
        return _rename_ssa_vars_in_expr(sigma.apply(expr), rename)

    step = _expr(s.step) if isinstance(s.step, Expr) else s.step
    return StridedLoop(
        axis=axis_fn(s.axis),
        start=_expr(s.start),
        step=step,
        body=tuple(rewrite(c, rename, sigma, axis_fn) for c in s.body),
        unroll=s.unroll,
        end=_expr(s.end) if s.end is not None else None,
        seed=s.seed,
    )


@_rewrite_kind.register
def _(s: Cond, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return Cond(
        cond=_rename_ssa_vars_in_expr(sigma.apply(s.cond), rename),
        body=tuple(rewrite(c, rename, sigma, axis_fn) for c in s.body),
        else_body=tuple(rewrite(c, rename, sigma, axis_fn) for c in s.else_body),
    )


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
    renamed = rewrite(stmt, lambda nm: alias.get(nm, nm), Sigma.IDENTITY, _axis_identity)
    bodies = stmt.nested()
    if not bodies:
        return renamed
    # ``rewrite`` just descended into the child scopes under the full alias. Redo each one with the
    # names that scope re-binds pruned out, and put those bodies back.
    inner = []
    for b in bodies:
        pruned = {k: v for k, v in alias.items() if k not in b.ssa_defs}
        inner.append(Body(tuple(rename_free(c, pruned) for c in b)))
    return renamed.with_bodies(tuple(inner))


# ---------------------------------------------------------------------------
# simplify — ctx-driven Expr simplification, threading axis ranges
# ---------------------------------------------------------------------------


@singledispatch
def simplify(stmt: Stmt, ctx: SimplifyCtx) -> Stmt:
    # Default: no Expr fields to simplify (Assign / Accum / Init).
    return stmt


@simplify.register
def _(s: Load, ctx: SimplifyCtx) -> Stmt:
    return Load(names=s.names, input=s.input, index=tuple(e.simplify(ctx) for e in s.index), dtype=s.dtype)


@simplify.register
def _(s: Write, ctx: SimplifyCtx) -> Stmt:
    return Write(
        output=s.output,
        index=tuple(e.simplify(ctx) for e in s.index),
        values=s.values,
        value_dtype=s.value_dtype,
        atomic=s.atomic,
        swizzle=s.swizzle,
    )


@simplify.register
def _(s: Select, ctx: SimplifyCtx) -> Stmt:
    """Predicates simplified, then every branch the constants DECIDE is dropped.

    Branches are ordered and the last one is the else, so a predicate that folds to false is
    unreachable and one that folds to true is the else from there on. Keeping such a branch costs
    far more than its own arithmetic: ``Select.deps`` names its value, so the tree-wide prune holds
    the whole producer cone alive, and a reduce inside a cone nothing can select does not read the
    output sweep -- which is what makes ``promoted_sweep`` refuse the grid and leave the kernel
    sweeping every cell in one block. Fusing a scatter at a literal coordinate decides a branch this
    way at every read it reaches, so this is the ordinary case, not a corner one.
    """
    branches = [SelectBranch(b.value, b.select.simplify(ctx)) for b in s.branches]
    kept: list[SelectBranch] = []
    for branch in branches[:-1]:
        if not isinstance(branch.select, Literal):
            kept.append(branch)
        elif branch.select.value:
            return Select(name=s.name, branches=(*kept, branch))
    return Select(name=s.name, branches=(*kept, branches[-1]))


@simplify.register
def _(s: Loop, ctx: SimplifyCtx) -> Stmt:
    inner = extend_simplify_ctx(ctx, s.axis)
    return Loop(
        axis=s.axis,
        body=tuple(simplify(c, inner) for c in s.body),
        unroll=s.unroll,
        seed=s.seed,
    )


@simplify.register
def _(s: StridedLoop, ctx: SimplifyCtx) -> Stmt:
    inner = extend_simplify_ctx(ctx, s.axis)
    step = s.step.simplify(ctx) if isinstance(s.step, Expr) else s.step
    return StridedLoop(
        axis=s.axis,
        start=s.start.simplify(ctx),
        step=step,
        body=tuple(simplify(c, inner) for c in s.body),
        unroll=s.unroll,
        end=s.end.simplify(ctx) if s.end is not None else None,
        seed=s.seed,
    )


@simplify.register
def _(s: Cond, ctx: SimplifyCtx) -> Stmt:
    return Cond(
        cond=s.cond.simplify(ctx),
        body=tuple(simplify(c, ctx) for c in s.body),
        else_body=tuple(simplify(c, ctx) for c in s.else_body),
    )


# Tile-IR Stmt registrations were DEMOLISHED along with the tile IR; pending
# rebuild.


def has_contraction_tail(stmts) -> bool:
    """True if the post-reduce tail contracts over a NEW free axis — a ``Loop`` whose body holds an
    inner reduce ``Loop``. This is the fused norm→linear shape, distinguished from a plain softmax
    tail (a single sweep over the SAME axis). ``Body.accums`` supplies the deep accumulator scan.

    A statement-SHAPE predicate, so it lives beside :func:`projection_distributes` rather than in
    the scheduler that asks: the reduce tiers read it to price a tail, and the shared-row stage
    gate to decide there is one to share a row with."""
    for s in stmts:
        if isinstance(s, Loop) and any(isinstance(c, Loop) and Body(c.body).accums for c in s.body):
            return True
        if any(has_contraction_tail(list(b)) for b in s.nested()):
            return True
    return False


def projection_distributes(body, states: tuple[str, ...]) -> bool:
    """True if the kernel's projection epilogue is a **linear-homogeneous** map of the carried
    state(s) — i.e. it distributes over the atomic-add combine, so applying it to each CTA's
    partition before the ``atomicAdd`` equals applying it once after the cross-CTA sum
    (``Σ c·xₛ = c·(Σ xₛ)``). A bare state write (``proj = id``) trivially distributes; a constant
    *scale* — ``mean``'s ``/N`` — does; an additive offset (a fused bias), a nonlinear unary
    (``relu`` / ``reciprocal`` of the *state*), or a product of two state-derived values do NOT.

    Conservative forward dataflow: ``linear`` is the set of SSA names that are a
    linear-homogeneous function of the state. Multiplication by or division by a state-independent
    operand grows this set; division requires the state in the numerator. Any other op that consumes a
    ``linear`` value — or any projection stmt we can't reason about — refuses. The final ``Write``
    must store only ``linear`` values."""
    linear = set(states)
    for s in body:
        if isinstance(s, Write):
            return all(v in linear for v in s.values)
        if isinstance(s, Load):
            continue  # reads memory (the count / a per-output operand) — state-independent
        if not isinstance(s, Assign):
            return False  # an unfamiliar projection stmt — can't prove distributivity
        hot = [a for a in s.args if a in linear]
        if not hot:
            continue  # state-independent — a constant w.r.t. the split
        if len(hot) == 1 and (s.op.name == "multiply" or s.op.name == "divide" and s.args[0] in linear):
            linear.add(s.name)  # state · constant or state / constant — still linear-homogeneous
            continue
        return False  # offset / inverse / nonlinear of a state value breaks distributivity
    return False  # no Write reached
