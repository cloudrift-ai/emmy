"""``Fold`` — the ONE reduce term: ``reduce(⊕) ∘ map(f)`` in the λ-foldMap spelling.

The whole stored vocabulary of Tile IR, and a PURE term throughout: an optional iteration
``axis``, a pure ``lift`` :class:`~emmy.compiler.ir.pure.lam.Lambda`, the monoid's ``(init, base)``
pair with the optional ``twist`` recipe that conjugates it, and a tuple of ``operands`` — the
closed inputs, each an edge bound positionally to a lift param. Every reading (the ⊕ itself, the
map at zero axes, the bilinear :class:`ContractionView`, the :class:`SlabView` leaf, the serial
step) is DERIVED from those params; nothing else is stored.

The fold's algebra is the twisted monoid of :mod:`~emmy.compiler.ir.pure.twist`, spelled in one
vocabulary with the recipes: ``lift`` is the per-element contribution to the BASE state, ``base``
the base monoid's componentwise ⊕, and ``twist`` the recipe carrying the bijection ``psi`` onto the
stable carrier. The ⊕ the fold actually folds with is then derived —
``combine(x, y) = psi(psi_inv(x) base psi_inv(y))`` — in the stable spelling the recipe certifies,
and ``twist=None`` is the planar case where ``psi`` is the identity and ``combine`` IS ``base``.

Nothing here is a :class:`~emmy.compiler.ir.stmt.Statement`. A composed step — flash's ``Σ Q·K``
ahead of its ``Σ_j P·V``, split-K's sliced contraction — is reached through ``operands``, and its
POSITION in the emitted nest is produced by the derivation (:meth:`Fold.lower` places every term
at the shallowest scope binding its free coordinates, operands ahead of their readers), not by
sitting in a statement list. The term becomes statements in exactly one place, :meth:`Fold.lower`,
which also places the kernel's boundary stores, each after the term defining its value.
See ``ir/ARCHITECTURE.md``, "Pure terms vs statements".

The schedule is deliberately absent: an accepted, site-indexed ``Schedule`` lives on the
``TileOp`` boundary (``ir/tile/ir.py``), so the term is IMMUTABLE across the whole schedule search
and kernel identity — the Loop IR the term lowers to — is the algebra alone.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace

import emmy.emmy_ir as _ir
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expression import Var
from emmy.compiler.ir.pure._fields import MISSING, declare
from emmy.compiler.ir.pure.lam import Lambda
from emmy.compiler.ir.pure.twist import Recipe, Twist
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Assign, Body, Load, Loop, OutputSpec, Statement, free_names
from emmy.utils import cached_method

#: The symbolic extents of the program being compiled. The launch binds them, so a term reads one free — like a
#: coordinate the grid binds — and never as a value some level defines: a scatter's guard compares its index with its
#: destination's extent. The pipeline binds them per graph (:func:`binding_extents`).
EXTENTS: ContextVar[frozenset[str]] = ContextVar("EXTENTS", default=frozenset())


@contextmanager
def binding_extents(names):
    """Read ``names`` as symbolic extents, on top of those already bound, for the enclosed block."""
    token = EXTENTS.set(EXTENTS.get() | frozenset(names))
    try:
        yield
    finally:
        EXTENTS.reset(token)


@dataclass(frozen=True)
class SlabView:
    """A Fold's reading as one gmem read — a leaf evaluated over the coordinates it indexes.

    ``load`` is the read itself. No operands, no monoid, a body of one ``Load``: what
    ``isinstance(edge, Load)`` used to ask, back when a statement could sit in ``operands``.
    Placement never offers one as a seam — there is nothing to materialize that the load does not
    already do.
    """

    load: Load


@dataclass(frozen=True)
class ContractionView:
    """A Fold's BILINEAR reading, as geometry: the axis its operands share and the free axis each
    one contributes.

    ``a[m,k] × b[n,k]`` reads as ``axis=k, left=m, right=n``. That is the whole of the recognition
    — no algebra walk, no product-argument matching, no canonical form to compare against. Which
    of ``left`` / ``right`` is physically M or N is the PLACEMENT's answer, not the term's.
    """

    axis: str
    left_axes: frozenset[str]
    """The free axes A carries and B does not — the output role it strides, plus any broadcast
    batch coordinate riding A alone (``x[b, m, k]`` against ``w[n, k]``): which of them is the
    row is the placement's answer, the others are grid offsets."""
    right_axes: frozenset[str]
    """The free axes the streamed operand carries alone. Empty for a MATVEC, whose B is a vector
    over the reduction alone: a contraction does not stop being one for want of a second output
    axis; whether the pair can be ORIENTED as (m, n) is the placement's question, not
    recognition's. A pair with NO output role on either side (a row's dot product with itself,
    the RMS statistic spelled as two casts of one load) is not a contraction at all: it reads as a
    planar reduce."""
    shared_axes: frozenset[str] = frozenset()
    """The free axes BOTH operands read besides the reduction: a batch the pair rides (attention's
    ``b, h``), a split-K partition coordinate, a value-dead reshape residue — or, when a side
    carries no role of its own, the row that side's every element moves with. Which of those a
    shared axis is takes the kernel's extents to tell, so it is the tile's question
    (:meth:`TileOp.contracts`), not the term's."""
    product: ElementwiseImpl | None = None
    """The ⊗ this contraction multiplies its operand pair with."""
    plus: ElementwiseImpl | None = None
    """The commutative-monoid ⊕ the products fold through. Together with :attr:`product` this is
    the semiring INSTANCE — the mma tier gates on ``(multiply, add)``, and a new instance reads
    back here without change."""
    b_trans: bool = False
    """Whether the streamed operand is stored N-major — its reduction axis LAST, ``B[n, k]``,
    against the canonical ``B[k, n]``. A gmem LAYOUT fact, so it is meaningful only for a
    materialized slab; a computed B answers ``False``. A stored on the same side is not a
    question: ``operands[0]`` is A by canonical form, k-last, which formation guarantees by
    orienting the pair."""
    channel: int = 0
    """Which carried state the products fold into — the position of this reading among the term's
    results. ``0`` for a plain contraction, whose only state IS the accumulator; a twisted
    carrier's expectation sits past the states its pivot and denominator carry, and the emission
    that accumulates into it needs the position to name it."""

    @property
    def left(self) -> str | None:
        """A's one own axis, or ``None`` when it carries none or several (a broadcast batch)."""
        return next(iter(self.left_axes)) if len(self.left_axes) == 1 else None

    @property
    def right(self) -> str | None:
        """B's one own axis, or ``None`` when it carries none or several."""
        return next(iter(self.right_axes)) if len(self.right_axes) == 1 else None


@dataclass(frozen=True)
class ReductionView:
    """A Fold's reading as a MONOID fold — the algebra its combine spells, read off the lambda.

    ``states`` are the carried state's names (the combine's results), ``other`` the names the
    combine binds its second operand to, ``terms`` the injection singleton (the lift's results,
    one per component). ``ops`` is the componentwise op vector when the combine has that shape —
    every result ``sᵢ = ⊕ᵢ(sᵢ, sᵢ′)`` independently, a PLANAR fold (sum, max, mean) — and ``None``
    for a TWISTED one (online softmax's rescaling program): the one discrimination every partition
    legality question asks. No family name: the program itself is the reading, and
    :meth:`Fold.merge` applies it.
    """

    axis: str
    states: tuple[str, ...]
    other: tuple[str, ...]
    terms: tuple[str, ...]
    ops: tuple[ElementwiseImpl, ...] | None

    @property
    def twisted(self) -> bool:
        return self.ops is None


@dataclass(frozen=True)
class AffineView:
    """A carried state's step read as AFFINE in the state, acting column-wise: every carrier read
    keeps the cell's own coordinate at every position but ``mixed``, so the step is ``S ← A·S + b``
    with ``A`` a matrix over the mixed axis alone, applied to every column of the block alike, and
    two steps compose as affine maps. ``kept`` is the own position a probe packs basis vectors
    along — the last one — so ``A`` is read off one walk of the step from the identity seed. A step
    that reads its own cell alone has no mixed position (``None``): ``A`` is diagonal, and one walk
    from a seed of ones reads it."""

    state: str
    mixed: int | None
    kept: int | None


class _Fold:
    """What Python still adds to the crate's :class:`Fold` (``emmy.emmy_ir.Fold``): the readings, the fusion and the
    lowering not yet in ``emmy-pure``, installed on the class below. The term itself — its fields, its formation
    checks, ``exposes``, ``bindings``, ``applied``, ``axis``, ``free_axes``, ``carries``, ``combine``, ``canonical``
    and ``as_carrier_read`` — is the crate's.

    The ONE reduce term — ``reduce(⊕) ∘ map(f)``, the typed successor of the annotated reduce
    ``Loop``. It splits the reduce's **algebra** (the base monoid ``base``, plus the ``twist``
    recipe whose ``psi`` conjugates it into the rescaling program online-softmax / flash folds
    with — :attr:`combine` derives that ⊕, and a planar ``sum`` / ``max`` / ``mean`` is the
    identity case) from its **structure** (the axis the lift binds and the per-element ``step`` it
    folds). Every reading is **derived** from the stored params (:attr:`combine`,
    :meth:`as_contraction`, :meth:`as_slab`, :meth:`as_reduction`, :meth:`step`), never stored, and the loop nest is
    **synthesized on demand** (:meth:`lower`), never stored — so the same term tiles under any
    :class:`~emmy.compiler.ir.schedule.Reduce`, which is not a field here: the reduce partition is
    a site choice in ``TileOp.schedule``, read through ``ops.Sched``.

    Everything a term reads arrives through its ``operands``, bound POSITIONALLY to the lift's
    params (:attr:`bindings`) — a gmem read as a slab, a nested reduce as the term itself, a
    projection as a zero-axis term — so a reduce that reads another (flash's ``Σ Q·K`` score ahead
    of its ``Σ_j P·V``) is one term with the other among its operands, and its position in the
    emitted nest is :meth:`lower`'s to place. A term holds no projection of its own: a bare reduce
    (``sum`` / ``max``) is the kernel root, and a reduce with a post-fold sweep (softmax / RMSNorm)
    is an operand of the zero-axis term whose lift is that projection.

    The reduce PARTITION (:class:`Reduce` — GRID split / BLOCK coop / REG ILP) is the schedule's,
    not the term's: it is selected for the term's site in ``TileOp.schedule`` and read through
    ``ops.Sched``, which is why ``lower`` cannot see it and ``identity_key(with_io=True, with_knobs=True)``
    stays byte-identical whichever partition the fork picked."""

    def fuse(self, recipe: Recipe, axes) -> Fold | None:
        """This reduce fused onto the reduce it reads, by ``recipe`` — one fold carrying both
        states under the recipe's twisted ⊕ — or ``None`` when no channel of the recipe clicks.
        The ONE generic algorithm over the declarative recipes (:mod:`~emmy.compiler.ir.pure.twist`).
        The pivot is found among this term's operands: a reduce folding the recipe's pivot ⊕, or a
        fold this recipe already fused (whose channel 0 is the pivot). The pivot's state is the
        lift param bound to it, positionally; the score is the sub-cone of the lift alpha-equal to
        the pivot's own per-element map, operand for operand (a projection's component by its own
        cone); and what remains of the lift with that cone cut out, its params in role order, must
        equal a channel's pattern by canonical form. A click gives the role-to-name map, and the
        fused fold is the recipe instantiated by renaming: the pivot's operands plus what the
        extras bind, the pivot's lift with the channel's injection appended, the states
        concatenated, the recipe's ⊕ program over them. The same recipe fuses online softmax and
        flash attention alike; the caller rewrites the tree's operands onto the result.
        """
        view = self.as_reduction()
        if view is None or self.observe is not None or view.ops is None or len(view.states) != 1:
            return None
        for pivot in self.operands:
            pview = pivot.as_reduction()
            if pview is None or pivot.observe is not None:
                continue
            if pview.ops is not None:
                if len(pview.states) != 1 or pview.ops[0].reduce_canon != recipe.pivot:
                    continue
            elif pivot.twist is None or pivot.twist.recipe is not recipe:
                continue  # a twisted pivot must already carry THIS recipe: another one's carrier is not its pivot
            if axes[self.axis].extent != axes[pivot.axis].extent or axes[self.axis].window != axes[pivot.axis].window:
                continue
            fused = self._twist(pivot, recipe, axes)
            if fused is not None:
                return fused
        return None

    def _twist(self, pivot: Fold, recipe, axes) -> Fold | None:
        view, pview = self.as_reduction(), pivot.as_reduction()

        def cone(fold: Fold, name: str) -> tuple:
            # The closed cone defining ``name`` in the lift, and the VALUE each of its params binds
            # (``None`` for a coordinate: a reduce's state binds that term, a projection's result
            # binds its own cone).
            for param, edge, index in fold.bindings:
                # A param this term only PASSES THROUGH is the edge's own cone: the value is the
                # same whether the statements sit in the lift or in the operand a factoring moved
                # them to, and a match on the score must not turn on which side of that line it is.
                if param == name and edge.axis is None:
                    return cone(edge, edge.exposes[index])
            by_param = {param: edge for param, edge, _ in fold.bindings}
            fn = fold.lift.cone(name)
            values = tuple(
                None if (edge := by_param.get(param)) is None else edge if edge.axis is not None else cone(edge, param)
                for param in fn.params
            )
            return fn, values

        def alike(x: Fold, y: Fold) -> bool:
            """Compare values under their actual coordinate scopes, independently of a lift's
            coordinate-parameter order and the names of the axes its operands bind."""
            from emmy.compiler.ir.stmt import canonicalize_identity  # noqa: PLC0415

            if self.axis != pivot.axis:
                x = x.rewrite(lambda name: name, Sigma({self.axis: Var(pivot.axis)}))
            scope = tuple(axes.values())
            return canonicalize_identity(x.lower(axes=scope)) == canonicalize_identity(y.lower(axes=scope))

        def same(a: tuple, b: tuple) -> bool:
            if a[0].canonical() != b[0].canonical() or len(a[1]) != len(b[1]):
                return False
            for x, y in zip(a[1], b[1], strict=True):
                if (x is None) != (y is None) or isinstance(x, tuple) != isinstance(y, tuple):
                    return False
                if x is not None and (not same(x, y) if isinstance(x, tuple) else not alike(x, y)):
                    return False
            return True

        bound = tuple((param, edge) for param, edge, _ in self.bindings)
        by_param = dict(bound)
        g = next(param for param, edge in bound if edge is pivot)
        pivot_params = {param for param, edge in bound if edge is pivot}
        score = cone(pivot, pivot.lift.results[0])
        candidates = [p for p in self.lift.params[1:] if p not in pivot_params] + [n for s in self.lift.body for n in s.defines()]
        for x in candidates:
            found = cone(self, x)
            # The cone may read coordinates free (the causal mask's ``key <= query``): this fold's
            # own axis is the pivot's once fused, so it is COMPARED under the pivot's name, while
            # the residual below is cut by the cone's own statements.
            spelled = found if self.axis == pivot.axis else (found[0].rename({self.axis: pivot.axis}), found[1])
            if not same(spelled, score):
                continue
            # What remains of the lift with the score cut out, closed over what it still reads —
            # ``(score, pivot, *extras)`` in role order, the extras operand-bound values.
            residual = tuple(stmt for stmt in self.lift.body if stmt not in found[0].body)
            fn = Lambda.closing((x, g), Body(residual), self.lift.results)
            extras = fn.params[2:]
            if any(p not in by_param or p in pivot_params for p in extras):
                continue
            # A channel whose base ⊕ is this fold's and whose pattern is what remains.
            plus = view.ops[0].reduce_canon
            channel = next(
                (
                    c
                    for index, c in enumerate(recipe.channels)
                    if c.pattern is not None and recipe.base[1 + index] == plus and fn.canonical() == c.pattern.canonical()
                ),
                None,
            )
            if channel is None:
                continue
            # Instantiate: every operand an extra binds joins, its axis spelled as the pivot's.
            old, new = self.axis, pivot.axis
            extra_edges = tuple(dict.fromkeys(edge for p, edge in bound if p in extras))
            extra_params = tuple(p for edge in extra_edges for p, e in bound if e is edge)
            if old != new:
                extra_edges = tuple(
                    edge.rewrite(
                        lambda n: n,
                        Sigma({old: Var(new)}),
                        lambda a, old=old, new=new: replace(a, name=new) if a.name == old else a,
                    )
                    for edge in extra_edges
                )
            # The carrier's states: the pivot's, then every channel in recipe order — the matched
            # one is this fold's own state, one without a pattern a state the recipe adds. Each
            # takes the recipe's BASE contribution for it (``lift``'s own result), instantiated at
            # the score and this channel's extras. STABLE coordinates throughout: the contribution
            # stored is the recipe's authored INJECTION — the singleton in the carrier's own state
            # space, where the pivot IS the score and softmax's ``exp(s)·v`` has already simplified
            # to ``v``. The base contribution is a derived reading (:meth:`based`) a matcher asks
            # for and nothing emits. Seeds are the carrier's, as the states are.
            state = view.states[0]
            score = pivot.lift.results[0]
            added = [
                (state if c is channel else f"{state}__{c.name}", index, self.init[0] if c is channel else c.init)
                for index, c in enumerate(recipe.channels)
                if c is channel or c.pattern is None
            ]
            roles = {recipe.lift.params[0]: score, **dict(zip(channel.pattern.params[2:], extras, strict=True))}
            arity = sum(len(edge.exposes) for edge in pivot.operands)
            held = [
                *zip(pivot.operands, _slots(pivot.lift.params[1 : 1 + arity], pivot.operands), strict=True),
                *zip(extra_edges, _slots(extra_params, extra_edges), strict=True),
            ]
            body, results, inits = list(pivot.lift.body), list(pivot.lift.results), list(pivot.init)
            for name, index, init in tuple(added):
                taken = {n for stmt in body for n in stmt.defines()} | {p for _, slot in held for p in slot}
                injection = recipe.channels[index].injection
                names = {param: roles[param] for param in injection.params}
                # Statement by statement, so a channel BINDS what the body already computes rather
                # than spelling it again: softmax's denominator and its expectation share the whole
                # weight prefix, and instantiating each cone whole put two ``exp(s)`` in one lift.
                have = {stmt.value: stmt.name for stmt in body if isinstance(stmt, Assign)}
                for stmt in injection.body:
                    spelled = stmt.rewrite(lambda read, names=names: names.get(read, read))
                    if isinstance(spelled, Assign) and (prior := have.get(spelled.value)) is not None:
                        names[stmt.name] = prior
                        continue
                    fresh = stmt.name if stmt.name not in taken else f"{name}__{stmt.name}"
                    names[stmt.name] = fresh
                    spelled = spelled.replace(name=fresh)
                    body.append(spelled)
                    taken.add(fresh)
                    if isinstance(spelled, Assign):
                        have[spelled.value] = fresh
                if name in pview.states:
                    slot = pview.states.index(name)
                    bound_roles = {**roles, recipe.lift.params[0]: x}
                    injected = injection.rename(lambda read, bound_roles=bound_roles, name=name: bound_roles.get(read, f"{name}__{read}"))
                    candidate = self.replace(
                        lift=Lambda.closing(self.lift.params, Body((*found[0].body, *injected.body)), injected.results)
                    )
                    contribution = cone(candidate, injected.results[0])
                    if old != new:
                        contribution = (contribution[0].rename({old: new}), contribution[1])
                    # A shared dependent can reach the pivot again through another score cone.
                    # Reuse its state only when both copies inject the same value into the same channel.
                    if (
                        slot == 0
                        or pivot.twist.channels[slot - 1] != index
                        or pivot.init[slot] != init
                        or not same(cone(pivot, pivot.lift.results[slot]), contribution)
                    ):
                        raise ValueError(f"conflicting definitions of twisted state {name!r}")
                    added.remove((name, index, init))
                    continue
                results.append(names[injection.results[0]])
                inits.append(init)
            if not added:
                return pivot
            # An operand the lift no longer names is nobody's — the recipe's own contribution
            # replaced the statements that read it.
            read = {name for stmt in body for name in stmt.deps()} | set(results)
            held = [entry for entry in held if not read.isdisjoint(entry[1])]
            # B SECOND. A leads by canonical form; a bilinear channel's other factor must follow it,
            # because staging and both atom tiers take B from ``operands[1]`` rather than asking the
            # channel. An operand with no free coordinates — attention's scale — is no channel's
            # factor at all, so it sorts behind the streamed value instead of between the pair.
            held = [held[0], *sorted(held[1:], key=lambda entry: not entry[0].free_axes)]
            # The carrier is the RECIPE's own vector: the pivot, then the channels it holds in the
            # order the recipe declares them, whatever order the tree happened to fuse them in.
            # Softmax's is (m, D, O) whether the denominator or the expectation clicked first, so a
            # state's position and its channel agree with ``Recipe.base`` and ``Recipe.lift``.
            channels = (*(() if pivot.twist is None else pivot.twist.channels), *(index for _, index, _ in added))
            order = sorted(range(len(channels)), key=lambda slot: channels[slot])

            def by_channel(pivot_slot, tail, order=order):
                """The pivot's slot, then the channel slots in the recipe's declared order."""
                return (pivot_slot, *(tail[slot] for slot in order))

            states = by_channel(pview.states[0], (*pview.states[1:], *(name for name, _, _ in added)))
            base_ops = pivot.base.components()
            ops = by_channel(base_ops[0], (*base_ops[1:], *(recipe.base[1 + index] for _, index, _ in added)))
            init = by_channel(inits[0], tuple(inits[1:]))
            channelled = by_channel(results[0], tuple(results[1:]))
            lift = Lambda(
                params=(pivot.axis, *(p for _, slot in held for p in slot), *pivot.lift.params[1 + arity :]),
                body=Body(tuple(Body(tuple(body)).backward_cone(tuple(dict.fromkeys(channelled))).members)),
                results=tuple(channelled),
            )
            return Fold(
                operands=tuple(edge for edge, _ in held),
                lift=lift,
                init=tuple(init),
                base=Lambda.componentwise(ops, states),
                twist=Twist(recipe=recipe, channels=tuple(channels[slot] for slot in order)),
            )
        return None

    def read_components(self, stored: frozenset[str] = frozenset()) -> dict[int, tuple[str, ...]]:
        """The components each term in this tree supplies to all its readers and boundary stores.

        Propagate demand through narrowed lifts. Shared terms keep the union of their readers'
        requests, so lowering never emits overlapping carriers with duplicate accumulator names.
        Interior stores keep the defining subtree's reads, without demanding its enclosing results.
        """
        taken: dict[int, tuple[str, ...]] = {}
        pending = [(self, self.exposes)]
        while pending:
            term, asked = pending.pop()
            wanted = set(taken.get(id(term), ())) | set(asked)
            names = tuple(name for name in term.exposes if name in wanted)
            if taken.get(id(term)) == names:
                continue
            taken[id(term)] = names
            narrowed = term if _writes_under(term, stored) else term.exposing(names) if names else None
            read = narrowed.step().ssa_uses | set(narrowed.exposes) if narrowed is not None else set()
            for edge in term.operands:
                keep = tuple(name for name in edge.exposes if name in read or name in stored)
                pending.append((edge, keep))
        return taken

    @cached_method
    def lower(self, bound: frozenset[str] | None = None, stores: tuple[OutputSpec, ...] = (), axes: tuple[Axis, ...] = ()) -> Body:
        """Flatten this term to the Loop IR nest the materializer expands, the kernel's boundary
        ``stores`` placed in it.

        ``bound`` names the coordinates the CALLER binds — the kernel grid, an enclosing loop's
        scope; ``None`` binds every free coordinate (the open body a term spells inside an
        enclosing scope) and ``frozenset()`` the closed program; ``axes`` is the kernel's axis table
        — the extent and window of every axis the tree binds or reads, which a term carries none
        of: a reduce loop takes its axis from it, and the closed program the coordinates it opens
        loops for. One rule then places every term
        of the tree, read straight off the representation: a term is materialized at the
        SHALLOWEST scope on its path that binds all of its :attr:`free_axes`, a value of the tree it
        reads by name (the expert a gathered row belongs to) standing for the coordinates of the term
        that defines it. The scopes are the
        plain loops this term opens for the free coordinates left unbound — outermost the
        coordinate the most terms are evaluated over, so what siblings share is hoisted above
        them, ties in declaration order — and the reduce loop of each term on the way down. That
        is the whole of the hoist: a declaration compared against the scopes above, not a body
        walked for free names. An operand that does not index its reader's reduce axis lands
        ahead of that loop; one that reads no coordinate a deeper loop binds lands ahead of every
        such loop, past the term that reads it.

        A boundary store follows the term that DEFINES the value it writes, at that term's scope:
        a projection's result after its step, a reduce's carried state after its loop, an observed
        per-step value inside the loop after the observer. A store alone evaluated over a
        coordinate no term declares (a broadcast, ``o[j] = acc``) opens that coordinate under its
        term, the spec's ``sweep`` naming the axis. Nothing is walked for the place a store goes:
        the term defining its value was just placed.

        The free loops form a TREE, not a chain. A term's operands sit on ITS path — the shallowest
        prefix of its own loops that binds them — so what a term reads is always in scope; only a
        wrapper with no step of its own leaves its operands free to take their own paths, which is
        how two output sweeps no term shares become sibling loops. Operands are placed before
        the statements that read them, a reduce term's step follows its operands inside the loop
        that binds its axis, and a SHARED term — one object reached through several operand
        positions — defines its names once per scope.

        The ONE lowering spelling — every consumer of a term's statements calls this. Memoized on
        the term per binding.
        """
        # The launch binds the symbolic extents, as the caller binds its coordinates.
        bound = (self.free_axes if bound is None else bound) | EXTENTS.get()
        # The stores are spelled in this term's own vocabulary; they render with the lift.
        spelled = dict(zip(self.lift.params, self.applied.params, strict=True))
        stores = tuple(
            replace(
                spec,
                write=spec.write.rewrite(lambda name: spelled.get(name, name)),
                when=None if spec.when is None else spec.when.rename(lambda name: spelled.get(name, name)),
            )
            for spec in stores
        )
        coordinates: dict[str, Axis] = {axis.name: axis for axis in axes}
        declared: list[str] = []  # the order the tree first reads its coordinates — the tie order below
        internal: set[str] = set()  # the axes terms of the tree bind: a store under a reduce loop may index one
        readers: dict[str, int] = {}
        origin: dict[str, tuple[int, str]] = {}  # a value's defining term and what it is there
        terms: dict[int, Fold] = {}
        seen: set[int] = set()
        pending = [self]
        while pending:
            term = pending.pop()
            if id(term) in seen:
                continue
            seen.add(id(term))
            terms[id(term)] = term
            declared.extend(name for name in term.lift.params[(term.axis is not None) + len(term.bindings) :] if name not in declared)
            if term.axis is not None:
                internal.add(term.axis)
            if term.carries:  # a carried state's cells, and what its step sweeps, are its own loops (``place``)
                internal.update(term.cells, *(edge.free_axes for edge in term.operands))
            for name in term.free_axes:
                readers[name] = readers.get(name, 0) + 1
            if term.base is None:
                origin.update((name, (id(term), "step")) for stmt in term.lift.body for name in stmt.defines())
            else:
                origin.update((name, (id(term), "state")) for name in term.base.results)
                if term.observe is not None:
                    origin.update((name, (id(term), "observed")) for name in term.observe.results)
                if term.carries:  # a per-step output the step streams, stored inside the loop like an observed value
                    origin.update((name, (id(term), "observed")) for name in term.exposes[len(term.base.results) :])
            pending.extend(reversed(term.operands))
        owned: dict[tuple[int, str], list[OutputSpec]] = {}
        # The order the terms are placed: a term's operands before its own step (:func:`place`).
        placed_at: dict[int, int] = {}

        def rank(term: Fold) -> None:
            if id(term) in placed_at:
                return
            for edge in term.operands:
                rank(edge)
            placed_at[id(term)] = len(placed_at)

        rank(self)
        # A store's coordinate the tree COMPUTES — a scatter's row, a value some term defines — is read
        # like a stored value: its term is kept, and the store follows the last of the terms it reads.
        indexed = frozenset(
            name
            for spec in stores
            for expression in _store_reads(spec)
            for name in expression.free_vars()
            if name in origin and name not in coordinates and name not in bound and name not in internal
        )
        stored = frozenset(name for spec in stores for name in spec.write.values) | indexed
        taken = self.read_components(stored)

        def placed(term: Fold) -> tuple[Fold, ...]:
            # An interior store retains its defining subtree, without exporting every state of
            # its enclosing carrier to a cut workspace.
            return tuple(
                edge if _writes_under(edge, stored) else edge.exposing(taken[id(edge)])
                for edge in term.operands
                if taken[id(edge)] or _writes_under(edge, stored)
            )

        for spec in stores:
            key = origin.get(spec.write.values[0])
            assert key is not None and all(origin.get(name) == key for name in spec.write.values), (
                f"a store over {spec.write.values} writes values no one term defines"
            )
            rows = {origin[name] for expression in _store_reads(spec) for name in expression.free_vars() if name in indexed}
            owned.setdefault(max((key, *rows), key=lambda k: placed_at[k[0]]), []).append(spec)
        for spec in stores:
            for axis in spec.sweep:
                coordinates.setdefault(axis.name, axis)
        declared.extend(name for name in coordinates if name not in declared)
        read = self.free_axes | (
            {name for spec in stores for expression in _store_reads(spec) for name in expression.free_vars()} - indexed
        )
        if missing := read - bound - internal - set(coordinates):
            raise ValueError(f"lower: no extent for coordinates {sorted(missing)} — the closed program takes them as axes")
        # A name a nested reduce binds is still a free coordinate here when the tree reads it free
        # (a stat over the row beside a slab of the same row): the reduce loop shadows it inside.
        opened = sorted(
            (name for name in coordinates if name in read and name not in bound and (name not in internal or name in self.free_axes)),
            key=lambda name: (-readers.get(name, 0), declared.index(name)),
        )

        class _Nest:
            """One scope's TREE of free loops: the coordinates it may open (outermost the one the
            most terms are evaluated over), what the scope already binds, and the statements at
            each path. The kernel has one; a carried state's loop holds one of its own for the
            terms evaluated inside the step, its cells first."""

            def __init__(self, opened: list[str], bound: frozenset[str]) -> None:
                self.opened, self.bound = opened, bound
                self.nest: dict[tuple[str, ...], list[Statement]] = {(): []}

            def path_of(self, free: frozenset[str], path: tuple[str, ...] | None) -> tuple[str, ...]:
                # The free loops a term sits under: the shallowest prefix of its reader's path
                # binding its coordinates, or — unconstrained — those coordinates in the opened order.
                needed = {name for name in self.opened if name in free}
                if path is None:
                    return tuple(name for name in self.opened if name in needed)
                return next(path[:depth] for depth in range(len(path) + 1) if needed <= set(path[:depth]))

            def sink(self, path: tuple[str, ...]) -> list[Statement]:
                for depth in range(len(path) + 1):
                    self.nest.setdefault(path[:depth], [])
                return self.nest[path]

            def assemble(self, path: tuple[str, ...] = ()) -> Body:
                body = list(self.nest[path])
                for name in self.opened[self.opened.index(path[-1]) + 1 :] if path else self.opened:
                    if (*path, name) in self.nest:
                        body.append(Loop(axis=coordinates[name], body=self.assemble((*path, name))))
                return Body(body).coalesce()

        def evaluated_over(names) -> frozenset[str]:
            # The coordinates ``names`` are evaluated over: a coordinate is its own, and a value a term of the tree
            # computes that is no coordinate (a scatter's row, the expert a gathered row belongs to) stands for the
            # coordinates of that term.
            values = {name for name in names if name in origin and name not in coordinates and name not in bound and name not in internal}
            return frozenset(names - values).union(*(evaluated_over(terms[origin[name][0]].free_axes) for name in values))

        def attach(
            term: Fold,
            kind: str,
            target: list[Statement],
            node: tuple[str, ...],
            scope: frozenset[str],
            nest: _Nest,
            computed: bool | None = None,
        ) -> None:
            # The stores over what ``term`` defines as ``kind`` — its step's results, its carried
            # state, or its observer's per-step values — land right after it, in ``target``. ``computed``
            # selects the stores whose coordinates name a value some term computes (``True``), the others
            # (``False``), or every store (``None``).
            for spec in owned.get((id(term), kind), ()):
                if computed is not None and computed != bool(
                    {name for expression in _store_reads(spec) for name in expression.free_vars()} & indexed
                ):
                    continue
                # The store sits under the loops binding every coordinate it spells, and every coordinate a
                # value it names is evaluated over (a scatter's row, computed per cell of a sweep).
                read = {name for expression in _store_reads(spec) for name in expression.free_vars()}
                extra = evaluated_over(read) - scope
                if not extra:
                    target.append(spec.stmt())
                    continue
                assert extra <= set(nest.opened), (
                    f"a store reads coordinates {sorted(extra - set(nest.opened))} no term declares and no sweep names"
                )
                nest.sink((*node, *(name for name in nest.opened if name in extra))).append(spec.stmt())

        emitted: dict[tuple[int, int], tuple[Fold, list[Statement]]] = {}

        def place(term: Fold, loops: list[tuple[str, frozenset[str], list[Statement]]], path: tuple[str, ...] | None, nest: _Nest) -> None:
            # ``loops``: the reduce loops enclosing this position, outermost first, as (axis, scope, stmts).
            if any(axis in term.free_axes for axis, _, _ in loops):
                depth = next(depth for depth, (_, scope, _) in enumerate(loops) if evaluated_over(term.free_axes) <= scope)
                _, scope, stmts = loops[depth]
                loops, node = loops[: depth + 1], path
            else:
                node = nest.path_of(evaluated_over(term.free_axes if term.axis is not None else _needs(term)), path)
                scope, stmts, loops = nest.bound | set(node), None, []
            if term.axis is not None or term.step():
                target = stmts if stmts is not None else nest.sink(node)
                key = (id(term), id(target))
                if key in emitted:
                    return
                # Retain both objects: a completed inner scope's list otherwise permits id reuse.
                emitted[key] = term, target
            if term.axis is None:
                step = term.step()
                # A step reading nothing the tree binds or defines (a multi-output root's constant result)
                # needs no coordinate, so it sits at the top of its path.
                reads = {name for stmt in step for name in free_names(stmt)} - {name for stmt in step for name in stmt.defines()}
                early = step and stmts is None and not reads & {*term.lift.params, *origin}
                if early:
                    node = nest.path_of(frozenset(), path)
                    nest.sink(node).extend(step)
                    attach(term, "step", nest.sink(node), node, nest.bound | set(node), nest, computed=False)
                    step = ()
                # An operand the step never reads — a value only stored, beside the step — sits on its own path, not
                # under the coordinates of the operands the step combines.
                read = {id(edge) for param, edge, index in term.bindings if {param, edge.exposes[index]} & reads}
                kept = [edge for edge in term.operands if taken[id(edge)] or _writes_under(edge, stored)]
                for edge, operand in zip(placed(term), kept, strict=True):
                    place(edge, loops, node if step and id(operand) in read else path, nest)
                if early:
                    # A store whose coordinate an operand computes follows that operand, not the step alone.
                    attach(term, "step", nest.sink(node), node, nest.bound | set(node), nest, computed=True)
                if step:
                    target = stmts if stmts is not None else nest.sink(node)
                    target.extend(step)
                    attach(term, "step", target, node, scope, nest)
                return
            if term.carries:
                # A CARRIED STATE: its reduce loop encloses the loops over its cells, where the Loop
                # IR puts a carrying loop, and holds a nest of its own for the terms evaluated inside
                # the step — the cells first, then the coordinates a per-step output sweeps. Each
                # opens only where the caller has not bound it (a serial launch binds the axis, a
                # grid the cells). The step and the stores of its own per-step results land under the
                # cells; the state is never stored after the loop: a ``Carry`` is read per step or
                # not at all.
                if owned.get((id(term), "state")):
                    raise ValueError("a carried state is stored per step, never after its loop")
                inside = scope | {term.axis}
                reads = set().union(*(edge.free_axes for edge in term.operands))
                cells = tuple(cell for cell in term.cells if cell not in inside)
                extras = [name for name in declared if name in reads and name not in inside and name not in term.cells]
                if missing := [name for name in (term.axis, *cells, *extras) if name not in coordinates and name not in bound]:
                    raise ValueError(f"lower: no extent for {missing} — the kernel's axis table names a carried state's axis and cells")
                local = _Nest([*cells, *extras], inside)
                for edge in placed(term):
                    # An operand over the cells sits on the step's path, as any operand sits on its
                    # reader's; one a per-step output sweeps takes its own path beside the cells.
                    place(edge, [], cells if edge.free_axes <= inside | set(cells) else None, local)
                target = local.sink(cells)
                target.extend(term.step())
                attach(term, "observed", target, cells, inside | set(cells), local)
                body = local.assemble()
                outer = stmts if stmts is not None else nest.sink(node)
                if term.axis in bound:
                    outer.extend(body)
                else:
                    outer.append(Loop(axis=coordinates[term.axis], body=body))
                return
            inner: list[Statement] = []
            for edge in placed(term):
                place(edge, [*loops, (term.axis, scope | {term.axis}, inner)], node, nest)
            inner.extend(term.step())
            attach(term, "observed", inner, node, scope | {term.axis}, nest)
            target = stmts if stmts is not None else nest.sink(node)
            if term.axis not in coordinates:
                raise ValueError(f"lower: no extent for reduce axis {term.axis!r} — the kernel's axis table names it")
            target.append(Loop(axis=coordinates[term.axis], body=Body(inner).coalesce()))
            attach(term, "state", target, node, scope, nest)

        def _needs(term: Fold) -> frozenset[str]:
            # The coordinates a map's step is evaluated over: what it reads itself and what the operands it reads
            # are evaluated over — not those of an operand it only carries to a store.
            step = term.step()
            reads = {name for stmt in step for name in free_names(stmt)}
            needed = reads.union(*(edge.free_axes for param, edge, index in term.bindings if {param, edge.exposes[index]} & reads))
            return term.free_axes & needed

        root = _Nest(opened, frozenset(bound))
        place(self, [], None, root)
        return root.assemble()

    def rewrite(self, rename, sigma: Sigma = Sigma.IDENTITY, axis_fn=lambda axis: axis) -> Fold:  # noqa: ANN001
        """This term with every SSA name through ``rename``, every expression through ``sigma`` and every axis through
        ``axis_fn``, the way ``Statement.rewrite`` rewrites a statement: the lift and the monoid rename in lockstep
        (params track the operand names positionally, the combine's results ARE the accumulator names), and σ is
        applied hygienically to the term's own binder — the reduce axis this term binds is a different variable from
        any σ key spelled alike, so its mapping is dropped for the subtree. At zero axes there is no iteration var to
        rename and no monoid to thread. Operand edges are terms and rewrite through this same entry."""
        if self.axis is not None and self.axis in sigma.mapping:
            sigma = Sigma({name: value for name, value in sigma.mapping.items() if name != self.axis})
        operands = tuple(edge.rewrite(rename, sigma, axis_fn) for edge in self.operands)
        lead = (rename(self.axis),) if self.axis is not None else ()  # the iteration var — a slab has none

        def _param(name: str) -> str:
            # The environment tail may hold AXIS names, which rename through sigma (the body's own
            # coordinate substitution), not through the SSA renamer. Taking sigma's answer when it has
            # one keeps a param and every use of it in the body spelled the same way.
            mapped = sigma.get(name)
            return mapped.name if isinstance(mapped, Var) else rename(name)

        bound_count = len(lead) + len(self.bindings)
        substitutions = {
            p: mapped for p in self.lift.params[bound_count:] if (mapped := sigma.get(p)) is not None and not isinstance(mapped, Var)
        }
        needed = set().union(*(expression.free_vars() for expression in substitutions.values()))
        params = (*lead, *(_param(p) for p in self.lift.params[len(lead) : bound_count]))
        params += tuple(_param(p) for p in self.lift.params[bound_count:] if p not in substitutions or p in needed)
        body = Body(tuple(st.rewrite(rename, sigma, axis_fn) for st in self.lift.body))
        results = tuple(rename(r) for r in self.lift.results)
        # Only an expression substitution can introduce a coordinate absent from the original tail.
        # A self-referential substitution still keeps its old binder in its original position.
        lift = Lambda.closing(params, body, results) if substitutions else Lambda(params, body, results)
        base = self.base.rename(rename) if self.base is not None else None
        cells = tuple(_param(cell) for cell in self.cells)  # coordinates, like the environment tail
        observe = None
        if self.observe is not None:
            # The observer renames in lockstep: param 0 tracks the axis, the state params track the
            # combine's renamed results, the body/results are ordinary SSA material.
            observe = Lambda(
                params=(*lead, *(rename(p) for p in self.observe.params[1:])),
                body=Body(tuple(st.rewrite(rename, sigma, axis_fn) for st in self.observe.body)),
                results=tuple(rename(r) for r in self.observe.results),
            )
        return self.replace(operands=operands, lift=lift, base=base, observe=observe, cells=cells)


def _store_reads(spec: OutputSpec) -> tuple:
    """The expressions a boundary store reads: its coordinates and its guard."""
    return (*spec.write.index, *(() if spec.when is None else (spec.when,)))


def _writes_under(term: Fold, stored: frozenset[str]) -> bool:
    """Whether any term of ``term``'s subtree defines a value the kernel's boundary stores."""
    pending, seen = [term], set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if (set(node.exposes) | node.step().defined) & stored:
            return True
        pending.extend(node.operands)
    return False


def _slots(params: tuple[str, ...], edges) -> list[tuple[str, ...]]:
    """``params`` cut into one group per edge — the positional binding, one param per result component."""
    out, cursor = [], 0
    for edge in edges:
        out.append(tuple(params[cursor : cursor + len(edge.exposes)]))
        cursor += len(edge.exposes)
    return out


Fold = _ir.Fold
declare(
    Fold,
    (
        ("operands", tuple[Fold, ...], ()),
        ("init", tuple[float | str, ...], ()),
        ("lift", Lambda, MISSING),
        ("cells", tuple[str, ...], ()),
        ("base", Lambda | None, None),
        ("twist", Twist | None, None),
        ("observe", Lambda | None, None),
    ),
)


def _install(owner: type, members: type) -> None:
    """Put every member ``members`` defines on ``owner``: the Python half of a crate class, as the statement glue
    installs its members."""
    for name, member in vars(members).items():
        if name.startswith("__") and name not in ("__doc__",):
            continue
        if name == "__doc__":
            continue
        setattr(owner, name, member)
        if hasattr(member, "__set_name__"):
            member.__set_name__(owner, name)


_install(Fold, _Fold)

__all__ = [
    "ReductionView",
    "Fold",
]
