"""Body-level normalization passes.

Pure ``body → body`` transforms applied via :func:`normalize_body` from
``LoopOp.__post_init__`` and from :meth:`Body.structural_key`, so a
constructed Loop-IR Op and every identity digest land in canonical form. The
passes operate on the shared Stmt vocabulary (``Loop``, ``Load``, ``Assign``,
``Accum``, ``Select``, ``Write``) and recurse through every block-structured
Stmt (``Loop`` / ``StridedLoop`` / ``Tile`` / ``Cond``).

A ``TileOp`` does NOT run these: it normalizes its TERM
(``normalize_fold_tree``), and the kernel it materializes is built straight
from ``Fold.lower``. So a body these passes could improve reaches the emitter
unchanged whenever it comes down the term path — the sibling-loop merge below
is reachable from Loop IR and from the digest, not from a materialized
``KernelOp``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace
from itertools import chain, count, product

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, CastExpr, Expr, FuncCallExpr, Literal, SimplifyCtx, TernaryExpr, Var, affine_form
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt.base import Stmt
from emmy.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop
from emmy.compiler.ir.stmt.body import Body, _exposed_defines, free_names
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Init, Load, Mma, SelectBranch, Write

__all__ = ["normalize_body"]


def normalize_body(
    stmts: Body,
    *,
    hoist: bool = True,
) -> Body:
    """Apply the structural and cosmetic normalization passes in order.

    Used by ``LoopOp.__post_init__`` so Loop-IR bodies land in a canonical shape before validation.
    Structural identity also runs it over the shared statement vocabulary before its identity-only
    transforms.

    ``hoist=False`` skips :func:`hoist_loop_invariants`. Structural identity turns it off because a
    Stage binding is scoped to the Loop where it is declared; hoisting a Load from a staged buffer
    above that declaration would leave the read referencing an undeclared name.

    External argument names remain readable. Identity assigns those arguments canonical names
    after this pass and may run normalization again; operation clustering never runs here because
    it would change executable semantics.
    """
    stmts = Body.coerce(stmts)
    return stmts._normalized if hoist else stmts._normalized_without_hoist


def _normalize_body(stmts: Body, *, hoist: bool) -> Body:
    """Uncached implementation owned by :class:`Body`'s normalization properties."""
    stmts = topo_sort_siblings(stmts)
    stmts = drop_size_one_free_axes(stmts)
    stmts = drop_size_one_reduce_axes(stmts)
    stmts = canonicalize_free_axis_order(stmts)
    stmts = eliminate_copy_aliases(stmts)
    stmts = unify_sibling_reduce_axes(stmts)
    stmts = merge_sibling_reduce_loops(stmts)
    if hoist:
        stmts = split_invariant_divides(stmts)
        stmts = hoist_loop_invariants(stmts)
    stmts = simplify_body(stmts)
    stmts = dedup_loads(stmts)
    return _canonicalize_order(stmts)


# ---------------------------------------------------------------------------
# Pass 1: drop size-1 free axes
# ---------------------------------------------------------------------------


def drop_size_one_free_axes(stmts: Body) -> Body:
    """Inline every free ``Loop(axis, extent=1)``: replace it with its body
    after substituting ``Var(axis.name) → Literal(0, "int")``. Reduce Loops
    keep their wrappers because dropping them would remove the accumulator.
    Recurses through StridedLoop / Tile / Cond bodies without rewriting
    those wrappers (their iteration semantics aren't a free Loop).

    Size-1 BLOCK / SPLITK_BLOCK protection used to live here when the
    planner stamped ``Loop.role`` for downstream launch_geometry to
    consume. The planner now constructs ``GridTile`` / ``ThreadTile``
    directly and applies its own size-1 filter (see
    ``010_partition_loops::_wrap_tower``), so by the time
    ``drop_size_one_free_axes`` runs on a LoopOp body, no Loop has any
    binding role — every size-1 free Loop is safely inlinable.
    """
    stmts = Body.coerce(stmts)

    def fn(s: Stmt) -> Stmt | Body:
        # Body.map post-order: ``s.body`` is already recursively mapped.
        if isinstance(s, Loop) and s.axis.extent.is_static and s.axis.extent.as_static() == 1 and not s.is_reduce:
            sub = Sigma({s.axis.name: Literal(0, "int")})
            return tuple(c.substitute(sub) for c in s.body)
        return s

    return stmts.map(fn)


def drop_size_one_reduce_axes(stmts: Body) -> Body:
    """Inline a canonical extent-one reduction as its single update.

    Fusion can hoist a singleton reduction's value into the enclosing scope (decode softmax is
    the common case).  Keeping the reduction wrapper then asks Tile IR to form a fold whose lift
    returns that enclosing value without defining it locally.  An extent-one fold is just one
    application of its monoid, so replace each distinct accumulator with an ordinary pure
    assignment before copy-alias elimination rewires the result.

    Only the canonical one-update form is collapsed.  Scans, nested effects, and repeated updates
    to one accumulator keep their loop because their sequential state is not an alias.
    """
    stmts = Body.coerce(stmts)

    def fn(stmt: Stmt) -> Stmt | Body:
        if not (isinstance(stmt, Loop) and stmt.is_reduce and stmt.axis.extent.is_static and stmt.axis.extent.as_static() == 1):
            return stmt
        accums = tuple(member for member in stmt.body if isinstance(member, Accum))
        if (
            not accums
            or len({accum.name for accum in accums}) != len(accums)
            or any(not (member.pure or isinstance(member, Accum)) for member in stmt.body)
        ):
            return stmt

        sub = Sigma({stmt.axis.name: Literal(0, "int")})
        out: list[Stmt] = []
        for member in stmt.body:
            member = member.substitute(sub)
            if not isinstance(member, Accum):
                out.append(member)
                continue
            if member.base is None or member.base == member.name:
                if not member.has_identity:
                    return stmt
                out.append(Assign(name=member.name, op="copy", args=(member.value,), dtype=member.dtype))
            else:
                out.append(Assign(name=member.name, op=member.op, args=(member.base, member.value), dtype=member.dtype))
        return Body(out)

    return stmts.map(fn)


# ---------------------------------------------------------------------------
# Pass 2: canonical free-axis ordering
# ---------------------------------------------------------------------------


def _recurse_canonicalize(s: Stmt) -> Stmt:
    nested = s.nested()
    if not nested:
        return s
    return s.with_bodies(tuple(canonicalize_free_axis_order(b) for b in nested))


def _output_storage_depth(stmts: Body, axis: str) -> int | None:
    """The row-major output-coordinate depth of one unit-affine free axis."""
    depths = []
    for write in stmts.iter_of_type(Write):
        positions = []
        for position, expr in enumerate(write.index):
            form = affine_form(expr, {axis})
            if form is None:
                return None
            coefficient = form[1].get(axis, 0)
            if coefficient:
                if coefficient != 1:
                    return None
                positions.append(position)
        if len(positions) > 1:
            return None
        if positions:
            depths.append(len(write.index) - positions[0] - 1)
    return depths[0] if depths and len(set(depths)) == 1 else None


def canonicalize_free_axis_order(stmts: Body) -> Body:
    """Sort an outer free-loop chain by row-major output storage order.

    Boundary writes provide the canonical geometry: larger coordinate depth is outer, so the
    innermost loop follows the output's contiguous dimension. If the writes do not totally order
    the chain, choose the least alpha-renamed structural form. Recursion continues into terminal
    block bodies (Loop / StridedLoop / Tile / Cond).
    """
    stmts = Body.coerce(stmts)
    chain: list[Loop] = []
    current = stmts
    while len(current) == 1 and isinstance(current[0], Loop):
        loop = current[0]
        if loop.is_reduce:
            break
        chain.append(loop)
        current = loop.body

    terminal = tuple(_recurse_canonicalize(s) for s in current)

    depths = [_output_storage_depth(Body(terminal), loop.axis.name) for loop in chain]
    if all(depth is not None for depth in depths) and len(set(depths)) == len(depths):
        chain_sorted = [loop for _, loop in sorted(zip(depths, chain, strict=True), key=lambda item: -item[0])]
    else:
        from emmy.compiler.structural import form  # noqa: PLC0415

        def rename_axes(body: Body, mapping: dict[str, str]) -> Body:
            sigma = Sigma({name: Var(replacement) for name, replacement in mapping.items()})
            return Body(stmt.substitute(sigma) for stmt in body)

        def axis_metadata(loop: Loop) -> str:
            mapping = {loop.axis.name: "__axis__"}
            parent = loop.axis.source_axis
            depth = 0
            seen: set[int] = set()
            while parent is not None and id(parent) not in seen:
                seen.add(id(parent))
                mapping.setdefault(parent.name, f"__parent{depth}__")
                depth += 1
                parent = parent.source_axis
            renamed = loop.rename(mapping)
            assert isinstance(renamed, Loop)
            return repr(form((renamed.axis, renamed.unroll, renamed.seed)))

        source_counts: dict[str, int] = {}
        for loop in chain:
            if loop.axis.source_axis is not None:
                name = loop.axis.source_axis.name
                source_counts[name] = source_counts.get(name, 0) + 1

        roles: dict[str, str] = {}
        for focus in chain:
            mapping = {loop.axis.name: "__self__" if loop is focus else "__other__" for loop in chain}
            focused = rename_axes(Body(terminal), mapping)
            source = focus.axis.source_axis
            source_arity = 1 if source is None else source_counts[source.name]
            roles[focus.axis.name] = repr((axis_metadata(focus), source_arity, form(focused)))

        groups: dict[str, list[Loop]] = {}
        for loop in chain:
            groups.setdefault(roles[loop.axis.name], []).append(loop)

        def axis_orders(group: list[Loop]) -> Iterator[tuple[Loop, ...]]:
            def interchangeable(left: Loop, right: Loop) -> bool:
                if axis_metadata(left) != axis_metadata(right):
                    return False
                left_source, right_source = left.axis.source_axis, right.axis.source_axis
                if (
                    left_source is not None
                    and right_source is not None
                    and left_source.name != right_source.name
                    and (source_counts[left_source.name] != 1 or source_counts[right_source.name] != 1)
                ):
                    return False
                swapped = rename_axes(Body(terminal), {left.axis.name: right.axis.name, right.axis.name: left.axis.name})
                return form(swapped) == form(terminal)

            yield from _orders_modulo_transpositions(group, interchangeable)

        def candidate(order: tuple[Loop, ...]) -> tuple[str, tuple[Loop, ...]]:
            result: Body = Body(terminal)
            for loop in reversed(order):
                result = Body((Loop(axis=loop.axis, body=result, unroll=loop.unroll, seed=loop.seed),))
            return repr(form(rename_ssa_sequential(result))), order

        ordered_groups = [groups[role] for role in sorted(groups)]
        chain_sorted = list(
            min(
                candidate(tuple(loop for choice in choices for loop in choice))
                for choices in product(*(axis_orders(group) for group in ordered_groups))
            )[1]
        )
    result: Body = terminal
    for loop in reversed(chain_sorted):
        result = Body((Loop(axis=loop.axis, body=result, unroll=loop.unroll, seed=loop.seed),))
    return result


# ---------------------------------------------------------------------------
# Pass 3: eliminate `y = copy(x)` identity aliases
# ---------------------------------------------------------------------------


def eliminate_copy_aliases(stmts: Body) -> Body:
    """Collapse ``y = copy(x)`` Assigns. The merge rule plants identity
    copies as bridges between producer writes and consumer reads; a long
    chain stacks them. Every such Assign is dropped and downstream
    references to ``y`` are rewired to the alias root. Pure IR hygiene."""
    from emmy.compiler.ir.stmt.passes import rename_free  # noqa: PLC0415

    def walk(body: Body) -> Body:
        alias: dict[str, str] = {}

        def resolve(name: str) -> str:
            seen: set[str] = set()
            while name in alias and name not in seen:
                seen.add(name)
                name = alias[name]
            return name

        out: list[Stmt] = []
        for stmt in body:
            if isinstance(stmt, Assign) and stmt.op.name == "copy" and len(stmt.args) == 1 and stmt.dtype is None:
                alias[stmt.name] = resolve(stmt.args[0])
                continue
            if stmt.nested():
                # Apply aliases from the enclosing scope hygienically, then give each child its
                # own alias table. A spelling reused by sibling bodies denotes separate binders.
                stmt = rename_free(stmt, alias)
                stmt = stmt.with_bodies(tuple(walk(child) for child in stmt.nested()))
                out.append(stmt)
            else:
                out.append(stmt.rewrite(resolve))
        return Body(out)

    return walk(Body.coerce(stmts))


# ---------------------------------------------------------------------------
# Pass 4: unify sibling reduce-loop axis names
# ---------------------------------------------------------------------------


def unify_sibling_reduce_axes(stmts: Body) -> Body:
    """At every scope, find sibling reduce ``Loop``s whose reduce axes
    index overlapping ``(Load.source, dim)`` positions and rename them
    to a single canonical axis name. Recurses through every block-
    structured Stmt (Loop / StridedLoop / Tile / Cond) to find nested
    scopes."""
    stmts = Body.coerce(stmts)

    def walk(body: Body) -> Body:
        # Recurse into nested bodies first (post-order) via the canonical
        # nested() / with_bodies() descent, then group siblings at this
        # scope. Splitting the recursion from the sibling-grouping keeps
        # this pass's scope-level logic isolated in ``_unify_siblings``.
        recursed: list[Stmt] = []
        for s in body:
            nested = s.nested()
            if nested:
                recursed.append(s.with_bodies(tuple(walk(b) for b in nested)))
            else:
                recursed.append(s)
        return _unify_siblings(Body(recursed))

    return walk(stmts)


def _unify_siblings(body: Body) -> Body:
    """Single-scope sibling grouping: rename reduce-axis vars across
    sibling reduce Loops whose bare-Var Load positions overlap on any
    ``(source, dim)`` pair so they share one canonical axis name.

    Two reduce Loops that bind different axis names but both index the
    same input slot (e.g. ``x[..., a2]`` and ``x[..., a3]`` for the
    same ``x``) are semantically the same reduction dimension. Union-
    find on the overlap relation merges all transitively-connected
    Loops into one group. Within a group, the first Loop's axis name
    wins; later Loops are rewritten to use it.

    Pairing on overlap rather than exact-set equality lets matmul-
    siblings that bring in distinct weight tensors (e.g.
    ``silu(x@Wg) * (x@Wu)`` — both reduce over K and index x, but only
    one indexes Wg and the other Wu) unify on the shared x position;
    the downstream :func:`merge_sibling_reduce_loops` pass then
    concatenates their bodies.
    """
    stmts = list(body)

    # Key on ``Dim.expr`` (the underlying ``Expr``) so structural equality on
    # extents matches both static and symbolic siblings: two ``Dim('seq_len')``
    # siblings unify (both back to ``Var('seq_len')``); two distinct symbolic
    # names don't. ``Expr`` is frozen + hashable so it slots into the tuple key.
    entries: list[tuple[int, str, object, frozenset[tuple[str, int, object, int]]]] = []
    for i, s in enumerate(stmts):
        if isinstance(s, Loop) and s.is_reduce:
            positions = _reduce_axis_source_positions(s.body, s.axis.name)
            if positions:
                entries.append((i, s.axis.name, s.axis.extent.expr, frozenset(positions)))

    if len(entries) < 2:
        return Body(stmts)

    parent = list(range(len(entries)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(len(entries)):
        for b in range(a + 1, len(entries)):
            if entries[a][2] != entries[b][2]:
                continue
            if entries[a][3] & entries[b][3]:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)

    for k, (idx, axis_name, extent, _) in enumerate(entries):
        canonical = entries[find(k)][1]
        if canonical == axis_name:
            continue
        loop = stmts[idx]
        assert isinstance(loop, Loop)
        # A RENAME — binders and uses through one map, which is why it may pass through the very
        # scopes it renames. Spelled as a σ plus an axis renamer it read as a substitution, and a
        # substitution must stop at a re-binding scope; these loops ARE the re-binding scopes.
        new_axis = replace(loop.axis, name=canonical, extent=extent)
        renamed = tuple(s.rename({loop.axis.name: canonical}) for s in loop.body)
        stmts[idx] = replace(loop, axis=new_axis, body=renamed)

    return Body(stmts)


def _reduce_axis_source_positions(body: Body, reduce_axis_name: str) -> set[tuple[str, int, object, int]]:
    """Collect ``(source, dim, anchor, coefficient)`` positions where a Load index within ``body``
    is AFFINE in ``Var(reduce_axis_name)`` (recursing into nested blocks).

    A bare ``Var`` is the ``(source, dim, 0, 1)`` case, so this generalizes the original bare-Var
    reading rather than replacing it. Affine matters because a BLOCKED reduce reads its stream at
    ``outer·B + inner``: the axis still walks that dimension, and refusing to see it left sibling
    loops over one block unmergeable for no semantic reason.

    The anchor and coefficient ride the key because that is what makes the reading sound. Two
    siblings indexing ``x[…, o·B + i]`` and ``x[…, o·B + j]`` walk the SAME dimension and unify;
    ``o·B + i`` against ``o·B + 32 + j`` walk different halves and must not. ``(source, dim)`` alone
    cannot tell those apart — seeing through the offset would be a miscompile, not a generalization.
    """
    out: set[tuple[str, int, object, int]] = set()
    for s in body.iter():
        if not isinstance(s, Load):
            continue
        for dim, e in enumerate(s.index):
            form = affine_form(e, {reduce_axis_name})
            if form is None:
                continue
            anchor, coeffs = form
            coeff = coeffs.get(reduce_axis_name, 0)
            if coeff:
                out.add((s.input, dim, anchor, coeff))
    return out


# ---------------------------------------------------------------------------
# Pass 4b: merge sibling reduce Loops with matching axis into one Loop.
# ---------------------------------------------------------------------------
#
# After :func:`unify_sibling_reduce_axes` renames sibling reduce axes
# that index overlapping ``(source, dim)`` positions to one canonical
# name, adjacent reduce Loops with the same axis name/extent become
# structurally identical iteration scopes. Merging concatenates their
# bodies into one Loop so the reduce axis is traversed once instead of
# twice. Later normalization by ``dedup_loads`` collapses the duplicate Loads
# both halves share — e.g. ``load x[0, a0, k]`` in the gated-MLP
# pattern ``silu(x@Wg) * (x@Wu)`` where both matmuls reduce over the
# same K and share x as a Load source. Symmetric staging follows: once
# wu lives in the same K-loop as wg, ``stage_inputs`` / ``use_tma`` /
# ``use_ring_buffers`` apply uniformly.
# ---------------------------------------------------------------------------


def merge_sibling_reduce_loops(stmts: Body) -> Body:
    """Merge sibling reduce ``Loop``s with matching ``axis.name`` and
    ``axis.extent`` into one Loop whose body is the concatenation.

    Gates a merge on three conditions, all phrased over what the second Loop
    reads from its enclosing scope (:func:`free_names` — what it uses and does
    not bind itself):

    1. It reads no SSA name the first Loop's body defines. When it does, the
       two reductions are sequentially dependent — e.g. softmax's sum-exp loop
       reads ``acc_max`` from the preceding max loop. Merging would replace
       that read of the *finalized* max with a read of the in-flight per-iter
       value, changing semantics.
    2. No statement between the two Loops defines a name it reads — otherwise
       the merge would move that read above its def.
    3. The names both bodies happen to bind are a COLLISION, not a dependence,
       and the incoming body's copies rename apart — which is what makes two
       alpha-equal copies of one cone mergeable at all. Only a name the incoming
       Loop still binds after it closes (:func:`_carried_out`) refuses: the
       rename cannot reach the readers that name has outside the loop.

    Statements that sit between the two original Loops stay in their
    original positions in the parent Body. References to the first
    Loop's ``Accum`` remain valid (Accum names cross the Loop
    boundary). References to the second Loop's ``Accum`` from
    statements that originally followed it now resolve to the merged
    Loop above them — still defs-before-uses.

    Recurses through every block-structured Stmt to find nested scopes.
    """
    stmts = Body.coerce(stmts)

    def walk(body: Body) -> Body:
        recursed: list[Stmt] = []
        for s in body:
            nested = s.nested()
            if nested:
                recursed.append(s.with_bodies(tuple(walk(b) for b in nested)))
            else:
                recursed.append(s)
        return _merge_sibling_reduce_loops(Body(recursed))

    return walk(stmts)


def _carried_out(body: Body) -> frozenset[str]:
    """The names a ``Loop`` over ``body`` still binds after it CLOSES.

    :meth:`Loop.render` declares the carriers of the immediate body ahead of the loop, so those —
    and, under a nested loop that does not seed its own, that loop's carriers too — are the names a
    later statement can still read. Every other definition lives inside the block the loop closes,
    which is what makes it renamable when two loops merge.
    """
    out = {name for stmt in body if isinstance(stmt, (Accum, Mma)) for name in stmt.carried_names()}
    for stmt in body:
        if isinstance(stmt, Loop) and not stmt.seed:
            out |= _carried_out(stmt.body)
    return frozenset(out)


def _rename_apart(body: Body, clashing: frozenset[str], taken: frozenset[str]) -> Body:
    """``body`` with each name in ``clashing`` renamed to one neither side spells."""
    used = set(body.ssa_defs) | set(body.ssa_uses) | set(taken)
    mapping: dict[str, str] = {}
    for name in sorted(clashing):
        fresh = next(candidate for n in count(1) if (candidate := f"{name}__m{n}") not in used)
        mapping[name] = fresh
        used.add(fresh)
    return Body(tuple(s.rename(mapping) for s in body))


def _merge_sibling_reduce_loops(body: Body) -> Body:
    items = list(body)
    if len(items) < 2:
        return body

    out: list[Stmt] = []
    consumed: set[int] = set()
    for i, s in enumerate(items):
        if i in consumed:
            continue
        if not (isinstance(s, Loop) and s.is_reduce):
            out.append(s)
            continue
        merged = s
        for j in range(i + 1, len(items)):
            if j in consumed:
                continue
            t = items[j]
            if not (
                isinstance(t, Loop)
                and t.is_reduce
                and t.axis.name == merged.axis.name
                and t.axis.extent == merged.axis.extent
                and t.unroll == merged.unroll
                and t.seed == merged.seed
            ):
                continue
            # ONE reading of what the incoming loop needs from around it: the names it reads and
            # does not bind itself. A name it both defines and uses is its own local — which is
            # what two alpha-equal copies of a single cone always share — and counting those as
            # reads reports a dependence that is not there.
            reads = free_names(t)
            merged_defs = Body.coerce(merged.body).ssa_defs
            if merged_defs & reads:
                continue
            incoming = Body.coerce(t.body)
            # What is left of the shared spellings is a COLLISION, not a dependence: two bodies
            # binding one name for unrelated values. Renaming the incoming body's copy apart is
            # sound for every name the loop closes over; a name it still binds afterwards has
            # readers the rename cannot reach, so that one refuses.
            clashing = merged_defs & incoming.ssa_defs
            if clashing & _carried_out(incoming):
                continue
            between_defs: set[str] = set()
            for k in range(i + 1, j):
                if k in consumed:
                    continue
                between_defs |= Body.coerce(Body((items[k],))).ssa_defs
            if between_defs & reads:
                continue
            merged = Loop(
                axis=merged.axis,
                body=Body(tuple(merged.body) + tuple(_rename_apart(incoming, clashing, merged_defs))),
                unroll=merged.unroll,
                seed=merged.seed,
            )
            consumed.add(j)
        out.append(merged)

    return Body(out)


# ---------------------------------------------------------------------------
# Pass 5a: split loop-invariant divides into reciprocal + multiply.
# ---------------------------------------------------------------------------
#
# ``divide(x, y)`` lowers to a single-precision divide on the XU pipe (the
# same pipe ``exp`` uses). When ``y`` is loop-invariant w.r.t. some
# enclosing Loop and ``x`` is not, the divide can't hoist as-is — its live
# set is the union of x's and y's. Splitting into::
#
#     recip_y = reciprocal(y)        # live = axes_of(y)
#     result  = multiply(x, recip_y) # live = axes_of(x) ∪ {recip_y}
#
# lets the next pass (``hoist_loop_invariants``) move ``recip_y`` out of
# every Loop axis that doesn't appear in ``y``. Inside the loop the
# divide turns into a multiply (FMA pipe), which is typically the
# under-utilized pipe on transcendental-heavy kernels (softmax,
# RMSNorm, attention output). One XU op per outer-axis iteration
# instead of one per inner-axis iteration.
#
# Gate: split iff ``axes_of(y)`` is a strict subset of ``axes_of(x)``.
# That's the precise structural condition for "splitting unblocks at
# least one Loop's worth of hoisting." Skip when y has axes x doesn't
# (no hoisting wins) or when both have identical axes (rcp would stay
# in the same scope as the original divide, no win and slight
# precision drift). When y is a true scalar (axes_of empty), the rcp
# hoists all the way to body root.
# ---------------------------------------------------------------------------


def split_invariant_divides(stmts: Body) -> Body:
    """Rewrite ``divide(x, y)`` → ``reciprocal(y) + multiply(x, recip)``
    when ``y``'s axis-dependency set is a strict subset of ``x``'s.

    Invariance is queried via :attr:`Body.axis_dependencies` over the
    pre-rewrite body. The strict-subset check means there's at least one
    axis ``x`` depends on that ``y`` doesn't — splitting moves the rcp out
    of that axis's Loop while the multiply stays. Generates fresh SSA names
    for the rcp; the trailing :func:`rename_ssa_sequential` pass renumbers
    them into ``vN`` form.
    """
    from emmy.compiler.ir.elementwise import ElementwiseImpl  # noqa: PLC0415

    stmts = Body.coerce(stmts)
    if not any(isinstance(stmt, Assign) and stmt.op.name == "divide" for stmt in stmts.iter()):
        return stmts
    axis_dependencies = dict(stmts.axis_dependencies)
    ssa_names: set[str] = set(axis_dependencies)
    fresh_counter = [0]

    def _fresh(prefix: str) -> str:
        while True:
            fresh_counter[0] += 1
            n = f"{prefix}_{fresh_counter[0]}"
            if n not in ssa_names:
                ssa_names.add(n)
                return n

    def _axes_of(name: str) -> frozenset[str]:
        return axis_dependencies.get(name, frozenset())

    def walk(body: Body) -> Body:
        out: list[Stmt] = []
        for s in body:
            nested = s.nested()
            if nested:
                # Generic descent — recurse into every nested body, rebuild
                # the wrapper via with_bodies. The closure was built once
                # over the whole body, so post-Loop Accum bookkeeping is
                # already baked in — no per-wrapper update needed here.
                out.append(s.with_bodies(tuple(walk(b) for b in nested)))
                continue
            if isinstance(s, Assign) and s.op == ElementwiseImpl("divide") and len(s.args) == 2:
                x_name, y_name = s.args
                if _axes_of(y_name) < _axes_of(x_name):  # strict subset → splitting unblocks at least one hoist
                    recip_name = _fresh(f"recip_{y_name}")
                    recip = Assign(name=recip_name, op=ElementwiseImpl("reciprocal"), args=(y_name,))
                    mult = Assign(name=s.name, op=ElementwiseImpl("multiply"), args=(x_name, recip_name))
                    # Patch dependencies for the freshly-introduced rcp so a
                    # later divide reading the same y in the same body
                    # still sees the correct axis set.
                    axis_dependencies[recip_name] = axis_dependencies.get(y_name, frozenset())
                    axis_dependencies[mult.name] = axis_dependencies.get(x_name, frozenset()) | axis_dependencies[recip_name]
                    out.append(recip)
                    out.append(mult)
                    continue
            out.append(s)
        return Body(out)

    return walk(stmts)


# ---------------------------------------------------------------------------
# Pass 5b: loop-invariant code motion
# ---------------------------------------------------------------------------


def hoist_loop_invariants(stmts: Body) -> Body:
    """Move stmts out of ``Loop``s whose axis they don't depend on.

    Hoists ``Load`` / ``Assign`` / ``Select`` (SSA values) and entire
    ``Loop`` / ``StridedLoop`` / ``Tile`` / ``Cond`` blocks whose contents
    transitively avoid the outer axis — provided the block contains no
    ``Write`` (a Write hoist would change observable side effects).
    Block-level hoisting is what lets a Loop and its downstream consumer
    move together: hoisting just the consumer would leave it referencing
    an Accum still defined inside the outer Loop body.

    ``Accum`` / ``Mma`` / ``Init`` / ``Write`` always stay (iteration-tied
    semantics). Loop-invariance is queried via :meth:`Body.depends_on`
    against the body's transitive read closure, so the hoisted set is
    automatically closed under SSA dependencies — no separate ordering
    check is needed.
    """
    stmts = Body.coerce(stmts)
    name_axes = stmts.axis_dependencies
    axis_names = stmts.axis_names
    axis_deps: dict[int, tuple[Stmt, frozenset[str]]] = {}

    def _axis_deps(s: Stmt) -> frozenset[str]:
        """Axes read by one immutable subtree, computed bottom-up once."""
        key = id(s)
        cached = axis_deps.get(key)
        if cached is not None and cached[0] is s:
            return cached[1]
        reads = set(s.deps())
        for expr in s.exprs():
            reads.update(expr.free_vars())
        deps = reads & axis_names
        for name in reads:
            deps.update(name_axes.get(name, frozenset()))
        for child in (child for body in s.nested() for child in body):
            deps.update(_axis_deps(child))
        result = frozenset(deps - s.binds_axes())
        axis_deps[key] = (s, result)
        return result

    def _hoistable(s: Stmt, axis: str) -> bool:
        # Accum / Init are scope-bound to their enclosing Loop's reduction (an Init seeds an
        # Accum or a Carrier's state per output cell) — they can't move alone, but the
        # whole enclosing block can. Side-effecting stmts (Write, or any block containing a
        # Write) pin their iteration count and stay put.
        if isinstance(s, (Accum, Mma, Init)) or s.has_side_effects:
            return False
        return axis not in _axis_deps(s)

    def _crossing_a_definition(inner: list[Stmt], hoisted: list[Stmt]) -> list[Stmt]:
        """``hoisted`` less every stmt reading a name the loop body still BINDS.

        Axis-invariance alone does not earn a hoist. A nested reduction can export an
        accumulator that varies with none of the outer axes while its own loop stays pinned
        (attention's denominator is produced inside the value sweep, which is pinned by the
        head-dim axis the value slab reads). Its consumer then reads as invariant and moves
        above the definition. Iterated: un-hoisting one candidate can pin the next."""
        while hoisted:
            ids = {id(c) for c in hoisted}
            bound = {name for c in inner if id(c) not in ids for name in _exposed_defines(c)}
            keep = [c for c in hoisted if not (free_names(c) & bound)]
            if len(keep) == len(hoisted):
                break
            hoisted = keep
        return hoisted

    def walk(body: Body) -> list[Stmt]:
        new_body: list[Stmt] = []
        for s in body:
            if isinstance(s, (Loop, StridedLoop)):
                inner = walk(s.body)
                axis = s.axis.name
                hoisted = _crossing_a_definition(inner, [c for c in inner if _hoistable(c, axis)])
                hoisted_ids = {id(c) for c in hoisted}
                stay = [c for c in inner if id(c) not in hoisted_ids]
                new_body.extend(hoisted)
                new_body.append(replace(s, body=tuple(stay)))
            elif isinstance(s, Cond):
                new_body.append(Cond(cond=s.cond, body=tuple(walk(s.body)), else_body=tuple(walk(s.else_body))))
            else:
                new_body.append(s)
        return new_body

    return tuple(walk(stmts))


# ---------------------------------------------------------------------------
# Pass 6: simplify Exprs inside body Stmts (constant folding, identity collapse,
# range-based comparison folding). The per-Expr rewrite logic lives on each
# ``Expr`` subclass as ``simplify(ctx)``; the walk over Stmts is dispatched
# in :mod:`.passes` (singledispatch + Stage introspection).
# ---------------------------------------------------------------------------


def simplify_body(body: Body) -> Body:
    """Simplify every Expr inside a body. Seeds ``SimplifyCtx`` from
    ``Loop`` / ``StridedLoop`` / ``Tile`` axis extents as the walker descends.
    Tile-IR Stmt registrations are loaded when ``tile.ir`` is imported."""
    from emmy.compiler.ir.stmt.passes import simplify  # noqa: PLC0415

    body = Body.coerce(body)
    ctx = SimplifyCtx.empty()
    return tuple(simplify(s, ctx) for s in body)


# ---------------------------------------------------------------------------
# Pass: deduplicate Load stmts with identical (input, index)
# ---------------------------------------------------------------------------


def dedup_loads(stmts: Body) -> Body:
    """Drop duplicate ``Load`` stmts within nested scopes.

    Two ``Load`` stmts with the same ``(input, index)`` read the same
    value; keep the first and rewire downstream SSA references to its
    name. Operates per-scope: a Load at an outer scope is reused by
    inner siblings (their identical ``index`` doesn't reference any
    inner-axis Var, so the values are equal). Loads inside a nested
    scope are not visible to outer / sibling scopes.

    Hygienic: an inner scope that re-binds a name the outer scope
    deduped keeps its own binding — those are different variables
    (see :func:`~emmy.compiler.ir.stmt.passes.rename_free`)."""
    from emmy.compiler.ir.stmt.passes import rename_free  # noqa: PLC0415

    stmts = Body.coerce(stmts)

    def written_buffers(stmt: Stmt) -> frozenset[str]:
        return frozenset(
            (*stmt.external_writes(), *(name for child in stmt.nested() for member in child.iter() for name in member.external_writes()))
        )

    def walk(body: Body, env: dict[tuple[str, tuple[str, ...], int, object], tuple[str, ...]]) -> Body:
        local = dict(env)
        alias: dict[str, str] = {}

        def rename(n: str) -> str:
            return alias.get(n, n)

        def descend(inner: Body, clobbered: frozenset[str]) -> Body:
            """Enter ``inner``'s scope, dropping every alias / kept name whose spelling ``inner``
            re-binds. SSA names bound inside a Loop / Cond body are scoped to it, so such a name is
            a DIFFERENT variable — following it out would rewire the inner arithmetic to the outer
            value and redeclare the survivor."""
            shadowed = Body.coerce(inner).ssa_defs
            return walk(inner, {k: v for k, v in local.items() if k[0] not in clobbered and not shadowed.intersection(v)})

        def invalidate(buffers: frozenset[str]) -> None:
            for key in tuple(local):
                if key[0] in buffers:
                    del local[key]

        out: list[Stmt] = []
        for s in body:
            if isinstance(s, Load):
                # Rewire any SSA names in this Load's *index* to their deduped
                # alias first — a gather ``weight[(int)in0, a]`` whose index
                # Load ``in0`` was itself deduped must follow ``in0`` to the
                # kept name, or the index dangles after the duplicate is
                # dropped. (No-op for plain axis indices: axes aren't aliased.)
                s = s.rewrite(rename)
                key = (s.input, tuple(e.pretty() for e in s.index), s.width, s.dtype)
                if key in local:
                    alias.update(dict(zip(s.names, local[key], strict=True)))
                    continue
                local[key] = s.names
                out.append(s)
            elif s.nested():
                clobbered = written_buffers(s)
                renamed = rename_free(s, alias)
                out.append(renamed.with_bodies(tuple(descend(child, clobbered) for child in renamed.nested())))
                invalidate(clobbered)
            else:
                out.append(rename_free(s, alias))
                invalidate(frozenset(s.external_writes()))
        return Body(out)

    return walk(stmts, {})


# ---------------------------------------------------------------------------
# Pass: topologically sort siblings so SSA defs precede their uses.
# ---------------------------------------------------------------------------


def topo_sort_siblings(stmts: Body) -> Body:
    """Reorder stmts within each Body so SSA defs precede their uses.

    Recurses into every child body via the Stmt protocol
    (:meth:`Stmt.nested` / :meth:`Stmt.with_bodies`), then runs a stable
    Kahn ordering over the current sibling list. A block stmt
    (``Loop`` / ``StridedLoop`` / ``Tile`` / ``Cond``) is opaque at the
    parent level: it ``defs`` any Accum names that escape its body
    (visible to siblings via Loop's cross-boundary Accum semantics) and
    ``uses`` its wrapper-level deps plus any free SSA names referenced
    inside (names referenced inside but not defined inside).

    Splicer worklists (and any future producer that emits stmts with
    sibling-dedup) can land a consumer above an already-emitted producer
    when the producer was reused from an earlier emission. Sorting at
    normalize time decouples final body order from producer subtleties
    and guarantees every constructed ``LoopOp`` / ``TileOp`` lands with
    defs above uses, which downstream passes (validator, renamer,
    codegen) rely on.

    Stable: when the dep edges leave a free choice, the original sibling
    order is preserved (heap-based Kahn with index tiebreak). Idempotent:
    bodies already in topo order round-trip unchanged.
    """
    return _topo(Body.coerce(stmts))


def _topo(body: Body) -> Body:
    import heapq

    items: list[Stmt] = []
    for s in body:
        nested = s.nested()
        if nested:
            items.append(s.with_bodies(tuple(_topo(b) for b in nested)))
        else:
            items.append(s)

    n = len(items)
    if n <= 1:
        return Body(tuple(items))

    defs_uses = [_sibling_defs_uses(s) for s in items]
    # First-writer wins: handles repeated Accum decls (idempotent at the
    # same name) and the rare aliasing edge case without crashing.
    def_idx: dict[str, int] = {}
    for i, (defs, _) in enumerate(defs_uses):
        for name in defs:
            def_idx.setdefault(name, i)

    incoming: list[set[int]] = [set() for _ in range(n)]
    outgoing: list[list[int]] = [[] for _ in range(n)]
    for i, (_, uses) in enumerate(defs_uses):
        for name in uses:
            j = def_idx.get(name)
            if j is not None and j != i and j not in incoming[i]:
                incoming[i].add(j)
                outgoing[j].append(i)

    ready: list[int] = [i for i in range(n) if not incoming[i]]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        i = heapq.heappop(ready)
        order.append(i)
        for k in outgoing[i]:
            incoming[k].discard(i)
            if not incoming[k]:
                heapq.heappush(ready, k)

    if len(order) != n:
        # Cycle through SSA names — leave order untouched so the validator
        # rejects it with a precise message instead of silently shuffling.
        return Body(tuple(items))
    return Body(tuple(items[i] for i in order))


def _sibling_defs_uses(stmt: Stmt) -> tuple[frozenset[str], frozenset[str]]:
    """Names ``stmt`` makes visible to siblings, and names it depends on
    from siblings.

    Leaves: ``defs = stmt.defines()``, ``uses = stmt.deps()``.
    Block stmts: ``defs`` = Accum names escaping the body (recursive);
    ``uses`` = wrapper's own deps ∪ ((all inner uses) − (all inner SSA
    defs)).
    """
    nested = stmt.nested()
    if not nested:
        return frozenset(stmt.defines()), frozenset(stmt.deps())
    defs: set[str] = set()
    all_uses: set[str] = set(stmt.deps())
    all_inner_defs: set[str] = set()
    for b in nested:
        defs |= _exported_accs(b)
        all_uses |= Body.coerce(b).ssa_uses
        all_inner_defs |= Body.coerce(b).ssa_defs
    return frozenset(defs), frozenset(all_uses - all_inner_defs)


def _ordered_sibling_defs(stmt: Stmt) -> tuple[str, ...]:
    """Names visible to siblings in structural body order."""
    children = stmt.nested()
    if not children:
        return stmt.defines()
    return tuple(dict.fromkeys(name for child in children for name in _ordered_exported_accs(child)))


def _ordered_exported_accs(body: Body) -> tuple[str, ...]:
    """Accumulator names exported by ``body``, deduplicated in structural order."""
    exported: list[str] = []
    seen: set[str] = set()
    for stmt in Body.coerce(body):
        if isinstance(stmt, (Accum, Mma)):
            for name in stmt.carried_names():
                if name not in seen:
                    seen.add(name)
                    exported.append(name)
        for child in stmt.nested():
            for name in _ordered_exported_accs(child):
                if name not in seen:
                    seen.add(name)
                    exported.append(name)
    return tuple(exported)


def _exported_accs(body: Body) -> frozenset[str]:
    return frozenset(_ordered_exported_accs(body))


# ---------------------------------------------------------------------------
# Pass 7: canonicalize SSA names to sequential v0, v1, ...
# ---------------------------------------------------------------------------


def rename_ssa_sequential(stmts: Body) -> Body:
    """Canonicalize names in a fused body:

    - Axes from every axis-bearing scope (``Loop`` / ``StridedLoop`` /
      ``Tile.axes`` / new tile flavors' axes) renamed to ``a0, a1, ...``.
      Window parent provenance is renamed to ``p0, p1, ...`` with its base and bound expressions.
    - Load SSA names renamed to ``in0, in1, ...`` in definition order.
    - Accum names renamed to ``acc0, acc1, ...`` in definition order.
    - Every other SSA definition renamed to ``v0, v1, ...`` in definition order.

    Each nested body is a lexical scope, while its assigned names remain globally unique. Reusing
    ``x`` or ``i`` in two sibling loops therefore cannot collapse two distinct binders into one
    canonical name. Loop-carried accumulators are allocated in the enclosing scope and deliberately
    keep that name through the reduce body.

    Idempotent: bodies already in canonical form round-trip unchanged."""

    def prefix(stmt: Stmt) -> str:
        if isinstance(stmt, Load):
            return "in"
        if isinstance(stmt, (Accum, Mma, Init)):
            return "acc"
        return "v"

    counters = {kind: 0 for kind in ("v", "in", "acc", "a", "p")}

    def bound_axes(stmt: Stmt) -> tuple[Axis, ...]:
        axis = getattr(stmt, "axis", None)
        if isinstance(axis, Axis):
            return (axis,)
        axes = getattr(stmt, "axes", ())
        return tuple(axis for axis in axes if isinstance(axis, Axis))

    def walk(
        body: Body,
        inherited_ssa: dict[str, str],
        inherited_axes: dict[str, str],
        inherited_sources: dict[str, str],
        fixed: frozenset[str] = frozenset(),
    ) -> Body:
        body = Body.coerce(body)
        ssa = dict(inherited_ssa)
        sources = dict(inherited_sources)
        owned: set[str] = set()

        def allocate(old: str, kind: str) -> None:
            if old in owned or old in fixed:
                return
            new = f"{kind}{counters[kind]}"
            counters[kind] += 1
            ssa[old] = new
            owned.add(old)

        out: list[Stmt] = []
        for stmt in body:
            children = stmt.nested()
            if children:
                for child in children:
                    for name in _ordered_exported_accs(child):
                        allocate(name, "acc")
            else:
                for name in stmt.defines():
                    allocate(name, prefix(stmt))

            axes = dict(inherited_axes)
            for old in stmt.binds_axes():
                axes[old] = f"a{counters['a']}"
                counters["a"] += 1
            for axis in bound_axes(stmt):
                parent = axis.source_axis
                seen: set[int] = set()
                while parent is not None and id(parent) not in seen:
                    seen.add(id(parent))
                    if parent.name not in sources:
                        sources[parent.name] = f"p{counters['p']}"
                        counters["p"] += 1
                    parent = parent.source_axis

            names = {**ssa, **sources, **axes}
            renamed = stmt.rename(names)
            if children:
                exported = frozenset(name for child in children for name in _exported_accs(child))
                renamed_children = tuple(walk(child, ssa, axes, sources, exported) for child in children)
                renamed = renamed.with_bodies(renamed_children)
            out.append(renamed)
        return Body(out)

    return walk(Body.coerce(stmts), {}, {}, {})


# ---------------------------------------------------------------------------
# Pass: sort args of commutative Assigns.
# ---------------------------------------------------------------------------


def sort_commutative_args(stmts: Body) -> Body:
    """Sort ``Assign.args`` for commutative ``op``s so two bodies that
    differ only by argument order land in the same canonical form.

    Acts on ``Assign`` only. Expression normalization handles equivalent index and condition
    spellings. Recurses
    through every block-structured Stmt (``Loop`` / ``StridedLoop`` /
    ``Tile`` / ``Cond``)."""
    stmts = Body.coerce(stmts)

    def fn(s: Stmt) -> Stmt:
        if isinstance(s, Assign) and s.op.commutative and len(s.args) > 1:
            sorted_args = tuple(sorted(s.args))
            if sorted_args != s.args:
                return replace(s, args=sorted_args)
        return s

    return stmts.map(fn)


def _canonicalize_order(stmts: Body) -> Body:
    """Canonicalize expressions and dependency-valid statement order."""
    from emmy.compiler.structural import form  # noqa: PLC0415

    stmts = _canonicalize_exprs(stmts)
    ordered, revisit = _canonicalize_scope_order(stmts)
    candidate = Body.coerce(sort_commutative_args(rename_ssa_sequential(ordered)))
    while revisit:
        refined, revisit = _revisit_scope_order(candidate, revisit)
        refined = Body.coerce(sort_commutative_args(rename_ssa_sequential(refined)))
        if form(refined) == form(candidate):
            return refined
        candidate = refined
    return candidate


def _renormalize_external_order(stmts: Body, buffers: frozenset[str]) -> Body:
    """Revisit only scopes whose order can change after an external-buffer rename."""
    from emmy.compiler.structural import form  # noqa: PLC0415

    def paths(body: Body) -> frozenset[_ScopePath]:
        found: set[_ScopePath] = set()
        touched = False
        for statement_index, stmt in enumerate(Body.coerce(body)):
            if buffers & {*stmt.external_reads(), *stmt.external_writes()}:
                touched = True
            for child_index, child in enumerate(stmt.nested()):
                child_paths = paths(child)
                if child_paths:
                    touched = True
                    found.update(((statement_index, child_index), *path) for path in child_paths)
        if touched:
            found.add(())
        return frozenset(found)

    revisit = paths(stmts)
    candidate = Body.coerce(stmts)
    seen: dict[str, int] = {}
    orbit: list[tuple[str, Body]] = []
    while revisit:
        rendered = repr(form(candidate))
        if rendered in seen:
            return min(orbit[seen[rendered] :])[1]
        seen[rendered] = len(orbit)
        orbit.append((rendered, candidate))
        candidate, revisit = _revisit_scope_order(candidate, revisit)
        candidate = Body.coerce(sort_commutative_args(rename_ssa_sequential(candidate)))
    return candidate


def _least_sibling_order(statements: list[Stmt]) -> tuple[Body, bool]:
    """Least dependency-valid order for one scope and whether a choice existed."""
    from emmy.compiler.structural import form  # noqa: PLC0415

    choices = iter(_canonicalize_sibling_order_variants(statements))
    first = next(choices)
    second = next(choices, None)
    if second is None:
        return Body(first), False

    def key(ordered: tuple[Stmt, ...]) -> str:
        candidate = Body.coerce(sort_commutative_args(rename_ssa_sequential(Body(ordered))))
        return repr(form(candidate))

    return Body(min(chain((first, second), choices), key=key)), True


type _ScopePath = tuple[tuple[int, int], ...]


def _scope_needs_context(body: Body) -> bool:
    """Whether a scope reads enclosing names or exports carried state."""
    visible_defs = frozenset(name for stmt in body for name in _sibling_defs_uses(stmt)[0])
    captures = frozenset(name for stmt in body for name in free_names(stmt)) - visible_defs
    return bool(captures or _exported_accs(body))


def _scope_paths(
    ordered: Body,
    child_paths: dict[int, tuple[frozenset[_ScopePath], ...]],
    *,
    current: bool,
) -> frozenset[_ScopePath]:
    """Rebase child scope paths under one chosen sibling order."""
    paths: set[_ScopePath] = {()} if current else set()
    for statement_index, stmt in enumerate(ordered):
        for child_index, nested in enumerate(child_paths.get(id(stmt), ())):
            paths.update(((statement_index, child_index), *path) for path in nested)
    return frozenset(paths)


def _canonicalize_scope_order(stmts: Body, *, nested: bool = False) -> tuple[Body, frozenset[_ScopePath]]:
    """Choose one canonical sibling order per scope without renaming its binders.

    Nested scopes are independent ordering problems. Keeping their original names until the final
    whole-body rename preserves outer captures and exported accumulator references without taking
    the Cartesian product of every child's valid orders.
    """
    statements: list[Stmt] = []
    child_paths: dict[int, tuple[frozenset[_ScopePath], ...]] = {}
    for stmt in Body.coerce(stmts):
        children = stmt.nested()
        if children:
            nested_results = tuple(_canonicalize_scope_order(child, nested=True) for child in children)
            stmt = stmt.with_bodies(tuple(body for body, _ in nested_results))
            child_paths[id(stmt)] = tuple(paths for _, paths in nested_results)
        statements.append(stmt)

    ordered, ambiguous = _least_sibling_order(statements)
    has_marked_child = any(paths for children_paths in child_paths.values() for paths in children_paths)
    current = has_marked_child or (nested and ambiguous and _scope_needs_context(Body(statements)))
    return ordered, _scope_paths(ordered, child_paths, current=current)


def _revisit_scope_order(stmts: Body, revisit: frozenset[_ScopePath]) -> tuple[Body, frozenset[_ScopePath]]:
    """Reorder marked scopes after their enclosing names become canonical.

    The bottom-up pass deliberately preserves names until the whole body is renamed. An ambiguous
    nested scope can therefore choose an order from the source spelling of a captured value or an
    exported accumulator. Revisit those scopes after the enclosing names are canonical, and revisit
    their ancestors because a changed child can change a block's sibling role. Unrelated scopes do
    not repeat their ordering search.
    """
    statements: list[Stmt] = []
    child_paths: dict[int, tuple[frozenset[_ScopePath], ...]] = {}
    for statement_index, stmt in enumerate(Body.coerce(stmts)):
        children = stmt.nested()
        if children:
            refined_children = []
            refined_paths = []
            for child_index, child in enumerate(children):
                prefix = (statement_index, child_index)
                nested_paths = frozenset(path[1:] for path in revisit if path and path[0] == prefix)
                if nested_paths:
                    refined, paths = _revisit_scope_order(child, nested_paths)
                else:
                    refined, paths = child, frozenset()
                refined_children.append(refined)
                refined_paths.append(paths)
            stmt = stmt.with_bodies(tuple(refined_children))
            child_paths[id(stmt)] = tuple(refined_paths)
        statements.append(stmt)

    current = () in revisit
    ordered = _least_sibling_order(statements)[0] if current else Body(statements)
    return ordered, _scope_paths(ordered, child_paths, current=current)


def _canonicalize_exprs(stmts: Body) -> Body:
    """Canonicalize equivalent integer coordinate and condition expressions."""
    from dataclasses import fields  # noqa: PLC0415

    from emmy.compiler.structural import form  # noqa: PLC0415

    commutative = frozenset({"+", "*", "==", "!=", "&&", "||", "&", "|", "^"})
    dual = {">": "<", ">=": "<="}
    axis_names = stmts.axis_names

    def affine(expr: Expr) -> Expr:
        variables = expr.free_vars()
        # Reassociation and coefficient folding are exact for integer coordinates. An SSA value
        # may be floating point, where changing the operation tree changes rounding and kernel work.
        if not variables or not variables <= axis_names or (decomposed := affine_form(expr, variables)) is None:
            return expr
        anchor, coefficients = decomposed
        anchor = anchor.simplify(SimplifyCtx.empty())
        terms: list[Expr] = []
        for name, coefficient in sorted(coefficients.items()):
            variable = Var(name)
            terms.append(variable if coefficient == 1 else BinaryExpr("*", Literal(coefficient, "int"), variable))
        if not (isinstance(anchor, Literal) and anchor.value == 0):
            terms.append(anchor)
        if not terms:
            return Literal(0, "int")
        result = terms[0]
        for term in terms[1:]:
            result = BinaryExpr("+", result, term)
        return result

    def expression(expr: Expr) -> Expr:
        if isinstance(expr, BinaryExpr):
            left, right = expression(expr.left), expression(expr.right)
            op = expr.op
            if op in dual:
                op, left, right = dual[op], right, left
            if op in commutative and repr(form(right)) < repr(form(left)):
                left, right = right, left
            result = BinaryExpr(op, left, right)
            return affine(result) if op in {"+", "-", "*"} else result
        if isinstance(expr, FuncCallExpr):
            return FuncCallExpr(expr.name, tuple(expression(arg) for arg in expr.args))
        if isinstance(expr, TernaryExpr):
            return TernaryExpr(expression(expr.cond), expression(expr.if_true), expression(expr.if_false))
        if isinstance(expr, CastExpr):
            return CastExpr(expr.dtype, expression(expr.expr))
        return expr

    def value(item):
        if isinstance(item, Expr):
            return expression(item)
        if isinstance(item, tuple):
            return tuple(value(member) for member in item)
        if isinstance(item, SelectBranch):
            return SelectBranch(value=item.value, select=expression(item.select))
        return item

    def statement(stmt: Stmt) -> Stmt:
        changes = {field.name: value(getattr(stmt, field.name)) for field in fields(stmt)}
        return replace(stmt, **changes)

    return Body.coerce(stmts).map(statement)


def _pure_tokens(stmts: Iterable[Stmt]) -> dict[int, str]:
    """Name- and order-free structural token for every pure definition.

    The forward half describes what a statement computes. The reverse half describes every use of
    its results. Equal producers that feed different operations or operand positions therefore get
    different tokens before the exact ordering search has to branch. Callers choose the scope:
    definitions from a nested lexical scope cannot participate in its parent's pure-value graph.
    """
    from emmy.compiler.structural import digest, form  # noqa: PLC0415

    members = tuple(stmts)
    definitions: dict[str, tuple[Stmt, int]] = {}
    for stmt in members:
        if stmt.pure:
            for slot, name in enumerate(stmt.defines()):
                definitions.setdefault(name, (stmt, slot))

    forward_tokens: dict[int, str] = {}
    visiting_forward: set[int] = set()

    def forward_token(stmt: Stmt) -> str:
        identity = id(stmt)
        if identity in forward_tokens:
            return forward_tokens[identity]
        if identity in visiting_forward:
            return "__cycle__"
        visiting_forward.add(identity)
        mapping = {name: f"__own{slot}" for slot, name in enumerate(stmt.defines())}
        for name in stmt.deps():
            owner = definitions.get(name)
            if owner is not None:
                producer, slot = owner
                mapping[name] = f"__dep_{digest(forward_token(producer))}_{slot}"
            else:
                mapping[name] = "__free__"
        renamed = stmt.rename(mapping)
        renamed = sort_commutative_args(Body((renamed,)))[0]
        result = repr(form(renamed))
        visiting_forward.remove(identity)
        forward_tokens[identity] = result
        return result

    consumers: dict[str, list[Stmt]] = {name: [] for name in definitions}
    for stmt in members:
        if stmt.pure:
            forward_token(stmt)
        for name in dict.fromkeys(stmt.deps()):
            if name in consumers:
                consumers[name].append(stmt)

    reverse_tokens: dict[str, str] = {}
    visiting_reverse: set[str] = set()

    def reverse_token(name: str) -> str:
        if name in reverse_tokens:
            return reverse_tokens[name]
        if name in visiting_reverse:
            return "__cycle__"
        visiting_reverse.add(name)
        contexts = []
        for consumer in consumers[name]:
            mapping = {defined: f"__own{slot}" for slot, defined in enumerate(consumer.defines())}
            for dependency in consumer.deps():
                if dependency == name:
                    mapping[dependency] = "__self__"
                    continue
                owner = definitions.get(dependency)
                if owner is not None:
                    producer, slot = owner
                    mapping[dependency] = f"__other_{digest(forward_token(producer))}_{slot}"
                else:
                    mapping[dependency] = "__free__"
            renamed = consumer.rename(mapping)
            renamed = sort_commutative_args(Body((renamed,)))[0]
            # Hash the recursive suffix before embedding it. Repr-nesting raw token strings doubles
            # their escaping at each producer in a long chain and grows exponentially in memory.
            downstream = tuple(digest(reverse_token(defined)) for defined in consumer.defines() if defined in definitions)
            contexts.append((repr(form(renamed)), downstream))
        result = repr(tuple(sorted(contexts)))
        visiting_reverse.remove(name)
        reverse_tokens[name] = result
        return result

    tokens: dict[int, str] = {}
    for stmt in members:
        if stmt.pure:
            tokens[id(stmt)] = repr((forward_token(stmt), tuple(reverse_token(name) for name in stmt.defines())))
    return tokens


def _canonicalize_sibling_order_variants(stmts: list[Stmt]) -> Iterator[tuple[Stmt, ...]]:
    """Yield every unresolved canonical Kahn order for one sibling scope."""
    if len(stmts) <= 1:
        yield tuple(stmts)
        return

    from emmy.compiler.structural import digest, form  # noqa: PLC0415

    defs_uses = [_sibling_defs_uses(stmt) for stmt in stmts]
    definitions: dict[str, list[int]] = {}
    for index, (defines, _) in enumerate(defs_uses):
        for name in defines:
            definitions.setdefault(name, []).append(index)

    def defining_stmt(name: str, consumer: int) -> int | None:
        sites = definitions.get(name, ())
        preceding = [site for site in sites if site < consumer]
        if preceding:
            return preceding[-1]
        return next((site for site in sites if site != consumer), None)

    incoming = []
    for index, (_, uses) in enumerate(defs_uses):
        sources = {source for name in uses if (source := defining_stmt(name, index)) is not None}
        incoming.append(sources)
    for reader, (_, uses) in enumerate(defs_uses):
        for name in uses:
            for later_definition in definitions.get(name, ()):
                if later_definition > reader:
                    incoming[later_definition].add(reader)

    def resources(stmt: Stmt) -> tuple[set[str], set[str], set[str]]:
        members = tuple(stmt for child in stmt.nested() for stmt in child.iter()) or (stmt,)
        reads = {name for member in members for name in member.external_reads()}
        writes = {name for member in members for name in member.external_writes()}
        state = {name for member in members for name in getattr(member, "carried_names", lambda: ())()}
        if isinstance(stmt, Init):
            state.update(stmt.defines())
        return reads, writes, state

    effects = [resources(stmt) for stmt in stmts]
    for later in range(len(stmts)):
        later_reads, later_writes, later_state = effects[later]
        for earlier in range(later):
            reads, writes, state = effects[earlier]
            if writes & (later_reads | later_writes) or reads & later_writes or state & later_state:
                incoming[later].add(earlier)

    edges = {(source, target) for target, sources in enumerate(incoming) for source in sources}
    body = Body(stmts)
    names = set(body.ssa_defs | body.ssa_uses | body.axis_names)
    names.update(name for stmt in body for name in free_names(stmt))
    for stmt in body.iter():
        axis = getattr(stmt, "axis", None)
        axes = (*((axis,) if isinstance(axis, Axis) else ()), *getattr(stmt, "axes", ()))
        for axis in (axis for axis in axes if isinstance(axis, Axis)):
            parent = axis.source_axis
            seen: set[int] = set()
            while parent is not None and id(parent) not in seen:
                seen.add(id(parent))
                names.add(parent.name)
                parent = parent.source_axis
    abstract = {name: "__name__" for name in names}

    def direct_token(stmt: Stmt, *, shallow: bool) -> str:
        children = stmt.nested()
        token_stmt = stmt.with_bodies(tuple(Body() for _ in children)) if shallow and children else stmt
        renamed = sort_commutative_args(Body((token_stmt.rename(abstract),)))[0]
        rendered = form(renamed)
        return repr((not children, rendered)) if shallow else digest(rendered)

    # Most ready statements already differ by their own operation, buffer, index, or wrapper.
    # Compare that cheap local shape first; build the downstream-sensitive pure graph only for a
    # remaining tie. This keeps exact canonical labeling while avoiding whole-scope graph work for
    # the common case of distinct loads and operations.
    coarse_tokens = {id(stmt): direct_token(stmt, shallow=True) for stmt in stmts}
    tokens: dict[int, str] | None = None
    graph_tokens: dict[int, int] | None = None

    def canonical_tokens() -> dict[int, str]:
        nonlocal tokens
        if tokens is not None:
            return tokens
        pure_tokens = _pure_tokens(stmts)

        def token(stmt: Stmt) -> str:
            if id(stmt) in pure_tokens:
                return pure_tokens[id(stmt)]
            return direct_token(stmt, shallow=False)

        tokens = {id(stmt): token(stmt) for stmt in stmts}
        return tokens

    def refined_tokens() -> dict[int, int]:
        """Refine tied local forms by their position in the sibling dependency graph.

        A statement's own form and immediate producer/use roles leave regular, non-symmetric
        graphs tied. Exhaustively ordering such a partition is factorial even though neighboring
        statements usually distinguish every member. Stable color refinement carries those
        distinctions through the graph before the exact transposition and ordering fallback.
        """
        nonlocal graph_tokens
        if graph_tokens is not None:
            return graph_tokens

        base_tokens = canonical_tokens()

        def ranks(values: list[object]) -> list[int]:
            ordered = {value: rank for rank, value in enumerate(sorted(set(values), key=repr))}
            return [ordered[value] for value in values]

        labels: dict[tuple[int, int], object] = {}
        for source, target in edges:
            mapping = dict(abstract)
            mapping.update({name: f"__source{slot}" for slot, name in enumerate(_ordered_sibling_defs(stmts[source]))})
            consumer = sort_commutative_args(Body((stmts[target].rename(mapping),)))[0]
            reads, writes, state = effects[source]
            target_reads, target_writes, target_state = effects[target]
            labels[source, target] = (
                digest(form(consumer)),
                tuple(sorted(writes & target_reads)),
                tuple(sorted(writes & target_writes)),
                tuple(sorted(reads & target_writes)),
                len(state & target_state),
            )

        base = [base_tokens[id(stmt)] for stmt in stmts]
        colors = ranks(base)
        for _ in stmts:
            descriptors = [
                (
                    base[index],
                    colors[index],
                    tuple(sorted((labels[source, index], colors[source]) for source in incoming[index])),
                    tuple(sorted((labels[index, target], colors[target]) for source, target in edges if source == index)),
                )
                for index in range(len(stmts))
            ]
            refined = ranks(descriptors)
            if refined == colors:
                break
            colors = refined
        graph_tokens = {id(stmt): colors[index] for index, stmt in enumerate(stmts)}
        return graph_tokens

    def interchangeable(left: int, right: int) -> bool:
        left_defs = _ordered_sibling_defs(stmts[left])
        right_defs = _ordered_sibling_defs(stmts[right])
        if len(left_defs) != len(right_defs):
            return False
        rename = {**dict(zip(left_defs, right_defs, strict=True)), **dict(zip(right_defs, left_defs, strict=True))}
        swap = {left: right, right: left}
        if {(swap.get(a, a), swap.get(b, b)) for a, b in edges} != edges:
            return False
        for index, stmt in enumerate(stmts):
            rewritten = sort_commutative_args(Body((stmt.rename(rename),)))[0]
            if form(rewritten) != form(stmts[swap.get(index, index)]):
                return False
        return True

    def walk(remaining: frozenset[int], ordered: tuple[int, ...]) -> Iterator[tuple[Stmt, ...]]:
        if not remaining:
            yield tuple(stmts[index] for index in ordered)
            return
        ready = [index for index in remaining if not incoming[index] & remaining]
        if not ready:
            yield tuple(stmts)
            return
        if len(ready) == 1:
            selected = ready[0]
            yield from walk(remaining - {selected}, (*ordered, selected))
            return
        least_coarse = min(coarse_tokens[id(stmts[index])] for index in ready)
        tied = [index for index in ready if coarse_tokens[id(stmts[index])] == least_coarse]
        if len(tied) > 1:
            tokens = canonical_tokens()
            least = min(tokens[id(stmts[index])] for index in tied)
            tied = [index for index in tied if tokens[id(stmts[index])] == least]
        if len(tied) > 1:
            tokens = refined_tokens()
            least = min(tokens[id(stmts[index])] for index in tied)
            tied = [index for index in tied if tokens[id(stmts[index])] == least]
        representatives: list[int] = []
        for selected in tied:
            if not any(interchangeable(selected, earlier) for earlier in representatives):
                representatives.append(selected)
        for selected in representatives:
            yield from walk(remaining - {selected}, (*ordered, selected))

    yield from walk(frozenset(range(len(stmts))), ())


def _orders_modulo_transpositions(items: list, interchangeable: Callable[[object, object], bool]) -> Iterator[tuple]:
    """Permute distinct items once modulo transpositions proven to preserve the whole form."""
    classes: list[list] = []
    for item in items:
        for group in classes:
            if interchangeable(item, group[0]):
                group.append(item)
                break
        else:
            classes.append([item])

    def walk(remaining: tuple[int, ...], positions: tuple[int, ...]) -> Iterator[tuple]:
        if not any(remaining):
            offsets = [0] * len(classes)
            out = []
            for class_index in positions:
                out.append(classes[class_index][offsets[class_index]])
                offsets[class_index] += 1
            yield tuple(out)
            return
        for class_index, count_left in enumerate(remaining):
            if not count_left:
                continue
            next_remaining = list(remaining)
            next_remaining[class_index] -= 1
            yield from walk(tuple(next_remaining), (*positions, class_index))

    yield from walk(tuple(len(group) for group in classes), ())
