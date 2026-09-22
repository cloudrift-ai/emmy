"""Body-level normalization passes.

Pure ``body → body`` transforms applied via :func:`normalize_body` from
``LoopOp.__post_init__`` and from :meth:`Body.structural_key`, so a
constructed Loop-IR Op and every identity digest land in one canonical form. The
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

from collections.abc import Callable, Iterator
from dataclasses import replace
from itertools import count, product

from emmy.compiler.ir.expr import BinaryExpr, CastExpr, Expr, FuncCallExpr, Literal, SimplifyCtx, TernaryExpr, Var, affine_form
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt.base import Stmt
from emmy.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop
from emmy.compiler.ir.stmt.body import Body, free_names
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Init, Load, SelectBranch, Write
from emmy.compiler.ir.stmt.order import _ordered_exported_accs, bound_axes, ordering_constraints, relation_graph, topological_sort

__all__ = ["normalize_body"]


def normalize_body(stmts: Body) -> Body:
    """Apply the structural and cosmetic normalization passes in order.

    Used by ``LoopOp.__post_init__`` so Loop-IR bodies land in a canonical shape before validation,
    and by structural identity, which labels the result's relation graph again with the external
    arguments colored by type. One executable normal form serves both: a body keys the same
    whether it was held bare or constructed as a Loop op.

    External argument names remain readable; operation clustering never runs here because it
    would change executable semantics.
    """
    return Body.coerce(stmts)._normalized


def _normalize_body(stmts: Body) -> Body:
    """Uncached implementation owned by :class:`Body`'s normalization property."""
    stmts = topo_sort_siblings(stmts)
    stmts = drop_size_one_free_axes(stmts)
    stmts = drop_size_one_reduce_axes(stmts)
    stmts = canonicalize_free_axis_order(stmts)
    stmts = eliminate_copy_aliases(stmts)
    stmts = unify_sibling_reduce_axes(stmts)
    stmts = merge_sibling_reduce_loops(stmts)
    stmts = hoist_loop_invariants(stmts)
    stmts = simplify_body(stmts)
    stmts = dedup_loads(stmts)
    # Hoisting, simplification, and a parent merge can expose sibling reductions after the first
    # merge. Close that dependency here: unifying their axes may enable a merge, which may then
    # expose duplicate loads and require one new canonical order. Every changed round removes a
    # loop or a load, so this reaches a fixed point without a fixed iteration bound.
    stmts = _canonical_order(stmts)
    while True:
        # Unification renames sibling reduce axes in place: an order-preserving alpha-rename the
        # relation graph never spelled, so the graph the order came from still describes it.
        unified = unify_sibling_reduce_axes(stmts)
        if unified == stmts:
            unified = stmts
        else:
            unified.__dict__["_ordering"] = stmts._ordering
        reduced = dedup_loads(merge_sibling_reduce_loops(unified))
        if reduced == unified:
            return unified
        stmts = _canonical_order(reduced)


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
    """The innermost row-major output-coordinate depth carrying one free axis."""
    depths = []
    for write in stmts.iter_of_type(Write):
        positions = []
        for position, expr in enumerate(write.index):
            if axis not in expr.free_vars():
                continue
            form = affine_form(expr, {axis})
            if form is not None and form[1].get(axis, 0) != 1:
                return None
            positions.append(position)
        if positions:
            depths.append(len(write.index) - positions[-1] - 1)
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
            for depth, source in enumerate(loop.axis.sources()):
                mapping.setdefault(source.name, f"__parent{depth}__")
            renamed = loop.rename(mapping)
            assert isinstance(renamed, Loop)
            return repr(form((renamed.axis, renamed.unroll, renamed.seed)))

        source_counts: dict[str, int] = {}
        for loop in chain:
            if loop.axis.source_axis is not None:
                name = loop.axis.source_axis.name
                source_counts[name] = source_counts.get(name, 0) + 1

        roles: dict[str, tuple[tuple[int, int], str]] = {}
        for focus, depth in zip(chain, depths, strict=True):
            mapping = {loop.axis.name: "__self__" if loop is focus else "__other__" for loop in chain}
            focused = rename_axes(Body(terminal), mapping)
            source = focus.axis.source_axis
            source_arity = 1 if source is None else source_counts[source.name]
            # A coordinate carried by a consistent output position has a fixed row-major role. An
            # unresolved coordinate is absent, inconsistently placed, or scaled; keep it outside
            # the known output suffix, then use its structural role to break ties.
            storage_order = (0, 0) if depth is None else (1, -depth)
            roles[focus.axis.name] = storage_order, repr((axis_metadata(focus), source_arity, form(focused)))

        groups: dict[tuple[tuple[int, int], str], list[Loop]] = {}
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
    out = {name for stmt in body if isinstance(stmt, Accum) for name in stmt.carried_names()}
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
# Pass 5: loop-invariant code motion
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

    ``Accum`` / ``Init`` / ``Write`` always stay (iteration-tied
    semantics). Axis-invariance alone does not earn a hoist: the hoisted set is closed under the
    scope's ordering constraints (:func:`~emmy.compiler.ir.stmt.order.ordering_constraints`), so
    a statement that must follow one that stays — the consumer of an accumulator a pinned
    reduction exports, a read of a buffer the loop writes, anything behind an ordered execution
    protocol such as a barrier or a declaration — stays with it.
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
        if isinstance(s, (Accum, Init)) or s.has_side_effects:
            return False
        return axis not in _axis_deps(s)

    def _closed_under_constraints(inner: list[Stmt], candidates: set[int]) -> set[int]:
        """``candidates`` less every statement that must follow one that stays.

        A nested reduction can export an accumulator that varies with none of the outer axes
        while its own loop stays pinned (attention's denominator is produced inside the value
        sweep, which is pinned by the head-dim axis the value slab reads); its consumer then
        reads as invariant and would move above the definition. Iterated: un-hoisting one
        candidate can pin the next."""
        if not candidates:
            return candidates
        incoming = ordering_constraints(Body(inner), effects=True)
        while pinned := {index for index in candidates if incoming[index] - candidates}:
            candidates -= pinned
        return candidates

    def walk(body: Body) -> list[Stmt]:
        new_body: list[Stmt] = []
        for s in body:
            if isinstance(s, (Loop, StridedLoop)):
                inner = walk(s.body)
                axis = s.axis.name
                hoisted = _closed_under_constraints(inner, {index for index, c in enumerate(inner) if _hoistable(c, axis)})
                new_body.extend(c for index, c in enumerate(inner) if index in hoisted)
                new_body.append(replace(s, body=tuple(c for index, c in enumerate(inner) if index not in hoisted)))
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
    """Drop duplicate ``Load`` stmts within nested scopes, and with them the duplicate pure
    statements they feed.

    Two ``Load`` stmts with the same ``(input, index)`` read the same
    value; keep the first and rewire downstream SSA references to its
    name. Operates per-scope: a Load at an outer scope is reused by
    inner siblings (their identical ``index`` doesn't reference any
    inner-axis Var, so the values are equal). Loads inside a nested
    scope are not visible to outer / sibling scopes.

    An ``Assign`` or ``Accum`` spelling the same operation over the same (already rewired) names
    as one before it in scope is the same value too — the fusion splice inlines a producer at
    every use, and two consumers in one reduce loop then carry two copies of one accumulation
    (a decoder half's gate and up channels each fold the o_proj result their norm reads). Keeping
    the first and aliasing the second is what makes those copies one cone the tile lift can cut
    once; the pass is named for the loads because that is where a duplicate chain starts.

    Hygienic: an inner scope that re-binds a name the outer scope
    deduped keeps its own binding — those are different variables
    (see :func:`~emmy.compiler.ir.stmt.passes.rename_free`)."""
    from emmy.compiler.ir.stmt.passes import rename_free  # noqa: PLC0415

    stmts = Body.coerce(stmts)

    def written_buffers(stmt: Stmt) -> frozenset[str]:
        return frozenset(
            (*stmt.external_writes(), *(name for child in stmt.nested() for member in child.iter() for name in member.external_writes()))
        )

    def walk(body: Body, env: dict[tuple, tuple[str, ...]], carried: dict[str, str]) -> Body:
        local = dict(env)
        alias: dict[str, str] = {}

        def rename(n: str) -> str:
            return alias.get(n, n)

        def descend(inner: Body, clobbered: frozenset[str], coordinates: frozenset[str]) -> Body:
            """Keep cached values only while their definitions and dependencies retain their bindings.
            Rebound coordinates change a read even when its index has the same spelling. Accumulator
            aliases carry out of the inner loop to the scope that reads the sum."""
            shadowed = Body.coerce(inner).ssa_defs | coordinates
            env = {k: v for k, v in local.items() if k[0] not in clobbered and not shadowed.intersection((*v, *k[-1]))}
            return walk(inner, env, alias)

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
                key = (s.input, tuple(e.pretty() for e in s.index), s.width, s.dtype, s.deps())
                if key in local:
                    alias.update(dict(zip(s.names, local[key], strict=True)))
                    continue
                local[key] = s.names
                out.append(s)
            elif isinstance(s, Assign | Accum):
                s = rename_free(s, alias)
                key = (
                    ("assign", s.op, s.args, s.dtype) if isinstance(s, Assign) else ("accum", s.value, s.op, s.dtype, s.axes, repr(s.base))
                ) + (s.deps(),)
                if key in local:
                    alias[s.name] = local[key][0]
                    if isinstance(s, Accum):
                        carried[s.name] = local[key][0]
                    continue
                local[key] = (s.name,)
                out.append(s)
            elif s.nested():
                clobbered = written_buffers(s)
                renamed = rename_free(s, alias)
                out.append(renamed.with_bodies(tuple(descend(child, clobbered, renamed.binds_axes()) for child in renamed.nested())))
                invalidate(clobbered)
            else:
                out.append(rename_free(s, alias))
                invalidate(frozenset(s.external_writes()))
        return Body(out)

    return walk(stmts, {}, {})


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
    return topological_sort(Body.coerce(stmts))


# ---------------------------------------------------------------------------
# Pass 7: canonicalize SSA names to sequential v0, v1, ...
# ---------------------------------------------------------------------------


def _ssa_prefix(stmt: Stmt) -> str:
    if isinstance(stmt, Load):
        return "in"
    if isinstance(stmt, (Accum, Init)):
        return "acc"
    return "v"


class _SequentialScope:
    """Mutable state for one lexical scope of sequential renaming."""

    def __init__(
        self,
        *,
        counters: dict[str, int] | None = None,
        ssa: dict[str, str] | None = None,
        sources: dict[str, str] | None = None,
        inherited_axes: dict[str, str] | None = None,
        owned: set[str] | None = None,
        fixed: frozenset[str] = frozenset(),
    ) -> None:
        self.counters = {kind: 0 for kind in ("v", "in", "acc", "a", "p")} if counters is None else counters
        self.ssa = {} if ssa is None else ssa
        self.sources = {} if sources is None else sources
        self.inherited_axes = {} if inherited_axes is None else inherited_axes
        self.owned = set() if owned is None else owned
        self.fixed = fixed

    def _allocate(self, old: str, kind: str) -> None:
        if old in self.owned or old in self.fixed:
            return
        self.ssa[old] = f"{kind}{self.counters[kind]}"
        self.counters[kind] += 1
        self.owned.add(old)

    def step(self, stmt: Stmt) -> Stmt:
        """Rename one next statement and advance this scope's allocation state."""
        children = stmt.nested()
        if children:
            for child in children:
                for name in _ordered_exported_accs(child):
                    self._allocate(name, "acc")
        else:
            for name in stmt.defines():
                self._allocate(name, _ssa_prefix(stmt))

        axes = dict(self.inherited_axes)
        for old in stmt.binds_axes():
            axes[old] = f"a{self.counters['a']}"
            self.counters["a"] += 1
        for axis in bound_axes(stmt):
            for source in axis.sources():
                if source.name not in self.sources:
                    self.sources[source.name] = f"p{self.counters['p']}"
                    self.counters["p"] += 1

        names = {**self.ssa, **self.sources, **axes}
        shell = stmt.with_bodies(tuple(Body() for _ in children)) if children else stmt
        renamed = shell.rename(names)
        if children:
            exported = frozenset(name for child in children for name in _ordered_exported_accs(child))
            renamed_children: list[Body] = []
            for child in children:
                scope = _SequentialScope(
                    counters=self.counters,
                    ssa=dict(self.ssa),
                    sources=dict(self.sources),
                    inherited_axes=dict(axes),
                    fixed=exported,
                )
                renamed_children.append(Body(tuple(scope.step(member) for member in child)))
            renamed = renamed.with_bodies(tuple(renamed_children))
        return renamed


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

    scope = _SequentialScope()
    return Body(tuple(scope.step(stmt) for stmt in Body.coerce(stmts)))


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


def _canonical_order(stmts: Body) -> Body:
    """Canonicalize expressions and dependency-valid statement order.

    The relation graph this labels rides the result: the sequential rename and the operand sort
    are order-preserving alpha-renames the graph never spelled, so identity labels the same graph
    with its own resource coloring instead of building it again.
    """
    stmts = _canonicalize_exprs(stmts)
    ordered, ordering = relation_graph(stmts).label().materialize(spelled=True)
    result = Body.coerce(sort_commutative_args(rename_ssa_sequential(ordered)))
    result.__dict__["_ordering"] = ordering
    return result


def _canonicalize_exprs(stmts: Body, axes: tuple[str, ...] = ()) -> Body:
    """Canonicalize integer expressions using lexical binding order, never axis spelling."""
    from dataclasses import fields  # noqa: PLC0415

    from emmy.compiler.structural import form  # noqa: PLC0415

    commutative = frozenset({"+", "*", "==", "!=", "&&", "||", "&", "|", "^"})
    dual = {">": "<", ">=": "<="}
    axis_order = {name: index for index, name in enumerate(axes)}

    def affine(expr: Expr) -> Expr:
        variables = expr.free_vars()
        # Reassociation and coefficient folding are exact for integer coordinates. An SSA value
        # may be floating point, where changing the operation tree changes rounding and kernel work.
        if not variables or not variables <= axis_order.keys() or (decomposed := affine_form(expr, variables)) is None:
            return expr
        anchor, coefficients = decomposed
        anchor = anchor.simplify(SimplifyCtx.empty())
        terms: list[Expr] = []
        for name, coefficient in sorted(coefficients.items(), key=lambda item: axis_order[item[0]]):
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
        rewritten = replace(stmt, **changes)
        bound = (*axes, *(axis.name for axis in bound_axes(stmt)))
        return rewritten.with_bodies(tuple(_canonicalize_exprs(child, bound) for child in rewritten.nested()))

    return Body(statement(stmt) for stmt in stmts)


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
