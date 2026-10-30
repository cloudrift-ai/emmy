"""Restore the contraction row a size-one output dimension carries.

Loop-IR normalization inlines every size-one free axis as ``Literal(0)`` — the coordinate is
constant, so the loop is not iteration. For a contraction that leaves the term with no free axis
of its own, though, the dropped coordinate was its ROW: what remains shares every coordinate with
the operand it is contracted against, and a B that moves with the row it multiplies is no slab per
tile (:meth:`TileOp.contracts`), so the whole family falls to the per-cell tier. Decode attention
is the standing instance — one query per head reduces ``q[0, h, 0, d]`` against the whole key set,
and every output channel then re-runs the score's own contraction.

Binding the coordinate back as an extent-one axis costs the term nothing and hands the tiers a row.
The per-cell choices stay exactly what they were — the catalog offers them beside the fragment ones
— so this widens the schedule space rather than choosing inside it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from math import prod

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Expr, Interval, Literal, SimplifyCtx, Var, affine_form
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.pure import Fold
from emmy.compiler.ir.pure.lam import Lambda
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Body, Load, Loop, Stmt, Write
from emmy.compiler.ir.stmt.passes import rewrite
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.ops import UnbindableProjection, output_regions
from emmy.compiler.ir.tile.path import sites
from emmy.compiler.pipeline.passes.tile._free_axes import canonical_free_axes
from emmy.compiler.pipeline.passes.tile._fromloop import lift_loop_op
from emmy.compiler.pipeline.passes.tile._twist import rewrite_twisted
from emmy.compiler.structural import form

ROW_AXIS = "_row"


def rowless(tile: TileOp) -> bool:
    """Whether some contraction shares a coordinate with its B and owns no free axis."""
    return any((view := node.as_contraction()) is not None and not view.left_axes for node in tile.views)


def _unit_positions(body: Body, shapes: Mapping) -> tuple[int, ...]:
    """Right-aligned positions where every buffer ``body`` stores carries a size-one dimension, read off ``shapes``:
    a program's loop op and a bare nest a cut is forming answer alike, so a piece binds the row its stored
    definition binds when it is lifted again.

    Right alignment is the broadcast correspondence the elementwise lifting already used when it
    turned these dimensions into ``Literal(0)``, so reading it back needs no other agreement.
    """
    shapes = [shapes[name] for name in dict.fromkeys(write.output for write in body.iter_of_type(Write)) if name in shapes]
    if not shapes:
        return ()
    depth = min(len(shape) for shape in shapes)
    return tuple(
        -offset for offset in range(1, depth + 1) if all(shape[-offset].is_static and shape[-offset].as_static() == 1 for shape in shapes)
    )


def _bind(body: Body, axis: Axis, position: int, shapes: dict) -> Body:
    """Substitute ``axis`` for the constant coordinate at ``position`` of every aligned buffer."""

    def rebind(stmt: Stmt) -> Stmt:
        buffer, index = (
            (stmt.input, stmt.index) if isinstance(stmt, Load) else (stmt.output, stmt.index) if isinstance(stmt, Write) else (None, None)
        )
        if buffer is None or buffer not in shapes:
            return stmt
        shape = shapes[buffer]
        if len(index) != len(shape) or -position > len(shape):
            return stmt
        dim = shape[position]
        if not (dim.is_static and dim.as_static() == 1) or index[position] != Literal(0, "int"):
            return stmt
        replaced = list(index)
        replaced[position] = Var(axis.name)
        return replace(stmt, index=tuple(replaced))

    return body.map(rebind)


def row_bound_body(op: LoopOp, position: int, name: str, shapes: Mapping) -> Body:
    """``op``'s body with the coordinate at ``position`` bound as one extent-one free axis."""
    axis = Axis(name=name, extent=1)
    return Body((Loop(axis=axis, body=_bind(Body.coerce(op.body), axis, position, shapes)),))


def binds_the_row(tile: TileOp, name: str) -> bool:
    """Whether ``name`` is the OWN free axis of some contraction — the row it was missing.

    Stated of the named axis rather than of the tile, because a contraction reorients: asking only
    whether some term now has a left axis accepts a binding that gave it to the other side, and a
    row an operand does not read is exactly the shape this module exists to stop producing.
    """
    return any((view := node.as_contraction()) is not None and name in view.left_axes for node in tile.views)


def row_candidates(op: LoopOp, tile: TileOp, shapes: Mapping) -> tuple[int, ...]:
    """The output positions worth binding: none unless a contraction is missing its row ENTIRELY.

    A placement that already carries an extent-one free axis has been served by
    ``_implicit_unit_row``, which proves the row from the stores and announces it without binding.
    That is the weaker statement, but it is the only one available where there is nothing to bind
    into — a matvec whose A is a bare vector — and rebinding those terms costs them the schedule
    they had: the reshaped-output matvec loses its TMA store descriptor and falls off the mma tier.
    So the two readings divide by what the term can support, and this one yields.
    """
    unit = any(axis.extent.is_static and axis.extent.as_static() == 1 for axis in tile.place.free)
    # The batched-matvec unit row is only placement geometry. A size-one output
    # coordinate may still bind into A, which gives the contraction its own row.
    batched_unit = any(axis.name == "_um" for axis in tile.place.free) and any(
        (view := node.as_contraction()) is not None and not view.left_axes and view.shared_axes for node in tile.views
    )
    if unit and not batched_unit:
        return ()
    return _unit_positions(Body.coerce(op.body), shapes) if rowless(tile) else ()


def io_shapes(op) -> dict:
    """The shape of every buffer ``op`` reads or writes, by name."""
    return {name: tensor.shape for name, tensor in {**op.inputs, **op.outputs}.items()}


def lift_kernel(loop: LoopOp, *, name: str, shapes: Mapping | None = None) -> TileOp:
    """One kernel formed from its program, a fused region and a piece alike: its free coordinates canonical over the
    buffer ``shapes`` (``loop``'s own io by default), the complete nest as one Fold tree, then, for a contraction
    that owns no free axis, its size-one row bound back so a tier has a row to tile. Every question reads ``shapes``,
    so a bare nest a cut is forming binds the row its stored definition binds when the lift takes it back: the piece
    and its stored definition are one kernel with one identity."""
    shapes = io_shapes(loop) if shapes is None else shapes
    if (body := canonical_free_axes(loop.body, shapes)) is not None:
        loop = replace(loop, body=body)
    tile = lift_loop_op(loop, name=name)
    # Independent output regions whose loops fusion paired by position re-form with the coordinates their shared
    # reads agree on; the swapped term is lowered and lifted again, so the grid comes out of the same lift.
    paired = _paired_by_loads(tile)
    if paired is not tile:
        body = paired.op.lower(bound=frozenset(), stores=paired.output_specs, axes=paired.axes)
        try:
            again = lift_loop_op(replace(loop, body=body), name=name)
            again.op.lower(bound=frozenset(), stores=again.output_specs, axes=again.axes)
        except ValueError:
            pass
        else:
            loop, tile = replace(loop, body=body), again
    # A contraction that owns no free axis has no row for any tier to tile. Its row is a size-one
    # output dimension Loop-IR normalization inlined; bound back, the term keeps every per-cell
    # choice it had and gains the fragment ones beside them.
    for position in row_candidates(loop, tile, shapes):
        bound = lift_loop_op(loop, name=name, body=row_bound_body(loop, position, ROW_AXIS, shapes))
        if binds_the_row(bound, ROW_AXIS):
            return bound
    return _store_ordered(tile)


def _region_loads(region: Fold) -> tuple[Load, ...]:
    """Every gmem read of ``region``'s term, in its own nodes and in its slab operands."""
    return tuple(
        stmt
        for site in sites(region)
        for node in (site.node, *(edge for edge in site.node.operands if edge.as_slab() is not None))
        for stmt in node.lift.body.iter()
        if isinstance(stmt, Load)
    )


class FoldingSigma(Sigma):
    """A substitution that FOLDS as it lands, under a fixed range context: ``((hi * c) + lo) / c`` collapses to
    ``hi`` and ``((hi * c) + lo) % c`` to ``lo``, so an index reaches the scheduler in the substituted coordinates.
    ``Sigma.apply`` substitutes and stops, which is right everywhere else: a σ that rewrote its result would hide what
    it did."""

    def __init__(self, mapping: dict[str, Expr], ctx: SimplifyCtx) -> None:
        super().__init__(mapping)
        object.__setattr__(self, "_ctx", ctx)

    def apply(self, expr):
        return expr.substitute(self.mapping).simplify(self._ctx)


def _paired_by_loads(tile: TileOp) -> TileOp:
    """``tile`` with one output region re-spelled in the coordinates another region reads a shared input at, where
    that makes more of their reads of it coincide and breaks none; else ``tile`` itself.

    Loop fusion merges loops of equal extent position by position, so two independent regions can share a grid
    coordinate that means different things to them: the GDN recurrence output ``o[h, v]`` and the new state
    ``S'[h, k, v]`` fuse with ``o``'s ``v`` on the store's ``k``, and every cell of the store's sweep over ``v`` folds
    its own column's correction again. Re-spelled, the store reads ``S[h, ·, v]`` where ``o`` does, in one cell, and
    sweeps ``k``. Two pieces spliced into one kernel can also share no loop at all (``o`` over ``(h // 3, h % 3, v)``
    beside the state over ``(h, k, v)``); re-spelled, the state region's ``h`` becomes ``3·g + j`` in ``o``'s
    coordinates and the two share a grid. A read that already coincides is never split: a QKV projection whose q
    column and token are both 4096 long reads ``hidden[t, ·]`` alike in every region and stays as it is."""
    op = tile.op
    if tile.schedule is not None or len(tile.output_specs) < 2 or not isinstance(op, Fold) or op.axis is not None:
        return tile
    try:
        regions = output_regions(op, tile.output_specs)
    except UnbindableProjection:
        return tile
    if len(regions) < 2 or any(tail for _, tail, _ in regions):
        return tile
    grid = {axis.name for axis in tile.place.free}
    axes = {axis.name: axis for axis in (*tile.axes, *(axis for spec in tile.output_specs for axis in spec.sweep))}
    if any(not axis.extent.is_static for axis in axes.values()):
        return tile
    ctx = SimplifyCtx(ranges={name: Interval(0, axis.extent.as_static() - 1) for name, axis in axes.items()})
    outer = [(grid & region.free_axes) | {axis.name for spec in stores for axis in spec.sweep} for region, _, stores in regions]
    loads = [_region_loads(region) for region, _, _ in regions]
    best: tuple[int, int, dict[str, Expr]] | None = None
    for index, (region, _, _) in enumerate(regions):
        theirs = tuple(load for other, found in enumerate(loads) if other != index for load in found)
        before = _coinciding(loads[index], theirs, {}, ctx, set().union(*outer) | grid)
        for other, anchor in enumerate(loads):
            if other == index:
                continue
            for mine in loads[index]:
                for read in anchor:
                    mapping = _read_coordinates(mine, read, outer[index], outer[other], axes, ctx)
                    full = None if mapping is None else _completed(mapping, outer[index], axes)
                    if full is None or not _renamable(region, outer[index] | {n for e in full.values() for n in e.free_vars()}):
                        continue
                    after = _coinciding(loads[index], theirs, full, ctx, set().union(*outer) | grid)
                    if before <= after and len(after) > (best[1] if best is not None else len(before)):
                        best = (index, len(after), full)
    if best is None:
        return tile
    index, _, full = best
    region, _, stores = regions[index]
    sigma = FoldingSigma(full, ctx)
    respelled = _recoordinated(region, sigma)
    # The re-spelled region sweeps every coordinate it now reads that the grid does not bind: the other regions'
    # sweeps in their order first, so the shared ones nest outermost, then its own.
    respelled_outer = {name for expr in full.values() for name in expr.free_vars()}
    own = (*(axis.name for spec in stores for axis in spec.sweep), *(axis.name for axis in tile.place.free))
    order = (
        *(axis.name for spec in tile.output_specs if spec not in stores for axis in spec.sweep),
        *(name for old in own if old in full for name in sorted(full[old].free_vars())),
    )
    sweep = tuple(axes[name] for name in dict.fromkeys(order) if name in respelled_outer - grid)
    specs = tuple(
        replace(spec, write=replace(spec.write, index=tuple(sigma.apply(expr) for expr in spec.write.index)), sweep=sweep)
        if spec in stores
        else spec
        for spec in tile.output_specs
    )
    operands = tuple(respelled if edge is region else edge for edge in op.operands)
    return replace(tile, op=replace(op, operands=operands), output_specs=specs)


def _read_coordinates(mine: Load, read: Load, own: set[str], theirs: set[str], axes: dict, ctx: SimplifyCtx) -> dict | None:
    """Each of ``own``'s coordinates ``mine`` reads at a position where ``read`` reads an expression of ``theirs``
    covering the same range, mapped to that expression; ``None`` unless the two reads then name one element."""
    if mine.input != read.input or len(mine.index) != len(read.index):
        return None
    mapping: dict[str, Expr] = {}
    pairs: dict[str, str] = {}
    for left, right in zip(mine.index, read.index, strict=True):
        if isinstance(left, Var) and left.name in own:
            span = right.range(ctx) if right.free_vars() <= theirs else None
            if span is None or (span.lo, span.hi) != (0, axes[left.name].extent.as_static() - 1):
                return None
            if mapping.setdefault(left.name, right) != right:
                return None
        elif isinstance(left, Var) and isinstance(right, Var) and right.name not in theirs:
            # Two reduce coordinates: any two of equal extent read alike, each paired with one.
            extents = {getattr(axes.get(name), "extent", None) for name in (left.name, right.name)}
            if len(extents) != 1 or pairs.setdefault(left.name, right.name) != right.name:
                return None
        elif left != right or left.free_vars() & own:
            return None
    if all(expr == Var(name) for name, expr in mapping.items()):
        return None
    return mapping


def _completed(mapping: dict[str, Expr], own: set[str], axes: dict) -> dict[str, Expr] | None:
    """``mapping`` extended to every coordinate in ``own``: one it leaves alone keeps its name, unless the mapped
    coordinates now use that name, in which case it takes a name they freed, of equal extent; ``None`` when none is
    free or the re-spelled coordinates cover another number of cells."""
    used = {name for expr in mapping.values() for name in expr.free_vars()}
    freed = sorted(name for name in mapping if name not in used)
    full = dict(mapping)
    for name in sorted(own - mapping.keys()):
        if name not in used:
            full[name] = Var(name)
            continue
        taken = next((other for other in freed if axes[other].extent == axes[name].extent), None)
        if taken is None:
            return None
        freed.remove(taken)
        full[name] = Var(taken)
    after = {name for expr in full.values() for name in expr.free_vars()}
    size = prod(axes[name].extent.as_static() for name in own)
    return full if prod(axes[name].extent.as_static() for name in after) == size else None


def _coinciding(mine: tuple, theirs: tuple, mapping: dict, ctx: SimplifyCtx, outer: set[str]) -> frozenset:
    """The pairs of ``mine`` (re-spelled by ``mapping``) and ``theirs`` that read one element."""
    sigma = FoldingSigma(mapping, ctx)
    return frozenset(
        (i, j)
        for i, load in enumerate(mine)
        for j, other in enumerate(theirs)
        if load.input == other.input
        and len(load.index) == len(other.index)
        and _equal_modulo_reduce(tuple(sigma.apply(expr) for expr in load.index), other.index, outer)
    )


def _recoordinated(node: Fold, sigma: FoldingSigma) -> Fold:
    """``node`` with its coordinates substituted per statement. A free coordinate is a lift PARAM, so a σ handed the
    whole term would be dropped as shadowed exactly where it has work to do: each lift is re-closed instead, the
    substituted names leaving its params and the names it now reads appended."""
    operands = tuple(_recoordinated(edge, sigma) for edge in node.operands)
    body = Body(rewrite(stmt, lambda name: name, sigma) for stmt in node.lift.body)
    lift = Lambda.closing(tuple(param for param in node.lift.params if param not in sigma.mapping), body, node.lift.results)
    return replace(node, operands=operands, lift=lift)


def _renamable(region: Fold, names: set[str]) -> bool:
    """Whether ``names`` are only coordinates ``region`` reads: no node binds, defines or observes one."""
    return not any(
        site.node.observe is not None
        or site.node.axis in names
        or set(site.node.cells) & names
        or set(site.node.lift.results) & names
        or {*site.node.lift.body.ssa_defs, *site.node.lift.body.axis_names} & names
        for site in sites(region)
    )


def _equal_modulo_reduce(left: tuple, right: tuple, outer: set[str]) -> bool:
    """Equal coordinate by coordinate, a reduce coordinate (one not in ``outer``) on one side matching one reduce
    coordinate on the other."""
    pairs: dict[str, str] = {}
    for a, b in zip(left, right, strict=True):
        if a == b:
            continue
        if not (isinstance(a, Var) and isinstance(b, Var)) or a.name in outer or b.name in outer:
            return False
        if pairs.setdefault(a.name, b.name) != b.name:
            return False
    return bool(left) and len(set(pairs.values())) == len(pairs)


def _store_ordered(tile: TileOp) -> TileOp:
    """A kernel with no contraction opens its grid in the order its first store writes: last coordinate fastest,
    within one coordinate the larger stride slower (``c + 8·s`` writes consecutive addresses along ``c``), and an axis
    the store does not name (a scatter's row, a split's partition) slowest, so consecutive threads write consecutive
    addresses. The lift peels loops in nest order, which put a RoPE piece's
    head-dim slowest: uncoalesced, 39 us on the H100 for a 2 MB write. A contraction's grid order is not
    linearization only: its last two axes are the fragment's rows and columns, and reordering them hands the rows to
    an axis an operand varies over."""
    if any(site.node.as_contraction() is not None for site in sites(tile.op)):
        return tile
    written = [name for spec in tile.output_specs[:1] for index in spec.write.index for name in _slowest_first(index)]
    free = tuple(sorted(tile.place.free, key=lambda axis: written.index(axis.name) if axis.name in written else -1))
    return replace(tile, place=replace(tile.place, free=free))


def _slowest_first(index: Expr) -> list[str]:
    """The axes one store coordinate names, the larger stride first; a coordinate not affine in them keeps its
    names' sorted order."""
    names = sorted(index.free_vars())
    form = affine_form(index, set(names))
    return names if form is None else sorted(names, key=lambda name: -abs(form[1].get(name, 0)))


def _orients_by_nest(tile: TileOp) -> bool:
    """Whether some single-product contraction of ``tile`` took its A from the loop nest: one with
    a computed operand, which the lift orients by nest order where two slabs orient by layout."""
    return any(
        node.as_contraction() is not None
        and len(node.bilinear_channels()) == 1
        and any(edge.as_slab() is None for edge in node.operands[:2])
        for node in tile.views
    )


def _align_owned_sweeps(piece: TileOp) -> TileOp:
    """Give independent, equal-domain output sweeps common coordinates before re-forming the piece."""
    op = piece.op
    if piece.schedule is not None or len(piece.output_specs) < 2 or not isinstance(op, Fold) or op.axis is not None:
        return piece
    try:
        regions = output_regions(op, piece.output_specs)
    except UnbindableProjection:
        return piece
    if len(regions) < 2 or any(tail or len(stores) != 1 or not stores[0].sweep for _, tail, stores in regions):
        return piece
    sweeps = tuple(stores[0].sweep for _, _, stores in regions)
    anchor = next((sweep for sweep in sweeps if len(sweep) == 1), sweeps[0])
    axes = tuple(axis for sweep in sweeps for axis in sweep)
    names = {axis.name for axis in axes}
    outputs = tuple(stores[0].write.output for _, _, stores in regions)
    if len(names) != len(axes) or len(set(outputs)) != len(outputs) or names & {axis.name for axis in piece.place.free}:
        return piece
    if len(anchor) == 1 and all(len(sweep) == 1 for sweep in sweeps):
        if any(sweep[0].extent != anchor[0].extent or form(sweep[0].window) != form(anchor[0].window) for sweep in sweeps):
            return piece
    elif any(axis.window is not None or not axis.extent.is_static for axis in axes) or any(
        prod(axis.extent.as_static() for axis in sweep) != prod(axis.extent.as_static() for axis in anchor) for sweep in sweeps
    ):
        return piece
    for (region, _, stores), sweep in zip(regions, sweeps, strict=True):
        owned = {axis.name for axis in sweep}
        store_axes = {name for index in stores[0].write.index for name in index.free_vars()}
        if region.free_axes & names != owned or store_axes & names != owned:
            return piece
        bound = set().union(
            *(
                {
                    site.node.axis,
                    *site.node.lift.params[: (site.node.axis is not None) + len(site.node.bindings)],
                    *site.node.exposes,
                    *site.node.lift.body.ssa_defs,
                    *site.node.lift.body.axis_names,
                }
                for site in sites(region)
            )
        )
        if bound & names or any(
            site.node.observe is not None
            or set(site.node.cells) & owned
            or set(site.node.lift.results) & owned
            or any(set(stmt.deps()) & owned for stmt in site.node.lift.body.iter() if not isinstance(stmt, Load))
            for site in sites(region)
        ):
            return piece

    def sigma(sweep: tuple[Axis, ...], mapping: dict[str, Var]) -> Sigma | None:
        remaining = tuple(axis for axis in sweep if axis.name not in mapping)
        available = tuple(axis for axis in anchor if axis.name not in {expr.name for expr in mapping.values()})
        if prod(axis.extent.as_static() for axis in remaining) != prod(axis.extent.as_static() for axis in available):
            return None
        flat = Var(available[0].name) if available else Literal(0, "int")
        for axis in available[1:]:
            flat = flat * Literal(axis.extent.as_static(), "int") + Var(axis.name)
        substitution: dict[str, Expr] = dict(mapping)
        for i, axis in enumerate(remaining):
            stride = prod(follower.extent.as_static() for follower in remaining[i + 1 :])
            expr = flat
            if stride > 1:
                expr = BinaryExpr("/", expr, Literal(stride, "int"))
            if i:
                expr = BinaryExpr("%", expr, Literal(axis.extent.as_static(), "int"))
            substitution[axis.name] = expr
        return Sigma(substitution)

    def loads(region: Fold) -> tuple[Load, ...]:
        return tuple(
            stmt
            for site in sites(region)
            for node in (site.node, *(edge for edge in site.node.operands if edge.as_slab() is not None))
            for stmt in node.lift.body.iter()
            if isinstance(stmt, Load)
        )

    anchor_region = next(region for (region, _, _), sweep in zip(regions, sweeps, strict=True) if sweep == anchor)
    anchor_loads = loads(anchor_region)
    extents = {axis.name: axis.extent for axis in piece.axes}
    reference = {axis.name for axis in anchor}

    def correspondence(region: Fold, sweep: tuple[Axis, ...]) -> dict[str, Var] | None:
        if len(anchor) == 1:
            return {}
        owned = {axis.name for axis in sweep}
        for load in loads(region):
            for other in anchor_loads:
                if load.input != other.input or len(load.index) != len(other.index):
                    continue
                coordinates = tuple(
                    (left, right)
                    for left, right in zip(load.index, other.index, strict=True)
                    if isinstance(left, Var)
                    and isinstance(right, Var)
                    and left.name in extents
                    and right.name in extents
                    and extents[left.name] == extents[right.name]
                )
                shared = {left.name: right for left, right in coordinates if left.name in owned and right.name in reference}
                reduced = {
                    left.name: right
                    for left, right in coordinates
                    if left.name not in region.free_axes and right.name not in anchor_region.free_axes
                }
                # Equal load indices prove the shared coordinates, modulo equal-domain reduce binders.
                # The rest of the output domain may flatten without mixing those coordinates.
                if (
                    shared
                    and len({expr.name for expr in shared.values()}) == len(shared)
                    and tuple(expr.substitute({**reduced, **shared}) for expr in load.index) == other.index
                ):
                    return shared
        return None

    substitutions: dict[tuple[Axis, ...], Sigma] = {}
    for (region, _, _), sweep in zip(regions, sweeps, strict=True):
        if sweep == anchor or len(sweep) == len(anchor) == 1:
            substitutions[sweep] = Sigma({sweep[0].name: Var(anchor[0].name)}) if sweep != anchor else Sigma.IDENTITY
            continue
        mapping = correspondence(region, sweep)
        substitution = sigma(sweep, mapping) if mapping is not None else None
        if substitution is None:
            return piece
        substitutions[sweep] = substitution
    operands = tuple(
        rewrite(region, lambda name: name, substitutions[sweep]) for (region, _, _), sweep in zip(regions, sweeps, strict=True)
    )
    specs = tuple(replace(spec, write=spec.write.substitute(substitutions[spec.sweep]), sweep=anchor) for spec in piece.output_specs)
    aligned = replace(piece, op=replace(op, operands=operands), output_specs=specs)
    return aligned if all(not spec.sweep for spec in aligned.output_specs) else piece


def reformed(piece: TileOp, shapes: Mapping) -> TileOp:
    """``piece`` formed as its own kernel: its tree lowered to the closed loop nest and lifted
    again, the way a kernel fusion had ended at a graph edge is formed.

    A piece minted by replacing cones in the parent's tree keeps the parent's structure, and that
    structure was formed around what is now a workspace read: a decoder half's gate and up
    contractions are two terms when a contraction feeds the norm ahead of them, and one twin term
    with both channels once the o_proj result is a load. The twin is the term the warp tier tiles;
    two terms give one of them the scalar tier. Formed fresh, the piece is the kernel the same
    program gets on its own, which is also the kernel the card's rows were recorded on. A nest the
    lift cannot take whole keeps the piece as minted. ``shapes`` names every buffer the piece
    touches, as its own program does."""
    piece = _align_owned_sweeps(piece)
    body = piece.op.lower(bound=frozenset(), stores=piece.output_specs, axes=piece.axes)
    try:
        # Through the LoopOp's normalization: that is where two reduce loops over one axis become
        # one loop with two accumulators, the twin the lift forms one term from.
        formed = lift_kernel(LoopOp(body=body), name=piece.name, shapes=shapes)
        if _orients_by_nest(formed):
            # The closed nest's grid order is ``lower``'s choice. Formed once, the piece is lowered
            # again inside its own grid loops: nothing sits ahead of that chain, so normalization
            # orders it by the output layout, and the lift orients the contraction by that order.
            # This pass lowers the formed terms, so it keeps the twin the first pass merged.
            # Normalization can hoist a table read's index out of the reduce loop there, a nest the lift
            # cannot take whole; such a piece keeps the first form.
            again = formed.op.lower(bound=frozenset(axis.name for axis in formed.place.free), stores=formed.output_specs, axes=formed.axes)
            for axis in reversed(formed.place.free):
                again = Body((Loop(axis=axis, body=again),))
            try:
                reoriented = lift_kernel(LoopOp(body=again), name=piece.name, shapes=shapes)
                reoriented.op.lower(bound=frozenset(), stores=reoriented.output_specs, axes=reoriented.axes)
            except ValueError:
                pass
            else:
                body, formed = again, reoriented
    except ValueError:
        return piece
    # The lift peels every outer plain loop into the grid, a store's sweep included when nothing
    # sits ahead of it. The piece takes that grid whatever grid it was minted with: formed alone it
    # binds those axes, so a row measured on it alone describes the kernel its program runs.
    grid = formed.place.free
    # The loop nest carries no buffer shapes, so a row the minted piece announced (an elided unit
    # dimension) cannot be proven again from it. Without one, a contraction whose only shared axis is
    # a split partition would tile that partition as its row, and every row past the first would
    # read the first partition's B.
    unit = next((axis for axis in piece.place.free if axis.extent.is_static and axis.extent.as_static() == 1), None)
    if unit is not None and rowless(formed) and not any(axis.extent.is_static and axis.extent.as_static() == 1 for axis in grid):
        grid = (unit, *grid)
    place = replace(formed.place, free=grid)
    # The loop op the piece was formed from rides as its ``source``, as a fused kernel's does: the body a
    # kernel row stores and a freeze re-lowers is the one the lift was given.
    loop = LoopOp(body=body, name=piece.name)
    return replace(
        piece, op=rewrite_twisted(formed.op, formed.axes), place=place, axes=formed.axes, output_specs=formed.output_specs, source=loop
    )
