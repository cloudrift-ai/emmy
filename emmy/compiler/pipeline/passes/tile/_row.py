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
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.pure import Fold
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


def _unit_positions(op: LoopOp) -> tuple[int, ...]:
    """Right-aligned positions where every stored output carries a size-one dimension.

    Right alignment is the broadcast correspondence the elementwise lifting already used when it
    turned these dimensions into ``Literal(0)``, so reading it back needs no other agreement.
    """
    shapes = [tensor.shape for tensor in op.outputs.values()]
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


def row_bound_body(op: LoopOp, position: int, name: str) -> Body:
    """``op``'s body with the coordinate at ``position`` bound as one extent-one free axis."""
    axis = Axis(name=name, extent=1)
    return Body((Loop(axis=axis, body=_bind(Body.coerce(op.body), axis, position, io_shapes(op))),))


def binds_the_row(tile: TileOp, name: str) -> bool:
    """Whether ``name`` is the OWN free axis of some contraction — the row it was missing.

    Stated of the named axis rather than of the tile, because a contraction reorients: asking only
    whether some term now has a left axis accepts a binding that gave it to the other side, and a
    row an operand does not read is exactly the shape this module exists to stop producing.
    """
    return any((view := node.as_contraction()) is not None and name in view.left_axes for node in tile.views)


def row_candidates(op: LoopOp, tile: TileOp) -> tuple[int, ...]:
    """The output positions worth binding: none unless a contraction is missing its row ENTIRELY.

    A placement that already carries an extent-one free axis has been served by
    ``_implicit_unit_row``, which proves the row from the stores and announces it without binding.
    That is the weaker statement, but it is the only one available where there is nothing to bind
    into — a matvec whose A is a bare vector — and rebinding those terms costs them the schedule
    they had: the reshaped-output matvec loses its TMA store descriptor and falls off the mma tier.
    So the two readings divide by what the term can support, and this one yields.
    """
    if any(axis.extent.is_static and axis.extent.as_static() == 1 for axis in tile.place.free):
        return ()
    return _unit_positions(op) if rowless(tile) else ()


def io_shapes(op) -> dict:
    """The shape of every buffer ``op`` reads or writes, by name."""
    return {name: tensor.shape for name, tensor in {**op.inputs, **op.outputs}.items()}


def lift_kernel(loop: LoopOp, *, name: str, shapes: Mapping | None = None) -> TileOp:
    """One kernel formed from its program, a fused region and a piece alike: its free coordinates canonical over the
    buffer ``shapes`` (``loop``'s own io by default), the complete nest as one Fold tree, then, for a contraction
    that owns no free axis, its size-one row bound back so a tier has a row to tile."""
    if (body := canonical_free_axes(loop.body, io_shapes(loop) if shapes is None else shapes)) is not None:
        loop = replace(loop, body=body)
    tile = lift_loop_op(loop, name=name)
    # A contraction that owns no free axis has no row for any tier to tile. Its row is a size-one
    # output dimension Loop-IR normalization inlined; bound back, the term keeps every per-cell
    # choice it had and gains the fragment ones beside them.
    for position in row_candidates(loop, tile):
        bound = lift_loop_op(loop, name=name, body=row_bound_body(loop, position, ROW_AXIS))
        if binds_the_row(bound, ROW_AXIS):
            return bound
    return tile


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
    lift cannot take whole keeps the piece as minted.

    A piece the rank rule left a SWEEP (``promoted_sweep``) keeps its minted form too. Re-forming
    walks the piece out to a full loop nest and lifts it back, and that nest opens the sweep loop
    AROUND the statistic the minted term holds beside it — the re-lifted reduce then reads the sweep
    axis, the rank rule promotes it, and the piece is back to folding its row statistic per output
    cell. The hoist is the form the reform is meant to preserve, so a piece that already has it is
    not re-formed. ``shapes`` names every buffer the piece touches, as its own program does."""
    piece = _align_owned_sweeps(piece)
    if any(store.sweep for store in piece.output_specs):
        return piece
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
    # sits ahead of it; the piece keeps the grid it was minted with, and an axis peeled past it
    # goes back to being the sweep of the stores that ride it.
    grid, peeled = formed.place.free[: len(piece.place.free)], formed.place.free[len(piece.place.free) :]
    # A piece with no contraction opens its grid in the order the store writes, last axis fastest,
    # so consecutive threads write consecutive addresses. The lift peels loops in nest order, which
    # put a RoPE piece's head-dim slowest: uncoalesced, 39 us on the H100 for a 2 MB write. A
    # contraction's grid order is not linearization only: its last two axes are the fragment's
    # rows and columns, and reordering them hands the rows to an axis an operand varies over.
    if not any(site.node.as_contraction() is not None for site in sites(formed.op)):
        written = [name for spec in formed.output_specs[:1] for index in spec.write.index for name in sorted(index.free_vars())]
        grid = tuple(sorted(grid, key=lambda axis: written.index(axis.name) if axis.name in written else len(written)))
    specs = tuple(
        replace(spec, sweep=(*spec.sweep, *(axis for axis in peeled if any(axis.name in index.free_vars() for index in spec.write.index))))
        for spec in formed.output_specs
    )
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
    return replace(piece, op=rewrite_twisted(formed.op, formed.axes), place=place, axes=formed.axes, output_specs=specs, source=loop)
