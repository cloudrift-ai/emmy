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

from dataclasses import replace

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Body, Load, Loop, Stmt, Write
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline.passes.tile._fromloop import lift_loop_op
from emmy.compiler.pipeline.passes.tile._twist import rewrite_twisted

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
    shapes = {buffer: tensor.shape for buffer, tensor in (*op.inputs.items(), *op.outputs.items())}
    return Body((Loop(axis=axis, body=_bind(Body.coerce(op.body), axis, position, shapes)),))


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


def lift_kernel(loop: LoopOp, *, name: str) -> TileOp:
    """One kernel's program lifted as the lift pass lifts it: the complete nest as one Fold tree,
    then, for a contraction that owns no free axis, its size-one row bound back so a tier has a
    row to tile."""
    tile = lift_loop_op(loop, name=name)
    # A contraction that owns no free axis has no row for any tier to tile. Its row is a size-one
    # output dimension Loop-IR normalization inlined; bound back, the term keeps every per-cell
    # choice it had and gains the fragment ones beside them.
    for position in row_candidates(loop, tile):
        bound = lift_loop_op(loop, name=name, body=row_bound_body(loop, position, ROW_AXIS))
        if binds_the_row(bound, ROW_AXIS):
            return bound
    return tile


def reformed(piece: TileOp) -> TileOp:
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
    not re-formed."""
    if any(store.sweep for store in piece.output_specs):
        return piece
    body = piece.op.lower(bound=frozenset(), stores=piece.output_specs, axes=piece.axes)
    try:
        # Through the LoopOp's normalization: that is where two reduce loops over one axis become
        # one loop with two accumulators, the twin the lift forms one term from.
        formed = lift_kernel(LoopOp(body=body), name=piece.name)
    except ValueError:
        return piece
    # The lift peels every outer plain loop into the grid, a store's sweep included when nothing
    # sits ahead of it; the piece keeps the grid it was minted with, and an axis peeled past it
    # goes back to being the sweep of the stores that ride it.
    grid, peeled = formed.place.free[: len(piece.place.free)], formed.place.free[len(piece.place.free) :]
    specs = tuple(
        replace(spec, sweep=(*spec.sweep, *(axis for axis in peeled if any(axis.name in index.free_vars() for index in spec.write.index))))
        for spec in formed.output_specs
    )
    place = replace(formed.place, free=grid)
    # The loop op the piece was formed from rides as its ``source``, as a fused kernel's does: the body a
    # kernel row stores and a freeze re-lowers is the one the lift was given.
    loop = LoopOp(body=body, name=piece.name)
    return replace(piece, op=rewrite_twisted(formed.op, formed.axes), place=place, axes=formed.axes, output_specs=specs, source=loop)
