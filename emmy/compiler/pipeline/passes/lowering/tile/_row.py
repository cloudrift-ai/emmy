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
