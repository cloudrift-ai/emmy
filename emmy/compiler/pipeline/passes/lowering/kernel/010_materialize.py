"""Materialize a ``TileOp``'s schedule into a ``KernelOp``.

Binds the schedule's grid axes to GPU threads and realizes the reduce partition through the **one**
node-kind dispatcher, ``_factor.factorize`` — every ``TileOp`` root (a tiled contraction, a
cooperative / ILP reduce, or a pointwise / scalar cell) lowers through that single emitter, which
reads the node kind + role + reduce plan off ``tile.op`` and picks the tier:

- **Tiled ``CONTRACTION``** (warp / register tile) — ``factorize`` synthesizes its bare
  grid-``Write`` (needs ``root.output``, so it can't ride the node) and expands it (mma → the
  ``RegFragment`` / fragment load / ``MmaSyncPtx`` / ``RegStore`` fragment soup; scalar → the
  per-thread register cell tile) through the shared four-level tiling layer (in ``_factor``).
- **Reduce tier** — a ``PLANAR`` / ``TWISTED`` reduce (or a non-output-tiled ``CONTRACTION``) whose
  ``Reduce`` carries a BLOCK ``coop`` and/or a REG ``reg`` stage: the reduce axis is partitioned
  ``coop`` ways across the CTA's threads (cooperation) and ``reg`` ways across per-thread register
  accumulators (ILP), then a REG-tree fold, the cross-thread combine, and the projection.
- **Scalar tier** — one thread per output cell (``op.lower()`` + an output-store glue).

The op tree + ``lower`` are shared across kinds; only the schedule's partition changes — the
article's "schedule separate from combine" thesis. The tier machinery all lives in ``_factor``.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.dim import Dim
from emmy.compiler.graph import Node
from emmy.compiler.ir.expression import BinaryExpression, Literal, Var
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.kernel.ir import ELEM_COL, ELEM_ROW, FRAG_COL, FRAG_ROW, THREAD_COORDS
from emmy.compiler.ir.pure.fold import EXTENTS
from emmy.compiler.ir.schedule.register import RegisterMaterialization
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Body, Load, Write, exposed_defines, free_names
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.ops import UnbindableProjection, reduce_plan, sched_of
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.kernel._factor import factorize

PATTERN = [Pattern("root", TileOp)]


def rewrite(match: Match, root: Node, ctx=None) -> KernelOp | None:
    tile: TileOp = root.op
    # By the kernel pass no schedule slice can carry a cross-CTA ``GRID`` stage: the split is the
    # structural ``tile/030_cut`` fork's, decided BEFORE scheduling (the walk's catalog
    # offers no ``g`` row, and its pin path strips the consumed ``g`` half). A surviving split
    # request is a bug — the materializer only lowers single-launch kernels.
    resident = isinstance(tile.materialization, RegisterMaterialization)
    if tile.carries and not resident:
        from emmy.compiler.pipeline.passes.tile._fromloop import serial_form  # noqa: PLC0415

        tile = serial_form(tile, root.id)
    rplan = reduce_plan(tile) if tile.op is not None and not resident else None
    assert rplan is None or not rplan.needs_split, "materialize: a GRID split stage reached the kernel pass past 030_cut"
    try:
        materialized = factorize(tile, root, sm_count=getattr(ctx, "sm_count", 0))
        if not resident:
            materialized = _pointwise_strip(tile, materialized)
        body = Body((materialized,))
        unbound = _unbound_names(tile, root, body)
        assert not unbound, f"materialize: kernel {tile.name!r} reads names it never binds: {sorted(unbound)}"
        kernel = KernelOp(body=body, name=tile.name, serial=() if resident else tuple(tile.place.serial))
        if resident:
            # The state's port — the buffer the lift added for the classic realization, which no
            # store of the term writes. Register storage has no global state allocation, so the
            # port goes: a graph splice; snapshots with external readers remain ordinary outputs.
            return _drop_private_ports(match, root, replace(kernel, source=tile, knobs=tile.knobs), "register")
        # An output no statement writes is a store the emission lost: the kernel would compile and
        # leave the buffer unwritten. Decline the row, so the retry moves on or the compile fails loudly.
        written = {name for stmt in body.iter() for name in stmt.external_writes()}
        if unwritten := [t.name for t in root.outputs if t.name not in written]:
            raise RuleSkipped(f"kernel {tile.name!r} declares outputs no statement writes: {unwritten}", reject=True)
        return kernel
    except UnbindableProjection as exc:
        # The offered row has no multi-root binding (e.g. it tiles two contraction operands of a
        # projection whose outputs do not partition by root). The row stays OFFERED — the
        # realization corpus pins that — and the compile declines it here: the skip is recorded,
        # the node stays a TileOp, and the greedy blocklist retry resolves onto the next row.
        raise RuleSkipped(f"kernel binder refuses this row's projection ownership: {exc}", reject=True) from exc


def _drop_private_ports(match: Match, root: Node, kernel: KernelOp, suffix: str):
    """Remove unused global state allocations after a resident materialization."""
    written = {name for stmt in kernel.body.iter() for name in stmt.external_writes()}
    ports = {t.name for t in root.outputs if t.name not in written and t.name not in match.graph.outputs}
    ports = {name for name in ports if not match.graph.buffer_users(name)}
    if not ports:
        return kernel
    from emmy.compiler.pipeline.passes.tile._cut import _input_fragment  # noqa: PLC0415

    outputs = tuple(t for t in root.outputs if t.name not in ports)
    names = {t.name: t.name + "__" + suffix for t in outputs}
    fragment = _input_fragment(match, root)
    fragment.add_node(
        replace(kernel, body=kernel.body.rename_buffers(names), outputs={}),
        list(root.inputs),
        outputs=(outputs[0], *(replace(t, name=names[t.name]) for t in outputs[1:])),
        node_id=names[outputs[0].name],
    )
    fragment.outputs = list(names.values())
    match.output = names
    return fragment


#: Names a statement may read with no binding in the tile's body: a thread's lane and warp, which the grid decode
#: binds (``THREAD_COORDS``), and names the RENDERER supplies — the reserved coordinates a
#: :class:`~emmy.compiler.ir.kernel.ir.FragmentApply` COORD / GMEM template is written over, which the render
#: substitutes per element (tile origin + layout offset), and the in-fragment offsets a
#: :class:`~emmy.compiler.ir.kernel.ir.RegStore` epilogue reads.
_RENDERED_HELPERS = frozenset({*THREAD_COORDS, FRAG_ROW, FRAG_COL, ELEM_ROW, ELEM_COL})


def _unbound_names(tile: TileOp, root: Node, body: Body) -> set[str]:
    """Names the emitted kernel body reads that its launch never supplies — always a compiler bug.

    A well-formed kernel reads its own buffers, the symbolic extents its launch passes beside them
    (the free vars of the axis table's extents — a ``Dim('seq_len')`` reduce bound, a ceil-div block
    count — and of its buffers' shapes, which a scatter's guard compares an index with) and the
    thread coordinates and the renderer's helpers, and nothing else: every other name is bound by a statement or by
    an enclosing axis, which is exactly what :func:`free_names` subtracts. What survives is a
    value some emission referred to under a spelling nothing defines — a per-cell rename whose shared
    coordinates missed an axis, a staged fill whose σ left a tile axis free, a workspace read the
    boundary store was not re-spelled for. Each of those reached ``nvcc`` as *identifier "x" is
    undefined*, a hundred errors deep in a generated source and attributed to whichever candidate the
    tuner happened to be benching. Asking it here names the kernel and the value instead, in the pass
    that built them."""
    serial = (axis.name for axis in tile.place.serial)
    bound = {*root.inputs, *root.buffer_names(), *_RENDERED_HELPERS, *serial, *tile.extents, *EXTENTS.get()}
    for axis in (*tile.axes, *tile.place.free):
        bound |= set(axis.extent_expression().free_vars())
    return set().union(*(free_names(stmt) for stmt in body)) - bound


def _pointwise_strip(tile: TileOp, materialized):
    """Realize a root projection's register strip after its stable schedule sites were consumed."""
    if tile.op is None or tile.op.axis is not None or tile.schedule is None:
        return materialized
    plan = sched_of(tile).get("TILE", tile.op)
    if plan is None:
        return materialized
    inner = tile.place.free[-1]
    width = plan.reg_n
    if not inner.extent.is_static or inner.extent.as_static() % width:
        raise ValueError("pointwise TILE must divide the static inner free extent")
    definitions = {name for stmt in materialized.body for name in exposed_defines(stmt)}
    loads = []
    computes = []
    writes = []
    for offset in range(width):

        def rename(name: str, offset: int = offset) -> str:
            return f"{name}__u{offset}" if name in definitions else name

        sigma = Sigma(
            {
                inner.name: BinaryExpression(
                    "+",
                    BinaryExpression("*", Var(inner.name), Literal(width, "int")),
                    Literal(offset, "int"),
                )
            }
        )
        for stmt in materialized.body:
            rewritten = stmt.rewrite(rename, sigma)
            target = loads if isinstance(rewritten, Load) else writes if isinstance(rewritten, Write) else computes
            target.append(rewritten)
    axes = tuple(
        replace(axis, extent=Dim(axis.extent.as_static() // width)) if axis.name == inner.name else axis for axis in materialized.axes
    )
    return materialized.replace(axes=axes, body=Body((*loads, *computes, *writes)))
