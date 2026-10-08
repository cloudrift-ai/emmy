"""Offer and realize the cross-CTA reduce split as a STRUCTURAL graph rewrite.

The split changes the kernel SET, exactly like a ``PLACE`` cut: a kernel that splits does not run, so
the split is decided from the pieces it leaves (``policy/greedy._kernel_set_pick``), which is why it is
the ``030_cut`` fork and not a schedule row. It runs BEFORE scheduling — the
rewrite consumes only the stored :class:`Fold` algebra, never a schedule decision — and each piece
is a fresh unmapped :class:`TileOp` that re-enters the pass scan and decides its own row at
``040_schedule`` like any newly lifted tree.

Each partial evaluates the same ``Fold(init, combine)`` over a contiguous axis slice and writes its
complete state tuple. The deferred finalize identity-lifts those tuples through the same ``init``
and ``combine``, then applies the original projection. This is the common path for additive and
exp-family monoids; the split does not recognize carrier families. A CONTRACTION slices through
``Fold.contraction`` over the σ-reindexed operand edges — the cone's row-invariant prologue stays
FULL-ROW in every partition (the redundant-statistic split); any other fold slices through the
generic ``Fold.rewrite``.

The atomic arm is the generic exception: it is legal only for a single additive state component
whose projection distributes over addition. Otherwise the deferred f32 workspace preserves the
full state until the finalize combines and projects it.

Every piece is a fresh unmapped :class:`TileOp`. A graph splice restarts the lowering pass scan:
scheduling offers each piece its own row. An axis :class:`Window` records that the partition has
already been consumed and prevents recursive splitting — the receipt is the IR itself, no flag.

A kernel that CARRIES A STATE has a split of its own, across the sequence (:func:`realize_carry_split`):
the parts of a step affine in the state compose as affine maps, so a probe reads each part's map
off two walks from known seeds, a prefix carries the state across the parts, and the walk itself
runs every part from the state it starts from. Its pieces are Loop IR lifted like the walk.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import BF16, F16, F32
from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.address import gmem_axis_step
from emmy.compiler.ir.axis import Axis, Window
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.pure import Lambda
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.pure.twist import Twist
from emmy.compiler.ir.schedule import Reduce, Work
from emmy.compiler.ir.schedule.catalog import splitk_moves
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Assign, Body, Carry, Let, Load, Loop, Pre, Select, SelectBranch, Write
from emmy.compiler.ir.stmt.passes import projection_distributes
from emmy.compiler.ir.tile import OutputSpec, Placement, TileOp
from emmy.compiler.ir.tile.ops import Sched, carries_partition, head, projection_regions, projection_root, projection_tail
from emmy.compiler.pipeline import Match
from emmy.compiler.pipeline.fork import DeferredFork
from emmy.compiler.pipeline.knob import axis_of, consume_kernel_row, kernel_pin
from emmy.compiler.pipeline.passes.tile._row import io_shapes, lift_kernel, reformed
from emmy.compiler.pipeline.search.space import REDUCE, WORK
from emmy.compiler.structural import digest

logger = logging.getLogger(__name__)

_SPLIT = "_ksplit"  # the cross-CTA split grid axis


# ---- what a kernel / an axis can carry -------------------------------------------------------- #


def splitk_width(k_axis: Axis, width: int) -> str | None:
    """A cross-CTA split needs a STATIC reduce axis its width divides evenly — the σ-reindex
    reconstructs an absolute k from ``ksplit·(K/w) + kslice``, which is a bijection only when the
    extent is known and ``w`` divides it. Total over a symbolic axis rather than raising out of
    ``as_static``, so the catalog drops the width and a pin reports the reason."""
    if not k_axis.extent.is_static:
        return f"cross-CTA split of the symbolic reduce axis {k_axis.name!r} is not built"
    big_k = k_axis.extent.as_static()
    if big_k % width == 0:
        return None
    return f"split-K width {width} does not divide K={big_k}; pick a dividing split width."


def _direct_atomic_output(outputs) -> str | None:
    """Whether direct cross-CTA partials avoid another low-precision rounding step.

    F16/BF16 destinations round once per CTA; the deferred finalize instead combines f32
    carrier state and rounds once at the output boundary."""
    lowp = sorted({str(t.dtype) for t in outputs.values() if t.dtype in (F16, BF16)})
    if not lowp:
        return None
    return (
        f"direct atomic REDUCE writes each partial into {'/'.join(lowp)} output storage; "
        "use the deferred f32 workspace finalize (REDUCE=g<n>k) so the output rounds once"
    )


def atomic_finalize(node: Fold, tail, outputs) -> str | None:
    """Whether the cross-CTA split may take its DIRECT ``atomicAdd`` arm — every condition, once,
    beside the move it filters. The deferred workspace finalize (``REDUCE=g<n>k``) carries any
    carrier and any projection, so each refusal names it as the alternative. Four things must
    hold, and they are stated together because the offer, the pin, and the realization all have to
    agree — a pin that reached the rewrite past a refusal the offer applied would crash the
    emitter instead of refusing:

    - ONE state component. ``atomicAdd`` folds a scalar; a twisted carrier's
      ``(maximum, denominator, …)`` tuple has no atomic instruction at all.
    - An ADDITIVE ⊕. The emitted instruction IS ``atomicAdd`` — a ``max`` carrier's partials
      would be SUMMED, silently wrong (the deferred finalize folds any monoid).
    - An output the partials can round into once per CTA (:func:`_direct_atomic_output`).
    - A projection that DISTRIBUTES over the add. The atomic arm applies the epilogue per
      partition, before the combine, so anything but a linear-homogeneous map mis-scales each
      CTA's contribution.

    Storage is asked BEFORE the projection: a narrowing store spells its rounding as a conversion
    in that same epilogue (``loop/lifting/090_spell_store_rounding``), which no linear-homogeneous
    reading admits — so both refuse a low-precision output and the storage reason is the one that
    names why.

    ``tail`` differs between the two askers: the OFFER passes the kernel's whole projection tail,
    the realization the MIMO-selected region's. The offer's read is the superset, so the only
    possible divergence is OVER-refusal at the offer (a sibling region's non-distributive stmt
    refusing an atomic the owned region could carry) — safe, since a withheld atomic row leaves
    the deferred finalize, never a crash past an offer the realizer cannot honor."""
    states = node.combine.results
    if len(states) != 1:
        return (
            f"atomic REDUCE folds ONE additive state component; this carrier has {len(states)} "
            f"({', '.join(states)}) — use the deferred f32 workspace finalize (REDUCE=g<n>k)"
        )
    ops = node.as_reduction().ops
    if ops is None or ops[0].name != "add":
        plus = "a twisted combine" if ops is None else f"⊕ = {ops[0].name}"
        return f"atomic REDUCE emits atomicAdd, which folds only an ADDITIVE carrier; this one has {plus} — use REDUCE=g<n>k"
    storage = _direct_atomic_output(outputs)
    if storage is not None:
        return storage
    if tail and not projection_distributes(tuple(tail), states):
        return (
            "atomic REDUCE applies the projection epilogue per partition, so it must distribute "
            "over the add; this one does not (a fused bias / activation) — use the deferred "
            "workspace finalize (REDUCE=g<n>k), which projects once after the combine"
        )
    return None


def _enforce(reason: str | None) -> None:
    """Raise a refusal a PIN ran into — the offer's catalog arm drops instead."""
    if reason is not None:
        raise ValueError(reason)


def _reducing_roots(op: Fold) -> tuple[Fold, ...]:
    """The DISTINCT reducing roots a projection's operands carry — the head fold reached through
    its epilogue and again as a shared operand is one root, and a provider term carries none."""
    out: list[Fold] = []
    for edge in op.operands:
        root = projection_root(edge)
        if root is not None and all(root is not seen for seen in out):
            out.append(root)
    return tuple(out)


def _projection_refusal(tile: TileOp, node) -> str | None:
    """Why the kernel's projection cannot survive a split of ``node`` (``None`` when it can) — the
    MIMO decomposition the realizer performs, asked at the OFFER so an unrealizable split is never
    offered: an independent-projection kernel must partition into output-owning regions and the
    split fold must own one of them (a projection reading SEVERAL roots into one output has no
    piece to hand the epilogue to). The residence guard leads: a head fold the realization cannot
    STRIP from the projection — neither the kernel's own node, an operand edge, nor a top-level
    projection member (``head``'s sweep case: a fold reading the boundary store's sweep axis lands
    inside the sweep ``Loop`` ``apply_output_specs`` wraps) — would leave the epilogue re-running
    the whole reduction and shadowing the workspace states, so the split declines there."""
    op = tile.op
    if (
        op is not node
        and all(edge is not node for edge in getattr(op, "operands", ()))
        and not any(stmt is node for stmt in projection_tail(tile))
    ):
        return "the head fold is nested inside the projection's sweep loop; the split cannot strip it"
    if any(axis.name in node.free_axes for spec in tile.output_specs for axis in spec.sweep):
        return "the head fold is evaluated inside the boundary store's sweep loop; the split cannot strip it"
    if not isinstance(op, Fold) or op.axis is not None or len(op.operands) < 2:
        return None
    if len(_reducing_roots(op)) < 2:
        return None  # one reducing root beside its providers (the chain form): the split owns the whole projection
    try:
        regions = projection_regions(op, tile.output_specs)
    except ValueError as e:
        return str(e)
    if not any(fold is node for fold, *_ in regions):
        return "the split fold does not own an independent projection region"
    return None


def _statistic_refusal(node: Fold) -> str | None:
    """Why the generic slicer cannot split ``node`` (``None`` when it can): an operand that reduces the
    fold's own axis is a statistic of the WHOLE row — the max under a softmax's sum, the mean square
    under a normed row's dot product — and the pieces' axis table holds one axis per name, so the slice
    narrows that axis for every fold that names it and each partition would take the statistic over
    one slice. A contraction head is not refused: its slicer keeps the statistic full-row."""

    def reduces_axis(edge) -> bool:
        return getattr(edge, "axis", None) == node.axis or any(map(reduces_axis, getattr(edge, "operands", ())))

    if node.as_contraction() is None and any(map(reduces_axis, node.operands)):
        return "an operand reduces the split axis itself; each partition would take that statistic over one slice"
    return None


# ---- the offer: the unsplit tree beside every split the head fold admits ---------------------- #


def split_pending(tile: TileOp) -> bool:
    """Whether this kernel still has a cross-CTA split decision for ``030_cut`` to consume."""
    node = head(tile.op)
    return (
        node is not None
        and node.axis is not None
        and node.combine is not None
        and node.observe is None
        and not node.carries
        and not carries_partition(tile)
        and not tile.split_consumed
    )


def split_forks(match: Match, root: Node, *, unsplit_tile: TileOp | None = None) -> list[DeferredFork] | None:
    """The split fork for ``root``'s kernel — the unsplit tree first, then one STRUCTURAL option
    per :func:`splitk_moves` member the head fold admits — or ``None`` when there is nothing to
    decide (no reduce fold, or the kernel is itself a piece of a realized split: the sliced axis's
    partition ``Window`` is the receipt, so the pieces re-entering the cut fixpoint and skip here;
    an ambient pin can never split twice). ``040_schedule`` then consumes the same receipt when it
    strips the pin's ``g`` half before composing each piece's own schedule.

    A ``REDUCE`` pin is authoritative over its cross-CTA ``g<n>[a|k]`` half and ONLY that half:
    the rest of the value (``coop`` / ``r<n>``) is the pieces' own schedule, which the walk reads
    off the same pin minus the consumed stage. A pin naming a split the head fold cannot carry
    raises the recorded refusal (``REDUCE`` has no choice of tier, so there is no drop layer);
    a pin with no ``g`` half decides UNSPLIT, exactly as a spelled row with no ``g`` half does."""
    tile: TileOp = root.op
    # Worker-built arms cannot publish Match mutations back to the parent that splices them.
    match.output = {name: f"{name}__split" for name in root.buffer_names()}
    node = head(tile.op)
    if node is not None and node.carries:
        return _carry_split_forks(match, root, tile, node)
    if not split_pending(tile):
        return None
    assert node is not None and node.axis is not None
    k_axis = tile.axis_of(node.axis)  # the node names its K; the kernel's axis table holds its extent
    key = Sched(tile).key("REDUCE", node) or "REDUCE"
    unsplit = DeferredFork(lambda: replace(unsplit_tile or tile, split_consumed=True), {key: ""})
    element = axis_of(key)
    # A kernel pin names the piece by its name, a ``node_`` pin the uncut root by its node id; the
    # schedule pass reads them the same way (``040_schedule.pin_row``).
    pin = kernel_pin("REDUCE", tile.name, root.id)
    if pin is None:
        pin = REDUCE.narrow_at(element) if element else REDUCE.raw()
    tail = projection_tail(tile)
    if pin is not None:
        work = kernel_pin("WORK", tile.name, root.id)
        plan = Reduce.parse(pin, Work.parse(work if work is not None else WORK.raw()))
        if not plan.needs_split:
            return [unsplit]
        _enforce(splitk_width(k_axis, plan.cta))
        _enforce(_projection_refusal(tile, node) or _statistic_refusal(node))
        if plan.finalize == "atomic":
            _enforce(atomic_finalize(node, tail, tile.outputs))
        return [_split_fork(match, root, key, plan.cta, plan.finalize)]
    if (why := _projection_refusal(tile, node) or _statistic_refusal(node)) is not None:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("no split offered: %s", why)
        return [unsplit]
    options: list[DeferredFork] = [unsplit]
    atomic_why = atomic_finalize(node, tail, tile.outputs)
    for plan in splitk_moves():
        why = splitk_width(k_axis, plan.cta) or (atomic_why if plan.finalize == "atomic" else None)
        if why is not None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("split g%d%s not offered: %s", plan.cta, plan.finalize[0], why)
            continue
        options.append(_split_fork(match, root, key, plan.cta, plan.finalize))
    return options


def _split_fork(match: Match, root: Node, key: str, cta: int, finalize: str) -> DeferredFork:
    spelling = Reduce.of(cta=cta, finalize=finalize).spell()
    return DeferredFork(lambda: realize_split(match, root, cta, finalize), {key: spelling}, structural=True)


# ---- a carried state: the split across the sequence ------------------------------------------- #


def _carry_split_forks(match: Match, root: Node, tile: TileOp, node: Fold) -> list[DeferredFork] | None:
    """The split fork for a kernel that CARRIES A STATE — the unsplit walk beside one structural
    option per width the step axis divides into — or ``None`` when there is nothing to decide: a
    step that is not affine in its state (:meth:`Fold.affine`), a block the probe cannot pack
    (fewer kept than mixed cells), a walk under a free loop outside the carrying one, a piece of a
    realized split. Only the deferred finalize exists here (``g<n>k``): the parts' maps compose in a
    kernel, there is nothing to add atomically."""
    view = node.affine()
    if view is None or carries_partition(tile) or tile.split_consumed or tile.place.free:
        return None  # a free loop outside the carrying one: the realizer rebuilds the carrying nest alone
    time = tile.axis_of(node.axis)
    if view.mixed is not None:
        mixed, kept = (tile.axis_of(node.cells[position]).extent for position in (view.mixed, view.kept))
        if not (mixed.is_static and kept.is_static) or kept.as_static() < mixed.as_static():
            return None
    key = Sched(tile).key("REDUCE", node) or "REDUCE"
    unsplit = DeferredFork(lambda: replace(tile, split_consumed=True), {key: ""})
    element = axis_of(key)
    pin = kernel_pin("REDUCE", tile.name, root.id)
    if pin is None:
        pin = REDUCE.narrow_at(element) if element else REDUCE.raw()

    def arm(cta: int) -> DeferredFork:
        return DeferredFork(lambda: realize_carry_split(match, root, cta), {key: Reduce.of(cta=cta).spell()}, structural=True)

    if pin is not None:
        work = kernel_pin("WORK", tile.name, root.id)
        plan = Reduce.parse(pin, Work.parse(work if work is not None else WORK.raw()))
        if not plan.needs_split:
            return [unsplit]
        _enforce(splitk_width(time, plan.cta))
        if plan.finalize != "kernel":
            raise ValueError("a carried state's split composes its parts in a finalize kernel: spell it REDUCE=g<n>k")
        return [arm(plan.cta)]
    options = [unsplit]
    for plan in splitk_moves():
        if plan.finalize == "kernel" and splitk_width(time, plan.cta) is None:
            options.append(arm(plan.cta))
    return options


def state_ports(tile: TileOp, prefix: str) -> tuple[Tensor, ...]:
    """The buffers the classic realization of ``tile``'s carried states owns, named under
    ``prefix`` — one per state over ``(time, *free, *cells)``, the index ``states_as_buffers``
    writes. The lift adds them to the node; register storage drops them again."""
    outer = tuple(axis.name for axis in tile.place.free)
    return tuple(
        Tensor(name=f"{prefix}__{state}", shape=tuple(tile.axis_of(axis).extent for axis in (node.axis, *outer, *node.cells)), dtype=F32)
        for node in (site.node for site in tile.sites if site.node.carries)
        for state in node.base.results
    )


def _nest(body: Body, axes: tuple) -> Body:
    """``body`` under loops over ``axes``, outermost first."""
    for axis in reversed(axes):
        body = Body((Loop(axis=axis, body=body),))
    return body


def realize_carry_split(match: Match, root: Node, cta: int) -> Graph:
    """Split a carried state's walk across the sequence into ``cta`` parts — three kernels, each
    lifted from Loop IR like the walk itself:

    1. the PROBE walks every part's range twice from known seeds, zero and the identity packed
       along the kept cells, and stores every step's state: a step affine in the state composes as
       an affine map, so a part's map is read off the two walks — its offset is the zero walk's last
       state, its matrix the difference of the two walks' last states;
    2. the PREFIX carries the state across the parts, applying each part's map to the state the
       part before it left, and stores the state every part starts from;
    3. the WALK is the kernel itself over each part's range from that state, one part per batch
       cell, storing what the kernel stored.

    Three walks of a part's range where the sequence took one, for ``cta`` times the parallelism:
    the price a card mostly idle on one CTA per head pays gladly, and evidence's to decide."""
    tile: TileOp = root.op
    node = head(tile.op)
    view = node.affine()
    assert view is not None, "the split offer fires on affine carried states only"
    (state,) = node.base.results
    cells = node.cells
    cell_axes = tuple(tile.axis_of(name) for name in cells)
    own = tuple(Var(name) for name in cells)
    time = tile.axis_of(node.axis)
    steps = time.extent.as_static() // cta
    out = root.output
    body = tile.loop_body
    (position,) = (index for index, stmt in enumerate(body) if isinstance(stmt, Loop) and stmt.carries)
    loop, before, after = body[position], tuple(body[:position]), tuple(body[position + 1 :])
    part, probe = Axis("_part", Dim(cta)), Axis("_probe", Dim(2))
    sliced = replace(loop.axis, extent=Dim(steps), window=Window(parent=loop.axis.source_axis or loop.axis, partition=True))
    sigma = Sigma({loop.axis.name: BinaryExpr("+", BinaryExpr("*", Var(part.name), Literal(steps, "int")), Var(loop.axis.name))})
    step = Body(tuple(stmt.substitute(sigma) for stmt in loop.body))
    probe_seed, probe_states, starts = f"{out.name}__probe_seed", f"{out.name}__probe", f"{out.name}__start"

    def indexed(stmts: Body, lead: tuple[Axis, ...], seed: str) -> Body:
        # The state under the lead coordinates too — a part, a probe — and seeded from ``seed``.
        prefix = tuple(Var(axis.name) for axis in lead)

        def rewrite(stmt):
            if isinstance(stmt, Carry):
                return replace(stmt, index=(*prefix, *stmt.index), seed=seed)
            if isinstance(stmt, Pre) and stmt.carrier == state:
                return replace(stmt, index=(*prefix, *stmt.index))
            return stmt

        return stmts.map(rewrite)

    # The probe: the walk with its stores replaced by one store of the state every step.
    def probed(stmt):
        if isinstance(stmt, Write):
            return None
        if isinstance(stmt, Loop) and not stmt.is_reduce and not stmt.body.carries and not stmt.body.iter_of_type(Write):
            return None  # an output sweep with nothing left to store
        if isinstance(stmt, Carry):
            return (stmt, Write(output=probe_states, index=(Var(probe.name), Var(part.name), Var(sliced.name), *own), value=stmt.value))
        return stmt

    walk = indexed(step, (probe, part), probe_seed).map(probed)
    probe_body = Body((*before, *_nest(walk, (sliced, probe, part)), *after))
    # The probe's seeds: zero, and the identity between the mixed and the kept cell — ones
    # everywhere for a step reading its own cell alone, whose map is diagonal.
    hit = BinaryExpr(">", Var(probe.name), Literal(0, "int"))
    if view.mixed is not None:
        hit = BinaryExpr("&&", hit, BinaryExpr("==", own[view.mixed], own[view.kept]))
    seed_body = Body(
        (
            Let(name="_one", value=1.0),
            Let(name="_zero", value=0.0),
            Select(name="_seed", branches=(SelectBranch("_one", hit), SelectBranch("_zero", Literal(True, "bool")))),
            Write(output=probe_seed, index=(Var(probe.name), Var(part.name), *own), value="_seed"),
        )
    )
    seed_kernel = _nest(seed_body, (probe, part, *cell_axes))
    # The prefix: S ← A_p · S + b_p across the parts, A_p[i, k] the identity walk less the zero walk
    # at kept cell k, b_p the zero walk; each part's start is the state before its step. A diagonal
    # map applies cell by cell, with no contraction.
    last = Literal(steps - 1, "int")
    carried = f"{state}__across"

    def at(position: int | None, expr) -> tuple:
        return tuple(expr if index == position else coordinate for index, coordinate in enumerate(own))

    def applied(along) -> tuple:
        # ``A_p`` at this cell's row and column ``along`` times the state at column ``along``.
        return (
            Load(name="_one_walk", input=probe_states, index=(Literal(1, "int"), Var(part.name), last, *at(view.kept, along))),
            Load(name="_zero_walk", input=probe_states, index=(Literal(0, "int"), Var(part.name), last, *at(view.kept, along))),
            Assign(name="_a", op="subtract", args=("_one_walk", "_zero_walk")),
            Pre(name="_s", carrier=carried, index=at(view.mixed, along)),
            Assign(name="_as", op="multiply", args=("_a", "_s")),
        )

    if view.mixed is None:
        mapped = applied(None)
        product = "_as"
    else:
        basis = Axis("_basis", cell_axes[view.mixed].extent)
        mapped = (Loop(axis=basis, body=Body((*applied(Var(basis.name)), Accum(name="_acc", value="_as")))),)
        product = "_acc"
    prefix_body = _nest(
        Body(
            (
                *mapped,
                Load(name="_b", input=probe_states, index=(Literal(0, "int"), Var(part.name), last, *own)),
                Pre(name="_from", carrier=carried, index=own),
                Assign(name="_next", op="add", args=(product, "_b")),
                Carry(name=carried, value="_next", index=own, seed=node.init[0]),
                Write(output=starts, index=(Var(part.name), *own), value="_from"),
            )
        ),
        (part, *cell_axes),
    )
    # The walk: the kernel over each part's range, from the state the prefix stored for it.
    walk_body = Body((*before, *_nest(indexed(step, (part,), starts), (sliced, part)), *after))

    # Every buffer a piece reads or writes: the kernel's own and the three the split adds.
    extents = tuple(axis.extent for axis in cell_axes)
    shapes = io_shapes(tile) | {
        probe_seed: (Dim(2), Dim(cta), *extents),
        probe_states: (Dim(2), Dim(cta), Dim(steps), *extents),
        starts: (Dim(cta), *extents),
    }

    def piece(body: Body, name: str) -> TileOp:
        lifted = lift_kernel(LoopOp(body=body), name=name, shapes=shapes)
        return replace(lifted, knobs=consume_kernel_row(lifted.knobs))

    frag = _frag(match, root)
    seed_tile = piece(seed_kernel, f"{tile.name}__probe_seed")
    frag.add_node(op=seed_tile, inputs=[], output=Tensor(probe_seed, shapes[probe_seed], F32), node_id=probe_seed)
    probe_tile = piece(probe_body, f"{tile.name}__probe")
    frag.add_node(
        op=probe_tile,
        inputs=_piece_inputs(root, probe_tile, probe_seed),
        outputs=(Tensor(probe_states, shapes[probe_states], F32), *state_ports(probe_tile, probe_states)),
        node_id=probe_states,
    )
    prefix_tile = replace(piece(prefix_body, f"{tile.name}__prefix"), split_consumed=True)
    seeded = [node.init[0]] if isinstance(node.init[0], str) else []
    frag.add_node(
        op=prefix_tile,
        inputs=[probe_states, *seeded],
        outputs=(Tensor(starts, shapes[starts], F32), *state_ports(prefix_tile, starts)),
        node_id=starts,
    )
    walk_tile = piece(walk_body, tile.name)
    # The walk owns the outputs the kernel stored, not the state port the lift gave the kernel:
    # its own port replaces that one, travelling under a temporary until the splice hands it the
    # name, the way every replaced buffer does.
    written = output_root(root, {spec.write.output for spec in tile.output_specs})
    ports = tuple(
        replace(port, name=f"{port.name}__split") if port.name in root.buffer_names() else port for port in state_ports(walk_tile, root.id)
    )
    result = add_output_piece(match, frag, written, walk_tile, _piece_inputs(root, walk_tile, starts), states=ports)
    replaced = {port.name.removesuffix("__split"): port.name for port in ports if port.name.endswith("__split")}
    result.outputs.extend(replaced.values())
    match.output = {**match.output, **replaced}
    return result


# ---- slicing the head fold -------------------------------------------------------------------- #


def _slice_fold(fold: Fold, axis: Axis, b: int, split: Axis) -> tuple[Axis, Fold]:
    """``(the sliced axis, the same monoid Fold over one CTA's absolute contiguous slice)`` — the
    generic (non-contraction) slicer: the whole fold rides ``Fold.rewrite`` under the σ-offset."""
    offset = BinaryExpr("+", Var(axis.name), BinaryExpr("*", Var(_SPLIT), Literal(b, "int")))
    sigma = Sigma({axis.name: offset})
    sliced_axis = replace(axis, extent=Dim(b), window=Window(parent=axis.source_axis or axis, partition=True))
    # RE-DERIVED over a narrower axis, not renamed and not substituted-through: the fold keeps its
    # own binder (same name, sliced extent) while its operands' coordinates take the σ-offset. A
    # blanket σ would be refused as capture — this fold BINDS the name σ maps — and rightly so;
    # what changes here is the axis itself, which only the caller can say.
    operands = tuple(_sliced_edge(edge, sigma, axis.name, sliced_axis, split) for edge in fold.operands)
    body = Body(tuple(stmt.substitute(sigma) for stmt in fold.lift.body))
    # Re-CLOSED, exactly as the contraction slicer closes its own: a lift that reads the reduce
    # coordinate directly — a causal mask's ``row <= key`` — reads the partition coordinate once the
    # σ-offset lands, and keeping the original param list leaves it unbound. The cut that mints a
    # piece whose head IS such a masked reduce makes the split pass offer ``g<n>k`` on it, and greedy
    # prices a fork by materializing every arm, so merely OFFERING the split raised.
    lift = Lambda.closing(fold.lift.params, body, fold.lift.results)
    return sliced_axis, replace(fold, operands=operands, lift=lift)


def _factor_k(k_axis: Axis, w: int) -> tuple[Axis, Axis, Sigma]:
    """Factor a STATIC contraction axis into ``ksplit × kslice``. ``ksplit`` (extent ``w``, name
    ``_<k>_ks``) becomes the partial's lead grid axis, parallelized across CTAs and combined in
    the finalize; ``kslice`` (extent ``K/w``, the ORIGINAL name) is the sliced contraction's. The
    ``sigma`` maps the original ``k`` to ``ksplit·(K/w) + kslice`` so the operand loads
    reconstruct the absolute index; distinct names are what avoid a double-reduce. The slice
    carries its parentage: a cross-CTA split is CONSUMED by the rewrite that realizes it, and an
    axis that is already a partition window is one nothing may partition again."""
    b = k_axis.extent.as_static() // w
    ksplit = Axis(name=f"_{k_axis.name}_ks", extent=Dim(w))
    kslice = replace(k_axis, extent=Dim(b), window=Window(parent=k_axis.source_axis or k_axis, partition=True))
    sigma = Sigma({k_axis.name: BinaryExpr("+", BinaryExpr("*", Var(ksplit.name), Literal(b, "int")), Var(k_axis.name))})
    return ksplit, kslice, sigma


def _sliced_edge(edge, sigma: Sigma, k_name: str, kslice=None, ksplit: Axis | None = None):
    """An operand edge σ-reindexed to absolute k for a split partition — the SAME rule on either
    edge. A MATERIALIZED edge rewrites its gmem index; a COMPUTED cone rewrites its per-cell BODY
    and every K-VARYING producer edge it composes (attention's per-cell score contraction — the
    slice's own k coordinate reaches gmem through that node, so leaving it unreindexed makes every
    partition recompute partition 0's scores). The cone's row-invariant prologue (the per-row
    statistic the K seam reads off the node boundary) spans the whole row and stays FULL-ROW in
    every partition, each recomputing it — the REDUNDANT-STATISTIC split. That redundancy is what
    the split trades for parallelism; whether it pays on a given shape is evidence's decision."""
    if isinstance(edge, Load):
        return replace(edge, index=tuple(sigma.apply(e) for e in edge.index))

    ops = tuple(_sliced_edge(e, sigma, k_name, kslice, ksplit) if k_name in e.free_axes else e for e in edge.operands)
    body = Body(tuple(s.substitute(sigma) for s in edge.lift.body))
    # Re-CLOSED, like every other σ-reindexed lift here. σ maps one coordinate onto an expression
    # over TWO (slice and partition), and expanding a param in PLACE shifted the operand
    # correspondence the prefix carries — by a ``free_vars()`` set order, so which name landed there
    # moved with the hash seed and ``_prune_unread`` then dropped a param the body still read.
    # Closing keeps the prefix exactly and appends the partition coordinate where a coordinate sits.
    lift = Lambda.closing(edge.lift.params, body, edge.lift.results)
    return replace(edge, operands=ops, lift=lift)


def _sliced_contraction(node: Fold, k_axis: Axis, w: int) -> tuple[Axis, Axis, Fold]:
    """``(ksplit, kslice, sliced)`` for a contraction head: the SAME bilinear node a non-split matmul
    builds, over ``kslice`` with operands σ-reindexed to absolute k, threading the node's OWN
    semiring (the reassociation ``fold_k = fold_{ksplit} ∘ fold_{kslice}`` is licensed by that
    ⊕-monoid's associativity). ``Fold.contraction`` regenerates the componentwise ⊕ over the same
    accumulator names, so the finalize folds the workspace states through the same monoid."""
    ksplit, kslice, sigma = _factor_k(k_axis, w)
    # Rebuilt DIRECTLY over the σ-reindexed operands, in stored order: the slice is the same term
    # with a narrower axis, so its monoid and seeds are the node's own — there is nothing for a
    # former to re-derive, and no role to re-name.
    operands = tuple(_sliced_edge(edge, sigma, node.axis, kslice, ksplit) for edge in node.operands)
    # The LIFT takes σ too, per statement so the binder is not shadowed away — the same rule
    # :func:`_slice_fold` applies on the generic side. A lift that only weighs its operands reads
    # no k and this changes nothing; one that reads the contraction coordinate DIRECTLY is
    # comparing against a partition-local index while its operands already reach absolute k.
    # Causal attention is exactly that lift: its mask is ``row < key``, and left unreindexed every
    # partition above the first admits keys far above the diagonal.
    # Re-CLOSED, not re-spelled: the head's first param is its iteration binder, which must keep
    # its position, so the partition coordinate the substituted body now reads joins as a TRAILING
    # param — where a coordinate already sits, and past the operand correspondence.
    body = Body(tuple(stmt.substitute(sigma) for stmt in node.lift.body))
    lift = Lambda.closing(node.lift.params, body, node.lift.results)
    return ksplit, kslice, replace(node, operands=operands, lift=lift)


# ---- the piece / fragment builders ------------------------------------------------------------ #


def _cell_index(stores: tuple, free) -> tuple:
    """The output-cell index the original kernel writes (the projection ``Write``'s index,
    or — for a bare carrier whose grid-cell store is glue — the free-axis vars)."""
    return stores[0].write.index if stores else tuple(Var(ax.name) for ax in free)


def _frag(match: Match, root: Node) -> Graph:
    """A fragment seeded with the split node's inputs — the graph a piece is stamped against (its
    structural features fold in its operands' dtypes, which need the buffers)."""
    frag = Graph()
    for inp in root.inputs:
        frag.add_node(op=InputOp(), inputs=[], output=match.graph.buffer(inp), node_id=inp)
    return frag


def _piece_inputs(root: Node, body, *first: str) -> list[str]:
    """Return fragment buffers followed by external inputs actually read by a piece."""
    if isinstance(body, TileOp):
        body = body.op.lower(axes=body.axes)
    reads = {load.input for load in Body.coerce(body).loads}
    return [*first, *(inp for inp in root.inputs if inp in reads)]


def add_output_piece(
    match: Match, frag: Graph, root: Node, piece: TileOp, inputs: list[str], *, suffix: str = "__split", states: tuple = ()
) -> Graph:
    """Add a fresh piece with its owned output ports and arrange their splice identities.

    ``root`` is the graph-node view of the ports this piece owns — :func:`output_root` narrows a
    MIMO node to a subset — so slot 0's edge key travels as the node id and every other as its own
    tensor name. ``suffix`` names the temporary those buffers travel under while the fragment is
    spliced in: the cross-CTA split mints ``__split``, the kernel-placement cut ``__placed``. ONE
    suffix per rewrite, so a fragment's temporary names say which decision minted them. ``states``
    are buffers the piece writes that the replaced node never had — a recurrence's state."""
    buffers = root.buffer_names()
    renamed = {name: f"{name}{suffix}" for name in buffers}
    piece = replace(
        piece,
        output_specs=tuple(
            replace(spec, write=replace(spec.write, output=renamed.get(spec.write.output, spec.write.output)))
            for spec in piece.output_specs
        ),
    )
    tensors = (
        replace(root.outputs[0], name=buffers[0]),
        *(replace(tensor, name=renamed[name]) for name, tensor in zip(buffers[1:], root.outputs[1:], strict=True)),
    )
    frag.add_node(op=piece, inputs=inputs, outputs=(*tensors, *states), node_id=renamed[buffers[0]])
    frag.outputs.extend(renamed.values())
    output = dict(match.output) if isinstance(match.output, dict) else {}
    output.update(renamed)
    match.output = output
    return frag


def _one(match: Match, frag: Graph, root: Node, piece: TileOp) -> Graph:
    """The ATOMIC arm's one-kernel fragment. It replaces the split kernel with ONE kernel, but it
    is a SPLICE, never an op rebind: a rebind is how the engine says "the same kernel, decided
    further", so it merges the replaced op's knobs forward and does not restart the pass scan. The
    atomic partial is a different kernel — its own placement, its own body — and it has to reach
    scheduling carrying nothing of the kernel it replaced."""
    return add_output_piece(match, frag, root, piece, list(root.inputs))


def _wrap(body: Body, operands: tuple) -> Fold:
    """A zero-axis term over ``body``, exposing its last definition — what a projection returns."""
    bound = tuple(name for edge in operands for name in edge.exposes)
    results = next((stmt.defines()[-1:] for stmt in reversed(tuple(body)) if stmt.defines()), ())
    lift = Lambda.closing(bound, body, results)
    return Fold(operands=operands, lift=lift)


def _with_axes(axes: tuple, *new: Axis) -> tuple:
    """The axis table with ``new`` entries replacing same-named ones — a slice keeps its axis's name."""
    return tuple({**{axis.name: axis for axis in axes}, **{axis.name: axis for axis in new}}.values())


def _piece(op: Fold, free, *, output_specs: tuple = (), axes: tuple, name: str = "", shapes: dict) -> TileOp:
    """One fresh unscheduled Tile kernel over ``op`` and the axis table ``axes``, formed as its own kernel
    (:func:`~._row.reformed`) over the buffers ``shapes`` names: its nest lowered and lifted again, the loop op it
    came from kept as its source. An unnamed piece launches under its graph node's id."""
    piece = reformed(TileOp(op=op, place=Placement(free=tuple(free)), output_specs=output_specs, axes=axes, name=name), shapes)
    # A split CONSUMES the kernel it replaces: the piece drops its schedule row and its structural
    # identity. Built fresh here, so this states the contract rather than doing work — and the rule
    # that mints a kernel is where that has to be said.
    return replace(piece, knobs=consume_kernel_row(piece.knobs))


def _state_fold(axis: Axis, algebra: Fold, loads: tuple[Load, ...]) -> Fold:
    """Fold already-reduced state tuples through ``algebra``'s unchanged monoid."""
    values = tuple(name for load in loads for name in load.defines())
    return Fold(
        # The workspace reads are slabs like any other gmem read, over the split axis and the
        # output coordinates the enclosing placement binds.
        operands=tuple(Fold.slab(load) for load in loads),
        lift=Lambda(params=(axis.name, *values), body=Body(), results=values),
        init=algebra.init,
        base=algebra.base,
        # The SAME ⊕ the partials were produced under — but over finished carrier states read out
        # of the workspace, so there is no per-element contribution left for ψ to be applied to.
        twist=None if algebra.twist is None else Twist(recipe=algebra.twist.recipe, channels=()),
    )


def _project(fold: Fold, body, axes: tuple) -> Fold:
    """Attach a pure projection body to one Fold, dropping the empty wrapper."""
    body = Body.coerce(body)
    return _wrap(body, (fold,)) if body else fold


def _rebind(term: Fold, node: Fold, replacement: Fold) -> Fold:
    """``term`` over ``replacement`` wherever it read ``node`` — a split piece keeps the projection
    whole and swaps the fold it is about. The replacement exposes the node's own state names, so
    every positional binding above it is unchanged."""
    if term is node:
        return replacement
    operands = tuple(_rebind(edge, node, replacement) for edge in term.operands)
    return term if operands == term.operands else replace(term, operands=operands)


def output_root(root: Node, outputs: set[str]) -> Node:
    """A graph-node view containing only the output ports owned by one projection Fold."""
    by_name = dict(zip(root.buffer_names(), root.outputs, strict=True))
    ordered = tuple(name for name in root.buffer_names() if name in outputs)
    if outputs != set(ordered):
        raise ValueError(f"projection stores target unknown output buffers: {sorted(outputs - set(ordered))}")
    tensors = tuple(replace(by_name[name], name=name) for name in ordered)
    return replace(root, id=ordered[0], outputs=tensors)


def _split_projection(tile: TileOp, root: Node, selected: Fold):
    """The region the split node owns — ``(graph node, region term, tail stmts, stores)`` — and the
    other regions as pieces. A kernel with one root keeps its whole term as the region; an
    independent MIMO projection partitions by producing root (:func:`projection_regions`)."""
    op = tile.op
    if not isinstance(op, Fold) or op.axis is not None or len(_reducing_roots(op)) < 2:
        return root, op, (), tuple(tile.output_specs), ()  # one root, its providers beside it: the whole term is the region
    pieces = []
    chosen = None
    for fold, region, body, stores in projection_regions(op, tile.output_specs):
        node = output_root(root, {store.write.output for store in stores})
        entry = (node, region, body, stores)
        if fold is selected:
            chosen = entry
        else:
            pieces.append(entry)
    if chosen is None:
        raise ValueError("the split Fold does not own an independent projection region")
    return (*chosen, tuple(pieces))


def _projection_piece_name(root: Node) -> str:
    """Keep a cut producer's output token when a split detaches its other projection roots.

    The output buffer already names the cut that produced it. A split can separate several
    independent roots of that producer, so each fresh sibling needs its own last ``__place_``
    token for child-scoped PLACE pins. Ordinary outputs use their own stable buffer identity.
    """
    parent = root.op.name
    suffix = root.id.rpartition("__place_")[2]
    token, separator, ordinal = suffix.rpartition("_")
    if not (separator and len(token) == 10 and all(c in "0123456789abcdef" for c in token) and ordinal.isdigit()):
        token = digest(*root.buffer_names())[:10]
    return f"{parent}__place_{token}"


def _add_projection_pieces(match: Match, frag: Graph, pieces: tuple, free: tuple, shapes: dict) -> Graph:
    """Add the unsplit independent projection Folds as fresh schedulable kernels. Each is a piece
    of the REALIZED split — the kernel-set decision was consumed by the kernel it addressed, and
    one pinned split means one split — so it carries the consumed-split receipt
    (``split_consumed``): a ``REDUCE`` pin's ``g`` half strips on it instead of splitting the
    sibling region again (or raising)."""
    for root, region, body, stores in pieces:
        tile = replace(
            _piece(
                _project(region, body, tuple(free)),
                free,
                output_specs=stores,
                axes=root.op.axes,
                name=_projection_piece_name(root),
                shapes=shapes,
            ),
            split_consumed=True,
        )
        add_output_piece(match, frag, root, tile, _piece_inputs(root, tile))
    return frag


# ---- the realization -------------------------------------------------------------------------- #


def realize_split(match: Match, root: Node, cta: int, finalize: str) -> Graph:
    """Build the split fragment: the partial + deferred finalize pair, or the atomic arm's one
    kernel. Always a ``Graph``, never a ``TileOp`` — this rewrite's whole job is to change the
    kernel SET, and a 1:1 op rebind is how the engine says the OPPOSITE (same kernel, decided
    further — knobs merged forward, no pass-scan restart). The one-kernel atomic arm splices too,
    via :func:`_one`."""
    tile: TileOp = root.op
    # The fold NODE carries the algebra — every algebra read below (state names, identities, the
    # cross-partition combine) is off the node, never a loop annotation. The projection (when the
    # kernel carries one) rides the zero-axis ``Fold`` wrapper — its ONE home; peel it here, with
    # its output specifications reconstituted, and retarget its root ``Write`` below.
    node = head(tile.op)
    assert node is not None, "the split offer fires on node-form kernels only"
    root, region, body, stores, projection_pieces = _split_projection(tile, root, node)
    free = tuple(tile.place.free)
    k_axis = tile.axis_of(node.axis)
    if node.as_contraction() is not None:
        split, kslice, partial_fold = _sliced_contraction(node, k_axis, cta)
    else:
        _enforce(splitk_width(k_axis, cta))
        split = Axis(name=_SPLIT, extent=Dim(cta))
        kslice, partial_fold = _slice_fold(node, k_axis, k_axis.extent.as_static() // cta, split)
    axes = _with_axes(tile.axes, split, kslice)  # the pieces' axis table: the slice under its own name
    states = partial_fold.combine.results
    n_comp = len(states)
    out = root.output
    cell = _cell_index(stores, free)
    # The epilogue the atomic arm would apply per partition: the region's projection and its stores.
    projection = (*(region.step() if region.axis is None else ()), *body, *(store.write for store in stores))
    frag = _frag(match, root)
    shapes = io_shapes(tile)

    if finalize == "atomic":
        # Direct atomic finalize: ONE kernel — each CTA atomicAdds its slice's state into the
        # output (zero-init'd per launch), the GRID stage consumed into the grid. ``projection``
        # is the kernel's epilogue (``mean``'s ``×1/N``, …); a bare carrier has just the output
        # ``Write``. Whether the arm can carry it is :func:`atomic_finalize`'s one answer — the
        # same predicate the offer and the pin applied, re-asked here over the SELECTED region's
        # projection (the offer read the whole tail, a conservative superset — see its docstring).
        _enforce(atomic_finalize(partial_fold, projection, tile.outputs))
        if stores:
            p_stores = tuple(replace(store, write=replace(store.write, atomic=True)) for store in stores)
        else:
            p_stores = (OutputSpec(write=Write(output=out.name, index=cell, values=states, atomic=True)),)
        piece = _piece(
            _project(_rebind(region, node, partial_fold), body, (split, *free)),
            (split, *free),
            output_specs=p_stores,
            axes=axes,
            name=tile.name,
            shapes=shapes,
        )
        result = _one(match, frag, root, piece)
        return _add_projection_pieces(match, result, projection_pieces, free, shapes)

    # Deferred finalize: write every raw component to ``ws[(comp,) ksplit, *cell]``. The workspace
    # shape MUST match the rank of the index the writes/loads use — ``render_index`` refuses
    # any other. ``ws_cell``
    # is the FREE-axis vars (the partial has no original ``Write`` to copy), so size the workspace
    # by the free extents, not ``out.shape`` (whose extent-1 batch dims the grid never carries). A
    # multi-component carrier packs its per-component states into a leading ``comp`` axis; the
    # single-component workspace stays ``ws[ksplit, *cell]``. The workspace is **f32**: it holds
    # raw pre-projection accumulator states (summed + ⊗-combined by the finalize), and the
    # pre-projection state must not round-trip through the output dtype (an fp16 round-trip can
    # saturate outlier partials to ±inf before the combine and costs the mantissa of every
    # partition sum).
    def output_stride(axis):
        steps = (gmem_axis_step(Load("", store.write.output, store.write.index), axis.name, tile.outputs) for store in stores)
        return min((abs(step[0]) for step in steps if step is not None and step[0]), default=float("inf"))

    # Preserve the output's contiguous axes across the split. Free-axis order is a traversal
    # choice; using it as workspace layout can make the partial tile heads instead of channels.
    ws_free = tuple(sorted(free, key=output_stride, reverse=True))
    ws_name = f"{out.name}__partial"
    ws_shape = (Dim(n_comp), Dim(cta), *(a.extent for a in ws_free)) if n_comp > 1 else (Dim(cta), *(a.extent for a in ws_free))
    ws_cell = tuple(Var(ax.name) for ax in ws_free)
    shapes[ws_name] = ws_shape

    def ws_index(i: int) -> tuple:
        lead = (Literal(i, "int"), Var(split.name)) if n_comp > 1 else (Var(split.name),)
        return (*lead, *ws_cell)

    # --- partial kernel: reduce a CTA's slice, write its carrier state to the workspace. The
    # split axis joins as a lead grid axis via the partial tile's OWN placement — the view derives
    # lead axes from the placement, so nothing is restamped on the node.
    ws_stores = tuple(OutputSpec(write=Write(output=ws_name, index=ws_index(i), value=states[i])) for i in range(n_comp))
    # The partial and the finalize keep the name of the kernel they split, so a kernel pin naming a
    # piece (``place_<token>_1``) still names both halves of its split; an unnamed kernel (a route's
    # uncut root) keeps launching under its graph node's id, which a ``node_`` pin names.
    partial_tile = _piece(
        partial_fold, (split, *free), output_specs=ws_stores, axes=axes, name=tile.name and f"{tile.name}__partial", shapes=shapes
    )

    # --- finalize kernel: identity-lift each workspace state tuple through the SAME monoid.
    # The merge axis carries the SAME consumed-split receipt the partial's slice does: the
    # finalize enumerates the partitions of a split that already happened, so the receipt must
    # read as a kernel that already realized one. Without it an ambient ``REDUCE`` pin splits the
    # finalize too and its workspace collides with the partial's (``<out>__partial`` exists).
    fin_axis = replace(split, window=Window(parent=split, partition=True))
    other = tuple(f"{nm}__p" for nm in states)
    loads = tuple(Load(name=other[i], input=ws_name, index=ws_index(i)) for i in range(n_comp))
    fin_fold = _state_fold(fin_axis, partial_fold, loads)
    fin_stores = stores
    if not fin_stores:
        # A bare carrier's grid-cell store is materializer glue; the finalize spells it over the
        # value the region exposes, or the carrier's primary state.
        out_val = region.exposes[0] if region.exposes else states[0]
        fin_stores = (OutputSpec(write=Write(output=out.name, index=cell, value=out_val)),)
    # The finalize is stamped AFTER the workspace joins the fragment: it reads that buffer, and a
    # kernel's structural features fold in its operands' dtypes, which only resolve once the
    # buffer is a graph node.
    frag.add_node(op=partial_tile, inputs=list(root.inputs), output=Tensor(ws_name, ws_shape, F32), node_id=ws_name)
    fin_tile = _piece(
        _project(_rebind(region, node, fin_fold), body, tuple(free)),
        free,
        output_specs=fin_stores,
        axes=_with_axes(tile.axes, fin_axis),
        name=tile.name,
        shapes=shapes,
    )
    result = add_output_piece(match, frag, root, fin_tile, _piece_inputs(root, fin_tile, ws_name))
    return _add_projection_pieces(match, result, projection_pieces, free, shapes)


__all__ = ["atomic_finalize", "realize_carry_split", "realize_split", "split_forks", "splitk_width", "state_ports"]
