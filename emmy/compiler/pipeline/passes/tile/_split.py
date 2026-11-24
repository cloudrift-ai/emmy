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
exp-family monoids; the split does not recognize carrier families. The fold keeps its body and walks
the range of one block (``ir/tile/split.py``), so a cone's row-invariant prologue stays FULL-ROW in
every partition (the redundant-statistic split).

The atomic arm is the generic exception: it is legal only for a single additive state component
whose projection distributes over addition. Otherwise the deferred f32 workspace preserves the
full state until the finalize combines and projects it.

Every piece is a fresh unmapped :class:`TileOp`. A graph splice restarts the lowering pass scan:
scheduling offers each piece its own row. An axis :class:`Window` records that the partition has
already been consumed and prevents recursive splitting — the receipt is the IR itself, no flag.

A kernel that CARRIES A STATE has a split of its own, across the sequence (:func:`realize_carry_split`):
the parts of a step affine in the state compose as affine maps, so a probe reads each part's map
off two walks from known seeds, a prefix carries the state across the parts, and the walk itself
runs every part from the state it starts from. The maps compose as matrix products, so the split
can halve them pairwise before the prefix. Its pieces are Loop IR lifted like the walk.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import BF16, F16, F32
from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.address import clamp_to, gmem_axis_step
from emmy.compiler.ir.axis import Axis, Window
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expression import BinaryExpression, Literal, Var, apply
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.pure import Lambda
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.schedule import Reduce, Work
from emmy.compiler.ir.schedule.catalog import splitk_moves
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Assign, Body, Carry, Let, Load, Loop, Select, SelectBranch, Write
from emmy.compiler.ir.tile import OutputSpec, Placement, TileOp
from emmy.compiler.ir.tile.ops import Sched, head, projection_regions, projection_root, projection_tail
from emmy.compiler.ir.tile.split import merged, split_axis
from emmy.compiler.pipeline import Match
from emmy.compiler.pipeline.fork import DeferredFork
from emmy.compiler.pipeline.knob import axis_of, consume_kernel_row, kernel_pin
from emmy.compiler.pipeline.passes.tile._fromloop import carrying_loops, seed_index, stored_states
from emmy.compiler.pipeline.passes.tile._row import io_shapes, lift_nest, reformed
from emmy.compiler.pipeline.search.space import REDUCE, WORK

logger = logging.getLogger(__name__)

# ---- what a kernel / an axis can carry -------------------------------------------------------- #


def splitk_width(k_axis: Axis, width: int) -> str | None:
    """A cross-CTA split needs a STATIC reduce axis its width divides evenly — blocks of ``K/w``
    positions tile the stream exactly only when the extent is known and ``w`` divides it. Total over
    a symbolic axis rather than raising out of ``as_static``, so the catalog drops the width and a
    pin reports the reason."""
    if not k_axis.extent.is_static:
        return f"cross-CTA split of the symbolic reduce axis {k_axis.name!r} is not built"
    big_k = k_axis.extent.as_static()
    if big_k % width == 0:
        return None
    return f"split-K width {width} does not divide K={big_k}; pick a dividing split width."


def _stream_refusal(node: Fold, k_axis: Axis, width: int) -> str | None:
    """Why the head fold's stream cannot split ``width`` ways (``None`` when it can): :func:`splitk_width` over a static
    stream. A symbolic one splits into blocks of ``⌈K/w⌉`` positions, the last ending where the stream does, which
    is built for a monoid and a twisted carrier."""
    if k_axis.extent.is_static:
        return splitk_width(k_axis, width)
    if node.observe is not None:
        return f"cross-CTA split of the symbolic reduce axis {k_axis.name!r} is built for a monoid or a twisted carrier"
    return None


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
    in that same epilogue (``loop/lowering/090_spell_store_rounding``), which no linear-homogeneous
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
    STRIP from the projection — one reading the boundary store's sweep axis lands inside the sweep
    ``Loop`` ``apply_output_specs`` wraps — would leave the epilogue re-running the whole reduction
    and shadowing the workspace states, so the split declines there."""
    op = tile.op
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
    one slice, including under a contraction head."""

    def reduces_axis(edge) -> bool:
        return getattr(edge, "axis", None) == node.axis or any(map(reduces_axis, getattr(edge, "operands", ())))

    if any(map(reduces_axis, node.operands)):
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
        and not tile.carries
        and not tile.split_taken
    )


def split_forks(match: Match | None, root: Node, *, unsplit_tile: TileOp | None = None) -> list[DeferredFork] | None:
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
    a pin with no ``g`` half decides UNSPLIT, exactly as a spelled row with no ``g`` half does.
    Enumeration-only callers may omit ``match``; materializing an arm requires it."""
    tile: TileOp = root.op
    if tile.carries:
        return _carry_split_forks(match, root, tile)
    node = head(tile.op)
    if not split_pending(tile):
        return None
    assert node is not None and node.axis is not None
    k_axis = tile.axis_of(node.axis)  # the node names its K; the kernel's axis table holds its extent
    key = Sched(tile).key("REDUCE", node) or "REDUCE"
    unsplit = DeferredFork(lambda: replace(unsplit_tile or tile, split_consumed=True), {key: ""})
    element = axis_of(key)
    # A kernel pin names the kernel by its name; the schedule pass reads them the same way (``040_schedule.pin_row``).
    pin = kernel_pin("REDUCE", tile.name)
    if pin is None:
        pin = REDUCE.narrow_at(element) if element else REDUCE.raw()
    tail = projection_tail(tile)
    if pin is not None:
        work = kernel_pin("WORK", tile.name)
        plan = Reduce.parse(pin, Work.parse(work if work is not None else WORK.raw()))
        if not plan.needs_split:
            return [unsplit]
        if plan.pairs:
            raise ValueError(f"REDUCE={pin}: only the parts of a carried state compose pairwise; a reduction's split is g<n>k")
        _enforce(_stream_refusal(node, k_axis, plan.cta))
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
        why = _stream_refusal(node, k_axis, plan.cta) or (atomic_why if plan.finalize == "atomic" else None)
        if why is not None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("split g%d%s not offered: %s", plan.cta, plan.finalize[0], why)
            continue
        options.append(_split_fork(match, root, key, plan.cta, plan.finalize))
    return options


def _split_fork(match: Match | None, root: Node, key: str, cta: int, finalize: str) -> DeferredFork:
    spelling = Reduce.of(cta=cta, finalize=finalize).spell()
    # An enumeration-only arm (no match) is never materialized, so it carries no redirect either.
    output = redirects(root) if match is not None else None
    return DeferredFork(lambda: realize_split(match, root, cta, finalize), {key: spelling}, structural=True, output=output)


# ---- a carried state: the split across the sequence ------------------------------------------- #


def _carrying_body(body: Body) -> Body | None:
    """The statements the carrying loop sits among, each size-one free loop outside it opened at its one
    coordinate; ``None`` under a free loop that holds more than one, whose nest the realizer does not rebuild."""
    while not any(isinstance(stmt, Loop) and stmt.carries for stmt in body):
        loops = [stmt for stmt in body if isinstance(stmt, Loop)]
        if len(loops) != 1 or loops[0].axis.extent != 1:
            return None
        (outer,) = loops
        here = Sigma({outer.axis.name: Literal(0, "int")})
        body = Body((*(stmt for stmt in body if stmt is not outer), *(stmt.substitute(here) for stmt in outer.body)))
    return body


def _step_maps(step: Body, state: str, mixed: int | None, column, stores) -> tuple[Body, Body] | None:
    """One step's affine map read off the step itself — ``(matrix, offset)`` bodies, the step with its carrier
    reads resolved and its ``Carry`` handed to ``stores`` — or ``None`` when a read has a shape this does not take.

    The matrix is the step from the identity less the offset, and the offset the step from zero. A contraction
    over the previous state (a reduction whose axis is the read's mixed coordinate, its term linear in the read)
    never runs: from the identity only the term at ``column`` is left, so the loop is that one term with the read
    at one, and from zero it is nothing. Any other read is the identity's own entry, one where its mixed
    coordinate is ``column``. No walk, and no contraction against a seed."""

    def reads_state(stmt) -> bool:
        return isinstance(stmt, Load) and stmt.input == state

    def contracts(loop: Loop) -> bool:
        reads = [stmt for stmt in loop.body if reads_state(stmt)]
        if not loop.is_reduce or not reads or mixed is None or any(read.index[1 + mixed] != Var(loop.axis.name) for read in reads):
            return False
        linear = {read.name for read in reads}
        for stmt in loop.body:
            if isinstance(stmt, Assign):
                kind = _homogeneity(stmt.value, linear)
                if kind is None:
                    return False
                if kind == "linear":
                    linear.add(stmt.name)
            elif isinstance(stmt, Accum):
                if (
                    stmt.op.name != "add"
                    or stmt.init != Literal(0.0, "float")
                    or not isinstance(stmt.value, Var)
                    or stmt.value.name not in linear
                ):
                    return False
            elif not isinstance(stmt, Load | Let):
                return False
        return True

    def resolved(body, identity: bool) -> tuple | None:
        out: list = []
        for stmt in body:
            if isinstance(stmt, Loop) and contracts(stmt):
                if not identity:
                    out += [Let(name=inner.name, value=0.0) for inner in stmt.body if isinstance(inner, Accum)]
                    continue
                at = Sigma({stmt.axis.name: column})
                for inner in stmt.body:
                    if reads_state(inner):
                        out.append(Let(name=inner.name, value=1.0))
                    elif isinstance(inner, Accum):
                        out.append(Assign(inner.name, inner.value))
                    else:
                        out.append(inner.substitute(at))
            elif isinstance(stmt, Loop):
                if (inner := resolved(stmt.body, identity)) is None:
                    return None
                out.append(stmt.replace(body=Body(inner)))
            elif reads_state(stmt):
                if not identity or mixed is None:
                    out.append(Let(name=stmt.name, value=1.0 if identity else 0.0))
                else:
                    hit = BinaryExpression("==", stmt.index[1 + mixed], column)
                    out += [Let(name=f"{stmt.name}_one", value=1.0), Let(name=f"{stmt.name}_zero", value=0.0)]
                    branches = (SelectBranch(f"{stmt.name}_one", hit), SelectBranch(f"{stmt.name}_zero", Literal(True, "bool")))
                    out.append(Select(name=stmt.name, branches=branches))
            elif isinstance(stmt, Carry):
                out += stores(stmt, identity)
            elif not isinstance(stmt, Write):
                out.append(stmt)
        return tuple(out)

    matrix, offset = resolved(step, True), resolved(step, False)
    return None if matrix is None or offset is None else (Body(matrix).pruned(()), Body(offset).pruned(()))


def _carry_stores(loop: Loop) -> str | None:
    """What the parts of a split walk may store: ``"steps"`` when every store is indexed by the step, so each
    part writes cells of its own; ``"last"`` when the one store is the state itself, each step over the last
    one's, which only the final state reaches; ``None`` for a mix, which parts running side by side cannot write."""
    stmts = tuple(loop.body.iter())
    writes = tuple(stmt for stmt in stmts if isinstance(stmt, Write))
    stepped = [any(loop.axis.name in expression.free_vars() for expression in write.index) for write in writes]
    if writes and all(stepped):
        return "steps"
    if len(writes) != 1 or not writes[0].is_scalar:
        return None
    carried = {stmt.value for stmt in stmts if isinstance(stmt, Carry)}
    copies = {stmt.name: stmt.value.name for stmt in stmts if isinstance(stmt, Assign) and isinstance(stmt.value, Var)}
    value = writes[0].value
    while value not in carried and value in copies:
        value = copies[value]
    return "last" if value in carried else None


def _affine_state(tile: TileOp):
    """The one state ``tile`` carries, as ``(its update, the affine reading of the step in it)``, or ``None``: several
    states, or a step that is not affine in its state (:meth:`Fold.affine`)."""
    if len(tile.states) != 1:
        return None
    carry = tile.states[0].write
    view = tile.op.affine(carry.name, carry.cells, carry.value)
    return None if view is None else (carry, view)


def _carry_split_forks(match: Match, root: Node, tile: TileOp) -> list[DeferredFork] | None:
    """The split fork for a kernel that CARRIES A STATE — the unsplit walk beside one structural
    option per width the step axis divides into — or ``None`` when there is nothing to decide: a
    step that is not affine in its state (:meth:`Fold.affine`), a block the probe cannot pack
    (fewer kept than mixed cells), a walk under a free loop of more than one coordinate outside the
    carrying one (:func:`_carrying_body`), stores the parts cannot write side by side
    (:func:`_carry_stores`), a piece of a realized split. Only the
    deferred finalize exists here (``g<n>k``): the parts' maps compose in a kernel, there is nothing
    to add atomically. The pairwise compositions of the maps (``g<n>k<p>``) are offered at one step a part,
    where the maps cost no probe; a pin may name them at any width."""
    found = _affine_state(tile)
    if found is None or tile.split_taken or (body := _carrying_body(tile.loop_body)) is None:
        return None
    carry, view = found
    (time,) = tile.place.serial
    if view.mixed is not None:
        mixed, kept = (tile.axis_of(carry.cells[position]).extent for position in (view.mixed, view.kept))
        if not (mixed.is_static and kept.is_static) or kept.as_static() < mixed.as_static():
            return None
    (loop,) = (stmt for stmt in body if isinstance(stmt, Loop) and stmt.carries)
    if (stores := _carry_stores(loop)) is None:
        return None
    written = {spec.write.output for spec in tile.output_specs if spec not in tile.states}
    # The split is the state's: bare where the step has no reduction of its own to split, else at the state's route.
    key = f"REDUCE@{tile.state_routes[0]}" if tile.family_sites["REDUCE"] else "REDUCE"
    unsplit = DeferredFork(lambda: replace(tile, split_consumed=True), {key: ""})
    element = axis_of(key)
    pin = kernel_pin("REDUCE", tile.name)
    if pin is None:
        pin = REDUCE.narrow_at(element) if element else REDUCE.raw()

    def arm(cta: int, pairs: int = 0) -> DeferredFork:
        spelling = Reduce.of(cta=cta, pairs=pairs).spell()
        # The piece that stores the kernel's outputs may carry nothing, and the state port goes with the state: the
        # one map a last-state walk composes to, applied to the seed; a walk of one step a part, read from its start.
        whole = time.extent.as_static() == cta
        ends = pairs == (cta - 1).bit_length() if stores == "last" else whole
        kept = output_root(root, written) if ends else root
        output = redirects(kept) if match is not None else None  # an enumeration-only arm carries none
        return DeferredFork(lambda: realize_carry_split(match, root, cta, pairs), {key: spelling}, structural=True, output=output)

    if pin is not None:
        work = kernel_pin("WORK", tile.name)
        plan = Reduce.parse(pin, Work.parse(work if work is not None else WORK.raw()))
        if not plan.needs_split:
            return [unsplit]
        _enforce(splitk_width(time, plan.cta))
        if plan.finalize != "kernel":
            raise ValueError("a carried state's split composes its parts in a finalize kernel: spell it REDUCE=g<n>k")
        if plan.pairs > (plan.cta - 1).bit_length():
            raise ValueError(f"REDUCE={pin}: {plan.cta} parts are one map after {(plan.cta - 1).bit_length()} compositions")
        return [arm(plan.cta, plan.pairs)]
    # The catalog's widths, then one step a part composed to every depth: at that width the maps are read off the
    # step, and how far composing them pays against walking what is left is the pieces' to price.
    steps = time.extent.as_static() if time.extent.is_static else 1
    widths = [
        plan.cta for plan in splitk_moves() if plan.finalize == "kernel" and plan.cta != steps and splitk_width(time, plan.cta) is None
    ]
    options = [unsplit, *(arm(width) for width in widths)]
    if steps > 1:
        options += [arm(steps, pairs) for pairs in range((steps - 1).bit_length() + 1)]
    return options


def state_ports(tile: TileOp, prefix: str) -> tuple[Tensor, ...]:
    """The buffers the classic realization of ``tile``'s carried states owns, named under
    ``prefix`` — one per state over the step and the state's cell, the index ``serial_form`` writes. The lift
    adds them to the node; register storage drops them again. A state the term stores that way itself
    (``stored_states``) has that output for its buffer and no port."""
    if not tile.states:
        return ()
    stored = {state for loop, above in carrying_loops(tile.loop_body) for state in stored_states(loop, above)}
    (time,) = tile.place.serial
    return tuple(
        Tensor(
            name=f"{prefix}__{carry.name}",
            shape=(time.extent, *(tile.axis_of(entry.name).extent if isinstance(entry, Var) else Dim(1) for entry in carry.cell)),
            dtype=F32,
        )
        for carry in (spec.write for spec in tile.states)
        if carry.name not in stored
    )


def _nest(body: Body, axes: tuple) -> Body:
    """``body`` under loops over ``axes``, outermost first."""
    for axis in reversed(axes):
        body = Body((Loop(axis=axis, body=body),))
    return body


def realize_carry_split(match: Match, root: Node, cta: int, pairs: int = 0) -> Graph:
    """Split a carried state's walk across the sequence into ``cta`` parts — kernels lifted from
    Loop IR like the walk itself:

    1. the PROBE walks every part's range twice from known seeds, zero and the identity packed
       along the kept cells, and stores every step's state: a step affine in the state composes as
       an affine map, so a part's map is read off the two walks — its offset is the zero walk's last
       state, its matrix the difference of the two walks' last states;
    2. ``pairs`` times, a PAIR of kernels halves the maps: ``(A, b)`` of parts ``2t`` and ``2t + 1``
       compose as ``(A₁·A₀, A₁·b₀ + b₁)``, one matrix product each, and an odd map out passes through;
    3. the PREFIX carries the state across the maps that are left, applying each to the state the
       one before it left, and stores the state each starts from;
    4. a FILL-IN per composition gives every map of the level below its start: the even ones the
       start of their pair, the odd ones that start through the even map;
    5. the WALK is the kernel itself over each part's range from its start, one part per batch
       cell, storing what the kernel stored.

    A kernel whose one store is the state itself, each step over the last one's, ends at the prefix:
    the final state is the prefix's last, which stores it, and nothing reads a part's start.

    Three walks of a part's range where the sequence took one, for ``cta`` times the parallelism:
    the price a card mostly idle on one CTA per head pays gladly, and evidence's to decide."""
    tile: TileOp = root.op
    found = _affine_state(tile)
    assert found is not None, "the split offer fires on affine carried states only"
    carry, view = found
    state, cells = carry.name, carry.cells
    cell_axes = tuple(tile.axis_of(name) for name in cells)
    own = tuple(Var(name) for name in cells)
    (time,) = tile.place.serial
    steps = time.extent.as_static() // cta
    out = root.output
    body = _carrying_body(tile.loop_body)
    (position,) = (index for index, stmt in enumerate(body) if isinstance(stmt, Loop) and stmt.carries)
    loop, before, after = body[position], tuple(body[:position]), tuple(body[position + 1 :])
    stores = _carry_stores(loop)
    part, probe = Axis("_part", Dim(cta)), Axis("_probe", Dim(2))
    here = Var(part.name)
    sliced = replace(loop.axis, extent=Dim(steps), window=Window(parent=loop.axis.source_axis or loop.axis, partition=True))
    sigma = Sigma({loop.axis.name: BinaryExpression("+", BinaryExpression("*", here, Literal(steps, "int")), Var(loop.axis.name))})
    step = Body(tuple(stmt.substitute(sigma) for stmt in loop.body))
    probe_seed, probe_states = f"{out.name}__probe_seed", f"{out.name}__probe"
    # Level 0 holds the parts' maps; each level above, half the maps of the one below.
    counts = [cta]
    for _ in range(pairs):
        counts.append((counts[-1] + 1) // 2)
    maps = [f"{out.name}__map{level}" for level in range(pairs + 1)]
    offsets = [f"{out.name}__offset{level}" for level in range(pairs + 1)]
    starts = [f"{out.name}__start{level or ''}" for level in range(pairs + 1)]

    def indexed(stmts: Body, lead: tuple[Axis, ...], seed: str) -> Body:
        # The state under the lead coordinates too — a part, a probe — and seeded from ``seed``, its step the trip of
        # the sliced loop.
        prefix, trip = tuple(Var(axis.name) for axis in lead), Var(loop.axis.name)

        def rewrite(stmt):
            if isinstance(stmt, Carry):
                return stmt.replace(index=(trip, *prefix, *stmt.cell), seed=seed)
            if isinstance(stmt, Load) and stmt.input == state:
                return stmt.replace(index=(trip - Literal(1, "int"), *prefix, *stmt.index[1:]))
            return stmt

        return stmts.map(rewrite)

    # The probe: the walk with its stores replaced by one store of the state every step.
    def probed(stmt):
        if isinstance(stmt, Write):
            return None
        if isinstance(stmt, Loop) and not stmt.is_reduce and not stmt.body.carries and not stmt.body.iter_of_type(Write):
            return None  # an output sweep with nothing left to store
        if isinstance(stmt, Carry):
            return (stmt, Write(output=probe_states, index=(Var(probe.name), here, Var(sliced.name), *own), values=(stmt.value,)))
        return stmt

    # What only a dropped store read goes with it: a second value the step stored beside its state.
    walk = indexed(step, (probe, part), probe_seed).map(probed).pruned(())
    probe_body = Body((*before, *_nest(walk, (sliced, probe, part)), *after))
    # The probe's seeds: zero, and the identity between the mixed and the kept cell — ones
    # everywhere for a step reading its own cell alone, whose map is diagonal.
    hit = BinaryExpression(">", Var(probe.name), Literal(0, "int"))
    if view.mixed is not None:
        hit = BinaryExpression("&&", hit, BinaryExpression("==", own[view.mixed], own[view.kept]))
    seed_body = Body(
        (
            Let(name="_one", value=1.0),
            Let(name="_zero", value=0.0),
            Select(name="_seed", branches=(SelectBranch("_one", hit), SelectBranch("_zero", Literal(True, "bool")))),
            Write(output=probe_seed, index=(Var(probe.name), here, *own), values=("_seed",)),
        )
    )
    seed_kernel = _nest(seed_body, (probe, part, *cell_axes))

    # A map ``S ← A·S + b``: ``A[i, k]`` at the cell whose kept coordinate is ``k``, the identity walk less the
    # zero walk at level 0; ``b`` the zero walk. A diagonal map applies cell by cell, with no contraction.
    last = Literal(steps - 1, "int")
    basis = None if view.mixed is None else Axis("_basis", cell_axes[view.mixed].extent)
    # A matrix's cells: the state's, its kept coordinate running over the columns the probe packed.
    map_axes = tuple(replace(axis, extent=basis.extent) if index == view.kept else axis for index, axis in enumerate(cell_axes))

    def at(position: int | None, expression) -> tuple:
        return tuple(expression if index == position else coordinate for index, coordinate in enumerate(own))

    # One step a part: the step's own matrix and offset, read off it, in place of the probe.
    # A step that is zero from zero has no offset at any level: its maps are their matrices.
    direct, linear = None, False
    if steps == 1:

        def stored(carry: Carry, identity: bool) -> tuple:
            if not identity:
                return (Write(output=offsets[0], index=(here, *own), values=(carry.value,)),)
            if linear:
                return (Write(output=maps[0], index=(here, *own), values=(carry.value,)),)
            return (
                Load(names=("_b",), input=offsets[0], index=(here, *own)),
                Assign("_a", apply("subtract", carry.value, "_b")),
                Write(output=maps[0], index=(here, *own), values=("_a",)),
            )

        once = Body(tuple(stmt.substitute(Sigma({loop.axis.name: here})) for stmt in loop.body))
        column = None if view.kept is None else own[view.kept]
        direct = _step_maps(once, state, view.mixed, column, stored)
        if direct is not None:
            # Normalization folds a product with zero, so an offset that is nothing stores a literal zero.
            formed = tuple(LoopOp(body=Body((*before, *_nest(direct[1], (part,)), *after))).body.iter())
            zeros = {stmt.name for stmt in formed if isinstance(stmt, Let) and stmt.value == Literal(0.0, "float")}
            linear = all(stmt.value in zeros for stmt in formed if isinstance(stmt, Write))
        if linear:
            direct = _step_maps(once, state, view.mixed, column, stored)
    if direct is not None and basis is not None:
        # The matrix has a column per mixed cell: its kept loop runs over those alone.
        columns = cells[view.kept]
        narrowed = direct[0].map(
            lambda stmt: stmt.replace(axis=map_axes[view.kept]) if isinstance(stmt, Loop) and stmt.axis.name == columns else stmt
        )
        direct = (narrowed, direct[1])

    def matrix(level: int, where, index: tuple, name: str) -> tuple:
        if level or direct is not None:
            return (Load(names=(name,), input=maps[level], index=(where, *index)),)
        return (
            Load(names=(f"{name}_one",), input=probe_states, index=(Literal(1, "int"), where, last, *index)),
            Load(names=(f"{name}_zero",), input=probe_states, index=(Literal(0, "int"), where, last, *index)),
            Assign(name, apply("subtract", f"{name}_one", f"{name}_zero")),
        )

    def offset(level: int, where, index: tuple, name: str) -> tuple:
        if linear:
            return (Let(name=name, value=0.0),)
        if level or direct is not None:
            return (Load(names=(name,), input=offsets[level], index=(where, *index)),)
        return (Load(names=(name,), input=probe_states, index=(Literal(0, "int"), where, last, *index)),)

    def applied(level: int, where, operand) -> tuple[tuple, str]:
        # ``A`` of map ``where`` times ``operand``, a value read at the cell whose mixed coordinate is the column.
        def term(along) -> tuple:
            return (
                *matrix(level, where, at(view.kept, along), "_a"),
                *operand(at(view.mixed, along), "_s"),
                Assign("_as", apply("multiply", "_a", "_s")),
            )

        if basis is None:
            return term(None), "_as"
        return (Loop(axis=basis, body=Body((*term(Var(basis.name)), Accum(name="_acc", value="_as")))),), "_acc"

    extents = tuple(axis.extent for axis in cell_axes)
    shapes = io_shapes(tile) | {
        probe_seed: (Dim(2), Dim(cta), *extents),
        probe_states: (Dim(2), Dim(cta), Dim(steps), *extents),
        **{maps[level]: (Dim(counts[level]), *(axis.extent for axis in map_axes)) for level in range(pairs + 1)},
        **{offsets[level]: (Dim(counts[level]), *extents) for level in range(pairs + 1)},
        **{name: (Dim(count), *extents) for name, count in zip(starts, counts, strict=True)},
    }

    def piece(body: Body) -> TileOp:
        lifted = lift_nest(body, shapes)
        return replace(lifted, knobs=consume_kernel_row(lifted.knobs))

    def add(name: str, body: Body, inputs: list[str], *, carried: bool = False) -> None:
        made = replace(piece(body), split_consumed=True) if carried else piece(body)
        ports = state_ports(made, name) if made.carries else ()
        frag.add_node(op=made, inputs=inputs, outputs=(Tensor(name, shapes[name], F32), *ports), node_id=name)

    frag = _frag(match, root)
    if direct is not None:
        for name, body, first in ((offsets[0], direct[1], ()), (maps[0], direct[0], (offsets[0],)))[linear:]:
            first = () if linear else first
            whole = Body((*before, *_nest(body, (part,)), *after))
            add(name, whole, _piece_inputs(root, whole, *first))
    else:
        add(probe_seed, seed_kernel, [])
        probe_tile = piece(probe_body)
        frag.add_node(
            op=probe_tile,
            inputs=_piece_inputs(root, probe_tile, probe_seed),
            outputs=(Tensor(probe_states, shapes[probe_states], F32), *state_ports(probe_tile, probe_states)),
            node_id=probe_states,
        )

    # The pairs: level by level, the maps ``2t`` and ``2t + 1`` of the level below composed as one.
    for level in range(1, pairs + 1):
        below, pair = level - 1, Axis(part.name, Dim(counts[level]))
        low = BinaryExpression("*", here, Literal(2, "int"))
        high = BinaryExpression("+", low, Literal(1, "int"))
        odd = counts[below] % 2 == 1
        whole = BinaryExpression("<", high, Literal(counts[below], "int"))
        if odd:
            high = clamp_to(high, Literal(counts[below], "int"))
        for target, axes, lower in ((maps, map_axes, matrix), (offsets, cell_axes, offset))[: 2 - linear]:
            product, value = applied(below, high, lambda index, name, lower=lower, below=below, low=low: lower(below, low, index, name))
            stmts = [*product]
            if target is offsets:
                stmts += [*offset(below, high, own, "_b"), Assign("_sum", apply("add", value, "_b"))]
                value = "_sum"
            if odd:
                keep = Select(name="_next", branches=(SelectBranch(value, whole), SelectBranch("_keep", Literal(True, "bool"))))
                stmts += [*lower(below, low, own, "_keep"), keep]
                value = "_next"
            stmts.append(Write(output=target[level], index=(here, *own), values=(value,)))
            composed = [maps[below]] if target is maps else [maps[below], offsets[below]]
            reads = [probe_states] if below == 0 and direct is None else composed
            add(target[level], _nest(Body(tuple(stmts)), (pair, *axes)), reads)

    # The prefix: the state carried across the maps of the top level, each one's start the state before it.
    # One map left applies to the seed, and nothing is carried.
    carried, seed, single = f"{state}__across", carry.seed, counts[pairs] == 1
    where = Literal(0, "int") if single else here

    def prior(index: tuple, name: str) -> tuple:
        if not single:
            return (Load(names=(name,), input=carried, index=(here - Literal(1, "int"), *index)),)
        if isinstance(seed, str):
            return (Load(names=(name,), input=seed, index=seed_index(index, shapes[seed])),)
        return (Let(name=name, value=seed),)

    product, value = applied(pairs, where, prior)
    if stores == "steps":
        kept = Write(output=starts[pairs], index=(where, *own), values=("_from",))
    else:
        (kept,) = (stmt.replace(values=("_next",)) for stmt in loop.body.iter() if isinstance(stmt, Write))
    prefix_body = _nest(
        Body(
            (
                *product,
                *offset(pairs, where, own, "_b"),
                *prior(own, "_from"),
                Assign("_next", apply("add", value, "_b")),
                *(() if single else (Carry(name=carried, value="_next", index=(here, *own), seed=seed),)),
                kept,
            )
        ),
        (*(() if single else (Axis(part.name, Dim(counts[pairs])),)), *cell_axes),
    )
    seeded = [seed] if isinstance(seed, str) else []
    read = [probe_states] if pairs == 0 and direct is None else [maps[pairs], *offsets[pairs : pairs + 1 - linear]]
    written = output_root(root, {spec.write.output for spec in tile.output_specs if spec not in tile.states})

    def owned(made: TileOp) -> tuple:
        # The piece that stores what the kernel stored owns the kernel's outputs, not the state port
        # the lift gave the kernel: its own port replaces that one, travelling under a temporary
        # until the splice hands it the name, the way every replaced buffer does.
        return tuple(
            replace(port, name=f"{port.name}__split") if port.name in root.buffer_names() else port for port in state_ports(made, root.id)
        )

    if stores == "last":
        prefix_tile = replace(piece(prefix_body), split_consumed=True) if not single else piece(prefix_body)
        ports = owned(prefix_tile)
        result = add_output_piece(frag, written, prefix_tile, [*read, *seeded], states=ports)
        result.outputs.extend(port.name for port in ports if port.name.endswith("__split"))
        return result
    add(starts[pairs], prefix_body, [*read, *seeded], carried=not single)

    # The fill-in: every map of the level below starts where its pair starts, the odd one through the even map.
    for level in range(pairs, 0, -1):
        below, half = level - 1, BinaryExpression("/", here, Literal(2, "int"))
        parity = BinaryExpression("%", here, Literal(2, "int"))
        even = BinaryExpression("-", here, parity)
        product, value = applied(
            below, even, lambda index, name, level=level, half=half: (Load(names=(name,), input=starts[level], index=(half, *index)),)
        )
        fill = Body(
            (
                *product,
                *offset(below, even, own, "_b"),
                Assign("_through", apply("add", value, "_b")),
                Load(names=("_same",), input=starts[level], index=(half, *own)),
                Select(
                    name="_from",
                    branches=(
                        SelectBranch("_through", BinaryExpression("==", parity, Literal(1, "int"))),
                        SelectBranch("_same", Literal(True, "bool")),
                    ),
                ),
                Write(output=starts[below], index=(here, *own), values=("_from",)),
            )
        )
        composed = [maps[below], *offsets[below : below + 1 - linear]]
        reads = [starts[level], *([probe_states] if below == 0 and direct is None else composed)]
        add(starts[below], _nest(fill, (Axis(part.name, Dim(counts[below])), *cell_axes)), reads)

    # The walk: the kernel over each part's range, from the state its start holds. One step a part carries
    # nothing: the step reads its start where it read the state, and is a kernel like any other.
    if steps == 1:

        def started(stmt):
            if isinstance(stmt, Load) and stmt.input == state:
                return Load(names=(stmt.name,), input=starts[0], index=(here, *stmt.index[1:]))
            return None if isinstance(stmt, Carry) else stmt

        once = Body(tuple(stmt.substitute(Sigma({loop.axis.name: here})) for stmt in loop.body))
        walk_tile = piece(Body((*before, *_nest(once.map(started), (part,)), *after)))
    else:
        walk_tile = piece(Body((*before, *_nest(indexed(step, (part,), starts[0]), (sliced, part)), *after)))
    ports = owned(walk_tile)
    result = add_output_piece(frag, written, walk_tile, _piece_inputs(root, walk_tile, starts[0]), states=ports)
    result.outputs.extend(port.name for port in ports if port.name.endswith("__split"))
    return result


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


def redirects(root: Node, suffix: str = "__split") -> dict[str, str]:
    """Each buffer of ``root`` mapped to the temporary it travels under in a fragment that replaces the
    kernel — the redirect map a structural arm carries (:attr:`DeferredFork.output`), and the names
    :func:`add_output_piece` gives the pieces' outputs."""
    return {name: f"{name}{suffix}" for name in root.buffer_names()}


def add_output_piece(frag: Graph, root: Node, piece: TileOp, inputs: list[str], *, suffix: str = "__split", states: tuple = ()) -> Graph:
    """Add a fresh piece with its owned output ports under the fragment's temporary names.

    ``root`` is the graph-node view of the ports this piece owns — :func:`output_root` narrows a
    MIMO node to a subset — so slot 0's edge key travels as the node id and every other as its own
    tensor name. ``suffix`` names the temporary those buffers travel under while the fragment is
    spliced in: the cross-CTA split mints ``__split``, the kernel-placement cut ``__placed``. ONE
    suffix per rewrite, so a fragment's temporary names say which decision minted them, and the arm
    that offers the fragment carries the same map for the whole kernel (:func:`redirects`). ``states``
    are buffers the piece writes that the replaced node never had — a recurrence's state."""
    buffers = root.buffer_names()
    renamed = redirects(root, suffix)
    piece = replace(
        piece,
        output_specs=tuple(replace(spec, write=spec.write.rename_buffers(renamed)) for spec in piece.output_specs),
    )
    tensors = (
        replace(root.outputs[0], name=buffers[0]),
        *(replace(tensor, name=renamed[name]) for name, tensor in zip(buffers[1:], root.outputs[1:], strict=True)),
    )
    frag.add_node(op=piece, inputs=inputs, outputs=(*tensors, *states), node_id=renamed[buffers[0]])
    frag.outputs.extend(renamed.values())
    return frag


def _one(frag: Graph, root: Node, piece: TileOp) -> Graph:
    """The ATOMIC arm's one-kernel fragment. It replaces the split kernel with ONE kernel, but it
    is a SPLICE, never an op rebind: a rebind is how the engine says "the same kernel, decided
    further", so it merges the replaced op's knobs forward and does not restart the pass scan. The
    atomic partial is a different kernel — its own placement, its own body — and it has to reach
    scheduling carrying nothing of the kernel it replaced."""
    return add_output_piece(frag, root, piece, list(root.inputs))


def _wrap(body: Body, operands: tuple) -> Fold:
    """A zero-axis term over ``body``, exposing its last definition — what a projection returns."""
    bound = tuple(name for edge in operands for name in edge.exposes)
    results = next((stmt.defines()[-1:] for stmt in reversed(tuple(body)) if stmt.defines()), ())
    lift = Lambda.closing(bound, body, results)
    return Fold(operands=operands, lift=lift)


def _with_axes(axes: tuple, *new: Axis) -> tuple:
    """The axis table with ``new`` entries replacing same-named ones — a slice keeps its axis's name."""
    return tuple({**{axis.name: axis for axis in axes}, **{axis.name: axis for axis in new}}.values())


def _piece(op: Fold, free, *, output_specs: tuple = (), axes: tuple, shapes: dict) -> TileOp:
    """One fresh unscheduled Tile kernel over ``op`` and the axis table ``axes``, formed as its own kernel
    (:func:`~._row.reformed`) over the buffers ``shapes`` names: its nest lowered and lifted again, the loop op it
    came from kept as its source."""
    piece = reformed(TileOp(op=op, place=Placement(free=tuple(free)), output_specs=output_specs, axes=axes), shapes)
    # A split CONSUMES the kernel it replaces: the piece drops its schedule row and its structural
    # identity. Built fresh here, so this states the contract rather than doing work — and the rule
    # that mints a kernel is where that has to be said.
    return replace(piece, knobs=consume_kernel_row(piece.knobs))


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
    return term if operands == term.operands else term.replace(operands=operands)


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
                shapes=shapes,
            ),
            split_consumed=True,
        )
        add_output_piece(frag, root, tile, _piece_inputs(root, tile))
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
    if node.as_contraction() is None:
        _enforce(_stream_refusal(node, k_axis, cta))
    split, kslice, partial_fold = split_axis(node, k_axis, cta)
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
            p_stores = tuple(replace(store, write=store.write.replace(atomic=True)) for store in stores)
        else:
            p_stores = (OutputSpec(write=Write(output=out.name, index=cell, values=states, atomic=True)),)
        piece = _piece(
            _project(_rebind(region, node, partial_fold), body, (split, *free)),
            (split, *free),
            output_specs=p_stores,
            axes=axes,
            shapes=shapes,
        )
        result = _one(frag, root, piece)
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
        steps = (gmem_axis_step(Load(("",), store.write.output, store.write.index), axis.name, tile.outputs) for store in stores)
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
    ws_stores = tuple(OutputSpec(write=Write(output=ws_name, index=ws_index(i), values=(states[i],))) for i in range(n_comp))
    partial_tile = _piece(partial_fold, (split, *free), output_specs=ws_stores, axes=axes, shapes=shapes)

    # --- finalize kernel: identity-lift each workspace state tuple through the SAME monoid.
    # The merge axis carries the SAME consumed-split receipt the partial's slice does: the
    # finalize enumerates the partitions of a split that already happened, so the receipt must
    # read as a kernel that already realized one. Without it an ambient ``REDUCE`` pin splits the
    # finalize too and its workspace collides with the partial's (``<out>__partial`` exists).
    fin_axis = replace(split, window=Window(parent=split, partition=True))
    other = tuple(f"{nm}__p" for nm in states)
    loads = tuple(Load(names=(other[i],), input=ws_name, index=ws_index(i)) for i in range(n_comp))
    fin_fold = merged(fin_axis, partial_fold, loads)
    fin_stores = stores
    if not fin_stores:
        # A bare carrier's grid-cell store is materializer glue; the finalize spells it over the
        # value the region exposes, or the carrier's primary state.
        out_val = region.exposes[0] if region.exposes else states[0]
        fin_stores = (OutputSpec(write=Write(output=out.name, index=cell, values=(out_val,))),)
    # The finalize is stamped AFTER the workspace joins the fragment: it reads that buffer, and a
    # kernel's structural features fold in its operands' dtypes, which only resolve once the
    # buffer is a graph node.
    frag.add_node(op=partial_tile, inputs=list(root.inputs), output=Tensor(ws_name, ws_shape, F32), node_id=ws_name)
    fin_tile = _piece(
        _project(_rebind(region, node, fin_fold), body, tuple(free)),
        free,
        output_specs=fin_stores,
        axes=_with_axes(tile.axes, fin_axis),
        shapes=shapes,
    )
    result = add_output_piece(frag, root, fin_tile, _piece_inputs(root, fin_tile, ws_name))
    return _add_projection_pieces(match, result, projection_pieces, free, shapes)


__all__ = ["atomic_finalize", "realize_carry_split", "realize_split", "split_forks", "splitk_width", "state_ports"]


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
        if isinstance(s, Let):
            if s.value.free_vars() & linear:
                return False
            continue  # a literal or coordinate expression independent of the carried state
        if not isinstance(s, Assign):
            return False  # an unfamiliar projection stmt — can't prove distributivity
        kind = _homogeneity(s.value, linear)
        if kind == "constant":
            continue  # state-independent — a constant w.r.t. the split
        if kind is None:
            return False  # offset / inverse / nonlinear of a state value breaks distributivity
        linear.add(s.name)
    return False  # no Write reached


def _homogeneity(value, linear: set[str]) -> str | None:
    """How ``value`` depends on the carried state: ``"linear"`` for a linear-homogeneous function of the ``linear``
    names (a sum of them, one scaled or divided by a constant), ``"constant"`` for none of them, ``None`` for anything
    else (an offset beside a state term, a product of two state values, a nonlinear unary of one)."""
    if isinstance(value, Var):
        return "linear" if value.name in linear else "constant"
    if isinstance(value, Literal):
        return "constant"
    op = value.operation
    if op is None:
        return None
    kinds = [_homogeneity(operand, linear) for operand in value.operands]
    if any(kind is None for kind in kinds):
        return None
    if all(kind == "constant" for kind in kinds):
        return "constant"
    if op.name in ("add", "subtract", "negative"):
        return "linear" if all(kind == "linear" for kind in kinds) else None
    if op.name == "multiply" and kinds.count("linear") == 1:
        return "linear"
    if op.name == "divide" and kinds == ["linear", "constant"]:
        return "linear"
    return None
