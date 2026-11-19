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
from emmy.compiler.ir.pure import Lambda
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.pure.twist import Twist
from emmy.compiler.ir.schedule import Reduce, Work
from emmy.compiler.ir.schedule.catalog import splitk_moves
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Assign, Body, Let, Load, Select, SelectBranch, Statement, Write
from emmy.compiler.ir.tile import OutputSpec, Placement, TileOp
from emmy.compiler.ir.tile.ops import Sched, carries_partition, head, projection_regions, projection_root, projection_tail
from emmy.compiler.pipeline import Match
from emmy.compiler.pipeline.fork import DeferredFork
from emmy.compiler.pipeline.knob import axis_of, consume_kernel_row, kernel_pin
from emmy.compiler.pipeline.passes.tile._row import io_shapes, reformed
from emmy.compiler.pipeline.search.space import REDUCE, WORK

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


def _stream_refusal(node: Fold, k_axis: Axis, width: int) -> str | None:
    """Why the head fold's stream cannot split ``width`` ways (``None`` when it can): :func:`splitk_width` over a static
    stream. A symbolic one splits into slices of ``⌈K/w⌉`` positions, the last overhanging ``K``, whose loads clamp
    to the stream's last position and whose lift masks every position past its end (:func:`_masked`) — a monoid's or
    a twisted carrier's, never an observed scan's."""
    if k_axis.extent.is_static:
        return splitk_width(k_axis, width)
    if node.observe is not None:
        return f"cross-CTA split of the symbolic reduce axis {k_axis.name!r} is built for a monoid or a twisted carrier"
    return None


def _slice_length(k_axis: Axis, width: int):
    """The positions one partition's slice of the stream spans: ``K/w``, or ``⌈K/w⌉`` of a symbolic stream."""
    if k_axis.extent.is_static:
        return Literal(k_axis.extent.as_static() // width, "int")
    return BinaryExpression("/", BinaryExpression("+", k_axis.extent.expression, Literal(width - 1, "int")), Literal(width, "int"))


def _clamped(k_axis: Axis, position):
    """``position`` as the stream's loads read it: a symbolic stream's overhanging slice reads its last position
    (:func:`~emmy.compiler.ir.address.clamp_to`, which the gmem-direct fragment loaders read back)."""
    return position if k_axis.extent.is_static else clamp_to(position, k_axis.extent.expression)


def _masked(fold: Fold, k_axis: Axis, position) -> Fold:
    """``fold`` folding nothing at any ``position`` past a symbolic stream's end, which the last slice overhangs; a
    static stream's slices tile it exactly. A monoid folds its seed there. A twisted carrier's score takes the
    pivot's identity, added through the `where` selects a program's own mask spells — the coordinate mask the chunk tier
    lands — so the carrier past the end weighs its channels by nothing, and an all-overhang slice folds a finite
    state the combine then scales away."""
    if k_axis.extent.is_static:
        return fold
    end = k_axis.extent.expression
    stmts: list[Statement] = []
    if fold.twist is None:
        inside = BinaryExpression("<", position, end)
        results = []
        for name, seed in zip(fold.lift.results, fold.init, strict=True):
            stmts += [
                Let(name=f"{name}__seed", value=seed),
                Select(name=f"{name}__in", branches=(SelectBranch(name, inside), SelectBranch(f"{name}__seed", Literal(True, "bool")))),
            ]
            results.append(f"{name}__in")
    else:
        score, *channels = fold.lift.results
        stmts += [
            Let(name=f"{score}__key", value=position),
            Let(name=f"{score}__end", value=end),
            Let(name=f"{score}__keep", value=Literal(0.0, "float")),
            Let(name=f"{score}__fill", value=Literal(float(fold.init[0]), "float")),
            Assign(f"{score}__inside", apply("less", f"{score}__key", f"{score}__end")),
            Assign(f"{score}__mask", apply("where", f"{score}__inside", f"{score}__keep", f"{score}__fill")),
            Assign(f"{score}__in", apply("add", score, f"{score}__mask")),
        ]
        results = [f"{score}__in", *channels]
    body = Body((*fold.lift.body, *stmts))
    return fold.replace(lift=Lambda.closing(fold.lift.params, body, tuple(results)))


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
        and not carries_partition(tile)
        and not tile.split_consumed
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


# ---- slicing the head fold -------------------------------------------------------------------- #


def _slice_fold(fold: Fold, axis: Axis, width: int, split: Axis) -> tuple[Axis, Fold]:
    """``(the sliced axis, the same monoid Fold over one CTA's absolute contiguous slice)`` — the
    generic (non-contraction) slicer: the whole fold rides ``Fold.rewrite`` under the σ-offset."""
    b = _slice_length(axis, width)
    offset = BinaryExpression("+", Var(axis.name), BinaryExpression("*", Var(_SPLIT), b))
    sigma = Sigma({axis.name: _clamped(axis, offset)})
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
    return sliced_axis, _masked(fold.replace(operands=operands, lift=lift), axis, offset)


def _factor_k(k_axis: Axis, w: int) -> tuple[Axis, Axis, Sigma, BinaryExpression]:
    """Factor a contraction axis into ``ksplit × kslice``. ``ksplit`` (extent ``w``, name
    ``_<k>_ks``) becomes the partial's lead grid axis, parallelized across CTAs and combined in
    the finalize; ``kslice`` (extent ``K/w``, or ``⌈K/w⌉`` of a symbolic axis, the ORIGINAL name)
    is the sliced contraction's. The ``sigma`` maps the original ``k`` to ``ksplit·(K/w) + kslice``,
    clamped to a symbolic axis's last position (:func:`_clamped`), so the operand loads reconstruct
    the absolute index, returned last; distinct names are what avoid a double-reduce. The slice
    carries its parentage: a cross-CTA split is CONSUMED by the rewrite that realizes it, and an
    axis that is already a partition window is one nothing may partition again."""
    b = _slice_length(k_axis, w)
    ksplit = Axis(name=f"_{k_axis.name}_ks", extent=Dim(w))
    kslice = replace(k_axis, extent=Dim(b), window=Window(parent=k_axis.source_axis or k_axis, partition=True))
    position = BinaryExpression("+", BinaryExpression("*", Var(ksplit.name), b), Var(k_axis.name))
    return ksplit, kslice, Sigma({k_axis.name: _clamped(k_axis, position)}), position


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
        return edge.replace(index=tuple(sigma.apply(e) for e in edge.index))

    ops = tuple(_sliced_edge(e, sigma, k_name, kslice, ksplit) if k_name in e.free_axes else e for e in edge.operands)
    body = Body(tuple(s.substitute(sigma) for s in edge.lift.body))
    # Re-CLOSED, like every other σ-reindexed lift here. σ maps one coordinate onto an expression
    # over TWO (slice and partition), and expanding a param in PLACE shifted the operand
    # correspondence the prefix carries — by a ``free_vars()`` set order, so which name landed there
    # moved with the hash seed and ``_prune_unread`` then dropped a param the body still read.
    # Closing keeps the prefix exactly and appends the partition coordinate where a coordinate sits.
    lift = Lambda.closing(edge.lift.params, body, edge.lift.results)
    return edge.replace(operands=ops, lift=lift)


def _sliced_contraction(node: Fold, k_axis: Axis, w: int) -> tuple[Axis, Axis, Fold]:
    """``(ksplit, kslice, sliced)`` for a contraction head: the SAME bilinear node a non-split matmul
    builds, over ``kslice`` with operands σ-reindexed to absolute k, threading the node's OWN
    semiring (the reassociation ``fold_k = fold_{ksplit} ∘ fold_{kslice}`` is licensed by that
    ⊕-monoid's associativity). ``Fold.contraction`` regenerates the componentwise ⊕ over the same
    accumulator names, so the finalize folds the workspace states through the same monoid."""
    ksplit, kslice, sigma, position = _factor_k(k_axis, w)
    # Rebuilt DIRECTLY over the σ-reindexed operands, in stored order: the slice is the same term
    # with a narrower axis, so its monoid and seeds are the node's own — there is nothing for a
    # former to re-derive, and no role to re-name.
    operands = tuple(_sliced_edge(edge, sigma, node.axis, kslice, ksplit) for edge in node.operands)
    # The LIFT takes σ too, per statement so the binder is not shadowed away — the same rule
    # :func:`_slice_fold` applies on the generic side. A lift that only weighs its operands reads
    # no k and this changes nothing; one that reads the contraction coordinate DIRECTLY is
    # comparing against a partition-local index while its operands already reach absolute k.
    # Causal attention is exactly that lift: its mask is ``row < key``, and left unreindexed every
    # partition above the first admits keys far above the diagonal. The lift reads the absolute key
    # UNCLAMPED: only a load needs the clamp, and a mask that compared a clamped key would admit the
    # overhang as the stream's last position — :func:`_masked` masks it, and the chunk tier reads
    # the comparison for its stream bounds, which a clamp would hide.
    # Re-CLOSED, not re-spelled: the head's first param is its iteration binder, which must keep
    # its position, so the partition coordinate the substituted body now reads joins as a TRAILING
    # param — where a coordinate already sits, and past the operand correspondence.
    body = Body(tuple(stmt.substitute(Sigma({node.axis: position})) for stmt in node.lift.body))
    lift = Lambda.closing(node.lift.params, body, node.lift.results)
    return ksplit, kslice, _masked(node.replace(operands=operands, lift=lift), k_axis, position)


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


def add_output_piece(frag: Graph, root: Node, piece: TileOp, inputs: list[str], *, suffix: str = "__split") -> Graph:
    """Add a fresh piece with its owned output ports under the fragment's temporary names.

    ``root`` is the graph-node view of the ports this piece owns — :func:`output_root` narrows a
    MIMO node to a subset — so slot 0's edge key travels as the node id and every other as its own
    tensor name. ``suffix`` names the temporary those buffers travel under while the fragment is
    spliced in: the cross-CTA split mints ``__split``, the kernel-placement cut ``__placed``. ONE
    suffix per rewrite, so a fragment's temporary names say which decision minted them, and the arm
    that offers the fragment carries the same map for the whole kernel (:func:`redirects`)."""
    buffers = root.buffer_names()
    renamed = redirects(root, suffix)
    piece = replace(
        piece,
        output_specs=tuple(
            replace(spec, write=spec.write.replace(output=renamed.get(spec.write.output, spec.write.output))) for spec in piece.output_specs
        ),
    )
    tensors = (
        replace(root.outputs[0], name=buffers[0]),
        *(replace(tensor, name=renamed[name]) for name, tensor in zip(buffers[1:], root.outputs[1:], strict=True)),
    )
    frag.add_node(op=piece, inputs=inputs, outputs=tensors, node_id=renamed[buffers[0]])
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
    if node.as_contraction() is not None:
        split, kslice, partial_fold = _sliced_contraction(node, k_axis, cta)
    else:
        _enforce(_stream_refusal(node, k_axis, cta))
        split = Axis(name=_SPLIT, extent=Dim(cta))
        kslice, partial_fold = _slice_fold(node, k_axis, cta, split)
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
    fin_fold = _state_fold(fin_axis, partial_fold, loads)
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


__all__ = ["atomic_finalize", "realize_split", "split_forks", "splitk_width"]


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
