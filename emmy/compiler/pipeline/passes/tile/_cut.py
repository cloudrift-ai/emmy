"""Materialize a stored Fold edge as a kernel boundary.

The cut is structural: the child Fold keeps its algebra and becomes its own kernel, the parent
reads what it produced through ordinary ``Load`` edges. Both pieces are fresh unmapped ``TileOp``
objects. Pinned cuts consume placement on both pieces before the cut pass continues with cross-CTA
reduction splitting; unpinned cuts may expose smaller seams.

A seam has three realizations, and the seam itself decides which — one site stays ONE decision,
because at each seam one of them dominates the others outright and there is no trade for the
evidence to weigh:

- the WORKSPACE cut, the general case: the piece writes every state component to a fresh buffer.
- the STORAGE-FRONTIER cut (contraction-operand seams whose cone passes through a decode, see
  :func:`storage_frontier`): the buffer holds the raw storage bits, exact and narrower than the
  re-rounded result, and the consumer keeps the decode-plus-factors residue.
- the OUTPUT-OWNING cut (:func:`_output_owners`): where the seam's cone solely produces some of the
  kernel's OWN outputs, the piece writes those outputs and the sibling piece keeps the rest. The
  workspace would have held the output's exact bytes at the output's exact dtype, leaving the
  sibling nothing to do for them but copy, so this deletes a buffer and a copy. It also leaves both
  pieces single-output, which is what lets the shared-sweep promotion
  (:func:`~emmy.compiler.ir.tile.ir.promoted_sweep`) bind a sweep the fused kernel had to serialize.

Several seams can also be ONE decision. :func:`realize` takes a group, and :func:`full_projection_seams` names one
such group off the tree: on a projection that owns more outputs than the binder can bind, every contraction
occurrence beside every output-owning branch. That group has to be one decision — each of its seams alone leaves the
rest of the shape standing — which is also what makes the route recordable, since a measured row names a decision
rather than a sequence of them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import gcd

from emmy.compiler.dtype import F32
from emmy.compiler.dtype import get as get_dtype
from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.axis import Axis, Dim
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import BinaryExpr, Interval, Literal, SimplifyCtx, Var
from emmy.compiler.ir.pure.fold import (
    Fold,
)
from emmy.compiler.ir.pure.lam import Lambda
from emmy.compiler.ir.schedule.packing import match_packed_pair_node
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Assign, Body, Load, Write
from emmy.compiler.ir.stmt.passes import rewrite as rewrite_stmt
from emmy.compiler.ir.tile import OutputSpec, Placement, TileOp
from emmy.compiler.ir.tile.ir import promoted_sweep
from emmy.compiler.ir.tile.ops import (
    UnbindableProjection,
    carries_partition,
    edge_dtypes,
    output_regions,
    owns_outputs_it_cannot_bind,
)
from emmy.compiler.ir.tile.path import family_sites, sites, spell
from emmy.compiler.pipeline import Match
from emmy.compiler.pipeline.knob import consume_kernel_row
from emmy.compiler.pipeline.passes.tile._row import reformed
from emmy.compiler.pipeline.passes.tile._split import add_output_piece, output_root
from emmy.compiler.structural import digest
from emmy.compiler.tensor import Tensor


@dataclass(frozen=True)
class CutSite:
    """All stored occurrences of one canonically shared child Fold. ``dtypes`` is the workspace's
    per-component materialization, decided at offer time so the realization stores exactly what
    was offered. ``frontier`` (contraction-operand seams only) moves the cut to the cone's storage
    waypoint — see :class:`Frontier`."""

    node: Fold
    spelling: str
    axes: tuple
    dtypes: tuple
    frontier: Frontier | None = None
    #: Duplicate cones this seam ALSO stands for — alpha-equivalent up to their captured axis
    #: names: each sibling is ``(node, ((rep axis name, sibling axis name), …), channels)`` — the
    #: positional capture correspondence the clustering proved, and for each component the sibling
    #: exposes the position of the representative's component that is the same value (a lone
    #: contraction beside the twin that folds it with another). One placement decision
    #: materializes the value once; the realization replaces every sibling with workspace loads
    #: spelled through its own axes. Object sharing is the degenerate case (identity, with the
    #: identity correspondence).
    siblings: tuple = ()
    #: The siblings' own spellings: a row or a pin that names any occurrence of the value names
    #: this one decision, and the arm that cuts it spells every one of them.
    aliases: tuple[str, ...] = ()
    #: ``(tail, stores)`` when this seam's cone solely produces some of the kernel's OWN output
    #: specifications — those stores and the projection statements only they read — else ``None``.
    #: A seam with ``owned`` realizes as the output-owning cut (module docstring) and writes no
    #: workspace, so its ``dtypes`` are empty.
    owned: tuple | None = None


@dataclass(frozen=True)
class Frontier:
    """A contraction-operand cone's STORAGE waypoint: a decode (the ``ElementwiseImpl.decodes``
    trait) of a value the cone itself computes. The seam materializes there instead of at the
    cone's result — the workspace holds the raw storage bits (exact, the element the graph's own
    quantize produced), the producer piece computes ``producer`` (the encode prefix), and the
    consumer keeps ``residue`` (the decode plus the factor chain), which the normalize-time
    decode hoist then absorbs into a raw storage-dtype load with the factors on the accumulator
    epilogue — the same ``sum_k a*(s*w) = s*sum_k a*w`` reassociation as the materialized case."""

    name: str  # the encoded value the workspace holds
    producer: Fold  # the encode prefix, retaining the operand edges it reads
    residue: tuple  # the decode + factor stmts the consumer keeps
    dtype: object  # the storage DataType the decode op names


def storage_frontier(node: Fold) -> Frontier | None:
    """``node``'s storage frontier, or ``None`` when it has none the cut can separate.

    The shape is semantic, not an op list: exactly one decode of a value DEFINED by the cone's own
    body (a decode of a materialized load was already absorbed by normalization). The residue
    reads the stored bits and recomputes any pure values shared with the encode, such as its scale.
    Each side keeps its operand edges, so a computed scale can contain a reduction and can be cut
    separately in the same placement decision."""
    if not isinstance(node, Fold) or node.axis is not None or len(node.lift.results) != 1:
        return None
    lift = node.applied
    body = lift.body
    if any(not isinstance(stmt, (Load, Assign)) for stmt in body):
        return None
    computed = {name for stmt in body if isinstance(stmt, Assign) for name in stmt.defines()}
    decodes = [
        stmt
        for stmt in body
        if isinstance(stmt, Assign) and stmt.op.decodes is not None and len(stmt.args) == 1 and stmt.args[0] in computed
    ]
    if len(decodes) != 1:
        return None
    decode = decodes[0]
    frontier = decode.args[0]
    prefix = tuple(body.backward_cone((frontier,)).members)
    remaining = Body(stmt for stmt in body if frontier not in stmt.defines())
    residue = tuple(remaining.backward_cone(lift.results).members)
    if decode not in residue:
        return None
    result = get_dtype(decode.op.decodes)

    operands = tuple(edge for edge in node.operands if set(edge.exposes) & Body(prefix).ssa_uses)
    params = tuple(name for edge in operands for name in edge.exposes)
    producer = Fold(operands=operands, lift=Lambda.closing(params, Body(prefix), (frontier,)))
    return Frontier(name=frontier, producer=producer, residue=residue, dtype=result)


def _external_reads(node: Fold) -> frozenset[str]:
    """Everything ``node`` needs supplied from outside — read off its DECLARATION.

    Was: lower the term and walk the result for free names. That asked a term to re-derive what it
    already states, re-lowered on every call, and returned a superset (names the term binds
    internally, which every caller here discards). :attr:`Fold.free_axes` is the declaration —
    the term's own axes unioned with its operands', asked of the term rather than derived here. A
    term is closed: its values arrive through its operand edges, so its coordinates are all it
    takes from outside."""
    return node.free_axes


def _closed_at(node: Fold, axes: tuple) -> bool:
    """Whether ``node`` has no capture other than the axes (by name) in scope at its incoming edge."""
    return _external_reads(node) <= set(axes)


def _fed_store_dtype(tile: TileOp, consumer: Fold):
    """The dtype ``consumer`` stores its result at: the output its accumulators transitively feed
    (a forward closure over the root's lowered stmts covers any epilogue between the two), or
    ``None`` when the fed dtypes are not a singleton. A multi-output kernel can store siblings at
    other dtypes (w8a8's fp8 encode beside the f16 linear), so only the contraction's own stores
    speak for its slabs — and when it feeds outputs at SEVERAL dtypes no one of them does, so the
    seam stays undetermined and unoffered rather than resolved by list order."""
    if not tile.output_specs:  # the default store: the root's result to the kernel's one output
        tensor = next(iter(tile.outputs.values()), None)
        return None if tensor is None else tensor.dtype
    dependent = set(consumer.exposes)
    stmts = tile.op.lower(axes=tile.axes)
    for _ in stmts:
        grown = False
        for stmt in stmts:
            defines = Body((stmt,)).ssa_defs
            if not defines <= dependent and Body((stmt,)).ssa_uses & dependent:
                dependent |= defines
                grown = True
        if not grown:
            break
    fed = {
        tensor.dtype
        for store in tile.output_specs
        if store.write.value in dependent
        if (tensor := tile.outputs.get(store.write.output)) is not None
    }
    return fed.pop() if len(fed) == 1 else None


def _workspace_dtypes(node: Fold, tile: TileOp, consumer: Fold | None, table: dict[int, tuple]) -> tuple | None:
    """The cut workspace's per-component dtypes, or ``None`` when they cannot be determined.
    Reduction carrier precision is a Kernel IR policy — every Fold state is f32 until lowering
    stamps the concrete Accum/Init pair; a zero-axis value has no carrier and is inferred from its
    typed pure program instead. A seam standing in for a contraction OPERAND (``consumer`` is the
    consuming contraction) is the exception: it materializes explicitly at the dtype that
    contraction's output is stored at — the element the fused slab would have stored — never the
    carrier its cone computed in (only the ``a`` edge has a converting fill, so an f32 workspace on
    a ``b`` edge could feed no warp atom). That exception is a ZERO-AXIS cone's; a REDUCING operand
    would have been no slab fused either, so it keeps the f32 carrier (:func:`cuttable_seams` names
    which edges the exception reaches). A seam whose dtypes stay undetermined is not offered: the
    offer and the realization must agree, and a raise past the offer would kill the compile."""
    names = node.exposes
    if consumer is not None:
        dtype = _fed_store_dtype(tile, consumer)
        return None if dtype is None else (dtype,) * len(names)
    dtypes = (F32,) * len(names) if node.axis is not None else table.get(id(node), ())
    if len(dtypes) != len(names) or any(dtype is None for dtype in dtypes):
        return None
    return dtypes


def _dtype_table(tile: TileOp) -> dict[int, tuple]:
    """Each stored edge's inferred result dtypes, keyed by edge identity.

    One edge, one answer: a term is CLOSED, so its values arrive through its own operand edges and
    it types the same wherever it occurs. The occurrence-agreement filter this used to apply — one
    shared cone under two scopes having two answers — cannot arise once no term captures."""
    cache: dict[int, tuple] = {}
    edge_dtypes(tile.op, tile.inputs, cache)
    return dict(cache)


def _output_owners(tile: TileOp) -> dict[int, tuple]:
    """The root operands that solely produce some of this kernel's outputs AND would bind a grid
    axis the fused kernel cannot — keyed by operand identity, each mapped to ``(tail, stores)``.

    Two conditions, both structural, both asked of rules that already exist.

    OWNERSHIP is :func:`~emmy.compiler.ir.tile.ops.output_regions`: every output specification must
    read exactly one operand, the operands' cones over the root body must be disjoint, and together
    they must cover it. Without it the pieces would not be a partition of the kernel — one of them
    would have to recompute what it no longer owns.

    RANK is :func:`~emmy.compiler.ir.tile.ir.promoted_sweep`, asked twice: once of the fused kernel,
    once of the candidate piece. A piece promotes a sweep the whole kernel could not exactly when
    its stores agree on an axis the sibling's stores do not, so the intersection over ALL stores
    came up empty — the NVFP4 encode, whose packed codes ride the feature axis and whose block
    scales ride one sixteenth of it. Where the piece promotes nothing more than the kernel already
    does, splitting buys a second launch and no grid, so the seam keeps its workspace reading.
    """
    op = tile.op
    if len(tile.output_specs) < 2 or not isinstance(op, Fold) or op.axis is not None or len(op.operands) < 2:
        return {}
    try:
        regions = output_regions(op, tile.output_specs)
    except UnbindableProjection:
        return {}
    fused = promoted_sweep(op, tile.output_specs, free=tile.place.free)
    return {
        id(region): (tail, stores)
        for region, tail, stores in regions
        if stores and not promoted_sweep(region, stores, free=tile.place.free) <= fused
    }


def cuttable_seams(tile: TileOp) -> tuple[CutSite, ...]:
    """Every semantically closed stored Fold edge a cut can hand its own kernel, grouped only by
    object sharing. A contraction's operand edges are seams too — cutting one materializes the cone
    feeding the operand into its own kernel and the contraction reads it back as an ordinary load —
    and they take the explicit contraction-operand dtype rule (`_workspace_dtypes`), except on a
    block-scaled packed pair, whose operand cones are not seams at all. A seam is offered only where
    the cone is closed at the axes of every occurrence; a term is closed by construction, so that
    check names a malformed tree rather than a capture to resolve.

    Workspace dtypes must be determined for the seam to be offered — the offer and the realization
    must agree, and a raise past the offer would kill the compile. An OUTPUT-OWNING seam
    (:func:`_output_owners`) writes no workspace, so it carries none and that condition does not
    apply to it."""
    all_sites = sites(tile.op)
    owners = _output_owners(tile)
    # Only a ZERO-AXIS operand cone: the rule says the workspace holds what the fused slab would
    # have stored, and a cone is exactly that element. A REDUCING operand — a twisted carrier's
    # score contraction — is no slab fused either: it stays an f32 accumulator in registers, so its
    # workspace is the carrier's own f32, and f16 scores would reach the softmax a percent off.
    store_dtype_consumers = {
        id(edge): site.node
        for site in all_sites
        if site.node.as_contraction() is not None
        for edge in site.node.operands
        if isinstance(edge, Fold) and edge.as_slab() is None and edge.axis is None
    }
    outer = tuple(axis.name for axis in (*tile.place.free, *(axis for store in tile.output_specs for axis in store.sweep)))
    occurrence_axes: dict[int, list[tuple]] = {}
    for site in all_sites[1:]:
        occurrence_axes.setdefault(id(site.node), []).append((*outer, *site.scope))
    dtype_table: dict[int, tuple] = {}
    if isinstance(tile.op, Fold):
        dtype_table = _dtype_table(tile)
    taken = _kept_components(tile)
    out: list[CutSite] = []
    seen: set[int] = set()
    for site in family_sites("PLACE", all_sites):
        node = site.node
        scopes = occurrence_axes.get(id(node), ())
        if not isinstance(node, Fold) or node.as_slab() is not None or id(node) in seen or not scopes:
            continue
        if not all(_closed_at(node, scope) for scope in scopes):
            continue
        if node.observe is not None:
            # An observed fold's per-step results exist only inside its stream — a cut would
            # separate the scan from its streamed boundary store, which no piece can then spell.
            continue
        if not taken.get(id(node), node.exposes):
            # No reader takes any component: lowering drops the edge outright, so a workspace
            # here would be written and never read.
            continue
        if node.scalar():
            # One value for the whole kernel (an sdpa scale and its mask fills). The piece would be
            # a kernel that writes those scalars to a workspace so its reader can read them back —
            # never the faster kernel set, and one more arm for the greedy to rank and price.
            continue
        consumer = store_dtype_consumers.get(id(node))
        if consumer is not None and match_packed_pair_node(consumer, tile.inputs) is not None:
            # An operand cone of a BLOCK-SCALED packed pair reaches gmem already: its codes and
            # its block scale are loads, and the cone only decodes them. Materializing it stores
            # the decoded values instead, so the consumer holds neither the codes nor the
            # per-block scale the cell multiplies — the reading is gone for every occurrence the
            # seam covers, and no piece can put it back. Nothing is hoisted either way, so this
            # is not a placement trade. The contraction's OWN seam stays offered, and that is the
            # cut that gives the piece the output-axis pair a fragment needs.
            continue
        # A frontier REPLACES the fed-store realization at this seam rather than joining the
        # offer: the raw bits dominate the fed-store workspace on both precision (exact vs
        # re-rounded) and footprint (storage width vs store width), so there is no trade for the
        # evidence to decide — one site stays one decision.
        owned = owners.get(id(node))
        frontier = storage_frontier(node) if consumer is not None and owned is None else None
        if owned is not None:
            dtypes = ()  # the piece writes the kernel's own outputs; there is no workspace to type
        else:
            dtypes = (frontier.dtype,) if frontier is not None else _workspace_dtypes(node, tile, consumer, dtype_table)
            if dtypes is None:
                continue
        seen.add(id(node))
        # Keep the term's read coordinates and unit axes that supply its matrix geometry.
        axes = tuple(
            axis
            for name in dict.fromkeys(name for scope in scopes for name in scope)
            for axis in (tile.axis_of(name),)
            if name in node.free_axes or axis.extent == 1
        )
        out.append(
            CutSite(
                node=node,
                spelling=spell(tile.op, "PLACE", node, all_sites=all_sites),
                axes=axes,
                dtypes=dtypes,
                frontier=frontier,
                owned=owned,
            )
        )
    return _cluster_value_seams(out, tile.axes)


def _hoisted_reduces(tile: TileOp) -> set[int]:
    """The reduce nodes evaluated ONCE ahead of the output sweep of the branch that holds them,
    keyed by identity — a branch's ROW STATISTIC, as opposed to a fold each cell of the sweep
    performs for itself.

    This is :func:`~emmy.compiler.ir.tile.ir.promoted_sweep`'s refusal read from the other side.
    That rule will not bind a sweep past a reduce invariant in it, because each cell would then
    recompute the whole statistic; so a piece holding one stays at a single grid axis however much
    else is cut away from it. Naming them here is what lets the cut hand them their own kernels
    instead, after which the piece reads one stored value and its sweep binds.
    """
    hoisted: set[int] = set()
    for region, _tail, stores in output_regions(tile.op, tile.output_specs):
        sweep = set.intersection(*({axis.name for axis in store.sweep} for store in stores)) if stores else set()
        if not sweep:
            continue
        hoisted.update(
            id(site.node)
            for site in sites(region)
            if isinstance(site.node, Fold) and site.node.axis is not None and not sweep <= site.node.free_axes
        )
    return hoisted


def full_projection_seams(tile: TileOp, seams) -> tuple[CutSite, ...]:
    """The seams of the FULL-PROJECTION cut — every contraction occurrence of this kernel, every
    reduce hoisted ahead of an output sweep, and every output-owning branch — or ``()`` where the
    kernel has no such cut to offer.

    Offered on a projection that owns more outputs than it can bind
    (:func:`~emmy.compiler.ir.tile.ops.owns_outputs_it_cannot_bind`). Such a kernel builds around
    one reduce and lowers every other serially inside the projection, so all but one of its
    contractions reach no tensor-core tier where they are — and its stores ride different axes, so
    the fused grid promotes none of them and even that one root has no ``(m, n)`` pair to tile.
    The cut answers both at once: each contraction becomes the sole root of its own kernel, and each
    owned output becomes a pointwise-or-small-reduce kernel over the sweep its store rides.

    It is ONE decision, not a sequence. Each seam alone leaves the rest of the shape standing, the
    pieces a partial cut mints respell what is left, and the evidence a route is recorded as names a
    decision rather than a sequence of them.

    The seams are the ones :func:`cuttable_seams` already offers; nothing new becomes cuttable. A
    contraction it does not offer stays where it is — the piece around it is no worse than the fused
    kernel was — so this takes what is on the ballot rather than declining the whole cut. A reduce
    each cell of a sweep performs for itself stays too: a block maximum over sixteen stored values is
    the small-reduce half of the target shape, and cutting it would buy a launch and a workspace for
    nothing.
    """
    if not owns_outputs_it_cannot_bind(tile.op, tile.output_specs):
        return ()
    hoisted = _hoisted_reduces(tile)
    chosen = tuple(seam for seam in seams if seam.owned is not None or seam.node.as_contraction() is not None or id(seam.node) in hoisted)
    return chosen if len(chosen) > 1 else ()


def _pruned(body: Body, roots: frozenset[str]) -> Body:
    """``body`` cut to what ``roots`` need — the statements a root reads through, in order, a
    loop kept with what its own body keeps."""
    from emmy.compiler.ir.stmt.body import free_names  # noqa: PLC0415

    kept: list = []
    needed = set(roots)
    for stmt in reversed(tuple(body)):
        if stmt.nested():
            inner = tuple(_pruned(child, frozenset(needed)) for child in stmt.nested())
            if any(inner):
                stmt = stmt.with_bodies(inner)
                kept.append(stmt)
                needed |= free_names(stmt)
        elif needed.intersection(stmt.defines()) or stmt.external_writes():
            kept.append(stmt)
            needed |= free_names(stmt)
    return Body(tuple(reversed(kept)))


def _value_forms(seam: CutSite, axes: tuple) -> tuple:
    """What each component of a seam's cone computes, spelled so two copies of one value key
    alike: the cone lowered over its captured axes renamed by position, cut to the component,
    under the exact statement identity (SSA names, commutative order and buffer declaration order
    do not reach it) beside the buffers that fill its roles. A copy under another scope binds the
    same coordinates by other names — the o_proj result feeds a norm's statistic inside a reduce
    and the residual add at the kernel's own free axis — and its lambda params may sit in another
    order, which :meth:`Fold.canonical` reads positionally; neither is a different value. Per
    component, because a value folded beside another in one twin (the k and v projections over
    one input) is the same value as the lone contraction that feeds its norm's statistic."""
    from emmy.compiler.ir.stmt.identity import canonicalize_identity  # noqa: PLC0415 — identity imports the tile IR

    scoped = tuple(axis.name for axis in seam.axes if axis.name in seam.node.free_axes)
    names = {name: f"_s{position}" for position, name in enumerate(scoped)}
    body = Body(tuple(stmt.rewrite(lambda name: names.get(name, name)) for stmt in seam.node.lower(bound=frozenset(scoped), axes=axes)))
    forms = []
    for exposed in seam.node.exposes:
        identity = canonicalize_identity(_pruned(body, frozenset((exposed,))))
        forms.append((identity.key, identity.arguments))
    return tuple(forms)


def _cluster_value_seams(seams: list[CutSite], axes: tuple) -> tuple[CutSite, ...]:
    """Fold duplicate cones — alpha-equivalent up to captured axis names — into ONE seam per value.

    Object sharing groups occurrences of one stored node; a traced graph can also hold several
    ALPHA-EQUIVALENT copies of the same computation captured under different axis names —
    attention's normalized K cone appears once per score contraction, and a fused decoder half
    reads its o_proj result under the post-attention statistic's reduce, under the pre-FFN
    statistic's and at the residual add. Those copies are one VALUE: the cluster's first seam
    becomes the decision for all of them, carrying each duplicate as a sibling with its
    positional capture correspondence (:class:`CutSite`), so one cut materializes the value once
    and every occurrence reads the workspace. Membership is :func:`_value_forms` inclusion; a member
    joins only when its paired axes agree on extent and window, its workspace dtypes match, and
    every workspace axis is a mapped capture — otherwise it stays its own seam. An output-owning
    seam writes the kernel's outputs and a frontier seam its raw storage bits; neither is a
    workspace another occurrence could read, so they cluster with nothing."""
    eligible = [index for index, seam in enumerate(seams) if seam.frontier is None and seam.owned is None]
    if len(eligible) < 2:
        return tuple(seams)
    captured = {index: tuple(axis.name for axis in seams[index].axes) for index in eligible}
    scoped = {index: tuple(axis for axis in captured[index] if axis in seams[index].node.free_axes) for index in eligible}
    forms = {index: _value_forms(seams[index], axes) for index in eligible}
    descendants = {index: {id(site.node) for site in sites(seams[index].node)[1:]} for index in eligible}
    drop: set[int] = set()
    merged: dict[int, CutSite] = {}
    # The representative exposes the most: a twin stands for the lone contractions that equal its
    # channels, never the other way round.
    for rep_index in sorted(eligible, key=lambda index: -len(forms[index])):
        if rep_index in drop:
            continue
        rep = seams[rep_index]
        rep_params = scoped[rep_index]
        rep_axes = {axis.name: axis for axis in rep.axes}
        if {axis.name for axis in _workspace_axes(rep, rep.node) if not _unit(axis)} - set(rep_params):
            continue  # a workspace axis with no capture to map has no sibling spelling; a unit axis reads at 0
        siblings = []
        aliases = []
        for member_index in eligible:
            if member_index == rep_index or member_index in drop or member_index in merged:
                continue
            member = seams[member_index]
            # A multi-result cone may expose a value it also consumes below another result.
            # Replacing that descendant by the cone's workspace would make the producer cyclic.
            if id(member.node) in descendants[rep_index] or id(rep.node) in descendants[member_index]:
                continue
            if not all(form in forms[rep_index] for form in forms[member_index]):
                continue
            channels = tuple(forms[rep_index].index(form) for form in forms[member_index])
            member_params = scoped[member_index]
            member_axes = {axis.name: axis for axis in member.axes}
            aligned = (
                len(member_params) == len(rep_params)
                and tuple(member.dtypes) == tuple(rep.dtypes[channel] for channel in channels)
                and all(
                    (a := rep_axes[rn]).extent == (b := member_axes[mn]).extent and a.window == b.window
                    for rn, mn in zip(rep_params, member_params, strict=True)
                )
            )
            if not aligned:
                continue
            siblings.append((member.node, tuple(zip(rep_params, member_params, strict=True)), channels))
            aliases.append(member.spelling)
            drop.add(member_index)
        if siblings:
            merged[rep_index] = replace(rep, siblings=tuple(siblings), aliases=tuple(aliases))
    return tuple(merged.get(index, seam) for index, seam in enumerate(seams) if index not in drop)


def _unchanged(pieces: tuple, members) -> bool:
    return len(pieces) == len(members) and all(piece is member for piece, member in zip(pieces, members, strict=True))


def _replace_member(member, targets: dict[int, tuple], renamed: dict[str, str]):
    if id(member) in targets:
        # A storage-frontier residue retains operand edges. Apply nested cuts there too, just
        # as in the encode producer, so both sides reuse a separately materialized scale.
        return tuple(_replace_fold(piece, targets, renamed) if isinstance(piece, Fold) else piece for piece in targets[id(member)])
    if isinstance(member, Fold):
        return (_replace_fold(member, targets, renamed),)
    nested = member.nested()
    if not nested:
        return (member,)
    bodies = []
    changed = False
    for body in nested:
        replaced = tuple(piece for child in body for piece in _replace_member(child, targets, renamed))
        changed = changed or not _unchanged(replaced, body)
        bodies.append(Body(replaced))
    return (member.with_bodies(tuple(bodies)) if changed else member,)


def _follow_reads(before: Fold, after: Fold, renamed: dict[str, str]) -> Fold:
    """``after`` with every value its statements DERIVE from a re-spelled read re-spelled the same way.

    :func:`_read_name` tags a workspace read because the value can still be computed in place
    beside it. The same holds one step on: a statement over that read is a different statement from
    the one still computed in place — ``v = in0__ws… + b`` beside ``v = in0 + b`` — and a lowered
    scope binds a name once, so under one name the pair is the same SSA fault (nvcc: *already
    declared*). Each such definition takes the tags of the reads it depends on; a statement that
    reads no workspace keeps its name and still shares with its in-place copy. The term's readers
    follow through :attr:`~emmy.compiler.ir.pure.fold.Fold.applied`, which is what walks the rename
    up the tree, one rebuilt term at a time. It stops at a reduce: a carried state keeps its name,
    because a twisted carrier's tile offer reads it. ``renamed`` collects what was minted, for the
    boundary stores.
    """
    was = {param: edge.exposes[slot] for param, edge, slot in before.bindings}
    tags = {param: now.removeprefix(was[param]) for param, edge, slot in after.bindings if (now := edge.exposes[slot]) != was[param]}
    for stmt in after.lift.body:
        read = sorted({tags[name] for name in Body((stmt,)).ssa_uses if name in tags})
        tags.update((name, "".join(read)) for name in stmt.defines() if read)
    names = {name: f"{name}{tag}" for name, tag in tags.items() if name not in was}
    renamed.update(names)
    return replace(after, lift=after.lift.rename(names)) if names else after


def _replace_fold(node: Fold, targets: dict[int, tuple], renamed: dict[str, str]) -> Fold:
    """Replace every stored occurrence of the target Folds in ONE walk — ``targets`` maps
    ``id(node)`` to its replacement stmts. One walk, because the rebuild copies every node on the
    way down: a second walk's target objects no longer exist in the first walk's output, so
    sequential replacement silently loses every decision after the first. IDENTITY-PRESERVING off
    the replacement spine: a subtree holding no target returns the SAME object, so untouched
    Lambdas are not reconstructed (construction normalization over a large fused body is where a
    copying walk turns quadratic) and shared-node grouping keeps its identities.

    An operand's replacement is POSITIONAL — one entry per component the edge exposed — and
    ``None`` there means the reader takes that component no more (:func:`_kept_components`). The
    param it bound goes with it: params past the iteration var bind the operands' components in
    order, so a dropped entry that left its param behind would turn a value into a free coordinate
    the kernel can hand no extent."""
    operands: list = []
    bound: list[bool] = []
    for edge in node.operands:
        pieces = _replace_member(edge, targets, renamed)
        components = len(edge.exposes) if isinstance(edge, Fold) else 0
        if any(piece is None for piece in pieces):
            bound.extend(piece is not None for piece in pieces)
            pieces = tuple(piece for piece in pieces if piece is not None)
        else:
            bound.extend([True] * components)
        operands.extend(pieces)
    operands = tuple(operands)
    body = tuple(piece for stmt in node.lift.body for piece in _replace_member(stmt, targets, renamed))
    if _unchanged(operands, node.operands) and _unchanged(body, node.lift.body):
        return node
    lift = replace(node.lift, body=Body(body))
    if not all(bound):
        lead = 1 if node.base is not None else 0
        params = node.lift.params
        head, slots, tail = params[:lead], params[lead : lead + len(bound)], params[lead + len(bound) :]
        lift = replace(lift, params=(*head, *(name for name, keep in zip(slots, bound, strict=True) if keep), *tail))
    return _follow_reads(node, replace(node, operands=operands, lift=lift), renamed)


def _kept_components(tile: TileOp) -> dict[int, tuple[str, ...]]:
    """Per stored edge, the result components its READERS take — what :meth:`Fold.lower` places.

    A term may carry more components than any one reader wants: six channels folded into one
    reduce, each occurrence read for a single accumulator. A cut that materialized all six would
    write five workspaces nothing loads back, and the backend's liveness plan refuses a scratch
    buffer with no consuming launch. An edge a boundary store reaches keeps every component — the
    store reads the term, not a reader's narrowing.
    """
    if not isinstance(tile.op, Fold):
        return {}
    # ``Fold.lower`` re-spells the stores into the root's applied vocabulary before it places
    # anything, so a store naming a bound param names the operand result it binds. Compare in that
    # same spelling or a store of an operand's own result reads as a name no edge exposes.
    spelled = dict(zip(tile.op.lift.params, tile.op.applied.params, strict=True))
    stored = {spelled.get(name, name) for store in tile.output_specs for name in store.write.values}
    return tile.op.read_components(frozenset(stored))


def _unit(axis) -> bool:
    return axis.extent.is_static and axis.extent.as_static() == 1


def _workspace_axes(seam: CutSite, produced: Fold) -> tuple:
    """The seam axes the PRODUCED piece actually sweeps — its workspace dimensions. ``produced``
    is the seam node, or the frontier prefix when the seam materializes at a storage waypoint.

    Static unit axes consume no additional storage, but retain the producer's schedule geometry.
    Dropping one lets a later split axis take its place as a contraction fragment axis even though
    operand indices still read it as the outer partition coordinate."""
    read = _external_reads(produced)
    return tuple(axis for axis in seam.axes if axis.name in read or _unit(axis))


def _workspace_strides(produced: Fold, axes: tuple) -> dict[str, int]:
    """A coordinate used only as ``i // d`` needs one stored value per group of ``d`` cells.

    Every occurrence must be the numerator of a positive integer division. Different divisors
    share their greatest common divisor; a direct read or a remainder keeps the full extent.
    Read the stored terms, including coordinate predicates, before choosing a representative.
    """
    factors = dict.fromkeys((axis.name for axis in axes), 0)
    pending = [produced]
    while pending:
        term = pending.pop()
        pending.extend(term.operands)
        for name in factors:
            if name not in term.free_axes:
                continue
            if term.observe is not None or name in term.lift.results:
                factors[name] = 1
            for stmt in term.lift.body.iter():
                if not isinstance(stmt, Load) and name in stmt.deps():
                    factors[name] = 1
                for expr in stmt.exprs():
                    parts = tuple(expr.subterms())
                    uses = sum(isinstance(part, Var) and part.name == name for part in parts)
                    divisors = [
                        int(part.right.value)
                        for part in parts
                        if isinstance(part, BinaryExpr)
                        and part.op in ("/", "//")
                        and part.left == Var(name)
                        and isinstance(part.right, Literal)
                        and part.right.dtype == "int"
                        and part.right.value > 0
                    ]
                    factors[name] = gcd(factors[name], *divisors) if uses == len(divisors) else 1
    return {name: factor for name, factor in factors.items() if factor > 1}


class _FoldingSigma(Sigma):
    """A substitution that FOLDS as it lands, under a fixed range context.

    ``Sigma.apply`` substitutes and stops, which is right everywhere else: a σ that rewrote its
    result would hide what it did. Here the fold is the point — the whole gain is that
    ``i / c`` and ``i % c`` collapse to the bare split coordinates, and an index left as
    ``((hi * c) + lo) / c`` reaches the scheduler as an expression of the fused name again."""

    def apply(self, expr):
        return expr.substitute(self.mapping).simplify(self._ctx)


def _substitute_and_fold(produced: Fold, name: str, sigma: Sigma, ctx: SimplifyCtx) -> Fold:
    """``produced`` with the coordinate ``name`` substituted and every index folded against ``ctx``.

    Applied per STATEMENT, not to the term: a free coordinate is a lift PARAM, so a σ handed the
    term is dropped as shadowed exactly where it has work to do. Each lift is re-closed instead —
    the substituted name leaves the param list and :meth:`Lambda.closing` appends the two the body
    now reads, as TRAILING params, which is where a coordinate already sat and is what keeps the
    operand correspondence in the prefix untouched."""
    folding = _FoldingSigma(dict(sigma.mapping))
    object.__setattr__(folding, "_ctx", ctx)
    operands = tuple(_substitute_and_fold(edge, name, sigma, ctx) for edge in produced.operands)
    body = Body(rewrite_stmt(stmt, lambda n: n, folding) for stmt in produced.lift.body)
    lift = Lambda.closing(tuple(p for p in produced.lift.params if p != name), body, produced.lift.results)
    return replace(produced, operands=operands, lift=lift)


def _divmod_in_edge(edge: Fold, name: str, factor: int) -> tuple[bool, bool]:
    """``(the edge reads name / factor, the edge reads name's low part)`` anywhere under it.

    The low part is ``name % factor`` or ``name`` itself: a plain read depends on both halves, as
    a workspace indexed by the fused name does when a cut materialized one operand at it."""
    div = low = False
    pending = [edge]
    while pending:
        term = pending.pop()
        pending.extend(term.operands)
        for stmt in term.lift.body.iter():
            for expr in stmt.exprs():
                uses = covered = 0
                for part in expr.subterms():
                    uses += isinstance(part, Var) and part.name == name
                    if (
                        isinstance(part, BinaryExpr)
                        and part.left == Var(name)
                        and isinstance(part.right, Literal)
                        and part.right.value == factor
                    ):
                        covered += 1
                        div = div or part.op in ("/", "//")
                        low = low or part.op == "%"
                low = low or uses > covered
    return div, low


def _straddles_a_contraction(produced: Fold, name: str, factor: int) -> bool:
    """Whether the pair straddles ONE contraction's operands — the div feeding a DIFFERENT edge
    from the mod.

    That straddle IS the pathology, and nothing weaker is. It says the contraction's A operand is
    indexed by ``name / factor`` while its B operand contracts into ``name % factor``, so A depends
    on the axis B reduces over and the term is not a matmul the warp tier can tile. A coordinate
    that merely happens to carry a divmod somewhere — a flat index a reduce walks, a strided read —
    is an ordinary coordinate, and splitting it only re-spells a kernel that was already scheduled.
    """
    pending = [produced]
    while pending:
        term = pending.pop()
        pending.extend(term.operands)
        if term.axis is None or len(term.free_axes) < 2:
            # TWO free axes or nothing: the pathology presupposes the contraction already HAS an m
            # and an n and that the fused pair is one of them, so splitting only un-entangles what
            # the tier could otherwise tile. A contraction with ONE free axis is a different shape
            # — splitting INVENTS its second dimension — and the DeepSeek V4 post block measured
            # what that costs: its piece stopped completing a single iteration in 90 s.
            continue
        divs = {position for position, edge in enumerate(term.operands) if _divmod_in_edge(edge, name, factor)[0]}
        mods = {position for position, edge in enumerate(term.operands) if _divmod_in_edge(edge, name, factor)[1]}
        if divs and mods and (divs - mods or mods - divs):
            return True
    return False


def _fused_pair_factor(produced: Fold, axes: tuple) -> tuple[str, int] | None:
    """A grid coordinate read as ``i / c`` beside ``i % c`` or ``i`` itself is TWO coordinates
    wearing one name — ``(the axis, c)``, or ``None``. The plain read is a workspace a cut stored
    at the fused index: the other operand of the same contraction still reads ``i / c``.

    Attention's (head, head-dim) pair arrives fused: the projection downstream reshapes the
    attention output to one flat width, and a cut inherits that spelling. While the pair stays
    fused the contraction's A operand is indexed by ``i / c`` — it DEPENDS on the axis the B
    operand contracts into — which is not a matmul, so the warp tier never offers a tile and the
    piece falls to a per-cell reduce that walks the whole reduction once per ``c``. Splitting the
    name restores the batched matmul the tier already schedules elsewhere.

    Distinct from :func:`_workspace_strides`, which claims a coordinate used only as ``i // d``
    and rightly declines this one: a remainder beside the division is not a narrower workspace,
    it is a second axis."""
    for axis in axes:
        name = axis.name
        if not axis.extent.is_static:
            continue
        extent = axis.extent.as_static()
        divisors: set[int] = set()
        remainders: set[int] = set()
        uses = covered = 0
        pending = [produced]
        while pending:
            term = pending.pop()
            pending.extend(term.operands)
            if name not in term.free_axes:
                continue
            for stmt in term.lift.body.iter():
                for expr in stmt.exprs():
                    for part in expr.subterms():
                        if isinstance(part, Var) and part.name == name:
                            uses += 1
                        if (
                            isinstance(part, BinaryExpr)
                            and part.left == Var(name)
                            and isinstance(part.right, Literal)
                            and part.right.dtype == "int"
                            and isinstance(part.right.value, int)
                            and part.right.value > 1
                        ):
                            if part.op in ("/", "//"):
                                divisors.add(int(part.right.value))
                                covered += 1
                            elif part.op == "%":
                                remainders.add(int(part.right.value))
                                covered += 1
        if len(divisors) == 1 and remainders <= divisors and (remainders or uses > covered):
            (factor,) = divisors
            if 1 < factor < extent and extent % factor == 0 and _straddles_a_contraction(produced, name, factor):
                return name, factor
    return None


def _split_fused_pair(produced: Fold, axes: tuple, index: tuple) -> tuple[Fold, tuple, tuple, tuple]:
    """``(tree, grid axes, write index, minted axes)`` with one fused pair split back into two.

    The substitution ``i -> hi * c + lo`` is an exact change of iteration variables, so the tree
    keeps its meaning whatever the detector decided; what changes is that the operands' ``i / c``
    and ``i % c`` fold to the bare coordinates the contraction wants. The WRITE keeps the fused
    index and the workspace its shape, so the sibling that reads it back needs no adjustment —
    only the grid is two-dimensional where it was one."""
    found = _fused_pair_factor(produced, axes)
    if found is None:
        return produced, axes, index, ()
    name, factor = found
    axis = next(a for a in axes if a.name == name)
    extent = axis.extent.as_static()
    taken = {a.name for a in axes} | set(produced.free_axes)
    hi, lo = (f"{name}_{suffix}" for suffix in ("hi", "lo"))
    while hi in taken or lo in taken:
        hi, lo = f"{hi}_", f"{lo}_"
    fused = BinaryExpr("+", BinaryExpr("*", Var(hi), Literal(factor, "int")), Var(lo))
    ranges = {hi: Interval(0, extent // factor - 1), lo: Interval(0, factor - 1)}
    produced = _substitute_and_fold(produced, name, Sigma({name: fused}), SimplifyCtx(ranges=ranges))
    minted = (Axis(name=hi, extent=Dim(extent // factor)), Axis(name=lo, extent=Dim(factor)))
    grid = tuple(part for a in axes for part in (minted if a.name == name else (a,)))
    return produced, grid, tuple(expr.substitute({name: fused}) for expr in index), minted


def _buffer_reads(node: Fold) -> set[str]:
    """The gmem buffers ``node``'s STORED tree reads — every lift body's loads, through the operand
    edges (a slab's body is its one load). Read off the tree, never by lowering it."""
    out = {load.input for load in node.lift.body.loads}
    for edge in node.operands:
        out |= _buffer_reads(edge)
    return out


def _piece_inputs(root: Node, fold: Fold, first: tuple[str, ...] = ()) -> list[str]:
    """A piece's graph inputs: its workspaces, then every buffer of ``root``'s the piece reads.

    The read set is the STORED tree's (:func:`_buffer_reads`), walked through the operand edges a
    body cannot reach. A walk that stopped at the stored body named fewer inputs than the kernel
    went on to read; the workspace producers then had no consumer edge, were pruned as orphans,
    and the launch asked for a buffer nothing had allocated."""
    reads = _buffer_reads(fold)
    return [*first, *(name for name in root.inputs if name in reads)]


def _input_fragment(match: Match, root: Node) -> Graph:
    fragment = Graph()
    for name in root.inputs:
        fragment.add_node(op=InputOp(), inputs=[], output=match.graph.buffer(name), node_id=name)
    return fragment


#: The temporary every piece of a placement cut travels under while the fragment is spliced in.
#: One suffix per rewrite (``_split`` mints ``__split``), so a fragment's names say which decision
#: minted them.
_PLACED = "__placed"


def output_map(root: Node) -> dict[str, str]:
    """Stable temporary output names used by every cut sibling of ``root``."""
    return {name: f"{name}{_PLACED}" for name in root.buffer_names()}


def _in_source_order(stores: tuple, order: list[str]) -> tuple:
    """``stores`` in the kernel's own output-specification order. The ownership partition groups by
    region; ``apply_output_specs`` reads consecutive same-path stores as one sweep nest, so the
    pieces must keep the order the kernel spelled rather than the order the partition returned."""
    return tuple(sorted(stores, key=lambda store: order.index(store.write.output)))


def _region_term(regions: tuple, body, results: tuple) -> Fold:
    """A zero-axis term over ``regions`` and the projection statements they own, exposing
    ``results`` — the values the stores that stay with it are written from. The results are named
    rather than derived from the body's last definition: a piece may own several stores, and one
    that owns a store read straight off a region has no body at all."""
    bound = tuple(name for region in regions for name in region.exposes)
    return Fold(operands=regions, lift=Lambda.closing(bound, Body.coerce(body), results))


def _region_piece(tile: TileOp, regions: tuple, tail, stores: tuple, placement_decided: bool, split_consumed: bool, spelling: str):
    """One output-owning piece: the regions' term, the outputs they produce, and the PARENT's free
    axes. The placement is deliberately the parent's and not the seam's own axes — the piece is a
    kernel writing the kernel's own outputs, so its grid is settled by the same shared-sweep
    promotion that settles any single-output kernel's, applied now that the sibling's stores are no
    longer in the intersection."""
    piece = TileOp(
        op=_region_term(regions, tail, tuple(dict.fromkeys(value for store in stores for value in store.write.values))),
        # The seam token keeps recursive pieces' kernel names distinct, as it does for a workspace
        # producer — two same-named pieces from different cut levels would launch one kernel twice.
        name=f"{tile.name}__place_{digest(tile.identity_key(structural=False) or '', spelling)[:10]}",
        place=Placement(free=tuple(tile.place.free)),
        axes=tile.axes,
        output_specs=stores,
        placement_decided=placement_decided,
        split_consumed=split_consumed,
    )
    return replace(reformed(piece), knobs=consume_kernel_row(piece.knobs))


def _read_name(name: str, token: str, ordinal: int | None = None) -> str:
    """The SSA name a workspace read binds: the cone's own result name, tagged with the seam whose
    workspace it now comes from (and, for a clustered duplicate, its ordinal in the cluster).

    A lowered body reads PRODUCER names throughout — a consumer's params are spelled as the result
    names of the edge they bind (:attr:`~emmy.compiler.ir.pure.fold.Fold.applied`) — so an edge's
    result name is what the emitted kernel declares, and the cone's own name is NOT unique among
    what one kernel binds. The value a cut materializes can still be computed in place beside the
    read: a structurally equal cone the replacement did not reach (replacement follows object
    sharing), or a second seam exposing that same value. Under one name those are two declarations
    at two different addresses, which is an SSA fault and which nvcc rejects (*already declared in
    the current scope*). Reads of ONE workspace at one address keep one name, so the emitted body
    still binds each value once. What a reader derives from the read carries the tag on
    (:func:`_follow_reads`).
    """
    return f"{name}__ws{token}" if ordinal is None else f"{name}__ws{token}s{ordinal}"


def _producer_order(pieces) -> list:
    """Topologically order cut producers by the workspaces their stored Fold reads.

    Containment can make one produced piece read another even when the corresponding seams have
    no provider requirement. Dependency COUNT is not an order: two pieces may each read one
    workspace while one of those workspaces is produced by the other piece.
    """
    workspaces = {buffer for *_, buffers in pieces for buffer in buffers}
    remaining = list(pieces)
    ordered = []
    available: set[str] = set()
    while remaining:
        ready = [piece for piece in remaining if _buffer_reads(piece[1]) & workspaces <= available]
        assert ready, "strict Fold containment makes the cut-workspace dependency graph acyclic"
        ordered.extend(ready)
        available.update(buffer for *_, buffers in ready for buffer in buffers)
        remaining = [piece for piece in remaining if not any(piece is chosen for chosen in ready)]
    return ordered


def realize(
    match: Match,
    root: Node,
    seams,
    *,
    placement_decided: bool = False,
) -> Graph:
    """Build the cut fragment for ``seams`` — one piece per seam plus the ONE sibling piece that
    reads what they produced. A single seam is the two-kernel cut; several seams are one COMPOSED
    placement decision (a pinned compile consumes every scoped PLACE pin that resolves on this
    kernel at once, so the pieces stay decided and the knob row records every spelling).

    A frontier seam cuts at the cone's storage waypoint: the piece computes the encode prefix, the
    workspace holds the raw bits, and the sibling keeps the decode + factor residue as its operand
    cone (which normalization then binds as a raw storage-dtype load with the factors hoisted onto
    the accumulator epilogue). An OUTPUT-OWNING seam writes the kernel's own outputs instead of a
    workspace, and the sibling keeps the rest; the two kinds compose, because a workspace cut nested
    under an output-owning region is applied to that region's term like any other consumer's.

    ``placement_decided`` consumes an authoritative pinned PLACE restriction on every piece.
    Unpinned cuts leave it false so fresh pieces can expose and decide smaller seams. A placement
    cut never erases an earlier cross-CTA decision: every piece inherits the parent's explicit or
    sliced-axis split receipt."""
    tile: TileOp = root.op
    split_consumed = tile.split_consumed or carries_partition(tile)
    owning = tuple(seam for seam in seams if seam.owned is not None)
    seams = tuple(seam for seam in seams if seam.owned is None)
    pieces = []
    # What each replaced cone's result is called once the consumer reads it back, and what every
    # value derived from such a read is called after it (:func:`_follow_reads`) — the renames this
    # pass mints, collected as they are minted. A term's readers follow them for free (a
    # consumer's params are spelled as the result names of the edge they bind), but the kernel's
    # boundary stores are NOT part of the term: ``TileOp.output_specs`` names the stored value as
    # a plain string, so a store of a renamed value has to be re-spelled here or it names a value
    # the consumer no longer defines.
    read_names: dict[str, str] = {}
    taken = _kept_components(tile)
    for seam in seams:
        child = seam.node
        front = seam.frontier
        # A workspace holds the components the consumer READS. The piece still folds every
        # channel — a reduce carries its accumulators together — but a component no reader takes
        # is not stored, so no launch is left loading a buffer nothing wrote for it.
        # One workspace serves the representative AND every clustered sibling, so a component any
        # occurrence reads is kept for all of them.
        shared = set(taken.get(id(child), set(child.exposes)))
        for sibling, _, channels in seam.siblings:
            read = taken.get(id(sibling), set(sibling.exposes))
            shared.update(child.exposes[channel] for position, channel in enumerate(channels) if sibling.exposes[position] in read)
        # ONE workspace per DISTINCT component, not one per position. A carrier seats a component
        # once per reader, so a value two readers share is exposed at SEVERAL positions naming the
        # one accumulator (flash's numerator, read straight and again through its normalizing
        # wrapper). Fused lowering collapses those onto that one SSA value; a workspace keyed by
        # position instead declares the accumulator's storage once per position and emits a
        # redeclaration no compiler accepts. The first position owns the buffer and the rest read it.
        owner = {name: position for position, name in reversed(list(enumerate(child.exposes))) if name in shared}
        slots = tuple(owner.get(name) == position for position, name in enumerate(child.exposes))
        wanted = tuple(name for position, name in enumerate(child.exposes) if slots[position])
        if front is not None:
            names = (front.name,)
            produced = front.producer
            dtypes = seam.dtypes
        else:
            names = wanted
            produced = child
            dtypes = tuple(dtype for dtype, keep in zip(seam.dtypes, slots, strict=True) if keep)
        axes = _workspace_axes(seam, produced)
        strides = _workspace_strides(produced, axes)
        axes = tuple(replace(axis, extent=axis.extent.ceil_div(strides[axis.name])) if axis.name in strides else axis for axis in axes)
        index = tuple(Var(axis.name) / strides[axis.name] if axis.name in strides else Var(axis.name) for axis in axes)
        token = digest(tile.identity_key(structural=False) or "", seam.spelling)[:10]
        # The ordinal names the component the workspace holds, so a narrowed seam keeps the
        # spelling of the components it did keep.
        ordinals = range(len(names)) if front is not None else (i for i, keep in enumerate(slots) if keep)
        buffers = tuple(f"{root.id}__place_{token}_{i}" for i in ordinals)

        # SLABS, not bare Loads: these replace an operand edge, and an operand is a term. The
        # workspace read declares the seam axes it indexes, exactly as any other gmem read does.
        # Positional over what the edge exposed, ``None`` where the reader took nothing. A
        # frontier's workspace is the one raw waypoint, which the block below spells instead.
        by_name = dict(zip(wanted, buffers, strict=True))
        held = {} if front is not None else {position: by_name[name] for position, name in enumerate(child.exposes) if name in by_name}
        loads: tuple = tuple(
            Fold.slab(Load(name=_read_name(name, token), input=held[position], index=index)) if position in held else None
            for position, name in enumerate(child.exposes)
        )
        if front is not None:
            # The raw storage read at the frontier's dtype stays INLINE under its decode residue —
            # the storage-decode cone the operand readers recognize (a raw ``b8`` fill), not a
            # projection over a slab. The residue is a lambda over that read, so tagging what it
            # EXPOSES renames its defining statements in lockstep and leaves the read's own
            # internal spelling alone.
            raw = Load(name=names[0], input=buffers[0], index=index, dtype=front.dtype)
            operands = tuple(edge for edge in child.operands if set(edge.exposes) & Body(front.residue).ssa_uses)
            params = tuple(name for edge in operands for name in edge.exposes)
            residue = Lambda.closing(params, Body((raw, *front.residue)), child.applied.results)
            loads = (Fold(operands=operands, lift=residue.rename({name: _read_name(name, token) for name in residue.results})),)
        # The names the consumer reads this workspace back under. A frontier seam's workspace holds
        # the raw storage waypoint, so its piece is named after the FRONTIER while the consumer
        # still exposes the cone's decoded results — the rename is over those.
        read_names.update({name: _read_name(name, token) for name in (names if front is None else child.lift.results)})
        replacements = {id(child): loads}
        for ordinal, (sibling, pairs, channels) in enumerate(seam.siblings):
            # A clustered duplicate reads the SAME workspace, spelled through its own captured
            # axes via the correspondence the clustering proved — and under its own read names,
            # since it reads that workspace at a DIFFERENT address than the representative. It
            # reads the component that is its value: a lone contraction reads one channel of the
            # twin it equals.
            mapping = {name: Var(other) for name, other in pairs}
            mapping.update({axis.name: Literal(0, "int") for axis in axes if _unit(axis) and axis.name not in mapping})
            sibling_index = tuple(expr.substitute(mapping) for expr in index)
            replacements[id(sibling)] = tuple(
                Fold.slab(Load(name=_read_name(own, token, ordinal), input=held[channel], index=sibling_index)) if channel in held else None
                for own, channel in zip(sibling.exposes, channels, strict=True)
            )
            # The representative wins a shared name: a boundary store of a value both occurrences
            # expose reads the term's own, and only the representative sits on the term's path.
            for name, channel in zip(sibling.exposes, channels, strict=True):
                if channel in held:
                    read_names.setdefault(name, _read_name(name, token, ordinal))
        pieces.append((replace(seam, dtypes=dtypes), produced, axes, strides, token, names, buffers, replacements))

    # Every replacement applies to the consumer AND to every OTHER seam's produced piece: a
    # composed decision may cut a cone nested inside another seam's value (attention's statistics
    # cone contains the score dots whose operand cones are cut beside it), and that producer must
    # read the workspace like any other consumer. Containment is strict, so order is free.
    everything = {target: loads for *_, replacements in pieces for target, loads in replacements.items()}
    parent_fold = _replace_fold(tile.op, everything, read_names)
    specs = tuple(
        replace(store, write=replace(store.write, values=tuple(read_names.get(value, value) for value in store.write.values)))
        for store in tile.output_specs
    )
    produced_pieces = []
    for seam, produced, axes, strides, token, names, buffers, replacements in pieces:
        others = {target: loads for target, loads in everything.items() if target not in replacements}
        # A piece that reads another seam's workspace re-spells what it derives from it too, and
        # its own stores name those values.
        derived: dict[str, str] = {}
        produced = _replace_fold(produced, others, derived) if others else produced
        if strides:
            produced = rewrite_stmt(produced, lambda name: name, Sigma({name: Var(name) * stride for name, stride in strides.items()}))
        index = tuple(Var(axis.name) for axis in axes)
        produced_pieces.append((seam, produced, axes, index, token, tuple(derived.get(name, name) for name in names), buffers))

    fragment = _input_fragment(match, root)
    all_buffers = [buffer for *_, buffers in produced_pieces for buffer in buffers]
    # A producer reading another seam's workspace must follow the node that writes it. Strict
    # containment makes this dependency graph acyclic, including chains whose members have the
    # same number of direct workspace reads.
    for seam, produced, axes, index, token, names, buffers in _producer_order(produced_pieces):
        # The workspace keeps the shape the sibling already reads it back at, so the split below
        # moves the GRID and nothing else.
        shape = tuple(axis.extent for axis in axes)
        produced, grid, index, minted = _split_fused_pair(produced, axes, index)
        producer = TileOp(
            op=produced,
            # The seam token keeps recursive pieces' kernel names distinct — the one-name-one-source
            # launch rule stated beside ``nvcc.load_cubin_function``: two same-named producers from
            # different cut levels would launch one kernel twice.
            name=f"{tile.name}__place_{token}",
            # The workspace axes are the store's SWEEP, and the one rank rule
            # (``promoted_sweep``, applied by ``TileOp.__post_init__``) binds the ones binding
            # replicates nothing over. A free axis per workspace dimension instead bound the sweep a
            # row statistic is invariant in, so the piece re-folded it once per output CELL: a
            # materialized q/k RoPE cone launched one cooperative block per element.
            place=Placement(free=()),
            axes=(
                *(next((axis for axis in grid if axis.name == original.name), original) for original in tile.axes),
                *minted,
            ),
            output_specs=tuple(
                OutputSpec(Write(output=buffer, index=index, value=name), sweep=grid) for name, buffer in zip(names, buffers, strict=True)
            ),
            placement_decided=placement_decided,
            split_consumed=split_consumed,
        )
        producer = replace(reformed(producer), knobs=consume_kernel_row(producer.knobs))
        workspace_tensors = tuple(Tensor(name=buffer, shape=shape, dtype=dtype) for buffer, dtype in zip(buffers, seam.dtypes, strict=True))
        reads = _buffer_reads(produced)
        fragment.add_node(
            op=producer,
            inputs=_piece_inputs(root, produced, tuple(buffer for buffer in all_buffers if buffer in reads)),
            outputs=workspace_tensors,
            node_id=buffers[0],
        )

    # A bare reduction carries NO output specification — its grid-cell store is materializer glue —
    # so the sibling's ports are read off the graph node, not off the stores.
    consumer_fold, consumer_stores = parent_fold, specs
    consumer_outputs = set(root.buffer_names())
    if owning:
        chosen = {index: seam for index, edge in enumerate(tile.op.operands) for seam in owning if seam.node is edge}
        # The ownership partition, re-derived over the REPLACED root: a workspace cut composed with
        # these swapped a cone for a slab under one of the regions, and the piece has to carry that
        # swap. With no such seam the replacement is the identity and this is the same call the
        # offer made, so the two agree by construction; a composed one replaces operand edges and
        # leaves the root body alone, which is what the partition reads. Either way it is positional
        # over the operands, so the regions line up with the seams that chose them.
        regions = output_regions(parent_fold, specs)
        order = [store.write.output for store in specs]
        for index, seam in chosen.items():
            region, tail, stores = regions[index]
            stores = _in_source_order(stores, order)
            piece = _region_piece(tile, (region,), tail, stores, placement_decided, split_consumed, seam.spelling)
            reads = _buffer_reads(piece.op)
            add_output_piece(
                match,
                fragment,
                output_root(root, {store.write.output for store in stores}),
                piece,
                _piece_inputs(root, piece.op, tuple(buffer for buffer in all_buffers if buffer in reads)),
                suffix=_PLACED,
            )
        kept = [regions[index] for index in range(len(regions)) if index not in chosen]
        consumer_stores = _in_source_order(tuple(store for _, _, stores in kept for store in stores), order)
        # A composed decision may hand every output away; then there is no sibling piece to emit.
        if not consumer_stores:
            return fragment
        consumer_outputs = {store.write.output for store in consumer_stores}
        consumer_fold = _region_term(
            tuple(region for region, _, _ in kept),
            Body(tuple(stmt for _, tail, _ in kept for stmt in tail)),
            tuple(dict.fromkeys(value for store in consumer_stores for value in store.write.values)),
        )

    consumer = TileOp(
        op=consumer_fold,
        name=tile.name,
        place=tile.place,
        axes=tile.axes,
        # ``specs`` already reads each stored value under the name the cut left it;
        # ``add_output_piece`` re-spells the BUFFER each write targets.
        output_specs=consumer_stores,
        placement_decided=placement_decided,
        split_consumed=split_consumed,
    )
    consumer = replace(reformed(consumer), knobs=consume_kernel_row(consumer.knobs))
    add_output_piece(
        match,
        fragment,
        output_root(root, consumer_outputs),
        consumer,
        _piece_inputs(root, consumer_fold, tuple(all_buffers)),
        suffix=_PLACED,
    )
    return fragment


__all__ = ["CutSite", "Frontier", "cuttable_seams", "full_projection_seams", "output_map", "realize", "storage_frontier"]
