"""Every per-choice legality rule of the classic model, asked of one value at a time: what a catalog value must
pass to be offered, and what a parsed row value must pass to be selected — one predicate, two callers. The
join-side rules (a plan against its placed geometry, a stage against its resolver, the fragment seams) live
here too, so the context composes without owning legality."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING

from frozendict import frozendict

from emmy.compiler.ir.address import gmem_axis_step, split_addressable
from emmy.compiler.ir.atom import ATOM_REGISTRY, AtomKind, atoms_for, wide_accumulate
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.schedule.catalog import (
    WARP_LANES,
    coop_reduce_moves,
    scalar_tile_moves,
    stage_moves,
    warp_tile_in_catalog,
    warp_tile_moves,
)
from emmy.compiler.ir.schedule.choices import PlacedTile, Reduce, ResolvedStage, Stage, Tile, Work
from emmy.compiler.ir.schedule.packing import block_scaled_atom
from emmy.compiler.ir.schedule.views import ContractionFacts, NodeId
from emmy.compiler.ir.stmt import Assign, Body, Load, Loop, Select, Write, mask_select_predicate
from emmy.compiler.ir.stmt.passes import has_contraction_tail

from .schedule import NodeSchedule, binds_root, node_id_spelling, packed_works

if TYPE_CHECKING:
    from emmy.compiler.ir.tile import TileOp


def _inner_free(tile: TileOp):
    """Return the innermost non-unit free axis, if one exists."""
    return next(
        (axis for axis in reversed(tile.place.free) if not (axis.extent.is_static and axis.extent.as_static() == 1)),
        None,
    )


def _transposed_reduction_ok(tile: TileOp) -> bool:
    """Whether this kernel has the structure required by a transposed cooperative band."""
    from emmy.compiler.ir.tile.ops import projection_tail  # noqa: PLC0415 — tile.ops reads this package; module level would cycle

    tail = projection_tail(tile)
    return _inner_free(tile) is not None and not any(isinstance(stmt, Loop) for stmt in tail) and not has_contraction_tail(tail)


def _reduction_domain(tile: TileOp, node, target=None) -> tuple[Reduce, ...]:
    """Project one plain reduction's legal choices from node and kernel facts only.

    The catalog is not capped by the axis extent: an over-wide band is legal and idles its extra
    lanes. Keeping it in the site's own factor lets a row select one instead of manufacturing a
    pin-only choice outside Algorithm 1.

    Shared by the contraction per-cell tier through :func:`_contraction_reductions`, and
    deliberately so: a contraction is a monoid with a ⊗ lift, so it inherits the same swept /
    streamed serial-only exclusions and the same transposed exclusion, with no carve-out of its own.
    """
    from emmy.compiler.ir.tile.ops import (  # noqa: PLC0415 — tile.ops reads this package; module level would cycle
        chain_form,
        chain_members,
        kernel_roots,
        merges_partition,
    )

    roots = kernel_roots(tile.op)
    is_root = any(node is root for root in roots)
    owner = node if is_root else next((root for root in roots if any(node is member for member in chain_members(root))), None)
    if node.carries or owner is None:
        return (Reduce(),)  # the binder partitions the roots it peels and their chain members; any other reduce lowers serially
    # A sweep the member's own ROOT is evaluated over wraps the whole chain, members included, and
    # only the serial fold spells that: the chain arm closes one grid cell.
    if {axis.name for spec in tile.output_specs for axis in spec.sweep} & (node.free_axes | owner.free_axes):
        return (Reduce(),)
    if is_root and merges_partition(tile):
        # A split's deferred finalize: one partial per split per cell, the parallelism is the cells,
        # and a band over the few partials pays a barrier per cell.
        return (Reduce(),)
    if node.observe is not None:
        # A prefix scan keeps every lane's inclusive state. Ordinary register partials,
        # transposed bands and cross-warp trees do not preserve those states.
        if not is_root or chain_form(node):
            return (Reduce(),)
        return (
            Reduce(),
            *(choice for choice in coop_reduce_moves() if 1 < choice.coop <= WARP_LANES and choice.reg == 1 and not choice.coop_transposed),
        )
    transposed_ok = _transposed_reduction_ok(tile) and is_root
    lanes = (32, 8) if target is not None and target.compute_capability == (7, 0) else (32,)
    return (
        Reduce(),
        *(
            choice
            for choice in coop_reduce_moves(transposed_lanes=lanes)
            if not choice.coop_transposed or (choice.coop % WARP_LANES == 0 and transposed_ok)
        ),
    )


def _contraction_reductions(tile: TileOp, node, facts: ContractionFacts, target=None) -> tuple[Reduce, ...]:
    """The per-cell tier's reductions of a contraction: the plain-reduction catalog, since a
    contraction is a monoid with a ⊗ lift and inherits the same serial-only exclusions with no
    carve-out of its own; the serial fold alone over a symbolic contraction extent."""
    return _reduction_domain(tile, node, target) if facts.k_axis.extent.is_static else (Reduce(),)


def _fragment_projection(tile: TileOp) -> tuple[list, frozenset[str]]:
    """The lowered work outside the computed roots, including nested sibling projections."""
    from emmy.compiler.ir.tile.ops import kernel_roots, refused_roots  # noqa: PLC0415 — tile.ops reads this package

    roots = () if refused_roots(tile.op, tuple(tile.output_specs)) else kernel_roots(tile.op)
    bound = frozenset(axis.name for axis in tile.place.free)
    computed = {stmt for root in roots for stmt in root.lower(bound, axes=tile.axes)}
    tail = [stmt for stmt in tile.op.lower(bound, tuple(tile.output_specs), tile.axes) if stmt not in computed]
    return tail, frozenset(name for root in roots for name in root.exposes)


def _fragment_epilogue_ok(tail: list, states: frozenset[str]) -> bool:
    """Whether every output is a straight-line projection of a Fold state."""
    definitions: set[str] = set()
    for stmt in tail:
        if isinstance(stmt, Loop):
            return False
        if isinstance(stmt, Load) and {name for index in stmt.index for name in index.free_vars()} & definitions:
            return False
        definitions.update(stmt.defines())
    body = Body(tail)
    return all(body.backward_cone(stmt.values).external_reads & states for stmt in tail if isinstance(stmt, Write))


def _channel_dtype(tile: TileOp, node, target):
    """Return the one tensor-core dtype shared by the contraction's streamed operands.

    The operand tuple past the shared first edge — a channel was never more than a position in it.
    """
    from emmy.compiler.ir.tile.ops import edge_dtypes  # noqa: PLC0415 — tile.ops reads this package; module level would cycle

    dtypes = {edge_dtypes(edge, tile.inputs)[0] for edge in node.operands[1:]}
    if len(dtypes) == 1:
        return next(iter(dtypes))
    eligible = {dtype for dtype in dtypes if dtype is not None and atoms_for(dtype, ctx=target)}
    return next(iter(eligible)) if len(eligible) == 1 else None


def _fragment_rows_refusal(tile: TileOp) -> str | None:
    """Why the M rows of a C fragment would have nowhere to land.

    The grid's second-to-last free axis carries them, and each row needs its own address in the store. An axis no
    output index mentions gives them one address between them: a plain store keeps the last row, a split's atomic
    finalize sums all of them into that cell. One row is the exception — a matvec with a reshaped output stores
    that way, the overhang guard predicating the rest. The case this rules out is a cross-CTA split's partition
    axis standing in for an M that a size-one extent left without an axis of its own.
    """
    axis = tile.place.free[-2]
    extent = getattr(axis.extent, "value", getattr(axis.extent, "extent", None))
    if extent == 1 or not tile.output_specs:
        # No specs is no statement: an authored tile carries the outputs alone, and the store this reads is the
        # lowering's. One row needs no dimension of its own.
        return None
    if any(axis.name in index.free_vars() for spec in tile.output_specs for index in spec.write.index):
        return None
    return f"no output dimension carries the fragment's M rows ({axis.name})"


def _node_refusal(tile: TileOp, target, node, fragment_epilogue: bool, packed: tuple = (None, None)) -> str | None:
    """Return why static node facts rule out every tensor-core atom."""
    from emmy.compiler.ir.tile.ops import edge_dtypes  # noqa: PLC0415 — tile.ops reads this package; module level would cycle

    view = node.as_contraction()
    if view is None or (view.product.name, view.plus.name) != ("multiply", "add"):
        return "the mma atom realizes only the (multiply, add) semiring instance"
    if not tile.inputs:
        return "no typed inputs expose operand dtypes"
    if len(tile.place.free) < 2:
        return "the grid supplies no output-axis pair for a fragment"
    if (why := _fragment_rows_refusal(tile)) is not None:
        return why
    if not fragment_epilogue:
        return "the projection epilogue is not a per-fragment straight-line program"
    # The operand tuple in stored order — there is no named A/B role any more, and a nested
    # scheduling site on ANY operand refuses the same way.
    if any(edge.axis is not None for edge in node.operands) and not node.chunked():
        return "a nested scheduling site inhabits an operand edge"
    # A CHUNKED carrier is the one exception: its A IS the score contraction, so A reduces. That is
    # the tier's own shape — the chunk's score is the producer's tile — and the fragment agreement
    # composed in ``extend`` is what holds the two to one atom. It only reached here as a zero-axis
    # cone with the contraction nested under it while the fusion still minted that cone, so the
    # blanket refusal never saw the case it was not written about.
    if node.chunked() and (why := _chunk_refusal(tile, node)) is not None:
        return why

    a_edge = node.operands[0]
    dtype = edge_dtypes(a_edge, tile.inputs)[0]
    if (pair := packed[1]) is not None:
        # The block-scaled cell's own three demands. Each states its reason here rather than
        # dropping the tier where the atom list is built, so a node that misses one says which.
        if any(operand.bits is None for operand in pair.b):
            return "a packed-pair channel computes its codes; the block-scaled cell loads its B fragments from a buffer"
        weights = {tile.inputs[operand.bits.input].dtype for operand in pair.b}
        if len(weights) != 1:
            return "the packed-pair channels store their codes at several dtypes; one cell takes one multiplicand dtype"
        weight = next(iter(weights))
        return None if atoms_for(weight, ctx=target) else f"no tensor-core atom takes a {weight} multiplicand on this target"
    if dtype is not None and dtype.logical_elems != 1:
        return f"a packed {dtype} A pairs with no packed peer; no atom multiplies packed codes against decoded ones"
    if dtype is not None and dtype.nbytes == 1:
        if a_edge.as_slab() is None:
            return "fp8 fragment loads require a materialized A edge"
        if _channel_dtype(tile, node, target) != dtype:
            return "fp8 fragment loads require one matching operand dtype"
        if not atoms_for(dtype, ctx=target):
            return f"no tensor-core atom takes a {dtype} multiplicand on this target"
        return None

    atom_dtype = dtype if atoms_for(dtype, ctx=target) else _channel_dtype(tile, node, target)
    if atom_dtype is None:
        return "no operand dtype selects a tensor-core atom family"
    if atom_dtype.nbytes == 1 and atom_dtype != dtype:
        return "a demoting compute fill cannot produce an fp8 fragment"
    if not (atoms_for(atom_dtype, ctx=target) or atoms_for(atom_dtype, acc=atom_dtype, ctx=target)):
        return f"no tensor-core atom takes a {atom_dtype} multiplicand on this target"
    return None


def _chunk_refusal(tile: TileOp, node) -> str | None:
    """Return why the CHUNK tier cannot fold this twisted carrier, whatever atom is offered.

    Stated at the enumeration, not at the binder: a row nothing realizes costs the greedy a
    blocklist retry per rank, and there are more ranked rows than the retry budget."""
    from emmy.compiler.ir.tile.ops import projection_tail  # noqa: PLC0415 — tile.ops reads this package; module level would cycle

    facts = tile.contractions.get(tile.node_id(node))
    score = facts.producer if facts is not None else None
    # The chunk's score is CONTRACTED into its fragments when a nested contraction supplies it, and
    # GATHERED into them when the carrier's own A edge is already the stored tile (softmax@V, whose
    # probabilities arrive as an input). Either way the tier gets a ``(row, chunk)`` C fragment; a
    # carrier that is neither has no chunk to fold.
    reads = (*score.operands, node.operands[1]) if score is not None else node.operands[:2]
    if score is None and node.operands[0].as_slab() is None:
        return "the chunk tier folds a carrier whose pivot a nested contraction or a stored tile supplies"
    # The chunk covers the score's OWN contraction in one pass and holds a query fragment per step,
    # so a symbolic extent there has no step count to hold them at.
    if score is not None and not tile.axis_of(score.axis).extent.is_static:
        return "the chunk tier covers the score's contraction in one pass, so its extent must be static"
    if any(edge.as_slab() is None for edge in reads):
        return "the chunk tier reads its score operands and its streamed value as slabs"
    # The score's own PREFIX is the CARRIER's lift cut to its score role — A is the score
    # contraction, and what scales its raw accumulator lives in the lift above it. Its leaves past
    # the producer are read once ahead of the chunk, so none of them may vary over the chunk.
    prefix = node.applied.cone(node.roles[0])
    if any(not isinstance(stmt, (Assign, Load, Select)) for stmt in prefix.body):
        return "the score's own prefix holds more than a straight-line program"
    coord_axes = {node.axis, *node.as_contraction().left_axes}
    uniform = {name for edge in node.operands if not edge.free_axes for name in edge.exposes}
    for stmt in (stmt for stmt in prefix.body if isinstance(stmt, Select)):
        consumers = [
            consumer
            for consumer in prefix.body
            if isinstance(consumer, Assign) and consumer.op.name == "add" and stmt.name in consumer.args
        ]
        if (
            mask_select_predicate(stmt) is None
            or len(consumers) != 1
            or not {branch.value for branch in stmt.branches} <= uniform
            or any(not branch.select.free_vars() <= coord_axes for branch in stmt.branches)
        ):
            return "the score's coordinate Select does not form a cell-uniform additive mask"
    leaves = [edge for edge in node.operands[1:] if set(edge.exposes) & set(prefix.params)]
    if any(node.axis in edge.free_axes for edge in leaves):
        return "the score's prefix reads an operand that varies over the chunk"
    # The tier holds ONE accumulator — the expectation — and every other carried state as a per-row
    # register. A projection may read those registers, and a cross-CTA split's partial stores each
    # of them WHOLE to its workspace (broadcast per row into a fragment); what the tier cannot write
    # is a per-row state computed into an output of its own beside the expectation.
    tail = projection_tail(tile)
    body = Body(tail)
    states = set(node.base.results)
    cell = {axis.name for axis in tile.place.free}
    expectation = node.base.results[node.bilinear_channels()[0][0]]
    for write in (stmt for stmt in tail if isinstance(stmt, Write)):
        if set(write.values) <= states and cell <= {name for index in write.index for name in index.free_vars()}:
            continue  # a carried state stored WHOLE per cell — a split partial's workspace write, broadcast per row
        reads = set(write.values) | set(body.backward_cone(tuple(write.values)).external_reads)
        if expectation not in reads:
            return "the chunk tier writes its expectation; a carried state beside it has no output of its own"
    return None


def _split_store_refusal(tail: list, free: tuple, atom_shape: tuple[int, int, int], shapes: dict) -> str | None:
    """Return why an atom cannot address a projection-tail load or store."""
    roles = [(free[-1].name, atom_shape[1], "n", True)]
    if len(free) >= 2:
        roles.append((free[-2].name, atom_shape[0], "m", False))
    for stmt in tail:
        if not isinstance(stmt, (Load, Write)):
            continue
        buffer = stmt.input if isinstance(stmt, Load) else stmt.output
        shape = getattr(shapes.get(buffer), "shape", None)
        for name, extent, role, trailing in roles:
            if not split_addressable(stmt.index, shape, name, extent, trailing):
                return f"warp TILE: the {role} axis reaches {buffer} through an unsupported split dimension"
    return None


def _atom_refusal(
    atom: AtomKind,
    a_dtype,
    a_step,
    a_is_load: bool,
    tail: list,
    free: tuple,
    shapes: dict,
) -> str | None:
    """Return why one otherwise available atom cannot bind this node."""
    converting = a_is_load and a_dtype is not None and a_dtype.nbytes >= 2 and a_dtype != atom.operand_dtype("a")
    if a_is_load and not converting and (a_step is None or a_step[0] != 1 or (a_step[1] and a_step[1] % atom.atom_k)):
        motion = "unknown" if a_step is None else f"{a_step[0]} elements per column"
        return (
            f"warp TILE: A fragment loaders read {atom.atom_k} contraction columns CONTIGUOUSLY, "
            f"but this operand's gmem index moves {motion}"
        )
    return _split_store_refusal(tail, free, atom.shape, shapes)


def _atom_families(tile: TileOp, target, node, tail: list, packed: tuple = (None, None)) -> tuple[str, ...]:
    """Project every tensor-core atom allowed by static node and target facts."""
    from emmy.compiler.ir.tile.ops import edge_dtypes  # noqa: PLC0415 — tile.ops reads this package; module level would cycle

    a_edge = node.operands[0]
    dtype = edge_dtypes(a_edge, tile.inputs)[0]
    a_is_load = a_edge.as_slab() is not None
    a_step = gmem_axis_step(a_edge.as_slab().load, node.axis, tile.inputs) if a_is_load else None
    shapes = {**tile.inputs, **tile.outputs}

    def bindable(names: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            name for name in names if _atom_refusal(ATOM_REGISTRY[name], dtype, a_step, a_is_load, tail, tile.place.free, shapes) is None
        )

    # The CHUNK tier hands its weight to the expectation's mma as a register repack of the score's
    # own C fragments, so only an atom whose two lane maps line up can carry it. Both accumulators
    # are offered: the cell named here is the EXPECTATION's, whose chunk partial promotes into the
    # f32 carrier once per chunk — the reduced one therefore runs that chain at the full consumer-die
    # rate without moving the softmax statistics off f32 (the score keeps ``wide_accumulate``).
    if node.chunked():
        # Scores accumulate at full width; the streamed value determines the repacked multiplicand width.
        dtype = _channel_dtype(tile, node, target)
        offered = bindable((*atoms_for(dtype, ctx=target), *atoms_for(dtype, acc=dtype, ctx=target)))
        return tuple(dict.fromkeys(name for name in offered if ATOM_REGISTRY[name].c_to_a_repack))
    if (pair := packed[1]) is not None:
        # ``_node_refusal`` already proved the channels share one stored code dtype that this
        # target has a cell for; the cell addresses its own operands, so no atom refusal applies.
        return atoms_for(tile.inputs[pair.b[0].bits.input].dtype, ctx=target)
    if dtype is not None and dtype.nbytes == 1:
        return bindable(atoms_for(dtype, ctx=target))
    atom_dtype = dtype if atoms_for(dtype, ctx=target) else _channel_dtype(tile, node, target)
    base = bindable(atoms_for(atom_dtype, ctx=target))
    reduced_acc = bindable(atoms_for(atom_dtype, acc=atom_dtype, ctx=target))
    return tuple(dict.fromkeys((*base, *reduced_acc)))


def _warp_atoms(tile: TileOp, target, node) -> tuple[str, ...]:
    """Project tensor-core atoms from contraction, dtype, address, and target facts."""
    tail, states = _fragment_projection(tile)
    packed = tile.packed_reading(node)
    if _node_refusal(tile, target, node, _fragment_epilogue_ok(tail, states), packed) is not None:
        return ()
    return _atom_families(tile, target, node, tail, packed)


@cache
def _scalar_catalog() -> frozenset[Tile]:
    """The scalar tile catalog as a set — what a parsed scalar spelling is checked against, so a
    row can select a catalog value, never manufacture one."""
    return frozenset(scalar_tile_moves())


def _atom_policy_ok(atom: AtomKind, *, allow_f16_accumulate: bool, allow_fp8: bool) -> bool:
    """The precision policy of UNPINNED enumeration: an f16-accumulate or FP8 atom is offered
    only where the caller allowed it. A row that names such a tile bypasses this — an authored
    value is a legal independent choice — so the policy filters the catalog, never a parse."""
    a = atom.operand_dtype("a")
    if a.logical_elems == 1 and a.nbytes == 1 and not allow_fp8:
        return False
    return atom.operand_dtype("c").nbytes != 2 or allow_f16_accumulate


def _warp_plan_ok(node, facts: ContractionFacts, plan: Tile) -> bool:
    if _kstep_refusal(facts.k_axis, plan) is not None or _wgmma_refusal(plan) is not None:
        return False
    chunk = plan.atom.atom_k * plan.bk
    return not node.chunked() or (chunk >= plan.atom.atom_n and chunk % plan.atom.atom_n == 0)


def _uniform_extras(node) -> bool:
    # The scalar register tier replicates the TERM's own step per cell, so a recipe folds there
    # like any other algebra — three states under their own ops, seeded by the ⊕'s identities.
    # Product channels read their streamed edges per register column. Any remaining operand is
    # read once ahead of the cells, so it must be uniform across the tile.
    streamed = {id(edge) for _, edge in node.bilinear_channels()}
    return len(node.operands) >= 2 and not any(edge.free_axes for edge in node.operands[1:] if id(edge) not in streamed)


def _warp_plans(node, facts: ContractionFacts, atoms: tuple[str, ...]) -> Iterator[Tile]:
    """The warp tier's catalog over ``atoms``: the bounded grid, then the wide plan per atom."""
    for plan in warp_tile_moves(atoms):
        if _warp_plan_ok(node, facts, plan):
            yield plan
    for name in atoms:
        plan = Tile(atom=ATOM_REGISTRY[name], regs=(26, 4), bk=2)
        if _warp_plan_ok(node, facts, plan):
            yield plan


def _contraction_plans(node, facts: ContractionFacts, atoms: tuple[str, ...]) -> Iterator[Tile]:
    """Every contraction tile plan the static facts allow, in catalog order: the scalar tier,
    then the warp tier over ``atoms``."""
    yield from scalar_tile_moves() if _uniform_extras(node) else (Tile(),)
    yield from _warp_plans(node, facts, atoms)


def _contraction_plan_refusal(node, facts: ContractionFacts, atoms: tuple[str, ...], plan: Tile) -> str | None:
    """Why one parsed contraction plan is not a value the catalog would have offered, or ``None``."""
    if not plan.is_warp:
        if _uniform_extras(node):
            return None if plan in _scalar_catalog() else "not a scalar tile of the catalog"
        return None if plan == Tile() else "this contraction takes no scalar register tile"
    if plan.atom.name not in atoms:
        return f"atom {plan.atom.name} is not offered here (offered: {', '.join(atoms) or 'none'})"
    if (why := _kstep_refusal(facts.k_axis, plan) or _wgmma_refusal(plan)) is not None:
        return why
    if not _warp_plan_ok(node, facts, plan):
        return f"the chunk width {plan.atom.atom_k * plan.bk} is not a multiple of the atom's N ({plan.atom.atom_n})"
    if not (warp_tile_in_catalog(plan) or (plan.regs == (26, 4) and plan.bk == 2)):
        return "not a warp tile of the catalog"
    return None


def fill_stage_moves() -> tuple[Stage, ...]:
    """The depths the smem fill tier names: single-buffer, and the peer prefetch ring.

    Stated once because it is asked twice — the edge catalog offers these, and the local support
    join re-checks that the stage it resolves is one of them. The two must agree: when they did
    not, the catalog's value was silently overruled by the join's and a widened ladder read as no
    change at all.

    Deeper rings are deliberately absent. The peers already fly asynchronously at depth 2
    (``resolve_fill_stage`` puts every gmem-read operand on ``cp.async``), so a third and fourth
    slot buy no overlap and cost the occupancy their smem takes: measured on the frozen Qwen3
    gate/up contraction, depth 3 and 4 run 1.38x and 2.22x slower than depth 2 on an A100 and
    monotonically slower on an H100. The ladder is what the shape can use, not what the codec can
    spell."""
    return (Stage(depth=1), Stage(depth=2))


def fill_tma_moves(ctx) -> tuple[Stage, ...]:
    """The compute fill's stages with its stored peers on TMA instead of cp.async: the fill's own
    depths, with and without the ``/p2`` register double buffer. The fill still evaluates every
    computed slab, single-buffer; only the stored slabs ring as box copies.

    Asked twice, like :func:`fill_stage_moves`: the edge catalog offers these, and the local
    support join admits exactly them."""
    depths = {stage.depth for stage in fill_stage_moves()}
    return tuple(stage for stage in stage_moves(warp=True, ctx=ctx) if stage.transport == "smem-tma" and stage.depth in depths)


def _stage_candidates(tile: TileOp, target, node, choice: NodeSchedule) -> tuple[Stage, ...]:
    """The transports one node choice can be fed by — the independent edge catalog."""
    direct = Stage.direct()
    if not choice.tile.is_tiled or (node.chunked() and not choice.tile.is_warp):
        # A chunked carrier's transport belongs to its own tier, which is the tensor-core one.
        # Its per-cell fallback folds the recipe in registers and reads no slab, so a stage
        # there would name a fill nothing performs.
        return (direct,)
    if _multi_channel_cell(tile, node, choice.tile):
        # The cell's copy transports: cp.async or TMA copies every stored slab, one codes and one
        # scale slab per channel, and beside cp.async a compute fill writes the one computed slab
        # (an activation's own codes) inside that stage (``staging._block_scaled_warp_stage``). Its
        # operands are decode CONES, so the generic reading below would send it to the compute
        # fill, which has no per-channel spelling of the pair; neither has the gmem-direct path.
        return tuple(stage for stage in stage_moves(warp=True, ctx=target) if stage.is_async)
    if _needs_fill(tile, node, choice.tile):
        candidates: tuple[Stage, ...] = fill_stage_moves()
        if tile.packed_reading(node)[0] is not None:
            candidates = (*candidates, *stage_moves(warp=True, ctx=target))
        else:
            candidates = (*candidates, *fill_tma_moves(target))
    elif _multi_fold_direct_refusal(node, choice.tile, direct) is not None:
        # Staged, a multi-channel fold is ordinary — one A slab beside one B per channel, the
        # operand list the compute fill already builds and the copy transports now build too, read
        # by the one drain. So both staging tiers are offered and only ``direct`` is withheld.
        candidates = (*fill_stage_moves(), *stage_moves(warp=True, ctx=target))
    else:
        candidates = (direct, *stage_moves(warp=choice.tile.is_warp, ctx=target))
    # Scalar multi-channel staging uses the blocking fill; the byte-copy catalogs below
    # keep their single-channel scalar geometry.
    if len(node.bilinear_channels()) > 1 and not choice.tile.is_warp:
        candidates = tuple(stage for stage in candidates if stage.transport not in ("smem-async", "smem-tma"))
    if not (choice.tile.is_warp and choice.tile.atom.is_wgmma):
        # An 8-deep ring has paid only on wgmma (the H100 down projection; depth 6 lost to 4 and 8).
        # Elsewhere it only multiplies the stage space every compile prices.
        candidates = tuple(stage for stage in candidates if stage.depth < 8)
    return candidates


def _multi_fold_direct_refusal(node: Fold, plan: Tile, stage: Stage) -> str | None:
    """Why a term folding several channels cannot run gmem-direct under ``plan`` — or ``None``.

    The mma leaf folds ONE B straight out of registers, so a node folding several channels has no
    direct form; staged it is ordinary, one A slab beside one B per channel. Stated once because
    it is asked twice: the edge catalog withholds ``direct`` and the local-support join re-checks
    the stage it was handed."""
    if plan.is_warp and stage.is_direct and len(node.bilinear_channels()) > 1:
        return "the gmem-direct mma leaf folds a single B; a multi-channel fold has to stage"
    return None


def _compute_filled(tile_op, node: Fold, plan: Tile) -> bool:
    """Whether this node's ``smem`` transport is the COMPUTE fill rather than a blocking byte copy.

    The fill is the only tier that evaluates a producer cone into its slab, and the only one that
    deposits through the drain's own threads — so a computed operand has no other spelling. A
    multi-channel fold reaches it too, beside the copy transports: both build the same operand list
    (one A slab, one B per channel), and only the fill can also compute one of them.
    :func:`_staged` materializes every ``smem`` stage through that fill, so the two agree."""
    return _needs_fill(tile_op, node, plan) or (plan.is_warp and len(node.bilinear_channels()) > 1)


def _computed_edge(node: Fold) -> bool:
    """Whether any operand is a computed cone rather than a gmem read.

    ``as_slab``, not ``axis is None``: a slab ITERATES (its coordinates are its own axes) and
    simply does not reduce, so the old test — written when a materialized edge was a bare ``Load``
    and only a cone was a Fold — now calls every operand computed.
    """
    return any(edge.as_slab() is None for edge in node.operands)


def _multi_channel_cell(tile_op, node: Fold, plan: Tile) -> bool:
    """Whether ``plan`` lowers ``node`` as the block-scaled cell over SEVERAL channels: a warp plan
    whose atom multiplies packed pairs, on a node that reads as one pair with more than one weight
    channel (:func:`~emmy.compiler.ir.schedule.packing.match_packed_pair_node` — a fused gate/up
    edge over one quantized activation). The reading is asked per plan because a 16-bit atom on
    the same node keeps the single-sided shape and its compute fill; only the pair-multiplying
    atom stages the pair. A single-channel pair keeps the general path, which offers it the same
    cell beside the compute fill."""
    return (
        plan.is_warp and len(node.bilinear_channels()) > 1 and tile_op.packed_reading(node)[1] is not None and block_scaled_atom(plan.atom)
    )


def _needs_fill(tile_op, node: Fold, plan: Tile) -> bool:
    from emmy.compiler.ir.schedule import staging  # noqa: PLC0415

    if _multi_channel_cell(tile_op, node, plan):
        # The cell's stage copies and fills its slabs itself (``staging._block_scaled_warp_stage``);
        # its operands are decode cones, so the generic reading below would send it to the compute
        # fill and the fill's cover and dtype rules would refuse every plan.
        return False
    if node.chunked():
        # The chunk tier's A is the WEIGHT, which never leaves registers: it is what the chunk's
        # own score fragments repack into. There is no operand to fill and no slab to fill it from.
        return False
    # Only a COMPUTED operand forces the fill: a copy transport moves bytes and cannot evaluate a
    # producer cone. A multi-channel node whose weights are all materialized is not that — it
    # stages one A slab beside one B per channel, which is the same operand list the fill builds
    # and the same order the drain reads (the gate/up pair on sm_80 and sm_90).
    return plan.is_warp and (
        _computed_edge(node)
        or staging.converting_a(node, plan.atom, tile_op.inputs)
        or any(not staging.copied_b(node, edge, tile_op.inputs) for edge in node.operands[1:])
    )


def _kstep_refusal(k_axis, plan: Tile) -> str | None:
    if not (plan.is_warp and plan.atom.operand_dtype("a").nbytes == 1):
        return None
    if not k_axis.extent.is_static:
        return f"atom {plan.atom.name}: fp8 fragment loads require a static K"
    step = plan.atom.atom_k * plan.bk
    extent = k_axis.extent.as_static()
    return None if extent % step == 0 else f"warp TILE K-step {step} does not divide the static contraction K={extent}"


def _wgmma_refusal(plan: Tile, stage: Stage | None = None) -> str | None:
    """Why a warp-group cell cannot run under ``plan`` — and under ``stage``, once the operand
    transport is known — or ``None``. The ONE statement of the wgmma legality rules: the catalog
    filter and the compatibility join drop a row through it, the pin path raises its message."""
    if not (plan.is_warp and plan.atom.is_wgmma):
        return None
    atom = plan.atom
    if plan.units_n != 1 or plan.units_m % 4:
        return "wgmma needs a w<4k>x1 warp grid: four M-adjacent warps issue one instruction"
    if plan.reg_m != 1 or plan.reg_n % atom.cells_per_instruction:
        return "wgmma issues whole m64nN instructions: the fragment grid must be f1x<C> with C a multiple of N/8"
    if atom.atom_k * plan.bk * atom.operand_dtype("a").nbytes != 128:
        return "wgmma reads one 128-byte swizzle row per descriptor: the K chunk must be 64 elements (k4)"
    if stage is not None and stage.is_direct:
        return "wgmma reads its operands through shared-memory descriptors: a direct stage cannot feed it"
    return None


#: Registers a ``wgmma`` lane keeps live beside its accumulators: descriptors, addresses, the ring
#: counters. ptxas asked for 154 at an m64n192 (96 accumulator registers) and refused the 128 a
#: sixteen-warp CTA leaves each lane.
_WGMMA_LIVE_REGISTERS = 58


def _wgmma_register_refusal(node: Fold, plan: Tile) -> str | None:
    """Why a warp-group row's accumulators cannot fit its CTA's register envelope, or ``None``. A
    ``wgmma`` holds the whole instruction's accumulator in registers at once, so a row past the
    envelope does not spill: ptxas refuses the kernel."""
    if not (plan.is_warp and plan.atom.is_wgmma):
        return None
    from emmy.compiler.ir.schedule.catalog import MAX_REGISTERS_PER_CTA, MAX_REGISTERS_PER_THREAD  # noqa: PLC0415

    channels = max(1, len(node.bilinear_channels()))
    required = channels * plan.reg_m * plan.reg_n * plan.atom.accumulator_registers_per_lane + _WGMMA_LIVE_REGISTERS
    available = min(MAX_REGISTERS_PER_THREAD, MAX_REGISTERS_PER_CTA // plan.block_threads)
    if required <= available:
        return None
    return (
        f"wgmma accumulators need about {required} registers/thread, over the {available}-register envelope "
        f"at {plan.block_threads} threads/CTA"
    )


def _plan_node_refusal(tile_op, node: Fold, plan: Tile, placed: PlacedTile, facts: ContractionFacts) -> str | None:
    from emmy.compiler.ir.schedule import staging  # noqa: PLC0415

    refusal = _kstep_refusal(facts.k_axis, plan) or _wgmma_refusal(plan) or _wgmma_register_refusal(node, plan)
    if refusal is not None or not _needs_fill(tile_op, node, plan):
        return refusal
    converting = staging.converting_a(node, plan.atom, tile_op.inputs)
    return staging.computed_operand_cover(
        node, placed, converting=converting, k_axis=facts.k_axis, inputs=tile_op.inputs
    ) or staging.computed_operand_copy_dtype(
        node,
        placed,
        tile_op.inputs,
        converting=converting,
    )


def _resolve_stage(
    tile_op,
    target,
    node: Fold,
    plan: Tile,
    placed: PlacedTile,
    choice: Stage,
    facts: ContractionFacts,
) -> ResolvedStage | None:
    from emmy.compiler.ir.schedule import staging  # noqa: PLC0415

    if node.chunked() and not plan.is_warp:
        return None  # the chunked carrier's per-cell fallback reads no slab (``_edge_domain``)
    packed = tile_op.packed_reading(node)
    packed_copy = packed[0] is not None and choice.transport in ("smem-async", "smem-tma")
    # A packed weight has its own copy stage; any other computed operand keeps the compute fill,
    # whose stored peers may ride TMA instead of cp.async.
    fill_tma = choice.transport == "smem-tma" and not packed_copy and _needs_fill(tile_op, node, plan)
    if (choice.transport == "smem" and _compute_filled(tile_op, node, plan) and not packed_copy) or fill_tma:
        return staging.resolve_fill_stage(
            node,
            placed,
            target.max_dynamic_smem,
            choice,
            inputs=tile_op.inputs,
            seam=facts.seam,
            k_axis=facts.k_axis,
            producer=facts.producer,
            producer_k=tile_op.axis_of(facts.producer.axis) if facts.producer is not None else None,
            axes=tile_op.axes,
        )
    if plan.is_warp:
        return staging.resolve_warp_stage(
            node,
            placed,
            choice,
            target.max_dynamic_smem,
            tile_op.inputs,
            readings=packed,
            k_axis=facts.k_axis,
            producer=facts.producer,
            producer_k=tile_op.axis_of(facts.producer.axis) if facts.producer is not None else None,
        )
    return staging.resolve_scalar_stage(node, placed, choice, tile_op.inputs, target.max_dynamic_smem, facts.k_axis, tile_op.axes)


@dataclass(frozen=True)
class _AxisAgreement:
    """One physical-axis geometry claim carried by a local schedule offer."""

    name: str
    tile: int
    units: int


@dataclass(frozen=True)
class _FragmentAgreement:
    """One producer or consumer claim at a fragment seam."""

    role: str
    edge: str
    value: tuple

    def __post_init__(self) -> None:
        if self.role not in ("need", "offer"):
            raise ValueError(f"fragment agreement role must be need or offer, got {self.role!r}")


@dataclass(frozen=True)
class _Relation:
    """What a prefix has decided that the compatibility of a later pick reads — the worker inventory it
    claimed, its physical-axis and fragment-seam agreements, and the decided nodes where a rule reads them:
    every one while no inventory is claimed (the canonical-form check runs over them), and all of them at a
    site that is a shared root or a chain member. Two prefixes equal here admit the same picks at a site,
    whatever else they decided, which is what lets a site keep one filtered answer per relation."""

    work: Work | None = None
    axes: Mapping[str, tuple[int, int]] = field(default_factory=frozendict)
    fragments: Mapping[tuple[str, str], tuple] = field(default_factory=frozendict)
    nodes: Mapping[NodeId, NodeSchedule] = field(default_factory=frozendict)

    def __post_init__(self) -> None:
        for name in ("axes", "fragments", "nodes"):
            object.__setattr__(self, name, frozendict(getattr(self, name)))


def _relation_refusal(site: NodeId, pick, relation: _Relation, *, roots, pairs, allowed_works) -> str | None:
    """Why ``pick`` cannot extend a prefix whose compatibility facts are ``relation``, or ``None``.

    ``pick`` is a site's tile choice or one of its supports; both carry ``node``, ``work``, ``axes`` and
    ``fragments``, and the rules read nothing else. The shared-root and chain rules (``roots``, ``pairs``: the
    kernel's) are the binder's dispatch applied at the offer; the inventory, canonical-form and physical-axis
    rules read the tile; the seam rule reads the claims, which only a support completes with its transport's K
    slab — so a choice passing here may still have no support that does."""
    node = pick.node
    if site in roots and binds_root(node) and any(binds_root(relation.nodes[other]) for other in roots if other in relation.nodes):
        return "a second scheduled root on a projection its outputs do not partition by root"
    for root, member in pairs:
        if site not in (root, member):
            continue
        picks = {**relation.nodes, site: node}
        if root not in picks or member not in picks or picks[member].reduce == Reduce():
            continue
        if picks[root].tile.is_tiled:
            return "a partitioned chain member under an output-tiled root, whose fill owns its cone"
        if picks[root].reduce.coop_transposed:
            return "a partitioned chain member under a transposed band, which binds its fold alone"
    work = relation.work
    if work is not None and pick.work is not None and pick.work != work:
        return "pick requires a different worker inventory"
    resolved = pick.work or work
    if (
        allowed_works is not None
        and resolved is not None
        and (resolved.kind, resolved.units) not in allowed_works
        and not any((packed.kind, packed.units) in allowed_works for packed in packed_works(resolved))
    ):
        return "pick cannot reach a kernel allowed by the schedule restriction"
    if work is None and resolved is not None:
        if not all(choice.tile.is_canonical_for(resolved) for choice in (*relation.nodes.values(), node)):
            return "pick is not canonical for the worker inventory"
    elif not node.tile.is_canonical_for(resolved):
        return "pick is not canonical for the worker inventory"
    axis_values = dict(relation.axes)
    for claim in pick.axes:
        value = (claim.tile, claim.units)
        if axis_values.get(claim.name, value) != value:
            return "pick disagrees on physical-axis geometry"
    fragment_values = dict(relation.fragments)
    for claim in pick.fragments:
        key = (claim.role, claim.edge)
        if fragment_values.setdefault(key, claim.value) != claim.value:
            return "pick repeats a fragment endpoint inconsistently"
        other_role = "need" if claim.role == "offer" else "offer"
        other = fragment_values.get((other_role, claim.edge))
        if other is None:
            continue
        need, offer = (claim.value, other) if claim.role == "need" else (other, claim.value)
        if need[0] == "chunk":
            rows, keys = offer[5] if offer[0] == "warp" else ((), ())
            if offer[0] != "warp":
                return "the chunk tier's score must be warp-tiled"
            if need[1:3] != offer[1:3]:
                return f"the score's cell {offer[1]} {offer[2]} is not the carrier's {need[1]} {need[2]}"
            if keys[0] != need[5]:  # the producer's N is the carrier's key: a (row, chunk) tile
                return "the score's N tile is not the carrier's key axis"
            if keys[1:3] != (1, need[3]):  # one warp column, and that column IS the chunk
                return f"the score's N tile ({keys[1]} warp columns, {keys[2]} wide) is not the carrier's chunk ({need[3]} wide)"
            if rows[3] != need[4]:  # the same register rows the carrier holds
                return f"the score holds {rows[3]} register rows where the carrier holds {need[4]}"
            compatible = True
        elif offer[0] == "free":
            compatible = need[0] != "step"
        else:
            compatible = (
                need[0] in ("warp", "step")
                and offer[0] == "warp"
                and need[1] == offer[1]
                and need[2] == offer[2]
                and offer[3] == 1
                and offer[4] == need[3]
            )
        if not compatible:
            return "pick is incompatible at a fragment seam"
    return None


def _fragment_agreements(
    site: NodeId,
    node: Fold,
    plan: Tile,
    placed: PlacedTile,
    stage: ResolvedStage | None,
    facts: ContractionFacts,
    producer_sites: frozenset[NodeId],
) -> tuple[_FragmentAgreement, ...]:
    out = []
    if site in producer_sites:
        if not plan.is_tiled:
            offer = ("free",)
        elif plan.is_warp:
            # The last entry names both output sides by AXIS. Which of them a consumer wants is the
            # consumer's question: the ordinary need wants the producer's N, a chunked one wants
            # whichever side carries ITS key, and the term's canonical orientation decides which
            # that is (a score whose A edge is the key tiles the key as M).
            sides = tuple((side.axis.name, side.units, side.tile, side.reg) for side in (placed.m, placed.n))
            offer = ("warp", plan.atom.shape, _seam_layout(plan.atom), placed.n.units, placed.n.tile, sides)
        else:
            offer = ("scalar",)
        out.append(_FragmentAgreement("offer", node_id_spelling(site), offer))
    if facts.need is not None and not (node.chunked() and not plan.is_warp):
        # A chunked carrier the tier does NOT fold — the serial arm — reads its score as a plain
        # value like any reduce and claims nothing at the seam, so the score keeps its own tile.
        if plan.is_warp and node.chunked():
            # A CHUNKED carrier does not merely tolerate a fragment at the seam, it is built on
            # one: the chunk's score IS the producer's tile, so the producer must be warp-tiled at
            # this atom with the chunk as its N tile, one warp column wide and the same register
            # rows. Stated as a need of its own because the ordinary one accepts an untiled
            # producer, and that row would be stamped on a kernel whose emission ignored it.
            need = ("chunk", plan.atom.shape, _seam_layout(plan.atom), plan.atom.atom_k * plan.bk, placed.m.reg, node.axis)
        elif plan.is_warp and stage is not None and stage.transport == "smem":
            need = ("step" if facts.need_step else "warp", plan.atom.shape, plan.atom.fragment_layout, stage.bk_elems)
        else:
            need = ("free",)
        out.append(_FragmentAgreement("need", node_id_spelling(facts.need), need))
    return tuple(out)


def _seam_layout(atom) -> str:
    """The register layout a fragment seam hands over. A ``wgmma`` cell's accumulator is, per warp,
    the ``m16n8k16`` C fragment repeated along N, and its register-A form takes that layout's A
    fragment, so across a seam it IS that layout: a warp-group expectation reads an ``mma.sync``
    score's fragments as they stand."""
    return "m16n8k16" if atom.is_wgmma else atom.fragment_layout


def _fragment_registers(atom, role: str) -> int:
    explicit = atom.fragment_nregs(role)
    if explicit is not None:
        return explicit
    m, n, k = atom.ptx_shape
    dtype = atom.operand_dtype(role)
    if role == "a":
        return m * k * dtype.nbytes // 128
    if role == "b":
        return n * k * dtype.nbytes // 128
    return m * n // (64 if dtype.nbytes == 2 else 32)


def _paired_registers(node: Fold, producer: Fold, placed: PlacedTile, stage: ResolvedStage, partial_columns: int) -> int:
    """The live fragment registers one thread of a paired contraction needs, with ``partial_columns``
    register columns of reduced partials live at once."""
    atom = placed.atom
    a_regs = _fragment_registers(atom, "a")
    b_regs = _fragment_registers(atom, "b")
    cells = placed.reg_m * placed.reg_n
    cell_regs = cells * _fragment_registers(atom, "c")
    if atom.operand_dtype("c").nbytes == 2:
        # A reduced accumulator promotes into an f32 shadow per cell, beside its live partials.
        cell_regs = cells * (atom.atom_m * atom.atom_n // 32) + placed.reg_m * partial_columns * _fragment_registers(atom, "c")
    depth = max(1, stage.reg_depth)
    # C fragment SETS the consumer holds. A fused multi-channel edge keeps one per streamed
    # operand; a TWISTED carrier keeps one — its bilinear channel — beside per-row registers for
    # the states that are no product, so counting its operands claimed fragments it never declares.
    channels = len(node.bilinear_channels()) if node.chunked() else len(node.operands) - 1
    consumer_c = channels * cell_regs
    consumer = placed.reg_m * depth * a_regs + channels * (placed.reg_n * depth * b_regs + cell_regs)
    producer_n = stage.bk_elems // atom.atom_n
    # The producer accumulates at ITS own cell: a chunked consumer on the reduced-accumulate cell
    # still scores in f32 (``wide_accumulate``), so its score tile is no wider for it.
    producer_c = _fragment_registers(wide_accumulate(atom), "c")
    producer_regs = placed.reg_m * a_regs + (len(producer.operands) - 1) * (producer_n * b_regs + placed.reg_m * producer_n * producer_c)
    return max(consumer, consumer_c + producer_regs)


def _register_envelope(placed: PlacedTile) -> int:
    from emmy.compiler.ir.schedule.catalog import MAX_REGISTERS_PER_CTA, MAX_REGISTERS_PER_THREAD  # noqa: PLC0415

    return min(MAX_REGISTERS_PER_THREAD, MAX_REGISTERS_PER_CTA // placed.block_threads)


def chunk_partial_columns(node: Fold, producer: Fold | None, placed: PlacedTile, stage: ResolvedStage | None) -> int:
    """How many register columns of reduced chunk partials a chunked carrier keeps live — the one rule the
    offer and the emitter share. The whole row while the thread's envelope holds it: every key step then drains
    the row at once, which measures faster (14.8 us against 16.6 at head width 128 on an RTX 5090). Past the
    envelope the chunk drains one column pair through all its steps and folds it at once — a pair, because that
    is what one paired ldmatrix fills."""
    row = placed.reg_n
    if not node.chunked() or producer is None or stage is None or stage.bk_elems % placed.atom.atom_n:
        return row
    return row if _paired_registers(node, producer, placed, stage, row) <= _register_envelope(placed) else min(row, 2)


def _paired_budget_refusal(node: Fold, producer: Fold | None, placed: PlacedTile, stage: ResolvedStage | None) -> str | None:
    if not (placed.is_warp and stage is not None and producer is not None):
        return None
    if stage.bk_elems % placed.atom.atom_n:
        return None
    required = _paired_registers(node, producer, placed, stage, chunk_partial_columns(node, producer, placed, stage))
    available = _register_envelope(placed)
    if required <= available:
        return None
    return (
        f"paired contractions require at least {required} live fragment registers/thread, over the "
        f"{available}-register envelope at {placed.block_threads} threads/CTA"
    )
