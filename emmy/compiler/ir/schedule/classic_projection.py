"""Classic schedule domain projection and materialization.

The scheduler has one candidate-space contract: kernel, node, and edge domains are projected
independently from static facts, and enumeration is exactly the compatible subset of their
Cartesian product. Algorithm 1(c, p, t) carries the immutable schedule restriction ``c`` intact
and evaluates it only on complete assignments. Traversal order may change evaluation cost, never
membership.

Projection, plain-reduction, scalar-contraction, precision-gated tensor-core, materialized-operand
copy staging, smem compute-fill staging, and kernel-global raster choices are live. Later schedule
families extend the same independent factors and the one compatibility relation; they do not add
another enumerator.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from functools import cached_property

from frozendict import frozendict

from emmy.compiler.ir.address import gmem_axis_step, split_addressable
from emmy.compiler.ir.atom import ATOM_REGISTRY, AtomKind, atoms_for
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.schedule import (
    PlacedTile,
    Raster,
    Reduce,
    Stage,
    Tile,
    WarpSpec,
    Work,
    derive_inventory,
    resolve_site_tile,
)
from emmy.compiler.ir.schedule.base import Schedule, ScheduleProblem, Site
from emmy.compiler.ir.schedule.catalog import (
    WARP_LANES,
    coop_reduce_moves,
    producer_band_moves,
    raster_moves,
    scalar_tile_moves,
    stage_moves,
    warp_tile_in_catalog,
    warp_tile_moves,
)
from emmy.compiler.ir.schedule.classic import (
    ClassicAssignment,
    ClassicDomains,
    ClassicMaterialization,
    EdgeSchedule,
    KernelSchedule,
    NodeSchedule,
    ProjectionSchedule,
    ReductionSchedule,
    _kstep_refusal,
    _needs_fill,
    _plan_node_refusal,
    _resolve_stage,
    _wgmma_refusal,
    classic_node_key,
    classic_stage_key,
    edge_site_spelling,
    no_site_claims_inventory,
    node_id_spelling,
)
from emmy.compiler.ir.schedule.staging import stage_target
from emmy.compiler.ir.schedule.views import ContractionFacts, NodeId
from emmy.compiler.ir.stmt import Assign, Body, Load, Loop, Select, Write, mask_select_predicate
from emmy.compiler.ir.stmt.passes import has_contraction_tail
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.ops import (
    Sched,
    chain_form,
    chain_members,
    edge_dtypes,
    kernel_roots,
    merges_partition,
    projection_tail,
    scheduled,
)
from emmy.utils import cached_method


class ClassicProjectionError(RuntimeError):
    """One projected site has no locally supported choice on this structural branch."""


def _inner_free(tile: TileOp):
    """Return the innermost non-unit free axis, if one exists."""
    return next(
        (axis for axis in reversed(tile.place.free) if not (axis.extent.is_static and axis.extent.as_static() == 1)),
        None,
    )


def _transposed_reduction_ok(tile: TileOp) -> bool:
    """Whether this kernel has the structure required by a transposed cooperative band."""
    tail = projection_tail(tile)
    return _inner_free(tile) is not None and not any(isinstance(stmt, Loop) for stmt in tail) and not has_contraction_tail(tail)


def _reduction_domain(tile: TileOp, node) -> tuple[Reduce, ...]:
    """Project one plain reduction's legal choices from node and kernel facts only.

    The catalog is not capped by the axis extent: an over-wide band is legal and idles its extra
    lanes. Keeping it in the independent node domain lets ``c`` restrict an existing assignment
    instead of manufacturing a pin-only choice outside Algorithm 1.

    Shared by the contraction per-cell tier through :func:`_contraction_domain`'s delegation, and
    deliberately so: a contraction is a monoid with a ⊗ lift, so it inherits the same swept /
    streamed serial-only exclusions and the same transposed exclusion, with no carve-out of its own.
    """
    roots = kernel_roots(tile.op)
    is_root = any(node is root for root in roots)
    if node.observe is not None or not (is_root or any(node is member for root in roots for member in chain_members(root))):
        return (Reduce(),)  # the binder partitions the roots it peels and their chain members; any other reduce lowers serially
    if {axis.name for spec in tile.output_specs for axis in spec.sweep} & node.free_axes:
        return (Reduce(),)
    if is_root and merges_partition(tile):
        # A split's deferred finalize: one partial per split per cell, the parallelism is the cells,
        # and a band over the few partials pays a barrier per cell.
        return (Reduce(),)
    transposed_ok = _transposed_reduction_ok(tile) and is_root and not chain_form(node)
    return (
        Reduce(),
        *(choice for choice in coop_reduce_moves() if not choice.coop_transposed or (choice.coop % WARP_LANES == 0 and transposed_ok)),
    )


def _fold_states(op) -> frozenset[str]:
    """Return the Fold state names visible to the projection tail."""
    if not isinstance(op, Fold):
        return frozenset()
    # What the term binds into its consumer, at either arity: a reducing fold exposes its carried
    # state, a projection its operands'. ``lift.body`` holds statements only, so the terms below a
    # projection are exactly its operands.
    if op.axis is not None:
        return frozenset(op.exposes)
    return frozenset(name for edge in op.operands for name in edge.exposes)


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
    dtypes = {edge_dtypes(edge, tile.inputs)[0] for edge in node.operands[1:]}
    if len(dtypes) == 1:
        return next(iter(dtypes))
    eligible = {dtype for dtype in dtypes if dtype is not None and atoms_for(dtype, ctx=target)}
    return next(iter(eligible)) if len(eligible) == 1 else None


def _node_refusal(tile: TileOp, target, node, fragment_epilogue: bool, packed: tuple = (None, None)) -> str | None:
    """Return why static node facts rule out every tensor-core atom."""
    view = node.as_contraction()
    if view is None or (view.product.name, view.plus.name) != ("multiply", "add"):
        return "the mma atom realizes only the (multiply, add) semiring instance"
    if not tile.inputs:
        return "no typed inputs expose operand dtypes"
    if len(tile.place.free) < 2:
        return "the grid supplies no output-axis pair for a fragment"
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
            or not set(stmt.deps()) <= uniform
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
        dtype = edge_dtypes(a_edge, tile.inputs)[0]
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
    tail = projection_tail(tile)
    packed = tile.packed_reading(node)
    if _node_refusal(tile, target, node, _fragment_epilogue_ok(tail, _fold_states(tile.op)), packed) is not None:
        return ()
    return _atom_families(tile, target, node, tail, packed)


def _contraction_domain(
    tile: TileOp,
    target,
    node,
    facts: ContractionFacts,
) -> tuple[ReductionSchedule, ...]:
    """Project one contraction's locally realizable scalar and tensor-core choices."""
    per_cell_reductions = _reduction_domain(tile, node) if facts.k_axis.extent.is_static else (Reduce(),)
    allowed_atoms = _warp_atoms(tile, target, node)

    def warp_plan_ok(plan: Tile) -> bool:
        if _kstep_refusal(facts.k_axis, plan) is not None or _wgmma_refusal(plan) is not None:
            return False
        chunk = plan.atom.atom_k * plan.bk
        return not node.chunked() or (chunk >= plan.atom.atom_n and chunk % plan.atom.atom_n == 0)

    wide_warp_tiles = tuple(plan for name in allowed_atoms if warp_plan_ok(plan := Tile(atom=ATOM_REGISTRY[name], regs=(26, 4), bk=2)))
    # The scalar register tier replicates the TERM's own step per cell, so a recipe folds there
    # like any other algebra — three states under their own ops, seeded by the ⊕'s identities.
    # What it has no residence for is an operand past the streamed one that VARIES: those are read
    # once, ahead of the cells, so every one of them must be uniform across the tile (attention's
    # scale and its mask fills are; a second streamed B is not, and rides the warp compute fill).
    uniform_extras = len(node.operands) >= 2 and not any(edge.free_axes for edge in node.operands[2:])
    scalar_tiles = scalar_tile_moves() if uniform_extras else (Tile(),)
    catalog = (
        *scalar_tiles,
        *(plan for plan in warp_tile_moves(allowed_atoms) if warp_plan_ok(plan)),
        *wide_warp_tiles,
    )
    return tuple(
        ReductionSchedule(plan, reduction) for plan in catalog for reduction in (per_cell_reductions if not plan.is_tiled else (Reduce(),))
    )


_SCALAR_CATALOG: frozenset[Tile] | None = None


def _scalar_plan_in_catalog(plan: Tile) -> bool:
    """Whether a parsed scalar spelling names a plan the scalar catalog offers — a row can select
    a catalog value, never manufacture one."""
    global _SCALAR_CATALOG  # noqa: PLW0603 — the catalog is a constant; built once, read per parse
    if _SCALAR_CATALOG is None:
        _SCALAR_CATALOG = frozenset(scalar_tile_moves())
    return plan in _SCALAR_CATALOG


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
    # What it has no residence for is an operand past the streamed one that VARIES: those are read
    # once, ahead of the cells, so every one of them must be uniform across the tile (attention's
    # scale and its mask fills are; a second streamed B is not, and rides the warp compute fill).
    return len(node.operands) >= 2 and not any(edge.free_axes for edge in node.operands[2:])


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


def _contraction_plan_allowed(node, facts: ContractionFacts, atoms: tuple[str, ...], plan: Tile) -> bool:
    """Whether one parsed contraction plan is a value the catalog would have offered."""
    if not plan.is_warp:
        return _scalar_plan_in_catalog(plan) if _uniform_extras(node) else plan == Tile()
    if plan.atom.name not in atoms or not _warp_plan_ok(node, facts, plan):
        return False
    return warp_tile_in_catalog(plan) or (plan.regs == (26, 4) and plan.bk == 2)


def _stage_candidates(tile: TileOp, target, node, choice: NodeSchedule) -> tuple[Stage, ...]:
    """The transports one node choice can be fed by — the independent edge catalog."""
    direct = Stage.direct()
    if not choice.tile.is_tiled or (node.chunked() and not choice.tile.is_warp):
        # A chunked carrier's transport belongs to its own tier, which is the tensor-core one.
        # Its per-cell fallback folds the recipe in registers and reads no slab, so a stage
        # there would name a fill nothing performs.
        return (direct,)
    if _needs_fill(tile, node, choice.tile):
        candidates: tuple[Stage, ...] = (Stage(depth=1), Stage(depth=2))
        if tile.packed_reading(node)[0] is not None:
            candidates = (*candidates, *stage_moves(warp=True, ctx=target))
    else:
        candidates = (direct, *stage_moves(warp=choice.tile.is_warp, ctx=target))
    # The prefetching transports deposit ONE slab per fold, so a term folding several channels has
    # no spelling there whatever tier carries it — the materializer emits a single deposit and then
    # refuses the channel count it was handed. The warp tier states this as "needs the compute
    # fill"; the per-cell tier has no fill to fall back to, so the refusal belongs on the transport.
    if len(node.bilinear_channels()) > 1:
        candidates = tuple(stage for stage in candidates if stage.transport not in ("smem-async", "smem-tma"))
    return candidates


def _select[T](
    named: str | None,
    catalog: Iterator[T] | tuple[T, ...],
    *,
    parse: Callable[[str], T | None],
    allowed: Callable[[T], bool],
    spell: Callable[[T], str],
    bare: str | None,
    validate_pins: bool,
) -> tuple[T, ...]:
    """One factor's values under the row.

    A NAMED value (the row spells this site's exact key) is parsed and checked; that one value is
    the factor. A spelling the parser cannot read alone — a warp tile with no ``WORK`` beside it —
    is matched against the catalog by spelling instead, still one value. A named value the site
    cannot take empties the factor, or, when pins are not validated (a row published across the
    peer kernels of a multi-kernel target), leaves the catalog whole. A BARE pin of an ambiguous
    family names one site among several: this factor keeps the pin's value and OFF, and the
    completed schedule is asked which site carried it."""
    if named is not None:
        value = parse(named)
        if value is not None and allowed(value):
            return (value,)
    values = tuple(catalog)  # the catalog is walked only past the named fast path
    if named is not None:
        matched = tuple(choice for choice in values if spell(choice) == named)
        if matched or validate_pins:
            return matched
    return values if bare is None else tuple(choice for choice in values if spell(choice) in ("", bare))


@dataclass(frozen=True, eq=False)
class ClassicNodeSite(Site[ClassicAssignment]):
    """One node site's independent factor: its node choices, the transport catalog of its incident
    edges, and their product as picks. Every derived read is memoized on the site, so a site's
    tuples keep one identity for the context's caches."""

    problem: ClassicProblem
    id: NodeId

    @cached_property
    def keys(self) -> tuple[str, ...]:
        tile = self.problem.tile
        keys = []
        if self.id in tile.family_sites["TILE"]:
            keys.append(classic_node_key(tile, "TILE", self.id))
        if self.id in tile.family_sites["REDUCE"]:
            keys.append(classic_node_key(tile, "REDUCE", self.id))
        if self.stage_key is not None:
            keys.append(self.stage_key)
        return tuple(keys)

    @cached_property
    def stage_key(self) -> str | None:
        tile = self.problem.tile
        return next((classic_stage_key(tile, edge) for edge in tile.stage_edges if edge[0] == self.id), None)

    @property
    def node(self) -> Fold:
        return self.problem.tile.sites[self.id].node

    def _named(self, family: str) -> str | None:
        """The row's value at this site's ``family`` key, when the row spells that exact key."""
        if self.id not in self.problem.tile.family_sites[family]:
            return None
        return self.problem.row.get(classic_node_key(self.problem.tile, family, self.id))

    def _wgmma_pin_refusal(self, named: str) -> None:
        """Refuse a named warp-group TILE that its WORK, fragment grid, K chunk or STAGE cannot
        feed, with that rule's own message: the catalog never offers such a row, so a silent empty
        site could only report an unsupported pin. Loud whatever the pin reading — the spelling is
        wrong wherever it is published."""
        atom = ATOM_REGISTRY.get(named.partition("/")[0])
        if atom is None or not atom.is_wgmma or not self.problem.loud_pins:
            return
        work = self.problem.work
        plan = Tile.parse(named, work if work is not None and work.kind == "warp" else Work(kind="warp", units=(4, 1)))
        stage = self.problem.row.get(self.stage_key) if self.stage_key is not None else None
        stage = self.problem.row.get("STAGE") if stage is None else stage
        if why := _wgmma_refusal(plan, None if stage is None else Stage.parse(stage)):
            raise ValueError(why)

    def _select_plans(self, named: str | None, catalog, *, allowed) -> tuple[Tile, ...]:
        if named is not None:
            self._wgmma_pin_refusal(named)

        def parse(spelling: str) -> Tile | None:
            try:
                return resolve_site_tile(spelling, self.problem.work)
            except ValueError:
                return None

        return _select(
            named,
            catalog,
            parse=parse,
            allowed=allowed,
            spell=Tile.spell,
            bare=self.problem.bare_value("TILE", self.keys),
            validate_pins=self.problem.validate_pins,
        )

    def _literal(self, family: str, choices: tuple, spell: Callable[[object], str]) -> tuple:
        """A hand-written factor under the row: the same selection the catalog gets, by spelling."""
        if family != "STAGE" and self.id not in self.problem.tile.family_sites[family]:
            return choices
        key = self.stage_key if family == "STAGE" else classic_node_key(self.problem.tile, family, self.id)
        if key is None:
            return choices
        named = self.problem.row.get(key)
        if named is not None:
            kept = tuple(choice for choice in choices if spell(choice) == named)
            choices = kept if kept or self.problem.strict(key) else choices
        bare = self.problem.bare_value(family, self.keys)
        return choices if bare is None else tuple(choice for choice in choices if spell(choice) in ("", bare))

    @cached_property
    def nodes(self) -> tuple[NodeSchedule, ...]:
        """The node choices: the row's value where it names this site, else the catalog."""
        literal = self.problem.domains_literal
        if literal is not None:
            choices = self._literal("TILE", literal.nodes[self.id], lambda choice: choice.tile.spell())
            return self._literal("REDUCE", choices, lambda choice: choice.reduce.spell() if isinstance(choice, ReductionSchedule) else "")
        tile, node = self.problem.tile, self.node
        view = tile.views[self.id]
        if view.axis is None:
            if self.id not in tile.family_sites["TILE"] or not tile.place.free:
                return (ProjectionSchedule(Tile()),)
            inner = tile.place.free[-1]
            extent = inner.extent.as_static() if inner.extent.is_static else 0

            def legal(plan: Tile) -> bool:
                return plan.units == (1, 1) and plan.reg_m == 1 and (plan.reg_n == 1 or (extent and extent % plan.reg_n == 0))

            catalog = tuple(dict.fromkeys(plan for plan in scalar_tile_moves() if legal(plan)))
            return tuple(
                ProjectionSchedule(plan)
                for plan in self._select_plans(self._named("TILE"), catalog, allowed=lambda p: legal(p) and _scalar_plan_in_catalog(p))
            )
        reductions = self._reductions()
        facts = tile.contractions.get(self.id)
        if facts is None:
            choices: Iterator[ReductionSchedule] = (ReductionSchedule(Tile(), reduction) for reduction in reductions)
        else:
            atoms = self.problem.atoms_of(self.id)
            plans = self._select_plans(
                self._named("TILE"),
                _contraction_plans(node, facts, self.problem.policy_atoms(self.id)),
                allowed=lambda plan: _contraction_plan_allowed(node, facts, atoms, plan),
            )
            # A tiled plan folds serially per cell; an untiled one takes every per-cell reduction.
            choices = (
                ReductionSchedule(plan, reduction) for plan in plans for reduction in (reductions if not plan.is_tiled else (Reduce(),))
            )
        return tuple(choice for choice in choices if self._placed_ok(choice))

    def _reductions(self) -> tuple[Reduce, ...]:
        tile, node = self.problem.tile, self.node
        facts = tile.contractions.get(self.id)
        catalog = _reduction_domain(tile, node) if facts is None or facts.k_axis.extent.is_static else (Reduce(),)

        def parse(spelling: str) -> Reduce | None:
            try:
                return Reduce.parse(spelling, self.problem.work)
            except ValueError:
                return None

        return _select(
            self._named("REDUCE"),
            catalog,
            parse=parse,
            allowed=lambda reduction: reduction in catalog,
            spell=Reduce.spell,
            bare=self.problem.bare_value("REDUCE", self.keys),
            validate_pins=self.problem.strict(classic_node_key(tile, "REDUCE", self.id)),
        )

    def _placed_ok(self, choice: NodeSchedule) -> bool:
        tile, node = self.problem.tile, self.node
        geometry = tile.grid_sched.placed(node, choice.tile)
        if choice.tile.is_tiled and not isinstance(geometry, PlacedTile):
            return False
        facts = tile.contractions.get(self.id)
        return not (
            isinstance(geometry, PlacedTile)
            and facts is not None
            and _plan_node_refusal(tile, node, choice.tile, geometry, facts) is not None
        )

    @cached_property
    def warp_eligible(self) -> bool:
        """Whether the CATALOG holds a warp plan for this site — a property of the offered space,
        read the same way whatever the row names, and found without walking the scalar tier."""
        tile, node = self.problem.tile, self.node
        facts = tile.contractions.get(self.id)
        if facts is None or self.problem.domains_literal is not None:
            return any(choice.tile.is_warp for choice in self.nodes)
        return any(self._placed_ok(ReductionSchedule(plan, Reduce())) for plan in _warp_plans(node, facts, self.problem.atoms_of(self.id)))

    @cached_property
    def node_set(self) -> frozenset[NodeSchedule]:
        return frozenset(self.nodes)

    @cached_property
    def edges(self) -> tuple[EdgeSchedule, ...]:
        """The transport choices of every incident edge — one tuple, shared by all of them."""
        literal = self.problem.domains_literal
        incident = self.problem.tile.incident_edges[self.id]
        if literal is not None:
            return self._literal("STAGE", literal.edges[incident[0]], lambda choice: choice.stage.spell()) if incident else ()
        if not incident:
            return ()
        tile, target, node = self.problem.tile, self.problem.target, self.node
        if self.id not in tile.contractions:
            return (EdgeSchedule(Stage.direct()),)
        candidates = {choice: _stage_candidates(tile, target, node, choice) for choice in self.nodes}
        catalog = tuple(dict.fromkeys(EdgeSchedule(stage) for stages in candidates.values() for stage in stages))

        def parse(spelling: str) -> EdgeSchedule | None:
            try:
                stage = Stage.parse(spelling)
            except ValueError:
                return None
            if target is not None and (why := stage_target(stage, target)):
                if self.problem.loud_pins:
                    raise ValueError(why)  # a spelling the card cannot run is wrong wherever it is published
                return None
            return EdgeSchedule(stage)

        return _select(
            None if self.stage_key is None else self.problem.row.get(self.stage_key),
            catalog,
            parse=parse,
            allowed=lambda choice: any(choice.stage in stages for stages in candidates.values()),
            spell=lambda choice: choice.stage.spell(),
            bare=self.problem.bare_value("STAGE", self.keys),
            validate_pins=self.problem.validate_pins,
        )

    @cached_property
    def edge_set(self) -> frozenset[EdgeSchedule]:
        return frozenset(self.edges)

    @cached_property
    def options(self) -> tuple[ClassicAssignment, ...]:
        """The site's picks: each node choice with each transport on every incident edge."""
        incident = self.problem.tile.incident_edges[self.id]
        edge_picks = tuple(frozendict({edge: choice for edge in incident}) for choice in self.edges) if incident else (frozendict(),)
        return tuple(Schedule(None, {self.id: node}, edges) for node in self.nodes for edges in edge_picks)


@dataclass(frozen=True, eq=False)
class ClassicKernelSite(Site[ClassicAssignment]):
    """The kernel-level factor: the worker inventory and raster, spelled bare (``WORK``,
    ``RASTER``). Its catalog is what the node sites' choices imply, so it is the last site."""

    problem: ClassicProblem

    @property
    def keys(self) -> tuple[str, ...]:
        return ("WORK", "RASTER")

    @cached_property
    def kernels(self) -> tuple[KernelSchedule, ...]:
        literal = self.problem.domains_literal
        if literal is not None:
            kernels = literal.kernel
            for key, spell in (("WORK", lambda kernel: kernel.work.spell()), ("RASTER", lambda kernel: kernel.raster.spell())):
                named = self.problem.row.get(key)
                if named is not None:
                    kept = tuple(kernel for kernel in kernels if spell(kernel) == named)
                    kernels = kept if kept or self.problem.strict(key) else kernels
            return kernels
        return tuple(KernelSchedule(work, raster) for work in self._works() for raster in self._rasters())

    def _inventories(self) -> Iterator[Work]:
        for site in self.problem.node_sites:
            for choice in site.nodes:
                work = derive_inventory((choice.tile,), coop=choice.reduce.coop if isinstance(choice, ReductionSchedule) else 1)
                if work is not None:
                    yield work

    def _sweep_widths(self) -> set[Work]:
        # A kernel whose work IS its shared output sweep — a bare elementwise map, the half a
        # placement cut leaves behind a reduction — has no site that folds out an inventory, so the
        # derived domain is the direct per-cell form alone: one worker per output cell with the
        # sweep serial inside it. Offer the widths a cooperative reduction would, so the sweep can
        # be split across workers (``_factor`` distributes it through ``_lane_close``). This widens
        # the worker inventory only; the grid stays the cell count, unlike promoting the axis into
        # ``place.free``, which multiplies the grid by its extent.
        tile = self.problem.tile
        if not (no_site_claims_inventory(tile) and tile.output_specs):
            return set()
        shared = set.intersection(*({axis.name for axis in store.sweep} for store in tile.output_specs))
        if not shared:
            return set()
        return {Work(kind="thread", units=(move.coop, 1)) for move in coop_reduce_moves() if move.coop > 1}

    def _works(self) -> tuple[Work, ...]:
        def catalog() -> tuple[Work, ...]:
            domain = {Work(), *self._inventories(), *self._sweep_widths()}
            return tuple(
                sorted(
                    {
                        Work(kind=work.kind, units=work.units, producer=producer)
                        for work in domain
                        for producer in (producer_band_moves() if work.kind == "warp" else (0,))
                    },
                    key=lambda work: work.spell(),
                )
            )

        def allowed(work: Work) -> bool:
            if work.producer and (work.kind != "warp" or work.producer not in producer_band_moves()):
                return False
            bare = Work(kind=work.kind, units=work.units)
            return bare == Work() or bare in self._sweep_widths() or any(inventory == bare for inventory in self._inventories())

        named = self.problem.row.get("WORK")
        if named is not None:
            work = self.problem.work
            if work is not None and allowed(work):
                return (work,)
            if self.problem.strict("WORK"):
                return ()
        return catalog()

    def _rasters(self) -> tuple[Raster, ...]:
        tile = self.problem.tile
        values = (
            raster_moves()
            if any(view.as_contraction() is not None for view in tile.views) and all(axis.extent.is_static for axis in tile.place.free)
            else ("",)
        )
        named = self.problem.row.get("RASTER")
        if named is not None:
            if named in values:
                return (Raster.parse(named),)
            if self.problem.strict("RASTER"):
                return ()
        # A transposed raster is never the catalog's own offer: it is taken only where a row names
        # it, the reading the restriction gave the ``gn`` values.
        return tuple(Raster.parse(value) for value in values if not value.startswith("gn"))

    @cached_property
    def kernel_set(self) -> frozenset[KernelSchedule]:
        return frozenset(self.kernels)

    @cached_property
    def options(self) -> tuple[ClassicAssignment, ...]:
        return tuple(Schedule(kernel, {}, {}) for kernel in self.kernels)


@dataclass(frozen=True, eq=False)
class ClassicProblem(ScheduleProblem[ClassicAssignment]):
    """``p + t`` and the row: one unscheduled ``TileOp``, its target, and the knob row whose
    values the sites offer where it names them. ``domains_literal`` substitutes hand-written
    factors for the projection — the tests' literal oracle. The precision policy and the pin
    reading (``validate_pins``: a named value the site cannot take empties it, else the site keeps
    its catalog — the reading a row published across peer kernels takes) are the problem's
    parameters, because they change what a site offers."""

    tile: TileOp
    target: object = None
    row: Mapping[str, str] = field(default_factory=frozendict)
    domains_literal: ClassicDomains | None = None
    allow_f16_accumulate: bool = True
    allow_fp8: bool = True
    validate_pins: bool = True
    #: A split's finalize reads a bare WORK / RASTER / REDUCE pin as the partial's: it spells its
    #: reduce serially only and its work at the thread level, so a warp WORK or a band names its
    #: sibling, and the finalize keeps its own catalog instead of offering nothing.
    tolerate_kernel_pins: bool = False
    #: Whether a named value the rules refuse outright — a warp-group tile its grid cannot feed, a
    #: transport the card cannot run, a stage no support resolves — is an error naming the rule.
    #: True for a hand pin, which is wrong wherever it is published; ``with_row`` turns it off, since
    #: a row a descent follows is answered by an empty site and the caller re-decides.
    loud_pins: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "row", frozendict({str(key): str(value) for key, value in self.row.items()}))

    def strict(self, key: str) -> bool:
        """Whether a named value at ``key`` the site cannot take empties the site (else the site
        keeps its catalog): pins are validated, and a bare kernel-family pin is not tolerated."""
        return self.validate_pins and not (self.tolerate_kernel_pins and key in ("WORK", "RASTER", "REDUCE"))

    def with_row(self, row: Mapping[str, str]) -> ClassicProblem:
        return replace(self, row=frozendict({**self.row, **{str(key): str(value) for key, value in row.items()}}), loud_pins=False)

    @cached_property
    def node_sites(self) -> tuple[ClassicNodeSite, ...]:
        return tuple(ClassicNodeSite(self, site) for site in self.tile.node_sites)

    @cached_property
    def kernel_site(self) -> ClassicKernelSite:
        return ClassicKernelSite(self)

    @cached_property
    def sites(self) -> tuple[Site[ClassicAssignment], ...]:
        return (*self.node_sites, self.kernel_site)

    @cached_method
    def node_site(self, site: NodeId) -> ClassicNodeSite:
        return self.node_sites[self.tile.node_sites.index(site)]

    @cached_property
    def work(self) -> Work | None:
        """The row's ``WORK`` inventory, the one every parsed site value decodes against."""
        named = self.row.get("WORK")
        if named is None:
            return None
        try:
            return Work.parse(named)
        except ValueError:
            return None

    @cached_property
    def allowed_works(self) -> frozenset[tuple[str, tuple[int, ...]]] | None:
        """The inventories a kernel may still take when the row names ``WORK`` — what lets a node
        pick that cannot reach it be refused before its subtree."""
        if "WORK" not in self.row:
            return None
        return frozenset((kernel.work.kind, kernel.work.units) for kernel in self.kernel_site.kernels)

    @cached_method
    def atoms_of(self, site: NodeId) -> tuple[str, ...]:
        """The tensor-core atoms the static facts allow at one contraction site."""
        return _warp_atoms(self.tile, self.target, self.tile.sites[site].node)

    @cached_method
    def policy_atoms(self, site: NodeId) -> tuple[str, ...]:
        """The atoms the unpinned catalog offers: :meth:`atoms_of` under the precision policy."""
        return tuple(
            name
            for name in self.atoms_of(site)
            if _atom_policy_ok(ATOM_REGISTRY[name], allow_f16_accumulate=self.allow_f16_accumulate, allow_fp8=self.allow_fp8)
        )

    @cached_property
    def bare_pins(self) -> frozendict[str, str]:
        """The bare pins of ambiguous families: a family key the row spells while this kernel
        spells the family at several sites. Such a pin names ONE site — some site carries the
        value and every other is OFF — so each site offers the value beside OFF, and
        :meth:`unrealized_bare_pin` asks the completed schedule which site carried it."""
        keys = [key for site in self.node_sites for key in site.keys]
        return frozendict(
            {
                family: value
                for family in ("TILE", "REDUCE", "STAGE")
                if (value := self.row.get(family)) is not None and sum(1 for key in keys if key.partition("@")[0] == family) > 1
            }
        )

    def bare_value(self, family: str, keys: tuple[str, ...]) -> str | None:
        """The bare pin that reaches a site spelling ``family`` at a scoped key, or ``None``."""
        value = self.bare_pins.get(family)
        if value is None or not any(key.partition("@")[0] == family for key in keys):
            return None
        return value

    def unrealized_bare_pin(self, assignment: ClassicAssignment) -> str | None:
        """Why a completed schedule leaves a bare pin unrealized — the half of the bare reading
        a site cannot decide alone. A pin no site can offer is ignored unless pins are validated,
        the reading a row published across the peer kernels of a multi-kernel target takes."""
        for family, value in self.bare_pins.items():
            if not value or value in self._spelled(assignment, family).values():
                continue
            if self.validate_pins or any(self._site_offers(site, family, value) for site in self.node_sites):
                return f"bare {family} pin {value} is realized by no site of this kernel"
        return None

    @staticmethod
    def _site_offers(site: ClassicNodeSite, family: str, value: str) -> bool:
        if family == "STAGE":
            return any(choice.stage.spell() == value for choice in site.edges)
        if family == "TILE":
            return any(choice.tile.spell() == value for choice in site.nodes)
        return any(isinstance(choice, ReductionSchedule) and choice.reduce.spell() == value for choice in site.nodes)

    def _spelled(self, assignment: ClassicAssignment, family: str) -> dict[str, str]:
        tile = self.tile
        if family == "TILE":
            return {classic_node_key(tile, "TILE", site): assignment.nodes[site].tile.spell() for site in tile.family_sites["TILE"]}
        if family == "REDUCE":
            return {
                classic_node_key(tile, "REDUCE", site): node.reduce.spell()
                for site in tile.family_sites["REDUCE"]
                if isinstance(node := assignment.nodes[site], ReductionSchedule)
            }
        return {classic_stage_key(tile, edge): assignment.edges[edge].stage.spell() for edge in tile.stage_edges}

    @cached_property
    def warp_eligible(self) -> bool:
        """Whether any site's catalog holds a warp plan — the offered space's own property."""
        return any(site.warp_eligible for site in self.node_sites)

    @cached_property
    def domains(self) -> ClassicDomains:
        """The literal independent factors — the product every enumeration is a subset of.
        Raises :class:`ClassicProjectionError` when a site offers nothing."""
        if self.domains_literal is not None:
            return self.domains_literal
        for site in self.node_sites:
            if not site.nodes or (self.tile.incident_edges[site.id] and not site.edges):
                raise ClassicProjectionError(f"classic site {node_id_spelling(site.id)} has no locally supported choice")
        if not self.kernel_site.kernels:
            raise ClassicProjectionError("classic kernel site has no locally supported choice")
        edges = {}
        for site in self.node_sites:
            edges.update({edge: site.edges for edge in self.tile.incident_edges[site.id]})
        return ClassicDomains(kernel=self.kernel_site.kernels, nodes={site.id: site.nodes for site in self.node_sites}, edges=edges)

    @cached_property
    def bounds(self) -> tuple[int, int]:
        size = descent = len(self.kernel_site.kernels)
        for site in self.node_sites:
            incident = len(self.tile.incident_edges[site.id])
            size *= len(site.nodes) * (len(site.edges) ** incident)
            descent += len(site.nodes) * max(len(site.edges), 1)
        return size, descent


def project_classic(tile: TileOp, target, row: Mapping[str, str] | None = None) -> ClassicDomains:
    """The independent kernel, node, and edge domains of one unscheduled tile: the catalog, or —
    under ``row`` — the row's values at the sites it names."""
    return ClassicProblem(tile, target, row=frozendict(row or {})).domains


def materialize_classic(
    tile: TileOp,
    *,
    name: str,
    knobs: dict,
    target,
    assignment: ClassicAssignment,
) -> TileOp:
    """Materialize one accepted classic assignment into a scheduled TileOp."""
    sched = Sched(tile, place=tile.place.on_grid())
    placed = {}
    resolved = {}
    for site, choice in assignment.nodes.items():
        node = tile.sites[site].node
        geometry = None
        if choice.tile.is_tiled and isinstance(choice, ReductionSchedule):
            geometry = sched.placed(node, choice.tile)
            if not isinstance(geometry, PlacedTile):
                raise ValueError(f"accepted TILE at {node_id_spelling(site)} has no placed geometry")
            placed[site] = geometry
        for edge, edge_choice in assignment.edges.items():
            if edge[0] != site or edge_choice.stage.is_direct:
                continue
            if not isinstance(geometry, PlacedTile):
                raise ValueError(f"accepted STAGE at {edge_site_spelling(edge)} has no placed consumer geometry")
            stage = _resolve_stage(
                tile,
                target,
                node,
                choice.tile,
                geometry,
                edge_choice.stage,
                tile.contractions[site],
            )
            if stage is None:
                raise ValueError(f"accepted STAGE at {edge_site_spelling(edge)} did not resolve")
            resolved[edge] = stage
    return scheduled(
        tile.op,
        name=name,
        place=tile.place.on_grid(),
        knobs=knobs,
        output_specs=tile.output_specs,
        schedule=assignment,
        axes=tile.axes,
        materialization=ClassicMaterialization(placed, resolved),
        workers=WarpSpec(assignment.kernel.work.producer) if assignment.kernel.work.producer else None,
    )


__all__ = [
    "ClassicProjectionError",
    "materialize_classic",
    "project_classic",
]
