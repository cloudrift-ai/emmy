"""The semantic model for the classic grid/CTA/warp/thread/register schedule.

``ClassicScheduleContext`` is the compatibility authority over one unscheduled ``TileOp`` and
target: the prefix ``c``, composing the options its ``ClassicProblem`` (``classic_projection``)
offers site by site. ``ClassicDomains`` is the literal independent product those sites span.
``ClassicScheduleCodec`` and ``ClassicMaterialization`` are the wire and lowering boundaries for
accepted assignments. Search state and pipeline Forks do not belong here.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING

from frozendict import frozendict

from emmy.compiler.ir.atom import wide_accumulate
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.structural import instance_memo

from .base import Schedule, ScheduleContext, ScheduleRefused
from .choices import (
    PlacedTile,
    Raster,
    Reduce,
    ResolvedStage,
    Stage,
    Tile,
    Work,
    derive_inventory,
    resolve_site_tile,
)
from .views import ContractionFacts, EdgeSite, NodeId

if TYPE_CHECKING:
    from .classic_projection import ClassicNodeSite, ClassicProblem

CLASSIC_FAMILIES = ("TILE", "REDUCE", "STAGE")


def node_id_spelling(node_id: NodeId) -> str:
    """Return one node identity's canonical wire spelling."""
    if type(node_id) is not int or node_id < 0:
        raise ValueError(f"node id must be a non-negative integer, got {node_id!r}")
    return f"n{node_id}"


def parse_node_id(value: str) -> NodeId:
    """Parse one canonical node identity."""
    if not value.startswith("n") or not value[1:].isdigit() or str(int(value[1:])) != value[1:]:
        raise ValueError(f"node id must be n<ordinal>, got {value!r}")
    return int(value[1:])


def edge_site_spelling(edge: EdgeSite) -> str:
    """Return one consumer operand position's canonical wire spelling."""
    if not _is_edge_site(edge):
        raise ValueError(f"edge site must be a (node id, operand) pair, got {edge!r}")
    return f"{node_id_spelling(edge[0])}.e{edge[1]}"


def parse_edge_site(value: str) -> EdgeSite:
    """Parse one canonical consumer operand position."""
    node, separator, operand = value.partition(".e")
    if separator != ".e" or not operand.isdigit() or str(int(operand)) != operand:
        raise ValueError(f"edge site must be n<ordinal>.e<operand>, got {value!r}")
    return parse_node_id(node), int(operand)


def _is_node_id(node_id: object) -> bool:
    return type(node_id) is int and node_id >= 0


def _is_edge_site(edge: object) -> bool:
    return isinstance(edge, tuple) and len(edge) == 2 and _is_node_id(edge[0]) and type(edge[1]) is int and edge[1] >= 0


@dataclass(frozen=True)
class KernelSchedule:
    """Kernel-scoped choices."""

    work: Work
    raster: Raster

    def __post_init__(self) -> None:
        if not isinstance(self.work, Work) or not isinstance(self.raster, Raster):
            raise TypeError("classic kernel choices must be Work and Raster values")


@dataclass(frozen=True)
class ProjectionSchedule:
    """The choices of a projection node."""

    tile: Tile

    def __post_init__(self) -> None:
        if not isinstance(self.tile, Tile):
            raise TypeError("classic projection TILE must be an unplaced Tile choice")


@dataclass(frozen=True)
class ReductionSchedule:
    """The choices of a reduction node, including a contraction-capable reduction."""

    tile: Tile
    reduce: Reduce

    def __post_init__(self) -> None:
        if not isinstance(self.tile, Tile):
            raise TypeError("classic reduction TILE must be an unplaced Tile choice")
        if not isinstance(self.reduce, Reduce):
            raise TypeError("classic reduction REDUCE must be a Reduce choice")


NodeSchedule = ProjectionSchedule | ReductionSchedule


@dataclass(frozen=True)
class EdgeSchedule:
    """The transport choice of one operand use."""

    stage: Stage

    def __post_init__(self) -> None:
        if not isinstance(self.stage, Stage):
            raise TypeError("classic edge STAGE must be a Stage choice")


type ClassicAssignment = Schedule[KernelSchedule, NodeSchedule, EdgeSchedule]


def classic_node_key(sites, family: str, site: NodeId) -> str:
    """Return the canonical key for one node-scoped classic family: bare when the family has one
    applicable site on this kernel, else ``FAMILY@<route>`` — the site's route from the root in the
    tree-path grammar (``TILE@map.1/twist.1/inner``), the spelling placement keys use too."""
    family_sites = sites.family_sites.get(family)
    if family_sites is None:
        raise ValueError(f"{family} is not a classic node family")
    if site not in family_sites:
        raise ValueError(f"{sites.sites[site].path} is not a {family} site")
    return family if len(family_sites) == 1 else f"{family}@{sites.sites[site].path}"


def classic_stage_key(sites, edge: EdgeSite) -> str:
    """Return the canonical key for one staged consumer: bare for one consumer, else its route."""
    if edge not in sites.stage_edges:
        raise ValueError(f"{edge_site_spelling(edge)} is not a STAGE edge")
    consumers = tuple(dict.fromkeys(candidate[0] for candidate in sites.stage_edges))
    return "STAGE" if len(consumers) == 1 else f"STAGE@{sites.sites[edge[0]].path}"


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
class _LocalSupport:
    """Static support for one node choice and its incident edge choices.

    The public domains are projections of these records.  A support record is not a schedule:
    placed geometry and fragment facts remain derived compatibility evidence and never enter a
    :class:`Schedule` value.
    """

    node: NodeSchedule
    edges: Mapping[EdgeSite, EdgeSchedule]
    work: Work | None = None
    axes: tuple[_AxisAgreement, ...] = ()
    fragments: tuple[_FragmentAgreement, ...] = ()
    raster_eligible: bool = False
    producer_eligible: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.node, (ProjectionSchedule, ReductionSchedule)):
            raise TypeError("classic local support requires a node schedule")
        if not isinstance(self.edges, Mapping) or any(
            not _is_edge_site(edge) or not isinstance(choice, EdgeSchedule) for edge, choice in self.edges.items()
        ):
            raise TypeError("classic local support edges must map consumer operand pairs to EdgeSchedule")
        if self.work is not None and not isinstance(self.work, Work):
            raise TypeError("classic local support work must be Work or None")
        object.__setattr__(self, "edges", frozendict(self.edges))


def _target_memo(tile_op, target, slot: str) -> dict:
    """The named memo of one kernel's ``p + t`` derivations, riding the tile it derives from.

    Every candidate schedule over one tile shares these tables, and the composition context is
    replaced at each step, so the tile owns them. The target is retained beside its table so the
    id keying it cannot be recycled while the table lives.

    That retention is the tell: this keys a table by ``id(target)`` and stores it on ``tile_op``,
    caching a fact about a PAIR on one member of it. STYLE.md forbids the shape, and it is the
    hardest of the ``instance_memo`` callers to retire — a ``cached_property`` cannot express it,
    so it wants a different owner or a cache threaded through the context.
    """
    table = instance_memo(tile_op, slot)
    if id(target) not in table:
        table[id(target)] = (target, {})
    return table[id(target)][1]


def _computed_edge(node: Fold) -> bool:
    """Whether any operand is a computed cone rather than a gmem read.

    ``as_slab``, not ``axis is None``: a slab ITERATES (its coordinates are its own axes) and
    simply does not reduce, so the old test — written when a materialized edge was a bare ``Load``
    and only a cone was a Fold — now calls every operand computed.
    """
    return any(edge.as_slab() is None for edge in node.operands)


def _needs_fill(tile_op, node: Fold, plan: Tile) -> bool:
    from . import staging  # noqa: PLC0415

    if node.chunked():
        # The chunk tier's A is the WEIGHT, which never leaves registers: it is what the chunk's
        # own score fragments repack into. There is no operand to fill and no slab to fill it from.
        return False
    # CHANNELS, not operand slots. The staged transports fill one slab per fold, so a fold count
    # above one belongs on the smem compute fill — and a B slab reused by several channels occupies
    # ONE operand slot, so counting operands reads a two-channel node as single-fold, offers it
    # cp.async, and the materializer then asserts on the channel count it actually emits (Qwen3-8B
    # decode on sm_80, channels=2 operands=2). The channel count subsumes the operand one.
    return plan.is_warp and (
        _computed_edge(node) or len(node.bilinear_channels()) > 1 or staging.converting_a(node, plan.atom, tile_op.inputs)
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


def _plan_node_refusal(tile_op, node: Fold, plan: Tile, placed: PlacedTile, facts: ContractionFacts) -> str | None:
    from . import staging  # noqa: PLC0415

    refusal = _kstep_refusal(facts.k_axis, plan) or _wgmma_refusal(plan)
    if refusal is not None or not _needs_fill(tile_op, node, plan):
        return refusal
    converting = staging.converting_a(node, plan.atom, tile_op.inputs)
    return staging.computed_operand_cover(node, placed, converting=converting, k_axis=facts.k_axis) or staging.computed_operand_copy_dtype(
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
    from . import staging  # noqa: PLC0415

    if node.chunked() and not plan.is_warp:
        return None  # the chunked carrier's per-cell fallback reads no slab (``_edge_domain``)
    packed = tile_op.packed_reading(node)
    packed_copy = packed[0] is not None and choice.transport in ("smem-async", "smem-tma")
    if _needs_fill(tile_op, node, plan) and not packed_copy:
        return staging.resolve_fill_stage(
            node,
            placed,
            target.max_dynamic_smem,
            choice.depth,
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
    return staging.resolve_scalar_stage(node, placed, choice, tile_op.inputs, target.max_dynamic_smem, facts.k_axis)


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
            offer = ("warp", plan.atom.shape, plan.atom.fragment_layout, placed.n.units, placed.n.tile, sides)
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
            need = ("chunk", plan.atom.shape, plan.atom.fragment_layout, plan.atom.atom_k * plan.bk, placed.m.reg, node.axis)
        elif plan.is_warp and stage is not None and stage.transport == "smem":
            need = ("step" if facts.need_step else "warp", plan.atom.shape, plan.atom.fragment_layout, stage.bk_elems)
        else:
            need = ("free",)
        out.append(_FragmentAgreement("need", node_id_spelling(facts.need), need))
    return tuple(out)


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


def _paired_budget_refusal(node: Fold, producer: Fold | None, placed: PlacedTile, stage: ResolvedStage | None) -> str | None:
    if not (placed.is_warp and stage is not None and producer is not None):
        return None
    from .catalog import MAX_REGISTERS_PER_CTA, MAX_REGISTERS_PER_THREAD  # noqa: PLC0415

    atom = placed.atom
    if stage.bk_elems % atom.atom_n:
        return None
    a_regs = _fragment_registers(atom, "a")
    b_regs = _fragment_registers(atom, "b")
    c_regs = _fragment_registers(atom, "c")
    if atom.operand_dtype("c").nbytes == 2:
        c_regs += atom.atom_m * atom.atom_n // 32
    depth = max(1, stage.reg_depth)
    # C fragment SETS the consumer holds. A fused multi-channel edge keeps one per streamed
    # operand; a TWISTED carrier keeps one — its bilinear channel — beside per-row registers for
    # the states that are no product, so counting its operands claimed fragments it never declares.
    channels = len(node.bilinear_channels()) if node.chunked() else len(node.operands) - 1
    consumer_c = channels * placed.reg_m * placed.reg_n * c_regs
    consumer = placed.reg_m * depth * a_regs + channels * (placed.reg_n * depth * b_regs + placed.reg_m * placed.reg_n * c_regs)
    producer_n = stage.bk_elems // atom.atom_n
    # The producer accumulates at ITS own cell: a chunked consumer on the reduced-accumulate cell
    # still scores in f32 (``wide_accumulate``), so its score tile is no wider for it.
    producer_c = _fragment_registers(wide_accumulate(atom), "c")
    producer_regs = placed.reg_m * a_regs + (len(producer.operands) - 1) * (producer_n * b_regs + placed.reg_m * producer_n * producer_c)
    required = max(consumer, consumer_c + producer_regs)
    available = min(MAX_REGISTERS_PER_THREAD, MAX_REGISTERS_PER_CTA // placed.block_threads)
    if required <= available:
        return None
    return (
        f"paired contractions require at least {required} live fragment registers/thread, over the "
        f"{available}-register envelope at {placed.block_threads} threads/CTA"
    )


@dataclass(frozen=True)
class ClassicDomains:
    """The literal independent kernel, node, and edge factors of Algorithm 1."""

    kernel: tuple[KernelSchedule, ...]
    nodes: Mapping[NodeId, tuple[NodeSchedule, ...]]
    edges: Mapping[EdgeSite, tuple[EdgeSchedule, ...]]

    def __post_init__(self) -> None:
        if not self.kernel or any(not isinstance(choice, KernelSchedule) for choice in self.kernel):
            raise TypeError("classic kernel domain must contain KernelSchedule choices")
        for name, values, site_test, choice_type in (
            ("node", self.nodes, _is_node_id, (ProjectionSchedule, ReductionSchedule)),
            ("edge", self.edges, _is_edge_site, EdgeSchedule),
        ):
            if not isinstance(values, Mapping) or any(not site_test(site) for site in values):
                raise TypeError(f"classic {name} domains have invalid site keys")
            if any(not choices or any(not isinstance(choice, choice_type) for choice in choices) for choices in values.values()):
                raise TypeError(f"classic {name} domains have invalid choices")
        object.__setattr__(self, "nodes", frozendict({site: tuple(choices) for site, choices in self.nodes.items()}))
        object.__setattr__(self, "edges", frozendict({edge: tuple(choices) for edge, choices in self.edges.items()}))

    def __getstate__(self):
        """Pickle declared domains, never derived membership indexes."""
        return {name: self.__dict__[name] for name in self.__dataclass_fields__ if name in self.__dict__}

    @property
    def product_size(self) -> int:
        """Number of assignments in the unfiltered Cartesian product."""
        size = len(self.kernel)
        for choices in (*self.nodes.values(), *self.edges.values()):
            size *= len(choices)
        return size


@dataclass(frozen=True)
class ClassicMaterialization:
    """Placed geometry and resolved transport facts derived from an accepted schedule."""

    tiles: Mapping[NodeId, PlacedTile]
    stages: Mapping[EdgeSite, ResolvedStage]

    def __post_init__(self) -> None:
        if not isinstance(self.tiles, Mapping) or not isinstance(self.stages, Mapping):
            raise TypeError("classic materialization tiles and stages must be mappings")
        if any(not _is_node_id(site) or not isinstance(tile, PlacedTile) for site, tile in self.tiles.items()):
            raise TypeError("classic materialization tiles must map node ids to PlacedTile")
        if any(not _is_edge_site(edge) or not isinstance(stage, ResolvedStage) for edge, stage in self.stages.items()):
            raise TypeError("classic materialization stages must map consumer operand pairs to ResolvedStage")
        object.__setattr__(self, "tiles", frozendict(self.tiles))
        object.__setattr__(self, "stages", frozendict(self.stages))

    def validate(self, schedule: ClassicAssignment, source: object, *, place: object, workers: object) -> None:
        """Validate classic lowering facts against their semantic assignment."""
        if not isinstance(schedule, Schedule):
            raise TypeError("classic materialization requires a Schedule")
        from emmy.compiler.ir.tile import TileOp  # noqa: PLC0415

        if not isinstance(source, TileOp):
            raise TypeError("classic materialization requires a TileOp")
        from emmy.compiler.ir.tile.ops import Sched  # noqa: PLC0415

        context = ClassicScheduleContext(source)
        try:
            context.extend(schedule)
        except ScheduleRefused as error:
            raise ValueError(f"TileOp carries a refused classic schedule: {error}") from error
        source_tile = context.tile_op
        expected_tiles = {
            site
            for site, assignment in schedule.nodes.items()
            if assignment.tile.is_tiled and source_tile.views[site].as_contraction() is not None
        }
        if set(self.tiles) != expected_tiles:
            raise ValueError("classic materialization must contain exactly the tiled node sites")
        expected_stages = {edge for edge, assignment in schedule.edges.items() if not assignment.stage.is_direct}
        if set(self.stages) != expected_stages:
            raise ValueError("classic materialization must contain exactly the staged edge sites")
        placement = Sched(source, place=place)
        for site, placed in self.tiles.items():
            choice = schedule.nodes[site].tile
            expected = placement.placed(source_tile.sites[site].node, choice)
            if placed.choice != choice or placed != expected:
                raise ValueError(f"materialized tile at {node_id_spelling(site)} does not derive from its classic choice")
        for edge, resolved in self.stages.items():
            if edge not in schedule.edges or resolved.choice != schedule.edges[edge].stage:
                raise ValueError(f"materialized stage at {edge_site_spelling(edge)} does not derive from its classic choice")
        producer = workers.producer_warps if workers is not None else 0
        if schedule.kernel.work.producer != producer:
            raise ValueError(f"classic producer band {schedule.kernel.work.producer} disagrees with WarpSpec producer band {producer}")


def no_site_claims_inventory(tile_op) -> bool:
    """Whether this kernel has no node site that could fold out a worker inventory.

    Only a tiled site or a cooperative reduction claims one, so a kernel with neither — a bare
    elementwise map, the half a placement cut leaves behind a reduction — has a kernel work that no
    node constrains. :func:`~...classic_projection.project_classic` offers such a kernel the sweep
    widths, and the two compatibility gates here let them through instead of filtering them back to
    the direct per-cell form. Read off the tile rather than the projected domains, so validation
    (which carries none) answers the same.
    """
    sites = tile_op.family_sites
    return not sites["TILE"] and not sites["REDUCE"]


@dataclass(frozen=True)
class ClassicScheduleContext(ScheduleContext[KernelSchedule, NodeSchedule, EdgeSchedule]):
    """Immutable classic ``c + p + t`` compatibility-composition state.

    The problem ``p`` is the unscheduled ``tile_op`` — its Fold root indexes every site through
    its own site index, and its typed inputs answer every operand-shape question — composed
    against the target ``t``. Derivations shared by every candidate ride memo tables on the tile.
    """

    tile_op: object
    target: object = None
    problem: ClassicProblem | None = None
    order: tuple[NodeId, ...] | None = None
    position: int = 0
    _assignment: ClassicAssignment = field(default_factory=lambda: Schedule(None, {}, {}), repr=False)
    _work: Work | None = field(default=None, repr=False)
    _axes: Mapping[str, tuple[int, int]] = field(default_factory=frozendict, repr=False)
    _fragments: Mapping[tuple[str, str], tuple] = field(default_factory=frozendict, repr=False)
    _raster_eligible: bool = field(default=False, repr=False)
    _producer_eligible: bool = field(default=True, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(getattr(self.tile_op, "op", None), Fold):
            raise TypeError("classic composition requires a TileOp owning a Fold root")
        order = self.tile_op.node_sites if self.order is None else tuple(self.order)
        if len(order) != len(self.tile_op.node_sites) or set(order) != set(self.tile_op.node_sites):
            raise ValueError("classic composition order must contain every node site exactly once")
        if self.problem is not None and self.problem.tile is not self.tile_op:
            raise ValueError("classic problem must be projected from this context's tile")
        if not 0 <= self.position <= len(order):
            raise ValueError("classic composition position is outside its node order")
        object.__setattr__(self, "order", order)
        if not isinstance(self._assignment, Schedule):
            raise TypeError("classic context assignment must be a Schedule")
        object.__setattr__(self, "_axes", frozendict(self._axes))
        object.__setattr__(self, "_fragments", frozendict(self._fragments))

    def _with_problem(self, problem: ClassicProblem) -> ClassicScheduleContext:
        return replace(self, problem=problem)

    def node(self, site: NodeId) -> Fold:
        if type(site) is not int or not 0 <= site < len(self.tile_op.sites):
            raise KeyError(f"unknown node site {site!r}")
        return self.tile_op.sites[site].node

    def site(self, node: Fold) -> NodeId:
        return self.tile_op.node_id(node)

    def operand(self, edge: EdgeSite):
        if not _is_edge_site(edge):
            raise KeyError(f"invalid edge site {edge!r}")
        consumer, operand = edge
        try:
            return self.node(consumer).operands[operand]
        except (TypeError, IndexError):
            raise KeyError(f"unknown classic edge site {edge!r}") from None

    def producer(self, edge: EdgeSite) -> NodeId | None:
        value = self.operand(edge)
        return self.tile_op.node_id(value) if isinstance(value, Fold) else None

    def incident_edges(self, site: NodeId) -> tuple[EdgeSite, ...]:
        try:
            return self.tile_op.incident_edges[site]
        except KeyError:
            raise KeyError(f"unknown node site {site!r}") from None

    def node_key(self, family: str, site: NodeId) -> str:
        return classic_node_key(self.tile_op, family, site)

    def stage_key(self, edge: EdgeSite) -> str:
        return classic_stage_key(self.tile_op, edge)

    def keys(self) -> tuple[str, ...]:
        stage_consumers = tuple(dict.fromkeys(edge[0] for edge in self.tile_op.stage_edges))
        return (
            "WORK",
            "RASTER",
            *(self.node_key("TILE", site) for site in self.tile_op.family_sites["TILE"]),
            *(self.node_key("REDUCE", site) for site in self.tile_op.family_sites["REDUCE"]),
            *(self.stage_key(next(edge for edge in self.tile_op.stage_edges if edge[0] == site)) for site in stage_consumers),
        )

    @property
    def nodes_complete(self) -> bool:
        assert self.order is not None
        return self.position == len(self.order)

    @property
    def next_site(self) -> NodeId | None:
        assert self.order is not None
        return None if self.nodes_complete else self.order[self.position]

    @property
    def assignment(self) -> ClassicAssignment:
        return self._assignment

    def extensions(self) -> Iterator[ClassicAssignment]:
        """Yield the next site's options that compose with this prefix: one node with its
        incident edges, or, past the last node, the kernel picks."""
        if self.assignment.kernel is not None:
            return
        if self.problem is None:
            raise ValueError("classic compatibility composition requires a projected problem")
        if self.nodes_complete:
            for pick in self.problem.kernel_site.options:
                if self._kernel_composes(pick.kernel):
                    yield pick
            return
        assert self.next_site is not None
        site = self.problem.node_site(self.next_site)
        for support in self._compatible_frontier(site):
            yield Schedule(None, {site.id: support.node}, support.edges)

    def _local_frontier(self, site: ClassicNodeSite) -> tuple[_LocalSupport, ...]:
        """The site's options with their local ``p + t`` support derived — once per site object,
        whatever prefix asks: the memo keys on the site, whose option tuple is fixed."""
        cache = _target_memo(self.tile_op, self.target, "_memo_local_frontier")
        key = (site.id, id(site))
        if key in cache:
            return cache[key]
        result = tuple(
            support for pick in site.options if (support := self._local_support(site.id, pick.nodes[site.id], pick.edges)) is not None
        )
        problem = self.problem
        if not result and site.stage_key is not None and problem.loud_pins and problem.validate_pins and problem.row.get(site.stage_key):
            # A hand-pinned non-direct transport no support resolves is a wrong spelling, not an empty pool.
            raise ValueError(f"STAGE pin {problem.row[site.stage_key]!r} does not resolve for this contraction")
        cache[key] = result
        return result

    def _compatible_frontier(self, site: ClassicNodeSite) -> tuple[_LocalSupport, ...]:
        """Filter one local frontier through this exact immutable prefix."""
        frontier = self._local_frontier(site)
        if self._work is not None:
            indexes = _target_memo(self.tile_op, self.target, "_memo_frontier_by_work")
            key = (site.id, id(site))
            if key not in indexes:
                by_work = {}
                for support in frontier:
                    by_work.setdefault(support.work, []).append(support)
                indexes[key] = {work: tuple(supports) for work, supports in by_work.items()}
            frontier = (*indexes[key].get(None, ()), *indexes[key].get(self._work, ()))
        return tuple(support for support in frontier if self._support_refusal(site.id, support) is None)

    def extend(self, pick: ClassicAssignment) -> ClassicScheduleContext:
        """Compose a frontier pick or validate and accept one complete assignment."""
        if not isinstance(pick, Schedule) or self.assignment.kernel is not None:
            self._refuse("classic extension requires an incomplete context and a Schedule pick")
        if pick.kernel is not None and (pick.nodes or pick.edges):
            return self._extend_complete(pick)
        if self.nodes_complete:
            return self._finish(pick)

        return self._extend_local(pick)

    def _extend_complete(self, pick: ClassicAssignment) -> ClassicScheduleContext:
        if any(pick.nodes.get(site) != choice for site, choice in self.assignment.nodes.items()) or any(
            pick.edges.get(edge) != choice for edge, choice in self.assignment.edges.items()
        ):
            self._refuse("complete assignment disagrees with the existing classic prefix")
        self._require_complete_shape(pick)
        context = self._restart()
        assert context.order is not None and pick.kernel is not None
        for site in context.order:
            context = context._extend_local(
                Schedule(None, {site: pick.nodes[site]}, {edge: pick.edges[edge] for edge in context.incident_edges(site)})
            )
        return context._finish(Schedule(pick.kernel, {}, {}))

    def _restart(self) -> ClassicScheduleContext:
        """Return the empty prefix carrying this context's unchanged ``c + p + t``."""
        return replace(
            self,
            position=0,
            _assignment=Schedule(None, {}, {}),
            _work=None,
            _axes=frozendict(),
            _fragments=frozendict(),
            _raster_eligible=False,
            _producer_eligible=True,
        )

    def _extend_local(self, pick: ClassicAssignment) -> ClassicScheduleContext:
        site = self.next_site
        incident = self.incident_edges(site)
        if site is None or pick.kernel is not None or set(pick.nodes) != {site} or set(pick.edges) != set(incident):
            self._refuse("pick is outside the next independent classic position", site)
        node = pick.nodes[site]
        if not isinstance(node, (ProjectionSchedule, ReductionSchedule)) or any(
            not isinstance(choice, EdgeSchedule) for choice in pick.edges.values()
        ):
            self._refuse("pick contains a value from another schedule family", site)
        view = self.tile_op.views[site]
        if view.axis is None and not isinstance(node, ProjectionSchedule):
            self._refuse("projection site requires a projection schedule", site)
        if view.axis is not None and not isinstance(node, ReductionSchedule):
            self._refuse("reduction site requires a reduction schedule", site)
        if isinstance(node.tile, PlacedTile):
            self._refuse("node choices cannot contain placed tile geometry", site)
        if self.problem is not None:
            offered = self.problem.node_site(site)
            if node not in offered.node_set or any(choice not in offered.edge_set for choice in pick.edges.values()):
                self._refuse("pick is outside the next independent classic position", site)
        support = self._local_support(site, node, pick.edges)
        if support is None:
            self._refuse("pick has no local classic support", site)
        if why := self._support_refusal(site, support):
            self._refuse(why, site)
        work = support.work or self._work
        nodes = {**self.assignment.nodes, site: support.node}
        axes = {**self._axes, **{claim.name: (claim.tile, claim.units) for claim in support.axes}}
        fragments = {**self._fragments, **{(claim.role, claim.edge): claim.value for claim in support.fragments}}
        return self._advance(
            position=self.position + 1,
            _assignment=Schedule(None, nodes, {**self.assignment.edges, **support.edges}),
            _work=work,
            _axes=frozendict(axes),
            _fragments=frozendict(fragments),
            _raster_eligible=self._raster_eligible or support.raster_eligible,
            _producer_eligible=self._producer_eligible and support.producer_eligible,
        )

    def _advance(self, **changed) -> ClassicScheduleContext:
        """This context with the fields ONE composition step changes, skipping ``__post_init__``.

        Everything that ``__post_init__`` derives or proves belongs to ``tile_op`` or ``problem`` —
        the node order covering every site exactly once, the problem projected from this tile — and a
        step touches neither, so a step re-derives only conclusions it already carries. Its own remaining checks do not reach a
        step either: the position bound holds because ``_extend_local`` advances only off a
        ``next_site``, and the stage restriction is validated at position 0. The caller passes
        ``_axes`` / ``_fragments`` already frozen, which is the one normalization lost with the
        skipped ``__post_init__``. ``replace`` re-ran all of it once per composition step —
        43.5k times for one SDPA_L schedule walk.

        The public ``extend`` keeps the validating path: a pick decoded from a golden row or handed
        in by a caller has proved none of this."""
        advanced = object.__new__(type(self))
        advanced.__dict__.update(self.__dict__)
        advanced.__dict__.update(changed)
        return advanced

    def _local_support(
        self,
        site: NodeId,
        node: NodeSchedule,
        edges: Mapping[EdgeSite, EdgeSchedule],
    ) -> _LocalSupport | None:
        """Derive the local ``p + t`` facts and decide their compatibility in one place."""
        facts = self.tile_op.contractions.get(site)
        if facts is None:
            return self._intrinsic_support(site, node, edges)
        materialization = getattr(self.tile_op, "materialization", None)
        if self.target is None and (
            materialization is None
            or (node.tile.is_tiled and site not in materialization.tiles)
            or any(not choice.stage.is_direct and edge not in materialization.stages for edge, choice in edges.items())
        ):
            return None
        cache = _target_memo(self.tile_op, self.target, "_memo_classic_local_support")
        key = (site, node, tuple(edges.items()))
        if key in cache:
            return cache[key]
        tile_op = self.tile_op
        fold = self.node(site)
        view = self.tile_op.views[site]
        incident = self.incident_edges(site)
        if set(edges) != set(incident):
            cache[key] = None
            return None
        if len(set(edges.values())) > 1:
            self._refuse("one contraction currently requires one transport choice across its operands", site)
        geometry = tile_op.grid_sched.placed(fold, node.tile)
        if node.tile.is_tiled and not isinstance(geometry, PlacedTile):
            cache[key] = None
            return None
        if isinstance(geometry, PlacedTile):
            if _plan_node_refusal(tile_op, fold, node.tile, geometry, facts) is not None:
                cache[key] = None
                return None
        stage = next(iter(edges.values())).stage if edges else Stage.direct()
        resolved_stage = None
        if _wgmma_refusal(node.tile, stage) is not None:
            cache[key] = None
            return None
        if view.as_contraction() is None or not node.tile.is_tiled:
            if not stage.is_direct:
                cache[key] = None
                return None
        elif self.target is None:
            materialization = getattr(tile_op, "materialization", None)
            resolved = {materialization.stages.get(edge) for edge in edges} if materialization is not None else set()
            resolved.discard(None)
            resolved_stage = next(iter(resolved)) if len(resolved) == 1 else None
        elif _needs_fill(tile_op, fold, node.tile):
            packed_copy = tile_op.packed_reading(fold)[0] is not None and stage.transport in ("smem-async", "smem-tma")
            if not packed_copy and stage not in (Stage(depth=1), Stage(depth=2)):
                cache[key] = None
                return None
            resolved_stage = _resolve_stage(tile_op, self.target, fold, node.tile, geometry, stage, facts)
        elif not stage.is_direct:
            resolved_stage = _resolve_stage(tile_op, self.target, fold, node.tile, geometry, stage, facts)
        if not stage.is_direct and (resolved_stage is None or resolved_stage.choice != stage):
            cache[key] = None
            return None
        if isinstance(geometry, PlacedTile):
            if _paired_budget_refusal(fold, facts.producer, geometry, resolved_stage) is not None:
                cache[key] = None
                return None
        support = _LocalSupport(
            node,
            edges,
            work=derive_inventory((node.tile,), coop=node.reduce.coop if isinstance(node, ReductionSchedule) else 1),
            axes=(
                tuple(_AxisAgreement(side.axis.name, side.tile, side.units) for side in geometry.mn)
                if node.tile.is_tiled and isinstance(geometry, PlacedTile)
                else ()
            ),
            fragments=(
                _fragment_agreements(
                    site,
                    fold,
                    node.tile,
                    geometry,
                    resolved_stage,
                    facts,
                    frozenset(candidate.need for candidate in self.tile_op.contractions.values() if candidate.need is not None),
                )
                if isinstance(geometry, PlacedTile)
                else ()
            ),
            raster_eligible=node.tile.is_tiled and view.as_contraction() is not None,
            # A producer band splits the staged K-loop's phases across warp bands, which only the
            # contraction tier's skeleton drives; the chunk tier runs every warp through one uniform
            # ring, where an aux band decoding onto warp 0 would re-issue its elected TMA arrive.
            producer_eligible=not fold.chunked() and not (tile_op.packed_reading(fold)[0] is not None and stage.transport == "smem-tma"),
        )
        cache[key] = support
        return support

    def _intrinsic_support(
        self,
        site: NodeId,
        node: NodeSchedule,
        edges: Mapping[EdgeSite, EdgeSchedule],
    ) -> _LocalSupport | None:
        """Derive the target-independent local relation when no finite domains are attached."""
        if site not in self.tile_op.family_sites["TILE"] and node.tile != Tile():
            return None
        if node.tile.is_warp and hasattr(self.target, node.tile.atom.target_feature):
            if not node.tile.atom.available_on(self.target):
                return None
        stages = {choice.stage for choice in edges.values()}
        if len(stages) > 1:
            self._refuse("one contraction currently requires one transport choice across its operands", site)
        if any(edge not in self.tile_op.stage_edges and not choice.stage.is_direct for edge, choice in edges.items()):
            return None
        if any(not choice.stage.is_direct and not node.tile.is_tiled for choice in edges.values()):
            return None
        if any(
            not choice.stage.is_direct and hasattr(self.target, "has_cp_async") and not choice.stage.available_on(self.target)
            for choice in edges.values()
        ):
            return None
        coop = node.reduce.coop if isinstance(node, ReductionSchedule) else 1
        try:
            work = derive_inventory((node.tile,), coop=coop)
        except ValueError:
            return None
        view = self.tile_op.views[site]
        return _LocalSupport(
            node,
            edges,
            work=work,
            raster_eligible=node.tile.is_tiled and view.as_contraction() is not None,
        )

    @cached_property
    def _shared_roots(self) -> frozenset[NodeId]:
        """The contraction roots that may not be output-tiled together
        (:func:`~emmy.compiler.ir.tile.ops.refused_roots`): one of them is the kernel's root and
        every other reduce lowers serially inside the projection, so a row tiling a second root
        spells a kernel the binder never builds. The binder's rule, applied at the offer. The
        placement lane asks a NEIGHBOURING question of the same projection
        (:func:`~emmy.compiler.ir.tile.ops.owns_outputs_it_cannot_bind`) and the two answers
        differ — a projection this one finds nothing shared in can still be one that cut takes
        apart."""
        from emmy.compiler.ir.tile.ops import refused_roots  # noqa: PLC0415

        return frozenset(self.tile_op.node_id(root) for root in refused_roots(self.tile_op.op, tuple(self.tile_op.output_specs)))

    def _support_refusal(self, site: NodeId, support: _LocalSupport) -> str | None:
        """Return why one locally supported pick cannot extend this prefix."""
        if (
            site in self._shared_roots
            and support.node.tile.is_tiled
            and any(self.assignment.nodes[other].tile.is_tiled for other in self._shared_roots if other in self.assignment.nodes)
        ):
            return "a second output-tiled root on a projection its outputs do not partition by root"
        return self._prefix_relation_refusal(
            support,
            work=self._work,
            previous_nodes=tuple(self.assignment.nodes.values()) if self._work is None else (),
            axes=tuple(self._axes.items()),
            fragments=tuple(self._fragments.items()),
            allowed_works=None if self.problem is None else self.problem.allowed_works,
        )

    @staticmethod
    def _prefix_relation_refusal(
        support: _LocalSupport,
        *,
        work: Work | None,
        previous_nodes: tuple[NodeSchedule, ...],
        axes: tuple[tuple[str, tuple[int, int]], ...],
        fragments: tuple[tuple[tuple[str, str], tuple], ...],
        allowed_works: frozenset[tuple[str, tuple[int, ...]]] | None,
    ) -> str | None:
        """Return the one work/axis/fragment refusal shared by frontier indexing and extension."""
        if work is not None and support.work is not None and support.work != work:
            return "pick requires a different worker inventory"
        resolved_work = support.work or work
        if allowed_works is not None and resolved_work is not None and (resolved_work.kind, resolved_work.units) not in allowed_works:
            return "pick cannot reach a kernel allowed by the schedule restriction"
        if work is None and resolved_work is not None:
            if not all(choice.tile.is_canonical_for(resolved_work) for choice in (*previous_nodes, support.node)):
                return "pick is not canonical for the worker inventory"
        elif not support.node.tile.is_canonical_for(resolved_work):
            return "pick is not canonical for the worker inventory"
        axis_values = dict(axes)
        for claim in support.axes:
            value = (claim.tile, claim.units)
            if axis_values.get(claim.name, value) != value:
                return "pick disagrees on physical-axis geometry"
        fragment_values = dict(fragments)
        for claim in support.fragments:
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
                compatible = (
                    offer[0] == "warp"
                    and need[1:3] == offer[1:3]
                    and keys[0] == need[5]  # the producer's N is the carrier's key: a (row, chunk) tile
                    and keys[1:3] == (1, need[3])  # one warp column, and that column IS the chunk
                    and rows[3] == need[4]  # the same register rows the carrier holds
                )
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

    def _finish(self, pick: ClassicAssignment) -> ClassicScheduleContext:
        if (
            not self.nodes_complete
            or self.assignment.kernel is not None
            or not isinstance(pick.kernel, KernelSchedule)
            or pick.nodes
            or pick.edges
            or (self.problem is not None and pick.kernel not in self.problem.kernel_site.kernel_set)
        ):
            self._refuse("pick is incompatible with the classic kernel position")
        work = self._work or Work()
        # Same rule as :meth:`_kernel_composes`: a kernel no node constrains takes any inventory
        # its own domain offers, so a bare elementwise map's output sweep is not left serial.
        if (pick.kernel.work.kind != work.kind or pick.kernel.work.units != work.units) and not (
            self._work is None and self._no_site_claims_inventory()
        ):
            self._refuse("kernel WORK does not realize the node choices")
        if not pick.kernel.raster.is_direct and not self._raster_eligible:
            self._refuse("RASTER requires a tiled contraction site")
        if pick.kernel.work.producer and not self._producer_eligible:
            self._refuse("producer band is incompatible with the selected transport")
        assignment = Schedule(pick.kernel, self.assignment.nodes, self.assignment.edges)
        self._require_kernel_prefix(assignment)
        if self.problem is not None and (why := self.problem.unrealized_bare_pin(assignment)):
            self._refuse(why)
        return replace(self, _assignment=assignment)

    def _require_kernel_prefix(self, schedule: ClassicAssignment) -> None:
        """Validate the kernel facts not already proved by local prefix composition."""
        kernel_work = Work(schedule.kernel.work.kind, schedule.kernel.work.units)
        warp_size = getattr(self.target, "warp_size", 32)
        compute_threads = kernel_work.count * (warp_size if kernel_work.kind == "warp" else 1)
        producer_threads = schedule.kernel.work.producer * warp_size
        if producer_threads > compute_threads:
            self._refuse("producer band cannot outnumber the compute band")
        if compute_threads + producer_threads > getattr(self.target, "max_threads_per_cta", 1024):
            self._refuse("worker inventory exceeds the target thread limit")
        if not schedule.kernel.work.producer:
            return
        for site, assignment in schedule.nodes.items():
            if not assignment.tile.is_tiled:
                continue
            if isinstance(assignment, ReductionSchedule) and assignment.reduce.needs_split:
                self._refuse("a producer band cannot accompany a cross-CTA reduction", site)
            edges = tuple(edge for edge in self.incident_edges(site) if edge in self.tile_op.stage_edges)
            if not edges or any(schedule.edges[edge].stage.transport != "smem-tma" for edge in edges):
                self._refuse("a producer band requires TMA transport at every tiled consumer", site)

    def _refuse(self, reason: str, site: NodeId | EdgeSite | None = None) -> None:
        if site is None:
            raise ScheduleRefused(reason)
        where = node_id_spelling(site) if type(site) is int else edge_site_spelling(site)
        raise ScheduleRefused(f"{where}: {reason}")

    def _require_complete_shape(self, schedule: ClassicAssignment) -> None:
        """Validate only complete-assignment structure before replaying normal transitions."""
        if not isinstance(schedule, Schedule) or not isinstance(schedule.kernel, KernelSchedule):
            self._refuse("assignment must contain a classic kernel schedule")
        if any(not isinstance(value, (ProjectionSchedule, ReductionSchedule)) for value in schedule.nodes.values()):
            self._refuse("classic node assignments must contain projection or reduction schedules")
        if any(not isinstance(value, EdgeSchedule) for value in schedule.edges.values()):
            self._refuse("classic edge assignments must contain edge schedules")
        expected_nodes = set(self.tile_op.node_sites)
        if missing := expected_nodes - schedule.nodes.keys():
            self._refuse("missing node assignment", min(missing))
        if extra := schedule.nodes.keys() - expected_nodes:
            self._refuse("node assignment is outside this problem", min(extra))
        expected_edges = set(self.tile_op.edge_sites)
        if missing := expected_edges - schedule.edges.keys():
            self._refuse("missing edge assignment", min(missing))
        if extra := schedule.edges.keys() - expected_edges:
            self._refuse("edge assignment is outside this problem", min(extra))

        for site in self.tile_op.node_sites:
            view = self.tile_op.views[site]
            assignment = schedule.nodes[site]
            if view.axis is None and not isinstance(assignment, ProjectionSchedule):
                self._refuse("projection site requires a projection schedule", site)
            if view.axis is not None and not isinstance(assignment, ReductionSchedule):
                self._refuse("reduction site requires a reduction schedule", site)
            if isinstance(assignment.tile, PlacedTile):
                self._refuse("node choices cannot contain placed tile geometry", site)

    def _no_site_claims_inventory(self) -> bool:
        return no_site_claims_inventory(self.tile_op)

    def _kernel_composes(self, kernel: KernelSchedule) -> bool:
        work = self._work or Work()
        # A kernel no node constrains takes any inventory its own domain offers: with nothing to
        # disagree with, holding it to ``Work()`` is not a compatibility rule but the collapse that
        # leaves an output sweep serial in one worker per cell.
        agrees = (kernel.work.kind == work.kind and kernel.work.units == work.units) or (
            self._work is None and self._no_site_claims_inventory()
        )
        return agrees and (not kernel.work.producer or self._producer_eligible) and (kernel.raster.is_direct or self._raster_eligible)

    def node_assignment(self, site: NodeId) -> NodeSchedule:
        return self.assignment.nodes[site]

    def edge_assignment(self, edge: EdgeSite) -> EdgeSchedule:
        return self.assignment.edges[edge]

    @property
    def work(self) -> Work | None:
        return self._work


class ClassicScheduleCodec:
    """Strict wire boundary for complete classic schedules.

    Kernel families are bare. A node family is bare when it has one applicable site and carries
    its site's route (``@map.1/twist.1/inner``) only when the family is ambiguous. STAGE is one
    transport decision per consumer node and follows the same rule. Decoding accepts no aliases, missing direct values,
    or unknown fields.
    """

    def __init__(self, context: ClassicScheduleContext) -> None:
        if not isinstance(context, ClassicScheduleContext):
            raise TypeError("classic codec requires a ClassicScheduleContext")
        self.context = context
        self.tile_op = context.tile_op
        stage_consumers = tuple(dict.fromkeys(edge[0] for edge in self.tile_op.stage_edges))
        self._key_order = (
            "WORK",
            "RASTER",
            *(classic_node_key(self.tile_op, "TILE", site) for site in self.tile_op.family_sites["TILE"]),
            *(classic_node_key(self.tile_op, "REDUCE", site) for site in self.tile_op.family_sites["REDUCE"]),
            *(
                classic_stage_key(self.tile_op, next(edge for edge in self.tile_op.stage_edges if edge[0] == site))
                for site in stage_consumers
            ),
        )
        self._keys = frozenset(self._key_order)

    def encode(self, schedule: ClassicAssignment) -> dict[str, str]:
        """Encode one accepted typed schedule in canonical scope order."""
        accepted = self.context.extend(schedule).assignment
        return self._encode(accepted)

    def _encode(self, schedule: ClassicAssignment) -> dict[str, str]:
        """Encode a schedule already accepted by this codec's context traversal."""
        row = {
            "WORK": schedule.kernel.work.spell(),
            "RASTER": schedule.kernel.raster.spell(),
        }
        for site in self.tile_op.family_sites["TILE"]:
            row[classic_node_key(self.tile_op, "TILE", site)] = schedule.nodes[site].tile.spell()
        for site in self.tile_op.family_sites["REDUCE"]:
            assignment = schedule.nodes[site]
            assert isinstance(assignment, ReductionSchedule)
            row[classic_node_key(self.tile_op, "REDUCE", site)] = assignment.reduce.spell()
        stage_consumers = tuple(dict.fromkeys(edge[0] for edge in self.tile_op.stage_edges))
        for site in stage_consumers:
            edges = tuple(edge for edge in self.tile_op.stage_edges if edge[0] == site)
            stages = {schedule.edges[edge].stage for edge in edges}
            if len(stages) != 1:
                raise ValueError(f"{node_id_spelling(site)}: one STAGE value must cover every operand edge")
            row[classic_stage_key(self.tile_op, edges[0])] = stages.pop().spell()
        return row

    def delta(self, before: ClassicScheduleContext, after: ClassicScheduleContext) -> dict[str, str]:
        """Encode the canonical row fields introduced by one compatibility step."""
        if any(context.tile_op != self.context.tile_op for context in (before, after)):
            raise ValueError("classic codec delta requires contexts for its problem")
        row = {}
        assert before.order is not None and after.order is not None
        for site in after.order[before.position : after.position]:
            node = after.node_assignment(site)
            if site in after.tile_op.family_sites["TILE"]:
                row[after.node_key("TILE", site)] = node.tile.spell()
            if site in after.tile_op.family_sites["REDUCE"]:
                assert isinstance(node, ReductionSchedule)
                row[after.node_key("REDUCE", site)] = node.reduce.spell()
            staged = tuple(edge for edge in after.incident_edges(site) if edge in after.tile_op.stage_edges)
            if staged:
                choices = {after.edge_assignment(edge) for edge in staged}
                if len(choices) == 1:
                    row[after.stage_key(staged[0])] = choices.pop().stage.spell()
        if after.work is not None:
            row["WORK"] = after.work.spell()
        return row

    def decode(self, row: Mapping[str, str]) -> ClassicAssignment:
        """Decode one complete canonical row and reject every other key set or assignment."""
        schedule = self._parse(row)
        return self._validate_row(schedule, row)

    def _parse(self, row: Mapping[str, str]) -> ClassicAssignment:
        """Parse typed values before a reconstructed TileOp supplies materialization for validation."""
        self._check_keys(row)

        work = Work.parse(row["WORK"])
        nodes: dict[NodeId, NodeSchedule] = {}
        for site in self.tile_op.node_sites:
            view = self.tile_op.views[site]
            reduce = None
            if view.axis is not None:
                reduce = Reduce.parse(row[classic_node_key(self.tile_op, "REDUCE", site)], work)
            tile = (
                resolve_site_tile(
                    row[classic_node_key(self.tile_op, "TILE", site)],
                    work,
                    reduce.coop if reduce is not None else 1,
                )
                if site in self.tile_op.family_sites["TILE"]
                else Tile()
            )
            nodes[site] = ProjectionSchedule(tile) if reduce is None else ReductionSchedule(tile, reduce)
        return Schedule(
            KernelSchedule(work, Raster.parse(row["RASTER"])),
            nodes,
            {
                edge: EdgeSchedule(Stage.parse(row[classic_stage_key(self.tile_op, edge)]))
                if edge in self.tile_op.stage_edges
                else EdgeSchedule(Stage.direct())
                for edge in self.tile_op.edge_sites
            },
        )

    def _validate_row(self, schedule: ClassicAssignment, row: Mapping[str, str]) -> ClassicAssignment:
        """Validate a parsed assignment and its claimed canonical row exactly once."""
        accepted = self.context.extend(schedule).assignment
        canonical = self._encode(accepted)
        if dict(row) != canonical:
            raise ValueError("classic schedule row is not its typed schedule's canonical encoding")
        return accepted

    def _check_keys(self, row: Mapping[str, str]) -> None:
        """Require the codec's exact key set before parsing or validating values."""
        expected = self._keys
        actual = set(row)
        if missing := expected - actual:
            raise ValueError(f"classic schedule row is missing {', '.join(sorted(missing))}")
        if extra := actual - expected:
            raise ValueError(f"classic schedule row has unknown keys {', '.join(sorted(extra))}")

    def keys(self) -> tuple[str, ...]:
        """Return accepted keys in canonical encoding order."""
        return self._key_order


__all__ = [
    "CLASSIC_FAMILIES",
    "ClassicAssignment",
    "ClassicMaterialization",
    "ClassicDomains",
    "ClassicScheduleCodec",
    "ClassicScheduleContext",
    "EdgeSchedule",
    "edge_site_spelling",
    "KernelSchedule",
    "NodeSchedule",
    "ProjectionSchedule",
    "ReductionSchedule",
    "node_id_spelling",
    "parse_edge_site",
    "parse_node_id",
]
