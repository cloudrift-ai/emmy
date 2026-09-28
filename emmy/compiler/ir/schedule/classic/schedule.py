"""What a classic schedule is written in: the kernel, node and edge choice types, the sites' wire spellings and
keys, and ``ClassicSchedule`` — this family's filling of the generic :class:`~emmy.compiler.ir.schedule.base.Schedule`.

Which of these values a site may actually take is the sites' business (``classic.sites``); this module only says
what they are and how each one is spelled."""

from __future__ import annotations

from dataclasses import dataclass, field

from emmy.compiler.ir.schedule.base import Schedule
from emmy.compiler.ir.schedule.catalog import coop_reduce_moves
from emmy.compiler.ir.schedule.choices import Raster, Reduce, Stage, Tile, Work
from emmy.compiler.ir.schedule.views import EdgeSite, NodeId

CLASSIC_FAMILIES = ("TILE", "REDUCE", "STAGE")

#: The ``STAGE`` key of a kernel's carried state — where the state lives across the steps of its sequential
#: axis. Always scoped, never the bare family: a bare ``STAGE`` pin is an operand-transport pin, and it must not
#: silently move a state on chip. Spelled only on a kernel whose placement has a sequential axis.
STATE_KEY = "STAGE@state"

#: The transports a carried state may take: global memory (one launch per step, the launch loop the runner
#: drives), registers (a warp owns its rows) and shared memory (a CTA owns its block).
STATE_TRANSPORTS = ("direct", "reg", "smem")


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


def _is_node_id(node_id: NodeId) -> bool:
    return type(node_id) is int and node_id >= 0


def _is_edge_site(edge: EdgeSite) -> bool:
    return isinstance(edge, tuple) and len(edge) == 2 and _is_node_id(edge[0]) and type(edge[1]) is int and edge[1] >= 0


@dataclass(frozen=True)
class KernelSchedule:
    """Kernel-scoped choices."""

    work: Work
    raster: Raster
    #: Where the carried state of a sequential axis lives across its steps (``STATE_KEY``). ``direct`` is the
    #: launch loop over a global buffer; an on-chip transport runs the whole step loop inside one launch, so
    #: the loop's scope is derived from this choice rather than chosen beside it. Direct on every kernel that
    #: carries no state, which is what keeps every such kernel's row byte-identical.
    state: Stage = field(default_factory=Stage.direct)

    def __post_init__(self) -> None:
        if not isinstance(self.work, Work) or not isinstance(self.raster, Raster):
            raise TypeError("classic kernel choices must be Work and Raster values")
        if not isinstance(self.state, Stage) or self.state.transport not in STATE_TRANSPORTS:
            raise TypeError("classic kernel state must be a direct, reg or smem Stage")

    @property
    def resident(self) -> bool:
        """Whether the carried state lives on chip, the step loop inside the launch."""
        return not self.state.is_direct


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


type ClassicSchedule = Schedule[KernelSchedule, NodeSchedule, EdgeSchedule]


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


def binds_root(choice: ProjectionSchedule | ReductionSchedule) -> bool:
    """Whether the kernel binder builds around this node: an output tile, or a reduce that claims
    a worker inventory. ``TILE`` and ``REDUCE`` both select the root the binder builds around, so
    the offer and the binder read one predicate."""
    return choice.tile.is_tiled or (isinstance(choice, ReductionSchedule) and (choice.reduce.coop > 1 or choice.reduce.reg > 1))


def carries_state(tile_op) -> bool:
    """Whether the kernel carries a state across a sequential axis — the kernels that spell ``STATE_KEY``."""
    return bool(tile_op.place.serial)


def resident_works() -> tuple[Work, ...]:
    """The thread inventories a shared-memory resident state's cell sweep may be striped across: the CTA
    holds the block, and each thread walks its own strided slice of the cells every step."""
    return tuple(Work(kind="thread", units=(width, 1)) for width in (32, 64, 128, 256, 512, 1024))


def output_sweep_works(tile_op, claimed_work: Work | None) -> frozenset[Work]:
    """The ``WORK`` values that may stripe every output sweep of a kernel whose nodes stay serial.

    Each output must own a sweep, because a scalar sibling would be repeated by every lane. A tiled
    or cooperative node choice that already claims an inventory must agree through the ordinary
    compatibility relation instead.
    """
    if claimed_work is not None or not tile_op.output_specs or not all(spec.sweep for spec in tile_op.output_specs):
        return frozenset()
    return frozenset(Work(kind="thread", units=(move.coop, 1)) for move in coop_reduce_moves() if move.coop > 1)
