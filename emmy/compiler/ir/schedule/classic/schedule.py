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


@dataclass(frozen=True, slots=True)
class KernelSchedule:
    """Kernel-scoped choices: the worker inventory, the raster, and where the state the kernel carries between
    steps is kept — direct (a buffer in global memory, one launch per step), shared memory (``d1/smem``: the
    steps run inside one launch)."""

    work: Work
    raster: Raster
    state: Stage = field(default_factory=Stage.direct)

    def __post_init__(self) -> None:
        if not isinstance(self.work, Work) or not isinstance(self.raster, Raster) or not isinstance(self.state, Stage):
            raise TypeError("classic kernel choices must be Work, Raster and Stage values")


@dataclass(frozen=True, slots=True)
class ProjectionSchedule:
    """The choices of a projection node."""

    tile: Tile

    def __post_init__(self) -> None:
        if not isinstance(self.tile, Tile):
            raise TypeError("classic projection TILE must be an unplaced Tile choice")


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
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


def classic_keys(tile_op) -> tuple[str, ...]:
    """The row keys a kernel's classic schedule spells, in canonical encoding order: the kernel's ``WORK`` and
    ``RASTER``, one ``TILE`` and one ``REDUCE`` key per site of the family, one ``STAGE`` key per staged consumer and
    one per carried state."""
    stage_consumers = tuple(dict.fromkeys(edge[0] for edge in tile_op.stage_edges))
    return (
        "WORK",
        "RASTER",
        *(classic_node_key(tile_op, "TILE", site) for site in tile_op.family_sites["TILE"]),
        *(classic_node_key(tile_op, "REDUCE", site) for site in tile_op.family_sites["REDUCE"]),
        *(classic_stage_key(tile_op, next(edge for edge in tile_op.stage_edges if edge[0] == site)) for site in stage_consumers),
        *classic_state_keys(tile_op),
    )


def classic_stage_key(sites, edge: EdgeSite) -> str:
    """Return the canonical key for one staged consumer: bare for one consumer, else its route."""
    if edge not in sites.stage_edges:
        raise ValueError(f"{edge_site_spelling(edge)} is not a STAGE edge")
    consumers = tuple(dict.fromkeys(candidate[0] for candidate in sites.stage_edges))
    return "STAGE" if len(consumers) == 1 and not sites.state_routes else f"STAGE@{sites.sites[edge[0]].path}"


def classic_state_keys(sites) -> tuple[str, ...]:
    """The canonical key of each carried state's storage, by the rule every ``STAGE`` site follows: bare when it
    is the kernel's one ``STAGE`` site, else the route of the fold that carries it."""
    routes = sites.state_routes
    if len(routes) == 1 and not sites.stage_edges:
        return ("STAGE",)
    keys = tuple(f"STAGE@{route}" for route in routes)
    if taken := {f"STAGE@{sites.sites[edge[0]].path}" for edge in sites.stage_edges} & set(keys):
        raise ValueError(f"a carried state and a staged consumer share the key {sorted(taken)[0]}")
    return keys


def holds_state(work: Work, raster: Raster) -> bool:
    """Whether a kernel under this inventory can keep its carried state in shared memory: one CTA owns a state
    whole and its threads stripe every cell copy, which the one-dimensional cooperative thread tile is."""
    return work.kind == "thread" and work.units[1] == 1 and work.units[0] > 1 and raster.is_direct


def binds_root(choice: ProjectionSchedule | ReductionSchedule) -> bool:
    """Whether the kernel binder builds around this node: an output tile, or a reduce that claims
    a worker inventory. ``TILE`` and ``REDUCE`` both select the root the binder builds around, so
    the offer and the binder read one predicate."""
    return choice.tile.is_tiled or (isinstance(choice, ReductionSchedule) and (choice.reduce.coop > 1 or choice.reduce.reg > 1))


#: Threads in a CTA that packs cells of a warp-wide cooperative reduce (``WORK=t<coop>x<cells>``).
PACKED_CTA_THREADS = 128


def packed_works(work: Work | None) -> frozenset[Work]:
    """The inventory that packs several cells of a cooperative reduce into one CTA.

    A cooperative reduce launches one CTA per output cell, ``coop`` threads each. When ``coop`` is
    at most a warp, its combine is a lane butterfly that never leaves the cell's lanes, so the
    launch can stack cells in one CTA (``t<coop>x<cells>``, ``PACKED_CTA_THREADS`` threads): the
    flat thread decode already hands consecutive cells to consecutive lane groups. At a 128-wide
    row the one-cell CTA is 32 or 64 threads, and the launch pays per CTA."""
    if work is None or work.kind != "thread" or work.units[1] != 1 or not 1 < work.units[0] <= 32:
        return frozenset()
    return frozenset({Work(kind="thread", units=(work.units[0], PACKED_CTA_THREADS // work.units[0]))})


def output_sweep_works(tile_op, claimed_work: Work | None) -> frozenset[Work]:
    """The ``WORK`` values that may stripe every output sweep of a kernel whose nodes stay serial.

    Each output must own a sweep, because a scalar sibling would be repeated by every lane. A tiled
    or cooperative node choice that already claims an inventory must agree through the ordinary
    compatibility relation instead.
    """
    if claimed_work is not None or not tile_op.output_specs or not all(spec.sweep for spec in tile_op.output_specs):
        return frozenset()
    return frozenset(Work(kind="thread", units=(move.coop, 1)) for move in coop_reduce_moves() if move.coop > 1)
