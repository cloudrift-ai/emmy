"""The classic assignment vocabulary: the kernel, node and edge choice types, the sites' wire spellings and
keys. The candidate values themselves belong to the sites (``classic.sites``), which are the one place
that says what may be chosen."""

from __future__ import annotations

from dataclasses import dataclass

from emmy.compiler.ir.schedule.base import Schedule
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


def no_site_claims_inventory(tile_op) -> bool:
    """Whether this kernel has no node site that could fold out a worker inventory.

    Only a tiled site or a cooperative reduction claims one, so a kernel with neither — a bare
    elementwise map, the half a placement cut leaves behind a reduction — has a kernel work that no
    node constrains. :class:`~emmy.compiler.ir.schedule.classic.sites.ClassicKernelSite` offers such a kernel the sweep
    widths, and the two compatibility gates here let them through instead of filtering them back to
    the direct per-cell form. Read off the tile rather than the projected domains, so validation
    (which carries none) answers the same.
    """
    sites = tile_op.family_sites
    return not sites["TILE"] and not sites["REDUCE"]
