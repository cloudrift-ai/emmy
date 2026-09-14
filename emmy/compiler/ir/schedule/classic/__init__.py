"""The classic grid/CTA/warp/thread/register schedule family, one role per module:

* ``schedule``   — the choice types, the sites' wire spellings and keys, ``ClassicSchedule``
* ``refusals``   — every per-choice legality rule, asked of a catalog value and of a parsed row value alike
* ``sites``      — the source: ``ClassicProblem`` factored into node sites and the kernel site
* ``context``    — the join: ``ClassicScheduleContext``, the compatibility prefix
* ``codec``      — the wire boundary
* ``materialize`` — the lowering boundary

``ir/tile/ops`` reads the choice types and key spellings through this package while a compile is importing it, so
nothing here may import the tile package at module level: the reads of ``ir/tile/ops`` sit inside the functions that
need them. Import order within this file therefore decides nothing — the dependency chain does.
"""

from .codec import ClassicScheduleCodec
from .context import ClassicScheduleContext
from .materialize import ClassicMaterialization, materialize_classic
from .schedule import (
    CLASSIC_FAMILIES,
    ClassicSchedule,
    EdgeSchedule,
    KernelSchedule,
    NodeSchedule,
    ProjectionSchedule,
    ReductionSchedule,
    classic_node_key,
    classic_stage_key,
    edge_site_spelling,
    no_site_claims_inventory,
    node_id_spelling,
    parse_edge_site,
    parse_node_id,
)
from .sites import ClassicKernelSite, ClassicNodeSite, ClassicProblem

__all__ = [
    "CLASSIC_FAMILIES",
    "ClassicSchedule",
    "ClassicKernelSite",
    "ClassicMaterialization",
    "ClassicNodeSite",
    "ClassicProblem",
    "ClassicScheduleCodec",
    "ClassicScheduleContext",
    "EdgeSchedule",
    "KernelSchedule",
    "NodeSchedule",
    "ProjectionSchedule",
    "ReductionSchedule",
    "classic_node_key",
    "classic_stage_key",
    "edge_site_spelling",
    "materialize_classic",
    "no_site_claims_inventory",
    "node_id_spelling",
    "parse_edge_site",
    "parse_node_id",
]
