"""The classic grid/CTA/warp/thread/register schedule family, one role per module:

* ``assignment`` — the choice types, the sites' wire spellings and keys, ``ClassicDomains``
* ``refusals``   — every per-choice legality rule, asked of a catalog value and of a parsed row value alike
* ``sites``      — the source: ``ClassicProblem`` factored into node sites and the kernel site
* ``context``    — the join: ``ClassicScheduleContext``, the compatibility prefix
* ``codec``      — the wire boundary
* ``materialize`` — the lowering boundary

``assignment`` is imported first on purpose: ``ir/tile/ops`` reads the assignment names through this package
while a compile imports it, and nothing below ``assignment`` imports the tile package at module level.
"""

from .assignment import (
    CLASSIC_FAMILIES,
    ClassicAssignment,
    ClassicDomains,
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
from .codec import ClassicScheduleCodec
from .context import ClassicScheduleContext
from .materialize import ClassicMaterialization, materialize_classic
from .sites import ClassicKernelSite, ClassicNodeSite, ClassicProblem, ClassicProjectionError, project_classic

__all__ = [
    "CLASSIC_FAMILIES",
    "ClassicAssignment",
    "ClassicDomains",
    "ClassicKernelSite",
    "ClassicMaterialization",
    "ClassicNodeSite",
    "ClassicProblem",
    "ClassicProjectionError",
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
    "project_classic",
]
