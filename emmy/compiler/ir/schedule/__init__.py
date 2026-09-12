"""Schedule interfaces and reusable choices."""

from emmy.compiler.ir.pure.fold import ContractionView

from .base import Schedule, ScheduleContext, ScheduleProblem, ScheduleRefused, Site, schedule
from .choices import (
    AtomKind,
    FoldMove,
    Level,
    PlacedTile,
    Placement,
    Raster,
    Reduce,
    ReduceStage,
    ResolvedStage,
    Side,
    Stage,
    Tile,
    WarpSpec,
    Work,
    derive_inventory,
    derive_workers,
    resolve_site_tile,
)
from .views import ContractionFacts, EdgeSite, NodeId

__all__ = [
    "AtomKind",
    "ContractionFacts",
    "ContractionView",
    "EdgeSite",
    "FoldMove",
    "Level",
    "NodeId",
    "Placement",
    "PlacedTile",
    "Raster",
    "Reduce",
    "ReduceStage",
    "ResolvedStage",
    "Schedule",
    "ScheduleContext",
    "ScheduleProblem",
    "ScheduleRefused",
    "Side",
    "Site",
    "Stage",
    "Tile",
    "WarpSpec",
    "Work",
    "derive_inventory",
    "derive_workers",
    "resolve_site_tile",
    "schedule",
]
