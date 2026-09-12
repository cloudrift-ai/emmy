"""The lowering boundary: ``materialize_classic`` turns one accepted schedule into a scheduled ``TileOp`` with
placed geometry and resolved transports, and ``ClassicMaterialization`` is what it derives."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from frozendict import frozendict

from emmy.compiler.ir.schedule.base import Schedule, ScheduleRefused
from emmy.compiler.ir.schedule.choices import PlacedTile, ResolvedStage, WarpSpec
from emmy.compiler.ir.schedule.views import EdgeSite, NodeId

from .context import ClassicScheduleContext
from .refusals import _resolve_stage
from .schedule import ClassicSchedule, ReductionSchedule, _is_edge_site, _is_node_id, edge_site_spelling, node_id_spelling

if TYPE_CHECKING:
    from emmy.compiler.ir.tile import TileOp


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

    def validate(self, schedule: ClassicSchedule, source: object, *, place: object, workers: object) -> None:
        """Validate classic lowering facts against their semantic schedule."""
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
            site for site, choice in schedule.nodes.items() if choice.tile.is_tiled and source_tile.views[site].as_contraction() is not None
        }
        if set(self.tiles) != expected_tiles:
            raise ValueError("classic materialization must contain exactly the tiled node sites")
        expected_stages = {edge for edge, choice in schedule.edges.items() if not choice.stage.is_direct}
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


def materialize_classic(
    tile: TileOp,
    *,
    name: str,
    knobs: dict,
    target,
    schedule: ClassicSchedule,
) -> TileOp:
    """Materialize one accepted classic schedule into a scheduled TileOp."""
    from emmy.compiler.ir.tile.ops import Sched, scheduled  # noqa: PLC0415 — tile.ops reads this package; module level would cycle

    sched = Sched(tile, place=tile.place.on_grid())
    placed = {}
    resolved = {}
    for site, choice in schedule.nodes.items():
        node = tile.sites[site].node
        geometry = None
        if choice.tile.is_tiled and isinstance(choice, ReductionSchedule):
            geometry = sched.placed(node, choice.tile)
            if not isinstance(geometry, PlacedTile):
                raise ValueError(f"accepted TILE at {node_id_spelling(site)} has no placed geometry")
            placed[site] = geometry
        for edge, edge_choice in schedule.edges.items():
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
        schedule=schedule,
        axes=tile.axes,
        materialization=ClassicMaterialization(placed, resolved),
        workers=WarpSpec(schedule.kernel.work.producer) if schedule.kernel.work.producer else None,
    )
