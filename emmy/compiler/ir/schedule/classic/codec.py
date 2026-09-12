"""The strict wire boundary of a classic schedule: ``ClassicScheduleCodec`` encodes an accepted schedule as
its canonical row and decodes one row into a typed schedule, validating through one context."""

from __future__ import annotations

from collections.abc import Mapping

from emmy.compiler.ir.schedule.base import Schedule
from emmy.compiler.ir.schedule.choices import Raster, Reduce, Stage, Tile, Work, resolve_site_tile
from emmy.compiler.ir.schedule.views import NodeId

from .context import ClassicScheduleContext
from .schedule import (
    ClassicSchedule,
    EdgeSchedule,
    KernelSchedule,
    NodeSchedule,
    ProjectionSchedule,
    ReductionSchedule,
    classic_node_key,
    classic_stage_key,
    node_id_spelling,
)


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

    def encode(self, schedule: ClassicSchedule) -> dict[str, str]:
        """Encode one accepted typed schedule in canonical scope order."""
        accepted = self.context.extend(schedule).schedule
        return self._encode(accepted)

    def _encode(self, schedule: ClassicSchedule) -> dict[str, str]:
        """Encode a schedule already accepted by this codec's context traversal."""
        row = {
            "WORK": schedule.kernel.work.spell(),
            "RASTER": schedule.kernel.raster.spell(),
        }
        for site in self.tile_op.family_sites["TILE"]:
            row[classic_node_key(self.tile_op, "TILE", site)] = schedule.nodes[site].tile.spell()
        for site in self.tile_op.family_sites["REDUCE"]:
            choice = schedule.nodes[site]
            assert isinstance(choice, ReductionSchedule)
            row[classic_node_key(self.tile_op, "REDUCE", site)] = choice.reduce.spell()
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
            node = after.node_choice(site)
            if site in after.tile_op.family_sites["TILE"]:
                row[after.node_key("TILE", site)] = node.tile.spell()
            if site in after.tile_op.family_sites["REDUCE"]:
                assert isinstance(node, ReductionSchedule)
                row[after.node_key("REDUCE", site)] = node.reduce.spell()
            staged = tuple(edge for edge in after.incident_edges(site) if edge in after.tile_op.stage_edges)
            if staged:
                choices = {after.edge_choice(edge) for edge in staged}
                if len(choices) == 1:
                    row[after.stage_key(staged[0])] = choices.pop().stage.spell()
        if after.work is not None:
            row["WORK"] = after.work.spell()
        return row

    def decode(self, row: Mapping[str, str]) -> ClassicSchedule:
        """Decode one complete canonical row and reject every other key set or schedule."""
        schedule = self._parse(row)
        return self._validate_row(schedule, row)

    def _parse(self, row: Mapping[str, str]) -> ClassicSchedule:
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

    def _validate_row(self, schedule: ClassicSchedule, row: Mapping[str, str]) -> ClassicSchedule:
        """Validate a parsed schedule and its claimed canonical row exactly once."""
        accepted = self.context.extend(schedule).schedule
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
