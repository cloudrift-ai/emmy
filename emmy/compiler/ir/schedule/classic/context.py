"""The classic compatibility prefix: ``ClassicScheduleContext`` composes the options its problem's sites offer,
one node with its incident edges at a time and the kernel last, and owns nothing but the join — worker
inventory, physical-axis agreement, fragment seams, raster eligibility, resource limits."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING

from frozendict import frozendict

from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.schedule.base import Schedule, ScheduleContext, ScheduleRefused
from emmy.compiler.ir.schedule.choices import PlacedTile, Stage, Tile, Work, derive_inventory
from emmy.compiler.ir.schedule.views import EdgeSite, NodeId
from emmy.compiler.structural import instance_memo

from .refusals import (
    _AxisAgreement,
    _fragment_agreements,
    _FragmentAgreement,
    _needs_fill,
    _paired_budget_refusal,
    _plan_node_refusal,
    _resolve_stage,
    _wgmma_refusal,
)
from .schedule import (
    ClassicSchedule,
    EdgeSchedule,
    KernelSchedule,
    NodeSchedule,
    ProjectionSchedule,
    ReductionSchedule,
    _is_edge_site,
    classic_node_key,
    classic_stage_key,
    edge_site_spelling,
    no_site_claims_inventory,
    node_id_spelling,
)

if TYPE_CHECKING:
    from emmy.compiler.context import Context
    from emmy.compiler.ir.tile import TileOp

    from .sites import ClassicNodeSite, ClassicProblem


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


@dataclass(frozen=True)
class ClassicScheduleContext(ScheduleContext[KernelSchedule, NodeSchedule, EdgeSchedule]):
    """Immutable classic ``c + p + t`` compatibility-composition state.

    The problem ``p`` is the unscheduled ``tile_op`` — its Fold root indexes every site through
    its own site index, and its typed inputs answer every operand-shape question — composed
    against the target ``t``. Derivations shared by every candidate ride memo tables on the tile.
    """

    tile_op: TileOp
    target: Context | None = None
    problem: ClassicProblem | None = None
    order: tuple[NodeId, ...] | None = None
    position: int = 0
    _schedule: ClassicSchedule = field(default_factory=lambda: Schedule(None, {}, {}), repr=False)
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
        if not isinstance(self._schedule, Schedule):
            raise TypeError("classic context prefix must be a Schedule")
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
    def schedule(self) -> ClassicSchedule:
        return self._schedule

    def extensions(self) -> Iterator[ClassicSchedule]:
        """Yield the next site's options that compose with this prefix: one node with its
        incident edges, or, past the last node, the kernel picks."""
        if self.schedule.kernel is not None:
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

    def extend(self, pick: ClassicSchedule) -> ClassicScheduleContext:
        """Compose a frontier pick or validate and accept one complete schedule."""
        if not isinstance(pick, Schedule) or self.schedule.kernel is not None:
            self._refuse("classic extension requires an incomplete context and a Schedule pick")
        if pick.kernel is not None and (pick.nodes or pick.edges):
            return self._extend_complete(pick)
        if self.nodes_complete:
            return self._finish(pick)

        return self._extend_local(pick)

    def _extend_complete(self, pick: ClassicSchedule) -> ClassicScheduleContext:
        if any(pick.nodes.get(site) != choice for site, choice in self.schedule.nodes.items()) or any(
            pick.edges.get(edge) != choice for edge, choice in self.schedule.edges.items()
        ):
            self._refuse("complete schedule disagrees with the existing classic prefix")
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
            _schedule=Schedule(None, {}, {}),
            _work=None,
            _axes=frozendict(),
            _fragments=frozendict(),
            _raster_eligible=False,
            _producer_eligible=True,
        )

    def _extend_local(self, pick: ClassicSchedule) -> ClassicScheduleContext:
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
        nodes = {**self.schedule.nodes, site: support.node}
        axes = {**self._axes, **{claim.name: (claim.tile, claim.units) for claim in support.axes}}
        fragments = {**self._fragments, **{(claim.role, claim.edge): claim.value for claim in support.fragments}}
        return self._advance(
            position=self.position + 1,
            _schedule=Schedule(None, nodes, {**self.schedule.edges, **support.edges}),
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
            and any(self.schedule.nodes[other].tile.is_tiled for other in self._shared_roots if other in self.schedule.nodes)
        ):
            return "a second output-tiled root on a projection its outputs do not partition by root"
        return self._prefix_relation_refusal(
            support,
            work=self._work,
            previous_nodes=tuple(self.schedule.nodes.values()) if self._work is None else (),
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

    def _finish(self, pick: ClassicSchedule) -> ClassicScheduleContext:
        if (
            not self.nodes_complete
            or self.schedule.kernel is not None
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
        schedule = Schedule(pick.kernel, self.schedule.nodes, self.schedule.edges)
        self._require_kernel_prefix(schedule)
        if self.problem is not None and (why := self.problem.unrealized_bare_pin(schedule)):
            self._refuse(why)
        return replace(self, _schedule=schedule)

    def _require_kernel_prefix(self, schedule: ClassicSchedule) -> None:
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
        for site, choice in schedule.nodes.items():
            if not choice.tile.is_tiled:
                continue
            if isinstance(choice, ReductionSchedule) and choice.reduce.needs_split:
                self._refuse("a producer band cannot accompany a cross-CTA reduction", site)
            edges = tuple(edge for edge in self.incident_edges(site) if edge in self.tile_op.stage_edges)
            if not edges or any(schedule.edges[edge].stage.transport != "smem-tma" for edge in edges):
                self._refuse("a producer band requires TMA transport at every tiled consumer", site)

    def _refuse(self, reason: str, site: NodeId | EdgeSite | None = None) -> None:
        if site is None:
            raise ScheduleRefused(reason)
        where = node_id_spelling(site) if type(site) is int else edge_site_spelling(site)
        raise ScheduleRefused(f"{where}: {reason}")

    def _require_complete_shape(self, schedule: ClassicSchedule) -> None:
        """Validate only complete-schedule structure before replaying normal transitions."""
        if not isinstance(schedule, Schedule) or not isinstance(schedule.kernel, KernelSchedule):
            self._refuse("a schedule must contain a classic kernel choice")
        if any(not isinstance(value, (ProjectionSchedule, ReductionSchedule)) for value in schedule.nodes.values()):
            self._refuse("classic node choices must be projection or reduction schedules")
        if any(not isinstance(value, EdgeSchedule) for value in schedule.edges.values()):
            self._refuse("classic edge choices must be edge schedules")
        expected_nodes = set(self.tile_op.node_sites)
        if missing := expected_nodes - schedule.nodes.keys():
            self._refuse("missing node choice", min(missing))
        if extra := schedule.nodes.keys() - expected_nodes:
            self._refuse("node choice is outside this problem", min(extra))
        expected_edges = set(self.tile_op.edge_sites)
        if missing := expected_edges - schedule.edges.keys():
            self._refuse("missing edge choice", min(missing))
        if extra := schedule.edges.keys() - expected_edges:
            self._refuse("edge choice is outside this problem", min(extra))

        for site in self.tile_op.node_sites:
            view = self.tile_op.views[site]
            choice = schedule.nodes[site]
            if view.axis is None and not isinstance(choice, ProjectionSchedule):
                self._refuse("projection site requires a projection schedule", site)
            if view.axis is not None and not isinstance(choice, ReductionSchedule):
                self._refuse("reduction site requires a reduction schedule", site)
            if isinstance(choice.tile, PlacedTile):
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

    def node_choice(self, site: NodeId) -> NodeSchedule:
        return self.schedule.nodes[site]

    def edge_choice(self, edge: EdgeSite) -> EdgeSchedule:
        return self.schedule.edges[edge]

    @property
    def work(self) -> Work | None:
        return self._work
