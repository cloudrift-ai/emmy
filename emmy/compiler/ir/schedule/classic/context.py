"""The classic compatibility prefix: ``ClassicScheduleContext`` composes the choices its problem's sites offer,
one node with its incident edges at a time and the kernel last, and owns nothing but the join — the relation a
prefix carries (its worker inventory, physical-axis and fragment-seam agreements) and the kernel-level rules
(raster eligibility, resource limits, the producer band). The sites keep every filtered answer, per relation."""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.schedule.base import Schedule, ScheduleContext, ScheduleRefused
from emmy.compiler.ir.schedule.choices import PlacedTile, Work
from emmy.compiler.ir.schedule.views import EdgeSite, NodeId

from .refusals import _Relation, _relation_refusal
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
    node_id_spelling,
    output_sweep_works,
    packed_works,
)
from .sites import local_support, stored_support

if TYPE_CHECKING:
    from emmy.compiler.context import Context
    from emmy.compiler.ir.tile import TileOp

    from .sites import ClassicProblem, _LocalSupport


@dataclass(frozen=True)
class ClassicScheduleContext(ScheduleContext[KernelSchedule, NodeSchedule, EdgeSchedule]):
    """Immutable classic ``c + p + t`` compatibility-composition state.

    The problem ``p`` is the unscheduled ``tile_op`` — its Fold root indexes every site through
    its own site index, and its typed inputs answer every operand-shape question — composed
    against the target ``t``. What the prefix has decided that a later pick's compatibility reads is
    its :class:`_Relation`; a site keeps its filtered answer per relation, so the context carries no table.
    """

    tile_op: TileOp
    target: Context | None = None
    problem: ClassicProblem | None = None
    order: tuple[NodeId, ...] | None = None
    position: int = 0
    _schedule: ClassicSchedule = field(default_factory=lambda: Schedule(None, {}, {}), repr=False)
    _relation: _Relation = field(default_factory=_Relation, repr=False)
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

    def _site_relation(self, site: NodeId) -> _Relation:
        """The relation ``site`` reads: the prefix's agreements, with the decided nodes beside them only where
        a rule reads those, so prefixes that decided different nodes but agree on the facts share one answer."""
        tile = self.tile_op
        if self._relation.work is None or site in tile.shared_roots or any(site in pair for pair in tile.chain_pairs):
            return replace(self._relation, nodes=self.schedule.nodes)
        return self._relation

    def extensions(self) -> Iterator[ClassicSchedule]:
        """Yield the next site's options that compose with this prefix: one node with its
        incident edges, or, past the last node, the kernel picks."""
        if self.schedule.kernel is not None:
            return
        if self.problem is None:
            raise ValueError("classic compatibility composition requires a projected problem")
        if self.nodes_complete:
            for kernel in self.problem.kernel_site.kernels:
                if self._kernel_composes(kernel):
                    yield Schedule(kernel, {}, {})
            return
        assert self.next_site is not None
        site = self.problem.node_site(self.next_site)
        for support in site.frontier(self._site_relation(site.id)):
            yield Schedule(None, {site.id: support.node}, support.edges)

    def random_step(self, rng: random.Random) -> ClassicScheduleContext | None:
        """One compatible extension drawn uniformly over the site's frontier, composed: a kernel pick past the last
        node, tried in random order and composed by :meth:`extend`, whose ``_finish`` proves more than the draw;
        else a node support drawn by :meth:`_random_support` and composed without :meth:`_extend_local`'s
        re-check, since the draw took it from the supports the site admits under this prefix's relation."""
        if self.schedule.kernel is not None:
            return None
        if self.problem is None:
            raise ValueError("classic compatibility composition requires a projected problem")
        if self.nodes_complete:
            kernels = list(self.problem.kernel_site.kernels)
            rng.shuffle(kernels)
            kernel = next((kernel for kernel in kernels if self._kernel_composes(kernel)), None)
            return None if kernel is None else self.extend(Schedule(kernel, {}, {}))
        support = self._random_support(rng)
        return None if support is None else self._compose(self.next_site, support)

    def _random_support(self, rng: random.Random) -> _LocalSupport | None:
        """A support of the next node site drawn uniformly over the admitted (choice, transport) pairs: a node
        choice among those the site admits under this prefix's relation, accepted as often as it has admitted
        supports, then one of those, deriving supports only for the choices it touches. A choice with none is dead
        under this prefix and leaves the draw."""
        assert self.next_site is not None
        site = self.problem.node_site(self.next_site)
        relation = self._site_relation(site.id)
        choices = list(site.compatible(relation))
        width = len(site.edge_picks)
        while choices:
            index = rng.randrange(len(choices))
            admitted = site.admitted(choices[index], relation)
            if not admitted:  # dead under this prefix: out of the draw
                choices[index] = choices[-1]
                choices.pop()
            elif rng.randrange(width) < len(admitted):  # a choice is taken as often as it has admitted supports
                return rng.choice(admitted)
        return None

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
            _relation=_Relation(),
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
        offered = None if self.problem is None else self.problem.node_site(site)
        if offered is not None and (node not in offered.node_set or any(choice not in offered.edge_set for choice in pick.edges.values())):
            self._refuse("pick is outside the next independent classic position", site)
        if offered is not None:
            support = offered.choice(node).support(pick.edges)
        else:
            support = (
                stored_support(self.tile_op, site, node, pick.edges)
                if self.target is None  # a kernel read back with no card: its materialization stands in for one
                else local_support(self.tile_op, self.target, site, node, pick.edges)
            )
        if support is None:
            self._refuse("pick has no local classic support", site)
        relation = self._site_relation(site)
        if offered is not None:
            why = offered.refusal(support, relation)
        else:
            why = _relation_refusal(
                site, support, relation, roots=self.tile_op.shared_roots, pairs=self.tile_op.chain_pairs, allowed_works=None
            )
        if why:
            self._refuse(why, site)
        return self._compose(site, support)

    def _compose(self, site: NodeId, support: _LocalSupport) -> ClassicScheduleContext:
        """This prefix with ``support`` decided at ``site``, its next position — a support already proved to compose."""
        composed = _Relation(
            work=support.work or self._relation.work,
            axes={**self._relation.axes, **{claim.name: (claim.tile, claim.units) for claim in support.axes}},
            fragments={**self._relation.fragments, **{(claim.role, claim.edge): claim.value for claim in support.fragments}},
        )
        return self._advance(
            position=self.position + 1,
            _schedule=Schedule(None, {**self.schedule.nodes, site: support.node}, {**self.schedule.edges, **support.edges}),
            _relation=composed,
            _raster_eligible=self._raster_eligible or support.raster_eligible,
            _producer_eligible=self._producer_eligible and support.producer_eligible,
        )

    def _advance(self, **changed) -> ClassicScheduleContext:
        """This context with the fields ONE composition step changes, skipping ``__post_init__``.

        Everything that ``__post_init__`` derives or proves belongs to ``tile_op`` or ``problem`` —
        the node order covering every site exactly once, the problem projected from this tile — and a
        step touches neither, so a step re-derives only conclusions it already carries. Its own remaining
        checks do not reach a step either: the position bound holds because ``_extend_local`` advances only
        off a ``next_site``. ``replace`` re-ran all of it once per composition step — 43.5k times for one
        SDPA_L schedule walk.

        The public ``extend`` keeps the validating path: a pick decoded from a golden row or handed
        in by a caller has proved none of this."""
        advanced = object.__new__(type(self))
        advanced.__dict__.update(self.__dict__)
        advanced.__dict__.update(changed)
        return advanced

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
        work = self._relation.work or Work()
        # Same rule as :meth:`_kernel_composes`: serial node choices leave output sweeps free to
        # take one of their own offered worker inventories.
        if (pick.kernel.work.kind != work.kind or pick.kernel.work.units != work.units) and not (
            self._output_sweeps_take(pick.kernel.work) or self._packs(pick.kernel.work)
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

    def _output_sweeps_take(self, work: Work) -> bool:
        """Whether this serial-node prefix may take this exact output-sweep WORK offer."""
        return work in output_sweep_works(self.tile_op, self._relation.work)

    def _packs(self, work: Work) -> bool:
        """Whether ``work`` stacks several cells of this prefix's cooperative reduce in one CTA
        (:func:`packed_works`). Every operand reads gmem directly: a staged row is one CTA-wide
        shared slab per cell, which a packed CTA would share between its cells."""
        cooperative = any(
            isinstance(choice, ReductionSchedule) and choice.reduce.coop > 1 and not choice.reduce.coop_transposed
            for choice in self.schedule.nodes.values()
        )
        return (
            cooperative
            and work in packed_works(self._relation.work)
            and all(choice.stage.is_direct for choice in self.schedule.edges.values())
        )

    def _kernel_composes(self, kernel: KernelSchedule) -> bool:
        work = self._relation.work or Work()
        # Serial node choices do not constrain a worker inventory used only to stripe output
        # sweeps; a node-owned inventory still follows the ordinary equality relation, or packs
        # several of its cells into one CTA.
        agrees = (
            (kernel.work.kind == work.kind and kernel.work.units == work.units)
            or self._output_sweeps_take(kernel.work)
            or self._packs(kernel.work)
        )
        return agrees and (not kernel.work.producer or self._producer_eligible) and (kernel.raster.is_direct or self._raster_eligible)

    def node_choice(self, site: NodeId) -> NodeSchedule:
        return self.schedule.nodes[site]

    def edge_choice(self, edge: EdgeSite) -> EdgeSchedule:
        return self.schedule.edges[edge]

    @property
    def work(self) -> Work | None:
        return self._relation.work
