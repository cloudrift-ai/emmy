"""The source of every classic candidate: ``ClassicProblem`` — the tile, the target and the knob row — factored
into ``ClassicNodeSite``s and one ``ClassicKernelSite``. A site the row names offers the row's value alone,
parsed and checked with the catalog's own rules; a site the row leaves free offers its catalog."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING

from frozendict import frozendict

from emmy.compiler.ir.atom import ATOM_REGISTRY
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.schedule.base import Schedule, ScheduleProblem, Site
from emmy.compiler.ir.schedule.catalog import coop_reduce_moves, producer_band_moves, raster_moves, scalar_tile_moves
from emmy.compiler.ir.schedule.choices import PlacedTile, Raster, Reduce, Stage, Tile, Work, derive_inventory, resolve_site_tile
from emmy.compiler.ir.schedule.staging import stage_target
from emmy.compiler.ir.schedule.views import NodeId
from emmy.utils import cached_method

from .assignment import (
    ClassicAssignment,
    ClassicDomains,
    EdgeSchedule,
    KernelSchedule,
    NodeSchedule,
    ProjectionSchedule,
    ReductionSchedule,
    classic_node_key,
    classic_stage_key,
    no_site_claims_inventory,
    node_id_spelling,
)
from .refusals import (
    _atom_policy_ok,
    _contraction_plan_allowed,
    _contraction_plans,
    _contraction_reductions,
    _plan_node_refusal,
    _reduction_domain,
    _scalar_catalog,
    _stage_candidates,
    _warp_atoms,
    _warp_plans,
    _wgmma_refusal,
)

if TYPE_CHECKING:
    from emmy.compiler.ir.tile import TileOp


class ClassicProjectionError(RuntimeError):
    """One projected site has no locally supported choice on this structural branch."""


def _select[T](
    named: str | None,
    catalog: Iterator[T] | tuple[T, ...],
    *,
    parse: Callable[[str], T | None],
    allowed: Callable[[T], bool],
    spell: Callable[[T], str],
    bare: str | None,
    validate_pins: bool,
) -> tuple[T, ...]:
    """One factor's values under the row.

    A NAMED value (the row spells this site's exact key) is parsed and checked; that one value is
    the factor. A spelling the parser cannot read alone — a warp tile with no ``WORK`` beside it —
    is matched against the catalog by spelling instead, still one value. A named value the site
    cannot take empties the factor, or, when pins are not validated (a row published across the
    peer kernels of a multi-kernel target), leaves the catalog whole. A BARE pin of an ambiguous
    family names one site among several: this factor keeps the pin's value and OFF, and the
    completed schedule is asked which site carried it."""
    if named is not None:
        value = parse(named)
        if value is not None and allowed(value):
            return (value,)
    values = tuple(catalog)  # the catalog is walked only past the named fast path
    if named is not None:
        matched = tuple(choice for choice in values if spell(choice) == named)
        if matched or validate_pins:
            return matched
    return values if bare is None else tuple(choice for choice in values if spell(choice) in ("", bare))


@dataclass(frozen=True, eq=False)
class ClassicNodeSite(Site[ClassicAssignment]):
    """One node site's independent factor: its node choices, the transport catalog of its incident
    edges, and their product as picks. Every derived read is memoized on the site, so a site's
    tuples keep one identity for the context's caches."""

    problem: ClassicProblem
    id: NodeId

    @cached_property
    def keys(self) -> tuple[str, ...]:
        tile = self.problem.tile
        keys = []
        if self.id in tile.family_sites["TILE"]:
            keys.append(classic_node_key(tile, "TILE", self.id))
        if self.id in tile.family_sites["REDUCE"]:
            keys.append(classic_node_key(tile, "REDUCE", self.id))
        if self.stage_key is not None:
            keys.append(self.stage_key)
        return tuple(keys)

    @cached_property
    def stage_key(self) -> str | None:
        tile = self.problem.tile
        return next((classic_stage_key(tile, edge) for edge in tile.stage_edges if edge[0] == self.id), None)

    @property
    def node(self) -> Fold:
        return self.problem.tile.sites[self.id].node

    def _named(self, family: str) -> str | None:
        """The row's value at this site's ``family`` key, when the row spells that exact key."""
        if self.id not in self.problem.tile.family_sites[family]:
            return None
        return self.problem.row.get(classic_node_key(self.problem.tile, family, self.id))

    def _wgmma_pin_refusal(self, named: str) -> None:
        """Refuse a named warp-group TILE that its WORK, fragment grid, K chunk or STAGE cannot
        feed, with that rule's own message: the catalog never offers such a row, so a silent empty
        site could only report an unsupported pin. Loud whatever the pin reading — the spelling is
        wrong wherever it is published."""
        atom = ATOM_REGISTRY.get(named.partition("/")[0])
        if atom is None or not atom.is_wgmma or not self.problem.loud_pins:
            return
        work = self.problem.work
        plan = Tile.parse(named, work if work is not None and work.kind == "warp" else Work(kind="warp", units=(4, 1)))
        stage = self.problem.row.get(self.stage_key) if self.stage_key is not None else None
        stage = self.problem.row.get("STAGE") if stage is None else stage
        if why := _wgmma_refusal(plan, None if stage is None else Stage.parse(stage)):
            raise ValueError(why)

    def _select_plans(self, named: str | None, catalog, *, allowed) -> tuple[Tile, ...]:
        if named is not None:
            self._wgmma_pin_refusal(named)

        def parse(spelling: str) -> Tile | None:
            try:
                return resolve_site_tile(spelling, self.problem.work)
            except ValueError:
                return None

        return _select(
            named,
            catalog,
            parse=parse,
            allowed=allowed,
            spell=Tile.spell,
            bare=self.problem.bare_value("TILE", self.keys),
            validate_pins=self.problem.validate_pins,
        )

    def _literal(self, family: str, choices: tuple, spell: Callable[[object], str]) -> tuple:
        """A hand-written factor under the row: the same selection the catalog gets, by spelling."""
        if family != "STAGE" and self.id not in self.problem.tile.family_sites[family]:
            return choices
        key = self.stage_key if family == "STAGE" else classic_node_key(self.problem.tile, family, self.id)
        if key is None:
            return choices
        named = self.problem.row.get(key)
        if named is not None:
            kept = tuple(choice for choice in choices if spell(choice) == named)
            choices = kept if kept or self.problem.strict(key) else choices
        bare = self.problem.bare_value(family, self.keys)
        return choices if bare is None else tuple(choice for choice in choices if spell(choice) in ("", bare))

    @cached_property
    def nodes(self) -> tuple[NodeSchedule, ...]:
        """The node choices: the row's value where it names this site, else the catalog."""
        literal = self.problem.domains_literal
        if literal is not None:
            choices = self._literal("TILE", literal.nodes[self.id], lambda choice: choice.tile.spell())
            return self._literal("REDUCE", choices, lambda choice: choice.reduce.spell() if isinstance(choice, ReductionSchedule) else "")
        tile, node = self.problem.tile, self.node
        view = tile.views[self.id]
        if view.axis is None:
            if self.id not in tile.family_sites["TILE"] or not tile.place.free:
                return (ProjectionSchedule(Tile()),)
            inner = tile.place.free[-1]
            extent = inner.extent.as_static() if inner.extent.is_static else 0

            def legal(plan: Tile) -> bool:
                return plan.units == (1, 1) and plan.reg_m == 1 and (plan.reg_n == 1 or (extent and extent % plan.reg_n == 0))

            catalog = tuple(dict.fromkeys(plan for plan in scalar_tile_moves() if legal(plan)))
            return tuple(
                ProjectionSchedule(plan)
                for plan in self._select_plans(self._named("TILE"), catalog, allowed=lambda p: legal(p) and p in _scalar_catalog())
            )
        reductions = self._reductions()
        facts = tile.contractions.get(self.id)
        if facts is None:
            choices: Iterator[ReductionSchedule] = (ReductionSchedule(Tile(), reduction) for reduction in reductions)
        else:
            atoms = self.problem.atoms_of(self.id)
            plans = self._select_plans(
                self._named("TILE"),
                _contraction_plans(node, facts, self.problem.policy_atoms(self.id)),
                allowed=lambda plan: _contraction_plan_allowed(node, facts, atoms, plan),
            )
            # A tiled plan folds serially per cell; an untiled one takes every per-cell reduction.
            choices = (
                ReductionSchedule(plan, reduction) for plan in plans for reduction in (reductions if not plan.is_tiled else (Reduce(),))
            )
        return tuple(choice for choice in choices if self._placed_ok(choice))

    def _reductions(self) -> tuple[Reduce, ...]:
        tile, node = self.problem.tile, self.node
        facts = tile.contractions.get(self.id)
        catalog = _reduction_domain(tile, node) if facts is None else _contraction_reductions(tile, node, facts)

        def parse(spelling: str) -> Reduce | None:
            try:
                return Reduce.parse(spelling, self.problem.work)
            except ValueError:
                return None

        return _select(
            self._named("REDUCE"),
            catalog,
            parse=parse,
            allowed=lambda reduction: reduction in catalog,
            spell=Reduce.spell,
            bare=self.problem.bare_value("REDUCE", self.keys),
            validate_pins=self.problem.strict(classic_node_key(tile, "REDUCE", self.id)),
        )

    def _placed_ok(self, choice: NodeSchedule) -> bool:
        tile, node = self.problem.tile, self.node
        geometry = tile.grid_sched.placed(node, choice.tile)
        if choice.tile.is_tiled and not isinstance(geometry, PlacedTile):
            return False
        facts = tile.contractions.get(self.id)
        return not (
            isinstance(geometry, PlacedTile)
            and facts is not None
            and _plan_node_refusal(tile, node, choice.tile, geometry, facts) is not None
        )

    @cached_property
    def warp_eligible(self) -> bool:
        """Whether the CATALOG holds a warp plan for this site — a property of the offered space,
        read the same way whatever the row names, and found without walking the scalar tier."""
        tile, node = self.problem.tile, self.node
        facts = tile.contractions.get(self.id)
        if facts is None or self.problem.domains_literal is not None:
            return any(choice.tile.is_warp for choice in self.nodes)
        return any(self._placed_ok(ReductionSchedule(plan, Reduce())) for plan in _warp_plans(node, facts, self.problem.atoms_of(self.id)))

    @cached_property
    def node_set(self) -> frozenset[NodeSchedule]:
        return frozenset(self.nodes)

    @cached_property
    def edges(self) -> tuple[EdgeSchedule, ...]:
        """The transport choices of every incident edge — one tuple, shared by all of them."""
        literal = self.problem.domains_literal
        incident = self.problem.tile.incident_edges[self.id]
        if literal is not None:
            return self._literal("STAGE", literal.edges[incident[0]], lambda choice: choice.stage.spell()) if incident else ()
        if not incident:
            return ()
        tile, target, node = self.problem.tile, self.problem.target, self.node
        if self.id not in tile.contractions:
            return (EdgeSchedule(Stage.direct()),)
        candidates = {choice: _stage_candidates(tile, target, node, choice) for choice in self.nodes}
        catalog = tuple(dict.fromkeys(EdgeSchedule(stage) for stages in candidates.values() for stage in stages))

        def parse(spelling: str) -> EdgeSchedule | None:
            try:
                stage = Stage.parse(spelling)
            except ValueError:
                return None
            if target is not None and (why := stage_target(stage, target)):
                if self.problem.loud_pins:
                    raise ValueError(why)  # a spelling the card cannot run is wrong wherever it is published
                return None
            return EdgeSchedule(stage)

        return _select(
            None if self.stage_key is None else self.problem.row.get(self.stage_key),
            catalog,
            parse=parse,
            allowed=lambda choice: any(choice.stage in stages for stages in candidates.values()),
            spell=lambda choice: choice.stage.spell(),
            bare=self.problem.bare_value("STAGE", self.keys),
            validate_pins=self.problem.validate_pins,
        )

    @cached_property
    def edge_set(self) -> frozenset[EdgeSchedule]:
        return frozenset(self.edges)

    @cached_property
    def options(self) -> tuple[ClassicAssignment, ...]:
        """The site's picks: each node choice with each transport on every incident edge."""
        incident = self.problem.tile.incident_edges[self.id]
        edge_picks = tuple(frozendict({edge: choice for edge in incident}) for choice in self.edges) if incident else (frozendict(),)
        return tuple(Schedule(None, {self.id: node}, edges) for node in self.nodes for edges in edge_picks)


@dataclass(frozen=True, eq=False)
class ClassicKernelSite(Site[ClassicAssignment]):
    """The kernel-level factor: the worker inventory and raster, spelled bare (``WORK``,
    ``RASTER``). Its catalog is what the node sites' choices imply, so it is the last site."""

    problem: ClassicProblem

    @property
    def keys(self) -> tuple[str, ...]:
        return ("WORK", "RASTER")

    @cached_property
    def kernels(self) -> tuple[KernelSchedule, ...]:
        literal = self.problem.domains_literal
        if literal is not None:
            kernels = literal.kernel
            for key, spell in (("WORK", lambda kernel: kernel.work.spell()), ("RASTER", lambda kernel: kernel.raster.spell())):
                named = self.problem.row.get(key)
                if named is not None:
                    kept = tuple(kernel for kernel in kernels if spell(kernel) == named)
                    kernels = kept if kept or self.problem.strict(key) else kernels
            return kernels
        return tuple(KernelSchedule(work, raster) for work in self._works() for raster in self._rasters())

    def _inventories(self) -> Iterator[Work]:
        for site in self.problem.node_sites:
            for choice in site.nodes:
                work = derive_inventory((choice.tile,), coop=choice.reduce.coop if isinstance(choice, ReductionSchedule) else 1)
                if work is not None:
                    yield work

    def _sweep_widths(self) -> set[Work]:
        # A kernel whose work IS its shared output sweep — a bare elementwise map, the half a
        # placement cut leaves behind a reduction — has no site that folds out an inventory, so the
        # derived domain is the direct per-cell form alone: one worker per output cell with the
        # sweep serial inside it. Offer the widths a cooperative reduction would, so the sweep can
        # be split across workers (``_factor`` distributes it through ``_lane_close``). This widens
        # the worker inventory only; the grid stays the cell count, unlike promoting the axis into
        # ``place.free``, which multiplies the grid by its extent.
        tile = self.problem.tile
        if not (no_site_claims_inventory(tile) and tile.output_specs):
            return set()
        shared = set.intersection(*({axis.name for axis in store.sweep} for store in tile.output_specs))
        if not shared:
            return set()
        return {Work(kind="thread", units=(move.coop, 1)) for move in coop_reduce_moves() if move.coop > 1}

    def _works(self) -> tuple[Work, ...]:
        def catalog() -> tuple[Work, ...]:
            domain = {Work(), *self._inventories(), *self._sweep_widths()}
            return tuple(
                sorted(
                    {
                        Work(kind=work.kind, units=work.units, producer=producer)
                        for work in domain
                        for producer in (producer_band_moves() if work.kind == "warp" else (0,))
                    },
                    key=lambda work: work.spell(),
                )
            )

        def allowed(work: Work) -> bool:
            if work.producer and (work.kind != "warp" or work.producer not in producer_band_moves()):
                return False
            bare = Work(kind=work.kind, units=work.units)
            return bare == Work() or bare in self._sweep_widths() or any(inventory == bare for inventory in self._inventories())

        named = self.problem.row.get("WORK")
        if named is not None:
            work = self.problem.work
            if work is not None and allowed(work):
                return (work,)
            if self.problem.strict("WORK"):
                return ()
        return catalog()

    def _rasters(self) -> tuple[Raster, ...]:
        tile = self.problem.tile
        values = (
            raster_moves()
            if any(view.as_contraction() is not None for view in tile.views) and all(axis.extent.is_static for axis in tile.place.free)
            else ("",)
        )
        named = self.problem.row.get("RASTER")
        if named is not None:
            if named in values:
                return (Raster.parse(named),)
            if self.problem.strict("RASTER"):
                return ()
        # A transposed raster is never the catalog's own offer: it is taken only where a row names
        # it, the reading the restriction gave the ``gn`` values.
        return tuple(Raster.parse(value) for value in values if not value.startswith("gn"))

    @cached_property
    def kernel_set(self) -> frozenset[KernelSchedule]:
        return frozenset(self.kernels)

    @cached_property
    def options(self) -> tuple[ClassicAssignment, ...]:
        return tuple(Schedule(kernel, {}, {}) for kernel in self.kernels)


@dataclass(frozen=True, eq=False)
class ClassicProblem(ScheduleProblem[ClassicAssignment]):
    """``p + t`` and the row: one unscheduled ``TileOp``, its target, and the knob row whose
    values the sites offer where it names them. ``domains_literal`` substitutes hand-written
    factors for the projection — the tests' literal oracle. The precision policy and the pin
    reading (``validate_pins``: a named value the site cannot take empties it, else the site keeps
    its catalog — the reading a row published across peer kernels takes) are the problem's
    parameters, because they change what a site offers."""

    tile: TileOp
    target: object = None
    row: Mapping[str, str] = field(default_factory=frozendict)
    domains_literal: ClassicDomains | None = None
    allow_f16_accumulate: bool = True
    allow_fp8: bool = True
    validate_pins: bool = True
    #: A split's finalize reads a bare WORK / RASTER / REDUCE pin as the partial's: it spells its
    #: reduce serially only and its work at the thread level, so a warp WORK or a band names its
    #: sibling, and the finalize keeps its own catalog instead of offering nothing.
    tolerate_kernel_pins: bool = False
    #: Whether a named value the rules refuse outright — a warp-group tile its grid cannot feed, a
    #: transport the card cannot run, a stage no support resolves — is an error naming the rule.
    #: True for a hand pin, which is wrong wherever it is published; ``with_row`` turns it off, since
    #: a row a descent follows is answered by an empty site and the caller re-decides.
    loud_pins: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "row", frozendict({str(key): str(value) for key, value in self.row.items()}))

    def strict(self, key: str) -> bool:
        """Whether a named value at ``key`` the site cannot take empties the site (else the site
        keeps its catalog): pins are validated, and a bare kernel-family pin is not tolerated."""
        return self.validate_pins and not (self.tolerate_kernel_pins and key in ("WORK", "RASTER", "REDUCE"))

    def with_row(self, row: Mapping[str, str]) -> ClassicProblem:
        return replace(self, row=frozendict({**self.row, **{str(key): str(value) for key, value in row.items()}}), loud_pins=False)

    @cached_property
    def node_sites(self) -> tuple[ClassicNodeSite, ...]:
        return tuple(ClassicNodeSite(self, site) for site in self.tile.node_sites)

    @cached_property
    def kernel_site(self) -> ClassicKernelSite:
        return ClassicKernelSite(self)

    @cached_property
    def sites(self) -> tuple[Site[ClassicAssignment], ...]:
        return (*self.node_sites, self.kernel_site)

    @cached_method
    def node_site(self, site: NodeId) -> ClassicNodeSite:
        return self.node_sites[self.tile.node_sites.index(site)]

    @cached_property
    def work(self) -> Work | None:
        """The row's ``WORK`` inventory, the one every parsed site value decodes against."""
        named = self.row.get("WORK")
        if named is None:
            return None
        try:
            return Work.parse(named)
        except ValueError:
            return None

    @cached_property
    def allowed_works(self) -> frozenset[tuple[str, tuple[int, ...]]] | None:
        """The inventories a kernel may still take when the row names ``WORK`` — what lets a node
        pick that cannot reach it be refused before its subtree."""
        if "WORK" not in self.row:
            return None
        return frozenset((kernel.work.kind, kernel.work.units) for kernel in self.kernel_site.kernels)

    @cached_method
    def atoms_of(self, site: NodeId) -> tuple[str, ...]:
        """The tensor-core atoms the static facts allow at one contraction site."""
        return _warp_atoms(self.tile, self.target, self.tile.sites[site].node)

    @cached_method
    def policy_atoms(self, site: NodeId) -> tuple[str, ...]:
        """The atoms the unpinned catalog offers: :meth:`atoms_of` under the precision policy."""
        return tuple(
            name
            for name in self.atoms_of(site)
            if _atom_policy_ok(ATOM_REGISTRY[name], allow_f16_accumulate=self.allow_f16_accumulate, allow_fp8=self.allow_fp8)
        )

    @cached_property
    def bare_pins(self) -> frozendict[str, str]:
        """The bare pins of ambiguous families: a family key the row spells while this kernel
        spells the family at several sites. Such a pin names ONE site — some site carries the
        value and every other is OFF — so each site offers the value beside OFF, and
        :meth:`unrealized_bare_pin` asks the completed schedule which site carried it."""
        keys = [key for site in self.node_sites for key in site.keys]
        return frozendict(
            {
                family: value
                for family in ("TILE", "REDUCE", "STAGE")
                if (value := self.row.get(family)) is not None and sum(1 for key in keys if key.partition("@")[0] == family) > 1
            }
        )

    def bare_value(self, family: str, keys: tuple[str, ...]) -> str | None:
        """The bare pin that reaches a site spelling ``family`` at a scoped key, or ``None``."""
        value = self.bare_pins.get(family)
        if value is None or not any(key.partition("@")[0] == family for key in keys):
            return None
        return value

    def unrealized_bare_pin(self, assignment: ClassicAssignment) -> str | None:
        """Why a completed schedule leaves a bare pin unrealized — the half of the bare reading
        a site cannot decide alone. A pin no site can offer is ignored unless pins are validated,
        the reading a row published across the peer kernels of a multi-kernel target takes."""
        for family, value in self.bare_pins.items():
            if not value or value in self._spelled(assignment, family).values():
                continue
            if self.validate_pins or any(self._site_offers(site, family, value) for site in self.node_sites):
                return f"bare {family} pin {value} is realized by no site of this kernel"
        return None

    @staticmethod
    def _site_offers(site: ClassicNodeSite, family: str, value: str) -> bool:
        if family == "STAGE":
            return any(choice.stage.spell() == value for choice in site.edges)
        if family == "TILE":
            return any(choice.tile.spell() == value for choice in site.nodes)
        return any(isinstance(choice, ReductionSchedule) and choice.reduce.spell() == value for choice in site.nodes)

    def _spelled(self, assignment: ClassicAssignment, family: str) -> dict[str, str]:
        tile = self.tile
        if family == "TILE":
            return {classic_node_key(tile, "TILE", site): assignment.nodes[site].tile.spell() for site in tile.family_sites["TILE"]}
        if family == "REDUCE":
            return {
                classic_node_key(tile, "REDUCE", site): node.reduce.spell()
                for site in tile.family_sites["REDUCE"]
                if isinstance(node := assignment.nodes[site], ReductionSchedule)
            }
        return {classic_stage_key(tile, edge): assignment.edges[edge].stage.spell() for edge in tile.stage_edges}

    @cached_property
    def warp_eligible(self) -> bool:
        """Whether any site's catalog holds a warp plan — the offered space's own property."""
        return any(site.warp_eligible for site in self.node_sites)

    @cached_property
    def domains(self) -> ClassicDomains:
        """The literal independent factors — the product every enumeration is a subset of.
        Raises :class:`ClassicProjectionError` when a site offers nothing."""
        if self.domains_literal is not None:
            return self.domains_literal
        for site in self.node_sites:
            if not site.nodes or (self.tile.incident_edges[site.id] and not site.edges):
                raise ClassicProjectionError(f"classic site {node_id_spelling(site.id)} has no locally supported choice")
        if not self.kernel_site.kernels:
            raise ClassicProjectionError("classic kernel site has no locally supported choice")
        edges = {}
        for site in self.node_sites:
            edges.update({edge: site.edges for edge in self.tile.incident_edges[site.id]})
        return ClassicDomains(kernel=self.kernel_site.kernels, nodes={site.id: site.nodes for site in self.node_sites}, edges=edges)

    @cached_property
    def bounds(self) -> tuple[int, int]:
        size = descent = len(self.kernel_site.kernels)
        for site in self.node_sites:
            incident = len(self.tile.incident_edges[site.id])
            size *= len(site.nodes) * (len(site.edges) ** incident)
            descent += len(site.nodes) * max(len(site.edges), 1)
        return size, descent


def project_classic(tile: TileOp, target, row: Mapping[str, str] | None = None) -> ClassicDomains:
    """The independent kernel, node, and edge domains of one unscheduled tile: the catalog, or —
    under ``row`` — the row's values at the sites it names."""
    return ClassicProblem(tile, target, row=frozendict(row or {})).domains
