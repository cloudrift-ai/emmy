"""The source of every classic candidate: ``ClassicProblem`` — the tile, the target and the knob row — factored
into ``ClassicNodeSite``s and one ``ClassicKernelSite``. A site the row names offers the row's value alone,
parsed and checked with the catalog's own rules; a site the row leaves free offers its catalog. A node site
holds one record per node choice (``_Choice``) with the ``p + t`` facts that are the tile's alone and, derived
only when asked, its supports over the site's transports — so a prefix filters choices and a draw derives the
supports it touches, never a site's product."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING

from frozendict import frozendict

from emmy.compiler.ir.atom import ATOM_REGISTRY
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.schedule.base import ScheduleProblem, ScheduleRefused, Site, note_pin_refusal
from emmy.compiler.ir.schedule.catalog import map_tile_moves, producer_band_moves, raster_moves
from emmy.compiler.ir.schedule.choices import PlacedTile, Raster, Reduce, Stage, Tile, Work, derive_inventory, resolve_site_tile
from emmy.compiler.ir.schedule.staging import stage_target
from emmy.compiler.ir.schedule.views import EdgeSite, NodeId
from emmy.utils import cached_method

from .refusals import (
    _atom_policy_ok,
    _AxisAgreement,
    _contraction_plan_refusal,
    _contraction_plans,
    _contraction_reductions,
    _fragment_agreements,
    _FragmentAgreement,
    _multi_fold_direct_refusal,
    _needs_fill,
    _paired_budget_refusal,
    _plan_node_refusal,
    _reduction_domain,
    _Relation,
    _relation_refusal,
    _resolve_stage,
    _stage_candidates,
    _warp_atoms,
    _warp_plans,
    _wgmma_refusal,
    fill_stage_moves,
    fill_tma_moves,
)
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
    output_sweep_works,
    packed_works,
)

if TYPE_CHECKING:
    from emmy.compiler.context import Context
    from emmy.compiler.ir.tile import TileOp


def _select[T](
    named: str | None,
    catalog: Iterator[T] | tuple[T, ...],
    *,
    parse: Callable[[str], T | None],
    allowed: Callable[[T], bool],
    spell: Callable[[T], str],
    bare: str | None,
    validate_pins: bool,
    exact: bool = False,
    key: str | None = None,
    why: Callable[[T], str | None] | None = None,
) -> tuple[T, ...]:
    """One factor's values under the row.

    A NAMED value (the row spells this site's exact key) is parsed and checked; that one value is
    the factor. A spelling the parser cannot read alone — a warp tile with no ``WORK`` beside it —
    is matched against the catalog by spelling instead, still one value. A named value the site
    cannot take empties the factor, or, when pins are not validated (a row published across the
    peer kernels of a multi-kernel target), leaves the catalog whole. A BARE pin of an ambiguous
    family names one site among several: this factor keeps the pin's value and OFF, and the
    completed schedule is asked which site carried it. An ``exact`` replay supplies the complete
    row, so a value that does not parse and pass its intrinsic checks can return empty without the
    catalog-spelling fallback needed by partial hand pins.

    A named value the factor does not keep is recorded against ``key`` (:func:`note_pin_refusal`) with
    ``why``'s reason, so the pin check can say which rule refused it."""
    out = _select_values(named, catalog, parse=parse, allowed=allowed, spell=spell, bare=bare, validate_pins=validate_pins, exact=exact)
    if named is not None and key is not None and not any(spell(choice) == named for choice in out):
        value = parse(named)
        reason = "it does not parse at this site" if value is None else (why(value) if why is not None else None)
        note_pin_refusal(key, named, reason or "this site does not offer it")
    return out


def _select_values[T](
    named: str | None,
    catalog: Iterator[T] | tuple[T, ...],
    *,
    parse: Callable[[str], T | None],
    allowed: Callable[[T], bool],
    spell: Callable[[T], str],
    bare: str | None,
    validate_pins: bool,
    exact: bool = False,
) -> tuple[T, ...]:
    if named is not None:
        value = parse(named)
        if value is not None and allowed(value):
            return (value,)
        if exact:
            return ()
    values = tuple(catalog)  # the catalog is walked only past the named fast path
    if named is not None:
        matched = tuple(choice for choice in values if spell(choice) == named)
        if matched or validate_pins:
            return matched
    return values if bare is None else tuple(choice for choice in values if spell(choice) in ("", bare))


@dataclass(frozen=True)
class _LocalSupport:
    """One node choice with its incident edge choices, resolved: the compatibility evidence a prefix reads
    (the inventory it claims, its axis and seam claims) beside the choices themselves. Not a schedule — placed
    geometry and fragment facts never enter a :class:`Schedule` value."""

    node: NodeSchedule
    edges: Mapping[EdgeSite, EdgeSchedule]
    work: Work | None = None
    axes: tuple[_AxisAgreement, ...] = ()
    fragments: tuple[_FragmentAgreement, ...] = ()
    raster_eligible: bool = False
    producer_eligible: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "edges", frozendict(self.edges))


def _producers(tile_op) -> frozenset[NodeId]:
    return frozenset(facts.need for facts in tile_op.contractions.values() if facts.need is not None)


def local_support(
    tile_op, target, site: NodeId, node: NodeSchedule, edges: Mapping[EdgeSite, EdgeSchedule], *, geometry=None, plan_checked=False
) -> _LocalSupport | None:
    """The ``p + t`` support of one node choice with its incident edge choices, or ``None`` where the pair
    resolves to nothing — the one statement of that derivation: a site's choice derives its supports through
    it, and a context without a problem (the codec validating a complete row) asks it directly. With no
    target, the kernel's own materialization stands in for the resolver."""
    facts = tile_op.contractions.get(site)
    if facts is None:
        return _intrinsic_support(tile_op, target, site, node, edges)
    materialization = getattr(tile_op, "materialization", None)
    if target is None and (
        materialization is None
        or (node.tile.is_tiled and site not in materialization.tiles)
        or any(not choice.stage.is_direct and edge not in materialization.stages for edge, choice in edges.items())
    ):
        return None
    fold = tile_op.sites[site].node
    view = tile_op.views[site]
    if set(edges) != set(tile_op.incident_edges[site]):
        return None
    if len(set(edges.values())) > 1:
        raise ScheduleRefused(f"{node_id_spelling(site)}: one contraction currently requires one transport choice across its operands")
    if geometry is None:
        geometry = tile_op.grid_sched.placed(fold, node.tile)
    if not plan_checked and _plan_refused(tile_op, site, node, geometry):
        return None
    stage = next(iter(edges.values())).stage if edges else Stage.direct()
    resolved_stage = None
    if _wgmma_refusal(node.tile, stage) is not None or _multi_fold_direct_refusal(fold, node.tile, stage) is not None:
        return None
    if view.as_contraction() is None or not node.tile.is_tiled:
        if not stage.is_direct:
            return None
    elif target is None:
        resolved = {materialization.stages.get(edge) for edge in edges} if materialization is not None else set()
        resolved.discard(None)
        resolved_stage = next(iter(resolved)) if len(resolved) == 1 else None
    elif _needs_fill(tile_op, fold, node.tile):
        packed_copy = tile_op.packed_reading(fold)[0] is not None and stage.transport in ("smem-async", "smem-tma")
        if not packed_copy and stage not in (*fill_stage_moves(), *fill_tma_moves(target)):
            return None
        resolved_stage = _resolve_stage(tile_op, target, fold, node.tile, geometry, stage, facts)
    elif not stage.is_direct:
        resolved_stage = _resolve_stage(tile_op, target, fold, node.tile, geometry, stage, facts)
    if not stage.is_direct and (resolved_stage is None or resolved_stage.choice != stage):
        return None
    if isinstance(geometry, PlacedTile) and _paired_budget_refusal(fold, facts.producer, geometry, resolved_stage) is not None:
        return None
    return _LocalSupport(
        node,
        edges,
        work=derive_inventory((node.tile,), coop=node.reduce.coop if isinstance(node, ReductionSchedule) else 1),
        axes=(
            tuple(_AxisAgreement(side.axis.name, side.tile, side.units) for side in geometry.mn)
            if node.tile.is_tiled and isinstance(geometry, PlacedTile)
            else ()
        ),
        fragments=(
            _fragment_agreements(site, fold, node.tile, geometry, resolved_stage, facts, _producers(tile_op))
            if isinstance(geometry, PlacedTile)
            else ()
        ),
        raster_eligible=node.tile.is_tiled and view.as_contraction() is not None,
        # A producer band splits the staged K-loop's phases across warp bands, which only the
        # contraction tier's skeleton drives; the chunk tier runs every warp through one uniform
        # ring, where an aux band decoding onto warp 0 would re-issue its elected TMA arrive.
        # TMA copies beside a compute fill (a packed weight's scales, a computed activation)
        # run as two groups of one uniform loop, which has no band split either.
        producer_eligible=not fold.chunked() and not (stage.transport == "smem-tma" and _needs_fill(tile_op, fold, node.tile)),
    )


def _plan_refused(tile_op, site: NodeId, node: NodeSchedule, geometry) -> bool:
    """Whether a contraction site's tile is refused whatever transport feeds it."""
    if node.tile.is_tiled and not isinstance(geometry, PlacedTile):
        return True
    return isinstance(geometry, PlacedTile) and (
        _plan_node_refusal(tile_op, tile_op.sites[site].node, node.tile, geometry, tile_op.contractions[site]) is not None
    )


def _intrinsic_support(tile_op, target, site: NodeId, node: NodeSchedule, edges: Mapping[EdgeSite, EdgeSchedule]) -> _LocalSupport | None:
    """The target-independent local relation of a site that contracts nothing."""
    if site not in tile_op.family_sites["TILE"] and node.tile != Tile():
        return None
    if node.tile.is_warp and hasattr(target, node.tile.atom.target_feature) and not node.tile.atom.available_on(target):
        return None
    if len({choice.stage for choice in edges.values()}) > 1:
        raise ScheduleRefused(f"{node_id_spelling(site)}: one contraction currently requires one transport choice across its operands")
    if any(edge not in tile_op.stage_edges and not choice.stage.is_direct for edge, choice in edges.items()):
        return None
    if any(not choice.stage.is_direct and not node.tile.is_tiled for choice in edges.values()):
        return None
    if any(
        not choice.stage.is_direct and hasattr(target, "has_cp_async") and not choice.stage.available_on(target)
        for choice in edges.values()
    ):
        return None
    try:
        work = derive_inventory((node.tile,), coop=node.reduce.coop if isinstance(node, ReductionSchedule) else 1)
    except ValueError:
        return None
    return _LocalSupport(node, edges, work=work, raster_eligible=node.tile.is_tiled and tile_op.views[site].as_contraction() is not None)


@dataclass(frozen=True, eq=False)
class _Choice:
    """One node choice at a site: the ``p + t`` facts that are the tile's alone — the inventory it claims,
    its placed geometry and axis agreements, the seam claims that read no transport — and, derived only when
    asked, its supports, the choice paired with each transport of the site's edge catalog that resolves."""

    site: ClassicNodeSite
    node: NodeSchedule

    @property
    def edges(self) -> Mapping[EdgeSite, EdgeSchedule]:
        """A choice names no transport; a support does."""
        return frozendict()

    @cached_property
    def geometry(self):
        return self.site.problem.tile.grid_sched.placed(self.site.node, self.node.tile)

    @cached_property
    def work(self) -> Work | None:
        try:
            return derive_inventory((self.node.tile,), coop=self.node.reduce.coop if isinstance(self.node, ReductionSchedule) else 1)
        except ValueError:
            return None

    @cached_property
    def axes(self) -> tuple[_AxisAgreement, ...]:
        if not (self.node.tile.is_tiled and isinstance(self.geometry, PlacedTile)):
            return ()
        return tuple(_AxisAgreement(side.axis.name, side.tile, side.units) for side in self.geometry.mn)

    @cached_property
    def fragments(self) -> tuple[_FragmentAgreement, ...]:
        """The seam claims the tile decides: its offer, and a chunked carrier's need. The ordinary need reads
        its transport's K slab, so it is a support's claim, checked where the support is."""
        tile = self.site.problem.tile
        facts = tile.contractions.get(self.site.id)
        if facts is None or not isinstance(self.geometry, PlacedTile):
            return ()
        claims = _fragment_agreements(self.site.id, self.site.node, self.node.tile, self.geometry, None, facts, _producers(tile))
        return tuple(claim for claim in claims if claim.role == "offer" or claim.value[0] == "chunk")

    @cached_property
    def plan_refused(self) -> bool:
        """Whether the tile is refused before any transport is asked — checked once, not per edge pick."""
        tile = self.site.problem.tile
        return self.site.id in tile.contractions and _plan_refused(tile, self.site.id, self.node, self.geometry)

    @cached_method
    def support(self, edges: Mapping[EdgeSite, EdgeSchedule]) -> _LocalSupport | None:
        """This choice with one transport on every incident edge, resolved — once per edge pick."""
        if self.plan_refused:
            return None
        return local_support(
            self.site.problem.tile, self.site.problem.target, self.site.id, self.node, edges, geometry=self.geometry, plan_checked=True
        )

    @cached_property
    def supports(self) -> tuple[_LocalSupport, ...]:
        if self.plan_refused:
            return ()
        return tuple(support for edges in self.site.edge_picks if (support := self.support(edges)) is not None)


@dataclass(frozen=True, eq=False)
class ClassicNodeSite(Site[ClassicSchedule]):
    """One node site's independent factor: its node choices and the transport catalog of its incident
    edges, each node choice a :class:`_Choice` whose supports are derived when a prefix reaches it. Every
    derived read is memoized on the site, including the choices a relation admits."""

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

    def _select_plans(self, named: str | None, catalog, *, allowed, work: Work | None = None, why=None) -> tuple[Tile, ...]:
        if named is not None:
            self._wgmma_pin_refusal(named)

        def parse(spelling: str) -> Tile | None:
            try:
                return resolve_site_tile(spelling, work)
            except ValueError:
                return None

        key = classic_node_key(self.problem.tile, "TILE", self.id)
        return _select(
            named,
            catalog,
            parse=parse,
            allowed=allowed,
            spell=Tile.spell,
            bare=self.problem.bare_value("TILE", self.keys),
            validate_pins=self.problem.strict(key),
            exact=self.problem._exact(key),
            key=key,
            why=why,
        )

    @cached_property
    def nodes(self) -> tuple[NodeSchedule, ...]:
        """The node choices: the row's value where it names this site, else the catalog."""
        tile, node = self.problem.tile, self.node
        view = tile.views[self.id]
        if view.axis is None:
            if self.id not in tile.family_sites["TILE"] or not tile.place.free:
                return (ProjectionSchedule(Tile()),)
            inner = tile.place.free[-1]
            extent = inner.extent.as_static() if inner.extent.is_static else 0

            catalog = tuple(plan for plan in map_tile_moves() if plan.reg_n == 1 or (extent and extent % plan.reg_n == 0))
            # A strip never carries the worker inventory: the row's WORK is the kernel's sweep
            # width, so a named strip parses bare, as the catalog spells it.
            return tuple(
                ProjectionSchedule(plan) for plan in self._select_plans(self._named("TILE"), catalog, allowed=lambda p: p in catalog)
            )
        reductions = self._reductions()
        facts = tile.contractions.get(self.id)
        if facts is None:
            choices: Iterator[ReductionSchedule] = (ReductionSchedule(Tile(), reduction) for reduction in reductions)
        else:
            # A hand-pinned tile is an authored choice past the precision policy; a followed row is not.
            followed = self.id in tile.family_sites["TILE"] and self.problem.followed(classic_node_key(tile, "TILE", self.id))
            atoms = self.problem.policy_atoms(self.id) if followed else self.problem.atoms_of(self.id)
            plans = self._select_plans(
                self._named("TILE"),
                _contraction_plans(node, facts, self.problem.policy_atoms(self.id)),
                allowed=lambda plan: _contraction_plan_refusal(node, facts, atoms, plan) is None,
                work=self.problem.work,
                why=lambda plan: _contraction_plan_refusal(node, facts, atoms, plan),
            )
            # A tiled plan folds serially per cell; an untiled one takes every per-cell reduction. So a
            # row whose reductions leave out the serial fold (a pinned ``coop`` band) rules the tiled
            # plans out too, instead of letting one realize the pin's site with a serial fold.
            serial = Reduce() in reductions
            choices = (
                ReductionSchedule(plan, reduction)
                for plan in plans
                if serial or not plan.is_tiled
                for reduction in (reductions if not plan.is_tiled else (Reduce(),))
            )
        return tuple(choice for choice in choices if self._placed_ok(choice))

    def _reductions(self) -> tuple[Reduce, ...]:
        tile, node, target = self.problem.tile, self.node, self.problem.target
        facts = tile.contractions.get(self.id)
        catalog = _reduction_domain(tile, node, target) if facts is None else _contraction_reductions(tile, node, facts, target)

        def parse(spelling: str) -> Reduce | None:
            try:
                return Reduce.parse(spelling, self.problem.work)
            except ValueError:
                return None

        key = classic_node_key(tile, "REDUCE", self.id)
        return _select(
            self._named("REDUCE"),
            catalog,
            parse=parse,
            allowed=lambda reduction: reduction in catalog,
            spell=Reduce.spell,
            bare=self.problem.bare_value("REDUCE", self.keys),
            validate_pins=self.problem.strict(key),
            exact=self.problem._exact(key),
            key=key,
            why=lambda reduction: f"this reduction offers {', '.join(r.spell() or 'serial' for r in catalog)}",
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
        if facts is None:
            return any(choice.tile.is_warp for choice in self.nodes)
        return any(self._placed_ok(ReductionSchedule(plan, Reduce())) for plan in _warp_plans(node, facts, self.problem.atoms_of(self.id)))

    @cached_property
    def node_set(self) -> frozenset[NodeSchedule]:
        return frozenset(self.nodes)

    @cached_property
    def edges(self) -> tuple[EdgeSchedule, ...]:
        """The transport choices of every incident edge — one tuple, shared by all of them."""
        incident = self.problem.tile.incident_edges[self.id]
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
            validate_pins=False if self.stage_key is None else self.problem.strict(self.stage_key),
            exact=self.stage_key is not None and self.problem._exact(self.stage_key),
            key=self.stage_key,
            why=lambda choice: (
                "register storage (d1/reg) is offered to a serial kernel's register program only"
                if choice.stage.transport == "reg"
                else "no tile this site offers is fed by that transport"
            ),
        )

    @cached_property
    def edge_set(self) -> frozenset[EdgeSchedule]:
        return frozenset(self.edges)

    @cached_property
    def edge_picks(self) -> tuple[Mapping[EdgeSite, EdgeSchedule], ...]:
        """Each transport of the catalog on every incident edge — the edge half of a support."""
        incident = self.problem.tile.incident_edges[self.id]
        return tuple(frozendict({edge: choice for edge in incident}) for choice in self.edges) if incident else (frozendict(),)

    @cached_method
    def choice(self, node: NodeSchedule) -> _Choice:
        return _Choice(self, node)

    @cached_property
    def choices(self) -> tuple[_Choice, ...]:
        return tuple(self.choice(node) for node in self.nodes)

    def refusal(self, pick: _Choice | _LocalSupport, relation: _Relation) -> str | None:
        """Why ``pick`` cannot extend a prefix with these compatibility facts, or ``None`` — recorded against
        any pinned value the pick spells (:func:`note_pin_refusal`), so the pin check can name the rule."""
        tile = self.problem.tile
        why = _relation_refusal(
            self.id, pick, relation, roots=tile.shared_roots, pairs=tile.chain_pairs, allowed_works=self.problem.allowed_works
        )
        if why is None or not any(key in self.problem.row for key in self.keys):
            return why
        families = tile.family_sites
        spelled = {classic_node_key(tile, "TILE", self.id): pick.node.tile.spell()} if self.id in families["TILE"] else {}
        if isinstance(pick.node, ReductionSchedule) and self.id in families["REDUCE"]:
            spelled[classic_node_key(tile, "REDUCE", self.id)] = pick.node.reduce.spell()
        if self.stage_key is not None:
            for edge in pick.edges.values():
                spelled[self.stage_key] = edge.stage.spell()
        for key, value in spelled.items():
            if self.problem.row.get(key) == value:
                note_pin_refusal(key, value, why)
        return why

    @cached_property
    def _choices_by_work(self) -> Mapping[Work | None, tuple[_Choice, ...]]:
        """The choices by the inventory they claim — what a prefix that claimed one reads, beside the choices
        claiming none, instead of refusing every other inventory one choice at a time."""
        by_work: dict[Work | None, list[_Choice]] = {}
        for choice in self.choices:
            by_work.setdefault(choice.work, []).append(choice)
        return {work: tuple(choices) for work, choices in by_work.items()}

    @cached_method
    def compatible(self, relation: _Relation) -> tuple[_Choice, ...]:
        """The choices whose tile-level facts extend a prefix with these compatibility facts — one filter per
        relation, kept, so every prefix that agrees on the facts reads the same answer."""
        self._check_stage_pin()
        if relation.work is None:
            candidates = self.choices
        else:
            candidates = (*self._choices_by_work.get(None, ()), *self._choices_by_work.get(relation.work, ()))
        return tuple(choice for choice in candidates if self.refusal(choice, relation) is None)

    def admitted(self, choice: _Choice, relation: _Relation) -> tuple[_LocalSupport, ...]:
        """``choice``'s supports that extend the prefix: the claim only a support carries, checked here."""
        return tuple(support for support in choice.supports if self.refusal(support, relation) is None)

    @cached_method
    def frontier(self, relation: _Relation) -> tuple[_LocalSupport, ...]:
        """Every support that extends the prefix — what a walk reads; a draw never asks for it."""
        return tuple(support for choice in self.compatible(relation) for support in self.admitted(choice, relation))

    def _check_stage_pin(self) -> None:
        """A hand-pinned non-direct transport no support resolves is a wrong spelling, not an empty pool:
        raised with the rule's own message the first time a prefix reads this site."""
        problem = self.problem
        pinned = None if self.stage_key is None else problem.row.get(self.stage_key)
        if pinned and problem.loud_pins and problem.validate_pins and not any(choice.supports for choice in self.choices):
            raise ValueError(f"STAGE pin {pinned!r} does not resolve for this contraction")


@dataclass(frozen=True, eq=False)
class ClassicKernelSite(Site[ClassicSchedule]):
    """The kernel-level factor: the worker inventory and raster, spelled bare (``WORK``,
    ``RASTER``). Its catalog is what the node sites' choices imply, so it is the last site."""

    problem: ClassicProblem

    @property
    def keys(self) -> tuple[str, ...]:
        return ("WORK", "RASTER")

    @cached_property
    def kernels(self) -> tuple[KernelSchedule, ...]:
        return tuple(KernelSchedule(work, raster) for work in self._works() for raster in self._rasters())

    def _inventories(self) -> Iterator[Work]:
        for site in self.problem.node_sites:
            for choice in site.nodes:
                work = derive_inventory((choice.tile,), coop=choice.reduce.coop if isinstance(choice, ReductionSchedule) else 1)
                if work is not None:
                    yield work

    def _packed(self) -> set[Work]:
        # Several cells of a warp-wide cooperative reduce in one CTA (:func:`packed_works`).
        return {
            packed
            for site in self.problem.node_sites
            for choice in site.nodes
            if isinstance(choice, ReductionSchedule) and choice.reduce.coop > 1 and not choice.reduce.coop_transposed
            for packed in packed_works(Work(kind="thread", units=(choice.reduce.coop, 1)))
        }

    def _sweep_widths(self) -> set[Work]:
        # A kernel whose work IS its output sweep — a bare elementwise map or a placement residual
        # whose reduction sites stay serial — otherwise has one worker per output cell with the
        # sweep serial inside it. Offer the widths a cooperative reduction would, so each sibling
        # sweep can be split across workers (``_factor`` distributes it through ``_lane_close``).
        # This widens only the worker inventory; the grid stays the cell count.
        tile = self.problem.tile
        return set(output_sweep_works(tile, None))

    def _works(self) -> tuple[Work, ...]:
        def catalog() -> tuple[Work, ...]:
            domain = {Work(), *self._inventories(), *self._sweep_widths(), *self._packed()}
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
            return (
                bare == Work()
                or bare in self._sweep_widths()
                or bare in self._packed()
                or any(inventory == bare for inventory in self._inventories())
            )

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


@dataclass(frozen=True, eq=False)
class ClassicProblem(ScheduleProblem[ClassicSchedule]):
    """``p + t`` and the row: one unscheduled ``TileOp``, its target, and the knob row whose
    values the sites offer where it names them. The precision policy and the pin
    reading (``validate_pins``: a named value the site cannot take empties it, else the site keeps
    its catalog — the reading a row published across peer kernels takes) are the problem's
    parameters, because they change what a site offers."""

    tile: TileOp
    target: Context | None = None
    row: Mapping[str, str] = field(default_factory=frozendict)
    allow_f16_accumulate: bool = True
    allow_fp8: bool = True
    validate_pins: bool = True
    #: A split's finalize reads a bare WORK / RASTER / REDUCE pin as the partial's: it spells its
    #: reduce serially only and its work at the thread level, so a warp WORK or a band names its
    #: sibling, and the finalize keeps its own catalog instead of offering nothing.
    tolerate_kernel_pins: bool = False
    #: Row keys whose values must be accepted exactly. Strict replay adds only the keys it supplies,
    #: leaving unrelated inherited pins under their original published-row reading.
    _strict_row_keys: frozenset[str] = frozenset()
    #: Row keys a descent supplied rather than the hand pins the fork was built with. Such a row is
    #: evidence, and evidence obeys the precision policy: only a hand pin authors a tile past it.
    _followed_row_keys: frozenset[str] = frozenset()
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
        return key in self._strict_row_keys or (
            self.validate_pins and not (self.tolerate_kernel_pins and key in ("WORK", "RASTER", "REDUCE"))
        )

    def _exact(self, key: str) -> bool:
        """Whether strict replay supplied this exact row key."""
        return key in self._strict_row_keys and key in self._followed_row_keys

    def with_row(self, row: Mapping[str, str], *, strict: bool = False) -> ClassicProblem:
        supplied = {str(key): str(value) for key, value in row.items()}
        strict_row_keys = self._strict_row_keys | supplied.keys() if strict else self._strict_row_keys
        # ``self.row`` is the live hand-pin restriction installed when the schedule fork was
        # built. A measured/prior row narrows that fork to one leaf, but cannot overwrite the
        # restriction: hard pins are authoritative over every ranking source.
        followed = self._followed_row_keys | {key for key in supplied if key not in self.row}
        return replace(
            self,
            row=frozendict({**supplied, **self.row}),
            _strict_row_keys=frozenset(strict_row_keys),
            _followed_row_keys=frozenset(followed),
            loud_pins=False,
        )

    def followed(self, key: str) -> bool:
        """Whether the row's value at ``key`` (or its bare family) came from a descent, not a hand pin."""
        return bool({key, key.partition("@")[0]} & self._followed_row_keys)

    @cached_property
    def node_sites(self) -> tuple[ClassicNodeSite, ...]:
        return tuple(ClassicNodeSite(self, site) for site in self.tile.node_sites)

    @cached_property
    def kernel_site(self) -> ClassicKernelSite:
        return ClassicKernelSite(self)

    @cached_property
    def sites(self) -> tuple[Site[ClassicSchedule], ...]:
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

    def unrealized_bare_pin(self, schedule: ClassicSchedule) -> str | None:
        """Why a completed schedule leaves a bare pin unrealized — the half of the bare reading
        a site cannot decide alone. A pin no site can offer is ignored unless pins are validated,
        the reading a row published across the peer kernels of a multi-kernel target takes."""
        for family, value in self.bare_pins.items():
            if not value or value in self._spelled(schedule, family).values():
                continue
            if (
                self.validate_pins
                or family in self._strict_row_keys
                or any(self._site_offers(site, family, value) for site in self.node_sites)
            ):
                return f"bare {family} pin {value} is realized by no site of this kernel"
        return None

    @staticmethod
    def _site_offers(site: ClassicNodeSite, family: str, value: str) -> bool:
        if family == "STAGE":
            return any(choice.stage.spell() == value for choice in site.edges)
        if family == "TILE":
            return any(choice.tile.spell() == value for choice in site.nodes)
        return any(isinstance(choice, ReductionSchedule) and choice.reduce.spell() == value for choice in site.nodes)

    def _spelled(self, schedule: ClassicSchedule, family: str) -> dict[str, str]:
        tile = self.tile
        if family == "TILE":
            return {classic_node_key(tile, "TILE", site): schedule.nodes[site].tile.spell() for site in tile.family_sites["TILE"]}
        if family == "REDUCE":
            return {
                classic_node_key(tile, "REDUCE", site): node.reduce.spell()
                for site in tile.family_sites["REDUCE"]
                if isinstance(node := schedule.nodes[site], ReductionSchedule)
            }
        return {classic_stage_key(tile, edge): schedule.edges[edge].stage.spell() for edge in tile.stage_edges}

    @cached_property
    def warp_eligible(self) -> bool:
        """Whether any site's catalog holds a warp plan — the offered space's own property."""
        return any(site.warp_eligible for site in self.node_sites)

    @cached_property
    def bound(self) -> int:
        size = len(self.kernel_site.kernels)
        for site in self.node_sites:
            incident = len(self.tile.incident_edges[site.id])
            size *= len(site.nodes) * (len(site.edges) ** incident)
        return size
