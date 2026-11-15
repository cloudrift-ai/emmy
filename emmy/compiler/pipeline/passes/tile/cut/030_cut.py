"""Enumerate every kernel-set cut before any schedule is composed.

Stored-Fold-edge placement is first, transposed-constant layout second, and cross-CTA reduction splitting third.
The rule runs to a fixpoint, so each successful choice and every fresh piece re-enters these ordered
domains before scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.schedule import Schedule, ScheduleContext, ScheduleRefused, schedule
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.path import MissingSiteError, parse_key, resolve, sites
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.fork import DeferredFork
from emmy.compiler.pipeline.knob import axis_of, family_of, family_pins, pinned_kernel
from emmy.compiler.pipeline.passes.tile._cut import (
    PIECE,
    cuttable_seams,
    full_projection_seams,
    join_pairs,
    joined_halves,
    joined_spelling,
    output_map,
    realize,
)
from emmy.compiler.pipeline.passes.tile._layout import layout_forks
from emmy.compiler.pipeline.passes.tile._split import split_forks
from emmy.compiler.pipeline.passes.tile.cut import takes_kernel_set_decision
from emmy.compiler.pipeline.search.pins import composed_cuts_for, note_place_key

PATTERN = [Pattern("root", TileOp)]
FIXPOINT = True


@dataclass(frozen=True)
class _CutContext(ScheduleContext[DeferredFork, object, object]):
    """One immutable frontier over the cut pass's already-restricted structural choices."""

    choices: tuple[DeferredFork, ...]
    _schedule: Schedule = field(default_factory=lambda: Schedule(None, {}, {}), repr=False)

    @property
    def schedule(self) -> Schedule:
        return self._schedule

    def extensions(self):
        if self.schedule.kernel is None:
            for choice in self.choices:
                yield Schedule(choice, {}, {})

    def extend(self, pick: Schedule) -> _CutContext:
        if self.schedule.kernel is not None or pick.nodes or pick.edges or pick.kernel not in self.choices:
            raise ScheduleRefused("pick is outside the cut frontier")
        return replace(self, _schedule=pick)


def _cut_arm(root: Node, materialize, seams, joins=()) -> DeferredFork:
    """The arm that cuts ``seams`` of ``root``'s kernel: it spells every occurrence of each clustered
    value, names the seam each of those spellings stands for, and carries the redirect of the kernel's
    buffers onto its pieces' outputs. A joined pair is spelled by its one joined key."""
    joined = {id(seam) for pair in joins for seam in pair}
    knobs = {spelling: "cut" for seam in seams if id(seam) not in joined for spelling in (seam.spelling, *seam.aliases)}
    knobs.update((joined_spelling(producer, region), "cut") for producer, region in joins)
    aliases = {alias: seam.spelling for seam in seams if id(seam) not in joined for alias in seam.aliases}
    return DeferredFork(materialize, knobs, structural=True, aliases=aliases, output=output_map(root))


def _resolved_seams(tile: TileOp, key: str, all_sites, by_node, *, strict: bool = True) -> tuple | None:
    """The seams a key's sites address on ``tile`` (two for a joined key), or ``None`` when a site is not on this
    tree. A site that resolves to no cuttable Fold edge is an addressing error in a pin (``strict``); in a recorded
    route it is another kernel's seam, and the key resolves to ``None``."""
    out = []
    for half in joined_halves(key):
        try:
            site = resolve(tile.op, half, all_sites=all_sites)
        except MissingSiteError:
            return None
        if id(site.node) not in by_node:
            if strict:
                raise ValueError(f"PLACE pin {key!r} does not address a cuttable Fold edge in this kernel")
            return None
        out.append(by_node[id(site.node)])
    return tuple(out)


def _seam_index(seams) -> dict[int, object]:
    """Map every seam node and clustered sibling to its shared decision."""
    by_node = {}
    for seam in seams:
        by_node[id(seam.node)] = seam
        by_node.update((id(sibling), seam) for sibling, *_ in seam.siblings)
        by_node.update((id(sibling), seam) for sibling, _ in seam.indexed_siblings)
        by_node.update((id(member), seam) for group in seam.contraction for member, *_ in group.members if id(member) not in by_node)
    return by_node


def _rootmost(seams, all_sites, refuse: frozenset[str] = frozenset()):
    """Return the root-most seam not excluded by a scoped fuse pin."""
    kept = [seam for seam in seams if seam.spelling not in refuse]
    if not kept:
        return None
    depth = {id(site.node): site.depth for site in all_sites}
    return min(kept, key=lambda candidate: depth[id(candidate.node)])


def _placement_pins(tile: TileOp, piece: bool) -> tuple[tuple[tuple[str, str], ...], dict[str, str]]:
    """The PLACE pins this kernel reads, by their local key, and the original key of each kernel pin among them: the
    kernel pins naming it (``PLACE@<name>``, ``PLACE@<name>/<site>`` with the site spelled relative to it), else —
    for a kernel no cut minted (``piece``) — the pins every kernel reads. A piece of a cut is a new kernel: the pins
    its parent read never address it."""
    own = {
        "PLACE" + ("@" + site if site else ""): (key, value)
        for key, value in family_pins("PLACE", kernels=True)
        if pinned_kernel(key) == tile.name
        for site in ((axis_of(key) or "").partition("/")[2],)
    }
    if own:
        return tuple((local, value) for local, (_, value) in own.items()), {local: key for local, (key, _) in own.items()}
    return (() if piece else family_pins("PLACE")), {}


def _placement_restriction(tile: TileOp, seams, piece: bool) -> tuple[tuple, str, tuple] | None:
    """The authoritative placement spelled by live PLACE pins, or ``None``.

    This restriction is consumed entirely by the cut pass before classic schedule enumeration.
    Every scoped ``PLACE@site=cut`` pin that resolves on this kernel joins ONE composed decision —
    the seams all live on this kernel's tree, so one realization cuts them together. A bare ``PLACE=cut`` names no
    site: it cuts the root-most seam of every kernel no cut minted, once (``PLACE@<name>=cut`` cuts the one kernel of
    that name). A scoped pin whose site path does not exist on this kernel addresses another kernel of the graph, and a
    kernel none of the pins address takes its own fork. A pin that resolves to a site no cut realizes is an addressing
    error and raises. A scoped
    ``PLACE@site=fuse`` excludes that seam from the composed cut (alone, it decides fuse under
    that spelling). A bare cut supplies the primary root-most seam and composes with scoped cuts;
    a bare fuse applies only when no scoped pin addressed this kernel. A JOINED key
    (``PLACE@<workspace seam>+<output-owning seam>=cut``) cuts both seams and splices their pieces into one; the
    last element of the result lists those pairs. A pin decides the forks it names: every piece a pinned cut mints
    takes its own fork, which a pin naming the piece decides, else evidence or the prior."""
    scoped_pins, source_keys = _placement_pins(tile, piece)
    pins = [(name, value) for name, value in scoped_pins if family_of(name) == "PLACE"]
    if not pins:
        return None
    for _, value in pins:
        if value not in {"fuse", "cut"}:
            raise ValueError(f"bad PLACE value {value!r}; expected 'fuse' or 'cut'")
    all_sites = sites(tile.op)
    by_node = _seam_index(seams)
    cut: list = []
    joins: list = []
    fused: list[str] = []
    addressed: list[tuple[str, str, tuple]] = []
    missing = False
    for name, value in pins:
        if name == "PLACE":
            continue
        found = _resolved_seams(tile, name, all_sites, by_node)
        if found is None:
            missing = True  # the key addresses a seam of another kernel in the graph
            continue
        addressed.append((source_keys.get(name, name), value, found))
        if value == "fuse":
            fused.extend(seam.spelling for seam in found if len(found) == 1)
            continue
        if len(found) == 2 and not any(pair[0] is found[0] and pair[1] is found[1] for pair in join_pairs(seams)):
            raise ValueError(f"PLACE pin {name!r} joins two seams no joined cut offers: a workspace seam, then an output region")
        cut.extend(seam for seam in found if not any(chosen is seam for chosen in cut))
        if len(found) == 2:
            joins.append(found)
    refused = frozenset(fused)
    cut = [seam for seam in cut if seam.spelling not in refused]
    joins = [pair for pair in joins if all(any(chosen is seam for chosen in cut) for seam in pair)]
    bare = next(((name, value) for name, value in pins if name == "PLACE"), None)
    if bare is not None and bare[1] == "cut":
        seam = _rootmost(seams, all_sites, refused)
        if seam is not None and not any(chosen is seam for chosen in cut):
            cut.append(seam)
    if cut:
        for original, value, found in addressed:
            if (value == "cut" and all(any(chosen is seam for chosen in cut) for seam in found)) or (
                value == "fuse" and all(seam.spelling in refused for seam in found)
            ):
                note_place_key(original)
        if bare is not None and bare[1] == "cut":
            note_place_key(source_keys.get("PLACE", "PLACE"))
        return tuple(cut), "cut", tuple(joins)
    if fused:
        for original, value, found in addressed:
            if value == "fuse" and all(seam.spelling in refused for seam in found):
                note_place_key(original)
        return (fused[0],), "fuse", ()
    if bare is not None:
        if bare[1] == "fuse":
            note_place_key(source_keys.get("PLACE", "PLACE"))
        return ("PLACE",), "fuse", ()
    if missing:
        return ("PLACE",), "fuse", ()
    return None


def _composed_forks(root: Node, tile: TileOp, seams, ctx, mint) -> list[DeferredFork]:
    """One composed arm per measured route of this kernel that names several of its seams
    (:func:`composed_cuts_for`) — the decision a pinned compile consumed those seams as, offered
    again so the row that measured it can spell it. A route is registered for every kernel of the
    compile, so most of its keys address another one: a scoped key whose path is not on this tree is
    skipped, and a BARE key is skipped whatever tree it meets. The codec spells bare for a family
    with ONE site, so the kernel that spelled it has one seam and composes with nothing — the key is
    always another kernel's, and resolving it here only asks this tree a question about somebody
    else, which on a tree with several sites is ambiguous and ended the compile. A route that
    resolves to fewer than two seams adds nothing the single arms lack."""
    if ctx is None:
        return []
    routes = composed_cuts_for(tile.identity_key(structural=False, with_io=True))
    if not routes:
        return []
    all_sites = sites(tile.op)
    by_node = _seam_index(seams)
    out: list[DeferredFork] = []
    for keys in routes:
        chosen: list = []
        joins: list = []
        for name in keys:
            if parse_key(joined_halves(name)[0]).bare:
                continue
            found = _resolved_seams(tile, name, all_sites, by_node, strict=False)
            if found is None:
                continue
            note_place_key(name)
            chosen.extend(seam for seam in found if not any(picked is seam for picked in chosen))
            if len(found) == 2:
                joins.append(found)
        if len(chosen) > 1:
            composed, joined = tuple(chosen), tuple(joins)
            out.append(_cut_arm(root, lambda composed=composed, joined=joined: mint(composed, joined), composed, joined))
    return out


def _placement_forks(match: Match, root: Node, tile: TileOp, ctx=None):
    """Return the next stored-edge cut fork, or ``None`` when that domain is consumed."""
    seams = cuttable_seams(tile)
    if not seams:
        return None

    def mint(chosen, joins: tuple = ()) -> Graph:
        return realize(match, root, chosen, joins=joins)

    pinned = _placement_restriction(tile, seams, root.hints.has(PIECE))
    if pinned is not None:
        chosen, value, joins = pinned
        if value == "fuse":
            (spelling,) = chosen
            return DeferredFork(lambda: replace(tile, placement_decided=True), {spelling: "fuse"})
        return _cut_arm(root, lambda: mint(chosen, joins), chosen, joins)

    options = [DeferredFork(lambda: replace(tile, placement_decided=True), {"PLACE": "fuse"})]
    options.extend(_cut_arm(root, lambda seam=seam: mint((seam,)), (seam,)) for seam in seams)
    whole = full_projection_seams(tile, seams)
    if whole:
        options.append(_cut_arm(root, lambda: mint(whole), whole))
    options.extend(_cut_arm(root, lambda pair=pair: mint(pair, (pair,)), pair, (pair,)) for pair in join_pairs(seams))
    options.extend(_composed_forks(root, tile, seams, ctx, mint))
    return options


def rewrite(match: Match, root: Node, ctx=None):
    tile: TileOp = root.op
    if not takes_kernel_set_decision(tile):
        raise RuleSkipped("TileOp already scheduled")
    if tile.carries:
        # The runner launches one kernel's steps to completion before the next kernel's first, so
        # the pieces of a cut could not interleave step by step: piece two's step c would read what
        # piece one stored at EVERY step. A kernel that carries a state takes no cut; its one
        # kernel-set decision is the split across the sequence (``_split.realize_carry_split``).
        choices = split_forks(match, root)
        if choices is None:
            raise RuleSkipped("a kernel that carries a state is one kernel")
        options = [cut.kernel for cut in schedule(_CutContext(tuple(choices)))]
        return options if len(options) > 1 else options[0]
    choices = None if tile.placement_decided else _placement_forks(match, root, tile, ctx)
    if choices is None:
        choices = layout_forks(match, root)
    if choices is None:
        choices = split_forks(match, root)
    if choices is None:
        raise RuleSkipped("no pending kernel-set cut")
    choices = choices if isinstance(choices, list) else [choices]
    options = [cut.kernel for cut in schedule(_CutContext(tuple(choices)))]
    return options if len(options) > 1 else options[0]
