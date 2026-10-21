"""A golden against the fresh lowering of its own programs — the check and the fix.

A golden's rows are evidence for the kernels it stores, and a deploy keys them by the kernels it lowers FRESH from
the model. When the compiler starts lowering a program differently the two drift apart: the rows stay in the file
while serving builds kernels none of them describes. :func:`restamp` rewrites the golden onto the fresh lowering:
every target kernel takes the body a fresh lowering of its program gives it; every kernel-set decision is taken
again on the fresh parent, so the pieces it mints take theirs; every row follows its kernel. A file the restamp
leaves unchanged is current, which is what ``emmy golden check`` and the suite ask. GPU-free.

What a restamp keeps is decided per row, never guessed: a target no fresh kernel writes (the layer regrouped) is
dropped with its decisions and rows, since a row is a schedule of one kernel; a decision the fresh parent no longer
takes the same way (another arm, another number of pieces) is dropped with its pieces' rows, unless only its seam
addresses moved and nearby seams mint exactly its stored pieces (:func:`_respelled`); a row whose kernel kept its
identity keeps its measurement, and one whose kernel was re-keyed keeps its schedule and loses its microseconds —
a proposal, no evidence until a record run on the card measures it again. Re-keyed means the stored body and the
fresh lowering are two kernels: their exact identities, both computed now, differ. A change to how identity is
computed therefore re-keys nothing. A kernel that kept its identity keeps its stored body too: one identity can be
minted by several parents, each spelling the body's buffers its own way. When the decisions that name one stored
piece (two programs' kernel sets sharing a kernel, say) now mint different kernels, the decisions that still mint
the stored kernel keep it and its rows, whatever order the file lists them in, and every other decision is
re-pointed to the kernel it mints; so a restamp of one program and of the whole file agree on whether that
program's kernel set is current. A row's whole-target latency names the kernel set it timed, and is cleared exactly
when one of those kernels is re-keyed or dropped.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from itertools import chain, count, product
from math import prod

from emmy.compiler import pipeline
from emmy.compiler.context import Context
from emmy.compiler.graph import Graph
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.ops import UnbindableProjection
from emmy.compiler.pipeline import Pipeline
from emmy.compiler.pipeline.fork import leaf_knobs
from emmy.compiler.pipeline.knob import family_of
from emmy.compiler.pipeline.passes.tile._layout import settled_layouts
from emmy.compiler.pipeline.pipeline import Run
from emmy.compiler.pipeline.search.bench_record import kernel_row
from emmy.compiler.pipeline.search.db import RoutingRow, knobs_json
from emmy.compiler.pipeline.search.inventory import KernelInventory
from emmy.compiler.pipeline.search.pins import composed_routes, seam_aliases, spelled_arm, unpinned_decisions
from emmy.compiler.specialize import specialize_program
from emmy.compiler.wire import declared_outputs, wire_writes

from .format import GoldenFile, Kernel, Row

#: A stored kernel body to the kernel-set forks offered on it.
CUT_PASSES = ["tile/lift", "tile/cut"]

#: The most re-spellings of one route's moved seams the restamp mints before it drops the route.
RESPELL_TRIES = 16


def definition(tile, name: str, **provenance) -> Kernel:
    """The :class:`Kernel` entry of a tile kernel: its ``kernel`` row
    (:func:`~emmy.compiler.pipeline.search.bench_record.kernel_row`) with ``provenance`` (``traced``, ``origins``,
    ``bindings``) beside it; the body is spelled under ``name``, the entry's own."""
    row = kernel_row(tile, name)
    return Kernel(loop_ir=_named(row.loop_ir, name), name=name, formed=row.formed, **provenance).keyed(row.exact_identity)


def lift_targets(graph, ctx: Context) -> dict[frozenset[str], TileOp]:
    """The kernels ``graph`` lowers to at ``ctx``, lifted, by the set of buffers each writes — the one lowering a
    trace inventory and a restamp share, so a stored target and its fresh lowering can only differ where the
    compiler differs."""
    lowered = Pipeline.build([*pipeline.LOOP_PASSES, "tile/lift"]).run(graph, ctx=ctx, db=None)
    out: dict[frozenset[str], TileOp] = {}
    for node in lowered.nodes.values():
        if isinstance(node.op, TileOp):
            tile = node.op.with_io(lowered, node)
            out[frozenset(declared_outputs(tile))] = tile
    return out


def mint(
    root: Kernel, path: list[RoutingRow], ctx: Context, *, document: GoldenFile | None = None, seams: dict | None = None
) -> list[tuple[RoutingRow, bool, list[Kernel]]]:
    """Take the compatible decisions of ``path`` again, from ``root`` down: the kernel's body through the lift and cut pass,
    each fork on a kernel ``path`` decides taking the arm its route spells, every other fork keeping the kernel whole.
    Decisions may follow sibling branches of the root, with at most one route per parent.
    Returns, per route of the path in the order the decisions were taken, the route, whether the fresh lowering
    takes it the same way (``False`` when the fresh parent takes it with another arm — a stale key dropped from a
    route the cut pass still offers — or mints another number of pieces) and the pieces' fresh definitions.
    With ``document``, fresh pieces align to stored children by exact identity first, then by what they read and
    write, then by order (:func:`_paired`). ``root`` is the kernel ``path[0]`` names as its parent. With ``seams``, each
    placement fork the path decides records there, under its parent's ``ref``, the seams its parent offers to cut."""
    return _replay(root, path, ctx, document=document, seams=seams)[1]


def isolated_program(document: GoldenFile, kernel: Kernel, bindings: Mapping[str, int]) -> Graph:
    """One kernel as a program: its formed body, or its exact live tile reconstructed through recorded decisions."""
    from emmy.compiler.pipeline.search.slice import single_node_graph  # noqa: PLC0415

    if kernel.formed:
        return document.executable(kernel, bindings)
    path = document.path_to(kernel.ref)
    if not path:
        raise ValueError(f"kernel {kernel.ref!r} has no recorded route to its derived body")
    root = document.kernel(path[0].parent)
    ctx = Context.from_target(tuple(document.compute_cap), gpu_name=document.gpu_name)
    graph, taken = _replay(root, path, ctx, document=document, bindings=bindings)
    if (
        graph is None
        or len(taken) != len(path)
        or any(
            not same
            or any(document.kernel(ref).exact_identity != child.exact_identity for ref, child in zip(route.children, children, strict=True))
            for route, same, children in taken
        )
    ):
        raise ValueError(f"kernel {kernel.ref!r} has a stale recorded route")
    matches = [
        node.id
        for node in graph.nodes.values()
        if isinstance(node.op, TileOp)
        and node.op.with_io(graph, node).identity_key(structural=False, with_io=True) == kernel.exact_identity
    ]
    if len(matches) != 1:
        raise ValueError(f"kernel {kernel.ref!r} has {len(matches)} exact matches after its recorded route")
    return single_node_graph(graph, matches[0])


def _replay(
    root: Kernel,
    path: list[RoutingRow],
    ctx: Context,
    *,
    document: GoldenFile | None,
    bindings: Mapping[str, int] | None = None,
    seams: dict | None = None,
) -> tuple[Graph | None, list[tuple[RoutingRow, bool, list[Kernel]]]]:
    """The shared recorded-route replay, keeping both its live graph and the definitions a restamp consumes."""
    ref_of: dict[str, str] = {root.exact_identity: path[0].parent}  # a fresh kernel's identity -> its ``ref`` in the file
    by_parent = {route.parent: route for route in path}
    # The layouts the path recorded are the program's: the replay compiles under them settled, as serving and the
    # bench walk compile a program whose family settled them, so a sibling the path says nothing about reads the
    # weight the way the recorded kernel does.
    settled = {str(key).partition("@")[2]: str(value) for route in path for key, value in route.arm.items() if family_of(key) == "LAYOUT"}
    out: list[tuple[RoutingRow, bool, list[Kernel]]] = []
    # Per decided parent, its fork's other seam spellings: the inventory reports the arm a pass took in each seam's
    # own spelling, while a stored route may spell a seam at any of its occurrences, as ``spelled_arm`` reads it.
    aliases: dict[str, dict[str, str]] = {}

    def decide(fp):
        if fp.structural and isinstance(fp.root_op, TileOp):
            route = by_parent.get(ref_of.get(fp.root_op.identity_key(structural=False, with_io=True)))
            if route is not None:
                if seams is not None:
                    seams[route.parent] = {key for option in fp.options for key, value in leaf_knobs(option).items() if value == "cut"}
                arm = spelled_arm(fp.options, route.arm)
                if arm is not None:
                    aliases[route.parent] = seam_aliases(fp.options)
                    return arm[0]
                # The fresh parent does not offer the stored decision: the restamp drops it, and nothing under it
                # is worth building — the first leaf would leave every piece open, a fork tree of arms deep.
                raise _NotTaken(route)
        return next(fp.leaves())

    def on_routing(parent, arm, pieces, _ids) -> None:
        route = by_parent.get(ref_of.get(parent.identity_key(structural=False, with_io=True)))
        if route is None:
            return
        kernels = [definition(piece, piece.name) for piece in pieces]
        spelled = aliases.get(route.parent, {})
        same = len(kernels) == len(route.children) and {str(k): str(v) for k, v in arm.items()} == {
            spelled.get(str(k), str(k)): str(v) for k, v in route.arm.items()
        }
        if same:
            if document is not None:
                kernels = _paired([document.kernel(ref) for ref in route.children], kernels, frozenset(root.loop_ir["inputs"]))
            for ref, kernel in zip(route.children, kernels, strict=True):
                ref_of[kernel.exact_identity] = ref
        out.append((route, same, kernels))

    pipeline = Pipeline.build(CUT_PASSES).with_strategies(KernelInventory(on_routing=on_routing))
    composed = []
    for route in path:
        keys = tuple(sorted(key for key, value in route.arm.items() if family_of(key) == "PLACE" and value == "cut"))
        if len(keys) > 1 and (None, keys) not in composed:
            composed.append((None, keys))
    graph = None
    with unpinned_decisions(), composed_routes(composed), settled_layouts(settled):
        program = (
            document.executable(root, bindings or {}) if document is not None and (settled or bindings is not None) else root.program({})
        )
        try:
            graph, _ = Run(pipeline=pipeline, ctx=ctx).resolve(program, decide)
        except _NotTaken:
            pass  # the refused route and everything under it stay out of the output: not taken
    return graph, out


class _NotTaken(Exception):
    """A stored decision the fresh parent's fork does not offer; the replay stops at it."""

    def __init__(self, route: RoutingRow) -> None:
        super().__init__(f"{route.parent} {route.arm}")
        self.route = route


@dataclass
class Report:
    kernels: int = 0
    rekeyed: list[str] = field(default_factory=list)
    dropped_kernels: list[str] = field(default_factory=list)
    dropped_routes: list[str] = field(default_factory=list)
    respelled: list[str] = field(default_factory=list)
    repointed: list[str] = field(default_factory=list)
    demoted: list[str] = field(default_factory=list)
    filled_latencies: list[str] = field(default_factory=list)
    cleared_latencies: list[str] = field(default_factory=list)
    dropped_rows: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return any(
            (
                self.rekeyed,
                self.dropped_kernels,
                self.dropped_routes,
                self.respelled,
                self.repointed,
                self.demoted,
                self.filled_latencies,
                self.cleared_latencies,
                self.dropped_rows,
            )
        )

    def lines(self) -> list[str]:
        out = [f"{len(self.rekeyed)} of {self.kernels} kernels re-keyed, {len(self.dropped_kernels)} dropped"]
        out.extend(f"re-keyed {name}" for name in self.rekeyed)
        out.extend(f"re-pointed {name}" for name in self.repointed)
        out.extend(f"dropped kernel {reason}" for reason in self.dropped_kernels)
        out.extend(f"dropped decision {reason}" for reason in self.dropped_routes)
        out.extend(f"re-spelled seam {change}" for change in self.respelled)
        out.extend(f"demoted to a proposal {name}" for name in self.demoted)
        out.extend(f"filled the kernel set of the whole-target latency {name}" for name in self.filled_latencies)
        out.extend(f"cleared whole-target latency {reason}" for reason in self.cleared_latencies)
        out.extend(f"dropped row {reason}" for reason in self.dropped_rows)
        return out


def restamp(document: GoldenFile, *, traced: int | None = None) -> tuple[GoldenFile, Report]:
    """The golden rewritten onto the fresh lowering of its programs — of traced program ``traced`` alone when
    given — and what that decided. Kernels outside the selected programs carry through unchanged, and so does a
    whole-target latency none of whose kernels is in scope. A file nothing survives in comes back with no kernels:
    deleting it or re-recording it on the card is a decision, not a restamp.

    A latency written before it named its kernel set (:attr:`Latency.kernels` empty) is filled here, once, with the
    kernels the file's decisions reach from the row's target — the set a restamp swept before the field existed —
    and is exact from then on. That fill is the one migration of files written before the field."""
    ctx = Context.from_target(tuple(document.compute_cap), gpu_name=document.gpu_name or None)
    report = Report()
    targets = [kernel for kernel in document.targets() if traced is None or kernel.traced == traced]
    scope = set().union(*(document.kernel_set(kernel.ref) for kernel in targets))
    report.kernels = len(scope)
    # A piece two programs' kernel sets share (one kernel by identity) is reached here through a route of a program
    # in scope, never through the other program's route, whose target this restamp does not lower.
    reached: dict[str, RoutingRow] = {}
    for route in document.routing:
        if route.parent in scope:
            for child in route.children:
                reached.setdefault(child, route)

    fresh: dict[str, Kernel | None] = {}  # a kernel's ref -> the kernel as the fresh lowering has it, None if dropped
    added: list[Kernel] = []  # pieces the fresh lowering mints that the file holds under no ref

    def entry(kernel: Kernel) -> str:
        """The ref of the kernel of ``kernel``'s identity: a fresh one, a stored one out of scope, or ``kernel`` stored anew."""
        identity = kernel.exact_identity
        for ref, candidate in fresh.items():
            if candidate is not None and candidate.exact_identity == identity:
                return ref
        for stored in (*added, *document.kernels):
            if stored.ref not in scope and _identity(stored) == identity:
                return stored.ref
        taken = {stored.ref for stored in (*document.kernels, *added)}
        ref = next(ref for ref in chain([kernel.name], (f"{kernel.name}#{n}" for n in count(2))) if ref not in taken)
        added.append(replace(kernel, key="" if ref == kernel.name else ref).keyed(identity))
        return ref

    groups: dict[tuple, list[Kernel]] = {}
    for kernel in targets:
        groups.setdefault((kernel.traced, tuple(sorted(kernel.bindings.items()))), []).append(kernel)
    for (index, bindings), kernels in groups.items():
        lifted = lift_targets(specialize_program(document.program(index), dict(bindings)), ctx)
        for kernel in kernels:
            tile = lifted.get(wire_writes(kernel.loop_ir))
            if tile is None:
                fresh[kernel.ref] = None
                report.dropped_kernels.append(f"{kernel.name}: no fresh kernel writes its outputs")
                continue
            fresh[kernel.ref] = _rekeyed(kernel, definition(tile, kernel.name), report)

    carried = {child for route in document.routing if route.parent not in scope for child in route.children}
    routing: list[RoutingRow | None] = list(document.routing)
    # A stored piece's ref -> (decision index, position, fresh body) for each decision that mints it.
    minted: dict[str, list[tuple[int, int, Kernel]]] = {}
    depth = _depths(document.routing, scope)

    def settle(child: str) -> None:
        """Give the stored piece ``child`` its fresh kernel once every decision in scope that mints it was taken again.
        The decisions minting the stored kernel keep the ``ref`` and its rows, whatever order the file lists them in;
        when none does and no decision out of scope keeps the stored kernel, the decisions replayed first keep it,
        re-keyed. Every other decision is re-pointed to the kernel of the identity it mints."""
        taken_by = minted.get(child)
        if not taken_by:
            return
        stored = document.kernel(child)
        identities = {kernel.exact_identity for _, _, kernel in taken_by}
        keep = _identity(stored)
        if keep not in identities:
            keep = None if child in carried else taken_by[0][2].exact_identity
        for index, position, kernel in taken_by:
            if kernel.exact_identity == keep:
                if child not in fresh or fresh[child] is None:
                    fresh[child] = _rekeyed(stored, kernel, report)
                    # Decisions on the piece replay along a path that mints it, through the arm the restamp took.
                    reached[child] = replace(document.routing[index], arm=routing[index].arm)
                continue
            route = routing[index]
            ref = entry(kernel)
            routing[index] = replace(route, children=tuple(ref if at == position else old for at, old in enumerate(route.children)))
            report.repointed.append(f"{route.parent} {child} -> {ref}: this decision mints another kernel now")

    # Decisions are taken again by depth, so every decision that mints a piece is taken before the piece is settled
    # and before any decision on it.
    # The outcome of each path a decision was replayed along, with the seams its parents offered.
    replays: dict[tuple, tuple[bool, list[Kernel], dict[str, set[str]]]] = {}
    for level in sorted({depth[route.parent] for route in document.routing if route.parent in scope}):
        for index, route in enumerate(document.routing):
            if route.parent not in scope or depth[route.parent] != level:
                continue
            path = [*_path_in(reached, route.parent), route]
            if any(fresh.get(step.parent) is None for step in path):
                routing[index] = None
                report.dropped_routes.append(f"{route.parent} {route.arm}: its parent is gone")
                for child in route.children:
                    fresh.setdefault(child, None)
                continue
            if (key := _spelled(path)) not in replays:
                # One replay down the plain decisions below this one takes each of them as its own replay would.
                extended = [*path, *_plain_descent(document.routing, route, scope)]
                offered: dict[str, set[str]] = {}
                for old, taken, pieces in mint(fresh[path[0].parent], extended, ctx, document=document, seams=offered):
                    at = next(at for at, step in enumerate(extended) if step is old)
                    replays[_spelled(extended[: at + 1])] = (taken, pieces, offered)
                replays.setdefault(key, (False, [], offered))
            same, kernels, offered = replays[key]
            if not same and (found := _respelled(document, fresh[path[0].parent], path, offered.get(route.parent, set()), ctx)):
                respelled, kernels = found
                report.respelled.extend(
                    f"{route.parent} {old} -> {new}" for old, new in zip(route.arm, respelled.arm, strict=True) if old != new
                )
                routing[index] = respelled
                reached.update({child: respelled for child, via in reached.items() if via is route})
                same = True
            if not same:
                routing[index] = None
                report.dropped_routes.append(f"{route.parent} {route.arm}: the fresh parent does not take it the same way")
                for child in route.children:
                    fresh.setdefault(child, None)
                continue
            for position, (child, kernel) in enumerate(zip(route.children, kernels, strict=True)):
                minted.setdefault(child, []).append((index, position, kernel))
        for child in [ref for ref, at in depth.items() if at == level + 1]:
            settle(child)

    # Two stored kernels the fresh lowering makes one are stored once, under the first's ref. A piece in scope that
    # no decision in scope mints any more is dropped, unless a decision out of scope still reaches it.
    kernels: list[Kernel] = []
    stored_as: dict[str, str] = {}  # every surviving kernel's ref -> the ref it is stored under
    by_identity: dict[str, str] = {}
    for kernel in (*document.kernels, *added):
        rekeyed = kernel if kernel.ref not in scope or (kernel.ref not in fresh and kernel.ref in carried) else fresh.get(kernel.ref)
        if rekeyed is None:  # dropped, or a piece no decision of the file reaches
            continue
        identity = _identity(rekeyed)
        if identity is not None and identity in by_identity:
            stored_as[kernel.ref] = by_identity[identity]
            report.dropped_kernels.append(f"{kernel.ref}: the fresh lowering makes it the kernel {by_identity[identity]}")
            continue
        if identity is not None:
            by_identity[identity] = kernel.ref
        stored_as[kernel.ref] = kernel.ref
        kernels.append(rekeyed)

    kept: list[RoutingRow] = []
    for route in routing:
        if route is None:
            continue
        if any(ref not in stored_as for ref in (route.parent, *route.children)):
            report.dropped_routes.append(f"{route.parent} {route.arm}: a piece is gone")
            continue
        kept.append(replace(route, parent=stored_as[route.parent], children=tuple(stored_as[child] for child in route.children)))

    def stale(ref: str) -> str | None:
        """Why the stored kernel ``ref`` is no longer what a compile builds: gone, or re-keyed; ``None`` when it is."""
        if ref not in stored_as:
            return f"{ref} is gone"
        return f"{ref} was re-keyed" if fresh.get(ref) not in (None, document.kernel(ref)) else None

    rows: list[Row] = []
    for row in document.rows:
        if row.kernel not in stored_as:
            report.dropped_rows.append(f"{row.name}: its kernel is gone")
            continue
        fresh_row = replace(row, kernel=stored_as[row.kernel])
        if row.measurements is not None and stale(row.kernel) is not None:
            fresh_row = replace(fresh_row, measurements=None)
            report.demoted.append(row.name)
        latencies = {}
        for card, latency in (row.latency or {}).items():
            if not latency.kernels:
                path = document.path_to(row.kernel)
                latency = replace(latency, kernels=document.kernel_set(path[0].parent if path else row.kernel))
                report.filled_latencies.append(row.name)
            reasons = [reason for reason in map(stale, latency.kernels) if reason is not None]
            if reasons:
                report.cleared_latencies.append(f"{row.name}: {reasons[0]}")
                continue
            latencies[card] = replace(latency, kernels=tuple(dict.fromkeys(stored_as[ref] for ref in latency.kernels)))
        rows.append(replace(fresh_row, latency=latencies or None))
    out = replace(document, kernels=kernels, routing=kept, rows=rows)
    return out, report


def _respelled(
    document: GoldenFile, root: Kernel, path: list[RoutingRow], offered: set[str], ctx: Context
) -> tuple[RoutingRow, list[Kernel]] | None:
    """The last route of ``path`` with its seams spelled where the fresh parent cuts its stored pieces, and those
    pieces; ``None`` when no re-spelling does. A seam address is an operand position, so a change of canonical operand
    order moves it while the computation stays: the stored pieces, whose identities this compiler computes from their
    stored bodies, are what the decision cut. Each seam the parent no longer offers is tried at the offered seams of
    its kind nearest to it — fewest route segments changed, added or removed. A candidate one operand index away is
    also tried as a swap of the two operands of that node, which moves every seam of the route under either; a swap
    can leave a moved seam on an offered spelling that now names the other operand. The nearest combinations are
    minted, at most :data:`RESPELL_TRIES`; one is taken when the fresh parent mints exactly the stored pieces. Where the
    change also re-keyed a piece, the candidates whose pieces read the same target inputs and write outputs of the same
    dtypes and sizes as the stored ones, one to one, stand for the decision; sibling cones over one input match alike,
    so the one that mints the most stored pieces again is taken, and a tie takes none. Two arms that mint one set of
    pieces are one kernel set: the first is taken."""
    route = path[-1]
    lost = [key for key, value in route.arm.items() if family_of(key) == "PLACE" and value == "cut" and key not in offered]
    if not lost or not offered:
        return None
    free = sorted(offered - set(route.arm))

    def moves(key: str) -> list[tuple[int, dict[tuple[str, ...], tuple[str, ...]]]]:
        """Each way to re-spell ``key``, nearest first, as its distance and a map of route prefixes to replace."""
        ranked = sorted((_distance(key, other), other) for other in free if other.rsplit("/", 1)[-1] == key.rsplit("/", 1)[-1])
        out = []
        for far, other in [(far, other) for far, other in ranked if far <= ranked[0][0] + 1][:4] if ranked else []:
            old, new = _segments(key), _segments(other)
            out.append((far, {old: new}))
            if far == 1 and len(old) == len(new):
                at = next(i for i, (a, b) in enumerate(zip(old, new, strict=True)) if a != b)
                out.append((far, {old[: at + 1]: new[: at + 1], new[: at + 1]: old[: at + 1]}))
        return out

    def respell(key: str, prefixes: dict[tuple[str, ...], tuple[str, ...]]) -> str:
        segments = _segments(key)
        for prefix, into in prefixes.items():
            if segments[: len(prefix)] == prefix:
                return "PLACE@" + "/".join(into + segments[len(prefix) :])
        return key

    combinations = sorted(product(*map(moves, lost)), key=lambda choice: sum(far for far, _ in choice))
    stored = [document.kernel(ref) for ref in route.children]
    want = sorted(_identity(kernel) or "" for kernel in stored)
    inputs = frozenset(root.loop_ir["inputs"])
    alike: list[tuple[int, RoutingRow, list[Kernel]]] = []
    seen: set[str] = set()
    for choice in combinations:
        prefixes = {prefix: into for _, move in choice for prefix, into in move.items()}
        arm = {respell(key, prefixes) if family_of(key) == "PLACE" else key: value for key, value in route.arm.items()}
        cuts = [key for key, value in arm.items() if family_of(key) == "PLACE" and value == "cut"]
        if len(arm) < len(route.arm) or any(key not in offered for key in cuts) or knobs_json(arm) in seen:
            continue
        seen.add(knobs_json(arm))
        candidate = replace(route, arm=arm)
        try:
            minted = mint(root, [*path[:-1], candidate], ctx, document=document)
        except UnbindableProjection:
            # Each seam is offered alone; a combination whose pieces cannot own their outputs is not the route.
            minted = []
        for taken, same, kernels in minted:
            if taken is candidate and same and sorted(kernel.exact_identity for kernel in kernels) == want:
                return candidate, kernels
            if taken is candidate and same and _io(stored, inputs) == _io(kernels, inputs):
                alike.append((sum((Counter(want) & Counter(kernel.exact_identity for kernel in kernels)).values()), candidate, kernels))
        if len(seen) == RESPELL_TRIES:
            break
    most = max((kept for kept, _, _ in alike), default=None)
    best = [(candidate, kernels) for kept, candidate, kernels in alike if kept == most]
    return best[0] if len(best) == 1 else None


def _segments(key: str) -> tuple[str, ...]:
    """A seam key's route, segment by segment."""
    return tuple(key.partition("@")[2].split("/"))


def _distance(key: str, other: str) -> int:
    """How many route segments of ``key`` must change, be added or be removed to spell ``other``."""
    a, b = _segments(key), _segments(other)
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, start=1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1]


def _depths(routing: list[RoutingRow], scope: set[str]) -> dict[str, int]:
    """Each kernel of ``scope`` by its depth: ``0`` for a kernel no decision in scope mints (a target); for a piece,
    one more than the greatest depth among the parents of the decisions in scope that mint it. A decision's pieces
    are cut from its parent, so routing has no cycles; ``seen`` only stops a malformed file from recursing forever,
    and the memoized depths of such a file then depend on visit order."""
    minters: dict[str, list[str]] = {}
    for route in routing:
        if route.parent in scope:
            for child in route.children:
                minters.setdefault(child, []).append(route.parent)
    out: dict[str, int] = {}

    def depth(ref: str, seen: frozenset[str]) -> int:
        if ref not in out:
            parents = [parent for parent in minters.get(ref, []) if parent not in seen]
            out[ref] = 1 + max((depth(parent, seen | {ref}) for parent in parents), default=-1)
        return out[ref]

    for ref in scope:
        depth(ref, frozenset())
    return out


def _spelled(path: list[RoutingRow]) -> tuple:
    """A replay path by value: the same decisions in the same order replay the same way."""
    return tuple((step.parent, knobs_json(step.arm), step.children) for step in path)


def _plain_descent(routing: list[RoutingRow], route: RoutingRow, scope) -> list[RoutingRow]:
    """The decisions straight below ``route``, each on a piece the one above mints, while each one cuts a single seam
    and settles no layout: adding such a decision to a replay changes neither the composed cuts nor the layouts the
    replay compiles under, so the decisions above it replay as they would alone. A nested cut a prior walk recorded
    is such a chain, often dozens deep, and replaying it once instead of once per decision keeps a restamp linear in
    its depth."""
    out: list[RoutingRow] = []
    while (below := next((step for step in routing if step.parent in route.children and step.parent in scope), None)) is not None:
        cuts = [key for key, value in below.arm.items() if family_of(key) == "PLACE" and value == "cut"]
        if len(cuts) > 1 or any(family_of(key) == "LAYOUT" for key in below.arm):
            break
        out.append(below)
        route = below
    return out


def _path_in(reached: dict[str, RoutingRow], ref: str) -> list[RoutingRow]:
    """The decisions from a target down to ``ref`` through the routes ``reached`` holds, in order (``path_to`` over
    one program's kernel sets)."""
    path: list[RoutingRow] = []
    seen = {ref}
    while (route := reached.get(ref)) is not None and route.parent not in seen:
        path.insert(0, route)
        ref = route.parent
        seen.add(ref)
    return path


def _paired(stored: list[Kernel], fresh: list[Kernel], inputs: frozenset[str]) -> list[Kernel]:
    """``fresh``, the pieces a decision mints again, in the order of the ``stored`` pieces they stand for. A fresh piece
    of a stored piece's exact identity is that piece; unmatched pieces pair by what they read and write, then by order.
    A piece reads its target's ``inputs`` (weights, activations) under their program names, which no re-key moves,
    and a cut's workspaces under names that do; so the target inputs a piece reads tell two re-keyed siblings of one
    shape apart (a q and a k projection piece), and the dtype and element count of each output it writes come next.
    Pieces that agree on a signature pair in their relative order."""

    def reads(kernel: Kernel) -> frozenset[str]:
        return frozenset(kernel.loop_ir["inputs"]) & inputs

    remaining = dict(enumerate(fresh))
    out: dict[int, Kernel] = {}
    for signature in (_identity, lambda kernel: (reads(kernel), _writes(kernel)), reads, _writes):
        for index, piece in enumerate(stored):
            if index in out:
                continue
            want = signature(piece)
            found = next((at for at, kernel in remaining.items() if signature(kernel) == want), None)
            if found is not None:
                out[index] = remaining.pop(found)
    rest = iter(remaining.values())
    return [out[index] if index in out else next(rest) for index in range(len(stored))]


def _writes(kernel: Kernel) -> tuple:
    """The dtype and element count of each output ``kernel`` writes."""
    wire = kernel.loop_ir
    return tuple(
        sorted(
            (str(dtype), str(prod(dims)) if all(isinstance(dim, int) for dim in dims) else str(dims))
            for node in wire["nodes"]
            for name, dtype, dims in node["outputs"]
            if name in wire["outputs"]
        )
    )


def _io(kernels: list[Kernel], inputs: frozenset[str]) -> list[tuple]:
    """What each of ``kernels`` reads of the target ``inputs`` and writes, as one sortable multiset."""
    return sorted((sorted(frozenset(kernel.loop_ir["inputs"]) & inputs), _writes(kernel)) for kernel in kernels)


def _identity(kernel: Kernel) -> str | None:
    """``kernel``'s exact identity, or ``None`` when its stored body no longer lowers to a kernel."""
    try:
        return kernel.exact_identity
    except Exception:  # noqa: BLE001 — a body the compiler stopped taking back is a stale kernel, not an error here
        return None


def _rekeyed(stored: Kernel, fresh: Kernel, report: Report) -> Kernel:
    """``stored`` as the fresh lowering defines it. A kernel that kept its identity keeps its entry — body and
    name; one the fresh lowering makes another kernel takes the fresh body, spelled under the stored name and known
    by the stored ``ref``, and is reported."""
    old = _identity(stored)
    if fresh.exact_identity == old:
        return stored
    out = replace(stored, loop_ir=_named(fresh.loop_ir, stored.name), formed=fresh.formed).keyed(fresh.exact_identity)
    report.rekeyed.append(f"{stored.ref} {(old or 'no kernel')[:12]} -> {out.exact_identity[:12]}")
    return out


def _named(wire: dict, name: str) -> dict:
    """``wire`` with its loop node spelled under ``name``: the one part of a body two lowerings of one kernel spell
    differently, and no part of its identity."""
    nodes = [{**node, "attrs": {**node["attrs"], "name": name}} if node.get("op") == "loop" else node for node in wire["nodes"]]
    return {**wire, "nodes": nodes}
