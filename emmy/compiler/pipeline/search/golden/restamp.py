"""A golden against the fresh lowering of its own programs — the check and the fix.

A golden's rows are evidence for the kernels it stores, and a deploy keys them by the kernels it lowers FRESH from
the model. When the compiler starts lowering a program differently the two drift apart: the rows stay in the file
while serving builds kernels none of them describes. :func:`restamp` rewrites the golden onto the fresh lowering:
every target kernel takes the body a fresh lowering of its program gives it; every kernel-set decision is taken
again on the fresh parent, so the pieces it mints take theirs; every row follows its kernel. A file the restamp
leaves unchanged is current, which is what ``emmy golden check`` and the suite ask. GPU-free.

What a restamp keeps is decided per row, never guessed: a target no fresh kernel writes (the layer regrouped) is
dropped with its decisions and rows, since a row is a schedule of one kernel; a decision the fresh parent no longer
takes the same way (another arm, another number of pieces) is dropped with its pieces' rows; a row whose kernel kept
its identity keeps its measurement, and one whose kernel was re-keyed keeps its schedule and loses its microseconds —
a proposal, no evidence until a record run on the card measures it again. Re-keyed means the stored body and the
fresh lowering are two kernels: their exact identities, both computed now, differ. A change to how identity is
computed therefore re-keys nothing. A kernel that kept its identity keeps its stored body too: one identity can be
minted by several parents, each spelling the body's buffers its own way.

A decision's fresh pieces are matched to the pieces the file stores for it by exact identity first, so pieces
minted in another order keep their entries and rows, listed in the fresh order. A fresh piece whose identity no
stored piece has takes the stored piece in its position, when that one has no fresh counterpart either. A stored
piece no decision still mints is dropped with its rows: they measured a kernel no decision mints now. A
fresh piece left without one joins the file with no rows, under its own name (``name#2``, ... where it is taken).

Every routing row that lists a piece among its children must mint that same kernel. When the decisions that name one
stored piece (two programs' kernel sets sharing a kernel, say) now mint different kernels, the file stores one entry
per kernel: the decisions that still mint the stored kernel keep the piece's ``ref`` and its rows — or, when none
does, the decisions replayed first, the rows then proposals — and every other kernel takes a ``ref`` of its own (or
that of a stored kernel it already is), which its decisions name. The report calls this a split of the shared piece,
not to be confused with a cross-CTA split. So a restamp of one program and of the whole file agree on whether that
program's kernel set is current.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from emmy.compiler import pipeline
from emmy.compiler.context import Context
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import Pipeline
from emmy.compiler.pipeline.knob import family_of
from emmy.compiler.pipeline.pipeline import Run
from emmy.compiler.pipeline.search.bench_record import kernel_row
from emmy.compiler.pipeline.search.db import RoutingRow, knobs_json
from emmy.compiler.pipeline.search.inventory import KernelInventory
from emmy.compiler.pipeline.search.pins import composed_routes, spelled_arm, unpinned_decisions
from emmy.compiler.specialize import specialize_program
from emmy.compiler.wire import declared_outputs, wire_writes

from .format import GoldenFile, Kernel, Row

#: A stored kernel body to the kernel-set forks offered on it.
CUT_PASSES = ["tile/lift", "tile/cut"]


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
    root: Kernel, path: list[RoutingRow], ctx: Context, *, document: GoldenFile | None = None
) -> list[tuple[RoutingRow, bool, list[Kernel]]]:
    """Take the decisions of ``path`` again, from ``root`` down: the kernel's body through the lift and the cut pass,
    each fork on a kernel ``path`` decides taking the arm its route spells, every other fork keeping the kernel whole.
    Returns, per route of the path in the order the decisions were taken, the route, whether the fresh lowering
    takes it the same way (``False`` when the fresh parent takes it with another arm — a stale key dropped from a
    route the cut pass still offers — or mints another number of pieces) and the pieces' fresh definitions.
    ``root`` is the kernel ``path[0]`` names as its parent."""
    ref_of: dict[str, str] = {root.exact_identity: path[0].parent}  # a fresh kernel's identity -> its ``ref`` in the file
    by_parent = {route.parent: route for route in path}
    out: list[tuple[RoutingRow, bool, list[Kernel]]] = []

    def decide(fp):
        if fp.structural and isinstance(fp.root_op, TileOp):
            route = by_parent.get(ref_of.get(fp.root_op.identity_key(structural=False, with_io=True)))
            arm = spelled_arm(fp.options, route.arm if route is not None else {})
            if arm is not None:
                return arm[0]
        return next(fp.leaves())

    def on_routing(parent, arm, pieces, _ids) -> None:
        route = by_parent.get(ref_of.get(parent.identity_key(structural=False, with_io=True)))
        if route is None:
            return
        kernels = [definition(piece, piece.name) for piece in pieces]
        same = len(kernels) == len(route.children) and {str(k): str(v) for k, v in arm.items()} == {
            str(k): str(v) for k, v in route.arm.items()
        }
        if same:
            identities = [_identity(document.kernel(ref)) if document is not None else None for ref in route.children]
            for ref, kernel in zip(_correspondence(route.children, identities, kernels), kernels, strict=True):
                if ref is not None:
                    ref_of[kernel.exact_identity] = ref
        out.append((route, same, kernels))

    pipeline = Pipeline.build(CUT_PASSES).with_strategies(KernelInventory(on_routing=on_routing))
    composed = []
    for route in path:
        keys = tuple(sorted(key for key, value in route.arm.items() if family_of(key) == "PLACE" and value == "cut"))
        if len(keys) > 1 and (None, keys) not in composed:
            composed.append((None, keys))
        if keys and document is not None:
            # Exact child identities bound which fresh pieces may continue cutting.
            entry = (document.kernel(route.parent).exact_identity, keys)
            if entry not in composed:
                composed.append(entry)
    with unpinned_decisions(), composed_routes(composed):
        has_layout = any(family_of(key) == "LAYOUT" for route in path for key in route.arm)
        program = document.executable(root, {}) if document is not None and has_layout else root.program({})
        Run(pipeline=pipeline, ctx=ctx).resolve(program, decide)
    return out


@dataclass
class Report:
    kernels: int = 0
    rekeyed: list[str] = field(default_factory=list)
    dropped_kernels: list[str] = field(default_factory=list)
    dropped_routes: list[str] = field(default_factory=list)
    split: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    reordered: list[str] = field(default_factory=list)
    demoted: list[str] = field(default_factory=list)
    dropped_rows: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(
            self.rekeyed
            or self.dropped_kernels
            or self.dropped_routes
            or self.split
            or self.added
            or self.reordered
            or self.demoted
            or self.dropped_rows
        )

    def lines(self) -> list[str]:
        out = [f"{len(self.rekeyed)} of {self.kernels} kernels re-keyed, {len(self.dropped_kernels)} dropped"]
        out.extend(f"re-keyed {name}" for name in self.rekeyed)
        out.extend(f"dropped kernel {reason}" for reason in self.dropped_kernels)
        out.extend(f"dropped decision {reason}" for reason in self.dropped_routes)
        out.extend(f"split shared piece {reason}" for reason in self.split)
        out.extend(f"new piece {reason}" for reason in self.added)
        out.extend(f"reordered the pieces of the decision on {reason}" for reason in self.reordered)
        out.extend(f"demoted to a proposal {name}" for name in self.demoted)
        out.extend(f"dropped row {reason}" for reason in self.dropped_rows)
        return out


def restamp(document: GoldenFile, *, traced: int | None = None) -> tuple[GoldenFile, Report]:
    """The golden rewritten onto the fresh lowering of its programs — of traced program ``traced`` alone when
    given, everything else carried through unchanged — and what that decided. A file nothing survives in comes back
    with no kernels: deleting it or re-recording it on the card is a decision, not a restamp."""
    ctx = Context.from_target(tuple(document.compute_cap), gpu_name=document.gpu_name or None)
    report = Report()
    targets = [kernel for kernel in document.targets() if traced is None or kernel.traced == traced]
    scope = {kernel.ref for kernel in targets}
    grown = True
    while grown:  # the subtree of every target in scope, whatever order the routing rows are stored in
        grown = False
        for route in document.routing:
            if route.parent in scope and not scope.issuperset(route.children):
                scope.update(route.children)
                grown = True
    report.kernels = len(scope)
    # A piece two programs' kernel sets share (one kernel by identity) is reached here through a route of a program
    # in scope, never through the other program's route, whose target this restamp does not lower.
    reached: dict[str, RoutingRow] = {}
    for route in document.routing:
        if route.parent in scope:
            for child in route.children:
                reached.setdefault(child, route)
    depth = _depths(document.routing, scope)

    fresh: dict[str, Kernel | None] = {}  # a kernel's ref -> the kernel as the fresh lowering has it, None if dropped
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

    routing: list[RoutingRow | None] = list(document.routing)
    minted: dict[str, list[tuple[int, Kernel]]] = {}  # a piece's ref -> each decision minting it (by index), its body
    splits: dict[str, list[Kernel]] = {}  # a stored piece's ref -> the kernels split off it
    added: dict[str, list[Kernel]] = {}  # a parent's ref -> the pieces its decisions mint that no stored piece is
    refs = {kernel.ref for kernel in document.kernels}  # every ref taken, the stored ones and each split's
    for level in sorted({depth[route.parent] for route in document.routing if route.parent in scope}):
        for index, route in enumerate(document.routing):
            if route.parent not in scope or depth[route.parent] != level:
                continue
            path = [*_path_in(reached, route.parent), route]
            if any(fresh.get(step.parent) is None for step in path):
                routing[index] = None
                report.dropped_routes.append(f"{route.parent} {route.arm}: its parent is gone")
                continue
            taken = {
                (old.parent, knobs_json(old.arm)): (same, kernels)
                for old, same, kernels in mint(fresh[path[0].parent], path, ctx, document=document)
            }
            same, kernels = taken.get((route.parent, knobs_json(route.arm)), (False, []))
            if not same:
                routing[index] = None
                report.dropped_routes.append(f"{route.parent} {route.arm}: the fresh parent does not take it the same way")
                continue
            identities = [_identity(document.kernel(child)) for child in route.children]
            children = []
            for child, kernel in zip(_correspondence(route.children, identities, kernels), kernels, strict=True):
                if child is None:  # no stored piece of the decision is this kernel: stored after its parent, no rows
                    kernel = _stored_as_new(kernel, kernel.name, refs)
                    added.setdefault(route.parent, []).append(kernel)
                    report.added.append(f"{kernel.ref} {kernel.exact_identity[:12]}: minted by the decision on {route.parent}")
                    children.append(kernel.ref)
                    continue
                minted.setdefault(child, []).append((index, kernel))
                children.append(child)
            named = [child for child in children if child in route.children]
            if named != [child for child in route.children if child in named]:
                report.reordered.append(f"{route.parent} {route.arm}")
            routing[index] = replace(route, children=tuple(children))
        # Every decision that mints a piece one level down is taken again or dropped: each piece gets its fresh kernel.
        for child in [kernel.ref for kernel in document.kernels if depth.get(kernel.ref) == level + 1]:
            taken_by = minted.get(child, [])
            if not taken_by:
                if any(routing[index] is not None and child in route.children for index, route in enumerate(document.routing)):
                    report.dropped_kernels.append(f"{child}: no piece the decisions that named it mint now is that kernel")
                fresh[child] = None
                continue
            keeper, *others = _split(document.kernel(child), taken_by, refs)
            fresh[child] = _rekeyed(document.kernel(child), keeper[0][1], report)
            reached[child] = document.routing[keeper[0][0]]  # decisions on the piece replay along this path
            for share in others:
                kernel = share[0][1]
                splits.setdefault(child, []).append(kernel)
                for index, _ in share:
                    route = routing[index]
                    routing[index] = replace(route, children=tuple(kernel.ref if ref == child else ref for ref in route.children))
                old = (_identity(document.kernel(child)) or "no kernel")[:12]
                parents = ", ".join(document.routing[index].parent for index, _ in share)
                report.split.append(f"{child} {old} -> {kernel.exact_identity[:12]} as {kernel.ref}, minted by the decisions on {parents}")

    # Two stored kernels the fresh lowering makes one are stored once, under the first's ref.
    kernels: list[Kernel] = []
    stored_as: dict[str, str] = {}  # every surviving kernel's ref -> the ref it is stored under
    by_identity: dict[str, str] = {}
    for kernel in document.kernels:
        rekeyed = fresh.get(kernel.ref) if kernel.ref in scope else kernel
        if rekeyed is None:  # dropped, or a piece no decision of the file reaches
            continue
        extra = [*splits.get(kernel.ref, []), *added.get(kernel.ref, [])]
        for ref, stored in ((kernel.ref, rekeyed), *((piece.ref, piece) for piece in extra)):
            identity = _identity(stored)
            if identity is not None and identity in by_identity:
                stored_as[ref] = by_identity[identity]
                report.dropped_kernels.append(f"{ref}: the fresh lowering makes it the kernel {by_identity[identity]}")
                continue
            if identity is not None:
                by_identity[identity] = ref
            stored_as[ref] = ref
            kernels.append(stored)

    rows: list[Row] = []
    for row in document.rows:
        if row.kernel not in stored_as:
            report.dropped_rows.append(f"{row.name}: its kernel is gone")
            continue
        fresh_row = replace(row, kernel=stored_as[row.kernel])
        if (row.measurements is not None or row.latency is not None) and fresh.get(row.kernel) not in (None, document.kernel(row.kernel)):
            fresh_row = replace(fresh_row, measurements=None, latency=None)
            report.demoted.append(row.name)
        rows.append(fresh_row)

    kept: list[RoutingRow] = []
    for route in routing:
        if route is None:
            continue
        if any(ref not in stored_as for ref in (route.parent, *route.children)):
            report.dropped_routes.append(f"{route.parent} {route.arm}: a piece is gone")
            continue
        kept.append(replace(route, parent=stored_as[route.parent], children=tuple(stored_as[child] for child in route.children)))
    out = replace(document, kernels=kernels, routing=kept, rows=rows)
    return out, report


def _depths(routing: list[RoutingRow], scope: set[str]) -> dict[str, int]:
    """Each kernel of ``scope`` by its depth: ``0`` for a target; for a piece, one more than the greatest depth
    among the parents of the in-scope decisions that mint it — so every decision minting a piece is replayed before
    any decision on it."""
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


def _split(stored: Kernel, minted: list[tuple[int, Kernel]], taken: set[str]) -> list[list[tuple[int, Kernel]]]:
    """The decisions that mint the piece ``stored`` (each by its index, with the body it mints), grouped by the
    kernel each mints: first the group that keeps the piece's ``ref`` — the one that mints the stored kernel, else
    the first in ``minted`` — then every other group, its body spelled under the stored name and known by a ``ref``
    not in ``taken``."""
    by_identity: dict[str, list[tuple[int, Kernel]]] = {}
    for index, kernel in minted:
        by_identity.setdefault(kernel.exact_identity, []).append((index, kernel))
    keep = _identity(stored)
    keep = keep if keep in by_identity else next(iter(by_identity))
    out = [by_identity.pop(keep)]
    for share in by_identity.values():
        body = _stored_as_new(share[0][1], stored.name, taken)
        out.append([(index, body) for index, _ in share])
    return out


def _stored_as_new(fresh: Kernel, name: str, taken: set[str]) -> Kernel:
    """``fresh`` as a kernel entry of its own: its body spelled under ``name`` and known by ``name`` — or the name
    numbered (``name#2``, ...) where a ref of ``taken`` has it — which then joins ``taken``."""
    ref = next(ref for ref in (name, *(f"{name}#{n}" for n in range(2, len(taken) + 3))) if ref not in taken)
    taken.add(ref)
    body = replace(fresh, key="" if ref == name else ref, name=name, loop_ir=_named(fresh.loop_ir, name))
    return body.keyed(fresh.exact_identity)


def _correspondence(children: tuple[str, ...], identities: list[str | None], pieces: list[Kernel]) -> list[str | None]:
    """For each piece a decision's fresh lowering mints, in that order, the stored piece ``children`` holds for it:
    the one of the same exact identity, wherever it is stored; else the one in the same position, when neither has
    a counterpart of its identity; else ``None``. ``identities`` holds each stored piece's identity, ``None`` where
    it is unknown."""
    out: list[str | None] = [None] * len(pieces)
    free = dict(enumerate(children))  # a stored piece's position -> its ref, until a fresh piece claims it
    for slot, piece in enumerate(pieces):
        match = next((position for position in free if identities[position] == piece.exact_identity), None)
        if match is not None:
            out[slot] = free.pop(match)
    for slot in range(len(pieces)):
        if out[slot] is None and slot in free:
            out[slot] = free.pop(slot)
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
