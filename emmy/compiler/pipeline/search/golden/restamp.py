"""A golden against the fresh lowering of its own programs — the check and the fix.

A golden's rows are evidence for the kernels it stores, and a deploy keys them by the kernels it lowers FRESH from
the model. When the compiler starts lowering a program differently the two drift apart: the rows stay in the file
while serving builds kernels none of them describes. :func:`restamp` rewrites the golden onto the fresh lowering:
every target kernel takes the identity, stamps and body a fresh lowering of its program gives it; every kernel-set
decision is taken again on the fresh parent, so the pieces it mints take theirs; every row follows its kernel. A
file the restamp leaves unchanged is current, which is what ``emmy golden check`` and the suite ask. GPU-free.

What a restamp keeps is decided per row, never guessed: a target no fresh kernel writes (the layer regrouped) is
dropped with its decisions and rows, since a row is a schedule of one kernel; a decision the fresh parent no longer
takes the same way (another arm, another number of pieces) is dropped with its pieces' rows; a row whose kernel kept
its identity keeps its measurement, and one whose kernel was re-keyed keeps its schedule and loses its microseconds —
a proposal, no evidence until a record run on the card measures it again. A kernel that kept its identity keeps its
stored body too: one identity can be minted by several parents, each spelling the body's buffers its own way, and
the identity is what the DB keys on; only its stamps are taken from the fresh lowering, so a featurizer change lands
here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace

from emmy.compiler import pipeline
from emmy.compiler.context import Context
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import Pipeline
from emmy.compiler.pipeline.fork import SCHEDULE_FORK_STAMPS
from emmy.compiler.pipeline.knob import family_of
from emmy.compiler.pipeline.pipeline import Run
from emmy.compiler.pipeline.search.bench_record import kernel_row
from emmy.compiler.pipeline.search.dataset.kernel import KernelDef
from emmy.compiler.pipeline.search.db import RoutingRow, knobs_json
from emmy.compiler.pipeline.search.inventory import KernelInventory
from emmy.compiler.pipeline.search.pins import composed_routes, spelled_arm, unpinned_decisions
from emmy.compiler.specialize import specialize_program

from .format import GoldenFile, Kernel, Row

#: A stored kernel body to the kernel-set forks offered on it.
CUT_PASSES = ["tile/lift", "tile/cut"]


def definition(tile, name: str, **provenance) -> Kernel:
    """The :class:`Kernel` entry of a tile kernel: its ``kernel`` row
    (:func:`~emmy.compiler.pipeline.search.bench_record.kernel_row`) with ``provenance`` (``traced``, ``origins``,
    ``bindings``) beside it. The stamps are the identity strategy's own, less the ones a schedule fork mints on its
    rows (``SCHEDULE_FORK_STAMPS``), so a kernel recorded off a compiled program and the same kernel re-derived from
    its program spell one entry; the body is spelled under ``name``, the entry's own."""
    row = kernel_row(tile, name)
    fields_ = {f.name: getattr(row, f.name) for f in fields(KernelDef) if f.name not in ("stamps", "loop_ir")}
    stamps = {key: value for key, value in row.stamps.items() if key not in SCHEDULE_FORK_STAMPS}
    return Kernel(**fields_, loop_ir=_named(row.loop_ir, name), stamps=stamps, **provenance)


def lift_targets(graph, ctx: Context) -> dict[frozenset[str], TileOp]:
    """The kernels ``graph`` lowers to at ``ctx``, lifted, by the set of buffers each writes — the one lowering a
    trace inventory and a restamp share, so a stored target and its fresh lowering can only differ where the
    compiler differs."""
    lowered = Pipeline.build([*pipeline.LOOP_PASSES, "tile/lift"]).run(graph, ctx=ctx, db=None)
    out: dict[frozenset[str], TileOp] = {}
    for node in lowered.nodes.values():
        if isinstance(node.op, TileOp):
            tile = node.op.with_io(lowered, node)
            out[frozenset(tile.outputs)] = tile
    return out


def mint(
    root: Kernel, path: list[RoutingRow], ctx: Context, *, root_identity: str | None = None
) -> list[tuple[RoutingRow, RoutingRow | None, list[Kernel]]]:
    """Take the decisions of ``path`` again, from ``root`` down: the kernel's body through the lift and the cut pass,
    each fork on a kernel ``path`` decides taking the arm its route spells, every other fork keeping the kernel whole.
    Returns, per route of the path in the order the decisions were taken, the route as stored, the route as the
    fresh lowering takes it (``None`` when the fresh parent takes it with another arm — a stale key dropped from a
    route the cut pass still offers — or mints another number of pieces) and the pieces' fresh definitions.
    ``root_identity`` is the identity ``path`` names the root by, when the root was re-keyed."""
    old_of: dict[str, str] = {root.exact_identity: root_identity or root.exact_identity}
    by_parent = {route.parent: route for route in path}
    out: list[tuple[RoutingRow, RoutingRow | None, list[Kernel]]] = []

    def decide(fp):
        if fp.structural and isinstance(fp.root_op, TileOp):
            route = by_parent.get(old_of.get(fp.root_op.identity_key(structural=False, with_io=True)))
            arm = spelled_arm(fp.options, route.arm if route is not None else {})
            if arm is not None:
                return arm[0]
        return next(fp.leaves())

    def on_routing(parent, arm, pieces, _ids) -> None:
        fresh_parent = parent.identity_key(structural=False, with_io=True)
        route = by_parent.get(old_of.get(fresh_parent))
        if route is None:
            return
        kernels = [definition(piece, piece.name) for piece in pieces]
        if len(kernels) != len(route.children) or {str(k): str(v) for k, v in arm.items()} != {
            str(k): str(v) for k, v in route.arm.items()
        }:
            out.append((route, None, kernels))  # another number of pieces, or another arm: not the decision stored
            return
        for old_child, kernel in zip(route.children, kernels, strict=True):
            old_of[kernel.exact_identity] = old_child
        out.append(
            (route, RoutingRow(fresh_parent, {str(k): str(v) for k, v in arm.items()}, tuple(k.exact_identity for k in kernels)), kernels)
        )

    pipeline = Pipeline.build(CUT_PASSES).with_strategies(KernelInventory(on_routing=on_routing))
    composed = []
    for route in path:
        keys = tuple(sorted(key for key, value in route.arm.items() if family_of(key) == "PLACE" and value == "cut"))
        if len(keys) > 1 and (None, keys) not in composed:
            composed.append((None, keys))
    with unpinned_decisions(), composed_routes(composed):
        Run(pipeline=pipeline, ctx=ctx).resolve(root.program({}), decide)
    return out


@dataclass
class Report:
    kernels: int = 0
    rekeyed: list[str] = field(default_factory=list)
    dropped_kernels: list[str] = field(default_factory=list)
    dropped_routes: list[str] = field(default_factory=list)
    demoted: list[str] = field(default_factory=list)
    dropped_rows: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.rekeyed or self.dropped_kernels or self.dropped_routes or self.demoted or self.dropped_rows)

    def lines(self) -> list[str]:
        out = [f"{len(self.rekeyed)} of {self.kernels} kernels re-keyed, {len(self.dropped_kernels)} dropped"]
        out.extend(f"re-keyed {name}" for name in self.rekeyed)
        out.extend(f"dropped kernel {reason}" for reason in self.dropped_kernels)
        out.extend(f"dropped decision {reason}" for reason in self.dropped_routes)
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
    scope = {kernel.exact_identity for kernel in targets}
    for route in document.routing:  # the subtree of every target in scope
        if route.parent in scope:
            scope.update(route.children)
    report.kernels = len(scope)

    fresh: dict[str, Kernel | None] = {}  # old exact identity -> the kernel as the fresh lowering has it, None if dropped
    groups: dict[tuple, list[Kernel]] = {}
    for kernel in targets:
        groups.setdefault((kernel.traced, tuple(sorted(kernel.bindings.items()))), []).append(kernel)
    for (index, bindings), kernels in groups.items():
        lifted = lift_targets(specialize_program(document.program(index), dict(bindings)), ctx)
        for kernel in kernels:
            tile = lifted.get(frozenset(kernel.loop_ir["outputs"]))
            if tile is None:
                fresh[kernel.exact_identity] = None
                report.dropped_kernels.append(f"{kernel.name}: no fresh kernel writes its outputs")
                continue
            fresh[kernel.exact_identity] = _rekeyed(kernel, tile, report)

    routing: list[RoutingRow | None] = []
    for route in document.routing:
        if route.parent not in scope:
            routing.append(route)
            continue
        path = [*document.path_to(route.parent), route]
        root = fresh.get(path[0].parent)
        if root is None or any(fresh.get(step.parent, None) is None for step in path):
            routing.append(None)
            report.dropped_routes.append(f"{route.parent[:12]} {route.arm}: its parent is gone")
            continue
        taken = {
            (old.parent, knobs_json(old.arm)): (new, kernels) for old, new, kernels in mint(root, path, ctx, root_identity=path[0].parent)
        }
        new, kernels = taken.get((route.parent, knobs_json(route.arm)), (None, []))
        if new is None:
            routing.append(None)
            report.dropped_routes.append(f"{route.parent[:12]} {route.arm}: the fresh parent does not take it the same way")
            for child in route.children:
                fresh.setdefault(child, None)
            continue
        routing.append(new)
        for old_child, kernel in zip(route.children, kernels, strict=True):
            if fresh.get(old_child) is None:  # a piece several decisions mint is re-keyed once, by the first
                fresh[old_child] = _rekeyed(document.kernel(old_child), kernel, report)

    rows: list[Row] = []
    for row in document.rows:
        if row.kernel not in scope:
            rows.append(row)
            continue
        kernel = fresh.get(row.kernel)
        if kernel is None:
            report.dropped_rows.append(f"{row.name}: its kernel is gone")
            continue
        fresh_row = replace(row, kernel=kernel.exact_identity)
        if (row.measurements is not None or row.latency is not None) and kernel.exact_identity != row.kernel:
            fresh_row = replace(fresh_row, measurements=None, latency=None)
            report.demoted.append(row.name)
        rows.append(fresh_row)

    kernels: list[Kernel] = []
    for kernel in document.kernels:
        rekeyed = fresh.get(kernel.exact_identity, kernel) if kernel.exact_identity in scope else kernel
        if rekeyed is not None and all(stored.exact_identity != rekeyed.exact_identity for stored in kernels):
            kernels.append(rekeyed)
    for route in [route for route in routing if route is not None]:
        for identity in (route.parent, *route.children):
            if all(stored.exact_identity != identity for stored in kernels):
                routing[routing.index(route)] = None
                report.dropped_routes.append(f"{route.parent[:12]} {route.arm}: a piece is gone")
                break
    out = replace(document, kernels=kernels, routing=[route for route in routing if route is not None], rows=rows)
    return out, report


def _rekeyed(stored: Kernel, tile_or_kernel, report: Report) -> Kernel:
    """``stored`` as the fresh lowering defines it. A kernel that kept its identity keeps its entry — body and name —
    with the fresh lowering's stamps; one the fresh lowering keys differently takes the fresh identity, stamps and
    body, spelled under the stored name, and is reported."""
    fresh = tile_or_kernel if isinstance(tile_or_kernel, Kernel) else definition(tile_or_kernel, stored.name)
    if fresh.exact_identity == stored.exact_identity:
        return replace(stored, stamps=fresh.stamps)
    out = replace(stored, **{f.name: getattr(fresh, f.name) for f in fields(KernelDef) if f.name != "name"})
    out = replace(out, loop_ir=_named(out.loop_ir, stored.name))
    report.rekeyed.append(f"{stored.name} {stored.exact_identity[:12]} -> {out.exact_identity[:12]}")
    return out


def _named(wire: dict, name: str) -> dict:
    """``wire`` with its loop node spelled under ``name``: the one part of a body two lowerings of one kernel spell
    differently, and no part of its identity."""
    nodes = [{**node, "attrs": {**node["attrs"], "name": name}} if node.get("op") == "loop" else node for node in wire["nodes"]]
    return {**wire, "nodes": nodes}
