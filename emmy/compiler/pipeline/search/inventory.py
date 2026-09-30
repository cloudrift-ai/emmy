"""The kernel inventory — how a run hears which kernels a lowering minted and which kernel-set
decisions it took — and the routing-row writer that stores such a decision in the tune DB."""

from __future__ import annotations

from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline.passes.identity import IdentityStrategy
from emmy.compiler.pipeline.search.db import SearchDB, knobs_json
from emmy.compiler.pipeline.strategy import SplicedEvent, SpliceEvent, discovered_strategies


def _identity() -> IdentityStrategy:
    """The discovered IdentityStrategy instance — the one spelling of structural identity."""
    return next(s for s in discovered_strategies() if isinstance(s, IdentityStrategy))

    """The splice watcher: how a run hears which kernels a lowering minted and which kernel-set
    decisions it took. The golden import and ``run --record-greedy`` compose one into the run's
    pipeline (``Pipeline.with_strategies``) to record the decisions. Reports each new kernel-bearing
    op — one whose structural identity has not been seen — to ``on_kernel(node_id, op, fragment)``.
    Cross-trajectory by design: a run re-minting the same piece reports it once, and the seen-set
    can be seeded with kernels already known so pieces structurally identical to one of them are
    not reported again. Identity is COMPUTED through the IdentityStrategy's read API, so nothing here
    depends on a stamp having happened or on strategy dispatch order. It derives from
    PipelineStrategy because the pipeline's strategy set is the channel the engine notifies —
    the event protocol is how a search shape hears about splices.

    It also reports each kernel-set decision once per run, to ``on_routing(parent, arm, pieces,
    ids)``: the tile kernel the fork was offered on, the arm's knobs with one key per seam cut (the
    fork's other spellings of a seam resolved through the event's ``aliases``), the pieces as they
    stand in the graph after the splice — a piece's buffers are bound only then, and its identity
    reads them — and the graph ids the splice consumed and minted, ``(root id, minted ids)``,
    which is how a recorded pick attributes its per-kernel launches to the decision that produced
    them. A run that starts over (a greedy retry) reports its decisions afresh: a retired decision
    must not stand."""

    def __init__(self, identity: IdentityStrategy | None = None, on_kernel=None, seen: set[str] | None = None, on_routing=None) -> None:
        self.identity = identity if identity is not None else _identity()
        self.on_kernel = on_kernel
        self.on_routing = on_routing
        self.seen = seen if seen is not None else set()
        self.seen_routes: set[tuple[str, str]] = set()
        self._open: tuple[object, dict, str] | None = None

    def on_run_start(self, e) -> None:
        del e
        self.seen_routes.clear()
        self._open = None

    def on_splice(self, e: SpliceEvent) -> None:
        for nid, node in e.fragment.nodes.items() if self.on_kernel is not None else ():
            op = node.op
            if op.dialect is None:
                continue
            key = self.identity.op_sig(op, e.fragment)
            if key in self.seen:
                continue
            self.seen.add(key)
            self.on_kernel(nid, op, e.fragment)
        self._open = (
            (e.root_op, {e.aliases.get(k, k): v for k, v in e.knobs.items()}, e.match.root_node_id)
            if isinstance(e.root_op, TileOp) and e.knobs
            else None
        )

    def on_spliced(self, e: SplicedEvent) -> None:
        if self._open is None:
            return
        parent, arm, root_id = self._open
        self._open = None
        key = parent.identity_key(structural=False, with_io=True)
        if key is None or self.on_routing is None or (key, knobs_json(arm)) in self.seen_routes:
            return
        self.seen_routes.add((key, knobs_json(arm)))
        pieces = []
        for nid in e.receipt.new_compute_ids:
            node = e.graph.nodes.get(nid)
            if node is not None and node.op.dialect is not None:
                pieces.append(node.op.with_io(e.graph, node))
        self.on_routing(parent, arm, pieces, (root_id, tuple(e.receipt.new_compute_ids)))


def record_routing(db: SearchDB, parent, arm: dict, pieces) -> None:
    """Store one kernel-set decision as definitions: the parent's ``kernel`` row, each piece's, and
    the ``routing`` row linking them by exact identity. A piece with no identity is not a kernel
    the DB can name and is left out of the row."""
    from emmy.compiler.pipeline.search.bench_record import kernel_row  # noqa: PLC0415
    from emmy.compiler.pipeline.search.db import RoutingRow  # noqa: PLC0415

    kernels = [(op.identity_key(structural=False, with_io=True), op) for op in (parent, *pieces)]
    for identity, op in kernels:
        if identity is not None:
            db.record_kernel(kernel_row(op, op.name))
    (parent_key, _), *children = kernels
    db.record_routing(RoutingRow(parent=parent_key, arm=arm, children=tuple(identity for identity, _ in children if identity is not None)))
