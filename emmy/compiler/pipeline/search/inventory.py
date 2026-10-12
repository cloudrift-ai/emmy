"""The kernel inventory — how a run hears which kernels a lowering minted and which kernel-set
decisions it took — and the routing-row writers that store such decisions in the tune DB."""

from __future__ import annotations

from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline.search.db import SearchDB, knobs_json
from emmy.compiler.pipeline.strategy import PipelineStrategy, SplicedEvent, SpliceEvent


class KernelInventory(PipelineStrategy):
    """The splice watcher: how a run hears which kernel-set decisions a lowering took. The golden
    restamp and ``run --golden PATH`` compose one into the run's pipeline
    (``Pipeline.with_strategies``) to record the decisions: ``--record-greedy`` into the golden,
    and, after a greedy re-bench, into the tune DB beside its perf rows (:func:`record_measured_routing`).
    It derives from PipelineStrategy because the pipeline's strategy set is the channel the engine
    notifies — the event protocol is how a search shape hears about splices.

    Each kernel-set decision is reported once per run, to ``on_routing(parent, arm, pieces,
    ids)``: the tile kernel the fork was offered on, the arm's knobs with one key per seam cut (the
    fork's other spellings of a seam resolved through the event's ``aliases``), the pieces as they
    stand in the graph after the splice — a piece's buffers are bound only then, and its identity
    reads them — and the graph ids the splice consumed and minted, ``(root id, minted ids)``,
    which is how a recorded pick attributes its per-kernel launches to the decision that produced
    them. A run that starts over (a greedy retry) reports its decisions afresh: a retired decision
    must not stand."""

    def __init__(self, on_routing=None) -> None:
        self.on_routing = on_routing
        self.seen_routes: set[tuple[str, str]] = set()
        self._open: tuple[object, dict, str] | None = None

    def on_run_start(self, e) -> None:
        del e
        self.seen_routes.clear()
        self._open = None

    def on_splice(self, e: SpliceEvent) -> None:
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


def distinct_decisions(decisions) -> list:
    """``decisions`` (``(parent, arm, pieces)`` as the splice watcher reports them) with each decision —
    a parent and an arm — once, as its last report: a greedy retry reports its decisions afresh."""
    last = {
        (parent.identity_key(structural=False, with_io=True), knobs_json(arm)): (parent, arm, pieces) for parent, arm, pieces in decisions
    }
    return list(last.values())


def record_measured_routing(db: SearchDB, decisions, measured: set[str]) -> tuple[int, list[str]]:
    """Store, through :func:`record_routing`, each kernel-set decision of ``decisions`` (``(parent,
    arm, pieces)``, the splice watcher's reports without the graph ids) whose every piece is a kernel
    in ``measured`` (exact identities) or the parent of another decision stored here — a piece cut
    again is priced through its own decision. A decision has no time of its own: it is priced from
    its pieces' rows (``SearchDB.priced_arms``), so storing one whose piece nothing measured, or whose
    piece has no identity and would be left out of the price, would price a decision nobody measured.
    Each decision counts once (:func:`distinct_decisions`). Returns the number of decisions stored as
    routing rows and one reason per decision left out."""
    from emmy.compiler.pipeline.search.db import is_placement_knob  # noqa: PLC0415

    def identity(op):
        return op.identity_key(structural=False, with_io=True)

    refused, pending = [], []
    for parent, arm, pieces in distinct_decisions(decisions):
        keys = [identity(op) for op in (parent, *pieces)]
        if not arm or not all(is_placement_knob(k, v) for k, v in arm.items()):
            refused.append(f"{parent.name} {arm}: not a kernel-set decision")
        elif None in keys:
            refused.append(f"{parent.name} {arm}: a kernel of the decision has no exact identity key")
        else:
            pending.append(((parent, arm, pieces), keys[0], keys[1:]))
    # A nested decision prices its parent's piece only once it is stored itself.
    priced, stored = set(measured), 0
    while ready := [entry for entry in pending if all(key in priced for key in entry[2])]:
        pending = [entry for entry in pending if not any(entry is done for done in ready)]
        for decision, parent_key, _ in ready:
            record_routing(db, *decision)
            priced.add(parent_key)
            stored += 1
    refused.extend(f"{parent.name} {arm}: a piece nothing measured" for (parent, arm, _), _, _ in pending)
    return stored, refused
