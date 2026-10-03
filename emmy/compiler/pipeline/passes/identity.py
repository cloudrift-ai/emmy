"""IdentityStrategy — a kernel's structural identity, owned end to end.

The ``S_*`` row is an extent-aware histogram used by the learned prior. ``I_kernel`` is the exact
schedule-free typed identity measured evidence is keyed by: the loop op's at birth, and the tile
kernel's once the lift has given it a body of its own — the identity the tune DB keys a kernel on
(``search.bench_record.kernel_key``), so a kernel's rows and its forks name it alike. Both are
materialized into ``op.knobs`` once per kernel, at birth:

- **fusion settled** — the end of the pipeline's last non-lowering pass (``on_pass_end`` at the
  computed stamp boundary; run start for a pipeline entering at lowering): the fused body is
  final, so the identity reflects the final form; earlier would give the same logical kernel two
  identities (pre- and post-stamp) and split the tune DB's keyings.
- **minted during lowering** — a cross-CTA split's pieces (``on_splice`` of a lowering pass):
  fresh knob-less TileOps stamped before the fragment enters the graph, so no rule can
  observe an unstamped kernel.
- **given a body of its own** — the lift of a loop op into a tile, the twist of a tile's
  exp-family cluster (``on_rebind`` of a lowering pass): the ``S_*`` row stays, the exact
  identity is re-derived, so the kernel's forks, its rows and its stored definition name it alike.

The ``S_*`` row of every kernel is the features of the loop body it was formed from — the fused loop op's for
a kernel the lift threads it in as ``source``, the loop nest a cut or split piece was re-formed through — which
is the body its ``kernel`` row stores, so re-lowering that definition stamps the kernel the same.

Materializing into knobs (rather than compute-on-read everywhere) is deliberate: the stamped row
rides the engine's rebind knob-merge into every later dialect, which is what keeps a terminal
CudaOp's cache key, its DB rows, and the prior's feature columns carrying the loop-birth
identity its own body could not reproduce. The read API below is knobs-first for the same
reason — compute is the fallback for an op nothing stamped yet.

Direct writes to shared ops are safe here: sibling candidates share op objects only before their
trajectories diverge, and the stamp is a deterministic function of the body — any "leak" writes
the values the sibling would have written. Writes are copy-on-write (a fresh dict), never a
mutation of a possibly-shared knob dict.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline.knob import KERNEL_IDENTITY, STRUCT_PREFIX
from emmy.compiler.pipeline.search.features import stamps, structure_features
from emmy.compiler.pipeline.strategy import PassEndEvent, PipelineStrategy, RebindEvent, RunStartEvent, SpliceEvent
from emmy.compiler.structural import digest

if TYPE_CHECKING:
    from emmy.compiler.graph import Graph

# The passes that lower a final fused body: a kernel minted or rebound there carries its own identity.
_LOWERING = ("tile/", "lowering/")


class IdentityStrategy(PipelineStrategy):
    """Stamp exact identity and structural features at birth; expose the feature signature to
    inventory and training readers (``signature`` / ``op_sig``)."""

    @staticmethod
    def _stamp_boundary(passes: tuple[str, ...]) -> str | None:
        """The pass whose END finalizes the fused kernel bodies for THIS pipeline: the last
        pass before the tile passes (``loop/stamp`` in the full pipeline; ``loop/fusion`` in a shorthand
        pipeline that skips the naming pass). ``None`` for a pipeline that starts at lowering
        (a loop-stage IR resume, a slice tune) — its entry kernels are already final. Computed
        per event from the pass list, never stored: this instance is shared across runs."""
        pre = [name for name in passes if name and not name.startswith(_LOWERING)]
        return pre[-1] if pre else None

    def on_run_start(self, e: RunStartEvent) -> None:
        # A pipeline entering AFTER the fusion boundary never fires a boundary pass end — but
        # its entry graph's kernels are already final, so they stamp at the door. A
        # pipeline that still runs fusion defers to the pass-end stamp: a premature stamp would
        # ride the rebind knob-merge onto fused bodies it no longer describes.
        if self._stamp_boundary(e.passes) is None:
            for node in e.graph.nodes.values():
                self._stamp(node, e.graph)

    def on_pass_end(self, e: PassEndEvent) -> None:
        if e.pass_name != self._stamp_boundary(e.passes):
            return
        for node in e.graph.nodes.values():
            self._stamp(node, e.graph)

    def on_splice(self, e: SpliceEvent) -> None:
        # Kernels minted inside lowering. Fusion-era splices are skipped: their kernels are
        # intermediate bodies whose identity is not final until the stamp boundary.
        if not e.pass_name.startswith(_LOWERING):
            return
        for node in e.fragment.nodes.values():
            op = node.op
            if not isinstance(op, (LoopOp, TileOp)):
                continue
            lifted = e.root_op.dialect == "loop"
            if op.source is None and lifted:
                node.op = op = replace(op, source=e.root_op)
            # Fragment buffers carry the operand Tensors (the pieces' builders add them), so the
            # dtype features read the same values the assembled graph would give. A kernel lifted
            # from a loop op has a body of its own (a twisted reduction is rewritten), and its
            # exact identity is re-derived from it: the ``S_*`` row stays the body's it was formed from.
            self._stamp(node, e.fragment, exact=lifted)

    def on_rebind(self, e: RebindEvent) -> None:
        # A lowering rewrite that gives a kernel a body of its own — the lift of a loop op into a
        # tile, the twist of a tile's exp-family cluster — is the same logical kernel under a new
        # exact identity, re-derived here; the ``S_*`` row stays the fused body's. A schedule keeps the
        # stamp, even one that realizes the kernel through another term (a carried state's serial form):
        # the kernel is the tile its schedule fork was offered, which its rows are filed under.
        op, old = e.node.op, e.replaced
        if not e.pass_name.startswith(_LOWERING) or not isinstance(op, (LoopOp, TileOp)):
            return
        scheduled = isinstance(op, TileOp) and op.schedule is not None and isinstance(old, TileOp) and old.schedule is None
        same_body = type(op) is type(old) and (op.op is old.op if isinstance(op, TileOp) else op.body is old.body)
        if not (scheduled or same_body):
            self._stamp(e.node, e.graph, exact=True)

    def _stamp(self, node, graph: Graph, *, exact: bool = False) -> None:
        op = node.op
        if not isinstance(op, (LoopOp, TileOp)):
            return
        op = with_stamps(op.with_io(graph, node), graph)
        knobs = dict(op.knobs)
        if exact or KERNEL_IDENTITY not in knobs:
            knobs[KERNEL_IDENTITY] = op.identity_key(structural=False, with_io=True)
        node.op = replace(op, knobs=knobs)

    # --- the read API: the one spelling of identity ------------------------------------------

    def signature(self, op, graph: Graph | None = None) -> tuple:
        """The sorted ``S_*`` row — golden-record identity. Knobs-first (the stamped row IS the
        identity every key already embeds); computed from the body only for an op nothing
        stamped yet (pass ``graph`` for the dtype features then)."""
        if isinstance(op, (LoopOp, TileOp)):
            op = with_stamps(op, graph)
        return tuple(sorted(stamps(op).items()))

    def op_sig(self, op, graph: Graph | None = None) -> str:
        """Digest of :meth:`signature` — the tune DB node-table key and the kernel-inventory
        dedup key."""
        return digest(*self.signature(op, graph))


def with_stamps(op: LoopOp | TileOp, graph: Graph | None = None) -> LoopOp | TileOp:
    """``op`` carrying its ``S_*`` row: as it is when stamped (the stamp stays the body's it was formed from,
    whatever a later rewrite does to the body), else stamped with :func:`structure_features` of its ``stamp_body``,
    ``graph`` supplying the operand dtypes. An op with no such body (a test's bare tile) stays unstamped."""
    body = op.stamp_body
    if body is None or any(k.startswith(STRUCT_PREFIX) for k in op.knobs):
        return op
    return replace(op, knobs={**op.knobs, **structure_features(body, graph)})


def stamp_pieces(fragment: Graph) -> Graph:
    """Stamp every kernel of a rewrite's ``fragment`` — the pieces a cut builds, whose rows the cut cleared — so a
    placement arm's kernels carry their ``S_*`` rows before any splice, as the spliced kernels will."""
    for node in fragment.nodes.values():
        if isinstance(node.op, (LoopOp, TileOp)):
            node.op = replace(node.op, knobs=with_stamps(node.op.with_io(fragment, node), fragment).knobs)
    return fragment
