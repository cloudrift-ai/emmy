"""Session-scoped reuse of fully-lowered kernels — fetch instead of re-lowering.

Greedy lowering of one kernel is a function: the tile its kernel-set decisions start from in, the
lowered ``KernelOp`` out (the cut and schedule picks baked in). This cache memoizes that function at its natural
boundary, so a compile that meets a kernel it has lowered before — the same computation under
different buffer names, in another program of the same session — splices the finished artifact
instead of enumerating, deciding and materializing again.

The boundary is deliberately the END OF THE KERNEL PASSES, before ``lowering/cuda``: the cuda
pass group holds the per-graph negotiations (``005_delegate_zero_init`` injects a
``ZeroPrologue`` into a *neighbor* kernel; ``010`` renders the launch), so artifacts
above the boundary stay a pure function of the kernel while the negotiations run fresh on every
assembled graph. Entries store the ``KernelOp`` with its buffers renamed to ROLE slots
(``Statement.rename_buffers`` over ``Op.canonical_buffers``); a hit renames the slots to the buffers
filling the same roles on the consumer — sound because the key types every role, whatever order
either side declared its io in. The consumer's tile becomes the hit's ``source``, so the hit's name and render are
a fresh lowering's.

The key is the exact variant key (``identity_key(structural=False, with_io=True,
with_knobs=True)`` — an artifact is exact code, never a cluster representative) folded with the
symbolic-dim hints and the live pin fingerprint: everything the pick depended on besides the
evidence. Evidence is the SESSION-freeze trade, exactly :mod:`~emmy.compiler.backend.plan_cache`'s:
the cache is process-local and CALLER-owned (installed on ``Context.kernel_cache`` by whoever
owns the session — nothing installs it by default), so one cache spans one compiler/evidence
view and no second persistent validity contract exists. Greedy-only: the tune search strips the
field (``prepare_ctx``) because an artifact bakes a conclusion and exploration must not inherit
conclusions.

A tile whose lowering produced SEVERAL kernels (a cross-CTA split) poisons its key: the artifact of a multi-kernel
origin is a graph fragment, deliberately out of scope. A cut is not one: each piece it leaves is a tile of its own,
cached and fetched as one kernel.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace

from emmy.compiler.dim import DEFAULT_SEQ_HINT
from emmy.compiler.structural import digest

#: A key whose origin lowered to more than one kernel — never served, never overwritten.
POISON = object()


@dataclass(frozen=True)
class _Entry:
    """One cached lowering: the slot-renamed ``KernelOp``, the slot order (origin inputs then
    outputs — the order a consumer's io maps back onto), and the minting origin's ``id()``
    (how the harvest tells a same-origin second kernel — poison — from an identical twin)."""

    kernel: object
    slots: tuple[str, ...]
    origin_id: int


class KernelCache:
    """``variant key ⊕ hints ⊕ pins → lowered KernelOp`` — LRU, process-local, caller-owned."""

    def __init__(self, cap: int = 128) -> None:
        self.cap = cap
        self._store: OrderedDict[str, object] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def origin_of(kernel):
        """The tile ``kernel``'s kernel-set decisions started from — the one the replay rule meets ahead of the cut:
        the oldest unscheduled tile on its rewrite chain of the kernel's own identity (a decision to keep the kernel
        whole adds knobs and keeps the identity; the twisted rewrite before it changes the identity). ``None`` for a
        kernel no tile stands behind."""
        from emmy.compiler.ir.tile import TileOp  # noqa: PLC0415
        from emmy.compiler.wire import kernel_identity  # noqa: PLC0415

        identity = kernel_identity(kernel)
        tiles = [
            op
            for op in kernel.source_chain()
            if isinstance(op, TileOp) and op.schedule is None and op.identity_key(structural=False, with_io=True) == identity
        ]
        return tiles[-1] if identity is not None and tiles else None

    @staticmethod
    def key_for(op) -> str | None:
        """The cache key of one kernel's origin tile (:meth:`origin_of`) — ``None`` when the op offers no identity.
        The ONE spelling of the composition (consult and harvest both call this)."""
        from emmy.compiler.pipeline.knob import schedule_pin_fingerprint  # noqa: PLC0415

        identity = op.identity_key(structural=False, with_io=True, with_knobs=True)
        if identity is None:
            return None
        io = (*op.inputs.values(), *op.outputs.values())
        hints = tuple(d.hint or DEFAULT_SEQ_HINT for t in io for d in t.shape if not d.is_static)
        return digest(identity, hints, schedule_pin_fingerprint())

    def fetch(self, key: str, target) -> object | None:
        """The cached lowering rebound to ``target``'s buffer names, or ``None`` (miss /
        poisoned). ``target`` is the consuming tile; its buffers fill the stored
        slots by role, which the key types."""
        entry = self._store.get(key)
        if entry is None or entry is POISON:
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return replace(
            entry.kernel,
            body=entry.kernel.body.rename_buffers(dict(zip(entry.slots, target.canonical_buffers(), strict=True))),
            inputs=dict(target.inputs),
            outputs=dict(target.outputs),
        )

    def harvest(self, key: str, kernel, origin) -> None:
        """Record one lowered kernel under its origin's key. A second, different kernel from the
        SAME origin means the lowering was multi-kernel — the key poisons. An equal key from a
        DIFFERENT origin is an identical twin: the stored artifact already covers it."""
        held = self._store.get(key)
        if held is POISON:
            return
        if held is not None:
            if held.origin_id == id(origin) and held.kernel is not kernel:
                self._store[key] = POISON
            return
        names = origin.canonical_buffers()
        decls = {n for s in kernel.body.iter() for n in s.local_decls()}
        buffers = {n for s in kernel.body.iter() for n in (*s.external_reads(), *s.external_writes())} - decls
        if not buffers <= set(names):
            return  # the kernel reads buffers beyond its origin's roles — not a pure single-kernel lowering
        slots = tuple(f"__kc{i}" for i in range(len(names)))
        slot_map = dict(zip(names, slots, strict=True))
        slotted = replace(
            kernel,
            body=kernel.body.rename_buffers(slot_map),
            inputs={slot_map[n]: t for n, t in kernel.inputs.items() if n in slot_map},
            outputs={slot_map[n]: t for n, t in kernel.outputs.items() if n in slot_map},
        )
        self._store[key] = _Entry(kernel=slotted, slots=slots, origin_id=id(origin))
        while len(self._store) > self.cap:
            self._store.popitem(last=False)
