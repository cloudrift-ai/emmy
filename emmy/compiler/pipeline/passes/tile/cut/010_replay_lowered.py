"""Fetch a kernel's finished lowering from the session kernel cache — the greedy shortcut.

The rule sits ahead of the cut, on the tile the kernel-set decisions start from: a tile whose exact variant key
(+ hints + pins) was already lowered this session rewrites STRAIGHT to the cached, io-rebound ``KernelOp`` — no
fork, no enumeration, no materialization. The tile becomes the kernel's ``source``, so its name and its buffer
slots are computed exactly as a fresh lowering's. Everything downstream treats it as any other kernel: the tile and
kernel passes don't match a ``KernelOp``, and the ``lowering/cuda`` group (zero-init delegation, code generation) runs
fresh on the assembled graph, per graph, as it must (those are the cross-kernel negotiations the cache boundary
deliberately excludes). A piece a cut leaves is a tile of its own, so it is fetched the same way.

Fires only when the caller installed ``Context.kernel_cache`` (see
:mod:`~emmy.compiler.pipeline.kernel_cache` — caller-owned, greedy-only; the tune search strips
the field, and the pricing probes strip it too, since a replayed kernel offers no fork to
price)."""

from __future__ import annotations

from emmy.compiler.graph import Node
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.kernel_cache import KernelCache

PATTERN = [Pattern("root", TileOp)]


def rewrite(match: Match, root: Node, ctx=None):  # -> the cached KernelOp (kernel IR stays unimported here)
    del match
    cache: KernelCache | None = getattr(ctx, "kernel_cache", None)
    if cache is None:
        raise RuleSkipped("no session kernel cache installed")
    if root.op.schedule is not None or root.op.place.is_mapped:
        raise RuleSkipped("the tile is past its kernel-set decisions")
    key = cache.key_for(root.op)
    kernel = cache.fetch(key, root.op) if key is not None else None
    if kernel is None:
        raise RuleSkipped("kernel not cached")
    return kernel
