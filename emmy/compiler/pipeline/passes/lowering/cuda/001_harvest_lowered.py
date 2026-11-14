"""Harvest every finished single-kernel lowering into the session kernel cache.

An observer rule at the head of ``lowering/cuda`` — the cache boundary: the ``KernelOp`` here is
fully scheduled and materialized but not yet touched by the per-graph negotiations (zero-init
delegation rewrites a NEIGHBOR kernel's body; rendering bakes graph buffer names), so it is the
pure function-of-the-kernel artifact. The origin is the tile the kernel's kernel-set decisions started from
(:meth:`~emmy.compiler.pipeline.kernel_cache.KernelCache.origin_of`), the tile the replay rule meets; a second,
different kernel from one origin poisons the key — a split lowered the origin to a graph fragment, which the
single-kernel cache deliberately refuses.

Never rewrites (always ``RuleSkipped``); fires only when the caller installed
``Context.kernel_cache``."""

from __future__ import annotations

from emmy.compiler.graph import Node
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.kernel_cache import KernelCache

PATTERN = [Pattern("root", KernelOp)]


def rewrite(match: Match, root: Node, ctx=None) -> None:
    del match
    cache: KernelCache | None = getattr(ctx, "kernel_cache", None)
    if cache is not None:
        origin = KernelCache.origin_of(root.op)
        if origin is not None and (key := cache.key_for(origin)) is not None:
            cache.harvest(key, root.op, origin)
    raise RuleSkipped("observer — harvest never rewrites")
