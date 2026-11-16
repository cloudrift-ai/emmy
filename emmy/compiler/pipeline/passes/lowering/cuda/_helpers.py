"""Shared helpers for the ``lowering/cuda`` rules.

Lives in a ``_``-prefixed module so the pass loader skips it (only ``NNN_<name>.py`` files load
as rules); ``005_delegate_zero_init`` and ``010_lower_kernelop`` both need the atomic-output
walk, so it is defined once here.
"""

from __future__ import annotations

from emmy.compiler.dim import paged_declarations
from emmy.compiler.graph import Graph
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.kernel.ir import RegStore
from emmy.compiler.ir.stmt import Write


def atomic_outputs(kernel: KernelOp) -> tuple[str, ...]:
    """Output buffers an atomic reduce-write (``030_cut``'s atomic finalize) accumulates
    into — they must be zero-init'd before each launch (``CudaOp.zero_outputs``), since every
    contributing CTA ``atomicAdd``\\ s into the same cell. The scalar tier's atomic ``Write``
    survives materialization verbatim; the mma tier's is a ``RegStore(atomic=True)``.
    Dict-keyed for stable order."""
    seen: dict[str, None] = {}
    for s in kernel.body.iter():
        if isinstance(s, Write) and s.atomic:
            seen.setdefault(s.output, None)
        elif isinstance(s, RegStore) and s.atomic:
            seen.setdefault(s.dst_buffer, None)
    return tuple(seen)


def paged_buffers(kernel: KernelOp, graph: Graph) -> tuple[tuple[str, int, int], ...]:
    """The kernel's buffers the graph declares paged, each as ``(name, axis, page)``: a device table of equal-sized
    pages rather than one allocation, so each read and write resolves its page first — the KV cache, whose pages
    belong to a request rather than to one contiguous slab. A graph-level hint, set by the caller before compile and
    ABI-level only: shapes are untouched, so the schedule search and the kernel cache never see it."""
    scope = {*kernel.inputs, *kernel.outputs}
    declared = paged_declarations(graph.hints.get("cuda.paged_buffers", ()), graph)
    return tuple((name, axis, page) for name, (axis, page, _) in declared.items() if name in scope)
