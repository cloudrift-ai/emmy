"""The Kernel IR optimizer's exact rewrites once more, after the decode, under the launch's ranges.

The early pass (``lowering/kernel/032_optimize``) runs before the memory rules spell vector loads, staging and fragment
stores, with no launch in view: a built-in takes what the hardware allows. Here each axis's decode is an expression
over the built-ins and the launch bounds them, so a decode reads the CTA index or the thread index alone, a value the
later rules compute twice is computed once, and what nothing reads drops. Then every index division and remainder by
a power of two over a dividend the ranges prove non-negative becomes a shift and a mask, the spelling a target
takes for them: last, since the simplifier reads ``/`` and ``%``, not shifts. Every rewrite is exact and removes work,
so it decides nothing (passes ``ARCHITECTURE.md``, "Performance is never a pass decision").
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.kernel import KernelOp, optimize_kernel_body, reduce_strength
from emmy.compiler.ir.kernel.ir import Grid, grid_launch_ranges
from emmy.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", KernelOp)]


def rewrite(root: Node) -> KernelOp:
    op: KernelOp = root.op
    grids = [stmt for stmt in op.body if isinstance(stmt, Grid)]
    if len(grids) != 1:
        raise RuleSkipped("not a single-grid kernel")
    launch = grid_launch_ranges(grids[0])
    body = reduce_strength(optimize_kernel_body(op.body, launch), launch)
    if body == op.body:
        raise RuleSkipped("the kernel body is optimized")
    return replace(op, body=body)
