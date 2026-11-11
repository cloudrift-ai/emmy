"""The Kernel IR optimizer's exact rewrites over each kernel body.

After the schedule is bound to threads and dtypes are stamped, a kernel body holds work its lowering spelled per cell
or per lane: the scalar tier's copies of one fold step under a ``__c{i}_{j}`` suffix, the decoded coordinates of a
cell and the index arithmetic a tile's axes make constant. The crate's ``optimize_kernel_body`` simplifies every
expression under the ranges of the axes around it, computes each pure value a scope computes twice once, merges
congruent carried states and drops what nothing reads. Each rewrite removes work and adds none, so it decides nothing
and answers to no knob (passes ``ARCHITECTURE.md``, "Performance is never a pass decision"); nothing moves to another
scope, since that is a trade of registers for work. It runs before the memory peepholes, which read the indices it
simplifies.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.kernel import KernelOp, optimize_kernel_body
from emmy.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", KernelOp)]


def rewrite(root: Node) -> KernelOp:
    op: KernelOp = root.op
    body = optimize_kernel_body(op.body)
    if body == op.body:
        raise RuleSkipped("the kernel body is optimized")
    return replace(op, body=body)
