"""Stamp per-statement dtypes on every Load / Assign / Accum / Write.

Runs over a materialized ``KernelOp`` body, after ``010_materialize`` and before the analytical codegen passes
(shared carry, vectorize) see the IR: those passes read dtypes off the statements, and match one-operation values
over names, so the stamps come early and the conversions (``120_explicit_conversions``) last. The rules are
:func:`emmy.compiler.ir.types.stamp_types`. A ``Load`` against a locally-declared buffer (smem) is stamped with the
conversions.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.types import stamp_types
from emmy.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", KernelOp)]


def rewrite(root: Node) -> KernelOp | None:
    op: KernelOp = root.op
    body = stamp_types(op.body, {n: t.dtype for n, t in {**op.inputs, **op.outputs}.items()})
    if body == op.body:
        raise RuleSkipped("every Load/Assign/Write already stamped")
    return replace(op, body=body, knobs=dict(op.knobs))
