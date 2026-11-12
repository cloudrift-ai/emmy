"""Type the lowered body in full: every operation at one dtype, every conversion a definition of its own.

The last kernel pass. Until here a conversion is implied: an ``Assign`` over operands of another dtype, a ``Write`` of
an f32 value into an f16 buffer. The rules that say where each operation computes (natively in f16 where the target
has the operation, else in f32 with the result converted once) are :func:`emmy.compiler.ir.types.explicit_conversions`;
after it a code generator reads the body and decides nothing, and ``check_types`` holds.

It runs last because the passes before it match one-operation values over the names the lowering wrote: a conversion
between a load and its reader would hide the pattern from the vector passes, the shared carry and the load reuse.
It decides nothing and answers to no knob: the kernel computes the same values.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.kernel.interface import buffer_dtypes
from emmy.compiler.ir.types import explicit_conversions
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.target import native_f16_ops

PATTERN = [Pattern("root", KernelOp)]


def rewrite(root: Node) -> KernelOp:
    op: KernelOp = root.op
    body = explicit_conversions(op.body, buffer_dtypes(op), native_f16_ops())
    if body == op.body:
        raise RuleSkipped("every conversion already explicit")
    return replace(op, body=body, knobs=dict(op.knobs))
