"""Store what every launch of an ordered kernel stores over once, behind the last launch.

A rolled scan whose states nothing reads but the next step keeps the state it ends with alone
(``loop/fusion/005_roll_recurrence``): its store does not read the step, so each launch of the ordered kernel
would write the same cells and only the last launch's would be read. The shared layout of ``035_shared_carry``
guards the same store inside its one launch.
"""

from dataclasses import replace
from importlib import import_module

from emmy.compiler.graph import Node
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", KernelOp)]
_at_the_last_step = import_module("emmy.compiler.pipeline.passes.lowering.kernel.035_shared_carry").at_the_last_step


def rewrite(root: Node) -> KernelOp:
    op = root.op
    if len(op.serial) != 1 or not op.serial[0].extent.is_static:
        raise RuleSkipped("no ordered launches of a known count")
    body = _at_the_last_step(op.body, op.serial[0], {tensor.name for tensor in root.outputs})
    if body == op.body:
        raise RuleSkipped("every store reads the step")
    return replace(op, body=body)
