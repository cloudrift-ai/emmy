"""Store what every launch of an ordered kernel stores over once, behind the last launch.

A rolled scan whose states nothing reads but the next step keeps the state it ends with alone
(``loop/fusion/005_roll_recurrence``): its store does not read the step, so each launch of the ordered kernel
would write the same cells and only the last launch's would be read. The shared layout of ``035_shared_carry``
stores it after its loop over the steps.
"""

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.expression import BinaryExpression, Literal, Var
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Body, Cond, Load, Write
from emmy.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", KernelOp)]


def rewrite(root: Node) -> KernelOp:
    op = root.op
    if len(op.serial) != 1 or not op.serial[0].extent.is_static:
        raise RuleSkipped("no ordered launches of a known count")
    steps = op.serial[0]
    last = BinaryExpression("==", Var(steps.name), Literal(steps.extent.as_static() - 1, "int"))
    # A buffer the body reads back is its state, one launch behind: every launch's store is read.
    kept = {tensor.name for tensor in root.outputs} - {load.input for load in op.body.iter() if isinstance(load, Load)}

    def guard(stmts: Body) -> Body:
        out = []
        for stmt in stmts:
            if isinstance(stmt, Write) and stmt.output in kept and not any(steps.name in e.free_vars() for e in stmt.index):
                stmt = Cond(last, Body((stmt,)), Body())
            elif stmt.nested() and not (isinstance(stmt, Cond) and stmt.cond == last):
                stmt = stmt.with_bodies(tuple(guard(child) for child in stmt.nested()))
            out.append(stmt)
        return Body(out)

    body = guard(op.body)
    if body == op.body:
        raise RuleSkipped("every store reads the step")
    return replace(op, body=body)
