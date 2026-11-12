"""Divide by a divisor that varies with fewer axes than its numerator once: multiply by its reciprocal, when
FAST_DIV permits the rounding change.

The reciprocal is bound beside the divide it replaces. The body's optimization moves it out of every loop it does
not vary with, as it moves any value.

Run after dtype stamping so integer division stays exact. Keep this policy out of body
normalization: structural identity must not depend on the live arithmetic flags.
"""

from dataclasses import replace
from itertools import count

from emmy.compiler.dtype import BF16, F16, F32, F64
from emmy.compiler.graph import Node
from emmy.compiler.ir.expression import Expression, apply
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Assign, Body, Statement
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.search.space import FAST_DIV, precision_pin

PATTERN = [Pattern("root", KernelOp)]


def rewrite(root: Node) -> KernelOp:
    if not precision_pin(FAST_DIV):
        raise RuleSkipped("FAST_DIV is off")
    op = root.op
    body = split_invariant_divides(op.body)
    if body == op.body:
        raise RuleSkipped("no invariant floating divisor")
    return replace(op, body=body)


def split_invariant_divides(body: Body) -> Body:
    axes = body.axis_dependencies
    names = set(axes)
    fresh = (f"recip_{i}" for i in count() if f"recip_{i}" not in names)

    def depends(expression: Expression) -> frozenset[str]:
        return frozenset().union(*(axes.get(name, frozenset()) for name in expression.free_vars()))

    def split(stmt: Statement):
        if not isinstance(stmt, Assign) or stmt.dtype not in (F16, BF16, F32, F64):
            return stmt
        hoisted: list[Statement] = []

        def reciprocal(node: Expression) -> Expression:
            op = node.operation
            if op is None or op.name != "divide":
                return node
            numerator, divisor = node.operands
            if not depends(divisor) < depends(numerator):
                return node
            name = next(fresh)
            hoisted.append(Assign(name, apply("reciprocal", divisor), dtype=stmt.dtype))
            return apply("multiply", numerator, name)

        value = stmt.value.rebuild(reciprocal)
        return (*hoisted, stmt.replace(value=value)) if hoisted else stmt

    return body.map(split)
