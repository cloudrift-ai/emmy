"""Hoist reciprocals of invariant floating divisors when FAST_MATH permits the rounding change.

Run after dtype stamping so integer division stays exact. Keep this policy out of body
normalization: structural identity must not depend on the live arithmetic flags.
"""

from dataclasses import replace
from itertools import count

from emmy.compiler.dtype import BF16, F16, F32, F64
from emmy.compiler.graph import Node
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Assign, Body, Stmt
from emmy.compiler.ir.stmt.normalize import hoist_loop_invariants
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.search.space import FAST_MATH, precision_pin

PATTERN = [Pattern("root", KernelOp)]


def rewrite(root: Node) -> KernelOp:
    if not precision_pin(FAST_MATH):
        raise RuleSkipped("FAST_MATH is disabled")
    op = root.op
    body = split_invariant_divides(op.body)
    if body == op.body:
        raise RuleSkipped("no invariant floating divisor")
    return replace(op, body=hoist_loop_invariants(body))


def split_invariant_divides(body: Body) -> Body:
    axes = body.axis_dependencies
    names = set(axes)
    fresh = (f"recip_{i}" for i in count() if f"recip_{i}" not in names)

    def split(stmt: Stmt):
        if isinstance(stmt, Assign) and stmt.op.name == "divide" and stmt.dtype in (F16, BF16, F32, F64):
            numerator, divisor = stmt.args
            if axes.get(divisor, frozenset()) < axes.get(numerator, frozenset()):
                reciprocal = next(fresh)
                return (
                    Assign(reciprocal, "reciprocal", (divisor,), dtype=stmt.dtype),
                    replace(stmt, op="multiply", args=(numerator, reciprocal)),
                )
        return stmt

    return body.map(split)
