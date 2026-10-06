"""Simplify the exact one-key softmax quotient after kernel materialization."""

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Assign, Body
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.search.space import SOFTMAX_ONE

PATTERN = [Pattern("root", KernelOp)]


def rewrite(root: Node) -> list[KernelOp]:
    op: KernelOp = root.op
    if SOFTMAX_ONE.name in op.knobs:
        raise RuleSkipped("one-element softmax already decided")
    body, changed = _walk(op.body)
    if not changed:
        raise RuleSkipped("no one-key softmax quotient")
    return [
        replace(op, body=body if enabled else op.body, source=op, knobs={**op.knobs, SOFTMAX_ONE.name: enabled})
        for enabled in SOFTMAX_ONE.narrow((0, 1))
    ]


def _walk(body: Body) -> tuple[Body, bool]:
    subtractions: dict[str, str] = {}
    exponents: dict[str, str] = {}
    changed = False
    stmts = []
    for stmt in body:
        nested = stmt.nested()
        if nested:
            bodies = [_walk(child) for child in nested]
            if any(was_changed for _, was_changed in bodies):
                stmt = stmt.with_bodies(tuple(child for child, _ in bodies))
                changed = True
        if isinstance(stmt, Assign):
            if stmt.op.name == "subtract" and len(stmt.args) == 2 and stmt.args[0] == stmt.args[1]:
                subtractions[stmt.name] = stmt.args[0]
            elif stmt.op.name in ("exp", "exp_fast") and len(stmt.args) == 1 and stmt.args[0] in subtractions:
                exponents[stmt.name] = subtractions[stmt.args[0]]
            elif stmt.op.name == "divide" and len(stmt.args) == 2 and stmt.args[0] == stmt.args[1] and stmt.args[0] in exponents:
                stmt = replace(stmt, op=ElementwiseImpl("softmax_one"), args=(exponents[stmt.args[0]],))
                changed = True
        stmts.append(stmt)
    return Body(tuple(stmts)), changed
