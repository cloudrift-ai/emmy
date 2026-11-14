"""Join the conditions that stand side by side and test the same thing into one.

A register tile's epilogue guards each element's store by its row, so the elements of one row stand under as many
conditions of the same test, one after another. Each was a branch of its own, and nothing two of them computed was
shared: the dynamic-M f32 matmul stored its 16 elements under 16 tests, each computing its row's remainder by the
launch's extent again (nvcc's build computed 4). The conditions of a run are one where neither has another side and
the first defines no name the test reads; their bodies keep their order. It runs after the unrolling, which writes
such runs too.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.context import Context
from emmy.compiler.graph import Node
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Body, Cond, Statement
from emmy.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", KernelOp)]


def rewrite(root: Node, ctx: Context) -> KernelOp:
    op: KernelOp = root.op
    body = _joined(op.body)
    if body == op.body:
        raise RuleSkipped("no two conditions side by side test the same thing")
    return replace(op, body=body, knobs=dict(op.knobs))


def _joined(body: Body) -> Body:
    out: list[Statement] = []
    for stmt in body:
        nested = stmt.nested()
        if nested:
            stmt = stmt.with_bodies(tuple(_joined(inner) for inner in nested))
        last = out[-1] if out else None
        if (
            isinstance(stmt, Cond)
            and isinstance(last, Cond)
            and not stmt.else_body
            and not last.else_body
            and stmt.cond == last.cond
            and not {name for inner in last.body.iter() for name in inner.defines()} & stmt.cond.free_vars()
        ):
            out[-1] = last.replace(body=Body((*last.body, *stmt.body)))
        else:
            out.append(stmt)
    return Body(out)
