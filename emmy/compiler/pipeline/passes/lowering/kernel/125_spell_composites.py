"""Write every composite function out in primitive operations.

A composite elementwise function (``sigmoid``, ``relu``, ``square``, ``reciprocal``) is defined by primitive
operations, the ones one machine instruction or one math-library call computes. The definitions are one table, beside
the operations' declarations in the ``emmy-ir`` crate (``Expression.definition``). Here each ``Assign`` of a composite
becomes its definition, one definition per primitive, so a code generator finds primitives only and the body's
optimization shares a primitive two composites compute (the exponential of a ``sigmoid`` and of what divides by it).

It runs after ``120_explicit_conversions``, the one place that decides where an operation computes. A composite is
typed as a whole there: an f16 ``sigmoid`` computes in f32 and converts once, where its primitives, typed one by one,
would each compute at f16 and round three times. After it the ``Assign`` says the dtype and its operand is already
there, so the primitives take that dtype and the body stays typed in full. No pass after ``120`` matches a pattern
this could hide. It decides nothing and answers to no knob: the kernel computes the same values.

A ``FragmentApply`` holds one operation and keeps it, and so does a program a statement carries (a combine, a
``RegStore`` epilogue): whatever generates their code applies the same table.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.expression import Expression, Var
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Assign, Body, Statement
from emmy.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", KernelOp)]


def rewrite(root: Node) -> KernelOp:
    op: KernelOp = root.op
    taken = {name for stmt in op.body.iter() for name in (*stmt.defines(), *stmt.deps())}
    body = _spelled(op.body, taken)
    if body == op.body:
        raise RuleSkipped("no composite function")
    return replace(op, body=body, knobs=dict(op.knobs))


def _spelled(body: Body, taken: set[str]) -> Body:
    out: list[Statement] = []
    for stmt in body:
        definition = stmt.value.definition if isinstance(stmt, Assign) else None
        if definition is not None:
            out.extend(_primitives(stmt, definition, taken))
            continue
        nested = stmt.nested()
        out.append(stmt.with_bodies(tuple(_spelled(inner, taken) for inner in nested)) if nested else stmt)
    return Body(out)


def _primitives(assign: Assign, definition: Expression, taken: set[str]) -> list[Assign]:
    """``assign`` as its composite's ``definition``: one definition per primitive, each at the dtype ``assign``
    computes at, the last one under its name."""
    out: list[Assign] = []

    def named(node: Expression) -> Expression:
        """``node`` as an operand: a name or a literal as it stands, an operation as a definition ahead."""
        if node.operation is None:
            return node
        value = node.with_children([named(child) for child in node.operands])
        index = 0
        while (name := f"{assign.name}__p{index}") in taken:
            index += 1
        taken.add(name)
        out.append(Assign(name, value, assign.dtype))
        return Var(name)

    out.append(Assign(assign.name, definition.with_children([named(child) for child in definition.operands]), assign.dtype))
    return out
