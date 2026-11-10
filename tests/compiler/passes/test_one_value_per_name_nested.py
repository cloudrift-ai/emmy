"""The one-value-per-name sweep reaches into nested scopes.

A projection distributed over a cooperative reduce's lanes is one sweep loop; inside it a sibling
reduce that re-derives the root's cell carries a hoisted copy of that cell under the cell's own name,
and the projection binds the name again for the store. Both live in the loop's one C scope, and the
sweep only compared top-level statements, so nvcc refused the DeepSeek-V4 divide kernels under any
cooperative reduce (``float v125`` declared twice, the serial arm being the only one that built).
Each nested body is its own scope: it sees the names bound around it and re-spells a second value
of a name inside it, without leaking that spelling out."""

from __future__ import annotations

from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expression import Var
from emmy.compiler.ir.stmt import Assign, Body, Loop, Write, parse_body
from emmy.compiler.pipeline.passes.lowering.kernel._factor import _one_value_per_name


def _sweep_loop() -> Loop:
    """The projection over the output sweep: a copy of the root's reduce (its accumulator already
    re-spelled) and the cell it derives, a consumer, then the projection's own cell and store."""
    [loop] = parse_body(
        """
        for a in 0..4:
            for k in 0..4:
                x_acc__s1 = load x[k]
                acc__s1 <- add(acc__s1, x_acc__s1) over k
            v = add(acc__s1, eps)
            w = multiply(v, v)
            for k in 0..4:
                x_acc = load x[k]
                acc <- add(acc, x_acc) over k
            v = add(acc, eps)
            y[a] = v
        """
    )
    return loop


def _names(stmt) -> tuple[str, ...]:
    """The names an ``Assign``'s value reads as direct operands (a bare-name value is its one operand)."""
    operands = (stmt.value,) if isinstance(stmt.value, Var) else stmt.value.operands
    return tuple(operand.name for operand in operands)


def _defines(body: Body) -> list[str]:
    return [name for stmt in body for name in stmt.defines()]


def test_a_second_value_of_a_name_inside_a_nested_scope_is_re_spelled() -> None:
    (swept,) = _one_value_per_name([_sweep_loop()])
    names = _defines(swept.body)
    assert names.count("v") == 1, f"the sweep loop binds v twice in one scope: {names}"
    second = [s for s in swept.body if isinstance(s, Assign) and _names(s) == ("acc", "eps")]
    assert len(second) == 1 and second[0].name != "v"
    store = next(s for s in swept.body if isinstance(s, Write))
    assert store.value == second[0].name, "the store reads the projection's own cell, the second binding"
    consumer = next(s for s in swept.body if isinstance(s, Assign) and s.value.operation == ElementwiseImpl("multiply"))
    assert _names(consumer) == ("v", "v"), "a use before the second binding keeps the first value"


def test_a_read_ahead_of_a_nested_second_binding_keeps_the_outer_name() -> None:
    """A loop reads the value bound around it, then binds that name inside to a different statement.
    The read means the outer value, so it keeps the outer name; the pass re-spells only the
    loop-local binding and the reads after it."""
    outer, loop = parse_body(
        """
        v = exp(x)
        for a in 0..4:
            u = multiply(v, v)
            y[a] = u
            w = load z[a]
            v = exp(w)
            t[a] = v
        """
    )
    kept, swept = _one_value_per_name([outer, loop])
    assert kept == outer
    use, _, _, second, store = swept.body
    assert _names(use) == ("v", "v"), f"the read ahead of the loop-local binding reads the outer v, got {_names(use)}"
    assert second.name != "v" and store.value == second.name
