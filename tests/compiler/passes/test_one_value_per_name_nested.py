"""The one-value-per-name sweep reaches into nested scopes.

A projection distributed over a cooperative reduce's lanes is one sweep loop; inside it a sibling
reduce that re-derives the root's cell carries a hoisted copy of that cell under the cell's own name,
and the projection binds the name again for the store. Both live in the loop's one C scope, and the
sweep only compared top-level statements, so nvcc refused the DeepSeek-V4 divide kernels under any
cooperative reduce (``float v125`` declared twice, the serial arm being the only one that built).
Each nested body is its own scope: it sees the names bound around it and re-spells a second value
of a name inside it, without leaking that spelling out."""

from __future__ import annotations

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.pipeline.passes.lowering.kernel._factor import _one_value_per_name


def _cell(acc: str) -> Assign:
    return Assign(name="v", op=ElementwiseImpl("add"), args=(acc, "eps"))


def _reduce(acc: str, source: str) -> Loop:
    load = Load(name=f"x_{acc}", input=source, index=(Var("k"),))
    fold = Accum(name=acc, value=f"x_{acc}", op=ElementwiseImpl("add"), axes=("k",))
    return Loop(axis=Axis("k", 4), body=Body((load, fold)))


def _sweep_loop() -> Loop:
    """The projection over the output sweep: a copy of the root's reduce (its accumulator already
    re-spelled) and the cell it derives, a consumer, then the projection's own cell and store."""
    return Loop(
        axis=Axis("a", 4),
        body=Body(
            (
                _reduce("acc__s1", "x"),
                _cell("acc__s1"),
                Assign(name="w", op=ElementwiseImpl("multiply"), args=("v", "v")),
                _reduce("acc", "x"),
                _cell("acc"),
                Write(output="y", index=(Var("a"),), value="v"),
            )
        ),
    )


def _defines(body: Body) -> list[str]:
    return [name for stmt in body for name in stmt.defines()]


def test_a_second_value_of_a_name_inside_a_nested_scope_is_re_spelled() -> None:
    (swept,) = _one_value_per_name([_sweep_loop()])
    names = _defines(swept.body)
    assert names.count("v") == 1, f"the sweep loop binds v twice in one scope: {names}"
    second = [s for s in swept.body if isinstance(s, Assign) and s.args == ("acc", "eps")]
    assert len(second) == 1 and second[0].name != "v"
    store = next(s for s in swept.body if isinstance(s, Write))
    assert store.value == second[0].name, "the store reads the projection's own cell, the second binding"
    consumer = next(s for s in swept.body if isinstance(s, Assign) and s.op == ElementwiseImpl("multiply"))
    assert consumer.args == ("v", "v"), "a use before the second binding keeps the first value"
