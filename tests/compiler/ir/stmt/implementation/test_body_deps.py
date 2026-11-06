"""The cached summaries of a ``Body`` that only the statement module reads: its dependency closure, free names
and carried names.
"""

from __future__ import annotations

import pytest

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.stmt import Accum, Assign, Body, Carry, Load, Loop, StridedLoop


def _ld(name: str, src: str, *idx: str) -> Load:
    return Load(name=name, input=src, index=tuple(Var(v) for v in idx))


def _asn(name: str, op: str, *args: str) -> Assign:
    return Assign(name=name, op=op, args=args)


def _acc(name: str, value: str, op: str = "add") -> Accum:
    return Accum(name=name, value=value, op=op)


def test_scope_summaries_share_nested_analysis(monkeypatch):
    bodies = [Body((_asn("value", "exp", "source"), _acc("sum", "value")))]
    for level in range(32):
        bodies.append(Body((Loop(Axis(f"i{level}", 4), bodies[-1]),)))
    visits = 0
    original = Assign.deps

    def counted(stmt):
        nonlocal visits
        visits += 1
        return original(stmt)

    monkeypatch.setattr(Assign, "deps", counted)
    for body in reversed(bodies):
        assert body._free_ssa == frozenset({"source"})
        assert tuple(stmt.name for stmt in body.accums) == ("sum",)
    assert visits == 1


def test_scope_summary_preserves_reads_above_a_shadowed_name():
    inner = Body((_asn("outer", "exp", "source"),))
    body = Body((_asn("use", "abs", "outer"), Loop(Axis("i", 4), inner)))
    assert inner._free_ssa == frozenset({"source"})
    assert body._free_ssa == frozenset({"outer", "source"})


def test_recursive_type_queries_share_children_and_preserve_mixed_order():
    carry = Carry("state", "value", (), 0)
    accum = _acc("sum", "value")
    child = Body((carry, accum))
    body = Body((Loop(Axis("i", 4), child), carry))
    assert child.carries == (carry,)
    assert child.accums == (accum,)
    assert body.iter_of_type(Accum, Carry) == (carry, accum, carry)
    assert body.iter_of_type(Accum, Carry) is body.iter_of_type(Accum, Carry)
    assert child.iter_of_type(Carry) is child.carries
    names = body._carried_names
    assert names == ("state", "sum")
    assert body._carried_names is names


@pytest.mark.parametrize("dependent", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_coalesce_preserves_finalized_reduction_dependencies(dependent, strided):
    axis = Axis("k", 8)
    first = (_ld("v", "x", "k"), _acc("maximum", "v", "maximum"))
    second = (_ld("v", "x", "k"),)
    if dependent:
        second += (_asn("shifted", "subtract", "v", "maximum"),)
    second += (_acc("sum", "shifted" if dependent else "v"),)
    loops = tuple(StridedLoop(axis, Literal(0, "int"), 1, body) if strided else Loop(axis, body) for body in (first, second))
    body = Body(loops).coalesce()
    assert len(body) == (2 if dependent else 1)
    assert len(body.loads) == (2 if dependent else 1)
    assert body._carried_names == ("maximum", "sum")
