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


def test_recursive_type_queries_share_children_and_preserve_mixed_order(monkeypatch):
    carry = Carry("state", "value", (), 0)
    accum = _acc("sum", "value")
    child = Body((carry, accum))
    body = Body((Loop(Axis("i", 4), child), carry))
    assert child.carries == (carry,)
    assert child.accums == (accum,)
    assert body.iter_of_type(Accum, Carry) == (carry, accum, carry)
    assert body.iter_of_type(Accum, Carry) is body.iter_of_type(Accum, Carry)
    assert child.iter_of_type(Carry) is child.carries
    visits = 0
    original = Carry.carried_names

    def counted(stmt):
        nonlocal visits
        visits += 1
        return original(stmt)

    monkeypatch.setattr(Carry, "carried_names", counted)
    names = body._carried_names
    assert names == ("state", "sum")
    assert body._carried_names is names
    assert visits == 2


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


def test_load_closure_includes_index_axis_vars():
    body = Body([_ld("v", "x", "k")])
    closure = body._deps_closure
    assert closure["v"] == frozenset({"k"})


def test_assign_closure_unions_args():
    body = Body(
        [
            _ld("a", "x", "i"),
            _ld("b", "y", "j"),
            _asn("c", "add", "a", "b"),
        ]
    )
    closure = body._deps_closure
    assert closure["c"] == frozenset({"a", "b", "i", "j"})


def test_accum_outside_form_subtracts_loop_axis():
    """``acc <- add(in0)`` inside ``Loop k`` exposes ``acc`` to the outer
    scope with ``k`` subtracted (reduced result no longer varies with k)."""
    inner = (_ld("in0", "x", "k"), _acc("acc", "in0", "add"))
    body = Body([Loop(axis=Axis("k", 32), body=inner)])
    closure = body._deps_closure
    assert closure["in0"] == frozenset({"k"})
    assert closure["acc"] == frozenset()  # k subtracted


def test_accum_outer_axis_kept():
    """An outer free axis remains in ``acc``'s closure even when its
    immediate reduce axis is subtracted."""
    inner = (
        Loop(
            axis=Axis("k", 32),
            body=(_ld("in0", "x", "a", "k"), _acc("acc", "in0", "add")),
        ),
    )
    body = Body([Loop(axis=Axis("a", 8), body=inner)])
    closure = body._deps_closure
    assert closure["acc"] == frozenset({"a"})


def test_sibling_reduce_invariance():
    """Mirror SDPA-reduce: first reduce produces ``acc0``, second reduce
    reads ``acc0`` cross-loop. ``acc0`` is loop-invariant w.r.t. the
    second reduce's axis, so depends_on(acc0, k2) should be False."""
    body = Body(
        [
            Loop(
                axis=Axis("k1", 32),
                body=(_ld("in0", "x", "k1"), _acc("acc0", "in0", "max")),
            ),
            Loop(
                axis=Axis("k2", 32),
                body=(
                    _ld("in1", "x", "k2"),
                    _asn("v0", "subtract", "in1", "acc0"),  # cross-loop read of acc0
                    _asn("v1", "exp", "v0"),
                    _acc("acc1", "v1", "add"),
                ),
            ),
        ]
    )
    closure = body._deps_closure
    # acc0 outside-form: closure[in0]={k1} minus {k1} = {} → invariant
    assert closure["acc0"] == frozenset()
    assert body.depends_on("acc0", "k2") is False
    # v0 inside the second loop reads in1 (which has k2) and acc0 (no k2).
    assert body.depends_on("v0", "k2") is True
    assert body.depends_on("v0", "k1") is False
