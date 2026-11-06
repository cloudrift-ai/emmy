"""Unit tests for ``Body`` dependency summaries and queries.

Exercises the dataflow-dependency primitives on small hand-built
bodies — softmax-style reduces, sibling cross-loop reads, hoisted
reciprocals — that mirror the patterns the SDPA-reduce kernel
gates 013/015 care about.
"""

from __future__ import annotations

import pytest

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, StridedLoop


def _ld(name: str, src: str, *idx: str) -> Load:
    return Load(name=name, input=src, index=tuple(Var(v) for v in idx))


def _asn(name: str, op: str, *args: str) -> Assign:
    return Assign(name=name, op=op, args=args)


def _acc(name: str, value: str, op: str = "add") -> Accum:
    return Accum(name=name, value=value, op=op)


# --- construction analyses ------------------------------------------


@pytest.mark.parametrize("fused_first", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_coalesce_declares_a_shared_fold_once_when_one_lowering_fuses_it(fused_first, strided):
    """One cone edge lowers the shared fold ``a`` alone; another fuses ``a`` and its sibling ``b`` into
    one pass over the same axis. In either order the scope declares ``a`` once and still folds ``b``
    (the GDN in_proj projections, read through two edges of one computed-A cone)."""
    axis = Axis("k", 8)
    alone = (_ld("x", "xs", "k"), _ld("wa", "was", "k"), _asn("pa", "multiply", "x", "wa"), _acc("a", "pa"))
    fused = (*alone, _ld("wb", "wbs", "k"), _asn("pb", "multiply", "x", "wb"), _acc("b", "pb"))
    bodies = (fused, alone) if fused_first else (alone, fused)
    loops = tuple(StridedLoop(axis, Literal(0, "int"), 1, body) if strided else Loop(axis, body) for body in bodies)
    # A statement between the two lowerings, as in the cone: the loops stay apart.
    loops = (loops[0], _asn("between", "exp", "a"), loops[1])
    body = Body(loops).coalesce()
    accumulators = [stmt.name for loop in body if isinstance(loop, (Loop, StridedLoop)) for stmt in loop.body if isinstance(stmt, Accum)]
    assert sorted(accumulators) == ["a", "b"]
    # The pass that keeps ``b`` alone drops ``a``'s own reads and keeps the one ``b`` shares.
    (last,) = [loop for loop in body if isinstance(loop, (Loop, StridedLoop)) and "b" in loop.body.ssa_defs]
    assert ("wa" in last.body.ssa_defs) == fused_first
    assert "x" in last.body.ssa_defs


# --- closure shape ---------------------------------------------------


def test_axis_dependencies_stay_compact_through_a_long_ssa_chain():
    """Axis dependence grows with definitions × axes, not transitive SSA edges."""
    chain_length = 4096
    stmts = [_ld("v0", "x", "a")]
    for i in range(1, chain_length + 1):
        stmts.append(_asn(f"v{i}", "negative", f"v{i - 1}"))
    body = Body([Loop(axis=Axis("a", 4), body=stmts)])

    dependencies = body.axis_dependencies

    assert len(dependencies) == chain_length + 1
    assert sum(map(len, dependencies.values())) == chain_length + 1
    assert all(axes == frozenset({"a"}) for axes in dependencies.values())


def test_axis_dependencies_remove_only_the_closed_reduce_axis():
    body = Body(
        [
            Loop(
                axis=Axis("a", 8),
                body=(
                    Loop(
                        axis=Axis("k", 32),
                        body=(_ld("value", "x", "a", "k"), _acc("acc", "value")),
                    ),
                ),
            )
        ]
    )

    assert body.axis_dependencies["value"] == frozenset({"a", "k"})
    assert body.axis_dependencies["acc"] == frozenset({"a"})


def test_axis_dependencies_keep_a_strided_partial_axis():
    body = Body(
        [
            StridedLoop(
                axis=Axis("k", 32),
                start=Var("start"),
                step=Var("step"),
                body=(_ld("value", "x", "k"), _acc("partial", "value")),
            )
        ]
    )

    assert body.axis_dependencies["partial"] == frozenset({"k"})


# --- depends_on ------------------------------------------------------


def test_depends_on_axis_one_way():
    body = Body([_ld("v", "x", "k")])
    assert body.depends_on("v", "k") is True
    assert body.depends_on("k", "v") is False  # k has no closure entry


def test_depends_on_set_args():
    body = Body(
        [
            _ld("a", "x", "i"),
            _ld("b", "y", "j"),
            _asn("c", "add", "a", "b"),
        ]
    )
    assert body.depends_on("c", {"i"}) is True
    assert body.depends_on("c", {"i", "j"}) is True
    assert body.depends_on("c", {"z"}) is False
    assert body.depends_on({"a", "b"}, "i") is True
    assert body.depends_on({"a"}, "j") is False


def test_depends_on_self_in_b():
    """A name listed in both ``a`` and ``b`` short-circuits true:
    ``depends_on(x, {x, ...})`` reads as 'is x related to anything in b?'"""
    body = Body([_ld("v", "x", "k")])
    assert body.depends_on("v", {"v", "z"}) is True


def test_depends_on_external_name_empty_closure():
    """External names (Tile-input buffers, names from enclosing scopes)
    aren't keys in the closure — depends_on treats them as having empty
    deps. Symmetric: querying with an external as ``a`` returns False."""
    body = Body([_ld("v", "x", "k")])
    assert body.depends_on("ext", "k") is False


def test_depends_on_empty_set():
    body = Body([_ld("v", "x", "k")])
    assert body.depends_on(set(), "k") is False
    assert body.depends_on("v", set()) is False


# --- independent -----------------------------------------------------


# --- SDPA-reduce-style sibling-reduce pattern ------------------------
