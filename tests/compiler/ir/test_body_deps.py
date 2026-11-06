"""Unit tests for ``Body`` dependency summaries and queries.

Exercises the dataflow-dependency primitives on small hand-built
bodies — softmax-style reduces, sibling cross-loop reads, hoisted
reciprocals — that mirror the patterns the SDPA-reduce kernel
gates 013/015 care about.
"""

from __future__ import annotations

import pytest

from emmy.compiler.ir.stmt import Accum, Loop, StridedLoop, parse_body

# --- construction analyses ------------------------------------------


@pytest.mark.parametrize("fused_first", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_coalesce_declares_a_shared_fold_once_when_one_lowering_fuses_it(fused_first, strided):
    """One cone edge lowers the shared fold ``a`` alone; another fuses ``a`` and its sibling ``b`` into
    one pass over the same axis. In either order the scope declares ``a`` once and still folds ``b``
    (the GDN in_proj projections, read through two edges of one computed-A cone)."""
    head = "for k in 0..8 step 1:" if strided else "for k in 0..8:"
    alone = (head, "    x = load xs[k]", "    wa = load was[k]", "    pa = multiply(x, wa)", "    a <- add(a, pa)")
    fused = (*alone, "    wb = load wbs[k]", "    pb = multiply(x, wb)", "    b <- add(b, pb)")
    bodies = (fused, alone) if fused_first else (alone, fused)
    # A statement between the two lowerings, as in the cone: the loops stay apart.
    body = parse_body("\n".join((*bodies[0], "between = exp(a)", *bodies[1]))).coalesce()
    accumulators = [stmt.name for loop in body if isinstance(loop, (Loop, StridedLoop)) for stmt in loop.body if isinstance(stmt, Accum)]
    assert sorted(accumulators) == ["a", "b"]
    # The pass that keeps ``b`` alone drops ``a``'s own reads and keeps the one ``b`` shares.
    (last,) = [loop for loop in body if isinstance(loop, (Loop, StridedLoop)) and "b" in loop.body.definitions]
    assert ("wa" in last.body.definitions) == fused_first
    assert "x" in last.body.definitions


# --- closure shape ---------------------------------------------------


def test_axis_dependencies_stay_compact_through_a_long_ssa_chain():
    """Axis dependence grows with definitions × axes, not transitive SSA edges."""
    chain_length = 4096
    stmts = ["    v0 = load x[a]"]
    for i in range(1, chain_length + 1):
        stmts.append(f"    v{i} = negative(v{i - 1})")
    body = parse_body("\n".join(("for a in 0..4:", *stmts)))

    dependencies = body.axis_dependencies

    assert len(dependencies) == chain_length + 1
    assert sum(map(len, dependencies.values())) == chain_length + 1
    assert all(axes == frozenset({"a"}) for axes in dependencies.values())


def test_axis_dependencies_remove_only_the_closed_reduce_axis():
    body = parse_body(
        """
        for a in 0..8:
            for k in 0..32:
                value = load x[a, k]
                acc <- add(acc, value)
        """
    )

    assert body.axis_dependencies["value"] == frozenset({"a", "k"})
    assert body.axis_dependencies["acc"] == frozenset({"a"})


def test_axis_dependencies_keep_a_strided_partial_axis():
    body = parse_body(
        """
        for k in start..32 step step:
            value = load x[k]
            partial <- add(partial, value)
        """
    )

    assert body.axis_dependencies["partial"] == frozenset({"k"})


# --- independent -----------------------------------------------------


# --- SDPA-reduce-style sibling-reduce pattern ------------------------
