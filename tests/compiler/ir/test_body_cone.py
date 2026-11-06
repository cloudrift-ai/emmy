"""Unit tests for ``Body.backward_cone``."""

from __future__ import annotations

from emmy.compiler.ir.stmt import Body, parse_body


def _cell() -> Body:
    """A demoted-matmul cell: ``p = cone(x) * w`` with a plain B load."""
    return parse_body(
        """
        a0 = load x[m, k]
        a1 = negative(a0)
        b = load w[k, n]
        p = multiply(a1, b)
        acc <- add(acc, p)
        """
    )


# --- backward_cone ----------------------------------------------------


def test_backward_cone_members_in_body_order():
    body = _cell()
    cone = body.backward_cone(["a1"])
    assert [s.defines()[0] for s in cone.members] == ["a0", "a1"]


def test_backward_cone_externals_carry_axes_not_members():
    cone = _cell().backward_cone(["a1"])
    # The cone reads its index axes from the enclosing scope; the resolved
    # SSA chain (a0) is internal, the untouched B side absent entirely.
    assert cone.external_reads == frozenset({"m", "k"})


def test_backward_cone_root_not_defined_here_is_external():
    """A root from another scope level yields no members and surfaces as an
    external read — the chaining signal, not an error."""
    cone = _cell().backward_cone(["stat"])
    assert cone.members == ()
    assert cone.external_reads == frozenset({"stat"})


def test_backward_cone_wrapper_member_joins_whole():
    """A reduce Loop resolves through its exposed Accum and joins as one
    member; its own loop axis is internal, the row axis it reads is not."""
    body = parse_body(
        """
        for r in 0..32:
            s0 = load x[m, r]
            stat <- maximum(stat, s0)
        e = exp(stat)
        """
    )
    cone = body.backward_cone(["e"])
    assert cone.members == (body[0], body[1])
    assert "r" not in cone.external_reads
    assert "m" in cone.external_reads


def test_backward_cone_select_predicate_axis_is_read():
    """An axis reaching the cone only through a Select predicate counts as
    read — materializing without it would drop a real dependence (the
    rotate-half rotary shape)."""
    body = parse_body(
        """
        a0 = load x[k]
        a1 = load x[m]
        v = select:
            a0 when n < m
            a1 when m
        """
    )
    cone = body.backward_cone(["v"])
    assert "n" in cone.external_reads
