"""The summaries of a ``Body`` that only the statement module reads: its free names and carried names."""

from __future__ import annotations

import pytest

from emmy.compiler.ir.stmt import Accum, Carry, parse_body


def test_scope_summary_preserves_reads_above_a_shadowed_name():
    body = parse_body(
        """
        use = abs(outer)
        for i in 0..4:
            outer = exp(source)
        """
    )
    inner = body[1].body
    assert inner._free_ssa == frozenset({"source"})
    assert body._free_ssa == frozenset({"outer", "source"})


def test_recursive_type_queries_share_children_and_preserve_mixed_order():
    body = parse_body(
        """
        for i in 0..4:
            state[] <- value seed 0
            sum <- add(sum, value)
        state[] <- value seed 0
        """
    )
    child = body[0].body
    carry, accum = child
    assert child.carries == (carry,)
    assert child.accums == (accum,)
    assert body.iter_of_type(Accum, Carry) == (carry, accum, carry)
    assert child.iter_of_type(Carry) == child.carries
    assert body._carried_names == ("state", "sum")


@pytest.mark.parametrize("dependent", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_coalesce_preserves_finalized_reduction_dependencies(dependent, strided):
    header = f"for k in 0..8{' step 1' if strided else ''}:"
    first = ("v = load x[k]", "maximum <- maximum(maximum, v)")
    second = ("v = load x[k]",)
    if dependent:
        second += ("shifted = subtract(v, maximum)",)
    second += (f"sum <- add(sum, {'shifted' if dependent else 'v'})",)
    loops = (header, *(f"    {line}" for line in first), header, *(f"    {line}" for line in second))
    body = parse_body("\n".join(loops)).coalesce()
    assert len(body) == (2 if dependent else 1)
    assert len(body.loads) == (2 if dependent else 1)
    assert body._carried_names == ("maximum", "sum")
