"""Scope hygiene for the register-tile scalar ``Load`` dedup.

``_dedup_loads`` collapses identical scalar ``Load``\\ s in one statement list and rewires the
dropped names to the survivor. The rewrite has to reach *into* nested scopes — an epilogue cell's
output sweep consumes the loads hoisted above it — but ``Statement.rewrite`` renames a stmt's own
bindings as well as its reads and descends unconditionally, while SSA names bound inside a ``Loop``
body are scoped to that body.

So an inner scope that merely re-uses the *spelling* of a dropped outer name used to be captured:
its binding was renamed to the survivor, redeclaring it inside the loop and rewiring the inner
arithmetic to the outer buffer's value. The two production call sites (the scalar drain, the
register-tile epilogue cell) pass flat-ish bodies, but the epilogue cell can carry nested output
sweeps or conditions, so the capture was reachable in principle.
"""

from emmy.compiler.ir.stmt import Load, parse_body
from emmy.compiler.pipeline.passes.lowering.kernel._atom import _dedup_loads
from tests.compiler.ir.stmt import listing


def test_dedup_loads_does_not_capture_a_rebinding_inner_scope() -> None:
    """The loop re-binds both ``in0`` and ``in1`` from different buffers — its own bindings and its
    arithmetic must survive the outer ``in1`` -> ``in0`` alias untouched."""
    inner = listing(
        """
        in0 = load x[0]
        in1 = load y[0]
        v = add(in0, in1)
        """
    )
    body = list(
        parse_body(
            """
            in0 = load const[0]
            in1 = load const[0]
            for a in 0..4:
                in0 = load x[0]
                in1 = load y[0]
                v = add(in0, in1)
            """
        )
    )  # the second load is a duplicate -> dropped, alias in1 -> in0

    kept = _dedup_loads(body)

    assert str(kept[0]) == "in0 = load const[0]"
    assert str(kept[1].body) == inner


def test_dedup_loads_still_rewires_an_inner_use_of_the_dropped_name() -> None:
    """The mirror case: the loop only *reads* the dropped name, so the rewrite must descend."""
    body = list(
        parse_body(
            """
            in0 = load const[0]
            in1 = load const[0]
            for a in 0..4:
                v = add(in0, in1)
            """
        )
    )

    kept = _dedup_loads(body)

    assert [s.name for s in kept if isinstance(s, Load)] == ["in0"]
    assert str(kept[-1].body) == "v = add(in0, in0)"
