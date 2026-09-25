"""Tests for ``merge_sibling_free_loops`` in ``stmt/normalize.py``.

Builds bodies by hand and asserts on the post-pass structure. The motivating case is a merged
region whose output stores landed in two free loops of one extent because the splicer named
the loops after their own axes; the pass makes that one loop, and the merge-order regression in
``tests/compiler/passes/test_fusion_merge_order.py`` covers the splicer end of it.
"""

from __future__ import annotations

import pytest

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.stmt.blocks import Loop
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.leaves import Assign, Load, Write
from emmy.compiler.ir.stmt.normalize import merge_sibling_free_loops


def _sweep(axis: Axis, *, source: str, output: str, tag: str) -> Loop:
    """A free loop that loads ``source[a]``, negates it and stores ``output[a]``; SSA names are
    namespaced by ``tag`` unless the caller wants a collision."""
    return Loop(
        axis=axis,
        body=(
            Load(name=f"in_{tag}", input=source, index=(Var(axis.name),)),
            Assign(name=f"v_{tag}", op="negative", args=(f"in_{tag}",)),
            Write(output=output, index=(Var(axis.name),), value=f"v_{tag}"),
        ),
    )


def test_merge_joins_equal_extent_free_loops_onto_the_first_axis() -> None:
    body = Body(
        (
            _sweep(Axis("a2", 256), source="x", output="out_a", tag="a"),
            _sweep(Axis("a3", 256), source="y", output="out_b", tag="b"),
        )
    )

    out = merge_sibling_free_loops(body)

    loops = [s for s in out if isinstance(s, Loop)]
    assert len(loops) == 1, f"expected one merged Loop, got {len(loops)}"
    (loop,) = loops
    assert loop.axis.name == "a2"
    writes = [s for s in loop.body if isinstance(s, Write)]
    assert [w.output for w in writes] == ["out_a", "out_b"]
    assert all(w.index == (Var("a2"),) for w in writes), "the incoming body is renamed onto the surviving axis"


def test_merge_keeps_loops_of_different_extents_apart() -> None:
    body = Body(
        (
            _sweep(Axis("a2", 256), source="x", output="out_a", tag="a"),
            _sweep(Axis("a3", 64), source="y", output="out_b", tag="b"),
        )
    )

    out = merge_sibling_free_loops(body)

    assert len([s for s in out if isinstance(s, Loop)]) == 2


def test_merge_refuses_when_the_second_loop_loads_what_the_first_writes() -> None:
    """One walk would let iteration ``a`` of the second body read ``buf[a]`` before every
    iteration of the first body had written its part of ``buf``."""
    body = Body(
        (
            _sweep(Axis("a2", 256), source="x", output="buf", tag="a"),
            _sweep(Axis("a3", 256), source="buf", output="out_b", tag="b"),
        )
    )

    out = merge_sibling_free_loops(body)

    assert len([s for s in out if isinstance(s, Loop)]) == 2


def test_merge_renames_a_colliding_local_apart() -> None:
    body = Body(
        (
            _sweep(Axis("a2", 256), source="x", output="out_a", tag="v"),
            _sweep(Axis("a3", 256), source="y", output="out_b", tag="v"),
        )
    )

    out = merge_sibling_free_loops(body)

    (loop,) = [s for s in out if isinstance(s, Loop)]
    defs = [s.name for s in loop.body if isinstance(s, Assign)]
    assert len(defs) == 2 and len(set(defs)) == 2, defs
    writes = {s.output: s.value for s in loop.body if isinstance(s, Write)}
    assert writes["out_a"] == "v_v"
    assert writes["out_b"] != "v_v" and writes["out_b"] in defs


def test_merge_refuses_when_a_statement_between_defines_what_the_second_reads() -> None:
    first = _sweep(Axis("a2", 256), source="x", output="out_a", tag="a")
    scale = Assign(name="s", op="negative", args=("t",))
    second = Loop(
        axis=Axis("a3", 256),
        body=(
            Load(name="in_b", input="y", index=(Var("a3"),)),
            Assign(name="v_b", op="multiply", args=("in_b", "s")),
            Write(output="out_b", index=(Var("a3"),), value="v_b"),
        ),
    )

    out = merge_sibling_free_loops(Body((first, scale, second)))

    assert len([s for s in out if isinstance(s, Loop)]) == 2


@pytest.mark.parametrize("nested", [False, True], ids=["direct-write", "nested-write"])
@pytest.mark.parametrize("output", ["out_b", "unrelated"])
def test_merge_preserves_intervening_write_order(output: str, nested: bool) -> None:
    """Moving the second loop before a write to its output would change the final value."""
    first = _sweep(Axis("a2", 256), source="x", output="out_a", tag="a")
    between = (
        Loop(axis=Axis("a4", 64), body=(Write(output=output, index=(Var("a4"),), value="value"),))
        if nested
        else Write(output=output, index=(Literal(0, "int"),), value="value")
    )
    second = _sweep(Axis("a3", 256), source="y", output="out_b", tag="b")
    body = Body((first, between, second))

    out = merge_sibling_free_loops(body)

    if output == "out_b":
        assert out == body, "the second loop must remain the last writer of out_b"
    else:
        assert len(out) == 2
        assert out[1] == between
        assert [s.output for s in out[0].body if isinstance(s, Write)] == ["out_a", "out_b"]
