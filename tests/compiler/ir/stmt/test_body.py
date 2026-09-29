"""Body transformations preserve unchanged subtrees and their analysis."""

from dataclasses import replace

import pytest

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Load
from emmy.compiler.ir.stmt.normalize import merge_sibling_reduce_loops, unify_sibling_reduce_axes


@pytest.mark.parametrize(
    "wrap",
    [
        lambda body: Loop(Axis("i", 4), body),
        lambda body: StridedLoop(Axis("i", 4), Literal(0, "int"), 1, body),
        lambda body: Cond(Literal(True), body, Body((Assign("other", "abs", ("input",)),))),
        lambda body: Tile(axes=(), body=body),
    ],
    ids=["loop", "strided", "condition", "tile"],
)
def test_map_preserves_unchanged_bodies_and_rebuilds_changed_path(wrap):
    value = Assign("value", "exp", ("input",))
    child = Body((value,))
    wrapper = wrap(child)
    untouched = Loop(Axis("j", 4), Body((Assign("tail", "abs", ("input",)),)))
    body = Body((wrapper, untouched))
    cached_reads = body.free_ssa
    visited = []

    def unchanged(stmt):
        visited.append(stmt)
        return stmt

    assert body.map(unchanged) is body
    assert body.free_ssa is cached_reads
    assert visited.index(value) < visited.index(wrapper)
    assert len(visited) == len(tuple(body.iter()))

    changed = replace(value, args=("replacement",))
    result = body.map(lambda stmt: changed if stmt is value else stmt)
    assert result is not body and result[0] is not wrapper
    assert result[0].nested()[0] == Body((changed,))
    assert result[0].nested()[1:] == wrapper.nested()[1:]
    assert result[1] is untouched
    assert body[0] is wrapper and child[0] is value
    assert result.free_ssa == frozenset({"input", "replacement"})


def test_map_keeps_drop_and_splice_order():
    dropped = Assign("drop", "abs", ("input",))
    kept = Assign("keep", "exp", ("input",))
    body = Body((dropped, Loop(Axis("i", 4), Body((dropped, kept)))))
    assert body.map(lambda stmt: None if stmt is dropped else stmt.body if isinstance(stmt, Loop) else stmt) == Body((kept,))


@pytest.mark.parametrize("transform", [unify_sibling_reduce_axes, merge_sibling_reduce_loops])
def test_unchanged_reduction_pass_preserves_cached_body(transform):
    body = Body(
        Loop(Axis(axis, 4), Body((Load(value, source, (Var(axis),)), Accum(state, value, axes=(axis,)))))
        for axis, value, source, state in (("i", "x", "X", "left"), ("j", "y", "Y", "right"))
    )
    for depth in range(8):
        body = Body((Cond(Var(f"condition{depth}"), body),))
    carried = body.carried_names
    assert transform(body) is body
    assert body.carried_names is carried
