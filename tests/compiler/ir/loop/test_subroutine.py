"""Fusion's subroutine calls expand to the body an inline expansion keys as: one form for executable CSE, identity
and Tile IR."""

from dataclasses import replace

import numpy as np
import pytest

from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.loop.assembly import definitions, expand_calls
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Assign, Body, Call, Load, Loop, Subroutine, Write, normalize_body, parse_body, prepare_body
from emmy.compiler.pipeline import Pipeline


def _project() -> Subroutine:
    return Subroutine(
        "project",
        (Axis("row", 3), Axis("col", 8)),
        Body(
            (
                Loop(
                    Axis("k", 16),
                    Body(
                        (
                            Load("xv", "x", (Var("row"), Var("k"))),
                            Load("wv", "weight", (Var("col"), Var("k"))),
                            Assign("product", "multiply", ("xv", "wv")),
                            Accum("sum", "product", "add", axes=("k",)),
                        )
                    ),
                ),
            )
        ),
        "sum",
    )


def _body(target: Subroutine, offset: int, *, other: Subroutine | None = None) -> Body:
    # The caller's k shadows the definition's internal k: expansion must be hygienic.
    return Body(
        (
            Loop(
                Axis("k", 3),
                Body(
                    (
                        Loop(
                            Axis("c", 8),
                            Body(
                                (
                                    Call("left", target, (Var("k"), Var("c"))),
                                    Call("right", other or target, (Var("k"), (Var("c") + Literal(offset, "int")) % Literal(8, "int"))),
                                    Assign("value", "add", ("left", "right")),
                                    Write("out", (Var("k"), Var("c")), "value"),
                                )
                            ),
                        ),
                    )
                ),
            ),
        )
    )


@pytest.mark.parametrize("offset", [0, 4])
def test_shifted_calls_preserve_values_and_share_common_loads(offset):
    body = _body(_project(), offset)
    prepared = prepare_body(body)
    assert len(definitions(prepared)) == 1
    assert len(tuple(prepared.iter_of_type(Call))) == (1 if offset == 0 else 2)

    op = LoopOp(body=expand_calls(body))
    assert not tuple(op.body.iter_of_type(Call))
    assert len(op.body.accums) == (1 if offset == 0 else 2)
    assert len([load for load in op.body.loads if load.input == "x"]) == 1
    x = np.linspace(-1, 1, 48, dtype=np.float32).reshape(3, 16)
    weight = np.linspace(-0.5, 0.5, 128, dtype=np.float32).reshape(8, 16)
    inputs = {"x": x, "weight": weight}
    actual = op.forward(*(inputs[name] for name in op.inputs))
    projected = x @ weight.T
    np.testing.assert_allclose(actual, projected + np.roll(projected, -offset, axis=1), rtol=2e-6, atol=2e-6)


def test_full_cse_crosses_distinct_definitions_and_tile_lift():
    target = _project()
    other = replace(target, name="another_projection")
    compact = _body(target, 0, other=other)
    shared = _body(target, 0)
    assert len(definitions(compact)) == 2
    assert expand_calls(compact).structural_key() == expand_calls(shared).structural_key()
    op = LoopOp(body=expand_calls(compact))
    assert len(op.body.accums) == 1
    graph = Graph()
    graph.add_node(op, [], Tensor("out", (3, 8)), node_id="out")
    graph.outputs = ["out"]
    tile = Pipeline.build(["tile/lift"], select=["lift"]).run(graph).nodes["out"].op
    assert len(tile.loop_body.accums) == 1


def test_a_body_that_holds_a_call_has_no_normal_form():
    with pytest.raises(ValueError, match="no normal form"):
        normalize_body(_body(_project(), 4))


def test_outlining_preserves_inline_identity():
    """Equivalent outlining affects neither the exact identity nor the compute-unit-clustered one."""
    target = _project()
    compact = _body(target, 4)

    def inline(stmt):
        if not isinstance(stmt, Call):
            return stmt
        rename = {name: f"{stmt.name}_{name}" for name in (*target.body.ssa_defs, *target.body.axis_names)}
        sigma = Sigma(dict(zip(target.params, stmt.args, strict=True)))
        return (*[s.rename(rename).substitute(sigma) for s in target.body], Assign(stmt.name, "copy", (rename[target.result],)))

    inlined, expanded = compact.map(inline), expand_calls(compact)
    for structural in (False, True):
        assert expanded.identity(structural=structural).key == inlined.identity(structural=structural).key


def test_a_compact_body_prints_a_shared_definition_once_and_reads_back():
    rendered = str(_body(_project(), 4))
    assert rendered.count("sub project(row in 0..3, col in 0..8) -> sum:") == 1
    assert "left = call project[x, weight](k, c)" in rendered
    assert "right = call project[x, weight](k," in rendered
    assert len(rendered.splitlines()) < 18
    back = parse_body(rendered)
    assert str(back) == rendered
    assert len(definitions(back)) == 1


def test_buffer_renaming_preserves_shared_definitions():
    compact = _body(_project(), 4)
    renamed = compact.rename_buffers({"x": "input", "weight": "matrix", "out": "output"})
    assert len(definitions(renamed)) == 1
    assert definitions(renamed)[0].buffers == ("input", "matrix")
    expanded = LoopOp(body=expand_calls(compact)).body.rename_buffers({"x": "input", "weight": "matrix", "out": "output"})
    assert expand_calls(renamed).structural_key() == expanded.structural_key()


def test_a_body_without_output_writes_expands_from_its_terminal_values():
    body = _body(_project(), 0).map(lambda stmt: None if isinstance(stmt, Write) else stmt)
    normalized = normalize_body(expand_calls(body))
    assert not tuple(normalized.iter_of_type(Call))
    assert len(normalized.accums) == 1
    assert len(normalized.loads) == 2


def test_nested_calls_freshen_locals_without_capturing_arguments():
    inner = _project()
    outer = Subroutine("wrapper", inner.axes, Body((Call("result", inner, tuple(Var(p) for p in inner.params)),)), "result")
    # A name resembling an inliner's generated local must remain a caller coordinate.
    compact = _body(outer, 4).map(lambda stmt: stmt.rename({"k": "k__call0"}))
    assert expand_calls(compact).structural_key() == expand_calls(_body(inner, 4)).structural_key()


def test_normalization_drops_an_unused_value_the_expansion_kept_beside_output_writes():
    """A value no store reaches is work the kernel never needed: the expansion keeps it, normalization drops it."""
    body = Body((*_body(_project(), 0), Assign("unused", "exp", ("external",))))
    normalized = normalize_body(expand_calls(body))
    assert len(normalized.writes) == 1
    assert not any(isinstance(stmt, Assign) and stmt.op.name == "exp" for stmt in normalized.iter())


def test_formal_coordinates_do_not_capture_canonical_axis_names():
    target = _project()
    target = replace(target, axes=(Axis("a0", 3), Axis("a1", 8)), body=Body(s.rename({"row": "a0", "col": "a1"}) for s in target.body))
    assert expand_calls(_body(target, 4)).structural_key() == expand_calls(_body(_project(), 4)).structural_key()


def test_expansion_does_not_capture_free_caller_coordinates():
    body = Body((Call("result", _project(), (Var("wv_s1"), Literal(0, "int"))),))
    normalized = normalize_body(expand_calls(body))
    (load,) = [stmt for stmt in normalized.loads if stmt.input == "x"]
    assert load.index[0] == Var("wv_s1")
    assert "wv_s1" not in normalized.ssa_defs
