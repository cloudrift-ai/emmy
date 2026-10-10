"""Value numbering with coordinates abstracted: a statement is a function of its coordinates, wherever it sits, and
a body's identity is the hash of its scope tree over those numbers."""

from __future__ import annotations

import glob
import json
import random
from dataclasses import replace

import numpy as np
import pytest

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.ir.stmt.blocks import Cond
from emmy.compiler.ir.stmt.normalize import normalize_body
from emmy.compiler.ir.stmt.order import ordering_constraints
from emmy.compiler.ir.stmt.values import digest, value_numbers


def _numbers(numbering) -> set[str]:
    return {number for number, _ in numbering.numbers.values()}


def _of(numbering, kind: str) -> list[tuple[str, tuple]]:
    return [numbering.numbers[statement] for statement, k in numbering.kind.items() if k == kind]


def _chain(axis: str, x: str, y: str, source: str = "X", sink: str = "Y") -> Body:
    return Body((Loop(Axis(axis, 8), (Load(x, source, (Var(axis),)), Assign(y, "exp", (x,)), Write(sink, (Var(axis),), y))),))


def test_a_statement_numbers_the_same_under_another_loop_name() -> None:
    assert _numbers(value_numbers(_chain("i", "x", "y"))) == _numbers(value_numbers(_chain("j", "p", "q")))


def test_a_composite_index_and_a_plain_axis_are_one_function() -> None:
    """``X[(i * 2) + k]`` under two loops and ``X[g]`` under one read the same function at different coordinates:
    the load and what follows it number alike, with different parameters; only the stores differ."""
    index = BinaryExpr("+", BinaryExpr("*", Var("i"), Literal(2, "int")), Var("k"))
    composite = Body(
        (
            Loop(
                Axis("i", 4),
                (Loop(Axis("k", 2), (Load("x", "X", (index,)), Assign("y", "exp", ("x",)), Write("Y", (Var("i"), Var("k")), "y"))),),
            ),
        )
    )
    plain = Body((Loop(Axis("g", 8), (Load("x", "X", (Var("g"),)), Assign("y", "exp", ("x",)), Write("Y", (Var("g"),), "y"))),))
    a, b = value_numbers(composite), value_numbers(plain)
    for kind in ("load", "assign"):
        [(number_a, params_a)], [(number_b, params_b)] = _of(a, kind), _of(b, kind)
        assert number_a == number_b
        assert params_a != params_b
    assert _of(a, "store")[0][0] != _of(b, "store")[0][0]


def _product(b_index, op: str = "multiply", a: str = "A", b: str = "B") -> Body:
    inner = (
        Load("a", a, (Var("i"), Var("j"))),
        Load("b", b, b_index),
        Assign("m", op, ("a", "b")),
        Write("Y", (Var("i"), Var("j")), "m"),
    )
    return Body((Loop(Axis("i", 4), (Loop(Axis("j", 4), inner),)),))


def test_parameter_maps_tell_a_transpose_apart() -> None:
    straight = value_numbers(_product((Var("i"), Var("j"))))
    transposed = value_numbers(_product((Var("j"), Var("i"))))
    assert _of(straight, "load")[1][0] == _of(transposed, "load")[1][0], "a load is one function of its coordinates"
    assert _of(straight, "assign")[0][0] != _of(transposed, "assign")[0][0], "the product reads them in another map"


def test_a_reduce_keeps_the_free_coordinates_of_a_bound_composite() -> None:
    index = BinaryExpr("+", BinaryExpr("*", Var("i"), Literal(8, "int")), Var("k"))
    body = Body(
        (
            Loop(
                Axis("i", 4),
                (
                    Loop(Axis("k", 8), (Load("x", "X", (index,)), Accum(name="acc", op="add", value="x", axes=("k",)))),
                    Write("Y", (Var("i"),), "acc"),
                ),
            ),
        )
    )
    [(_, params)] = _of(value_numbers(body), "reduce")
    assert params == (("expr", ("Var", 0)),), "the free coordinate, spelled by its binding depth"


@pytest.mark.parametrize("stride,offset", [(1, 0), (1, 4), (2, 0)])
def test_reductions_preserve_their_bound_coordinate_maps(stride, offset) -> None:
    i, k = Var("i"), Var("k")
    body = Body(
        (
            Loop(
                Axis("i", 2),
                (
                    Loop(
                        Axis("k", 4),
                        (
                            Load("a", "X", (i, k)),
                            Load("b", "X", (i, stride * k + offset)),
                            Accum(name="left", value="a", axes=("k",)),
                            Accum(name="right", value="b", axes=("k",)),
                        ),
                    ),
                    Write("Y", (i, Literal(0, "int")), "left"),
                    Write("Y", (i, Literal(1, "int")), "right"),
                ),
            ),
        )
    )
    op = LoopOp(body=body)
    values = np.arange(16, dtype=np.float32).reshape(2, 8)
    expected = np.stack((values[:, :4].sum(1), values[:, offset : offset + 4 * stride : stride].sum(1)), axis=1)
    np.testing.assert_array_equal(op.forward(values), expected)
    assert len(op.body.accums) == (1 if (stride, offset) == (1, 0) else 2)
    renamed = Body(stmt.rename({"i": "row", "k": "column", "left": "first", "right": "second"}) for stmt in body)
    assert _of(value_numbers(body), "reduce") == _of(value_numbers(renamed), "reduce")
    nested = Body((Loop(Axis("batch", 3), body),))
    assert [number for number, _ in _of(value_numbers(body), "reduce")] == [number for number, _ in _of(value_numbers(nested), "reduce")]
    assert normalize_body(op.body) == op.body


@pytest.mark.parametrize("nested", [False, True])
def test_bound_coordinate_maps_survive_multiple_reduced_axes(nested) -> None:
    i, j, k = Var("i"), Var("j"), Var("k")
    axes = ("k",) if nested else ("j", "k")
    reductions = (
        Load("a", "X", (12 * i + 4 * j + k,)),
        Load("b", "X", (12 * i + 4 * j + 2 * k,)),
        Accum(name="left", value="a", axes=axes),
        Accum(name="right", value="b", axes=axes),
    )
    outer = (
        (
            Accum(name="left_outer", value="left", axes=("j",)),
            Accum(name="right_outer", value="right", axes=("j",)),
        )
        if nested
        else ()
    )
    names = ("left_outer", "right_outer") if nested else ("left", "right")
    body = Body(
        (
            Loop(
                Axis("i", 2),
                (
                    Loop(Axis("j", 2), (Loop(Axis("k", 3), reductions), *outer)),
                    *(Write("Y", (i, Literal(column, "int")), name) for column, name in enumerate(names)),
                ),
            ),
        )
    )
    values = np.arange(24, dtype=np.float32)
    expected = np.array(
        [
            [sum(values[12 * row + 4 * col + stride * inner] for col in range(2) for inner in range(3)) for stride in (1, 2)]
            for row in range(2)
        ],
        dtype=np.float32,
    )
    numbered = _of(value_numbers(body), "reduce")
    assert numbered[0][0] != numbered[1][0]
    if nested:
        assert numbered[2][0] != numbered[3][0]
        np.testing.assert_array_equal(LoopOp(body=body).forward(values), expected)


def test_the_key_is_spelling_free_and_the_roles_follow_the_operands() -> None:
    """Buffers are ranked by how the body uses them: the left operand of a subtraction is role 0 whatever it is
    called, and two interchangeable operands of an addition may take either role."""
    key, roles = digest(_chain("i", "x", "y"))
    assert digest(_chain("j", "p", "q", "foo", "bar")) == (key, tuple({"X": "foo", "Y": "bar"}[role] for role in roles))
    straight, swapped = digest(_product((Var("i"), Var("j")), "subtract")), digest(_product((Var("i"), Var("j")), "subtract", "B", "A"))
    assert straight[0] == swapped[0]
    assert straight[1].index("A") == swapped[1].index("B"), "the left operand takes one role whatever it is called"
    assert digest(_product((Var("i"), Var("j")), "add"))[0] == digest(_product((Var("i"), Var("j")), "add", "B", "A"))[0]


def test_types_color_the_roles() -> None:
    body = _product((Var("i"), Var("j")), "subtract")
    narrow_a, narrow_b = digest(body, {"A": "f16", "B": "f32", "Y": "f32"}.get), digest(body, {"A": "f32", "B": "f16", "Y": "f32"}.get)
    assert narrow_a[0] != narrow_b[0], "a narrow left operand and a narrow right operand are two kernels"
    assert narrow_a[0] == digest(_product((Var("i"), Var("j")), "subtract", "B", "A"), {"B": "f16", "A": "f32", "Y": "f32"}.get)[0]


def test_where_a_value_sits_against_a_branch_is_in_the_key() -> None:
    """A load inside a branch and the same load ahead of it are two placements, so two kernels."""
    guard = BinaryExpr("<", Var("i"), Literal(4, "int"))
    inside = Body((Loop(Axis("i", 8), (Cond(guard, (Load("x", "X", (Var("i"),)), Write("W", (Var("i"),), "x"))),)),))
    ahead = Body((Loop(Axis("i", 8), (Load("x", "X", (Var("i"),)), Cond(guard, (Write("W", (Var("i"),), "x"),)))),))
    assert len({digest(inside)[0], digest(ahead)[0], digest(_chain("i", "x", "y", "X", "W"))[0]}) == 3


def _corpus_bodies() -> list[tuple[str, Body]]:
    out = []
    for path in sorted(glob.glob("tests/compiler/realization/**/*.json", recursive=True)):
        document = json.loads(open(path).read())
        for kernel in document.get("kernels", []):
            for node in kernel.get("loop_ir", {}).get("nodes", []):
                if node.get("op") == "loop":
                    out.append((f"{path}:{kernel.get('name')}", normalize_body(Body.from_wire(node["attrs"]["body"]))))
    return out


def _shuffled_and_renamed(body: Body, rng: random.Random) -> tuple[Body, dict[str, str]]:
    """A dependency-valid random order in every scope, then a random spelling of every name, axis and buffer; the
    spelling each buffer took."""

    def reorder(stmts: Body) -> Body:
        stmts = Body.coerce(stmts)
        order = stmts.topological_permutation(ordering_constraints(stmts, effects=True), lambda _index, _stmt: rng.random())
        out = []
        for index in order:
            stmt = stmts[index]
            children = stmt.nested()
            out.append(stmt.with_bodies(tuple(reorder(child) for child in children)) if children else stmt)
        return Body(out)

    shuffled = reorder(body)
    names = {name for stmt in shuffled.iter() for name in stmt.defines()} | {name for stmt in shuffled.iter() for name in stmt.binds_axes()}
    buffers = {name for stmt in shuffled.iter() for name in (*stmt.external_reads(), *stmt.external_writes())}
    fresh = {name: f"r{rng.randrange(10**9)}_{index}" for index, name in enumerate(sorted(names | buffers))}
    renamed = Body(
        [
            stmt.rewrite(
                lambda name: fresh.get(name, name), Sigma.IDENTITY, lambda axis: replace(axis, name=fresh.get(axis.name, axis.name))
            )
            for stmt in shuffled
        ]
    ).rename_buffers(fresh)
    return renamed, {name: fresh[name] for name in buffers}


def test_the_key_and_the_roles_survive_a_reorder_and_a_rename_over_the_corpus() -> None:
    """Over every corpus kernel: a dependency-valid reordering plus a renaming of every name, axis and buffer keeps
    the key, and the roles bind the same buffers — two interchangeable buffers (a gate and an up projection the
    kernel treats alike) may swap roles, so the roles are checked by the key of the body spelled in them."""
    bodies = _corpus_bodies()
    assert len(bodies) > 100
    rng = random.Random(0)
    for index, (name, body) in enumerate(bodies):
        if index % 5:
            continue
        key, arguments = digest(body)
        renamed, spelled = _shuffled_and_renamed(body, rng)
        renamed = normalize_body(renamed)
        other_key, other_arguments = digest(renamed)
        assert other_key == key, name
        roles = {argument: f"b{role}" for role, argument in enumerate(arguments)}
        other_roles = {argument: f"b{role}" for role, argument in enumerate(other_arguments)}
        assert digest(renamed.rename_buffers(other_roles), str)[0] == digest(body.rename_buffers(roles), str)[0], name
