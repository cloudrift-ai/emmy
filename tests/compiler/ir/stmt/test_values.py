"""Value numbering with coordinates abstracted: a statement is a function of its coordinates, wherever it sits."""

from __future__ import annotations

import glob
import json
import random
from dataclasses import replace

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.ir.stmt.normalize import normalize_body
from emmy.compiler.ir.stmt.order import ordering_constraints
from emmy.compiler.ir.stmt.values import digest, value_numbers


def _numbers(numbering) -> set[str]:
    return {number for number, _ in numbering.numbers.values()}


def _of(numbering, kind: str) -> list[tuple[str, tuple]]:
    return [numbering.numbers[statement] for statement, k in numbering.kind.items() if k == kind]


def test_a_statement_numbers_the_same_under_another_loop_name() -> None:
    def chain(axis: str, x: str, y: str) -> Body:
        return Body((Loop(Axis(axis, 8), (Load(x, "X", (Var(axis),)), Assign(y, "exp", (x,)), Write("Y", (Var(axis),), y))),))

    assert _numbers(value_numbers(chain("i", "x", "y"))) == _numbers(value_numbers(chain("j", "p", "q")))


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


def test_parameter_maps_tell_a_transpose_apart() -> None:
    def product(b_index) -> Body:
        inner = (
            Load("a", "A", (Var("i"), Var("j"))),
            Load("b", "B", b_index),
            Assign("m", "multiply", ("a", "b")),
            Write("Y", (Var("i"), Var("j")), "m"),
        )
        return Body((Loop(Axis("i", 4), (Loop(Axis("j", 4), inner),)),))

    straight = value_numbers(product((Var("i"), Var("j"))))
    transposed = value_numbers(product((Var("j"), Var("i"))))
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
    numbering = value_numbers(body)
    [(_, params)] = _of(numbering, "reduce")
    [statement] = [s for s, kind in numbering.kind.items() if kind == "reduce"]
    assert numbering.free[statement] == {"i"}
    assert params == (("expr", repr(("Var", "i"))),)


def _corpus_bodies() -> list[tuple[str, Body]]:
    out = []
    for path in sorted(glob.glob("tests/compiler/realization/**/*.json", recursive=True)):
        document = json.loads(open(path).read())
        for kernel in document.get("kernels", []):
            for node in kernel.get("loop_ir", {}).get("nodes", []):
                if node.get("op") == "loop":
                    out.append((f"{path}:{kernel.get('name')}", normalize_body(Body.from_wire(node["attrs"]["body"]))))
    return out


def _shuffled_and_renamed(body: Body, rng: random.Random) -> Body:
    """A dependency-valid random order in every scope, then a random spelling of every name and axis."""

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
    fresh = {name: f"r{rng.randrange(10**9)}_{index}" for index, name in enumerate(sorted(names))}
    return Body(
        [
            stmt.rewrite(
                lambda name: fresh.get(name, name), Sigma.IDENTITY, lambda axis: replace(axis, name=fresh.get(axis.name, axis.name))
            )
            for stmt in shuffled
        ]
    )


def test_the_digest_partitions_the_corpus_as_identity_does() -> None:
    """The digest of the stores' numbers, buffers keyed by use, is identity material: it keys two corpus kernels alike
    exactly when ``canonicalize_identity`` does, and a reordering or renaming of a body never reaches it."""
    bodies = _corpus_bodies()
    assert len(bodies) > 100
    by_identity: dict[str, set[str]] = {}
    by_digest: dict[str, set[str]] = {}
    rng = random.Random(0)
    for index, (name, body) in enumerate(bodies):
        key = digest(body)
        by_identity.setdefault(body.structural_key(), set()).add(name)
        by_digest.setdefault(key, set()).add(name)
        if index % 7 == 0:
            assert digest(normalize_body(_shuffled_and_renamed(body, rng))) == key, name
    assert set(map(frozenset, by_identity.values())) == set(map(frozenset, by_digest.values()))
