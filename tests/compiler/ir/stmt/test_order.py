"""Exact graph labeling and statement-order normalization tests."""

from __future__ import annotations

from itertools import product

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import CpAsyncCommit, CpAsyncWait, Sync, WgmmaCommit, WgmmaFence, WgmmaWait
from emmy.compiler.ir.stmt import Assign, Body, Cond, Let, Load, Loop, Write, canonicalize_identity, normalize_body, ordering_constraints


def test_kahn_tie_break_is_optional() -> None:
    body = Body((Assign(name="right", op="exp", args=("x",)), Assign(name="left", op="abs", args=("x",))))
    incoming = (frozenset(), frozenset())
    assert body.topological_order(incoming) == body
    assert body.topological_order(incoming, lambda _index, stmt: stmt.op.name) == Body(reversed(body))


def test_effect_constraints_keep_only_intervening_resource_hazards() -> None:
    writes = Body(Write(output="X", index=(), value=f"value_{index}") for index in range(100))
    incoming = ordering_constraints(writes, effects=True)
    assert incoming[0] == set()
    assert incoming[1:] == [{index - 1} for index in range(1, len(writes))]

    body = Body(
        (
            Write(output="X", index=(), value="first"),
            Load(name="left", input="X", index=()),
            Load(name="right", input="X", index=()),
            Write(output="X", index=(), value="second"),
            Load(name="tail", input="X", index=()),
        )
    )
    incoming = ordering_constraints(body, effects=True)
    assert incoming == [set(), {0}, {0}, {0, 1, 2}, {3}]


def test_protocol_statement_order_is_identity() -> None:
    protocols = (
        (CpAsyncCommit(), CpAsyncWait(), Sync()),
        (CpAsyncWait(), CpAsyncCommit(), Sync()),
        (WgmmaFence(), WgmmaCommit(), WgmmaWait()),
        (WgmmaCommit(), WgmmaFence(), WgmmaWait()),
    )
    assert len({Body(protocol).structural_key(structural=False) for protocol in protocols}) == len(protocols)


def test_identity_combines_buffer_rename_with_nested_reordering() -> None:
    def make(*, reverse: bool, renamed: bool) -> Body:
        axis = "renamed_axis" if renamed else "axis"
        names = ("renamed_left", "renamed_right") if renamed else ("left", "right")
        buffers = ("renamed_X", "renamed_Y") if renamed else ("X", "Y")
        chains = tuple(
            Cond(
                cond=BinaryExpr("<", Var(axis), Literal(4, "int")),
                body=(
                    Load(name=name, input=buffer, index=(Var(axis),)),
                    Assign(name=f"{name}_value", op="abs", args=(name,)),
                    Write(output=f"{buffer}_output", index=(Var(axis),), value=f"{name}_value"),
                ),
            )
            for name, buffer in zip(names, buffers, strict=True)
        )
        return Body((Loop(axis=Axis(axis, 4), body=tuple(reversed(chains)) if reverse else chains),))

    keys = {
        canonicalize_identity(normalize_body(make(reverse=reverse, renamed=renamed))).key
        for reverse, renamed in product((False, True), repeat=2)
    }
    assert len(keys) == 1


def test_repeated_definition_capture_uses_nearest_predecessor() -> None:
    def make(name: str, inner: str, result: str, *, late_before_capture: bool) -> Body:
        first = Load(name=name, input="A", index=())
        capture = Cond(cond=Literal(True), body=(Assign(name=inner, op="abs", args=(name,)), Write(output="P", index=(), value=inner)))
        second = Load(name=name, input="B", index=())
        tail = Assign(name=result, op="exp", args=(name,))
        middle = (second, capture) if late_before_capture else (capture, second)
        return Body((first, *middle, tail, Write(output="O", index=(), value=result)))

    before = normalize_body(make("value", "inside", "result", late_before_capture=False))
    renamed = normalize_body(make("renamed", "renamed_inside", "renamed_result", late_before_capture=False))
    after = normalize_body(make("value", "inside", "result", late_before_capture=True))
    assert before == renamed
    assert before != after


def test_shadowing_after_the_read_keeps_the_outer_dependency() -> None:
    """A deeper scope's rebind of a name binds nothing read above it: the block still depends on
    the enclosing definition, so it cannot sort above it."""
    body = Body(
        (
            Cond(
                cond=Literal(True),
                body=(
                    Assign(name="y", op="abs", args=("x",)),
                    Cond(cond=Literal(True), body=(Let(name="x", value=2.0), Write(output="P", index=(), value="x"))),
                    Write(output="O", index=(), value="y"),
                ),
            ),
            Load(name="x", input="X", index=()),
        )
    )
    normalized = normalize_body(body)
    assert isinstance(normalized[0], Load)
    assert isinstance(normalized[1], Cond)


def test_structural_key_is_invariant_under_random_renaming_and_reordering() -> None:
    """Seeded property check: independent load/compute/write chains keyed under every
    interleaving and spelling. The relation graph never reads a spelling or a source position."""
    import random

    def chains(seed: int, *, shuffle: bool) -> Body:
        rng = random.Random(seed)
        stmts: list = []
        for chain in range(4):
            prefix = f"c{chain}_" if shuffle else f"r{rng.randrange(1000)}_"
            axis = f"{prefix}i"
            stmts.append(
                Loop(
                    axis=Axis(axis, 4),
                    body=(
                        Load(name=f"{prefix}x", input=f"{prefix}X", index=(Var(axis),)),
                        Assign(name=f"{prefix}y", op=("abs", "exp", "negative")[chain % 3], args=(f"{prefix}x",)),
                        Write(output=f"{prefix}O", index=(Var(axis),), value=f"{prefix}y"),
                    ),
                )
            )
        if shuffle:
            rng.shuffle(stmts)
        return Body(stmts)

    keys = {chains(seed, shuffle=shuffle).structural_key() for seed in range(12) for shuffle in (False, True)}
    assert len(keys) == 1


def test_identity_ignores_a_buffer_spelling_that_reorders_the_executable_body() -> None:
    """The executable order breaks symmetric ties by buffer spelling; identity labels the same
    graph without it, so a rename that flips the executable order keys the same."""

    def make(left: str, right: str) -> Body:
        return Body(
            (
                Load(name="x", input=left, index=()),
                Load(name="y", input=right, index=()),
                Assign(name="z", op="add", args=("x", "y")),
                Write(output="O", index=(), value="z"),
            )
        )

    first, second = normalize_body(make("X", "Y")), normalize_body(make("z_input", "a_input"))
    assert [stmt.input for stmt in first[:2]] == ["X", "Y"]
    assert [stmt.input for stmt in second[:2]] == ["a_input", "z_input"]
    assert first.structural_key() == second.structural_key()
    assert first.structural_key(structural=False) == second.structural_key(structural=False)
