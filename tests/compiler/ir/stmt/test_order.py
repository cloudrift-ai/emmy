"""Exact graph labeling and statement-order normalization tests."""

from __future__ import annotations

from itertools import product

from emmy.compiler.ir.kernel.ir import CpAsyncCommit, CpAsyncWait, Sync, WgmmaCommit, WgmmaFence, WgmmaWait
from emmy.compiler.ir.stmt import Body, canonicalize_identity, normalize_body, ordering_constraints, parse_body
from tests.compiler.ir.stmt import listing


def test_topological_order_keeps_an_order_the_constraints_allow() -> None:
    body = parse_body(
        """
        right = exp(x)
        left = abs(x)
        """
    )
    incoming = (frozenset(), frozenset())
    assert body.topological_order(incoming) == body


def test_effect_constraints_keep_only_intervening_resource_hazards() -> None:
    writes = parse_body("\n".join(f"X[] = value_{index}" for index in range(100)))
    incoming = ordering_constraints(writes, effects=True)
    assert incoming[0] == set()
    assert incoming[1:] == [{index - 1} for index in range(1, len(writes))]

    body = parse_body(
        """
        X[] = first
        left = load X[]
        right = load X[]
        X[] = second
        tail = load X[]
        """
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
        chains = [
            f"""
    if {axis} < 4:
        {name} = load {buffer}[{axis}]
        {name}_value = abs({name})
        {buffer}_output[{axis}] = {name}_value"""
            for name, buffer in zip(names, buffers, strict=True)
        ]
        return parse_body(f"for {axis} in 0..4:" + "".join(reversed(chains) if reverse else chains))

    keys = {
        canonicalize_identity(normalize_body(make(reverse=reverse, renamed=renamed))).key
        for reverse, renamed in product((False, True), repeat=2)
    }
    assert len(keys) == 1


def test_repeated_definition_capture_uses_nearest_predecessor() -> None:
    def make(name: str, inner: str, result: str, *, late_before_capture: bool) -> Body:
        capture = f"""
            if true:
                {inner} = abs({name})
                P[] = {inner}"""
        second = f"""
            {name} = load B[]"""
        return parse_body(
            f"""
            {name} = load A[]{second + capture if late_before_capture else capture + second}
            {result} = exp({name})
            O[] = {result}
            """
        )

    before = normalize_body(make("value", "inside", "result", late_before_capture=False))
    renamed = normalize_body(make("renamed", "renamed_inside", "renamed_result", late_before_capture=False))
    after = normalize_body(make("value", "inside", "result", late_before_capture=True))
    assert str(before) == str(renamed)
    assert str(before) != str(after)


def test_shadowing_after_the_read_keeps_the_outer_dependency() -> None:
    """A deeper scope's rebind of a name binds nothing read above it: the block still depends on
    the enclosing definition, so it cannot sort above it."""
    body = parse_body(
        """
        if true:
            y = abs(x)
            if true:
                let x = 2.0
                P[] = x
            O[] = y
        x = load X[]
        """
    )
    first, second, *_ = str(normalize_body(body)).splitlines()
    assert first.endswith(" = load X[]")
    assert second.startswith("if ")


def test_structural_key_is_invariant_under_random_renaming_and_reordering() -> None:
    """Seeded property check: independent load/compute/write chains keyed under every
    interleaving and spelling. The relation graph never reads a spelling or a source position."""
    import random

    def chains(seed: int, *, shuffle: bool) -> Body:
        rng = random.Random(seed)
        loops: list[str] = []
        for chain in range(4):
            prefix = f"c{chain}_" if shuffle else f"r{rng.randrange(1000)}_"
            loops.append(
                listing(
                    f"""
                    for {prefix}i in 0..4:
                        {prefix}x = load {prefix}X[{prefix}i]
                        {prefix}y = {("abs", "exp", "negative")[chain % 3]}({prefix}x)
                        {prefix}O[{prefix}i] = {prefix}y
                    """
                )
            )
        if shuffle:
            rng.shuffle(loops)
        return parse_body("\n".join(loops))

    keys = {chains(seed, shuffle=shuffle).structural_key() for seed in range(12) for shuffle in (False, True)}
    assert len(keys) == 1


def test_identity_ignores_a_buffer_spelling_that_reorders_the_executable_body() -> None:
    """The executable order breaks symmetric ties by buffer spelling; identity labels the same
    graph without it, so a rename that flips the executable order keys the same."""

    def make(left: str, right: str) -> Body:
        return parse_body(
            f"""
            x = load {left}[]
            y = load {right}[]
            z = add(x, y)
            O[] = z
            """
        )

    def left_leads(body: Body, left: str) -> bool:
        """Whether the normal form reads the left operand's buffer first."""
        return str(body).splitlines()[0].endswith(f" = load {left}[]")

    first, second = normalize_body(make("X", "Y")), normalize_body(make("z_input", "a_input"))
    # The rename flips which operand's load comes first, whichever way the order breaks the tie.
    assert left_leads(first, "X") != left_leads(second, "z_input")
    assert first.structural_key() == second.structural_key()
    assert first.structural_key(structural=False) == second.structural_key(structural=False)
