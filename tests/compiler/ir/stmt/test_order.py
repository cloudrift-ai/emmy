"""Exact graph labeling and statement-order normalization tests."""

from __future__ import annotations

from itertools import islice, permutations, product

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import CpAsyncCommit, CpAsyncWait, Sync, WgmmaCommit, WgmmaFence, WgmmaWait
from emmy.compiler.ir.stmt.blocks import Cond, Loop
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.identity import canonicalize_identity
from emmy.compiler.ir.stmt.leaves import Assign, Const, Load, Write
from emmy.compiler.ir.stmt.normalize import normalize_body
from emmy.compiler.ir.stmt.order import _canonical_ranks, _equitable_partition, ordering_constraints


def _certificate(colors, edges, ranks: tuple[int, ...]) -> tuple:
    order = tuple(sorted(range(len(colors)), key=ranks.__getitem__))
    return (
        tuple(repr(colors[vertex]) for vertex in order),
        tuple(sorted((ranks[source], ranks[target], repr(color)) for source, target, color in edges)),
    )


def _permuted(colors, edges, order):
    positions = {old: new for new, old in enumerate(order)}
    return tuple(colors[old] for old in order), tuple((positions[source], positions[target], color) for source, target, color in edges)


def test_canonical_label_is_invariant_for_every_vertex_permutation() -> None:
    graphs = (
        (
            ("same", "same", "sink"),
            ((0, 0, "loop"), (0, 2, "edge"), (0, 2, "edge"), (1, 2, "edge")),
        ),
        (
            ("twin", "twin", "middle", "sink"),
            ((0, 2, "in"), (1, 2, "in"), (2, 3, "out")),
        ),
        (
            ("node",) * 6,
            ((0, 1, "a"), (1, 2, "b"), (3, 4, "a"), (4, 5, "b"), (2, 2, "loop")),
        ),
    )
    for colors, edges in graphs:
        exhaustive_certificates = set()
        for order in permutations(range(len(colors))):
            permuted_colors, permuted_edges = _permuted(colors, edges, order)
            exhaustive = _canonical_ranks(permuted_colors, permuted_edges, _prune=False)
            exhaustive_certificates.add(_certificate(permuted_colors, permuted_edges, exhaustive))
        assert len(exhaustive_certificates) == 1

        expected = exhaustive_certificates.pop()
        for order in islice(permutations(range(len(colors))), 24):
            permuted_colors, permuted_edges = _permuted(colors, edges, order)
            pruned = _canonical_ranks(permuted_colors, permuted_edges)
            assert _certificate(permuted_colors, permuted_edges, pruned) == expected


def test_worklist_refinement_visits_relations_logarithmically() -> None:
    visits = [0]

    class CountedEdges:
        def __init__(self, values) -> None:
            self.values = values

        def __iter__(self):
            for value in self.values:
                visits[0] += 1
                yield value

    count = 1024
    incoming = [[] for _ in range(count)]
    outgoing = [[] for _ in range(count)]
    for source in range(count - 1):
        outgoing[source].append((0, source + 1))
        incoming[source + 1].append((0, source))
    refined = _equitable_partition(
        (tuple(range(count)),),
        [CountedEdges(values) for values in incoming],
        [CountedEdges(values) for values in outgoing],
    )
    assert all(len(cell) == 1 for cell in refined)
    assert visits[0] <= 4 * count * count.bit_length()


def test_canonical_label_splits_regular_asymmetric_graph() -> None:
    """The Frucht graph stays canonical even though color refinement leaves all vertices tied."""
    undirected = (
        (0, 1),
        (0, 6),
        (0, 7),
        (1, 2),
        (1, 7),
        (2, 3),
        (2, 8),
        (3, 4),
        (3, 9),
        (4, 5),
        (4, 9),
        (5, 6),
        (5, 10),
        (6, 10),
        (7, 11),
        (8, 9),
        (8, 11),
        (10, 11),
    )
    colors = ("vertex",) * 12
    edges = tuple((source, target, "edge") for left, right in undirected for source, target in ((left, right), (right, left)))
    ranks = _canonical_ranks(colors, edges)
    for order in (tuple(reversed(range(12))), (3, 8, 1, 10, 5, 0, 11, 4, 7, 2, 9, 6)):
        permuted_colors, permuted_edges = _permuted(colors, edges, order)
        permuted_ranks = _canonical_ranks(permuted_colors, permuted_edges)
        positions = {old: new for new, old in enumerate(order)}
        assert tuple(permuted_ranks[positions[old]] for old in range(12)) == ranks


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


def test_normalize_order_matches_all_small_dependency_valid_forms() -> None:
    statements = (
        Load(name="x", input="X", index=()),
        Assign(name="absolute", op="abs", args=("x",)),
        Load(name="y", input="Y", index=()),
        Assign(name="result", op="add", args=("absolute", "y")),
        Write(output="O", index=(), value="result"),
    )
    normalized = {normalize_body(Body(order)) for order in permutations(statements)}
    assert len(normalized) == 1


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
        first = Const(name=name, value=1.0)
        capture = Cond(cond=Literal(True), body=(Assign(name=inner, op="abs", args=(name,)),))
        second = Const(name=name, value=2.0)
        tail = Assign(name=result, op="exp", args=(name,))
        middle = (second, capture) if late_before_capture else (capture, second)
        return Body((first, *middle, tail))

    before = normalize_body(make("value", "inside", "result", late_before_capture=False))
    renamed = normalize_body(make("renamed", "renamed_inside", "renamed_result", late_before_capture=False))
    after = normalize_body(make("value", "inside", "result", late_before_capture=True))
    assert before == renamed
    assert before != after


def test_topological_sort_refuses_a_captured_rebind() -> None:
    """A scope that reads an enclosing binding and later rebinds the same spelling is refused:
    sorting the rebind above the read would silently retarget it."""
    import pytest

    body = Body(
        (
            Const(name="x", value=1.0),
            Cond(
                cond=Literal(True),
                body=(
                    Assign(name="y", op="abs", args=("x",)),
                    Const(name="x", value=2.0),
                    Assign(name="z", op="exp", args=("x",)),
                    Write(output="O", index=(), value="y"),
                    Write(output="P", index=(), value="z"),
                ),
            ),
        )
    )
    with pytest.raises(ValueError, match="rebinds"):
        normalize_body(body)


def test_shadowing_after_the_read_keeps_the_outer_dependency() -> None:
    """A deeper scope's rebind of a name binds nothing read above it: the block still depends on
    the enclosing definition, so it cannot sort above it."""
    body = Body(
        (
            Cond(
                cond=Literal(True),
                body=(
                    Assign(name="y", op="abs", args=("x",)),
                    Cond(cond=Literal(True), body=(Const(name="x", value=2.0), Write(output="P", index=(), value="x"))),
                    Write(output="O", index=(), value="y"),
                ),
            ),
            Const(name="x", value=1.0),
        )
    )
    normalized = normalize_body(body)
    assert isinstance(normalized[0], Const)
    assert isinstance(normalized[1], Cond)


def test_relation_graph_binds_every_name_of_a_closed_body() -> None:
    """Every name a closed body spells is a binder the graph owns. A name it could not bind
    keeps its spelling as a vertex color, which would make identity spelling-dependent."""
    from emmy.compiler.ir.stmt.leaves import Accum
    from emmy.compiler.ir.stmt.order import relation_graph

    body = Body(
        (
            Loop(
                axis=Axis("i", 4),
                body=(
                    Load(name="x", input="X", index=(Var("i"),)),
                    Loop(
                        axis=Axis("k", 8),
                        body=(
                            Load(name="w", input="W", index=(Var("i"), Var("k"))),
                            Assign(name="p", op="multiply", args=("x", "w")),
                            Accum(name="acc", op="add", value="p", axes=("k",)),
                        ),
                    ),
                    Cond(cond=BinaryExpr("<", Var("i"), Literal(2, "int")), body=(Write(output="O", index=(Var("i"),), value="acc"),)),
                ),
            ),
        )
    )
    assert relation_graph(normalize_body(body)).fixed_names == ()


def test_identity_shares_normalizations_graph() -> None:
    body = Body((Load(name="x", input="X", index=()), Assign(name="y", op="abs", args=("x",)), Write(output="O", index=(), value="y")))
    normalized = normalize_body(body)
    assert "_ordering" in normalized.__dict__
    assert canonicalize_identity(normalized).body == canonicalize_identity(body).body


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
