"""Focused tests for program-backed golden enumeration."""

from collections import Counter
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from emmy.compiler.context import Context
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.schedule import Placement
from emmy.compiler.ir.stmt import Select, SelectBranch
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline.fork import DeferredFork, Fork
from emmy.compiler.pipeline.pipeline import ForkPoint
from emmy.compiler.pipeline.search import features
from emmy.compiler.pipeline.search.features import Featurizer
from emmy.compiler.pipeline.search.ranking import _place_ballot, enumerate_graph
from tests.compiler.terms import contraction, projection, slab


@dataclass(frozen=True)
class _EmptyBranch(Fork):
    knobs: dict = field(default_factory=dict)
    is_leaf = False

    def expand(self):
        return []


def _ballot(monkeypatch, pieces: dict[int, tuple[tuple[str, str], ...]], rows: list[dict], aliases=None):
    """Fake leaves whose expansion is the ``(name, identity)`` pieces each arm leaves, read through ``kernel_pieces``."""
    monkeypatch.setattr(
        features,
        "kernel_pieces",
        lambda left: [(SimpleNamespace(name=name, identity_key=lambda key=key, **_: key), False) for name, key in left],
    )
    return [DeferredFork(lambda i=i: pieces.get(i, ()), knobs=row, aliases=aliases or {}) for i, row in enumerate(rows)]


@pytest.mark.parametrize("complete_offered", [False, True])
@pytest.mark.parametrize("alias", ["PLACE@left", "PLACE@left_alias"])
def test_placement_labels_prefer_the_complete_natural_arm(monkeypatch, complete_offered, alias):
    """A golden that left ``left``, ``right`` and the remainder ``rest``: each single seam carving one of them is
    positive, and a natural arm leaving exactly that kernel set is positive instead of them. An alias spells the
    same seam, and the label reads the seam."""
    rows = [{"PLACE": "fuse"}, {alias: "cut"}, {"PLACE@right": "cut"}]
    pieces = {1: (("k__place_l", "left"), ("k", "k_left")), 2: (("k__place_r", "right"), ("k", "k_right"))}
    if complete_offered:
        rows.append({"PLACE@left": "cut", "PLACE@right": "cut"})
        pieces[3] = (("k__place_l", "left"), ("k__place_r", "right"), ("k", "rest"))
    leaves = _ballot(monkeypatch, pieces, rows, aliases={"PLACE@left_alias": "PLACE@left"})
    pending = Counter({"left": 1, "right": 1, "rest": 1})

    positives, labels, carved = _place_ballot(leaves, rows, "k", "k", pending, frozenset())

    assert labels == ["fuse", "left", "right", *(["left right"] if complete_offered else [])]
    assert positives == ([3] if complete_offered else [1, 2])
    assert carved[1] == Counter({"left": 1}) and carved[0] is None


def test_placement_labels_require_pending_pieces_with_multiplicity(monkeypatch):
    """An arm is positive only when every piece it carves is still pending, as many times as it carves it — or is
    a kernel the golden cut in its turn, whose own fork carries the chain on. The remainder is the kernel that
    keeps the fork kernel's name, and a kernel that is itself a pending piece stays whole."""
    rows = [{"PLACE": "fuse"}, {"PLACE@a": "cut"}, {"PLACE@c": "cut"}, {"PLACE@a": "cut", "PLACE@a2": "cut"}, {"PLACE@p": "cut"}]
    pieces = {
        1: (("k__place_a", "a"), ("k", "k_a")),
        2: (("k__place_c", "c"), ("k", "k_c")),
        3: (("k__place_a", "a"), ("k__place_a2", "a"), ("k", "k_aa")),
        4: (("k__place_p", "parent"), ("k", "k_p")),
    }
    leaves = _ballot(monkeypatch, pieces, rows)
    pending = Counter({"a": 1, "b": 1, "rest": 1})

    positives, _labels, _carved = _place_ballot(leaves, rows, "k", "k", pending, frozenset({"parent"}))
    assert positives == [1, 4]
    assert _place_ballot(leaves, rows, "k", "k", Counter({"a": 2, "rest": 1}), frozenset())[0] == [1, 3]
    assert _place_ballot(leaves, rows, "k", "k", Counter({"c": 1}), frozenset())[0] == [2]
    assert _place_ballot(leaves, rows, "rest", "k", pending, frozenset())[0] == [0], "a pending kernel stays whole"
    assert _place_ballot(leaves, rows, "k", "k", None, frozenset())[0] == [0], "no cut recorded: fuse"
    assert _place_ballot(leaves, rows, "k", "k", Counter({"z": 1}), frozenset())[0] == [], "the recorded set is out of reach"


def test_enumeration_skips_an_empty_pinned_branch(monkeypatch) -> None:
    row = {"WORK": "w1x1", "TILE": "mma"}
    live = DeferredFork(materialize=lambda: None, knobs=row)

    def resolve(_self, graph, decide):
        assert decide(ForkPoint(match=None, options=[_EmptyBranch(), live], root_op=None, ctx=None)) is live
        return graph, []

    monkeypatch.setattr("emmy.compiler.pipeline.pipeline.Run.resolve", resolve)

    candidates = enumerate_graph(Graph(), Context.from_target((8, 0)))

    assert candidates.rows == [row]


def test_placement_separates_contraction_roots_from_equal_loop_histograms() -> None:
    m, n, k = Axis("m", 4), Axis("n", 8), Axis("k", 16)
    b = slab("b", "w", "k", "n")
    product = contraction(k, slab("a", "x", "m", "k"), (b, "acc0"), (b, "acc1"))
    selected = Select(
        name="masked",
        branches=(SelectBranch("acc0", BinaryExpr("<", Var("n"), Literal(4, "int"))), SelectBranch("acc1", Literal(1, "int"))),
    )
    roots = (product, projection((product,), (selected,), results=("masked", "acc1")))
    tiles = [TileOp(op=root, place=Placement(free=(m, n)), axes=(m, n, k)) for root in roots]
    rows = [Featurizer({}).features(tile, pieces=tile) for tile in tiles]

    assert [row["P_n_whole_contraction_roots"] for row in rows] == [1.0, 0.0]
    assert {key: value for key, value in rows[0].items() if key != "P_n_whole_contraction_roots"} == {
        key: value for key, value in rows[1].items() if key != "P_n_whole_contraction_roots"
    }
