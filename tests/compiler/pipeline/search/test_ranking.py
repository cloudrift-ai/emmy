"""Focused tests for program-backed golden enumeration."""

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


@pytest.mark.parametrize("complete_offered", [False, True])
@pytest.mark.parametrize("alias", ["PLACE@left", "PLACE@left_alias"])
def test_placement_labels_prefer_the_complete_natural_arm(complete_offered, alias):
    route = {alias: "cut", "PLACE@right": "cut"}
    rows = [{"PLACE": "fuse"}, {"PLACE@left": "cut"}, {"PLACE@right": "cut"}]
    if complete_offered:
        rows.append(dict(route))
    rows.append(dict(route))  # The registered route steers the walk but is not a training candidate.
    leaves = [DeferredFork(TileOp, knobs=row, aliases={"PLACE@left_alias": "PLACE@left"}) for row in rows]

    arms, positives, followed, labels = _place_ballot(leaves, rows, route)

    assert arms == list(range(len(rows) - 1)) and followed == len(rows) - 1
    assert [labels[i] for i in positives] == (["left right"] if complete_offered else ["left", "right"])


@pytest.mark.parametrize(
    "candidate,recorded,equivalent",
    [
        (("b", "a"), ("a", "b"), True),
        (("a", "c"), ("a", "b"), False),
        (("a", "a", "b"), ("a", "b"), False),
        ((None,), (None,), False),
    ],
)
def test_placement_labels_require_every_exact_piece_with_multiplicity(monkeypatch, candidate, recorded, equivalent):
    monkeypatch.setattr(
        features,
        "kernel_pieces",
        lambda keys: [(SimpleNamespace(identity_key=lambda key=key, **_: key), False) for key in keys],
    )
    route = {"PLACE@left": "cut", "PLACE@right": "cut"}
    rows = [{"PLACE": "fuse"}, {"PLACE@left": "cut"}, {"PLACE@right": "cut"}, {**route, "PLACE@extra": "cut"}, route]
    pieces = [("fused",), ("left",), ("right",), candidate, recorded]
    leaves = [DeferredFork(lambda keys=keys: keys, knobs=row) for row, keys in zip(rows, pieces, strict=True)]

    arms, positives, followed, _ = _place_ballot(leaves, rows, route)

    assert arms == [0, 1, 2, 3] and followed == 4
    assert positives == ([3] if equivalent else [1, 2])


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
