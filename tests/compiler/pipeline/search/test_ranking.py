"""Focused tests for program-backed golden enumeration."""

from dataclasses import dataclass, field

from emmy.compiler.context import Context
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.schedule import Placement
from emmy.compiler.ir.stmt import Select, SelectBranch
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline.fork import DeferredFork, Fork
from emmy.compiler.pipeline.pipeline import ForkPoint
from emmy.compiler.pipeline.search.ranking import arm_features, enumerate_graph
from tests.compiler.terms import contraction, projection, slab


@dataclass(frozen=True)
class _EmptyBranch(Fork):
    knobs: dict = field(default_factory=dict)
    is_leaf = False

    def expand(self):
        return []


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
    rows = [arm_features(None, TileOp(op=root, place=Placement(free=(m, n)), axes=(m, n, k)), Graph()) for root in roots]

    assert [row["P_n_whole_contraction_roots"] for row in rows] == [1.0, 0.0]
    assert {key: value for key, value in rows[0].items() if key != "P_n_whole_contraction_roots"} == {
        key: value for key, value in rows[1].items() if key != "P_n_whole_contraction_roots"
    }
