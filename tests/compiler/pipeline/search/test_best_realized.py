"""``TuningSearch.best_realized`` — the one row a finished search reports as its winner.

A terminal is a Σ over the kernels it lowered to. When it lowered to ONE, that kernel's row is the
winner's row; when a structural fork made it several, the kernels carry different decisions and no
single schedule row describes them — the search then reports the structural route that produced them
(its routing row), or nothing, never a merge of the pieces' rows that no kernel realized.
"""

from __future__ import annotations

from types import SimpleNamespace

from emmy.compiler.pipeline.knob import complete_kernel_row
from emmy.compiler.pipeline.search.db import PerfStats
from emmy.compiler.pipeline.search.policy.mcts import SearchNode, SearchTree, TuningSearch


def _classic(*, work: str, tile: str, reduce: str = "", stage: str = "", raster: str = "") -> dict[str, str]:
    return {
        "WORK": work,
        "RASTER": raster,
        "TILE": tile,
        "REDUCE": reduce,
        "STAGE": stage,
    }


def _ok_leaf(us: float, *, realized_knobs: dict | None, cuda_ops: int, cuda_knobs: list[dict] | None = None) -> SearchNode:
    node = SearchNode(candidate=object())
    node.realized_knobs = realized_knobs
    node.realized_cuda_ops = cuda_ops
    node.realized_cuda_knobs = cuda_knobs
    node.bench_status = "ok"
    node.bench_stats = PerfStats(median=us, min=us, max=us, mean=us, variance=0.0, n_samples=1)
    return node


def test_best_realized_does_not_fall_back_from_a_faster_unrepresentable_terminal() -> None:
    tree = SearchTree()
    tree.root.children = [
        _ok_leaf(18.0, realized_knobs={"WORK": "t64"}, cuda_ops=1),
        _ok_leaf(6.0, realized_knobs=None, cuda_ops=2),
    ]

    search = TuningSearch.__new__(TuningSearch)
    search.tree = tree

    assert search.best_realized() is None


def test_a_validated_structural_input_names_the_multi_kernel_winner() -> None:
    route = _classic(work="w2x1", tile="mma_m16n8k16_f16_f32/f4x8/k8", reduce="g4k", stage="d1/smem-async")
    tree = SearchTree()
    leaf = _ok_leaf(
        59.61,
        realized_knobs=None,
        cuda_ops=2,
        cuda_knobs=[
            {**route, "REDUCE": ""},
            _classic(work="", tile=""),
        ],
    )
    leaf.visits = 1
    leaf.best_reward = 1.0 / 59.61
    tree.root.children = [leaf]
    tree.root.visits = 1
    tree.root.best_reward = leaf.best_reward

    search = TuningSearch.__new__(TuningSearch)
    search.tree = tree
    search._base_knobs = {"H_opt": 3.0, "S_loop": 1.0}

    assert search.best_realized() is None
    assert search.best_realized(validated_input_route=route) == (complete_kernel_row(route), 59.61, 2, True)


def test_best_realized_returns_the_fastest_terminal_with_its_structural_replay_row() -> None:
    tree = SearchTree()
    row = _classic(work="w1x1", tile="mma_m16n8k16_f16_f32/f1x4/k8", reduce="g8k", stage="d1/smem")
    route = SearchNode(candidate=SimpleNamespace(resolved_knobs=row), parent=tree.root)
    fast = _ok_leaf(
        6.0,
        realized_knobs=None,
        cuda_ops=2,
        cuda_knobs=[
            {**row, "REDUCE": ""},
            _classic(work="", tile=""),
        ],
    )
    fast.parent = route
    route.children = [fast]
    tree.root.children = [_ok_leaf(18.0, realized_knobs={"WORK": "t64"}, cuda_ops=1), route]

    search = TuningSearch.__new__(TuningSearch)
    search.tree = tree

    assert search.best_realized() == (complete_kernel_row(row), 6.0, 2, True)


def test_best_realized_rejects_a_structural_parent_that_names_a_different_child_schedule() -> None:
    tree = SearchTree()
    row = _classic(work="w4x2", tile="mma_m16n8k16_f16_f32/f1x2/k8", reduce="g8k", stage="d1/smem")
    route = SearchNode(candidate=SimpleNamespace(resolved_knobs=row), parent=tree.root)
    fast = _ok_leaf(
        6.0,
        realized_knobs=None,
        cuda_ops=2,
        cuda_knobs=[
            {**row, "WORK": "w1x2", "REDUCE": ""},
            _classic(work="", tile=""),
        ],
    )
    fast.parent = route
    route.children = [fast]
    tree.root.children = [_ok_leaf(18.0, realized_knobs={"WORK": "t64"}, cuda_ops=1), route]

    search = TuningSearch.__new__(TuningSearch)
    search.tree = tree

    assert search.best_realized() is None


def test_best_realized_uses_a_compatible_multi_cuda_placement_route() -> None:
    tree = SearchTree()
    tree.root.children = [
        _ok_leaf(
            6.0,
            realized_knobs={"WORK": "w1x1", "TILE": "mma_m16n8k16_f16_f32/f1x4/k8", "PLACE@inner.1/map": "cut"},
            cuda_ops=2,
            cuda_knobs=[{"WORK": "w1x1"}, {"WORK": ""}],
        )
    ]

    search = TuningSearch.__new__(TuningSearch)
    search.tree = tree

    assert search.best_realized() == ({"PLACE@inner.1/map": "cut"}, 6.0, 2, True)


def test_best_realized_keeps_only_the_routing_row_for_a_placement_cut() -> None:
    tree = SearchTree()
    route = SearchNode(candidate=SimpleNamespace(resolved_knobs={"PLACE@inner.1/map": "cut", "WORK": "w1x1"}), parent=tree.root)
    fast = _ok_leaf(6.0, realized_knobs=None, cuda_ops=2, cuda_knobs=[{"WORK": "w1x1"}, {"WORK": ""}])
    fast.parent = route
    route.children = [fast]
    tree.root.children = [route]

    search = TuningSearch.__new__(TuningSearch)
    search.tree = tree

    assert search.best_realized() == ({"PLACE@inner.1/map": "cut"}, 6.0, 2, True)


def test_best_realized_keeps_the_schedule_row_for_fused_placement() -> None:
    tree = SearchTree()
    row = complete_kernel_row(_classic(work="w1x4", tile="mma_m16n8k16_f16_f32/f4x2/k8", stage="d1/smem"))
    route = SearchNode(candidate=SimpleNamespace(resolved_knobs={"PLACE": "fuse"}), parent=tree.root)
    fast = _ok_leaf(6.0, realized_knobs=row, cuda_ops=1, cuda_knobs=[row])
    fast.parent = route
    route.children = [fast]
    tree.root.children = [route]

    search = TuningSearch.__new__(TuningSearch)
    search.tree = tree

    assert search.best_realized() == (row, 6.0, 1, False)


def test_best_realized_returns_an_ordinary_one_kernel_row() -> None:
    tree = SearchTree()
    tree.root.children = [_ok_leaf(6.0, realized_knobs={"WORK": "t64"}, cuda_ops=1)]

    search = TuningSearch.__new__(TuningSearch)
    search.tree = tree

    assert search.best_realized() == ({"WORK": "t64"}, 6.0, 1, False)
