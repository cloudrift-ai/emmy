"""Search stopping: failed lowering must be bounded and cache replay must leave room for new evidence."""

from types import SimpleNamespace

import pytest

from emmy.compiler.context import Context
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline.fork import OptionFork
from emmy.compiler.pipeline.pipeline import Pass, Pattern, Pipeline, Rule, RuleSkipped
from emmy.compiler.pipeline.search.policy.mcts import TuningSearch
from emmy.compiler.pipeline.search.policy.terminal_bench import point_stats
from tests.compiler.helpers import drain_tune


def test_patience_bounds_consecutive_lowering_failures() -> None:
    patience = 3
    attempted = []

    def offer(root):
        if root.op.knobs:
            raise RuleSkipped("already selected")
        return [OptionFork(option=TileOp(name="test", knobs={"WORK": f"t{n}"}), knobs={"WORK": f"t{n}"}) for n in range(1, 9)]

    def lower(root):
        attempted.append(root.op.knobs["WORK"])
        raise ValueError("synthetic unsupported lowering")

    rules = [
        Rule(name="offer", pattern=[Pattern(name="root", op_type=TileOp)], rewrite=offer, param_names=("root",)),
        Rule(name="lower", pattern=[Pattern(name="root", op_type=TileOp)], rewrite=lower, param_names=("root",)),
    ]
    passes = [Pass(name=rule.name, rules=[rule], index=i) for i, rule in enumerate(rules)]
    for rule, pass_ in zip(rules, passes, strict=True):
        rule.pass_ = pass_
    graph = Graph()
    graph.add_node(TileOp(name="test"), [], Tensor("out", (1,), "f32"), node_id="out")
    graph.outputs = ["out"]
    search = TuningSearch(patience=patience, max_measurements=1)

    terminals = drain_tune(Pipeline(passes=passes), graph, search=search, ctx=Context.from_target((8, 0)))

    assert terminals == []
    assert search.measurements == 0
    assert 0 < len(attempted) <= patience, f"lowering attempted {len(attempted)} candidates despite patience={patience}"


def test_cached_replay_does_not_exhaust_live_measurement_patience() -> None:
    search = TuningSearch(patience=3, max_measurements=1)
    # Equal-prior sibling leaves are visited in order. Four cached rows precede one new, faster row.
    search.push(*(SimpleNamespace(fork=None, resolved_knobs={"WORK": f"t{n}"}) for n in range(1, 6)))
    observed = []
    while (popped := search.pop()) is not None:
        token, candidate = popped
        measured = candidate.resolved_knobs["WORK"] == "t5"
        search.observe(token, point_stats(5.0 if measured else 10.0), "ok", measured=measured)
        observed.append(candidate.resolved_knobs["WORK"])

    assert search.measurements == 1, f"stopped after {observed} with no new measurement: {search.stop_reason}"
    assert search.tree.best_reward == pytest.approx(1 / 5.0)


@pytest.mark.parametrize(
    "events,limits,expected,reason",
    [
        (["live"] * 8, {}, 4, "patience"),
        (["reject"] * 8, {}, 3, "patience"),
        (["fail"] * 8, {}, 3, "patience"),
        (["cached"] * 8, {}, 8, None),
        (["live", "reject", "cached", "reject", "cached", "reject", "live"], {}, 6, "patience"),
        (["live", "reject", "reject", "better", "reject", "reject", "live", "live"], {}, 7, "patience"),
        (["cached"] * 8, {"max_visits": 2}, 2, "max_visits"),
        (["reject"] * 8, {"max_visits": 2}, 2, "max_visits"),
        (["cached", "live", "cached", "live", "live"], {"max_measurements": 2}, 4, "max_measurements"),
    ],
)
def test_stopping_accounts_for_evaluation_origin(events, limits, expected, reason) -> None:
    search = TuningSearch(patience=3, **limits)
    search.push(*(SimpleNamespace(fork=None, resolved_knobs={}) for _ in events))
    for event in events[:expected]:
        token, _ = search.pop()
        if event == "reject":
            search.reject(token)
            assert token.bench_stats is None
        else:
            search.observe(
                token,
                point_stats(5.0 if event == "better" else 10.0),
                "bench_fail" if event == "fail" else "ok",
                measured=event not in {"cached", "better"},
            )
    assert search.pop() is None
    assert search.tree.root.visits == expected
    assert search.measurements == sum(event in {"live", "fail"} for event in events[:expected])
    assert (search.stop_reason.split(" ")[0] if search.stop_reason else None) == reason
