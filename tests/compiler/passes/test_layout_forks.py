"""A folded weight transpose and its source storage produce the same values."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from emmy.compiler.backend.numpy import NumpyBackend
from emmy.compiler.context import Context
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.frontend.ir import TransposeOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Load, Write
from emmy.compiler.ir.tile import OutputSpec, Placement, TileOp
from emmy.compiler.pipeline import Match, Pipeline, Rule
from emmy.compiler.pipeline.fork import DeferredFork
from emmy.compiler.pipeline.passes.tile._layout import layout_forks
from emmy.compiler.pipeline.pipeline import ForkPoint, Run, _is_structural_option
from emmy.compiler.pipeline.search.pins import pinned_knobs, spelled_arm, unreproducible_pin_flag
from emmy.compiler.pipeline.search.policy.greedy import _EMPTY_MEASURED, _layout_candidates, _Measured, _route_candidates
from emmy.compiler.wire import kernel_wire
from tests.compiler.terms import contraction


def _graph(*, grouped: bool = False) -> Graph:
    n, k = Axis("n", 4), Axis("k", 8)
    channels = [(Load(name="wv", input="w", index=(Var("k"), Var("n"))), "acc")]
    if grouped:
        channels.append((Load(name="w2v", input="w2", index=(Var("k"), Var("n"))), "acc2"))
    tile = TileOp(
        op=contraction(k, Load(name="xv", input="x", index=(Var("k"),)), *channels),
        name="y",
        place=Placement(free=(n,)),
        axes=(n, k),
        output_specs=(OutputSpec(Write(output="y", index=(Var("n"),), value="acc")),)
        + ((OutputSpec(Write(output="y2", index=(Var("n"),), value="acc2")),) if grouped else ()),
    )
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (8,), "f32"), node_id="x")
    weights = ("w", "w2") if grouped else ("w",)
    for name in weights:
        graph.add_node(
            ConstantOp(name=name, load_ops=(TransposeOp((-2, -1)),), source_path=name, source_shape=(4, 8)),
            [],
            Tensor(name, (8, 4), "f32"),
            node_id=name,
        )
    outputs = (Tensor("y", (4,), "f32"), Tensor("y2", (4,), "f32")) if grouped else (Tensor("y", (4,), "f32"),)
    graph.add_node(tile, ["x", *weights], outputs=outputs, node_id="y")
    graph.inputs, graph.outputs = ["x"], ["y", "y2"] if grouped else ["y"]
    return graph


def _lower(source: bool, *, grouped: bool = False, tile: bool = False) -> Graph:
    graph = _graph(grouped=grouped)

    def decide(point):
        keys = {key for option in point.options for key in option.knobs}
        if any(key.startswith("LAYOUT@") for key in keys):
            want = "source" if source else "folded"
            option = max(
                (option for option in point.options if all(value == want for value in option.knobs.values())),
                key=lambda item: len(item.knobs),
            )
            assert spelled_arm(point.options, option.knobs)[0] is option
            return option
        return next((option for option in point.options if not _is_structural_option(option)), point.options[0])

    graph, _ = Run(Pipeline.build(["tile/cut"]), Context.from_target((7, 0))).resolve(graph, decide)
    graph.validate()
    if not tile:
        for node in graph.nodes.values():
            if isinstance(node.op, TileOp):
                op = node.op
                node.op = LoopOp(body=op.op.lower(bound=frozenset(), stores=op.output_specs, axes=op.axes))
    return graph


def test_source_layout_matches_folded_layout() -> None:
    rng = np.random.default_rng(7)
    x = rng.standard_normal((8,)).astype(np.float32)
    weight = rng.standard_normal((4, 8)).astype(np.float32)

    def run(graph: Graph):
        constants = {
            name: weight.T if node.output.shape == (8, 4) else weight
            for name, node in graph.nodes.items()
            if isinstance(node.op, ConstantOp)
        }
        result, _ = NumpyBackend().run(graph, input_data={"x": x, **constants})
        return result.outputs["y"]

    folded, source = _lower(False), _lower(True)
    assert any(name.endswith("__source") for name, op in source.loadable_constants())
    assert len(list(source.loadable_constants())) == 1
    np.testing.assert_allclose(run(source), run(folded), rtol=1e-6, atol=1e-6)


def test_joint_source_layout_matches_folded_layout() -> None:
    rng = np.random.default_rng(9)
    x = rng.standard_normal((8,)).astype(np.float32)
    weights = {name: rng.standard_normal((4, 8)).astype(np.float32) for name in ("w", "w2")}

    def run(graph: Graph):
        constants = {
            name: weights[name.removesuffix("__source")] if name.endswith("__source") else weights[name].T
            for name, node in graph.nodes.items()
            if isinstance(node.op, ConstantOp)
        }
        result, _ = NumpyBackend().run(graph, input_data={"x": x, **constants})
        return result.outputs

    folded, source = _lower(False, grouped=True), _lower(True, grouped=True)
    assert {name for name, _ in source.loadable_constants()} == {"w__source", "w2__source"}
    actual = run(source)
    for name, want in run(folded).items():
        np.testing.assert_allclose(actual[name], want, rtol=1e-6, atol=1e-6)


def test_source_layout_golden_body_reads_source_storage() -> None:
    graph = _lower(True, tile=True)
    node = next(node for node in graph.nodes.values() if isinstance(node.op, TileOp))
    wire = Graph.from_wire(kernel_wire(node.op.with_io(graph, node)))
    reads = [load for node in wire.nodes.values() if isinstance(node.op, LoopOp) for load in node.op.body.loads]
    assert any(load.input == "w__source" for load in reads)
    assert all(load.input != "w" for load in reads)


def test_layout_prices_its_own_measured_kernel_and_not_its_child_route() -> None:
    graph = _graph()
    root = graph.nodes["y"]
    match = Match(graph=graph, root_node_id="y", rule=Rule(name="test", pattern=[]))
    options = layout_forks(match, root)
    assert options is not None
    bound = root.op.with_io(graph, root)
    point = ForkPoint(match=match, options=options, root_op=bound, ctx=Context.from_target((7, 0)))
    folded = bound.identity_key(structural=False, with_io=True)
    source = next(option for option in options if option.knobs.get("LAYOUT@w") == "source")
    source_graph = source.materialize()
    source_node = next(node for node in source_graph.nodes.values() if isinstance(node.op, TileOp))
    source_key = source_node.op.with_io(source_graph, source_node).identity_key(structural=False, with_io=True)
    assert folded != source_key

    db = SimpleNamespace(
        priced_arms=lambda _ctx, kernel, **_kw: [({"LAYOUT@w": "source"}, 17.0), ({"REDUCE": "g2k"}, 20.0)] if kernel == folded else [],
        best_per_op_time=lambda *_args, **_kwargs: None,
        has_perf=lambda *_args, **_kwargs: True,
    )
    index = _Measured({source_key: [({"WORK": "t128"}, 17.0)]}, {})
    prices = {option.knobs["LAYOUT@w"]: us for option, us in _layout_candidates(point, index, db)}
    assert prices == {"folded": 20.0, "source": 17.0}
    assert _route_candidates(point, _EMPTY_MEASURED, db) == []
    fuse = DeferredFork(lambda: bound, {"PLACE": "fuse"})
    cut = DeferredFork(lambda: graph, {"PLACE": "cut"}, structural=True)
    place_point = ForkPoint(match=match, options=[fuse, cut], root_op=bound, ctx=point.ctx)
    assert _route_candidates(place_point, _EMPTY_MEASURED, db) == [(fuse, 20.0)]


def test_layout_pin_selects_a_weight_storage_choice() -> None:
    graph = _graph()
    match = Match(graph=graph, root_node_id="y", rule=Rule(name="test", pattern=[]))
    with pinned_knobs({"LAYOUT@w": "source"}):
        assert [option.knobs for option in layout_forks(match, graph.nodes["y"])] == [{"LAYOUT@w": "source"}]
    with pinned_knobs({"LAYOUT@w": "folded"}):
        assert [option.knobs for option in layout_forks(match, graph.nodes["y"])] == [{"LAYOUT@w": "folded"}]
    grouped = _graph(grouped=True)
    group_match = Match(graph=grouped, root_node_id="y", rule=Rule(name="test", pattern=[]))
    with pinned_knobs({"LAYOUT@w": "source", "LAYOUT@w2": "source"}):
        assert [option.knobs for option in layout_forks(group_match, grouped.nodes["y"])] == [{"LAYOUT@w": "source", "LAYOUT@w2": "source"}]
    assert unreproducible_pin_flag({"LAYOUT@w": "source"}, [{}], placement_knobs=[{"LAYOUT@w": "source"}]) is None
    assert unreproducible_pin_flag({"LAYOUT@w": "source"}, [{}], placement_knobs=[{"LAYOUT@w": "folded"}])
