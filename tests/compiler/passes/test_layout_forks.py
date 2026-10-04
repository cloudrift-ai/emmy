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
from emmy.compiler.pipeline.search.pins import spelled_arm
from emmy.compiler.pipeline.search.policy.greedy import _EMPTY_MEASURED, _Measured, _layout_candidates, _route_candidates
from tests.compiler.terms import contraction


def _graph() -> Graph:
    n, k = Axis("n", 4), Axis("k", 8)
    tile = TileOp(
        op=contraction(
            k,
            Load(name="xv", input="x", index=(Var("k"),)),
            (Load(name="wv", input="w", index=(Var("k"), Var("n"))), "acc"),
        ),
        name="y",
        place=Placement(free=(n,)),
        axes=(n, k),
        output_specs=(OutputSpec(Write(output="y", index=(Var("n"),), value="acc")),),
    )
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (8,), "f32"), node_id="x")
    graph.add_node(
        ConstantOp(name="w", load_ops=(TransposeOp((-2, -1)),), source_path="weight", source_shape=(4, 8)),
        [],
        Tensor("w", (8, 4), "f32"),
        node_id="w",
    )
    graph.add_node(tile, ["x", "w"], Tensor("y", (4,), "f32"), node_id="y")
    graph.inputs, graph.outputs = ["x"], ["y"]
    return graph


def _lower(source: bool) -> Graph:
    graph = _graph()

    def decide(point):
        keys = {key for option in point.options for key in option.knobs}
        if any(key.startswith("LAYOUT@") for key in keys):
            want = "source" if source else "folded"
            option = next(option for option in point.options if option.knobs.get("LAYOUT@w") == want)
            assert spelled_arm(point.options, {"LAYOUT@w": want})[0] is option
            return option
        return next((option for option in point.options if not _is_structural_option(option)), point.options[0])

    graph, _ = Run(Pipeline.build(["tile/cut"]), Context.from_target((7, 0))).resolve(graph, decide)
    graph.validate()
    for node in graph.nodes.values():
        if isinstance(node.op, TileOp):
            tile = node.op
            node.op = LoopOp(body=tile.op.lower(bound=frozenset(), stores=tile.output_specs, axes=tile.axes))
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
        priced_arms=lambda _ctx, kernel, **_kw: [({"LAYOUT@w": "source"}, 17.0), ({"REDUCE": "g2k"}, 20.0)]
        if kernel == folded else [],
        best_per_op_time=lambda *_args, **_kwargs: None,
    )
    index = _Measured({source_key: [({"WORK": "t128"}, 17.0)]}, {})
    prices = {option.knobs["LAYOUT@w"]: us for option, us in _layout_candidates(point, index, db)}
    assert prices == {"folded": 20.0, "source": 17.0}
    assert _route_candidates(point, _EMPTY_MEASURED, db) == []
    fuse = DeferredFork(lambda: bound, {"PLACE": "fuse"})
    cut = DeferredFork(lambda: graph, {"PLACE": "cut"}, structural=True)
    place_point = ForkPoint(match=match, options=[fuse, cut], root_op=bound, ctx=point.ctx)
    assert _route_candidates(place_point, _EMPTY_MEASURED, db) == [(fuse, 20.0)]
