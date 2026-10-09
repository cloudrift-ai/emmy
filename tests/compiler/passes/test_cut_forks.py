"""Kernel-placement forks over closed stored Fold edges."""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest

from emmy.compiler.backend.cuda.backend import CudaBackend
from emmy.compiler.context import Context
from emmy.compiler.dtype import F16
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis, Window
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.cuda.ir import CudaOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.frontend.ir import SdpaOp, SoftmaxOp
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.stmt import Assign, Load, Write
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.ir.tile import OutputSpec, Placement, TileOp
from emmy.compiler.ir.tile.path import resolve
from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, TILE_PASSES, Match, Pipeline, Rule
from emmy.compiler.pipeline.fork import Fork
from emmy.compiler.pipeline.passes.tile._cut import (
    CutSite,
    _fuse_sibling_producers,
    _producer_order,
    _workspace_axes,
    cuttable_seams,
    output_map,
    realize,
)
from emmy.compiler.pipeline.pipeline import RuleSkipped, Run, _is_structural_option, _structural_domain
from emmy.compiler.pipeline.search.db import RoutingRow, SearchDB
from emmy.compiler.pipeline.search.golden import Measurements, Row, evidence_scope, import_rows, mint, restamp
from emmy.compiler.pipeline.search.pins import pinned_knobs, spelled_arm, tracking_place_keys, unmatched_place_pins
from tests.compiler.helpers import case_target_tile, direct_classic_leaf, inventory_document, requires_cuda
from tests.compiler.terms import contraction, projection, reduction, slab

_CTX = Context.from_target((12, 0))
_CUT = import_module("emmy.compiler.pipeline.passes.tile.cut.030_cut")


def _input(graph: Graph, name: str, shape, dtype="f16") -> None:
    graph.add_node(InputOp(), [], Tensor(name, shape, dtype), node_id=name)


def test_cut_and_schedule_passes_share_the_generic_schedule_driver() -> None:
    from emmy.compiler.ir.schedule import schedule

    assert _CUT.schedule is schedule
    assert import_module("emmy.compiler.pipeline.fork").schedule is schedule


@pytest.mark.parametrize("siblings", [False, True])
def test_placement_cut_preserves_a_cross_cta_split_receipt(siblings: bool) -> None:
    """A split piece can re-enter placement; cutting it must not make REDUCE pending again."""
    from emmy.compiler.pipeline.passes.tile._split import split_pending

    if siblings:
        graph, root = _mimo_case(_REQUANT)
    else:
        graph = _computed_operand_graph("a")
        root = graph.nodes["out"]
    tile = root.op
    # The partition receipt is the reduce axis's window in the kernel's axis table; the term names it only.
    axes = tuple(replace(axis, window=Window(parent=axis, partition=True)) for axis in tile.axes)
    root.op = replace(tile, axes=axes)
    pipeline = Pipeline.build(["tile/cut"])
    match = pipeline.match(graph, pipeline.passes[0].rules[0])[0]
    seams = cuttable_seams(match.root.op)

    fragment = _composed_arm(graph, root)[0].materialize() if siblings else realize(match, match.root, (seams[0],))

    pieces = [node.op for node in fragment.nodes.values() if isinstance(node.op, TileOp)]
    if siblings:
        assert any(len(piece.output_specs) > 1 for piece in pieces)
    assert pieces and all(piece.split_consumed for piece in pieces)
    assert not any(split_pending(piece) for piece in pieces)


def _computed_operand_graph(side: str) -> Graph:
    m, n, k = Axis("m", 8), Axis("n", 8), Axis("k", 16)
    computed = projection(
        (),
        (
            Load(name="raw", input="computed", index=(Var("m" if side == "a" else "n"), Var("k"))),
            Assign(name="scaled", op="multiply", args=("raw", "raw")),
        ),
    )
    direct = Load(
        name="direct",
        input="direct",
        index=(Var("k"), Var("n")) if side == "a" else (Var("m"), Var("k")),
    )
    a, b = (computed, direct) if side == "a" else (direct, computed)
    tile = TileOp(op=contraction(k, a, (b, "acc")), name="out", place=Placement(free=(m, n)), axes=(m, n, k))
    graph = Graph()
    _input(graph, "computed", (8, 16))
    _input(graph, "direct", (16, 8) if side == "a" else (8, 16))
    graph.add_node(tile, ["computed", "direct"], Tensor("out", (8, 8), "f16"), node_id="out")
    graph.inputs, graph.outputs = ["computed", "direct"], ["out"]
    return graph


def _mimo_graph() -> Graph:
    m, n, k = Axis("m", 8), Axis("n", 8), Axis("k", 16)

    def matmul(a: str, b: str, acc: str) -> Fold:
        return contraction(
            k,
            Load(name=f"{a}_v", input=a, index=(Var("m"), Var("k"))),
            (Load(name=f"{b}_v", input=b, index=(Var("k"), Var("n"))), acc),
        )

    first, second = matmul("a", "b", "first"), matmul("c", "d", "second")
    tile = TileOp(
        op=projection((first, second), results=("first", "second")),
        name="out0",
        place=Placement(free=(m, n)),
        axes=(m, n, k),
        output_specs=(
            OutputSpec(Write(output="out0", index=(Var("m"), Var("n")), value="first")),
            OutputSpec(Write(output="out1", index=(Var("m"), Var("n")), value="second")),
        ),
    )
    graph = Graph()
    for name in ("a", "c"):
        _input(graph, name, (8, 16))
    for name in ("b", "d"):
        _input(graph, name, (16, 8))
    graph.add_node(
        tile,
        ["a", "b", "c", "d"],
        outputs=(Tensor("out0", (8, 8), "f16"), Tensor("out1", (8, 8), "f16")),
        node_id="out0",
    )
    graph.inputs, graph.outputs = ["a", "b", "c", "d"], ["out0", "out1"]
    return graph


def _sdpa_graph() -> Graph:
    graph = Graph()
    for name in ("q", "k", "v"):
        _input(graph, name, (1, 2, 8, 16))
    graph.add_node(SdpaOp(), ["q", "k", "v"], Tensor("out", (1, 2, 8, 16), "f16"), node_id="out")
    graph.inputs, graph.outputs = ["q", "k", "v"], ["out"]
    return graph


def _softmax_graph() -> Graph:
    graph = Graph()
    _input(graph, "x", (4, 32))
    graph.add_node(SoftmaxOp(axis=-1), ["x"], Tensor("out", (4, 32), "f16"), node_id="out")
    graph.inputs, graph.outputs = ["x"], ["out"]
    return graph


def _offered(graph: Graph, *, frontend: bool = False) -> list[dict]:
    offered: list[dict] = []
    passes = TILE_PASSES if frontend else ["tile/cut"]
    select = None if frontend else {"cut"}

    def decide(fork):
        place = [option for option in fork.options if any(str(key).startswith("PLACE") for key in option.knobs)]
        if place:
            offered.extend(dict(option.knobs) for option in place)
            return next(option for option in place if not _is_structural_option(option))
        option = fork.options[0]
        while isinstance(option, Fork) and not option.is_leaf:
            option = option.expand()[0]
        return option

    Run(Pipeline.build(passes, select=select), _CTX).resolve(graph, decide)
    return offered


def _lower(graph: Graph, placement: dict[str, str]) -> Graph:
    with pinned_knobs(placement):
        lowered, _ = Run(Pipeline.build(CUDA_PASSES), _CTX).resolve(graph, direct_classic_leaf)
    lowered.validate()
    return lowered


def _lower_cut(graph: Graph, spelling: str) -> Graph:
    return _lower(graph, {spelling: "cut"})


def _case_match(case: str) -> tuple[Match, Graph]:
    """A one-node match on a corpus case's lifted target — the fork point the cut pass rewrites."""
    tile = case_target_tile(case)
    graph = Graph()
    for name, tensor in tile.inputs.items():
        graph.add_node(InputOp(), [], tensor, node_id=name)
    graph.add_node(tile, list(tile.inputs), next(iter(tile.outputs.values())), node_id=tile.name)
    return Match(graph=graph, root_node_id=tile.name, rule=Rule(name="test", pattern=[])), graph


def _nested_attention_cut(pins: dict[str, str]) -> Graph:
    match, graph = _case_match("attention/rmsnorm-gqa-b-cut.json")
    with pinned_knobs(pins):
        result = _CUT.rewrite(match, graph.nodes[match.root_node_id])
    options = result if isinstance(result, list) else [result]
    cut = next(option for option in options if "cut" in option.knobs.values())
    return cut.expand()[0]


def _piece_with_seam(fragment: Graph):
    return next(node for node in fragment.nodes.values() if isinstance(node.op, TileOp) and cuttable_seams(node.op))


def test_a_pipeline_that_stops_at_the_cut_pass_decides_the_kernel_set_and_schedules_nothing(monkeypatch) -> None:
    """``compile --passes dolfstp``: a kernel-set fork is decided from what its arms are, never by scheduling
    them, so a greedy compile that never reaches ``tile/schedule`` decides the offered cuts the way a full compile
    does — and scores no schedule row and leaves every kernel unscheduled."""
    from emmy.compiler.pipeline.search.db import SearchDB
    from emmy.compiler.pipeline.search.policy import greedy as policy

    class NoSchedule:
        def mean_scores_features(self, rows):
            raise AssertionError("a kernel-set fork must not score a schedule row")

    monkeypatch.setattr(policy, "_load_prior_safe", NoSchedule)
    assert any(value == "cut" for offer in _offered(_softmax_graph(), frontend=True) for value in offer.values())
    result = Pipeline.build([*LOOP_PASSES, "tile/lift", "tile/cut"]).run(_softmax_graph(), ctx=_CTX, db=SearchDB())
    kernels = [node.op for node in result.nodes.values() if isinstance(node.op, TileOp)]
    assert kernels and all(kernel.schedule is None for kernel in kernels)


def test_cut_workspace_retains_static_unit_axes() -> None:
    """A unit seam axis remains workspace geometry even when the produced value is invariant in it."""
    unit, unused, column = Axis("batch", 1), Axis("unused", 8), Axis("n", 64)
    produced = projection((), (Load(name="value", input="x", index=(Var("n"),)),), results=("value",))
    seam = CutSite(
        node=produced,
        spelling="PLACE",
        axes=(unit, unused, column),
        dtypes=(F16,),
    )

    assert _workspace_axes(seam, produced) == (unit, column)


@pytest.mark.parametrize("second_divisor,expected_rows", [(8, 3), (2, 5), (1, 10)])
def test_cut_stores_one_value_per_repeated_coordinate_group(second_divisor, expected_rows):
    """A partial final group and mixed divisors preserve the full consumer's outputs."""
    from emmy.compiler.backend.numpy import NumpyBackend
    from emmy.compiler.ir.loop import LoopOp

    m, n, k = Axis("m", 10), Axis("n", 3), Axis("k", 7)
    computed = projection(
        (),
        (
            Load(name="x_value", input="x", index=(Var("m") / 4, Var("k"))),
            Load(name="y_value", input="y", index=(Var("m") / second_divisor, Var("k"))),
            Assign(name="scaled", op="multiply", args=("x_value", "y_value")),
        ),
    )
    fold = contraction(k, computed, (Load(name="weight", input="w", index=(Var("k"), Var("n"))), "acc"))
    tile = TileOp(
        op=fold,
        place=Placement(free=(m, n)),
        axes=(m, n, k),
        output_specs=(OutputSpec(Write(output="out", index=(Var("m"), Var("n")), value="acc")),),
    )
    graph = Graph()
    for name, shape in (("x", (3, 7)), ("y", (10, 7)), ("w", (7, 3))):
        _input(graph, name, shape, "f32")
    graph.add_node(tile, ["x", "y", "w"], Tensor("out", (10, 3)), node_id="out")
    graph.inputs, graph.outputs = ["x", "y", "w"], ["out"]
    tile = tile.with_io(graph, graph.nodes["out"])
    graph.nodes["out"].op = tile
    match = Match(graph=graph, root_node_id="out", rule=Rule(name="test", pattern=[]))
    fragment = realize(match, match.root, cuttable_seams(tile), placement_decided=True)
    fragment.inputs = list(graph.inputs)
    workspace = next(node for node in fragment.nodes.values() if isinstance(node.op, TileOp) and node.id.startswith("out__place_"))
    assert tuple(d.as_static() for d in workspace.outputs[0].shape) == (expected_rows, 7)

    rng = np.random.default_rng(1)
    inputs = {name: rng.standard_normal(tuple(d.as_static() for d in graph.buffer(name).shape)).astype(np.float32) for name in graph.inputs}

    def run(g):
        g = g.copy()
        for node in g.nodes.values():
            if isinstance(node.op, TileOp):
                op = node.op
                node.op = LoopOp(body=op.op.lower(bound=frozenset(), stores=op.output_specs, axes=op.axes))
        backend = NumpyBackend()
        return backend.run(backend.compile(g), input_data=inputs)[0].outputs

    for got, want in zip(run(fragment).values(), run(graph).values(), strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("live_dependents", [0, 1, 2])
def test_sibling_producer_fusion_prunes_unused_later_group(live_dependents: int) -> None:
    """Splicing the first group can remove unused producers from the next group."""
    from emmy.compiler.backend.numpy import NumpyBackend
    from emmy.compiler.ir.loop import LoopOp

    graph = Graph()
    _input(graph, "x", (4,), "f32")
    axis = Axis("i", 4)
    for name, source in (("a", "x"), ("b", "x"), ("c", "a"), ("d", "a")):
        fold = projection((), (Load(name="value", input=source, index=(Var("i"),)),))
        tile = TileOp(
            op=fold,
            name=name,
            place=Placement(free=(axis,)),
            axes=(axis,),
            output_specs=(OutputSpec(Write(output=name, index=(Var("i"),), value="value")),),
        )
        graph.add_node(tile, [source], Tensor(name, (4,), "f32"), node_id=name)
    graph.inputs, graph.outputs = ["x"], ["a", "b", *("c", "d")[:live_dependents]]
    before = graph.copy()
    parent = graph.nodes["a"].op.with_io(graph, graph.nodes["a"])

    _fuse_sibling_producers(graph, ("a", "b", "c", "d"), parent)

    assert len([node for node in graph.nodes.values() if isinstance(node.op, TileOp)]) == 1 + bool(live_dependents)
    assert all(graph.producer(name) is None for name in ("c", "d")[live_dependents:])
    inputs = {"x": np.array([-3.0, 0.5, 2.0, 7.0], dtype=np.float32)}

    def run(g):
        for node in g.nodes.values():
            if isinstance(node.op, TileOp):
                node.op = LoopOp(body=node.op.loop_body)
        backend = NumpyBackend()
        return backend.run(backend.compile(g), input_data=inputs)[0].outputs

    for got, want in zip(run(graph).values(), run(before).values(), strict=True):
        np.testing.assert_array_equal(got, want)


def test_composed_cut_topologically_orders_equal_degree_workspace_chain() -> None:
    """Counting direct workspace reads cannot order A->C->B when A and C each read one."""

    def piece(name: str, source: str | None):
        produced = projection((), (Load(name=f"{name}_value", input=source or "input", index=()),), results=(f"{name}_value",))
        return (None, produced, (), (), name, (f"{name}_value",), (name,))

    pieces = [piece("a", "c"), piece("c", "b"), piece("b", None)]

    assert [buffers[0] for *_, buffers in _producer_order(pieces)] == ["b", "c", "a"]


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("computed", [False, True])
def test_cut_does_not_rematerialize_load_only_bundles(channels: int, computed: bool) -> None:
    axis = Axis("i", 8)
    names = tuple(f"v{channel}" for channel in range(channels))
    loads = tuple(Load(name=name, input=f"workspace{channel}", index=(Var("i"),)) for channel, name in enumerate(names))
    values = tuple(f"computed{channel}" for channel in range(channels)) if computed else names
    body = (*loads, *(Assign(name=value, op="negative", args=(name,)) for name, value in zip(names, values, strict=True) if computed))
    bundle = projection((), body, values)
    outputs = tuple(f"out{channel}" for channel in range(channels))
    tile = TileOp(
        op=projection(
            (bundle,), tuple(Assign(name=out, op="negative", args=(value,)) for out, value in zip(outputs, values, strict=True)), outputs
        ),
        name="out",
        place=Placement(free=(axis,)),
        axes=(axis,),
        output_specs=tuple(OutputSpec(Write(output=out, index=(Var("i"),), value=out)) for out in outputs),
    )
    graph = Graph()
    for channel in range(channels):
        _input(graph, f"workspace{channel}", (8,))
    nid = graph.add_node(
        tile, [f"workspace{channel}" for channel in range(channels)], outputs=tuple(Tensor(out, (8,), "f16") for out in outputs)
    )
    assert bool(cuttable_seams(tile.with_io(graph, graph.nodes[nid]))) is computed


def test_pinned_fusion_lowers_one_computed_operand_kernel() -> None:
    lowered = _lower(_computed_operand_graph("a"), {"PLACE": "fuse"})
    assert sum(type(node.op).__name__ == "CudaOp" for node in lowered.nodes.values()) == 1


@pytest.mark.parametrize("side", ("a", "b"))
def test_computed_operand_offers_fused_and_cut_and_pinned_cut_lowers(side: str) -> None:
    offered = _offered(_computed_operand_graph(side))
    assert {frozenset(row.items()) for row in offered} == {
        frozenset({("PLACE", "fuse")}),
        frozenset({("PLACE", "cut")}),
    }
    lowered = _lower_cut(_computed_operand_graph(side), "PLACE")
    cuda = [node for node in lowered.nodes.values() if type(node.op).__name__ == "CudaOp"]
    assert len(cuda) == 2
    assert len(cuda[1].inputs) == 2 and any("__place_" in name for name in cuda[1].inputs)


def test_sdpa_score_cut_is_offered_and_pinned_cut_lowers() -> None:
    offered = _offered(_sdpa_graph(), frontend=True)
    assert {"PLACE": "fuse"} in offered
    assert {"PLACE@map.1/twist.1/inner": "cut"} in offered
    lowered = _lower_cut(_sdpa_graph(), "PLACE@map.1/twist.1/inner")
    cuda = [node for node in lowered.nodes.values() if type(node.op).__name__ == "CudaOp"]
    assert len(cuda) == 2  # the two pieces of the cut
    workspace = next(node.output for node in cuda if "__place_" in node.id)
    assert workspace.dtype.name == "f32"


def _sdpa_document(gpu_name: str | None = None):
    """The sdpa program's inventory on a 12.0 card: one target kernel, one traced-only row."""
    return inventory_document(_sdpa_graph(), (12, 0), gpu_name=gpu_name)


def _routed(document, arm: dict, *, measured: bool = True, target=None):
    """``document`` with the decision ``arm`` taken on its target and recorded as the DB holds it: the routing row, the
    pieces it minted, and a row per piece (measured at one microsecond when ``measured``)."""
    if target is None:
        [target] = document.targets()
    probe = RoutingRow(target.ref, arm, ("a piece",))
    ctx = Context.from_target((12, 0), gpu_name=document.gpu_name or None)
    taken = [kernels for route, _same, kernels in mint(target, [probe], ctx) if route is probe]
    if not taken:
        return document, None
    pieces = [document.add_kernel(piece) for piece in taken[0]]
    route = RoutingRow(target.ref, dict(arm), tuple(piece.ref for piece in pieces))
    document.add_routing(route)
    stand_in = Measurements(emmy_us=1.0, reference_us=2.0, reference_backend="torch") if measured else None
    for piece in pieces:
        document.rows.append(Row(name=f"sdpa.{piece.exact_identity[:12]}", kernel=piece.ref, knobs={}, measurements=stand_in))
    return document, route


def test_a_recorded_cut_is_taken_again_and_a_stale_route_mints_nothing() -> None:
    """A routing row's arm is taken again on the fresh parent by the restamp's mint: a seam the parent offers mints
    the pieces; a key the parent has no site for decides nothing, and a route naming one beside a real seam is not
    the decision the fresh parent takes (it takes the one seam), so the restamp drops it."""
    document, route = _routed(_sdpa_document(), {"PLACE@map.1/twist.1/inner": "cut"})
    assert route is not None and len(route.children) >= 2
    fresh, report = restamp(document)
    assert fresh == document and not report.changed

    document, route = _routed(_sdpa_document(), {"PLACE@missing": "cut"})
    assert route is None, "a key that names no site of the tree decides nothing"
    # The stale seam here is well formed and stands on no site of this tree: one hop past the score contraction,
    # where the operand is a gmem slab and takes no hop of its own.
    partial = {"PLACE@map.1/twist.1/inner": "cut", "PLACE@map.1/twist.1/inner.1/map": "cut"}
    document, route = _routed(_sdpa_document(), partial)
    assert route is not None, "the fresh parent takes the one seam it offers"
    stale = replace(document, routing=[replace(route, arm=partial)])
    fresh, report = restamp(stale)
    assert fresh.routing == [] and report.dropped_routes
    assert all(row.kernel not in route.children for row in fresh.rows), "the pieces' rows go with the decision"


@requires_cuda
def test_softmax_state_cut_is_offered_and_pinned_cut_lowers() -> None:
    offered = _offered(_softmax_graph(), frontend=True)
    # Direct division leaves one cuttable seam: the maximum and denominator carrier.
    assert {"PLACE": "fuse"} in offered and {"PLACE": "cut"} in offered
    lowered = _lower_cut(_softmax_graph(), "PLACE")
    cuda = [node for node in lowered.nodes.values() if type(node.op).__name__ == "CudaOp"]
    assert len(cuda) == 2
    assert len(next(node for node in cuda if "__place_" in node.id).outputs) == 2  # maximum + denominator state
    values = np.linspace(-3, 3, 128, dtype=np.float16).reshape(4, 32)
    got = CudaBackend().run(lowered, input_data={"x": values})[0].outputs["out"]
    shifted = values.astype(np.float32) - values.max(axis=-1, keepdims=True).astype(np.float32)
    expected = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(got, expected, rtol=2e-3, atol=2e-3)


def test_mimo_cut_preserves_both_outputs_and_lowers_both_pieces() -> None:
    offered = _offered(_mimo_graph())
    cuts = [next(iter(row)) for row in offered if next(iter(row.values())) == "cut"]
    assert len(cuts) == 2
    lowered = _lower_cut(_mimo_graph(), cuts[0])
    assert lowered.outputs == ["out0", "out1"]
    assert sum(type(node.op).__name__ == "CudaOp" for node in lowered.nodes.values()) == 2


def test_scoped_place_cut_is_consumed_once_by_both_pieces() -> None:
    fragment = _nested_attention_cut({"PLACE@map.1/twist.1/inner.2/map": "cut"})
    pieces = [node for node in fragment.nodes.values() if isinstance(node.op, TileOp)]

    assert pieces and all(node.op.placement_decided for node in pieces)
    node = _piece_with_seam(fragment)
    match = Match(graph=fragment, root_node_id=node.id, rule=Rule(name="test", pattern=[]))
    # The pin is consumed: the rule offers nothing under PLACE again — its remaining domain (split
    # forks) or, with none pending, its skip.
    with pinned_knobs({"PLACE@map.1/twist.1/inner.2/map": "cut"}):
        try:
            result = _CUT.rewrite(match, node)
        except RuleSkipped as skipped:
            assert "no pending kernel-set cut" in str(skipped)
            return
    options = result if isinstance(result, list) else [result]
    assert options and all(not any(name.startswith("PLACE") for name in option.knobs) for option in options)


def test_bare_place_cut_is_consumed_once_by_both_pieces() -> None:
    fragment = _nested_attention_cut({"PLACE": "cut"})
    node = _piece_with_seam(fragment)
    match = Match(graph=fragment, root_node_id=node.id, rule=Rule(name="test", pattern=[]))

    assert node.op.placement_decided
    with pinned_knobs({"PLACE": "cut"}):
        result = _CUT.rewrite(match, node)
    options = result if isinstance(result, list) else [result]
    assert options and all(not any(name.startswith("PLACE") for name in option.knobs) for option in options)


def test_unpinned_place_keeps_offering_fuse_and_recursive_cuts() -> None:
    fragment = _nested_attention_cut({})
    node = _piece_with_seam(fragment)
    match = Match(graph=fragment, root_node_id=node.id, rule=Rule(name="test", pattern=[]))

    assert not node.op.placement_decided
    options = _CUT.rewrite(match, node)
    assert {"fuse", "cut"} <= {value for option in options for value in option.knobs.values()}


def test_composed_scoped_place_pins_cut_together_and_foreign_pins_are_skipped() -> None:
    """Scoped cuts compose; sibling workspace producers fuse, while their consumer stays cut."""
    match, graph = _case_match("attention/rmsnorm-qk-sdpa-composed-cut.json")
    pins = {
        "PLACE@map.1/twist.1/inner.1/map": "cut",  # the normalized-Q cone
        "PLACE@map.1/twist.1/inner.2/map": "cut",  # the normalized-K cone
        "PLACE@map.1/twist.1/inner.2/map.3/map.1/reduce": "cut",  # the K statistic nested inside it
        "PLACE@map.9/map": "cut",  # no such site here — another kernel's pin
    }
    with pinned_knobs(pins):
        fork = _CUT.rewrite(match, graph.nodes[match.root_node_id])
    assert set(fork.knobs) == {
        "PLACE@map.1/twist.1/inner.1/map",
        "PLACE@map.1/twist.1/inner.2/map",
        "PLACE@map.1/twist.1/inner.2/map.3/map.1/reduce",
    }
    (fragment,) = fork.expand()
    pieces = [node for node in fragment.nodes.values() if isinstance(node.op, TileOp)]
    producers = [node for node in pieces if "__place_" in node.id]
    assert len(producers) == 2 and len(pieces) == 3
    fused = next(node for node in producers if len(node.outputs) == 2)
    assert len([seam for seam in cuttable_seams(fused.op) if seam.owned]) == 2
    assert not fused.op.placement_decided, "a later output cut must still be offered"
    assert all(node.op.placement_decided for node in pieces if node is not fused)
    assert any(set(node.inputs) & {producer.id for producer in producers} for node in producers)


def test_bare_and_scoped_place_cuts_compose_in_one_decision() -> None:
    match, graph = _case_match("attention/rmsnorm-qk-sdpa-composed-cut.json")
    pins = {"PLACE": "cut", "PLACE@map.1/twist.1/inner.2/map": "cut"}

    with pinned_knobs(pins):
        fork = _CUT.rewrite(match, graph.nodes[match.root_node_id])

    assert len(fork.knobs) == 2 and set(fork.knobs.values()) == {"cut"}
    (fragment,) = fork.expand()
    pieces = [node for node in fragment.nodes.values() if isinstance(node.op, TileOp)]
    assert len(pieces) == 3 and all(node.op.placement_decided for node in pieces)


def test_a_composed_route_skips_a_bare_key_and_still_fails_on_a_broken_one() -> None:
    """A composed route is registered for EVERY kernel of the compile, so most of its keys address
    another one. A scoped key off this tree is skipped already; a BARE key must be too, because the
    codec spells bare for a family with ONE site — so the kernel that spelled it has one seam and
    composes with nothing, and asking this tree about it is asking about somebody else. On a tree
    with several sites that question is ambiguous, and letting the ambiguity out ended the compile
    over another kernel's evidence: every strict decode of a golden holding one bare and one scoped
    routing row for the same kernel set died here, and so would the deploy compile reading it.

    The skip stays that narrow. The codec still calls a bare key on a several-site tree ambiguous,
    and a route key off the grammar is still a broken stored row that raises."""
    from emmy.compiler.pipeline.search.pins import composed_routes  # noqa: PLC0415

    graph = _mimo_graph()
    match = Match(graph=graph, root_node_id="out0", rule=Rule(name="test", pattern=[]))
    root = graph.nodes[match.root_node_id]
    with pytest.raises(ValueError, match="PLACE is ambiguous"):
        resolve(root.op.op, "PLACE")

    with composed_routes([(None, ("PLACE", "PLACE@map.1/inner"))]):
        options = _CUT.rewrite(match, root, _CTX)

    options = options if isinstance(options, list) else [options]
    assert options, "the ordinary fuse and single-seam arms still stand"
    assert all(option.knobs.get("PLACE") != "cut" for option in options), "no arm cuts under the unattributable bare key"

    with composed_routes([(None, ("PLACE@map.1/inner", "PLACE@map.1/not-a-kind"))]), pytest.raises(ValueError):
        _CUT.rewrite(match, root, _CTX)


def test_import_files_each_row_under_the_kernel_it_decides(monkeypatch) -> None:
    """Golden evidence is per kernel, in the DB as in the file: the decision is a routing row on the parent, and
    each piece's row is that piece's perf row, under the golden's source. The parent ran as no kernel and has none."""
    monkeypatch.setenv("EMMY_FAST_MATH", "0")
    document, route = _routed(_sdpa_document(), {"PLACE@map.1/twist.1/inner": "cut"})
    db = SearchDB()
    assert import_rows(db, Context.from_target((12, 0)), document, document.rows, source="golden:test") == len(route.children)
    assert [stored.arm for stored in db.iter_routing()] == [route.arm]
    rows = list(db.iter_perf_rows())
    identity = document.identities()
    assert {row.kernel for row in rows} == {identity[child] for child in route.children}
    assert all((row.stats.median, row.captured, row.source) == (1.0, True, "golden:test") for row in rows)


def test_a_multi_output_kernels_entry_carries_the_identity_its_live_fork_carries() -> None:
    """A kernel that writes SEVERAL output buffers is stored under the identity its live fork carries. Every evidence
    row a golden contributes is keyed by that identity, so an entry derived from one output slot alone would key the
    rows off a fingerprint no fork can produce and the deploy would read none of them."""
    from emmy.compiler.wire import kernel_tile

    graph = Graph()
    _input(graph, "x", (8,))
    graph.add_node(ElementwiseOp("relu"), ["x"], Tensor("hot", (8,), "f16"), node_id="hot")
    graph.add_node(ElementwiseOp("negative"), ["hot"], Tensor("cold", (8,), "f16"), node_id="cold")
    graph.inputs, graph.outputs = ["x"], ["hot", "cold"]
    document = inventory_document(graph, (12, 0))
    [kernel] = document.kernels
    assert set(kernel.loop_ir["outputs"]) == {"hot", "cold"}, "the fused target is ONE kernel writing two buffers"
    with evidence_scope([]):
        lowered = Pipeline.build(CUDA_PASSES).run(kernel.program({}), ctx=_CTX, db=None)
    [op] = [node.op for node in lowered.nodes.values() if isinstance(node.op, CudaOp)]
    assert kernel_tile(op).identity_key(structural=False, with_io=True) == kernel.exact_identity


# ---------------------------------------------------------------------------
# The routing lane: a recorded ROUTING row decides a placement fork.
# ---------------------------------------------------------------------------

#: A card in the ``emmy.gpu`` registry, so the record's context reconstructs without a live device.
_ROUTING_CARD = "NVIDIA GeForce RTX 5090"


def _deploy_kernels(document) -> list[str]:
    """Resolve the sdpa program through the deploy policy with ``document`` as the card's golden scope — imported into
    the compile's DB, as every compile imports its golden scope — and return the resolved kernel set. ``prior=None``
    pins the non-recorded forks to emission order, so the recorded evidence is the only thing that can move the
    answer."""
    from emmy.compiler.pipeline.search.golden.evidence import evidence_db
    from emmy.compiler.pipeline.search.policy.greedy import greedy_decide

    ctx = Context.from_target((12, 0), gpu_name=_ROUTING_CARD)
    lowered = Pipeline.build(LOOP_PASSES).run(_sdpa_graph(), ctx=ctx)
    with evidence_scope([document] if document is not None else []):
        db = evidence_db(None, ctx)
        terminal, _trace = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx).resolve(lowered, greedy_decide(prior=None, db=db))
    return sorted(node.id for node in terminal.nodes.values() if isinstance(node.op, TileOp))


#: The root-most of the sdpa kernel's offered seams, as the route codec spells it: the twist that
#: carries the softmax statistics.
_SDPA_ROUTE = "PLACE@map.1/twist"


def test_a_recorded_kernel_set_deploys_the_cut_its_routing_row_records(monkeypatch) -> None:
    """A routing row on the target's kernel, priced from the rows of the pieces it minted, is what makes the deploy
    take the cut: with no rows the fork falls to emission order (fuse), a routing row whose pieces carry no row prices
    nothing, and the routing row beside its pieces' rows deploys the one seam it records."""
    monkeypatch.setenv("EMMY_FAST_MATH", "0")
    fused = _deploy_kernels(None)
    assert len(fused) == 1, f"with no recorded route the fork falls to emission order (fuse): {fused}"

    unpriced, route = _routed(_sdpa_document(_ROUTING_CARD), {_SDPA_ROUTE: "cut"}, measured=False)
    assert route is not None and _deploy_kernels(unpriced) == fused, "a routing row alone prices nothing: its pieces have no row"

    routed, _route = _routed(_sdpa_document(_ROUTING_CARD), {_SDPA_ROUTE: "cut"})
    deployed = _deploy_kernels(routed)
    assert sum(1 for name in deployed if "__place_" in name) == 1, f"the routing row deploys its one seam: {deployed}"


def _cone_seam() -> CutSite:
    """A bare seam record standing in for a clustered operand cone."""
    node = projection((), (Load(name="w", input="w", index=(Var("n"), Var("k"))),), results=("w",))
    return CutSite(node=node, spelling="PLACE@map.1/twist.1/inner.2/map", axes=(Axis("n", 8), Axis("k", 8)), dtypes=(F16,))


def test_alpha_equivalent_operand_cones_cluster_into_one_seam() -> None:
    """Two operand cones spelling the same value are ONE placement decision: the representative
    carries the other as a sibling with its capture correspondence."""
    from emmy.compiler.pipeline.passes.tile._cut import _cluster_value_seams

    same = [_cone_seam(), _cone_seam()]

    clustered = _cluster_value_seams(same, (Axis("n", 8), Axis("k", 8)))
    assert len(clustered) == 1 and len(clustered[0].siblings) == 1


def test_a_multi_result_cone_does_not_materialize_its_own_dependency() -> None:
    from emmy.compiler.pipeline.passes.tile._cut import _cluster_value_seams

    norm = projection(body=(Load("x", "x", (Var("m"),)), Assign("norm", "rsqrt", ("x",))))
    maximum = reduction("k", (norm, slab("y", "y", "m", "k")), (Assign("amax__v", "multiply", ("norm", "y")),), ("amax",), "maximum")
    combined = projection((norm, maximum), results=("norm", "amax"))
    axes = (Axis("m", 8), Axis("k", 16))
    seams = [CutSite(combined, "PLACE@map.1/map", axes[:1], (F16, F16)), CutSite(norm, "PLACE@map.1/map.2/reduce.1/map", axes[:1], (F16,))]

    clustered = _cluster_value_seams(seams, axes)

    assert len(clustered) == 2
    assert not any(seam.siblings for seam in clustered)


def _norm_residual_graph(m: int = 16) -> Graph:
    """``y = x @ w`` read twice: under the norm's statistic reduce and at the residual add — the
    shape of a fused decoder half, whose o_proj result feeds the post-attention norm and the
    residual stream. Fusion keeps one definition; the lifted tree holds one cone per scope. At
    ``m == 1`` (decode) every workspace also carries the kernel's unit row axis."""
    from emmy.commands.trace import graph_from_code

    code = (
        "(lambda y: y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-6) + y)"
        f"(torch.matmul(torch.randn({m}, 64, dtype=torch.float16), torch.randn(64, 32, dtype=torch.float16)))"
    )
    return graph_from_code(code)[0]


def _lifted_parent(graph: Graph) -> TileOp:
    """The one fused kernel of ``graph`` as the cut pass first sees it."""
    lowered = Pipeline.build(LOOP_PASSES).run(graph, ctx=_CTX)
    lifted = Pipeline.build(["tile/lift"], select={"lift", "twisted"}).run(lowered, ctx=_CTX)
    (tile,) = [node.op for node in lifted.nodes.values() if isinstance(node.op, TileOp)]
    return tile


def _projection_views_graph() -> Graph:
    """One matrix result read as flat rows, adjacent pairs, and blocks of sixteen."""
    from emmy.commands.trace import graph_from_code

    code = (
        "(lambda x,w: (lambda y: (y.reshape(2,2,16).abs().amax(-1), "
        "y.reshape(2,16,2)[:,:,0] + y.reshape(2,16,2)[:,:,1], y))(torch.matmul(x,w)))"
        "(torch.randn(2,32,dtype=torch.float16), torch.randn(32,32,dtype=torch.float16))"
    )
    return graph_from_code(code)[0]


def _block_pair_projection_graph() -> Graph:
    """One matrix result read as packed pairs and blocks, with no flat-row reader."""
    from emmy.commands.trace import graph_from_code

    code = (
        "(lambda x,w: (lambda y: (y.reshape(2,2,16).abs().amax(-1), "
        "y.reshape(2,16,2)[:,:,0] + y.reshape(2,16,2)[:,:,1]))(torch.matmul(x,w)))"
        "(torch.randn(2,32,dtype=torch.float16), torch.randn(32,32,dtype=torch.float16))"
    )
    return graph_from_code(code)[0]


def test_block_projection_cut_reuses_one_producer_for_packed_pairs() -> None:
    """A row-major two-axis workspace covers both channels of a packed pair."""
    parent = _lifted_parent(_block_pair_projection_graph())
    (seam,) = [seam for seam in cuttable_seams(parent) if seam.node.as_contraction() is not None]
    assert [axis.extent.as_static() for axis in seam.axes] == [2, 2, 16]
    assert len(seam.indexed_siblings) == 1 and len(seam.indexed_siblings[0][1]) == 2

    graph = _block_pair_projection_graph()
    loop = Pipeline.build(LOOP_PASSES).run(graph, ctx=_CTX)
    lifted = Pipeline.build(["tile/lift"], select={"lift", "twisted"}).run(loop, ctx=_CTX)
    root = next(node for node in lifted.nodes.values() if isinstance(node.op, TileOp))
    seam = next(seam for seam in cuttable_seams(root.op) if seam.node.as_contraction() is not None)
    match = Match(graph=lifted, root_node_id=root.id, rule=Rule(name="test", pattern=[]))
    fragment = realize(match, root, (seam,))
    from emmy.compiler.ir.tile.path import sites

    contractions = sum(
        site.node.as_contraction() is not None
        for node in fragment.nodes.values()
        if isinstance(node.op, TileOp)
        for site in sites(node.op.op)
    )
    assert contractions == 1


def test_block_projection_refuses_an_out_of_bounds_packed_pair() -> None:
    """The last pair cannot read a row past the block workspace's final cell."""
    from emmy.compiler.pipeline.passes.tile._cut import _cluster_reindexed_contractions

    m, block, lane, pair, k = Axis("m", 2), Axis("block", 2), Axis("lane", 16), Axis("pair", 17), Axis("k", 8)
    flat_row = Var("block") * 16 + Var("lane")
    block_contraction = contraction(k, slab("a", "A", "m", "k"), (slab("w", "W", flat_row, "k"), "acc"))
    pair_contraction = contraction(k, slab("a", "A", "m", "k"), (slab("w", "W", Var("pair") * 2, "k"), "acc"))
    seams = (
        CutSite(block_contraction, "PLACE@map.1/inner", (m, block, lane), (F16,)),
        CutSite(pair_contraction, "PLACE@map.2/inner", (m, pair), (F16,)),
    )

    assert len(_cluster_reindexed_contractions(seams, (m, block, lane, pair, k))) == 2


def test_indexed_reader_omits_an_unread_workspace_component() -> None:
    """A twin may expose a second channel that no reader takes; no buffer is written for it."""
    from emmy.compiler.pipeline.passes.tile._cut import _indexed_read

    m, block, lane, k = Axis("m", 2), Axis("block", 2), Axis("lane", 16), Axis("k", 8)
    row = Var("block") * 16 + Var("lane")
    twin = contraction(
        k,
        slab("a", "A", "m", "k"),
        (slab("gate", "G", row, "k"), "gate_acc"),
        (slab("up", "U", row, "k"), "up_acc"),
    )
    seam = CutSite(twin, "PLACE@map.1/inner", (m, block, lane), (F16, F16))
    address = (("m", Var("m")), ("block", Var("pair") / 16), ("lane", Var("pair") % 16))
    held = {0: "gate_workspace"}
    indexes = {0: (Var("m"), Var("block"), Var("lane"))}

    read = _indexed_read(seam, "gate_pair", 0, address, held, indexes, "cut", 0)
    assert read is not None and read.exposes == ("gate_pair__wscuts0",)
    assert _indexed_read(seam, "unused_up_pair", 1, address, held, indexes, "cut", 0) is None


@requires_cuda
def test_block_projection_cut_matches_packed_pair_and_block_oracles() -> None:
    graph = _block_pair_projection_graph()
    (seam,) = [seam for seam in cuttable_seams(_lifted_parent(graph.copy())) if seam.node.as_contraction() is not None]
    cut = _lower_cut(graph, seam.spelling)
    rng = np.random.default_rng(31)
    x = rng.standard_normal((2, 32)).astype(np.float16)
    w = rng.standard_normal((32, 32)).astype(np.float16)
    result = CudaBackend().run(cut, input_data=dict(zip(cut.inputs, (x, w), strict=True)))[0].outputs
    y = (x.astype(np.float32) @ w.astype(np.float32)).astype(np.float16)
    np.testing.assert_allclose(result["add"], y.reshape(2, 16, 2).astype(np.float32).sum(-1).astype(np.float16), rtol=2e-2, atol=3e-2)
    np.testing.assert_allclose(result["amax"], np.abs(y).reshape(2, 2, 16).max(-1), rtol=2e-2, atol=3e-2)


def test_flat_projection_cut_reuses_one_producer_for_pair_and_block_views() -> None:
    """An exact row-address match cuts the matrix once and serves both alternate layouts."""
    parent = _lifted_parent(_projection_views_graph())
    contractions = [seam for seam in cuttable_seams(parent) if seam.node.as_contraction() is not None]
    (projection_seam,) = contractions
    assert [(axis.extent.as_static()) for axis in projection_seam.axes] == [2, 32]
    assert len(projection_seam.indexed_siblings) == 2
    assert sorted(len(addresses) for _, addresses in projection_seam.indexed_siblings) == [1, 2]

    from emmy.compiler.ir.tile.path import sites

    graph = _projection_views_graph()
    loop = Pipeline.build(LOOP_PASSES).run(graph, ctx=_CTX)
    lifted = Pipeline.build(["tile/lift"], select={"lift", "twisted"}).run(loop, ctx=_CTX)
    root = next(node for node in lifted.nodes.values() if isinstance(node.op, TileOp))
    seam = next(seam for seam in cuttable_seams(root.op) if seam.node.as_contraction() is not None)
    match = Match(graph=lifted, root_node_id=root.id, rule=Rule(name="test", pattern=[]))
    fragment = realize(match, root, (seam,))
    contractions = sum(
        site.node.as_contraction() is not None
        for node in fragment.nodes.values()
        if isinstance(node.op, TileOp)
        for site in sites(node.op.op)
    )
    assert contractions == 1


def test_reindexed_projection_refuses_a_row_outside_the_workspace() -> None:
    """Equal per-row algebra does not authorize an out-of-bounds workspace read."""
    from emmy.compiler.pipeline.passes.tile._cut import _cluster_reindexed_contractions

    m, n, p, k = Axis("m", 2), Axis("n", 32), Axis("p", 16), Axis("k", 8)
    flat = contraction(k, slab("a", "A", "m", "k"), (slab("w", "W", "n", "k"), "acc"))
    shifted = contraction(k, slab("a", "A", "m", "k"), (slab("w", "W", Var("p") * 2 + 32, "k"), "acc"))
    seams = (
        CutSite(flat, "PLACE@map.1/inner", (m, n), (F16,)),
        CutSite(shifted, "PLACE@map.2/inner", (m, p), (F16,)),
    )

    assert len(_cluster_reindexed_contractions(seams, (m, n, p, k))) == 2


@requires_cuda
def test_reindexed_projection_cut_matches_pair_and_block_oracles() -> None:
    """Workspace reads preserve each original channel's row, including the two packed-pair lanes."""
    graph = _projection_views_graph()
    (seam,) = [seam for seam in cuttable_seams(_lifted_parent(graph.copy())) if seam.node.as_contraction() is not None]
    cut = _lower_cut(graph, seam.spelling)
    rng = np.random.default_rng(21)
    x = rng.standard_normal((2, 32)).astype(np.float16)
    w = rng.standard_normal((32, 32)).astype(np.float16)
    result = CudaBackend().run(cut, input_data=dict(zip(cut.inputs, (x, w), strict=True)))[0].outputs
    y = (x.astype(np.float32) @ w.astype(np.float32)).astype(np.float16)
    expected = {
        "matmul": y,
        "add": y.reshape(2, 16, 2).astype(np.float32).sum(-1).astype(np.float16),
        "amax": np.abs(y).reshape(2, 2, 16).max(-1),
    }
    for name, value in expected.items():
        np.testing.assert_allclose(result[name], value, rtol=2e-2, atol=3e-2)


@pytest.mark.parametrize("m", [16, 1])
def test_a_value_read_under_a_reduce_and_at_the_free_axis_is_one_seam(m: int) -> None:
    """The two copies of the contraction bind the hidden coordinate under different names (the
    reduce's own axis, the kernel's free axis) and their slab params may sit in another order, so
    their canonical forms differ; they are one value, and cutting it once must materialize it once
    with every occurrence reading the workspace."""
    parent = _lifted_parent(_norm_residual_graph(m))
    contractions = [seam for seam in cuttable_seams(parent) if seam.node.as_contraction() is not None]
    assert len(contractions) == 1 and len(contractions[0].siblings) == 1, [seam.spelling for seam in cuttable_seams(parent)]
    cut = _lower_cut(_norm_residual_graph(m), contractions[0].spelling)
    cuda = [node for node in cut.nodes.values() if type(node.op).__name__ == "CudaOp"]
    assert len(cuda) == 2


@requires_cuda
@pytest.mark.parametrize("m", [16, 1])
def test_a_clustered_value_cut_once_computes_the_right_answer(m: int) -> None:
    graph = _norm_residual_graph(m)
    (seam,) = [seam for seam in cuttable_seams(_lifted_parent(graph.copy())) if seam.node.as_contraction() is not None]
    cut = _lower_cut(graph, seam.spelling)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((m, 64)).astype(np.float16)
    w = rng.standard_normal((64, 32)).astype(np.float16)
    inputs = dict(zip(cut.inputs, (x, w), strict=True))
    (out_name,) = cut.outputs
    got = CudaBackend().run(cut, input_data=inputs)[0].outputs[out_name].astype(np.float32)
    y = x.astype(np.float32) @ w.astype(np.float32)
    expected = y / np.sqrt((y * y).mean(-1, keepdims=True) + 1e-6) + y
    np.testing.assert_allclose(got, expected, rtol=2e-2, atol=2e-1)


def _column_seam(column, spelling: str) -> CutSite:
    """A cone reading its weight at ``column``, an expression of the captured ``n``."""
    node = projection((), (Load(name="w", input="w", index=(column, Var("k"))),), results=("w",))
    return CutSite(node=node, spelling=spelling, axes=(Axis("n", 8), Axis("k", 8)), dtypes=(F16,))


def test_copies_read_at_shifted_columns_cluster_onto_the_plain_read() -> None:
    """RoPE's rotate-half reads one projection at its own column and at the two half-shifted ones:
    one value at three addresses. The plain copy computes it, and each shifted copy reads its
    workspace at the column expression it spelled."""
    from emmy.compiler.ir.expr import Literal, TernaryExpr
    from emmy.compiler.pipeline.passes.tile._cut import _cluster_value_seams

    n, low = Var("n"), Var("n").lt(4)
    lower = TernaryExpr(cond=low, if_true=Literal(0, "int"), if_false=n + Literal(-4, "int"))
    upper = Literal(4, "int") + TernaryExpr(cond=low, if_true=n, if_false=Literal(0, "int"))
    seams = [_column_seam(lower, "PLACE@map.1/inner"), _column_seam(upper, "PLACE@map.2/inner"), _column_seam(n, "PLACE@map.3/inner")]

    (clustered,) = _cluster_value_seams(seams, (Axis("n", 8), Axis("k", 8)))

    assert clustered.spelling == "PLACE@map.3/inner"
    assert sorted(dict(pairs)["n"].pretty() for _, pairs, _ in clustered.siblings) == sorted((lower.pretty(), upper.pretty()))


def test_a_column_read_past_the_plain_copy_is_not_its_value() -> None:
    """A shifted read reaching past the plain copy's axis reads columns that copy never computes."""
    from emmy.compiler.ir.expr import Literal
    from emmy.compiler.pipeline.passes.tile._cut import _cluster_value_seams

    seams = [_column_seam(Var("n") + Literal(4, "int"), "PLACE@map.1/inner"), _column_seam(Var("n"), "PLACE@map.2/inner")]

    assert not any(seam.siblings for seam in _cluster_value_seams(seams, (Axis("n", 8), Axis("k", 8))))


def test_a_conversion_to_the_dtype_a_workspace_stores_reads_the_workspace_itself() -> None:
    """A reader's rounding of a value the workspace already stores rounded is a no-op copy; left in
    place it keeps the edge a computed cone, which the chunk tier refuses for its streamed value."""
    from emmy.compiler.dtype import F32
    from emmy.compiler.pipeline.passes.tile._cut import _without_identity_casts

    rounded = projection((slab("x", "ws", "m"),), (Assign(name="y", op="copy", args=("x",), dtype=F16),), ("y",))
    root = projection((rounded,), (Assign(name="z", op="exp", args=("y",)),), ("z",))

    kept = _without_identity_casts(root, {"ws": F16}).operands[0]
    assert kept.as_slab() is not None and kept.exposes == ("y",)
    assert _without_identity_casts(root, {"ws": F32}) is root


def test_a_row_spelled_at_any_occurrence_of_a_clustered_value_names_its_cut() -> None:
    """The arm that cuts a clustered seam spells every occurrence, and a route recorded at one of
    them — a row from before the clustering, a pin at the copy a hand found — selects that arm."""
    from emmy.compiler.pipeline.search.pins import spelled_arm

    graph = _norm_residual_graph()
    lowered = Pipeline.build(LOOP_PASSES).run(graph, ctx=_CTX)
    lifted = Pipeline.build(["tile/lift"], select={"lift", "twisted"}).run(lowered, ctx=_CTX)
    node = next(node for node in lifted.nodes.values() if isinstance(node.op, TileOp))
    (seam,) = [seam for seam in cuttable_seams(node.op) if seam.node.as_contraction() is not None]
    (alias,) = seam.aliases
    match = Match(graph=lifted, root_node_id=node.id, rule=Rule(name="test", pattern=[]))
    options = _CUT.rewrite(match, node)
    (arm,) = [option for option in options if option.knobs.get(seam.spelling) == "cut"]
    assert arm.knobs[alias] == "cut" and arm.aliases == {alias: seam.spelling}
    assert spelled_arm(options, {alias: "cut"}) == (arm, {key: str(value) for key, value in arm.knobs.items()})
    assert spelled_arm(options, {seam.spelling: "cut"})[0] is arm
    assert spelled_arm(options, {"WORK": "t256"})[0].knobs == {"PLACE": "fuse"}


def _twin_norm_graph() -> Graph:
    """``k = x @ wk`` and ``v = x @ wv`` over one input, with ``k`` normed: the k/v projection pair
    of a fused pre-attention half. Lifting folds the two contractions into one twin; the norm's
    statistic recomputes ``k`` alone under its reduce."""
    from emmy.commands.trace import graph_from_code

    code = (
        "(lambda x, wk, wv: (lambda k, v: k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-6) + v)"
        "(torch.matmul(x, wk), torch.matmul(x, wv)))"
        "(torch.randn(16, 64, dtype=torch.float16), torch.randn(64, 32, dtype=torch.float16), torch.randn(64, 32, dtype=torch.float16))"
    )
    return graph_from_code(code)[0]


def test_a_lone_contraction_is_a_channel_of_the_twin_it_equals() -> None:
    """The twin exposes two values; the cone under the reduce exposes one of them. One placement
    decision materializes the twin, and the lone copy reads the channel that is its value."""
    parent = _lifted_parent(_twin_norm_graph())
    contractions = [seam for seam in cuttable_seams(parent) if seam.node.as_contraction() is not None]
    assert len(contractions) == 1, [seam.spelling for seam in cuttable_seams(parent)]
    (twin,) = contractions
    ((sibling, _, channels),) = twin.siblings
    assert len(twin.node.exposes) == 2 and len(sibling.exposes) == 1 and channels == (0,)
    cut = _lower_cut(_twin_norm_graph(), twin.spelling)
    cuda = [node for node in cut.nodes.values() if type(node.op).__name__ == "CudaOp"]
    assert len(cuda) == 2


@requires_cuda
def test_a_twin_cut_once_serves_its_lone_channel_reader() -> None:
    graph = _twin_norm_graph()
    (seam,) = [seam for seam in cuttable_seams(_lifted_parent(graph.copy())) if seam.node.as_contraction() is not None]
    cut = _lower_cut(graph, seam.spelling)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, 64)).astype(np.float16)
    wk = rng.standard_normal((64, 32)).astype(np.float16)
    wv = rng.standard_normal((64, 32)).astype(np.float16)
    inputs = dict(zip(cut.inputs, (x, wk, wv), strict=True))
    (out_name,) = cut.outputs
    got = CudaBackend().run(cut, input_data=inputs)[0].outputs[out_name].astype(np.float32)
    k = x.astype(np.float32) @ wk.astype(np.float32)
    v = x.astype(np.float32) @ wv.astype(np.float32)
    expected = k / np.sqrt((k * k).mean(-1, keepdims=True) + 1e-6) + v
    np.testing.assert_allclose(got, expected, rtol=2e-2, atol=2e-1)


def _rope_graph() -> Graph:
    """RoPE over ``q = x @ wq`` (four heads of eight) and ``k = x @ wk`` (two heads, each read by two):
    ``t * cos + rotate_half(t) * sin`` for each. Lifting folds both projections and their half-shifted
    copies into one six-channel twin."""
    from emmy.commands.trace import graph_from_code

    code = (
        "(lambda x, wq, wk, c, s: (lambda q, k: q * c + torch.cat((-q[..., 4:], q[..., :4]), -1) * s"
        " + k * c + torch.cat((-k[..., 4:], k[..., :4]), -1) * s)"
        "(torch.matmul(x, wq).view(1, 4, 8), torch.matmul(x, wk).view(1, 2, 8).repeat_interleave(2, 1)))"
        "(torch.randn(1, 64, dtype=torch.float16), torch.randn(64, 32, dtype=torch.float16), torch.randn(64, 16, dtype=torch.float16),"
        " torch.randn(1, 1, 8, dtype=torch.float16), torch.randn(1, 1, 8, dtype=torch.float16))"
    )
    return graph_from_code(code)[0]


def _rope_seam() -> CutSite:
    (seam,) = [seam for seam in cuttable_seams(_lifted_parent(_rope_graph())) if seam.node.as_contraction() is not None]
    return seam


def test_a_twin_channel_read_at_shifted_columns_reads_the_plain_channels_workspace() -> None:
    """RoPE's rotate-half copies are channels of the projection twin, each reading its weight again, and
    the repeated k head reads its weight once per q head. Cut at the projection, each weight is read
    once: the q piece stores four heads, the k piece two, and the reader loads each workspace at the
    three columns a copy reads."""
    seam = _rope_seam()
    assert len(seam.node.exposes) == 6
    with pinned_knobs({seam.spelling: "cut"}):
        lowered = Pipeline.build(LOOP_PASSES).run(_rope_graph(), ctx=_CTX)
        cut, _ = Run(pipeline=Pipeline.build(["tile/lift", "tile/cut"]), ctx=_CTX).resolve(lowered, lambda fork: fork.options[0])

    def loads(node) -> list[tuple[str, str]]:
        body = node.op.op.lower(frozenset(), node.op.output_specs, node.op.axes)
        return [(stmt.input, stmt.index[-1].pretty()) for stmt in body.iter() if isinstance(stmt, Load)]

    *producers, consumer = (cut.nodes[nid] for nid in cut.topological_order() if isinstance(cut.nodes[nid].op, TileOp))
    workspaces = [tensor for node in producers for tensor in node.outputs]
    assert sorted(tuple(d.as_static() for d in tensor.shape) for tensor in workspaces) == [(2, 1, 8), (4, 1, 8)], "k keeps its two heads"
    assert sorted(name for node in producers for name, _ in loads(node) if name in ("x1", "x2")) == ["x1", "x2"], "each weight is read once"
    for node in producers:
        for buffer in node.buffer_names():
            assert len({column for name, column in loads(consumer) if name == buffer}) == 3, "the reader loads a workspace at three columns"


@requires_cuda
def test_a_rope_projection_cut_once_computes_the_right_answer() -> None:
    cut = _lower_cut(_rope_graph(), _rope_seam().spelling)
    rng = np.random.default_rng(0)
    x, wq, wk = (rng.standard_normal(shape).astype(np.float16) for shape in ((1, 64), (64, 32), (64, 16)))
    c, s = (rng.standard_normal((1, 1, 8)).astype(np.float16) for _ in range(2))
    inputs = dict(zip(cut.inputs, (x, wq, wk, c, s), strict=True))
    (out_name,) = cut.outputs
    got = CudaBackend().run(cut, input_data=inputs)[0].outputs[out_name].astype(np.float32)

    def rope(t):
        return t * c.astype(np.float32) + np.concatenate((-t[..., 4:], t[..., :4]), -1) * s.astype(np.float32)

    q = (x.astype(np.float32) @ wq.astype(np.float32)).reshape(1, 4, 8)
    k = np.repeat((x.astype(np.float32) @ wk.astype(np.float32)).reshape(1, 2, 8), 2, axis=1)
    np.testing.assert_allclose(got, rope(q) + rope(k), rtol=2e-2, atol=2e-1)


def _gqa_value_graph() -> Graph:
    """``v = x @ w`` over two heads of eight, each head repeated for two query heads and contracted at the
    flattened (head, head-dim) channel, as the output projection reads a GQA value."""
    from emmy.commands.trace import graph_from_code

    code = (
        "(lambda x, w, y: torch.matmul(torch.matmul(x, w).view(1, 2, 8).repeat_interleave(2, 1).reshape(1, 32), y))"
        "(torch.randn(1, 64, dtype=torch.float16), torch.randn(64, 16, dtype=torch.float16), torch.randn(32, 16, dtype=torch.float16))"
    )
    return graph_from_code(code)[0]


def _gqa_value_cut():
    graph = _gqa_value_graph()
    (seam,) = [seam for seam in cuttable_seams(_lifted_parent(graph.copy())) if seam.node.as_contraction() is not None]
    with pinned_knobs({seam.spelling: "cut"}):
        lowered = Pipeline.build(LOOP_PASSES).run(graph, ctx=_CTX)
        cut, _ = Run(pipeline=Pipeline.build(["tile/lift", "tile/cut"]), ctx=_CTX).resolve(lowered, lambda fork: fork.options[0])
    return cut


def test_a_gqa_value_read_at_its_flat_channel_is_stored_once_per_kv_head() -> None:
    """The value's head is read as ``(i // 8) // 2`` beside its head-dim ``i % 8``: the piece stores 16
    cells, not 32, and reads its weight at the plain column instead of through the repeat. The reader
    loads it at ``(i // 16) * 8 + i % 8``."""
    producer, _consumer = (node for node in _gqa_value_cut().nodes.values() if isinstance(node.op, TileOp))
    assert [d.as_static() for d in producer.outputs[0].shape][-1] == 16
    body = producer.op.op.lower(frozenset(), producer.op.output_specs, producer.op.axes)
    (weight,) = [stmt for stmt in body.iter() if isinstance(stmt, Load) and stmt.input == "x1"]
    assert "/" not in weight.index[-1].pretty() and "%" not in weight.index[-1].pretty()


@requires_cuda
def test_a_gqa_value_stored_once_per_kv_head_computes_the_right_answer() -> None:
    graph = _gqa_value_graph()
    (seam,) = [seam for seam in cuttable_seams(_lifted_parent(graph.copy())) if seam.node.as_contraction() is not None]
    cut = _lower_cut(graph, seam.spelling)
    rng = np.random.default_rng(0)
    x, w, y = (rng.standard_normal(shape).astype(np.float16) for shape in ((1, 64), (64, 16), (32, 16)))
    (out_name,) = cut.outputs
    got = CudaBackend().run(cut, input_data=dict(zip(cut.inputs, (x, w, y), strict=True)))[0].outputs[out_name].astype(np.float32)
    v = np.repeat((x.astype(np.float32) @ w.astype(np.float32)).reshape(1, 2, 8), 2, axis=1).reshape(1, 32)
    np.testing.assert_allclose(got, v @ y.astype(np.float32), rtol=2e-2, atol=5e-1)


def test_a_scalar_operand_is_no_seam() -> None:
    """A value uniform over the kernel — an sdpa scale beside its mask fills — offers no cut. The
    piece would be a kernel writing scalars to a workspace so its reader could read them back, and
    a seam nothing realizes costs the greedy an arm per rank."""
    from emmy.compiler.ir.expr import Literal

    scale = projection(
        body=(
            Load(name="s0", input="sdpa_scale", index=(Literal(0, "int"),)),
            Load(name="s1", input="sdpa_mask_fill", index=(Literal(0, "int"),)),
        ),
        results=("s0", "s1"),
    )
    scores = contraction(
        "k",
        Load(name="q", input="q", index=(Var("m"), Var("k"))),
        (Load(name="kk", input="k", index=(Var("n"), Var("k"))), "acc0"),
    )
    root = projection(
        (scores, scale),
        (Assign(name="v0", op="multiply", args=("acc0", "s0")), Assign(name="v1", op="add", args=("v0", "s1"))),
    )
    tile = TileOp(
        op=root,
        name="k_scores",
        place=Placement(free=(Axis("m", 8), Axis("n", 8))),
        axes=(Axis("m", 8), Axis("n", 8), Axis("k", 8)),
        output_specs=(OutputSpec(write=Write(output="out", index=(Var("m"), Var("n")), values=("v1",))),),
        inputs={name: Tensor(name, (8, 8), "f16") for name in ("q", "k", "sdpa_scale", "sdpa_mask_fill")},
        outputs={"out": Tensor("out", (8, 8), "f16")},
    )
    assert scale.scalar() and not scores.scalar()
    assert [seam.node for seam in cuttable_seams(tile)] == [scores]


def test_every_seam_is_an_unpinned_arm() -> None:
    """The unpinned fork offers every cuttable seam as its own structural arm, spelled by the same
    key the pin path resolves."""
    match, graph = _case_match("attention/rmsnorm-qk-sdpa-composed-cut.json")
    node = next(node for node in graph.nodes.values() if isinstance(node.op, TileOp))
    options = _CUT.rewrite(match, node)
    arms = [dict(option.knobs) for option in options if "cut" in option.knobs.values()]
    seams = cuttable_seams(node.op)
    assert [set(arm) for arm in arms] == [{seam.spelling} for seam in seams]


# ---- the output-owning cut -------------------------------------------------------------------- #


def _mimo_case(case: str) -> tuple[Graph, object]:
    """A corpus case's multi-output target as a graph node carrying ALL of its output ports.

    ``_case_match`` keeps one port, which is enough for a single-output target; an output-owning
    cut hands ports between pieces, so this one has to declare them."""
    tile = case_target_tile(case)
    graph = Graph()
    for name, tensor in tile.inputs.items():
        graph.add_node(InputOp(), [], tensor, node_id=name)
    others = [name for name in tile.outputs if name != tile.name]
    graph.add_node(
        tile,
        list(tile.inputs),
        outputs=(tile.outputs[tile.name], *(tile.outputs[name] for name in others)),
        node_id=tile.name,
    )
    return graph, graph.nodes[tile.name]


def _realized(case: str, spelling: str) -> Graph:
    graph, node = _mimo_case(case)
    match = Match(graph=graph, root_node_id=node.id, rule=Rule(name="test", pattern=[]))
    renamed = output_map(node)
    match.output = renamed
    seam = next(seam for seam in cuttable_seams(node.op) if seam.spelling == spelling)
    return realize(match, node, (seam,))


def test_disjoint_output_sweeps_offer_an_output_owning_seam() -> None:
    """The NVFP4 encode writes packed codes over the feature axis and one block scale per 16 of
    them. No axis rides both stores, so the fused kernel promotes nothing and its contractions get
    no output-axis pair. Each branch owns one store, and owning it is what gives the piece a grid."""
    tile = case_target_tile("fused/nvfp4-gate-up-requant-place-cut.json")
    owning = {seam.spelling: seam for seam in cuttable_seams(tile) if seam.owned is not None}

    assert set(owning) == {"PLACE@map.1/map", "PLACE@map.2/map"}
    assert [store.write.output for store in owning["PLACE@map.1/map"].owned[1]] == ["mul_static_fp4_bits"]
    assert [store.write.output for store in owning["PLACE@map.2/map"].owned[1]] == ["mul_static_fp4_scale_bits"]
    assert all(seam.dtypes == () for seam in owning.values()), "an output-owning seam writes no workspace to type"


def test_output_owning_cut_leaves_single_output_pieces_that_promote() -> None:
    """Realizing it gives two kernels, each writing ONE of the kernel's own outputs — no workspace
    between them — and each binding the sweep its store rides as a grid axis. Rank two is what the
    contraction sites need to name an ``(m, n)`` pair at all."""
    fragment = _realized("fused/nvfp4-gate-up-requant-place-cut.json", "PLACE@map.1/map")
    pieces = [node for node in fragment.nodes.values() if isinstance(node.op, TileOp)]

    assert len(pieces) == 2
    assert sorted(fragment.outputs) == ["mul_static_fp4_bits__placed", "mul_static_fp4_scale_bits__placed"]
    for piece in pieces:
        assert len(piece.op.output_specs) == 1
        assert len(piece.op.place.free) >= 2, f"{piece.id} kept a rank-1 placement"
        assert not any(spec.sweep for spec in piece.op.output_specs), f"{piece.id} still sweeps its store"
    widths = {tuple(sorted(axis.extent.as_static() for axis in piece.op.place.free)) for piece in pieces}
    assert widths == {(128, 256), (32, 128)}, "each piece binds its OWN store's width, not the other's"


def test_peeling_all_but_one_output_leaves_every_piece_single_output() -> None:
    """Three outputs on three widths: peeling TWO of them in one decision has to leave the sibling
    holding the third, not nothing. The serving post blocks that emit codes, block scales and a mean
    take this shape, and one peel there would still leave a two-output sibling."""
    m, k = Axis("m", 8), Axis("k", 16)
    widths = {"wide": Axis("n0", 16), "mid": Axis("n1", 8), "narrow": Axis("n2", 4)}
    edges = tuple(
        contraction(
            k,
            Load(name=f"a_{out}", input="a", index=(Var("m"), Var("k"))),
            (Load(name=f"b_{out}", input=f"{out}_w", index=(Var("k"), Var(axis.name))), out),
        )
        for out, axis in widths.items()
    )
    tile = TileOp(
        op=projection(edges, results=tuple(widths)),
        name="wide",
        place=Placement(free=(m,)),
        axes=(m, k, *widths.values()),
        output_specs=tuple(
            OutputSpec(Write(output=out, index=(Var("m"), Var(axis.name)), value=out), sweep=(axis,)) for out, axis in widths.items()
        ),
    )
    graph = Graph()
    _input(graph, "a", (8, 16))
    for out, axis in widths.items():
        _input(graph, f"{out}_w", (16, axis.extent.as_static()))
    graph.add_node(
        tile,
        ["a", *(f"{out}_w" for out in widths)],
        outputs=tuple(Tensor(out, (8, axis.extent.as_static()), "f16") for out, axis in widths.items()),
        node_id="wide",
    )
    root = graph.nodes["wide"]
    owning = sorted(seam.spelling for seam in cuttable_seams(root.op) if seam.owned is not None)
    assert len(owning) == 3, "each branch solely produces one output, so each is an output-owning seam"

    match = Match(graph=graph, root_node_id=root.id, rule=Rule(name="test", pattern=[]))
    renamed = output_map(root)
    match.output = renamed
    seams = {seam.spelling: seam for seam in cuttable_seams(root.op)}
    fragment = realize(match, root, tuple(seams[spelling] for spelling in owning[:2]))
    pieces = {piece.id: piece.op for piece in fragment.nodes.values() if isinstance(piece.op, TileOp)}

    assert len(pieces) == 3, "two peeled pieces and the sibling that keeps the third output"
    assert all(len(op.output_specs) == 1 for op in pieces.values())
    assert {op.place.free[-1].extent.as_static() for op in pieces.values()} == {16, 8, 4}, "each piece binds its own width"


def _epilogue_kernel(shared: bool) -> tuple[Graph, object]:
    """Two contractions over disjoint output widths under ONE projection body.

    Both NVFP4 encode shapes join their branches with an empty root body, so the ownership
    partition there is a plain operand split. This shape gives the root a body to divide: an
    epilogue statement per store when ``shared`` is false, and one statement both stores read when
    it is true — the case the cover-and-disjointness rule has to refuse, because neither piece can
    take it without the other losing it."""
    m, wide, narrow, k = Axis("m", 8), Axis("n", 16), Axis("n2", 4), Axis("k", 8)
    first = contraction(
        k, Load(name="a_v", input="a", index=(Var("m"), Var("k"))), (Load(name="b_v", input="b", index=(Var("k"), Var("n"))), "first")
    )
    second = contraction(
        k, Load(name="c_v", input="c", index=(Var("m"), Var("k"))), (Load(name="d_v", input="d", index=(Var("k"), Var("n2"))), "second")
    )
    body = [
        Assign(name="wide_out", op=ElementwiseImpl("negative"), args=("first",)),
        Assign(name="narrow_out", op=ElementwiseImpl("negative"), args=("second" if not shared else "first",)),
    ]
    tile = TileOp(
        op=projection((first, second), body=body, results=("wide_out", "narrow_out")),
        name="wide",
        place=Placement(free=(m,)),
        axes=(m, wide, narrow, k),
        output_specs=(
            OutputSpec(Write(output="wide", index=(Var("m"), Var("n")), value="wide_out"), sweep=(wide,)),
            OutputSpec(Write(output="narrow", index=(Var("m"), Var("n2")), value="narrow_out"), sweep=(narrow,)),
        ),
    )
    graph = Graph()
    for name in ("a", "c"):
        _input(graph, name, (8, 8))
    _input(graph, "b", (8, 16))
    _input(graph, "d", (8, 4))
    graph.add_node(
        tile,
        ["a", "b", "c", "d"],
        outputs=(Tensor("wide", (8, 16), "f16"), Tensor("narrow", (8, 4), "f16")),
        node_id="wide",
    )
    return graph, graph.nodes["wide"]


def test_an_output_owning_piece_takes_the_epilogue_statements_its_store_reads() -> None:
    """A root body divides with the outputs: each piece keeps the statements only its own store
    reads, so the term it becomes still defines the value it writes."""
    _, node = _epilogue_kernel(shared=False)
    owning = {seam.spelling: seam for seam in cuttable_seams(node.op) if seam.owned is not None}

    assert set(owning) == {"PLACE@map.1/inner", "PLACE@map.2/inner"}
    tail, stores = owning["PLACE@map.1/inner"].owned
    assert [store.write.output for store in stores] == ["wide"]
    assert [stmt.defines() for stmt in tail] == [("wide_out",)], "the piece takes its OWN epilogue, not the sibling's"


def test_an_output_owning_piece_carries_its_epilogue_into_a_lowerable_kernel() -> None:
    """Realizing it: two single-output kernels, each rank two, each still computing its epilogue."""
    graph, node = _epilogue_kernel(shared=False)
    match = Match(graph=graph, root_node_id=node.id, rule=Rule(name="test", pattern=[]))
    renamed = output_map(node)
    match.output = renamed
    seam = next(seam for seam in cuttable_seams(node.op) if seam.spelling == "PLACE@map.1/inner")
    fragment = realize(match, node, (seam,))
    pieces = {piece.id: piece.op for piece in fragment.nodes.values() if isinstance(piece.op, TileOp)}

    assert sorted(pieces) == ["narrow__placed", "wide__placed"]
    # Re-formed as its own kernel, a piece spells its axes canonically; the grid is read by extent.
    assert [axis.extent.as_static() for axis in pieces["wide__placed"].place.free] == [8, 16]
    assert [axis.extent.as_static() for axis in pieces["narrow__placed"].place.free] == [8, 4]
    for name, piece in pieces.items():
        stored = {value for spec in piece.output_specs for value in spec.write.values}
        defined = {name for stmt in piece.op.lower(axes=piece.axes) for name in stmt.defines()}
        assert stored <= defined, f"{name} stores a value its term never defines: {sorted(stored - defined)}"


def test_a_shared_epilogue_statement_refuses_the_output_owning_cut() -> None:
    """One statement both stores read belongs to no single piece, so the partition refuses and every
    seam keeps its workspace reading — the pieces are never a partition of the kernel."""
    _, node = _epilogue_kernel(shared=True)

    assert all(seam.owned is None for seam in cuttable_seams(node.op))


def test_an_output_owning_cut_is_offered_without_a_grid_gain() -> None:
    """Independent outputs can be split again even when both pieces use the same grid."""
    tile = case_target_tile("fused/nvfp4-quantize-cut-shared-normalizer.json")

    assert len(tile.output_specs) == 2
    assert len([seam for seam in cuttable_seams(tile) if seam.owned]) == 2


# ---- the full-projection cut --------------------------------------------------------------------- #

#: The serving W4A4 MLP shape, whose requant projection owns more outputs than the binder can bind.
_REQUANT = "fused/nvfp4-gate-up-requant-place-cut.json"


def _cut_arms(graph: Graph, node) -> list:
    """The unpinned placement fork's cut arms on ``node``, each paired with its knob row."""
    match = Match(graph=graph, root_node_id=node.id, rule=Rule(name="test", pattern=[]))
    return [(option, dict(option.knobs)) for option in _CUT.rewrite(match, node) if "cut" in option.knobs.values()]


def _composed_arm(graph: Graph, node):
    """The one arm that cuts the FULL projection: its knob row marks every owning seam ``cut``. A
    clustered sibling seam (two alpha-equivalent operand cones, one decision) also spells several
    seams, and is not this arm."""
    owning = {seam.spelling for seam in cuttable_seams(node.op) if seam.owned is not None}
    composed = [(option, knobs) for option, knobs in _cut_arms(graph, node) if len(knobs) > 1 and owning <= set(knobs)]
    assert len(composed) == 1, f"expected one full-projection arm, got {[sorted(knobs) for _, knobs in composed]}"
    return composed[0]


def _contraction_spellings(tile: TileOp) -> list[str]:
    """Every contraction occurrence of this kernel, spelled the way a placement key names it. A
    contraction standing at the kernel's own root is no PLACE site and reads as ``ROOT``."""
    from emmy.compiler.ir.tile.path import sites, spell  # noqa: PLC0415

    all_sites = sites(tile.op)
    return [
        spell(tile.op, "PLACE", site.node, all_sites=all_sites) if site.hops else "ROOT"
        for site in all_sites
        if site.node.as_contraction() is not None
    ]


def _piece_ops(fragment: Graph) -> list[TileOp]:
    return [node.op for node in fragment.nodes.values() if isinstance(node.op, TileOp)]


def _pinned_requant_cut(pins: dict[str, str], *, allow_unpinned: bool = False):
    graph, root = _mimo_case(_REQUANT)
    graph.inputs, graph.outputs = list(root.inputs), list(root.buffer_names())

    def decide(fork):
        if _structural_domain(fork.options) == ("PLACE",) and not allow_unpinned:
            assert len(fork.options) == 1, f"{fork.node_id} still offers an unpinned placement choice"
        return fork.options[0]

    with pinned_knobs(pins), tracking_place_keys() as resolved:
        result, trace = Run(Pipeline.build(["tile/cut"]), _CTX).resolve(graph, decide)
        unmatched = unmatched_place_pins(resolved)
    pieces = [node.op for node in result.nodes.values() if isinstance(node.op, TileOp)]
    return pieces, trace, unmatched


def test_parent_and_child_site_pins_cut_only_the_named_piece() -> None:
    """A named child can cut one of its nested contractions after the parent cut; its sibling stays put."""
    parent, producer, token = _two_site_child()
    before, parent_trace, unmatched = _pinned_requant_cut(parent)
    assert _contraction_spellings(producer) == ["PLACE@map.1/reduce.1/inner", "PLACE@map.2/inner"]
    child = {f"PLACE@place_{token}/map.2/inner": "cut"}

    after, trace, unmatched_with_child = _pinned_requant_cut({**parent, **child})

    assert not unmatched and not unmatched_with_child
    assert len(before) == 2 and producer.placement_decided, "parent-only pins retain the nested producer"
    assert len(after) == 3 and all(piece.placement_decided for piece in after if not _contraction_spellings(piece))
    assert len(trace) < 20, "the two levels of cuts must reach a fixpoint"
    assert [decision.knob_delta for decision in trace if "cut" in decision.knob_delta.values()] == [
        parent,
        {"PLACE@map.2/inner": "cut"},
    ]
    assert len([decision for decision in parent_trace if "cut" in decision.knob_delta.values()]) == 1
    assert all(len(_contraction_spellings(piece)) <= 1 for piece in after)
    siblings = {piece.name: piece for piece in before if piece is not producer}
    assert {piece.name for piece in after if piece.name in siblings} == set(siblings)
    assert all(
        {spec.write.output for spec in next(piece for piece in after if piece.name == name).output_specs}
        == {spec.write.output for spec in sibling.output_specs}
        for name, sibling in siblings.items()
    ), "the sibling keeps the outputs it owned"


def _two_site_child() -> tuple[dict[str, str], TileOp, str]:
    """The parent cut that leaves one child with two contraction sites, the child and its placement token. The full
    projection cut shares one gate/up producer, so it leaves no such child; the single ``map.1/map`` seam does."""
    parent = {"PLACE@map.1/map": "cut"}
    before, _, _ = _pinned_requant_cut(parent)
    child = next(piece for piece in before if len(_contraction_spellings(piece)) > 1)
    return parent, child, child.name.rsplit("__place_", 1)[1]


def test_child_site_pins_cut_the_same_remainder_in_two_stages() -> None:
    # Peeling one output leaves its statistic and contraction available for successive child cuts.
    parent, child, token = _two_site_child()
    pins = {
        **parent,
        f"PLACE@place_{token}/map.1/reduce": "cut",
        f"PLACE@place_{token}/step.1/map.1/inner": "cut",
    }

    pieces, trace, unmatched = _pinned_requant_cut(pins)

    assert not unmatched
    assert len(trace) < 20
    assert len(pieces) == 4
    assert [decision.knob_delta for decision in trace if "cut" in decision.knob_delta.values()] == [
        parent,
        {"PLACE@map.1/reduce": "cut"},
        {"PLACE@map.1/inner": "cut"},
    ]
    assert next(piece for piece in pieces if piece.name == child.name).placement_step == 2
    assert all(piece.placement_decided for piece in pieces if cuttable_seams(piece))
    assert not any(piece.name == child.name and cuttable_seams(piece) for piece in pieces)


def test_child_pin_replays_after_an_unstaged_site_is_exposed() -> None:
    graph, root = _mimo_case(_REQUANT)
    graph.inputs, graph.outputs = list(root.inputs), list(root.buffer_names())
    root.op = replace(root.op, name=f"{root.op.name}__place_deadbeef00")
    pins = {
        "PLACE@place_deadbeef00/map.1/map": "cut",
        "PLACE@place_deadbeef00/map.1/reduce": "cut",
    }

    with pinned_knobs(pins), tracking_place_keys() as resolved:
        result, trace = Run(Pipeline.build(["tile/cut"]), _CTX).resolve(graph, lambda fork: fork.options[0])
        unmatched = unmatched_place_pins(resolved)

    pieces = _piece_ops(result)
    assert not unmatched
    assert len(pieces) == 3
    assert [decision.knob_delta for decision in trace if "cut" in decision.knob_delta.values()] == [
        {"PLACE@map.1/map": "cut"},
        {"PLACE@map.1/reduce": "cut"},
    ]


@requires_cuda
def test_staged_child_cut_preserves_both_requant_outputs() -> None:
    graph, root = _mimo_case(_REQUANT)
    graph.inputs, graph.outputs = list(root.inputs), list(root.buffer_names())
    parent, child, token = _two_site_child()
    first = f"PLACE@place_{token}/map.1/reduce"
    second = f"PLACE@place_{token}/step.1/map.1/inner"
    inputs = {
        name: np.full(tuple(dim.as_static() for dim in tensor.shape), 1, dtype=tensor.dtype.np) for name, tensor in root.op.inputs.items()
    }
    inputs["mul_static_fp4_shift"][:] = 0
    backend = CudaBackend()
    single = backend.run(_lower(graph.copy(), {**parent, first: "cut"}), input_data=inputs)[0].outputs
    staged = backend.run(_lower(graph.copy(), {**parent, first: "cut", second: "cut"}), input_data=inputs)[0].outputs

    for name in graph.outputs:
        np.testing.assert_array_equal(staged[name], single[name])


def test_unknown_later_child_pin_stays_unmatched_and_terminates() -> None:
    parent, child, token = _two_site_child()
    stale = f"PLACE@place_{token}/step.1/map.9/inner"

    pieces, trace, unmatched = _pinned_requant_cut({**parent, f"PLACE@place_{token}/map.1/reduce": "cut", stale: "cut"})

    assert unmatched == [stale]
    assert len(trace) < 20
    assert len(pieces) == 3


def test_staged_child_pin_cannot_alias_an_ordinary_pin() -> None:
    parent, child, token = _two_site_child()
    pins = {
        **parent,
        f"PLACE@place_{token}/map.1/reduce": "cut",
        f"PLACE@place_{token}/step.0/map.1/reduce": "cut",
    }

    with pytest.raises(ValueError, match="address the same site"):
        _pinned_requant_cut(pins)


def test_stale_child_site_pin_is_reported_unmatched() -> None:
    from emmy.compiler.pipeline.search.pins import unreproducible_pin_flag

    graph, root = _mimo_case(_REQUANT)
    _, parent = _composed_arm(graph, root)
    stale = "PLACE@place_deadbeef00/map.1/inner"

    pieces, trace, unmatched = _pinned_requant_cut({**parent, stale: "cut"})

    assert unmatched == [stale]
    assert unreproducible_pin_flag({stale: "cut"}, [{}], placement_knobs=[decision.knob_delta for decision in trace])
    assert len([decision for decision in trace if "cut" in decision.knob_delta.values()]) == 1
    assert len(pieces) == 3 and sum(bool(_contraction_spellings(piece)) for piece in pieces) == 1


def test_parent_place_pin_is_consumed_on_the_uncut_remainder() -> None:
    """A child pin opens its named piece, without cutting the parent's remaining matmul again."""
    from emmy.compiler.ir.tile.path import sites

    graph = _computed_operand_graph("a")
    root = graph.nodes["out"]
    second = contraction(
        root.op.axes[2],
        Load(name="b_x", input="computed", index=(Var("m"), Var("k"))),
        (Load(name="b_w", input="direct", index=(Var("k"), Var("n"))), "acc2"),
    )
    root.op = replace(
        root.op,
        op=projection((root.op.op, second), (Assign(name="sum", op="add", args=("acc", "acc2")),), ("sum",)),
    )
    pipeline = Pipeline.build(["tile/cut"])
    with pinned_knobs({"PLACE": "cut"}):
        before, _ = Run(pipeline, _CTX).resolve(graph.copy(), lambda fork: fork.options[0])
    child = next(piece for piece in _piece_ops(before) if "__place_" in piece.name)
    (seam,) = cuttable_seams(child)
    path = next(site.path for site in sites(child.op) if site.node is seam.node)
    token = child.name.rsplit("__place_", 1)[1]
    with pinned_knobs({"PLACE": "cut", f"PLACE@place_{token}/{path}": "cut"}):
        after, trace = Run(pipeline, _CTX).resolve(graph.copy(), lambda fork: fork.options[0])

    pieces = _piece_ops(after)
    assert len(pieces) == 3, "the parent remainder keeps its matmul and epilogue together"
    remainder = next(piece for piece in pieces if piece.name == root.op.name)
    assert remainder.placement_decided and len(_contraction_spellings(remainder)) == 1
    assert len([decision for decision in trace if "cut" in decision.knob_delta.values()]) == 2


def test_scoped_pins_cut_the_same_remainder_in_two_stages() -> None:
    pins = {"PLACE@map.1/map": "cut", "PLACE@step.1/map.1/reduce": "cut"}
    pieces, trace, unmatched = _pinned_requant_cut(pins)

    assert not unmatched
    assert len(trace) < 20
    assert len([decision for decision in trace if "cut" in decision.knob_delta.values()]) == 2
    assert len(pieces) == 3
    assert any(piece.name == "mul_static_fp4_scale_bits" for piece in pieces)


@requires_cuda
def test_two_staged_cuts_preserve_norm_residual_values() -> None:
    pins = {"PLACE@map.1/map.1/reduce.1/inner": "cut", "PLACE@step.1/map.1/map": "cut"}
    cut = _lower(_norm_residual_graph(1), pins)
    assert sum(isinstance(node.op, CudaOp) for node in cut.nodes.values()) == 3

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 64)).astype(np.float16)
    w = rng.standard_normal((64, 32)).astype(np.float16)
    got = CudaBackend().run(cut, input_data=dict(zip(cut.inputs, (x, w), strict=True)))[0].outputs[cut.outputs[0]]
    y = x.astype(np.float32) @ w.astype(np.float32)
    expected = y / np.sqrt((y * y).mean(-1, keepdims=True) + 1e-6) + y
    np.testing.assert_allclose(got, expected, rtol=2e-2, atol=2e-1)


@pytest.mark.parametrize(
    "pins",
    [
        {"PLACE@map.1/map": "cut", "PLACE@step.0/map.1/map": "cut"},
        {
            "PLACE@map.1/map": "cut",
            "PLACE@map.1/reduce": "cut",
            "PLACE@step.1/map.1/reduce": "cut",
        },
    ],
)
def test_staged_and_ordinary_pins_cannot_alias_one_site(pins: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="address the same site"):
        _pinned_requant_cut(pins)


def test_future_stage_without_a_preceding_cut_stays_unmatched() -> None:
    stale = "PLACE@step.9/map.1/reduce"
    pieces, trace, unmatched = _pinned_requant_cut({"PLACE@map.1/map": "cut", stale: "cut"})

    assert unmatched == [stale]
    assert len(trace) < 20
    assert len(pieces) == 2


def test_unknown_later_root_pin_stays_unmatched_and_terminates() -> None:
    stale = "PLACE@map.9/inner"
    pieces, trace, unmatched = _pinned_requant_cut({"PLACE@map.1/map": "cut", stale: "cut"}, allow_unpinned=True)

    assert unmatched == [stale]
    assert len(trace) < 20
    assert len(pieces) == 2


def test_a_projection_owning_more_than_it_binds_offers_one_full_projection_cut() -> None:
    """The requant projection partitions its outputs by ownership — the packed codes and the block
    scales each have one producing branch — but not by producing root: the code branch alone holds
    six contractions. The placement fork offers taking the whole projection apart as ONE decision,
    every contraction and every owned output its own kernel, spelled by seams it already offers one
    at a time."""
    graph, node = _mimo_case(_REQUANT)
    _, knobs = _composed_arm(graph, node)
    owning = [seam.spelling for seam in cuttable_seams(node.op) if seam.owned is not None]

    assert set(knobs.values()) == {"cut"}
    assert sorted(knobs) == sorted([*_contraction_spellings(node.op), *owning])


def test_the_full_projection_cut_leaves_one_contraction_per_piece_on_a_grid() -> None:
    """The cut fixpoint terminates after output cuts separate the fused producers again."""
    graph, node = _mimo_case(_REQUANT)
    graph.inputs, graph.outputs = list(node.inputs), list(node.buffer_names())
    _, knobs = _composed_arm(graph, node)
    parent = node.op.identity_key(with_io=True)
    decisions = 0

    def decide(fork):
        nonlocal decisions
        decisions += 1
        assert decisions < 30, "cut/fuse did not reach a fixpoint"
        owned = [seam for seam in cuttable_seams(fork.root_op) if seam.owned]
        row = knobs if fork.root_op.identity_key(with_io=True) == parent else {owned[0].spelling: "cut"} if owned else {}
        arm = spelled_arm(fork.options, row)
        assert arm is not None
        return arm[0]

    fragment, _ = Run(Pipeline.build(["tile/cut"]), _CTX).resolve(graph, decide)
    pieces = _piece_ops(fragment)

    assert len(pieces) == 3, "the grouped gate/up contraction needs one producer beside the two owned outputs"
    for piece in pieces:
        contractions = _contraction_spellings(piece)
        assert len(contractions) <= 1, f"{piece.name} still holds several contractions"
        # A pointwise piece forms with its free coordinates fused into one axis, as its own program does.
        assert not contractions or len(piece.place.free) >= 2, f"{piece.name} kept a one-axis placement"


def _one_root_kernel() -> tuple[Graph, object]:
    """Two owned outputs, one binder root — the serving post-attention shape in miniature.

    The wide branch multiplies a contraction by two folds, so it reads several reduces and the
    binder has no single node to build it around; ``kernel_roots`` therefore sees only the narrow
    branch's contraction and reports one. The outputs still partition by ownership, and the fused
    kernel still cannot tile the wide branch — which is why the offer asks ownership, not how many
    roots the binder found.

    The two folds are deliberately different. ``stat`` does not read the wide store's sweep: it is
    the branch's row statistic, evaluated once ahead of the sweep, the shape of the rms mean in the
    serving kernel. ``blockmax`` does read it, so each cell of the sweep folds its own — the shape
    of the NVFP4 encode's per-block maximum.
    """
    m, wide, narrow, k, r, q = Axis("m", 8), Axis("n", 16), Axis("n2", 4), Axis("k", 8), Axis("r", 4), Axis("q", 4)
    acc = contraction(
        k, Load(name="a_v", input="a", index=(Var("m"), Var("k"))), (Load(name="b_v", input="b", index=(Var("k"), Var("n"))), "acc")
    )
    stat = reduction(r, (slab("s_v", "s", "m", "r"),), (Assign(name="stat__v", op="copy", args=("s_v",)),), ("stat",))
    blockmax = reduction(
        q, (slab("t_v", "t", "m", "n", "q"),), (Assign(name="blockmax__v", op="copy", args=("t_v",)),), ("blockmax",), ops="maximum"
    )
    scaled = Assign(name="scaled", op=ElementwiseImpl("multiply"), args=("acc", "stat"))
    wide_out = Assign(name="wide_out", op=ElementwiseImpl("multiply"), args=("scaled", "blockmax"))
    first = projection((acc, stat, blockmax), (scaled, wide_out), ("wide_out",))
    second = contraction(
        k, Load(name="c_v", input="c", index=(Var("m"), Var("k"))), (Load(name="d_v", input="d", index=(Var("k"), Var("n2"))), "narrow_out")
    )
    tile = TileOp(
        op=projection((first, second), results=("wide_out", "narrow_out")),
        name="wide",
        place=Placement(free=(m,)),
        axes=(m, wide, narrow, k, r, q),
        output_specs=(
            OutputSpec(Write(output="wide", index=(Var("m"), Var("n")), value="wide_out"), sweep=(wide,)),
            OutputSpec(Write(output="narrow", index=(Var("m"), Var("n2")), value="narrow_out"), sweep=(narrow,)),
        ),
    )
    graph = Graph()
    for name in ("a", "c"):
        _input(graph, name, (8, 8))
    _input(graph, "b", (8, 16))
    _input(graph, "d", (8, 4))
    _input(graph, "s", (8, 4))
    _input(graph, "t", (8, 16, 4))
    graph.add_node(
        tile,
        ["a", "b", "c", "d", "s", "t"],
        outputs=(Tensor("wide", (8, 16), "f16"), Tensor("narrow", (8, 4), "f16")),
        node_id="wide",
    )
    return graph, graph.nodes["wide"]


def test_the_offer_reads_ownership_not_how_many_roots_the_binder_found() -> None:
    """A projection can own two outputs it cannot bind while ``kernel_roots`` reports ONE: a branch
    reading several reduces is no root of its own, so counting roots misses the kernel entirely. The
    serving post-attention output+norm+requant kernel is that shape, and it is the one this cut
    exists for. Asking ownership finds it."""
    from emmy.compiler.ir.tile.ops import kernel_roots, owns_outputs_it_cannot_bind, refused_roots  # noqa: PLC0415

    graph, node = _one_root_kernel()
    tile = node.op

    assert len(kernel_roots(tile.op)) == 1 and refused_roots(tile.op, tile.output_specs) == ()
    assert owns_outputs_it_cannot_bind(tile.op, tile.output_specs)
    assert len(_composed_arm(graph, node)[1]) > 1


def test_the_cut_takes_a_row_statistic_but_leaves_a_per_cell_fold() -> None:
    """Which folds the cut hands away, and why each answer is the one the piece needs.

    A reduce evaluated once ahead of a branch's output sweep keeps that branch's piece at one grid
    axis — the sweep-promotion rule will not replicate a row statistic per cell, and it is right not
    to — so the cut hands it its own kernel and the piece reads the one value back. A reduce each
    cell of the sweep folds for itself replicates nothing, so it stays where it is and the piece
    binds its sweep around it.
    """
    graph, node = _one_root_kernel()
    seams = {seam.node.axis: seam.spelling for seam in cuttable_seams(node.op) if seam.node.axis in ("r", "q")}

    _, knobs = _composed_arm(graph, node)
    assert seams["r"] in knobs, "the row statistic is part of the decision, not left behind"
    assert seams["q"] not in knobs, "the per-cell fold is the piece's own work"

    owning = _composed_arm(graph, node)[0].materialize().nodes["wide__placed"].op
    assert [axis.extent.as_static() for axis in owning.place.free] == [8, 16], "the piece binds its store's sweep around what it kept"


def test_restamp_replays_nested_cuts_when_the_fresh_child_identity_changes() -> None:
    """A fresh lowering re-keys the pieces without changing their structural cut paths."""
    document, route = _routed(
        inventory_document(_norm_residual_graph(16)),
        {"PLACE@map.1/map": "cut", "PLACE@map.1/map.1/reduce.1/inner": "cut"},
    )
    assert route is not None
    # The parent route alone keeps the norm statistic fused. A route on that fresh piece
    # takes its cut; unrelated pieces remain whole under the replay's ordinary decisions.
    unchanged, report = restamp(document)
    assert unchanged == document and not report.changed
    child = next(document.kernel(ref) for ref in route.children if cuttable_seams(document.kernel(ref).op()))
    document, nested = _routed(document, {"PLACE": "cut"}, target=child)
    assert nested is not None
    unchanged, report = restamp(document)
    assert unchanged == document and not report.changed

    changed = replace(document, programs=inventory_document(_norm_residual_graph(32)).programs)
    fresh, report = restamp(changed)

    assert len(fresh.routing) == 2 and not report.dropped_routes
    assert report.rekeyed and report.demoted
    assert all(row.measurements is None for row in fresh.rows)


def test_a_recorded_route_selects_the_arm_spelling_its_whole_cut_set() -> None:
    """A composed arm and the single-seam arms it composes all carry keys a composed row marks, so
    only the cut-key SET tells them apart: a row spelling the whole set selects the composed arm, and
    a row spelling one seam still selects that seam's own arm."""
    from emmy.compiler.pipeline.search.pins import spelled_arm  # noqa: PLC0415

    graph, node = _mimo_case(_REQUANT)
    match = Match(graph=graph, root_node_id=node.id, rule=Rule(name="test", pattern=[]))
    options = _CUT.rewrite(match, node)
    _, whole = _composed_arm(graph, node)

    composed = spelled_arm(options, dict.fromkeys(whole, "cut"))
    assert composed is not None and sorted(composed[1]) == sorted(whole)

    one = next(iter(sorted(whole)))
    single = spelled_arm(options, {one: "cut"})
    assert single is not None and list(single[1]) == [one]


@pytest.mark.parametrize("computed_scale", [False, True])
def test_storage_frontier_recomputes_the_encode_scale_in_the_consumer(computed_scale):
    """A scale computed before the encode can also feed the decode without fusing the encode."""
    from emmy.compiler.dtype import F8E4M3, F32

    m, n, k = Axis("m", 2), Axis("n", 3), Axis("k", 16)
    scale = Load(name="scale_value", input="scale", index=(Var("m"), Var("k") / 8), dtype=F32)
    operands = ()
    if computed_scale:
        scale = reduction(
            "r",
            (Load(name="sample", input="scale", index=(Var("m"), Var("k") / 8, Var("r")), dtype=F32),),
            (Assign(name="scale_value__v", op="copy", args=("sample",)),),
            ("scale_value",),
        )
        operands = (scale,)
    quantized = projection(
        operands,
        (
            Load(name="x_value", input="x", index=(Var("m"), Var("k")), dtype=F16),
            *((scale,) if not computed_scale else ()),
            Assign(name="scale_squared", op="multiply", args=("scale_value", "scale_value"), dtype=F32),
            Assign(name="scaled", op="divide", args=("x_value", "scale_squared"), dtype=F32),
            Assign(name="encoded", op="to_f8e4m3", args=("scaled",), dtype=F8E4M3),
            Assign(name="decoded", op="from_f8e4m3", args=("encoded",), dtype=F16),
            Assign(name="quantized", op="multiply", args=("decoded", "scale_squared"), dtype=F16),
        ),
    )
    tile = TileOp(
        op=contraction(k, quantized, (Load(name="weight", input="w", index=(Var("k"), Var("n"))), "acc")),
        place=Placement(free=(m, n)),
        axes=(m, n, k, Axis("r", 3)),
        output_specs=(OutputSpec(Write(output="out", index=(Var("m"), Var("n")), value="acc")),),
    )
    graph = Graph()
    for name, shape in (("x", (2, 16)), ("scale", (2, 2, 3) if computed_scale else (2, 2)), ("w", (16, 3))):
        _input(graph, name, shape, "f32")
    graph.add_node(tile, ["x", "scale", "w"], Tensor("out", (2, 3)), node_id="out")
    graph.inputs, graph.outputs = ["x", "scale", "w"], ["out"]
    graph.nodes["out"].op = tile.with_io(graph, graph.nodes["out"])
    match = Match(graph=graph, root_node_id="out", rule=Rule(name="test", pattern=[]))
    seam = next(seam for seam in cuttable_seams(match.root.op) if seam.node.exposes == ("quantized",))
    assert seam.frontier is not None
    assert seam.dtypes == (F8E4M3,)
    assert any("scale_squared" in stmt.defines() for stmt in seam.frontier.residue)
    assert not any("encoded" in stmt.defines() for stmt in seam.frontier.residue)
    cuts = tuple(s for s in cuttable_seams(match.root.op) if s is seam or s.spelling == seam.spelling or s.node.axis == "r")
    fragment = realize(match, match.root, cuts, placement_decided=True)
    fragment.inputs = list(graph.inputs)
    pieces = [node for node in fragment.nodes.values() if isinstance(node.op, TileOp)]
    assert len(pieces) == (3 if computed_scale else 2)
    assert sum(tensor.dtype == F8E4M3 for node in pieces for tensor in node.outputs) == 1
    if computed_scale:
        assert sum("scale" in node.inputs for node in pieces) == 1, "both encode and decode reuse the separately computed scale"
    for node in pieces:
        node.op.op.lower(bound=frozenset(), stores=node.op.output_specs, axes=node.op.axes)


def test_a_bare_cut_is_one_decision_at_the_root_most_seam(monkeypatch) -> None:
    """A bare ``PLACE=cut`` spells one cut — the root-most seam of the kernel it is recorded on — and is spent by it:
    the pieces are read against their own rows, so the mint takes the one decision the recording took, not a cut at
    every piece that offers a seam."""
    monkeypatch.setenv("EMMY_FAST_MATH", "0")
    document, route = _routed(_sdpa_document(_ROUTING_CARD), {"PLACE": "cut"})
    assert route is not None and len(document.routing) == 1 and len(route.children) >= 2


def _row_statistic_graph() -> Graph:
    """``out[m, n] = f(sum_k x[m, k], w[n])`` — a cut piece whose cone holds a ROW statistic the
    output sweep is invariant in, which is the shape every norm-plus-rotation operand cone has."""
    m, n, k = Axis("m", 8), Axis("n", 16), Axis("k", 32)
    statistic = reduction(k, (slab("x", "x", "m", "k"),), (Assign(name="acc__v", op="multiply", args=("x", "x")),), ("acc",))
    cone = projection(
        (statistic, slab("w", "w", "n")),
        (Assign(name="scaled", op="multiply", args=("acc", "w")),),
    )
    tile = TileOp(
        op=projection((cone,), (Assign(name="out_v", op="multiply", args=("scaled", "scaled")),)),
        name="out",
        place=Placement(free=(m, n)),
        axes=(m, n, k),
    )
    graph = Graph()
    _input(graph, "x", (8, 32))
    _input(graph, "w", (16,))
    graph.add_node(tile, ["x", "w"], Tensor("out", (8, 16), "f16"), node_id="out")
    graph.inputs, graph.outputs = ["x", "w"], ["out"]
    return graph


def test_cut_piece_sweeps_the_axis_its_row_statistic_is_invariant_in() -> None:
    """The piece's grid binds ``m`` and sweeps ``n``: binding ``n`` too would re-fold the statistic
    once per output cell, which is what a materialized q/k RoPE cone did — one cooperative block
    per element."""
    graph = _row_statistic_graph()
    pipeline = Pipeline.build(["tile/cut"])
    match = pipeline.match(graph, pipeline.passes[0].rules[0])[0]
    seams = cuttable_seams(match.root.op)

    fragment = realize(match, match.root, (seams[0],))

    producer = next(node.op for name, node in fragment.nodes.items() if isinstance(node.op, TileOp) and "__place_" in name)
    assert [axis.name for axis in producer.place.free] == ["m"]
    assert [axis.name for store in producer.output_specs for axis in store.sweep] == ["n"]


def _mlp_graph() -> Graph:
    """``exp(down(silu(x @ wg) * (x @ wu)))`` at one token: cut at the gate/up twin and at the down
    projection, the down piece's A operand is the computed SiLU product and its weight is laid out
    ``[k, n]``."""
    from emmy.commands.trace import graph_from_code

    code = (
        "(lambda x, wg, wu, wd: torch.matmul(torch.nn.functional.silu(torch.matmul(x, wg)) * torch.matmul(x, wu), wd).exp())"
        "(torch.randn(1, 64, dtype=torch.float16), torch.randn(64, 256, dtype=torch.float16),"
        " torch.randn(64, 256, dtype=torch.float16), torch.randn(256, 128, dtype=torch.float16))"
    )
    return graph_from_code(code)[0]


def _mlp_cuts() -> dict[str, str]:
    """Both contraction seams cut: the gate/up twin and the down projection."""
    return {seam.spelling: "cut" for seam in cuttable_seams(_lifted_parent(_mlp_graph())) if seam.node.as_contraction() is not None}


def _mlp_down() -> TileOp:
    with pinned_knobs(_mlp_cuts()):
        lowered = Pipeline.build(LOOP_PASSES).run(_mlp_graph(), ctx=_CTX)
        cut, _ = Run(pipeline=Pipeline.build(["tile/lift", "tile/cut"]), ctx=_CTX).resolve(lowered, lambda fork: fork.options[0])
    (down,) = [
        node.op
        for node in cut.nodes.values()
        if isinstance(node.op, TileOp) and "__place_" in node.op.name and any(t.shape[0] == 256 for t in node.op.inputs.values())
    ]
    return down


def test_a_computed_input_is_a_formed_gemv_pieces_a_operand() -> None:
    """A GEMV with a computed input and a ``[k, n]`` weight orients, fused or formed, so the
    computed operand is A. With the weight as A no fragment loader reads its k column, and the SiLU
    down projection lost every tensor-core tier (the RTX 5090 s1 layer: 48 -> 65 us with the prologue
    fused; 41 us once it is A)."""
    from emmy.compiler.ir.schedule.classic.refusals import _warp_atoms
    from emmy.compiler.ir.tile.path import sites

    down = _mlp_down()
    (site,) = [site for site in sites(down.op) if site.node.as_contraction() is not None]
    assert site.node.operands[0].as_slab() is None, "the SiLU product is A"
    assert _warp_atoms(down, _CTX, site.node), "the tensor-core tier is offered"


def test_split_projection_sibling_keeps_its_child_place_address() -> None:
    """An unsplit projection sibling keeps a distinct PLACE scope when its twin splits K."""
    graph = _mimo_graph()
    with pinned_knobs({"REDUCE": "g2k"}):
        result, _ = Run(Pipeline.build(["tile/cut"]), _CTX).resolve(graph, lambda fork: fork.options[0])

    sibling = result.producer("out1").op
    assert isinstance(sibling, TileOp)
    assert sibling.name.startswith("out0__place_")
    assert sibling.name != result.producer("out0").op.name
    token = sibling.name.rsplit("__place_", 1)[1]
    with pinned_knobs({f"PLACE@place_{token}/inner": "cut"}):
        scoped, sources = _CUT._placement_pins(sibling)
    assert scoped == (("PLACE@inner", "cut"),)
    assert sources == {"PLACE@inner": f"PLACE@place_{token}/inner"}


def test_a_split_keeps_the_name_of_the_piece_it_splits() -> None:
    """A split piece's partial and finalize launch under the piece's own name, so a kernel pin naming
    the piece (its ordinal included) names both halves; they used to take the name of the workspace
    buffer, whose ordinal counts components, not pieces."""
    import re

    down = _mlp_down()
    token = re.search(r"__place_(\w+)$", down.name).group(1)
    from emmy.compiler.pipeline.fork import iter_leaves

    with pinned_knobs({**_mlp_cuts(), f"REDUCE@place_{token}": "g4k"}):
        lowered, _ = Run(Pipeline.build(CUDA_PASSES), _CTX).resolve(_mlp_graph(), lambda fork: next(iter_leaves(fork.options)))
    names = {node.op.kernel_name for node in lowered.nodes.values() if type(node.op).__name__ == "CudaOp"}
    assert {down.name, f"{down.name}__partial"} <= names, names


def test_a_kernel_pin_that_leaves_no_row_is_refused_by_name() -> None:
    """A kernel-scoped pin the named piece cannot take fails the compile with the pins that did it,
    instead of leaving the piece unscheduled and the pin realized by nothing."""
    import re

    token = re.search(r"__place_([0-9a-f]+)", _mlp_down().name).group(1)
    with pytest.raises(ValueError, match="leave no schedule row"):
        _lower(_mlp_graph(), {**_mlp_cuts(), f"TILE@place_{token}": "mma_m16n8k16_f16_f32/f64x64"})


@pytest.mark.parametrize("rows", [1, 2])
def test_computed_operand_cut_keeps_bf16_before_fp8_output(rows) -> None:
    """A later FP8 encode cannot change the precision of an earlier contraction operand."""
    from emmy.compiler.backend.cuda.nvcc import compile_to_cubin, nvcc_path
    from emmy.compiler.dtype import BF16, F8E4M3

    m, n, k = Axis("m", rows), Axis("n", 4), Axis("k", 8)
    operand = projection(
        (),
        (
            Load(name="xv", input="x", index=(Var("m"), Var("k"))),
            Assign(name="squared", op="multiply", args=("xv", "xv")),
            Assign(name="rounded", op="copy", args=("squared",), dtype=BF16),
        ),
        ("rounded",),
    )
    product = contraction(k, operand, (Load(name="wv", input="w", index=(Var("k"), Var("n"))), "acc"))
    tile = TileOp(
        op=projection((product,), (Assign(name="encoded", op="to_f8e4m3", args=("acc",), dtype=F8E4M3),), ("encoded",)),
        name="out",
        place=Placement(free=(m, n)),
        axes=(m, n, k),
        output_specs=(OutputSpec(Write(output="out", index=(Var("m"), Var("n")), value="encoded")),),
    )
    graph = Graph()
    _input(graph, "x", (rows, 8), dtype="bf16")
    _input(graph, "w", (8, 4), dtype="bf16")
    graph.add_node(tile, ["x", "w"], Tensor("out", (rows, 4), "f8e4m3"))
    graph.inputs, graph.outputs = ["x", "w"], ["out"]
    tile = tile.with_io(graph, graph.nodes["out"])
    seam = next(seam for seam in cuttable_seams(tile) if seam.node.exposes == ("rounded",))
    assert seam.dtypes == (BF16,)
    lowered = _lower_cut(graph, seam.spelling)
    producer = next(node for node in lowered.nodes.values() if isinstance(node.op, CudaOp) and "__place_" in node.id)
    assert producer.output.dtype == BF16
    assert lowered.buffer("out").dtype == F8E4M3
    if nvcc_path() is None:
        pytest.skip("nvcc unavailable")
    for node in lowered.nodes.values():
        if isinstance(node.op, CudaOp):
            assert compile_to_cubin(node.op.kernel_source, node.op.kernel_name, arch="sm_120a").exists()


def test_parallel_split_preserves_every_output_buffer() -> None:
    """Worker-built split arms redirect the secondary port before a later cut reads it."""
    from emmy.compiler.pipeline.fork import parallel_expand

    n, k = Axis("n", 4), Axis("k", 256)
    tile = TileOp(
        op=contraction(
            k,
            Load(name="xv", input="x", index=(Var("k"),)),
            (Load(name="av", input="a", index=(Var("k"), Var("n"))), "first"),
            (Load(name="bv", input="b", index=(Var("k"), Var("n"))), "second"),
        ),
        name="out0",
        place=Placement(free=(n,)),
        axes=(n, k),
        output_specs=tuple(
            OutputSpec(Write(output=name, index=(Var("n"),), value=value)) for name, value in (("out0", "first"), ("out1", "second"))
        ),
        placement_decided=True,
    )
    graph = Graph()
    _input(graph, "x", (256,))
    for name in ("a", "b"):
        _input(graph, name, (256, 4))
    graph.add_node(tile, ["x", "a", "b"], outputs=(Tensor("out0", (4,), "f16"), Tensor("out1", (4,), "f16")))
    graph.add_node(ElementwiseOp("negative"), ["out1"], Tensor("consumer", (4,), "f16"))
    graph.inputs, graph.outputs = ["x", "a", "b"], ["out0", "consumer"]

    def choose(point):
        point.match.graph.validate()
        arms = [option for option in point.options if _is_structural_option(option)]
        if point.node_id == "out0":
            assert len(arms) > 1
            parallel_expand(arms, workers=2)
            return next(arm for arm in arms if "g2k" in arm.knobs.values())
        return point.options[0]

    result, _ = Run(Pipeline.build(["tile/cut"]), _CTX).resolve(graph, choose)
    result.validate()
    assert result.outputs == ["out0", "consumer"]
    assert result.nodes["consumer"].inputs == ["out1"]
    assert result.buffer("out1") is not None
