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
    _producer_order,
    _workspace_axes,
    cuttable_seams,
    output_map,
    realize,
)
from emmy.compiler.pipeline.pipeline import RuleSkipped, Run, _is_structural_option
from emmy.compiler.pipeline.search.golden import GoldenFile, GoldenRecord, Measurements, decode_record, kernel_identity
from emmy.compiler.pipeline.search.golden.decode import _replay
from emmy.compiler.pipeline.search.golden.record import _lifted_target, _target_kernel_nodes
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import case_target_tile, direct_classic_leaf, loop_record_fields, loop_target, requires_cuda
from tests.compiler.terms import contraction, projection, reduction, slab

_CTX = Context.from_target((12, 0))
_CUT = import_module("emmy.compiler.pipeline.passes.tile.cut.030_cut")


def _input(graph: Graph, name: str, shape, dtype="f16") -> None:
    graph.add_node(InputOp(), [], Tensor(name, shape, dtype), node_id=name)


def test_cut_and_schedule_passes_share_the_generic_schedule_driver() -> None:
    from emmy.compiler.ir.schedule import schedule

    assert _CUT.schedule is schedule
    assert import_module("emmy.compiler.pipeline.fork").schedule is schedule


def test_placement_cut_preserves_a_cross_cta_split_receipt() -> None:
    """A split piece can re-enter placement; cutting it must not make REDUCE pending again."""
    from emmy.compiler.pipeline.passes.tile._split import split_pending

    graph = _computed_operand_graph("a")
    tile = graph.nodes["out"].op
    # The partition receipt is the reduce axis's window in the kernel's axis table; the term names it only.
    axes = tuple(replace(axis, window=Window(parent=axis, partition=True)) if axis.name == tile.op.axis else axis for axis in tile.axes)
    graph.nodes["out"].op = replace(tile, axes=axes)
    pipeline = Pipeline.build(["tile/cut"])
    match = pipeline.match(graph, pipeline.passes[0].rules[0])[0]
    seams = cuttable_seams(match.root.op)

    fragment = realize(match, match.root, (seams[0],))

    pieces = [node.op for node in fragment.nodes.values() if isinstance(node.op, TileOp)]
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
    match, graph = _case_match("attention/rmsnorm-gqa-b-cut.yaml")
    with pinned_knobs(pins):
        result = _CUT.rewrite(match, graph.nodes[match.root_node_id])
    options = result if isinstance(result, list) else [result]
    cut = next(option for option in options if "cut" in option.knobs.values())
    return cut.expand()[0]


def _piece_with_seam(fragment: Graph):
    return next(node for node in fragment.nodes.values() if isinstance(node.op, TileOp) and cuttable_seams(node.op))


def test_a_pipeline_that_stops_at_the_cut_pass_keeps_the_fused_tree_and_schedules_nothing(monkeypatch) -> None:
    """``compile --passes dolfnstp``: a kernel-set arm is priced by scheduling its pieces, so a greedy
    compile that never reaches ``tile/schedule`` must not price one. The offered state cut resolves to
    the fused tree (pins alone could pick the cut), and no kernel comes out scheduled."""
    from emmy.compiler.pipeline.search.db import SearchDB
    from emmy.compiler.pipeline.search.policy import greedy as policy

    def no_price(*_args, **_kwargs):
        raise AssertionError("a pipeline without tile/schedule must not price an arm by scheduling its pieces")

    monkeypatch.setattr(policy, "_price_kernel", no_price)
    assert any(value == "cut" for offer in _offered(_softmax_graph(), frontend=True) for value in offer.values())
    result = Pipeline.build([*LOOP_PASSES, "tile/lift", "tile/cut"]).run(_softmax_graph(), ctx=_CTX, db=SearchDB())
    kernels = [node.op for node in result.nodes.values() if isinstance(node.op, TileOp)]
    assert len(kernels) == 1
    assert kernels[0].schedule is None


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


def test_composed_cut_topologically_orders_equal_degree_workspace_chain() -> None:
    """Counting direct workspace reads cannot order A->C->B when A and C each read one."""

    def piece(name: str, source: str | None):
        produced = projection((), (Load(name=f"{name}_value", input=source or "input", index=()),), results=(f"{name}_value",))
        return (None, produced, (), (), name, (f"{name}_value",), (name,))

    pieces = [piece("a", "c"), piece("c", "b"), piece("b", None)]

    assert [buffers[0] for *_, buffers in _producer_order(pieces)] == ["b", "c", "a"]


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


def test_recorded_sdpa_cut_decodes_exactly_and_stale_path_fails_loudly() -> None:
    wire = _sdpa_graph().to_wire()
    fields = {
        "name": "sdpa.route",
        "gpu_name": "",
        "compute_cap": (12, 0),
        "model": None,
        "program_index": 0,
        "program_wire": wire,
        **loop_record_fields(_sdpa_graph(), ["out"]),
        "bindings": (),
        "pins": (),
        "measurements": None,
        "ranking": None,
    }
    assert decode_record(GoldenRecord(knobs={"PLACE@map.1/twist.1/inner": "cut"}, **fields)) is None
    reason = decode_record(GoldenRecord(knobs={"PLACE@missing": "cut"}, **fields))
    assert reason is not None and "does not resolve" in reason
    # A route that resolves SOME of its seams is refused the same way. Evidence import is
    # best-effort per record — it keeps the arms the replay did resolve — so the strict decode is
    # the one place a record that would deploy a shorter kernel set than the measured one is loud.
    # The stale seam here is well formed and stands on no site of this tree: one hop past the score
    # contraction, where the operand is a gmem slab and takes no hop of its own.
    partial_route = {"PLACE@map.1/twist.1/inner": "cut", "PLACE@map.1/twist.1/inner.1/map": "cut"}
    partial = decode_record(GoldenRecord(knobs=partial_route, **fields))
    assert partial is not None and "does not resolve" in partial


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
    """Every scoped PLACE pin that resolves on one kernel joins ONE realization — a producer per
    seam and one consumer, with a producer reading another seam's workspace when its value nests
    inside it — while a pin whose site path exists on no kernel here is another kernel's and is
    skipped, never an error."""
    match, graph = _case_match("attention/rmsnorm-qk-sdpa-composed-cut.yaml")
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
    assert len(producers) == 3 and len(pieces) == 4
    assert all(node.op.placement_decided for node in pieces)
    workspaces = {node.id for node in producers}
    assert any(set(node.inputs) & workspaces for node in producers), "the nested value's producer must read a sibling workspace"


def test_bare_and_scoped_place_cuts_compose_in_one_decision() -> None:
    match, graph = _case_match("attention/rmsnorm-qk-sdpa-composed-cut.yaml")
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


def _receipt_fields() -> dict:
    return {
        "name": "sdpa.child",
        "gpu_name": "",
        "compute_cap": (12, 0),
        "model": None,
        "program_index": 0,
        "program_wire": _sdpa_graph().to_wire(),
        **loop_record_fields(_sdpa_graph(), ["out"]),
        "bindings": (),
        "pins": (("PLACE@map.1/twist.1/inner", "cut"),),
        "measurements": None,
        "ranking": None,
    }


def test_child_identity_receipts_decode_per_child_and_join_by_stored_identity() -> None:
    """Conflicting per-child schedules behind one pinned cut persist as sibling receipts: each
    stored child identity selects its own kernel's rows, a sibling child's row does not vouch for
    it, and the strict decode joins by the stored identity instead of the pre-cut lift."""
    fields = _receipt_fields()
    parent = GoldenRecord(knobs={}, **fields)
    lift_identity = _lifted_target(parent).identity_key(with_io=True)
    children = {i: rows for i, rows in _replay(parent, exhaustive=True).rows.items() if i is not None and i != lift_identity}
    assert len(children) == 2, "the pinned cut must resolve to two distinctly identified child kernels"
    # One child's rows are a subset of the other's, and which identity digest sorts first is not a
    # fact about the kernels — take the child that HAS a row its sibling does not offer.
    (id_a, rows_a), (id_b, rows_b) = sorted(children.items(), key=lambda child: len(child[1]), reverse=True)
    row_a = next(iter(rows_a - rows_b), None)
    assert row_a is not None, "the children must offer at least one distinguishing schedule row"

    receipt = GoldenRecord(knobs=dict(row_a), identity=id_a, **fields)
    assert decode_record(receipt) is None
    assert kernel_identity(receipt) == id_a

    sibling = GoldenRecord(knobs=dict(row_a), identity=id_b, **fields)
    reason = decode_record(sibling)
    assert reason is not None and "no enumerated row of the identified kernel" in reason

    stale = GoldenRecord(knobs=dict(row_a), identity="0" * 64, **fields)
    reason = decode_record(stale)
    assert reason is not None and "equals none" in reason


def test_child_decode_verdict_changes_with_sibling_route_owner() -> None:
    """A cached child miss must not survive a sibling route-owner repair.

    The child row itself is unchanged. Only the route sibling's identity changes from stale to the
    current pre-cut owner, which makes the replay take the cut and expose the child's schedule.
    """
    fields = {**_receipt_fields(), "pins": ()}
    common = {key: value for key, value in fields.items() if key != "name"}
    route = {"PLACE@map.1/twist.1/inner": "cut"}
    parent = GoldenRecord(name="cache.parent", knobs={}, **common)
    owner = _lifted_target(parent).identity_key(with_io=True)
    current_route = GoldenRecord(name="cache.route", knobs=route, identity=owner, **common)
    children = {
        identity: rows
        for identity, rows in _replay(current_route, exhaustive=True).rows.items()
        if identity is not None and identity != owner
    }
    child_identity, child_rows = max(children.items(), key=lambda child: len(child[1]))
    child = GoldenRecord(name="cache.child", knobs=dict(next(row for row in child_rows if row)), identity=child_identity, **common)

    stale_route = replace(current_route, identity="0" * 64)
    reason = decode_record(child, (stale_route,))
    assert reason is not None and "stored identity equals none of the kernel identities" in reason
    assert decode_record(child, (current_route,)) is None

    # An explicit kernel set supplies its route even after the pre-cut identity changes.
    lead = replace(parent, kernel_set=(stale_route.name,))
    assert decode_record(child, (lead, stale_route)) is None
    assert decode_record(stale_route, (lead, child)) is None


def test_post_schedule_receipt_does_not_steer_an_unowned_peer(monkeypatch) -> None:
    """A receipt identity that appears after scheduling selects only that materialized kernel.

    Another cut child accepts the same schedule row, but must retain the lead's distinct row and
    must not realize the receipt's.
    """
    from emmy.compiler.pipeline.knob import evidence_row_vouches

    fields = _receipt_fields()
    parent = GoldenRecord(knobs={}, **fields)
    children = {identity: rows for identity, rows in _replay(parent, exhaustive=True).rows.items() if identity is not None}
    (target_identity, target_rows), (peer_identity, peer_rows) = sorted(children.items(), key=lambda item: len(item[1]), reverse=True)
    target_row = next(iter(target_rows & peer_rows), None)
    peer_row = next(iter(peer_rows - {target_row}), None)
    assert target_row is not None and peer_row is not None, "the two children need one shared and one distinct schedule row"

    post_identity = "f" * 64
    original_identity_key = TileOp.identity_key

    def identity_after_schedule(self, *, structural=True, with_io=False, with_knobs=False):
        identity = original_identity_key(self, structural=structural, with_io=with_io, with_knobs=with_knobs)
        if self.schedule is not None and with_io and not with_knobs and identity == target_identity:
            return post_identity
        return identity

    monkeypatch.setattr(TileOp, "identity_key", identity_after_schedule)
    lead = GoldenRecord(name="sdpa.lead", knobs=dict(peer_row), **{key: value for key, value in fields.items() if key != "name"})
    receipt = GoldenRecord(
        name="sdpa.receipt",
        knobs=dict(target_row),
        identity=post_identity,
        **{key: value for key, value in fields.items() if key != "name"},
    )

    replay = _replay(receipt, siblings=(lead,), lead=lead)

    assert evidence_row_vouches(replay.realized[post_identity], dict(target_row))
    assert evidence_row_vouches(replay.realized[peer_identity], dict(peer_row))
    assert not evidence_row_vouches(replay.realized[peer_identity], dict(target_row))


def test_child_identity_receipt_selects_one_kernel_from_multi_kernel_loop_target() -> None:
    """A stored child identity is the selector when a regenerated target now lowers to several
    kernels; strict decoding must consult that identity's rows before requiring a one-kernel lift."""
    graph = _sdpa_graph()
    _input(graph, "x", (4, 32))
    graph.add_node(SoftmaxOp(axis=-1), ["x"], Tensor("softmax", (4, 32), "f16"), node_id="softmax")
    graph.inputs.append("x")
    graph.outputs.append("softmax")
    loop = Pipeline.build(LOOP_PASSES).run(graph.copy(), ctx=_CTX)
    fields = {
        **_receipt_fields(),
        "program_wire": graph.to_wire(),
        "origins": (),
        "loop_index": 0,
        "loop_wire": loop.to_wire(),
    }
    parent = GoldenRecord(knobs={}, **fields)
    with pytest.raises(ValueError, match="target lowers to 2 kernels"):
        _lifted_target(parent)
    identity, rows = next((identity, rows) for identity, rows in _replay(parent, exhaustive=True).rows.items() if identity is not None)
    receipt = GoldenRecord(knobs=dict(next(iter(rows))), identity=identity, **fields)
    assert decode_record(receipt) is None


def test_import_files_each_row_under_the_kernel_it_decides(monkeypatch) -> None:
    """Golden evidence is per kernel, in the DB as in the file. A target's entries walk one path: the
    leading entry (the routing record here) decides the parent's placement fork and is its routing
    row — the parent's exact identity, the arm, the pieces — and the child-identity receipt decides
    only the forks of the kernel it names and is that child's perf row, its schedule row as
    recorded, captured, under the golden's source. The parent ran as no kernel and has no row; a
    piece inherits nothing from the kernel it replaced."""
    monkeypatch.setenv("EMMY_FAST_MATH", "0")
    from emmy.compiler.pipeline.search.db import SearchDB
    from emmy.compiler.pipeline.search.golden.evidence import import_goldens

    fields = {**_receipt_fields(), "measurements": {"emmy_us": 1.0, "reference_us": 2.0, "reference_backend": "torch"}}
    route = {"PLACE@map.1/twist.1/inner": "cut"}
    routing = GoldenRecord(knobs=route, **{**fields, "pins": ()})
    parent = GoldenRecord(knobs={}, **fields)
    lift_identity = _lifted_target(parent).identity_key(with_io=True)
    replay = _replay(parent, exhaustive=True)
    child, rows = next((identity, rows) for identity, rows in replay.rows.items() if identity is not None and identity != lift_identity)
    receipt = GoldenRecord(knobs=dict(next(iter(rows))), identity=child, **fields)

    db = SearchDB()
    counts = import_goldens(db, Context.from_target((12, 0)), [routing, receipt], source="golden:test")

    assert counts == {"routing rows": 1, "perf rows": 1}
    [decision] = db.iter_routing()
    assert decision.arm == route and len(decision.children) >= 2
    [row] = db.iter_perf_rows()
    assert row.kernel in decision.children and row.kernel != decision.parent
    schedule = {k: str(v) for k, v in row.knobs.items() if not k.startswith(("S_", "I_"))}
    assert schedule == {k: str(v) for k, v in receipt.schedule_row.items()}
    assert (row.stats.median, row.captured, row.source) == (1.0, True, "golden:test")
    assert next(kernel for kernel in db.iter_kernels() if kernel.exact_identity == row.kernel).structural_identity == child


def test_multi_output_kernel_record_derives_the_identity_its_live_fork_carries() -> None:
    """A record whose one target kernel writes SEVERAL output buffers must derive the identity its
    live fork carries. Every evidence row a golden contributes is keyed by that identity, so a
    derivation that kept only output slot 0 keys the record's rows off a fingerprint no fork can
    produce and the deploy reads none of them. The derivation lifts the persisted kernel and the
    fork root op is whatever the matcher's ``with_io`` produced — a map holding every output slot,
    which is why the lift goes through the same call."""
    graph = Graph()
    _input(graph, "x", (8,))
    graph.add_node(ElementwiseOp("relu"), ["x"], Tensor("hot", (8,), "f16"), node_id="hot")
    graph.add_node(ElementwiseOp("negative"), ["hot"], Tensor("cold", (8,), "f16"), node_id="cold")
    graph.inputs, graph.outputs = ["x"], ["hot", "cold"]
    loop = Pipeline.build(LOOP_PASSES).run(graph.copy(), ctx=_CTX)
    fields = {
        **_receipt_fields(),
        "name": "fused.multi_output",
        "pins": (),
        "program_wire": graph.to_wire(),
        "origins": (),
        "loop_index": 0,
        "loop_wire": loop.to_wire(),
    }
    record = GoldenRecord(knobs={}, **fields)
    _lowered, nodes = _target_kernel_nodes(record)
    assert len(nodes) == 1 and len(nodes[0].outputs) == 2, "the fused target must be ONE kernel writing two buffers"

    identity = kernel_identity(record)
    rows = _replay(record, exhaustive=True).rows
    assert identity in rows, "the derived identity names no kernel the live resolve offers"
    # The join is the subject; spelling one of that kernel's own rows shows the record decodes
    # strictly through it too.
    assert decode_record(GoldenRecord(knobs=dict(next(iter(rows[identity]))), **fields)) is None


def test_receipt_validation_requires_child_identity_and_place_pins_stay_live(monkeypatch) -> None:
    monkeypatch.setenv("EMMY_FAST_MATH", "0")
    from types import SimpleNamespace

    from emmy.compiler.pipeline.search.golden import regime_live

    fields = _receipt_fields()
    loops: list[dict] = []
    document = {
        "compute_cap": [12, 0],
        "programs": [fields["program_wire"]],
        "loops": loops,
        "configs": [
            {
                "program": 0,
                "target": loop_target(_sdpa_graph(), ["out"], loops),
                "realizations": [
                    {"name": "sdpa.child", "bindings": {}, "pins": {"PLACE@map.1/twist.1/inner": "cut"}, "knobs": {"WORK": "w4x2"}}
                ],
            }
        ],
    }
    with pytest.raises(ValueError, match="child-identity schedule receipt"):
        GoldenFile.from_wire(document).check()
    document["configs"][0]["realizations"][0]["identity"] = "0" * 64
    GoldenFile.from_wire(document).check()
    receipt = SimpleNamespace(pin_map={"PLACE@map.1/twist.1/inner": "cut"})
    assert regime_live(receipt), "a receipt's routing pins are its route, never a dead env regime"


def test_pool_group_fuses_node_id_respellings_and_keys_on_pins() -> None:
    """``pool_group`` composes the target kernels' identity keys, so two recordings of ONE
    program whose node ids differ (separate recording sessions) fuse into one enumeration
    group — the wire digest this replaced split them — while a different pin regime still
    keys apart."""
    fields = _receipt_fields()
    respelled = _sdpa_graph()
    for nid in [n for n in respelled.nodes if n not in respelled.inputs]:
        respelled.rename_node(nid, f"session2_{nid}")
    twin_fields = {
        **fields,
        "program_wire": respelled.to_wire(),
        **loop_record_fields(respelled, [f"session2_{o}" for o in fields["origins"]]),
    }
    a = GoldenRecord(knobs={}, **fields)
    b = GoldenRecord(knobs={}, **twin_fields)
    assert a.pool_group == b.pool_group, "node-id spelling must not split an enumeration group"

    unpinned = GoldenRecord(knobs={}, **{**fields, "pins": ()})
    assert unpinned.pool_group != a.pool_group, "the pin regime is a group-key term"


# ---------------------------------------------------------------------------
# The routing lane: a recorded ROUTING row decides a placement fork.
# ---------------------------------------------------------------------------

#: A card in the ``emmy.gpu`` registry, so the record's context reconstructs without a live device.
_ROUTING_CARD = "NVIDIA GeForce RTX 5090"


def _sdpa_kernel_identity() -> str:
    """The deploy identity carried by the sdpa program's PLACEMENT fork — the PRE-CUT kernel, which
    is the kernel a routing row names: the route it records is the one decision taken on that
    kernel, before any piece of it exists. Probed off a resolve rather than restated here, so these
    tests pin the routing lane and not a second copy of the identity derivation."""
    from emmy.compiler.pipeline.fork import flatten_leaves

    ctx = Context.from_target((12, 0), gpu_name=_ROUTING_CARD)
    lowered = Pipeline.build(LOOP_PASSES).run(_sdpa_graph(), ctx=ctx)
    seen: list[str] = []

    def decide(fp):
        if not seen and isinstance(fp.root_op, TileOp) and fp.match.rule.name == _CUT.__name__.rsplit(".", 1)[-1]:
            seen.append(fp.root_op.identity_key(with_io=True))
        return flatten_leaves(fp.options)[0]

    Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx).resolve(lowered, decide)
    assert seen, "the sdpa program must offer a placement fork"
    return seen[0]


def _routing_record(knobs: dict, *, name: str = "sdpa.route") -> GoldenRecord:
    """A measured ROUTING row over the sdpa kernel — nothing but ``PLACE`` keys, which is what
    makes it a recorded placement rather than a recorded schedule. The identity is stored so this
    exercises the routing LANE and not the record-side identity derivation, which has its own
    tests."""
    return GoldenRecord(
        name=name,
        gpu_name=_ROUTING_CARD,
        compute_cap=(12, 0),
        model=None,
        program_index=0,
        program_wire=_sdpa_graph().to_wire(),
        **loop_record_fields(_sdpa_graph(), ["out"]),
        bindings=(),
        pins=(),
        knobs=knobs,
        identity=_sdpa_kernel_identity(),
        measurements=Measurements(emmy_us=1.0, reference_us=2.0, reference_backend="torch"),
        ranking=None,
    )


def _deploy_kernels(records: list) -> list[str]:
    """Resolve the sdpa program through the deploy policy with ``records`` as the card's corpus —
    imported into the compile's DB, as every compile imports its golden scope — and return the
    resolved kernel set. ``prior=None`` pins the non-recorded forks to emission
    order, so the recorded evidence is the only thing that can move the answer. The records are
    evidence in any nvcc regime: a golden row is scoped by the card and by its own input pins
    (``regime_live``), never by the optimization level the suite compiles at."""
    from emmy.compiler.pipeline.search.golden import records_override
    from emmy.compiler.pipeline.search.golden.evidence import evidence_db
    from emmy.compiler.pipeline.search.policy.greedy import greedy_decide

    ctx = Context.from_target((12, 0), gpu_name=_ROUTING_CARD)
    lowered = Pipeline.build(LOOP_PASSES).run(_sdpa_graph(), ctx=ctx)
    with records_override(records):
        db = evidence_db(None, ctx)
        terminal, _trace = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx).resolve(lowered, greedy_decide(prior=None, db=db))
    return sorted(node.id for node in terminal.nodes.values() if isinstance(node.op, TileOp))


#: The root-most of the sdpa kernel's offered seams, as the route codec spells it: the twist that
#: carries the softmax statistics.
_SDPA_ROUTE = "PLACE@map.1/twist"


def test_a_recorded_kernel_set_deploys_the_cut_every_entry_spells(monkeypatch) -> None:
    """A cut mints brand-new kernels, so a kernel set cut twice over is recorded per kernel and not
    as one row spelling both seams: the leading entry spells the seam offered on the target's own
    kernel, and an entry naming a piece by its stored identity spells the seam that piece offers on
    its own tree. Each entry's decision is a routing row on the kernel whose fork it decided, priced
    from the receipts of the kernels the set finally ran as, so the deploy composes the whole
    recorded set — and a decision whose pieces carry no receipt prices nothing."""
    monkeypatch.setenv("EMMY_FAST_MATH", "0")
    fused = _deploy_kernels([])
    assert len(fused) == 1, f"with no recorded route the fork falls to emission order (fuse): {fused}"

    parent = _routing_record({_SDPA_ROUTE: "cut"})
    assert _deploy_kernels([parent]) == fused, "a routing row alone prices nothing: its pieces have no receipt"
    pieces = sorted(_replay(parent).kernels)
    receipts = [replace(parent, name=f"sdpa.receipt{i}", knobs={}, identity=identity) for i, identity in enumerate(pieces)]
    routed = _deploy_kernels([parent, *receipts])
    assert sum(1 for name in routed if "__place_" in name) == 1, f"the parent's entry deploys its one seam: {routed}"

    cuts = [replace(parent, name=f"sdpa.piece{i}", knobs={"PLACE": "cut"}, identity=identity) for i, identity in enumerate(pieces)]
    leaves = sorted(_replay(parent, siblings=tuple(cuts), lead=parent).kernels)
    leaf_receipts = [replace(parent, name=f"sdpa.leaf{i}", knobs={}, identity=identity) for i, identity in enumerate(leaves)]
    composed = _deploy_kernels([parent, *cuts, *leaf_receipts])
    assert sum(1 for name in composed if "__place_" in name) >= 2, f"every recorded seam must be cut: {composed}"


def test_a_recorded_schedule_row_never_routes() -> None:
    """The lanes do not cross: a schedule row carries no ``PLACE`` key, so it is not a route and
    the placement fork stays with pricing even though the row joins the same kernel identity."""
    schedule_row = _routing_record({"WORK": "w4x1", "TILE": ""}, name="sdpa.schedule")
    assert not schedule_row.is_routing
    assert _deploy_kernels([schedule_row]) == _deploy_kernels([])


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
    match, graph = _case_match("attention/rmsnorm-qk-sdpa-composed-cut.yaml")
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
    tile = case_target_tile("fused/nvfp4-gate-up-requant-place-cut.yaml")
    owning = {seam.spelling: seam for seam in cuttable_seams(tile) if seam.owned is not None}

    assert set(owning) == {"PLACE@map.1/map", "PLACE@map.2/map"}
    assert [store.write.output for store in owning["PLACE@map.1/map"].owned[1]] == ["mul_static_fp4_bits"]
    assert [store.write.output for store in owning["PLACE@map.2/map"].owned[1]] == ["mul_static_fp4_scale_bits"]
    assert all(seam.dtypes == () for seam in owning.values()), "an output-owning seam writes no workspace to type"


def test_output_owning_cut_leaves_single_output_pieces_that_promote() -> None:
    """Realizing it gives two kernels, each writing ONE of the kernel's own outputs — no workspace
    between them — and each binding the sweep its store rides as a grid axis. Rank two is what the
    contraction sites need to name an ``(m, n)`` pair at all."""
    fragment = _realized("fused/nvfp4-gate-up-requant-place-cut.yaml", "PLACE@map.1/map")
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


def test_an_output_owning_cut_is_declined_where_no_piece_would_gain_a_grid_axis() -> None:
    """The same partition over a purely pointwise quantize: both branches own a store, but neither
    holds a contraction reading it, so promotion has nothing to lift and splitting would buy a
    second launch and no grid. The seams keep their workspace reading."""
    tile = case_target_tile("fused/nvfp4-quantize-cut-shared-normalizer.yaml")

    assert len(tile.output_specs) == 2
    assert all(seam.owned is None for seam in cuttable_seams(tile))


# ---- the full-projection cut --------------------------------------------------------------------- #

#: The serving W4A4 MLP shape, whose requant projection owns more outputs than the binder can bind.
_REQUANT = "fused/nvfp4-gate-up-requant-place-cut.yaml"


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
    """Taking it. Every piece holds at most one contraction, which is the committed shape: each
    expensive contraction computed once as the sole root of its own kernel, the owned outputs read
    back. And every piece binds at least two grid axes — the pointwise pieces included, which is
    what the sweep-promotion rule has to supply once no contraction is left under them."""
    graph, node = _mimo_case(_REQUANT)
    option, knobs = _composed_arm(graph, node)
    pieces = _piece_ops(option.materialize())

    assert len(pieces) == len(knobs), "one piece per seam; the projection hands away all of its outputs"
    for piece in pieces:
        assert len(_contraction_spellings(piece)) <= 1, f"{piece.name} still holds several contractions"
        assert len(piece.place.free) >= 2, f"{piece.name} kept a one-axis placement"


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


def test_a_decision_consumes_the_key_that_spelled_it_on_import(monkeypatch) -> None:
    """A bare ``PLACE=cut`` spells one cut — the root-most seam of the kernel the entry decides — and is
    spent by it: the pieces are read against what the entry has left to say, so the import writes the
    one decision the recording took, not a cut at every piece that offers a seam."""
    monkeypatch.setenv("EMMY_FAST_MATH", "0")
    from emmy.compiler.pipeline.search.db import SearchDB
    from emmy.compiler.pipeline.search.golden.evidence import import_goldens

    db = SearchDB()
    counts = import_goldens(
        db, Context.from_target((12, 0), gpu_name=_ROUTING_CARD), [_routing_record({"PLACE": "cut"})], source="golden:t"
    )
    assert counts["routing rows"] == 1 and len(list(db.iter_routing())) == 1


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
