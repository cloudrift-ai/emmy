"""Explicit working-golden replay for ``compile`` / ``run``: a row is selected by name, the file is the compile's golden
evidence, the kernel-set decisions that mint a row's kernel are its route, and what a run records is what the next
compile picks."""

import asyncio
import re
from dataclasses import replace
from types import SimpleNamespace

import pytest

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.cuda.ir import CudaOp
from emmy.compiler.ir.frontend.ir import MatmulOp, ReshapeOp, RmsNormOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from emmy.compiler.pipeline.knob import schedule_row_key
from emmy.compiler.pipeline.search.golden import GoldenFile, Measurements, Row, evidence_scope, record_greedy_pick, sole_evidence
from emmy.compiler.pipeline.search.inventory import KernelInventory
from emmy.compiler.pipeline.search.pins import pinned_knobs
from emmy.compiler.wire import kernel_tile
from tests.compiler.helpers import inventory_document

_CUT = {"PLACE@inner.1/map": "cut"}
# The residual matmul's cross-CTA split, pinned on its graph node: the route a recorded golden replays is the test's
# input, not the prior's pick.
_SPLIT = {"REDUCE@node_y": "g2k"}


def _relu_graph() -> Graph:
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (16,)), node_id="x")
    graph.add_node(ElementwiseOp("relu"), ["x"], Tensor("y", (16,)), node_id="y")
    graph.inputs, graph.outputs = ["x"], ["y"]
    return graph


def _working_loop(path, *, state="inventory", pins=None) -> GoldenFile:
    """A working golden of one relu kernel with no Torch twin, its one row ``working.relu`` traced only
    (``inventory``), scheduled (``proposal``) or scheduled and measured (``verified``)."""
    document = inventory_document(_relu_graph(), (8, 9))
    [row], [kernel] = document.rows, document.kernels
    row = replace(row, name="working.relu", pins=dict(pins) if pins is not None else row.pins)
    if state in {"proposal", "verified"}:
        row = replace(row, knobs={"WORK": "w1x1"})
    if state == "verified":
        row = replace(row, measurements=Measurements(emmy_us=1.0, reference_us=2.0, reference_backend="torch"))
    document = replace(document, kernels=[replace(kernel, origins=())], rows=[row])
    document.dump(path, overwrite=True)
    return document


def _norm_matmul_graph() -> Graph:
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (Dim(1), Dim(2), Dim(16)), dtype=F16), node_id="x")
    graph.add_node(InputOp(), [], Tensor("wn", (Dim(16),), dtype=F16), node_id="wn")
    graph.add_node(InputOp(), [], Tensor("w", (Dim(16), Dim(16)), dtype=F16), node_id="w")
    graph.add_node(RmsNormOp(), ["x", "wn"], Tensor("xn", (Dim(1), Dim(2), Dim(16)), dtype=F16), node_id="xn")
    graph.add_node(MatmulOp(), ["xn", "w"], Tensor("y", (Dim(1), Dim(2), Dim(16)), dtype=F16), node_id="y")
    graph.inputs, graph.outputs = ["x", "wn", "w"], ["y"]
    return graph


def _cuda_nodes(graph):
    return [graph.nodes[nid] for nid in graph.topological_order() if isinstance(graph.nodes[nid].op, CudaOp)]


@pytest.mark.parametrize("selection", ["--kernel", "--realization"])
def test_cold_golden_selection_reaches_bench_in_its_regime(tmp_path, monkeypatch, selection):
    import argparse

    import torch

    from emmy.commands import run
    from emmy.compiler.pipeline.search.space import cold_cache

    path = tmp_path / "working.json"
    document = _working_loop(path)
    parser = argparse.ArgumentParser()
    run.register_run_command(parser.add_subparsers())
    name = document.kernels[0].ref if selection == "--kernel" else "working.relu"
    args = parser.parse_args(["run", "--golden", str(path), selection, name, "--bench", "--cold-cache"])
    seen = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(run, "_handle_run_ir", lambda *args: seen.append(cold_cache()))
    run.handle_run(args)
    assert seen == [True]


def _picked(graph) -> list[tuple[str, dict]]:
    """Each CUDA kernel of a compiled graph in launch order: its tile's exact identity and its schedule row."""
    return [
        (kernel_tile(node.op).identity_key(structural=False, with_io=True), dict(schedule_row_key(dict(node.op.knobs or {}))))
        for node in _cuda_nodes(graph)
    ]


def _compile_pinned(document: GoldenFile, pins: dict, cap=(8, 9), gpu_name=None):
    """The document's one target compiled under hand pins with no golden evidence, and the decisions the compile took."""
    taken: list = []
    watcher = KernelInventory(on_routing=lambda parent, arm, pieces, _ids: taken.append((parent, arm, pieces)))
    ctx = Context.from_target(cap, gpu_name=gpu_name)
    [target] = document.targets()
    with evidence_scope([]), pinned_knobs(pins):
        picked = Pipeline.build(CUDA_PASSES).with_strategies(watcher).run(target.program({}), ctx=ctx, db=None)
    return picked, taken


def _working_placement_route(path, cap=(8, 9), gpu_name=None) -> GoldenFile:
    """A working golden whose one target, a norm into a matmul, was cut once and its residual matmul split: the seed row
    ``working.route``, each decision as a routing row, and a measured row per piece — what ``run --record-greedy`` writes."""
    document = inventory_document(_norm_matmul_graph(), cap, gpu_name=gpu_name)
    [row] = document.rows
    document = replace(document, rows=[replace(row, name="working.route", pins={"FAST_MATH": False})])
    document.dump(path, overwrite=True)
    picked, taken = _compile_pinned(document, {"FAST_MATH": False, **_CUT, **_SPLIT}, cap, gpu_name)
    assert taken and taken[0][1] == _CUT
    record_greedy_pick(
        path,
        "working.route",
        decisions=taken,
        kernels=[(node.op, 1.0, 2.0, None) for node in _cuda_nodes(picked)],
        reference_backend="torch",
    )
    return GoldenFile.load(path)


def _piece_name(document: GoldenFile) -> str:
    """The name of a row on a piece the decision on the file's target minted."""
    [target] = document.targets()
    route = next(route for route in document.routing if route.parent == target.ref)
    return next(row.name for row in document.rows if row.kernel in route.children)


@pytest.mark.parametrize("child_first", [False, True])
@pytest.mark.parametrize("traced", [None, 0])
def test_restamp_preserves_nested_routes_in_either_file_order(tmp_path, child_first, traced):
    from emmy.compiler.pipeline.search.golden.restamp import restamp

    document = _working_placement_route(tmp_path / "nested.json")
    parent, child = document.routing
    assert child.parent in parent.children
    if child_first:
        document.routing.reverse()
    fresh, report = restamp(document, traced=traced)
    assert not report.changed, report.lines()
    assert fresh == document, "route order cannot change the kernels, decisions, or measured rows"


@pytest.mark.parametrize("changed", [False, True])
def test_restamp_matches_reordered_children_before_rekeying(tmp_path, changed):
    from emmy.compiler.pipeline.search.golden.restamp import restamp

    document = _working_placement_route(tmp_path / "reordered.json")
    parent, nested = document.routing
    document.routing[0] = replace(parent, children=parent.children[::-1])
    altered = next(ref for ref in parent.children if ref != nested.parent)
    if changed:
        [other] = inventory_document(_relu_graph(), (8, 9)).kernels
        document.kernels[:] = [replace(k, loop_ir=other.loop_ir, formed=other.formed) if k.ref == altered else k for k in document.kernels]
    fresh, report = restamp(document)
    assert fresh.routing == document.routing, "the nested decision must follow its exact child despite reordering"
    assert not report.dropped_kernels and not report.dropped_routes and not report.dropped_rows
    for old, new in zip(document.rows, fresh.rows, strict=True):
        if changed and old.kernel == altered:
            assert old.measured and new == replace(old, measurements=None, latency=None)
        else:
            assert new == old, "an unchanged kernel keeps its measured row"
    if changed:
        assert len(report.rekeyed) == 1
        assert report.demoted == [row.name for row in document.rows if row.kernel == altered]
    else:
        assert not report.changed and fresh == document


def _args(path, **overrides):
    values = {
        "realization": "working.relu",
        "golden": str(path),
        "code": None,
        "input": None,
        "ir": "cuda",
        "dynamic": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_compile_working_file_uses_exact_loop_target(run_cli, tmp_path):
    path = tmp_path / "working.json"
    _working_loop(path)
    rc, stdout, stderr = run_cli("compile", "--golden", str(path), "--realization", "working.relu", "--target", "sm_89", "--ir", "loop")
    assert rc == 0, stderr
    assert "relu" in stdout


def test_working_file_requires_name_and_reports_its_own_available_rows(run_cli, tmp_path):
    path = tmp_path / "working.json"
    _working_loop(path)
    rc, stdout, stderr = run_cli("compile", "--golden", str(path), "--target", "sm_89")
    assert rc == 2 and "requires --realization NAME" in stdout + stderr
    rc, stdout, stderr = run_cli("compile", "--golden", str(path), "--realization", "missing", "--target", "sm_89")
    assert rc == 2 and "working.relu" in stdout + stderr


def test_working_file_golden_conflicts_with_direct_input(run_cli, tmp_path):
    path = tmp_path / "working.json"
    _working_loop(path)
    rc, stdout, stderr = run_cli("compile", "--golden", str(path), "--realization", "working.relu", "--code", "torch.randn(4)")
    assert rc == 2
    assert "mutually exclusive" in stdout + stderr


def test_a_realization_substring_resolves_to_the_exact_name_on_args(tmp_path):
    """``--realization`` accepts an unambiguous substring, and everything after resolution — the record path above
    all — must see the exact name it selected."""
    from emmy.commands.compile import resolve_golden_arg

    path = tmp_path / "working.json"
    _working_loop(path)
    args = _args(path)
    args.realization = "relu"
    resolve_golden_arg(args)
    assert args.realization == "working.relu"


def test_a_specialized_row_uses_its_own_kernels_reference(tmp_path):
    """A row template with sizes specializes the program: the kernel at that size is a kernel of its own, and the
    symbolic row's reference runs at any size while the specialized one's does not."""
    import torch

    from emmy.commands.compile import resolve_golden_arg
    from emmy.compiler.backend import torch_ref

    tokens = Dim("num_tokens")
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (tokens, 4)), node_id="x")
    graph.add_node(ReshapeOp(shape=(1, "num_tokens", 4)), ["x"], Tensor("y", (1, tokens, 4)), node_id="y")
    graph.inputs, graph.outputs = ["x"], ["y"]
    templates = [
        {"name": "m1", "bindings": {"num_tokens": 1}, "pins": {"FAST_MATH": False}},
        {"name": "dynamic", "bindings": {}, "pins": {"FAST_MATH": False}},
    ]
    document = inventory_document(graph, (8, 9), realizations=templates)
    assert len(document.kernels) == 2 and {row.name.rsplit(".", 1)[1] for row in document.rows} == {"m1", "dynamic"}
    path = tmp_path / "working-dynamic.json"
    document.dump(path)

    args = _args(path, realization="dynamic")
    resolve_golden_arg(args)
    static = document.kernel(next(row.kernel for row in document.rows if row.name.endswith(".m1")))
    assert static.bindings == {"num_tokens": 1}
    x = torch.arange(24).reshape(6, 4)
    first_fn, first_inputs = torch_ref.build_callable(document.reference_program(static), {"x": x})
    with pytest.raises(RuntimeError, match="shape"):
        first_fn(*first_inputs)
    fn, inputs = torch_ref.build_callable(args._golden_reference, {"x": x})
    assert fn(*inputs).shape == (1, 6, 4)


def test_a_name_recorded_on_two_kernels_is_ambiguous(tmp_path, caplog):
    """A repeated row name must not silently choose between distinct kernels."""
    from emmy.commands.compile import resolve_golden_arg

    document = inventory_document(_relu_graph(), (8, 9))
    other = inventory_document(_norm_matmul_graph(), (8, 9))
    [relu], [norm] = document.rows, other.rows
    both = replace(
        document,
        programs=[*document.programs, *other.programs],
        kernels=[*document.kernels, *(replace(k, traced=1) for k in other.kernels)],
        rows=[replace(relu, name="working.relu"), replace(norm, name="working.relu")],
    )
    path = tmp_path / "working.json"
    both.dump(path)
    with pytest.raises(SystemExit) as exc:
        resolve_golden_arg(_args(path))
    assert exc.value.code == 2
    assert "resolves to 2 different kernels" in caplog.text


def test_named_proposal_is_pinned_and_a_file_walk_leaves_it_unbenched(tmp_path):
    """Naming a realization asks for that row: it benches as a pinned row whatever its measurement state. A whole-file
    walk benches a target's measured rows only."""
    from emmy.commands.compile import resolve_golden_arg

    path = tmp_path / "working.json"
    _working_loop(path, state="proposal")
    explicit = _args(path)
    resolve_golden_arg(explicit)
    assert [sample.knobs for sample in explicit.golden_configs] == [{"WORK": "w1x1"}]
    walk = _args(path, _explicit_realization=False)
    resolve_golden_arg(walk)
    assert walk.golden_configs == []


@pytest.mark.parametrize(
    "knobs,measured,timed,automatic",
    [
        ({}, False, True, False),
        (None, False, True, False),
        ({"WORK": "w1x1"}, False, True, True),
        ({}, True, True, True),
        ({}, False, False, True),
        (None, False, False, True),
    ],
    ids=["canonical-latency", "working-latency", "scheduled-latency", "measured-defaults", "proposal-defaults", "untuned"],
)
def test_target_latency_without_a_schedule_adds_no_automatic_comparison(tmp_path, knobs, measured, timed, automatic):
    from emmy.commands.compile import golden_regime, resolve_golden_arg
    from emmy.compiler.pipeline.search.golden.format import Latency

    path = tmp_path / "working-route.json"
    document = _working_placement_route(path)
    seed = replace(
        document.rows[0],
        knobs=knobs,
        measurements=Measurements(emmy_us=1.0, reference_us=2.0, reference_backend="torch") if measured else None,
        latency={"test-card": Latency(emmy_us=3.0, tcompile_us=4.0)} if timed else None,
    )
    document.rows[0] = seed
    document.dump(path, overwrite=True)
    args = _args(path, realization=seed.name)

    resolve_golden_arg(args)

    assert bool(args.golden_configs) == automatic
    assert args._golden_rows == [seed]
    assert args._golden_scope == [document]
    assert args._golden_reference is not None
    assert golden_regime(args) == {"FAST_MATH": False}
    if automatic:
        assert args.golden_configs[0].knobs == (knobs or {})


@pytest.mark.parametrize("descendant", [False, True])
def test_latency_only_selection_keeps_its_explicit_route(tmp_path, monkeypatch, descendant):
    from emmy.commands.compile import resolve_golden_arg, selected_decisions
    from emmy.compiler.pipeline.search.golden.format import Latency

    monkeypatch.delenv("EMMY_KNOBS", raising=False)
    path = tmp_path / "working-route.json"
    document = _working_placement_route(path)
    row = next(row for row in document.rows if bool(document.path_to(row.kernel)) == descendant)
    seed = replace(row, knobs={}, measurements=None, latency={"test-card": Latency(emmy_us=3.0)})
    document.rows[document.rows.index(row)] = seed
    document.dump(path, overwrite=True)
    args = _args(path, realization=seed.name, pin_route=True)

    resolve_golden_arg(args)

    assert args.golden_configs == []
    assert selected_decisions(args) == (_CUT if descendant else {})


def test_a_recorded_kernel_set_is_the_evidence_a_compile_cuts_by(tmp_path, monkeypatch):
    """The kernel set a file records is the one a compile of its target takes: the routing row priced from its pieces'
    rows outranks the fused arm, which nothing measured, so the cut is taken with no pin anywhere."""
    monkeypatch.setenv("EMMY_FAST_MATH", "0")
    path = tmp_path / "working-route.json"
    document = _working_placement_route(path)
    [target] = document.targets()
    with evidence_scope([document]), pinned_knobs({"FAST_MATH": False}):
        picked = Pipeline.build(CUDA_PASSES).run(target.program({}), ctx=Context.from_target((8, 9)), db=None)
    names = [node.op.kernel_name for node in _cuda_nodes(picked)]
    assert len(names) >= 2 and sum("__place_" in name for name in names) == 1, names


def test_a_file_walk_does_not_pin_a_pieces_schedule_across_its_target(tmp_path):
    """A seedless route is named by a piece's row. Its schedule belongs to that piece only; the target replays
    from the file's measured evidence, in the selected row's regime even when the file also has another regime."""
    from emmy.commands.compile import golden_regime, resolve_golden_arg

    path = tmp_path / "working-route.json"
    document = _working_placement_route(path)
    pieces = [row for row in document.rows if row.measured]
    document = replace(document, rows=[*pieces, replace(pieces[0], name="fast", pins={"FAST_MATH": True})])
    document.dump(path, overwrite=True)
    args = _args(path, realization=_piece_name(document), _explicit_realization=False)

    resolve_golden_arg(args)

    assert args.golden_configs == []
    assert golden_regime(args) == {"FAST_MATH": False}
    assert args._golden_scope == [document]
    assert args._golden_reference is not None


@pytest.mark.parametrize("explicit", [False, True], ids=["ordinary", "explicit"])
def test_recorded_route_cuts_the_selected_compile_target(run_cli, tmp_path, monkeypatch, explicit):
    """``--pin-route`` compiles a named piece's row under the decisions that mint its kernel: the route is pinned for
    the compile (`compile.selected_decisions`), so the pass's own cut arm is taken and the compile splits into the
    placed producer plus its consumers. A hand pin of the same route through ``EMMY_KNOBS`` lands identically."""
    path = tmp_path / "working-route.json"
    document = _working_placement_route(path)
    monkeypatch.delenv("EMMY_KNOBS", raising=False)
    monkeypatch.setenv("EMMY_NVCC_FLAGS", "")
    monkeypatch.setenv("EMMY_READABLE", "1")
    monkeypatch.setenv("EMMY_TUNE_DB", str(tmp_path / "tune.db"))
    if explicit:
        monkeypatch.setenv("EMMY_KNOBS", "PLACE@inner.1/map=cut")

    rc, stdout, stderr = run_cli(
        "compile", "--golden", str(path), "--realization", _piece_name(document), "--pin-route", "--target", "sm_89", "--ir", "tile"
    )

    # Kernel names carry volatile identity digests, so assert the kernel set's shape, not the spelled names.
    assert rc == 0, stderr
    headers = [line for line in stdout.splitlines() if line.startswith("=== ")]
    assert len(headers) == 3, stdout
    assert sum("__place_" in line for line in headers) == 1, stdout


def test_a_pieces_row_carries_the_decisions_that_mint_it(tmp_path):
    """A row on a piece replays under the decisions that mint its kernel — its route — beside its own schedule;
    a row on a kernel that ran whole pins the kernel whole."""
    from emmy.commands.compile import resolve_golden_arg
    from emmy.commands.run import _sample_replay_knobs

    path = tmp_path / "working-route.json"
    document = _working_placement_route(path)
    args = _args(path, realization=_piece_name(document))
    resolve_golden_arg(args)
    (sample,) = args.golden_configs
    assert sample.route == _CUT and sample.pins == {"FAST_MATH": False}
    assert _sample_replay_knobs(sample) == {"FAST_MATH": False, **_CUT, **sample.knobs}

    whole = tmp_path / "working.json"
    _working_loop(whole, state="verified")
    args = _args(whole)
    resolve_golden_arg(args)
    (sample,) = args.golden_configs
    assert sample.route == {"PLACE": "fuse"}
    assert _sample_replay_knobs(sample) == {"FAST_MATH": True, "PLACE": "fuse", "WORK": "w1x1"}


def test_selected_file_scopes_the_evidence_and_a_split_regime_publishes_nothing(monkeypatch, tmp_path):
    """The selected file is the compile's whole golden scope; the input regime (the precision pins) reaches the
    environment only when every selected row agrees on it."""
    from emmy.commands.compile import golden_regime, resolve_golden_arg
    from emmy.compiler.pipeline.search.golden import repository

    path = tmp_path / "working.json"
    document = _working_loop(path, pins={"FAST_MATH": False})
    [row] = document.rows
    document = replace(document, rows=[row, replace(row, pins={"FAST_MATH": True})])
    document.dump(path, overwrite=True)
    args = _args(path)

    resolve_golden_arg(args)

    assert [sample.record.pins for sample in args.golden_configs] == [{"FAST_MATH": False}, {"FAST_MATH": True}]
    assert golden_regime(args) == {}
    args._golden_rows = args._golden_rows[:1]
    assert golden_regime(args) == {"FAST_MATH": False}
    assert [len(scope.rows) for scope in args._golden_scope] == [2]

    # Without --golden PATH the live card's repository goldens are searched, and a match scopes the compile to its file.
    monkeypatch.setattr(repository, "live_gpu_key", lambda: None)
    monkeypatch.setattr(repository, "repository_documents", lambda *_: [document])
    canonical = _args(path, golden=None)
    resolve_golden_arg(canonical)
    assert [sample.name for sample in canonical.golden_configs] == ["working.relu", "working.relu"]
    assert canonical._golden_scope == [document]


def test_named_run_records_only_the_selected_precision_regime(monkeypatch, tmp_path):
    """A row recorded in one regime leaves the other regime's rows of the same kernel untouched."""
    import torch

    from emmy.commands import compile as compile_module
    from emmy.commands import run as run_module
    from emmy.compiler.pipeline.search.pins import measured_regime_pins

    path = tmp_path / "working.json"
    document = _working_loop(path, pins={"FAST_MATH": False})
    [kernel] = document.kernels
    picked, _taken = _compile_pinned(document, {"FAST_MATH": False})
    [node] = _cuda_nodes(picked)
    fast = Row(
        name=f"working.relu.{kernel.exact_identity[:12]}",
        kernel=kernel.ref,
        pins={"FAST_MATH": True},
        knobs=dict(schedule_row_key(dict(node.op.knobs or {}))),
        measurements=Measurements(emmy_us=9.0, reference_us=10.0, reference_backend="same-input-greedy"),
    )
    with GoldenFile.edit(path) as editing:
        editing.rows.append(fast)
    monkeypatch.delenv("EMMY_FAST_MATH", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(compile_module, "validate_trace_adapter_args", lambda _args: None)

    def record(args, *_):
        assert [sample.record.pins for sample in args.golden_configs] == [{"FAST_MATH": False}]
        assert measured_regime_pins()["FAST_MATH"] is False
        record_greedy_pick(path, args.realization, decisions=[], kernels=[(node.op, 1.0, 2.0, None)], reference_backend="same-input-greedy")

    monkeypatch.setattr(run_module, "_handle_run_ir", record)
    run_module._handle_run_once(_args(path, ir=None, ab=[], bench=True, strict_correctness=False, json=None))

    rows = GoldenFile.load(path).rows
    assert sorted((row.pins["FAST_MATH"], row.measurements.emmy_us) for row in rows if row.measured) == [(False, 1.0), (True, 9.0)]


def test_working_verified_row_is_automatically_pinned(tmp_path):
    from emmy.commands.compile import resolve_golden_arg
    from emmy.commands.run import _sample_replay_knobs

    path = tmp_path / "working.json"
    _working_loop(path, state="verified")
    args = _args(path)

    resolve_golden_arg(args)

    assert len(args.golden_configs) == 1
    assert args.golden_configs[0].knobs == {"WORK": "w1x1"}
    assert args.golden_configs[0].pins == {"FAST_MATH": True}
    assert _sample_replay_knobs(args.golden_configs[0]) == {"FAST_MATH": True, "PLACE": "fuse", "WORK": "w1x1"}


def test_run_replays_embedded_loop_golden_through_structural_stamps(tmp_path):
    from emmy.commands.compile import resolve_golden_arg
    from emmy.commands.run import _passes_after_stage, _replay_stage_and_passes

    path = tmp_path / "working.json"
    _working_loop(path)
    args = _args(path)
    resolve_golden_arg(args)

    stage, passes = _replay_stage_and_passes(args._golden_graph, embedded_golden=True)
    assert stage == "golden Loop"
    assert passes == CUDA_PASSES

    stage, passes = _replay_stage_and_passes(args._golden_graph, embedded_golden=False)
    assert stage == "loop"
    assert passes == _passes_after_stage("loop")
    assert passes != CUDA_PASSES


def test_emmy_only_benchmark_returns_same_input_reference():
    """Embedded Loop replay can return its greedy inputs/outputs without a Torch twin."""
    import numpy as np

    from emmy.commands.run import bench_lowered_vs_torch

    graph = Graph()
    graph.add_node(ConstantOp(name="y", value=2.0), [], Tensor("y", (1,)), node_id="y")
    graph.outputs = ["y"]
    outputs = {"y": np.array([2.0], dtype=np.float32)}

    class FakeBackend:
        def run(self, _graph, *, input_data, taps=()):
            return SimpleNamespace(outputs=outputs), None

        async def benchmark_async(self, *_args, **_kwargs):
            return SimpleNamespace(time_ms=0.001, captured=True)

    refs = []
    asyncio.run(
        bench_lowered_vs_torch(None, graph, FakeBackend(), seed=0, do_bench=True, warmup=1, iters=1, bench_backends="emmy", ref_out=refs)
    )
    assert len(refs) == 1
    assert refs[0][0] == {"y": [2.0]}
    assert refs[0][1] is outputs


def test_constant_cast_fragment_has_no_whole_op_reference(monkeypatch):
    """A cast's synthetic boundary cannot be compared with the original constant's value."""
    from emmy.compiler import pipeline
    from emmy.compiler.ir.expr import Var
    from emmy.compiler.ir.tensor.ir import IndexMapOp, IndexSource

    # Keep the cast boundary that a larger unfusable consumer region leaves behind.
    monkeypatch.setattr(pipeline, "LOOP_PASSES", [p for p in pipeline.LOOP_PASSES if p != "loop/fusion"])
    graph = Graph()
    graph.add_node(ConstantOp(name="weight", source_path="weight"), [], Tensor("weight", (256,), "f16"), node_id="weight")
    graph.add_node(
        IndexMapOp(out_shape=(Dim(256),), sources=(IndexSource(0, (Var("out_coord_0"),)),)),
        ["weight"],
        Tensor("out", (256,), "f32"),
        node_id="out",
    )
    graph.outputs = ["out"]
    document = inventory_document(graph, (7, 0))
    assert len(document.kernels) == 2, "the cast boundary stays: two kernels"
    fragment = next(kernel for kernel in document.targets() if not kernel.origins)
    assert document.reference_program(fragment) is None


def test_emmy_only_benchmark_does_not_duplicate_inputs_on_torch(monkeypatch):
    """A reference-free Loop target owns one device input allocation, not a redundant Torch copy."""
    import numpy as np

    from emmy.commands import run as run_module

    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (8,), "f16"), node_id="x")
    graph.outputs = ["x"]

    class FakeBackend:
        def run(self, _graph, *, input_data, taps=()):
            assert input_data["x"].shape == (8,)
            return SimpleNamespace(outputs={"x": np.ones(8, dtype=np.float16)}, time_ms=0.001), None

        async def benchmark_async(self, *_args, **_kwargs):
            return SimpleNamespace(time_ms=0.001, captured=True)

    monkeypatch.setattr(
        run_module,
        "_to_cuda_tensor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("reference-free replay must not make a Torch copy")),
    )
    asyncio.run(
        run_module.bench_lowered_vs_torch(None, graph, FakeBackend(), seed=0, do_bench=True, warmup=1, iters=1, bench_backends="emmy")
    )


def _run_args(path, **overrides):
    return _args(
        path,
        ir=None,
        bench=True,
        ab=None,
        debug=False,
        dump_dir=None,
        bench_backends="emmy",
        warmup=5,
        iters=20,
        seed=0,
        json=None,
        profile=False,
        **overrides,
    )


class _FakeDump:
    @staticmethod
    def resolve(_path):
        return None


def test_embedded_loop_pins_receive_greedy_output_reference(monkeypatch, tmp_path, caplog):
    """Exact Loop targets have no Torch twin, so pinned replay must compare against the greedy Loop execution."""
    from emmy.commands import run as run_module
    from emmy.commands.compile import resolve_golden_arg

    path = tmp_path / "working.json"
    _working_loop(path, state="verified")
    args = _run_args(path)
    resolve_golden_arg(args)

    reference = ({"x": object()}, {"y": object()})
    returned = {"reference": reference, "greedy_error": None, "reference_run_us": None, "accuracy_error": None}
    seen = {}

    class FakePipeline:
        def run(self, graph, **_kwargs):
            return graph

    class FakeBackend:
        name = "cuda"
        tune_db = None
        bench_compile_timeout_s = 1.0
        bench_run_timeout_s = 1.0

        def __init__(self, **_kwargs):
            pass

        async def benchmark_compare_async(self, _graph, **kwargs):
            seen["want_ref"] = kwargs["want_ref"]
            return {
                "results": {},
                "result": None,
                "captured": False,
                "torch_available": False,
                "accuracy_error": returned["accuracy_error"],
                "run_io": returned["reference"],
                "greedy_error": returned["greedy_error"],
                "reference_run_us": returned["reference_run_us"],
            }

        async def aclose_async_worker(self):
            pass

    async def fake_isolated(*_args, **_kwargs):
        return None

    scopes = []

    async def fake_pinned(_backend, _source, _pins, **kwargs):
        from emmy.compiler.pipeline.search.golden import repository

        scopes.append(repository.SCOPE)  # the pinned rows compile under the file's own scope
        seen["ref"] = kwargs["ref"]
        if kwargs["strict_correctness"]:
            seen["strict_reference"] = kwargs["strict_reference"]
        return []

    monkeypatch.setattr(Pipeline, "build", lambda _passes: FakePipeline())
    monkeypatch.setattr(run_module, "_bench_greedy_isolated", fake_isolated)
    monkeypatch.setattr(run_module, "_bench_golden_variants", fake_pinned)
    monkeypatch.setattr(run_module, "_print_kernel_stats", lambda *_args, **_kwargs: None)

    run_module._handle_run_ir(args, FakeBackend, _FakeDump)

    assert seen == {"want_ref": True, "ref": reference}
    assert scopes == [args._golden_scope]

    args.strict_correctness = True
    returned["accuracy_error"] = "strict eager correctness unavailable: frontend IR is not runnable"
    returned["reference"] = ({"x": [1.0]}, {"y": [1.0]})
    seen.clear()
    with pytest.raises(SystemExit) as exc:
        run_module._handle_run_ir(args, FakeBackend, _FakeDump)
    assert exc.value.code == 1
    assert seen == {"want_ref": True, "ref": returned["reference"], "strict_reference": "same-input-greedy"}

    args.strict_correctness = False
    returned["accuracy_error"] = None
    returned["reference"] = reference

    async def fail_if_isolated(*_args, **_kwargs):
        raise AssertionError("a failed greedy timing must not be re-benched or made eligible")

    seen.clear()
    returned["greedy_error"] = "HungKernelError: repeated timing crossed the watchdog"
    returned["reference_run_us"] = 4_000_000.0
    monkeypatch.setattr(run_module, "_bench_greedy_isolated", fail_if_isolated)
    with pytest.raises(SystemExit) as exc:
        run_module._handle_run_ir(args, FakeBackend, _FakeDump)
    assert exc.value.code == 1
    assert seen == {"want_ref": True, "ref": reference}
    assert "untimed greedy is ineligible; pinned rows still bench" in caplog.text

    seen.clear()
    returned["greedy_error"] = None
    returned["reference_run_us"] = None
    returned["reference"] = None
    monkeypatch.setattr(run_module, "_bench_greedy_isolated", fake_isolated)
    with pytest.raises(SystemExit) as exc:
        run_module._handle_run_ir(args, FakeBackend, _FakeDump)
    assert exc.value.code == 1
    assert seen == {"want_ref": True}
    assert "requires same-input greedy outputs" in caplog.text


def test_a_walk_recording_the_greedy_pick_still_times_it_isolated(monkeypatch, tmp_path):
    """``--record-greedy`` records the greedy pick from its per-kernel ISOLATED re-bench, and asks the worker for the
    greedy's own outputs — the only reference a target with no Torch twin has — on a walk over a fresh trace
    inventory too, which holds no measured row to pin."""
    from emmy.commands import run as run_module
    from emmy.commands.compile import resolve_golden_arg

    path = tmp_path / "working.json"
    _working_loop(path)
    args = _run_args(path, record_greedy=True)
    resolve_golden_arg(args)
    args._explicit_realization = False
    args.golden_configs = []  # a walk pins a target's measured rows; a trace inventory has none
    seen = {}
    returned = {"accuracy_error": None}

    class FakePipeline:
        def with_strategies(self, _taken):
            return self

        def run(self, graph, **_kwargs):
            return graph

    class FakeBackend:
        name = "cuda"
        tune_db = None
        bench_compile_timeout_s = 1.0
        bench_run_timeout_s = 1.0

        def __init__(self, **_kwargs):
            pass

        async def benchmark_compare_async(self, _graph, **kwargs):
            seen["want_ref"] = kwargs["want_ref"]
            return {
                "results": {},
                "result": None,
                "captured": False,
                "torch_available": False,
                "accuracy_error": returned["accuracy_error"],
                "run_io": ({"x": [1.0]}, {"y": [1.0]}),
                "greedy_error": None,
                "reference_run_us": None,
            }

        async def aclose_async_worker(self):
            pass

    async def fake_isolated(*_args, **_kwargs):
        seen["isolated"] = True
        return None

    monkeypatch.setattr(Pipeline, "build", lambda _passes: FakePipeline())
    monkeypatch.setattr(run_module, "_bench_greedy_isolated", fake_isolated)
    monkeypatch.setattr(run_module, "_print_kernel_stats", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_module, "_record_greedy_pick", lambda *_args, **_kwargs: seen.__setitem__("recorded", True))
    monkeypatch.setattr(run_module, "_record_bench_evidence", lambda *_args, **_kwargs: None)

    run_module._handle_run_ir(args, FakeBackend, _FakeDump)

    assert seen["want_ref"] is True, "the greedy outputs are the only reference a twinless target has"
    assert seen.get("isolated") is True, "the recorded row's per-kernel timings come from the isolated re-bench"
    assert seen.get("recorded") is True

    # And a row whose ANSWER --strict rejected is not recorded: on sm_70 a wrong answer can run faster than the
    # right neighbour, and a recorded row outranks every later compile.
    seen.pop("recorded")
    args.strict_correctness = True
    returned["accuracy_error"] = "strict eager correctness failed: output 'y' exceeds rtol=0.001"
    with pytest.raises(SystemExit) as exit_code:
        run_module._handle_run_ir(args, FakeBackend, _FakeDump)
    assert exit_code.value.code == 1
    assert "recorded" not in seen

    # A strict file walk also needs the reference when only the cut pieces have measured rows.
    # It has no whole-target pin and is not recording anything.
    args.record_greedy = False
    returned["accuracy_error"] = None
    seen.clear()
    with pytest.raises(SystemExit) as exit_code:
        run_module._handle_run_ir(args, FakeBackend, _FakeDump)
    assert exit_code.value.code == 1  # the fake worker has no captured timing
    assert seen == {"want_ref": True}


@pytest.mark.parametrize("card,cap", [("NVIDIA GeForce RTX 4090", (8, 9)), ("NVIDIA A100-SXM4-40GB", (8, 0))])
def test_recorded_greedy_pick_is_picked_again_under_strict_evidence(tmp_path, card, cap, monkeypatch):
    """The kernel set a compile picked, recorded as the DB holds it — a routing row per kernel-set decision it took
    and a measured row per kernel — is evidence enough: those rows alone yield the same kernels with the same rows
    under strict evidence, with no prior and no tune DB."""
    monkeypatch.setenv("EMMY_FAST_MATH", "0")
    from emmy import config

    path = tmp_path / "working-route.json"
    document = inventory_document(_norm_matmul_graph(), cap, gpu_name=card)
    [row] = document.rows
    document = replace(document, rows=[replace(row, name="working.route", pins={"FAST_MATH": False})])
    document.dump(path)
    picked, taken = _compile_pinned(document, {"FAST_MATH": False, **_CUT}, cap, card)
    rows = _picked(picked)
    # The routing row's cut first; the pieces may take further kernel-set decisions of their own (a cross-CTA split
    # of the residual), each recorded as a routing row of its own kernel.
    assert len(rows) >= 2 and taken[0][1] == _CUT

    written = record_greedy_pick(
        path,
        "working.route",
        decisions=taken,
        kernels=[(node.op, 1.0, 2.0, None) for node in _cuda_nodes(picked)],
        reference_backend="same-input-greedy",
    )

    reloaded = GoldenFile.load(path)
    added = [r for r in reloaded.rows if r.name in written]
    assert len(added) == len(rows) and len(reloaded.routing) == len(taken)
    assert all(r.measured and r.pins == {"FAST_MATH": False} for r in added)
    [target] = reloaded.targets()
    with sole_evidence([reloaded]), pinned_knobs({"FAST_MATH": False}), config.strict_evidence_override(True):
        again = Pipeline.build(CUDA_PASSES).run(target.program({}), ctx=Context.from_target(cap, gpu_name=card), db=None)
    assert _picked(again) == rows


def test_recorded_composed_pick_is_picked_again_under_strict_evidence(tmp_path, monkeypatch):
    """A pinned compile consumes every scoped PLACE pin that resolves on one kernel as ONE composed decision, and
    ``--record-greedy`` records it as one routing row naming every seam. Those rows are evidence enough for the same
    composed cut under strict evidence — the cut pass offers the composed arm the row spells beside its single
    seams on the deploy that reads it."""
    monkeypatch.setenv("EMMY_FAST_MATH", "0")
    from emmy import config

    path = tmp_path / "working-route.json"
    document = inventory_document(_norm_matmul_graph(), (8, 9))
    [row] = document.rows
    document = replace(document, rows=[replace(row, name="working.route", pins={"FAST_MATH": False})])
    document.dump(path)
    both = {**_CUT, "PLACE@inner.1/map.3/map": "cut"}
    picked, taken = _compile_pinned(document, {"FAST_MATH": False, **both})
    rows = _picked(picked)
    assert len(rows) >= 3 and taken[0][1] == both, "one composed decision minting at least two pieces"

    record_greedy_pick(
        path,
        "working.route",
        decisions=taken,
        kernels=[(node.op, 1.0, 2.0, None) for node in _cuda_nodes(picked)],
        reference_backend="same-input-greedy",
    )
    reloaded = GoldenFile.load(path)
    [target] = reloaded.targets()
    with sole_evidence([reloaded]), pinned_knobs({"FAST_MATH": False}), config.strict_evidence_override(True):
        again = Pipeline.build(CUDA_PASSES).run(target.program({}), ctx=Context.from_target((8, 9)), db=None)
    assert _picked(again) == rows


def test_record_replays_a_cut_pinned_on_a_cut_piece(tmp_path, monkeypatch):
    """A composed route leaves its fresh pieces eligible for their own recorded cut decisions."""
    from emmy import config  # noqa: PLC0415
    from emmy.commands.compile import golden_row, selected_decisions
    from emmy.commands.run import _applied_place_pins, _placement_knob_dicts
    from emmy.compiler.pipeline.search.pins import unreproducible_pin_flag

    monkeypatch.setenv("EMMY_FAST_MATH", "0")
    path = tmp_path / "working-route.json"
    document = inventory_document(_norm_matmul_graph(), (8, 9))
    [row] = document.rows
    document = replace(document, rows=[replace(row, name="working.route", pins={"FAST_MATH": False})])
    document.dump(path)
    both = {**_CUT, "PLACE@inner.1/map.3/map": "cut"}
    _, [(_, _, pieces)] = _compile_pinned(document, {"FAST_MATH": False, **both})
    token = pieces[0].name.rsplit("__place_", 1)[1]
    picked, taken = _compile_pinned(document, {"FAST_MATH": False, **both, f"PLACE@place_{token}/map.1/reduce": "cut"})
    assert len(taken) == 2, "the pin on the piece cuts it again as a decision of its own"

    record_greedy_pick(
        path,
        "working.route",
        decisions=taken,
        kernels=[(node.op, 1.0, 2.0, None) for node in _cuda_nodes(picked)],
        reference_backend="same-input-greedy",
    )
    reloaded = GoldenFile.load(path)
    assert len(reloaded.routing) == 2
    [target] = reloaded.targets()
    with sole_evidence([reloaded]), pinned_knobs({"FAST_MATH": False}), config.strict_evidence_override(True):
        again = Pipeline.build(CUDA_PASSES).run(target.program({}), ctx=Context.from_target((8, 9)), db=None)
    assert _picked(again) == _picked(picked)
    assert reloaded.routing[1].arm == {"PLACE": "cut"}, "the child has one site, so its recorded cut is bare"
    samples = [golden_row(reloaded, row) for row in reloaded.rows if row.measured]
    route = selected_decisions(SimpleNamespace(golden_configs=samples, pin_route=True))
    with sole_evidence([reloaded]), pinned_knobs({"FAST_MATH": False, **route}), config.strict_evidence_override(True):
        pinned = Pipeline.build(CUDA_PASSES).run(target.program({}), ctx=Context.from_target((8, 9)), db=None)
    assert _picked(pinned) == _picked(picked), "--pin-route must apply the bare cut to its recorded child"
    assert (
        unreproducible_pin_flag(route, [{}], placement_knobs=_placement_knob_dicts(pinned), applied_place_pins=_applied_place_pins(pinned))
        is None
    )


def _branching_route_document():
    from emmy.compiler.pipeline.search.db import RoutingRow
    from emmy.compiler.pipeline.search.golden.restamp import definition

    document = inventory_document(_norm_matmul_graph(), (8, 9))
    both = {**_CUT, "PLACE@inner.1/map.3/map": "cut"}
    _, [(_, _, pieces)] = _compile_pinned(document, {"FAST_MATH": False, **both})
    token = pieces[0].name.rsplit("__place_", 1)[1]
    _, taken = _compile_pinned(document, {"FAST_MATH": False, **both, f"PLACE@place_{token}/map.1/reduce": "cut", **_SPLIT})
    for parent, arm, pieces in taken:
        stored = document.add_kernel(definition(parent, parent.name))
        children = [document.add_kernel(definition(piece, piece.name)) for piece in pieces]
        document.add_routing(RoutingRow(stored.ref, arm, tuple(child.ref for child in children)))
    assert len(document.routing) == 3
    return document


def test_record_validates_sibling_routes_in_one_replay(monkeypatch):
    from emmy.compiler.pipeline.search.golden import working

    document = _branching_route_document()
    replays = []
    mint = working.mint

    def observed(*args, **kwargs):
        result = mint(*args, **kwargs)
        replays.append(result)
        return result

    monkeypatch.setattr(working, "mint", observed)
    working._refuse_unreplayable(document, document.routing)
    assert len(replays) == 1, "the common root lowers once for both child decisions"
    assert all(any(taken == route and same for taken, same, _ in replays[0]) for route in document.routing)


@pytest.mark.parametrize("branch", [1, 2])
@pytest.mark.parametrize("corruption", ["missing_arm", "child_count"])
def test_record_refuses_each_invalid_sibling_route(branch, corruption):
    from emmy.compiler.pipeline.search.golden import working

    document = _branching_route_document()
    routes = list(document.routing)
    route = routes[branch]
    routes[branch] = (
        replace(route, arm={"PLACE@missing": "cut"})
        if corruption == "missing_arm"
        else replace(route, children=(*route.children, route.children[0]))
    )
    with pytest.raises(ValueError, match="restamp would drop it"):
        working._refuse_unreplayable(document, routes)


def test_record_replays_conflicting_route_alternatives_separately(monkeypatch):
    from emmy.compiler.pipeline.search.db import RoutingRow
    from emmy.compiler.pipeline.search.golden import working
    from emmy.compiler.pipeline.search.golden.restamp import definition

    document = _branching_route_document()
    _, taken = _compile_pinned(document, {"FAST_MATH": False, **_CUT, **_SPLIT})
    parent, arm, pieces = taken[0]
    stored = document.add_kernel(definition(parent, parent.name))
    children = [document.add_kernel(definition(piece, piece.name)) for piece in pieces]
    document.add_routing(RoutingRow(stored.ref, arm, tuple(child.ref for child in children)))
    replays = []
    mint = working.mint

    def observed(*args, **kwargs):
        result = mint(*args, **kwargs)
        replays.append(result)
        return result

    monkeypatch.setattr(working, "mint", observed)
    working._refuse_unreplayable(document, document.routing)
    assert len(replays) == 2, "different decisions on the same parent remain independent replays"
    assert all(any(taken == route and same for replay in replays for taken, same, _ in replay) for route in document.routing)


def test_run_records_the_greedy_pick_of_an_embedded_golden(monkeypatch, tmp_path):
    """``run --golden PATH --realization NAME --bench --record-greedy``: the greedy row compiles with the file as its
    golden evidence (here the routing row and its pieces' rows, so the cut is taken), and after the bench the kernel
    set it picked is written back — a routing row per kernel-set decision, a row per kernel with its isolated launch
    timing, the greedy comparison row as every reference — while the per-kernel perf rows every embedded-golden
    bench records by default are recorded too."""
    monkeypatch.setenv("EMMY_FAST_MATH", "0")
    from emmy.commands import run as run_module
    from emmy.commands.compile import resolve_golden_arg
    from emmy.compiler import target as target_mod

    path = tmp_path / "working-route.json"
    before = _working_placement_route(path)
    args = _run_args(path, realization="working.route", record=False, record_greedy=True, pin_route=True, strict_correctness=False)
    resolve_golden_arg(args)

    def launches(graph, ms_per_launch):
        n = len(_cuda_nodes(graph))
        # ``time_ms`` is the per-launch median; the samples' minimum sits below it, so a record that took the
        # minimum would miss the ``[1, 2, ...]`` the assertion below expects.
        per_launch = [
            SimpleNamespace(idx=i, time_ms=ms_per_launch * (i + 1), samples=[ms_per_launch * (i + 1) * f for f in (0.5, 1.0, 1.5)])
            for i in range(n)
        ]
        total = sum(launch.time_ms for launch in per_launch)
        return SimpleNamespace(min_ms=total, time_ms=total, e2e_min_ms=None, captured=True, num_launches=n, per_launch=per_launch)

    class FakeBackend:
        name = "cuda"
        tune_db = None
        bench_compile_timeout_s = 1.0
        bench_run_timeout_s = 1.0

        def __init__(self, **_kwargs):
            pass

        async def benchmark_compare_async(self, graph, **_kwargs):
            return {
                "results": {"Emmy": 1.0},
                "result": launches(graph, 0.002),
                "captured": True,
                "torch_available": False,
                "accuracy_error": None,
                "run_io": ({"x": object()}, {"y": object()}),
                "greedy_error": None,
                "reference_run_us": None,
            }

        async def aclose_async_worker(self):
            pass

    async def fake_isolated(_backend, compiled, *, warmup, iters, ref=None, ref_key=None):
        sample = SimpleNamespace(name="greedy (isolated)", knobs={}, shape=None, dynamic=None)
        return run_module._GoldenBench(sample, compiled, launches(compiled, 0.001), [], "ok")

    async def fake_pinned(_backend, _source, _pins, **_kwargs):
        return []

    recorded = {}
    monkeypatch.setattr(run_module, "_bench_greedy_isolated", fake_isolated)
    monkeypatch.setattr(run_module, "_bench_golden_variants", fake_pinned)
    monkeypatch.setattr(run_module, "_print_kernel_stats", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_module, "_record_bench_evidence", lambda _args, benches, iso: recorded.update(benches=benches, iso=iso))
    target_mod.set_target((8, 9))
    try:
        run_module._handle_run_ir(args, FakeBackend, _FakeDump)
    finally:
        target_mod.set_target(None)

    assert recorded["benches"] == [] and recorded["iso"].status == "ok"
    document = GoldenFile.load(path)
    assert document.routing[0].arm == _CUT and len(document.routing) == 2, "the cut, then the residual's cross-CTA split"
    assert re.fullmatch(r"g\d+[ak]", next(iter(document.routing[1].arm.values())))
    measured = [row for row in document.rows if row.measured]
    assert len(measured) >= 2 and len(document.rows) == len(before.rows)
    # Each kernel's row at its isolated launch timing, the greedy comparison row as its reference.
    assert sorted(row.measurements.emmy_us for row in measured) == [float(i + 1) for i in range(len(measured))]
    assert all(row.measurements.reference_us == 2 * row.measurements.emmy_us for row in measured)
    assert all(row.measurements.reference_backend == "same-input-greedy" for row in measured)


def test_run_files_a_hung_greedy_kernel_as_bench_fail_evidence(monkeypatch, tmp_path):
    """A greedy pick that hangs the watchdog is evidence too: the kernel the watchdog NAMED earns a ``bench_fail``
    perf row in the tune DB and no other kernel does, so the next compile's evidence pick disqualifies that arm
    instead of electing the identical route and hanging again."""
    monkeypatch.setenv("EMMY_FAST_MATH", "0")
    from emmy.commands import run as run_module
    from emmy.commands.compile import resolve_golden_arg
    from emmy.compiler import target as target_mod
    from emmy.compiler.backend.cuda.program import BenchWorkerJobError
    from emmy.compiler.pipeline.search.db import SearchDB
    from emmy.compiler.wire import kernel_bindings

    db_path = tmp_path / "autotune.db"
    monkeypatch.setenv("EMMY_TUNE_DB", str(db_path))
    monkeypatch.setenv("EMMY_NVCC_FLAGS", "")  # the deployable regime: perf evidence is recorded only there
    path = tmp_path / "working-route.json"
    _working_placement_route(path)
    args = _run_args(path, realization="working.route", record=False, record_greedy=True, pin_route=True, strict_correctness=False)
    resolve_golden_arg(args)
    seen = {}

    class FakeBackend:
        name = "cuda"
        tune_db = None
        bench_compile_timeout_s = 1.0
        bench_run_timeout_s = 2.0

        def __init__(self, **_kwargs):
            pass

        async def benchmark_compare_async(self, graph, **_kwargs):
            seen["nodes"] = _cuda_nodes(graph)
            culprit = seen["nodes"][-1].op.kernel_name
            hang = f"kernel '{culprit} (iter 0)' did not complete within 60000 ms — variant marked bench_fail"
            raise BenchWorkerJobError(f'bench worker error: HungKernelError("{hang}")')

        async def aclose_async_worker(self):
            pass

    monkeypatch.setattr(run_module, "_print_kernel_stats", lambda *_args, **_kwargs: None)
    target_mod.set_target((8, 9))
    try:
        with pytest.raises(SystemExit):
            run_module._handle_run_ir(args, FakeBackend, _FakeDump)
        probed = Context.probe()
    finally:
        target_mod.set_target(None)

    nodes = seen["nodes"]
    assert len(nodes) >= 2, "the route must hold an innocent kernel beside the culprit"
    db = SearchDB(db_path)
    try:
        tiles = {n.op.kernel_name: (kernel_tile(n.op), dict(n.op.knobs or {})) for n in nodes}
        rows = {
            name: db.lookup_perf(
                probed, tile.identity_key(structural=False, with_io=True), bindings=kernel_bindings(tile), knobs=knobs, backend="cuda"
            )
            for name, (tile, knobs) in tiles.items()
        }
    finally:
        db.close()
    filed = {name: row.status for name, row in rows.items() if row is not None}
    assert filed == {nodes[-1].op.kernel_name: "bench_fail"}, "only the kernel the watchdog named is evidence"
    assert rows[nodes[-1].op.kernel_name].stats.median == pytest.approx(2.0e6), "priced at the run budget's fail sentinel"


def test_run_skips_pinned_rebench_of_the_same_election_after_a_greedy_hang(monkeypatch, tmp_path):
    """An embedded Loop golden has no Torch twin, so its greedy job can complete the same-input reference and only
    then have its repeated timing cross the watchdog. The bench_fail evidence for the failed election is recorded
    BEFORE any pinned compile begins, and the knob-less automatic pin is not re-benched: with nothing pinning it away
    from the greedy compile's own choices it would re-elect and re-hang the identical program."""
    from emmy.commands import run as run_module
    from emmy.commands.compile import resolve_golden_arg

    path = tmp_path / "working.json"
    _working_loop(path)  # default state: an untuned seed realization with no recorded knobs
    args = _run_args(path, realization="working.relu", record=False, record_greedy=False, strict_correctness=False)
    resolve_golden_arg(args)
    assert args.golden_configs and not args.golden_configs[0].knobs, "the seed must carry no knobs"

    class FakeBackend:
        name = "cuda"
        tune_db = None
        bench_compile_timeout_s = 1.0
        bench_run_timeout_s = 1.0

        def __init__(self, **_kwargs):
            pass

        async def benchmark_compare_async(self, _graph, **_kwargs):
            return {
                "results": {},
                "result": None,
                "captured": False,
                "torch_available": False,
                "accuracy_error": None,
                "run_io": ({"x": object()}, {"y": object()}),
                "greedy_error": "HungKernelError: repeated timing crossed the watchdog",
                "reference_run_us": 4_000_000.0,
            }

        async def aclose_async_worker(self):
            pass

    calls = []

    def fake_record_failure(*_args, **_kwargs):
        calls.append("record")

    async def fake_bench_golden_variants(_backend, _source, pinned, **_kwargs):
        calls.append(("bench_golden_variants", list(pinned)))
        return []

    async def fail_if_isolated(*_args, **_kwargs):
        raise AssertionError("a failed greedy timing must not be re-benched or made eligible")

    monkeypatch.setattr(run_module, "_record_greedy_failure", fake_record_failure)
    monkeypatch.setattr(run_module, "_bench_golden_variants", fake_bench_golden_variants)
    monkeypatch.setattr(run_module, "_bench_greedy_isolated", fail_if_isolated)
    monkeypatch.setattr(run_module, "_print_kernel_stats", lambda *_args, **_kwargs: None)

    with pytest.raises(SystemExit):
        run_module._handle_run_ir(args, FakeBackend, _FakeDump)

    assert calls[0] == "record", "the bench_fail row must be recorded before the pinned walk starts"
    assert calls[1] == ("bench_golden_variants", []), "the knob-less seed pin must not be re-compiled and re-benched"


def test_pin_route_pins_the_decisions_the_named_rows_agree_on(monkeypatch):
    """Under ``--pin-route`` the kernel-set decisions that mint the named rows' kernels are one hand pin for their
    compile — a cut, a cross-CTA split — never a schedule knob; rows whose routes disagree pin nothing; without the
    flag nothing is pinned. It is the hand pin ``EMMY_KNOBS`` publishes, so one already set on a seam with another
    value is refused, and one that agrees is not."""
    from emmy.commands.compile import selected_decisions

    monkeypatch.delenv("EMMY_KNOBS", raising=False)
    cut = SimpleNamespace(route=_CUT)
    split = SimpleNamespace(route={**_CUT, "REDUCE": "g2k"})
    fused = SimpleNamespace(route={"PLACE@inner.1/map": "fuse"})
    assert selected_decisions(SimpleNamespace(golden_configs=[cut, split], pin_route=True)) == {**_CUT, "REDUCE": "g2k"}
    assert selected_decisions(SimpleNamespace(golden_configs=[cut, split])) == {}
    assert selected_decisions(SimpleNamespace(golden_configs=[cut, fused], pin_route=True)) == {}
    assert selected_decisions(SimpleNamespace(golden_configs=[SimpleNamespace(route={})], pin_route=True)) == {}
    monkeypatch.setenv("EMMY_KNOBS", "PLACE@inner.1/map=cut")
    assert selected_decisions(SimpleNamespace(golden_configs=[cut], pin_route=True)) == _CUT
    monkeypatch.setenv("EMMY_KNOBS", "PLACE@inner.1/map=fuse")
    with pytest.raises(SystemExit):
        selected_decisions(SimpleNamespace(golden_configs=[cut], pin_route=True))


@pytest.mark.parametrize("bare", (False, True))
def test_recorded_route_addresses_successive_remainders_and_cut_producers(monkeypatch, bare: bool) -> None:
    from emmy.commands.compile import _route_pins

    names = {
        "root": "k",
        "remainder": "k",
        "producer": "k__place_aaaa",
        "producer_remainder": "k__place_aaaa",
        "nested": "k__place_aaaa__place_bbbb",
        "final": "k__place_aaaa__place_bbbb__place_cccc",
    }
    path = [
        SimpleNamespace(parent="root", arm={"PLACE@map.1/map": "cut"}, children=("remainder",)),
        SimpleNamespace(parent="remainder", arm={"PLACE@map.2/inner": "cut"}, children=("producer",)),
        SimpleNamespace(parent="producer", arm={"PLACE@map.1/reduce": "cut"}, children=("producer_remainder",)),
        SimpleNamespace(parent="producer_remainder", arm={"PLACE@map.2/inner": "cut"}, children=("nested",)),
        SimpleNamespace(parent="nested", arm={"PLACE@map.3/inner": "cut"}, children=("final",)),
    ]
    if bare:
        for decision in path:
            decision.arm = {"PLACE": "cut"}
    import importlib

    restamp = importlib.import_module("emmy.compiler.pipeline.search.golden.restamp")
    fresh = {ref: SimpleNamespace(name=name) for ref, name in names.items()}
    document = SimpleNamespace(
        path_to=lambda ref: path,
        kernel=lambda ref: SimpleNamespace(name=names[ref] + ("__place_obsolete" if ref in {"producer", "nested"} else "")),
        compute_cap=(8, 9),
        gpu_name="",
    )
    monkeypatch.setattr(restamp, "mint", lambda *_args, **_kwargs: [(step, True, [fresh[c] for c in step.children]) for step in path])

    expected = {
        "PLACE@map.1/map": "cut",
        "PLACE@step.1/map.2/inner": "cut",
        "PLACE@place_aaaa/map.1/reduce": "cut",
        "PLACE@place_aaaa/step.1/map.2/inner": "cut",
        "PLACE@place_bbbb/map.3/inner": "cut",
    }
    if bare:
        expected = dict.fromkeys(("PLACE", "PLACE@step.1", "PLACE@place_aaaa", "PLACE@place_aaaa/step.1", "PLACE@place_bbbb"), "cut")
    assert _route_pins(document, "final") == expected


def test_ab_rows_compile_under_the_pinned_route(monkeypatch):
    """An ``--ab`` row under ``--pin-route`` compiles the same kernel set as the greedy it is compared with: the route
    rides every row, and the row's own knobs win where they name the same key."""
    from emmy.commands.run import _pinned_samples_for_ir, _sample_replay_knobs

    monkeypatch.delenv("EMMY_KNOBS", raising=False)
    route = SimpleNamespace(name="r", pins={"FAST_MATH": False}, route=_CUT, knobs={})
    args = SimpleNamespace(golden_configs=[route], pin_route=True, ab=["REDUCE=g8k"], dynamic=None)
    (_, ab) = _pinned_samples_for_ir(args, embedded=object())
    assert _sample_replay_knobs(ab) == {**_CUT, "REDUCE": "g8k"}
    args.pin_route = False
    (_, ab) = _pinned_samples_for_ir(args, embedded=object())
    assert _sample_replay_knobs(ab) == {"REDUCE": "g8k"}


@pytest.mark.parametrize(
    ("proposed", "realized"),
    [
        ({}, {"LOOPIFY": "0"}),
        ({"LOOPIFY": "0x0"}, {"LOOPIFY": "0"}),
        ({"LOOPIFY": "0x4"}, {"LOOPIFY": 4}),
        ({"VECTORIZE_LOADS": "off"}, {"VECTORIZE_LOADS": False}),
    ],
)
def test_record_replaces_a_proposal_with_explicit_off_defaults(proposed, realized):
    knobs = {"WORK": "t256", "REDUCE": "coop-t/v2", **proposed}
    proposal = Row(name="proposed", kernel="k", pins={"FAST_MATH": False}, knobs=knobs, note="Keep this name.")
    document = GoldenFile(compute_cap=(8, 9), rows=[proposal])
    recorded = replace(
        proposal,
        name="generated",
        knobs={**knobs, "TILE": "", "RASTER": "", "STAGE": "", "SHARED_CARRY": "0", **realized},
        measurements=Measurements(emmy_us=1.0, reference_us=2.0, reference_backend="same-input-greedy"),
        note=None,
    )

    merged = document.upsert_row(recorded)

    assert document.rows == [merged]
    assert merged.name == proposal.name and merged.note == proposal.note
    assert merged.measurements == recorded.measurements and merged.knobs == recorded.knobs


@pytest.mark.parametrize("decision", [{"LOOPIFY": "4"}, {"SHARED_CARRY": "1"}, {"VECTORIZE_LOADS": True}])
def test_record_does_not_conflate_later_decisions(decision):
    proposal = Row(name="proposed", kernel="k", knobs={"WORK": "t256"})
    document = GoldenFile(compute_cap=(8, 9), rows=[proposal])
    recorded = replace(
        proposal,
        name="different",
        knobs={**proposal.knobs, **decision},
        measurements=Measurements(emmy_us=1.0, reference_us=2.0, reference_backend="same-input-greedy"),
    )

    assert document.upsert_row(recorded) == recorded
    assert document.rows == [proposal, recorded]


def test_record_greedy_writes_the_regime_the_compile_measured_when_both_regimes_seed_the_name(tmp_path, monkeypatch):
    """A name seeded in both precision regimes: the recorded row takes the regime the compile ran in, not whichever
    seed comes first in the file."""
    path = tmp_path / "working.json"
    document = _working_loop(path, pins={"FAST_MATH": False})
    with GoldenFile.edit(path) as editing:
        editing.rows.append(replace(editing.rows[0], pins={"FAST_MATH": True}))
    picked, _taken = _compile_pinned(document, {"FAST_MATH": True})
    [node] = _cuda_nodes(picked)
    monkeypatch.setenv("EMMY_FAST_MATH", "1")

    record_greedy_pick(path, "working.relu", decisions=[], kernels=[(node.op, 1.0, 2.0, None)], reference_backend="same-input-greedy")

    assert [row.pins for row in GoldenFile.load(path).rows if row.measured] == [{"FAST_MATH": True}]


def test_the_seed_row_of_a_name_both_regimes_share_is_the_live_regimes(tmp_path, monkeypatch):
    """``--record-greedy`` writes the whole pick's latency onto the seed row of the regime it measured."""
    from emmy.compiler.pipeline.search.golden.working import seed_row

    path = tmp_path / "working.json"
    _working_loop(path, pins={"FAST_MATH": False})
    with GoldenFile.edit(path) as editing:
        editing.rows.append(replace(editing.rows[0], pins={"FAST_MATH": True}))
    for raw, fast in (("1", True), ("0", False)):
        monkeypatch.setenv("EMMY_FAST_MATH", raw)
        assert seed_row(GoldenFile.load(path), "working.relu").pins == {"FAST_MATH": fast}


def test_record_greedy_cold_rows_do_not_replace_hot_measurements(tmp_path):
    path = tmp_path / "working.json"
    document = _working_loop(path, pins={"FAST_MATH": False})
    picked, _ = _compile_pinned(document, {"FAST_MATH": False})
    [node] = _cuda_nodes(picked)
    for cold, latency in ((False, 1.0), (True, 3.0)):
        with pinned_knobs({"FAST_MATH": False, "COLD_CACHE": cold}):
            record_greedy_pick(
                path,
                "working.relu",
                decisions=[],
                kernels=[(node.op, latency, latency, None)],
                reference_backend="same-input-greedy",
            )
    rows = [row for row in GoldenFile.load(path).rows if row.measured]
    assert {(bool(row.pins.get("COLD_CACHE")), row.measurements.emmy_us) for row in rows} == {(False, 1.0), (True, 3.0)}
