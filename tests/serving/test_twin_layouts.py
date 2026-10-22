"""Serving, the serving audit and a golden's record run build a layer's twins narrowest first, each under the weight
layouts the twins before it bound (CPU-only)."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from emmy.compiler.context import Context
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.frontend.ir import TransposeOp
from emmy.compiler.ir.stmt import Load, Write
from emmy.compiler.ir.tile import OutputSpec, Placement, TileOp
from emmy.compiler.pipeline import Pipeline
from emmy.serving.twins import twins_in_build_order
from tests.compiler.terms import contraction


def _traced(width, source: str | None) -> Graph:
    """A traced program with ``width`` tokens (``"num_tokens"``: symbolic) binding checkpoint tensor ``source``."""
    from emmy.compiler.dim import Dim

    graph = Graph()
    tokens = Dim(width) if isinstance(width, str) else width
    graph.add_node(InputOp(), [], Tensor("x", (tokens, 8), "f32"), node_id="x")
    if source is not None:
        graph.add_node(ConstantOp(name="w", source_path=source, source_shape=(8, 8)), [], Tensor("w", (8, 8), "f32"), node_id="w")
    graph.inputs, graph.outputs = ["x"], ["x"]
    return graph


def test_twins_bind_one_layers_tensors_and_build_narrowest_first():
    """Twins are grouped by the checkpoint tensors they bind and ordered by their own sizes, not by their names."""
    graphs = {
        "a": _traced(64, "layers.0.w"),
        "b": _traced("num_tokens", "layers.3.w"),
        "c": _traced(1, "layers.0.w"),
        "d": _traced(16, "layers.3.w"),
        "e": _traced(16, "layers.0.w"),
        "f": _traced(1, None),
        "g": _traced(1, None),
    }
    order = list(twins_in_build_order(graphs, lambda _graph: set()))
    assert [key for key, _ in order] == ["c", "e", "a", "d", "b", "f", "g"]
    scopes = dict(order)
    assert scopes["c"] is scopes["a"] and scopes["d"] is scopes["b"]
    assert scopes["c"] is not scopes["d"] and scopes["f"] is not scopes["g"]


def _constants(*names):
    def constant(name):
        transposed = not name.endswith("__source")
        return ConstantOp(name="w", load_ops=(TransposeOp((-2, -1)),) if transposed else (), source_path="w", source_shape=(4, 8))

    return SimpleNamespace(loadable_constants=lambda: [(name, constant(name)) for name in names], inputs=[], outputs=[])


def test_the_audit_compiles_a_later_twin_under_the_layout_an_earlier_twin_bound(caplog):
    from emmy.commands.eval import audit_twins
    from emmy.compiler.pipeline.passes.tile import _layout

    seen = []

    def compile_twin(name, _graph):
        seen.append((name, dict(_layout._SETTLED.get() or {})))
        if name == "gdn64":
            raise RuntimeError("no row")
        return _constants("w__source")

    twins = {"gdn64": _traced(64, "w"), "gdn16": _traced(16, "w"), "gdn1": _traced(1, "w")}
    with caplog.at_level(logging.INFO, logger="emmy.commands.eval"):
        broken = audit_twins(twins, compile_twin, "std", lambda _graph: set())
    assert seen == [("gdn1", {}), ("gdn16", {"w": "source"}), ("gdn64", {"w": "source"})]
    assert broken == 1
    assert "std: gdn16 compiles with w=source settled by gdn1" in caplog.text


def test_a_fork_less_reader_in_a_wider_twin_settles_the_family_folded():
    """``gdn16`` holds a reader the cut pass offers no layout fork for (a 1-D read), so ``w`` stays folded there. The
    family settles ``w`` folded before ``gdn1``, which would otherwise pick source, decides: no twin is refused."""
    from dataclasses import replace

    from emmy.commands.eval import audit_twins
    from emmy.compiler.pipeline.passes.tile import _layout

    gdn16 = _twin(16)
    reader = gdn16.nodes["y"].op
    axis_k = Axis("k", 8)
    flat = replace(
        reader,
        name="z",
        op=contraction(axis_k, Load(name="xv", input="x", index=(Var("k"),)), (Load(name="wv", input="w", index=(Var("k"),)), "acc")),
        output_specs=(OutputSpec(Write(output="z", index=(Var("n"),), value="acc")),),
    )
    gdn16.add_node(flat, ["x", "w"], outputs=(Tensor("z", (16,), "f32"),), node_id="z")
    gdn16.outputs = ["y", "z"]
    assert _layout.fork_less_reads(gdn16) == {"w"}
    assert _layout.fork_less_reads(_twin(4)) == set()
    seen = []

    def compile_twin(name, _graph):
        settled = dict(_layout._SETTLED.get() or {})
        seen.append((name, settled))
        return _constants("w" if settled.get("w") == "folded" else "w__source")

    broken = audit_twins({"gdn16": gdn16, "gdn1": _twin(4)}, compile_twin, "std", _layout.fork_less_reads)
    assert broken == 0
    assert seen == [("gdn1", {"w": "folded"}), ("gdn16", {"w": "folded"})]


def _linear(width: int) -> Graph:
    """A traced width twin: ``x @ weight.T`` over ``width`` tokens, binding checkpoint tensor ``weight``."""
    import torch

    from emmy.compiler.trace.torch import trace_module

    torch.manual_seed(0)
    return trace_module(torch.nn.Linear(8, 4, bias=False), (torch.randn(width, 8),))


def test_the_record_run_compiles_each_target_with_its_familys_settled_layouts(monkeypatch):
    """The golden target walk compiles each target through the golden replay path (``_handle_run_ir``) with the
    layouts its family settled so far, and records what that compile bound."""
    from emmy.commands import run
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.pipeline.dump import CompilerDump
    from emmy.compiler.pipeline.passes.tile import _layout
    from emmy.compiler.pipeline.search import golden
    from emmy.serving.twins import graph_layouts

    rows = {
        ("t16", ()): [SimpleNamespace(name="gdn16.k_linear.m16", kernel="t16")],
        ("t1", ()): [SimpleNamespace(name="gdn1.k_linear.m1", kernel="t1")],
    }
    programs = [_linear(16), _linear(1)]
    document = SimpleNamespace(
        rows=[row for group in rows.values() for row in group],
        target_rows=lambda: rows,
        targets=lambda: [SimpleNamespace(ref=ref, traced=index) for index, ref in enumerate(("t16", "t1"))],
        program=lambda index: programs[index],
        shared_regime=lambda rows=None: {},
        compute_cap=(8, 9),
        gpu_name=None,
    )
    monkeypatch.setattr(golden.GoldenFile, "load", staticmethod(lambda _path: document))

    async def no_bench(*_args, **_kwargs):
        return None

    monkeypatch.setattr(run, "bench_lowered_vs_torch", no_bench)
    seen = []

    def run_once(args):  # the golden branch of ``_handle_run_once``, past its GPU check and row resolution
        args._golden_graph = programs[0 if args.realization.startswith("gdn16") else 1]
        settled = dict(_layout._SETTLED.get() or {})
        run._handle_run_ir(args, CudaBackend, CompilerDump)
        seen.append(
            (
                args.realization,
                settled,
                sorted(name for name, transposed in graph_layouts(args._compiled) if transposed or name.endswith("__source")),
            )
        )

    monkeypatch.setattr(run, "_handle_run_once", run_once)
    args = SimpleNamespace(
        input=None,
        code=None,
        ir=None,
        json=None,
        golden="g.json",
        bench=False,
        dump_dir=None,
        debug=False,
        dynamic=None,
        seed=0,
        warmup=1,
        iters=1,
        bench_backends="",
        strict_correctness=False,
        ab=None,
    )
    run._run_golden_targets(args)
    (narrow, first, bound), (wide, then, wide_bound) = seen
    assert (narrow, first, wide) == ("gdn1.k_linear.m1", {}, "gdn16.k_linear.m16")
    (weight,) = bound
    layout = "source" if weight.endswith("__source") else "folded"
    assert then == {weight.removesuffix("__source"): layout}
    assert wide_bound == bound


def _twin(n: int) -> Graph:
    """A traced width twin: one kernel reading the transposed weight ``w`` at output width ``n``."""
    axis_n, axis_k = Axis("n", n), Axis("k", 8)
    tile = TileOp(
        op=contraction(
            axis_k, Load(name="xv", input="x", index=(Var("k"),)), (Load(name="wv", input="w", index=(Var("k"), Var("n"))), "acc")
        ),
        name="y",
        place=Placement(free=(axis_n,)),
        axes=(axis_n, axis_k),
        output_specs=(OutputSpec(Write(output="y", index=(Var("n"),), value="acc")),),
    )
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (8,), "f32"), node_id="x")
    graph.add_node(
        ConstantOp(name="w", load_ops=(TransposeOp((-2, -1)),), source_path="w", source_shape=(n, 8)),
        [],
        Tensor("w", (8, n), "f32"),
        node_id="w",
    )
    graph.add_node(tile, ["x", "w"], outputs=(Tensor("y", (n,), "f32"),), node_id="y")
    graph.inputs, graph.outputs = ["x"], ["y"]
    return graph


def _measure(db, graph, layout, ctx, pipeline):
    """Record one perf row for the kernel ``graph`` forms under ``layout``."""
    from emmy.compiler.pipeline.search.bench_record import kernel_row
    from emmy.compiler.pipeline.search.db import PerfStats
    from emmy.compiler.pipeline.search.golden import evidence_scope
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    with evidence_scope([]), pinned_knobs({"PLACE": "fuse", "LAYOUT@w": layout}):
        formed = pipeline.run(graph.copy(), ctx=ctx)
    node = next(node for node in formed.nodes.values() if isinstance(node.op, TileOp))
    kernel = kernel_row(node.op.with_io(formed, node), node.op.name)
    db.record_kernel(kernel)
    stats = PerfStats(median=17, min=17, max=17, mean=17, variance=0, n_samples=1)
    db.record_perf(ctx, kernel.exact_identity, bindings={}, knobs={"WORK": "t32"}, status="ok", stats=stats)


def test_a_twin_whose_rows_cover_only_the_other_layout_fails_the_audit():
    """gdn1's rows measure only its source-layout kernel, gdn16's only its folded one. Alone, gdn16 deploys folded;
    built after gdn1, as serving builds it, it must read ``w`` in source layout too, and strict evidence refuses the
    kernel no row measures — the error serving would hit."""
    from emmy import config
    from emmy.commands.eval import audit_twins
    from emmy.compiler.pipeline.search.db import SearchDB
    from emmy.compiler.pipeline.search.golden import evidence_scope
    from emmy.compiler.pipeline.search.policy.greedy import EvidenceError

    ctx, pipeline, db = Context.from_target((7, 0)), Pipeline.build(["tile/cut"]), SearchDB()
    twins = {"gdn1": _twin(4), "gdn16": _twin(16)}
    _measure(db, twins["gdn1"], "source", ctx, pipeline)
    _measure(db, twins["gdn16"], "folded", ctx, pipeline)

    def compile_twin(_name, graph):
        return pipeline.run(graph.copy(), ctx=ctx, db=db)

    with evidence_scope([]), config.strict_evidence_override(True):
        assert {name for name, _ in compile_twin("gdn16", twins["gdn16"]).loadable_constants()} == {"w"}
        with pytest.raises(EvidenceError):
            from emmy.compiler.pipeline.passes.tile._layout import fork_less_reads, settled_layouts

            with settled_layouts({"w": "source"}):
                compile_twin("gdn16", twins["gdn16"])
        assert audit_twins(twins, compile_twin, "std", fork_less_reads) == 1


def test_settled_layouts_leave_the_fork_one_arm():
    from emmy.compiler.pipeline import Match, Rule
    from emmy.compiler.pipeline.passes.tile._layout import layout_forks, settled_layouts

    graph = _twin(4)
    match = Match(graph=graph, root_node_id="y", rule=Rule(name="test", pattern=[]))
    with settled_layouts({"w": "source"}):
        assert [option.knobs for option in layout_forks(match, graph.nodes["y"])] == [{"LAYOUT@w": "source"}]
    assert len(layout_forks(match, graph.nodes["y"])) == 2


def test_the_audit_reads_a_rows_lane_from_the_regime_its_pins_spell():
    """A row that spells every precision gate and one that spells only the umbrella are the same lane; the other
    lane is not — a recorded regime must not fall out of the audit because the recorder spelled it in full."""
    from emmy.commands.eval import _lane

    umbrella = {"FAST_MATH": False}
    spelled = {"FAST_MATH": False, "FAST_EXP": False, "F16_MMA_F32_ACC": False, "FP8_MMA": False}
    assert _lane(spelled) == _lane(umbrella)
    assert _lane({"FAST_MATH": True}) != _lane(umbrella)
    assert _lane({"FAST_MATH": True, "FAST_EXP": True, "F16_MMA_F32_ACC": True, "FP8_MMA": True}) == _lane({"FAST_MATH": True})
