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
from emmy.serving.twins import twin_build_order, twin_family, twins_in_build_order
from tests.compiler.terms import contraction


def test_twins_build_narrowest_first_per_family():
    names = ["gdn64", "pre-sym", "gdn1", "pre16-global", "pre16", "gdn16", "pre1", "k_matmul_a1ae2d"]
    assert twin_build_order(names) == ["gdn1", "gdn16", "gdn64", "pre1", "pre16", "pre-sym", "pre16-global", "k_matmul_a1ae2d"]
    assert twin_family("gdn16-dense-linear@nvfp4") == "gdn-dense-linear@nvfp4"
    assert twin_family("k_matmul_a1ae2d") is None
    scopes = dict(twins_in_build_order(names))
    assert scopes["gdn1"] is scopes["gdn64"] and scopes["pre1"] is scopes["pre-sym"]
    assert scopes["pre16"] is not scopes["pre16-global"] and scopes["gdn1"] is not scopes["pre1"]


def _constants(*names):
    def constant(name):
        transposed = not name.endswith("__source")
        return ConstantOp(name="w", load_ops=(TransposeOp((-2, -1)),) if transposed else (), source_path="w", source_shape=(4, 8))

    return SimpleNamespace(loadable_constants=lambda: [(name, constant(name)) for name in names])


def test_the_audit_compiles_a_later_twin_under_the_layout_an_earlier_twin_bound(caplog):
    from emmy.commands.eval import audit_twins
    from emmy.compiler.pipeline.passes.tile import _layout

    seen = []

    def compile_twin(graph):
        seen.append((graph, dict(_layout._SETTLED.get() or {})))
        if graph == "gdn64":
            raise RuntimeError("no row")
        return _constants("w__source")

    with caplog.at_level(logging.INFO, logger="emmy.commands.eval"):
        broken = audit_twins({"gdn64": "gdn64", "gdn16": "gdn16", "gdn1": "gdn1"}, compile_twin, "std")
    assert seen == [("gdn1", {}), ("gdn16", {"w": "source"}), ("gdn64", {"w": "source"})]
    assert broken == 1
    assert "std: gdn16 compiles under LAYOUT@w=source (from gdn1)" in caplog.text


def test_the_record_run_compiles_a_later_twin_under_the_layout_an_earlier_twin_bound(monkeypatch):
    from emmy.commands import run
    from emmy.compiler.pipeline.passes.tile import _layout
    from emmy.compiler.pipeline.search import golden

    rows = {
        ("t64", ()): [SimpleNamespace(name="gdn64.k_conv.m64", kernel="t64")],
        ("t1", ()): [SimpleNamespace(name="gdn1.k_conv.m1", kernel="t1")],
        ("t16", ()): [SimpleNamespace(name="gdn16.k_conv.m16", kernel="t16")],
    }
    document = SimpleNamespace(rows=[row for group in rows.values() for row in group], target_rows=lambda: rows)
    monkeypatch.setattr(golden.GoldenFile, "load", staticmethod(lambda _path: document))
    seen = []

    def run_once(args):
        seen.append((args.realization, dict(_layout._SETTLED.get() or {})))
        args._compiled = _constants("w__source")

    monkeypatch.setattr(run, "_handle_run_once", run_once)
    run._run_golden_targets(SimpleNamespace(input=None, code=None, ir=None, json=None, golden="g.json"))
    assert seen == [("gdn1.k_conv.m1", {}), ("gdn16.k_conv.m16", {"w": "source"}), ("gdn64.k_conv.m64", {"w": "source"})]


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
    db.record_perf(ctx, kernel.exact_identity, bindings={}, knobs={"WORK": "t32"}, backend="cuda", status="ok", stats=stats)


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

    def compile_twin(graph):
        return pipeline.run(graph.copy(), ctx=ctx, db=db)

    with evidence_scope([]), config.strict_evidence_override(True):
        assert {name for name, _ in compile_twin(twins["gdn16"]).loadable_constants()} == {"w"}
        with pytest.raises(EvidenceError):
            from emmy.compiler.pipeline.passes.tile._layout import settled_layouts

            with settled_layouts({"w": "source"}):
                compile_twin(twins["gdn16"])
        assert audit_twins(twins, compile_twin, "std") == 1


def test_settled_layouts_leave_the_fork_one_arm():
    from emmy.compiler.pipeline import Match, Rule
    from emmy.compiler.pipeline.passes.tile._layout import layout_forks, settled_layouts

    graph = _twin(4)
    match = Match(graph=graph, root_node_id="y", rule=Rule(name="test", pattern=[]))
    with settled_layouts({"w": "source"}):
        assert [option.knobs for option in layout_forks(match, graph.nodes["y"])] == [{"LAYOUT@w": "source"}]
    assert len(layout_forks(match, graph.nodes["y"])) == 2
