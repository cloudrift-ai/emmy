"""Warp eligibility is computed from the kernel wherever a schedule row is featurized.

Whether a kernel's schedule space holds a warp plan used to be a stamp the schedule pass minted onto the rows it
offered. When the classic scheduler's materialization dropped it, one op's rows fractured into two ``S_*``
signatures (fork rows stamped, leaf rows not), the deploy's evidence index never joined the measured -O3 rows, and
greedy shipped the prior's unbenched per-cell extrapolation (the 2026-07-07 RTX 5090 gate: 1157 µs per-cell b256 vs
the 3.5 µs mma golden, ~330x). Nothing carries it now: the featurizer asks the scheduling problem
(``Featurizer.warp_eligible``), so the fork's candidates and the op a candidate materializes to read one answer.
"""

from __future__ import annotations

from emmy.compiler import dtype as _dt
from emmy.compiler.context import Context
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Literal
from emmy.compiler.ir.frontend.ir import MatmulOp
from emmy.compiler.ir.stmt import Assign, Let
from emmy.compiler.ir.tile import Placement, TileOp
from emmy.compiler.pipeline import TILE_PASSES, Pipeline
from emmy.compiler.pipeline.fork import Fork, leaf_knobs
from emmy.compiler.pipeline.pipeline import Run
from emmy.compiler.pipeline.search.features import Featurizer
from tests.compiler.terms import contraction, projection, slab


def _matmul_graph(M: int, N: int, K: int, dtype: str) -> Graph:
    graph = Graph()
    dt = _dt.get({"fp16": "f16", "fp32": "f32"}.get(dtype, dtype))
    graph.add_node(InputOp(), [], Tensor("a", (M, K), dt), node_id="a")
    graph.add_node(InputOp(), [], Tensor("b", (K, N), dt), node_id="b")
    graph.add_node(MatmulOp(), ["a", "b"], Tensor("o", (M, N), dt), node_id="o")
    graph.inputs, graph.outputs = ["a", "b"], ["o"]
    return graph


def _resolve_option0(graph, ctx):
    """Option-0 resolution; returns the terminal graph and, per schedule fork, the offered kernel and the row taken."""
    offered = []

    def decide(fp):
        option = fp.options[0]
        while isinstance(option, Fork) and not option.is_leaf:
            option = option.expand()[0]
        if isinstance(fp.root_op, TileOp) and any(key.split("@")[0] == "WORK" for key in leaf_knobs(option)):
            offered.append((fp.root_op, leaf_knobs(option)))
        return option

    resolved, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx).resolve(graph, decide)
    return resolved, offered


def test_a_fork_and_the_op_it_materializes_read_one_eligibility():
    """An fp16 matmul on a tensor-core-capable target is warp-eligible at its schedule fork and on the scheduled op
    the fork's pick materializes to: one kernel, one answer, and the row's features agree. No op carries a stamp.

    A split row may also mint a distinct f32 finalize reduction; that kernel is not itself warp-eligible."""
    ctx = Context.from_target((12, 0))
    featurizer = Featurizer.of(ctx)
    resolved, offered = _resolve_option0(_matmul_graph(512, 512, 512, "fp16"), ctx)
    assert offered, "no schedule fork was offered"
    scheduled = [node.op for node in resolved.nodes.values() if isinstance(node.op, TileOp) and node.op.schedule is not None]
    assert scheduled, "no tile-scheduled op in the resolved graph"
    assert any(featurizer.warp_eligible(kernel) for kernel, _row in offered)
    for kernel, row in offered:
        eligible = featurizer.warp_eligible(kernel)
        assert featurizer.features(kernel, row).get("S_warp_eligible", 0.0) == float(eligible)
        materialized = [op for op in scheduled if kernel in tuple(op.source_chain())]
        assert all(featurizer.warp_eligible(op) == eligible for op in materialized)
        assert all(featurizer.features(op, row) == featurizer.features(kernel, row) for op in materialized)
    assert not any(str(key).startswith("S_") for op in scheduled for key in op.knobs)


def test_an_fp32_matmul_is_not_warp_eligible():
    ctx = Context.from_target((12, 0))
    _resolved, offered = _resolve_option0(_matmul_graph(64, 64, 64, "fp32"), ctx)
    assert offered and not any(Featurizer.of(ctx).warp_eligible(kernel) for kernel, _row in offered)


def test_typed_literal_operands_preserve_the_tensor_core_feature():
    """The declared dtype reaches both computed operands without any input dtype fallback."""
    m, n, k = Axis("m", 32), Axis("n", 32), Axis("k", 32)

    def operand(name, *axes):
        constant = projection((), (Let(f"{name}_scale", Literal(2.0), dtype=_dt.F16),), (f"{name}_scale",))
        return projection(
            (slab(name, name, *axes), constant),
            (Assign(f"{name}_scaled", "multiply", (name, f"{name}_scale")),),
            (f"{name}_scaled",),
        )

    tile = TileOp(
        op=contraction(k, operand("a", "m", "k"), (operand("b", "k", "n"), "acc")),
        name="out",
        place=Placement(free=(m, n)),
        axes=(m, n, k),
        inputs={name: Tensor(name, (32, 32), "f16") for name in ("a", "b")},
        outputs={"out": Tensor("out", (32, 32), "f16")},
    )
    featurizer = Featurizer.of(Context.from_target((8, 0)))
    assert featurizer.warp_eligible(tile)
    assert featurizer.features(tile)["S_warp_eligible"] == 1.0
