"""Maximal loop fusion is one schedule-blind fixpoint."""

import numpy as np
import pytest

from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import LinearOp
from emmy.compiler.ir.loop import Loop, LoopOp
from emmy.compiler.pipeline import LOOP_PASSES, Pipeline
from tests.compiler.passes.test_roll_recurrence import _SINKHORN, _delta


def _nests_reduce(loop_op: LoopOp) -> bool:
    reduce_names = loop_op.reduce_axis_names

    def walk(body, inside_reduce: bool) -> bool:
        for stmt in body:
            if isinstance(stmt, Loop):
                is_reduce = stmt.axis.name in reduce_names
                if is_reduce and inside_reduce:
                    return True
                if walk(stmt.body, inside_reduce or is_reduce):
                    return True
        return False

    return walk(loop_op.body, False)


def _chained_matmuls(m=8, k0=4, k1=6, n=5) -> Graph:
    """``(x @ w0) @ w1`` — the smallest graph with nested contractions after fusion."""
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (m, k0)), node_id="x")
    graph.add_node(InputOp(), [], Tensor("w0", (k1, k0)), node_id="w0")
    graph.add_node(InputOp(), [], Tensor("w1", (n, k1)), node_id="w1")
    graph.add_node(LinearOp(), ["x", "w0"], Tensor("h", (m, k1)), node_id="h")
    graph.add_node(LinearOp(), ["h", "w1"], Tensor("y", (m, n)), node_id="y")
    graph.inputs, graph.outputs = ["x", "w0", "w1"], ["y"]
    return graph


def test_chained_matmuls_fuse_into_one_nested_reduction() -> None:
    result = Pipeline.build(LOOP_PASSES).run(_chained_matmuls())
    kernels = [node for node in result.nodes.values() if isinstance(node.op, LoopOp)]
    assert [node.id for node in kernels] == ["y"]
    assert _nests_reduce(kernels[0].op)


def test_nested_reduction_fusion_preserves_numerics() -> None:
    from emmy.compiler.backend.numpy import NumpyBackend

    rng = np.random.default_rng(0)
    inputs = {
        "x": rng.standard_normal((8, 4)).astype(np.float32),
        "w0": rng.standard_normal((6, 4)).astype(np.float32),
        "w1": rng.standard_normal((5, 6)).astype(np.float32),
    }
    backend = NumpyBackend()
    fused = Pipeline.build(LOOP_PASSES).run(_chained_matmuls())
    got = next(iter(backend.run(backend.compile(fused), input_data=inputs)[0].outputs.values()))
    want = (inputs["x"] @ inputs["w0"].T) @ inputs["w1"].T
    np.testing.assert_allclose(got.reshape(want.shape), want, rtol=1e-5, atol=1e-5)


def _chain_beside_a_sibling(stages: dict[str, LoopOp], widths: dict[str, int] | None = None) -> tuple[Graph, str]:
    """A producer feeding both ``stages`` — a chain of states, each reading the last, eight wide
    unless ``widths`` says otherwise — through a broadcast, and a plain elementwise sibling;
    ``(graph, the chain's last state)``."""
    from dataclasses import replace

    from emmy.compiler.ir.expr import Literal, Var
    from emmy.compiler.ir.loop import Assign, Axis, Load, Write

    axis = Axis("i", 8)
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("b0", (9,)), node_id="b0")
    producer = LoopOp(
        body=(
            Loop(
                axis=axis,
                body=(
                    Load(name="p", input="b0", index=(Var("i"),)),
                    Assign(name="pv", op="relu", args=("p",)),
                    Write(output="root", index=(Var("i"),), value="pv"),
                ),
            ),
        ),
    )
    graph.add_node(producer, ["b0"], Tensor("root", (8,)), node_id="root")
    broadcast = LoopOp(
        body=(
            Loop(
                axis=axis,
                body=(
                    Loop(
                        axis=Axis("j", 32),
                        body=(
                            Load(name="v", input="root", index=(Var("i"),)),
                            Write(output="broadcast", index=(Var("i"), Var("j")), value="v"),
                        ),
                    ),
                ),
            ),
        ),
    )
    graph.add_node(broadcast, ["root"], Tensor("broadcast", (8, 32)), node_id="broadcast")
    upstream = "broadcast"
    for tag, op in stages.items():
        # Keep the affine recurrence's exponentially distinct bindings inside the input.
        def bound_load(stmt):
            if not isinstance(stmt, Load):
                return stmt
            index = (stmt.index[0] % Literal(8, "int"),)
            return replace(stmt, index=(*index, Literal(0, "int")) if stmt.input == "b0" else index)

        rebound = replace(op, body=op.body.map(bound_load)).rename_buffers({"b0": "broadcast"})
        graph.add_node(rebound, [upstream], Tensor(tag, ((widths or {}).get(tag, 8),)), node_id=tag)
        upstream = tag
    sibling = LoopOp(
        body=(
            Loop(
                axis=axis,
                body=(
                    Load(name="q", input="root", index=(Var("i"),)),
                    Assign(name="qv", op="negative", args=("q",)),
                    Write(output="easy", index=(Var("i"),), value="qv"),
                ),
            ),
        ),
    )
    graph.add_node(sibling, ["root"], Tensor("easy", (8,)), node_id="easy")
    graph.inputs, graph.outputs = ["b0"], [upstream, "easy"]
    return graph, upstream


def test_a_recurrence_chain_rolls_and_the_rest_of_the_region_merges():
    """The maximal region contains a chain no splice can construct: each stage reads the last at
    two affine maps, so stage 0 is demanded under two to the twelfth σs. The roller replaces the
    chain by one kernel that carries its state, and everything else — the producer, the broadcast
    the first state reads, the plain sibling — is one region and one kernel. Fusion never declines
    a region: leaving the chain unrolled and shrinking around it is what shattered DeepSeek-V4's
    post block into 433 kernels where pre-maximal fusion produced 92, and it made the kernel set
    depend on the order the producers were visited in."""
    from emmy.compiler.backend.numpy import NumpyBackend
    from emmy.compiler.pipeline.passes.loop.fusion._region import carries_state
    from tests.compiler.ir.loop.test_splicer import _affine_recurrence_chain

    stages, _edges, _roots = _affine_recurrence_chain(12)
    graph, last = _chain_beside_a_sibling(stages)
    backend = NumpyBackend()
    inputs = {"b0": np.linspace(-1, 1, 9, dtype=np.float32)}
    before = backend.run(backend.compile(graph), input_data=inputs)[0].outputs

    fused = Pipeline.build(["loop/fusion"]).run(graph)

    kernels = {nid: node.op for nid, node in fused.nodes.items() if isinstance(node.op, LoopOp)}
    (rolled,) = (op for op in kernels.values() if carries_state(op))
    (steps,) = (stmt for stmt in rolled.body if isinstance(stmt, Loop))
    assert steps.axis.extent.as_static() == 11, "eleven states carried from the first, which reads the broadcast"
    (merged,) = (nid for nid, op in kernels.items() if not carries_state(op) and "easy" in op.outputs)
    assert set(kernels[merged].outputs) >= {"easy", "s0"}, "the producer, the broadcast, the first state and the sibling are one kernel"
    assert len(kernels) == 3, sorted(kernels)  # ... plus the slice of the last state the graph exports
    after = backend.run(backend.compile(fused), input_data=inputs)[0].outputs
    for name in (last, "easy"):
        np.testing.assert_allclose(after[name], before[name])


def test_a_chain_the_roller_does_not_roll_is_an_error_not_a_smaller_region():
    """Every stage of this chain has a shape of its own, so no two are states of one recurrence
    and nothing rolls; the region the chain multiplies in raises at the splicer's construction
    bound. Nothing catches it — the kernel set is never a fall-through."""
    from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
    from emmy.compiler.ir.loop import Assign, Axis, Load, UnfusableStmt, Write

    stages: dict[str, LoopOp] = {}
    for k in range(16):
        src = "b0" if k == 0 else f"s{k - 1}"
        left = BinaryExpr("+", BinaryExpr("*", Literal(2, "int"), Var("i")), Literal(1, "int"))
        right = BinaryExpr("+", BinaryExpr("*", Literal(3, "int"), Var("i")), Literal(2, "int"))
        body = (
            Load(name=f"x{k}", input=src, index=(left,)),
            Load(name=f"y{k}", input=src, index=(right,)),
            Assign(name=f"v{k}", op="add", args=(f"x{k}", f"y{k}")),
            Write(output=f"s{k}", index=(Var("i"),), value=f"v{k}"),
        )
        stages[f"s{k}"] = LoopOp(body=(Loop(axis=Axis("i", 8 + k), body=body),))
    graph, _ = _chain_beside_a_sibling(stages, widths={f"s{k}": 8 + k for k in range(16)})
    with pytest.raises(UnfusableStmt, match="bindings per source statement"):
        Pipeline.build(["loop/fusion"]).run(graph)


@pytest.mark.parametrize("code", [_delta(1, 6, 4, 2), _SINKHORN.format(iters=3, n=2, r=3, c=3)])
def test_the_kernel_set_does_not_depend_on_the_order_the_producers_are_visited_in(code: str) -> None:
    """Fusion decides its regions from the graph before it merges any, so the kernels are the same
    whatever order the producers are visited in. Topological order breaks ties by node id: renaming
    every node in reverse visits them in another order, and a rolled recurrence's boundary is
    where a walk from one producer or another used to disagree."""
    from emmy.commands.trace import graph_from_code

    def kernels(graph) -> list[str]:
        fused = Pipeline.build(LOOP_PASSES).run(graph)
        return sorted(node.op.body.structural_key(structural=False) for node in fused.nodes.values() if isinstance(node.op, LoopOp))

    graph, _, _ = graph_from_code(code)
    renamed, _, _ = graph_from_code(code)
    for index, nid in enumerate(reversed(renamed.topological_order())):
        renamed.rename_node(nid, f"z{index}")
    assert kernels(renamed) == kernels(graph)
