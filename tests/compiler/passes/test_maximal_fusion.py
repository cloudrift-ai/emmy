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


def _unexplained_boundaries(graph: Graph) -> list[tuple[str, str]]:
    """Every edge between two kernels of a fused graph that no correctness boundary explains — a
    fusion gate. Derived from the op kinds alone, not from ``regions()``: the edge ``u -> v`` could
    have been one kernel when every node on a path between them is a loop fusion may merge (no
    carried state, no running-accumulator output) and none of them writes a packed buffer. Such an
    edge is a region fusion declined, whatever the reason."""
    from emmy.compiler.ir.loop import observes_running_accumulator
    from emmy.compiler.pipeline.passes.loop.fusion._region import carries_state

    nodes = graph.nodes

    def mergeable(nid: str) -> bool:
        op = nodes[nid].op
        return isinstance(op, LoopOp) and not carries_state(op) and not observes_running_accumulator(op)

    def packed(nid: str) -> bool:
        return any((tensor := graph.buffer(buf)) is not None and tensor.dtype.logical_elems > 1 for buf in nodes[nid].buffer_names())

    def producers(nid: str) -> set[str]:
        return {producer.id for buf in nodes[nid].inputs if (producer := graph.producer(buf)) is not None}

    upstream: dict[str, set[str]] = {}
    for nid in graph.topological_order():
        upstream[nid] = set().union(*({p} | upstream[p] for p in producers(nid)))
    gates = []
    for consumer, above in upstream.items():
        for producer in producers(consumer) if mergeable(consumer) else ():
            between = {nid for nid in above if producer in upstream[nid]} | {producer}
            if all(mergeable(nid) and not packed(nid) for nid in between):
                gates.append((producer, consumer))
    return gates


_TEMPTING = {
    # a reduction over 64k elements, normalized and projected
    "long_reduction": """
import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(65536, 16, bias=False)
    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        return self.proj(x)
m = M()
m(torch.randn(2, 65536))
""",
    # a projection read under a wide axis it does not depend on: the merged kernel recomputes it 4096 times
    "recompute": """
import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(64, 64, bias=False)
    def forward(self, x, table):
        h = self.proj(x)
        return (h[:, :, None] * table[None]).sum(1)
m = M()
m(torch.randn(4, 64), torch.randn(64, 4096))
""",
    # attention and its output projection: three nested contractions
    "attention": """
import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.o = nn.Linear(32, 32, bias=False)
    def forward(self, q, k, v):
        p = torch.softmax(q @ k.transpose(-1, -2) / 4.0, dim=-1)
        return self.o(p @ v)
m = M()
m(torch.randn(16, 32), torch.randn(24, 32), torch.randn(24, 32))
""",
    # sixty-four stages, long enough to trip any bound on a region's size; the pointwise stages are
    # seven different ops and every projection has a width of its own, so no two stages are steps of
    # one recurrence the roller would carry
    "long_chain": """
import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.ws = nn.ParameterList([nn.Parameter(torch.randn(16 + i, 17 + i)) for i in range(8)])
    def forward(self, x):
        for w in self.ws:
            for f in (torch.tanh, torch.sigmoid, torch.sin, torch.cos, torch.relu, torch.abs, torch.neg):
                x = f(x)
            x = x @ w
        return x
m = M()
m(torch.randn(4, 16))
""",
    # a decoder MLP block with its residual, exporting two outputs
    "multi_output_block": """
import torch, torch.nn as nn
import torch.nn.functional as F
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(32, 96, bias=False)
        self.up = nn.Linear(32, 96, bias=False)
        self.down = nn.Linear(96, 32, bias=False)
    def forward(self, x):
        h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        return x + self.down(F.silu(self.gate(h)) * self.up(h)), h
m = M()
m(torch.randn(8, 32))
""",
}


@pytest.mark.parametrize("name", sorted(_TEMPTING))
def test_a_graph_that_tempts_a_gate_is_one_kernel(name: str) -> None:
    """Each graph is a region a size, cost, recompute or recognizer bound would decline. None has a
    correctness boundary, so fusion makes each one kernel; a cut offers the pieces back later."""
    from emmy.commands.trace import graph_from_code

    graph, _, _ = graph_from_code(_TEMPTING[name])
    fused = Pipeline.build(LOOP_PASSES).run(graph)
    assert _unexplained_boundaries(fused) == []
    assert sum(isinstance(node.op, LoopOp) for node in fused.nodes.values()) == 1


#: Programs with more traced nodes than this take 5 s to minutes each to fuse (the Qwen3.8 layers);
#: the rest still hold a layer of every other model family the goldens store, and fuse in under a
#: second.
_MAX_TRACED_NODES = 200


def _repository_goldens() -> list:
    from emmy.compiler.pipeline.search.golden import _repository_golden_paths

    with _repository_golden_paths() as paths:
        return [pytest.param(path, id=f"{path.parent.parent.name}/{path.name}") for path in paths]


@pytest.mark.parametrize("path", _repository_goldens())
def test_every_boundary_of_a_golden_program_is_a_correctness_boundary(path) -> None:
    """The traced programs the repository goldens store are real models' layers: none of them has
    an edge between two kernels that fusion could have merged. One program per traced size — a
    golden stores the same layer at many widths."""
    import yaml

    from emmy.compiler.pipeline.search.golden import _SAFE_LOADER, stored_program

    document = yaml.load(path.read_text(), Loader=_SAFE_LOADER)
    sizes: set[int] = set()
    for index in range(len(document["programs"])):
        graph = stored_program(document, index)
        if len(graph.nodes) <= _MAX_TRACED_NODES and len(graph.nodes) not in sizes:
            sizes.add(len(graph.nodes))
            assert _unexplained_boundaries(Pipeline.build(LOOP_PASSES).run(graph)) == [], f"program {index}"


def test_a_region_the_splicer_cannot_build_raises(monkeypatch) -> None:
    """No fallback shrinks a region the splice refuses: the kernel set is never a fall-through."""
    from emmy.compiler.pipeline.passes.loop.fusion import _region

    # The pass loads its rule module afresh, so the rule imports the refusing splice.
    monkeypatch.setattr(_region, "build_merged_region", lambda *_: None)
    with pytest.raises(ValueError, match="fusion cannot splice the region"):
        Pipeline.build(LOOP_PASSES).run(_chained_matmuls())
