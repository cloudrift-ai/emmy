"""Tests for the structural CUDA emitter with the pattern-based fusion pipeline.

Exercises source-level assertions and end-to-end GPU runs. CUDA-specific
by design (source-level assertions on emitted C code); not parameterized
over backends.
"""

from __future__ import annotations

import pytest

from emmy.compiler.backend.cuda.backend import CudaBackend
from emmy.compiler.dtype import BOOL
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.loop import Assign, LoopOp
from emmy.compiler.ir.stmt import Accum, Loop, StridedLoop, Write
from emmy.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp  # noqa: F401
from emmy.compiler.pipeline import LOOP_PASSES, Pipeline
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import compile_with_kernels, matmul_graph, operations, requires_cuda, statements, text


def _folds_in_a_loop(kernel, op: str) -> bool:
    """Whether ``kernel`` holds a loop whose body folds an accumulator with ``op``."""
    loops = statements(kernel, Loop, StridedLoop)
    return any(isinstance(stmt, Accum) and stmt.op.name == op for loop in loops for stmt in loop.body.iter())


def _pointwise_add_graph() -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4,)), node_id="x")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("y", (4,)), node_id="y")
    g.add_node(op=ElementwiseOp("add"), inputs=["x", "y"], output=Tensor("z", (4,)), node_id="z")
    g.inputs = ["x", "y"]
    g.outputs = ["z"]
    return g


def _reduce_sum_graph() -> Graph:
    # K=256: the size rule steps the K loop 32 copies a trip, eight trips, so a loop survives in the kernel; at K=128
    # the four trips left are written out in full, as ptxas wrote them out.
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4, 256)), node_id="x")
    g.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["x"], output=Tensor("y", (4, 1)), node_id="y")
    g.inputs = ["x"]
    g.outputs = ["y"]
    return g


def _boolean_not_graph() -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("mask", (8,), BOOL), node_id="mask")
    g.add_node(op=ElementwiseOp("bitwise_not"), inputs=["mask"], output=Tensor("inverted", (8,), BOOL), node_id="inverted")
    g.inputs = ["mask"]
    g.outputs = ["inverted"]
    return g


def _matmul_graph() -> Graph:
    # K=128: the size rule keeps a K loop of this contraction stepping, so it survives in the kernel.
    return matmul_graph(4, 128, 4)


def _softmax_graph() -> Graph:
    """A 2-axis softmax graph, reduced on axis=1."""
    from emmy.compiler.ir.frontend.ir import SoftmaxOp

    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4, 8)), node_id="x")
    g.add_node(
        op=SoftmaxOp(axis=-1),
        inputs=["x"],
        output=Tensor("y", (4, 8)),
        node_id="y",
    )
    g.inputs = ["x"]
    g.outputs = ["y"]
    return g


def _cuda_nodes(graph: Graph) -> list:
    return [n for n in graph.nodes.values() if isinstance(n.op, CudaOp)]


# ---------------------------------------------------------------------------
# Source-level structure assertions
# ---------------------------------------------------------------------------


def test_pointwise_emits_correct_source():
    compiled = CudaBackend().compile(_pointwise_add_graph())
    nodes = _cuda_nodes(compiled)
    assert len(nodes) == 1
    source = nodes[0].op.kernel_source
    assert "%tid.x" in source
    assert "ld.global" in source and "st.global" in source
    assert "buf0[" in source and "buf1[" in source  # buffers are spelled by their role, in the statement comments


def test_boolean_bitwise_not_lowers_from_graph_loop_to_a_logical_not():
    loop_graph = Pipeline.build(LOOP_PASSES).run(_boolean_not_graph())
    loops = [node.op for node in loop_graph.nodes.values() if isinstance(node.op, LoopOp)]
    assert len(loops) == 1
    next(
        stmt
        for stmt in loops[0]
        if isinstance(stmt, Assign) and stmt.value.operation is not None and stmt.value.operation.name == "bitwise_not"
    )

    compiled = CudaBackend().compile(loop_graph)
    # The negation is a comparison of the mask with zero, its truth stored as a byte.
    source = _cuda_nodes(compiled)[0].op.kernel_source
    assert "setp." in source and "st.global.u8" in source


def test_reduce_emits_k_loop():
    # The scalar serial emitter with its K loop retained; the greedy pick may unroll the loop away.
    with pinned_knobs({"UNROLL": 0, "TILE": "", "REDUCE": ""}):
        _, kernels = compile_with_kernels(_reduce_sum_graph())
    assert _folds_in_a_loop(next(iter(kernels.values())), "add")


def test_contraction_emits_matmul():
    # Inspect the scalar serial emitter with its K loop retained.
    with pinned_knobs({"UNROLL": 0, "TILE": "", "REDUCE": ""}):
        _, kernels = compile_with_kernels(_matmul_graph())
    assert any(_folds_in_a_loop(kernel, "add") for kernel in kernels.values())  # accumulator fold


def test_buffer_roles():
    compiled = CudaBackend().compile(_pointwise_add_graph())
    assert "x" in compiled.inputs
    assert "y" in compiled.inputs
    assert len(compiled.outputs) == 1


def test_softmax_sweeps_k_twice():
    """Softmax sweeps K twice: the online reduce (running max + the rescaled running
    denominator, the twisted state-merge) then the per-element div — not a single
    collapsed pass. (The online formulation folds max + sum in one sweep via the
    ``exp(x−m)`` twist, so the sum accumulates as a state-merge, not a literal ``+=``.)
    At K=8 the unroll passes write both sweeps out, so they are told apart by order:
    every fold of the reduce stands ahead of every store of the output, and no loop
    holds both."""
    with pinned_knobs({"PLACE": "fuse"}):
        compiled, kernels = compile_with_kernels(_softmax_graph())
    # Find the softmax-bearing kernel (it folds a maximum).
    softmax = next((kernel for kernel in kernels.values() if "maximum" in operations(kernel)), None)
    assert softmax is not None, f"no kernel folds a maximum: {text(kernels)}"
    order = statements(softmax)
    folds = [at for at, stmt in enumerate(order) if isinstance(stmt, Accum)]
    stores = [at for at, stmt in enumerate(order) if isinstance(stmt, Write) and stmt.output == compiled.outputs[0]]
    assert folds and stores and folds[-1] < stores[0], f"the reduce must finish before the div stores\n{text(softmax)}"
    for loop in statements(softmax, Loop, StridedLoop):
        inner = list(loop.body.iter())
        assert not (any(isinstance(s, Accum) for s in inner) and any(isinstance(s, Write) for s in inner)), text(softmax)
    assert operations(softmax) & {"exp", "exp_fast"}  # the softmax numerator / online-reduce normalizer


def test_softmax_stores_every_element():
    """Softmax output is per-element: each of a row's eight elements is stored as its
    own value (the unroll passes write the K loop out, so the stores stand in its copies)."""
    compiled, kernels = compile_with_kernels(_softmax_graph())
    out_name = compiled.outputs[0]
    stored = {str(value) for write in statements(kernels, Write) if write.output == out_name for value in write.values}
    assert len(stored) == 8, text(kernels)


def test_chained_pointwise_single_kernel():
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4,)), node_id="x")
    g.add_node(op=ElementwiseOp("exp"), inputs=["x"], output=Tensor("e", (4,)), node_id="e")
    g.add_node(op=ElementwiseOp("negative"), inputs=["e"], output=Tensor("n", (4,)), node_id="n")
    g.inputs = ["x"]
    g.outputs = ["n"]

    compiled = CudaBackend().compile(g)
    assert len(_cuda_nodes(compiled)) == 1


# ---------------------------------------------------------------------------
# GPU execution
# ---------------------------------------------------------------------------


@requires_cuda
def test_pointwise_runs_on_gpu():
    compiled = CudaBackend().compile(_pointwise_add_graph())
    result, _ = CudaBackend().run(compiled, input_data={"x": [1, 2, 3, 4], "y": [10, 20, 30, 40]})
    assert list(result.outputs.values())[0].flatten().tolist() == pytest.approx([11, 22, 33, 44])


@requires_cuda
def test_reduce_runs_on_gpu():
    compiled = CudaBackend().compile(_reduce_sum_graph())
    x_data = [float(i) for i in range(4 * 256)]
    result, _ = CudaBackend().run(compiled, input_data={"x": x_data})
    expected = [sum(x_data[row * 256 : (row + 1) * 256]) for row in range(4)]
    assert list(result.outputs.values())[0].flatten().tolist() == pytest.approx(expected)


@requires_cuda
def test_softmax_runs_on_gpu():
    import math

    compiled = CudaBackend().compile(_softmax_graph())
    x_data = [float(i) for i in range(32)]
    result, _ = CudaBackend().run(compiled, input_data={"x": x_data})
    expected = []
    for row in range(4):
        row_vals = x_data[row * 8 : (row + 1) * 8]
        mx = max(row_vals)
        exps = [math.exp(v - mx) for v in row_vals]
        s = sum(exps)
        expected.extend(e / s for e in exps)
    assert list(result.outputs.values())[0].flatten().tolist() == pytest.approx(expected, rel=1e-3)


@requires_cuda
def test_matmul_runs_on_gpu():
    compiled = CudaBackend().compile(_matmul_graph())
    a_data = [float(i) for i in range(4 * 128)]
    b_data = [float(i) for i in range(128 * 4)]
    result, _ = CudaBackend().run(compiled, input_data={"a": a_data, "b": b_data})
    expected = []
    for mi in range(4):
        for ni in range(4):
            s = sum(a_data[mi * 128 + k] * b_data[k * 4 + ni] for k in range(128))
            expected.append(s)
    assert list(result.outputs.values())[0].flatten().tolist() == pytest.approx(expected)
