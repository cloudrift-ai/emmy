"""Tests for the execution facade over the runtime: host fill policy, the run and bench entry
points, and the bench loop's budget policy."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from emmy.compiler.backend.cuda.program import _numpy_storage, benchmark_program, run_program
from emmy.compiler.dtype import BF16, decode_bf16
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.cuda import CudaOp
from tests.compiler.helpers import requires_cuda


def test_bf16_host_values_materialize_as_bits():
    values = np.array([1.0, -2.0, np.pi], dtype=np.float32)
    storage = _numpy_storage(values, BF16)

    assert storage.dtype == np.uint16
    np.testing.assert_array_equal(decode_bf16(storage), np.array([1.0, -2.0, 3.140625], dtype=np.float32))
    np.testing.assert_array_equal(_numpy_storage(storage, BF16), storage)


EW_ADD_SOURCE = """
extern "C" __global__ void ew_add(const float* A, const float* B, float* C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 8) C[i] = A[i] + B[i];
}
"""


def _make_add_graph(n: int = 8) -> Graph:
    """Simple elementwise add: C = A + B."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("A", (n,)), node_id="A")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("B", (n,)), node_id="B")
    g.add_node(
        op=CudaOp(
            kernel_source=EW_ADD_SOURCE,
            kernel_name="ew_add",
            arg_order=("A", "B", "C"),
            grid=((n + 255) // 256, 1, 1),
            block=(256, 1, 1),
        ),
        inputs=["A", "B"],
        output=Tensor("C", (n,)),
        node_id="C",
    )
    g.inputs = ["A", "B"]
    g.outputs = ["C"]
    return g


@requires_cuda
def test_run_program_elementwise_add():
    result, _ = run_program(_make_add_graph(8))
    assert "C" in result.outputs
    assert result.outputs["C"].shape == (8,)
    assert all(v == v for v in result.outputs["C"].tolist())  # NaN check


def _make_chain_graph(n: int = 8) -> Graph:
    """Two-stage chain with a real scratch buffer: T = A + B (scratch), C = T + T."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("A", (n,)), node_id="A")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("B", (n,)), node_id="B")
    g.add_node(
        op=CudaOp(
            kernel_source=EW_ADD_SOURCE, kernel_name="ew_add", arg_order=("A", "B", "T"), grid=((n + 255) // 256, 1, 1), block=(256, 1, 1)
        ),
        inputs=["A", "B"],
        output=Tensor("T", (n,)),
        node_id="T",
    )
    g.add_node(
        op=CudaOp(
            kernel_source=EW_ADD_SOURCE, kernel_name="ew_add", arg_order=("T", "T", "C"), grid=((n + 255) // 256, 1, 1), block=(256, 1, 1)
        ),
        inputs=["T"],
        output=Tensor("C", (n,)),
        node_id="C",
    )
    g.inputs = ["A", "B"]
    g.outputs = ["C"]
    return g


@requires_cuda
def test_chain_through_scratch_is_correct():
    """A graph with a scratch buffer computes through the intermediate ``T``."""
    from emmy.compiler.backend.cuda.program import CompiledProgram
    from emmy.compiler.backend.gpu_lock import gpu_lock

    graph = _make_chain_graph(8)
    a = np.arange(8, dtype=np.float32)
    b = np.arange(8, dtype=np.float32) * 10
    with gpu_lock():
        prog = CompiledProgram.build(graph, {"A": a, "B": b})
        prog.iter_once()
        out = prog.outputs()["C"]
    np.testing.assert_array_equal(out, 2 * (a + b))  # C = T + T = 2(A+B)


@requires_cuda
def test_benchmark_program_returns_timing():
    result = benchmark_program(_make_add_graph(1024), warmup=2, num_iters=5)
    assert result.time_ms > 0
    assert result.num_launches == 1
    assert result.per_launch is not None
    assert len(result.per_launch) == 1


def _fake_benchmark_program(monkeypatch, iter_ms: float):
    import emmy.compiler.backend.cuda.program as program_mod
    import emmy.compiler.backend.gpu_lock as lock_mod

    class _FakeProgram:
        def __init__(self) -> None:
            self.plan = SimpleNamespace(launches=[SimpleNamespace(kernel_name="k")])
            self.calls = 0

        def iter_once(self, *, batch_sizes=None, pre_iter=None):
            self.calls += 1
            return [iter_ms]

    fake = _FakeProgram()
    monkeypatch.setattr(program_mod.CompiledProgram, "build", classmethod(lambda _cls, *_args, **_kwargs: fake))
    monkeypatch.setattr(lock_mod, "gpu_lock", nullcontext)
    return fake


def test_benchmark_program_explicit_warmup_count_is_unchanged(monkeypatch):
    fake = _fake_benchmark_program(monkeypatch, iter_ms=5.0)

    result = benchmark_program(Graph(), warmup=5, num_iters=1, run_timeout_s=2.0, capture_graphs=False)

    assert fake.calls == 6
    assert result.time_ms == 5.0
    assert result.per_launch[0].samples == (5.0,)


def test_benchmark_program_run_budget_still_fails_on_first_slow_iteration(monkeypatch):
    fake = _fake_benchmark_program(monkeypatch, iter_ms=2100.0)

    with pytest.raises(RuntimeError, match="benchmark run stage exceeded 2.0s of GPU time"):
        benchmark_program(Graph(), warmup=1, num_iters="auto", run_timeout_s=2.0, capture_graphs=False)

    assert fake.calls == 1
