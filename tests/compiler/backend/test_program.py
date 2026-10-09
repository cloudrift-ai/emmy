"""Tests for the execution facade over the runtime: host fill policy, the run and bench entry
points, and the bench loop's budget policy."""

import pickle
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from emmy.compiler.backend.cuda.program import (
    _LAYOUT_MEMO,
    CompiledProgram,
    _AsyncBenchWorker,
    _numpy_storage,
    benchmark_program,
    run_program,
)
from emmy.compiler.dtype import BF16, decode_bf16
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import requires_cuda


def test_bf16_host_values_materialize_as_bits():
    values = np.array([1.0, -2.0, np.pi], dtype=np.float32)
    storage = _numpy_storage(values, BF16)

    assert storage.dtype == np.uint16
    np.testing.assert_array_equal(decode_bf16(storage), np.array([1.0, -2.0, 3.140625], dtype=np.float32))
    np.testing.assert_array_equal(_numpy_storage(storage, BF16), storage)


def test_layout_is_kept_per_environment_up_to_a_bound():
    """A program asks the runtime for the layout once per environment and keeps the most recently used
    ones: a routed MoE prefill asks five times per expert launch, and a long-lived server sees every
    prompt length, so the memo must neither rebuild a hot entry nor grow with the lengths served."""
    asked = []
    runtime = SimpleNamespace(layout=lambda env: asked.append(dict(env)) or {"at": dict(env)})
    program = CompiledProgram(plan=None, program=runtime, executor=None)

    first = program._layout({"num_tokens": 0})
    assert program._layout({"num_tokens": 0}) is first and asked == [{"num_tokens": 0}]
    for width in range(1, _LAYOUT_MEMO):
        program._layout({"num_tokens": width})
        program._layout({"num_tokens": 0})  # stays the most recently used
    program._layout({"num_tokens": _LAYOUT_MEMO})  # one past the bound: the least recently used (width 1) goes
    assert len(program._layouts) == _LAYOUT_MEMO and len(asked) == _LAYOUT_MEMO + 1
    assert program._layout({"num_tokens": 0}) is first
    program._layout({"num_tokens": 1})
    assert len(asked) == _LAYOUT_MEMO + 2 and len(program._layouts) == _LAYOUT_MEMO


def test_bf16_buffer_view_has_logical_dtype_and_shares_storage():
    from types import SimpleNamespace

    import torch

    from emmy.compiler.backend.cuda.program import CompiledProgram

    bits = torch.tensor([0x3F80, 0xC000, 0x4049], dtype=torch.uint16)
    backing = torch.cat((torch.zeros(2, dtype=torch.uint8), bits.view(torch.uint8)))
    buffer = SimpleNamespace(name="x", dtype=BF16, resolve_shape=lambda _sym: (3,))
    runtime = SimpleNamespace(layout=lambda _sym: {"buffers": {"x": {"region": "r", "offset": 2, "bytes": 6}}})
    program = CompiledProgram(SimpleNamespace(buffers=[buffer]), runtime, None, _tensors={"r": backing})

    view = program.buffer_view("x")
    assert view.dtype == torch.bfloat16
    assert view.data_ptr() == backing.data_ptr() + 2
    torch.testing.assert_close(view.float(), torch.tensor([1.0, -2.0, 3.140625]))
    view[0] = 2.0
    assert backing[2:4].view(torch.uint16).item() == 0x4000


EW_ADD_SOURCE = """
extern "C" __global__ void ew_add(const float* A, const float* B, float* C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ELEMENTS) C[i] = A[i] + B[i];
}
"""


def _make_add_graph(n: int = 8) -> Graph:
    """Simple elementwise add: C = A + B."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("A", (n,)), node_id="A")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("B", (n,)), node_id="B")
    g.add_node(
        op=CudaOp(
            kernel_source=EW_ADD_SOURCE.replace("ELEMENTS", str(n)),
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
            kernel_source=EW_ADD_SOURCE.replace("ELEMENTS", str(n)),
            kernel_name="ew_add",
            arg_order=("A", "B", "T"),
            grid=((n + 255) // 256, 1, 1),
            block=(256, 1, 1),
        ),
        inputs=["A", "B"],
        output=Tensor("T", (n,)),
        node_id="T",
    )
    g.add_node(
        op=CudaOp(
            kernel_source=EW_ADD_SOURCE.replace("ELEMENTS", str(n)),
            kernel_name="ew_add",
            arg_order=("T", "T", "C"),
            grid=((n + 255) // 256, 1, 1),
            block=(256, 1, 1),
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

        def on_torch_stream(self):
            return nullcontext()

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


# Cold-cache timing


def test_eviction_precedes_every_launch_window():
    calls = []

    def time_launch(index, batch, deadline):
        calls.append((index, batch))
        return 1.0

    program = CompiledProgram(
        plan=SimpleNamespace(launches=["first", "second"]),
        program=None,
        executor=SimpleNamespace(time_launch=time_launch),
    )
    result = program.iter_once(pre_launch=lambda: calls.append("evict"))
    assert calls == ["evict", (0, 1), "evict", (1, 1)]
    assert result == [1.0, 1.0]


def test_worker_jobs_carry_each_calls_cache_regime():
    for cold in (True, False, True):
        with pinned_knobs({"COLD_CACHE": cold}):
            assert pickle.loads(_AsyncBenchWorker._encode({}))["cold_cache"] is cold


def test_cold_benchmark_never_batches_or_times_a_hot_program(monkeypatch):
    import contextlib

    from emmy.compiler.backend.cuda import cache, program

    calls, captures = [], []
    fake = SimpleNamespace(
        plan=SimpleNamespace(launches=[SimpleNamespace(name="k", kernel_name="k", grid=(1, 1, 1), block=(1, 1, 1))]),
        on_torch_stream=contextlib.nullcontext,
        capture_launch_graphs=lambda sizes: captures.append(list(sizes)),
    )

    def once(*, batch_sizes, pre_iter, pre_launch):
        calls.append(list(batch_sizes))
        pre_launch()
        return [1.0]

    fake.iter_once = once
    monkeypatch.setattr(program.CompiledProgram, "build", lambda *a, **kw: fake)
    monkeypatch.setattr(cache, "L2Eviction", lambda: lambda: None)
    with pinned_knobs({"COLD_CACHE": True}):
        measured = benchmark_program(object(), warmup=1, num_iters=2)
    assert calls and all(batch == [1] for batch in calls)
    assert captures == [[1]]
    assert measured.e2e_ms is None


@requires_cuda
def test_weight_streaming_is_slower_after_l2_eviction():
    import torch

    # Three float buffers occupy less than a quarter of L2: repeated hot launches reuse their data.
    n = torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size // 64
    graph = _make_add_graph(n)
    with pinned_knobs({"COLD_CACHE": False}):
        hot = benchmark_program(graph, warmup=5, num_iters=30)
    with pinned_knobs({"COLD_CACHE": True}):
        cold = benchmark_program(graph, warmup=5, num_iters=30)
    assert hot.captured and cold.captured
    assert cold.time_ms > hot.time_ms
    assert cold.e2e_ms is None
