"""Cold timing is one replay after eviction, and a separate worker job regime."""

import pickle
from types import SimpleNamespace

from emmy.compiler.backend.cuda.program import CompiledProgram, _AsyncBenchWorker, benchmark_program
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import requires_cuda

from .test_program import _make_add_graph


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
