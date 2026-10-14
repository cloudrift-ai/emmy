"""Every captured comparison window submits its complete batch once, outside warmup."""

import asyncio
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

from emmy.commands import run
from emmy.compiler.backend.base import BenchmarkResult, LaunchTime


@pytest.fixture
def fake_cuda(monkeypatch):
    import torch

    state = SimpleNamespace(active=None, timeline=[], graphs=[], events=[])
    stream = SimpleNamespace(wait_stream=lambda other: None)

    class Graph:
        def __init__(self):
            self.calls = []
            state.graphs.append(self)

        def replay(self):
            state.timeline.append(("replay", tuple(self.calls)))

    @contextmanager
    def capture(graph):
        state.timeline.append(("capture",))
        state.active = graph
        try:
            yield
        finally:
            state.active = None

    class Event:
        def __init__(self, **kwargs):
            self.index = len(state.events)
            state.events.append(self)

        def record(self):
            state.timeline.append(("event", self.index))

        def elapsed_time(self, stop):
            # Two backends per iteration. Warmup is deliberately faster than every measured sample.
            return [0.00001, 0.00001, 0.00001, 0.016, 0.024][self.index // 4]

    monkeypatch.setattr(torch.cuda, "Stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "graph", capture)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    def forward(name):
        if state.active is not None:
            state.active.calls.append(name)
        else:
            state.timeline.append(("forward", name))

    state.fns = {name: lambda name=name: forward(name) for name in ("eager", "compiled")}
    return state


def _result(*, captured=True, launches=1, samples=(2.0, 3.0)):
    return BenchmarkResult(
        time_ms=2.0,
        min_ms=2.0,
        num_launches=launches,
        per_launch=[LaunchTime(0, "k", 2.0, samples=samples)],
        captured=captured,
    )


def test_torch_windows_capture_the_full_batch_once_and_share_the_measured_suffix(fake_cuda):
    async def bench(graph, *, on_iter, **kwargs):
        for batch in (1, 8, 8, 8, 8):
            on_iter(batch)
        return _result()

    results, _, captured = asyncio.run(
        run._bench_interleaved_captured(
            None,
            (),
            {},
            SimpleNamespace(benchmark_async=bench),
            None,
            1,
            2,
            torch_fns=fake_cuda.fns,
        )
    )
    assert captured
    assert results["eager"] == results["compiled"] == 2.0
    assert sorted((g.calls[0], len(g.calls)) for g in fake_cuda.graphs) == [
        ("compiled", 1),
        ("compiled", 8),
        ("eager", 1),
        ("eager", 8),
    ]
    # Nothing between the start and stop events except one full-batch replay, for either backend.
    timeline = fake_cuda.timeline
    for event in range(0, 20, 2):
        start = timeline.index(("event", event))
        stop = timeline.index(("event", event + 1))
        window = timeline[start + 1 : stop]
        assert len(window) == 1 and window[0][0] == "replay"
        assert len(window[0][1]) == (1 if event < 4 else 8)


@pytest.mark.parametrize("failure", ["initial_torch", "calibrated_torch", "emmy_program"])
def test_capture_failure_reruns_every_backend_uncaptured(monkeypatch, fake_cuda, failure):
    original = run._capture_torch_fns
    calls = []

    def capture(fns, batch_size=1):
        if failure == "initial_torch" or (failure == "calibrated_torch" and batch_size == 8):
            return None
        return original(fns, batch_size)

    async def bench(graph, *, on_iter, capture_graphs, **kwargs):
        calls.append(capture_graphs)
        on_iter(1)
        on_iter(8)
        return _result(captured=capture_graphs, launches=2 if failure == "emmy_program" else 1)

    monkeypatch.setattr(run, "_capture_torch_fns", capture)
    _, _, captured = asyncio.run(
        run._bench_interleaved_captured(
            None,
            (),
            {},
            SimpleNamespace(benchmark_async=bench),
            None,
            1,
            2,
            torch_fns=fake_cuda.fns,
        )
    )
    assert not captured
    assert calls == [True, False]


def test_unrelated_benchmark_failure_is_not_retried(fake_cuda):
    calls = []

    async def bench(graph, **kwargs):
        calls.append(kwargs["capture_graphs"])
        raise ValueError("invalid executable")

    with pytest.raises(ValueError, match="invalid executable"):
        asyncio.run(
            run._bench_interleaved_captured(
                None,
                (),
                {},
                SimpleNamespace(benchmark_async=bench),
                None,
                1,
                2,
                torch_fns=fake_cuda.fns,
            )
        )
    assert calls == [True]


def test_cold_emmy_only_capture_does_not_require_a_whole_program_window(fake_cuda):
    calls = []

    async def bench(graph, **kwargs):
        calls.append(kwargs["capture_graphs"])
        return _result(launches=2)

    _, result, captured = asyncio.run(
        run._bench_interleaved_captured(
            None,
            (),
            {},
            SimpleNamespace(benchmark_async=bench),
            None,
            1,
            2,
            torch_fns={},
        )
    )
    assert captured and result.e2e_ms is None
    assert calls == [True]


@pytest.mark.parametrize("failure", [None, "capture", "timing"])
def test_calibrated_program_batch_is_prepared_before_timing(monkeypatch, failure):
    from emmy.compiler.backend.cuda import program

    calls = []

    class Program:
        plan = SimpleNamespace(launches=[SimpleNamespace(name="k", kernel_name="k", grid=(1, 1, 1), block=(1, 1, 1))] * 2)

        def on_torch_stream(self):
            return nullcontext()

        def capture_launch_graphs(self, sizes):
            pass

        def iter_once(self, **kwargs):
            return [0.001, 0.001]

        def capture_program_graph(self, batch_size=1):
            calls.append(("capture", batch_size))
            if failure == "capture" and batch_size > 1:
                raise program.GraphCaptureError("calibrated capture refused")

        def time_program_window(self, batch_size):
            calls.append(("timing", batch_size))
            if failure == "timing":
                raise RuntimeError("event watchdog failed")
            return 0.002

    prog = Program()
    prog.capture_program_graph()  # serving's single-forward graph does not establish batch-N capture
    monkeypatch.setattr(program.CompiledProgram, "build", classmethod(lambda cls, *a, **k: prog))
    if failure == "timing":
        with pytest.raises(RuntimeError, match="event watchdog"):
            program.benchmark_program(object(), warmup=0, num_iters=2)
    else:
        result = program.benchmark_program(object(), warmup=0, num_iters=2)
        assert result.captured
        assert (result.e2e_ms is None) == (failure == "capture")
    batch = calls[1][1]
    assert batch > 1 and calls[:2] == [("capture", 1), ("capture", batch)]
    assert calls[2:] == ([] if failure == "capture" else [("timing", batch)] * (1 if failure == "timing" else 2))
