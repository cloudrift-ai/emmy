"""The bench warmup outlasts a clock ramp.

A card coming out of idle runs its first kernels slow while the clocks ramp; a fixed warmup floor
of GPU time did not cover that on an A100 (the first measured kernel ran ~30% slow). The warmup
now keeps going while each batched iter still beats the last one by more than noise."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

from emmy.compiler.backend.cuda import program


class _RampingProgram:
    """One launch whose per-call time falls from 2x to 1x over ``ramp`` iters, then holds."""

    def __init__(self, ramp: int) -> None:
        self.plan = SimpleNamespace(launches=[SimpleNamespace(name="k", kernel_name="k", grid=(1, 1, 1), block=(1, 1, 1))])
        self.ramp, self.calls = ramp, 0

    def on_torch_stream(self):
        return contextlib.nullcontext()

    def capture_launch_graphs(self, sizes) -> None:
        pass

    def iter_once(self, *, batch_sizes, pre_iter=None):
        self.calls += 1
        slow = max(0, self.ramp - self.calls) / self.ramp
        return [2.0 * (1.0 + slow)]  # ms per call: above the batch target, so batches stay 1


def test_warmup_runs_until_the_clock_ramp_settles(monkeypatch) -> None:
    fake = _RampingProgram(ramp=20)
    monkeypatch.setattr(program.CompiledProgram, "build", classmethod(lambda cls, *a, **k: fake))
    result = program.benchmark_program(object(), warmup=3, num_iters=5, capture_graphs=False)
    assert fake.calls > 20, "the warmup must outlast the ramp"
    assert result.per_launch[0].samples == (2.0,) * 5, "every measured iter runs at settled clocks"
