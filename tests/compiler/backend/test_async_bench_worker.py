"""Unit tests for the async bench worker's request framing."""

from __future__ import annotations

import pickle
from unittest.mock import AsyncMock

from emmy.compiler.backend.cuda.program import _AsyncBenchWorker


async def test_worker_warmup_uses_separate_readiness_request() -> None:
    worker = _AsyncBenchWorker()
    worker.run_job = AsyncMock(return_value={"warmed": True})
    await worker.warmup(wall_timeout_s=45.0)
    worker.run_job.assert_awaited_once_with({"worker_warmup": True}, wall_timeout_s=45.0)


def test_each_worker_request_carries_its_measurement_regime(monkeypatch):
    worker = _AsyncBenchWorker()
    for value, cold in (("0", "1"), ("1", "0"), ("0", "0")):
        monkeypatch.setenv("EMMY_FAST_MATH", value)
        monkeypatch.setenv("EMMY_COLD_CACHE", cold)
        request = pickle.loads(worker._encode({"graph": None}))
        assert request == {"graph": None, "fast_math": value == "1", "cold_cache": cold == "1"}
