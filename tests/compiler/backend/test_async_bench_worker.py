"""Unit tests for the async bench worker's request framing."""

from __future__ import annotations

import pickle

from emmy.compiler.backend.cuda.program import _AsyncBenchWorker


def test_each_worker_request_carries_its_measurement_regime(monkeypatch):
    from emmy.compiler.pipeline.search.space import precision_pins

    worker = _AsyncBenchWorker()
    monkeypatch.delenv("EMMY_FAST_DIV", raising=False)
    for value, cold in (("0", "1"), ("1", "0"), ("0", "0")):
        monkeypatch.setenv("EMMY_FAST_MATH", value)
        monkeypatch.setenv("EMMY_COLD_CACHE", cold)
        request = pickle.loads(worker._encode({"graph": None}))
        assert request == {"graph": None, "precision": precision_pins({"FAST_MATH": value == "1"}), "cold_cache": cold == "1"}
    # An independent override travels with the umbrella: a persistent worker spawned earlier never saw it.
    monkeypatch.setenv("EMMY_FAST_DIV", "0")
    assert pickle.loads(worker._encode({"graph": None}))["precision"]["FAST_DIV"] is False
