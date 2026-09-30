"""The stock MLP oracle must feed all twelve stored leaves through vLLM shard loaders."""

from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace

import torch


def _script():
    return runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/compare_stock_vllm_nvfp4_mlp.py"))


def test_stock_mlp_oracle_loads_exact_gate_up_and_down_leaves():
    script = _script()
    mapping = script["layer_keys"]("model.language_model", 3)
    assert len(mapping) == 12
    assert {key.rsplit(".", 1)[-1] for key in mapping} == {"weight", "weight_scale", "weight_scale_2", "input_scale"}

    seen = []

    def linear(name):
        leaves = {}
        for leaf in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
            param = SimpleNamespace()
            param.weight_loader = lambda p, value, *shard, name=name, leaf=leaf: seen.append((name, leaf, value, shard))
            leaves[leaf] = param
        return SimpleNamespace(**leaves)

    module = SimpleNamespace(gate_up_proj=linear("gate_up_proj"), down_proj=linear("down_proj"))
    tensors = {key: key for key in mapping}
    script["load_mlp_leaves"](module, mapping, tensors)

    assert len(seen) == 12
    for key, (target, shard) in mapping.items():
        leaf = key.rsplit(".", 1)[-1]
        assert (target, leaf, key, () if shard is None else (shard,)) in seen


def test_oracle_reports_bf16_ulp_distance_and_outliers():
    expected = torch.tensor([1.0, 2.0, -4.0], dtype=torch.bfloat16)
    actual = expected.clone()
    bits = actual.view(torch.int16)
    bits[0] += 1  # next larger positive BF16 value
    metrics = _script()["error_metrics"](actual, expected, atol=0.0, rtol=0.0)
    assert metrics["ulp_counts"] == {"0": 2, "1": 1, "2": 0, "3": 0, "4_or_more": 0}
    assert metrics["max_ulp"] == 1
    assert metrics["outside_tolerance"] == 1
    assert metrics["rms_relative"] > 0
