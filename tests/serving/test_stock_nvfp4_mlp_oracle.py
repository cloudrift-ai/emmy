"""The stock MLP oracle must feed all twelve stored leaves through vLLM shard loaders."""

from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace


def test_stock_mlp_oracle_loads_exact_gate_up_and_down_leaves():
    script = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/compare_stock_vllm_nvfp4_mlp.py"))
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
