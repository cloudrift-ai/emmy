"""The mixed model keeps vLLM's non-MLP loader and owns exact MLP keys."""

import importlib
import sys
import types
import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from emmy.serving.mlp import checkpoint_keys


@pytest.fixture
def adapter(monkeypatch):
    module_name = "vllm.model_executor.models.qwen3_5"
    fake_vllm = types.ModuleType(module_name)

    class StockModel(nn.Module):
        def __init__(self, *, vllm_config, prefix):
            super().__init__()
            self.language_model = nn.Module()
            self.language_model.model = nn.Module()
            self.language_model.model.layers = nn.ModuleList([nn.Module()])
            layer = self.language_model.model.layers[0]
            layer.mlp = nn.Linear(2, 2, device="meta")
            # Match vLLM's Parameter -> bound weight_loader -> module cycle.
            layer.mlp.weight._weight_loader = layer.mlp.forward
            self.old_mlp_ref = weakref.ref(layer.mlp)
            layer.self_attn = nn.Linear(2, 2, device="meta")

        def load_weights(self, weights):
            self.stock_names = [name for name, _ in weights]
            return set(self.stock_names)

    fake_vllm.Qwen3_5ForConditionalGeneration = StockModel
    monkeypatch.setitem(sys.modules, module_name, fake_vllm)
    import emmy.serving as serving

    adapter_name = "emmy.serving.vllm_qwen_mlp"
    old_adapter = sys.modules.pop(adapter_name, None)
    old_attribute = getattr(serving, "vllm_qwen_mlp", None)
    implementation = importlib.import_module(adapter_name)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setenv("EMMY_KNOBS", "FAST_MATH=false,WORK=w1x4")
    yield implementation
    sys.modules.pop(adapter_name, None)
    if old_adapter is not None:
        sys.modules[adapter_name] = old_adapter
    if old_attribute is None:
        delattr(serving, "vllm_qwen_mlp")
    else:
        serving.vllm_qwen_mlp = old_attribute


def _config():
    text = SimpleNamespace(model_type="qwen3_5_text", hidden_act="silu", hidden_size=2, intermediate_size=4, num_hidden_layers=1)
    model = SimpleNamespace(
        dtype=torch.bfloat16,
        max_model_len=4096,
        multimodal_config=SimpleNamespace(language_model_only=True),
        enforce_eager=True,
        hf_text_config=text,
        model="pinned/checkpoint",
        revision="exact-revision",
    )
    return SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
        model_config=model,
        scheduler_config=SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=64),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        speculative_config=None,
        load_config=SimpleNamespace(download_dir="/vllm-cache"),
    )


def test_stock_module_ownership_is_preserved(adapter):
    model = adapter.EmmyQwen35MlpModel(vllm_config=_config())
    layer = model.language_model.model.layers[0]
    assert isinstance(layer.mlp, adapter._CompiledMLP)
    assert layer.self_attn.__class__ is nn.Linear
    assert all("mlp." not in name for name, _ in model.named_parameters())
    assert "owner" not in layer.mlp._modules
    assert model.old_mlp_ref() is None
    with pytest.raises(RuntimeError, match="have not been bound"):
        layer.mlp(torch.zeros(1, 2, dtype=torch.bfloat16))


def test_lazy_download_happens_before_resolving_and_exact_keys_are_filtered(adapter, monkeypatch):
    import emmy.compiler.loader.safetensors as safetensors

    model = adapter.EmmyQwen35MlpModel(vllm_config=_config())
    advanced = False
    required = sorted(checkpoint_keys(0, prefix="model.language_model"))
    seen = {}

    def weights():
        nonlocal advanced
        advanced = True
        yield "model.language_model.layers.0.self_attn.q_proj.weight", torch.zeros(1)
        for name in required:
            yield name, torch.zeros(1)

    def resolve(name, revision, *, cache_dir, local_files_only):
        assert advanced
        assert (name, revision, cache_dir, local_files_only) == ("pinned/checkpoint", "exact-revision", "/vllm-cache", True)
        return "/vllm-cache/snapshot"

    monkeypatch.setattr(safetensors, "_resolve_model_dir", resolve)
    monkeypatch.setattr(adapter, "text_prefix", lambda _: "model.language_model")

    def bind(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(adapter, "MLPPrograms", bind)
    loaded = model.load_weights(weights())
    assert loaded == {"model.language_model.layers.0.self_attn.q_proj.weight"}
    assert model.stock_names == list(loaded)
    assert seen["args"] == ("/vllm-cache/snapshot", 2, 4, 1)
    assert seen["kwargs"] == {"dtype": torch.bfloat16, "capacity": 64}
    assert model._emmy_mlp_programs is not None


@pytest.mark.parametrize(
    "defect, match",
    [
        ("missing", "missing MLP checkpoint"),
        ("duplicate", "duplicate MLP checkpoint"),
        ("extra", "unclaimed MLP checkpoint"),
    ],
)
def test_bad_mlp_ledger_is_rejected(adapter, monkeypatch, defect, match):
    import emmy.compiler.loader.safetensors as safetensors

    monkeypatch.setattr(safetensors, "_resolve_model_dir", lambda *a, **kw: "/snapshot")
    monkeypatch.setattr(adapter, "text_prefix", lambda _: "model.language_model")
    monkeypatch.setattr(adapter, "MLPPrograms", lambda *a, **kw: object())
    model = adapter.EmmyQwen35MlpModel(vllm_config=_config())
    keys = sorted(checkpoint_keys(0, prefix="model.language_model"))
    if defect == "missing":
        keys.pop()
    elif defect == "duplicate":
        keys.append(keys[0])
    else:
        keys.append("model.language_model.layers.0.mlp.mystery")
    with pytest.raises(ValueError, match=match):
        model.load_weights((name, torch.zeros(1)) for name in keys)
    assert model._emmy_mlp_programs is None


def test_unpinned_mlp_serving_is_rejected(adapter, monkeypatch):
    monkeypatch.delenv("EMMY_KNOBS")
    monkeypatch.delenv("EMMY_FAST_MATH", raising=False)
    with pytest.raises(ValueError, match="FAST_MATH=false"):
        adapter.EmmyQwen35MlpModel(vllm_config=_config())

    monkeypatch.setenv("EMMY_KNOBS", "FAST_MATH=false")
    monkeypatch.delenv("EMMY_GOLDEN_FILE", raising=False)
    with pytest.raises(ValueError, match="schedule pins or EMMY_GOLDEN_FILE"):
        adapter.EmmyQwen35MlpModel(vllm_config=_config())
