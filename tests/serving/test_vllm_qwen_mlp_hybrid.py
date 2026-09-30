"""The mixed Qwen3.5 adapter retains vLLM's hybrid decoder modules."""

import pytest


def test_real_qwen35_hybrid_modules_survive_mlp_replacement(tmp_path, monkeypatch, request):
    vllm = pytest.importorskip("vllm", reason="requires the pinned vLLM 0.23.0 image")
    if vllm.__version__ != "0.23.0":
        pytest.skip("requires vLLM 0.23.0")
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers.models.qwen3_5")

    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration

    from emmy.serving.vllm_qwen_mlp import EmmyQwen35MlpModel, _CompiledMLP

    text = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_conv_kernel_dim=4,
        max_position_embeddings=64,
        layer_types=["linear_attention", "full_attention"],
    )
    vision = Qwen3_5VisionConfig(depth=1, hidden_size=64, intermediate_size=128, num_heads=4, out_hidden_size=64)
    Qwen3_5Config(text_config=text, vision_config=vision, architectures=["Qwen3_5ForConditionalGeneration"]).save_pretrained(tmp_path)
    vllm_config = EngineArgs(
        model=str(tmp_path),
        dtype="bfloat16",
        max_model_len=64,
        max_num_seqs=1,
        max_num_batched_tokens=64,
        language_model_only=True,
        enforce_eager=True,
        enable_prefix_caching=False,
        skip_tokenizer_init=True,
    ).create_engine_config()
    monkeypatch.setenv("EMMY_FAST_MATH", "false")
    monkeypatch.setenv("EMMY_WORK", "w1x4")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    original_init = Qwen3_5ForConditionalGeneration.__init__
    before = {}

    def capture_stock_modules(self, *, vllm_config, prefix):
        original_init(self, vllm_config=vllm_config, prefix=prefix)
        before["modules"] = {name: module for name, module in self.named_modules() if ".mlp" not in name}
        before["parameters"] = {name: param for name, param in self.named_parameters() if ".mlp." not in name}

    monkeypatch.setattr(Qwen3_5ForConditionalGeneration, "__init__", capture_stock_modules)
    if not torch.distributed.is_initialized():
        request.addfinalizer(destroy_distributed_environment)
        request.addfinalizer(destroy_model_parallel)
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1, rank=0, local_rank=0, distributed_init_method=f"file://{tmp_path / 'dist_init'}", backend="gloo"
        )
        ensure_model_parallel_initialized(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, backend="gloo")
        with torch.device("meta"):
            model = EmmyQwen35MlpModel(vllm_config=vllm_config)

    layers = model.language_model.model.layers
    assert [layer.layer_type for layer in layers] == ["linear_attention", "full_attention"]
    assert len(layers) == sum(isinstance(layer.mlp, _CompiledMLP) for layer in layers) == 2
    assert hasattr(layers[0], "linear_attn") and hasattr(layers[1], "self_attn")
    assert all(hasattr(layer, "input_layernorm") and hasattr(layer, "post_attention_layernorm") for layer in layers)
    after_modules = dict(model.named_modules())
    assert all(after_modules[name] is module for name, module in before["modules"].items())
    after_parameters = dict(model.named_parameters())
    assert set(after_parameters) == set(before["parameters"])
    assert all(after_parameters[name] is param for name, param in before["parameters"].items())
    assert all(".mlp." not in name for name in after_parameters)

    stock = Qwen3_5ForConditionalGeneration
    for method in ("get_mamba_state_shape_from_config", "get_mamba_state_dtype_from_config", "get_mamba_state_copy_func"):
        args = (vllm_config,) if method != "get_mamba_state_copy_func" else ()
        assert getattr(EmmyQwen35MlpModel, method)(*args) == getattr(stock, method)(*args)
    assert EmmyQwen35MlpModel.is_hybrid
