"""CPU tests for the generative plugin's served-length RoPE cache construction."""

from types import SimpleNamespace

import pytest


def test_rope_cache_limit_prefers_vllm_served_length():
    pytest.importorskip("vllm")
    from emmy.serving.vllm_model_gen import _rope_cache_limit

    hf_config = SimpleNamespace(max_position_embeddings=1_048_576)
    assert _rope_cache_limit(SimpleNamespace(max_model_len=4096), hf_config) == 4096
    assert _rope_cache_limit(SimpleNamespace(max_model_len=None), hf_config) == 1_048_576


def test_laguna_multi_rope_is_served_length_bounded_and_dtype_correct():
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from vllm.config import VllmConfig
    from vllm.config.vllm import set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import get_rope

    from emmy.serving.vllm_model_gen import _build_rotaries

    yarn = {
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 16,
        "rope_theta": 10_000.0,
        "partial_rotary_factor": 0.5,
        "beta_fast": 32,
        "beta_slow": 1,
    }
    config = SimpleNamespace(
        layer_types=["sliding_attention", "full_attention", "sliding_attention"],
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 1_000_000.0},
            "full_attention": yarn,
        },
    )
    runner = SimpleNamespace(layer_meta=lambda _i: (8, 2, 1, 8**-0.5), global_layer_id=lambda i: i)

    with set_current_vllm_config(VllmConfig()):
        rotaries = _build_rotaries(config, runner, 3, max_position=7, dtype=torch.float16)
        full_yarn = get_rope(8, max_position=64, rope_parameters=yarn, dtype=torch.float16)

    assert rotaries[0] is rotaries[2]
    assert tuple(rotaries[0].cos_sin_cache.shape) == (7, 8)
    assert tuple(rotaries[1].cos_sin_cache.shape) == (7, 4)
    assert rotaries[0].cos_sin_cache.dtype == torch.float16
    assert rotaries[1].cos_sin_cache.dtype == torch.float16
    # Bounding only changes the row count: YaRN's correction ramp is still anchored at the
    # original training context, so every served row matches vLLM's full cache exactly.
    torch.testing.assert_close(rotaries[1].cos_sin_cache, full_yarn.cos_sin_cache[:7], rtol=0, atol=0)


def test_rope_takes_the_fused_path_under_inductor_default_custom_ops():
    """Bare ``vllm serve`` compiles with inductor, whose default ``custom_ops`` is ``none``: vLLM
    then hands every RoPE its ``forward_native`` for inductor to fuse. Nothing compiles the plugin,
    so that path ran as 17 small kernels per Gemma 4 decode layer; its RoPEs dispatch as enabled."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from vllm.config import CompilationConfig, VllmConfig
    from vllm.config.compilation import CompilationMode
    from vllm.config.vllm import set_current_vllm_config

    from emmy.serving.vllm_model_gen import _build_rotaries

    # Gemma 4's pair: default RoPE on sliding layers, proportional partial RoPE on global ones. Head
    # widths no other test builds, so get_rope's instance cache cannot hand back an earlier dispatch.
    config = SimpleNamespace(
        layer_types=["sliding_attention", "full_attention"],
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
            "full_attention": {"rope_type": "proportional", "rope_theta": 1_000_000.0, "partial_rotary_factor": 0.25},
        },
    )
    runner = SimpleNamespace(layer_meta=lambda i: ((24, 48)[i], 1, 1, 1.0), global_layer_id=lambda i: i)
    inductor_default = CompilationConfig(mode=CompilationMode.VLLM_COMPILE, custom_ops=["none"])
    with set_current_vllm_config(VllmConfig(compilation_config=inductor_default)):
        rotaries = _build_rotaries(config, runner, 2, max_position=16, dtype=torch.float16)

    assert [type(r).__name__ for r in rotaries] == ["RotaryEmbedding", "Gemma4RotaryEmbedding"]
    assert all(r._forward_method.__name__ != "forward_native" for r in rotaries)


def test_qwen3_5_rope_matches_hugging_face_on_text_positions():
    """Qwen3.5's full-attention layers rotate a quarter of each head with a three-axis (MRoPE)
    schedule. For text every axis carries the same position, and the served rotary must then turn
    ``q`` and ``k`` exactly as the Hugging Face ``qwen3_5`` text model does."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    pytest.importorskip("transformers.models.qwen3_5")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding, apply_rotary_pos_emb
    from vllm.config import VllmConfig
    from vllm.config.vllm import set_current_vllm_config

    from emmy.serving.vllm_model_gen import _build_rotaries

    # Inferact/Qwen3.8-27B-NVFP4's full-attention geometry and rotary parameters, at two heads.
    head_dim, heads, kv_heads, t = 256, 2, 1, 48
    config = Qwen3_5TextConfig(
        head_dim=head_dim,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        layer_types=["full_attention"],
        num_hidden_layers=1,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
        },
    )
    runner = SimpleNamespace(layer_meta=lambda _i: (head_dim, heads, kv_heads, head_dim**-0.5), global_layer_id=lambda i: i)
    with set_current_vllm_config(VllmConfig()):
        (rotary,) = _build_rotaries(config, runner, 1, max_position=64, dtype=torch.float32)

    torch.manual_seed(0)
    q, k = torch.randn(t, heads * head_dim), torch.randn(t, kv_heads * head_dim)
    positions = torch.arange(t)
    got_q, got_k = rotary.forward_native(positions, q.clone(), k.clone())

    cos, sin = Qwen3_5TextRotaryEmbedding(config)(q[None], positions[None])
    want_q, want_k = apply_rotary_pos_emb(
        q.view(1, t, heads, head_dim).transpose(1, 2), k.view(1, t, kv_heads, head_dim).transpose(1, 2), cos, sin
    )
    torch.testing.assert_close(got_q, want_q.transpose(1, 2).reshape(t, -1), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(got_k, want_k.transpose(1, 2).reshape(t, -1), rtol=1e-5, atol=1e-5)
