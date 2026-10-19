"""`EmmyGenModel` hosting the 1Cat fork's sparse attention (Qwen3.8-Flash-Next).

The model class must CONSTRUCT under a real vLLM config — one fork ``Qwen4ExpQSAAttention`` per local
sparse-attention layer, each registering its main cache and the indexer's two side caches under the absolute
layer's ``self_attn`` prefix, all sharing one top-k buffer — and the checkpoint's sparse-attention weights must LOAD
into the real fork modules through the ownership table, query, key and value as the fused projection's shards.

Serving such a checkpoint end to end needs its Gated DeltaNet layers too: the fork lays out the sparse attention's
caches together with their recurrent state and refuses a model without them. Runs only where the fork is importable
and the card is the one it was built for (the pinned 1Cat image on sm_70); skipped everywhere else.
"""

from __future__ import annotations

import json

import pytest

pytestmark = [pytest.mark.perf, pytest.mark.xdist_group("cuda")]

LAYERS = 2


def _tiny_qwen4_exp(path, transformers, torch):
    """Sparse-attention layers with the published attention geometry; everything else shrinks."""
    text = dict(
        hidden_size=512, hc_count=4, hc_lowrank=64, num_attention_heads=24, num_key_value_heads=2, head_dim=256,
        indexer_n_heads=4, indexer_kv_heads=1, indexer_head_dim=128, indexer_budget=2048, indexer_compress_ratio=4,
        partial_rotary_factor=0.25, rope_parameters={"mrope_interleaved": True, "mrope_section": [11, 11, 10],
                                                     "partial_rotary_factor": 0.25, "rope_theta": 10000000,
                                                     "rope_type": "default"},
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=128, shared_expert_intermediate_size=128,
        vocab_size=1024, max_position_embeddings=4096, mtp_num_hidden_layers=0, ple_layer_ids=[],
        num_hidden_layers=LAYERS, layer_types=["full_attention"] * LAYERS, tie_word_embeddings=False,
    )  # fmt: skip
    vision = dict(depth=1, hidden_size=64, intermediate_size=128, num_heads=2, out_hidden_size=text["hidden_size"])
    config = transformers.Qwen4ExpConfig(text_config=text, vision_config=vision, tie_word_embeddings=False)
    torch.manual_seed(0)
    transformers.Qwen4ExpForConditionalGeneration(config).eval().to(torch.bfloat16).save_pretrained(path)
    # Spelled the way the published file spells it: Transformers saves its internal "indexed_attention", which the
    # fork's config does not read.
    config_path = path / "config.json"
    saved = json.loads(config_path.read_text())
    saved["text_config"]["layer_types"] = text["layer_types"]
    config_path.write_text(json.dumps(saved))


def _build_model(tmp_path, torch, transformers):
    """The in-process build a worker does: engine config, single-rank groups, construction on the card."""
    import socket

    from vllm.config import set_current_vllm_config
    from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
    from vllm.engine.arg_utils import EngineArgs

    from emmy.serving.vllm_model_gen import EmmyGenModel

    _tiny_qwen4_exp(tmp_path, transformers, torch)
    vllm_config = EngineArgs(
        model=str(tmp_path),
        runner="generate",
        dtype="float16",
        max_model_len=256,
        max_num_batched_tokens=256,
        enforce_eager=True,
        language_model_only=True,
        skip_tokenizer_init=True,
        hf_overrides={"architectures": ["EmmyGenModel"]},
    ).create_engine_config()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(world_size=1, rank=0, distributed_init_method=f"tcp://127.0.0.1:{port}", local_rank=0)
        ensure_model_parallel_initialized(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        with torch.device("cuda"):
            model = EmmyGenModel(vllm_config=vllm_config, prefix="model")
    return vllm_config, model


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("vllm.models.qwen4_exp.nvidia.qsa")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    path = tmp_path_factory.mktemp("qwen4_exp")
    with pytest.MonkeyPatch.context() as env:
        # Construction and loading need only the symbolic programs. The static decode and prefill tiers would add
        # their compiles and the boot audit's runs of picks nothing measured for this checkpoint.
        env.setenv("EMMY_GEN_DECODE_BUCKET", "0")
        env.setenv("EMMY_GEN_PREFILL_BUCKET", "0")
        vllm_config, model = _build_model(path, torch, transformers)
    return path, vllm_config, model


def test_sparse_attention_layers_register_their_own_caches(built):
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAAttention

    _path, vllm_config, model = built
    assert [type(m) for m in model.fork_attn] == [Qwen4ExpQSAAttention] * LAYERS
    assert len(model.attn) == 0 and len(model.rotary_emb) == 0
    registered = set(vllm_config.compilation_config.static_forward_context)
    for layer in range(LAYERS):
        site = f"model.layers.{layer}.self_attn"
        assert {f"{site}.attn", f"{site}.indexer.raw_key_cache", f"{site}.indexer.compressed_key_cache"} <= registered
    # The layers run one after another and consume their selection before the next overwrites it: one buffer.
    assert len({m.topk_indices_buffer.data_ptr() for m in model.fork_attn}) == 1
    assert model.runner.carrier_size == 4 * 512


def test_checkpoint_sparse_attention_weights_load_into_the_fork_modules(built):
    """Stream the checkpoint itself through ``load_weights``: every fork parameter is claimed, and holds the
    checkpoint's values, the fused projection in query, key, value order."""
    import torch
    from safetensors.torch import load_file

    path, _vllm_config, model = built
    tensors = load_file(str(path / "model.safetensors"))

    loaded = model.load_weights(iter(tensors.items()))

    for layer, module in enumerate(model.fork_attn):
        params = dict(module.named_parameters())
        assert {f"fork_attn.{layer}.{name}" for name in params} <= loaded
        source = {k.removeprefix(f"model.language_model.layers.{layer}.self_attn."): v for k, v in tensors.items()}
        expected = {"qkv_proj.weight": torch.cat([source["q_proj.weight"], source["k_proj.weight"], source["v_proj.weight"]])}
        for name in params:
            want = expected.get(name, source.get(name))
            assert want is not None, f"layer {layer}: no checkpoint source for {name}"
            assert torch.equal(params[name].detach().cpu(), want.to(params[name].dtype)), f"layer {layer}: {name}"
    assert "lm_head.weight" in loaded


def test_the_last_gated_residual_closes_the_streams(built):
    """The checkpoint has no final norm: its last gated residual collapses the four streams into the head's input."""
    import torch
    import transformers

    path, _vllm_config, model = built
    reference = transformers.Qwen4ExpForConditionalGeneration.from_pretrained(path, dtype=torch.float16)
    mixer = reference.model.language_model.hyper_connection_mixer.cuda()
    torch.manual_seed(1)
    streams = torch.randn(3, 4 * 512, dtype=torch.float16, device="cuda")
    with torch.no_grad():
        torch.testing.assert_close(model.runner.final_norm_device(streams), mixer(streams))
