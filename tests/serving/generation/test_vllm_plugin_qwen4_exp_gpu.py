"""`EmmyGenHybridModel` hosting the 1Cat fork's token mixers (Qwen3.8-Flash-Next).

The model class must CONSTRUCT under a real vLLM config — vLLM's ``QwenGatedDeltaNetAttention`` per local Gated
DeltaNet layer, registered under the absolute layer's ``linear_attn`` prefix, and the fork's ``Qwen4ExpQSAAttention``
per local sparse-attention layer, each registering its main cache and the indexer's two side caches under the
absolute layer's ``self_attn`` prefix, all sharing one top-k buffer, and the fork's n-gram PLE on layer 2 (one-indexed), under its
``ple`` prefix. The hybrid class must declare the state the Gated DeltaNet layers hold, and the checkpoint's mixer and
PLE weights must LOAD into the real modules: the sparse attention's query, key and value as its fused projection's
shards, the Gated DeltaNet's projections as the shards of its two merged ones, the PLE through its own loader.

Runs only where the fork is importable and the card is the one it was built for (the pinned 1Cat image on sm_70);
skipped everywhere else. ``test_vllm_plugin_qwen4_exp_engine_gpu`` serves the same checkpoint end to end.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.perf, pytest.mark.xdist_group("cuda")]

# The published order: layer 2 (one-indexed) reads the n-gram PLE, and the config allows it only on a Gated DeltaNet layer.
LAYER_TYPES = ["linear_attention", "linear_attention", "full_attention"]


def _build_model(tmp_path, torch):
    """The in-process build a worker does: engine config, single-rank groups, construction on the card."""
    import socket

    from vllm.config import set_current_vllm_config
    from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
    from vllm.engine.arg_utils import EngineArgs

    from emmy.serving.vllm_model_gen import EmmyGenHybridModel
    from tests.serving.helpers import save_tiny_qwen4_exp

    save_tiny_qwen4_exp(tmp_path, LAYER_TYPES)
    vllm_config = EngineArgs(
        model=str(tmp_path),
        runner="generate",
        dtype="float16",
        max_model_len=256,
        max_num_batched_tokens=256,
        enforce_eager=True,
        enable_prefix_caching=False,
        language_model_only=True,
        skip_tokenizer_init=True,
        hf_overrides={"architectures": ["EmmyGenHybridModel"]},
    ).create_engine_config()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(world_size=1, rank=0, distributed_init_method=f"tcp://127.0.0.1:{port}", local_rank=0)
        ensure_model_parallel_initialized(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        with torch.device("cuda"):
            model = EmmyGenHybridModel(vllm_config=vllm_config, prefix="model")
    return vllm_config, model


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("vllm.models.qwen4_exp.nvidia.qsa")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    path = tmp_path_factory.mktemp("qwen4_exp")
    with pytest.MonkeyPatch.context() as env:
        # Construction and loading need only the symbolic programs. The static decode and prefill tiers would add
        # their compiles and the boot audit's runs of picks nothing measured for this checkpoint.
        env.setenv("EMMY_GEN_DECODE_BUCKET", "0")
        env.setenv("EMMY_GEN_PREFILL_BUCKET", "0")
        # The fork lays out the sparse attention's ring cache only in its Model Runner V2.
        env.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
        # The PLE's n-gram table goes to host memory whole. Left to place itself, the fork sizes the KV cache for it
        # before vLLM has widened the attention pages to the Gated DeltaNet state, and refuses the narrower pages.
        env.setenv("VLLM_QWEN4EXP_PLE_HOST_GIB", "1")
        vllm_config, model = _build_model(path, torch)
        yield path, vllm_config, model
    # The engine module boots its own groups in this process.
    cleanup_dist_env_and_memory()


def test_each_layer_hosts_the_forks_token_mixer(built):
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
    from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpPLELayer
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAAttention

    _path, vllm_config, model = built
    assert [type(m) for m in model.fork_attn] == [QwenGatedDeltaNetAttention, QwenGatedDeltaNetAttention, Qwen4ExpQSAAttention]
    assert len(model.attn) == 0 and len(model.rotary_emb) == 0
    assert all(state is None for state in model.gdn_state)  # Emmy runs no GDN layer of its own here
    registered = vllm_config.compilation_config.static_forward_context
    assert registered["model.layers.1.linear_attn"] is model.fork_attn[1]
    site = "model.layers.2.self_attn"
    assert {f"{site}.attn", f"{site}.indexer.raw_key_cache", f"{site}.indexer.compressed_key_cache"} <= set(registered)
    assert list(model.fork_ple) == ["1"] and isinstance(model.fork_ple["1"], Qwen4ExpPLELayer)
    assert registered["model.layers.1.ple"] is model.fork_ple["1"]
    assert model.runner.carrier_size == 4 * 512


def test_the_hybrid_class_declares_the_state_the_hosted_layer_holds(built):
    """vLLM sizes every request's state pages from the class before the model exists; the Gated DeltaNet layer then
    allocates exactly what it declares itself."""
    from emmy.serving.vllm_model_gen import EmmyGenHybridModel

    _path, vllm_config, model = built
    layer = model.fork_attn[0]
    assert tuple(EmmyGenHybridModel.get_mamba_state_shape_from_config(vllm_config)) == tuple(layer.get_state_shape())
    assert tuple(EmmyGenHybridModel.get_mamba_state_dtype_from_config(vllm_config)) == tuple(layer.get_state_dtype())


def test_checkpoint_mixer_weights_load_into_the_hosted_modules(built):
    """Stream the checkpoint itself through ``load_weights``: every hosted parameter is claimed and holds the
    checkpoint's values, the fused projections in their shard order."""
    import torch
    from safetensors.torch import load_file
    from vllm.config import set_current_vllm_config

    path, vllm_config, model = built
    tensors = load_file(str(path / "model.safetensors"))

    with set_current_vllm_config(vllm_config):  # as a worker loads: the PLE reads the engine config to place its table
        loaded = model.load_weights(iter(tensors.items()))

    for layer, module in enumerate(model.fork_attn):
        params = dict(module.named_parameters())
        assert {f"fork_attn.{layer}.{name}" for name in params} <= loaded
        site = f"model.language_model.layers.{layer}.{'self_attn' if LAYER_TYPES[layer] == 'full_attention' else 'linear_attn'}."
        source = {k.removeprefix(site): v for k, v in tensors.items() if k.startswith(site)}
        fused = {
            "qkv_proj.weight": ("q_proj.weight", "k_proj.weight", "v_proj.weight"),
            "in_proj_qkvz.weight": ("in_proj_qkv.weight", "in_proj_z.weight"),
            "in_proj_ba.weight": ("in_proj_b.weight", "in_proj_a.weight"),
        }
        for name, param in params.items():
            want = torch.cat([source[part] for part in fused[name]]) if name in fused else source.get(name)
            assert want is not None, f"layer {layer}: no checkpoint source for {name}"
            got = param.detach().cpu()
            assert torch.equal(got, want.to(param.dtype).reshape(got.shape)), f"layer {layer}: {name}"
    ple = model.fork_ple["1"]
    assert {f"fork_ple.1.{name}" for name, _ in ple.named_parameters()} <= loaded
    site = "model.language_model.layers.1.ple."
    source = {k.removeprefix(site): v for k, v in tensors.items() if k.startswith(site)}
    # The n-gram table's own parameter is a placeholder: the fork keeps the rows in tables of its own.
    for name, param in ple.named_parameters():
        if name in source:
            assert torch.equal(param.detach().cpu(), source[name].to(param.dtype).reshape(param.shape)), f"PLE: {name}"
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
