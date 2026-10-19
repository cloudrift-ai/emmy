"""What ``EmmyGenModel`` refuses on a checkpoint with GDN (gated DeltaNet) layers, and why. CPU; needs vllm."""

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")


def _vllm_config(*, prefix_caching=False, eager=True, cudagraph_mode="NONE"):
    """The settings the GDN boot check reads, as a stand-in for a ``VllmConfig``."""
    return SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=prefix_caching),
        speculative_config=None,
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name=cudagraph_mode)),
        model_config=SimpleNamespace(enforce_eager=eager),
    )


def test_gdn_layers_boot_only_on_the_hybrid_class():
    """vLLM allocates per-request state only for a class that carries the hybrid flag, so the plain class names
    the class to use instead of serving GDN layers without state."""
    from emmy.serving.vllm_model_gen import EmmyGenHybridModel, EmmyGenModel

    with pytest.raises(ValueError, match="EmmyGenHybridModel"):
        EmmyGenModel._check_gdn_serving(_vllm_config())
    EmmyGenHybridModel._check_gdn_serving(_vllm_config())


@pytest.mark.parametrize(
    ("setting", "error", "reason"),
    [
        ({"prefix_caching": True}, NotImplementedError, "prefix caching"),
        ({"eager": False, "cudagraph_mode": "FULL_AND_PIECEWISE"}, ValueError, "FULL_DECODE_ONLY"),
        ({"eager": False, "cudagraph_mode": "FULL"}, ValueError, "FULL_DECODE_ONLY"),
        ({"eager": False, "cudagraph_mode": "PIECEWISE"}, ValueError, "FULL_DECODE_ONLY"),
    ],
)
def test_gdn_serving_refuses_what_would_corrupt_or_cannot_record_the_state(setting, error, reason):
    """A request keeps one recurrent state per GDN layer, with no snapshot a cached prefix could resume from. A
    capture of a prefill or mixed step would replay its own per-request token ranges, which a GDN layer reads on the
    host, on steps whose ranges differ."""
    from emmy.serving.vllm_model_gen import EmmyGenHybridModel

    with pytest.raises(error, match=reason):
        EmmyGenHybridModel._check_gdn_serving(_vllm_config(**setting))


def test_gdn_serving_captures_decode_steps():
    """A uniform decode step repeats the token ranges its capture recorded: one token per request."""
    from emmy.serving.vllm_model_gen import EmmyGenHybridModel

    EmmyGenHybridModel._check_gdn_serving(_vllm_config(eager=False, cudagraph_mode="FULL_DECODE_ONLY"))


def test_gdn_state_layout_matches_the_layer_programs():
    """The state vLLM allocates per request must have the shapes of a GDN program's ``history`` and ``state``
    inputs: the convolution history in the trunk dtype, then the float32 recurrent matrix."""
    import torch
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from emmy.serving.vllm_model_gen import _gdn_state_layout
    from tests.serving.helpers import QWEN3_5_TINY

    config = Qwen3_5TextConfig(**QWEN3_5_TINY)
    mixer = Qwen3_5ForCausalLM(config).model.layers[0].linear_attn
    shapes, dtypes = _gdn_state_layout(config, torch.float16)
    assert shapes == ((mixer.conv_dim, mixer.conv_kernel_size), (mixer.num_v_heads, mixer.head_k_dim, mixer.head_v_dim))
    assert dtypes == (torch.float16, torch.float32)


def test_mrope_positions_for_text_are_the_token_positions_on_every_axis():
    """vLLM asks a model for M-RoPE positions whenever the config carries ``mrope_section``, as Qwen3.5 / Qwen3.8
    do, and would otherwise refuse the first request. For text every axis carries the token position; vision
    inputs are refused."""
    import torch

    from emmy.serving.vllm_model_gen import EmmyGenHybridModel, EmmyGenModel

    model = EmmyGenHybridModel.__new__(EmmyGenHybridModel)  # the method reads no state
    positions, delta = EmmyGenModel.get_mrope_input_positions(model, [5, 7, 11], [])
    assert positions.shape == (3, 3) and delta == 0
    assert torch.equal(positions, torch.arange(3).expand(3, -1))
    with pytest.raises(NotImplementedError, match="text only"):
        EmmyGenModel.get_mrope_input_positions(model, [5], [object()])
