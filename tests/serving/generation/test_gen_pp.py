"""Pipeline ownership tests for the vLLM generative plugin (CPU-only)."""

from types import SimpleNamespace

import pytest


def test_generative_trunk_dtype_accepts_bfloat16():
    import torch

    from emmy.serving.trunk_dtype import _trunk_dtype_str

    assert _trunk_dtype_str(torch.bfloat16) == "bfloat16"
    assert _trunk_dtype_str(torch.bfloat16, allow_bf16=False) == "float16"


def test_bf16_host_fallback_names_width():
    pytest.importorskip("vllm")
    import torch

    from emmy.serving.vllm_model_gen import EmmyGenModel

    model = EmmyGenModel.__new__(EmmyGenModel)
    torch.nn.Module.__init__(model)
    model.runner = SimpleNamespace(
        maybe_log_routing_histogram=lambda: None, has_device_decode=False, decode_bucket=0,
        prefill_capacity=2, prefill_bucket=0, rider_width=0, residual_dtype=torch.bfloat16,
    )
    model.config = SimpleNamespace(vocab_size=32)
    model._is_first_rank = False
    model.fork_attn = None
    hidden = torch.zeros(3, 8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="BF16 token width 3.*lower --max-num-batched-tokens"):
        model.forward(None, torch.arange(3), intermediate_tensors={"hidden_states": hidden})


def test_pipeline_ranges_cover_every_layer_once_with_uneven_partition(monkeypatch):
    pytest.importorskip("vllm")
    from emmy.serving.vllm_model_gen import _pipeline_layer_range

    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    ranges = [_pipeline_layer_range(50, SimpleNamespace(rank_in_group=rank, world_size=8)) for rank in range(8)]

    assert ranges[0][0] == 0
    assert ranges[-1][1] == 50
    assert all(left[1] == right[0] for left, right in zip(ranges[:-1], ranges[1:], strict=True))
    assert sorted(layer for start, end in ranges for layer in range(start, end)) == list(range(50))


def test_pipeline_ranges_assign_exact_laguna_intervals(monkeypatch):
    pytest.importorskip("vllm")
    from emmy.serving.vllm_model_gen import _pipeline_layer_range

    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    ranges = [_pipeline_layer_range(48, SimpleNamespace(rank_in_group=rank, world_size=8)) for rank in range(8)]

    assert ranges == [(0, 6), (6, 12), (12, 18), (18, 24), (24, 30), (30, 36), (36, 42), (42, 48)]


def test_pipeline_hidden_buffers_use_runner_residual_dtype():
    pytest.importorskip("vllm")
    import torch

    from emmy.serving.vllm_model_gen import _hidden_intermediate_tensors_factory

    make = _hidden_intermediate_tensors_factory(3072, torch.float32)
    tensors = make(batch_size=2, dtype=torch.float16, device=torch.device("cpu"))

    assert tensors["hidden_states"].shape == (2, 3072)
    assert tensors["hidden_states"].dtype == torch.float32
