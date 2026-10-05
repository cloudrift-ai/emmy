"""The split Llama programs apply each adapter projection to selected tokens."""

import copy

import pytest

from emmy.serving.lora import PROJECTIONS, build_lora_attention_split_wrapper, projection_modules


@pytest.mark.parametrize("projection", PROJECTIONS)
def test_lora_split_matches_weight_update_for_mixed_tokens(projection):
    torch = pytest.importorskip("torch")
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    torch.manual_seed(4)
    config = LlamaConfig(hidden_size=32, intermediate_size=64, num_attention_heads=4, num_key_value_heads=2)
    block = LlamaDecoderLayer(config, layer_idx=0)
    updated = copy.deepcopy(block)
    module = projection_modules(block)[projection]
    updated_module = projection_modules(updated)[projection]
    rank = 2
    a = torch.randn(rank, module.in_features) * 0.1
    b = torch.randn(module.out_features, rank) * 0.1
    with torch.no_grad():
        updated_module.weight.add_(b @ a)

    adapters = {
        name: (
            (a, b)
            if name == projection
            else (torch.zeros(rank, part.in_features), torch.zeros(part.out_features, rank))
        )
        for name, part in projection_modules(block).items()
    }
    mask = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    hidden = torch.randn(4, config.hidden_size)
    attn_out = torch.randn(4, config.hidden_size)
    pre, post = build_lora_attention_split_wrapper(block)
    pre_updated, post_updated = build_lora_attention_split_wrapper(updated)

    pre_args = [tensor for name in PROJECTIONS[:3] for tensor in adapters[name]]
    post_args = [tensor for name in PROJECTIONS[3:] for tensor in adapters[name]]
    actual_pre = pre(hidden, mask, *pre_args)
    actual_post = post(attn_out, hidden, mask, *post_args)
    base_pre = pre(hidden, torch.zeros_like(mask), *pre_args)
    base_post = post(attn_out, hidden, torch.zeros_like(mask), *post_args)
    updated_pre = pre_updated(hidden, torch.zeros_like(mask), *[torch.zeros_like(tensor) for tensor in pre_args])
    updated_post = post_updated(attn_out, hidden, torch.zeros_like(mask), *[torch.zeros_like(tensor) for tensor in post_args])

    for actual, base, adapted in zip(actual_pre, base_pre, updated_pre, strict=True):
        torch.testing.assert_close(actual[::2], base[::2], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual[1::2], adapted[1::2], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual_post[::2], base_post[::2], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_post[1::2], updated_post[1::2], atol=1e-5, rtol=1e-5)
    affected = actual_pre[PROJECTIONS.index(projection)] if projection in PROJECTIONS[:3] else actual_post
    baseline = base_pre[PROJECTIONS.index(projection)] if projection in PROJECTIONS[:3] else base_post
    assert not torch.allclose(affected[1::2], baseline[1::2])
