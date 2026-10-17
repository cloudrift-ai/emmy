"""Hermetic equivalence tests for the Qwen4Exp (Qwen3.8-Flash-Next) hyper-connection seam (no GPU/vLLM).

Proves the carve on a tiny random Qwen4Exp layer of each kind: the n-gram PLE (layer 2 only) and the token mixer —
Gated DeltaNet or sparse attention — are the block's own modules standing in for the 1Cat fork's, between ``pre``
(the attention-site gated residual) and ``post`` (mixer injection, MLP-site gated residual, gated shared expert);
the runner-style torch half (HF router → per-expert ``expert(x, w_gate_up, w_down)`` → weighted combine →
per-stream placement) closes the layer. The result must reproduce the eager ``Qwen4ExpTextDecoderLayer`` over the
flattened ``[num_tokens, hc_count * hidden]`` carrier. Pure eager, CPU, fp32 — no compile.
"""

import pytest

from tests.serving.helpers import QWEN4_EXP_TINY, per_expert


def _random_model(torch):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextModel

    torch.manual_seed(0)
    model = Qwen4ExpTextModel(Qwen4ExpTextConfig(**QWEN4_EXP_TINY)).eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.2)
    for layer in model.layers:
        layer.mlp.gate.weight.data.mul_(5.0)  # spread the router so tokens pick distinct experts
    return model


def _mixer_output(torch, model, block, pre, t):
    """The eager layer's output for random streams, and what ``post`` receives at the seam: the carrier (after the
    fork's n-gram PLE on layer 2) and the token mixer's output, both from the block's own modules standing in for
    the fork's."""
    from transformers.masking_utils import create_causal_mask

    h, hc = model.config.hidden_size, model.config.hc_count
    ids = torch.randint(0, model.config.vocab_size, (1, t))
    positions = torch.arange(t).unsqueeze(0)
    streams = torch.randn(1, t, hc * h)
    rope = model.rotary_emb(streams, positions)
    mask = create_causal_mask(
        config=model.config, inputs_embeds=streams, attention_mask=None, past_key_values=None, position_ids=positions,
        allow_is_causal_skip=False,
    )  # fmt: skip
    ref = block(streams, rope, attention_mask=mask, ple_input_ids=ids).reshape(t, hc * h)
    carrier = streams
    if block.ple is not None:  # the fork's n-gram PLE stands here
        carrier = carrier + block.ple(carrier, ids, None)
    carrier = carrier.reshape(t, hc * h)
    x = pre(carrier)
    assert x.shape == (t, h)
    if block.layer_type == "linear_attention":  # the fork's Gated DeltaNet stands here
        mixer_out = block.linear_attn(x.unsqueeze(0), cache_params=None)
    else:  # the fork's sparse attention stands here
        mixer_out, _ = block.self_attn(x.unsqueeze(0), rope, attention_mask=mask)
    return ref, carrier, mixer_out.squeeze(0)


@pytest.mark.parametrize("layer_idx", [0, 1, 3])  # Gated DeltaNet, Gated DeltaNet + n-gram PLE, sparse attention
def test_gated_residual_split_matches_eager_layer(layer_idx):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers.models.qwen4_exp")

    from emmy.compiler.trace.huggingface import build_moe_split_wrapper, hyper_connection_seam, moe_block_parts, place_routed_streams
    from emmy.serving.gen_runner import combine_routed_experts

    model = _random_model(torch)
    block = model.layers[layer_idx]
    t, h, hc = 12, model.config.hidden_size, model.config.hc_count
    assert hyper_connection_seam(block) == (hc * h, h)

    pre, post, expert = build_moe_split_wrapper(block)
    gate, experts = moe_block_parts(block.mlp)
    with torch.no_grad():
        ref, carrier, mixer_out = _mixer_output(torch, model, block, pre, t)
        mixed, xn, mix = post(mixer_out, carrier)
        assert mixed.shape == (t, hc * h) and xn.shape == (t, h) and mix.shape == (t, hc)
        routed = combine_routed_experts(
            xn,
            gate(xn),
            per_expert(lambda e, rows: expert(rows, experts.gate_up_proj[e], experts.down_proj[e])),
            num_experts=experts.gate_up_proj.shape[0],
        )
        got = place_routed_streams(mixed, routed, mix)
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


def test_tensor_parallel_ranks_sum_to_the_eager_layer():
    """At tensor parallelism every rank runs ``pre`` and ``post`` whole, the gated shared expert included, and
    its slice of every routed expert. The all-reduce sums only the routed partials, which are then placed into
    the streams. Two ranks together must reproduce the eager layer."""
    import copy

    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers.models.qwen4_exp")

    from emmy.compiler.trace.huggingface import build_moe_split_wrapper, moe_block_parts, place_routed_streams, slice_routed_experts
    from emmy.serving.gen_runner import combine_routed_experts

    model, layer_idx, t, world = _random_model(torch), 3, 12, 2
    block = model.layers[layer_idx]
    with torch.no_grad():
        ref, carrier, mixer_out = _mixer_output(torch, model, block, build_moe_split_wrapper(block)[0], t)
        halves, routed = [], 0
        for rank in range(world):
            sliced = copy.deepcopy(model)
            slice_routed_experts(sliced, rank, world)
            rank_block = sliced.layers[layer_idx]
            _, post, expert = build_moe_split_wrapper(rank_block)
            gate, experts = moe_block_parts(rank_block.mlp)
            assert experts.down_proj.shape[-1] == model.config.moe_intermediate_size // world
            mixed, xn, mix = post(mixer_out, carrier)
            halves.append((mixed, xn, mix))
            routed = routed + combine_routed_experts(
                xn,
                gate(xn),
                per_expert(lambda e, rows, experts=experts, expert=expert: expert(rows, experts.gate_up_proj[e], experts.down_proj[e])),
                num_experts=experts.gate_up_proj.shape[0],
            )
        for other in halves[1:]:  # post is replicated: every rank computes the same streams and routing input
            for a, b in zip(halves[0], other, strict=True):
                assert torch.equal(a, b)
        mixed, _, mix = halves[0]
        got = place_routed_streams(mixed, routed, mix)
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)
