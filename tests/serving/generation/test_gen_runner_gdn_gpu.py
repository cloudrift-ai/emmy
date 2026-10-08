"""GDN (gated DeltaNet) layers through ``EmmyGenRunner``, without vLLM.

Needs CUDA (skips itself otherwise). The model is a tiny Qwen3.5 whose layers are all linear attention, so a
whole-model step is ``embed`` → one chain of GDN programs per layer → ``final_norm`` → lm_head, with no attention
stitch. The caller owns each request's recurrent state and convolution history; the runner keeps none. The
reference is the Hugging Face model, with its own cache where a request spans several steps. fp32.
"""

import pytest

# NOT perf-marked, for the reason ``test_gen_runner_gpu`` gives: these are correctness pins.
pytestmark = [
    pytest.mark.xdist_group("cuda"),
    pytest.mark.skip(
        reason="a regenerated serving golden authors a one-CTA leftover GDN piece that holds the GPU for minutes, "
        "and each runner build spends about 24 minutes pricing kernel sets; skipped on every card until fixed",
    ),
]

RUNNER = "qwen3_5.gdn.l2"


def _fresh_state(model):
    """Zero recurrent state and convolution history per GDN layer — what starts a request. An attention layer
    holds ``None``."""
    import torch

    dtype = model.lm_head.weight.dtype  # the history is in the trunk dtype; the recurrent state is always float32
    return [
        (
            torch.zeros(1, mixer.num_v_heads, mixer.head_k_dim, mixer.head_v_dim, device="cuda"),
            torch.zeros(1, mixer.conv_dim, mixer.conv_kernel_size, dtype=dtype, device="cuda"),
        )
        if mixer is not None
        else None
        for mixer in (getattr(layer, "linear_attn", None) for layer in model.model.layers)
    ]


def _step(pair, input_ids, states):
    """One step of ONE request through the runner: ``input_ids`` continue the request that ``states`` belongs
    to. Returns the ``[T, vocab]`` logits and leaves ``states`` advanced."""
    import torch

    runner = pair.runner
    hidden = runner.embed_device(torch.tensor(input_ids, device="cuda"))
    for layer, (state, history) in enumerate(states):
        hidden = runner.forward_layer_gdn_device(layer, hidden, state, history)
    with torch.no_grad():
        return pair.model.lm_head(runner.final_norm_device(hidden).cpu())


def _eager(model, input_ids):
    """The Hugging Face model's ``[T, vocab]`` logits for the whole sequence, from zero state. Always without its
    cache: with linear-attention layers only, ``DynamicCache`` has no layer to read a sequence length from."""
    import torch

    with torch.no_grad():
        return model(torch.tensor([input_ids]), use_cache=False).logits[0]


def _prompt(length):
    return [1 + i % 63 for i in range(length)]


@pytest.mark.parametrize("length", [1, 5, 16, 17, 63, 64, 65, 131])
def test_gdn_prompt_of_any_length_matches_eager(built, length):
    """No GDN program pads, so a prompt is decomposed into the static widths (16, 4, 1 here) and the state threads
    through the calls. The lengths sit on both sides of those widths and of the 64-token chunk the Hugging Face
    layer computes in."""
    import torch

    pair = built(RUNNER)
    ids = _prompt(length)
    logits = _step(pair, ids, _fresh_state(pair.model))
    torch.testing.assert_close(logits, _eager(pair.model, ids), rtol=2e-3, atol=2e-3)


def test_gdn_decode_steps_continue_the_prompt(built):
    """Prefill hands its state to decode: a prompt, then single-token steps. Each step's logits must equal the
    eager model's logits for those positions of the whole sequence so far."""
    import torch

    pair = built(RUNNER)
    prompt, tail = _prompt(21), [7, 3, 11]
    states = _fresh_state(pair.model)
    torch.testing.assert_close(_step(pair, prompt, states), _eager(pair.model, prompt), rtol=2e-3, atol=2e-3)
    for end in range(1, len(tail) + 1):
        eager = _eager(pair.model, prompt + tail[:end])[-1:]
        torch.testing.assert_close(_step(pair, tail[end - 1 : end], states), eager, rtol=2e-3, atol=2e-3)


def test_gdn_request_starts_clean_after_another(built):
    """The runner keeps no state of its own: a request gives the same logits whether or not another request,
    with its own state, ran before it. Same up to rounding, not bit for bit: a cross-CTA split that combines its
    partials with atomic adds sums them in launch order, which moves the last bits; leaked state moves far more."""
    import torch

    pair = built(RUNNER)
    alone = _step(pair, _prompt(9), _fresh_state(pair.model))
    other = _fresh_state(pair.model)
    _step(pair, _prompt(21)[::-1], other)
    _step(pair, [5], other)
    after = _step(pair, _prompt(9), _fresh_state(pair.model))
    torch.testing.assert_close(after, alone, rtol=0, atol=1e-6)


def test_gdn_padded_step_corrupts_the_state(built):
    """Why no GDN program pads: a zero row still decays the recurrent state and enters the convolution history.
    Three tokens padded to the width-4 program leave a different state than the same three tokens alone. If the
    runner ever padded a short call up to a program width, both paths would run the same padded program and the
    states would agree, so this test also pins that it does not."""
    import torch

    pair = built(RUNNER)
    runner = pair.runner
    hidden = runner.embed_device(torch.tensor(_prompt(3), device="cuda"))
    (state, history), (padded_state, padded_history) = _fresh_state(pair.model)[0], _fresh_state(pair.model)[0]
    runner.forward_layer_gdn_device(0, hidden, state, history)
    runner.forward_layer_gdn_device(0, torch.cat([hidden, torch.zeros_like(hidden[:1])]), padded_state, padded_history)
    assert not torch.allclose(padded_history, history)
    assert not torch.allclose(padded_state, state)


@pytest.mark.parametrize("length", [1, 5, 21])
def test_gdn_bf16_prompt_matches_eager(built, length):
    """The Qwen3.8 checkpoints are BF16: the same GDN programs with BF16 activations and history, and the float32
    recurrent state. The reference is the eager model computing in float32 on the same BF16 weights. BF16 keeps 8
    significant bits, so the tolerance is a few BF16 steps at the logits' magnitude, not the float32 tolerance."""
    import copy

    import torch

    pair = built("qwen3_5.gdn.l2.bf16")
    ids = _prompt(length)
    logits = _step(pair, ids, _fresh_state(pair.model)).float()
    eager = _eager(copy.deepcopy(pair.model).float(), ids)
    # The logits are of magnitude 0.5 here, where one BF16 step is 0.002: allow a few steps.
    torch.testing.assert_close(logits, eager, rtol=0, atol=1e-2)


@pytest.fixture(scope="module")
def hybrid():
    """The tiny hybrid Qwen3.5: one GDN layer, then one full-attention layer that carries an output gate. The lane's
    golden decides the GDN kernels — an uncut GDN kernel takes minutes per call. It holds no row for the attention
    programs, so those compile cold, as the gated-attention tests in ``test_gen_runner_gpu`` do. That is why this
    runner is not a ``RUNNERS`` entry: the table builds under strict evidence, which a cold attention compile fails."""
    import torch
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from emmy.compiler.pipeline.search.golden import evidence_scope
    from emmy.compiler.pipeline.search.pins import pinned_knobs
    from emmy.serving.gen_runner import EmmyGenRunner
    from tests.serving import helpers
    from tests.serving.conftest import Built
    from tests.serving.helpers import QWEN3_5_TINY

    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**QWEN3_5_TINY)).eval()
    document = helpers.golden_document()
    with pinned_knobs(document.shared_regime()), evidence_scope([document]):
        runner = EmmyGenRunner.from_model(model, dtype_str="float32", decode_bucket=4, prefill_bucket=16, max_tokens=32)
    return Built(runner, model)


# 1: single-token programs; 3: the decode bucket; 9: the symbolic attention program; 16: the prefill bucket;
# 18: the attention rider split, and GDN widths 16 + 1 + 1.
@pytest.mark.parametrize("length", [1, 3, 9, 16, 18])
def test_hybrid_prompt_matches_eager(hybrid, length):
    """GDN and attention layers in one model. The GDN layer runs its exact-width programs; the attention layer runs
    ``pre`` → the Hugging Face rotary and causal attention → ``post``, at whichever tier serves this length."""
    import torch
    import torch.nn.functional as F
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    from emmy.compiler.trace.huggingface import build_causal_mask
    from tests.serving.generation.test_gen_runner_gpu import _repeat_kv

    runner, model = hybrid.runner, hybrid.model
    ids = _prompt(length)
    hidden = runner.embed_device(torch.tensor(ids, device="cuda"))
    cos, sin = (t.cuda() for t in model.model.rotary_emb(hidden[None].cpu(), torch.arange(length)[None]))
    mask = build_causal_mask(length, torch.float32).cuda()
    for layer, gdn_state in enumerate(_fresh_state(model)):
        if gdn_state is not None:
            hidden = runner.forward_layer_gdn_device(layer, hidden, *gdn_state)
            continue
        hd, nh, nkv, scaling = runner.layer_meta(layer)
        q, k, v, gate = runner.forward_layer_pre_device(layer, hidden)
        q = q.view(length, nh, hd).transpose(0, 1)[None]
        k = k.view(length, nkv, hd).transpose(0, 1)[None]
        v = v.view(length, nkv, hd).transpose(0, 1)[None]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k, v = _repeat_kv(k, nh // nkv), _repeat_kv(v, nh // nkv)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scaling).transpose(1, 2).reshape(length, nh * hd)
        hidden = runner.forward_layer_post_device(layer, attn.contiguous(), hidden, gate)
    with torch.no_grad():
        logits = model.lm_head(runner.final_norm_device(hidden).cpu())
    torch.testing.assert_close(logits, _eager(model, ids), rtol=2e-3, atol=2e-3)
