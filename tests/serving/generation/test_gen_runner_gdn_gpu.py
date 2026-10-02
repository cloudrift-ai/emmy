"""GDN (gated DeltaNet) layers through ``EmmyGenRunner``, without vLLM.

Needs CUDA (skips itself otherwise). The model is a tiny Qwen3.5 whose layers are all linear attention, so a
whole-model step is ``embed`` → one chain of GDN programs per layer → ``final_norm`` → lm_head, with no attention
stitch. The caller owns each request's recurrent state and convolution history; the runner keeps none. The
reference is the Hugging Face model, with its own cache where a request spans several steps. fp32.
"""

import pytest

# NOT perf-marked, for the reason ``test_gen_runner_gpu`` gives: these are correctness pins.
pytestmark = [pytest.mark.xdist_group("cuda")]

RUNNER = "qwen3_5.gdn.l2"


def _fresh_state(model):
    """Zero recurrent state and convolution history per layer — what starts a request."""
    import torch

    return [
        (
            torch.zeros(1, mixer.num_v_heads, mixer.head_k_dim, mixer.head_v_dim, device="cuda"),
            torch.zeros(1, mixer.conv_dim, mixer.conv_kernel_size, device="cuda"),
        )
        for mixer in (layer.linear_attn for layer in model.model.layers)
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
    """The runner keeps no state of its own: a request gives bit-identical logits whether or not another
    request, with its own state, ran before it."""
    import torch

    pair = built(RUNNER)
    alone = _step(pair, _prompt(9), _fresh_state(pair.model))
    other = _fresh_state(pair.model)
    _step(pair, _prompt(21)[::-1], other)
    _step(pair, [5], other)
    after = _step(pair, _prompt(9), _fresh_state(pair.model))
    torch.testing.assert_close(after, alone, rtol=0, atol=0)


def test_gdn_padded_step_corrupts_the_state(built):
    """Why no GDN program pads: a zero row still decays the recurrent state and enters the convolution history.
    Three tokens padded to the width-4 program leave a different state than the same three tokens alone."""
    import torch

    pair = built(RUNNER)
    runner = pair.runner
    hidden = runner.embed_device(torch.tensor(_prompt(3), device="cuda"))
    (state, history), (padded_state, padded_history) = _fresh_state(pair.model)[0], _fresh_state(pair.model)[0]
    runner.forward_layer_gdn_device(0, hidden, state, history)
    runner.forward_layer_gdn_device(0, torch.cat([hidden, torch.zeros_like(hidden[:1])]), padded_state, padded_history)
    assert not torch.allclose(padded_history, history)
    assert not torch.allclose(padded_state, state)
