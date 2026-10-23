"""EmmyGenModel's speculative-decoding shim — the ``.model.embed_tokens`` surface vLLM's MTP
drafter shares off the target model (no GPU, no vllm).

The drafter reaches into ``target.model.embed_tokens`` and adopts it as the draft head's own embedding, so the
shim gathers from the table the runner gathers from, with no copy. vLLM's Gemma-4 MTP drafter applies a
``* sqrt(hidden)`` normalizer itself, so a target that scales its embedding must share the raw tied ``lm_head``
weight the runner adopted. These tests pin that contract against the shim classes, without constructing the full
model (needs vllm/CUDA).
"""

import types

import pytest

torch = pytest.importorskip("torch")


def _shim(model):
    vllm_model_gen = pytest.importorskip("emmy.serving.vllm_model_gen")
    return vllm_model_gen._EmmyTargetInner(vllm_model_gen._SharedRawEmbedding(model))


def _target(embed_dev_is_tied, scale=1.0):
    """Stand-in for a constructed EmmyGenModel exposing only what the shim reads: ``lm_head.weight`` and the
    runner's table and embed scale. ``embed_dev_is_tied`` chooses whether the runner gathers from the SAME tensor
    as ``lm_head.weight`` (the tied gemma-4 path) or from its own table beside a distinct output head."""
    param = torch.nn.Parameter(torch.randn(7, 4))
    embed_dev = param.data if embed_dev_is_tied else torch.randn(7, 4)
    return types.SimpleNamespace(
        lm_head=types.SimpleNamespace(weight=param),
        runner=types.SimpleNamespace(_embed_weight_dev=embed_dev, _embed_scale=scale),
    )


@pytest.mark.parametrize("tied", [True, False])
def test_drafter_shares_the_table_the_runner_gathers_from(tied):
    model = _target(embed_dev_is_tied=tied)
    embed = _shim(model).embed_tokens
    assert embed.weight.data_ptr() == model.runner._embed_weight_dev.data_ptr()
    ids = torch.tensor([0, 3, 6])
    assert torch.equal(embed(ids), torch.nn.functional.embedding(ids, model.runner._embed_weight_dev))
    embed.validate()


def test_scaled_embedding_must_be_the_raw_tied_head():
    _shim(_target(embed_dev_is_tied=True, scale=2.0)).embed_tokens.validate()
    embed = _shim(_target(embed_dev_is_tied=False, scale=2.0)).embed_tokens
    with pytest.raises(RuntimeError, match="tied"):
        embed.validate()
    # forward itself carries no guard (torch.compile traces it): it gathers regardless.
    assert embed(torch.tensor([0, 1])).shape == (2, 4)


def test_drafter_computes_logits_with_a_coded_target_head():
    """vLLM's MTP proposer sets the draft model's ``lm_head`` to the target's (``_maybe_share_lm_head``). A coded
    head is a module that computes f32 logits itself and holds no weight, so the draft passes its logits through a
    ``logits_as_input`` processor instead of asking the head for a weight."""
    pytest.importorskip("vllm")
    from vllm.config import VllmConfig
    from vllm.config.vllm import set_current_vllm_config
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    from emmy.serving.vllm_model_mtp import EmmyMTPModel
    from tests.serving.generation.helpers import coded_head_double

    weight = torch.randn(6, 4)
    target = types.SimpleNamespace(lm_head=coded_head_double(weight))
    draft = EmmyMTPModel.__new__(EmmyMTPModel)
    torch.nn.Module.__init__(draft)
    with set_current_vllm_config(VllmConfig()):
        # The two processors ``EmmyMTPModel.__init__`` builds; the rest of it needs the engine.
        draft.logits_processor = LogitsProcessor(6)
        draft.coded_logits_processor = LogitsProcessor(6, logits_as_input=True)
    draft.lm_head = target.lm_head  # the proposer's share
    hidden = torch.randn(3, 4)
    torch.testing.assert_close(draft.compute_logits(hidden), hidden @ weight.T)
