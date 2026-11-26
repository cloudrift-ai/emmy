"""``EmmyGenModel.compute_logits`` honors the generation config's ``suppress_tokens`` (no GPU).

gemma-4 checkpoints list the mm delimiter tokens ('<image|>'/'<audio|>') there; a degenerate text
prompt can genuinely rank one top-1 (verified vs HF eager fp16), and an emitted mm token decodes to
empty text. HF generate and stock vLLM -inf those ids; the plugin must match.
"""

import pytest

torch = pytest.importorskip("torch")


def _model_with(suppress_ids):
    vllm_model_gen = pytest.importorskip("emmy.serving.vllm_model_gen")
    m = vllm_model_gen.EmmyGenModel.__new__(vllm_model_gen.EmmyGenModel)
    torch.nn.Module.__init__(m)
    # The surface compute_logits reads; the identity processor makes the hidden-states argument
    # stand in for the logits tensor. Only the last pipeline stage owns the head, so the double
    # is that stage — the single-rank case every non-pipelined deployment runs.
    m._is_last_rank = True
    m.logits_processor = lambda lm_head, hidden: hidden
    m.lm_head = None
    m._suppress_token_ids = suppress_ids
    return m


def test_suppressed_ids_are_neg_inf_others_untouched():
    out = _model_with([3, 5]).compute_logits(torch.zeros((2, 8)))
    assert torch.isneginf(out[:, [3, 5]]).all()
    keep = [i for i in range(8) if i not in (3, 5)]
    assert (out[:, keep] == 0).all()


def test_no_suppress_list_is_noop():
    out = _model_with(None).compute_logits(torch.zeros((2, 8)))
    assert (out == 0).all()


def test_coded_head_supplies_logits_before_the_processor():
    from tests.serving.generation.helpers import coded_head_double

    model = _model_with(None)
    model._coded_head_spec = object()
    model.lm_head = coded_head_double(torch.eye(8) * 7)
    out = model.compute_logits(torch.ones((2, 8)))
    assert (out == 7).all()
    model.lm_head = None
    with pytest.raises(RuntimeError, match="coded lm_head was not loaded"):
        model.compute_logits(torch.ones((2, 8)))


def test_earlier_pipeline_stage_refuses_to_compute_logits():
    """The head lives on the last stage only — the mirror of ``embed_input_ids``' first-stage guard."""
    model = _model_with(None)
    model._is_last_rank = False
    with pytest.raises(RuntimeError, match="last pipeline stage"):
        model.compute_logits(torch.zeros((2, 8)))
