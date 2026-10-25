"""The multi-token-prediction head of an Emmy-served target, as the draft model of vLLM's MTP proposer.

vLLM's proposer keeps what it owns for any MTP head: the draft's KV cache, its positions and attention metadata,
the draft steps and the sampling. This class replaces only the stock head module, so that the head runs Emmy's
programs: the prologue, then the head's attention layer split around vLLM's paged attention as the target's layers
are, then the final norm (``EmmyGenRunner.create_mtp``). The proposer gives it the target's embedding and output
head, so it holds no weight of its own.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.logits_processor import LogitsProcessor

from emmy import config as emmy_config
from emmy.serving.coded_head import CodedHead
from emmy.serving.gen_runner import EmmyGenRunner
from emmy.serving.vllm_model import _trunk_dtype_str, pinned_model_id
from emmy.serving.vllm_model_gen import _build_rotaries, _rope_cache_limit, prefill_widths

#: The stock head classes this one stands in for when the target is Emmy's.
STOCK_HEADS = ("Qwen3_5MTP", "Qwen3_5MoeMTP")


class _Inner(nn.Module):
    """The ``.model`` the proposer reaches into: it sets ``embed_tokens`` to the target's embedding, after vLLM 0.25
    compares that embedding's width with this one's. The placeholder has the width and no row."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.embed_tokens = nn.Module()
        self.embed_tokens.register_buffer("weight", torch.empty(0, hidden_size), persistent=False)


class EmmyMTPModel(nn.Module):
    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()
        mc = vllm_config.model_config
        config = mc.hf_text_config
        self.config = config
        self.dtype = mc.dtype
        capacity, prefill_bucket = prefill_widths(vllm_config)
        self.runner, self.prologue = EmmyGenRunner.create_mtp(
            pinned_model_id(mc),
            dtype_str=_trunk_dtype_str(mc.dtype),
            decode_bucket=emmy_config.gen_decode_bucket(),
            max_tokens=capacity,
            prefill_bucket=prefill_bucket,
        )
        hd, nh, nkv, scaling = self.runner.layer_meta(0)
        self.attn = Attention(
            nh,
            hd,
            scaling,
            num_kv_heads=nkv,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix="mtp.layers.0.self_attn.attn",
        )
        (self.rotary_emb,) = _build_rotaries(config, self.runner, 1, _rope_cache_limit(mc, config), mc.dtype)
        self.model = _Inner(config.hidden_size)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        # The proposer shares the target's head, and a coded one (``CodedHead``) computes f32 logits itself.
        self.coded_logits_processor = LogitsProcessor(config.vocab_size, logits_as_input=True)

    def embed_input_ids(self, input_ids, *args, **kwargs):
        return self.model.embed_tokens(input_ids)

    def forward(self, input_ids, positions, hidden_states, intermediate_tensors=None, inputs_embeds=None, spec_step_idx=0):
        embeds = self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
        x = self.prologue(embeds.to(self.dtype), hidden_states.to(self.dtype))
        q, k, v, *gate = self.runner.forward_layer_pre_device(0, x)
        q, k = self.rotary_emb(positions, q, k)
        attn_out = self.attn(q.to(self.dtype), k.to(self.dtype), v)
        return self.runner.final_norm_device(self.runner.forward_layer_post_device(0, attn_out, x, *gate))

    def compute_logits(self, hidden_states):
        if isinstance(self.lm_head, CodedHead):
            return self.coded_logits_processor(self.lm_head, self.lm_head(hidden_states))
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights):
        # The runner read the head from the checkpoint; the proposer shares the target's embedding and head.
        for _ in weights:
            pass
        return set()


def use_for_draft(vllm_config) -> None:
    """Serve the MTP head with :class:`EmmyMTPModel` when the draft is one of the stock heads it stands in for.

    Called from the target's constructor: the proposer builds the draft after the target, and vLLM caches a model
    class per architecture name, so the draft config is renamed to Emmy's registered head rather than the stock
    name re-registered."""
    spec = vllm_config.speculative_config
    if spec is None or spec.method != "mtp":
        return
    hf_config = spec.draft_model_config.hf_config
    if list(hf_config.architectures or ()) in ([head] for head in STOCK_HEADS):
        hf_config.architectures = ["EmmyMTPModel"]
