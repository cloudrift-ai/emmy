"""One-adapter Llama generation with Emmy programs and vLLM's LoRA scheduler."""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.models.interfaces import SupportsLoRA

from emmy import config as emmy_config
from emmy.compiler.trace.dynamic import DYNAMIC_DIM_MAX
from emmy.serving.gen_runner import EmmyGenRunner
from emmy.serving.lora import POST_PROJECTIONS, PRE_PROJECTIONS


class _EmmyLoRASlot(LinearBase, BaseLayerWithLoRA):
    """Stable GPU adapter buffers that vLLM's LoRA manager fills by module name.

    The base projection is compiled by Emmy, so this module has no base weight
    and is never called. Its shape and slot methods are the LoRA manager's
    ordinary loading contract; the compiled program reads the same buffers.
    """

    def __init__(self, input_size, output_size, rank, max_loras, dtype):
        nn.Module.__init__(self)
        self.input_size = input_size
        self.output_size = output_size
        self.register_buffer("a", torch.zeros(max_loras, 1, rank, input_size, dtype=dtype, device="cuda"))
        self.register_buffer("b", torch.zeros(max_loras, 1, output_size, rank, dtype=dtype, device="cuda"))
        self.lora_a_stacked = (self.a,)
        self.lora_b_stacked = (self.b,)

    def forward(self, input_):
        raise RuntimeError("Emmy's compiled projection consumes this LoRA slot")

    def slice_lora_a(self, lora_a):
        return lora_a

    def slice_lora_b(self, lora_b):
        return lora_b

    def reset_lora(self, index):
        self.a[index].zero_()
        self.b[index].zero_()

    def set_lora(self, index, lora_a, lora_b):
        if lora_a.shape[1] != self.input_size or lora_b.shape[0] != self.output_size or lora_a.shape[0] != lora_b.shape[1]:
            raise ValueError("LoRA matrices do not match the compiled projection")
        rank = lora_a.shape[0]
        if rank > self.a.shape[2]:
            raise ValueError(f"LoRA rank {rank} exceeds compiled rank {self.a.shape[2]}")
        self.reset_lora(index)
        self.a[index, 0, :rank].copy_(lora_a)
        self.b[index, 0, :, :rank].copy_(lora_b)


def _make_slots(config, rank, max_loras, dtype):
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    head_dim = getattr(config, "head_dim", None) or hidden // config.num_attention_heads
    q_width = config.num_attention_heads * head_dim
    kv_width = config.num_key_value_heads * head_dim
    dims = {
        "q_proj": (hidden, q_width),
        "k_proj": (hidden, kv_width),
        "v_proj": (hidden, kv_width),
        "o_proj": (q_width, hidden),
        "gate_proj": (hidden, intermediate),
        "up_proj": (hidden, intermediate),
        "down_proj": (intermediate, hidden),
    }
    model = nn.Module()
    layers = nn.ModuleList()
    for _ in range(config.num_hidden_layers):
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.mlp = nn.Module()
        for name in PRE_PROJECTIONS + ("o_proj",):
            setattr(layer.self_attn, name, _EmmyLoRASlot(*dims[name], rank, max_loras, dtype))
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(layer.mlp, name, _EmmyLoRASlot(*dims[name], rank, max_loras, dtype))
        layers.append(layer)
    model.layers = layers
    return model


class EmmyGenLoRAModel(nn.Module, SupportsLoRA):
    """Llama serving with one adapter slot and per-token base/adapter selection."""

    packed_modules_mapping = {}
    embedding_modules = {}

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        mc = vllm_config.model_config
        lora = vllm_config.lora_config
        if getattr(config, "model_type", None) != "llama" or mc.dtype != torch.float16:
            raise ValueError("EmmyGenLoRAModel currently requires an FP16 Llama checkpoint")
        if lora is None or lora.max_loras != 1:
            raise ValueError("EmmyGenLoRAModel requires --enable-lora and --max-loras 1")
        if vllm_config.parallel_config.tensor_parallel_size != 1 or vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("EmmyGenLoRAModel currently requires one GPU")
        if vllm_config.speculative_config is not None:
            raise ValueError("EmmyGenLoRAModel does not support speculative decoding")
        max_batched = vllm_config.scheduler_config.max_num_batched_tokens
        if max_batched > DYNAMIC_DIM_MAX:
            raise ValueError(f"LoRA serving needs --max-num-batched-tokens <= {DYNAMIC_DIM_MAX}")
        self.config = config
        self.dtype = mc.dtype
        self._lora_rank = lora.max_lora_rank
        model_id = str(mc.model)
        if not model_id.startswith("/") and mc.revision:
            model_id = f"{model_id}@{mc.revision}"
        self.runner = EmmyGenRunner.create(
            model_id=model_id,
            dtype_str="float16",
            decode_bucket=emmy_config.gen_decode_bucket(),
            max_tokens=DYNAMIC_DIM_MAX,
            prefill_bucket=0,
            lora_rank=self._lora_rank,
        )
        self.model = _make_slots(config, self._lora_rank, 1, self.dtype)
        self.attn = nn.ModuleList()
        self.rotary_emb = nn.ModuleList()
        for layer in range(config.num_hidden_layers):
            head_dim, num_heads, num_kv_heads, scaling = self.runner.layer_meta(layer)
            self.attn.append(
                Attention(
                    num_heads,
                    head_dim,
                    scaling,
                    num_kv_heads=num_kv_heads,
                    cache_config=vllm_config.cache_config,
                    quant_config=vllm_config.quant_config,
                    prefix=f"{prefix}.model.layers.{layer}.self_attn.attn".lstrip("."),
                )
            )
            self.rotary_emb.append(
                get_rope(
                    head_dim,
                    max_position=config.max_position_embeddings,
                    rope_parameters=getattr(config, "rope_parameters", None),
                    is_neox_style=True,
                )
            )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.lm_head".lstrip("."),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids):
        return self.runner.embed_device(input_ids.clamp(0, self.config.vocab_size - 1))

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        del intermediate_tensors, kwargs
        hidden = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        return self._forward_device(hidden, positions)

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights):
        param = self.lm_head.weight
        loader = getattr(param, "weight_loader", default_weight_loader)
        loaded = set()
        for name, weight in weights:
            if name == "lm_head.weight":
                loader(param, weight)
                loaded.add(name)
        if not loaded:
            raise RuntimeError("checkpoint has no lm_head.weight")
        return loaded

    def _lora_mask(self, tokens):
        # vLLM writes this stable buffer before both ordinary forwards and CUDA graph
        # replays. Slot 0 is the sole adapter; -1 is the base model.
        wrapper = self.model.layers[0].self_attn.q_proj.punica_wrapper
        indices = wrapper._token_lora_indices[:tokens]
        return (indices == 0).to(self.dtype).unsqueeze(-1)

    def _lora_inputs(self, layer, mask, names):
        modules = self.model.layers[layer]
        weights = []
        for name in names:
            parent = modules.self_attn if name in PRE_PROJECTIONS + ("o_proj",) else modules.mlp
            slot = getattr(parent, name)
            weights.extend((slot.a[0, 0], slot.b[0, 0]))
        return (mask, *weights)

    def _forward_device(self, hidden, positions, token_ids=None):
        mask = self._lora_mask(hidden.shape[0])
        for layer in range(self.runner.num_layers):
            residual = hidden
            q, k, v = self.runner.forward_layer_pre_device(layer, hidden, lora=self._lora_inputs(layer, mask, PRE_PROJECTIONS))
            q, k = self.rotary_emb[layer](positions, q, k)
            q, k = q.to(self.dtype), k.to(self.dtype)
            attn_out = self.attn[layer](q, k, v)
            hidden = self.runner.forward_layer_post_device(
                layer, attn_out, residual, lora=self._lora_inputs(layer, mask, POST_PROJECTIONS)
            )
        return self.runner.final_norm_device(hidden)
