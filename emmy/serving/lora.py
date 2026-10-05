"""LoRA inputs and Llama attention-split programs for the vLLM generation runner.

The adapter matrices are program inputs. vLLM owns their stable GPU slots and the
per-token selection; the compiler keeps the base weights as shared constants.
"""

from __future__ import annotations


PRE_PROJECTIONS = ("q_proj", "k_proj", "v_proj")
POST_PROJECTIONS = ("o_proj", "gate_proj", "up_proj", "down_proj")
PROJECTIONS = PRE_PROJECTIONS + POST_PROJECTIONS

PRE_INPUTS = ("hidden", "lora_mask", "q_a", "q_b", "k_a", "k_b", "v_a", "v_b")
POST_INPUTS = (
    "attn_out",
    "residual",
    "lora_mask",
    "o_a",
    "o_b",
    "gate_a",
    "gate_b",
    "up_a",
    "up_b",
    "down_a",
    "down_b",
)


def projection_modules(block):
    """The seven dense projections whose adapter weights the Llama split consumes."""
    attn, mlp = block.self_attn, block.mlp
    return {
        "q_proj": attn.q_proj,
        "k_proj": attn.k_proj,
        "v_proj": attn.v_proj,
        "o_proj": attn.o_proj,
        "gate_proj": mlp.gate_proj,
        "up_proj": mlp.up_proj,
        "down_proj": mlp.down_proj,
    }


def weight_examples(block, rank, dtype):
    """Traced A/B inputs in the order the two programs declare them."""
    import torch

    projections = projection_modules(block)
    return {
        name: (
            torch.zeros(rank, module.in_features, dtype=dtype),
            torch.zeros(module.out_features, rank, dtype=dtype),
        )
        for name, module in projections.items()
    }


def build_lora_attention_split_wrapper(block):
    """Llama's split with a masked low-rank update at every projection.

    The mask is one scalar per token. vLLM maps a base request to zero and an
    adapter request to one, so mixed batches keep their original row order for
    paged attention. The B matrices already include the PEFT scale when vLLM
    installs an adapter in its GPU slot.
    """
    import torch.nn as nn
    import torch.nn.functional as F

    attn = block.self_attn
    mlp = block.mlp
    if any(getattr(attn, name, None) is None for name in PRE_PROJECTIONS + ("o_proj",)):
        raise ValueError("LoRA serving requires separate Llama q, k, v and o projections")
    if any(getattr(mlp, name, None) is None for name in ("gate_proj", "up_proj", "down_proj")):
        raise ValueError("LoRA serving requires Llama gate, up and down projections")
    if any(getattr(attn, name, None) is not None for name in ("q_norm", "k_norm", "v_norm", "g_proj")):
        raise ValueError("LoRA serving currently requires the Llama attention layout")
    if any(getattr(block, name, None) is not None for name in ("pre_feedforward_layernorm", "post_feedforward_layernorm")):
        raise ValueError("LoRA serving currently requires the Llama two-norm layout")

    head_dim = attn.head_dim
    num_heads = attn.q_proj.out_features // head_dim
    num_kv_heads = attn.k_proj.out_features // head_dim

    def project(x, module, a, b, mask):
        return module(x) + F.linear(F.linear(x, a), b) * mask

    class Pre(nn.Module):
        emits_gate = False

        def __init__(self):
            super().__init__()
            self.input_layernorm = block.input_layernorm
            self.q_proj, self.k_proj, self.v_proj = attn.q_proj, attn.k_proj, attn.v_proj

        def forward(self, hidden, lora_mask, q_a, q_b, k_a, k_b, v_a, v_b):
            h = self.input_layernorm(hidden)
            t = h.shape[0]
            q = project(h, self.q_proj, q_a, q_b, lora_mask).reshape(t, num_heads * head_dim)
            k = project(h, self.k_proj, k_a, k_b, lora_mask).reshape(t, num_kv_heads * head_dim)
            v = project(h, self.v_proj, v_a, v_b, lora_mask).reshape(t, num_kv_heads * head_dim)
            return q, k, v

    class Post(nn.Module):
        def __init__(self):
            super().__init__()
            self.o_proj = attn.o_proj
            self.post_attention_layernorm = block.post_attention_layernorm
            self.mlp = mlp

        def forward(self, attn_out, residual, lora_mask, o_a, o_b, gate_a, gate_b, up_a, up_b, down_a, down_b):
            h = residual + project(attn_out, self.o_proj, o_a, o_b, lora_mask)
            xn = self.post_attention_layernorm(h)
            gate = self.mlp.act_fn(project(xn, self.mlp.gate_proj, gate_a, gate_b, lora_mask))
            up = project(xn, self.mlp.up_proj, up_a, up_b, lora_mask)
            out = project(gate * up, self.mlp.down_proj, down_a, down_b, lora_mask)
            return h + out

    return Pre(), Post()
