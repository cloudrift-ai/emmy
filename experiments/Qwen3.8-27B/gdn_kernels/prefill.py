"""The Transformers GDN prefill used by Qwen 3.8, with TP-local head dimensions."""

import os

import torch
import torch.nn as nn
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule


class Prefill(nn.Module):
    def forward(self, q, k, v, g, beta):
        q = q.repeat_interleave(3, dim=2)
        k = k.repeat_interleave(3, dim=2)
        return torch_chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=64, use_qk_l2norm_in_kernel=True)[0]


heads = int(os.environ.get("GDN_HEADS", "12"))
tokens = int(os.environ.get("GDN_TOKENS", "128"))
torch.manual_seed(0)
model = Prefill()
model(
    torch.randn(1, tokens, heads // 3, 128, dtype=torch.float16),
    torch.randn(1, tokens, heads // 3, 128, dtype=torch.float16),
    torch.randn(1, tokens, heads, 128, dtype=torch.float16),
    -torch.rand(1, tokens, heads),
    torch.rand(1, tokens, heads),
)
