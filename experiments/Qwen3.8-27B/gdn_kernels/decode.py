"""One Qwen GDN decode step, including the FP32 state read and write."""

import os

import torch
import torch.nn as nn
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule


class Decode(nn.Module):
    def forward(self, q, k, v, g, beta, state):
        q = q.repeat_interleave(3, dim=2)
        k = k.repeat_interleave(3, dim=2)
        return torch_recurrent_gated_delta_rule(q, k, v, g, beta, state, True, use_qk_l2norm_in_kernel=True)


heads = int(os.environ.get("GDN_HEADS", "12"))
torch.manual_seed(0)
model = Decode()
model(
    torch.randn(1, 1, heads // 3, 128, dtype=torch.float16),
    torch.randn(1, 1, heads // 3, 128, dtype=torch.float16),
    torch.randn(1, 1, heads, 128, dtype=torch.float16),
    -torch.rand(1, 1, heads),
    torch.rand(1, 1, heads),
    torch.randn(1, heads, 128, 128) * 0.01,
)
