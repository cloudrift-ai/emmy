"""Qwen's inter-chunk GDN update, with chunk-local transforms supplied as inputs."""

import os

import torch
import torch.nn as nn


class ChunkState(nn.Module):
    def forward(self, w, k, u, decay):
        state = torch.zeros_like(k[:, 0, 0, :, None] * u[:, 0, 0, None, :])
        corrected = []
        for chunk in range(w.shape[1]):
            value = u[:, chunk] - w[:, chunk] @ state
            state = state * decay[:, chunk, None, None] + k[:, chunk].transpose(-1, -2) @ value
            corrected.append(value)
        return torch.stack(corrected, 1), state


heads = int(os.environ.get("GDN_HEADS", "12"))
chunks = int(os.environ.get("GDN_TOKENS", "128")) // 64
torch.manual_seed(0)
model = ChunkState()
model(
    torch.randn(heads, chunks, 64, 128) * 0.02,
    torch.randn(heads, chunks, 64, 128) * 0.02,
    torch.randn(heads, chunks, 64, 128) * 0.1,
    torch.rand(heads, chunks) * 0.2 + 0.8,
)
