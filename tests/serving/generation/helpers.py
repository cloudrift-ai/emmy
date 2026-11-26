"""Doubles the generation tests share."""

from __future__ import annotations


def coded_head_double(weight):
    """A :class:`~emmy.serving.coded_head.CodedHead` without a compiled program: f32 logits ``rows @ weight.T``,
    what the program computes, for tests of the code around it on a host with no GPU."""
    import torch

    from emmy.serving.coded_head import CodedHead

    class _Head(CodedHead):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.register_buffer("weight_f32", weight.float())

        def forward(self, hidden_states):
            return hidden_states.float() @ self.weight_f32.T

    return _Head()
