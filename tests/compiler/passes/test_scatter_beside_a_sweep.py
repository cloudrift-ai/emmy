"""A scatter fused into one kernel beside a store over another sweep keeps its store through the kernel lowering
(CPU-only: the program is lowered to Kernel IR for an RTX 5090 and nothing runs)."""

import pytest
import torch

from emmy.compiler.context import Context
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from emmy.compiler.trace.torch import trace_module


class _RankedSlots(torch.nn.Module):
    """A row's softmax, each element's rank among the row, and every element ranked below 2 scattered to its slot:
    the scatter and the softmax are two stores of one fused kernel over different sweeps."""

    def forward(self, x):
        rows, width, k = x.shape[0], x.shape[1], 2
        ids = torch.arange(width)
        p = torch.softmax(x, dim=-1)
        other, own = p.unsqueeze(1), p.unsqueeze(2)
        rank = ((other > own) | ((other == own) & (ids.view(1, 1, -1) < ids.view(1, -1, 1)))).long().sum(dim=2)
        slots = torch.full((rows, k + 1), width - 1).scatter(1, torch.where(rank < k, rank, k), ids.expand(rows, width))
        return slots, p


def _written(rows: int) -> set[str]:
    graph = trace_module(_RankedSlots(), (torch.zeros(rows, 8),))
    lowered = Pipeline.build(CUDA_PASSES[:-1]).run(graph, ctx=Context.from_target((12, 0)), db=None)
    kernels = [node.op for node in lowered.nodes.values() if isinstance(node.op, KernelOp)]
    return {name for kernel in kernels for stmt in kernel.body.iter() for name in stmt.external_writes()}


def test_one_rows_scatter_keeps_its_store():
    assert {"scatter", "softmax"} <= _written(1)


@pytest.mark.xfail(
    strict=True,
    reason="at two rows the kernel lowering drops the scatter's store and nothing raises (#1248)",
)
def test_two_rows_scatter_keeps_its_store():
    assert {"scatter", "softmax"} <= _written(2)
