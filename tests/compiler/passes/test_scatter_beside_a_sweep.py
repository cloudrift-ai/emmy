"""A scatter fused into one kernel beside a store over another sweep keeps its store through the kernel lowering
(CPU-only: the program is lowered to Kernel IR for an RTX 5090 and nothing runs)."""

import pytest
import torch

from emmy.compiler.context import Context
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.pipeline import CUDA_PASSES, LoweringError, Pipeline
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


def _written(module, rows: int, width: int = 8) -> set[str]:
    """The buffers the kernels of ``module`` store, lowered at ``rows`` rows of ``width``."""
    graph = trace_module(module, (torch.zeros(rows, width),))
    lowered = Pipeline.build(CUDA_PASSES[:-1]).run(graph, ctx=Context.from_target((12, 0)), db=None)
    kernels = [node.op for node in lowered.nodes.values() if isinstance(node.op, KernelOp)]
    return {name for kernel in kernels for stmt in kernel.body.iter() for name in stmt.external_writes()}


def test_one_rows_scatter_keeps_its_store():
    assert {"scatter", "softmax"} <= _written(_RankedSlots(), 1)


@pytest.mark.xfail(
    raises=LoweringError,
    strict=True,
    reason="at two rows the kernel lowering drops the fused kernel's scatter store, so every row is declined (#1248)",
)
def test_two_rows_scatter_keeps_its_store():
    """Kept fused: the cut that hands each output its own kernel avoids the dropped store, and is offered beside this."""
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    with pinned_knobs({"PLACE": "fuse"}):
        assert {"scatter", "softmax"} <= _written(_RankedSlots(), 2)


def test_the_native_router_keeps_both_scatter_stores_at_a_chunks_rows():
    """Native generation's router scatters its picks and their probabilities from one rank sweep, which keeps the
    dropped store above away from it (#1248); at a chunk's rows both stores still reach the kernels."""
    from emmy.serving.native.prepare import router_module

    class Gate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts, self.top_k = 8, 2
            self.weight = torch.nn.Parameter(torch.randn(8, 16))

    assert {"scatter", "scatter_1"} <= _written(router_module(Gate(), 3), 3, width=16)


def test_the_native_rotary_cache_store_spreads_over_the_rows_and_heads_it_writes():
    """Native generation's rotary program stores a chunk's keys and values into the cache at its rows' positions,
    beside the rotated queries over another sweep, so the fused kernel binds no grid. The store's row is computed once
    per row, above the cell it stores, and rides with the keys' and values' region: each output then owns a kernel of
    its own, and every kernel spreads over more blocks than one, where the store used to sweep the whole chunk in
    one."""
    from emmy.compiler.ir.cuda.ir import CudaOp
    from emmy.serving.native.prepare import rope_module

    rows, heads, kv, head_dim, rotary, context = 16, 8, 4, 128, 64, 256
    cache = torch.zeros(context, kv, head_dim, dtype=torch.float8_e4m3fn)
    projections = (torch.zeros(rows, width * head_dim, dtype=torch.bfloat16) for width in (heads, kv, kv))
    examples = (*projections, torch.zeros(1, dtype=torch.int64), torch.zeros(context, rotary), torch.zeros(context, rotary), cache, cache.clone())
    graph = trace_module(rope_module(heads, kv, head_dim, rotary, rows, cache.dtype), examples)
    lowered = Pipeline.build(CUDA_PASSES).run(graph, ctx=Context.from_target((12, 0)), db=None)
    launches = {tuple(t.name for t in node.outputs): node.op.grid[0][0] for node in lowered.nodes.values() if isinstance(node.op, CudaOp)}

    assert set(launches) == {("view_1",), ("scatter", "scatter_1")}
    assert all(blocks > 1 for blocks in launches.values()), launches
