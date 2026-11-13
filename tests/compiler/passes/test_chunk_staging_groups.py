"""The chunk tier stages its key and its value as two operand groups, each refilled at its own kill point."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from emmy.commands.trace import graph_from_code
from emmy.compiler.backend.cuda.backend import CudaBackend
from emmy.compiler.context import Context
from emmy.compiler.ir.kernel.ir import CpAsyncCopy, FragmentLoad, LdmatrixLoad, MbarrierWait, MmaSyncPtx, TmaLoad
from emmy.compiler.ir.stmt import Body, StridedLoop
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import compile_with_kernels, requires_cuda, requires_sm90, statements, text


def _chunk_loop(stage: str) -> str:
    """The emitted chunk loop of a causal SDPA whose key and value both stage at ``stage``."""
    q = "torch.randn(1, 8, 512, 64, dtype=torch.float16)"
    graph = graph_from_code(f"F.scaled_dot_product_attention({q}, {q}, {q}, is_causal=True)")[0]
    pins = {
        "PLACE": "fuse",
        "REDUCE@map.1/twist": "",
        "REDUCE@map.1/twist.1/inner": "",
        "TILE@map.1/twist": "mma_m16n8k16_f16_f32/f1x8/k4",
        "TILE@map.1/twist.1/inner": "mma_m16n8k16_f16_f32/f1x8/k4",
        "STAGE@map.1/twist": stage,
        "STAGE@map.1/twist.1/inner": "",
        "WORK": "w2x1",
    }
    with pinned_knobs(pins):
        _, kernels = compile_with_kernels(graph, ctx=Context.from_target((12, 0)))
    (kernel,) = kernels.values()
    (loop,) = [loop for loop in statements(kernel, StridedLoop) if loop.axis.name == "a2__ck"]
    return list(loop.body.iter())


def test_a_tma_staged_chunk_loop_gives_the_key_and_the_value_their_own_barrier() -> None:
    """Two groups, two barriers: the score drains the key's slab and the expectation the value's,
    so each group's fill, wait and release derive from its own live range instead of one group
    spanning the whole body."""
    loop = _chunk_loop("d1/smem-tma")
    assert {"_mbar_k", "_mbar_v"} <= {wait.mbar for wait in loop if isinstance(wait, MbarrierWait)}


def test_a_deeper_ring_keeps_one_group_and_one_barrier() -> None:
    """A ring of two or more prefetches both operands at the top of the body whatever the grouping,
    so a second group there would only add a release barrier between the score and the softmax on
    every chunk: the key and the value stay one group with one barrier."""
    loop = _chunk_loop("d2/smem-tma")
    assert {wait.mbar for wait in loop if isinstance(wait, MbarrierWait)} == {"_mbar"}
    assert not any(name in text(Body(loop)) for name in ("_mbar_k", "_mbar_v"))


def test_a_single_slot_ring_refills_each_operand_at_its_own_kill_point() -> None:
    """At ring depth one the key is dead once the score is contracted and the value once the
    expectation is, so the key's refill is issued between the score and the value's drain — the
    copy overlaps the softmax and the expectation — and the value's after the expectation: the
    single-slab schedule FlashAttention-2 runs, which one group spanning the body cannot give."""
    loop = _chunk_loop("d1/smem-async")

    def first(found) -> int:
        return next(i for i, stmt in enumerate(loop) if found(stmt))

    score = first(lambda stmt: isinstance(stmt, MmaSyncPtx) and stmt.c_frag.startswith("_s"))  # the score's first mma
    key_refill = first(lambda stmt: isinstance(stmt, CpAsyncCopy) and stmt.smem == "_a_smem")  # the key slab's next-chunk fill
    # The expectation's first read of the value slab.
    value_drain = first(lambda stmt: isinstance(stmt, (LdmatrixLoad, FragmentLoad)) and stmt.src_buffer == "_b_smem")
    value_refill = first(lambda stmt: isinstance(stmt, CpAsyncCopy) and stmt.smem == "_b_smem")
    assert score < key_refill < value_drain < value_refill


@requires_sm90
@requires_cuda
@pytest.mark.parametrize("stage", ["d1/smem-tma", "d2/smem-tma"])
def test_tma_refills_preserve_causal_attention(stage: str) -> None:
    """Shared slabs may be refilled only after their generic-proxy reads have completed."""
    shape = (1, 8, 512, 128)
    operand = "torch.randn(1, 8, 512, 128, dtype=torch.float16)"
    graph = graph_from_code(f"F.scaled_dot_product_attention({operand}, {operand}, {operand}, is_causal=True)")[0]
    pins = {
        "WORK": "w2x1",
        "TILE@map.1/twist": "mma_m16n8k16_f16_f32/f1x1",
        "TILE@map.1/twist.1/inner": "mma_m16n8k16_f16_f32/f1x2",
        "REDUCE@map.1/twist": "",
        "REDUCE@map.1/twist.1/inner": "",
        "STAGE@map.1/twist": stage,
        "STAGE@map.1/twist.1/inner": "",
    }
    with pinned_knobs(pins):
        compiled, kernels = compile_with_kernels(graph, ctx=Context.probe())
    (kernel,) = kernels.values()
    products = {(tuple(mma.shape), mma.ab_dtype, mma.c_dtype) for mma in statements(kernel, MmaSyncPtx)}
    assert statements(kernel, TmaLoad) and products == {((16, 8, 16), "f16", "f32")}
    backend = CudaBackend()
    for seed in range(3):
        rng = np.random.default_rng(seed)
        feed = {name: rng.standard_normal(shape).astype(np.float16) for name in graph.inputs}
        tensors = [torch.as_tensor(feed[name], device="cuda") for name in graph.inputs]
        expected = torch.nn.functional.scaled_dot_product_attention(*tensors, is_causal=True).cpu().numpy()
        actual = backend.run(compiled, input_data=feed)[0].outputs[graph.outputs[0]]
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
