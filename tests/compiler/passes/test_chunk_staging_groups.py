"""The chunk tier stages its key and its value as two operand groups, each refilled at its own kill point."""

from __future__ import annotations

from emmy.commands.trace import graph_from_code
from emmy.compiler.context import Context
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from emmy.compiler.pipeline.search.pins import pinned_knobs


def _chunk_loop(stage: str) -> str:
    """The emitted chunk loop of a causal SDPA whose key and value both stage at ``stage``."""
    q = "torch.randn(1, 8, 512, 64, dtype=torch.float16)"
    graph = graph_from_code(f"F.scaled_dot_product_attention({q}, {q}, {q}, is_causal=True)")[0]
    pins = {
        "TILE@map.1/twist": "mma_m16n8k16_f16_f32/f1x8/k4",
        "TILE@map.1/twist.1/inner": "mma_m16n8k16_f16_f32/f1x8/k4",
        "STAGE@map.1/twist": stage,
        "STAGE@map.1/twist.1/inner": stage,
        "WORK": "w2x1",
    }
    with pinned_knobs(pins):
        lowered = Pipeline.build(CUDA_PASSES).run(graph, ctx=Context.from_target((12, 0)))
    (source,) = [node.op.kernel_source for node in lowered.nodes.values() if isinstance(node.op, CudaOp)]
    return source[source.index("for (int a2__ck") :]


def test_a_tma_staged_chunk_loop_gives_the_key_and_the_value_their_own_barrier() -> None:
    """Two groups, two barriers: the score drains the key's slab and the expectation the value's,
    so each group's fill, wait and release derive from its own live range instead of one group
    spanning the whole body."""
    loop = _chunk_loop("d1/smem-tma")
    assert "mbarrier_wait_parity(&_mbar_k" in loop and "mbarrier_wait_parity(&_mbar_v" in loop


def test_a_deeper_ring_keeps_one_group_and_one_barrier() -> None:
    """A ring of two or more prefetches both operands at the top of the body whatever the grouping,
    so a second group there would only add a release barrier between the score and the softmax on
    every chunk: the key and the value stay one group with one barrier."""
    loop = _chunk_loop("d2/smem-tma")
    assert "_mbar_k" not in loop and "_mbar_v" not in loop
    assert "mbarrier_wait_parity(&_mbar[" in loop


def test_a_single_slot_ring_refills_each_operand_at_its_own_kill_point() -> None:
    """At ring depth one the key is dead once the score is contracted and the value once the
    expectation is, so the key's refill is issued between the score and the value's drain — the
    copy overlaps the softmax and the expectation — and the value's after the expectation: the
    single-slab schedule FlashAttention-2 runs, which one group spanning the body cannot give."""
    loop = _chunk_loop("d1/smem-async")
    score = loop.index("emmy_mma_m16n8k16_f16_f32(_s")  # the score's first mma
    key_refill = loop.index("emmy_cp_async_cg(&_a_smem")  # the key slab's next-chunk fill
    value_drain = loop.index("&_b_smem[")  # the expectation's first read of the value slab
    value_refill = loop.index("emmy_cp_async_cg(&_b_smem")
    assert score < key_refill < value_drain < value_refill
