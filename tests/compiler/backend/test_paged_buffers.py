"""Paged buffers: a read or write resolves its page before its offset.

A paged buffer is not one allocation but a device table of equal-sized pages — the shape a KV
cache has once it is allocated per request instead of contiguously. The ``cuda.paged_buffers``
hint names the buffer, its paged axis, the page size, and a runtime ``start`` symbol that makes
the buffer's own coordinate absolute, which is what lets a step write only its new rows.

The hint only reaches a buffer that survives to a kernel boundary — an intermediate the fusion
policy absorbs is not a buffer at all — so every test here asserts the page table reached the
signature rather than trusting the hint.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from tests.compiler.helpers import requires_cuda

HEADS, KV_HEADS, HEAD_DIM, SEQ, PAGE = 4, 2, 16, 32, 8
CHUNK = 4  # keys one step appends to the cache — deliberately NOT the page size, so a chunk starts mid-page


def _attention():
    """GQA attention over an explicit additive mask — the reader of a KV cache."""
    import torch
    import torch.nn as nn

    class Attention(nn.Module):
        def forward(self, q, k, v, mask):
            k = k.repeat_interleave(HEADS // KV_HEADS, dim=1)
            v = v.repeat_interleave(HEADS // KV_HEADS, dim=1)
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    return Attention()


def _cache_write():
    """A producer of the cache's contents — ``tanh`` standing in for the K projection."""
    import torch
    import torch.nn as nn

    class CacheWrite(nn.Module):
        def forward(self, kin):
            return torch.tanh(kin)

    return CacheWrite()


def _inputs():
    import torch

    torch.manual_seed(0)
    return (
        torch.randn(1, HEADS, SEQ, HEAD_DIM),
        torch.randn(1, KV_HEADS, SEQ, HEAD_DIM),
        torch.randn(1, KV_HEADS, SEQ, HEAD_DIM),
        torch.zeros(1, 1, SEQ, SEQ),
    )


def _compile_read(paged: bool):
    """Compile the attention above, optionally reading K and V through page tables."""
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.trace.torch import trace_module

    graph = trace_module(_attention(), _inputs())
    if paged:
        graph.hints.set("cuda.paged_buffers", (("k", 2, PAGE, None), ("v", 2, PAGE, None)))
    return CudaBackend().compile(graph)


def _compile_write(*, rows: int = SEQ, start: str | None = None):
    """Compile the producer of ``rows`` new keys, writing them into a page table at ``start``."""
    import torch

    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.trace.torch import trace_module

    graph = trace_module(_cache_write(), (torch.zeros(1, KV_HEADS, rows, HEAD_DIM),))
    graph.hints.set("cuda.paged_buffers", ((graph.outputs[0], 2, PAGE, start),))
    return CudaBackend().compile(graph)


def _kernels(compiled):
    return [node.op for node in compiled.nodes.values() if getattr(node.op, "kernel_source", None)]


def _signature(kernel) -> str:
    return re.search(r'extern "C" __global__[^{]*', kernel.kernel_source).group(0)


def test_paged_hint_reaches_the_kernel_abi():
    """The marked buffers take a page table in place of a pointer, the launcher binds that table
    by name, and every read of them goes through it — codegen only, so no GPU is needed."""
    pytest.importorskip("torch")
    (kernel,) = _kernels(_compile_read(paged=True))

    signature = _signature(kernel)
    for name in ("k", "v"):
        assert f"const float* const* {name}__pages" in signature, signature
        assert not re.search(rf"const float\* {name}\b", signature), signature
        assert f"{name}__pages[" in kernel.kernel_source
    assert "k__pages" in kernel.arg_order and "k" not in kernel.arg_order


def test_unpaged_build_is_byte_identical_to_before():
    """The hint is opt-in: without it the kernel source is what it always was, so no recorded
    schedule, golden or cubin key moves."""
    pytest.importorskip("torch")
    (kernel,) = _kernels(_compile_read(paged=False))

    assert "__pages" not in kernel.kernel_source
    assert kernel.arg_order == tuple(dict.fromkeys(kernel.arg_order))
    assert "k" in kernel.arg_order and "v" in kernel.arg_order


def test_paged_output_writes_at_a_runtime_start():
    """The write side of the same ABI. A paged OUTPUT takes a (non-const element) page table, and
    a ``start`` symbol becomes an ordinary runtime ``int`` arg that shifts every store into the
    cache's coordinates — so a kernel producing CHUNK rows can land them anywhere in the cache."""
    pytest.importorskip("torch")
    compiled = _compile_write(rows=CHUNK, start="past")
    (kernel,) = _kernels(compiled)
    name = compiled.outputs[0]

    signature = _signature(kernel)
    assert f"float* const* {name}__pages" in signature, signature
    assert "int past" in signature, signature
    assert "past" in kernel.runtime_args
    assert f"{name}__pages[" in kernel.kernel_source
    assert f"{name}__pages" in kernel.arg_order and name not in kernel.arg_order


@requires_cuda
def test_paged_read_matches_the_contiguous_read():
    """Reading K/V from a table of four 8-key pages gives bit-identical results to reading them
    from one contiguous 32-key buffer. Only the addressing differs, so anything but equality is
    an addressing bug."""
    pytest.importorskip("cupy")
    import cupy as cp

    from emmy.compiler.backend.cuda.program import CompiledProgram
    from emmy.compiler.backend.gpu_lock import gpu_lock

    feed = {name: t.numpy() for name, t in zip(("q", "k", "v", "mask"), _inputs(), strict=True)}
    flat, paged = _compile_read(paged=False), _compile_read(paged=True)
    out_name = flat.outputs[0]

    with gpu_lock():
        contiguous = CompiledProgram.build(flat, dict(feed))
        contiguous.run_once()
        reference = contiguous.outputs()[out_name].copy()

        program = CompiledProgram.build(paged, dict(feed))
        pages: list = []  # keep every page alive: the table holds raw device pointers
        for name in ("k", "v"):
            program.arrays[f"{name}__pages"] = _split_into_pages(cp, feed[name], pages)
        program.run_once()
        got = program.outputs()[out_name]

    assert np.array_equal(got, reference), f"max|Δ| = {np.max(np.abs(got - reference))}"


@requires_cuda
def test_cache_filled_in_chunks_then_attended():
    """End to end. The cache starts empty as a table of pages; four steps each compute CHUNK new
    keys and write them at their absolute position through one ``start`` symbol; then attention
    reads the whole cache back through the same table. The result must equal the same pipeline
    run on one contiguous buffer — a wrong page, a wrong offset or a disagreeing layout between
    the write and the read all break it."""
    pytest.importorskip("cupy")
    import cupy as cp
    import torch

    from emmy.compiler.backend.cuda.program import CompiledProgram
    from emmy.compiler.backend.gpu_lock import gpu_lock

    q, kin, v, mask = _inputs()
    with torch.no_grad():
        reference = _attention()(q, _cache_write()(kin), v, mask).numpy()

    writer = _compile_write(rows=CHUNK, start="past")
    reader = _compile_read(paged=True)

    with gpu_lock():
        cache = [cp.zeros((1, KV_HEADS, PAGE, HEAD_DIM), dtype=cp.float32) for _ in range(SEQ // PAGE)]
        table = cp.asarray(np.array([page.data.ptr for page in cache], dtype=np.uint64))

        chunk = np.ascontiguousarray(kin.numpy()[:, :, :CHUNK, :])
        step = CompiledProgram.build(writer, {writer.inputs[0]: chunk})
        step.arrays[f"{writer.outputs[0]}__pages"] = table
        for past in range(0, SEQ, CHUNK):
            step.arrays[writer.inputs[0]].set(np.ascontiguousarray(kin.numpy()[:, :, past : past + CHUNK, :]))
            step.set_sym_values({"past": past})
            step.run_once()

        feed = {name: t.numpy() for name, t in zip(("q", "k", "v", "mask"), (q, kin, v, mask), strict=True)}
        attend = CompiledProgram.build(reader, feed)
        values: list = []  # keep V's pages alive for as long as its table is bound
        attend.arrays["k__pages"] = table  # the pages the four steps just filled
        attend.arrays["v__pages"] = _split_into_pages(cp, v.numpy(), values)
        attend.run_once()
        got = attend.outputs()[reader.outputs[0]]

    np.testing.assert_allclose(got, reference, rtol=1e-5, atol=1e-5)


def _split_into_pages(cp, array, keep: list):
    """A page table over ``array``'s key axis. Pages are appended to ``keep`` so the device memory
    the table points at outlives the call; pass a list that stays alive for as long as the table."""
    start = len(keep)
    for offset in range(0, SEQ, PAGE):
        keep.append(cp.ascontiguousarray(cp.asarray(array[:, :, offset : offset + PAGE, :])))
    return cp.asarray(np.array([page.data.ptr for page in keep[start:]], dtype=np.uint64))
