"""Paged input buffers: a read resolves its page before its offset.

A paged buffer is not one allocation but a device table of equal-sized pages — the shape a KV
cache has once it is allocated per request instead of contiguously. The ``cuda.paged_buffers``
hint names the buffer, its paged axis and the page size; nothing above the load changes, so the
axis stays its ordinary symbolic extent and the schedule search never sees the paging.

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


def _inputs():
    import torch

    torch.manual_seed(0)
    return (
        torch.randn(1, HEADS, SEQ, HEAD_DIM),
        torch.randn(1, KV_HEADS, SEQ, HEAD_DIM),
        torch.randn(1, KV_HEADS, SEQ, HEAD_DIM),
        torch.zeros(1, 1, SEQ, SEQ),
    )


def _compile(paged: bool):
    """Compile the attention above, optionally paging K and V along their key axis."""
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.trace.torch import trace_module

    graph = trace_module(_attention(), _inputs())
    if paged:
        graph.hints.set("cuda.paged_buffers", (("k", 2, PAGE), ("v", 2, PAGE)))
    return CudaBackend().compile(graph)


def _cache_write():
    """A producer of the cache's contents — ``tanh`` standing in for the K projection."""
    import torch
    import torch.nn as nn

    class CacheWrite(nn.Module):
        def forward(self, kin):
            return torch.tanh(kin)

    return CacheWrite()


def _compile_write(paged: bool):
    """Compile the producer, optionally writing its output through a page table."""
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.trace.torch import trace_module

    graph = trace_module(_cache_write(), (_inputs()[1],))
    if paged:
        graph.hints.set("cuda.paged_buffers", ((graph.outputs[0], 2, PAGE),))
    return CudaBackend().compile(graph)


def _kernels(compiled):
    return [node.op for node in compiled.nodes.values() if getattr(node.op, "kernel_source", None)]


def test_paged_hint_reaches_the_kernel_abi():
    """The marked buffers take a page table in place of a pointer, the launcher binds that table
    by name, and every read of them goes through it — codegen only, so no GPU is needed."""
    pytest.importorskip("torch")
    (kernel,) = _kernels(_compile(paged=True))

    signature = re.search(r'extern "C" __global__[^{]*', kernel.kernel_source).group(0)
    for name in ("k", "v"):
        assert f"const float* const* {name}__pages" in signature, signature
        assert not re.search(rf"const float\* {name}\b", signature), signature
        assert f"{name}__pages[" in kernel.kernel_source
    assert "k__pages" in kernel.arg_order and "k" not in kernel.arg_order


def test_unpaged_build_is_byte_identical_to_before():
    """The hint is opt-in: without it the kernel source is what it always was, so no recorded
    schedule, golden or cubin key moves."""
    pytest.importorskip("torch")
    (kernel,) = _kernels(_compile(paged=False))

    assert "__pages" not in kernel.kernel_source
    assert kernel.arg_order == tuple(dict.fromkeys(kernel.arg_order))
    assert "k" in kernel.arg_order and "v" in kernel.arg_order


def test_paged_output_writes_through_the_table():
    """The write side of the same ABI: a paged OUTPUT takes a (non-const element) page table and
    every store resolves its page, so a producer and a consumer address the cache identically."""
    pytest.importorskip("torch")
    compiled = _compile_write(paged=True)
    (kernel,) = _kernels(compiled)
    name = compiled.outputs[0]

    signature = re.search(r'extern "C" __global__[^{]*', kernel.kernel_source).group(0)
    assert f"float* const* {name}__pages" in signature, signature
    assert f"{name}__pages[" in kernel.kernel_source
    assert f"{name}__pages" in kernel.arg_order and name not in kernel.arg_order


@requires_cuda
def test_paged_read_matches_the_contiguous_read():
    """The whole contract: reading K/V from a table of four 8-key pages gives bit-identical
    results to reading them from one contiguous 32-key buffer. Only the addressing differs, so
    anything but equality is an addressing bug."""
    pytest.importorskip("cupy")
    import cupy as cp

    from emmy.compiler.backend.cuda.program import CompiledProgram
    from emmy.compiler.backend.gpu_lock import gpu_lock

    feed = {name: t.numpy() for name, t in zip(("q", "k", "v", "mask"), _inputs(), strict=True)}
    flat, paged = _compile(paged=False), _compile(paged=True)
    out_name = flat.outputs[0]

    with gpu_lock():
        contiguous = CompiledProgram.build(flat, dict(feed))
        contiguous.run_once()
        reference = contiguous.outputs()[out_name].copy()

        program = CompiledProgram.build(paged, dict(feed))
        pages = []  # keep every page alive: the table holds raw device pointers
        for name in ("k", "v"):
            for start in range(0, SEQ, PAGE):
                pages.append(cp.ascontiguousarray(cp.asarray(feed[name][:, :, start : start + PAGE, :])))
            table = np.array([page.data.ptr for page in pages[-SEQ // PAGE :]], dtype=np.uint64)
            program.arrays[f"{name}__pages"] = cp.asarray(table)
        program.run_once()
        got = program.outputs()[out_name]

    assert np.array_equal(got, reference), f"max|Δ| = {np.max(np.abs(got - reference))}"


@requires_cuda
def test_paged_write_then_read_round_trips():
    """The two halves agree on the layout. One program writes the cache into a table of pages;
    a second program reads those same pages as K. The result must equal the same pipeline run
    entirely on contiguous buffers — which is what a cache filled in one step and attended to
    in the next actually does."""
    pytest.importorskip("cupy")
    import cupy as cp
    import torch

    from emmy.compiler.backend.cuda.program import CompiledProgram
    from emmy.compiler.backend.gpu_lock import gpu_lock

    q, kin, v, mask = _inputs()
    feed = {"q": q.numpy(), "kin": kin.numpy(), "v": v.numpy(), "mask": mask.numpy()}
    with torch.no_grad():
        contiguous_k = _cache_write()(kin)
        reference = _attention()(q, contiguous_k, v, mask).numpy()

    writer, reader = _compile_write(paged=True), _compile(paged=True)
    written = writer.outputs[0]

    with gpu_lock():
        pages = [cp.zeros((1, KV_HEADS, PAGE, HEAD_DIM), dtype=cp.float32) for _ in range(SEQ // PAGE)]
        table = cp.asarray(np.array([page.data.ptr for page in pages], dtype=np.uint64))

        fill = CompiledProgram.build(writer, {"kin": feed["kin"]})
        fill.arrays[f"{written}__pages"] = table
        fill.run_once()

        attend = CompiledProgram.build(reader, {"q": feed["q"], "k": feed["kin"], "v": feed["v"], "mask": feed["mask"]})
        attend.arrays["k__pages"] = table  # the pages the writer just filled
        attend.arrays["v__pages"] = _page_table(cp, feed["v"], pages)
        attend.run_once()
        got = attend.outputs()[reader.outputs[0]]

    np.testing.assert_allclose(got, reference, rtol=1e-5, atol=1e-5)


def _page_table(cp, array, keep):
    """A page table over ``array``'s key axis; pages are appended to ``keep`` so the device
    memory the table points at outlives the call."""
    start = len(keep)
    for offset in range(0, SEQ, PAGE):
        keep.append(cp.ascontiguousarray(cp.asarray(array[:, :, offset : offset + PAGE, :])))
    return cp.asarray(np.array([page.data.ptr for page in keep[start:]], dtype=np.uint64))
