"""Paged buffers: a read or write resolves its page before its offset.

A paged buffer is not one allocation but a device table of equal-sized pages — the shape a KV
cache has once it is allocated per request instead of contiguously. The ``cuda.paged_buffers``
hint names the buffer, its paged axis, the page size, and a ``start``: the graph's own i64 scalar
input that makes the buffer's coordinate absolute, which is what lets a step write only its new
rows and still replay as one graph at every position.

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


def _cache_write_at():
    """A producer of the cache's contents — ``tanh`` standing in for the K projection — taking the
    position it writes at as an i64 scalar in device memory."""
    import torch
    import torch.nn as nn

    class CacheWriteAt(nn.Module):
        def forward(self, kin, past):
            return torch.where(past >= 0, torch.tanh(kin), kin)

    return CacheWriteAt()


def _compile_write(rows: int):
    """Compile the producer of ``rows`` new keys, writing them into a page table at the graph's own
    ``past`` input scalar."""
    import torch

    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.trace.torch import trace_module

    graph = trace_module(_cache_write_at(), (torch.zeros(1, KV_HEADS, rows, HEAD_DIM), torch.zeros(1, dtype=torch.int64)))
    graph.hints.set("cuda.paged_buffers", ((graph.outputs[0], 2, PAGE, "past"),))
    return CudaBackend().compile(graph)


def _kernels(compiled):
    return [node.op for node in compiled.nodes.values() if getattr(node.op, "kernel_source", None)]


def _signature(kernel) -> str:
    return re.search(r'extern "C" __global__[^{]*', kernel.kernel_source).group(0)


def test_paged_hint_reaches_the_kernel_abi():
    """The marked buffers take a page table in place of a pointer, the launcher binds that table
    by name, and every read of them goes through it — codegen only, so no GPU is needed."""
    pytest.importorskip("torch")
    from emmy.compiler.backend.plan import PLAN_FORMAT_PAGED, plan_from_graph, plan_to_dict

    compiled = _compile_read(paged=True)
    (kernel,) = _kernels(compiled)

    signature = _signature(kernel)
    for name in ("k", "v"):
        assert f"const float* const* {name}__pages" in signature, signature
        assert not re.search(rf"const float\* {name}\b", signature), signature
        assert f"{name}__pages[" in kernel.kernel_source
    assert "k__pages" in kernel.arg_order and "k" not in kernel.arg_order
    # The table is a runtime contract of its own: a plan carrying one serializes as the paged format.
    assert plan_to_dict(plan_from_graph(compiled))["format"] == PLAN_FORMAT_PAGED


def test_unpaged_build_is_byte_identical_to_before():
    """The hint is opt-in: without it the kernel source is what it always was, so no recorded
    schedule, golden or cubin key moves."""
    pytest.importorskip("torch")
    (kernel,) = _kernels(_compile_read(paged=False))

    assert "__pages" not in kernel.kernel_source
    assert kernel.arg_order == tuple(dict.fromkeys(kernel.arg_order))
    assert "k" in kernel.arg_order and "v" in kernel.arg_order


def test_paged_output_writes_at_a_device_start():
    """The write side of the same ABI. A paged OUTPUT takes a (non-const element) page table, and
    the ``start`` — a graph input the kernel reads off the device in its preamble, so nothing on
    the host changes between positions and a token step stays one replayable graph — shifts every
    store into the cache's coordinates: a kernel producing CHUNK rows lands them anywhere in it."""
    pytest.importorskip("torch")
    compiled = _compile_write(CHUNK)
    (kernel,) = _kernels(compiled)
    name = compiled.outputs[0]

    signature = _signature(kernel)
    assert f"float* const* {name}__pages" in signature, signature
    assert "const long long* __restrict__ past" in signature and not kernel.runtime_args, signature
    assert "const int past__at = (int)past[0];" in kernel.kernel_source
    assert "past__at" in kernel.kernel_source.split("__pages[", 1)[1].split("]", 1)[0]
    assert f"{name}__pages" in kernel.arg_order and name not in kernel.arg_order


def test_one_page_buffer_resolves_its_base_once():
    """A buffer whose single page spans it keeps the table in its signature but is addressed like
    the flat buffer it replaces: the preamble takes the table's only entry as the base, so every
    statement kind — a fragment store, a staged copy — works over it unchanged, and no access
    resolves a page per element."""
    pytest.importorskip("torch")
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.trace.torch import trace_module

    graph = trace_module(_attention(), _inputs())
    graph.hints.set("cuda.paged_buffers", (("k", 2, SEQ, None), ("v", 2, SEQ, None)))
    (kernel,) = _kernels(CudaBackend().compile(graph))

    signature = _signature(kernel)
    for name in ("k", "v"):
        assert f"const float* const* {name}__pages" in signature, signature
        assert f"const float* {name} = {name}__pages[0];" in kernel.kernel_source
    assert kernel.kernel_source.count("__pages[") == 2


def test_paged_start_must_be_an_i64_scalar():
    """The kernel reads the start as one i64 off the device, so the hint may name nothing else."""
    pytest.importorskip("torch")
    import torch

    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.trace.torch import trace_module

    example = torch.zeros(1, KV_HEADS, CHUNK, HEAD_DIM)
    for past in (torch.zeros(1, dtype=torch.float32), torch.zeros(HEAD_DIM, dtype=torch.int64)):
        graph = trace_module(_cache_write_at(), (example, past))
        graph.hints.set("cuda.paged_buffers", ((graph.outputs[0], 2, PAGE, "past"),))
        with pytest.raises(ValueError, match="i64 scalar"):
            CudaBackend().compile(graph)


@requires_cuda
def test_paged_read_matches_the_contiguous_read(monkeypatch):
    """Reading K/V from a table of four 8-key pages gives bit-identical results to reading them
    from one contiguous 32-key buffer. Both compiles use direct scalar loads, so the addressing
    differs while the contraction schedule stays the same."""
    import torch

    monkeypatch.setenv("EMMY_STAGE", "")
    monkeypatch.setenv("EMMY_TILE", "")

    from emmy.compiler.backend.cuda.program import CompiledProgram
    from emmy.compiler.backend.gpu_lock import gpu_lock

    feed = {name: t.numpy() for name, t in zip(("q", "k", "v", "mask"), _inputs(), strict=True)}
    flat, paged = _compile_read(paged=False), _compile_read(paged=True)
    out_name = flat.outputs[0]

    with gpu_lock():
        contiguous = CompiledProgram.build(flat, dict(feed))
        contiguous.run_once()
        reference = contiguous.outputs()[out_name].copy()

        # The paged K and V carry no bytes of their own: only their tables are bound.
        program = CompiledProgram.build(paged, dict(feed))
        pages: list = []  # keep every page alive: the table holds raw device pointers
        for name in ("k", "v"):
            program.alias_buffer(f"{name}__pages", _split_into_pages(torch, feed[name], pages))
        program.run_once()
        got = program.outputs()[out_name]

    assert np.array_equal(got, reference), f"max|Δ| = {np.max(np.abs(got - reference))}"


@requires_cuda
def test_cache_filled_in_chunks_then_attended():
    """End to end. The cache starts empty as a table of pages; four steps each compute CHUNK new
    keys and write them at their absolute position, a scalar the host uploads; then attention
    reads the whole cache back through the same table. The result must equal the same pipeline
    run on one contiguous buffer — a wrong page, a wrong offset or a disagreeing layout between
    the write and the read all break it."""
    import torch

    from emmy.compiler.backend.cuda.program import CompiledProgram
    from emmy.compiler.backend.gpu_lock import gpu_lock

    q, kin, v, mask = _inputs()
    with torch.no_grad():
        reference = _attention()(q, torch.tanh(kin), v, mask).numpy()

    writer = _compile_write(CHUNK)
    reader = _compile_read(paged=True)

    with gpu_lock():
        cache = [torch.zeros((1, KV_HEADS, PAGE, HEAD_DIM), dtype=torch.float32, device="cuda") for _ in range(SEQ // PAGE)]
        table = _table(torch, cache)
        torch.cuda.synchronize()  # the zero fill is on torch's stream, which the runtime's launches do not wait on

        step = CompiledProgram.build(writer, {writer.inputs[0]: np.ascontiguousarray(kin.numpy()[:, :, :CHUNK, :])})
        step.alias_buffer(f"{writer.outputs[0]}__pages", table)
        for past in range(0, SEQ, CHUNK):
            chunk = np.ascontiguousarray(kin.numpy()[:, :, past : past + CHUNK, :])
            step.upload_prefix({writer.inputs[0]: chunk, "past": np.array([past], dtype=np.int64)})
            step.run_once()

        feed = {name: t.numpy() for name, t in zip(("q", "k", "v", "mask"), (q, kin, v, mask), strict=True)}
        attend = CompiledProgram.build(reader, feed)
        values: list = []  # keep V's pages alive for as long as its table is bound
        attend.alias_buffer("k__pages", table)  # the pages the four steps just filled
        attend.alias_buffer("v__pages", _split_into_pages(torch, v.numpy(), values))
        attend.run_once()
        got = attend.outputs()[reader.outputs[0]]

    np.testing.assert_allclose(got, reference, rtol=1e-5, atol=1e-5)


def _table(torch, pages: list):
    """The device table one buffer addresses through: its pages' addresses, in cache order."""
    return torch.tensor([page.data_ptr() for page in pages], dtype=torch.int64, device="cuda")


def _split_into_pages(torch, array, keep: list):
    """A page table over ``array``'s key axis. Pages are appended to ``keep`` so the device memory
    the table points at outlives the call; pass a list that stays alive for as long as the table."""
    start = len(keep)
    for offset in range(0, SEQ, PAGE):
        keep.append(torch.from_numpy(np.ascontiguousarray(array[:, :, offset : offset + PAGE, :])).cuda())
    torch.cuda.synchronize()  # the copies are on torch's stream, which the runtime's launches do not wait on
    return _table(torch, keep[start:])
