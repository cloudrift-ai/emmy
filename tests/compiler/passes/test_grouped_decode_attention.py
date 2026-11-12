"""A decode or verify step attends per key-value head: native's grouped twin puts a key-value head's query heads on the
axis the chunk tier folds as its rows, so the group shares one read of the head's cache, and a key split's finalize
combines the slices' partial states, never the window. The per-head spelling a prefill chunk keeps reads the cache once
per query head instead."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from emmy.compiler.backend.gpu_lock import gpu_lock
from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import decode_bf16
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.schedule.classic import refusals
from emmy.compiler.ir.stmt import Load
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, Pipeline
from emmy.compiler.pipeline.search.pins import pinned_knobs
from emmy.compiler.trace.torch import trace_module
from emmy.serving.native.prepare import WINDOW, attend_module
from tests.compiler.helpers import requires_cuda

# The 27B's geometry: 24 query heads in groups of six over four key-value heads.
HEADS, KV, D, CONTEXT = 24, 4, 256, 4096
GROUP = HEADS // KV
CACHES = {"bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}
SLICES = 4  # a key split across the window, ``g<SLICES>k``
CTX = Context.from_target((12, 0))


def _pins(cache, split: int = 0) -> dict[str, str]:
    """The chunk tier on the twin, one warp over the group and the whole head dimension (32 n8 columns), so a CTA
    covers what one read of a key-value head's cache serves. A split's partial names its sites from its own root."""
    site = "" if split else "map.1/map.1/"
    pins = {
        "PLACE": "fuse",
        "WORK": "w1x1",
        "REDUCE@map.1/map.1/twist": f"g{split}k" if split else "",
        f"TILE@{site}twist": "mma_m16n8k16_bf16_f32/f1x32/k4",
        f"TILE@{site}twist.1/inner": "mma_m16n8k16_bf16_f32/f1x8/k4",
    }
    if cache != torch.bfloat16:
        pins[f"STAGE@{site}twist"] = ""  # an FP8 value stages no slab
    return pins


def _attention(rows, cache=torch.bfloat16, grouped=True):
    """Native's attention twin at ``rows`` over a symbolic window, traced as the export traces it."""
    module = attend_module(HEADS, KV, D, rows, grouped=grouped)
    window = torch.export.Dim(WINDOW, min=rows, max=CONTEXT)
    shapes = {n: {0: window} if n in ("keys", "values") else {} for n in inspect.signature(module.forward).parameters}
    # Two tensors, not one twice: the tracer takes a shared example tensor as one aliased input.
    caches = (torch.zeros(512, KV, D, dtype=cache), torch.zeros(512, KV, D, dtype=cache))
    examples = (torch.zeros(rows, HEADS * D, dtype=torch.bfloat16), *caches, torch.zeros(1, dtype=torch.int64))
    return module, trace_module(module, examples, dynamic_shapes=shapes)


def _lifted(graph):
    """The lifted kernel and its twisted carrier."""
    tiled = Pipeline.build(["tile/lift"]).run(Pipeline.build(LOOP_PASSES).run(graph), ctx=CTX)
    (root,) = [node.op for node in tiled.nodes.values() if isinstance(node.op, TileOp)]

    def folds(node):
        return [node, *(f for e in node.operands for f in folds(e))] if isinstance(node, Fold) else []

    (twist,) = [f for f in folds(root.op) if f.twist is not None]
    return root, twist


def _cache_reads(node):
    """Every load of the cache in ``node``'s fold tree, with the index it reads at."""
    loads = [stmt for stmt in node.lift.body if isinstance(stmt, Load) and stmt.input in ("keys", "values")]
    return loads + [load for edge in node.operands if isinstance(edge, Fold) for load in _cache_reads(edge)]


@pytest.mark.parametrize("cache", CACHES)
@pytest.mark.parametrize("rows", [1, 3])
def test_a_key_value_head_s_query_group_is_the_fold_s_rows(rows, cache) -> None:
    """The output's storage order puts the group second to last, so the tensor-core fold takes it as its rows and the
    head dimension as its columns, and no read of a key or a value depends on that axis: one read serves the group. The
    per-head spelling reads the cache through the query head instead, once per head of the group."""
    root, twist = _lifted(_attention(rows, CACHES[cache])[1])
    *_, fold_rows, columns = root.place.free
    assert (fold_rows.extent, columns.extent) == (Dim(GROUP), Dim(D))
    reads = _cache_reads(twist)
    assert {load.input for load in reads} == {"keys", "values"}
    assert all(fold_rows.name not in {name for index in load.index for name in index.free_vars()} for load in reads)
    assert twist.chunked() and refusals._chunk_refusal(root, twist) is None
    assert "mma_m16n8k16_bf16_f32" in refusals._warp_atoms(root, CTX, twist)
    per_head, folded = _lifted(_attention(rows, CACHES[cache], grouped=False)[1])
    (heads,) = [axis for axis in per_head.place.free if axis.extent == Dim(HEADS)]
    assert all(heads.name in load.index[1].free_vars() for load in _cache_reads(folded))


@pytest.mark.parametrize("rows", [1, 3])
def test_a_key_split_combines_the_slices_partials_not_the_window(rows) -> None:
    """Split across the window, a partial per row, key-value head and slice reads its slice of the head's cache once for
    the whole group; the finalize reads the partials alone and never the window, which is no extent of it."""
    _, graph = _attention(rows)
    with pinned_knobs(_pins(torch.bfloat16, SLICES)):
        lowered = Pipeline.build(CUDA_PASSES).run(graph, ctx=CTX)
    kernels = {nid: node for nid, node in lowered.nodes.items() if isinstance(node.op, CudaOp)}
    assert len(kernels) == 2, sorted(kernels)
    (partial,) = [node for nid, node in kernels.items() if nid.endswith("__partial")]
    (finalize,) = [node for node in kernels.values() if node is not partial]
    assert "mma.sync" in partial.op.kernel_source and "__ck" in partial.op.kernel_source
    assert np.prod([extent for axis in partial.op.grid for extent in axis]) == rows * KV * SLICES
    assert list(finalize.inputs) == [partial.id]
    assert WINDOW not in finalize.op.kernel_source


@requires_cuda
@pytest.mark.parametrize("split", [0, SLICES], ids=["whole", "split"])
@pytest.mark.parametrize("cache", CACHES)
@pytest.mark.parametrize("rows", [1, 3])
def test_the_grouped_attention_matches_the_per_head_twin(rows, cache, split) -> None:
    """The grouped twin, one tensor-core kernel or a partial per key slice and its finalize, computes the per-head
    twin's attention at every position and window a step can set, one the slices overhang included."""
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.cuda.program import CompiledProgram

    dtype = CACHES[cache]
    _, graph = _attention(rows, dtype)
    with pinned_knobs(_pins(dtype, split)):
        compiled = CudaBackend().compile(graph)
    sources = [node.op.kernel_source for node in compiled.nodes.values() if isinstance(node.op, CudaOp)]
    assert len(sources) == (2 if split else 1) and "mma.sync" in sources[0]
    per_head = attend_module(HEADS, KV, D, rows)
    rng = np.random.default_rng(11)
    query = torch.from_numpy(rng.normal(size=(rows, HEADS * D)).astype(np.float32)).to(torch.bfloat16)
    stored = torch.from_numpy(rng.normal(size=(2, CONTEXT, KV, D)).astype(np.float32)).to(dtype)

    def bits(t):
        # The host carries bf16 as its bit pattern and an FP8 buffer as its codes.
        return t.view(torch.int16).numpy().view(np.uint16) if t.dtype == torch.bfloat16 else t.view(torch.uint8).numpy()

    with gpu_lock():
        program = CompiledProgram.build(
            compiled, {"q": bits(query), "position": np.array([0], np.int64), "keys": bits(stored[0]), "values": bits(stored[1])}
        )
        for position in (0, 5, 937, CONTEXT - rows):
            # The window the step's rows reach, the runtime's 4096-position step, and a window no split width divides.
            for span in sorted({position + rows, CONTEXT, max(position + rows, 1001)}):
                program.set_sym_values({WINDOW: span})  # the window moves the buffers, so the cache uploads after it
                program.upload_prefix(
                    {"position": np.array([position], np.int64), "keys": bits(stored[0][:span]), "values": bits(stored[1][:span])}
                )
                program.run_once()
                actual = decode_bf16(program.outputs()[compiled.outputs[0]]).reshape(rows, HEADS * D)
                expected = per_head(query, stored[0][:span], stored[1][:span], torch.tensor([position])).float().numpy()
                np.testing.assert_allclose(actual, expected, rtol=3e-2, atol=3e-2)
