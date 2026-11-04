"""The chunk tier folds native's causal attention, whose mask compares the key against the row and a position read from
device memory: the mask reaches the tier as a comparison and a ``where`` over values that alias the coordinates, and the
tier spells it as the coordinate Select it lands, the position a uniform value in the predicate."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from emmy.compiler.backend.gpu_lock import gpu_lock
from emmy.compiler.context import Context
from emmy.compiler.dtype import decode_bf16
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.schedule.classic import refusals
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import LOOP_PASSES, Pipeline
from emmy.compiler.pipeline.search.pins import pinned_knobs
from emmy.compiler.trace.torch import trace_module
from emmy.serving.native.prepare import WINDOW, attend_module
from tests.compiler.helpers import requires_cuda

HEADS, KV, D, ROWS, CONTEXT = 4, 2, 128, 64, 1024
FLASH = {
    "PLACE": "fuse",
    "TILE@map.1/map.1/twist": "mma_m16n8k16_bf16_f32/f1x16/k4",
    "TILE@map.1/map.1/twist.1/inner": "mma_m16n8k16_bf16_f32/f1x8/k4",
    "WORK": "w4x1",
}


def _attention(cache=torch.bfloat16):
    """Native's 64-row attention twin over a symbolic window, traced as the export traces it."""
    module = attend_module(HEADS, KV, D, ROWS)
    window = torch.export.Dim(WINDOW, min=ROWS, max=CONTEXT)
    shapes = {n: {0: window} if n in ("keys", "values") else {} for n in inspect.signature(module.forward).parameters}
    # Two tensors, not one twice: the tracer takes a shared example tensor as one aliased input.
    examples = (
        torch.zeros(ROWS, HEADS * D, dtype=torch.bfloat16),
        torch.zeros(512, KV, D, dtype=cache),
        torch.zeros(512, KV, D, dtype=cache),
    )
    return module, trace_module(module, (*examples, torch.zeros(1, dtype=torch.int64)), dynamic_shapes=shapes)


def _twist(graph):
    ctx = Context.from_target((12, 0))
    tiled = Pipeline.build(["tile/lift"]).run(Pipeline.build(LOOP_PASSES).run(graph), ctx=ctx)
    (tile,) = [node.op for node in tiled.nodes.values() if isinstance(node.op, TileOp)]

    def folds(node):
        return [node, *(f for e in node.operands for f in folds(e))] if isinstance(node, Fold) else []

    (twist,) = [f for f in folds(tile.op) if f.twist is not None]
    return tile, ctx, twist


def test_the_position_mask_is_a_coordinate_select_the_chunk_tier_takes() -> None:
    """The twin's mask hoists its row alias and the position into a leaf over the row axis; that leaf is uniform to
    the tier, so the expectation is bilinear, the carrier chunks, and the tensor-core atom is offered."""
    _, graph = _attention()
    tile, ctx, twist = _twist(graph)
    assert twist.chunked(), "the value expectation must join the twist as its one bilinear channel"
    assert refusals._chunk_refusal(tile, twist) is None
    assert refusals._warp_atoms(tile, ctx, twist) == ("mma_m16n8k16_bf16_f32",)


def test_an_fp8_cache_takes_a_16_bit_cell_whose_value_load_converts() -> None:
    """The weight repacks from f32 registers into a 16-bit fragment whatever the cache stores; an FP8 value converts
    in its gmem-direct fragment load, so the 16-bit families are offered and no staged slab is."""
    _, graph = _attention(torch.float8_e4m3fn)
    tile, ctx, twist = _twist(graph)
    assert twist.chunked()
    assert refusals._node_refusal(tile, ctx, twist, True) is None
    assert set(refusals._warp_atoms(tile, ctx, twist)) >= {"mma_m16n8k16_bf16_f32", "mma_m16n8k16_f16_f32"}


@requires_cuda
@pytest.mark.parametrize("cache", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"])
def test_the_folded_attention_matches_the_twin_over_the_causal_prefix(cache) -> None:
    """One tensor-core kernel at the symbolic window, at every position and window a step can set: the keys past a
    row's own position never reach the output, whatever the cache stores."""
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.cuda.program import CompiledProgram

    module, graph = _attention(cache)
    pins = FLASH if cache == torch.bfloat16 else {**FLASH, "STAGE@map.1/map.1/twist": ""}  # an FP8 value stages no slab
    with pinned_knobs(pins):
        compiled = CudaBackend().compile(graph)
    (source,) = [node.op.kernel_source for node in compiled.nodes.values() if isinstance(node.op, CudaOp)]
    assert "mma.sync" in source and "__ck" in source, "the chunk tier's one kernel"
    rng = np.random.default_rng(7)
    query = torch.from_numpy(rng.normal(size=(ROWS, HEADS * D)).astype(np.float32)).to(torch.bfloat16)
    stored = torch.from_numpy(rng.normal(size=(2, CONTEXT, KV, D)).astype(np.float32)).to(cache)

    def bits(t):
        # The host carries bf16 as its bit pattern and an FP8 buffer as its codes.
        return t.view(torch.int16).numpy().view(np.uint16) if t.dtype == torch.bfloat16 else t.view(torch.uint8).numpy()

    with gpu_lock():
        program = CompiledProgram.build(
            compiled, {"q": bits(query), "position": np.array([0], np.int64), "keys": bits(stored[0]), "values": bits(stored[1])}
        )
        for position in (0, 100, CONTEXT - ROWS):
            for span in sorted({-(-(position + ROWS) // 512) * 512, CONTEXT}):
                program.set_sym_values({WINDOW: span})  # the window moves the buffers, so the cache uploads after it
                program.upload_prefix(
                    {"position": np.array([position], np.int64), "keys": bits(stored[0][:span]), "values": bits(stored[1][:span])}
                )
                program.run_once()
                actual = decode_bf16(program.outputs()[compiled.outputs[0]]).reshape(ROWS, HEADS * D)
                expected = module(query, stored[0][:span], stored[1][:span], torch.tensor([position])).float().numpy()
                np.testing.assert_allclose(actual, expected, rtol=3e-2, atol=3e-2)
