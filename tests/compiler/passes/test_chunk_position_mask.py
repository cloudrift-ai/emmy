"""The chunk tier folds native's causal attention, whose mask compares the key against the row and a position read from
device memory: the mask reaches the tier as a comparison and a ``where`` over values that alias the coordinates, and the
tier spells it as the coordinate Select it lands, the position a uniform value in the predicate. A key split across the
symbolic window folds the same way per slice, the partition coordinate in the predicate and the overhang masked."""

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
from emmy.compiler.pipeline.passes.tile._split import split_forks
from emmy.compiler.pipeline.search.pins import pinned_knobs
from emmy.compiler.trace.torch import trace_module
from emmy.serving.native.prepare import WINDOW, attend_module
from tests.compiler.helpers import requires_cuda

HEADS, KV, D, ROWS, CONTEXT = 4, 2, 128, 64, 1024
# One CTA per head: 4 heads launch one wave, where the tier keeps the stream whole; 192 heads launch more CTAs than
# any card has SMs, where the stream stops at the block end plus the position, a bound decided at runtime.
WIDE = 192
STOP = "__ck_end"  # the chunk loop's bound, a variable its header carries only when a mask bounds the stream
SPLIT = "g4k"  # a key split across the window: four CTAs per head, each over a quarter of the keys
CELLS = {  # the twist's cell; the score inner stays bf16_f32, the query's own width
    "bf16": (torch.bfloat16, "mma_m16n8k16_bf16_f32"),
    "fp8": (torch.float8_e4m3fn, "mma_m16n8k16_bf16_f32"),
    "fp8-f16": (torch.float8_e4m3fn, "mma_m16n8k16_f16_f32"),
    "fp8-f16acc": (torch.float8_e4m3fn, "mma_m16n8k16_f16_f16"),
}


def _flash(cell: str, cache, split: str = "") -> dict[str, str]:
    """The chunk tier's pins on the twin, or on the partial piece of its key split, whose site keys drop the map prefix."""
    pins = {"TILE@twist": f"{cell}/f1x16/k4", "TILE@twist.1/inner": "mma_m16n8k16_bf16_f32/f1x8/k4"}
    if cache != torch.bfloat16:
        pins["STAGE@twist"] = ""  # an FP8 value stages no slab
    site = "" if split else "map.1/map.1/"
    return {"PLACE": "fuse", "WORK": "w4x1", "REDUCE@map.1/map.1/twist": split, **{k.replace("@", f"@{site}"): v for k, v in pins.items()}}


def _attention(cache=torch.bfloat16, heads=HEADS):
    """Native's 64-row attention twin over a symbolic window, traced as the export traces it."""
    module = attend_module(heads, KV, D, ROWS)
    window = torch.export.Dim(WINDOW, min=ROWS, max=CONTEXT)
    shapes = {n: {0: window} if n in ("keys", "values") else {} for n in inspect.signature(module.forward).parameters}
    # Two tensors, not one twice: the tracer takes a shared example tensor as one aliased input.
    examples = (
        torch.zeros(ROWS, heads * D, dtype=torch.bfloat16),
        torch.zeros(512, KV, D, dtype=cache),
        torch.zeros(512, KV, D, dtype=cache),
    )
    return module, trace_module(module, (*examples, torch.zeros(1, dtype=torch.int64)), dynamic_shapes=shapes)


def _twist(graph):
    """The lifted kernel's graph node, the context and its twisted carrier."""
    ctx = Context.from_target((12, 0))
    tiled = Pipeline.build(["tile/lift"]).run(Pipeline.build(LOOP_PASSES).run(graph), ctx=ctx)
    (root,) = [node for node in tiled.nodes.values() if isinstance(node.op, TileOp)]

    def folds(node):
        return [node, *(f for e in node.operands for f in folds(e))] if isinstance(node, Fold) else []

    (twist,) = [f for f in folds(root.op.op) if f.twist is not None]
    return root, ctx, twist


def test_the_position_mask_is_a_coordinate_select_the_chunk_tier_takes() -> None:
    """The twin's mask hoists its row alias and the position into a leaf over the row axis; that leaf is uniform to
    the tier, so the expectation is bilinear, the carrier chunks, and the tensor-core atom is offered."""
    _, graph = _attention()
    root, ctx, twist = _twist(graph)
    assert twist.chunked(), "the value expectation must join the twist as its one bilinear channel"
    assert refusals._chunk_refusal(root.op, twist) is None
    assert refusals._warp_atoms(root.op, ctx, twist) == ("mma_m16n8k16_bf16_f32",)


def test_the_key_split_is_offered_across_the_symbolic_window() -> None:
    """The window's length is symbolic, so the last slice overhangs it: a twisted carrier masks the overhang through its
    score, as a monoid folds its seed there, and the split offers every width it has."""
    _, graph = _attention()
    root, _, _ = _twist(graph)
    offered = {spelling for fork in split_forks(None, root) for spelling in fork.knobs.values()}
    assert {"", SPLIT, "g8k", "g32k"} <= offered, offered


def test_an_fp8_cache_takes_a_16_bit_cell_whose_value_load_converts() -> None:
    """The weight repacks from f32 registers into a 16-bit fragment whatever the cache stores; an FP8 value converts
    in its gmem-direct fragment load, so the 16-bit families are offered and no staged slab is."""
    _, graph = _attention(torch.float8_e4m3fn)
    root, ctx, twist = _twist(graph)
    assert twist.chunked()
    assert refusals._node_refusal(root.op, ctx, twist, True) is None
    assert set(refusals._warp_atoms(root.op, ctx, twist)) >= {cell for _cache, cell in CELLS.values()}


@requires_cuda
@pytest.mark.parametrize("split", ["", SPLIT], ids=["whole", "split"])
@pytest.mark.parametrize("heads", [HEADS, WIDE], ids=["one-wave", "wide"])
@pytest.mark.parametrize("cell", CELLS)
def test_the_folded_attention_matches_the_twin_over_the_causal_prefix(cell, heads, split) -> None:
    """One tensor-core kernel at the symbolic window, or a partial per key slice and its finalize, at every position and
    window a step can set, one the slices overhang included: the keys past a row's own position never reach the
    output, whatever the cache stores and whether or not the stream stops early."""
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.cuda.program import CompiledProgram

    cache, atom = CELLS[cell]
    module, graph = _attention(cache, heads)
    with pinned_knobs(_flash(atom, cache, split)):
        compiled = CudaBackend().compile(graph)
    sources = [node.op.kernel_source for node in compiled.nodes.values() if isinstance(node.op, CudaOp)]
    assert len(sources) == (2 if split else 1), "the chunk tier's one kernel, or the split's partial and finalize"
    source = sources[0]
    assert "mma.sync" in source and "__ck" in source, "the chunk tier folds the stream, or each slice of it"
    assert (STOP in source) == (heads == WIDE), "the position bounds the stream on a launch of more than one wave"
    rng = np.random.default_rng(7)
    query = torch.from_numpy(rng.normal(size=(ROWS, heads * D)).astype(np.float32)).to(torch.bfloat16)
    stored = torch.from_numpy(rng.normal(size=(2, CONTEXT, KV, D)).astype(np.float32)).to(cache)

    def bits(t):
        # The host carries bf16 as its bit pattern and an FP8 buffer as its codes.
        return t.view(torch.int16).numpy().view(np.uint16) if t.dtype == torch.bfloat16 else t.view(torch.uint8).numpy()

    with gpu_lock():
        program = CompiledProgram.build(
            compiled, {"q": bits(query), "position": np.array([0], np.int64), "keys": bits(stored[0]), "values": bits(stored[1])}
        )
        for position in (0, 100, 937, CONTEXT - ROWS):
            # The step's 512-wide buckets, the whole context, and a window no split width divides.
            for span in sorted({-(-(position + ROWS) // 512) * 512, CONTEXT, max(position + ROWS, 1001)}):
                program.set_sym_values({WINDOW: span})  # the window moves the buffers, so the cache uploads after it
                program.upload_prefix(
                    {"position": np.array([position], np.int64), "keys": bits(stored[0][:span]), "values": bits(stored[1][:span])}
                )
                program.run_once()
                actual = decode_bf16(program.outputs()[compiled.outputs[0]]).reshape(ROWS, heads * D)
                expected = module(query, stored[0][:span], stored[1][:span], torch.tensor([position])).float().numpy()
                np.testing.assert_allclose(actual, expected, rtol=3e-2, atol=3e-2)
