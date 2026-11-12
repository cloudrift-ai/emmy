"""The chunk tier folds native's causal attention, whose mask compares the key against the row and a position read from
device memory: the mask reaches the tier as a comparison and a ``where`` over values that alias the coordinates, and the
tier spells it as the coordinate Select it lands, the position a uniform value in the predicate. A key split across the
symbolic window folds the same way per slice, the partition coordinate in the predicate and the overhang masked."""

from __future__ import annotations

import inspect
import re

import numpy as np
import pytest
import torch

from emmy.compiler.backend.gpu_lock import gpu_lock
from emmy.compiler.context import Context
from emmy.compiler.dtype import F8E4M3, decode_bf16
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.schedule import Stage, Tile, Work
from emmy.compiler.ir.schedule.classic import refusals
from emmy.compiler.ir.schedule.staging import chunk_slab_dtype, chunk_slab_pad
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, Pipeline
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
BYTES = "d1/smem-async"  # an FP8 cache copied verbatim into byte slabs, its fragment reads converting
CELLS = {  # the twist's cell and an FP8 cache's transport; the score inner stays bf16_f32, the query's own width
    "bf16": (torch.bfloat16, "mma_m16n8k16_bf16_f32", None),
    "fp8": (torch.float8_e4m3fn, "mma_m16n8k16_bf16_f32", ""),
    "fp8-f16": (torch.float8_e4m3fn, "mma_m16n8k16_f16_f32", ""),
    "fp8-f16acc": (torch.float8_e4m3fn, "mma_m16n8k16_f16_f16", ""),
    "fp8-staged": (torch.float8_e4m3fn, "mma_m16n8k16_bf16_f32", BYTES),
    "fp8-f16acc-staged": (torch.float8_e4m3fn, "mma_m16n8k16_f16_f16", BYTES),
}


def _flash(cell: str, stage: str | None, split: str = "") -> dict[str, str]:
    """The chunk tier's pins on the twin, or on the partial piece of its key split, whose site keys drop the map prefix."""
    pins = {"TILE@twist": f"{cell}/f1x16/k4", "TILE@twist.1/inner": "mma_m16n8k16_bf16_f32/f1x8/k4"}
    if stage is not None:
        pins["STAGE@twist"] = stage
    site = "" if split else "map.1/map.1/"
    return {"PLACE": "fuse", "WORK": "w4x1", "REDUCE@map.1/map.1/twist": split, **{k.replace("@", f"@{site}"): v for k, v in pins.items()}}


def _attention(cache=torch.bfloat16, heads=HEADS, offset=0, row=D, query=None):
    """Native's 64-row attention twin over a symbolic window, traced as the export traces it. ``offset`` and ``row``
    read each head's dims as a slice of a padded cache row; ``query`` reads the queries as a slice at that offset of
    a row 16 elements wider."""
    attend = attend_module(heads, KV, D, ROWS)

    class Sliced(torch.nn.Module):
        def forward(self, q, keys, values, position):
            q = q if query is None else q[:, query : query + heads * D]
            return attend(q, keys[:, :, offset : offset + D], values[:, :, offset : offset + D], position)

    module = attend if (offset, row, query) == (0, D, None) else Sliced()
    window = torch.export.Dim(WINDOW, min=ROWS, max=CONTEXT)
    shapes = {n: {0: window} if n in ("keys", "values") else {} for n in inspect.signature(module.forward).parameters}
    # Two tensors, not one twice: the tracer takes a shared example tensor as one aliased input.
    examples = (
        torch.zeros(ROWS, heads * D + (0 if query is None else 16), dtype=torch.bfloat16),
        torch.zeros(512, KV, row, dtype=cache),
        torch.zeros(512, KV, row, dtype=cache),
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
    in its fragment load, so the 16-bit families are offered. On cp.async the cache also stages, its key and value
    copied into byte slabs whose rows take the 16-byte pad; a blocking copy or a TMA box takes no byte slab."""
    _, graph = _attention(torch.float8_e4m3fn)
    root, ctx, twist = _twist(graph)
    assert twist.chunked()
    assert refusals._node_refusal(root.op, ctx, twist, True) is None
    assert set(refusals._warp_atoms(root.op, ctx, twist)) >= {cell for _cache, cell, _stage in CELLS.values()}
    tile, inputs = Tile.parse("mma_m16n8k16_f16_f16/f1x16/k4", Work.parse("w4x1")), root.op.inputs
    assert chunk_slab_dtype(tile, Stage.parse(BYTES), inputs, "keys") == F8E4M3
    assert chunk_slab_pad(tile, Stage.parse(BYTES), F8E4M3) == 16
    assert chunk_slab_dtype(tile, Stage.parse("d1/smem-tma"), inputs, "keys") is None
    bf16_inputs = _twist(_attention()[1])[0].op.inputs
    assert chunk_slab_dtype(tile, Stage.parse(BYTES), bf16_inputs, "keys") is None, "a 16-bit cache under an f16 cell stages nothing"


@pytest.mark.parametrize(
    ("cache", "target", "score", "spelled"),
    [
        (torch.float8_e4m3fn, (12, 0), "mma_m16n8k16_bf16_f32", ("__nv_fp8_e4m3 _a_smem", "emmy_cp_async")),
        (torch.bfloat16, (9, 0), "wgmma_m64n64k16_bf16_f32", ("__nv_bfloat16 _a_smem", "emmy_cp_async", "wgmma")),
        (torch.float8_e4m3fn, (9, 0), "wgmma_m64n64k16_bf16_f32", ()),
    ],
    ids=["fp8-mma", "bf16-wgmma", "fp8-wgmma"],
)
def test_a_staged_key_is_read_only_by_a_cell_that_converts_it(cache, target, score, spelled) -> None:
    """GPU-free. Under the staged pins an FP8 key stages as a byte slab on cp.async, which the ``mma.sync`` score
    converts as it reads. A warp-group score reads its key slab through a descriptor, which converts nothing: it takes
    a bf16 key, and refuses an FP8 one, so no kernel reads the key's bytes as bf16."""
    _, graph = _attention(cache)
    pins = {**_flash("mma_m16n8k16_bf16_f32", BYTES), "TILE@map.1/map.1/twist.1/inner": f"{score}/f1x8/k4"}
    with pinned_knobs(pins):
        compiled = Pipeline.build(CUDA_PASSES).run(graph, ctx=Context.from_target(target))
    (source,) = [node.op.kernel_source for node in compiled.nodes.values() if isinstance(node.op, CudaOp)]
    assert all(text in source for text in spelled), spelled
    assert ("wgmma" in source) == ("wgmma" in spelled)


@pytest.mark.parametrize(
    ("cache", "offset", "row", "staged"),
    [
        (torch.float8_e4m3fn, 16, D + 16, True),
        (torch.float8_e4m3fn, 1, D + 16, False),
        (torch.float8_e4m3fn, 0, D + 8, False),
        (torch.bfloat16, 8, D + 16, True),
        (torch.bfloat16, 1, D + 16, False),
    ],
    ids=["fp8-16B-offset", "fp8-1B-offset", "fp8-136B-rows", "bf16-16B-offset", "bf16-2B-offset"],
)
def test_a_cache_slice_stages_only_on_the_16_byte_grid(cache, offset, row, staged) -> None:
    """GPU-free. A cp.async copy moves 16 B from the address the cache's index names, so a slice of a padded cache
    stages only when its offset and its row strides keep every copy on the 16 B grid; else the pin does not resolve."""
    _, graph = _attention(cache, offset=offset, row=row)

    def compile_():
        with pinned_knobs(_flash("mma_m16n8k16_bf16_f32", BYTES)):
            return Pipeline.build(CUDA_PASSES).run(graph, ctx=Context.from_target((12, 0)))

    if not staged:
        with pytest.raises(ValueError, match="does not resolve"):
            compile_()
        return
    (source,) = [node.op.kernel_source for node in compile_().nodes.values() if isinstance(node.op, CudaOp)]
    assert "emmy_cp_async" in source


def test_the_expectation_reads_the_weight_the_denominator_computed() -> None:
    """GPU-free. The twin's softmax weight ``exp(score - pivot)`` is one fragment per score element, folded into the
    denominator and repacked as the expectation's operand: one ``exp`` per element. The normal form spells the
    difference as ``-pivot + score``; evaluated one operation at a time from that spelling, the negated pivot became a
    row temp under each channel's own name, the fragment memo matched nothing across the two channels, and the
    expectation computed every exp a second time (Qwen3.6's 4-token decode twin 127 → 171 us on an RTX 5090)."""
    _, graph = _attention()
    with pinned_knobs(_flash("mma_m16n8k16_bf16_f32", None)):
        compiled = Pipeline.build(CUDA_PASSES).run(graph, ctx=Context.from_target((12, 0)))
    (source,) = [node.op.kernel_source for node in compiled.nodes.values() if isinstance(node.op, CudaOp)]
    # One element-wise exponential per score fragment of the inner f1x8 tile; a recomputed weight would double them.
    weights = re.findall(r"(\w+)\[_e\] = __expf\(", source)
    assert len(weights) == len(set(weights)) == 8, weights


def test_a_sum_with_a_negated_term_is_evaluated_as_a_subtraction() -> None:
    """GPU-free. The atom lowering evaluates a sum one operation at a time; a negated term, or one scaled by a negative
    literal, is the subtraction of its positive form, exact by IEEE's definition of subtraction, so no sum leaves a
    negate temp behind. Only the one node is re-spelled: a sum every leading term of which is negated keeps its
    negate, since reassociating it would round differently."""
    from importlib import import_module

    from emmy.compiler.ir.stmt import parse_body

    atom = import_module("emmy.compiler.pipeline.passes.lowering.kernel._atom")

    def spelled(text: str) -> str:
        return "; ".join(str(stmt) for stmt in atom._one_operation(parse_body(text)))

    assert spelled("v = exp(-m + s)") == "v__e0 = s - m; v = exp(v__e0)"
    assert spelled("v = exp(s + -2.0 * m)") == "v__e0 = 2.0 * m; v__e1 = s - v__e0; v = exp(v__e1)"
    assert spelled("v = exp(s - m - c)") == "v__e0 = s - m; v__e1 = v__e0 - c; v = exp(v__e1)"
    assert spelled("v = exp(-a - b + c)") == "v__e0 = -a; v__e1 = v__e0 - b; v__e2 = v__e1 + c; v = exp(v__e2)"


def test_the_mask_bound_reads_the_negation_the_normal_form_spells() -> None:
    """GPU-free. The normal form spells ``key - row`` as ``-row + key``; the stream bound reads that affine form as
    the same ``{key: 1, row: -1}``, or a causal mask on a wide launch would bound nothing."""
    from importlib import import_module

    from emmy.compiler.ir.stmt import parse_body

    atom = import_module("emmy.compiler.pipeline.passes.lowering.kernel._atom")
    (negated,) = parse_body("bool v = -a1 + a2 > in0")
    (subtracted,) = parse_body("bool v = a2 - a1 > in0")
    assert atom._mask_affine(negated.value, frozenset({"a1", "a2"}))[0] == {"a1": -1, "a2": 1}
    assert atom._mask_affine(negated.value, frozenset({"a1", "a2"}))[0] == atom._mask_affine(subtracted.value, frozenset({"a1", "a2"}))[0]


@pytest.mark.parametrize(("query", "wgmma"), [(8, True), (1, False)], ids=["16B-offset", "2B-offset"])
def test_a_warp_group_score_takes_a_query_slice_only_on_the_16_byte_grid(query, wgmma) -> None:
    """GPU-free. A warp-group score reads its query off a slab the kernel copies on cp.async ahead of the chunk loop,
    so a query sliced off the 16 B grid keeps the ``mma.sync`` score, whose fragments load straight from gmem."""
    _, graph = _attention(query=query)
    pins = {**_flash("mma_m16n8k16_bf16_f32", BYTES), "TILE@map.1/map.1/twist.1/inner": "wgmma_m64n64k16_bf16_f32/f1x8/k4"}
    with pinned_knobs(pins):
        compiled = Pipeline.build(CUDA_PASSES).run(graph, ctx=Context.from_target((9, 0)))
    (source,) = [node.op.kernel_source for node in compiled.nodes.values() if isinstance(node.op, CudaOp)]
    assert ("wgmma" in source) is wgmma


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

    cache, atom, stage = CELLS[cell]
    module, graph = _attention(cache, heads)
    with pinned_knobs(_flash(atom, stage, split)):
        compiled = CudaBackend().compile(graph)
    sources = [node.op.kernel_source for node in compiled.nodes.values() if isinstance(node.op, CudaOp)]
    assert len(sources) == (2 if split else 1), "the chunk tier's one kernel, or the split's partial and finalize"
    source = sources[0]
    assert "mma.sync" in source and "__ck" in source, "the chunk tier folds the stream, or each slice of it"
    if stage is not None:
        assert ("emmy_cp_async" in source) == bool(stage), "an FP8 cache stages exactly when its row spells a transport"
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
