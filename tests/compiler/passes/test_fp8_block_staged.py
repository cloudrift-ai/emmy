"""The packed byte-slab stage over a block-scaled fp8 weight, and the fill's per-chunk statistic.

An fp8 weight whose scale changes every 128 elements along K cannot move its scale onto the
epilogue, so it reaches the tensor cores as a computed B. The packed byte-slab stage reads it the
way it reads a packed-pair weight: the bytes copy verbatim, the block scales fill a small f32 slab,
and the fragment load converts, scales and rounds once — the compute fill's own arithmetic, so the
two are bit-identical. The grouped activation quantize in front of it is a per-row statistic over
each 128-wide K group, which a staged chunk inside that group evaluates once per row.
"""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.dtype import F8E4M3, F32
from emmy.compiler.graph import Tensor
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.schedule.packing import match_packed_kblock_b
from emmy.compiler.ir.schedule.views import cone_seam
from emmy.compiler.ir.stmt import Assign, Load
from tests.compiler.helpers import requires_cuda

pytest.importorskip("torch")

K16 = "mma_m16n8k16_f16_f32"
K16_BF16 = "mma_m16n8k16_bf16_f32"


def _linear(tmp_path, *, m=16, k=256, n=256, dtype="bfloat16"):
    """``nn.Linear`` written as a block-FP8 checkpoint and read back: the weight's decode cone and
    the grouped dynamic activation in front of the matmul."""
    from emmy.commands.trace import graph_from_code
    from emmy.compiler.loader.synthesize import quantize_and_spell

    code = f"nn.Linear({k},{n},bias=False,dtype=torch.{dtype})(torch.randn({m},{k},dtype=torch.{dtype}))"
    graph, _, bundle = graph_from_code(code)
    ckpt, spelled, marked = quantize_and_spell(graph, bundle, tmp_path / "ckpt", scheme="fp8-block")
    assert (spelled, marked) == (1, 1)
    return graph, bundle, ckpt


def _pins(place: str, stage: str, *, atom: str = K16_BF16, chunk: str = "k8") -> dict:
    return {"PLACE": place, "WORK": "w1x4", "TILE": f"{atom}/f1x1/{chunk}", "STAGE": stage}


def _lower(graph, pins):
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    with pinned_knobs(pins):
        lowered = Pipeline.build(CUDA_PASSES).run(graph, ctx=Context.from_target((12, 0)))
    return [s for node in lowered.nodes.values() if (s := getattr(node.op, "kernel_source", None))]


@pytest.mark.parametrize(
    ("scale_index", "block"),
    [((Var("n"),), None), ((BinaryExpr("/", Var("n"), Literal(128, "int")), BinaryExpr("/", Var("k"), Literal(128, "int"))), 128)],
)
def test_an_fp8_weight_reads_as_one_value_bytes_only_under_a_k_block_scale(scale_index, block):
    """The decode cast of a stored fp8 byte times its scale: a scale that changes along K is a
    128-block reading one byte per element; one constant along K is not — it moves onto the
    epilogue instead."""
    cone = [
        Load(name="in0", input="w_scale", index=scale_index, dtype=None),
        Load(name="in1", input="w_bits", index=(Var("n"), Var("k")), dtype=None),
        Assign(name="v0", op="from_f8e4m3", args=("in1",)),
        Assign(name="v1", op="multiply", args=("in0", "v0")),
    ]
    inputs = {"w_bits": Tensor("w_bits", (256, 256), F8E4M3), "w_scale": Tensor("w_scale", (2, 2) if block else (256,), F32)}
    read = match_packed_kblock_b(cone, "k", inputs)
    if block is None:
        assert read is None
    else:
        assert (read.per_byte, read.block, read.bits.input, read.factor) == (1, 128, "w_bits", "in0")


def test_the_cut_linear_lowers_to_the_fp8_byte_slab(tmp_path):
    """Under a copy stage the weight stages as raw e4m3 bytes in padded rows beside an f32 scale
    slab, and one drain reads both; no 16-bit weight tile and no packed-pair table appear."""
    graph, _, _ = _linear(tmp_path)
    src = "\n".join(_lower(graph, _pins("cut", "d2/smem-async")))
    assert "emmy_mma_load_b_smem_trans_f8s_bf16" in src
    # tile_n = 32 rows of (128 bytes + the 16 B pad), two ring slots; one f32 scale per row.
    assert "__nv_fp8_e4m3 _b_smem[9216]" in src and "float _bs_smem[32]" in src
    assert "EMMY_F4_LUT" not in src


def test_the_fused_quantize_refills_its_group_statistic_every_chunk(tmp_path):
    """Fused into the matmul, the activation's group maximum is a warp reduce at the head of each
    chunk's fill rather than a 128-element scan per slab cell."""
    graph, _, _ = _linear(tmp_path)
    (src,) = _lower(graph, _pins("fuse", "d2/smem-async"))
    loop = src.index("for (int _ks")
    assert "__shared__ float _a_stat_acc0" in src[:loop], "the statistic's row is declared once, ahead of the K loop"
    assert "__shfl_xor_sync" in src[loop:] and "_a_stat_acc0[_sr] = acc0" in src[loop:]


def test_the_seam_reads_the_group_maximum_as_a_per_chunk_statistic(tmp_path):
    """The activation's group maximum varies with K only through its 128-wide group, so the seam
    splits it off the per-cell body as a chunk statistic over that block; the rest stays per cell."""
    from emmy.compiler.context import Context
    from emmy.compiler.ir.tile import TileOp
    from emmy.compiler.pipeline import LOOP_PASSES, Pipeline
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    graph, _, _ = _linear(tmp_path)
    with pinned_knobs({"PLACE": "fuse"}):
        tiled = Pipeline.build([*LOOP_PASSES, "lowering/tile"]).run(graph, ctx=Context.from_target((12, 0)))
    tile = next(node.op for node in tiled.nodes.values() if isinstance(node.op, TileOp))
    node = next(site.node for site in tile.sites if site.node.as_contraction() is not None)
    _pro, cell, _stats, (chunk_pro, chunk_stats, block) = cone_seam(node.operands[0], node.axis, tile.axes)
    assert block == 128 and len(chunk_stats) == 1
    assert chunk_stats[0] not in {name for stmt in cell for name in stmt.defines()}, "the cell reads the statistic, it never computes it"


def _run(graph, bundle, ckpt, pins):
    import torch

    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.loader.safetensors import load_constants_from_safetensors
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    backend = CudaBackend()
    with pinned_knobs(pins):
        compiled = backend.compile(graph)
    data = dict(load_constants_from_safetensors(compiled, str(ckpt)))
    x = bundle[1][0].detach()
    data[compiled.inputs[0]] = x.view(torch.uint16).numpy() if x.dtype == torch.bfloat16 else x.numpy()
    result, _ = backend.run(compiled, input_data=data)
    return np.asarray(result.outputs[compiled.outputs[0]])


@requires_cuda
@pytest.mark.parametrize(
    ("dtype", "atom", "stage"),
    [("bfloat16", K16_BF16, "d2/smem-async"), ("float16", K16, "d2/smem-async"), ("bfloat16", K16_BF16, "d2/smem-tma")],
)
@pytest.mark.xdist_group("cuda")
def test_the_byte_slab_matches_the_compute_fill_bit_for_bit(tmp_path, dtype, atom, stage):
    """The staged fp8 drain and the compute fill hand the tensor cores the same 16-bit values, so
    the matmul agrees to the bit on either copy transport and either fragment dtype."""
    graph, bundle, ckpt = _linear(tmp_path, k=512, n=512, dtype=dtype)
    staged = _run(graph.copy(), bundle, ckpt, _pins("cut", stage, atom=atom))
    filled = _run(graph.copy(), bundle, ckpt, _pins("cut", "d1/smem", atom=atom))
    assert np.array_equal(staged, filled)


@requires_cuda
@pytest.mark.xdist_group("cuda")
def test_the_fused_quantize_matches_the_cut_one_bit_for_bit(tmp_path):
    """The per-chunk group statistic computes the activation the quantize kernel writes, value for
    value, so fusing it into the matmul's fill changes nothing the tensor cores see."""
    graph, bundle, ckpt = _linear(tmp_path, k=512, n=512)
    fused = _run(graph.copy(), bundle, ckpt, _pins("fuse", "d2/smem-async"))
    cut = _run(graph.copy(), bundle, ckpt, _pins("cut", "d2/smem-async"))
    assert np.array_equal(fused, cut)
