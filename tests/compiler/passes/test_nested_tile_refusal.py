"""A schedule must not stamp a TILE or a STAGE below a kernel root that the binder would silently discard."""

from dataclasses import replace
from importlib import import_module

import pytest
from frozendict import frozendict

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16, F32
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.cuda.ir import CudaOp
from emmy.compiler.ir.frontend.ir import MatmulOp, ReshapeOp, SdpaOp
from emmy.compiler.ir.kernel.ir import MmaSyncPtx, WgmmaMma
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.schedule.base import clear_pin_refusals, pin_refusal
from emmy.compiler.ir.tensor.ir import ReduceOp
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.ops import UnbindableProjection
from emmy.compiler.pipeline import LOOP_PASSES, TILE_PASSES, Pipeline
from emmy.compiler.pipeline.fork import iter_leaves
from emmy.compiler.pipeline.passes.lowering.kernel._factor import factorize
from emmy.compiler.pipeline.passes.tile._fromloop import lift_loop_op
from emmy.compiler.pipeline.search.pins import pinned_knobs
from emmy.compiler.pipeline.search.pool import PoolSample
from emmy.compiler.pipeline.search.ranking import enumerate_graph
from tests.compiler.helpers import compile_with_kernels, statements

classic_forks = import_module("emmy.compiler.pipeline.passes.tile.schedule.040_schedule").classic_forks

_CARDS = [("NVIDIA GeForce RTX 5090", (12, 0)), ("NVIDIA GeForce RTX 4090", (8, 9)), ("NVIDIA A100-SXM4-40GB", (8, 0))]


def _block_maximum() -> Graph:
    """The maximum over each group of 16 columns of a matmul: one fused kernel whose root is the
    maximum, with the contraction nested below it."""
    graph = Graph()
    for name in ("a", "b"):
        graph.add_node(InputOp(), [], Tensor(name, (128, 128), F16), node_id=name)
    graph.add_node(MatmulOp(), ["a", "b"], Tensor("mm", (128, 128), F32), node_id="mm")
    graph.add_node(ReshapeOp((128, 8, 16)), ["mm"], Tensor("blocks", (128, 8, 16), F32), node_id="blocks")
    graph.add_node(ReduceOp("maximum", -1), ["blocks"], Tensor("out", (128, 8, 1), F32), node_id="out")
    graph.inputs, graph.outputs = ["a", "b"], ["out"]
    return graph


def _back_to_back() -> Graph:
    """``(a @ b) @ c``: one fused kernel whose root is the second matmul, with the first nested in its A cone."""
    graph = Graph()
    for name, shape in (("a", (128, 64)), ("b", (64, 128)), ("c", (128, 128))):
        graph.add_node(InputOp(), [], Tensor(name, shape, F16), node_id=name)
    graph.add_node(MatmulOp(), ["a", "b"], Tensor("mm", (128, 128), F16), node_id="mm")
    graph.add_node(MatmulOp(), ["mm", "c"], Tensor("out", (128, 128), F32), node_id="out")
    graph.inputs, graph.outputs = ["a", "b", "c"], ["out"]
    return graph


def _attention(heads: int = 2, seq: int = 64, dim: int = 32) -> Graph:
    graph = Graph()
    shape = (Dim(1), Dim(heads), Dim(seq), Dim(dim))
    for name in ("q", "k", "v"):
        graph.add_node(InputOp(), [], Tensor(name, shape, dtype="f16"), node_id=name)
    graph.add_node(SdpaOp(is_causal=False), ["q", "k", "v"], Tensor("o", shape, dtype="f16"), node_id="o")
    graph.inputs, graph.outputs = ["q", "k", "v"], ["o"]
    return graph


#: The attention's chunked carrier and its score on the tensor core, the score's transport left to the offer.
_PAIRED = {
    "WORK": "w1x1",
    "RASTER": "",
    "REDUCE": "",
    "TILE@map.1/twist": "mma_m16n8k16_f16_f32/f1x1",
    "TILE@map.1/twist.1/inner": "mma_m16n8k16_f16_f32/f1x2",
    "STAGE@map.1/twist": "d1/smem-async",
}
#: The same pair on the warp-group atoms, whose operands a shared-memory descriptor reads: the score's are
#: the carrier's slabs.
_PAIRED_WGMMA = {
    "WORK": "w4x1",
    "RASTER": "",
    "REDUCE": "",
    "TILE@map.1/twist": "wgmma_m64n128k16_f16_f32/f1x16/k4",
    "TILE@map.1/twist.1/inner": "wgmma_m64n64k16_f16_f32/f1x8/k4",
    "STAGE@map.1/twist": "d2/smem-async",
}


def _lifted(graph: Graph) -> TileOp:
    """The graph's one fused kernel, lifted and not yet scheduled."""
    with pinned_knobs({"PLACE": "fuse"}):
        loop = next(node.op for node in Pipeline.build(LOOP_PASSES).run(graph).nodes.values() if isinstance(node.op, LoopOp))
    return replace(lift_loop_op(loop), knobs=dict(loop.knobs), inputs=dict(loop.inputs), outputs=dict(loop.outputs))


def _tiled(knobs) -> int:
    return sum(1 for key, value in knobs.items() if key.partition("@")[0] == "TILE" and value)


@pytest.mark.parametrize(("gpu_name", "cap"), _CARDS)
def test_block_maximum_offers_no_nested_matmul_tile(gpu_name, cap) -> None:
    """The binder lowers the maximum serially and would drop a tile selected at the matmul inside it,
    so the schedule offer leaves such rows out: the greedy compile never ranks one, and a pin naming
    one is refused at the offer with the rule's reason."""
    ctx = Context.from_target(cap, gpu_name=gpu_name)
    tile = _lifted(_block_maximum())
    assert tile.nested_tiles, "the matmul is a TILE site below the maximum"

    leaves = list(iter_leaves(classic_forks(tile, tile.knobs, ctx)))
    assert leaves
    assert not any(_tiled(leaf.knobs) for leaf in leaves)

    pins = {"TILE": "mma_m16n8k16_f16_f32/f1x2/k2", "WORK": "w1x1", "STAGE": "", "REDUCE": ""}
    clear_pin_refusals()
    with pinned_knobs(pins):
        assert not list(iter_leaves(classic_forks(tile, tile.knobs, ctx)))
    assert "does not fold it" in (pin_refusal("TILE", pins["TILE"]) or "")

    with pinned_knobs({"PLACE": "fuse"}):
        compiled, bodies = compile_with_kernels(_block_maximum(), ctx=ctx, db=None)
    kernels = {node_id: node.op for node_id, node in compiled.nodes.items() if isinstance(node.op, CudaOp)}
    assert kernels
    for node_id, kernel in kernels.items():
        if _tiled(kernel.knobs):
            assert statements(bodies[node_id], MmaSyncPtx, WgmmaMma)


@pytest.mark.parametrize(("gpu_name", "cap"), _CARDS)
def test_an_output_tiled_root_offers_no_tile_on_the_matmul_in_its_cone(gpu_name, cap) -> None:
    """An output-tiled root evaluates its A cone per cell in the fill, so a tile or a transport at the
    matmul nested there was stamped on the row and realized as a serial per-cell loop: the row read
    a schedule that never ran. Only the root is tiled now, and a pin on the nested matmul is refused
    at the offer with the rule's reason."""
    ctx = Context.from_target(cap, gpu_name=gpu_name)
    tile = _lifted(_back_to_back())
    (nested,) = (site for _root, site in tile.nested_tiles)
    tile_key, stage_key = (f"{family}@{tile.sites[nested].path}" for family in ("TILE", "STAGE"))

    leaves = iter_leaves(classic_forks(tile, tile.knobs, ctx))
    assert not any(leaf.knobs[tile_key] or leaf.knobs[stage_key] for leaf in leaves)
    assert any(_tiled(leaf.knobs) for leaf in iter_leaves(classic_forks(tile, tile.knobs, ctx))), "the root still tiles"

    clear_pin_refusals()
    with pinned_knobs({tile_key: "mma_m16n8k16_f16_f32/f1x2"}):
        assert not list(iter_leaves(classic_forks(tile, tile.knobs, ctx)))
    assert "does not fold it" in (pin_refusal(tile_key, "mma_m16n8k16_f16_f32/f1x2") or "")


def test_binder_refuses_a_nested_tile_that_reaches_it(monkeypatch) -> None:
    """The binder is the backstop for a row the offer's rule misses. With that rule switched off the
    offer holds tiled rows again, and the binder must refuse one instead of lowering the maximum
    serially and dropping the tile."""
    ctx = Context.from_target((12, 0), gpu_name="NVIDIA GeForce RTX 5090")
    tile = _lifted(_block_maximum())
    monkeypatch.setattr(TileOp, "nested_tiles", property(lambda _self: frozendict()))
    leaf = next(leaf for leaf in iter_leaves(classic_forks(tile, tile.knobs, ctx)) if _tiled(leaf.knobs))
    (realized,) = leaf.expand()
    with pytest.raises(UnbindableProjection, match="serial lowering of reduce cannot realize TILE at reduce.1/inner"):
        factorize(realized, root=None)


def test_binder_refuses_a_tile_in_an_output_tiled_roots_cone(monkeypatch) -> None:
    """The same backstop on the tiled arm: a row tiling both matmuls used to lower the nested one as
    a serial per-cell loop in the root's fill, byte for byte the kernel of the row that tiles only
    the root."""
    ctx = Context.from_target((8, 9), gpu_name="NVIDIA GeForce RTX 4090")
    tile = _lifted(_back_to_back())
    monkeypatch.setattr(TileOp, "nested_tiles", property(lambda _self: frozendict()))
    leaf = next(leaf for leaf in iter_leaves(classic_forks(tile, tile.knobs, ctx)) if _tiled(leaf.knobs) == 2)
    (realized,) = leaf.expand()
    with pytest.raises(UnbindableProjection, match="tiled lowering of inner cannot realize TILE at inner.1/map.1/inner"):
        factorize(realized, root=None)


@pytest.mark.parametrize(
    ("cap", "pins", "graph"),
    [((8, 9), _PAIRED, _attention), ((9, 0), _PAIRED_WGMMA, lambda: _attention(heads=8, seq=512, dim=128))],
    ids=["mma", "wgmma"],
)
def test_a_chunk_score_keeps_its_tile_and_takes_no_transport(cap, pins, graph) -> None:
    """The chunk tier is the one place the binder folds a nested tile, the score's, and it stages the
    score's operands through the carrier's transport. The paired row is still offered, a warp-group
    score included; a transport at the score is not, since the kernel it built was the one without it."""
    ctx = replace(Context.from_target(cap), pool_sample=PoolSample(rows=64, seed=0))
    with pinned_knobs(pins):
        rows = enumerate_graph(graph(), ctx).rows
    assert rows
    assert {row["STAGE@map.1/twist.1/inner"] for row in rows} == {""}


def test_binder_refuses_a_transport_below_the_root(monkeypatch) -> None:
    """With the offer's rule switched off the score's transport is offered again, and the binder
    refuses it instead of building the kernel without it."""
    ctx = Context.from_target((8, 9), gpu_name="NVIDIA GeForce RTX 4090")
    monkeypatch.setattr(TileOp, "nested_tiles", property(lambda _self: frozendict()))
    with pinned_knobs({**_PAIRED, "STAGE@map.1/twist.1/inner": "d2/smem-async"}):
        (tile,) = (node.op for node in Pipeline.build(TILE_PASSES).run(_attention(), ctx=ctx).nodes.values() if isinstance(node.op, TileOp))
    with pytest.raises(UnbindableProjection, match="cannot realize STAGE at map.1/twist.1/inner"):
        factorize(tile, root=None)
