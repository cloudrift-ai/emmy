"""A scalar outer fold must not silently discard its operand's selected contraction tile."""

from dataclasses import replace
from importlib import import_module

import pytest

from emmy.compiler.context import Context
from emmy.compiler.dtype import F16, F32
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.cuda.ir import CudaOp
from emmy.compiler.ir.frontend.ir import MatmulOp, ReshapeOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.schedule.base import clear_pin_refusals, pin_refusal
from emmy.compiler.ir.tensor.ir import ReduceOp
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.ops import UnbindableProjection
from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, Pipeline
from emmy.compiler.pipeline.fork import iter_leaves
from emmy.compiler.pipeline.passes.lowering.kernel._factor import factorize
from emmy.compiler.pipeline.passes.tile._fromloop import lift_loop_op
from emmy.compiler.pipeline.search.pins import pinned_knobs

classic_forks = import_module("emmy.compiler.pipeline.passes.tile.schedule.040_schedule").classic_forks


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


def _lifted():
    """The block maximum's one fused kernel, lifted and not yet scheduled."""
    loop = next(node.op for node in Pipeline.build(LOOP_PASSES).run(_block_maximum()).nodes.values() if isinstance(node.op, LoopOp))
    return replace(lift_loop_op(loop, name=loop.name), knobs=dict(loop.knobs), inputs=dict(loop.inputs), outputs=dict(loop.outputs))


def _tiled(knobs) -> bool:
    return any(value for key, value in knobs.items() if key.partition("@")[0] == "TILE")


@pytest.mark.parametrize(
    ("gpu_name", "cap"),
    [("NVIDIA GeForce RTX 5090", (12, 0)), ("NVIDIA GeForce RTX 4090", (8, 9)), ("NVIDIA A100-SXM4-40GB", (8, 0))],
)
def test_block_maximum_offers_no_nested_matmul_tile(gpu_name, cap) -> None:
    """The binder lowers the maximum serially and would drop a tile selected at the matmul inside it,
    so the schedule offer leaves such rows out: the greedy compile never ranks one, and a pin naming
    one is refused at the offer with the rule's reason."""
    ctx = Context.from_target(cap, gpu_name=gpu_name)
    tile = _lifted()
    assert tile.nested_tiles, "the matmul is a TILE site below the maximum"

    leaves = list(iter_leaves(classic_forks(tile, tile.name, tile.knobs, ctx)))
    assert leaves
    assert not any(_tiled(leaf.knobs) for leaf in leaves)

    pins = {"TILE": "mma_m16n8k16_f16_f32/f1x2/k2", "WORK": "w1x1", "STAGE": "", "REDUCE": ""}
    clear_pin_refusals()
    with pinned_knobs(pins):
        assert not list(iter_leaves(classic_forks(tile, tile.name, tile.knobs, ctx)))
    assert "not output-tiled" in (pin_refusal("TILE", pins["TILE"]) or "")

    with pinned_knobs({"PLACE": "fuse"}):
        compiled = Pipeline.build(CUDA_PASSES).run(_block_maximum(), ctx=ctx, db=None)
    kernels = [node.op for node in compiled.nodes.values() if isinstance(node.op, CudaOp)]
    assert kernels
    for kernel in kernels:
        if _tiled(kernel.knobs):
            assert "mma" in kernel.kernel_source


def test_binder_refuses_a_nested_tile_that_reaches_it(monkeypatch) -> None:
    """The binder is the backstop for a row the offer's rule misses. With that rule switched off the
    offer holds tiled rows again, and the binder must refuse one instead of lowering the maximum
    serially and dropping the tile."""
    ctx = Context.from_target((12, 0), gpu_name="NVIDIA GeForce RTX 5090")
    tile = _lifted()
    monkeypatch.setattr(TileOp, "nested_tiles", property(lambda _self: ()))
    leaf = next(leaf for leaf in iter_leaves(classic_forks(tile, tile.name, tile.knobs, ctx)) if _tiled(leaf.knobs))
    (realized,) = leaf.expand()
    with pytest.raises(UnbindableProjection, match="serial lowering of reduce cannot realize TILE at reduce.1/inner"):
        factorize(realized, root=None)
