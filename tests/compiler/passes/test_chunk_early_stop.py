"""The chunk tier stops a causally masked key stream at the CTA's own diagonal."""

from __future__ import annotations

import pytest

from emmy.commands.trace import graph_from_code
from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.stmt import Assign, Select, SelectBranch
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from emmy.compiler.pipeline.passes.lowering.kernel._atom import _mask_key_end
from emmy.compiler.pipeline.passes.lowering.kernel._tiling import AxisOffset
from emmy.compiler.pipeline.search.pins import pinned_knobs

KEY = Axis(name="a2", extent=Dim(512))
ROW_BLOCK = AxisOffset(atom_dim=16, reg=1, block_var="a1_b", unit_var="a1_u", unit_count=4)  # a 64-row CTA


def _masked_score(keep_op: str, mask_op: str, key_lead: int = 0, row_lead: int = 0) -> list:
    """``s3 = s2 + (0 when key <= row, -1e9 when key > row)`` — the prefix a causal SDPA lifts."""
    key = Var("a2") if not key_lead else BinaryExpr("+", Var("a2"), Literal(key_lead, "int"))
    row = Var("a1") if not row_lead else BinaryExpr("+", Var("a1"), Literal(row_lead, "int"))
    mask = Select(
        name="v1",
        branches=(
            SelectBranch(select=BinaryExpr(keep_op, key, row), value="zero"),
            SelectBranch(select=BinaryExpr(mask_op, key, row), value="fill"),
        ),
    )
    return [mask, Assign(name="s3", op="add", args=("v1", "s2"))]


def _pretty(expr) -> str:
    return expr.pretty() if hasattr(expr, "pretty") else str(expr)


def test_a_causal_mask_stops_the_stream_at_the_block_end() -> None:
    end = _mask_key_end(_masked_score("<=", ">"), "s2", KEY, "a1", ROW_BLOCK)
    assert isinstance(end, TernaryExpr), "the stop is the block end clamped to the extent"
    assert _pretty(end.if_true) == _pretty(ROW_BLOCK.block_end())
    assert _pretty(end.if_false) == "512"


def test_a_lead_on_the_row_side_pushes_the_stop_out_and_a_strict_mask_pulls_it_in() -> None:
    lagged = _mask_key_end(_masked_score("<=", ">", row_lead=32), "s2", KEY, "a1", ROW_BLOCK)
    assert _pretty(lagged.if_true) == _pretty(BinaryExpr("+", ROW_BLOCK.block_end(), Literal(32, "int")))
    strict = _mask_key_end(_masked_score("<", ">=", row_lead=1), "s2", KEY, "a1", ROW_BLOCK)
    assert _pretty(strict.if_true) == _pretty(ROW_BLOCK.block_end()), "key >= row + 1 masks the same keys as key > row"


@pytest.mark.parametrize(
    ("stmts", "why"),
    [
        (_masked_score("<=", ">", key_lead=1), "a negative lead could stop a block before its first chunk"),
        (_masked_score("<", ">="), "key >= row masks the block's own last row"),
        ([Assign(name="s3", op="add", args=("s2", "bias"))], "a bias is not a coordinate mask"),
    ],
)
def test_masks_that_do_not_bound_every_row_from_above_leave_the_stream_whole(stmts, why) -> None:
    assert _mask_key_end(stmts, "s2", KEY, "a1", ROW_BLOCK) is None, why


def _sdpa(causal: bool) -> str:
    q = "torch.randn(1, 1, 64, 32, dtype=torch.float16)"
    return f"F.scaled_dot_product_attention({q}, {q}, {q}{', is_causal=True' if causal else ''})"


@pytest.mark.parametrize("causal", [True, False], ids=["causal", "global"])
def test_the_emitted_chunk_loop_carries_the_stop_only_under_a_causal_mask(causal: bool) -> None:
    graph = graph_from_code(_sdpa(causal))[0]
    pins = {
        "TILE@map.1/twist": "mma_m16n8k16_f16_f32/f1x4/k2",
        "TILE@map.1/twist.1/inner": "mma_m16n8k16_f16_f32/f1x4/k2",
        "STAGE@map.1/twist": "d2/smem-async",
        "STAGE@map.1/twist.1/inner": "d2/smem-async",
        "WORK": "w1x1",
    }
    with pinned_knobs(pins):
        lowered = Pipeline.build(CUDA_PASSES).run(graph, ctx=Context.from_target((12, 0)))
    sources = [node.op.kernel_source for node in lowered.nodes.values() if isinstance(node.op, CudaOp)]
    assert len(sources) == 1 and "__ck" in sources[0], "one fused kernel with a chunk loop"
    assert ("__ck_end" in sources[0]) is causal
