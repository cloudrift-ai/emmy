"""The chunk tier skips the chunks a coordinate mask puts outside the CTA's rows: a causal stream stops at the
CTA's diagonal, a banded one starts at its near edge, and the bound is read off the mask where the loop opens."""

from __future__ import annotations

from dataclasses import replace

import pytest

from emmy.commands.trace import graph_from_code
from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.frontend.ir import SdpaOp
from emmy.compiler.ir.stmt import Assign, Select, SelectBranch
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from emmy.compiler.pipeline.passes.lowering.kernel._atom import _mask_key_bounds
from emmy.compiler.pipeline.passes.lowering.kernel._tiling import AxisOffset
from emmy.compiler.pipeline.search.pins import pinned_knobs

KEY = Axis(name="a2", extent=Dim(512))
ROW_BLOCK = AxisOffset(atom_dim=16, reg=1, block_var="a1_b", unit_var="a1_u", unit_count=4)  # a 64-row CTA
BK = 32


def _mask(name: str, keep_op: str, mask_op: str, key_lead: int = 0, row_lead: int = 0) -> Select:
    """``zero when key <keep_op> row + lead, fill otherwise`` — one coordinate mask as the SDPA decomposition spells it."""
    key = Var("a2") if not key_lead else BinaryExpr("+", Var("a2"), Literal(key_lead, "int"))
    row = Var("a1") if not row_lead else BinaryExpr("+" if row_lead > 0 else "-", Var("a1"), Literal(abs(row_lead), "int"))
    return Select(
        name=name,
        branches=(
            SelectBranch(select=BinaryExpr(keep_op, key, row), value="zero"),
            SelectBranch(select=BinaryExpr(mask_op, key, row), value="fill"),
        ),
    )


def _masked_score(*masks: Select) -> list:
    """``s3 = s2 + v1``, ``s4 = s3 + v2`` … — the prefix an SDPA lifts, one add per mask."""
    stmts: list = []
    held = "s2"
    for index, mask in enumerate(masks, start=3):
        stmts += [mask, Assign(name=f"s{index}", op="add", args=(mask.name, held))]
        held = f"s{index}"
    return stmts


CAUSAL = _mask("v1", "<=", ">")  # keep key <= row
BAND = _mask("v2", ">", "<=", row_lead=-32)  # keep key > row - 32


def _bounds(stmts) -> tuple:
    return _mask_key_bounds(stmts, "s2", KEY, "a1", ROW_BLOCK, BK)


def _pretty(expr) -> str:
    return expr.pretty() if hasattr(expr, "pretty") else str(expr)


def test_a_causal_mask_stops_the_stream_at_the_block_end() -> None:
    first, end = _bounds(_masked_score(CAUSAL))
    assert first is None
    assert isinstance(end, TernaryExpr), "the stop is the block end clamped to the extent"
    assert _pretty(end.if_true) == _pretty(ROW_BLOCK.block_end())
    assert _pretty(end.if_false) == "512"


def test_a_band_starts_the_stream_on_the_chunk_holding_the_first_row_near_edge() -> None:
    first, end = _bounds(_masked_score(BAND))
    assert end is None
    edge = BinaryExpr("-", ROW_BLOCK.block_base(), Literal(31, "int"))  # key <= row - 32 masks every key below row - 31
    clamped = TernaryExpr(cond=BinaryExpr(">", edge, Literal(0, "int")), if_true=edge, if_false=Literal(0, "int"))
    assert _pretty(first) == _pretty(BinaryExpr("*", BinaryExpr("/", clamped, Literal(BK, "int")), Literal(BK, "int")))


def test_a_causal_band_bounds_both_ends_whichever_mask_comes_first() -> None:
    first, end = _bounds(_masked_score(CAUSAL, BAND))
    assert first is not None and end is not None
    assert tuple(map(_pretty, (first, end))) == tuple(map(_pretty, _bounds(_masked_score(BAND, CAUSAL))))


def test_a_lead_on_the_row_side_pushes_the_stop_out_and_a_strict_mask_pulls_it_in() -> None:
    _, lagged = _bounds(_masked_score(_mask("v1", "<=", ">", row_lead=32)))
    assert _pretty(lagged.if_true) == _pretty(BinaryExpr("+", ROW_BLOCK.block_end(), Literal(32, "int")))
    _, strict = _bounds(_masked_score(_mask("v1", "<", ">=", row_lead=1)))
    assert _pretty(strict.if_true) == _pretty(ROW_BLOCK.block_end()), "key >= row + 1 masks the same keys as key > row"


@pytest.mark.parametrize(
    ("stmts", "why"),
    [
        (
            _masked_score(_mask("v1", "<=", ">", key_lead=1)),
            "key + 1 > row takes the row's own key: a block could stop before its last row",
        ),
        (_masked_score(_mask("v1", "<", ">=")), "key >= row masks the block's own last row"),
        (
            _masked_score(_mask("v1", ">=", "<", row_lead=1)),
            "key < row + 1 takes the row's own key: a block could start after its first row",
        ),
        (_masked_score(_mask("v1", ">", "<=")), "key <= row masks the row's own key from below"),
        ([Assign(name="s3", op="add", args=("s2", "bias"))], "a bias is not a coordinate mask"),
    ],
)
def test_masks_that_could_take_a_row_own_key_leave_the_stream_whole(stmts, why) -> None:
    assert _bounds(stmts) == (None, None), why


def _sdpa(causal: bool, heads: int = 1, rows: int = 64) -> str:
    q = f"torch.randn(1, {heads}, {rows}, 32, dtype=torch.float16)"
    return f"F.scaled_dot_product_attention({q}, {q}, {q}{', is_causal=True' if causal else ''})"


def _sources(code: str, window: int | None = None) -> list[str]:
    graph = graph_from_code(code)[0]
    if window is not None:  # the HF-wrapper stamp path: F.scaled_dot_product_attention has no window arg to trace
        for node in graph.nodes.values():
            if isinstance(node.op, SdpaOp):
                node.op = replace(node.op, sliding_window=window)
    pins = {
        "TILE@map.1/twist": "mma_m16n8k16_f16_f32/f1x4/k2",
        "TILE@map.1/twist.1/inner": "mma_m16n8k16_f16_f32/f1x4/k2",
        "STAGE@map.1/twist": "d2/smem-async",
        "STAGE@map.1/twist.1/inner": "d2/smem-async",
        "WORK": "w1x1",
    }
    with pinned_knobs(pins):
        lowered = Pipeline.build(CUDA_PASSES).run(graph, ctx=Context.from_target((12, 0)))
    return [node.op.kernel_source for node in lowered.nodes.values() if isinstance(node.op, CudaOp)]


def _chunk_loop(source: str) -> str:
    return next(line.strip() for line in source.splitlines() if "for (int " in line and "__ck" in line)


@pytest.mark.parametrize(
    ("code", "window", "starts", "stops", "why"),
    [
        (
            _sdpa(True, heads=8, rows=512),
            None,
            False,
            True,
            "256 CTAs of 16 rows over 8 heads, several waves: the stop shortens the kernel",
        ),
        (_sdpa(True), None, False, False, "4 CTAs: one wave, the kernel takes as long as its longest CTA either way"),
        (_sdpa(False, heads=8, rows=512), None, False, False, "no mask, nothing to stop at"),
        (_sdpa(True, heads=8, rows=512), 64, True, True, "a causal band is skipped at both ends"),
        (_sdpa(True, rows=128), 32, True, True, "8 CTAs, one wave: bounded at both ends every CTA shortens"),
    ],
    ids=["causal-multi-wave", "causal-one-wave", "global", "band-multi-wave", "band-one-wave"],
)
def test_the_emitted_chunk_loop_carries_the_bounds_only_where_they_can_shorten_the_kernel(code, window, starts, stops, why) -> None:
    sources = _sources(code, window)
    assert len(sources) == 1 and "__ck" in sources[0], "one fused kernel with a chunk loop"
    loop = _chunk_loop(sources[0])
    assert ("__ck_end" in loop) is stops, f"{why}: {loop}"
    assert ("__ck = 0" not in loop) is starts, f"{why}: {loop}"


def test_the_launch_count_is_the_lead_extents_times_the_block_counts() -> None:
    from emmy.compiler.ir.schedule.choices import Side
    from emmy.compiler.pipeline.passes.lowering.kernel._factor import launch_ctas

    m = Side(axis=Axis(name="a1", extent=Dim(512)), tile=64, units=4, reg=1, block="a1_b", unit="a1_u")
    n = Side(axis=Axis(name="a5", extent=Dim(256)), tile=256, units=1, reg=32, block="a5_b", unit="a5_u")
    heads = Axis(name="a0", extent=Dim(16))
    assert launch_ctas((heads,), (m, n)) == 16 * 8
    assert launch_ctas((), (m, None)) == 8
    ragged = Side(axis=Axis(name="a1", extent=Dim(100)), tile=64, units=4, reg=1, block="a1_b", unit="a1_u")
    assert launch_ctas((), (ragged, None)) == 2, "a partial block still launches"
    assert launch_ctas((Axis(name="s", extent=Dim("seq_len")),), (m, n)) is None, "a symbolic extent is unknown"
