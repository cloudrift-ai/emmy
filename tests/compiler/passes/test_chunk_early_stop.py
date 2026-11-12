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
from emmy.compiler.ir.expression import BinaryExpression, Literal, TernaryExpression, Var
from emmy.compiler.ir.frontend.ir import SdpaOp
from emmy.compiler.ir.stmt import Select, parse_body
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from emmy.compiler.pipeline.passes.lowering.kernel._atom import _mask_key_bounds
from emmy.compiler.pipeline.passes.lowering.kernel._tiling import AxisOffset
from emmy.compiler.pipeline.search.pins import pinned_knobs

KEY = Axis(name="a2", extent=Dim(512))
ROW_BLOCK = AxisOffset(atom_dim=16, reg=1, block_var="a1_b", unit_var="a1_u", unit_count=4)  # a 64-row CTA
BK = 32


def _mask(name: str, keep_op: str, mask_op: str, key_lead: int = 0, row_lead: int = 0) -> Select:
    """``zero when key <keep_op> row + lead, fill otherwise`` — one coordinate mask as the SDPA decomposition spells it."""
    key = "a2" if not key_lead else f"a2 + {key_lead}"
    row = "a1" if not row_lead else f"a1 {'+' if row_lead > 0 else '-'} {abs(row_lead)}"
    [select] = parse_body(f"{name} = select:\n    `zero` when {key} {keep_op} {row}\n    fill when {key} {mask_op} {row}")
    return select


def _masked_score(*masks: Select) -> list:
    """``s3 = s2 + v1``, ``s4 = s3 + v2`` … — the prefix an SDPA lifts, one add per mask."""
    lines: list[str] = []
    held = "s2"
    for index, mask in enumerate(masks, start=3):
        lines += [str(mask), f"s{index} = add({mask.name}, {held})"]
        held = f"s{index}"
    return list(parse_body("\n".join(lines)))


CAUSAL = _mask("v1", "<=", ">")  # keep key <= row
BAND = _mask("v2", ">", "<=", row_lead=-32)  # keep key > row - 32


def _bounds(stmts) -> tuple:
    return _mask_key_bounds(stmts, "s2", KEY, "a1", ROW_BLOCK, BK)


def _pretty(expression) -> str:
    return str(expression)


def test_a_causal_mask_stops_the_stream_at_the_block_end() -> None:
    first, end = _bounds(_masked_score(CAUSAL))
    assert first is None
    assert isinstance(end, TernaryExpression), "the stop is the block end clamped to the extent"
    assert _pretty(end.if_true) == _pretty(ROW_BLOCK.block_end())
    assert _pretty(end.if_false) == "512"


def test_a_band_starts_the_stream_on_the_chunk_holding_the_first_row_near_edge() -> None:
    first, end = _bounds(_masked_score(BAND))
    assert end is None
    edge = BinaryExpression("-", ROW_BLOCK.block_base(), Literal(31, "int"))  # key <= row - 32 masks every key below row - 31
    clamped = TernaryExpression(cond=BinaryExpression(">", edge, Literal(0, "int")), if_true=edge, if_false=Literal(0, "int"))
    assert _pretty(first) == _pretty(BinaryExpression("*", BinaryExpression("/", clamped, Literal(BK, "int")), Literal(BK, "int")))


def test_a_causal_band_bounds_both_ends_whichever_mask_comes_first() -> None:
    first, end = _bounds(_masked_score(CAUSAL, BAND))
    assert first is not None and end is not None
    assert tuple(map(_pretty, (first, end))) == tuple(map(_pretty, _bounds(_masked_score(BAND, CAUSAL))))


def test_a_comparison_canonicalized_by_swapping_operands_still_bounds_the_stream() -> None:
    [causal] = parse_body("v1 = select:\n    `zero` when a2 <= a1\n    fill when a1 < a2")
    first, end = _bounds(_masked_score(causal))
    assert first is None and end is not None

    [band] = parse_body("v1 = select:\n    `zero` when a1 - 32 < a2\n    fill when a2 <= a1 - 32")
    first, end = _bounds(_masked_score(band))
    assert first is not None and end is None


def test_a_lead_on_the_row_side_pushes_the_stop_out_and_a_strict_mask_pulls_it_in() -> None:
    _, lagged = _bounds(_masked_score(_mask("v1", "<=", ">", row_lead=32)))
    assert _pretty(lagged.if_true) == _pretty(BinaryExpression("+", ROW_BLOCK.block_end(), Literal(32, "int")))
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
        (list(parse_body("s3 = add(s2, bias)")), "a bias is not a coordinate mask"),
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
        "PLACE": "fuse",  # one fused kernel's chunk loop is what these tests read; the placement prior would decide the cut
        "TILE@map.1/twist": "mma_m16n8k16_f16_f32/f1x4/k2",
        "TILE@map.1/twist.1/inner": "mma_m16n8k16_f16_f32/f1x4/k2",
        "STAGE@map.1/twist": "d2/smem-async",
        "STAGE@map.1/twist.1/inner": "",
        "REDUCE": "",
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


def test_a_position_read_from_memory_bounds_the_stream_at_runtime() -> None:
    """Native's causal mask compares ``key - row`` against a position the kernel loads: the anchor is a uniform value,
    so the stop is the block end plus the position, the sign test kept at runtime, and nothing starts the stream."""
    [select] = parse_body("v1 = select:\n    `zero` when -1 * a1 + a2 <= in0\n    fill when -1 * a1 + a2 > in0")
    first, end = _mask_key_bounds(_masked_score(select), "s2", KEY, "a1", ROW_BLOCK, BK, frozenset({"in0"}))
    assert first is None
    shift = BinaryExpression("-", Literal(0, "int"), BinaryExpression("-", Literal(0, "int"), Var("in0")))  # the anchor, ``-in0``, negated
    lead = TernaryExpression(cond=BinaryExpression(">", shift, Literal(0, "int")), if_true=shift, if_false=Literal(0, "int"))
    assert _pretty(end.if_true) == _pretty(BinaryExpression("+", ROW_BLOCK.block_end(), lead))
    assert _bounds(_masked_score(select)) == (None, None), "a value the caller does not name as uniform bounds nothing"


def test_a_stop_bound_under_a_name_is_read_not_declared_again() -> None:
    """The staged chunk loop binds its early stop as ``<axis>_end`` ahead of the loop. The loop reads that value: a
    for-init local of the same spelling, initialised from the name, would read itself."""
    from emmy.compiler.ir.render import RenderCtx, render_stmt
    from emmy.compiler.ir.stmt import Body, StridedLoop

    step = Literal(64, "int")
    loop = StridedLoop(axis=KEY, start=Literal(0, "int"), step=step, body=Body(()), unroll=False, end=Var("a2_end"))
    assert "for (int a2 = 0; a2 < a2_end; a2 += 64) {" in [line.strip() for line in render_stmt(loop, RenderCtx())]
    computed = loop.replace(end=BinaryExpression("+", Var("row"), step))
    assert any("a2_end = row + 64" in line for line in render_stmt(computed, RenderCtx()))
