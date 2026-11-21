"""The chunk tier skips the chunks a carrier's range puts outside the CTA's rows: a causal stream stops at the
CTA's diagonal, a banded one starts at its near edge, and the bound is read off the range where the loop opens."""

from __future__ import annotations

from dataclasses import replace

import pytest

from emmy.commands.trace import graph_from_code
from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expression import BinaryExpression, Literal, TernaryExpression, Var, apply
from emmy.compiler.ir.frontend.ir import SdpaOp
from emmy.compiler.ir.stmt import Loop
from emmy.compiler.pipeline.passes.lowering.kernel._atom import _key_bounds, _outside
from emmy.compiler.pipeline.passes.lowering.kernel._tiling import AxisOffset
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import compile_with_kernels, statements

KEY = Axis(name="a2", extent=Dim(512))
ROW_BLOCK = AxisOffset(atom_dim=16, reg=1, block_var="a1_b", unit_var="a1_u", unit_count=4)  # a 64-row CTA
BK = 32


ROW, ZERO = Var("a1"), Literal(0, "int")


def _past(lead: int) -> BinaryExpression | Var:
    """``a1 + lead``: a bound ``lead`` keys past the row."""
    return ROW if not lead else BinaryExpression("+" if lead > 0 else "-", ROW, Literal(abs(lead), "int"))


CAUSAL = (ZERO, _past(1))  # the keys at or before the row
BAND = (_past(-31), Literal(512, "int"))  # the keys past ``row - 32``


def _bounds(*ranges: tuple, uniform: frozenset[str] = frozenset()) -> tuple:
    """The stream bounds of a carrier that walks what ``ranges`` share."""
    start, end = ranges[0]
    for more in ranges[1:]:
        start, end = apply("maximum", start, more[0]), apply("minimum", end, more[1])
    return _key_bounds(_outside((start, end), KEY, False), KEY, "a1", ROW_BLOCK, BK, uniform)


def _pretty(expression) -> str:
    return str(expression)


def test_a_causal_range_stops_the_stream_at_the_block_end() -> None:
    first, end = _bounds(CAUSAL)
    assert first is None
    assert isinstance(end, TernaryExpression), "the stop is the block end clamped to the extent"
    assert _pretty(end.if_true) == _pretty(ROW_BLOCK.block_end())
    assert _pretty(end.if_false) == "512"


def test_a_band_starts_the_stream_on_the_chunk_holding_the_first_row_near_edge() -> None:
    first, end = _bounds(BAND)
    assert end is None
    edge = BinaryExpression("-", ROW_BLOCK.block_base(), Literal(31, "int"))  # no row folds a key below row - 31
    clamped = TernaryExpression(cond=BinaryExpression(">", edge, Literal(0, "int")), if_true=edge, if_false=Literal(0, "int"))
    assert _pretty(first) == _pretty(BinaryExpression("*", BinaryExpression("/", clamped, Literal(BK, "int")), Literal(BK, "int")))


def test_a_causal_band_bounds_both_ends_whichever_bound_comes_first() -> None:
    first, end = _bounds(CAUSAL, BAND)
    assert first is not None and end is not None
    assert tuple(map(_pretty, (first, end))) == tuple(map(_pretty, _bounds(BAND, CAUSAL)))


def test_an_end_past_the_row_pushes_the_stop_out() -> None:
    _, lagged = _bounds((ZERO, _past(33)))
    assert _pretty(lagged.if_true) == _pretty(BinaryExpression("+", ROW_BLOCK.block_end(), Literal(32, "int")))


@pytest.mark.parametrize(
    ("walked", "why"),
    [
        ((ZERO, _past(0)), "an end at the row leaves out the row's own key: a block could stop before its last row"),
        ((_past(1), Literal(512, "int")), "a start past the row leaves out the row's own key: a block could start after its first row"),
        ((ZERO, Literal(512, "int")), "the whole axis bounds nothing"),
    ],
)
def test_ranges_that_could_take_a_row_own_key_leave_the_stream_whole(walked, why) -> None:
    assert _bounds(walked) == (None, None), why


def _sdpa(causal: bool, heads: int = 1, rows: int = 64) -> str:
    q = f"torch.randn(1, {heads}, {rows}, 32, dtype=torch.float16)"
    return f"F.scaled_dot_product_attention({q}, {q}, {q}{', is_causal=True' if causal else ''})"


def _kernels(code: str, window: int | None = None) -> list:
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
        _, kernels = compile_with_kernels(graph, ctx=Context.from_target((12, 0)))
    return list(kernels.values())


def _chunk_loops(kernel) -> list[Loop]:
    return [loop for loop in statements(kernel, Loop) if loop.axis.name.endswith("__ck")]


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
    kernels = _kernels(code, window)
    assert len(kernels) == 1 and _chunk_loops(kernels[0]), "one fused kernel with a chunk loop"
    loop = _chunk_loops(kernels[0])[0]
    start, end, _ = loop.walk()
    assert (end != loop.axis.extent_expression()) is stops, f"{why}: {end}"
    assert (start != Literal(0, "int")) is starts, f"{why}: {start}"


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
    """Native's causal attention folds the keys at or before ``row + position``, the position a value the kernel
    loads: the stop is the block end plus the position, the sign test kept at runtime, and nothing starts the stream."""
    walked = (ZERO, BinaryExpression("+", BinaryExpression("+", ROW, Var("in0")), Literal(1, "int")))
    first, end = _bounds(walked, uniform=frozenset({"in0"}))
    assert first is None
    shift = BinaryExpression("-", Literal(0, "int"), BinaryExpression("-", Literal(0, "int"), Var("in0")))  # the anchor, ``-in0``, negated
    lead = TernaryExpression(cond=BinaryExpression(">", shift, Literal(0, "int")), if_true=shift, if_false=Literal(0, "int"))
    assert _pretty(end.if_true) == _pretty(BinaryExpression("+", ROW_BLOCK.block_end(), lead))
    assert _bounds(walked) == (None, None), "a value the caller does not name as uniform bounds nothing"
