"""The staged slab address split: ``swz(lane + base) = (swz(lane) ^ swz(col)) + row·ldm`` whenever the base's
row is a multiple of the lane's row span and its column of the lane's column step — the XOR swizzle is linear
over bit-disjoint parts. Checked against the swizzle itself over every lane offset a drain or a fill adds."""

import pytest

from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.kernel.ir import swizzle_fn, swizzle_xor, swizzled_slab_index
from emmy.compiler.ir.stmt.base import RenderCtx


def _swz(mode: str):
    shift, mask = swizzle_xor(mode)
    return lambda e: e ^ (((e >> shift) & mask) << 3)


@pytest.mark.parametrize(
    ("mode", "ldm", "lane_rows", "lane_col_mod"),
    [
        ("B128@7", 128, 16, 16),  # the cp.async slab at head width 128: a 16-row x4 drain
        ("B128@7", 128, 8, 16),  # the unpaired transposed-B x2 drain
        ("B128", 128, 16, 16),  # the TMA slab at head width 128
        ("B128", 256, 16, 16),  # the TMA slab at head width 256
        ("B128", 64, 16, 8),  # a 64-element row: the x2.trans drain reads columns 8 apart
        ("B64", 32, 16, 16),
        ("B32", 16, 8, 8),
        ("B128@7", 128, 8, 128),  # a fill stripe: 8 rows, every 16 B chunk of the row
    ],
)
def test_the_split_address_is_the_swizzle_of_the_sum(mode, ldm, lane_rows, lane_col_mod) -> None:
    swz = _swz(mode)
    ctx = RenderCtx()
    for row in range(0, 8 * lane_rows, lane_rows):
        for col in range(0, ldm, lane_col_mod):
            split = swizzled_slab_index(
                mode, ldm, "L", Literal(row, "int"), Literal(col, "int"), ctx, lane_rows=lane_rows, lane_col_mod=lane_col_mod
            )
            assert split is not None, (mode, ldm, row, col)
            for lane_row in range(lane_rows):
                for lane_col in range(0, lane_col_mod, 8) if lane_col_mod > 8 else (0,):
                    lane = lane_row * ldm + lane_col
                    names = {"L": lane, swizzle_fn(mode): swz}
                    assert eval(split, names) == swz(row * ldm + col + lane), (mode, ldm, row, col, lane)


def test_a_base_the_reading_cannot_prove_keeps_the_whole_index() -> None:
    ctx = RenderCtx()
    row, col = Literal(8, "int"), Literal(16, "int")
    assert swizzled_slab_index("B128@7", 128, "L", row, col, ctx, lane_rows=16, lane_col_mod=16) is None  # row 8 inside a 16-row lane span
    assert swizzled_slab_index("B128@7", 128, "L", Literal(16, "int"), Literal(8, "int"), ctx, lane_rows=16, lane_col_mod=16) is None
    assert swizzled_slab_index("NONE", 128, "L", row, col, ctx, lane_rows=16, lane_col_mod=16) is None
    slot = Var("_slot")  # a ring slot's row offset is a multiple of the slab's rows, whatever the slot
    from emmy.compiler.ir.expr import BinaryExpr

    ring = BinaryExpr("+", BinaryExpr("*", slot, Literal(64, "int")), Literal(16, "int"))
    assert swizzled_slab_index("B128@7", 128, "L", ring, col, ctx, lane_rows=16, lane_col_mod=16) is not None
