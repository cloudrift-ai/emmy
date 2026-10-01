"""A contraction whose B slab reads the row it is contracted against is no mma tile site — and a
gmem-direct fragment load never leaks the sibling output axis's unsplit name.

The DeepSeek-V4-Flash ``post4096`` twin's matmul piece reads ``out[a0, m, n] = Σ_k A[a0, m, k] · B[a0, m, k, n]``:
a batch of per-row matvecs whose B carries a live ``m`` stride (a placement cut materialized the frontend's
per-row broadcast). ``TileOp.contracts`` declines such a site — an mma B fragment is ``B[k, n]``, one tile
for every row — but the schedule handed the site the tile catalog anyway, because the catalog keyed off
``Fold.tiles_whole`` while ``contracts`` decided only the ``TILE`` key spelling. Placed on the grid's trailing
pair, the mma emission then wrote B's address with the unsplit row axis, which the kernel no longer defines
after the tile split (nvcc on sm_70: ``identifier "a26_1" is undefined``). The catalog now follows the one
reading; the site takes the reduction domain, where the row is an ordinary grid coordinate.

The emission half mirrors the staged fill's sibling binding (``test_staged_fill_sibling_axis``): a value-dead
reshape residue of the sibling axis passes ``contracts`` and still appears SYNTACTICALLY in the gmem index, so
the gmem-direct leaves bind it to the sibling's block base the way every slab fill does.
"""

from __future__ import annotations

import pytest

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Expr, Literal, Var
from emmy.compiler.ir.kernel.ir import LdmatrixLoad
from emmy.compiler.ir.schedule import Placement, Tile, Work
from emmy.compiler.ir.schedule.classic import ClassicProblem
from emmy.compiler.ir.stmt import Body, Load, Write
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline.passes.lowering.kernel._atom import reduce_codegen, store_sink
from emmy.compiler.pipeline.passes.lowering.kernel._tiling import atomize, grid_tile, register_tile, unit_tile
from tests.compiler.terms import contraction

_A0, _M, _N, _K = Axis("a0", 2), Axis("m", 4), Axis("n", 64), Axis("k", 4)


def _lit(value: int) -> Literal:
    return Literal(value, "int")


def _row_residue(sibling: Expr, own: Expr) -> Expr:
    """``((s·1024 + o) / 128 % 8) · 128 + (s·1024 + o) % 128`` — value-equal to ``o`` for ``o < 1024``."""
    flat = sibling * _lit(1024) + own
    return (flat / _lit(128)) % _lit(8) * _lit(128) + flat % _lit(128)


def _free(exprs) -> set[str]:
    return set().union(*(set(e.free_vars()) for e in exprs))


def _fragment_loads(a: Load, b: Load, atom: str, inputs=None) -> dict[str, LdmatrixLoad]:
    """The gmem-direct operand fragment loads the mma atom emits for ``a ⊗ b`` on a 1×1 warp tile — sealed
    through ``grid_tile`` the way ``_factor._bind``'s output-tiled arm seals it. ``inputs`` gives the
    operand tensors their shapes, which is what lets the leaf read a stride rather than a dim position."""
    c = contraction(_K, a, (b, "acc"))
    plan = Tile.parse(f"{atom}/f1x1", Work.parse("w1x1")).at(_M, _N)
    axes = (_M, _N, _K)
    state, region = reduce_codegen(c, plan, inputs=inputs, k_axis=_K, axes=axes)
    epilogue = Body((Write(output="out", index=(Var("m"), Var("n")), value=c.combine.results[0]),))
    tile = grid_tile(
        unit_tile(register_tile(atomize(plan.atom.shape[:2]), plan.mn), plan.mn),
        mn=plan.mn,
        block_threads=plan.launch_threads,
        lanes=plan.atom.lanes,
        state_decls=state,
        reduce_region=region,
        store=store_sink(c, plan, epilogue, k_axis=_K, axes=axes),
    )
    return {s.role: s for s in Body(tile.body).iter() if isinstance(s, LdmatrixLoad)}


@pytest.mark.parametrize("cc", [(7, 0), (12, 0)])
def test_row_varying_b_slab_takes_the_reduction_domain(cc) -> None:
    """The reproducer's shape: B reads the row axis with a live stride, so the site is no tile site and its
    domain carries no tiled plan and no transport — the tile catalog follows ``contracts``."""
    a = Load(name="a", input="A", index=(Var("a0"), Var("m"), Var("k")))
    b = Load(name="b", input="B", index=(Var("a0"), Var("m"), Var("k"), Var("n")))
    tile = TileOp(op=contraction(_K, a, (b, "acc")), place=Placement(free=(_A0, _M, _N)), axes=(_A0, _M, _N, _K))
    site = tile.node_id(tile.op)
    assert not tile.contracts(site)
    offers = ClassicProblem(tile, Context.from_target(cc))
    assert all(not choice.tile.is_tiled for choice in offers.node_site(site).nodes)
    assert all(choice.stage.is_direct for choice in offers.node_site(site).edges)


@pytest.mark.parametrize("atom", ["mma_m8n8k4_f16_f32", "mma_m16n8k16_f16_f32"])
def test_gmem_direct_b_fragment_binds_the_row_residue(atom) -> None:
    """The merged-weight column residue: a value-dead ``m`` in B's index is spelled through the row's block
    base, never the bare axis name the tiled kernel does not define."""
    a = Load(name="a", input="A", index=(Var("m"), Var("k")))
    b = Load(name="b", input="B", index=(Var("k"), _row_residue(Var("m"), Var("n"))))
    load = _fragment_loads(a, b, atom)["b"]
    assert "m" not in _free(load.src_index) and "m_b" in _free(load.src_index)


@pytest.mark.parametrize("atom", ["mma_m8n8k4_f16_f32", "mma_m16n8k16_f16_f32"])
def test_gmem_direct_a_fragment_binds_the_column_residue(atom) -> None:
    """The mirror on A: a value-dead ``n`` in A's index is spelled through the column's block base."""
    a = Load(name="a", input="A", index=(Var("m"), _row_residue(Var("n"), Var("k")) * _lit(0) + Var("k")))
    b = Load(name="b", input="B", index=(Var("k"), Var("n")))
    load = _fragment_loads(a, b, atom)["a"]
    assert "n" not in _free(load.src_index) and "n_b" in _free(load.src_index)


def _packed(k_extent: int, n_extent: int) -> tuple[Load, Load, dict]:
    """``out[m, n] = Σ_k a[m, k] · x[m, k * n_extent + n]`` — B's own coordinate and the reduction
    coordinate packed into ONE tensor dim, the shape DeepSeek-V4's hyper-connection mixing reads
    (``hidden[stream * 4096 + channel]``, 24 streams over one 16384-wide row)."""
    a = Load(name="a", input="A", index=(Var("m"), Var("k")))
    b = Load(name="b", input="X", index=(Var("m"), Var("k") * _lit(n_extent) + Var("n")))
    inputs = {
        "A": Tensor("A", (Dim(4), Dim(k_extent)), "f16"),
        "X": Tensor("X", (Dim(4), Dim(k_extent * n_extent)), "f16"),
    }
    return a, b, inputs


@pytest.mark.parametrize("atom", ["mma_m8n8k4_f16_f32", "mma_m16n8k16_f16_f32"])
def test_a_packed_operand_dim_reads_its_stride_not_its_position(atom) -> None:
    """One dim holding both coordinates is K-major at the reduction coordinate's own stride.

    The dim-position reading calls it N-major (its LAST dim holds the reduction axis) at the
    tensor's row stride, and the fragment then reads the wrong elements with nothing raised — the
    DeepSeek-V4 V100 pre-attention kernel, wrong by up to 64x against eager on 3594 of 4096 outputs.
    """
    a, b, inputs = _packed(k_extent=4, n_extent=64)
    load = _fragment_loads(a, b, atom, inputs)["b"]
    assert not load.b_trans
    assert load.ldm == 64


@pytest.mark.parametrize("atom", ["mma_m8n8k4_f16_f32", "mma_m16n8k16_f16_f32"])
def test_separate_dims_keep_the_answer_the_dim_positions_gave(atom) -> None:
    """Where the dims separate the coordinates, the stride reading says what the positions said: a
    ``b[n, k]`` operand is N-major with the row stride its trailing extent gives."""
    a = Load(name="a", input="A", index=(Var("m"), Var("k")))
    b = Load(name="b", input="B", index=(Var("n"), Var("k")))
    inputs = {"A": Tensor("A", (Dim(4), Dim(64)), "f16"), "B": Tensor("B", (Dim(64), Dim(64)), "f16")}
    load = _fragment_loads(a, b, atom, inputs)["b"]
    assert load.b_trans
    assert load.ldm == 64
