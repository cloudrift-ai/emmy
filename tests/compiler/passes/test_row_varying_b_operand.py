"""A per-row matvec can tile with a unit fragment row, and fragment loads bind sibling axes.

The DeepSeek-V4-Flash ``post4096`` twin's matmul piece reads ``out[a0, m, n] = Σ_k A[a0, m, k] · B[a0, m, k, n]``:
a batch of per-row matvecs whose B carries a live ``m`` stride (a placement cut materialized the frontend's
per-row broadcast). That ``m`` coordinate is a batch dimension of independent matrix-vector products.
Putting a size-one row beside the trailing column lets the MMA tile each product while ``m`` stays a grid
coordinate and B's per-batch address remains intact.

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
from emmy.compiler.ir.kernel.ir import LdmatrixLoad
from emmy.compiler.ir.schedule import Placement, Tile, Work
from emmy.compiler.ir.schedule.classic import ClassicProblem
from emmy.compiler.ir.stmt import Body, Load, parse_body
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline.passes.lowering.kernel._atom import reduce_codegen, store_sink
from emmy.compiler.pipeline.passes.lowering.kernel._tiling import atomize, grid_tile, register_tile, unit_tile
from tests.compiler.terms import contraction

_A0, _M, _N, _K = Axis("a0", 2), Axis("m", 4), Axis("n", 64), Axis("k", 4)


def _row_residue(sibling: str, own: str) -> str:
    """``((s·1024 + o) / 128 % 8) · 128 + (s·1024 + o) % 128`` — value-equal to ``o`` for ``o < 1024``."""
    flat = f"({sibling} * 1024 + {own})"
    return f"{flat} / 128 % 8 * 128 + {flat} % 128"


def _free(expressions) -> set[str]:
    return set().union(*(set(e.free_vars()) for e in expressions))


def _fragment_loads(a: Load, b: Load, atom: str, inputs=None) -> dict[str, LdmatrixLoad]:
    """The gmem-direct operand fragment loads the mma atom emits for ``a ⊗ b`` on a 1×1 warp tile — sealed
    through ``grid_tile`` the way ``_factor._bind``'s output-tiled arm seals it. ``inputs`` gives the
    operand tensors their shapes, which is what lets the leaf read a stride rather than a dim position."""
    c = contraction(_K, a, (b, "acc"))
    plan = Tile.parse(f"{atom}/f1x1", Work.parse("w1x1")).at(_M, _N)
    axes = (_M, _N, _K)
    state, region = reduce_codegen(c, plan, inputs=inputs, k_axis=_K, axes=axes)
    epilogue = parse_body(f"out[m, n] = {c.combine.results[0]}")
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
def test_batched_matvec_uses_a_unit_fragment_row(cc) -> None:
    """A and B both read m, so m is a batch coordinate and each batch has one output row.

    B varies between batches, which is legal; it must never vary within a fragment row.
    """
    a, b = parse_body("a = load A[a0, m, k]\nb = load B[a0, m, k, n]")
    tile = TileOp(op=contraction(_K, a, (b, "acc")), place=Placement(free=(_A0, _M, _N)), axes=(_A0, _M, _N, _K))
    site = tile.node_id(tile.op)
    assert tuple(axis.name for axis in tile.grid_sched._mn_for(tile.op)) == ("_um", "n")
    assert tile.contracts(site)
    offers = ClassicProblem(tile, Context.from_target(cc))
    assert any(choice.tile.is_tiled for choice in offers.node_site(site).nodes)


@pytest.mark.parametrize("atom", ["mma_m8n8k4_f16_f32", "mma_m16n8k16_f16_f32"])
def test_gmem_direct_b_fragment_binds_the_row_residue(atom) -> None:
    """The merged-weight column residue: a value-dead ``m`` in B's index is spelled through the row's block
    base, never the bare axis name the tiled kernel does not define."""
    a, b = parse_body(f"a = load A[m, k]\nb = load B[k, {_row_residue('m', 'n')}]")
    load = _fragment_loads(a, b, atom)["b"]
    assert "m" not in _free(load.src_index) and "m_b" in _free(load.src_index)


@pytest.mark.parametrize("atom", ["mma_m8n8k4_f16_f32", "mma_m16n8k16_f16_f32"])
def test_gmem_direct_a_fragment_binds_the_column_residue(atom) -> None:
    """The mirror on A: a value-dead ``n`` in A's index is spelled through the column's block base."""
    a, b = parse_body(f"a = load A[m, {_row_residue('n', 'k')}]\nb = load B[k, n]")
    load = _fragment_loads(a, b, atom)["a"]
    assert "n" not in _free(load.src_index) and "n_b" in _free(load.src_index)


def _packed(k_extent: int, n_extent: int) -> tuple[Load, Load, dict]:
    """``out[m, n] = Σ_k a[m, k] · x[m, k * n_extent + n]`` — B's own coordinate and the reduction
    coordinate packed into ONE tensor dim, the shape DeepSeek-V4's hyper-connection mixing reads
    (``hidden[stream * 4096 + channel]``, 24 streams over one 16384-wide row)."""
    a, b = parse_body(f"a = load A[m, k]\nb = load X[m, k * {n_extent} + n]")
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
    a, b = parse_body("a = load A[m, k]\nb = load B[n, k]")
    inputs = {"A": Tensor("A", (Dim(4), Dim(64)), "f16"), "B": Tensor("B", (Dim(64), Dim(64)), "f16")}
    load = _fragment_loads(a, b, atom, inputs)["b"]
    assert load.b_trans
    assert load.ldm == 64
