"""Target selection and schedule legality for the Hopper ``wgmma`` warp-group family.

The ``wgmma_m64n<N>k16_{f16,bf16}_f32`` atoms are the m16n8k16 warp cell whose PTX instruction four
M-adjacent warps issue together, so the accumulator, epilogue and P→A repack keep the mma.sync lane
maps and only the schedule carries the group: a ``w<4k>x1`` warp grid, one fragment row of whole
instructions (``f1x<C>``, ``C`` a multiple of ``N/8``), a K chunk of one 128-byte swizzle row (``k4``)
and every operand staged in shared memory. Each rule drops an unpinned row silently and refuses a
pinned one with its own message. An N-contiguous B stages ATOM-MAJOR — its 128-byte swizzle atoms
stacked along the slab rows, one TMA box per atom — which is the descriptor's MN-major canonical
layout at any tile width. sm_90 only. No GPU: the atom registry, the classic domains, and the
staged operand geometry the lowering builds.
"""

from __future__ import annotations

import pytest

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import BF16, F16
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.atom import ATOM_REGISTRY, atoms_for, wide_accumulate
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.frontend.ir import MatmulOp
from emmy.compiler.ir.kernel.ir import Smem, TmaDescriptor, WgmmaDescriptor
from emmy.compiler.ir.schedule import Stage, Tile, Work
from emmy.compiler.ir.schedule.classic import ClassicProblem, ClassicScheduleContext, ReductionSchedule, project_classic
from emmy.compiler.ir.schedule.classic import refusals as classic
from emmy.compiler.ir.schedule.classic.refusals import _wgmma_refusal
from emmy.compiler.ir.stmt import Load
from emmy.compiler.ir.stmt.leaves import Assign
from emmy.compiler.ir.tile import Placement, TileOp
from emmy.compiler.pipeline import TILE_PASSES, Pipeline
from emmy.compiler.pipeline.passes.lowering.kernel._atom import _slab_operands, _sync_operands, _tile_base, _wgmma_drain
from emmy.compiler.pipeline.passes.lowering.kernel._stage import CtaTile, TmaTransport
from tests.compiler.terms import contraction, projection

WGMMA = tuple(name for name in ATOM_REGISTRY if name.startswith("wgmma_"))
N128 = "wgmma_m64n128k16_bf16_f32"
M, N, K = 512, 1024, 1024


def _family(names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(name for name in names if name.startswith("wgmma_"))


def test_family_is_offered_on_hopper_only_and_moves_no_option_0() -> None:
    for dtype, tag in ((F16, "f16"), (BF16, "bf16")):
        offered = atoms_for(dtype, ctx=Context.from_target((9, 0)))
        assert offered[0] == f"mma_m16n8k16_{tag}_f32"
        assert _family(offered) == tuple(f"wgmma_m64n{n}k16_{tag}_f32" for n in (64, 128, 256))
        for cap in ((8, 0), (8, 9), (10, 0), (12, 0)):
            assert _family(atoms_for(dtype, ctx=Context.from_target(cap))) == (), cap
    assert atoms_for(F16) == ("mma_m16n8k16_f16_f32",)


def test_cell_is_the_m16n8k16_sub_cell_of_its_instruction() -> None:
    assert len(WGMMA) == 6
    for name in WGMMA:
        atom = ATOM_REGISTRY[name]
        assert atom.shape == (16, 8, 16) and atom.is_wgmma
        assert atom.cells_per_instruction == atom.ptx_shape[1] // 8 in (8, 16, 32)
        assert atom.accumulator_registers_per_lane == 4
        assert atom.c_to_a_repack
        assert wide_accumulate(atom) is atom
    for name in ("mma_m16n8k16_f16_f32", "mma_m8n8k4_f16_f32"):
        assert not ATOM_REGISTRY[name].is_wgmma and ATOM_REGISTRY[name].cells_per_instruction == 1
    assert wide_accumulate(ATOM_REGISTRY["mma_m16n8k16_f16_f16"]).name == "mma_m16n8k16_f16_f32"


@pytest.mark.parametrize(
    ("work", "tile", "stage", "message"),
    [
        ("w4x1", f"{N128}/f1x16/k4", "d2/smem-tma", None),
        ("w8x1", f"{N128}/f1x32/k4", "d2/smem-async", None),
        ("w2x4", f"{N128}/f1x16/k4", "d2/smem-tma", "w<4k>x1 warp grid"),
        ("w2x1", f"{N128}/f1x16/k4", "d2/smem-tma", "w<4k>x1 warp grid"),
        ("w4x1", f"{N128}/f2x8/k4", "d2/smem-tma", "fragment grid must be f1x<C>"),
        ("w4x1", f"{N128}/f1x8/k4", "d2/smem-tma", "fragment grid must be f1x<C>"),
        ("w4x1", f"{N128}/f1x16/k8", "d2/smem-tma", "K chunk must be 64 elements"),
        ("w4x1", f"{N128}/f1x16/k2", "d2/smem-tma", "K chunk must be 64 elements"),
        ("w4x1", f"{N128}/f1x16/k4", "", "direct stage cannot feed it"),
        ("w2x4", "mma_m16n8k16_bf16_f32/f2x2/k2", "", None),
    ],
)
def test_rules_read_the_tile_and_its_stage(work, tile, stage, message) -> None:
    plan = Tile.parse(tile, Work.parse(work))
    why = _wgmma_refusal(plan, Stage.parse(stage))
    assert (why is None) if message is None else (message in why), why


def _matmul(b_trans: bool = False) -> TileOp:
    """A bf16 matmul whose B is N-contiguous (``[K, N]``) or, with ``b_trans``, K-contiguous (``[N, K]``,
    the serving ``F.linear`` weight layout)."""
    m, n, k = Axis("m", M), Axis("n", N), Axis("k", K)
    a = Load(name="a_e", input="a", index=(Var("m"), Var("k")))
    b_index, b_shape = ((Var("n"), Var("k")), (N, K)) if b_trans else ((Var("k"), Var("n")), (K, N))
    root = contraction(k, a, (Load(name="b_e", input="b", index=b_index), "acc"))
    return TileOp(
        op=root,
        place=Placement(free=(m, n)),
        axes=(m, n, k),
        inputs={"a": Tensor("a", (M, K), "bf16"), "b": Tensor("b", b_shape, "bf16")},
        outputs={"out": Tensor("out", (M, N), "bf16")},
    )


def _domains(monkeypatch, b_trans: bool = False):
    """The bf16 matmul's sm_90 domains over a catalog cut to three warp grids and one TMA stage."""
    moves = classic.warp_tile_moves
    monkeypatch.setattr(classic, "scalar_tile_moves", lambda: [Tile()])
    monkeypatch.setattr(classic, "warp_tile_moves", lambda atoms: [plan for plan in moves(atoms) if plan.units in ((2, 4), (4, 1), (8, 1))])
    monkeypatch.setattr(classic, "stage_moves", lambda *, warp, ctx=None: [Stage(depth=2, transport="smem-tma")])
    tile, target = _matmul(b_trans), Context.from_target((9, 0))
    return tile, target, project_classic(tile, target)


FULL_WGMMA_ROWS = {
    (64, (1, 8), 4),
    (64, (1, 16), 4),
    (64, (1, 32), 4),
    (128, (1, 16), 4),
    (128, (1, 32), 4),
    (256, (1, 32), 4),
}


@pytest.mark.parametrize("b_trans", [False, True])
def test_domain_offers_only_group_aligned_wgmma_rows_and_stages_them(monkeypatch, b_trans: bool) -> None:
    """The catalog keeps every wgmma row whose grid, fragment grid and K chunk the group allows,
    whatever the B orientation (an N-contiguous B stages atom-major, so its tile is not bound to
    one swizzle atom), and the compatibility join lets none of them read a direct stage — while
    the mma.sync rows beside them still do, so the drop is the rule's, not the stage domain's."""
    tile, target, domains = _domains(monkeypatch, b_trans=b_trans)
    site = tile.node_sites[0]

    rows = tuple(choice.tile for choice in domains.nodes[site] if isinstance(choice, ReductionSchedule) and choice.tile.is_warp)
    wgmma = tuple(plan for plan in rows if plan.atom.is_wgmma)
    assert {plan.units for plan in wgmma} == {(4, 1), (8, 1)}
    assert {(plan.atom.ptx_shape[1], plan.regs, plan.bk) for plan in wgmma} == FULL_WGMMA_ROWS
    assert {(2, 4), (4, 1), (8, 1)} <= {plan.units for plan in rows if not plan.atom.is_wgmma}
    assert any(plan.bk == 8 for plan in rows if not plan.atom.is_wgmma)

    context = ClassicScheduleContext(tile, target, ClassicProblem(tile, target, allow_f16_accumulate=False, allow_fp8=False))
    picks = tuple(context.extensions())
    staged = {pick.nodes[site].tile.atom.name for pick in picks if all(not choice.stage.is_direct for choice in pick.edges.values())}
    direct = {pick.nodes[site].tile.atom.name for pick in picks if any(choice.stage.is_direct for choice in pick.edges.values())}
    assert {name for name in WGMMA if name.endswith("_bf16_f32")} <= staged
    assert not set(WGMMA) & direct and "mma_m16n8k16_bf16_f32" in direct


@pytest.mark.parametrize(
    ("pins", "message"),
    [
        ({"WORK": "w2x4", "TILE": f"{N128}/f1x16/k4"}, "w<4k>x1 warp grid"),
        ({"WORK": "w4x1", "TILE": f"{N128}/f2x8/k4"}, "fragment grid must be f1x<C>"),
        ({"WORK": "w4x1", "TILE": f"{N128}/f1x16/k8"}, "K chunk must be 64 elements"),
        ({"WORK": "w4x1", "TILE": f"{N128}/f1x16/k4", "STAGE": ""}, "direct stage cannot feed it"),
        ({"TILE": f"{N128}/f1x16/k4", "STAGE": ""}, "direct stage cannot feed it"),
    ],
)
def test_pinned_wgmma_row_refuses_with_its_rule(monkeypatch, pins, message) -> None:
    """A pin the catalog never offered is refused with the rule's message, not as an unsupported
    pin; a wgmma TILE with no WORK pin still meets the rules that do not read the grid."""
    tile, target, _ = _domains(monkeypatch)
    with pytest.raises(ValueError, match=message):
        _ = ClassicProblem(tile, target, row=pins).domains


def _graph(m: int, n: int, k: int) -> Graph:
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("a", (m, k), dtype=BF16), node_id="a")
    graph.add_node(InputOp(), [], Tensor("b", (k, n), dtype=BF16), node_id="b")
    graph.add_node(MatmulOp(), ["a", "b"], Tensor("c", (m, n), dtype=BF16), node_id="c")
    graph.inputs, graph.outputs = ["a", "b"], ["c"]
    return graph


def test_env_pins_reach_the_wgmma_rules_and_a_legal_row_schedules(monkeypatch) -> None:
    monkeypatch.setenv("EMMY_TILE", f"{N128}/f1x16/k4")
    monkeypatch.setenv("EMMY_STAGE", "d2/smem-tma")
    monkeypatch.setenv("EMMY_WORK", "w2x4")
    with pytest.raises(ValueError, match="w<4k>x1 warp grid"):
        Pipeline.build(TILE_PASSES).run(_graph(128, 256, 256), ctx=Context.from_target((9, 0)))
    monkeypatch.setenv("EMMY_WORK", "w4x1")
    out = Pipeline.build(TILE_PASSES).run(_graph(128, 256, 256), ctx=Context.from_target((9, 0)))
    tile_op = next(node.op for node in out.nodes.values() if isinstance(node.op, TileOp))
    (placed,) = tile_op.materialization.tiles.values()
    assert placed.choice.atom.name == N128 and placed.choice.units == (4, 1)


def _lit(n: int) -> Literal:
    return Literal(n, "int")


def _at(expr, **env: int) -> int:
    return expr.eval(dict(env))


def _atom_major_operands():
    """The ``w8x1 n128/f1x16/k4`` tile's staged operands over the bf16 matmul with an N-contiguous B: a
    128 x 128 CTA tile whose B slab is two swizzle atoms wide."""
    mn = Tile.parse(f"{N128}/f1x16/k4", Work.parse("w8x1")).at(Axis("m", Dim(M)), Axis("n", Dim(N))).mn
    ka = Axis("k", Dim(K))
    ops = _slab_operands(
        index_srcs=((Var("m"), Var("k")), (Var("k"), Var("n"))),
        bufs=("a", "b"),
        mn=mn,
        k_axis=ka,
        bk_elems=64,
        base=_tile_base(mn),
        swizzles=("B128", "B128"),
        b_atoms=2,
    )
    return mn, ka, ops


def test_an_n_contiguous_b_stages_atom_major() -> None:
    """The B slab stacks its atoms along the rows — slab row ``r`` is K row ``r % 64`` of the atom whose
    columns start at ``(r / 64)·64`` — so its own row is one swizzle row; the TMA fill deposits one
    ``(64, 64)`` box per atom, 64 rows down the slot and 64 columns along the weight, under one
    expect-tx for the whole slot; and the drain's B descriptors read each k16 step 16 rows down the
    atom, with the atom stride (8 KiB) as the leading offset and the eight-row core group (1 KiB) as
    the stride."""
    mn, _, (a_op, b_op) = _atom_major_operands()
    assert (a_op.shape, a_op.atoms, b_op.shape, b_op.atoms, b_op.box_extents) == ((128, 64), 1, (128, 64), 2, (64, 64))
    k, n = b_op.index(_lit(0))(_lit(70), _lit(3))
    assert (_at(k), _at(n, n_b=2)) == (6, 2 * 128 + 64 + 3)

    transport = TmaTransport(
        operands=(a_op, b_op), slab_dtype="__nv_bfloat16", elem_bytes=2, cta=CtaTile(linear_tid=Var("_t"), n_threads=256)
    )
    decls = transport.slab_decls(ring=2)
    assert next(d for d in decls if isinstance(d, TmaDescriptor) and d.name == b_op.desc).box_extents == (64, 64)
    assert next(d for d in decls if isinstance(d, Smem) and d.name == b_op.slab).extents == (2 * 128, 64)
    (cond,) = transport.fill(k0=Var("k0"), slot=Var("_s"))
    expect, *loads = cond.body
    assert expect.bytes_ == 2 * 128 * 64 * 2
    assert [load.smem for load in loads] == [a_op.slab, b_op.slab, b_op.slab]
    assert [_at(load.smem_index[0], _s=1) for load in loads[1:]] == [128, 128 + 64]
    assert [(_at(load.coords[0], k0=64), _at(load.coords[1], n_b=2)) for load in loads[1:]] == [(64, 256), (64, 256 + 64)]

    drain = _wgmma_drain(operands=(a_op, b_op), slot=Var("_s"), mn=mn, atom=ATOM_REGISTRY[N128], bk_elems=64, frag_ns="", n_folds=1)
    descs = [s for s in drain if isinstance(s, WgmmaDescriptor) and s.smem == b_op.slab]
    assert len(descs) == 4 and {(d.lbo_bytes, d.sbo_bytes, d.swizzle) for d in descs} == {(64 * 128, 1024, "B128")}
    assert [_at(d.smem_index, **{mn[1].unit: 0, "_s": 1}) for d in descs] == [(128 + 16 * step) * 64 for step in range(4)]


def test_a_computed_n_contiguous_b_fills_atom_major() -> None:
    """The ``smem`` compute fill writes the same stacked slab: cell ``(70, 3)`` of the two-atom B evaluates
    its producer at K row 6, column 3 of the second atom."""
    mn, ka, _ = _atom_major_operands()
    a = Load(name="a_e", input="a", index=(Var("m"), Var("k")), dtype=BF16)
    w = Load(name="w_e", input="w", index=(Var("k"), Var("n")), dtype=BF16)
    b = projection(body=(w, Assign(name="bv", op=ElementwiseImpl("multiply"), args=("w_e", "w_e"))), results=("bv",))
    c = contraction(ka, a, (b, "acc"))
    _, sync_ops, _, _ = _sync_operands(c, 64, mn, CtaTile(linear_tid=Var("_t"), n_threads=256), ("B128", "B128"), k_axis=ka, b_atoms=2)
    b_op = next(op for op in sync_ops if op.tag == "b")
    assert b_op.shape == (128, 64)
    stmts, _ = b_op.value(_lit(0), _lit(70), _lit(3))
    read = next(s for s in stmts if isinstance(s, Load) and s.input == "w")
    assert (_at(read.index[0]), _at(read.index[1], n_b=2)) == (6, 2 * 128 + 64 + 3)
