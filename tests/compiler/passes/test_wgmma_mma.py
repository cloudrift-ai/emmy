"""Target selection and schedule legality for the Hopper ``wgmma`` warp-group family.

The ``wgmma_m64n<N>k16_{f16,bf16}_f32`` atoms are the m16n8k16 warp cell whose PTX instruction four
M-adjacent warps issue together, so the accumulator, epilogue and P→A repack keep the mma.sync lane
maps and only the schedule carries the group: a ``w<4k>x1`` warp grid, one fragment row of whole
instructions (``f1x<C>``, ``C`` a multiple of ``N/8``), a K chunk of one 128-byte swizzle row (``k4``)
and every operand staged in shared memory. Each rule drops an unpinned row silently and refuses a
pinned one with its own message. sm_90 only. No GPU: the atom registry and the classic domains.
"""

from __future__ import annotations

import pytest

from emmy.compiler.context import Context
from emmy.compiler.dtype import BF16, F16
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.atom import ATOM_REGISTRY, atoms_for, wide_accumulate
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.frontend.ir import MatmulOp
from emmy.compiler.ir.schedule import Stage, Tile, Work
from emmy.compiler.ir.schedule import classic_projection as classic
from emmy.compiler.ir.schedule.classic import ClassicScheduleContext, ReductionSchedule, _wgmma_refusal
from emmy.compiler.ir.schedule.classic_projection import project_classic
from emmy.compiler.ir.stmt import Load
from emmy.compiler.ir.tile import Placement, TileOp
from emmy.compiler.pipeline import TILE_PASSES, Pipeline
from tests.compiler.terms import contraction

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


def _matmul() -> TileOp:
    m, n, k = Axis("m", M), Axis("n", N), Axis("k", K)
    a = Load(name="a_e", input="a", index=(Var("m"), Var("k")))
    root = contraction(k, a, (Load(name="b_e", input="b", index=(Var("k"), Var("n"))), "acc"))
    return TileOp(
        op=root,
        place=Placement(free=(m, n)),
        axes=(m, n, k),
        inputs={"a": Tensor("a", (M, K), "bf16"), "b": Tensor("b", (K, N), "bf16")},
        outputs={"out": Tensor("out", (M, N), "bf16")},
    )


def _domains(monkeypatch):
    """The bf16 matmul's sm_90 domains over a catalog cut to three warp grids and one TMA stage."""
    moves = classic.warp_tile_moves
    monkeypatch.setattr(classic, "scalar_tile_moves", lambda: [Tile()])
    monkeypatch.setattr(classic, "warp_tile_moves", lambda atoms: [plan for plan in moves(atoms) if plan.units in ((2, 4), (4, 1), (8, 1))])
    monkeypatch.setattr(classic, "stage_moves", lambda *, warp, ctx=None: [Stage(depth=2, transport="smem-tma")])
    tile, target = _matmul(), Context.from_target((9, 0))
    return tile, target, project_classic(tile, target)


def test_domain_offers_only_group_aligned_wgmma_rows_and_stages_them(monkeypatch) -> None:
    """The catalog keeps every wgmma row whose grid, fragment grid and K chunk the group allows,
    and the compatibility join lets none of them read a direct stage — while the mma.sync rows
    beside them still do, so the drop is the rule's, not the stage domain's."""
    tile, target, domains = _domains(monkeypatch)
    site = tile.node_sites[0]

    rows = tuple(choice.tile for choice in domains.nodes[site] if isinstance(choice, ReductionSchedule) and choice.tile.is_warp)
    wgmma = tuple(plan for plan in rows if plan.atom.is_wgmma)
    assert {plan.units for plan in wgmma} == {(4, 1), (8, 1)}
    assert {(plan.atom.ptx_shape[1], plan.regs, plan.bk) for plan in wgmma} == {
        (64, (1, 8), 4),
        (64, (1, 16), 4),
        (64, (1, 32), 4),
        (128, (1, 16), 4),
        (128, (1, 32), 4),
        (256, (1, 32), 4),
    }
    assert {(2, 4), (4, 1), (8, 1)} <= {plan.units for plan in rows if not plan.atom.is_wgmma}
    assert any(plan.bk == 8 for plan in rows if not plan.atom.is_wgmma)

    context = ClassicScheduleContext(tile, target, domains).restrict({}, allow_f16_accumulate=False, allow_fp8=False)
    picks = tuple(context.extensions())
    staged = {pick.nodes[site].tile.atom.name for pick in picks if all(not choice.stage.is_direct for choice in pick.edges.values())}
    direct = {pick.nodes[site].tile.atom.name for pick in picks if any(choice.stage.is_direct for choice in pick.edges.values())}
    assert set(WGMMA) & staged == set(_family(atoms_for(BF16, ctx=target)))
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
    tile, target, domains = _domains(monkeypatch)
    with pytest.raises(ValueError, match=message):
        ClassicScheduleContext(tile, target, domains).restrict({family: ((family, value),) for family, value in pins.items()})


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
