"""A computed operand cone is evaluated once per value it stands for — never once per reader.

Two ways the cone re-derived what it had already computed:

- Attention's output is normalized by its own row sum, and the tree forms the softmax fold twice on
  the way — once under the normalize, once under the reciprocal it derives — as two equal ``Fold``
  nodes it could not share. The o_proj's compute fill replicates the cone per cell, so lowering both
  declared every state of that fold twice in one scope (``acc3__c3 has already been declared``: the
  prefill SDPA + o_proj + residual target on every card). The seam keeps the first lowering only.
- The cone's row-invariant prologue belongs ahead of the K-loop, and the gmem-direct register tile
  emitted it inside — so a fused norm→linear re-derived the row's whole statistic once per
  contraction step, and once per register row on top.

No GPU."""

from __future__ import annotations

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.schedule import Tile, Work
from emmy.compiler.ir.schedule.views import cone_seam
from emmy.compiler.ir.stmt import Body, Load, Loop, Write
from emmy.compiler.ir.stmt.body import dedup_recomputes
from emmy.compiler.ir.stmt.leaves import Assign
from emmy.compiler.pipeline.passes.lowering.kernel._atom import reduce_codegen, store_sink
from emmy.compiler.pipeline.passes.lowering.kernel._tiling import atomize, grid_tile, register_tile, unit_tile
from tests.compiler.terms import contraction, projection, reduction, slab

S = Axis("s", Dim(8))
M, N, K, J = Axis("m", Dim(64)), Axis("n", Dim(64)), Axis("k", Dim(32)), Axis("j", Dim(32))


def _pv():
    """The P·V fold over the key axis, reading the contraction axis ``k`` of the o_proj it feeds — built
    afresh each time, the way the tree forms one traced value twice."""
    p = Load(name="p", input="p", index=(Var("m"), Var("s")))
    v = Load(name="v", input="v", index=(Var("s"), Var("k")))
    return contraction(S, p, (v, "acc"))


def _cone():
    """``out = acc · f(acc)``: the fold read directly and through a derived edge."""
    derived = projection(
        operands=(_pv(),), body=(Assign(name="inv", op=ElementwiseImpl("multiply"), args=("acc", "acc")),), results=("inv",)
    )
    return projection(
        operands=(_pv(), derived),
        body=(Assign(name="out", op=ElementwiseImpl("multiply"), args=("acc", "inv")),),
        results=("out",),
    )


def test_the_seam_lowers_a_twice_read_fold_once() -> None:
    _, cell, _, _ = cone_seam(_cone(), "k", axes=(S,))
    loops = [stmt for stmt in cell if isinstance(stmt, Loop)]
    assert len(loops) == 1
    defined = [name for stmt in cell for name in stmt.defines()]
    assert len(defined) == len(set(defined)), defined
    assert {"inv", "out"} <= Body(cell).ssa_defs


def test_dedup_keeps_a_repeated_effect_and_the_first_definition() -> None:
    fold = tuple(_pv().lower(axes=(S,)))
    twice = (*fold, *fold)
    assert dedup_recomputes(twice) == fold
    assert dedup_recomputes(fold) == fold


def _norm_linear() -> tuple:
    """``out[m, n] = Σ_k w[n, k] · (x[m, k] · Σ_j x[m, j]²)`` — the fused norm→linear shape, whose A is a
    cone over a row statistic (the ``j`` reduce, invariant in the contraction axis) and a k-varying read."""
    stat = reduction(J, (slab("xj", "x", "m", "j"),), (Assign(name="sq", op=ElementwiseImpl("multiply"), args=("xj", "xj")),), ("acc0",))
    cone = projection(
        operands=(stat, slab("xk", "x", "m", "k")),
        body=(Assign(name="a", op=ElementwiseImpl("multiply"), args=("acc0", "xk")),),
        results=("a",),
    )
    return cone, contraction(K, cone, (slab("w", "w", "n", "k"), "acc"))


def _register_tiled(spelling: str) -> Body:
    """The gmem-direct register tile's body for the norm→linear shape, sealed the way ``_factor._bind`` seals it."""
    cone, c = _norm_linear()
    axes = (M, N, K, J)
    plan = Tile.parse(spelling, Work.parse("w1x1")).at(M, N)
    state, region = reduce_codegen(c, plan, seam=cone_seam(cone, K.name, axes=axes), k_axis=K, axes=axes)
    epilogue = Body((Write(output="out", index=(Var("m"), Var("n")), value=c.combine.results[0]),))
    return grid_tile(
        unit_tile(register_tile(atomize(plan.atom.shape[:2]), plan.mn), plan.mn),
        mn=plan.mn,
        block_threads=plan.launch_threads,
        lanes=plan.atom.lanes,
        state_decls=state,
        reduce_region=region,
        store=store_sink(c, plan, epilogue, k_axis=K, axes=axes),
    ).body


def _statistic_loops(body: Body, under: str | None = None) -> int:
    """How many ``j`` reduce loops sit under ``under`` (``None`` = at any depth outside the ``k`` loop)."""
    def walk(stmts, enclosing):
        found = 0
        for stmt in stmts:
            name = stmt.axis.name if isinstance(stmt, Loop) else None
            if name == J.name and enclosing == under:
                found += 1
            for nested in stmt.nested():
                found += walk(nested, name if name in (K.name, J.name) else enclosing)
        return found
    return walk(body, None)


def test_the_row_statistic_lifts_out_of_the_contraction_loop() -> None:
    """One evaluation per register ROW, ahead of the K-loop — not one per row per contraction step."""
    body = _register_tiled("f2x2")
    assert _statistic_loops(body, under=None) == 2
    assert _statistic_loops(body, under=K.name) == 0


def test_every_register_row_keeps_its_own_statistic() -> None:
    """The lift is per row: a taller tile hoists one statistic for each of its rows, since each reads a
    different row of the input."""
    assert _statistic_loops(_register_tiled("f2x4"), under=None) == 4
