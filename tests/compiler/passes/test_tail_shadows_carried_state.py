"""A tail that recomputes the carrier's own fold must not redeclare the cell's accumulator.

The DeepSeek-V4-Flash ``post1`` twin's ``k_div_11_reduce`` carries one value twice: the register tile
folds it into ``acc0``, and the projection tail below re-derives the same sum in its own loop, which
the trace also named ``acc0`` because in Loop IR the two live in separate scopes. The scalar tier
replicates the tail under the same ``__c{i}_{j}`` suffix the states take, so both landed on
``acc0__c0_0`` in ONE emitted scope and nvcc refused the kernel — eight collisions, one per cell.

The tail's value is its own, so it takes its own name. A tail that merely READS the state is
untouched: that read is how a per-cell epilogue reaches its accumulator.
"""

from __future__ import annotations

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.schedule import Tile
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.pipeline.passes.lowering.kernel._atom import reduce_codegen, store_sink
from emmy.compiler.pipeline.passes.lowering.kernel._tiling import atomize, grid_tile, register_tile, unit_tile
from tests.compiler.terms import contraction

_M, _N, _K = Axis("m", 8), Axis("n", 8), Axis("k", 4)
_PLAN = Tile(units=(2, 2), regs=(2, 2))


def _tile(epilogue: Body):
    """The scalar contraction ``a ⊗ b`` with ``epilogue`` as its projection tail, sealed through
    ``grid_tile`` the way ``_factor._bind``'s output-tiled arm seals it."""
    c = contraction(
        _K,
        Load(name="a", input="A", index=(Var("m"), Var("k"))),
        (Load(name="b", input="B", index=(Var("k"), Var("n"))), "acc"),
    )
    plan = _PLAN.at(_M, _N)
    state, region = reduce_codegen(c, plan, k_axis=_K, axes=(_M, _N, _K))
    return grid_tile(
        unit_tile(register_tile(atomize(plan.atom.shape[:2]), plan.mn), plan.mn),
        mn=plan.mn,
        block_threads=plan.launch_threads,
        lanes=plan.atom.lanes,
        state_decls=state,
        reduce_region=region,
        store=store_sink(c, plan, epilogue, k_axis=_K, axes=(_M, _N, _K)),
    )


def _definitions(tile, name: str) -> int:
    return sum(1 for stmt in Body(tile.body).iter() for defined in stmt.defines() if defined == name)


def _recomputing_tail() -> Body:
    """A tail that folds the same sum again into a name the term also carries."""
    return Body(
        (
            Loop(
                axis=_K,
                body=Body(
                    (
                        Load(name="again", input="A", index=(Var("m"), Var("k"))),
                        Accum(name="acc", value="again", op="add", axes=("k",)),
                    )
                ),
            ),
            Assign(name="out", op="multiply", args=("acc", "acc")),
            Write(output="out", index=(Var("m"), Var("n")), value="out"),
        )
    )


def test_a_tail_that_recomputes_a_carried_state_does_not_redeclare_the_cell_accumulator() -> None:
    tile = _tile(_recomputing_tail())
    assert _definitions(tile, "acc__c0_0") == 1, "the cell's accumulator is declared once — the tail's sum is its own value"
    assert _definitions(tile, "acc__own__c0_0") == 1, "and the tail's own sum keeps a cell copy under its own name"


def test_a_tail_that_only_reads_the_state_still_reads_the_cell_accumulator() -> None:
    """The ordinary epilogue: no definition of its own, so nothing is renamed and the read resolves
    to the cell's accumulator through the plain suffix."""
    tail = Body(
        (
            Assign(name="scaled", op="multiply", args=("acc", "acc")),
            Write(output="out", index=(Var("m"), Var("n")), value="scaled"),
        )
    )
    reads = {dep for stmt in Body(_tile(tail).body).iter() for dep in stmt.deps()}
    assert "acc__c0_0" in reads and not any(name.startswith("acc__own") for name in reads)
