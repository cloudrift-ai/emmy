"""A carried state is an operand of its own step, and its ``STAGE`` says where it lives across the steps.

``direct`` keeps the state in its global buffer and launches once per step; ``smem`` gives one CTA the
block — the cells under one batch coordinate — and walks every step inside the launch, the reads of a
step behind a barrier from its writes. The block proof decides which coordinates are the grid's and
which the block's, and refuses a state a step reads outside its own block.
"""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.context import Context
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.loop.runner import execute_loop_op_cpp
from emmy.compiler.ir.schedule.base import schedule
from emmy.compiler.ir.schedule.classic import STATE_KEY, ClassicProblem, ClassicScheduleCodec, ClassicScheduleContext
from emmy.compiler.ir.stmt import Accum, Assign, Body, Carry, Let, Load, Loop, Pre, Select, SelectBranch, Write
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import requires_cuda
from tests.compiler.ir.test_carried_state import _graph as _column_graph
from tests.compiler.ir.test_carried_state import _inputs as _column_inputs
from tests.compiler.ir.test_carried_state import _reference as _column_reference

STEPS, ROWS, BATCH = 3, 4, 2
c, b, i, j, s = Var("c"), Var("b"), Var("i"), Var("j"), Var("s")
_ZERO, _ONE, _TRUE = Literal(0, "int"), Literal(1, "int"), Literal(True, "bool")


def _solve_step(stored: str = "next") -> Body:
    """The forward substitution of a strictly lower triangular system, one row per step: row ``c + 1``
    takes ``A[r, j] += sum_{s < r} A[r, s] * A[s, j]`` for ``j < r`` and every other cell keeps its
    value. The reads under the row's mask address the batch coordinate through the roll's clamped
    spelling, ``mask ? b : 0``, the value discarded where the mask is false."""
    row = BinaryExpr("+", c, _ONE)
    mask = BinaryExpr("&&", BinaryExpr("&&", BinaryExpr("<", i, BinaryExpr("+", row, _ONE)), BinaryExpr("<=", row, i)), BinaryExpr("<", j, row))
    bb, jj = TernaryExpr(mask, b, _ZERO), TernaryExpr(mask, j, _ZERO)
    mix = Loop(
        axis=Axis("s", ROWS),
        body=(
            Pre(name="p1", carrier="A", index=(bb, row, s)),
            Pre(name="p2", carrier="A", index=(bb, s, jj)),
            Assign(name="prod", op="multiply", args=("p1", "p2")),
            Select(name="term", branches=(SelectBranch("prod", BinaryExpr("<", s, row)), SelectBranch("zero", _TRUE))),
            Accum(name="acc", value="term"),
        ),
    )
    cell = (
        mix,
        Pre(name="own_row", carrier="A", index=(bb, row, jj)),
        Assign(name="upd", op="add", args=("acc", "own_row")),
        Pre(name="own", carrier="A", index=(b, i, j)),
        Select(name="next", branches=(SelectBranch("upd", mask), SelectBranch("own", _TRUE))),
        Carry(name="A", value="next", index=(b, i, j), seed="A0"),
        Write(output="out", index=(c, b, i, j), value=stored),
    )
    cells = Loop(axis=Axis("b", BATCH), body=(Loop(axis=Axis("i", ROWS), body=(Loop(axis=Axis("j", ROWS), body=cell),)),))
    return Body((Let(name="zero", value=0.0), Loop(axis=Axis("c", STEPS), body=(cells,))))


def _solve_graph(stored: str = "next") -> Graph:
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("A0", (BATCH, ROWS, ROWS), "f32"), node_id="A0")
    graph.add_node(LoopOp(body=_solve_step(stored), name="k_solve"), ["A0"], Tensor("out", (STEPS, BATCH, ROWS, ROWS), "f32"), node_id="out")
    graph.inputs, graph.outputs = ["A0"], ["out"]
    return graph


def _solve_inputs() -> dict[str, np.ndarray]:
    return {"A0": (np.random.default_rng(1).standard_normal((BATCH, ROWS, ROWS)) * 0.5).astype(np.float32)}


def _solve_reference(arrays: dict[str, np.ndarray]) -> np.ndarray:
    state, want = arrays["A0"].copy(), np.zeros((STEPS, BATCH, ROWS, ROWS), np.float32)
    for step in range(STEPS):
        r = step + 1
        following = state.copy()
        for batch in range(BATCH):
            for col in range(r):
                following[batch, r, col] = state[batch, r, col] + sum(state[batch, r, k] * state[batch, k, col] for k in range(r))
        state = following
        want[step] = state
    return want


def _elementwise_graph() -> Graph:
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (8,), "f32"), node_id="x")
    body = Body((Loop(axis=Axis("i", 8), body=(Load(name="v", input="x", index=(i,)), Assign(name="w", op="exp", args=("v",)), Write(output="y", index=(i,), value="w"))),))
    graph.add_node(LoopOp(body=body, name="k_exp"), ["x"], Tensor("y", (8,), "f32"), node_id="y")
    graph.inputs, graph.outputs = ["x"], ["y"]
    return graph


def _lifted(graph: Graph) -> TileOp:
    lifted = Pipeline.build(["tile/lift"], select=["lift"]).run(graph)
    (node,) = (node for node in lifted.nodes.values() if isinstance(node.op, TileOp))
    return node.op


def _rows(tile: TileOp, target=(7, 0)) -> set[tuple[str, str, str, str]]:
    ctx = Context.from_target(target)
    context = ClassicScheduleContext(tile, ctx, ClassicProblem(tile, ctx))
    codec = ClassicScheduleCodec(context)
    rows = (codec.encode(accepted) for accepted in schedule(context))
    return {(row.get(STATE_KEY, "-"), row["WORK"], row["RASTER"], row.get("REDUCE", "-")) for row in rows}


def test_the_state_key_is_spelled_on_a_kernel_that_carries_a_state_and_on_no_other() -> None:
    carried, plain = _lifted(_column_graph()), _lifted(_elementwise_graph())
    ctx = Context.from_target((7, 0))
    assert STATE_KEY in ClassicScheduleCodec(ClassicScheduleContext(carried, ctx, ClassicProblem(carried, ctx))).keys()
    assert STATE_KEY not in ClassicScheduleCodec(ClassicScheduleContext(plain, ctx, ClassicProblem(plain, ctx))).keys()


def test_the_block_proof_splits_the_state_into_the_grid_and_the_block() -> None:
    column = _lifted(_column_graph()).block_program
    # ``S = D * S + W @ S``: a column reads only its own column, so the columns are the grid and a
    # column's rows are the block.
    assert [axis.name for axis in column.batch] == ["a2"] and [axis.name for axis in column.cells] == ["a1"]
    assert column.seed == 0.0 and column.bytes == 16
    solve = _lifted(_solve_graph()).block_program
    assert [axis.name for axis in solve.batch] == ["a1"] and [axis.name for axis in solve.cells] == ["a2", "a3"]
    assert solve.seed == "A0" and solve.bytes == ROWS * ROWS * 4


def test_the_block_proof_refuses_a_guarded_read_whose_value_survives() -> None:
    # Storing the masked update itself keeps the guarded reads live where the mask is false, so the
    # read at coordinate zero is another CTA's block and the state cannot be held per CTA.
    assert _lifted(_solve_graph(stored="upd")).block_program is None


def test_the_solve_model_matches_its_reference_on_the_launch_loop() -> None:
    arrays = _solve_inputs()
    tile = _lifted(_solve_graph())
    closed = LoopOp(body=tile.loop_body)
    shapes = dict.fromkeys(("out", tile.block_program.state.write.output), (STEPS, BATCH, ROWS, ROWS))
    got = dict(zip(closed.outputs, execute_loop_op_cpp(closed, arrays, shapes), strict=True))
    np.testing.assert_allclose(got["out"], _solve_reference(arrays), rtol=1e-5, atol=1e-6)


def test_the_classic_walk_offers_the_launch_loop_and_the_resident_block() -> None:
    rows = _rows(_lifted(_column_graph()))
    assert ("", "", "", "") in rows
    assert {("d1/smem", "t32", "", ""), ("d2/smem", "t32", "", ""), ("d1/smem", "t1024", "", "")} <= rows
    # A resident block stripes its cells across a thread inventory and takes no raster and no
    # cooperating reduce of its own.
    assert all(work.startswith("t") and raster == "" and reduce == "" for state, work, raster, reduce in rows if state)


@pytest.mark.parametrize("depth", [1, 2])
def test_a_resident_state_lowers_to_one_launch_holding_the_block(depth: int) -> None:
    with pinned_knobs({STATE_KEY: f"d{depth}/smem", "WORK": "t32"}):
        graph = Pipeline.build(CUDA_PASSES).run(_column_graph(), ctx=Context.from_target((7, 0)))
    (op,) = (node.op for node in graph.nodes.values() if isinstance(node.op, CudaOp))
    assert not op.serial and op.smem_bytes == 16 * depth and op.block == ((32,), (1,), (1,))
    # The private state has no global port; the snapshot the graph reads stays an output.
    assert op.arg_order == ("D", "W", "U", "out")
    # One barrier after the seed fill; per step, one at depth two and two at depth one.
    assert op.kernel_source.count("__syncthreads()") == 1 + (3 - depth)


@requires_cuda
@pytest.mark.xdist_group("cuda")
@pytest.mark.parametrize("width", ["t32", "t64"])
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("model", ["column", "solve"])
def test_a_resident_state_matches_the_reference_on_the_gpu(model: str, depth: int, width: str) -> None:
    from emmy.compiler.backend.cuda.program import run_program  # noqa: PLC0415

    build, inputs, reference = {
        "column": (_column_graph, _column_inputs, _column_reference),
        "solve": (_solve_graph, _solve_inputs, _solve_reference),
    }[model]
    arrays = inputs()
    with pinned_knobs({STATE_KEY: f"d{depth}/smem", "WORK": width}):
        graph = Pipeline.build(CUDA_PASSES).run(build())
    (op,) = (node.op for node in graph.nodes.values() if isinstance(node.op, CudaOp))
    assert not op.serial
    result, _ = run_program(graph, arrays)
    want = reference(arrays)
    np.testing.assert_allclose(np.asarray(result.outputs["out"]).reshape(want.shape), want, rtol=1e-5, atol=1e-6)
