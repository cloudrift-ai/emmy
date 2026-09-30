"""A carried state: a recurrence whose step reads OTHER cells of its own state.

A scalar ``Accum`` lives inside the loops over the cells, so a step sees only its own cell. A delta
rule's step contracts over the previous state (``k @ S``), which needs the loop over the steps
OUTSIDE the cells and the state kept across them. ``Carry`` spells that: the statement defines
the next value of one cell, a ``Pre`` read sees what the previous step left at any cell, and the
loop that carries it is read off the body.
"""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.loop.runner import execute_loop_op_cpp
from emmy.compiler.ir.stmt import Accum, Assign, Body, Carry, Load, Loop, Pre, Write
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.ir import loaded_buffers
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from emmy.compiler.wire import decode, encode
from tests.compiler.helpers import requires_cuda

STEPS, N = 3, 4
c, i, j, k = Var("c"), Var("i"), Var("j"), Var("k")
ZERO = Literal(0, "int")


def _step(out_index: tuple, seed: float | str = 0.0, u_index: tuple = (c, i, j), steps: int = STEPS) -> Body:
    """``S_c = decay_c * S_{c-1} + W @ S_{c-1} + U_c``, storing the state each step READ, from ``seed``."""
    mix = Loop(
        axis=Axis("k", N),
        body=(
            Load(name="w", input="W", index=(i, k)),
            Pre(name="other", carrier="S", index=(k, j)),
            Assign(name="p", op="multiply", args=("w", "other")),
            Accum(name="mixed", value="p"),
        ),
    )
    cell = (
        mix,
        Load(name="u", input="U", index=u_index),
        Pre(name="own", carrier="S", index=(i, j)),
        Assign(name="kept", op="multiply", args=("own", "decay")),
        Assign(name="moved", op="add", args=("kept", "mixed")),
        Assign(name="next", op="add", args=("moved", "u")),
        Carry(name="S", value="next", index=(i, j), seed=seed),
        Write(output="out", index=out_index, value="own"),
    )
    cells = Loop(axis=Axis("i", N), body=(Loop(axis=Axis("j", N), body=cell),))
    return Body((Loop(axis=Axis("c", steps), body=(Load(name="decay", input="D", index=(c,)), cells)),))


def _loops(body: Body) -> list[Loop]:
    return [stmt for stmt in body.iter() if isinstance(stmt, Loop)]


def _inputs(seed: float | str = 0.0, batch: int | None = None, steps: int = STEPS) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    arrays = {
        "D": rng.standard_normal(steps).astype(np.float32),
        "W": (rng.standard_normal((N, N)) * 0.3).astype(np.float32),
        "U": rng.standard_normal((*(() if batch is None else (batch,)), steps, N, N)).astype(np.float32),
    }
    if isinstance(seed, str):
        arrays[seed] = rng.standard_normal((N, N)).astype(np.float32)
    return arrays


def _reference(arrays: dict[str, np.ndarray], seed: float | str = 0.0) -> np.ndarray:
    if arrays["U"].ndim == 4:  # a batch of recurrences, one per leading row of U
        return np.stack([_reference({**arrays, "U": u}, seed) for u in arrays["U"]])
    state = arrays[seed].copy() if isinstance(seed, str) else np.full((N, N), seed, np.float32)
    steps = arrays["U"].shape[0]
    want = np.zeros((steps, N, N), np.float32)
    for step in range(steps):
        want[step] = state
        state = arrays["D"][step] * state + arrays["W"] @ state + arrays["U"][step]
    return want


@pytest.mark.parametrize("step_major", [True, False], ids=["out[c,i,j]", "out[i,j,c]"])
def test_a_step_reads_other_cells_of_its_own_state(step_major: bool) -> None:
    """Both output layouts: storage order sorts the FREE loops, and sorting the loop over the steps
    under the cells would hand a step cells the previous step has not stored yet."""
    op = LoopOp(body=_step((c, i, j) if step_major else (i, j, c)))

    outer = op.body[0]
    assert outer.is_reduce and list(outer.carries) == [carry.name for carry in op.body.carries]
    assert not any(loop.carries for loop in _loops(outer.body))

    arrays = _inputs()
    got = np.asarray(execute_loop_op_cpp(op, arrays, {"out": (STEPS, N, N) if step_major else (N, N, STEPS)}))

    np.testing.assert_allclose(got if step_major else np.moveaxis(got, -1, 0), _reference(arrays), rtol=1e-5, atol=1e-6)


def _graph(seed: float | str = 0.0, steps: int = STEPS) -> Graph:
    """The step as a graph; a named ``seed`` is an input tensor of the state's shape the loop starts from."""
    graph = Graph()
    inputs = [("D", (steps,)), ("W", (N, N)), ("U", (steps, N, N)), *([(seed, (N, N))] if isinstance(seed, str) else [])]
    for name, shape in inputs:
        graph.add_node(InputOp(), [], Tensor(name, shape, "f32"), node_id=name)
    names = [name for name, _ in inputs]
    graph.add_node(
        LoopOp(body=_step((c, i, j), seed, steps=steps), name="k_step"), names, Tensor("out", (steps, N, N), "f32"), node_id="out"
    )
    graph.inputs, graph.outputs = names, ["out"]
    return graph


def _batched_graph(batch: int = 2, steps: int = STEPS) -> Graph:
    """The step under a FREE loop outside the carrying one — one recurrence per row of the batch,
    ``U`` and ``out`` carrying the batch coordinate first. The roll never makes this shape (its
    batch loops sit inside the step, as cells); the lift takes it all the same."""
    b = Var("b")
    graph = Graph()
    for name, shape in (("D", (steps,)), ("W", (N, N)), ("U", (batch, steps, N, N))):
        graph.add_node(InputOp(), [], Tensor(name, shape, "f32"), node_id=name)
    body = Body((Loop(axis=Axis("b", batch), body=_step((b, c, i, j), u_index=(b, c, i, j), steps=steps)),))
    graph.add_node(LoopOp(body=body, name="k_step"), ["D", "W", "U"], Tensor("out", (batch, steps, N, N), "f32"), node_id="out")
    graph.inputs, graph.outputs = ["D", "W", "U"], ["out"]
    return graph


def test_a_carrying_loop_under_an_outer_loop_keeps_that_coordinate_in_its_port() -> None:
    """The classic buffer is indexed ``(time, *outer, *cells)``, so the port the lift adds spells
    the outer coordinate too; the serial form the classic schedule lifts writes that port at that
    rank, keeps the kernel's bound I/O, and the register program declines the shape."""
    from importlib import import_module

    serial_form = import_module("emmy.compiler.pipeline.passes.tile.schedule.040_schedule").serial_form
    lifted = Pipeline.build(["tile/lift"], select=["lift"]).run(_batched_graph())
    (node,) = (node for node in lifted.nodes.values() if isinstance(node.op, TileOp))
    tile: TileOp = node.op

    assert tile.carries and [axis.extent.as_static() for axis in tile.place.free] == [2] and tile.register_program is None
    (state,) = (tensor for tensor in node.outputs if tensor.name != "out")
    assert tuple(dim.as_static() for dim in state.shape) == (STEPS, 2, N, N)
    serial = serial_form(tile, node.id)
    assert serial.inputs == tile.inputs and serial.outputs == tile.outputs
    (spec,) = (spec for spec in serial.output_specs if spec.write.output == state.name)
    (carrying,) = (site.node for site in tile.sites if site.node.carries)
    assert len(spec.write.index) == len(state.shape) and [axis.name for axis in serial.place.serial] == [carrying.axis]

    arrays = _inputs(batch=2)
    for closed in (LoopOp(body=tile.loop_body), LoopOp(body=serial.loop_body)):
        shapes = {"out": (2, STEPS, N, N), state.name: (STEPS, 2, N, N)}
        got = execute_loop_op_cpp(closed, arrays, {name: shape for name, shape in shapes.items() if name in closed.outputs})
        got = dict(zip(closed.outputs, got if isinstance(got, tuple) else (got,), strict=True))["out"]
        np.testing.assert_allclose(np.asarray(got), _reference(arrays), rtol=1e-5, atol=1e-6)


def test_the_lift_carries_the_state_in_the_term() -> None:
    """The term carries the state: the kernel root folds the action ``next`` over the steps, its
    cells the two free coordinates the block is indexed by, its seed the loop's; the step's ``W @ S``
    stays a contraction, its B a read of the carrier one step back. The node gains the state's port
    for the classic realization, and the closed program spells the carrying loop outside the cells."""
    lifted = Pipeline.build(["tile/lift"], select=["lift"]).run(_graph())
    (node,) = (node for node in lifted.nodes.values() if isinstance(node.op, TileOp))
    tile: TileOp = node.op

    assert tile.carries and not tile.place.serial and not tile.place.free
    (root,) = (site.node for site in tile.sites if site.node.carries)  # under the projection passing its streamed value on
    assert root.init == (0.0,) and len(root.cells) == 2
    assert tile.axis_of(root.axis).extent.as_static() == STEPS and all(tile.axis_of(cell).extent.as_static() == N for cell in root.cells)
    (state,) = (tensor for tensor in node.outputs if tensor.name != "out")
    assert tuple(dim.as_static() for dim in state.shape) == (STEPS, N, N) and lifted.outputs == ["out"]
    assert not [load for load in loaded_buffers(root) if load.input == state.name], "the term reads no buffer of its own"
    (mix,) = (edge for edge in root.operands if edge.axis is not None)
    assert mix.as_contraction() is not None and any(edge.as_carrier_read() is not None for edge in mix.operands)
    (carrying,) = (loop for loop in tile.loop_body if isinstance(loop, Loop))
    assert list(carrying.carries) == list(root.base.results)

    arrays = _inputs()
    closed = LoopOp(body=tile.loop_body)
    got = np.asarray(execute_loop_op_cpp(closed, arrays, {"out": (STEPS, N, N)}))
    np.testing.assert_allclose(got, _reference(arrays), rtol=1e-5, atol=1e-6)


@requires_cuda
@pytest.mark.xdist_group("cuda")
@pytest.mark.parametrize("batched", [False, True], ids=["root", "under-an-outer-loop"])
def test_the_lifted_step_runs_one_launch_per_step_on_the_gpu(batched: bool) -> None:
    from emmy.compiler.backend.cuda.program import run_program  # noqa: PLC0415

    arrays = _inputs(batch=2 if batched else None)
    result, _ = run_program(Pipeline.build(CUDA_PASSES).run(_batched_graph() if batched else _graph()), arrays)

    want = _reference(arrays)
    np.testing.assert_allclose(np.asarray(result.outputs["out"]).reshape(want.shape), want, rtol=1e-5, atol=1e-6)


def test_a_loop_over_the_cells_carries_nothing() -> None:
    """One rule places every carrier — the nearest enclosing loop its index does not read — so a
    scalar fold is carried by its own loop and a state by the loop outside its cells."""
    outer = _step((c, i, j))[0]
    over_i, over_j, over_k = _loops(outer.body)

    assert list(outer.carries) == ["S"] and outer.carries["S"] == (over_i.axis, over_j.axis)
    assert not over_i.is_reduce and not over_j.is_reduce
    assert over_k.is_reduce and not over_k.carries


def _fold_then_read() -> LoopOp:
    """``S`` summed over the steps, then a tail loop reading what the last step left — the smallest
    body with one carrying loop and one loop that carries nothing."""
    fold = Loop(
        axis=Axis("c", STEPS),
        body=(
            Loop(
                axis=Axis("i", N),
                body=(
                    Load(name="u", input="U", index=(c, i)),
                    Pre(name="own", carrier="S", index=(i,)),
                    Assign(name="next", op="add", args=("own", "u")),
                    Carry(name="S", value="next", index=(i,), seed=0.0),
                ),
            ),
        ),
    )
    r = Var("r")
    after = Loop(axis=Axis("r", N), body=(Pre(name="last", carrier="S", index=(r,)), Write(output="out", index=(r,), value="last")))
    return LoopOp(body=Body((fold, after)))


def test_the_last_step_is_what_a_read_sees_after_the_loop_closes() -> None:
    op = _fold_then_read()

    update = np.arange(STEPS * N, dtype=np.float32).reshape(STEPS, N)
    got = np.asarray(execute_loop_op_cpp(op, {"U": update}, {"out": (N,)}))

    np.testing.assert_allclose(got, update.sum(0))


def test_a_carried_state_round_trips_the_wire() -> None:
    body = LoopOp(body=_step((c, i, j))).body

    assert decode(encode(body)) == body


def _define() -> Carry:
    return Carry(name="S", value="u", index=(i,), seed=0.0)


@pytest.mark.parametrize(
    ("stmts", "message"),
    [
        ((_define(),), "no loop carries its cells"),  # every enclosing loop is a cell loop
        ((_define(), Assign(name="bad", op="exp", args=("S",))), "read one cell with Pre"),
        ((_define(), Pre(name="bad", carrier="S", index=(i, ZERO))), "no state with 2 cell axes"),
        ((_define(), _define()), "one statement defines a carried state"),
    ],
)
def test_validation_refuses(stmts: tuple, message: str) -> None:
    cell = Loop(axis=Axis("i", N), body=(Load(name="u", input="U", index=(c, i)), *stmts, Write(output="out", index=(c, i), value="u")))
    body = Body((cell,)) if message.startswith("no loop") else Body((Loop(axis=Axis("c", STEPS), body=(cell,)),))

    with pytest.raises(ValueError, match=message):
        LoopOp(body=body)


def _plain_fold() -> LoopOp:
    """The same shape with a scalar ``Accum`` in place of the carried state: a reduce loop with
    nothing to outline."""
    over_i = Loop(axis=Axis("i", N), body=(Load(name="u", input="U", index=(c, i)), Accum(name="total", value="u")))
    return LoopOp(body=Body((Loop(axis=Axis("c", STEPS), body=(over_i, Write(output="out", index=(c,), value="total"))),)))


def test_a_dump_outlines_the_loop_that_carries_the_state() -> None:
    """The dump says where a state lives: the header names it with its cell extents, a bar runs down
    the carrying loop's column, and the commit closes the section."""
    carrying, tail = (stmt.pretty() for stmt in _fold_then_read().body)

    assert carrying[0].endswith("  # carries acc0[4]")
    assert all(line.startswith("|") for line in carrying[1:])
    assert carrying[-1] == "|   # commit acc0"
    assert tail == ["for a2 in 0..4", "    v2 = pre acc0[a2]", "    out[a2] = v2"]


def test_a_loop_that_carries_nothing_prints_unchanged() -> None:
    assert _plain_fold().pretty_body() == (
        "    for a0 in 0..3\n"
        "        for a1 in 0..4\n"
        "            in0 = load U[a0, a1]\n"
        "            acc0 <- add(acc0, in0)\n"
        "        out[a0] = acc0"
    )
