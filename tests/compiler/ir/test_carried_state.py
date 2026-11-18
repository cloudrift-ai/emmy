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
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.loop.runner import execute_loop_op_cpp
from emmy.compiler.ir.stmt import Body, Load, Loop, Write, parse_body
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.ir import loaded_buffers
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from emmy.compiler.wire import decode, encode
from tests.compiler.helpers import requires_cuda

STEPS, N = 3, 4


def _step(out_index: str, seed: float | str = 0.0, u_index: str = "c, i, j", steps: int = STEPS, stored: str = "own") -> Body:
    """``S_c = decay_c * S_{c-1} + W @ S_{c-1} + U_c``, from ``seed``, storing each step the state it READ
    (``own``) or the one it defined (``next``, the roll's store)."""
    return parse_body(
        f"""
        for c in 0..{steps}:
            decay = load D[c]
            for i in 0..{N}:
                for j in 0..{N}:
                    for k in 0..{N}:
                        w = load W[i, k]
                        other = pre S[k, j]
                        p = multiply(w, other)
                        mixed <- add(mixed, p)
                    u = load U[{u_index}]
                    own = pre S[i, j]
                    kept = multiply(own, decay)
                    moved = add(kept, mixed)
                    next = add(moved, u)
                    S[i, j] <- next seed {seed}
                    out[{out_index}] = {stored}
        """
    )


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


def _reference(arrays: dict[str, np.ndarray], seed: float | str = 0.0, stored: str = "own") -> np.ndarray:
    if arrays["U"].ndim == 4:  # a batch of recurrences, one per leading row of U
        return np.stack([_reference({**arrays, "U": u}, seed, stored) for u in arrays["U"]])
    state = arrays[seed].copy() if isinstance(seed, str) else np.full((N, N), seed, np.float32)
    steps = arrays["U"].shape[0]
    want = np.zeros((steps, N, N), np.float32)
    for step in range(steps):
        if stored == "own":
            want[step] = state
        state = arrays["D"][step] * state + arrays["W"] @ state + arrays["U"][step]
        if stored == "next":
            want[step] = state
    return want


@pytest.mark.parametrize("step_major", [True, False], ids=["out[c,i,j]", "out[i,j,c]"])
def test_a_step_reads_other_cells_of_its_own_state(step_major: bool) -> None:
    """Both output layouts: storage order sorts the FREE loops, and sorting the loop over the steps
    under the cells would hand a step cells the previous step has not stored yet."""
    op = LoopOp(body=_step("c, i, j" if step_major else "i, j, c"))

    outer = op.body[0]
    assert outer.is_reduce and list(outer.carries) == [carry.name for carry in op.body.carries]
    assert not any(loop.carries for loop in _loops(outer.body))

    arrays = _inputs()
    got = np.asarray(execute_loop_op_cpp(op, arrays, {"out": (STEPS, N, N) if step_major else (N, N, STEPS)}))

    np.testing.assert_allclose(got if step_major else np.moveaxis(got, -1, 0), _reference(arrays), rtol=1e-5, atol=1e-6)


def _graph(seed: float | str = 0.0, steps: int = STEPS, stored: str = "own") -> Graph:
    """The step as a graph; a named ``seed`` is an input tensor of the state's shape the loop starts from."""
    graph = Graph()
    inputs = [("D", (steps,)), ("W", (N, N)), ("U", (steps, N, N)), *([(seed, (N, N))] if isinstance(seed, str) else [])]
    for name, shape in inputs:
        graph.add_node(InputOp(), [], Tensor(name, shape, "f32"), node_id=name)
    names = [name for name, _ in inputs]
    graph.add_node(
        LoopOp(body=_step("c, i, j", seed, steps=steps, stored=stored)),
        names,
        Tensor("out", (steps, N, N), "f32"),
        node_id="out",
    )
    graph.inputs, graph.outputs = names, ["out"]
    return graph


def _batched_graph(batch: int = 2, steps: int = STEPS) -> Graph:
    """The step under a FREE loop outside the carrying one — one recurrence per row of the batch,
    ``U`` and ``out`` carrying the batch coordinate first. The roll never makes this shape (its
    batch loops sit inside the step, as cells); the lift takes it all the same."""
    graph = Graph()
    for name, shape in (("D", (steps,)), ("W", (N, N)), ("U", (batch, steps, N, N))):
        graph.add_node(InputOp(), [], Tensor(name, shape, "f32"), node_id=name)
    step = _step("b, c, i, j", u_index="b, c, i, j", steps=steps)
    body = parse_body(f"for b in 0..{batch}:\n" + "\n".join("    " + line for line in str(step).splitlines()))
    graph.add_node(LoopOp(body=body), ["D", "W", "U"], Tensor("out", (batch, steps, N, N), "f32"), node_id="out")
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


def _unit_seeded_graph() -> Graph:
    """The step over a state with a size-one leading dim, started from a seed tensor of that shape: the
    Loop IR spells that dim's coordinate 0 in every index over the state, as the Sinkhorn loop of
    DeepSeek V4's hyper-connections does."""
    graph = Graph()
    inputs = [("D", (STEPS,)), ("W", (N, N)), ("U", (STEPS, N, N)), ("S0", (1, N, N))]
    for name, shape in inputs:
        graph.add_node(InputOp(), [], Tensor(name, shape, "f32"), node_id=name)
    names = [name for name, _ in inputs]
    step = LoopOp(body=parse_body(str(_step("c, i, j", "S0")).replace("S[", "S[0, ")))
    graph.add_node(step, names, Tensor("out", (STEPS, N, N), "f32"), node_id="out")
    graph.inputs, graph.outputs = names, ["out"]
    return graph


def test_the_serial_form_reads_a_seed_at_every_dim_the_seed_has() -> None:
    """The lift keeps a state's cells alone and drops the coordinate a size-one dim pins to 0; the seed
    tensor keeps that dim, so the serial form reads it with a 0 there. Read at the cells alone, the
    address loses its strides and the first step starts from the wrong cells."""
    from importlib import import_module

    serial_form = import_module("emmy.compiler.pipeline.passes.tile.schedule.040_schedule").serial_form
    lifted = Pipeline.build(["tile/lift"], select=["lift"]).run(_unit_seeded_graph())
    (node,) = (node for node in lifted.nodes.values() if isinstance(node.op, TileOp))
    (state,) = (tensor for tensor in node.outputs if tensor.name != "out")
    serial = LoopOp(body=serial_form(node.op, node.id).loop_body)

    arrays = {**_inputs(), "S0": np.random.default_rng(1).standard_normal((1, N, N)).astype(np.float32)}
    shapes = {"out": (STEPS, N, N), state.name: tuple(dim.as_static() for dim in state.shape)}
    got = execute_loop_op_cpp(serial, arrays, {name: shape for name, shape in shapes.items() if name in serial.outputs})
    got = dict(zip(serial.outputs, got if isinstance(got, tuple) else (got,), strict=True))["out"]
    np.testing.assert_allclose(np.asarray(got), _reference({**arrays, "S0": arrays["S0"][0]}, "S0"), rtol=1e-5, atol=1e-6)


def test_a_state_the_term_stores_every_step_needs_no_port() -> None:
    """The roll stores a rolled recurrence's state at every step beside its ``Carry``, so that output holds
    every step: the lift adds no buffer for the state, and the serial form reads the output one step back,
    the seed before the first step. A step storing the state it READ keeps the port (the tests above)."""
    from importlib import import_module

    serial_form = import_module("emmy.compiler.pipeline.passes.tile.schedule.040_schedule").serial_form
    lifted = Pipeline.build(["tile/lift"], select=["lift"]).run(_graph(stored="next"))
    (node,) = (node for node in lifted.nodes.values() if isinstance(node.op, TileOp))
    assert node.op.carries and [tensor.name for tensor in node.outputs] == ["out"]
    serial = serial_form(node.op, node.id)
    assert {load.input for load in serial.loop_body.iter() if isinstance(load, Load)} == {"D", "W", "U", "out"}
    assert {write.output for write in serial.loop_body.iter() if isinstance(write, Write)} == {"out"}

    arrays = _inputs()
    got = execute_loop_op_cpp(LoopOp(body=serial.loop_body), arrays, {"out": (STEPS, N, N)})
    np.testing.assert_allclose(np.asarray(got), _reference(arrays, stored="next"), rtol=1e-5, atol=1e-6)


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


def test_the_classic_realization_is_filed_under_the_kernel_its_fork_was_offered() -> None:
    """The classic schedule runs the serial form, a term of its own, but the kernel stays the lifted
    one — the tile its schedule fork was offered, the nearest unscheduled tile on the rewrite chain
    (``kernel_tile``): its measurements are filed under that tile's identity, so a measured row vouches
    at the fork that offered the schedule, and the definition a row stores computes that identity again."""
    from emmy.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415
    from emmy.compiler.pipeline.search.bench_record import kernel_row  # noqa: PLC0415
    from emmy.compiler.pipeline.search.dataset import KernelDef  # noqa: PLC0415
    from emmy.compiler.wire import kernel_identity, kernel_tile  # noqa: PLC0415

    lifted = Pipeline.build(["tile/lift"], select=["lift"]).run(_graph())
    (tile,) = (node.op.with_io(lifted, node) for node in lifted.nodes.values() if isinstance(node.op, TileOp))
    compiled = Pipeline.build(CUDA_PASSES).run(_graph())
    (op,) = (node.op for node in compiled.nodes.values() if isinstance(node.op, CudaOp))
    scheduled = next(ancestor for ancestor in op.source_chain() if isinstance(ancestor, TileOp))

    assert scheduled.place.serial, "the classic schedule realizes the serial form"
    identity = tile.identity_key(structural=False, with_io=True)
    assert scheduled.identity_key(structural=False, with_io=True) != identity, "the serial form is another term"
    assert kernel_tile(op).schedule is None and kernel_identity(op) == kernel_identity(scheduled) == identity
    row = kernel_row(kernel_tile(op))
    assert KernelDef(loop_ir=row.loop_ir, formed=row.formed).exact_identity == identity


@requires_cuda
@pytest.mark.xdist_group("cuda")
@pytest.mark.parametrize(
    ("batched", "stored"), [(False, "own"), (True, "own"), (False, "next")], ids=["root", "under-an-outer-loop", "no-port"]
)
def test_the_lifted_step_runs_one_launch_per_step_on_the_gpu(batched: bool, stored: str) -> None:
    from emmy.compiler.backend.cuda.program import run_program  # noqa: PLC0415

    arrays = _inputs(batch=2 if batched else None)
    result, _ = run_program(Pipeline.build(CUDA_PASSES).run(_batched_graph() if batched else _graph(stored=stored)), arrays)

    want = _reference(arrays, stored=stored)
    np.testing.assert_allclose(np.asarray(result.outputs["out"]).reshape(want.shape), want, rtol=1e-5, atol=1e-6)


def _fold_then_read() -> LoopOp:
    """``S`` summed over the steps, then a tail loop reading what the last step left — the smallest
    body with one carrying loop and one loop that carries nothing."""
    return LoopOp(
        body=parse_body(
            """
            for c in 0..3:
                for i in 0..4:
                    u = load U[c, i]
                    own = pre S[i]
                    next = add(own, u)
                    S[i] <- next seed 0
            for r in 0..4:
                last = pre S[r]
                out[r] = last
            """
        )
    )


def test_the_last_step_is_what_a_read_sees_after_the_loop_closes() -> None:
    op = _fold_then_read()

    update = np.arange(STEPS * N, dtype=np.float32).reshape(STEPS, N)
    got = np.asarray(execute_loop_op_cpp(op, {"U": update}, {"out": (N,)}))

    np.testing.assert_allclose(got, update.sum(0))


def test_a_carried_state_round_trips_the_wire() -> None:
    body = LoopOp(body=_step("c, i, j")).body

    assert decode(encode(body)) == body


_DEFINE = "S[i] <- u seed 0"


@pytest.mark.parametrize(
    ("stmts", "message"),
    [
        ((_DEFINE,), "no loop carries its cells"),  # every enclosing loop is a cell loop
        ((_DEFINE, "bad = exp(S)", "bad_out[c, i] = bad"), "read one cell with Pre"),
        ((_DEFINE, "bad = pre S[i, 0]"), "no state with 2 cell axes"),
        ((_DEFINE, _DEFINE), "one statement defines a carried state"),
    ],
)
def test_validation_refuses(stmts: tuple[str, ...], message: str) -> None:
    cell = "\n".join(("for i in 0..4:", "    u = load U[c, i]", *(f"    {stmt}" for stmt in stmts), "    out[c, i] = u"))
    text = cell if message.startswith("no loop") else "for c in 0..3:\n" + "\n".join(f"    {line}" for line in cell.splitlines())

    with pytest.raises(ValueError, match=message):
        LoopOp(body=parse_body(text))


# ---- more than one level of steps ------------------------------------------------------------------ #


def _levels(inner: str = " seed=false", between: str = "") -> Body:
    return parse_body(
        f"""
        for b in 0..2:
            for c in 0..3{inner}:
                for i in 0..{N}:
                    own = pre S[i]
                    x = load X[b, c, i]
                    next = add(own, x)
                    S[i] <- next seed 0.0
                    out[b, c, i] = next
            {between}
        """
    )


def test_a_loop_that_does_not_seed_is_a_further_level_of_the_steps() -> None:
    """Two levels of 2 and 3 steps are one walk of 6: the state runs on through the inner loop from where the outer
    step left it. The same nest with a seeding inner loop is a walk started again for every outer coordinate."""
    x = np.arange(2 * 3 * N, dtype=np.float32).reshape(2, 3, N)
    body = _levels()
    outer, inner = _loops(body)[:2]
    assert list(outer.carries) == ["S"] and list(inner.carries) == ["S"] and outer.seed and not inner.seed
    got = execute_loop_op_cpp(LoopOp(body=body), {"X": x}, {"out": (2, 3, N)})
    np.testing.assert_allclose(got, x.reshape(6, N).cumsum(0).reshape(2, 3, N))

    apart = _levels(inner="")
    assert not _loops(apart)[0].carries
    np.testing.assert_allclose(execute_loop_op_cpp(LoopOp(body=apart), {"X": x}, {"out": (2, 3, N)}), x.cumsum(1))


def test_a_read_beside_the_inner_level_sees_the_state_the_inner_steps_left() -> None:
    """A statement between the levels runs once per outer step, and its read of the state follows the inner loop,
    as a read after any loop that carries a state does."""
    x = np.arange(2 * 3 * N, dtype=np.float32).reshape(2, 3, N)
    between = f"for i in 0..{N}:\n                last = pre S[i]\n                seen[b, i] = last"
    op = LoopOp(body=_levels(between=between))
    got = dict(zip(op.outputs, execute_loop_op_cpp(op, {"X": x}, {"seen": (2, N), "out": (2, 3, N)}), strict=True))
    walk = x.reshape(6, N).cumsum(0)
    np.testing.assert_allclose(got["seen"], walk[[2, 5]])


def _solve_in_blocks(n: int, width: int) -> Body:
    """Forward substitution in blocks of rows, as one kernel: the outer level walks the blocks, the inner one the
    rows of a block, summing over the block's own rows alone. The part of every row's sum over earlier blocks is
    a second state, carried by the blocks: computed once a block, beside the inner level, for the block after."""
    return parse_body(
        f"""
        let zero = 0.0
        for b in 0..{n // width}:
            for s in 0..{width} seed=false:
                for i in 0..{n}:
                    for j in 0..{n}:
                        for k in 0..{width}:
                            coefficient = pre S[{width} * b + s, {width} * b + k]
                            above = pre S[{width} * b + k, j]
                            product = multiply(coefficient, above)
                            term = select:
                                product when k < s
                                zero when true
                            own_rows <- add(own_rows, term) over k
                        lifted = pre E[s, j]
                        mine = pre S[i, j]
                        sum = add(own_rows, lifted)
                        new = add(sum, mine)
                        next = select:
                            new when i == {width} * b + s && j < {width} * b + s
                            mine when true
                        S[i, j] <- next seed A
                        out[i, j] = next
            for u in 0..{width}:
                for j in 0..{n}:
                    for k in 0..{n}:
                        coefficient = pre S[{width} * b + {width} + u < {n} ? {width} * b + {width} + u : 0, k]
                        above = pre S[k, j]
                        product = multiply(coefficient, above)
                        term = select:
                            product when k < {width} * b + {width}
                            zero when true
                        earlier_rows <- add(earlier_rows, term) over k
                    E[u, j] <- earlier_rows seed 0.0
        """
    )


def _system(n: int) -> np.ndarray:
    return np.tril(np.random.default_rng(0).standard_normal((n, n)) * 0.3, -1).astype(np.float32)


def test_a_solve_in_blocks_is_one_kernel() -> None:
    n = 16
    a = _system(n)
    got = execute_loop_op_cpp(LoopOp(body=_solve_in_blocks(n, 4)), {"A": a}, {"out": (n, n)})
    np.testing.assert_allclose(got + np.eye(n), np.linalg.inv(np.eye(n) - a), rtol=1e-4, atol=1e-5)


def _kernel(body: Body, inputs: dict[str, tuple[int, ...]], out: tuple[int, ...]) -> Graph:
    graph = Graph()
    for name, shape in inputs.items():
        graph.add_node(InputOp(), [], Tensor(name, shape, "f32"), node_id=name)
    graph.add_node(LoopOp(body=body), list(inputs), Tensor("out", out, "f32"), node_id="out")
    graph.inputs, graph.outputs = list(inputs), ["out"]
    return graph


def test_a_further_level_of_steps_is_a_fold_under_the_fold() -> None:
    """The lift reads the inner level as a fold of its own, an operand of the fold over the outer level, which
    carries its state on and does not seed. Lowered, the kernel is the two levels again, and still the solve."""
    from emmy.compiler.ir.tile.ops import head  # noqa: PLC0415

    n = 16
    lifted = Pipeline.build(["tile/lift"], select=["lift"]).run(_kernel(_solve_in_blocks(n, 4), {"A": (n, n)}, (n, n)))
    (node,) = (node for node in lifted.nodes.values() if isinstance(node.op, TileOp))
    outer = head(node.op.op)
    (inner,) = (operand for operand in outer.operands if operand.carries)
    assert outer.carries and outer.seed and not inner.seed and len(inner.cells) == len(outer.cells) == 2
    assert set(inner.cells).isdisjoint(outer.cells), "the matrix's cells and the workspace's are two nests"

    lowered = LoopOp(body=node.op.loop_body)
    blocks, rows = _loops(lowered.body)[:2]
    assert sorted(blocks.carries) == sorted(["acc0", "acc1"]) and list(rows.carries) == ["acc0"] and blocks.seed and not rows.seed
    a = _system(n)
    shapes = {tensor.name: tuple(dim.as_static() for dim in tensor.shape) for tensor in node.outputs}
    result = execute_loop_op_cpp(lowered, {"A": a}, shapes)
    got = dict(zip(lowered.outputs, result if isinstance(result, tuple) else (result,), strict=True))
    (out,) = (value for name, value in got.items() if name.startswith("out"))
    np.testing.assert_allclose(out.reshape(n, n) + np.eye(n), np.linalg.inv(np.eye(n) - a), rtol=1e-4, atol=1e-5)


def _run_kernels(graph: Graph, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every kernel of ``graph`` run from its Loop IR, in order: each buffer by its name."""
    values = dict(arrays)
    for node_id in graph.topological_order():
        node = graph.nodes[node_id]
        if not isinstance(node.op, TileOp):
            continue
        op = LoopOp(body=node.op.loop_body)
        shapes = {tensor.name: tuple(dim.as_static() for dim in tensor.shape) for tensor in node.outputs}
        result = execute_loop_op_cpp(op, {name: values[name] for name in node.inputs}, {name: shapes[name] for name in op.outputs})
        values.update(zip(op.outputs, result if isinstance(result, tuple) else (result,), strict=True))
    return values


@pytest.mark.parametrize(("n", "width"), [(16, 4), (16, 8), (16, 16)], ids=["four-blocks", "two-blocks", "one-block"])
def test_the_cut_runs_the_outer_level_as_kernels_in_order(n: int, width: int) -> None:
    """One kernel's launches are one level of steps, so the cut unrolls the outer level: per block a kernel that
    carries the state through the block's rows, and between two blocks a kernel that carries nothing and computes
    the workspace. Every one is a kernel scheduling already takes."""
    blocks = n // width
    cut = Pipeline.build(["tile/lift", "tile/cut"], select=["lift", "cut"]).run(_kernel(_solve_in_blocks(n, width), {"A": (n, n)}, (n, n)))
    kernels = [node for node in cut.nodes.values() if isinstance(node.op, TileOp)]
    walks = [node for node in kernels if node.op.carries]
    assert len(kernels) == 2 * blocks - 1 and len(walks) == blocks
    for node in walks:
        (steps,) = (stmt for stmt in node.op.loop_body.iter() if isinstance(stmt, Loop) and stmt.carries)
        assert steps.seed and steps.axis.extent.as_static() == width, "a level of its own: it seeds, and walks one block"

    a = _system(n)
    got = _run_kernels(cut, {"A": a})["out"]
    np.testing.assert_allclose(got + np.eye(n), np.linalg.inv(np.eye(n) - a), rtol=1e-4, atol=1e-5)
    scheduled = Pipeline.build(["tile/schedule", "lowering/kernel"]).run(cut)
    assert not any(isinstance(node.op, TileOp) for node in scheduled.nodes.values())


@requires_cuda
@pytest.mark.xdist_group("cuda")
def test_a_solve_in_blocks_runs_on_the_gpu() -> None:
    from emmy.compiler.backend.cuda.program import run_program  # noqa: PLC0415

    n = 32
    a = _system(n)
    passes = CUDA_PASSES[CUDA_PASSES.index("tile/lift") :]
    result, _ = run_program(Pipeline.build(passes).run(_kernel(_solve_in_blocks(n, 8), {"A": (n, n)}, (n, n))), {"A": a})
    got = np.asarray(result.outputs["out"]).reshape(n, n)
    np.testing.assert_allclose(got + np.eye(n), np.linalg.inv(np.eye(n) - a), rtol=1e-4, atol=1e-5)


def test_what_is_not_built_for_a_further_level_is_refused_by_name() -> None:
    """A fold that runs on carries a state. A level with no state of its own beside the level under it does not
    lift."""
    from emmy.compiler.ir.tile.ops import head  # noqa: PLC0415

    with pytest.raises(ValueError, match="no state of its own"):
        Pipeline.build(["tile/lift"], select=["lift"]).run(_kernel(_levels(), {"X": (2, 3, N)}, (2, 3, N)))

    lifted = Pipeline.build(["tile/lift"], select=["lift"]).run(
        _kernel(_step("c, i, j"), {"D": (STEPS,), "W": (N, N), "U": (STEPS, N, N)}, (STEPS, N, N))
    )
    (node,) = (node for node in lifted.nodes.values() if isinstance(node.op, TileOp))
    walk = head(node.op.op)
    (product,) = (operand for operand in walk.operands if operand.axis is not None)
    assert walk.seed and walk.replace(seed=False) != walk and "seed=False" in repr(walk.replace(seed=False))
    with pytest.raises(ValueError, match="carries a state"):
        product.replace(seed=False)
