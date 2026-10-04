"""A CTA owns its state rows across chunks and reuses matrix results in registers."""

from dataclasses import replace

import numpy as np
import pytest

from emmy.compiler.context import Context
from emmy.compiler.graph import Graph
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.kernel.ir import FragmentPromote, FragmentRepack, LdmatrixLoad, MmaSyncPtx, RegFragment
from emmy.compiler.ir.schedule import Stage
from emmy.compiler.ir.schedule.base import ScheduleRefused
from emmy.compiler.ir.schedule.register import RegisterCodec, RegisterContext, RegisterProblem, materialize_register
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, Pipeline
from emmy.compiler.pipeline.passes.lowering.kernel._register import factorize_register
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import requires_cuda
from tests.compiler.ir.test_carried_state import _graph, _inputs, _reference, _unit_seeded_graph


def _lift(graph):
    """``graph`` lifted, each tile bound to its node's buffers — as the matcher hands a rule the op it matched."""
    lifted = Pipeline.build(["tile/lift"], select=["lift"]).run(graph)
    for node in lifted.nodes.values():
        if isinstance(node.op, TileOp):
            node.op = node.op.with_io(lifted, node)
    return lifted


def _context(tile, target=(12, 0)):
    return RegisterContext(RegisterProblem(tile, Context.from_target(target), allow_f16=True))


@pytest.mark.parametrize("target", [(7, 0), (12, 0)], ids=["volta", "modern"])
def test_register_schedule_round_trip_and_precision_gate(target):
    graph = _lift(_graph())
    (node,) = (n for n in graph.nodes.values() if isinstance(n.op, TileOp))
    tile = node.op
    context = _context(tile, target)
    assert not tuple(RegisterContext(replace(context.problem, allow_f16=False)).extensions())
    assert not tuple(RegisterContext(replace(context.problem, target=Context.from_target((6, 0)))).extensions())
    codec = RegisterCodec(context)
    for schedule in context.extensions():
        assert codec.decode(codec.encode(schedule)) == schedule
        node.op = materialize_register(tile, schedule, {})
        assert "STAGE=d1/reg" in node.op.pretty_body()
        restored = Graph.from_dict(graph.to_dict())
        assert restored.nodes[node.id].op.schedule == schedule
        assert restored.nodes[node.id].op.register_program == node.op.register_program
    row = {**codec.encode(schedule), "TILE": schedule.kernel.tile.spell().replace("/k4", "/k2")}
    assert codec.decode(row).kernel.tile.bk == 2
    invalid = replace(schedule, kernel=replace(schedule.kernel, tile=replace(schedule.kernel.tile, regs=(2, 1))))
    with pytest.raises(ScheduleRefused):
        context.extend(invalid)
    with pytest.raises(ValueError, match="one live value"):
        Stage.parse("d2/reg")


def test_register_storage_refuses_cross_warp_state_reads():
    from emmy.compiler.ir.loop import LoopOp
    from emmy.compiler.ir.stmt import Pre
    from tests.compiler.ir.test_carried_state import _step, c, i, j, k

    graph = _graph()
    body = _step((c, i, j)).map(lambda s: replace(s, index=(i, k)) if isinstance(s, Pre) and s.index == (k, j) else s)
    graph.nodes["out"].op = LoopOp(body=body)
    (tile,) = (n.op for n in _lift(graph).nodes.values() if isinstance(n.op, TileOp))
    assert tile.register_program is None


def test_register_step_keeps_a_time_coordinate_read_by_its_lift():
    from emmy.compiler.ir.expr import Var
    from emmy.compiler.ir.stmt import Assign, Body, Load

    (tile,) = (n.op for n in _lift(_graph()).nodes.values() if isinstance(n.op, TileOp))
    (carrying,) = (site.node for site in tile.sites if site.node.carries)
    lift = carrying.lift
    changed = replace(
        carrying,
        lift=replace(
            lift,
            body=Body(
                (
                    Load(name="phase", input="D", index=(Var(carrying.axis),)),
                    *[
                        replace(stmt, args=(stmt.args[0], "phase")) if isinstance(stmt, Assign) and stmt.name == lift.results[0] else stmt
                        for stmt in lift.body
                    ],
                )
            ),
        ),
    )

    def substitute(term):
        return changed if term is carrying else replace(term, operands=tuple(substitute(edge) for edge in term.operands))

    program = replace(tile, op=substitute(tile.op)).register_program
    assert program is not None
    assert program.time in program.roots[-1].free_axes


def test_a_descent_row_naming_other_families_offers_no_register_leaf():
    """A descent narrows every offered tier with the kernel's whole row: a row naming a family the register tier does
    not own describes another tier, so the register tier offers nothing for it, and a strict row must be its own."""
    (tile,) = (n.op for n in _lift(_graph()).nodes.values() if isinstance(n.op, TileOp))
    context = _context(tile)
    assert tuple(context.narrowed({}).extensions()), "the unnarrowed tier offers leaves"
    row = {"WORK": "t16x8", "TILE@map.2/inner": "f26x26", "REDUCE@map.1/inner": ""}
    for narrowing in (row, {"REDUCE@map.1/inner": ""}):
        assert not tuple(context.narrowed(narrowing).extensions())
    with pytest.raises(ValueError, match="accepts only WORK, TILE and STAGE"):
        context.narrowed(row, strict=True)


@pytest.mark.parametrize("target", [(7, 0), (12, 0)], ids=["volta", "modern"])
def test_chunk_loop_is_inside_one_launch(target):
    with pinned_knobs({"FAST_MATH": True, "STAGE": "d1/reg"}):
        graph = Pipeline.build(CUDA_PASSES).run(_graph(), ctx=Context.from_target(target))
    (op,) = (n.op for n in graph.nodes.values() if isinstance(n.op, CudaOp))
    assert not op.serial and op.smem_bytes == 0
    assert op.arg_order == ("D", "W", "U", "out")
    assert [t.name for t in graph.nodes["out"].outputs] == ["out"]
    assert f"float _state0[{8 if target == (7, 0) else 4}]" in op.kernel_source and "for (int a0" in op.kernel_source
    assert "out__acc1[" not in op.kernel_source
    assert "#include <cuda_fp16.h>" in op.kernel_source  # The buffers are FP32; the direct loader constructs FP16 operands.


def _with_output_batch_axis(graph, batch_extent):
    from emmy.compiler.dim import Dim
    from emmy.compiler.ir.expr import Literal
    from emmy.compiler.ir.stmt import Write

    node = graph.nodes["out"]
    tensor = node.outputs[0]
    tensor.shape = (tensor.shape[0], Dim(batch_extent), *tensor.shape[1:])
    node.op = replace(
        node.op,
        body=node.op.body.map(lambda s: replace(s, index=(s.index[0], Literal(0, "int"), *s.index[1:])) if isinstance(s, Write) else s),
    )
    return graph


@pytest.mark.parametrize("batch_extent", [1, 2])
def test_register_output_keeps_a_unit_batch_coordinate(batch_extent):
    graph = _with_output_batch_axis(_graph(), batch_extent)
    node = graph.nodes["out"]
    (tile,) = (n.op for n in _lift(graph).nodes.values() if isinstance(n.op, TileOp))
    assert (tile.register_program is not None) == (batch_extent == 1)
    if batch_extent == 1:
        schedule = next(iter(_context(tile).extensions()))
        node.op = materialize_register(tile, schedule, {})
        lowered = Pipeline.build(["lowering/kernel", "lowering/cuda"]).run(graph, ctx=Context.from_target((12, 0)))
        (cuda,) = (n.op for n in lowered.nodes.values() if isinstance(n.op, CudaOp))
        assert "emmy_mma_m16n8k16" in cuda.kernel_source


def test_classic_recurrence_does_not_reopen_the_launch_axis():
    with pinned_knobs({"FAST_MATH": False}):
        graph = Pipeline.build(CUDA_PASSES).run(_graph(), ctx=Context.from_target((12, 0)))
    (cuda,) = (n.op for n in graph.nodes.values() if isinstance(n.op, CudaOp))
    assert cuda.serial
    for name, _extent in cuda.serial:
        assert f"for (int {name} =" not in cuda.kernel_source


def test_register_fork_keeps_the_carrier_when_classic_is_also_offered():
    from importlib import import_module

    from emmy.compiler.pipeline.fork import iter_leaves

    classic_forks = import_module("emmy.compiler.pipeline.passes.tile.schedule.040_schedule").classic_forks
    (tile,) = (n.op for n in _lift(_graph()).nodes.values() if isinstance(n.op, TileOp))
    with pinned_knobs({"FAST_MATH": True}):
        forks = classic_forks(tile, tile.name, {}, Context.from_target((12, 0)))
        register = next(leaf for leaf in iter_leaves(forks) if leaf.knobs.get("STAGE") == "d1/reg")
        (scheduled,) = register.expand()
    assert scheduled.carries and not scheduled.place.serial
    assert scheduled.register_program == tile.register_program


@pytest.mark.parametrize("target", [(7, 0), (12, 0)], ids=["volta", "modern"])
@pytest.mark.parametrize("stride", [1, 2])
def test_register_operands_use_direct_loads_when_the_address_allows_it(target, stride):
    from emmy.compiler.ir.expr import Literal
    from emmy.compiler.ir.stmt import Load

    graph = _graph()
    weight = graph.nodes["W"].outputs[0]
    weight.shape = (weight.shape[0], weight.shape[1] * stride)
    op = graph.nodes["out"].op
    graph.nodes["out"].op = replace(
        op,
        body=op.body.map(
            lambda s: replace(s, index=(s.index[0], s.index[1] * Literal(stride, "int"))) if isinstance(s, Load) and s.input == "W" else s
        ),
    )
    (tile,) = (n.op for n in _lift(graph).nodes.values() if isinstance(n.op, TileOp))
    schedule = next(iter(_context(tile, target).extensions()))
    statements = tuple(factorize_register(materialize_register(tile, schedule, {})).body.iter())
    reads = [s for s in statements if isinstance(s, LdmatrixLoad)]
    assert bool(reads) == (stride == 1)
    assert all(s.src_buffer == "W" and s.b_trans and not s.staged and s.gmem_guard for s in reads)
    assert any(isinstance(s, FragmentRepack) and s.role == "b" for s in statements) == (stride != 1)
    assert any(isinstance(s, FragmentRepack) and s.role == "a" for s in statements)


@pytest.mark.parametrize("target", [(7, 0), (12, 0)], ids=["volta", "modern"])
def test_gdn_reuses_the_corrected_values(target):
    from emmy.commands.trace import graph_from_code
    from tests.compiler.passes.test_roll_recurrence import _GATED_DELTA_RULE

    graph = _lift(Pipeline.build(LOOP_PASSES).run(graph_from_code(_GATED_DELTA_RULE)[0]))
    (tile,) = (n.op for n in graph.nodes.values() if isinstance(n.op, TileOp) and n.op.carries)
    for schedule in _context(tile, target).extensions():
        body = factorize_register(materialize_register(tile, schedule, {})).body
        statements = list(body.iter())
        # One product computes the correction, one updates the state. The separate correction
        # output shares the first product instead of evaluating the old state a second time.
        assert sum(isinstance(s, MmaSyncPtx) for s in statements) == (3 if target == (7, 0) else 2)
        half = schedule.kernel.tile.atom.operand_dtype("c").nbytes == 2
        assert sum(isinstance(s, FragmentPromote) for s in statements) == (2 if half else 0)
        assert isinstance(body[0], RegFragment) and body[0].dtype.name == "f32"


@requires_cuda
@pytest.mark.xdist_group("cuda")
@pytest.mark.parametrize("half", [False, True], ids=["f32-acc", "f16-acc"])
@pytest.mark.parametrize("warps", [1, 2])
def test_register_state_preserves_old_reads_on_cuda(half, warps):
    from emmy.compiler.backend.cuda.program import run_program

    target = Context.probe()
    atom = ("mma_m8n8k4" if target.has_volta_mma else "mma_m16n8k16") + "_f16_" + ("f16" if half else "f32")
    with pinned_knobs({"STAGE": "d1/reg", "WORK": f"w{warps}x1", "TILE": f"{atom}/f1x1/k4"}):
        graph = Pipeline.build(CUDA_PASSES).run(_graph())
    (op,) = (n.op for n in graph.nodes.values() if isinstance(n.op, CudaOp))
    assert not op.serial
    arrays = _inputs()
    result, _ = run_program(graph, arrays)
    np.testing.assert_allclose(
        np.asarray(result.outputs["out"]).reshape(_reference(arrays).shape), _reference(arrays), rtol=2e-3, atol=2e-3
    )


@requires_cuda
@pytest.mark.xdist_group("cuda")
@pytest.mark.parametrize("unit_batch", [False, True])
@pytest.mark.parametrize("unit", [False, True], ids=["seed", "seed-with-a-size-one-dim"])
def test_register_state_starts_from_the_seed_tensor_on_cuda(unit, unit_batch):
    """A loop that starts from a tensor rather than zeros: the first step reads the seed at its cell,
    and at 0 on a size-one dim the seed has and the state's cells do not."""
    from emmy.compiler.backend.cuda.program import run_program

    atom = ("mma_m8n8k4" if Context.probe().has_volta_mma else "mma_m16n8k16") + "_f16_f32"
    source = _unit_seeded_graph() if unit else _graph(seed="S0")
    if unit_batch:
        source = _with_output_batch_axis(source, 1)
    with pinned_knobs({"STAGE": "d1/reg", "WORK": "w1x1", "TILE": f"{atom}/f1x1/k4"}):
        graph = Pipeline.build(CUDA_PASSES).run(source)
    (op,) = (n.op for n in graph.nodes.values() if isinstance(n.op, CudaOp))
    assert not op.serial and "S0" in op.arg_order
    arrays = _inputs(seed="S0")
    result, _ = run_program(graph, {**arrays, "S0": arrays["S0"][None]} if unit else arrays)
    want = _reference(arrays, seed="S0")
    np.testing.assert_allclose(np.asarray(result.outputs["out"]).reshape(want.shape), want, rtol=2e-3, atol=2e-3)


@requires_cuda
@pytest.mark.xdist_group("cuda")
@pytest.mark.parametrize("register", [False, True], ids=["classic", "register"])
def test_seed_tensor_keeps_unit_dimensions_on_cuda(register):
    from emmy.compiler.backend.cuda.program import run_program
    from emmy.compiler.dim import Dim
    from emmy.compiler.ir.expr import Literal
    from emmy.compiler.ir.stmt import Carry, Pre

    source = _graph(seed="S0")
    seed = source.nodes["S0"].outputs[0]
    seed.shape = (Dim(1), seed.shape[0], Dim(1), seed.shape[1])
    op = source.nodes["out"].op
    zero = Literal(0, "int")
    source.nodes["out"].op = replace(
        op, body=op.body.map(lambda s: replace(s, index=(zero, s.index[0], zero, s.index[1])) if isinstance(s, (Carry, Pre)) else s)
    )
    pins = {"FAST_MATH": False}
    if register:
        atom = ("mma_m8n8k4" if Context.probe().has_volta_mma else "mma_m16n8k16") + "_f16_f32"
        pins.update(STAGE="d1/reg", WORK="w1x1", TILE=f"{atom}/f1x1/k4")
    with pinned_knobs(pins):
        graph = Pipeline.build(CUDA_PASSES).run(source)
    arrays = _inputs(seed="S0")
    want = _reference(arrays, seed="S0")
    arrays["S0"] = arrays["S0"].reshape(1, *arrays["S0"].shape[:1], 1, -1)
    result, _ = run_program(graph, arrays)
    np.testing.assert_allclose(result.outputs["out"], want, rtol=2e-3 if register else 1e-5, atol=2e-3 if register else 1e-6)


@requires_cuda
@pytest.mark.xdist_group("cuda")
@pytest.mark.parametrize("shape", [(17, 80, 35), (64, 128, 128)], ids=["tails", "gdn128"])
@pytest.mark.parametrize("half", [False, True], ids=["f32-acc", "f16-acc"])
def test_gdn_chunk_step_matches_loop_on_cuda(shape, half):
    from emmy import config
    from emmy.commands.trace import graph_from_code
    from emmy.compiler.backend.cuda.program import CompiledProgram, kernel_attributes
    from emmy.compiler.backend.gpu_lock import gpu_lock
    from emmy.compiler.ir.loop import LoopOp
    from emmy.compiler.ir.loop.runner import execute_loop_op_cpp

    # The inter-chunk recurrence, with the chunk-local triangular solve supplied as inputs.
    # The real model's trace is covered above; this keeps large and uneven shape checks small.
    chunk, keys, values = shape
    code = f"""
import torch, torch.nn as nn
class Step(nn.Module):
    def forward(self, w, k, u, d):
        s = torch.zeros(2, {keys}, {values})
        states, corrected = [], []
        for c in range(4):
            v = u[:, c] - w[:, c] @ s
            s = s * d[:, c, None, None] + k[:, c].transpose(-1, -2) @ v
            states.append(s)
            corrected.append(v)
        return torch.stack(states, 1), torch.stack(corrected, 1)
m = Step()
m(torch.randn(2,4,{chunk},{keys}), torch.randn(2,4,{chunk},{keys}),
  torch.randn(2,4,{chunk},{values}), torch.rand(2,4))
"""
    lifted = _lift(Pipeline.build(LOOP_PASSES).run(graph_from_code(code)[0]))
    (node,) = (n for n in lifted.nodes.values() if isinstance(n.op, TileOp) and n.op.carries)
    tile = node.op
    context = _context(tile, Context.probe().compute_capability)
    schedule = next(
        s for s in context.extensions() if (s.kernel.tile.atom.operand_dtype("c").nbytes == 2) == half and s.kernel.work.units == (2, 1)
    )
    graph = Graph()
    rng = np.random.default_rng(0)
    arrays = {}
    for name in node.inputs:
        tensor = lifted.buffer(name)
        graph.add_node(InputOp(), [], tensor, node_id=name)
        arrays[name] = (rng.standard_normal(tuple(d.as_static() for d in tensor.shape)) * 0.1).astype(np.float32)
    graph.add_node(materialize_register(tile, schedule, {}), list(node.inputs), outputs=node.outputs, node_id=node.id)
    graph.inputs = list(node.inputs)
    graph.outputs = [s.write.output for s in tile.register_program.outputs]
    shapes = {name: tuple(d.as_static() for d in graph.buffer(name).shape) for name in node.buffer_names()}
    loop = LoopOp(body=tile.loop_body)
    expected = dict(zip(loop.outputs, execute_loop_op_cpp(loop, arrays, shapes), strict=True))
    lowered = Pipeline.build(["lowering/kernel", "lowering/cuda"]).run(graph)
    lowered.validate()
    (op,) = (n.op for n in lowered.nodes.values() if isinstance(n.op, CudaOp))
    (port,) = (t.name for t in node.outputs if t.name not in graph.outputs)  # the state's buffer, dropped with the port
    assert not op.serial and port not in op.arg_order
    # The carried state's register residency is a property of the deployable build: at the
    # correctness lane's `-Xcicc -O1` the fragment arrays stay in local memory.
    with gpu_lock(), config.nvcc_flags_override(""):
        program = CompiledProgram.build(lowered, arrays)
        program.iter_once()
        result = program.outputs()
        attributes = {name: kernel_attributes(name, spec) for name, spec in program.plan.kernels.items()}
    for name, actual in result.items():
        actual = np.asarray(actual).reshape(expected[name].shape)
        # FAST_MATH rounds matrix operands at every step, even with f32 accumulators. Bound
        # both error near cancellation and the relative error over the whole recurrence.
        np.testing.assert_allclose(actual, expected[name], rtol=3e-3, atol=1e-3)
        assert np.linalg.norm(actual - expected[name]) / np.linalg.norm(expected[name]) < 2e-3

    spills = [(a["num_regs"], a["local_size_bytes"]) for a in attributes.values()]
    if Context.probe().has_volta_mma and half and values == 128:
        # Volta's m8n8k4 C fragment holds eight f32 per thread per 8x8 tile, so the 128-column
        # state alone is 128 registers and the walk spills 64 bytes at the 255-register cap. The
        # numerics above hold; register residency of this shape needs an 8-row program on this card.
        assert all(local <= 64 for _, local in spills), spills
    else:
        assert all(local == 0 for _, local in spills), spills
