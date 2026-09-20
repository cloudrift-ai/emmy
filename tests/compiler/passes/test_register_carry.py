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
from tests.compiler.ir.test_carried_state import _graph, _inputs, _reference


def _lift(graph):
    return Pipeline.build(["lowering/tile"], select=["lift"]).run(graph)


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
    (tile,) = (n.op for n in graph.nodes.values() if isinstance(n.op, TileOp) and n.op.place.serial)
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
@pytest.mark.parametrize("shape", [(17, 80, 35), (64, 128, 128)], ids=["tails", "gdn128"])
@pytest.mark.parametrize("half", [False, True], ids=["f32-acc", "f16-acc"])
def test_gdn_chunk_step_matches_loop_on_cuda(shape, half):
    from emmy.commands.trace import graph_from_code
    from emmy.compiler.backend.cuda.program import CompiledProgram
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
    (node,) = (n for n in lifted.nodes.values() if isinstance(n.op, TileOp) and n.op.place.serial)
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
    assert not op.serial and tile.register_program.state.write.output not in op.arg_order
    with gpu_lock():
        program = CompiledProgram.build(lowered, arrays)
        program.iter_once()
        result = program.outputs()
    for name, actual in result.items():
        actual = np.asarray(actual).reshape(expected[name].shape)
        # FAST_MATH rounds matrix operands at every step, even with f32 accumulators. Bound
        # both error near cancellation and the relative error over the whole recurrence.
        np.testing.assert_allclose(actual, expected[name], rtol=3e-3, atol=1e-3)
        assert np.linalg.norm(actual - expected[name]) / np.linalg.norm(expected[name]) < 2e-3

    assert all(kernel.local_size_bytes == 0 for kernel in program.compiled.kernels.values()), [
        (kernel.num_regs, kernel.local_size_bytes) for kernel in program.compiled.kernels.values()
    ]
