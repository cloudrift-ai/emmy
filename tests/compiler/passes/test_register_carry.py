"""A CTA owns its state rows across chunks and reuses matrix results in registers."""

from dataclasses import replace

import numpy as np
import pytest

from emmy.compiler.context import Context
from emmy.compiler.graph import Graph
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.kernel.ir import FragmentPromote, MmaSyncPtx, RegFragment
from emmy.compiler.ir.schedule.register import RegisterCodec, RegisterContext, RegisterProblem, materialize_register
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, Pipeline
from emmy.compiler.pipeline.passes.lowering.kernel._register import factorize_register
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import requires_cuda
from tests.compiler.ir.test_carried_state import _graph, _inputs, _reference


def _lift(graph):
    return Pipeline.build(["lowering/tile"], select=["lift"]).run(graph)


def _context(tile):
    return RegisterContext(RegisterProblem(tile, Context.from_target((12, 0)), allow_f16=True))


def test_register_schedule_round_trip_and_precision_gate():
    graph = _lift(_graph())
    (node,) = (n for n in graph.nodes.values() if isinstance(n.op, TileOp))
    tile = node.op
    context = _context(tile)
    assert not tuple(RegisterContext(replace(context.problem, allow_f16=False)).extensions())
    assert not tuple(RegisterContext(replace(context.problem, target=Context.from_target((7, 0)))).extensions())
    codec = RegisterCodec(context)
    for schedule in context.extensions():
        assert codec.decode(codec.encode(schedule)) == schedule
        node.op = materialize_register(tile, schedule, {})
        restored = Graph.from_dict(graph.to_dict())
        assert restored.nodes[node.id].op.schedule == schedule
        assert restored.nodes[node.id].op.register_program == node.op.register_program


def test_chunk_loop_is_inside_one_launch():
    with pinned_knobs({"FAST_MATH": True, "STAGE": "d1/reg"}):
        graph = Pipeline.build(CUDA_PASSES).run(_graph(), ctx=Context.from_target((12, 0)))
    (op,) = (n.op for n in graph.nodes.values() if isinstance(n.op, CudaOp))
    assert not op.serial and op.smem_bytes == 0
    assert "float _state0[4]" in op.kernel_source and "for (int a0" in op.kernel_source
    assert "out__acc1[" not in op.kernel_source


def test_gdn_reuses_the_corrected_values():
    from emmy.commands.trace import graph_from_code
    from tests.compiler.passes.test_roll_recurrence import _GATED_DELTA_RULE

    graph = _lift(Pipeline.build(LOOP_PASSES).run(graph_from_code(_GATED_DELTA_RULE)[0]))
    (tile,) = (n.op for n in graph.nodes.values() if isinstance(n.op, TileOp) and n.op.place.serial)
    for schedule in _context(tile).extensions():
        body = factorize_register(materialize_register(tile, schedule, {})).body
        statements = list(body.iter())
        # One product computes the correction, one updates the state. The separate correction
        # output shares the first product instead of evaluating the old state a second time.
        assert sum(isinstance(s, MmaSyncPtx) for s in statements) == 2
        half = schedule.kernel.tile.atom.operand_dtype("c").nbytes == 2
        assert sum(isinstance(s, FragmentPromote) for s in statements) == (2 if half else 0)
        assert isinstance(body[0], RegFragment) and body[0].dtype.name == "f32"


@requires_cuda
@pytest.mark.xdist_group("cuda")
@pytest.mark.parametrize("half", [False, True], ids=["f32-acc", "f16-acc"])
@pytest.mark.parametrize("warps", [1, 2])
def test_register_state_preserves_old_reads_on_cuda(half, warps):
    from emmy.compiler.backend.cuda.program import run_program

    atom = "mma_m16n8k16_f16_" + ("f16" if half else "f32")
    with pinned_knobs({"STAGE": "d1/reg", "WORK": f"w{warps}x1", "TILE": f"{atom}/f1x1/k4"}):
        graph = Pipeline.build(CUDA_PASSES).run(_graph())
    (op,) = (n.op for n in graph.nodes.values() if isinstance(n.op, CudaOp))
    assert not op.serial
    arrays = _inputs()
    result, _ = run_program(graph, arrays)
    np.testing.assert_allclose(np.asarray(result.outputs["out"]).reshape(_reference(arrays).shape),
                               _reference(arrays), rtol=2e-3, atol=2e-3)
