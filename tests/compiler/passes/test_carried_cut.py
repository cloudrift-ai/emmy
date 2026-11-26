"""``tile/cut/030_cut`` on a kernel that carries a state: a cone that reads no state is a seam."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from emmy.commands.trace import graph_from_code
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import LOOP_PASSES, Pipeline
from emmy.compiler.pipeline.passes.tile._cut import _buffer_reads, cuttable_seams
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import compile_with_kernels, requires_cuda

_WALK = """
import torch, torch.nn as nn
class Walk(nn.Module):
    # a state walked over four chunks; a step multiplies it by the chunk's Gram matrix and adds a product of the
    # chunk's keys and values: neither product reads the state
    def forward(self, k, v, state):
        for c in range(4):
            kc, vc = k[:, c*4:(c+1)*4], v[:, c*4:(c+1)*4]
            state = state + (kc.transpose(1, 2) @ kc) @ state + kc.transpose(1, 2) @ vc
        return state
m = Walk()
m(torch.randn(2, 16, 8) * 0.3, torch.randn(2, 16, 8) * 0.3, torch.randn(2, 8, 8))
"""
_NAMES = ("k", "v", "state")


def _tiles(graph) -> list:
    return [node for node in graph.nodes.values() if isinstance(node.op, TileOp) and node.op.op is not None]


def _walk(graph):
    """The lifted kernel that carries the state, and its seams."""
    (walk,) = (node for node in _tiles(Pipeline.build([*LOOP_PASSES, "tile/lift"]).run(graph)) if node.op.carries)
    return walk, cuttable_seams(walk.op)


def _cuts(walk, seams) -> dict[str, str]:
    """The pins that cut every seam of ``walk``, each named on its kernel."""
    sites = (seam.spelling.partition("@")[2] for seam in seams)
    return {f"PLACE@{walk.op.name}" + (f"/{site}" if site else ""): "cut" for site in sites}


def test_a_cone_that_reads_no_state_is_a_seam_of_the_kernel_that_carries_it() -> None:
    """The two products of the chunk's own operands are the same at whichever step reads them, so each is a seam.
    The product with the state runs inside the step, after the step before it, and is none."""
    walk, seams = _walk(graph_from_code(_WALK)[0])
    states = {spec.write.name for spec in walk.op.states}
    assert len(seams) == 2 and not any(states & _buffer_reads(seam.node) for seam in seams)
    assert all(walk.op.place.serial[0].name in seam.node.free_axes for seam in seams), "each reads its chunk by the step"


def test_a_cut_computes_the_cone_ahead_of_the_loop_for_every_step() -> None:
    """The kernel of a cone that reads no state carries nothing and has no serial axis: it computes every step's
    cells ahead, the step a coordinate of its workspace. The kernel that keeps the state reads them back, and keeps
    the buffer the state starts from among its inputs."""
    graph = graph_from_code(_WALK)[0]
    walk, seams = _walk(graph)
    (seed,) = (spec.write.seed for spec in walk.op.states)
    with pinned_knobs(_cuts(walk, seams)):
        cut = Pipeline.build([*LOOP_PASSES, "tile/lift", "tile/cut"]).run(graph)
    (keeps,) = (node for node in _tiles(cut) if node.op.carries)
    steps = keeps.op.place.serial[0].extent.as_static()
    workspaces = [name for name in keeps.inputs if isinstance(cut.producer(name).op, TileOp)]
    assert len(workspaces) == 2 and all(cut.buffer(name).shape[0].as_static() == steps for name in workspaces)
    ahead = [cut.producer(name).op for name in workspaces]
    assert not any(tile.carries or tile.place.serial for tile in ahead)
    assert seed in keeps.inputs and not cuttable_seams(keeps.op), "what is left of the step reads the state"


def _held(got, want: torch.Tensor) -> None:
    want = want.double().numpy()
    error = np.asarray(got).reshape(want.shape) - want
    assert float(np.sqrt(np.mean(error**2))) < 1e-4 * float(np.sqrt(np.mean(want**2)))


def test_the_walk_matches_eager_on_every_backend(run_graph) -> None:
    graph, _, (module, example, _) = graph_from_code(_WALK)
    with torch.no_grad():
        exact = module.double()(*(tensor.double() for tensor in example))
    (got,) = run_graph(graph, {name: tensor.numpy() for name, tensor in zip(_NAMES, example, strict=True)}).values()
    _held(got, exact)


@requires_cuda
@pytest.mark.xdist_group("cuda")
def test_the_cut_kernels_match_eager_on_the_gpu() -> None:
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    graph, _, (module, example, _) = graph_from_code(_WALK)
    walk, seams = _walk(graph)
    with pinned_knobs({"PLACE": "fuse", "FAST_MATH": "off"}):
        _, fused = compile_with_kernels(graph)
    with pinned_knobs({**_cuts(walk, seams), "FAST_MATH": "off"}):
        compiled, kernels = compile_with_kernels(graph)
    assert len(kernels) == len(fused) + 1, "the two cones are one kernel ahead of the walk"
    result, _ = CudaBackend().run(compiled, input_data={name: tensor.numpy() for name, tensor in zip(_NAMES, example, strict=True)})
    with torch.no_grad():
        exact = module.double()(*(tensor.double() for tensor in example))
    (got,) = result.outputs.values()
    _held(got, exact)
