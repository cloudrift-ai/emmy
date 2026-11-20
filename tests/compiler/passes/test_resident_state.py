"""A carried state kept in register fragments (``STAGE=d1/reg``): the steps as one loop inside the CTA."""

import numpy as np
import pytest

from emmy.compiler.backend.cuda.backend import CudaBackend
from emmy.compiler.context import Context
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.atom import ATOM_REGISTRY
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.kernel.ir import FragmentApply, MmaSyncPtx, RegStore
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Loop, Write, parse_body
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from emmy.compiler.pipeline.passes.lowering.kernel._resident import Unresident, resident_walk
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import compile_with_kernels, requires_cuda, statements

_HEADS, _STEPS = 2, 3


def _graph(n: int = 32) -> Graph:
    """A delta rule over ``_STEPS`` tokens: ``S = clamp(g) * S + k (v - k S) / 16``, every step's state stored."""
    body = parse_body(
        f"""
for a0 in 0..{_STEPS}:
  for a1 in 0..{_HEADS}:
    in0 = load g[a1, a0]
    v0 = maximum(0.0, in0)
    v1 = minimum(0.5, v0)
    for a2 in 0..{n}:
      in1 = load k[a1, a0, a2]
      v2 = 0.0625 * in1
      for a3 in 0..{n}:
        for a4 in 0..{n}:
          in2 = load k[a1, a0, a4]
          v3 = 0.0625 * in2
          v4 = pre acc0[a1, a4, a3]
          v5 = v3 * v4
          acc1 <- add(acc1, v5) over a4
        in3 = load v[a1, a0, a3]
        v6 = in3 - acc1
        v7 = v6 * v2
        v8 = pre acc0[a1, a2, a3]
        v9 = v1 * v8
        v10 = v7 + v9
        out[a0, a1, a2, a3] = v10
        acc0[a1, a2, a3] <- v10 seed s
"""
    )
    graph = Graph()
    shapes = {"g": (_HEADS, _STEPS), "k": (_HEADS, _STEPS, n), "s": (_HEADS, n, n), "v": (_HEADS, _STEPS, n)}
    for name, shape in shapes.items():
        graph.add_node(InputOp(), [], Tensor(name, shape, F32), node_id=name)
    graph.add_node(LoopOp(body=body), list(shapes), Tensor("out", (_STEPS, _HEADS, n, n), F32), node_id="out")
    graph.inputs, graph.outputs = list(shapes), ["out"]
    return graph


def _row(n: int, atom: str = "mma_m16n8k16_f16_f32") -> dict:
    return {"FAST_MATH": True, "PLACE": "fuse", "WORK": "w2x1", "STAGE": "d1/reg", "TILE": f"{atom}/f1x{n // 8}/k4"}


def _feed(n: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "g": rng.uniform(-0.2, 0.8, (_HEADS, _STEPS)).astype(np.float32),
        "k": (rng.standard_normal((_HEADS, _STEPS, n)) * 0.3).astype(np.float32),
        "s": (rng.standard_normal((_HEADS, n, n)) * 0.3).astype(np.float32),
        "v": (rng.standard_normal((_HEADS, _STEPS, n)) * 0.3).astype(np.float32),
    }


def _reference(feed: dict) -> np.ndarray:
    state, out = feed["s"].astype(np.float64), []
    for t in range(_STEPS):
        k, v = feed["k"][:, t].astype(np.float64) / 16, feed["v"][:, t].astype(np.float64)
        gate = np.clip(feed["g"][:, t].astype(np.float64), 0.0, 0.5)[:, None, None]
        read = np.einsum("hk,hkv->hv", k, state)
        state = k[:, :, None] * (v - read)[:, None, :] + gate * state
        out.append(state)
    return np.stack(out)


def _lifted(n: int) -> TileOp:
    graph = Pipeline.build(["tile/lift"], select=["lift"]).run(_graph(n))
    (tile,) = (node.op for node in graph.nodes.values() if isinstance(node.op, TileOp))
    return tile


def test_the_steps_run_as_one_loop_and_the_state_has_no_store():
    """The kernel holds the step loop itself, so the launch takes no step coordinate; it stores the output every
    step and never the state; and the node keeps the output alone."""
    n = 32
    with pinned_knobs(_row(n)):
        compiled, kernels = compile_with_kernels(_graph(n), ctx=Context.from_target((12, 0)))
    (kernel,) = kernels.values()
    (loop,) = [stmt for stmt in statements(kernel, Loop) if stmt.axis.name == "a0"]
    assert not kernel.serial and not statements(kernel, Write)
    assert {store.dst_buffer for store in statements(kernel, RegStore)} == {"out"}
    assert isinstance(loop.body[-1], FragmentApply) and loop.body[-1].in_place, "the state's fragments are overwritten last"
    assert [tensor.name for node in compiled.nodes.values() if node.id == "out" for tensor in node.outputs] == ["out"]


def test_a_value_that_does_not_read_the_column_is_computed_once():
    """``k S`` reads the state's row and no column, so it is one product for every column block of the state:
    its atom steps number the product's axis over ``atom_k``, not that times the column blocks."""
    n, atom = 32, ATOM_REGISTRY["mma_m16n8k16_f16_f32"]
    body = resident_walk(_lifted(n), atom, 2, 4).body
    assert len(list(body.iter_of_type(MmaSyncPtx))) == n // atom.atom_k


def test_a_state_the_fragments_do_not_tile_is_not_offered():
    """Sixteen rows a warp and eight columns a fragment: a state of 24 rows has no register row, and the builder
    says why."""
    assert _lifted(24).resident_state is None
    with pytest.raises(Unresident, match="whole fragments"):
        resident_walk(_lifted(24), ATOM_REGISTRY["mma_m16n8k16_f16_f32"], 2, 4)


@requires_cuda
@pytest.mark.parametrize("atom", ["mma_m16n8k16_f16_f32", "mma_m16n8k16_f16_f16"])
@pytest.mark.parametrize("n", [32, 64])
def test_a_state_in_registers_matches_the_recurrence(atom, n):
    with pinned_knobs(_row(n, atom)):
        compiled = Pipeline.build(CUDA_PASSES).run(_graph(n), ctx=Context.probe())
    backend = CudaBackend()
    for seed in (0, 1):
        feed = _feed(n, seed)
        got = np.asarray(backend.run(compiled, input_data=dict(feed))[0].outputs["out"], dtype=np.float64)
        np.testing.assert_allclose(got, _reference(feed), atol=2e-3)
