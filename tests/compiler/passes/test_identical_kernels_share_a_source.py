"""Two kernels of one exact identity are one kernel: one name, one CUDA source, each launch binding its own buffers.

The kernels below compute ``out[i] = 2 * x[i] - y[N - 1 - i]`` over buffers named and declared in different orders:
the second declares its ``y`` first. The reversed read tells the two operands apart by structure, so the declaration
order is no part of the identity. A kernel's name is its identity and its source is rendered on role slots, so both
launch the one source; the launch arguments follow each kernel's roles, never its declaration order.
"""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import parse_body
from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, Pipeline
from tests.compiler.helpers import requires_cuda

N = 256


def _graph() -> Graph:
    """``p = 2 * a - reversed(b)`` and ``q = 2 * d - reversed(c)``, each its own kernel; ``q``'s node lists ``c`` (its
    ``y``) first."""
    graph = Graph()
    for name in ("a", "b", "c", "d"):
        graph.add_node(InputOp(), [], Tensor(name, (N,), "f32"), node_id=name)
    for out, x, y, inputs in (("p", "a", "b", ["a", "b"]), ("q", "d", "c", ["c", "d"])):
        body = parse_body(
            f"""
            for i in 0..{N}:
                xv = load {x}[i]
                yv = load {y}[{N - 1} - i]
                twice = add(xv, xv)
                v = subtract(twice, yv)
                {out}[i] = v
            """
        )
        graph.add_node(LoopOp(body=body), inputs, Tensor(out, (N,), "f32"), node_id=out)
    graph.inputs, graph.outputs = ["a", "b", "c", "d"], ["p", "q"]
    return graph


def _lowered() -> Graph:
    """The graph lowered with each kernel kept apart: fusion would merge the two into one kernel."""
    passes = [p for p in CUDA_PASSES if p not in LOOP_PASSES or p == "loop/lowering"]
    return Pipeline.build(passes).run(_graph())


def test_kernels_of_one_identity_share_a_name_and_a_source() -> None:
    cuda = {node.id: node.op for node in _lowered().nodes.values() if isinstance(node.op, CudaOp)}
    p, q = cuda["p"], cuda["q"]
    assert p.kernel_name == q.kernel_name and p.kernel_source == q.kernel_source
    # One role order for both, whatever each node declared first: position by position, the same role.
    roles = {"a": "x", "b": "y", "p": "out", "d": "x", "c": "y", "q": "out"}
    assert [roles[arg] for arg in p.arg_order] == [roles[arg] for arg in q.arg_order]


@requires_cuda
@pytest.mark.xdist_group("cuda")
def test_kernels_of_one_identity_bind_their_own_buffers_on_the_gpu() -> None:
    from emmy.compiler.backend.cuda.program import run_program  # noqa: PLC0415

    rng = np.random.default_rng(0)
    arrays = {name: rng.standard_normal(N).astype(np.float32) for name in ("a", "b", "c", "d")}
    result, _ = run_program(_lowered(), arrays)

    np.testing.assert_allclose(np.asarray(result.outputs["p"]), 2 * arrays["a"] - arrays["b"][::-1], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(result.outputs["q"]), 2 * arrays["d"] - arrays["c"][::-1], rtol=1e-6)
