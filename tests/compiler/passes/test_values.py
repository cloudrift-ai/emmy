"""A body may define a value once and read it by index at two or more places.

A ``Carry`` without a seed defines a value over the loops around it and a ``Pre`` reads one cell of it. No tier
keeps such a value for its readers, so a kernel computes it where each reader reads it (``computed_where_read``):
the kernel is the one the body that writes the value out at each reader gives.
"""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.loop.runner import execute_loop_op_cpp
from emmy.compiler.ir.stmt import Accum, Carry, Load, Pre, parse_body
from emmy.compiler.pipeline import Pipeline
from emmy.compiler.pipeline.passes.tile._fromloop import computed_where_read, lift_loop_op
from emmy.compiler.pipeline.passes.tile._row import lift_kernel
from tests.compiler.helpers import requires_cuda

# The scaled keys of a product with itself: read as its rows and as its columns.
READ_TWICE = """
for h in 0..4:
  for n in 0..8:
    for m in 0..8:
      for d in 0..16:
        b = pre kn[h, n, d]
        c = pre kn[h, m, d]
        p = multiply(b, c)
        s <- add(s, p) over d
      out[h, n, m] = s
for g in 0..4:
  for a in 0..8:
    for e in 0..16:
      k = load K[g, a, e]
      w = load W[g]
      v = multiply(k, w)
      kn[g, a, e] <- v
"""

COMPUTED_AT_EACH_READ = """
for h in 0..4:
  for n in 0..8:
    for m in 0..8:
      for d in 0..16:
        k0 = load K[h, n, d]
        w = load W[h]
        b = multiply(k0, w)
        k1 = load K[h, m, d]
        c = multiply(k1, w)
        p = multiply(b, c)
        s <- add(s, p) over d
      out[h, n, m] = s
"""

# A softmax written with its exponentials defined once: the sum over a row reads them, and so does each cell.
SOFTMAX = """
for i in 0..4:
  for k in 0..8:
    x = load X[i, k]
    m <- maximum(m, x) over k
  for j in 0..8:
    y = load X[i, j]
    d = subtract(y, m)
    e = exp(d)
    ex[i, j] <- e
for r in 0..4:
  for c in 0..8:
    a = pre ex[r, c]
    s <- add(s, a) over c
  for o in 0..8:
    b = pre ex[r, o]
    q = divide(b, s)
    out[r, o] = q
"""

_SHAPES = {"K": (4, 8, 16), "W": (4,), "X": (4, 8)}


def _run(body, outputs: dict[str, tuple[int, ...]]) -> np.ndarray:
    op = LoopOp(body=body)
    rng = np.random.default_rng(0)
    arrays = {name: rng.standard_normal(_SHAPES[name]).astype(np.float32) for name in op.inputs}
    return execute_loop_op_cpp(op, arrays, outputs)


def test_a_body_that_defines_a_value_is_resolved_to_the_body_that_computes_it_at_each_read() -> None:
    resolved = computed_where_read(LoopOp(body=parse_body(READ_TWICE)).body, {})
    assert not any(isinstance(stmt, (Carry, Pre)) for stmt in resolved.iter())
    assert sum(isinstance(stmt, Load) and stmt.input == "K" for stmt in resolved.iter()) == 2
    assert resolved == LoopOp(body=parse_body(COMPUTED_AT_EACH_READ)).body


def test_a_body_that_defines_no_value_is_the_body_itself() -> None:
    body = LoopOp(body=parse_body(COMPUTED_AT_EACH_READ)).body
    assert computed_where_read(body, {}) is body


def test_a_term_two_reads_take_at_the_same_coordinates_is_computed_once() -> None:
    """The row maximum stands under the sum over a row and under each cell of it: one reduction, not two."""
    resolved = computed_where_read(LoopOp(body=parse_body(SOFTMAX)).body, {})
    maxima = [stmt for stmt in resolved.iter() if isinstance(stmt, Accum) and stmt.op.name == "maximum"]
    assert len(maxima) == 1
    assert sum(isinstance(stmt, Load) for stmt in resolved.iter()) == 3


@pytest.mark.parametrize(("text", "output"), [(READ_TWICE, (4, 8, 8)), (SOFTMAX, (4, 8))], ids=["product", "softmax"])
def test_a_body_runs_the_same_as_written_and_resolved(text: str, output: tuple[int, ...]) -> None:
    body = LoopOp(body=parse_body(text)).body
    as_written = _run(body, {"out": output})
    resolved = _run(computed_where_read(body, {}), {"out": output})
    np.testing.assert_allclose(resolved, as_written, rtol=1e-6, atol=1e-6)


def test_the_lift_takes_a_body_that_defines_a_value_to_the_kernel_of_the_resolved_body() -> None:
    defined = lift_kernel(LoopOp(body=parse_body(READ_TWICE)))
    resolved = lift_loop_op(LoopOp(body=parse_body(COMPUTED_AT_EACH_READ)))
    assert defined.identity_key(structural=False) == resolved.identity_key(structural=False)


def _program() -> Graph:
    """A kernel written with its scaled keys defined once, and a reader of its output: ``-(kn @ knᵀ)``."""
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("K", _SHAPES["K"]), node_id="K")
    graph.add_node(InputOp(), [], Tensor("W", _SHAPES["W"]), node_id="W")
    graph.add_node(LoopOp(body=parse_body(READ_TWICE)), ["K", "W"], Tensor("out", (4, 8, 8)), node_id="out")
    negate = LoopOp(
        body=parse_body(
            """
            for h in 0..4:
              for n in 0..8:
                for m in 0..8:
                  x = load out[h, n, m]
                  y = negative(x)
                  neg[h, n, m] = y
            """
        )
    )
    graph.add_node(negate, ["out"], Tensor("neg", (4, 8, 8)), node_id="neg")
    graph.inputs, graph.outputs = ["K", "W"], ["neg"]
    return graph


def _inputs() -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(0)
    inputs = {name: rng.standard_normal(_SHAPES[name]).astype(np.float32) for name in ("K", "W")}
    scaled = inputs["K"] * inputs["W"][:, None, None]
    return inputs, -(scaled @ scaled.transpose(0, 2, 1))


def test_a_kernel_that_defines_a_value_fuses_with_its_reader() -> None:
    from emmy.compiler.backend.numpy import NumpyBackend

    fused = Pipeline.build(["loop/fusion"]).run(_program())
    (kernel,) = [node.op for node in fused.nodes.values() if isinstance(node.op, LoopOp)]
    assert not any(isinstance(stmt, (Carry, Pre)) for stmt in kernel.body.iter())
    inputs, want = _inputs()
    backend = NumpyBackend()
    got = next(iter(backend.run(backend.compile(fused), input_data=inputs)[0].outputs.values()))
    np.testing.assert_allclose(got.reshape(want.shape), want, rtol=1e-5, atol=1e-5)


@requires_cuda
@pytest.mark.xdist_group("cuda")
def test_a_kernel_that_defines_a_value_computes_what_its_program_does_on_the_card() -> None:
    from emmy.compiler.backend.cuda.backend import CudaBackend

    inputs, want = _inputs()
    backend = CudaBackend()
    got = next(iter(backend.run(backend.compile(_program()), input_data=inputs)[0].outputs.values()))
    np.testing.assert_allclose(np.asarray(got).reshape(want.shape), want, rtol=1e-4, atol=1e-4)
