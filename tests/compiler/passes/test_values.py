"""A body may define a value once and read it by index at two or more places.

A ``Define`` stores one cell of a value the kernel keeps for itself and a ``Load`` reads a cell of it, the way it
reads any buffer. No tier keeps such a value for its readers, so a kernel computes it where each reader reads it
(``computed_where_read``): the kernel is the one the body that writes the value out at each reader gives.
"""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.loop.runner import execute_loop_op_cpp
from emmy.compiler.ir.stmt import Accum, Define, Load, parse_body
from emmy.compiler.pipeline import Pipeline
from emmy.compiler.pipeline.passes.tile._fromloop import computed_where_read, lift_loop_op
from emmy.compiler.pipeline.passes.tile._row import lift_kernel
from tests.compiler.helpers import requires_cuda

# The scaled keys of a product with itself: read as its rows and as its columns.
READ_TWICE = """
for g in 0..4:
  for a in 0..8:
    for e in 0..16:
      k = load K[g, a, e]
      w = load W[g]
      v = multiply(k, w)
      kn[g, a, e] <- v
for h in 0..4:
  for n in 0..8:
    for m in 0..8:
      for d in 0..16:
        b = load kn[h, n, d]
        c = load kn[h, m, d]
        p = multiply(b, c)
        s <- add(s, p) over d
      out[h, n, m] = s
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
    a = load ex[r, c]
    s <- add(s, a) over c
  for o in 0..8:
    b = load ex[r, o]
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
    assert not any(isinstance(stmt, Define) for stmt in resolved.iter())
    assert {stmt.input for stmt in resolved.iter() if isinstance(stmt, Load)} == {"K", "W"}
    assert sum(isinstance(stmt, Load) and stmt.input == "K" for stmt in resolved.iter()) == 2
    assert resolved == LoopOp(body=parse_body(COMPUTED_AT_EACH_READ)).body


def test_a_defined_value_is_a_buffer_of_the_kernel_and_no_input_or_output_of_it() -> None:
    op = LoopOp(body=parse_body(READ_TWICE))
    assert (list(op.inputs), list(op.outputs)) == (["K", "W"], ["out"])
    assert sum(isinstance(stmt, Load) and stmt.input == "kn" for stmt in op.body.iter()) == 2


def test_a_load_ahead_of_the_definition_it_reads_is_refused() -> None:
    definition, readers = READ_TWICE.split("for h in 0..4:")
    with pytest.raises(ValueError, match="ahead of its definition"):
        LoopOp(body=parse_body(f"for h in 0..4:{readers}{definition}"))


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
    assert not any(isinstance(stmt, Define) for stmt in kernel.body.iter())
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


# FlashInfer's walk over the chunks of a Gated DeltaNet layer, for one head. Per chunk ``c``, from the state ``S`` the
# chunk starts with:  w = v - k S,  u = X w,  out = q S + qk u,  S' = decay S + kT u.  ``X`` is the inverse of the
# chunk's own system, which reads no state. The correction ``u`` is read by the output and by the next state.
WALK = """
for c in 0..4:
  for r in 0..4:
    for j in 0..8:
      for i in 0..8:
        kk = load k[c, r, i]
        s0 = pre S[i, j]
        p0 = multiply(kk, s0)
        a0 <- add(a0, p0) over i
      vv = load v[c, r, j]
      d0 = subtract(vv, a0)
      w[c, r, j] <- d0
  for r2 in 0..4:
    for j2 in 0..8:
      for m in 0..4:
        x = load X[c, r2, m]
        ww = load w[c, m, j2]
        p1 = multiply(x, ww)
        a1 <- add(a1, p1) over m
      u[c, r2, j2] <- a1
  for r3 in 0..4:
    for j3 in 0..8:
      for i3 in 0..8:
        qq = load q[c, r3, i3]
        s1 = pre S[i3, j3]
        p2 = multiply(qq, s1)
        a2 <- add(a2, p2) over i3
      for m3 in 0..4:
        g = load qk[c, r3, m3]
        uu = load u[c, m3, j3]
        p3 = multiply(g, uu)
        a3 <- add(a3, p3) over m3
      o = add(a2, a3)
      out[c, r3, j3] = o
  for i4 in 0..8:
    for j4 in 0..8:
      for r4 in 0..4:
        k4 = load k[c, r4, i4]
        u4 = load u[c, r4, j4]
        p4 = multiply(k4, u4)
        a4 <- add(a4, p4) over r4
      dd = load decay[c]
      s2 = pre S[i4, j4]
      t = multiply(dd, s2)
      n = add(t, a4)
      S[i4, j4] <- n seed state0
      last[i4, j4] = n
"""

_WALK_SHAPES = {"k": (4, 4, 8), "v": (4, 4, 8), "q": (4, 4, 8), "X": (4, 4, 4), "qk": (4, 4, 4), "decay": (4,), "state0": (8, 8)}
_WALK_OUTPUTS = {"out": (4, 4, 8), "last": (8, 8)}


def _walk_arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Inputs of the walk and what it computes, in numpy."""
    rng = np.random.default_rng(0)
    arrays = {name: (0.5 * rng.standard_normal(shape)).astype(np.float32) for name, shape in _WALK_SHAPES.items()}
    state = arrays["state0"].copy()
    out = np.zeros(_WALK_OUTPUTS["out"], np.float32)
    for chunk in range(4):
        correction = arrays["X"][chunk] @ (arrays["v"][chunk] - arrays["k"][chunk] @ state)
        out[chunk] = arrays["q"][chunk] @ state + arrays["qk"][chunk] @ correction
        state = arrays["decay"][chunk] * state + arrays["k"][chunk].T @ correction
    return arrays, {"out": out, "last": state}


def test_the_walk_over_chunks_runs_as_one_body_with_its_correction_defined_once() -> None:
    """A value a step of a recurrence defines for itself stands beside the state's cells, and reads the state the
    step started from like any statement of the step."""
    arrays, want = _walk_arrays()
    op = LoopOp(body=parse_body(WALK))
    resolved = LoopOp(body=computed_where_read(op.body, {}), outputs=op.outputs)
    assert not any(isinstance(stmt, Define) for stmt in resolved.body.iter())
    for kernel in (op, resolved):
        got = dict(zip(kernel.outputs, execute_loop_op_cpp(kernel, arrays, _WALK_OUTPUTS), strict=True))
        for name, expected in want.items():
            np.testing.assert_allclose(got[name], expected, rtol=1e-4, atol=1e-5)
    assert lift_kernel(op).carries


@requires_cuda
@pytest.mark.xdist_group("cuda")
def test_the_walk_over_chunks_compiles_to_one_kernel_on_the_card() -> None:
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.ir.cuda import CudaOp

    arrays, want = _walk_arrays()
    loop = LoopOp(body=parse_body(WALK))
    graph = Graph()
    for name, shape in _WALK_SHAPES.items():
        graph.add_node(InputOp(), [], Tensor(name, shape), node_id=name)
    outputs = tuple(Tensor(name, shape) for name, shape in _WALK_OUTPUTS.items())
    graph.add_node(loop, list(loop.inputs), outputs=outputs, node_id="out")
    graph.inputs, graph.outputs = list(_WALK_SHAPES), list(_WALK_OUTPUTS)
    backend = CudaBackend()
    compiled = backend.compile(graph)
    assert sum(isinstance(node.op, CudaOp) for node in compiled.nodes.values()) == 1
    got = backend.run(compiled, input_data=arrays)[0].outputs
    for name, expected in want.items():
        np.testing.assert_allclose(np.asarray(got[name]).reshape(expected.shape), expected, rtol=1e-4, atol=1e-5)
