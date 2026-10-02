"""CpuBackend against NumpyBackend: one graph per kernel strategy, 16-bit storage, runtime sizes, fallback."""

import numpy as np
import pytest

pytest.importorskip("llvmlite")

from emmy.compiler import dtype as dt  # noqa: E402
from emmy.compiler.backend.cpu import CpuBackend  # noqa: E402
from emmy.compiler.backend.cpu import backend as cpu_backend  # noqa: E402
from emmy.compiler.backend.cpu.codegen import Unsupported  # noqa: E402
from emmy.compiler.backend.numpy import NumpyBackend  # noqa: E402
from emmy.compiler.dim import Dim  # noqa: E402
from emmy.compiler.graph import Graph, Tensor  # noqa: E402
from emmy.compiler.ir.base import InputOp  # noqa: E402
from emmy.compiler.ir.frontend.ir import MatmulOp, RmsNormOp, SoftmaxOp  # noqa: E402
from emmy.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp  # noqa: E402
from tests.compiler.helpers import inject_constants  # noqa: E402


def _graph(inputs: dict, nodes: list, dtype) -> Graph:
    g = Graph()
    for name, shape in inputs.items():
        g.add_node(InputOp(), [], Tensor(name, shape, dtype), node_id=name)
    for op, args, name, shape in nodes:
        g.add_node(op, args, Tensor(name, shape, dtype), node_id=name)
    g.inputs, g.outputs = list(inputs), [nodes[-1][2]]
    return g


def _pointwise(dtype):
    return _graph(
        {"x": (64, 96), "y": (64, 96)},
        [
            (ElementwiseOp("multiply"), ["x", "y"], "m", (64, 96)),
            (ElementwiseOp("silu"), ["m"], "o", (64, 96)),
        ],
        dtype,
    )


def _sum(dtype):
    return _graph({"x": (1, 4096)}, [(ReduceOp("sum", -1), ["x"], "o", (1, 1))], dtype)


def _row_max(dtype):
    return _graph({"x": (48, 300)}, [(ReduceOp("maximum", -1), ["x"], "o", (48, 1))], dtype)


def _matmul(dtype):
    return _graph({"a": (24, 160), "b": (160, 72)}, [(MatmulOp(), ["a", "b"], "o", (24, 72))], dtype)


def _matvec(dtype):
    return _graph({"a": (1, 512), "b": (512, 384)}, [(MatmulOp(), ["a", "b"], "o", (1, 384))], dtype)


def _softmax(dtype):
    return _graph({"x": (16, 200)}, [(SoftmaxOp(axis=-1), ["x"], "o", (16, 200))], dtype)


def _rmsnorm(dtype):
    return _graph({"x": (1, 8, 256), "w": (256,)}, [(RmsNormOp(), ["x", "w"], "o", (1, 8, 256))], dtype)


GRAPHS = [_pointwise, _sum, _row_max, _matmul, _matvec, _softmax, _rmsnorm]


def _inputs(graph: Graph, sizes: dict | None = None) -> dict:
    rng = np.random.default_rng(0)
    data = {}
    for name in graph.inputs:
        t = graph.buffer(name)
        shape = tuple(d.as_static() if d.is_static else sizes[d.value] for d in t.shape)
        data[name] = rng.standard_normal(shape).astype(t.dtype.np)
    return data


def _check(graph: Graph, data: dict, *, threads: int, rtol: float) -> None:
    cpu = CpuBackend(threads=threads)
    program = cpu.compile(graph)
    got = cpu.run(program, input_data=inject_constants(dict(data), program.graph))[0].outputs
    ref_backend = NumpyBackend()
    ref_graph = ref_backend.compile(graph)
    want = ref_backend.run(ref_graph, input_data=inject_constants(dict(data), ref_graph))[0].outputs
    for name, expected in want.items():
        expected = np.asarray(expected, np.float32)
        actual = np.asarray(got[name], np.float32).reshape(expected.shape)
        scale = max(float(np.abs(expected).max()), 1e-30)
        assert float(np.abs(actual - expected).max()) / scale < rtol, name


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("dtype", ["f32", "f16"])
@pytest.mark.parametrize("build", GRAPHS, ids=[b.__name__.strip("_") for b in GRAPHS])
def test_matches_numpy(build, dtype, threads):
    graph = build(dt.get(dtype))
    _check(graph, _inputs(graph), threads=threads, rtol=1e-5 if dtype == "f32" else 2e-3)


def test_every_kernel_is_native():
    for build in GRAPHS:
        program = CpuBackend().compile(build(dt.F32))
        assert program.kernels and not program.fallbacks, (build.__name__, program.fallbacks)


def test_runtime_size_one_compile():
    seq = Dim("seq_len")
    graph = _graph(
        {"x": (Dim(1), seq, Dim(128)), "w": (Dim(128),)},
        [
            (RmsNormOp(), ["x", "w"], "n", (Dim(1), seq, Dim(128))),
            (ElementwiseOp("exp"), ["n"], "o", (Dim(1), seq, Dim(128))),
        ],
        dt.F32,
    )
    cpu = CpuBackend(threads=4)
    program = cpu.compile(graph)
    assert any("seq_len" in k.plan.sizes for k in program.kernels.values())
    ref = NumpyBackend()
    for s in (1, 7, 33):
        data = _inputs(graph, {"seq_len": s})
        got = cpu.run(program, input_data=data)[0].outputs["o"]
        want = ref.run(ref.compile(graph), input_data=data)[0].outputs["o"]
        assert got.shape == (1, s, 128)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_unsupported_kernel_falls_back(monkeypatch):
    def unsupported(*_args, **_kwargs):
        raise Unsupported("forced")

    monkeypatch.setattr(cpu_backend, "generate", unsupported)
    graph = _softmax(dt.F32)
    program = CpuBackend().compile(graph)
    assert not program.kernels and program.fallbacks
    _check(graph, _inputs(graph), threads=1, rtol=1e-5)
