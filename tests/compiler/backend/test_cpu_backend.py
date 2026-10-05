"""CpuBackend against NumpyBackend: one graph per kernel strategy, parallel launches, 16-bit storage, runtime
sizes, fallback, determinism. Every case runs through both the Python graph walk and the Rust runtime."""

import shutil

import numpy as np
import pytest

pytest.importorskip("llvmlite")

from emmy.compiler import dtype as dt  # noqa: E402
from emmy.compiler.backend.cpu import CpuBackend  # noqa: E402
from emmy.compiler.backend.numpy import NumpyBackend  # noqa: E402
from emmy.compiler.dim import Dim  # noqa: E402
from emmy.compiler.dtype import decode_bf16, encode_bf16  # noqa: E402
from emmy.compiler.graph import Graph, Tensor  # noqa: E402
from emmy.compiler.ir.base import InputOp  # noqa: E402
from emmy.compiler.ir.frontend.ir import MatmulOp, RmsNormOp, SoftmaxOp  # noqa: E402
from emmy.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp  # noqa: E402
from tests.compiler.helpers import inject_constants  # noqa: E402


def _native_available() -> bool:
    try:
        from emmy import emmy_runtime  # noqa: PLC0415
    except ImportError:
        return False
    return hasattr(emmy_runtime, "CpuExecutor") and bool(shutil.which("cc") or shutil.which("clang"))


NATIVE = _native_available()
PATHS = [
    pytest.param(False, id="python"),
    pytest.param(True, id="rust", marks=pytest.mark.skipif(not NATIVE, reason="needs the runtime extension and a C linker")),
]


def _graph(inputs: dict, nodes: list, dtype) -> Graph:
    g = Graph()
    for name, shape in inputs.items():
        g.add_node(InputOp(), [], Tensor(name, shape, dtype), node_id=name)
    for op, args, name, shape in nodes:
        g.add_node(op, args, Tensor(name, shape, dtype), node_id=name)
    g.inputs, g.outputs = list(inputs), [nodes[-1][2]]
    return g


def _pointwise(dtype, rows=64, cols=96):
    return _graph(
        {"x": (rows, cols), "y": (rows, cols)},
        [(ElementwiseOp("multiply"), ["x", "y"], "m", (rows, cols)), (ElementwiseOp("silu"), ["m"], "o", (rows, cols))],
        dtype,
    )


def _sum(dtype, n=4096):
    return _graph({"x": (1, n)}, [(ReduceOp("sum", -1), ["x"], "o", (1, 1))], dtype)


def _row_max(dtype, rows=48, cols=300):
    return _graph({"x": (rows, cols)}, [(ReduceOp("maximum", -1), ["x"], "o", (rows, 1))], dtype)


def _matmul(dtype, m=24, k=160, n=72):
    return _graph({"a": (m, k), "b": (k, n)}, [(MatmulOp(), ["a", "b"], "o", (m, n))], dtype)


def _matvec(dtype, k=512, n=384):
    return _graph({"a": (1, k), "b": (k, n)}, [(MatmulOp(), ["a", "b"], "o", (1, n))], dtype)


def _softmax(dtype):
    return _graph({"x": (16, 200)}, [(SoftmaxOp(axis=-1), ["x"], "o", (16, 200))], dtype)


def _rmsnorm(dtype):
    return _graph({"x": (1, 8, 256), "w": (256,)}, [(RmsNormOp(), ["x", "w"], "o", (1, 8, 256))], dtype)


# Each graph and the strategies its cut pieces compile to.
CASES = [
    (_pointwise, ["pointwise"]),
    (_sum, ["full reduction"]),
    (_row_max, ["row reduction"]),
    (_matmul, ["contraction"]),
    (_matvec, ["contraction, split-K"]),
    (_softmax, ["pointwise", "serial"]),
    (_rmsnorm, ["pointwise", "pointwise", "row reduction"]),
]

# Large enough that every piece splits across threads.
PARALLEL = [
    (lambda d: _pointwise(d, 512, 1024), "pointwise"),
    (lambda d: _sum(d, 1 << 20), "full reduction"),
    (lambda d: _row_max(d, 1024, 1024), "row reduction"),
    (lambda d: _matmul(d, 64, 256, 256), "contraction"),
    (lambda d: _matvec(d, 2048, 512), "contraction, split-K"),
]


def _inputs(graph: Graph, sizes: dict | None = None) -> dict:
    rng = np.random.default_rng(0)
    data = {}
    for name in graph.inputs:
        t = graph.buffer(name)
        shape = tuple(d.as_static() if d.is_static else sizes[d.value] for d in t.shape)
        values = rng.standard_normal(shape).astype(np.float32)
        data[name] = encode_bf16(values) if t.dtype.name == "bf16" else values.astype(t.dtype.np)
    return data


def _run(graph: Graph, data: dict, *, native: bool, threads: int = 4):
    cpu = CpuBackend(threads=threads, native=native)
    program = cpu.compile(graph)
    assert not program.fallbacks, program.fallbacks
    assert (program.native is not None) == native, program.native_reason
    return program, cpu.run(program, input_data=inject_constants(dict(data), program.graph))[0].outputs


def _reference(graph: Graph, data: dict) -> dict:
    ref = NumpyBackend()
    compiled = ref.compile(graph)
    return ref.run(compiled, input_data=inject_constants(dict(data), compiled))[0].outputs


def _assert_close(got: dict, want: dict, rtol: float) -> None:
    for name, expected in want.items():
        expected = np.asarray(expected, np.float32)
        actual = np.asarray(got[name], np.float32).reshape(expected.shape)
        scale = max(float(np.abs(expected).max()), 1e-30)
        assert float(np.abs(actual - expected).max()) / scale < rtol, name


@pytest.mark.parametrize("native", PATHS)
@pytest.mark.parametrize("dtype", ["f32", "f16"])
@pytest.mark.parametrize("case", CASES, ids=[b.__name__.strip("_") for b, _ in CASES])
def test_matches_numpy(case, dtype, native):
    build, strategies = case
    graph = build(dt.get(dtype))
    data = _inputs(graph)
    program, got = _run(graph, data, native=native)
    assert sorted(k.plan.strategy for k in program.kernels.values()) == strategies
    _assert_close(got, _reference(graph, data), 1e-5 if dtype == "f32" else 2e-3)


@pytest.mark.parametrize("native", PATHS)
@pytest.mark.parametrize("case", PARALLEL, ids=[s for _, s in PARALLEL])
def test_parallel_launches_match_numpy(case, native):
    build, strategy = case
    graph = build(dt.F32)
    data = _inputs(graph)
    program, got = _run(graph, data, native=native)
    assert [(k.plan.strategy, k.plan.parallel) for k in program.kernels.values()] == [(strategy, True)]
    _assert_close(got, _reference(graph, data), 1e-5)


@pytest.mark.parametrize("native", PATHS)
def test_bf16_storage_rounds_only_at_the_store(native):
    graph = _graph(
        {"x": (64, 300), "y": (64, 300)},
        [(ElementwiseOp("multiply"), ["x", "y"], "m", (64, 300)), (ReduceOp("sum", -1), ["m"], "o", (64, 1))],
        dt.BF16,
    )
    data = _inputs(graph)
    _, got = _run(graph, data, native=native)
    want = (decode_bf16(data["x"]) * decode_bf16(data["y"])).sum(axis=-1, keepdims=True)
    # One bf16 step of the largest output: the sum itself accumulates in f32.
    _assert_close({"o": decode_bf16(got["o"])}, {"o": want}, 2**-7)


@pytest.mark.parametrize("native", PATHS)
def test_runtime_size_one_compile(native):
    seq = Dim("seq_len")
    graph = _graph(
        {"x": (Dim(1), seq, Dim(128)), "w": (Dim(128),)},
        [(RmsNormOp(), ["x", "w"], "n", (Dim(1), seq, Dim(128))), (ElementwiseOp("exp"), ["n"], "o", (Dim(1), seq, Dim(128)))],
        dt.F32,
    )
    cpu = CpuBackend(threads=4, native=native)
    program = cpu.compile(graph)
    assert any("seq_len" in k.plan.sizes for k in program.kernels.values())
    for s in (1, 7, 33):
        data = _inputs(graph, {"seq_len": s})
        got = cpu.run(program, input_data=data)[0].outputs["o"]
        assert got.shape == (1, s, 128)
        np.testing.assert_allclose(got, _reference(graph, data)["o"], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("native", PATHS)
def test_missing_input_raises(native):
    graph = _pointwise(dt.F32)
    cpu = CpuBackend(native=native)
    with pytest.raises(KeyError):
        cpu.run(cpu.compile(graph), input_data={"x": _inputs(graph)["x"]})


def test_split_reduction_is_the_same_bits_on_any_thread_count_and_path():
    graph = _matvec(dt.F32, 4096, 256)
    data = _inputs(graph)
    outputs = []
    for native in [False, True] if NATIVE else [False]:
        for threads in (1, 3, 4):
            program, got = _run(graph, data, native=native, threads=threads)
            assert [k.plan.mode for k in program.kernels.values()] == ["reduce"]
            outputs.append(got["o"])
    assert all(np.array_equal(o, outputs[0]) for o in outputs)


def test_unsupported_piece_runs_through_the_loop_interpreter():
    # The generator has no ``pow``; the loop interpreter does.
    graph = _graph(
        {"x": (64, 300), "e": (64, 1)},
        [(ReduceOp("sum", -1), ["x"], "s", (64, 1)), (ElementwiseOp("pow"), ["s", "e"], "o", (64, 1))],
        dt.F32,
    )
    data = {"x": np.abs(_inputs(graph)["x"]), "e": np.full((64, 1), 0.5, np.float32)}
    cpu = CpuBackend()
    program = cpu.compile(graph)
    assert [k.plan.strategy for k in program.kernels.values()] == ["row reduction"]
    assert list(program.fallbacks) == ["o"]
    assert program.native is None and "no native kernel" in program.native_reason
    _assert_close(cpu.run(program, input_data=data)[0].outputs, _reference(graph, data), 1e-5)
