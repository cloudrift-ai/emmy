"""CpuBackend against NumpyBackend: one graph per kernel strategy, parallel launches, 16-bit storage, runtime
sizes, unsupported pieces, determinism."""

import shutil

import numpy as np
import pytest

pytest.importorskip("llvmlite")

from emmy.compiler import dtype as dt  # noqa: E402
from emmy.compiler.backend.cpu import CpuBackend  # noqa: E402
from emmy.compiler.backend.cpu.codegen import Unsupported  # noqa: E402
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


pytestmark = pytest.mark.skipif(not _native_available(), reason="needs the runtime extension and a C linker")


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


def _run(graph: Graph, data: dict, *, threads: int = 4):
    cpu = CpuBackend(threads=threads)
    program = cpu.compile(graph)
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


@pytest.mark.parametrize("dtype", ["f32", "f16"])
@pytest.mark.parametrize("case", CASES, ids=[b.__name__.strip("_") for b, _ in CASES])
def test_matches_numpy(case, dtype):
    build, strategies = case
    graph = build(dt.get(dtype))
    data = _inputs(graph)
    program, got = _run(graph, data)
    assert sorted(k.plan.strategy for k in program.kernels.values()) == strategies
    _assert_close(got, _reference(graph, data), 1e-5 if dtype == "f32" else 2e-3)


@pytest.mark.parametrize("case", PARALLEL, ids=[s for _, s in PARALLEL])
def test_parallel_launches_match_numpy(case):
    build, strategy = case
    graph = build(dt.F32)
    data = _inputs(graph)
    program, got = _run(graph, data)
    assert [(k.plan.strategy, k.plan.parallel) for k in program.kernels.values()] == [(strategy, True)]
    _assert_close(got, _reference(graph, data), 1e-5)


def test_bf16_storage_rounds_only_at_the_store():
    graph = _graph(
        {"x": (64, 300), "y": (64, 300)},
        [(ElementwiseOp("multiply"), ["x", "y"], "m", (64, 300)), (ReduceOp("sum", -1), ["m"], "o", (64, 1))],
        dt.BF16,
    )
    data = _inputs(graph)
    _, got = _run(graph, data)
    want = (decode_bf16(data["x"]) * decode_bf16(data["y"])).sum(axis=-1, keepdims=True)
    # One bf16 step of the largest output: the sum itself accumulates in f32.
    _assert_close({"o": decode_bf16(got["o"])}, {"o": want}, 2**-7)


def test_runtime_size_one_compile():
    seq = Dim("seq_len")
    graph = _graph(
        {"x": (Dim(1), seq, Dim(128)), "w": (Dim(128),)},
        [(RmsNormOp(), ["x", "w"], "n", (Dim(1), seq, Dim(128))), (ElementwiseOp("exp"), ["n"], "o", (Dim(1), seq, Dim(128)))],
        dt.F32,
    )
    cpu = CpuBackend(threads=4)
    program = cpu.compile(graph)
    assert any("seq_len" in k.plan.sizes for k in program.kernels.values())
    for s in (1, 7, 33):
        data = _inputs(graph, {"seq_len": s})
        got = cpu.run(program, input_data=data)[0].outputs["o"]
        assert got.shape == (1, s, 128)
        np.testing.assert_allclose(got, _reference(graph, data)["o"], rtol=1e-5, atol=1e-6)


def test_missing_input_raises():
    graph = _pointwise(dt.F32)
    cpu = CpuBackend()
    with pytest.raises(KeyError):
        cpu.run(cpu.compile(graph), input_data={"x": _inputs(graph)["x"]})


def test_split_reduction_is_the_same_bits_on_any_thread_count():
    graph = _matvec(dt.F32, 4096, 256)
    data = _inputs(graph)
    outputs = []
    for threads in (1, 3, 4):
        program, got = _run(graph, data, threads=threads)
        assert [k.plan.mode for k in program.kernels.values()] == ["reduce"]
        outputs.append(got["o"])
    assert all(np.array_equal(o, outputs[0]) for o in outputs)


def test_unsupported_piece_fails_the_compile():
    # The generator has no ``pow``.
    graph = _graph(
        {"x": (64, 300), "e": (64, 1)},
        [(ReduceOp("sum", -1), ["x"], "s", (64, 1)), (ElementwiseOp("pow"), ["s", "e"], "o", (64, 1))],
        dt.F32,
    )
    with pytest.raises(Unsupported, match="pow"):
        CpuBackend().compile(graph)


def test_every_piece_takes_a_cpu_schedule_under_the_cpus_context():
    from emmy.compiler.backend.cpu.backend import _cut_everywhere  # noqa: PLC0415
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.ir.schedule.cpu import CpuSchedule  # noqa: PLC0415
    from emmy.compiler.ir.tile.ir import TileOp  # noqa: PLC0415
    from emmy.compiler.pipeline import Pipeline  # noqa: PLC0415

    ctx = Context.for_cpu(4)
    assert (ctx.backend_name, ctx.threads) == ("cpu", 4) and ctx.arch() == ctx.host_cpu != ""
    assert ctx.hardware_id().endswith("(4 threads)")
    assert ctx.structural_key() != Context.for_cpu(6).structural_key(), "a thread count is a regime of its own"
    scheduled = Pipeline.build(["tile/schedule"]).run(_cut_everywhere(_matvec(dt.F32, 512, 64), ctx), ctx=ctx)
    tiles = [node.op for node in scheduled.nodes.values() if isinstance(node.op, TileOp)]
    assert tiles and all(isinstance(tile.schedule.kernel, CpuSchedule) for tile in tiles)


def test_symbolic_rows_split_over_the_static_columns():
    """A decode step binds the rows to 1: a projection over symbolic rows splits its output columns across threads, so
    every thread streams its own slice of the weight at any row count."""
    rows = Dim("rows")
    graph = _graph({"a": (rows, Dim(256)), "b": (Dim(256), Dim(512))}, [(MatmulOp(), ["a", "b"], "o", (rows, Dim(512)))], dt.F32)
    program = CpuBackend(threads=4).compile(graph)
    assert [k.plan.extent for k in program.kernels.values()] == [512]
    for n in (1, 3):
        data = _inputs(graph, {"rows": n})
        got = CpuBackend(threads=4).run(program, input_data=data)[0].outputs["o"]
        np.testing.assert_allclose(got, _reference(graph, data)["o"], rtol=1e-4, atol=1e-4)


def test_default_threads_is_one_per_physical_core():
    import os

    from emmy.compiler.backend.cpu.runtime import default_threads

    assert 1 <= default_threads() <= (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count())


@pytest.mark.parametrize("cpu_tile", ["1x1", "2x4", "4x2", "8x1"])
def test_register_tiles_match_the_reference_with_rows_and_columns_left_over(cpu_tile):
    """A projection over symbolic rows runs in ``CPU_TILE`` register tiles: 5 rows and 72 columns (over 4 threads)
    leave rows and columns over that the tile's tail loops take one at a time."""
    from emmy.commands.trace import graph_from_code
    from emmy.compiler.pipeline.search.pins import pinned_knobs
    from emmy.compiler.trace.dynamic import build_torch_dynamic_shapes

    code = "x = torch.randn(5, 96)\nw = torch.randn(72, 96)\nF.linear(x, w)"
    graph, _, (module, args, _kwargs) = graph_from_code(code, dynamic_shapes=build_torch_dynamic_shapes([("rows", "x", 0)]))
    with pinned_knobs({"CPU_TILE": cpu_tile}):
        program = CpuBackend(threads=4).compile(graph)
    (tile,) = program.tiles.values()
    assert tile.knobs["CPU_TILE"] == cpu_tile
    data = inject_constants({name: a.numpy() for name, a in zip(graph.inputs, args, strict=True)}, program.graph)
    (got,) = CpuBackend(threads=4).run(program, input_data=data)[0].outputs.values()
    np.testing.assert_allclose(got, module(*args).detach().numpy(), rtol=1e-5, atol=1e-4)


def test_a_pin_outranks_a_measured_cpu_tile():
    """A measured row narrows the CPU schedule within the hand pins, never over them, as in the classic family."""
    from emmy.compiler.ir.schedule.cpu import CpuProblem

    (tile,) = CpuBackend(threads=2).compile(_matvec(dt.F32)).tiles.values()
    assert CpuProblem(tile, None, row={"CPU_TILE": "1x1"}).with_row({"CPU_TILE": "2x2"}).row == {"CPU_TILE": "1x1"}


@pytest.mark.parametrize("cpu_tile", ["1x1", "4x4"])
def test_nvfp4_weights_decode_inside_the_projection(cpu_tile, tmp_path):
    """An NVFP4 weight's decode stays inside its projection, reading the packed bytes and e4m3 scales, and the pair
    table lookup is the arithmetic e2m1 decode: the same values as the numpy backend on the same graph."""
    from emmy.commands.run import _bind_inputs
    from emmy.commands.trace import graph_from_code
    from emmy.compiler.loader.synthesize import quantize_and_spell
    from emmy.compiler.pipeline.search.pins import pinned_knobs
    from emmy.compiler.trace.dynamic import build_torch_dynamic_shapes

    code = "import torch.nn as nn\nlin = nn.Linear(256, 64, bias=False).half()\nx = torch.randn(5, 256, dtype=torch.float16)\nlin(x)"
    graph, _, (module, args, kwargs) = graph_from_code(code, dynamic_shapes=build_torch_dynamic_shapes([("rows", "input", 0)]))
    checkpoint, *_ = quantize_and_spell(graph, (module, args, kwargs), tmp_path, scheme="nvfp4-w4a16")
    numpy = NumpyBackend()
    data = _bind_inputs(graph, module, args, kwargs, checkpoint=checkpoint)
    (want,) = numpy.run(numpy.compile(graph), input_data=data)[0].outputs.values()
    with pinned_knobs({"CPU_TILE": cpu_tile}):
        cpu = CpuBackend(threads=4)
        program = cpu.compile(graph)
    assert len(program.kernels) == 1, "the decode is not a kernel of its own"
    (got,) = cpu.run(program, input_data=_bind_inputs(program.graph, module, args, kwargs, checkpoint=checkpoint))[0].outputs.values()
    np.testing.assert_allclose(got.astype(np.float32), want.astype(np.float32), rtol=1e-2, atol=2e-3)


def test_a_lent_weight_is_read_in_place():
    """``alias_buffer`` points a constant at host memory the caller holds, as the serving runner will for each expert:
    the next run reads it in place, and swapping it back restores the first result."""
    graph = _matvec(dt.F32, 256, 64)
    data = _inputs(graph)
    program, first = _run(graph, data, threads=2)
    other = np.ascontiguousarray(np.random.default_rng(1).standard_normal(data["b"].shape, dtype=np.float32))
    program.native.alias_buffer("b", other)
    program.native.run_once()
    np.testing.assert_allclose(program.native.outputs()["o"], data["a"] @ other, rtol=1e-4, atol=1e-4)
    program.native.alias_buffer("b", np.ascontiguousarray(data["b"]))
    program.native.run_once()
    np.testing.assert_array_equal(program.native.outputs()["o"], first["o"])
