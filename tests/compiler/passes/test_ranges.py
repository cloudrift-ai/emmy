"""A range, from the Loop IR body to the kernel a tier compiles.

A reduction over a triangle states the part of its axis it walks as its loop's range. The range reaches the term on
the fold's axis, and the classic schedule and the kernel lowering read it there (``TileOp.scheduled``). The lowered
kernel is held to numpy on the CPU backend.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from emmy.compiler.backend.numpy import NumpyBackend
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import parse_body
from emmy.compiler.pipeline import Pipeline
from tests.compiler.helpers import inject_constants, requires_cuda

#: ``O[i, j] = sum(X[i, k] * Y[k, j] for k <= i)``: the sum walks the keys at or before the row.
TO_THE_ROW = """
for i in 0..8:
  for j in 0..8:
    for k in 0..i + 1 of 8:
      x = load X[i, k]
      y = load Y[k, j]
      p = multiply(x, y)
      s <- add(s, p) over k
    O[i, j] = s
"""

#: The same sum over the keys at or past the row: a range that does not start at zero.
FROM_THE_ROW = """
for i in 0..8:
  for j in 0..8:
    for k in i..8 of 8:
      x = load X[i, k]
      y = load Y[k, j]
      p = multiply(x, y)
      s <- add(s, p) over k
    O[i, j] = s
"""


def _graph(text: str, inputs: dict[str, tuple[int, ...]], shape: tuple[int, ...]) -> Graph:
    graph = Graph()
    for name, dims in inputs.items():
        graph.add_node(InputOp(), [], Tensor(name, dims), node_id=name)
    graph.add_node(LoopOp(body=parse_body(text)), list(inputs), Tensor("O", shape), node_id="O")
    graph.inputs, graph.outputs = list(inputs), ["O"]
    return graph


def _tile(text: str, inputs: dict[str, tuple[int, ...]], shape: tuple[int, ...]):
    return Pipeline.build(["tile/lift"], select=["lift"]).run(_graph(text, inputs, shape)).nodes["O"].op


def _data(inputs: dict[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {name: rng.standard_normal(shape).astype(np.float32) for name, shape in inputs.items()}


def test_a_loops_range_is_in_the_body_and_on_the_fold() -> None:
    square = {"X": (8, 8), "Y": (8, 8)}
    tile = _tile(TO_THE_ROW, square, (8, 8))
    assert "for a2 in 0..a0 + 1 of 8:" in str(tile.loop_body)
    start, end = tile.op.range
    assert (str(start), str(end)) == ("0", "a0 + 1")
    assert "a0" in tile.op.free_axes, "the range is evaluated where the term is"


def test_a_term_keeps_its_range_in_the_form_a_tier_takes() -> None:
    """Only a block of a split is rewritten for a tier, to walk from zero. Any other term that walks a range is the
    term itself in the scheduled form, a start that moves with the row included: no end lies a fixed way past it,
    and the tier reads the range as it stands."""
    for text, walked in ((TO_THE_ROW, ["0", "a0 + 1"]), (FROM_THE_ROW, ["a0", "8"])):
        tile = _tile(text, {"X": (8, 8), "Y": (8, 8)}, (8, 8))
        assert [str(bound) for bound in tile.op.range] == walked
        assert tile.scheduled() is tile


@pytest.mark.parametrize(("text", "expected"), [(TO_THE_ROW, np.tril), (FROM_THE_ROW, np.triu)], ids=["to_the_row", "from_the_row"])
def test_a_ranged_fold_computes_the_sum_over_its_range(text: str, expected) -> None:
    """The body a ranged fold lowers to, run on the Loop IR runner: the loop takes the trips of the range alone."""
    from emmy.compiler.ir.loop.runner import execute_loop_op_cpp  # noqa: PLC0415

    shapes = {"X": (8, 8), "Y": (8, 8)}
    data = _data(shapes)
    op = LoopOp(body=_tile(text, shapes, (8, 8)).loop_body)
    assert "select" not in str(op.body)
    got = execute_loop_op_cpp(op, {name: data[name] for name in op.inputs}, {"O": (8, 8)})
    np.testing.assert_allclose(got, expected(data["X"]) @ data["Y"], rtol=1e-4, atol=1e-4)


def test_a_block_of_a_split_walks_the_part_of_the_folds_range_inside_it(monkeypatch) -> None:
    """A block of a cross-CTA split walks ``length * block .. length * (block + 1)``, and a fold that walked a range
    before the split walks what the two share. The schedule addresses the block from zero, with what the fold's own
    end leaves of it as the range. The partial and the finalize, run in order on the Loop IR runner, compute the
    masked product."""
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.ir.loop.runner import execute_loop_op_cpp  # noqa: PLC0415
    from emmy.compiler.ir.tile.ir import TileOp  # noqa: PLC0415
    from emmy.compiler.pipeline import TILE_PASSES  # noqa: PLC0415
    from emmy.compiler.pipeline.pipeline import Run  # noqa: PLC0415

    monkeypatch.setenv("EMMY_REDUCE", "g2k")
    shapes = {"X": (16, 64), "Y": (64, 16)}
    text = TO_THE_ROW.replace("0..8", "0..16").replace("0..i + 1 of 8", "0..4 * i + 4 of 64")
    run = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((12, 0)))
    out, _ = run.resolve(_graph(text, shapes, (16, 16)), lambda fork: next(fork.leaves()))
    assert {"O", "O__partial"} <= set(out.nodes)
    piece = str(out.nodes["O__partial"].op.loop_body)
    assert "for a3 in 0..minimum(32, 4 * a1 - 32 * a0 + 4) of 32" in piece and "load X[a1, 32 * a0 + a3]" in piece
    env = _data(shapes)
    for nid in out.topological_order():
        node = out.nodes[nid]
        if isinstance(node.op, TileOp):
            op = LoopOp(body=node.op.loop_body)
            shape = {tensor.name: tuple(dim.as_static() for dim in tensor.shape) for tensor in node.outputs}
            got = execute_loop_op_cpp(op, {buffer: env[buffer] for buffer in op.inputs}, shape)
            env.update(zip(op.outputs, got if isinstance(got, tuple) else (got,), strict=True))
    inside = np.arange(64)[None, :] < 4 * np.arange(16)[:, None] + 4
    np.testing.assert_allclose(env["O"], (env["X"] * inside) @ env["Y"], rtol=1e-4, atol=1e-4)


def _attention(queries: int, keys: int, **attrs) -> Graph:
    """``softmax(q kᵀ / sqrt(d)) v`` for two heads of ``queries`` rows over ``keys`` keys of width 8."""
    from emmy.compiler.ir.frontend.ir import SdpaOp  # noqa: PLC0415

    graph = Graph()
    for name, rows in (("q", queries), ("k", keys), ("v", keys)):
        graph.add_node(InputOp(), [], Tensor(name, (1, 2, rows, 8)), node_id=name)
    graph.add_node(SdpaOp(**attrs), ["q", "k", "v"], Tensor("o", (1, 2, queries, 8)), node_id="o")
    graph.inputs, graph.outputs = ["q", "k", "v"], ["o"]
    return graph


def _masked_attention(data: dict[str, np.ndarray], keep: np.ndarray) -> np.ndarray:
    scores = np.where(keep, data["q"] @ data["k"].transpose(0, 1, 3, 2) / np.sqrt(8.0), -np.inf)
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    return (weights / weights.sum(-1, keepdims=True)) @ data["v"]


@pytest.mark.parametrize(
    ("queries", "keys", "attrs", "walked"),
    [
        (16, 16, {"is_causal": True}, ("0", "a1 + 1")),
        (16, 16, {"is_causal": True, "sliding_window": 4}, ("maximum(0, a1 - 3)", "a1 + 1")),
        (8, 16, {"is_causal": True}, ("0", "a1 + 1")),
    ],
    ids=["causal", "window", "fewer_queries_than_keys"],
)
def test_an_attention_folds_each_rows_keys_over_a_range(queries: int, keys: int, attrs: dict, walked: tuple[str, str]) -> None:
    """Causal attention is a range of the key axis, the keys at or before the row, and a sliding window starts it
    ``window`` keys back. Fusion keeps the range on the loops it merges, the softmax fuses with its product where
    both walk the same range, and the carrier keeps it. The chunk tier lands one coordinate mask for each bound. A
    row that folds no key past the queries leaves the key axis that short."""
    from emmy.compiler.ir.loop.runner import execute_loop_op_cpp  # noqa: PLC0415
    from emmy.compiler.ir.tile.ops import head  # noqa: PLC0415
    from emmy.compiler.pipeline import LOOP_PASSES  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.lowering.kernel._atom import _outside  # noqa: PLC0415

    fused = Pipeline.build(LOOP_PASSES).run(_attention(queries, keys, **attrs))
    (kernel,) = (node.op for node in fused.nodes.values() if isinstance(node.op, LoopOp))
    assert "mask" not in " ".join(kernel.inputs), "no fill constant is an input"
    tile = Pipeline.build(["tile/lift"]).run(fused).nodes["o"].op
    carrier = head(tile.op)
    assert carrier.twist is not None and tuple(str(bound) for bound in carrier.range) == walked
    assert tile.axis_of(carrier.axis).extent.as_static() == min(queries, keys)
    assert len(_outside(carrier.range, tile.axis_of(carrier.axis), False)) == 1 + ("sliding_window" in attrs)

    data = _data({"q": (1, 2, queries, 8), "k": (1, 2, keys, 8), "v": (1, 2, keys, 8)})
    got = execute_loop_op_cpp(kernel, {name: data[name] for name in kernel.inputs}, {"o": (1, 2, queries, 8)})
    row, key = np.arange(queries)[:, None], np.arange(keys)[None, :]
    keep = (key <= row) & (key > row - attrs.get("sliding_window", keys))
    np.testing.assert_allclose(got, _masked_attention(data, keep), rtol=1e-4, atol=1e-4)


def test_a_scalar_walk_and_a_cooperative_reduce_read_a_range_as_their_loops_bounds() -> None:
    """The schedule addresses a kernel with each range left on its term, and the lowering takes it so. A scalar walk
    opens its loop over the range, for an attention's carrier and for a plain sum alike. A cooperative reduce gives
    each lane every ``coop``-th position of the range, from the lane's own start to the range's end. Neither masks
    a position inside the range."""
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.ir.stmt import Loop, pretty_body  # noqa: PLC0415
    from emmy.compiler.pipeline.search.pins import pinned_knobs  # noqa: PLC0415
    from tests.compiler.helpers import compile_with_kernels, statements  # noqa: PLC0415

    def lowered(graph: Graph, **pins):
        with pinned_knobs({"PLACE": "fuse", "TILE": "", "STAGE": "", **pins}):
            _, kernels = compile_with_kernels(graph, ctx=Context.from_target((12, 0)), db=None)
        (kernel,) = kernels.values()
        return kernel, "\n".join(pretty_body(kernel.body))

    def walks(kernel) -> list[tuple[str, ...]]:
        return [tuple(map(str, loop.walk())) for loop in statements(kernel, Loop) if loop.range is not None]

    attention = _attention(16, 16, is_causal=True)
    scalar, text = lowered(attention, WORK="", REDUCE="")
    assert walks(scalar) == [("0", "a1 + 1", "1")]
    assert "where(" not in text and "select" not in text

    triangle, text = lowered(_graph(TO_THE_ROW, {"X": (8, 8), "Y": (8, 8)}, (8, 8)), WORK="", REDUCE="")
    assert walks(triangle) == [("0", "a0 + 1", "1")]
    assert "select" not in text

    cooperative, text = lowered(attention, WORK="t32", REDUCE="coop")
    ((start, end, step),) = walks(cooperative)
    assert (end, step) == ("a1 + 1", "32") and start.endswith("_co"), (start, end, step)
    assert "where(" not in text and "select" not in text


def _native_available() -> bool:
    try:
        import llvmlite  # noqa: F401, PLC0415

        from emmy import emmy_runtime  # noqa: PLC0415
    except ImportError:
        return False
    return hasattr(emmy_runtime, "CpuExecutor") and bool(shutil.which("cc") or shutil.which("clang"))


@pytest.mark.skipif(not _native_available(), reason="needs llvmlite, the runtime extension and a C linker")
@pytest.mark.parametrize(
    ("text", "inputs", "shape"),
    [
        (TO_THE_ROW, {"X": (8, 8), "Y": (8, 8)}, (8, 8)),
        (FROM_THE_ROW, {"X": (8, 8), "Y": (8, 8)}, (8, 8)),
    ],
    ids=["to_the_row", "from_the_row"],
)
def test_the_lowered_kernel_computes_what_the_body_means(text: str, inputs: dict, shape: tuple) -> None:
    from emmy.compiler.backend.cpu import CpuBackend  # noqa: PLC0415

    data = _data(inputs)
    reference = NumpyBackend()
    compiled = reference.compile(_graph(text, inputs, shape))
    want = reference.run(compiled, input_data=inject_constants(dict(data), compiled))[0].outputs["O"]
    cpu = CpuBackend(threads=2)
    program = cpu.compile(_graph(text, inputs, shape))
    got = cpu.run(program, input_data=inject_constants(dict(data), program.graph))[0].outputs["O"]
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
    if text is TO_THE_ROW:
        np.testing.assert_allclose(want, np.tril(data["X"]) @ data["Y"], rtol=1e-4, atol=1e-4)
    if text is FROM_THE_ROW:
        np.testing.assert_allclose(want, np.triu(data["X"]) @ data["Y"], rtol=1e-4, atol=1e-4)


#: A recurrence over six steps whose sum walks a range of the step alone: ``S[t] = W[:, :t + 2] @ S[t - 1][:t + 2]``.
OVER_THE_STEP = """
for a0 in 0..6:
  for a1 in 0..8:
    for a2 in 0..64:
      for a3 in 0..a0 + 2 of 8:
        in0 = load w[a1, a3]
        v0 = load acc0[a0 - 1, a3, a2]
        v1 = in0 * v0
        acc1 <- add(acc1, v1) over a3
      O[a0, a1, a2] = acc1
      acc0[a0, a1, a2] <- acc1 seed s
"""

_STEP_INPUTS = {"w": (8, 8), "s": (8, 64)}
_COLUMN_BAND = {"PLACE": "fuse", "TILE": "", "WORK": "t32", "REDUCE": "coop-t/v2", "FAST_MATH": False}


def _column_band(text: str, inputs: dict, shape: tuple, ctx):
    """``text`` compiled for ``ctx`` under a transposed band of two columns, and why the band was not taken."""
    from emmy.compiler.ir.cuda import CudaOp  # noqa: PLC0415
    from emmy.compiler.pipeline import CUDA_PASSES  # noqa: PLC0415
    from emmy.compiler.pipeline.search.pins import pinned_knobs, unreproducible_pin_flag  # noqa: PLC0415

    with pinned_knobs(_COLUMN_BAND):
        compiled = Pipeline.build(CUDA_PASSES).run(_graph(text, inputs, shape), ctx=ctx, db=None)
    built = [dict(node.op.knobs or {}) for node in compiled.nodes.values() if isinstance(node.op, CudaOp)]
    return compiled, unreproducible_pin_flag({"REDUCE": _COLUMN_BAND["REDUCE"]}, built)


def test_a_range_over_the_step_alone_keeps_the_columns_of_a_transposed_band() -> None:
    """A lane of a transposed band folds the columns it owns in one loop, so it takes one column where the range
    reads an output axis: each column would walk its own. A rolled recurrence's range reads the step, a serial
    axis, and is one range for every cell, so the band keeps its columns there."""
    from emmy.compiler.context import Context  # noqa: PLC0415

    ctx = Context.from_target((12, 0))
    assert _column_band(OVER_THE_STEP, _STEP_INPUTS, (6, 8, 64), ctx)[1] is None
    over_the_row = TO_THE_ROW.replace("0..8", "0..64").replace("of 8", "of 64")
    assert "coop-t/v2" in _column_band(over_the_row, {"X": (64, 64), "Y": (64, 64)}, (64, 64), ctx)[1]


@requires_cuda
def test_a_transposed_band_of_two_columns_computes_a_sum_over_a_range_of_the_step() -> None:
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415
    from emmy.compiler.context import Context  # noqa: PLC0415

    feed = _data(_STEP_INPUTS)
    feed["w"] *= 0.3
    compiled, unrealized = _column_band(OVER_THE_STEP, _STEP_INPUTS, (6, 8, 64), Context.probe())
    assert unrealized is None
    got = CudaBackend().run(compiled, input_data=dict(feed))[0].outputs["O"]
    state, want = feed["s"].astype(np.float64), []
    for step in range(6):
        state = feed["w"][:, : step + 2].astype(np.float64) @ state[: step + 2]
        want.append(state)
    np.testing.assert_allclose(got, np.stack(want), rtol=1e-4, atol=1e-5)
