"""A copy is read through by every loop that reads it (``loop/fusion/008_read_through_copies``), and a kernel
no region holds stores its buffer where the one loop that reads it would only have moved it
(``loop/fusion/020_store_through_copies``).

A loop that only re-lays out or widens a buffer was stored wherever a kernel outside its region read it: for a kernel
that carries a state, which is in no region, and for a second region. Every reader now reads the buffer behind the
copy, the copy is stored for no one, and it joins no two loops into one kernel. On the other side of such a kernel,
the loop that puts its outputs where the program wants them, rounded to the dtype it wants them at, was a kernel of
its own.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from emmy.commands.trace import graph_from_code
from emmy.compiler.dtype import BF16, F32
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Assign, parse_body
from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, Pipeline, RuleSkipped
from tests.compiler.helpers import requires_cuda

_RULE = importlib.import_module("emmy.compiler.pipeline.passes.loop.fusion.008_read_through_copies")
_STORE = importlib.import_module("emmy.compiler.pipeline.passes.loop.fusion.020_store_through_copies")

_WALK = """
import torch, torch.nn as nn
class Walk(nn.Module):
    # a state carried over the chunks of a sequence whose keys and values arrive token by token{cast}
    def forward(self, q, k, v):
        B, T, H, D = k.shape
        C = {chunk}
        chunked = lambda x: x{widen}.transpose(1, 2).reshape(B, H, T // C, C, D)
        q, k, v = chunked(q), chunked(k), chunked(v)
        S = torch.zeros(B, H, D, D)
        outs = []
        for c in range(T // C):
            outs.append(q[:, :, c] @ S)
            S = S * 0.5 + k[:, :, c].transpose(-1, -2) @ v[:, :, c]
        return torch.stack(outs, 2)
m = Walk()
m(torch.randn(1, {t}, 2, 8){dtype}, torch.randn(1, {t}, 2, 8){dtype}, torch.randn(1, {t}, 2, 8){dtype})
"""


def _walk(chunk: int = 4, t: int = 16, bf16: bool = False) -> str:
    cast, widen, dtype = (", as bf16", ".float()", ".to(torch.bfloat16)") if bf16 else ("", "", "")
    return _WALK.format(chunk=chunk, t=t, cast=cast, widen=widen, dtype=dtype)


def _loops(graph) -> list:
    return [node for node in graph.nodes.values() if isinstance(node.op, LoopOp)]


@pytest.mark.parametrize("bf16", [False, True], ids=["f32", "bf16"])
def test_a_kernel_that_carries_a_state_reads_its_operands_where_they_are_stored(bf16: bool) -> None:
    graph, _, _ = graph_from_code(_walk(bf16=bf16))
    graph = Pipeline.build(LOOP_PASSES).run(graph)

    (walk,) = (node for node in _loops(graph) if node.op.body.carries)
    # Every operand is the traced input, read at the token its chunk and step name: no kernel lays the chunks out
    # for the walk, and none widens them.
    operands = [load for load in walk.op.body.loads if load.input in ("q", "k", "v")]
    assert {load.input for load in operands} == {"q", "k", "v"} and all(len(load.index) == 4 for load in operands)
    # The last chunk's output is computed from the stored state by a product of its own; it reads its queries the
    # same way. Nothing that reads an input only copies it.
    assert all(node.op.body.accums for node in _loops(graph) if set(node.inputs) & {"q", "k", "v"})


_TWO_REGIONS = """
import torch, torch.nn as nn
class TwoRegions(nn.Module):
    # a transposed x read ahead of a running sum and again behind it: two regions, the sum between them
    def forward(self, x):
        t = x{widen}.transpose(1, 2)
        return t * torch.cumsum(t.exp(), -1)
m = TwoRegions()
m(torch.randn(2, 8, 4){dtype})
"""


def _two_regions(bf16: bool = False) -> str:
    return _TWO_REGIONS.format(widen=".float()" if bf16 else "", dtype=".to(torch.bfloat16)" if bf16 else "")


def test_a_copy_two_regions_read_is_stored_by_neither() -> None:
    graph, _, _ = graph_from_code(_two_regions())
    graph = Pipeline.build(LOOP_PASSES).run(graph)

    readers = [node for node in _loops(graph) if "x" in node.inputs]
    # The exponential and the product each read ``x`` itself. The product is behind the running sum, which no region
    # holds, so it is another region, and the first kernel stores its exponential alone.
    assert len(readers) == 2 and all(len(node.outputs) == 1 for node in readers)
    assert len(_loops(graph)) == 3, "the exponential, the running sum and the product"


_TWO_READERS = """
import torch, torch.nn as nn
class TwoReaders(nn.Module):
    def forward(self, x):
        t = {shared}
        return torch.relu(t), torch.exp(t)
m = TwoReaders()
m(torch.randn(4, 8))
"""


@pytest.mark.parametrize(
    ("shared", "kernels"),
    [("x", [1, 1]), ("x.transpose(0, 1)", [1, 1]), ("torch.sin(x)", [2])],
    ids=["the-input", "a-transpose-of-it", "a-computed-value"],
)
def test_loops_are_joined_by_a_value_that_is_computed_never_by_one_that_is_moved(shared: str, kernels: list[int]) -> None:
    """Two loops that read one input are two kernels, and a transpose between the input and them changes nothing:
    it is read through by both and joins neither to the other. A value that is computed joins its readers: one
    kernel computes it once and stores both of their results."""
    graph = Pipeline.build(LOOP_PASSES).run(graph_from_code(_TWO_READERS.format(shared=shared))[0])
    assert [len(node.outputs) for node in _loops(graph)] == kernels


def _copy_graph(dtype, stored_over: bool = False) -> Graph:
    """``c[i, j] = x[j, i]`` at ``dtype``, and (``stored_over``) a loop that stores over ``x``."""
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (4, 8), F32), node_id="x")
    copy = parse_body("for a0 in 0..8:\n  for a1 in 0..4:\n    in0 = load x[a1, a0]\n    c[a0, a1] = in0")
    graph.add_node(LoopOp(body=copy), ["x"], Tensor("c", (8, 4), dtype), node_id="c")
    if stored_over:
        store = parse_body("for a0 in 0..8:\n  in0 = load c[a0, 0]\n  x2[0, a0] = in0")
        graph.add_node(LoopOp(body=store), ["c", "x"], Tensor("x2", (4, 8), F32, alias="x"), node_id="x2")
    graph.inputs, graph.outputs = ["x"], ["x2" if stored_over else "c"]
    return graph


def test_only_a_copy_that_keeps_every_value_and_whose_source_stands_is_read_through() -> None:
    cells, read = _RULE.copied(_copy_graph(F32), "c")
    assert [str(cell) for cell in cells] == ["a0", "a1"] and read.input == "x"
    # A copy to a narrower float rounds: it is arithmetic, and its reader reads the rounded buffer.
    assert _RULE.copied(_copy_graph(BF16), "c") is None
    # A kernel stores over the source: the copy holds the values from before that store.
    assert _RULE.copied(_copy_graph(F32, stored_over=True), "c") is None


_STEPS = """
import torch, torch.nn as nn
class Steps(nn.Module):
    # a walk whose step computes its output, the outputs put back in sequence order{also}
    def forward(self, q, k, v, state):
        B, T, D = q.shape
        outs = []
        for c in range(T // 4):
            kc = k[:, c*4:(c+1)*4]
            new = v[:, c*4:(c+1)*4] - kc @ state
            outs.append(q[:, c*4:(c+1)*4] @ state)
            state = state + kc.transpose(1, 2) @ new
        return torch.stack(outs, 1).reshape(B, T, D){cast}, state{first}
m = Steps()
m(torch.randn(2, 16, 8), torch.randn(2, 16, 8), torch.randn(2, 16, 8), torch.randn(2, 8, 8))
"""


def _steps(also: str = "", first: str = "", cast: str = "") -> str:
    return _STEPS.format(also=also, first=first, cast=cast)


@requires_cuda
@pytest.mark.xdist_group("cuda")
@pytest.mark.parametrize(
    "code",
    [_walk(), _walk(bf16=True), _two_regions(), _two_regions(bf16=True), _steps(cast=".to(torch.bfloat16)")],
    ids=["walk", "walk-bf16", "regions", "regions-bf16", "steps-rounded"],
)
def test_copies_read_and_stored_through_match_eager_on_the_gpu(code: str) -> None:
    from emmy.compiler.backend.cuda.program import run_program  # noqa: PLC0415
    from emmy.compiler.pipeline.search.pins import pinned_knobs  # noqa: PLC0415

    graph, _, (module, example, _) = graph_from_code(code)
    arrays = {
        name: tensor.view(torch.uint16).numpy() if tensor.dtype == torch.bfloat16 else tensor.numpy()
        for name, tensor in zip(graph.inputs, example, strict=True)
    }

    # The numbers are held to f32: under fast math a pick may read ten bits of each operand.
    with pinned_knobs({"FAST_MATH": False}):
        result, _ = run_program(Pipeline.build(CUDA_PASSES).run(graph), arrays)

    want = module(*example)
    want = want if isinstance(want, tuple) else (want,)
    for got, reference in zip(result.outputs.values(), want, strict=True):
        got, rtol = np.asarray(got), 1e-4
        if reference.dtype == torch.bfloat16:  # the bits, held to one step of bf16
            got, rtol = torch.from_numpy(got.astype(np.uint16)).view(torch.bfloat16).float().numpy(), 2.0**-7
        np.testing.assert_allclose(got.reshape(reference.shape), reference.float().numpy(), rtol=rtol, atol=1e-5)


@pytest.mark.parametrize("cast", ["", ".to(torch.bfloat16)"], ids=["moved", "rounded"])
def test_a_walk_stores_its_outputs_where_their_one_reader_would_move_them(cast: str) -> None:
    graph, _, _ = graph_from_code(_steps(cast=cast))
    graph = Pipeline.build(LOOP_PASSES).run(graph)

    (walk,) = _loops(graph)
    # Each step stores its four rows of the joined tensor, at a row the step moves, and nothing joins them after.
    assert walk.op.body.carries and set(graph.outputs) == {tensor.name for tensor in walk.outputs}
    (joined,) = (write for write in walk.op.body.writes if len(write.index) == 3 and len(write.index[1].free_vars()) == 2)
    assert "4" in str(joined.index[1])
    # A reader that rounded what it moved is gone the same way: the walk rounds the value it stores.
    rounded = [stmt.dtype for stmt in walk.op.body.iter() if isinstance(stmt, Assign) and stmt.name == joined.value]
    assert rounded == ([BF16] if cast else []) and graph.buffer(joined.output).dtype == (BF16 if cast else F32)


def test_a_buffer_read_elsewhere_is_stored_as_it_is() -> None:
    graph, _, _ = graph_from_code(_steps(also=", and the first one alone", first=", outs[0]"))
    graph = Pipeline.build(LOOP_PASSES).run(graph)

    # Two loops read the steps' outputs: the walk keeps its own buffer, and the join is a kernel.
    (walk,) = (node for node in _loops(graph) if node.op.body.carries)
    assert len(_loops(graph)) > 1 and not set(graph.outputs) <= {tensor.name for tensor in walk.outputs}


def _stored_through(reader: str, kept: Tensor, shape: tuple = (3, 1, 4)) -> str:
    """The loop that stores ``s``, three steps of four cells, once the rule took ``reader``, the one loop that reads
    ``s`` and stores ``kept``."""
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (3, 4), F32), node_id="x")
    steps = "for a0 in 0..3:\n  for a1 in 0..4:\n    in0 = load x[a0, a1]\n    v0 = exp(in0)\n    s[a0, 0, a1] = v0"
    graph.add_node(LoopOp(body=parse_body(steps)), ["x"], Tensor("s", shape, F32), node_id="s")
    graph.add_node(LoopOp(body=parse_body(reader)), ["s"], kept, node_id=kept.name)
    graph.inputs, graph.outputs = ["x"], [kept.name]
    fragment = _STORE.rewrite(SimpleNamespace(graph=graph), graph.nodes[kept.name])
    (stored,) = _loops(fragment)
    return str(stored.op.body)


_CHOSEN = """
for a0 in 0..3:
  for a1 in 0..4:
    in0 = load s[0, 0, a1]
    in1 = load s[1, 0, a1]
    in2 = load s[2, 0, a1]
    v0 = select:
      in0 when 0 == a0
      in1 when 1 == a0
      in2 when true
    c[4 * a0 + a1] = v0
"""
# The same choice as a region writes it whose last loop is not the join: over the joined coordinate, the step a digit.
_JOINED = """
for a0 in 0..12:
  in0 = load s[0, 0, a0 % 4]
  in1 = load s[1, 0, a0 % 4]
  in2 = load s[2, 0, a0 % 4]
  v0 = select:
    in0 when 0 == a0 / 4
    in1 when 1 == a0 / 4
    in2 when true
  bf16 v1 = v0
  c[a0] = v1
"""


def test_a_loop_that_moves_every_cell_once_is_stored_through() -> None:
    # The steps read back by position and chosen by the joined coordinate are one read at that coordinate: the
    # producer stores each step's cells at the step's place in the join.
    assert "c[4 * a0 + a1] = v0" in _stored_through(_CHOSEN, Tensor("c", (12,), F32))
    # A reader that rounds what it moves is gone too, and the producer rounds what it stores.
    stored = _stored_through(_JOINED, Tensor("c", (12,), BF16))
    assert "bf16 v1 = v0" in stored and "c[4 * a0 + a1] = v1" in stored
    transposed = "for a0 in 0..4:\n  for a1 in 0..3:\n    in0 = load s[a1, 0, a0]\n    bf16 v0 = in0\n    c[a0, a1] = v0"
    stored = _stored_through(transposed, Tensor("c", (4, 3), BF16))
    assert "bf16 v1 = v0" in stored and "load x[a1, a0]" in stored and "c[a0, a1] = v1" in stored


@pytest.mark.parametrize(
    ("reader", "shape"),
    [
        (_CHOSEN.replace("in1 = load s[1, 0, a1]", "in1 = load s[0, 0, a1]"), (12,)),
        (_CHOSEN, (16,)),
        ("for a0 in 0..3:\n  for a1 in 0..4:\n    in0 = load s[a0, 0, a1]\n    v0 = exp(in0)\n    c[a0, a1] = v0", (3, 4)),
        ("for a0 in 0..3:\n  for a1 in 0..4:\n    in0 = load s[a0, 0, a1]\n    bf16 v0 = in0\n    f32 v1 = v0\n    c[a0, a1] = v1", (3, 4)),
        ("for a0 in 0..3:\n  for a1 in 0..4:\n    for a2 in 0..2:\n      in0 = load s[a0, 0, a1]\n      c[a0, a1, a2] = in0", (3, 4, 2)),
        ("for a0 in 0..3:\n  for a1 in 0..4:\n    in0 = load s[2 - a0, 0, a1]\n    c[a0, a1] = in0", (3, 4)),
        ("for a0 in 0..3:\n  for a1 in 0..2:\n    in0 = load s[a0, 0, a1]\n    c[a0, a1] = in0", (3, 2)),
    ],
    ids=["a-step-missing", "a-step-more-stored", "computed", "rounded-twice", "read-twice", "reversed", "a-part"],
)
def test_a_loop_that_does_more_than_move_every_cell_once_keeps_its_kernel(reader: str, shape: tuple) -> None:
    stored = (4, 1, 4) if shape == (16,) else (3, 1, 4)
    with pytest.raises(RuleSkipped):
        _stored_through(reader, Tensor("c", shape, F32), shape=stored)
