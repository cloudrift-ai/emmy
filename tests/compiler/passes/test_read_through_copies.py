"""A copy is read through by every loop outside its region (``loop/fusion/008_read_through_copies``), and a kernel
no region holds stores its buffer where the one loop that reads it would only have moved it
(``loop/fusion/020_store_through_copies``).

A loop that only re-lays out or widens a buffer was stored wherever a kernel outside its region read it: for a kernel
that carries a state, which is in no region, and for a second region. The reader now reads the buffer behind the
copy, and the copy is stored for no one. On the other side of such a kernel, the loop that puts its outputs where the
program wants them was a kernel of its own.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from emmy.commands.trace import graph_from_code
from emmy.compiler.dtype import BF16, F32
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import parse_body
from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, Pipeline
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
    # The transpose is in the region of the exponential. The product is behind the running sum, which no region
    # holds, so it is another region; it reads ``x`` itself, and the first kernel stores its exponential alone.
    assert len(readers) == 2 and all(len(node.outputs) == 1 for node in readers)
    assert len(_loops(graph)) == 3, "the exponential, the running sum and the product"


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


@requires_cuda
@pytest.mark.xdist_group("cuda")
@pytest.mark.parametrize(
    "code", [_walk(), _walk(bf16=True), _two_regions(), _two_regions(bf16=True)], ids=["walk", "walk-bf16", "regions", "regions-bf16"]
)
def test_readers_through_a_copy_match_eager_on_the_gpu(code: str) -> None:
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
        np.testing.assert_allclose(np.asarray(got).reshape(reference.shape), reference.float().numpy(), rtol=1e-4, atol=1e-5)


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
        return torch.stack(outs, 1).reshape(B, T, D), state{first}
m = Steps()
m(torch.randn(2, 16, 8), torch.randn(2, 16, 8), torch.randn(2, 16, 8), torch.randn(2, 8, 8))
"""


def test_a_walk_stores_its_outputs_where_their_one_reader_would_move_them() -> None:
    graph, _, _ = graph_from_code(_STEPS.format(also="", first=""))
    graph = Pipeline.build(LOOP_PASSES).run(graph)

    (walk,) = _loops(graph)
    # Each step stores its four rows of the joined tensor, at a row the step moves, and nothing joins them after.
    assert walk.op.body.carries and set(graph.outputs) == {tensor.name for tensor in walk.outputs}
    (rows,) = (write.index[1] for write in walk.op.body.writes if len(write.index) == 3 and len(write.index[1].free_vars()) == 2)
    assert "4" in str(rows)


def test_a_buffer_read_elsewhere_is_stored_as_it_is() -> None:
    graph, _, _ = graph_from_code(_STEPS.format(also=", and the first one alone", first=", outs[0]"))
    graph = Pipeline.build(LOOP_PASSES).run(graph)

    # Two loops read the steps' outputs: the walk keeps its own buffer, and the join is a kernel.
    (walk,) = (node for node in _loops(graph) if node.op.body.carries)
    assert len(_loops(graph)) > 1 and not set(graph.outputs) <= {tensor.name for tensor in walk.outputs}


def _moved(body: str):
    return _STORE.moved(LoopOp(body=parse_body(body)))


def test_only_a_loop_that_moves_every_cell_once_is_stored_through() -> None:
    chosen = """
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
    # The steps read back by position and chosen by the joined coordinate are one read at that coordinate.
    source, cell, write = _moved(chosen)
    assert source == "s" and [str(entry) for entry in cell] == ["a0", "0", "a1"] and write.output == "c"
    # A step missing from the choice, a value computed on the way, a cell read twice: each stays a kernel.
    assert _moved(chosen.replace("in1 = load s[1, 0, a1]", "in1 = load s[0, 0, a1]")) is None
    assert _moved("for a0 in 0..4:\n  in0 = load s[a0]\n  v0 = exp(in0)\n  c[a0] = v0") is None
    assert _moved("for a0 in 0..4:\n  for a1 in 0..2:\n    in0 = load s[a0]\n    c[a0, a1] = in0") is None
    assert _moved("for a0 in 0..4:\n  in0 = load s[3 - a0]\n  c[a0] = in0") is None
