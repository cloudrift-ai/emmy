"""A copy is read through by every loop outside its region (``loop/fusion/008_read_through_copies``).

A loop that only re-lays out or widens a buffer was stored wherever a kernel outside its region read it: for a kernel
that carries a state, which is in no region, and for a second region. The reader now reads the buffer behind the
copy, and the copy is stored for no one.
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
