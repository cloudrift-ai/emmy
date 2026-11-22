"""``loop/fusion/015_merge_into_step``: the kernels around a kernel that carries a state join its step."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from emmy.commands.trace import graph_from_code
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.schedule.classic.refusals import _swept_root_refusal
from emmy.compiler.ir.stmt import Assign, Loop
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.ops import kernel_roots
from emmy.compiler.pipeline import LOOP_PASSES, Pipeline
from emmy.compiler.pipeline.passes.tile._cut import _buffer_reads, cuttable_seams
from emmy.compiler.pipeline.passes.tile._fromloop import scheduled_form
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import compile_with_kernels, requires_cuda

_STEPS = """
import torch, torch.nn as nn
class Steps(nn.Module):
    # a state walked over four chunks; each chunk's output reads the state its step starts from{also}
    def forward(self, q, k, v, g, x, state):
        {weight}
        outs = []
        for c in range(4):
            kc = k[:, c*4:(c+1)*4]
            new = v[:, c*4:(c+1)*4] - kc @ state
            outs.append(q[:, c*4:(c+1)*4] @ state)
            state = state + kc.transpose(1, 2) @ new
        y = torch.stack(outs, 1).reshape(2, 16, 8)
        return {reader}, state
m = Steps()
m(torch.randn(2, 16, 8), torch.randn(2, 16, 8), torch.randn(2, 16, 8), torch.rand(2, 16), torch.randn(2, 16, 8), torch.randn(2, 8, 8))
"""
_COMPUTED = _STEPS.format(
    also=", a computed weight on the keys ahead of the loop and a reader behind it that computes",
    weight="k = k * torch.exp(-g).unsqueeze(-1)",
    reader="(y * 2.0 + x).to(torch.bfloat16)",
)
_LAST = _STEPS.format(also="; the reader takes the state the loop ends with", weight="pass", reader="y.sum(1) @ state")
_FOLDED = _STEPS.format(also="; the reader folds the steps into one row", weight="pass", reader="y.sum(1)")


def _loops(code: str) -> list:
    graph = Pipeline.build(LOOP_PASSES).run(graph_from_code(code)[0])
    return [node for node in graph.nodes.values() if isinstance(node.op, LoopOp)]


def test_a_producer_and_a_reader_that_compute_join_the_step() -> None:
    """The weight only the step reads is computed where the step reads it. The reader's loop over the sixteen rows
    becomes the step and the four rows of that step, so the kernel stores the program's output from inside the step,
    at a row the step moves."""
    (kernel,) = _loops(_COMPUTED)
    body = kernel.op.body
    assert body.carries and {"g", "x"} <= set(kernel.inputs)
    (steps,) = (stmt for stmt in body if isinstance(stmt, Loop))
    assert steps.axis.extent.as_static() == 4 and steps.carries
    (joined,) = (write for write in body.writes if len(write.index) == 3 and write.index[1].free_vars() > {steps.axis.name})
    assert [stmt.dtype.name for stmt in body.iter() if isinstance(stmt, Assign) and stmt.name == joined.value] == ["bf16"]


@pytest.mark.parametrize("code", [_LAST, _FOLDED], ids=["the-last-state", "a-fold-over-the-steps"])
def test_a_reader_that_no_step_can_run_stays_a_kernel(code: str) -> None:
    """A reader of the state the loop ends with runs after the last step, and a fold over the steps reads every step
    in one trip: neither is a part of one step."""
    kernels = _loops(code)
    assert len(kernels) == 2 and sum(bool(node.op.body.carries) for node in kernels) == 1


def test_the_merged_step_matches_eager(run_graph) -> None:
    graph, _, (module, example, _) = graph_from_code(_COMPUTED)
    names = ("q", "k", "v", "g", "x", "state")
    with torch.no_grad():
        exact = module.double()(*(tensor.double() for tensor in example))
    outputs = run_graph(graph, {name: tensor.numpy() for name, tensor in zip(names, example, strict=True)})
    for got, want in zip(outputs.values(), exact, strict=True):
        got = np.asarray(got)
        if want.dtype == torch.bfloat16 and got.dtype != np.float32:  # the bits the kernel stored
            got = torch.from_numpy(got.astype(np.uint16).astype(np.int16)).view(torch.bfloat16).double().numpy()
        want = want.double().numpy()
        error = got.reshape(want.shape) - want
        assert float(np.sqrt(np.mean(error**2))) < 2.0**-7 * float(np.sqrt(np.mean(want**2)))


_AHEAD = _STEPS.format(
    also="; the keys are a product computed ahead of the loop",
    weight="k = k @ x[:, :8].transpose(1, 2)",
    reader="y",
)
_SETUP = """
import torch, torch.nn as nn
class Walk(nn.Module):
    # a walk whose outputs read a matrix computed before the loop, as the values each step adds do
    def forward(self, q, k, v, state):
        B, T, D = q.shape
        C = 4
        chunked = lambda x: x.reshape(B, T // C, C, D)
        weights = (chunked(q) @ chunked(k).transpose(-1, -2)).tril()
        values = weights @ chunked(v)
        outs = []
        for c in range(T // C):
            kc = k[:, c*C:(c+1)*C]
            new = values[:, c] - kc @ state
            outs.append(q[:, c*C:(c+1)*C] @ state + weights[:, c] @ new)
            state = state + kc.transpose(1, 2) @ new
        return torch.stack(outs, 1), state
m = Walk()
m(torch.randn(2, 16, 8), torch.randn(2, 16, 8), torch.randn(2, 16, 8), torch.randn(2, 8, 8))
"""


def _tiles(graph) -> list:
    return [node for node in graph.nodes.values() if isinstance(node.op, TileOp) and node.op.op is not None]


def _cut_ahead(graph) -> dict[str, str]:
    """The pin that cuts the product ahead of the loop out of the merged kernel: its one seam, which reads no state."""
    (merged,) = _tiles(Pipeline.build([*LOOP_PASSES, "tile/lift"]).run(graph))
    (seam,) = cuttable_seams(merged.op)
    assert merged.op.carries and not {spec.write.name for spec in merged.op.states} & _buffer_reads(seam.node)
    return {f"PLACE@{merged.op.name}/{seam.spelling.partition('@')[2]}": "cut"}


def test_a_product_the_step_reads_is_cut_into_a_kernel_ahead_of_the_loop() -> None:
    """The product is the same at whichever step reads it, so its kernel computes every step's rows ahead and
    carries nothing; the kernel that keeps the state reads them back."""
    graph = graph_from_code(_AHEAD)[0]
    with pinned_knobs(_cut_ahead(graph)):
        pieces = _tiles(Pipeline.build([*LOOP_PASSES, "tile/lift", "tile/cut"]).run(graph))
    (keeps,) = (node for node in pieces if node.op.carries)
    ahead = [node for node in pieces if not node.op.carries]
    assert ahead and not any(node.op.place.serial for node in ahead)
    assert {name for node in ahead for name in node.buffer_names()} & set(keeps.inputs)
    assert "state" in keeps.inputs, "the kernel that keeps the state reads the buffer it starts from"


@requires_cuda
@pytest.mark.xdist_group("cuda")
def test_the_cut_kernels_match_eager_on_the_gpu() -> None:
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    graph, _, (module, example, _) = graph_from_code(_AHEAD)
    names = ("q", "k", "v", "g", "x", "state")
    with pinned_knobs(_cut_ahead(graph)):
        compiled, kernels = compile_with_kernels(graph)
    assert len(kernels) > 1
    result, _ = CudaBackend().run(compiled, input_data={name: tensor.numpy() for name, tensor in zip(names, example, strict=True)})
    with torch.no_grad():
        exact = module.double()(*(tensor.double() for tensor in example))
    for got, want in zip(result.outputs.values(), exact, strict=True):
        want = want.double().numpy()
        error = np.asarray(got).reshape(want.shape) - want
        assert float(np.sqrt(np.mean(error**2))) < 2.0**-7 * float(np.sqrt(np.mean(want**2)))


def test_a_product_no_root_holds_is_offered_no_tile() -> None:
    """The matrix computed before the loop joins the step under both of its readers. The kernel is built around
    the product that updates the state, so the products the output's branch holds are lowered a cell at a time
    by the reader that holds them, and the offer gives none of them a tile. The binder refused such a row; with
    enough of them ranked first the kernel ran with no schedule."""
    (merged,) = _tiles(Pipeline.build([*LOOP_PASSES, "tile/lift"]).run(graph_from_code(_SETUP)[0]))
    tile = scheduled_form(merged.op, merged.id)
    (root,) = (tile.sites[tile.node_id(root)] for root in kernel_roots(tile.op))
    products = [tile.sites[at] for at in tile.family_sites["TILE"]]
    apart = [site for site in products if site is not root and not site.under(root)]
    assert len(apart) > 2 and all(_swept_root_refusal(tile, site.node) for site in apart)
    assert [site for site in products if site.under(root) and _swept_root_refusal(tile, site.node) is None]
