"""Sibling contraction roots whose stores ride different widths keep their sweeps serial: no output tile can own a
swept coordinate, so the schedule offers none there and the binder declines one that reaches it."""

from dataclasses import replace
from importlib import import_module

import pytest

from emmy.commands.trace import graph_from_code
from emmy.compiler.context import Context
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.schedule.base import clear_pin_refusals, pin_refusal
from emmy.compiler.ir.schedule.classic import sites as classic_sites
from emmy.compiler.ir.tile.ops import UnbindableProjection
from emmy.compiler.pipeline import LOOP_PASSES, Pipeline
from emmy.compiler.pipeline.fork import iter_leaves
from emmy.compiler.pipeline.passes.lowering.kernel._factor import factorize
from emmy.compiler.pipeline.passes.tile._fromloop import lift_loop_op
from emmy.compiler.pipeline.search.pins import pinned_knobs

classic_forks = import_module("emmy.compiler.pipeline.passes.tile.schedule.040_schedule").classic_forks

# One normalized query read by two contractions: against a state over 32 value columns and against 16 keys. The
# stores ride 32 and 16 wide, so neither sweep reaches the grid.
CODE = """
q = torch.randn(16, 4, 32); s = torch.randn(12, 32, 32); k = torch.randn(12, 16, 32)
def f(q, s, k):
    qv = (q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)).repeat_interleave(3, dim=1)
    return torch.einsum("tvd,vdj->tvj", qv, s), torch.einsum("tvd,vmd->tvm", qv, k)
f(q, s, k)
"""
CTX = Context.from_target((12, 0), gpu_name="NVIDIA GeForce RTX 5090")


def _lifted():
    """The two contractions' one fused kernel, lifted and not yet scheduled."""
    lowered = Pipeline.build(LOOP_PASSES).run(graph_from_code(CODE)[0], ctx=CTX)
    (loop,) = (node.op for node in lowered.nodes.values() if isinstance(node.op, LoopOp))
    return replace(lift_loop_op(loop, name=loop.name), knobs=dict(loop.knobs), inputs=dict(loop.inputs), outputs=dict(loop.outputs))


def _tiled(knobs) -> bool:
    return any(value for key, value in knobs.items() if key.partition("@")[0] == "TILE")


def test_swept_roots_offer_no_tile() -> None:
    tile = _lifted()
    assert all(spec.sweep for spec in tile.output_specs), "each store keeps its own sweep"
    leaves = list(iter_leaves(classic_forks(tile, tile.name, tile.knobs, CTX)))
    assert leaves
    assert not any(_tiled(leaf.knobs) for leaf in leaves)

    clear_pin_refusals()
    with pinned_knobs({"TILE@map.1/inner": "f1x2"}):
        list(iter_leaves(classic_forks(tile, tile.name, tile.knobs, CTX)))
    assert "stays a serial loop" in (pin_refusal("TILE", "f1x2") or "")


def test_binder_declines_a_swept_tiled_root(monkeypatch) -> None:
    """With the offer's rule switched off, a row tiling both roots reaches the binder, which must decline it rather
    than emit register cells over a coordinate nothing binds."""
    tile = _lifted()
    monkeypatch.setattr(classic_sites, "_swept_root_refusal", lambda _tile, _node: None)
    both = lambda knobs: all(value for key, value in knobs.items() if key.partition("@")[0] == "TILE")  # noqa: E731
    leaf = next(leaf for leaf in iter_leaves(classic_forks(tile, tile.name, tile.knobs, CTX)) if both(leaf.knobs))
    (realized,) = leaf.expand()
    with pytest.raises(UnbindableProjection, match="output sweep axis"):
        factorize(realized, root=None)
