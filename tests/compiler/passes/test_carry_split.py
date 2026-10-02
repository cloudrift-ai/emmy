"""A carried state's split across the sequence — the cross-CTA fork of a recurrence.

A step affine in its state composes as an affine map, so the sequence splits into parts: a probe
walks every part from two known seeds to read its map, a prefix carries the state across the parts,
and the walk itself runs every part from the state it starts from. The three are kernels lifted
from Loop IR like the walk, so each decides its own row; the numerics are checked against the
sequential walk, off-GPU through the Loop IR runner and on a card through the CUDA pipeline.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.loop.runner import execute_loop_op_cpp
from emmy.compiler.ir.stmt import Assign
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.ops import carries_partition, head
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from emmy.compiler.pipeline.passes.tile._split import split_forks
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import requires_cuda
from tests.compiler.ir.test_carried_state import N, _graph, _inputs, _reference, _step, c, i, j

STEPS = 8


def _run(graph) -> dict[str, np.ndarray]:
    """Every kernel of ``graph`` as Loop IR, in order, over the fixture's inputs."""
    arrays = _inputs(steps=STEPS)
    for node_id in graph.topological_order():
        node = graph.nodes[node_id]
        if isinstance(node.op, TileOp):
            loop = LoopOp(body=node.op.loop_body)
            shapes = {tensor.name: tuple(dim.as_static() for dim in tensor.shape) for tensor in node.outputs}
            result = execute_loop_op_cpp(loop, arrays, shapes)
            arrays.update(zip(loop.outputs, result if isinstance(result, tuple) else (result,), strict=True))
    return arrays


def _cut(graph, pins: dict) -> object:
    with pinned_knobs(pins):
        return Pipeline.build(["tile/lift", "tile/cut"], select=["lift", "cut"]).run(graph)


def test_a_stored_carried_kernel_relifts_with_one_state_port() -> None:
    """A golden stores the lifted node's state port beside its public output. Replaying that
    body must reuse the port, preserving the kernel's identity and output interface."""
    from emmy.compiler.pipeline.search.golden.restamp import definition  # noqa: PLC0415

    lifted = Pipeline.build(["tile/lift"], select=["lift"]).run(_graph(steps=STEPS))
    (node,) = (n for n in lifted.nodes.values() if isinstance(n.op, TileOp))
    stored = definition(node.op.with_io(lifted, node), node.op.name)
    replayed = Pipeline.build(["tile/lift"], select=["lift"]).run(stored.program({}))
    (fresh,) = (n for n in replayed.nodes.values() if isinstance(n.op, TileOp))
    assert len(fresh.outputs) == len(node.outputs) == 2
    assert fresh.op.with_io(replayed, fresh).identity_key(structural=False, with_io=True) == stored.exact_identity


def test_the_walk_is_offered_its_split_and_declines_it_by_default() -> None:
    """The unsplit walk beside one arm per width the step count divides into, the row spelled on
    the carrying site; a step that squares its state is not affine and offers nothing."""
    lifted = Pipeline.build(["tile/lift"], select=["lift"]).run(_graph(steps=STEPS))
    (node,) = (n for n in lifted.nodes.values() if isinstance(n.op, TileOp))
    from emmy.compiler.pipeline import Match  # noqa: PLC0415

    forks = split_forks(Match(graph=lifted, root_node_id=node.id, rule=None), node)
    assert [fork.knobs for fork in forks] == [{"REDUCE@map.1/scan": v} for v in ("", "g2k", "g4k", "g8k")]

    squared = _step((c, i, j), steps=STEPS).map(
        lambda s: Assign(name="kept", op="multiply", args=("own", "own")) if isinstance(s, Assign) and s.name == "kept" else s
    )
    graph = _graph(steps=STEPS)
    graph.nodes["out"].op = LoopOp(body=squared)
    lifted = Pipeline.build(["tile/lift"], select=["lift"]).run(graph)
    (node,) = (n for n in lifted.nodes.values() if isinstance(n.op, TileOp))
    assert head(node.op.op).affine() is None and split_forks(Match(graph=lifted, root_node_id=node.id, rule=None), node) is None


def test_a_coefficient_that_varies_with_the_kept_cell_offers_no_split() -> None:
    """``S ← S·d[c, j] + W·S + U`` is affine, but a different matrix acts on every column: the
    probe's packed identity would read column ``k`` of the ``k``-th map and the prefix apply it
    to every column, so the step is not read as affine and offers nothing."""
    from emmy.compiler.graph import Graph, Tensor  # noqa: PLC0415
    from emmy.compiler.ir.base import InputOp  # noqa: PLC0415
    from emmy.compiler.ir.stmt import Load  # noqa: PLC0415
    from emmy.compiler.pipeline import Match  # noqa: PLC0415

    per_column = _step((c, i, j), steps=STEPS).map(
        lambda s: (
            (Load(name="dj", input="D2", index=(c, j)), Assign(name="kept", op="multiply", args=("own", "dj")))
            if isinstance(s, Assign) and s.name == "kept"
            else s
        )
    )
    graph = Graph()
    for name, shape in (("D", (STEPS,)), ("W", (N, N)), ("U", (STEPS, N, N)), ("D2", (STEPS, N))):
        graph.add_node(InputOp(), [], Tensor(name, shape, "f32"), node_id=name)
    graph.add_node(LoopOp(body=per_column, name="k_step"), ["D", "W", "U", "D2"], Tensor("out", (STEPS, N, N), "f32"), node_id="out")
    graph.inputs, graph.outputs = ["D", "W", "U", "D2"], ["out"]
    lifted = Pipeline.build(["tile/lift"], select=["lift"]).run(graph)
    (node,) = (n for n in lifted.nodes.values() if isinstance(n.op, TileOp))
    assert head(node.op.op).affine() is None and split_forks(Match(graph=lifted, root_node_id=node.id, rule=None), node) is None


def test_a_walk_under_an_outer_loop_offers_no_split() -> None:
    """The realizer rebuilds the carrying nest alone, so a walk under a free loop outside it keeps
    the sequence, whatever its step count divides into."""
    from emmy.compiler.pipeline import Match  # noqa: PLC0415
    from tests.compiler.ir.test_carried_state import _batched_graph  # noqa: PLC0415

    lifted = Pipeline.build(["tile/lift"], select=["lift"]).run(_batched_graph(steps=STEPS))
    (node,) = (n for n in lifted.nodes.values() if isinstance(n.op, TileOp))
    assert head(node.op.op).affine() is not None and node.op.place.free
    assert split_forks(Match(graph=lifted, root_node_id=node.id, rule=None), node) is None


@pytest.mark.parametrize("parts", [2, 4])
def test_the_split_parts_match_the_sequential_walk(parts: int) -> None:
    """Pinned, the split mints the probe, the prefix and the walk; run as Loop IR they reproduce
    the sequential walk, and the walk's step axis carries the partition receipt."""
    graph = _cut(_graph(steps=STEPS), {"REDUCE": f"g{parts}k"})
    tiles = {node.id: node.op for node in graph.nodes.values() if isinstance(node.op, TileOp)}
    assert sorted(tiles) == ["out", "out__probe", "out__probe_seed", "out__start"]
    walk = tiles["out"]
    assert walk.carries and carries_partition(walk) and walk.axis_of(head(walk.op).axis).extent.as_static() == STEPS // parts
    assert tiles["out__probe"].carries and tiles["out__start"].carries and not tiles["out__probe_seed"].carries

    arrays = _run(graph)
    np.testing.assert_allclose(arrays["out"], _reference(arrays), rtol=1e-4, atol=1e-5)


@requires_cuda
@pytest.mark.xdist_group("cuda")
def test_the_split_parts_match_the_sequential_walk_on_the_gpu() -> None:
    from emmy.compiler.backend.cuda.program import run_program  # noqa: PLC0415

    with pinned_knobs({"REDUCE": "g2k"}):
        graph = Pipeline.build(CUDA_PASSES).run(_graph(steps=STEPS))
    arrays = _inputs(steps=STEPS)
    result, _ = run_program(graph, arrays)
    want = _reference(arrays)
    np.testing.assert_allclose(np.asarray(result.outputs["out"]).reshape(want.shape), want, rtol=1e-4, atol=1e-5)


def test_the_delta_rule_splits_and_matches_eager() -> None:
    """The inter-chunk delta rule of ``test_roll_recurrence``, rolled, split two ways and run as
    Loop IR against eager PyTorch: its step ``S·exp(g) + kᵀv`` reads the state at its own cell
    alone, so the map is diagonal and the probe walks from a seed of ones."""
    import torch  # noqa: PLC0415

    from emmy.commands.trace import graph_from_code  # noqa: PLC0415
    from emmy.compiler.pipeline import LOOP_PASSES  # noqa: PLC0415
    from tests.compiler.passes.test_roll_recurrence import _delta  # noqa: PLC0415
    from tests.compiler.passes.test_roll_recurrence import _run as run_loops

    graph, _, (module, _, _) = graph_from_code(_delta(b=2, t=36, d=8, chunk=4))
    with pinned_knobs({"REDUCE": "g2k"}):
        graph = Pipeline.build(["tile/lift", "tile/cut"], select=["lift", "cut"]).run(Pipeline.build(LOOP_PASSES).run(graph))
    carrying = [node.op for node in graph.nodes.values() if isinstance(node.op, TileOp) and node.op.carries]
    assert len(carrying) == 3 and sum(carries_partition(tile) for tile in carrying) == 2

    arrays = run_loops(graph)
    reference = module(*(torch.from_numpy(arrays[name]) for name in ("q", "k", "v", "g"))).numpy()
    # The chunks' outputs by chunk order; the bare pin split the consumers' contractions too, so
    # their workspaces sit beside them under longer names.
    chunks = sorted((name for name in arrays if re.fullmatch(r"matmul(_\d+)?", name)), key=lambda name: int(name[7:] or 0))
    np.testing.assert_allclose(np.concatenate([arrays[name] for name in chunks], axis=1), reference, rtol=1e-4, atol=1e-5)
