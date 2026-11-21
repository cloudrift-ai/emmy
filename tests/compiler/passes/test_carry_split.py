"""A carried state's split across the sequence — the cross-CTA fork of a recurrence.

A step affine in its state composes as an affine map, so the sequence splits into parts: a probe
walks every part from two known seeds to read its map, a prefix carries the state across the parts,
and the walk itself runs every part from the state it starts from. The three are kernels lifted
from Loop IR like the walk, so each decides its own row; the numerics are checked against the
sequential walk, off-GPU through the Loop IR runner and on a card through the CUDA pipeline.
"""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
import pytest

from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.loop.runner import execute_loop_op_cpp
from emmy.compiler.ir.stmt import Assign, parse_body
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.ops import carries_partition, head
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from emmy.compiler.pipeline.passes.tile._split import split_forks
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import requires_cuda
from tests.compiler.ir.test_carried_state import N, _graph, _inputs, _reference, _step

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
    stored = definition(node.op.with_io(lifted, node))
    replayed = Pipeline.build(["tile/lift"], select=["lift"]).run(stored.program({}))
    (fresh,) = (n for n in replayed.nodes.values() if isinstance(n.op, TileOp))
    assert len(fresh.outputs) == len(node.outputs) == 2
    assert fresh.op.with_io(replayed, fresh).identity_key(structural=False, with_io=True) == stored.exact_identity


def test_an_unmapped_carried_kernel_uses_ordered_cuda_launches() -> None:
    """The scalar fallback owns the same global state port as a classic schedule."""
    from emmy.compiler.ir.cuda import CudaOp  # noqa: PLC0415

    graph = Pipeline.build(["tile/lift", "lowering/kernel", "lowering/cuda"]).run(_graph(steps=STEPS))
    (op,) = (node.op for node in graph.nodes.values() if isinstance(node.op, CudaOp))
    assert len(op.serial) == 1 and op.serial[0][1] == STEPS
    # The carried state is in registers: the kernel takes no local memory.
    assert ".local" not in op.kernel_source


def test_the_walk_is_offered_its_split_and_declines_it_by_default() -> None:
    """The unsplit walk beside one arm per width the step count divides into and, at one step a part,
    one per pairwise composition of the maps, the row spelled on the carrying site; a step that squares
    its state is not affine and offers nothing."""
    lifted = Pipeline.build(["tile/lift"], select=["lift"]).run(_graph(steps=STEPS))
    (node,) = (n for n in lifted.nodes.values() if isinstance(n.op, TileOp))
    from emmy.compiler.pipeline import Match  # noqa: PLC0415

    forks = split_forks(Match(graph=lifted, root_node_id=node.id, rule=None), node)
    offered = ("", "g2k", "g4k", "g8k", "g8k1", "g8k2", "g8k3")  # the catalog's widths, then one step a part to every depth
    assert [fork.knobs for fork in forks] == [{"REDUCE@map.1/scan": v} for v in offered]
    # A measured row and a stored decision reach the arm of their own depth, not the first of their width.
    from emmy.compiler.pipeline.search.pins import spelled_arm  # noqa: PLC0415

    assert [spelled_arm(forks, {"REDUCE@map.1/scan": v})[1] for v in offered] == [fork.knobs for fork in forks]

    squared = _step("c, i, j", steps=STEPS).map(
        lambda s: parse_body("kept = multiply(own, own)") if isinstance(s, Assign) and s.name == "kept" else s
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
    from emmy.compiler.pipeline import Match  # noqa: PLC0415

    per_column = _step("c, i, j", steps=STEPS).map(
        lambda s: parse_body("dj = load D2[c, j]\nkept = multiply(own, dj)") if isinstance(s, Assign) and s.name == "kept" else s
    )
    graph = Graph()
    for name, shape in (("D", (STEPS,)), ("W", (N, N)), ("U", (STEPS, N, N)), ("D2", (STEPS, N))):
        graph.add_node(InputOp(), [], Tensor(name, shape, "f32"), node_id=name)
    graph.add_node(LoopOp(body=per_column), ["D", "W", "U", "D2"], Tensor("out", (STEPS, N, N), "f32"), node_id="out")
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


SCAN = "REDUCE@map.1/scan"  # the carrying site: a bare pin would reach the product pieces too


@pytest.mark.parametrize("pin", ["g2k1", "g4k1", "g4k2", "g8k1", "g8k3"])
def test_the_parts_maps_compose_pairwise_and_match_the_sequential_walk(pin: str) -> None:
    """``g<n>k<p>`` composes the parts' maps pairwise ``p`` times: a matrix and an offset kernel per
    composition, each a reduction over the column the two maps meet at, and a fill-in per composition
    that gives the parts below their starts. The walk stores what it stored. The products carry no
    state, so each is a kernel with a fork and a row of its own."""
    parts, pairs = int(pin[1]), int(pin[-1])
    # The products unsplit: the runner takes no atomic store. And uncut: a prior may cut a fill-in, which is its
    # own decision and not what this test lists.
    graph = _cut(_graph(steps=STEPS), {SCAN: pin, "REDUCE": "", "PLACE": "fuse"})
    tiles = {node.id: node.op for node in graph.nodes.values() if isinstance(node.op, TileOp)}
    # One step a part reads the step's maps off the step; longer parts walk a probe.
    read = ["out__map0", "out__offset0"] if parts == STEPS else ["out__probe", "out__probe_seed"]
    composed = [f"out__{kind}{level}" for level in range(1, pairs + 1) for kind in ("map", "offset", "start")]
    assert sorted(tiles) == sorted(["out", "out__start", *read, *composed])
    carrying = {name for name, tile in tiles.items() if tile.carries}
    left = -(-parts // 2**pairs)
    # One step a part carries nothing in the probe's place or the walk's; one map left applies to the seed.
    walked = {*(["out", "out__probe"] if parts < STEPS else []), *([f"out__start{pairs}"] if left > 1 else [])}
    assert carrying == walked
    if left > 1:
        assert tiles[f"out__start{pairs}"].axis_of(head(tiles[f"out__start{pairs}"].op).axis).extent.as_static() == left

    arrays = _run(graph)
    np.testing.assert_allclose(arrays["out"], _reference(arrays), rtol=1e-4, atol=1e-5)


def test_a_pairwise_pin_names_a_carried_state_and_a_depth_its_parts_have() -> None:
    """A reduction's parts combine through its own ⊕, so the digit means nothing there; and four
    parts are one map after two compositions."""
    with pytest.raises(ValueError, match="one map after 2 compositions"):
        _cut(_graph(steps=STEPS), {SCAN: "g4k3"})
    with pytest.raises(ValueError, match="only the parts of a carried state compose pairwise"):
        _cut(_graph(steps=STEPS), {"REDUCE": "g4k1"})


_INVERSE = """
import torch, torch.nn as nn
class Inverse(nn.Module):
    # The inverse of a unit lower triangular matrix a row a step, the system read from its own tensor: the
    # step is the state plus one row of the system times the state, which is affine in the state.
    def forward(self, a):
        low = a.tril(-1)
        eye = torch.eye({n})
        a = low
        for i in range(1, {n}):
            a = a + eye[i].unsqueeze(-1) * (low[..., i:i+1, :] @ a)
        return a + eye
m = Inverse()
m(torch.randn(2, {n}, {n}) * 0.1)
"""


@pytest.mark.parametrize("pin", ["", "g2k", "g6k", "g6k1", "g6k2", "g6k3"])
def test_the_row_substitution_composes_to_its_last_state_and_matches_eager(pin: str) -> None:
    """The substitution stores its state alone, each step over the last one's, so parts running side by
    side would race on it: the split ends at the prefix, which stores the final state, and mints no walk.
    Six steps compose as three maps, then two with the odd one passed through, then one."""
    import torch  # noqa: PLC0415

    from emmy.commands.trace import graph_from_code  # noqa: PLC0415
    from emmy.compiler.pipeline import LOOP_PASSES  # noqa: PLC0415
    from tests.compiler.passes.test_roll_recurrence import _run as run_loops  # noqa: PLC0415

    graph, _, (module, _, _) = graph_from_code(_INVERSE.format(n=8))
    with pinned_knobs({SCAN: pin} if pin else {}):
        graph = Pipeline.build(["tile/lift", "tile/cut"], select=["lift", "cut"]).run(Pipeline.build(LOOP_PASSES).run(graph))
    carrying = [node.op for node in graph.nodes.values() if isinstance(node.op, TileOp) and node.op.carries]
    # Two parts walk a probe and a prefix; six read their maps off the step, and one map left carries nothing.
    assert len(carrying) == {"": 1, "g2k": 2, "g6k3": 0}.get(pin, 1), "no walk: the prefix stores the last state"
    assert sum("__map" in node.id for node in graph.nodes.values()) == (int(pin[3:] or 0) + 1 if pin.startswith("g6") else 0)

    arrays = run_loops(graph)
    (out,) = graph.outputs
    reference = module(torch.from_numpy(arrays["a"])).numpy()
    np.testing.assert_allclose(arrays[out].reshape(reference.shape), reference, rtol=1e-4, atol=1e-5)


_WRAPPED_RULE = """
import torch, torch.nn as nn
import transformers.models.qwen3_5.modeling_qwen3_5 as scope
from emmy.compiler.trace.huggingface import _chunk_rule_scan_then_read
class Rule(nn.Module):
    def forward(self, q, k, v, g, beta):
        out, _ = _chunk_rule_scan_then_read(vars(scope), q, k, v, g, beta, chunk_size=8, use_qk_l2norm_in_kernel=True)
        return out
m = Rule()
m(torch.randn(1, 32, 2, 8), torch.randn(1, 32, 2, 8), torch.randn(1, 32, 2, 8), -torch.rand(1, 32, 2), torch.rand(1, 32, 2))
"""


@pytest.mark.parametrize("pin", ["g3k", "g3k2"])
def test_the_chunk_loop_of_the_gated_delta_rule_composes_and_matches_eager(pin: str) -> None:
    """The rule as the Gated DeltaNet wrapper traces it holds two recurrences. The chunk loop is affine: it
    stores a second value beside its state every step, under the batch's one coordinate, and three parts
    of one step read their maps off the step. The row substitution multiplies two reads of its state, so
    the pin that reaches both carrying sites decides nothing on it."""
    import torch  # noqa: PLC0415

    from emmy.commands.trace import graph_from_code  # noqa: PLC0415
    from emmy.compiler.pipeline import LOOP_PASSES  # noqa: PLC0415
    from tests.compiler.passes.test_roll_recurrence import _run as run_loops  # noqa: PLC0415

    graph, _, (module, _, _) = graph_from_code(_WRAPPED_RULE)
    with pinned_knobs({SCAN: pin, "REDUCE": ""}):
        graph = Pipeline.build(["tile/lift", "tile/cut"], select=["lift", "cut"]).run(Pipeline.build(LOOP_PASSES).run(graph))
    names = [node.id for node in graph.nodes.values() if isinstance(node.op, TileOp)]
    assert not any(name.endswith("__probe") for name in names)
    assert sum("__map" in name for name in names) == int(pin[3:] or 0) + 1

    arrays = run_loops(graph)
    (out,) = graph.outputs
    reference = module(*(torch.from_numpy(arrays[name]) for name in ("q", "k", "v", "g", "beta"))).numpy()
    np.testing.assert_allclose(arrays[out].reshape(reference.shape), reference, rtol=1e-3, atol=1e-4)


def test_a_store_of_another_value_without_the_step_offers_no_split() -> None:
    """A store the step does not index holds what the last step wrote. The state is the prefix's to
    write; the state the step READ is not, and parts side by side would each write it."""
    from emmy.compiler.pipeline import Match  # noqa: PLC0415

    graph = _graph(steps=STEPS)
    graph.nodes["out"].op = LoopOp(body=_step("i, j", steps=STEPS))
    graph.nodes["out"].outputs = (replace(graph.nodes["out"].outputs[0], shape=graph.nodes["out"].outputs[0].shape[1:]),)
    lifted = Pipeline.build(["tile/lift"], select=["lift"]).run(graph)
    (node,) = (n for n in lifted.nodes.values() if isinstance(n.op, TileOp))
    assert head(node.op.op).affine() is not None
    assert split_forks(Match(graph=lifted, root_node_id=node.id, rule=None), node) is None


@requires_cuda
@pytest.mark.xdist_group("cuda")
def test_an_unmapped_carried_kernel_matches_the_sequential_walk_on_the_gpu() -> None:
    from emmy.compiler.backend.cuda.program import run_program  # noqa: PLC0415

    graph = Pipeline.build(["tile/lift", "lowering/kernel", "lowering/cuda"]).run(_graph(steps=STEPS))
    arrays = _inputs(steps=STEPS)
    result, _ = run_program(graph, arrays)
    want = _reference(arrays)
    np.testing.assert_allclose(np.asarray(result.outputs["out"]).reshape(want.shape), want, rtol=1e-4, atol=1e-5)


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


@requires_cuda
@pytest.mark.xdist_group("cuda")
def test_the_composed_parts_match_the_sequential_walk_on_the_gpu() -> None:
    from emmy.compiler.backend.cuda.program import run_program  # noqa: PLC0415

    with pinned_knobs({SCAN: "g4k2"}):
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
    reference = torch.cat(module(*(torch.from_numpy(arrays[name]) for name in ("q", "k", "v", "g"))), 1).numpy()
    # The chunks' outputs by chunk order; the bare pin split the consumers' contractions too, so
    # their workspaces sit beside them under longer names.
    chunks = sorted((name for name in arrays if re.fullmatch(r"matmul(_\d+)?", name)), key=lambda name: int(name[7:] or 0))
    np.testing.assert_allclose(np.concatenate([arrays[name] for name in chunks], axis=1), reference, rtol=1e-4, atol=1e-5)


@requires_cuda
@pytest.mark.xdist_group("cuda")
@pytest.mark.parametrize("pin", ["g2k", "g6k", "g6k3"])
def test_the_row_substitution_composes_to_its_last_state_on_the_gpu(pin: str) -> None:
    """The prefix owns the kernel's output here, and stores the final state from ordered launches."""
    import torch  # noqa: PLC0415

    from emmy.commands.trace import graph_from_code  # noqa: PLC0415
    from emmy.compiler.backend.cuda.program import run_program  # noqa: PLC0415

    graph, _, (module, _, _) = graph_from_code(_INVERSE.format(n=8))
    # The atoms are the prior's to pick, and the numbers are held to f32: fast math off.
    with pinned_knobs({SCAN: pin, "FAST_MATH": False}):
        graph = Pipeline.build(CUDA_PASSES).run(graph)
    assert sum(node.id.endswith("__probe") for node in graph.nodes.values()) == (pin == "g2k")
    a = (np.random.default_rng(0).standard_normal((2, 8, 8)) * 0.1).astype(np.float32)
    result, _ = run_program(graph, {"a": a})
    (out,) = graph.outputs
    want = module(torch.from_numpy(a)).numpy()
    np.testing.assert_allclose(np.asarray(result.outputs[out]).reshape(want.shape), want, rtol=1e-4, atol=1e-5)
