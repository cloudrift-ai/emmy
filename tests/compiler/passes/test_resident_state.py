"""A carried state kept in register fragments (``STAGE=d1/reg``): the steps as one loop inside the CTA."""

from importlib import import_module

import numpy as np
import pytest

from emmy.compiler.backend.cuda.backend import CudaBackend
from emmy.compiler.context import Context
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.atom import ATOM_REGISTRY
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.kernel.ir import FragmentApply, FragmentRepack, MmaSyncPtx, RegStore
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.schedule import Stage
from emmy.compiler.ir.stmt import Body, Cond, Loop, Write, parse_body
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from emmy.compiler.pipeline.passes.lowering.kernel._resident import Unresident, resident_walk
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import compile_with_kernels, requires_cuda, statements

_HEADS, _STEPS = 2, 3

#: The atoms registers keep a state for, each with how far its recurrence may stand from the 64-bit one: ten bits
#: of mantissa an operand for halves and TF32, eight for bfloat16, and f32's own for the split.
_ATOMS = {
    "mma_m16n8k16_f16_f32": 2e-3,
    "mma_m16n8k16_f16_f16": 2e-3,
    "mma_m16n8k16_bf16_f32": 8e-3,
    "mma_m16n8k8_tf32_f32": 2e-3,
    "mma_m16n8k8_tf32x3_f32": 2e-6,
}


def _graph(n: int = 32) -> Graph:
    """A delta rule over ``_STEPS`` tokens: ``S = clamp(g) * S + k (v - k S) / 16``, every step's state stored."""
    body = parse_body(
        f"""
for a0 in 0..{_STEPS}:
  for a1 in 0..{_HEADS}:
    in0 = load g[a1, a0]
    v0 = maximum(0.0, in0)
    v1 = minimum(0.5, v0)
    for a2 in 0..{n}:
      in1 = load k[a1, a0, a2]
      v2 = 0.0625 * in1
      for a3 in 0..{n}:
        for a4 in 0..{n}:
          in2 = load k[a1, a0, a4]
          v3 = 0.0625 * in2
          v4 = pre acc0[a1, a4, a3]
          v5 = v3 * v4
          acc1 <- add(acc1, v5) over a4
        in3 = load v[a1, a0, a3]
        v6 = in3 - acc1
        v7 = v6 * v2
        v8 = pre acc0[a1, a2, a3]
        v9 = v1 * v8
        v10 = v7 + v9
        out[a0, a1, a2, a3] = v10
        acc0[a1, a2, a3] <- v10 seed s
"""
    )
    graph = Graph()
    shapes = {"g": (_HEADS, _STEPS), "k": (_HEADS, _STEPS, n), "s": (_HEADS, n, n), "v": (_HEADS, _STEPS, n)}
    for name, shape in shapes.items():
        graph.add_node(InputOp(), [], Tensor(name, shape, F32), node_id=name)
    graph.add_node(LoopOp(body=body), list(shapes), Tensor("out", (_STEPS, _HEADS, n, n), F32), node_id="out")
    graph.inputs, graph.outputs = list(shapes), ["out"]
    return graph


def _row(n: int, atom: str = "mma_m16n8k16_f16_f32") -> dict:
    return {"FAST_MATH": True, "PLACE": "fuse", "WORK": "w2x1", "STAGE": "d1/reg", "TILE": f"{atom}/f1x{n // 8}/k4"}


def _feed(n: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "g": rng.uniform(-0.2, 0.8, (_HEADS, _STEPS)).astype(np.float32),
        "k": (rng.standard_normal((_HEADS, _STEPS, n)) * 0.3).astype(np.float32),
        "s": (rng.standard_normal((_HEADS, n, n)) * 0.3).astype(np.float32),
        "v": (rng.standard_normal((_HEADS, _STEPS, n)) * 0.3).astype(np.float32),
    }


def _reference(feed: dict) -> np.ndarray:
    state, out = feed["s"].astype(np.float64), []
    for t in range(_STEPS):
        k, v = feed["k"][:, t].astype(np.float64) / 16, feed["v"][:, t].astype(np.float64)
        gate = np.clip(feed["g"][:, t].astype(np.float64), 0.0, 0.5)[:, None, None]
        read = np.einsum("hk,hkv->hv", k, state)
        state = k[:, :, None] * (v - read)[:, None, :] + gate * state
        out.append(state)
    return np.stack(out)


def _lifted_of(graph: Graph) -> TileOp:
    graph = Pipeline.build(["tile/lift"], select=["lift"]).run(graph)
    (tile,) = (node.op for node in graph.nodes.values() if isinstance(node.op, TileOp))
    return tile


def _lifted(n: int) -> TileOp:
    return _lifted_of(_graph(n))


def test_the_steps_run_as_one_loop_and_the_state_has_no_store():
    """The kernel holds the step loop itself, so the launch takes no step coordinate; it stores the output every
    step and never the state; and the node keeps the output alone."""
    n = 32
    with pinned_knobs(_row(n)):
        compiled, kernels = compile_with_kernels(_graph(n), ctx=Context.from_target((12, 0)))
    (kernel,) = kernels.values()
    (loop,) = [stmt for stmt in statements(kernel, Loop) if stmt.axis.name == "a0"]
    assert not kernel.serial and not statements(kernel, Write)
    assert {store.dst_buffer for store in statements(kernel, RegStore)} == {"out"}
    assert isinstance(loop.body[-1], FragmentApply) and loop.body[-1].in_place, "the state's fragments are overwritten last"
    assert [tensor.name for node in compiled.nodes.values() if node.id == "out" for tensor in node.outputs] == ["out"]


def test_a_value_that_does_not_read_the_column_is_computed_once():
    """``k S`` reads the state's row and no column, so it is one product for every column block of the state:
    its atom steps number the product's axis over ``atom_k``, not that times the column blocks."""
    n, atom = 32, ATOM_REGISTRY["mma_m16n8k16_f16_f32"]
    body = resident_walk(_lifted(n), atom, 2, 4).body
    assert len(list(body.iter_of_type(MmaSyncPtx))) == n // atom.atom_k


def test_a_state_the_fragments_do_not_tile_is_not_offered():
    """Sixteen rows a warp and eight columns a fragment: a state of 24 rows has no register row, and the builder
    says why."""
    assert _lifted(24).resident_state is None
    with pytest.raises(Unresident, match="whole fragments"):
        resident_walk(_lifted(24), ATOM_REGISTRY["mma_m16n8k16_f16_f32"], 2, 4)


@pytest.mark.parametrize("name", _ATOMS)
def test_a_states_fragments_become_the_operands_of_the_atom_that_reads_them(name):
    """The repack is told which atom it serves: halves for a 16-bit one, each at that atom's own format, and the
    f32 elements as they are for a TF32 one. Left untold it packed halves for every atom, and a bfloat16 product
    read them as bfloat16."""
    atom = ATOM_REGISTRY[name]
    body = resident_walk(_lifted(32), atom, 2, 4).body
    repacks = list(body.iter_of_type(FragmentRepack))
    assert repacks and {repack.ab_dtype for repack in repacks} == {atom.ab_dtype}
    # An operand of 32-bit floats is the columns of one accumulator; a pair of halves takes two.
    assert {len(repack.srcs) for repack in repacks if repack.role == "a"} == {atom.atom_k // atom.atom_n}


def _rows_graph(n: int = 32, seed: str = "0") -> Graph:
    """The inverse of a unit lower triangle less its diagonal, a row a step: row ``a0 + 1`` of the state becomes
    the matrix's row plus that row times the state, and every other row is kept. The reads of the written row choose
    their coordinates on the row's own test, as the roll of a row assignment writes them. The state starts from
    ``seed``: zero, or the tensor ``s``."""
    row = "a2 < a0 + 2 && a0 + 1 <= a2"
    body = parse_body(
        f"""
for a0 in 0..{n - 2}:
  for a1 in 0..{_HEADS}:
    for a2 in 0..{n}:
      for a3 in 0..{n}:
        for a4 in 0..{n}:
          in0 = load m[0, {row} ? a1 : 0, a0 + 1, a4]
          v0 = pre acc0[0, {row} ? a1 : 0, a4, {row} ? a3 : 0]
          v1 = in0 * v0
          acc1 <- add(acc1, v1) over a4
        in1 = load m[0, {row} ? a1 : 0, a0 + 1, {row} ? a3 : 0]
        v2 = acc1 + in1
        v3 = pre acc0[0, a1, a2, a3]
        v4 = select:
          v2 when {row}
          v3 when true
        out[0, a1, a2, a3] = v4
        acc0[0, a1, a2, a3] <- v4 seed {seed}
"""
    )
    graph = Graph()
    inputs = ["m", "s"] if seed == "s" else ["m"]
    for name in inputs:
        graph.add_node(InputOp(), [], Tensor(name, (1, _HEADS, n, n), F32), node_id=name)
    graph.add_node(LoopOp(body=body), inputs, Tensor("out", (1, _HEADS, n, n), F32), node_id="out")
    graph.inputs, graph.outputs = inputs, ["out"]
    return graph


def _rows_reference(matrix: np.ndarray, seed: np.ndarray | None = None) -> np.ndarray:
    matrix, n = matrix.astype(np.float64), matrix.shape[-1]
    state = np.zeros_like(matrix) if seed is None else seed.astype(np.float64)
    for i in range(1, n - 1):
        state[..., i, :] = matrix[..., i, :] + np.einsum("...k,...kc->...c", matrix[..., i, :], state)
    return state


@pytest.mark.parametrize("name", ["mma_m16n8k16_f16_f32", "mma_m16n8k8_tf32_f32"])
def test_a_step_that_writes_one_row_runs_the_product_of_one_block(name):
    """The state's next value is a choice: the product on the row the step writes, the state as it stands on every
    other. A block of eight rows that the written row is not in computes nothing, and the step loop is written out,
    so each step's kernel code is one block's product, its tests decided. The state is stored once, by the last
    step, and it lies under the size-one axis the row lift bound."""
    n, atom = 32, ATOM_REGISTRY[name]
    with pinned_knobs(_row(n, name)):
        _, kernels = compile_with_kernels(_rows_graph(n), ctx=Context.from_target((12, 0)))
    (kernel,) = kernels.values()
    assert not kernel.serial and not [stmt for stmt in statements(kernel, Loop) if stmt.axis.name == "a0"]
    assert not statements(kernel, Cond)
    assert len(statements(kernel, MmaSyncPtx)) == (n - 2) * (n // atom.atom_k)
    assert len(statements(kernel, RegStore)) == n // atom.atom_n


def test_a_step_that_keeps_part_of_its_state_tests_each_block_before_its_product():
    """Before the step loop is written out, each block's branch stands behind the test that the branch's condition
    reaches a row of the block; the read of the matrix's row is then the atom's own load, a row of memory under each
    row of the block, where the condition holds the written row to the step's."""
    n, atom = 32, ATOM_REGISTRY["mma_m16n8k16_f16_f32"]
    (loop,) = [stmt for stmt in resident_walk(_lifted_of(_rows_graph(n)), atom, 2, 4).body if isinstance(stmt, Loop)]
    tests = [stmt for stmt in loop.body if isinstance(stmt, Cond) and Body(stmt.body).iter_of_type(MmaSyncPtx)]
    assert loop.unroll and len(tests) == n // atom.atom_n
    assert [str(stmt.cond) for stmt in tests[:2]] == ["a0 + 1 < 8 && 0 < a0 + 2", "a0 + 1 < 16 && 8 < a0 + 2"]
    assert all(len(list(Body(stmt.body).iter_of_type(MmaSyncPtx))) == n // atom.atom_k for stmt in tests)
    assert {repack.role for repack in loop.body.iter_of_type(FragmentRepack)} == {"a"}


def test_registers_offer_the_tf32_atoms_under_their_gates():
    """The state is f32, and a TF32 operand is its fragments as they are. The catalog offers each atom where its
    gate allows; what a card's fragments can hold is the same whatever the gates say."""
    from emmy.compiler.ir.schedule.classic import ClassicProblem

    schedule = import_module("emmy.compiler.pipeline.passes.tile.schedule.040_schedule")
    tile, ctx = _lifted(32), Context.from_target((12, 0))
    tiles = schedule._resident_tiles(tile, ctx)
    assert {plan.atom.name for plan in tiles} >= {"mma_m16n8k16_f16_f32", "mma_m16n8k8_tf32_f32", "mma_m16n8k8_tf32x3_f32"}
    serial = schedule.serial_form(tile, "out").scheduled()

    def offered(**gates) -> set[str]:
        problem = ClassicProblem(serial, ctx, state_stages=(Stage(transport="reg"),), state_tiles=tiles, **gates)
        return {kernel.state_tile.atom.name for kernel in problem.kernel_site.kernels if kernel.state_tile is not None}

    assert "mma_m16n8k8_tf32_f32" in offered(allow_tf32=True) and "mma_m16n8k8_tf32_f32" not in offered(allow_tf32=False)
    assert "mma_m16n8k8_tf32x3_f32" not in offered(allow_tf32=True)
    assert "mma_m16n8k8_tf32x3_f32" in offered(allow_tf32_split=True)


@requires_cuda
@pytest.mark.parametrize("atom", _ATOMS)
@pytest.mark.parametrize("n", [32, 64])
def test_a_state_in_registers_matches_the_recurrence(atom, n):
    with pinned_knobs(_row(n, atom)):
        compiled = Pipeline.build(CUDA_PASSES).run(_graph(n), ctx=Context.probe())
    backend = CudaBackend()
    for seed in (0, 1):
        feed = _feed(n, seed)
        got = np.asarray(backend.run(compiled, input_data=dict(feed))[0].outputs["out"], dtype=np.float64)
        np.testing.assert_allclose(got, _reference(feed), atol=_ATOMS[atom])


@requires_cuda
@pytest.mark.parametrize("atom", _ATOMS)
@pytest.mark.parametrize(("n", "seed"), [(32, "0"), (32, "s"), (64, "s")])
def test_a_row_written_a_step_matches_the_substitution(atom, n, seed):
    """From zero the fragments are the seed as declared; from a tensor they are filled ahead of the steps, since a
    step that writes one row reads no seed into the rows it keeps."""
    with pinned_knobs(_row(n, atom)):
        compiled = Pipeline.build(CUDA_PASSES).run(_rows_graph(n, seed), ctx=Context.probe())
    backend = CudaBackend()
    for draw in (0, 1):
        rng = np.random.default_rng(draw)
        feed = {"m": np.tril(rng.standard_normal((1, _HEADS, n, n)) * 0.2, -1).astype(np.float32)}
        if seed == "s":
            feed["s"] = (rng.standard_normal((1, _HEADS, n, n)) * 0.2).astype(np.float32)
        got = np.asarray(backend.run(compiled, input_data=dict(feed))[0].outputs["out"], dtype=np.float64)
        np.testing.assert_allclose(got, _rows_reference(feed["m"], feed.get("s")), atol=_ATOMS[atom])


@requires_cuda
def test_a_row_step_left_a_loop_takes_its_tests_at_run_time():
    """With the unroll budget at zero the steps stay one loop, each block's branch behind its test of the step; the
    answer is the written-out kernel's."""
    n = 32
    with pinned_knobs({**_row(n), "UNROLL": 0}):
        compiled, kernels = compile_with_kernels(_rows_graph(n, "s"), ctx=Context.probe())
    (kernel,) = kernels.values()
    assert [stmt for stmt in statements(kernel, Loop) if stmt.axis.name == "a0"] and statements(kernel, Cond)
    rng = np.random.default_rng(0)
    feed = {
        "m": np.tril(rng.standard_normal((1, _HEADS, n, n)) * 0.2, -1).astype(np.float32),
        "s": (rng.standard_normal((1, _HEADS, n, n)) * 0.2).astype(np.float32),
    }
    got = np.asarray(CudaBackend().run(compiled, input_data=dict(feed))[0].outputs["out"], dtype=np.float64)
    np.testing.assert_allclose(got, _rows_reference(feed["m"], feed["s"]), atol=2e-3)
