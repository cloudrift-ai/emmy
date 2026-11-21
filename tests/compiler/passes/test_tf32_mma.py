"""The TF32 tensor-core atom: a product of two f32 operands on tensor cores.

``mma_m16n8k8_tf32_f32`` multiplies f32 values the instruction reads at ten bits of mantissa
(``mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32``, sm_80 and later) into an f32 accumulator. Its operands stay
f32 in memory and in registers, one to a fragment register, so K is 8. It is offered where both operands of a
product are f32, which otherwise has no tensor-core cell; an f32 operand beside a 16-bit one keeps the 16-bit cell
that converts it. Enumeration is precision-gated (``TF32_MMA``, under the ``FAST_MATH`` umbrella); a ``TILE`` pin
naming the atom bypasses the gate. The product rounds each operand to TF32 first, since the instruction drops the
low bits and a dot product does not average that away.

``mma_m16n8k8_tf32x3_f32`` is the split: the same fragments, three products a step over each operand's rounding and
what the rounding left, each step added to the accumulator by a float add. Its sums are as close to the exact one
as the scalar tier's. Its gate (``TF32_SPLIT_MMA``) is off unless pinned, whatever fast math says.
"""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import BF16, F16, F32
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.atom import ATOM_REGISTRY, atoms_for
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import MatmulOp
from emmy.compiler.ir.kernel.ir import FragmentLoad, MmaSyncPtx
from emmy.compiler.ir.schedule import Placement
from emmy.compiler.ir.schedule.classic import ClassicProblem
from emmy.compiler.ir.schedule.classic import refusals as sched
from emmy.compiler.ir.stmt import parse_body
from emmy.compiler.ir.tile.ir import TileOp
from tests.compiler.helpers import compile_with_kernels, requires_cuda, requires_sm, statements
from tests.compiler.terms import contraction

TF32 = "mma_m16n8k8_tf32_f32"
SPLIT = "mma_m16n8k8_tf32x3_f32"


def _term(cap=(12, 0), *, a_dtype=F32, b_dtype=F32, k=512):
    """A product of ``a[m, k]`` and ``w[k, n]`` as a kernel, its target and its contraction."""
    ka = Axis("k", Dim(k))
    a, b = parse_body(f"{a_dtype.name} ab = load a[m, k]\n{b_dtype.name} wb = load w[k, n]")
    node = contraction(ka, a, (b, "acc"))
    inputs = {"a": Tensor("a", (32, k), a_dtype), "w": Tensor("w", (k, 512), b_dtype)}
    free = (Axis("m", Dim(32)), Axis("n", Dim(512)))
    tile = TileOp(op=node, place=Placement(free=free), axes=(*free, ka), inputs=inputs)
    return tile, Context.from_target(cap), node


def _offered(tile, ctx, node) -> tuple[str, ...]:
    """The atoms the node's static facts allow, whatever the precision policy says."""
    return sched._warp_atoms(tile, ctx, node)


def test_the_atom_is_one_32_bit_element_to_a_register_and_k_is_8() -> None:
    atom = ATOM_REGISTRY[TF32]
    assert (atom.shape, atom.ptx_shape, atom.ab_dtype) == ((16, 8, 8), (16, 8, 8), "tf32")
    assert [atom.operand_dtype(role) for role in "abc"] == [F32, F32, F32]
    assert [atom.fragment_registers(role) for role in "abc"] == [4, 2, 4]
    # A state held in accumulator fragments does not become this atom's operand by a move of registers alone: the
    # lanes of the two layouts do not line up, so the register and chunk storage do not offer it.
    assert not atom.c_to_a_repack and not atom.c_to_b_repack
    # Appended last, so no atom offered before them moved.
    assert list(ATOM_REGISTRY)[-2:] == [TF32, SPLIT] and atoms_for(F32) == (TF32, SPLIT)


def test_the_split_is_the_same_cell_under_a_token_of_its_own() -> None:
    plain, split = ATOM_REGISTRY[TF32], ATOM_REGISTRY[SPLIT]
    assert (split.shape, split.operand_dtypes, split.fragment_layout) == (plain.shape, plain.operand_dtypes, plain.fragment_layout)
    assert (plain.ab_dtype, split.ab_dtype) == ("tf32", "tf32x3")


def test_a_product_of_two_f32_operands_is_offered_the_two_atoms_and_no_other() -> None:
    assert _offered(*_term()) == (TF32, SPLIT)
    # The instruction needs sm_80; Volta has no cell for two f32 operands.
    assert _offered(*_term(cap=(8, 0))) == (TF32, SPLIT)
    assert _offered(*_term(cap=(7, 0))) == ()


@pytest.mark.parametrize("narrow", [F16, BF16])
def test_an_f32_operand_beside_a_16_bit_one_keeps_the_16_bit_cell(narrow) -> None:
    """The cell of 16 bits converts the f32 side in its fill, as it did before the TF32 atom existed."""
    for a_dtype, b_dtype in ((F32, narrow), (narrow, F32), (narrow, narrow)):
        atoms = _offered(*_term(a_dtype=a_dtype, b_dtype=b_dtype))
        assert atoms and all(ATOM_REGISTRY[name].operand_dtype("a") == narrow for name in atoms), (a_dtype, b_dtype)


def test_each_gate_decides_whether_the_catalog_offers_its_atom() -> None:
    tile, ctx, _ = _term()
    (site,) = ClassicProblem(tile, ctx).node_sites

    def offered(tf32: bool, split: bool) -> tuple[str, ...]:
        return ClassicProblem(tile, ctx, allow_tf32=tf32, allow_tf32_split=split).policy_atoms(site.id)

    assert (offered(True, False), offered(False, True), offered(True, True), offered(False, False)) == (
        (TF32,),
        (SPLIT,),
        (TF32, SPLIT),
        (),
    )
    # What the node allows is the same either way: a row or a pin that names an atom is a choice of its own.
    assert ClassicProblem(tile, ctx, allow_tf32=False, allow_tf32_split=False).atoms_of(site.id) == (TF32, SPLIT)


def test_the_gate_follows_fast_math_unless_pinned(monkeypatch) -> None:
    from emmy.compiler.pipeline.search.space import PRECISION_KNOBS, TF32_MMA, precision_pin

    assert TF32_MMA in PRECISION_KNOBS
    for var in ("EMMY_TF32_MMA", "EMMY_FAST_MATH"):
        monkeypatch.delenv(var, raising=False)
    assert precision_pin(TF32_MMA) is True
    monkeypatch.setenv("EMMY_FAST_MATH", "0")
    assert precision_pin(TF32_MMA) is False
    monkeypatch.setenv("EMMY_TF32_MMA", "1")
    assert precision_pin(TF32_MMA) is True
    monkeypatch.setenv("EMMY_FAST_MATH", "1")
    monkeypatch.setenv("EMMY_TF32_MMA", "0")
    assert precision_pin(TF32_MMA) is False


def test_the_splits_gate_is_off_unless_pinned_whatever_fast_math_says(monkeypatch) -> None:
    """The priors cannot tell the split from the TF32 atom, and it is the slower of the two: only a pin offers it."""
    from emmy.compiler.pipeline.search.space import PRECISION_KNOBS, TF32_SPLIT_MMA, precision_pin

    assert TF32_SPLIT_MMA in PRECISION_KNOBS
    monkeypatch.delenv("EMMY_TF32_SPLIT_MMA", raising=False)
    for fast_math in ("0", "1"):
        monkeypatch.setenv("EMMY_FAST_MATH", fast_math)
        assert precision_pin(TF32_SPLIT_MMA) is False
    monkeypatch.setenv("EMMY_FAST_MATH", "0")
    monkeypatch.setenv("EMMY_TF32_SPLIT_MMA", "1")
    assert precision_pin(TF32_SPLIT_MMA) is True


def test_the_priors_reading_of_tensor_core_eligibility_does_not_count_the_atom() -> None:
    """The priors were fit with a product of f32 operands on the scalar tier's ranking; the atom is offered beside
    it and that reading stays the one they learned."""
    tile, ctx, _ = _term()
    problem = ClassicProblem(tile, ctx)
    assert problem.atoms_of(problem.node_sites[0].id) == (TF32, SPLIT) and not problem.warp_eligible
    narrow = ClassicProblem(*_term(a_dtype=F16, b_dtype=F16)[:2])
    assert narrow.warp_eligible


def _matmul(m: int, n: int, k: int) -> Graph:
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("a", (m, k), F32), node_id="a")
    graph.add_node(InputOp(), [], Tensor("b", (k, n), F32), node_id="b")
    graph.add_node(MatmulOp(), ["a", "b"], Tensor("o", (m, n), F32), node_id="o")
    graph.inputs, graph.outputs = ["a", "b"], ["o"]
    return graph


def _products(kernels) -> list:
    return [mma for kernel in kernels.values() for mma in statements(kernel, MmaSyncPtx) if tuple(mma.shape) == (16, 8, 8)]


@requires_cuda
@pytest.mark.xdist_group("cuda")
@requires_sm(8, 0)
@pytest.mark.parametrize("stage", ["", "d2/smem-async"], ids=["from global memory", "staged"])
def test_a_product_of_two_f32_matrices_runs_on_the_atom_and_matches_numpy(stage: str) -> None:
    """Each operand is rounded to TF32 and the sum is f32. Small integers are exact at ten bits, so the first
    product is exact; the second is within what ten bits of mantissa a side give a sum of 256 terms."""
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    m, n, k = 64, 128, 256
    rng = np.random.default_rng(3)
    backend = CudaBackend()
    with pinned_knobs({"TILE": f"{TF32}/f2x2", "WORK": "w1x4", "REDUCE": "", "STAGE": stage}):
        compiled, kernels = compile_with_kernels(_matmul(m, n, k))
    products = _products(kernels)
    assert products and all((mma.ab_dtype, mma.c_dtype) == ("tf32", "f32") for mma in products)
    loads = [load for kernel in kernels.values() for load in statements(kernel, FragmentLoad)]
    assert {load.role for load in loads} == {"a", "b"}
    assert all(load.staged == bool(stage) and load.dtype == F32 for load in loads)

    a = rng.integers(-8, 9, (m, k)).astype(np.float32)
    b = rng.integers(-8, 9, (k, n)).astype(np.float32)
    result, _ = backend.run(compiled, input_data={"a": a, "b": b})
    np.testing.assert_array_equal(result.outputs["o"].reshape(m, n), a @ b)

    a = rng.standard_normal((m, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)
    result, _ = backend.run(compiled, input_data={"a": a, "b": b})
    exact = a.astype(np.float64) @ b.astype(np.float64)
    # Rounding to nearest leaves each product within 2**-10 of itself and of either sign, so the sum of 256 of
    # them stays well inside this; dropping the low bits instead makes every output about 1e-3 small.
    assert float(np.abs(result.outputs["o"].reshape(m, n) - exact).max()) < 2e-2
    assert abs(float(np.mean((result.outputs["o"].reshape(m, n) - exact) * np.sign(exact)))) < 2e-4


@requires_cuda
@pytest.mark.xdist_group("cuda")
@requires_sm(8, 0)
@pytest.mark.parametrize("k", [256, 4096])
def test_the_split_is_as_close_to_the_exact_product_as_f32_arithmetic(k: int) -> None:
    """Three TF32 products a step recover what one drops, and each step is added by a float add, since the
    instruction truncates as it accumulates: left inside it, the sum of 512 steps drifted by 2e-5 of the result."""
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    m, n = 64, 128
    rng = np.random.default_rng(5)
    backend = CudaBackend()
    with pinned_knobs({"TILE": f"{SPLIT}/f2x2", "WORK": "w1x4", "REDUCE": "", "STAGE": ""}):
        compiled, kernels = compile_with_kernels(_matmul(m, n, k))
    products = _products(kernels)
    assert products and all(mma.ab_dtype == "tf32x3" for mma in products)
    a = rng.standard_normal((m, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)
    result, _ = backend.run(compiled, input_data={"a": a, "b": b})
    exact = a.astype(np.float64) @ b.astype(np.float64)
    scale = float(np.sqrt(np.mean(exact**2)))
    error = result.outputs["o"].reshape(m, n) - exact
    # What numpy's own f32 product leaves is the bar: the split is within twice it, and without a sign.
    bar = float(np.sqrt(np.mean(((a @ b) - exact) ** 2)))
    assert float(np.sqrt(np.mean(error**2))) < 2 * bar
    assert abs(float(np.mean(error * np.sign(exact)))) < 2e-7 * scale
