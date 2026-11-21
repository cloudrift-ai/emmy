"""A product of two f32 operands on the bf16 tensor-core atoms, each operand rounded to bf16 in its fill.

An f32 operand beside a bf16 one has always taken the bf16 cell: the smem compute fill evaluates it cell by cell and
the typed store into its slab converts it. A product whose operands are all f32 takes the same cell by the same
fill, on both sides. bf16 keeps f32's exponent range and eight bits of mantissa, so the offer is precision-gated
(``F32_AS_BF16_MMA``, under the ``FAST_MATH`` umbrella); a ``TILE`` pin naming the atom bypasses the gate. The
accumulator stays f32.
"""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.dtype import BF16, F8E4M3, F16, F32
from emmy.compiler.graph import Tensor
from emmy.compiler.ir.atom import ATOM_REGISTRY, atoms_for
from emmy.compiler.ir.kernel.ir import LdmatrixLoad, MmaSyncPtx
from emmy.compiler.ir.schedule.classic import ClassicProblem
from emmy.compiler.ir.schedule.classic import refusals as sched
from emmy.compiler.ir.schedule.staging import copied_b
from tests.compiler.helpers import compile_with_kernels, requires_cuda, requires_sm, statements
from tests.compiler.passes.test_tf32_mma import SPLIT, TF32, _matmul, _offered, _term

BF16_ATOM = "mma_m16n8k16_bf16_f32"


def test_a_product_of_two_f32_operands_is_offered_the_bf16_atoms_beside_the_tf32_ones() -> None:
    tile, ctx, node = _term()
    rounded = sched._rounded_atoms(tile, ctx, node)
    assert BF16_ATOM in rounded and rounded == atoms_for(BF16, ctx=ctx)
    assert all([ATOM_REGISTRY[name].operand_dtype(role) for role in "abc"] == [BF16, BF16, F32] for name in rounded)
    assert _offered(tile, ctx, node) == (TF32, SPLIT, *rounded)
    # Volta multiplies f16 alone, and f16's range is not f32's: no cell there.
    assert sched._rounded_atoms(*_term(cap=(7, 0))[:2], node) == () == _offered(*_term(cap=(7, 0)))


@pytest.mark.parametrize("dtypes", [(F32, BF16), (BF16, F32), (F16, F32), (BF16, BF16)])
def test_a_product_with_a_16_bit_operand_takes_that_operands_cell_and_no_rounded_one(dtypes) -> None:
    tile, ctx, node = _term(a_dtype=dtypes[0], b_dtype=dtypes[1])
    narrow = next(dtype for dtype in dtypes if dtype != F32)
    assert sched._rounded_atoms(tile, ctx, node) == ()
    assert all(ATOM_REGISTRY[name].operand_dtype("a") == narrow for name in _offered(tile, ctx, node))


def test_the_gate_decides_whether_the_catalog_offers_the_rounded_atoms() -> None:
    tile, ctx, node = _term()
    (site,) = ClassicProblem(tile, ctx).node_sites
    rounded = sched._rounded_atoms(tile, ctx, node)
    assert ClassicProblem(tile, ctx, allow_f32_as_bf16=True).policy_atoms(site.id) == (TF32, *rounded)
    assert ClassicProblem(tile, ctx, allow_f32_as_bf16=False).policy_atoms(site.id) == (TF32,)
    assert ClassicProblem(tile, ctx, allow_tf32=False, allow_f32_as_bf16=True).policy_atoms(site.id) == rounded
    # What the node allows is the same either way: a row or a pin that names an atom is a choice of its own.
    assert ClassicProblem(tile, ctx, allow_f32_as_bf16=False).atoms_of(site.id) == (TF32, SPLIT, *rounded)
    # The gate speaks for a product of f32 operands alone: a bf16 operand's own cell is offered whatever it says.
    narrow, narrow_ctx, _ = _term(a_dtype=BF16, b_dtype=BF16)
    (narrow_site,) = ClassicProblem(narrow, narrow_ctx).node_sites
    assert BF16_ATOM in ClassicProblem(narrow, narrow_ctx, allow_f32_as_bf16=False).policy_atoms(narrow_site.id)


def test_the_gate_follows_fast_math_unless_pinned(monkeypatch) -> None:
    from emmy.compiler.pipeline.search.space import F32_AS_BF16_MMA, PRECISION_KNOBS, precision_pin

    assert F32_AS_BF16_MMA in PRECISION_KNOBS
    for var in ("EMMY_F32_AS_BF16_MMA", "EMMY_FAST_MATH"):
        monkeypatch.delenv(var, raising=False)
    assert precision_pin(F32_AS_BF16_MMA) is True
    monkeypatch.setenv("EMMY_FAST_MATH", "0")
    assert precision_pin(F32_AS_BF16_MMA) is False
    monkeypatch.setenv("EMMY_F32_AS_BF16_MMA", "1")
    assert precision_pin(F32_AS_BF16_MMA) is True
    monkeypatch.setenv("EMMY_FAST_MATH", "1")
    monkeypatch.setenv("EMMY_F32_AS_BF16_MMA", "0")
    assert precision_pin(F32_AS_BF16_MMA) is False


def test_the_priors_reading_of_tensor_core_eligibility_does_not_count_the_rounded_atoms() -> None:
    """The priors were fit with a product of f32 operands on the scalar tier's ranking; the rounded atoms are offered
    beside it and that reading stays the one they learned."""
    assert not ClassicProblem(*_term()[:2], allow_f32_as_bf16=True).warp_eligible
    assert ClassicProblem(*_term(a_dtype=F32, b_dtype=BF16)[:2]).warp_eligible


def test_the_fill_converts_a_stored_operand_of_another_dtype_where_a_copy_moves_its_bytes() -> None:
    """A copy transport moves bytes, so a stored operand that holds another dtype than its slab's is filled cell by
    cell like a computed one, and the typed store converts it. A slab of bytes stays a copy: its fragment loads
    convert it."""
    tile, _, node = _term()
    edge = node.operands[1]
    assert copied_b(node, edge, tile.inputs)  # by its address alone, as the copy transports ask
    assert not copied_b(node, edge, tile.inputs, BF16)
    assert copied_b(node, edge, tile.inputs, F32)
    for stored in (BF16, F8E4M3):
        inputs = {**tile.inputs, "w": Tensor("w", tile.inputs["w"].shape, stored)}
        assert copied_b(node, edge, inputs, BF16)


@requires_cuda
@pytest.mark.xdist_group("cuda")
@requires_sm(8, 0)
def test_a_product_of_two_f32_matrices_runs_on_the_bf16_atom_and_matches_numpy() -> None:
    """Both slabs are filled with the operands rounded to bf16 and read back by ``ldmatrix``; the sum is f32.
    Small integers are exact at eight bits, so the first product is exact; the second is within what eight bits of
    mantissa a side give a sum of 256 terms, and has no sign."""
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    m, n, k = 64, 128, 256
    rng = np.random.default_rng(7)
    backend = CudaBackend()
    with pinned_knobs({"TILE": f"{BF16_ATOM}/f2x2/k4", "WORK": "w1x4", "REDUCE": "", "STAGE": "d1/smem"}):
        compiled, kernels = compile_with_kernels(_matmul(m, n, k))
    products = [mma for kernel in kernels.values() for mma in statements(kernel, MmaSyncPtx)]
    assert products and all((mma.ab_dtype, mma.c_dtype) == ("bf16", "f32") for mma in products)
    assert {load.role for kernel in kernels.values() for load in statements(kernel, LdmatrixLoad)} == {"a", "b"}

    a = rng.integers(-8, 9, (m, k)).astype(np.float32)
    b = rng.integers(-8, 9, (k, n)).astype(np.float32)
    result, _ = backend.run(compiled, input_data={"a": a, "b": b})
    np.testing.assert_array_equal(result.outputs["o"].reshape(m, n), a @ b)

    a = rng.standard_normal((m, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)
    result, _ = backend.run(compiled, input_data={"a": a, "b": b})
    exact = a.astype(np.float64) @ b.astype(np.float64)
    error = result.outputs["o"].reshape(m, n) - exact
    scale = float(np.sqrt(np.mean(exact**2)))
    # Each operand is within 2**-9 of itself, so a product is within about 2**-8 of itself and the sum of 256 of
    # either sign stays near that fraction of the result's size.
    assert float(np.sqrt(np.mean(error**2))) < 4e-3 * scale
    assert abs(float(np.mean(error * np.sign(exact)))) < 4e-4 * scale
