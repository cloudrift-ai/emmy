"""The tensor-core atoms of a product of two f32 operands, which otherwise runs on the scalar tier.

``mma_m16n8k8_tf32_f32`` multiplies f32 values the instruction reads at ten bits of mantissa
(``mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32``, sm_80 and later) into an f32 accumulator. Its operands stay
f32 in memory and in registers, one to a fragment register, so K is 8. The product rounds each operand to TF32
first, since the instruction drops the low bits and a dot product does not average that away. Its gate is
``TF32_MMA``.

``mma_m16n8k8_tf32x3_f32`` is the split: the same fragments, three products a step over each operand's rounding and
what the rounding left, each step added to the accumulator by a float add. Its sums are as close to the exact one
as the scalar tier's, so no gate speaks for it; the catalog never offers it, and a row or a pin that names it takes
it.

The bf16 atoms take the same product with each operand rounded to bf16 in its fill: the smem compute fill evaluates
a stored operand of another dtype cell by cell and the typed store into its slab converts it, as it always did for
an f32 operand beside a bf16 one. Their gate on this product is ``F32_AS_BF16_MMA``.

Both gates follow ``FAST_MATH``; a ``TILE`` pin naming an atom bypasses its gate. An f32 operand beside a 16-bit one
keeps the 16-bit cell that converts it.
"""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import BF16, F8E4M3, F16, F32
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.atom import ATOM_REGISTRY, atoms_for
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import MatmulOp
from emmy.compiler.ir.kernel.ir import FragmentLoad, LdmatrixLoad, MmaSyncPtx
from emmy.compiler.ir.schedule import Placement
from emmy.compiler.ir.schedule.classic import ClassicProblem
from emmy.compiler.ir.schedule.classic import refusals as sched
from emmy.compiler.ir.schedule.classic.schedule import classic_node_key
from emmy.compiler.ir.schedule.staging import copied_b
from emmy.compiler.ir.stmt import parse_body
from emmy.compiler.ir.tile.ir import TileOp
from tests.compiler.helpers import compile_with_kernels, requires_cuda, requires_sm, statements
from tests.compiler.terms import contraction

TF32 = "mma_m16n8k8_tf32_f32"
SPLIT = "mma_m16n8k8_tf32x3_f32"
BF16_ATOM = "mma_m16n8k16_bf16_f32"


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


def _offered(tile, ctx, node) -> tuple[str, ...]:
    """The atoms the node's static facts allow, whatever the precision policy says."""
    return sched._warp_atoms(tile, ctx, node)


def test_the_split_is_the_tf32_cell_under_a_token_of_its_own_and_both_stand_last() -> None:
    plain, split = ATOM_REGISTRY[TF32], ATOM_REGISTRY[SPLIT]
    assert (split.shape, split.operand_dtypes, split.fragment_layout) == (plain.shape, plain.operand_dtypes, plain.fragment_layout)
    assert (plain.ab_dtype, split.ab_dtype) == ("tf32", "tf32x3")
    # Appended last, so no atom offered before them moved.
    assert list(ATOM_REGISTRY)[-2:] == [TF32, SPLIT] and atoms_for(F32) == (TF32, SPLIT)


def test_a_product_of_two_f32_operands_is_offered_the_tf32_atoms_and_the_bf16_ones() -> None:
    tile, ctx, node = _term()
    rounded = sched._f32_as_bf16_atoms(tile, ctx, node)
    assert BF16_ATOM in rounded and rounded == atoms_for(BF16, ctx=ctx)
    assert all([ATOM_REGISTRY[name].operand_dtype(role) for role in "abc"] == [BF16, BF16, F32] for name in rounded)
    assert _offered(tile, ctx, node) == (TF32, SPLIT, *rounded)
    assert _offered(*_term(cap=(8, 0)))[:2] == (TF32, SPLIT)
    # The instructions need sm_80. Volta multiplies f16 alone, and f16's range is not f32's: no cell there.
    assert sched._f32_as_bf16_atoms(*_term(cap=(7, 0))[:2], node) == () == _offered(*_term(cap=(7, 0)))


@pytest.mark.parametrize("dtypes", [(F32, F16), (F16, F32), (F16, F16), (F32, BF16), (BF16, F32), (BF16, BF16)])
def test_a_product_with_a_16_bit_operand_takes_that_operands_cell(dtypes) -> None:
    """The cell of 16 bits converts the f32 side in its fill, as it did before a product of f32 operands had cells."""
    tile, ctx, node = _term(a_dtype=dtypes[0], b_dtype=dtypes[1])
    narrow = next(dtype for dtype in dtypes if dtype != F32)
    atoms = _offered(tile, ctx, node)
    assert atoms and all(ATOM_REGISTRY[name].operand_dtype("a") == narrow for name in atoms)
    assert sched._f32_as_bf16_atoms(tile, ctx, node) == ()


def test_each_gate_decides_whether_the_catalog_offers_its_atoms() -> None:
    tile, ctx, node = _term()
    (site,) = ClassicProblem(tile, ctx).node_sites
    rounded = sched._f32_as_bf16_atoms(tile, ctx, node)

    def catalog(**policy: bool) -> tuple[str, ...]:
        return ClassicProblem(tile, ctx, **policy).catalog_atoms(site.id)

    assert catalog(allow_f32_as_bf16=True) == (TF32, *rounded)
    assert catalog() == (TF32,)
    assert catalog(allow_tf32=False, allow_f32_as_bf16=True) == rounded
    assert catalog(allow_tf32=False) == ()
    # What the node allows is the same either way: a pin that names an atom is a choice of its own.
    assert ClassicProblem(tile, ctx, allow_tf32=False).atoms_of(site.id) == (TF32, SPLIT, *rounded)
    # The second gate speaks for a product of f32 operands alone: a bf16 operand's own cell is offered whatever it says.
    narrow, narrow_ctx, _ = _term(a_dtype=BF16, b_dtype=BF16)
    (narrow_site,) = ClassicProblem(narrow, narrow_ctx).node_sites
    assert BF16_ATOM in ClassicProblem(narrow, narrow_ctx).catalog_atoms(narrow_site.id)


def test_the_split_is_taken_where_a_row_names_it_and_is_never_the_catalogs_offer() -> None:
    """The priors cannot tell the split from the TF32 atom, and it is the slower of the two, so no enumeration
    offers it. It trades no precision, so a row a descent follows deploys it whatever the gates say."""
    tile, ctx, _ = _term()
    problem = ClassicProblem(tile, ctx, allow_tf32=False)
    (site,) = problem.node_sites
    assert SPLIT not in problem.catalog_atoms(site.id) and SPLIT in problem.policy_atoms(site.id)
    assert not any(choice.tile.is_warp and choice.tile.atom.name == SPLIT for choice in site.nodes)
    followed = problem.with_row({classic_node_key(tile, "TILE", site.id): f"{SPLIT}/f2x2", "WORK": "w1x4"})
    assert [choice.tile.spell() for choice in followed.node_sites[0].nodes] == [f"{SPLIT}/f2x2"]
    # The TF32 atom is a precision decision: a followed row does not carry it past its gate.
    refused = problem.with_row({classic_node_key(tile, "TILE", site.id): f"{TF32}/f2x2", "WORK": "w1x4"})
    assert not any(choice.tile.is_warp for choice in refused.node_sites[0].nodes)


def test_the_priors_reading_of_tensor_core_eligibility_does_not_count_a_product_of_f32_operands() -> None:
    """The priors were fit with a product of f32 operands on the scalar tier's ranking; its atoms are offered beside
    it and that reading stays the one they learned."""
    tile, ctx, _ = _term()
    problem = ClassicProblem(tile, ctx)
    assert problem.atoms_of(problem.node_sites[0].id)[:2] == (TF32, SPLIT) and not problem.warp_eligible
    assert ClassicProblem(*_term(a_dtype=F16, b_dtype=F16)[:2]).warp_eligible
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
