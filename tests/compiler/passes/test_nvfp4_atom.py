"""The block-scaled FP4 atom — ``m16n8k64`` with the scales the instruction applies itself.

Every other registered mma takes three fragments (A, B, and the accumulator standing in for both
C and D). This one takes two more: a packed b32 of ue4m3 block scales per side, one scale per 16
K elements, which the hardware multiplies into each block's partial product. That extra pair is
what the IR node and the fragment loaders here exist to carry. The product on a card is the PTX
compiler's test (``tests/compiler/ptx/test_tensor_cores.py``).

It also compiles differently. ptxas assembles this instruction only for the arch-SUFFIXED target
(``sm_120a``) and refuses it on ``sm_100a``, so a kernel carrying it has the same requirement TMA
has, for an unrelated reason — hence the shared ``arch_specific`` flag rather than a TMA-named one.

Nothing offers this atom to the scheduler yet: both operands would have to arrive already
quantized, and quantizing an activation is its own step. These tests cover the pieces that exist.
"""

from __future__ import annotations

import pytest

from emmy.compiler.context import Context
from emmy.compiler.dtype import F32, F4E2M1x2
from emmy.compiler.ir.atom import ATOM_REGISTRY
from emmy.compiler.ir.kernel.ir import MmaSyncPtx

ATOM = "mma_m16n8k64_e2m1_f32"


def test_the_atom_is_registered_with_packed_operands_and_an_f32_accumulator():
    atom = ATOM_REGISTRY[ATOM]
    assert atom.shape == (16, 8, 64)  # k counts VALUES, which is twice the bytes a row holds
    assert atom.operand_dtype("a") is F4E2M1x2
    assert atom.operand_dtype("b") is F4E2M1x2
    assert atom.operand_dtype("c") is F32
    assert (atom.fragment_registers("a"), atom.fragment_registers("b")) == (4, 2)
    assert atom.target_feature == "has_block_scaled_f4_mma"


def test_the_atom_names_the_ptx_dtype_token():
    # ``ab_dtype`` is the PTX dtype token the compiler spells the instruction with.
    assert ATOM_REGISTRY[ATOM].ab_dtype == "e2m1"


def test_the_atom_declares_no_staged_drain():
    """Conservative first scope: no staged slab drain exists for a packed-plus-scales operand.

    This used to ride an ``AtomKind.materialized_edges_only`` flag. That flag is gone — a pass may
    not refuse for profitability — and the refusal it stood for was never about profitability: there
    is simply no code to emit. It now lives in the stage resolver, and the atom only declares that
    it has no synchronous-copy staging either."""
    assert not ATOM_REGISTRY[ATOM].sync_copy_staging


@pytest.mark.parametrize(
    ("cc", "available"),
    [((12, 0), True), ((10, 0), False), ((9, 0), False), ((8, 9), False), ((8, 0), False)],
)
def test_only_the_consumer_blackwell_family_offers_the_atom(cc, available):
    """cc 10.0 is datacenter Blackwell, where ptxas refuses this instruction — so the check asks
    for the 12 family rather than 'Blackwell or newer'."""
    assert ATOM_REGISTRY[ATOM].available_on(Context(compute_capability=cc)) is available


def test_the_mma_node_carries_both_scale_fragments():
    node = MmaSyncPtx(c_frag="c", a_frag="a", b_frag="b", shape=(16, 8, 64), ab_dtype="e2m1", sfa_frag="sa", sfb_frag="sb")
    assert node.sfa_frag is not None
    # The scales are read, so a reordering pass has to keep them defined before the mma.
    assert node.deps() == ("c", "a", "b", "sa", "sb")


def test_a_plain_mma_node_carries_no_scales():
    node = MmaSyncPtx(c_frag="c", a_frag="a", b_frag="b", shape=(16, 8, 16))
    assert node.sfa_frag is None
    assert node.deps() == ("c", "a", "b")


def test_a_kernel_carrying_the_block_scaled_mma_asks_for_the_arch_suffixed_target():
    """ptxas assembles the block-scaled fp4 ``mma`` only for the arch-suffixed target, as it does TMA; the kernel's
    statements say which a kernel holds."""
    from emmy.compiler.ir.kernel import KernelOp
    from emmy.compiler.ir.kernel.interface import needs_arch_isa
    from emmy.compiler.ir.kernel.ir import TmaDescriptor
    from emmy.compiler.ir.stmt import Body

    scaled = MmaSyncPtx(c_frag="c", a_frag="a", b_frag="b", shape=(16, 8, 64), ab_dtype="e2m1", sfa_frag="sa", sfb_frag="sb")
    plain = MmaSyncPtx(c_frag="c", a_frag="a", b_frag="b", shape=(16, 8, 16))
    tma = TmaDescriptor(name="desc", src_buf="x", src_shape=(16, 16), box_extents=(16, 16))
    assert needs_arch_isa(KernelOp(body=Body((scaled,)), name="k"))
    assert needs_arch_isa(KernelOp(body=Body((tma,)), name="k"))
    assert not needs_arch_isa(KernelOp(body=Body((plain,)), name="k"))
