"""Hopper's instructions compiled to PTX by the ``emmy-ptx`` crate and launched on a card: a tensor copy by the copy
engine with the barrier its bytes arrive at, ``wgmma`` over swizzled slabs through descriptors and with A in
registers, and the limit on a warp's registers. Each is compared with NumPy bit for bit."""

from __future__ import annotations

import textwrap

import numpy as np
import pytest

from emmy.compiler.ir.cuda import TmaDescMeta
from tests.compiler.helpers import device_compute_capability, requires_cuda, requires_sm
from tests.compiler.ptx import launch
from tests.compiler.ptx.launch import same_bits, under

pytestmark = [
    requires_cuda,
    requires_sm(9, 0),
    pytest.mark.xdist_group("cuda"),
    pytest.mark.parametrize("level", launch.LEVELS, ids=["default", "O0"]),
]

#: ``wgmma`` is assembled for ``sm_90a`` alone.
hopper_only = pytest.mark.skipif((device_compute_capability() or (0, 0))[0] != 9, reason="wgmma is assembled for sm_90a alone")

BLOCKS = 3


def test_a_box_of_a_tensor_is_copied_into_a_slab_every_thread_waits_for(level):
    """Two boxes of 16 x 16 floats out of a 32 x 16 tensor, one a phase of one barrier. Thread 0 starts the
    barrier, and for each phase states the copy's bytes and issues the copy; every thread waits for the phase, by
    its parity, before it reads its element of the slab. The block's barrier between the two phases keeps a thread
    that is late to the first wait from meeting the barrier a whole turn on."""
    x = np.arange(BLOCKS * 512, dtype=np.float32).reshape(BLOCKS * 32, 16)
    grid = f"Grid(axes=(r in 0..{BLOCKS}, t in 0..256), decode=(block_idx.x, thread_idx.x), block_threads=256):"
    body = under(
        grid,
        """
        Smem(name=tile, extents=(2, 16, 16), dtype=f32, align=128)
        Smem(name=bar, extents=(1), dtype=u64)
        TmaDescriptor(name=x_desc, src_buf=x, src_shape=(96, 16), box_extents=(16, 16))
        if t == 0:
            MbarrierInit(mbar=bar, count=1, slot=0)
        Sync
        if t == 0:
            MbarrierArriveExpectTx(mbar=bar, bytes_=1024, slot=0)
            TmaLoad(smem=tile, smem_index=(0, 0, 0), desc=x_desc, coords=(r * 32, 0), mbar=bar, mbar_slot=0)
        MbarrierWait(mbar=bar, phase=0, slot=0)
        Sync
        if t == 0:
            MbarrierArriveExpectTx(mbar=bar, bytes_=1024, slot=0)
            TmaLoad(smem=tile, smem_index=(1, 0, 0), desc=x_desc, coords=(r * 32 + 16, 0), mbar=bar, mbar_slot=0)
        MbarrierWait(mbar=bar, phase=1, slot=0)
        f32 first = load tile[t]
        f32 second = load tile[256 + t]
        f32 u = first + 1.0
        f32 v = second + 1.0
        f32 y[r * 512 + t] = u
        f32 y[r * 512 + 256 + t] = v
        """,
    )
    descriptors = (TmaDescMeta("x_desc", "x", (16, 16)),)
    out = launch.run(body, {"x": x}, {"y": (np.float32, x.size)}, blocks=BLOCKS, threads=256, level=level, descriptors=descriptors)["y"]
    same_bits(out, x.ravel() + 1)


def _moved(index: str) -> str:
    """Where the element at ``index`` of a slab with 64 halves a row stands under the 128-byte swizzle."""
    return f"({index}) ^ (((({index}) >> 6) & 7) << 3)"


@hopper_only
@pytest.mark.parametrize("form", ["ss", "rs"])
def test_four_warps_multiply_a_64_by_16_tile_from_swizzled_slabs(level, form):
    """A 64 x 16 tile of ``A @ B`` over K = 64, four ``wgmma`` steps of 16. The 128 threads stage A, and B with K
    along its rows, into slabs under the 128-byte swizzle with plain stores. Both operands are read through
    descriptors (``ss``), or A is loaded into each warp's own A fragment by ``ldmatrix`` (``rs``). A warp's
    accumulator is two m16n8 fragments side by side, rows ``16 * warp`` on."""
    k = 64
    rng = np.random.default_rng(31)
    a = rng.integers(-8, 9, size=(BLOCKS, 64, k)).astype(np.float32)
    b = rng.integers(-8, 9, size=(BLOCKS, k, 16)).astype(np.float32)
    descriptor = "smem_index=s * 16, swizzle=B128, lbo_bytes=16, sbo_bytes=1024)"
    if form == "ss":
        step = f"""
        WgmmaDescriptor(name=da, smem=sa, {descriptor}
        WgmmaDescriptor(name=db, smem=sb, {descriptor}
        WgmmaMma(c_frags=(c0, c1), b_desc=db, shape=(64, 16, 16), a_desc=da)
        """
    else:
        step = f"""
        LdmatrixLoad(frag=fa, src_buffer=sa, src_index=(w * 16, s * 16), role=a, ldm=64, swizzle=B128)
        WgmmaDescriptor(name=db, smem=sb, {descriptor}
        WgmmaMma(c_frags=(c0, c1), b_desc=db, shape=(64, 16, 16), a_frag=fa)
        """
    store = "\n".join(
        line
        for j in range(2)
        for i in range(4)
        for line in (
            f"f32 c{j}_{i} = load c{j}[{i}]",
            f"f32 o[r * 1024 + (w * 16 + g + {8 * (i // 2)}) * 16 + {8 * j} + q * 2 + {i % 2}] = c{j}_{i}",
        )
    )
    grid = f"Grid(axes=(r in 0..{BLOCKS}, t in 0..128), decode=(block_idx.x, thread_idx.x), block_threads=128):"
    statements = [
        f"""
        let i32 w = t / 32
        let i32 g = t % 32 / 4
        let i32 q = t % 4
        Smem(name=sa, extents=(64, 64), dtype=f16, align=1024)
        Smem(name=sb, extents=(16, 64), dtype=f16, align=1024)
        for i in 0..32:
            f16 x = load a[r * 4096 + i * 128 + t]
            f16 sa[{_moved("i * 128 + t")}] = x
        for i in 0..8:
            f16 y = load bt[r * 1024 + i * 128 + t]
            f16 sb[{_moved("i * 128 + t")}] = y
        Sync
        RegFragment(name=c0, role=c, dtype=f32)
        RegFragment(name=c1, role=c, dtype=f32)
        RegFragment(name=fa, role=a, dtype=f16)
        WgmmaFence
        """,
        "for s in 0..4:\n" + textwrap.indent(textwrap.dedent(step).strip("\n"), "    "),
        """
        WgmmaCommit
        WgmmaWait(group=0)
        """,
        store,
    ]
    body = under(grid, "\n".join(textwrap.dedent(part).strip("\n") for part in statements))
    inputs = {"a": a.astype(np.float16).ravel(), "bt": np.swapaxes(b, 1, 2).astype(np.float16).ravel()}
    out = launch.run(body, inputs, {"o": (np.float32, BLOCKS * 1024)}, blocks=BLOCKS, threads=128, level=level)["o"]
    same_bits(out.reshape(BLOCKS, 64, 16), a @ b)


def test_a_warp_gives_registers_back_and_claims_them_again(level):
    """Every warp lowers its register limit, computes, and raises it back to what it gave up: the kernel's result
    is what it is without the two statements."""
    a = launch.floats(BLOCKS * 128, np.float32)
    grid = f"Grid(axes=(r in 0..{BLOCKS}, t in 0..128), decode=(block_idx.x, thread_idx.x), block_threads=128):"
    body = under(
        grid,
        """
        SetMaxNReg(count=32, direction=dec)
        f32 x = load a[r * 128 + t]
        f32 y = x * x
        SetMaxNReg(count=40, direction=inc)
        f32 z = y + x
        f32 o[r * 128 + t] = z
        """,
    )
    out = launch.run(body, {"a": a}, {"o": (np.float32, a.size)}, blocks=BLOCKS, threads=128, level=level)["o"]
    same_bits(out, a * a + a)
