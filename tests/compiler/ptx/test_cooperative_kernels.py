"""Cooperative kernels compiled to PTX by the ``emmy-ptx`` crate and launched on a card: shared memory and barriers,
warp shuffles, the tree reduction, vector access, atomic adds, async copies and carried states. Each is compared with
NumPy bit for bit, the fold replayed in the kernel's order; only an atomic sum of arbitrary floats, whose order the
card picks, is compared within a tolerance."""

from __future__ import annotations

import numpy as np
import pytest

from tests.compiler.helpers import requires_cuda
from tests.compiler.ptx import launch
from tests.compiler.ptx.launch import floats, same_bits, under

pytestmark = [requires_cuda, pytest.mark.xdist_group("cuda"), pytest.mark.parametrize("level", launch.LEVELS, ids=["default", "O0"])]

ROWS = 37


def _row_grid(threads: int) -> str:
    """One block per row ``r``, one thread per element ``t`` of it."""
    return f"Grid(axes=(r in 0..{ROWS}, t in 0..{threads}), decode=(block_idx.x, thread_idx.x), block_threads={threads}):"


def _butterfly(values: np.ndarray, width: int) -> np.ndarray:
    """``values`` (rows of lanes) after a butterfly sum over aligned groups of ``width`` lanes."""
    lanes = np.arange(values.shape[-1])
    stride = width // 2
    while stride:
        values = values + values[..., lanes ^ stride]
        stride //= 2
    return values


def test_a_row_is_summed_by_a_warp_shuffle_and_a_tree_over_the_warps(level):
    """Each of 128 threads holds an element of its row. A butterfly sums each warp, the first lane of each warp
    stages the warp's sum, and the tree over the four staged sums, which the lanes of a warp share a slot of, is
    folded by the first warp. Every thread ends with the row's sum, equal to NumPy's taken in the same order."""
    threads = 128
    a = floats(ROWS * threads, np.float32)
    body = under(
        _row_grid(threads),
        f"""
        let i32 lane = t % 32
        let i32 warp = t / 32
        Smem(name=slab, extents=(4), dtype=f32)
        f32 acc = load a[r * {threads} + t]
        WarpShuffle(state=(acc), state_b=(acc__o), length=32):
            combine_states:
                f32 acc = acc + acc__o
        if lane == 0:
            f32 slab[warp] = acc
        Sync
        TreeHalve(bufs=(slab), state=(acc), state_b=(acc__o), length=4, tid_var=warp):
            combine_states:
                f32 acc = acc + acc__o
        f32 o[r * {threads} + t] = acc
        """,
    )
    out = launch.run(body, {"a": a}, {"o": (np.float32, a.size)}, blocks=ROWS, threads=threads, level=level)["o"]
    warps = _butterfly(a.reshape(ROWS, 4, 32), 32)[:, :, 0]
    total = _butterfly(warps, 4)[:, 0]
    same_bits(out.reshape(ROWS, threads), np.repeat(total[:, None], threads, axis=1))
    np.testing.assert_allclose(total, a.reshape(ROWS, threads).sum(axis=1, dtype=np.float64), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("inner", [False, True], ids=["flat", "segmented"])
def test_a_tree_reduction_folds_a_slab_each_thread_has_a_slot_of(level, inner):
    """64 threads stage a value each and halve the slab in shared memory, a barrier after each step. Flat, the slab
    is one tree and every thread ends with the row's sum; segmented, eight threads apart share a tree of eight
    slots, so each thread ends with the sum of its own segment, and the steps meet at a named barrier."""
    threads = 64
    a = floats(ROWS * threads, np.float32)
    if inner:
        tree = "TreeHalve(bufs=(slab), state=(acc), state_b=(acc__o), length=8, tid_var=k, barrier_id=1, barrier_count=64, inner=(j, 8)):"
    else:
        tree = "TreeHalve(bufs=(slab), state=(acc), state_b=(acc__o), length=64, tid_var=t):"
    body = under(
        _row_grid(threads),
        f"""
        let i32 k = t / 8
        let i32 j = t % 8
        Smem(name=slab, extents=(64), dtype=f32)
        f32 acc = load a[r * {threads} + t]
        f32 slab[t] = acc
        Sync
        {tree}
            combine_states:
                f32 acc = acc + acc__o
        f32 o[r * {threads} + t] = acc
        """,
    )
    out = launch.run(body, {"a": a}, {"o": (np.float32, a.size)}, blocks=ROWS, threads=threads, level=level)["o"]
    # The slab as [slot of the tree, lanes that share nothing]: each halving folds the upper half into the lower.
    slab = a.reshape(ROWS, 8, 8) if inner else a.reshape(ROWS, threads, 1)
    while slab.shape[1] > 1:
        half = slab.shape[1] // 2
        slab = slab[:, :half] + slab[:, half:]
    expected = np.repeat(slab, 8 if inner else threads, axis=1).reshape(ROWS, threads)
    same_bits(out.reshape(ROWS, threads), np.ascontiguousarray(expected))


def test_a_combine_of_two_states_runs_its_program_in_order(level):
    """The combine online softmax runs, a running maximum and a sum rescaled to it, with a polynomial standing in
    for the exponential: the sum's update reads the maximum both states had before the maximum's own."""
    threads = 32
    a = floats(ROWS * threads, np.float32)
    body = under(
        _row_grid(threads),
        f"""
        f32 m = load a[r * {threads} + t]
        let f32 l = 1.0
        WarpShuffle(state=(m, l), state_b=(m__o, l__o), length=32):
            combine_states:
                f32 big = maximum(m, m__o)
                f32 da = m - big
                f32 db = m__o - big
                f32 ha = da * 0.5
                f32 hb = db * 0.5
                f32 ia = ha + 1.0
                f32 ib = hb + 1.0
                f32 ja = da * ia
                f32 jb = db * ib
                f32 fa = ja + 1.0
                f32 fb = jb + 1.0
                f32 sa = l * fa
                f32 sb = l__o * fb
                f32 l = sa + sb
                f32 m = big
        f32 top[r * {threads} + t] = m
        f32 mass[r * {threads} + t] = l
        """,
    )
    size = (np.float32, a.size)
    out = launch.run(body, {"a": a}, {"top": size, "mass": size}, blocks=ROWS, threads=threads, level=level)
    m, mass = a.reshape(ROWS, threads), np.ones((ROWS, threads), np.float32)
    lanes, half, one = np.arange(threads), np.float32(0.5), np.float32(1)
    for stride in (16, 8, 4, 2, 1):
        other_m, other_mass = m[:, lanes ^ stride], mass[:, lanes ^ stride]
        big = np.maximum(m, other_m)
        da, db = m - big, other_m - big
        fa, fb = da * (da * half + one) + one, db * (db * half + one) + one
        m, mass = big, mass * fa + other_mass * fb
    same_bits(out["top"].reshape(ROWS, threads), np.ascontiguousarray(m))
    same_bits(out["mass"].reshape(ROWS, threads), np.ascontiguousarray(mass))


def test_a_prefix_shuffle_scans_each_group_and_a_broadcast_spreads_its_last_lane(level):
    """Groups of eight lanes of half-precision values: a prefix shuffle leaves each lane the sum of the lanes up to
    itself, in the scan's own order, and a broadcast then gives every lane its group's last."""
    threads = 32
    a = floats(ROWS * threads, np.float16)
    body = under(
        _row_grid(threads),
        f"""
        f16 acc = load a[r * {threads} + t]
        WarpShuffle(state=(acc), state_b=(acc__o), length=8, dtype=f16, prefix=true):
            combine_states:
                f16 acc = acc + acc__o
        f16 scan[r * {threads} + t] = acc
        WarpBroadcast(state=(acc), length=8)
        f16 last[r * {threads} + t] = acc
        """,
    )
    size = (np.float16, a.size)
    out = launch.run(body, {"a": a}, {"scan": size, "last": size}, blocks=ROWS, threads=threads, level=level)
    groups = a.reshape(-1, 8)
    for stride in (1, 2, 4):
        below = np.roll(groups, stride, axis=1)
        groups = np.where(np.arange(8) >= stride, groups + below, groups)
    same_bits(out["scan"].reshape(-1, 8), np.ascontiguousarray(groups))
    same_bits(out["last"].reshape(-1, 8), np.repeat(groups[:, 7:], 8, axis=1))


@pytest.mark.parametrize("pooled", [False, True], ids=["arrays", "pool"])
def test_a_row_staged_in_shared_memory_is_read_back_across_a_barrier(level, pooled):
    """Every thread stages its element and its double, the block meets at a barrier, and each thread reads what
    the thread opposite staged. The two buffers are arrays of the module, or places in one pool the launch sizes:
    there the second buffer starts past 64 KiB of the first."""
    threads = 64
    a = floats(ROWS * threads, np.float32)
    first = 16384 if pooled else threads
    shared = ({"wide": 0, "twice": first * 4}, first * 4 + threads * 4) if pooled else ({}, 0)
    body = under(
        _row_grid(threads),
        f"""
        Smem(name=wide, extents=({first}), dtype=f32)
        Smem(name=twice, extents=({threads}), dtype=f32)
        f32 x = load a[r * {threads} + t]
        f32 d = x + x
        f32 wide[t * {first // threads}] = x
        f32 twice[t] = d
        Sync
        f32 y = load wide[{first - first // threads} - t * {first // threads}]
        f32 z = load twice[{threads - 1} - t]
        f32 o[r * {threads} + t] = y
        f32 p[r * {threads} + t] = z
        """,
    )
    size = (np.float32, a.size)
    out = launch.run(body, {"a": a}, {"o": size, "p": size}, blocks=ROWS, threads=threads, level=level, shared=shared)
    rows = a.reshape(ROWS, threads)
    same_bits(out["o"].reshape(ROWS, threads), np.ascontiguousarray(rows[:, ::-1]))
    same_bits(out["p"].reshape(ROWS, threads), np.ascontiguousarray((rows + rows)[:, ::-1]))


def test_vector_loads_and_stores_move_several_elements_in_one_access(level):
    """Runs of f32, f16 and bytes loaded by one access and stored reversed by one access: two and four elements
    directly, eight halves and eight or sixteen bytes through 32-bit words."""
    threads = 64
    a = floats(threads * 16, np.float32)
    h = floats(threads * 16, np.float16, 1)
    c = np.random.default_rng(2).integers(0, 256, threads * 32).astype(np.uint8)
    grid = f"Grid(axes=(t in 0..{threads}), decode=(thread_idx.x), block_threads={threads}):"

    def run(dtype: str, source: str, target: str, width: int, offset: int) -> str:
        names = [f"{source}{offset}_{k}" for k in range(width)]
        load = f"{dtype} {', '.join(names)} = load {source}[t * 16 + {offset}]\n"
        return load + f"{dtype} {target}[t * 16 + {offset}] = ({', '.join(reversed(names))})\n"

    statements = (
        run("f32", "a", "o", 4, 0)
        + run("f32", "a", "o", 2, 4)
        + run("f32", "a", "o", 2, 6)
        + run("f32", "a", "o", 4, 8)
        + run("f32", "a", "o", 4, 12)
        + run("f16", "h", "g", 8, 0)
        + run("f16", "h", "g", 4, 8)
        + run("f16", "h", "g", 2, 12)
        + run("f16", "h", "g", 2, 14)
        + run("u8", "c", "d", 16, 0).replace("* 16", "* 32")
        + run("u8", "c", "d", 8, 16).replace("* 16", "* 32")
        + run("u8", "c", "d", 4, 24).replace("* 16", "* 32")
        + run("u8", "c", "d", 2, 28).replace("* 16", "* 32")
        + run("u8", "c", "d", 2, 30).replace("* 16", "* 32")
    )
    outputs = {"o": (np.float32, a.size), "g": (np.float16, h.size), "d": (np.uint8, c.size)}
    out = launch.run(under(grid, statements), {"a": a, "h": h, "c": c}, outputs, blocks=1, threads=threads, level=level)

    def reversed_runs(data: np.ndarray, period: int, widths: tuple[int, ...]) -> np.ndarray:
        rows, at, parts = data.reshape(-1, period), 0, []
        for width in widths:
            parts.append(rows[:, at : at + width][:, ::-1])
            at += width
        return np.concatenate(parts, axis=1).reshape(-1)

    same_bits(out["o"], reversed_runs(a, 16, (4, 2, 2, 4, 4)))
    same_bits(out["g"], reversed_runs(h, 16, (8, 4, 2, 2)))
    same_bits(out["d"], reversed_runs(c, 32, (16, 8, 4, 2, 2)))


def test_atomic_adds_from_every_thread_land_in_one_cell(level):
    """256 threads of each block add into their block's cell and into one cell of the whole grid. Whole numbers sum
    exactly in any order, as floats and as integers; arbitrary positive floats, sixteen to a cell, sum to the
    float64 total within the rounding of fifteen additions."""
    blocks, threads = 23, 256
    whole = np.random.default_rng(0).integers(-100, 100, blocks * threads).astype(np.float32)
    part = (np.random.default_rng(1).random(blocks * threads) + 0.5).astype(np.float32)
    body = under(
        f"Grid(axes=(r in 0..{blocks}, t in 0..{threads}), decode=(block_idx.x, thread_idx.x), block_threads={threads}):",
        f"""
        f32 x = load whole[r * {threads} + t]
        f32 y = load part[r * {threads} + t]
        i32 n = t + r
        f32 rows[r] += x
        f32 cells[r * 16 + t / 16] += y
        i32 count[0] += n
        """,
    )
    outputs = {"rows": (np.float32, blocks), "cells": (np.float32, blocks * 16), "count": (np.int32, 1)}
    out = launch.run(body, {"whole": whole, "part": part}, outputs, blocks=blocks, threads=threads, level=level)
    same_bits(out["rows"], whole.reshape(blocks, threads).sum(axis=1, dtype=np.float64).astype(np.float32))
    np.testing.assert_allclose(out["cells"], part.reshape(blocks * 16, 16).sum(axis=1, dtype=np.float64), rtol=1e-6, atol=0)
    assert out["count"][0] == sum(t + r for r in range(blocks) for t in range(threads))


def test_one_kernel_zeroes_the_words_of_another_kernels_accumulator(level):
    """A first kernel fills a buffer with ones; a second zeroes its first 3,000 words, every thread of the grid a
    stride of them, and leaves the rest."""
    blocks, threads, words = 5, 256, 3000
    a = np.ones(blocks * threads, np.float32)
    grid = f"Grid(axes=(i in 0..{blocks * threads}), decode=(block_idx.x * {threads} + thread_idx.x), block_threads={threads}):"
    fill = under(grid, "f32 x = load a[i]\nf32 acc[i * 4] = (x, x, x, x)\nf32 o[i] = x")
    zero = under(grid, f"zero acc words={words}\nf32 x = load a[i]\nf32 y = x + x\nf32 o[i] = y")
    outputs = {"acc": (np.float32, 4 * a.size), "o": (np.float32, a.size)}
    out = launch.run([fill, zero], {"a": a}, outputs, blocks=blocks, threads=threads, level=level)
    same_bits(out["acc"], np.concatenate([np.zeros(words, np.float32), np.ones(4 * a.size - words, np.float32)]))
    same_bits(out["o"], a + a)


@pytest.mark.parametrize("pooled", [False, True], ids=["array", "pool"])
def test_async_copies_stage_a_row_in_shared_memory(level, pooled):
    """Every thread issues copies of its 16, 8 and 4 bytes of the row into shared memory, and a 16-byte copy that
    reads its bytes in the lower half of the block and writes zeros in the upper. The group is committed and waited
    for, the block meets at a barrier, and each thread reads what the thread opposite copied. In the pool the
    slab starts past another buffer, so its place in the block's shared memory is not zero."""
    threads = 64
    a = floats(ROWS * threads * 4, np.float32)
    shared = ({"pad": 0, "slab": 65536}, 65536 + threads * 16 * 4) if pooled else ({}, 0)
    body = under(
        _row_grid(threads),
        f"""
        Smem(name=pad, extents=({16384 if pooled else 4}), dtype=f32)
        Smem(name=slab, extents=({threads * 16}), dtype=f32, align=16)
        CpAsyncCopy(smem=slab, smem_index=(t * 16), src=a, src_index=(r * {threads * 4} + t * 4), nbytes=16)
        CpAsyncCopy(smem=slab, smem_index=(t * 16 + 4), src=a, src_index=(r * {threads * 4} + t * 4), nbytes=16, valid=t < 32)
        CpAsyncCopy(smem=slab, smem_index=(t * 16 + 8), src=a, src_index=(r * {threads * 4} + t * 4 + 2), nbytes=8)
        CpAsyncCopy(smem=slab, smem_index=(t * 16 + 10), src=a, src_index=(r * {threads * 4} + t * 4 + 1), nbytes=4)
        CpAsyncCommit
        CpAsyncWait(group=0)
        Sync
        f32 s0, s1, s2, s3 = load slab[{(threads - 1) * 16} - t * 16]
        f32 z0, z1, z2, z3 = load slab[{(threads - 1) * 16 + 4} - t * 16]
        f32 e0, e1 = load slab[{(threads - 1) * 16 + 8} - t * 16]
        f32 one = load slab[{(threads - 1) * 16 + 10} - t * 16]
        f32 o[r * {threads * 16} + t * 16] = (s0, s1, s2, s3)
        f32 o[r * {threads * 16} + t * 16 + 4] = (z0, z1, z2, z3)
        f32 o[r * {threads * 16} + t * 16 + 8] = (e0, e1)
        f32 o[r * {threads * 16} + t * 16 + 10] = one
        """,
    )
    outputs = {"o": (np.float32, ROWS * threads * 16)}
    out = launch.run(body, {"a": a}, outputs, blocks=ROWS, threads=threads, level=level, shared=shared)["o"]
    staged = a.reshape(ROWS, threads, 4)
    expected = np.zeros((ROWS, threads, 16), np.float32)
    expected[:, :, 0:4] = staged
    expected[:, :32, 4:8] = staged[:, :32]
    expected[:, :, 8:10] = staged[:, :, 2:4]
    expected[:, :, 10] = staged[:, :, 1]
    same_bits(out.reshape(ROWS, threads, 16), np.ascontiguousarray(expected[:, ::-1]))


def test_a_carried_state_steps_through_its_loop(level):
    """Two carried states over eight steps: one seeded with a constant and defined every step, one seeded from a
    buffer and defined in the first four. A read sees what the step before left, in the loop and after it."""
    n, steps = 1000, 8
    a, seeds = floats(n * steps, np.float32), floats(2, np.float32, 1)
    grid = f"Grid(axes=(i in 0..{n}), decode=(block_idx.x * 256 + thread_idx.x), block_threads=256):\n"
    grid += f"    if block_idx.x * 256 + thread_idx.x < {n}:"
    body = under(
        grid,
        f"""
        for k in 0..{steps}:
            f32 x = load a[i * {steps} + k]
            f32 p = pre s[0]
            f32 q = pre u[1]
            f32 h = p * 0.5
            f32 w = h + x
            f32 v = q + w
            f32 s[0] <- w seed 0.25
            if k < 4:
                f32 u[1] <- v seed seeds
        f32 last = pre s[0]
        f32 kept = pre u[1]
        f32 o[i] = last
        f32 g[i] = kept
        """,
        depth=2,
    )
    size = (np.float32, 1024)
    out = launch.run(body, {"a": a, "seeds": seeds}, {"o": size, "g": size}, blocks=4, level=level)
    rows = a.reshape(n, steps)
    s, u = np.full(n, 0.25, np.float32), np.full(n, seeds[1], np.float32)
    for k in range(steps):
        w = s * np.float32(0.5) + rows[:, k]
        s, u = w, (u + w if k < 4 else u)
    same_bits(out["o"][:n], s)
    same_bits(out["g"][:n], u)
    assert not out["o"][n:].any() and not out["g"][n:].any()


def test_a_staged_copy_is_loaded_again_into_its_registers(level):
    """Four elements loaded ahead of a loop and loaded again at the end of each trip: every trip reads the copy
    the trip before staged."""
    threads, trips = 64, 5
    a = floats(threads * 4 * (trips + 1), np.float32)
    body = under(
        f"Grid(axes=(t in 0..{threads}), decode=(thread_idx.x), block_threads={threads}):",
        f"""
        f32 x0, x1, x2, x3 = load a[t * 4]
        for k in 0..{trips}:
            f32 first <- add(first, x0) over k
            f32 third <- add(third, x2) over k
            f32 x0, x1, x2, x3 = load a[k * {threads * 4} + t * 4 + {threads * 4}] carried=load
        f32 o[t * 4] = (first, third, x1, x3)
        """,
    )
    out = launch.run(body, {"a": a}, {"o": (np.float32, threads * 4)}, blocks=1, threads=threads, level=level)["o"]
    staged = a.reshape(trips + 1, threads, 4)
    first, third = np.zeros(threads, np.float32), np.zeros(threads, np.float32)
    for k in range(trips):
        first, third = first + staged[k, :, 0], third + staged[k, :, 2]
    same_bits(out.reshape(threads, 4), np.stack([first, third, staged[trips, :, 1], staged[trips, :, 3]], axis=1))
