//! One test per cooperative rule: shared memory, barriers, warp shuffles, the tree reduction, vector access, atomic
//! adds, async copies and carried states. Each compares the instruction list a body in the text form gives, and
//! ptxas assembles the printed kernel at `-O0` and at its default level wherever it is installed.

mod common;

use emmy_ir::dtype::DType;
use emmy_ptx::inst::Module;
use emmy_ptx::print::print;
use emmy_ptx::{Interface, ParamKind, Target};

use DType::{F16, F32, I32, I64, U8};
use ParamKind::{Input, Output};
use common::{A_O, EXACT, Params, SM_120, listing, lowered_in};

const HALVES: Params = &[
    ("a", Input, Some(F16)),
    ("o", Output, Some(F16)),
    ("c", Input, Some(I32)),
    ("d", Output, Some(I32)),
];
const VECTORS: Params = &[
    ("a", Input, Some(F32)),
    ("o", Output, Some(F32)),
    ("h", Input, Some(F16)),
    ("g", Output, Some(F16)),
    ("q", Input, Some(I64)),
];
const BYTES: Params = &[("c", Input, Some(U8)), ("d", Output, Some(U8))];
const SUMS: Params = &[
    ("a", Input, Some(F32)),
    ("o", Output, Some(F32)),
    ("n", Output, Some(I32)),
    ("m", Output, Some(I64)),
];
const SEEDED: Params = &[
    ("a", Input, Some(F32)),
    ("o", Output, Some(F32)),
    ("s0", Input, Some(F32)),
];

/// A kernel of one block of `threads` threads a row `r`, one thread an element `t` of it, with the statements of a
/// test, written eight spaces deep, under its grid. `pool` gives each shared buffer its place where the launch
/// sizes a pool.
fn row_kernel(
    statements: &str,
    params: Params,
    threads: u32,
    pool: &[(&str, u64)],
    target: Target,
) -> emmy_ir::Result<Module> {
    let grid = format!(
        "Grid(axes=(r in 0..64, t in 0..{threads}), decode=(block_idx.x, thread_idx.x), block_threads={threads}):"
    );
    let interface = Interface {
        shared_offsets: pool
            .iter()
            .map(|(name, offset)| ((*name).into(), *offset))
            .collect(),
        ..common::interface(params, threads)
    };
    let body = format!("{grid}\n{}", common::indented(statements));
    lowered_in(&body, &interface, target, EXACT)
}

fn row_module(statements: &str, params: Params, threads: u32, pool: &[(&str, u64)]) -> Module {
    row_kernel(statements, params, threads, pool, SM_120).unwrap()
}

/// The code of such a kernel after the decode of `r` and `t`, which are `%r0` and `%r1`.
fn row_code(module: &Module) -> Vec<String> {
    let all = listing(module);
    assert_eq!(all[..2], ["mov.u32 %r0, %ctaid.x", "mov.u32 %r1, %tid.x"]);
    all[2..].to_vec()
}

/// The message such a kernel of 64 threads a block is refused with.
fn row_refusal(statements: &str, params: Params, pool: &[(&str, u64)]) -> String {
    row_kernel(statements, params, 64, pool, SM_120)
        .unwrap_err()
        .to_string()
}

/// A load of an input is `ld.global.nc`, as nvcc loads through a `const` pointer it was promised nothing else
/// writes. A load of an output, which the kernel may have stored to, is a plain load.
#[test]
fn a_load_of_a_buffer_the_kernel_never_writes_goes_past_the_coherent_cache() {
    let module = row_module(
        "
        f32 x = load a[t]
        f32 o[t] = x
        f32 y = load o[t + 1]
        f32 o[t + 2] = y
        ",
        A_O,
        64,
        &[],
    );
    assert_eq!(
        row_code(&module),
        [
            "mul.wide.s32 %rd2, %r1, 4",
            "add.s64 %rd3, %rd0, %rd2",
            "ld.global.nc.f32 %f0, [%rd3]",
            "add.s64 %rd4, %rd1, %rd2",
            "st.global.f32 [%rd4], %f0",
            "ld.global.f32 %f1, [%rd4+4]",
            "st.global.f32 [%rd4+8], %f1",
        ]
    );
}

/// A shared declaration is an array of bytes of the module, aligned to its elements or to what it asks for. Its
/// address is the symbol's own, read into a 32-bit register, and every address into it is 32-bit arithmetic: the
/// variable part scaled by a shift or a multiply, the constant in the address.
#[test]
fn a_shared_buffer_is_an_array_of_the_module_addressed_in_32_bits() {
    let module = row_module(
        "
        Smem(name=s, extents=(8, 8), dtype=f32)
        Smem(name=h, extents=(64), dtype=f16, align=16)
        f32 x = load a[r * 64 + t]
        f16 n = (f16)x
        f32 s[t] = x
        f16 h[t] = n
        Sync
        f32 y = load s[63 - t]
        f16 m = load h[t + 1]
        f32 w = (f32)m
        f32 v = y + w
        f32 o[r * 64 + t] = v
        ",
        A_O,
        64,
        &[],
    );
    let text = print(&module);
    assert!(
        text.contains(
            ".shared .align 4 .b8 k_shared_0[256];\n.shared .align 16 .b8 k_shared_1[128];\n"
        ),
        "{text}"
    );
    assert_eq!(
        row_code(&module),
        [
            "mov.u32 %r2, k_shared_0",
            "mov.u32 %r3, k_shared_1",
            "mul.wide.s32 %rd2, %r0, 256",
            "add.s64 %rd3, %rd0, %rd2",
            "mul.wide.s32 %rd4, %r1, 4",
            "add.s64 %rd5, %rd3, %rd4",
            "ld.global.nc.f32 %f0, [%rd5]",
            "cvt.rn.f16.f32 %h0, %f0",
            "shl.b32 %r4, %r1, 2",
            "add.s32 %r5, %r2, %r4",
            "st.shared.f32 [%r5], %f0",
            "shl.b32 %r6, %r1, 1",
            "add.s32 %r7, %r3, %r6",
            "st.shared.b16 [%r7], %h0",
            "bar.sync 0",
            "mul.lo.s32 %r8, %r1, -4",
            "add.s32 %r9, %r2, %r8",
            "ld.shared.f32 %f1, [%r9+252]",
            "ld.shared.b16 %h1, [%r7+2]",
            "cvt.f32.f16 %f2, %h1",
            "add.rn.f32 %f3, %f1, %f2",
            "add.s64 %rd6, %rd1, %rd2",
            "add.s64 %rd7, %rd6, %rd4",
            "st.global.f32 [%rd7], %f3",
        ]
    );
}

/// Where the interface gives a pool, the module declares one external array, aligned to the strictest buffer in
/// it, and each buffer is its byte offset from the pool's address. That address is read from the symbol and
/// never taken as zero: on `sm_120` the pool does not start where the block's shared memory does.
#[test]
fn a_pooled_shared_buffer_stands_at_its_offset_from_the_pools_own_address() {
    let module = row_module(
        "
        Smem(name=s, extents=(64), dtype=f32)
        Smem(name=big, extents=(16384), dtype=f32, align=128)
        f32 x = load a[t]
        f32 s[t] = x
        f32 big[t * 4 + 1] = x
        Sync
        f32 y = load big[t]
        f32 o[t] = y
        ",
        A_O,
        64,
        &[("s", 0), ("big", 256)],
    );
    let text = print(&module);
    assert!(
        text.contains(".extern .shared .align 128 .b8 k_shared_0[];\n"),
        "{text}"
    );
    assert_eq!(
        row_code(&module),
        [
            "mov.u32 %r2, k_shared_0",
            "mul.wide.s32 %rd2, %r1, 4",
            "add.s64 %rd3, %rd0, %rd2",
            "ld.global.nc.f32 %f0, [%rd3]",
            "shl.b32 %r3, %r1, 2",
            "add.s32 %r4, %r2, %r3",
            "st.shared.f32 [%r4], %f0",
            "shl.b32 %r5, %r1, 4",
            "add.s32 %r6, %r2, %r5",
            "st.shared.f32 [%r6+260], %f0",
            "bar.sync 0",
            "ld.shared.f32 %f1, [%r4+256]",
            "add.s64 %rd4, %rd1, %rd2",
            "st.global.f32 [%rd4], %f1",
        ]
    );
}

/// A barrier of the block, of a warp, and of a named barrier with its thread count. Each stands where every
/// thread meant to meet at it reaches it, or where that is not decided: under a condition on a loaded value or
/// on the block's index, and inside the grid's mask, whose other threads have left the kernel.
#[test]
fn a_barrier_is_one_instruction_wherever_nothing_decides_against_it() {
    let module = row_module(
        "
        Sync
        Sync(warp=true)
        if t < 32:
            Sync(barrier_id=1, count=32)
        else:
            Sync(barrier_id=2, count=32)
        f32 x = load a[t]
        if x > 0.0:
            Sync
        if block_idx.x * 64 + thread_idx.x < 4000:
            if r < 3:
                Sync
            Sync
        ",
        A_O,
        64,
        &[],
    );
    assert_eq!(
        row_code(&module),
        [
            "bar.sync 0",
            "bar.warp.sync -1",
            "setp.lt.s32 %p0, %r1, 32",
            "@!%p0 bra L1",
            "bar.sync 1, 32",
            "bra L2",
            "L1:",
            "bar.sync 2, 32",
            "L2:",
            "mul.wide.s32 %rd2, %r1, 4",
            "add.s64 %rd3, %rd0, %rd2",
            "ld.global.nc.f32 %f0, [%rd3]",
            "setp.gt.f32 %p1, %f0, 0f00000000",
            "@!%p1 bra L3",
            "bar.sync 0",
            "L3:",
            "shl.b32 %r2, %r0, 6",
            "add.s32 %r3, %r2, %r1",
            "setp.lt.s32 %p2, %r3, 4000",
            "@!%p2 bra L4",
            "setp.lt.s32 %p3, %r0, 3",
            "@!%p3 bra L5",
            "bar.sync 0",
            "L5:",
            "bar.sync 0",
            "L4:",
        ]
    );
}

/// A butterfly over four lanes of two states. At each stride every state goes through its bits to the lane that
/// stride away and back into the name the program reads it by; the program then runs in its order, a
/// temporary in a register of its own and each state stepped in place: by the instruction that computes its
/// value where that reads a state, by a move where the value is another name's.
#[test]
fn a_warp_shuffle_folds_every_state_through_its_combine_program() {
    let module = row_module(
        "
        f32 m = load a[r * 32 + t]
        let f32 l = 1.0
        WarpShuffle(state=(m, l), state_b=(m__o, l__o), length=4):
            combine_states:
                f32 big = maximum(m, m__o)
                f32 d = m - m__o
                f32 l = l + l__o
                f32 l = l * d
                f32 m = big
        f32 v = m + l
        f32 o[r * 32 + t] = v
        ",
        A_O,
        32,
        &[],
    );
    assert_eq!(
        row_code(&module),
        [
            "mul.wide.s32 %rd2, %r0, 128",
            "add.s64 %rd3, %rd0, %rd2",
            "mul.wide.s32 %rd4, %r1, 4",
            "add.s64 %rd5, %rd3, %rd4",
            "ld.global.nc.f32 %f0, [%rd5]",
            "mov.f32 %f1, 0f3F800000",
            "activemask.b32 %r2",
            "mov.b32 %r3, %f0",
            "shfl.sync.bfly.b32 %r4, %r3, 2, 31, %r2",
            "mov.b32 %f2, %r4",
            "mov.b32 %r5, %f1",
            "shfl.sync.bfly.b32 %r6, %r5, 2, 31, %r2",
            "mov.b32 %f3, %r6",
            "max.f32 %f4, %f0, %f2",
            "sub.rn.f32 %f5, %f0, %f2",
            "add.rn.f32 %f1, %f1, %f3",
            "mul.rn.f32 %f1, %f1, %f5",
            "mov.f32 %f0, %f4",
            "mov.b32 %r7, %f0",
            "shfl.sync.bfly.b32 %r8, %r7, 1, 31, %r2",
            "mov.b32 %f8, %r8",
            "mov.b32 %r9, %f1",
            "shfl.sync.bfly.b32 %r10, %r9, 1, 31, %r2",
            "mov.b32 %f9, %r10",
            "max.f32 %f10, %f0, %f8",
            "sub.rn.f32 %f11, %f0, %f8",
            "add.rn.f32 %f1, %f1, %f9",
            "mul.rn.f32 %f1, %f1, %f11",
            "mov.f32 %f0, %f10",
            "add.rn.f32 %f14, %f0, %f1",
            "add.s64 %rd6, %rd1, %rd2",
            "add.s64 %rd7, %rd6, %rd4",
            "st.global.f32 [%rd7], %f14",
        ]
    );
}

/// A prefix over groups of four lanes reads the lane a doubling stride below, clamped to the group, and steps
/// the state under a guard where such a lane stands: no branch, so the warp stays together. A half goes through
/// a 32-bit register. A broadcast gives every lane of a group of eight the state of its last lane, an integer
/// as it is.
#[test]
fn a_prefix_shuffle_combines_under_a_guard_and_a_broadcast_reads_the_last_lane() {
    let module = row_module(
        "
        f16 acc = load a[r * 32 + t]
        i32 n = load c[t]
        WarpShuffle(state=(acc), state_b=(acc__o), length=4, dtype=f16, prefix=true):
            combine_states:
                f16 acc = acc + acc__o
        WarpBroadcast(state=(acc, n), length=8)
        f16 o[r * 32 + t] = acc
        i32 d[t] = n
        ",
        HALVES,
        32,
        &[],
    );
    assert_eq!(
        row_code(&module),
        [
            "mul.wide.s32 %rd4, %r0, 64",
            "add.s64 %rd5, %rd0, %rd4",
            "mul.wide.s32 %rd6, %r1, 2",
            "add.s64 %rd7, %rd5, %rd6",
            "ld.global.nc.b16 %h0, [%rd7]",
            "mul.wide.s32 %rd8, %r1, 4",
            "add.s64 %rd9, %rd2, %rd8",
            "ld.global.nc.s32 %r2, [%rd9]",
            "activemask.b32 %r3",
            "cvt.u32.u16 %r4, %h0",
            "shfl.sync.up.b32 %r5, %r4, 1, 7168, %r3",
            "cvt.u16.u32 %h1, %r5",
            "and.b32 %r6, %r1, 3",
            "setp.ge.s32 %p0, %r6, 1",
            "@%p0 add.rn.f16 %h0, %h0, %h1",
            "cvt.u32.u16 %r7, %h0",
            "shfl.sync.up.b32 %r8, %r7, 2, 7168, %r3",
            "cvt.u16.u32 %h3, %r8",
            "setp.ge.s32 %p1, %r6, 2",
            "@%p1 add.rn.f16 %h0, %h0, %h3",
            "activemask.b32 %r9",
            "cvt.u32.u16 %r10, %h0",
            "shfl.sync.idx.b32 %r11, %r10, 7, 6175, %r9",
            "cvt.u16.u32 %h5, %r11",
            "mov.b16 %h0, %h5",
            "shfl.sync.idx.b32 %r12, %r2, 7, 6175, %r9",
            "mov.s32 %r2, %r12",
            "add.s64 %rd10, %rd1, %rd4",
            "add.s64 %rd11, %rd10, %rd6",
            "st.global.b16 [%rd11], %h0",
            "add.s64 %rd12, %rd3, %rd8",
            "st.global.s32 [%rd12], %r2",
        ]
    );
}

/// Where each thread has a slot of its own, every halving step is a condition over the threads of the low half,
/// which read their slot and the one a stride above, combine them and write their own, and a barrier of the
/// block. Every thread then reads the first slot into its state.
#[test]
fn a_tree_reduction_halves_a_slab_each_thread_has_a_slot_of() {
    let module = row_module(
        "
        Smem(name=slab, extents=(64), dtype=f32)
        f32 acc = load a[r * 64 + t]
        f32 slab[t] = acc
        Sync
        TreeHalve(bufs=(slab), state=(acc), state_b=(acc__o), length=4, tid_var=t):
            combine_states:
                f32 acc = acc + acc__o
        f32 o[r * 64 + t] = acc
        ",
        A_O,
        64,
        &[],
    );
    assert_eq!(
        row_code(&module),
        [
            "mov.u32 %r2, k_shared_0",
            "mul.wide.s32 %rd2, %r0, 256",
            "add.s64 %rd3, %rd0, %rd2",
            "mul.wide.s32 %rd4, %r1, 4",
            "add.s64 %rd5, %rd3, %rd4",
            "ld.global.nc.f32 %f0, [%rd5]",
            "shl.b32 %r3, %r1, 2",
            "add.s32 %r4, %r2, %r3",
            "st.shared.f32 [%r4], %f0",
            "bar.sync 0",
            "setp.lt.s32 %p0, %r1, 2",
            "@!%p0 bra L1",
            "ld.shared.f32 %f1, [%r4]",
            "ld.shared.f32 %f2, [%r4+8]",
            "add.rn.f32 %f1, %f1, %f2",
            "st.shared.f32 [%r4], %f1",
            "L1:",
            "bar.sync 0",
            "setp.lt.s32 %p1, %r1, 1",
            "@!%p1 bra L2",
            "ld.shared.f32 %f4, [%r4]",
            "ld.shared.f32 %f5, [%r4+4]",
            "add.rn.f32 %f4, %f4, %f5",
            "st.shared.f32 [%r4], %f4",
            "L2:",
            "bar.sync 0",
            "ld.shared.f32 %f0, [%r2]",
            "add.s64 %rd6, %rd1, %rd2",
            "add.s64 %rd7, %rd6, %rd4",
            "st.global.f32 [%rd7], %f0",
        ]
    );
}

/// Eight trees of four slots, the lanes of a tree eight slots apart: a thread's slot is addressed once for all
/// the steps, the slot a stride above is that many segments further, and every thread ends with the root of
/// its own segment. The barrier is the named one the statement gives, with its thread count.
#[test]
fn a_segmented_tree_reduction_halves_each_segment_at_a_named_barrier() {
    let module = row_module(
        "
        let i32 k = t / 8
        let i32 j = t % 8
        Smem(name=slab, extents=(64), dtype=f32)
        f32 acc = load a[r * 64 + t]
        f32 slab[t] = acc
        Sync(barrier_id=3, count=64)
        TreeHalve(bufs=(slab), state=(acc), state_b=(acc__o), length=4, tid_var=k, barrier_id=3, barrier_count=64, inner=(j, 8)):
            combine_states:
                f32 acc = maximum(acc, acc__o)
        f32 o[r * 64 + t] = acc
        ",
        A_O,
        64,
        &[],
    );
    assert_eq!(
        row_code(&module),
        [
            "shr.s32 %r2, %r1, 3",
            "and.b32 %r3, %r1, 7",
            "mov.u32 %r4, k_shared_0",
            "mul.wide.s32 %rd2, %r0, 256",
            "add.s64 %rd3, %rd0, %rd2",
            "mul.wide.s32 %rd4, %r1, 4",
            "add.s64 %rd5, %rd3, %rd4",
            "ld.global.nc.f32 %f0, [%rd5]",
            "shl.b32 %r5, %r1, 2",
            "add.s32 %r6, %r4, %r5",
            "st.shared.f32 [%r6], %f0",
            "bar.sync 3, 64",
            "shl.b32 %r7, %r2, 5",
            "add.s32 %r8, %r4, %r7",
            "shl.b32 %r9, %r3, 2",
            "add.s32 %r10, %r8, %r9",
            "setp.lt.s32 %p0, %r2, 2",
            "@!%p0 bra L1",
            "ld.shared.f32 %f1, [%r10]",
            "ld.shared.f32 %f2, [%r10+64]",
            "max.f32 %f1, %f1, %f2",
            "st.shared.f32 [%r10], %f1",
            "L1:",
            "bar.sync 3, 64",
            "setp.lt.s32 %p1, %r2, 1",
            "@!%p1 bra L2",
            "ld.shared.f32 %f4, [%r10]",
            "ld.shared.f32 %f5, [%r10+32]",
            "max.f32 %f4, %f4, %f5",
            "st.shared.f32 [%r10], %f4",
            "L2:",
            "bar.sync 3, 64",
            "add.s32 %r11, %r4, %r9",
            "ld.shared.f32 %f0, [%r11]",
            "add.s64 %rd6, %rd1, %rd2",
            "add.s64 %rd7, %rd6, %rd4",
            "st.global.f32 [%rd7], %f0",
        ]
    );
}

/// Where the lanes of a warp share a slot, no thread reads, combines and writes it in shared memory: the
/// block's first four lanes take one slot each into registers, a butterfly under the mask of exactly those
/// lanes folds them, the first lane writes the first slot, and one barrier stands ahead of the read of it.
#[test]
fn a_tree_reduction_over_slots_a_warp_shares_is_folded_in_registers() {
    let module = row_module(
        "
        let i32 lane = t % 32
        let i32 warp = t / 32
        Smem(name=slab, extents=(4), dtype=f32)
        f32 acc = load a[r * 128 + t]
        if lane == 0:
            f32 slab[warp] = acc
        Sync
        TreeHalve(bufs=(slab), state=(acc), state_b=(acc__o), length=4, tid_var=warp):
            combine_states:
                f32 acc = acc + acc__o
        f32 o[r * 128 + t] = acc
        ",
        A_O,
        128,
        &[],
    );
    assert_eq!(
        row_code(&module),
        [
            "and.b32 %r2, %r1, 31",
            "shr.s32 %r3, %r1, 5",
            "mov.u32 %r4, k_shared_0",
            "mul.wide.s32 %rd2, %r0, 512",
            "add.s64 %rd3, %rd0, %rd2",
            "mul.wide.s32 %rd4, %r1, 4",
            "add.s64 %rd5, %rd3, %rd4",
            "ld.global.nc.f32 %f0, [%rd5]",
            "setp.eq.s32 %p0, %r2, 0",
            "@!%p0 bra L1",
            "shl.b32 %r5, %r3, 2",
            "add.s32 %r6, %r4, %r5",
            "st.shared.f32 [%r6], %f0",
            "L1:",
            "bar.sync 0",
            "setp.lt.s32 %p1, %r1, 4",
            "@!%p1 bra L2",
            "shl.b32 %r7, %r1, 2",
            "add.s32 %r8, %r4, %r7",
            "ld.shared.f32 %f1, [%r8]",
            "mov.b32 %r9, %f1",
            "shfl.sync.bfly.b32 %r10, %r9, 2, 31, 15",
            "mov.b32 %f2, %r10",
            "add.rn.f32 %f1, %f1, %f2",
            "mov.b32 %r11, %f1",
            "shfl.sync.bfly.b32 %r12, %r11, 1, 31, 15",
            "mov.b32 %f4, %r12",
            "add.rn.f32 %f1, %f1, %f4",
            "setp.eq.s32 %p2, %r1, 0",
            "@!%p2 bra L3",
            "st.shared.f32 [%r4], %f1",
            "L3:",
            "L2:",
            "bar.sync 0",
            "ld.shared.f32 %f0, [%r4]",
            "add.s64 %rd6, %rd1, %rd2",
            "add.s64 %rd7, %rd6, %rd4",
            "st.global.f32 [%rd7], %f0",
        ]
    );
}

/// Two or four f32, two or four halves and two 64-bit integers are one access of that many registers. Eight
/// halves are one access of four 32-bit words, each split into its two halves by a move and joined by one.
#[test]
fn a_load_or_a_store_of_several_elements_is_one_vector_access() {
    let module = row_module(
        "
        f32 x0, x1, x2, x3 = load a[r * 256 + t * 4]
        f32 o[r * 256 + t * 4] = (x3, x2, x1, x0)
        f32 y0, y1 = load a[r * 256 + t * 4 + 2]
        f32 o[t * 4 + 2] = (y1, y0)
        f16 p0, p1 = load h[t * 8 + 2]
        f16 g[t * 8 + 2] = (p1, p0)
        f16 q0, q1, q2, q3 = load h[t * 8 + 4]
        f16 g[t * 8 + 4] = (q3, q2, q1, q0)
        f16 w0, w1, w2, w3, w4, w5, w6, w7 = load h[t * 8]
        f16 g[t * 8] = (w7, w6, w5, w4, w3, w2, w1, w0)
        i64 u0, u1 = load q[t * 2]
        ",
        VECTORS,
        64,
        &[],
    );
    assert_eq!(
        row_code(&module),
        [
            "mul.wide.s32 %rd5, %r0, 1024",
            "add.s64 %rd6, %rd0, %rd5",
            "mul.wide.s32 %rd7, %r1, 16",
            "add.s64 %rd8, %rd6, %rd7",
            "ld.global.nc.v4.f32 {%f0, %f1, %f2, %f3}, [%rd8]",
            "add.s64 %rd9, %rd1, %rd5",
            "add.s64 %rd10, %rd9, %rd7",
            "st.global.v4.f32 [%rd10], {%f3, %f2, %f1, %f0}",
            "ld.global.nc.v2.f32 {%f4, %f5}, [%rd8+8]",
            "add.s64 %rd11, %rd1, %rd7",
            "st.global.v2.f32 [%rd11+8], {%f5, %f4}",
            "add.s64 %rd12, %rd2, %rd7",
            "ld.global.nc.v2.b16 {%h0, %h1}, [%rd12+4]",
            "add.s64 %rd13, %rd3, %rd7",
            "st.global.v2.b16 [%rd13+4], {%h1, %h0}",
            "ld.global.nc.v4.b16 {%h2, %h3, %h4, %h5}, [%rd12+8]",
            "st.global.v4.b16 [%rd13+8], {%h5, %h4, %h3, %h2}",
            "ld.global.nc.v4.b32 {%r2, %r3, %r4, %r5}, [%rd12]",
            "mov.b32 {%h6, %h7}, %r2",
            "mov.b32 {%h8, %h9}, %r3",
            "mov.b32 {%h10, %h11}, %r4",
            "mov.b32 {%h12, %h13}, %r5",
            "mov.b32 %r6, {%h13, %h12}",
            "mov.b32 %r7, {%h11, %h10}",
            "mov.b32 %r8, {%h9, %h8}",
            "mov.b32 %r9, {%h7, %h6}",
            "st.global.v4.b32 [%rd13], {%r6, %r7, %r8, %r9}",
            "add.s64 %rd14, %rd4, %rd7",
            "ld.global.nc.v2.s64 {%rd15, %rd16}, [%rd14]",
        ]
    );
}

/// A byte is an integer in a 32-bit register: loaded zero-extended, with its range known to what reads it, and
/// stored from the low byte. Two or four bytes are one access of that many registers; eight are one access of
/// two words, each split into its bytes by shifts and masks and joined by masks, shifts and sums.
#[test]
fn a_run_of_bytes_is_a_vector_access_or_32_bit_words() {
    let module = row_module(
        "
        u8 b0, b1 = load c[t * 16 + 2]
        u8 d[t * 16 + 2] = (b1, b0)
        u8 e0, e1, e2, e3 = load c[t * 16 + 4]
        u8 d[t * 16 + 4] = (e3, e2, e1, e0)
        u8 k0, k1, k2, k3, k4, k5, k6, k7 = load c[t * 16 + 8]
        u8 d[t * 16 + 8] = (k7, k6, k5, k4, k3, k2, k1, k0)
        u8 one = load c[t]
        i32 third = one / 3
        u8 d[t] = third
        ",
        BYTES,
        64,
        &[],
    );
    assert_eq!(
        row_code(&module),
        [
            "mul.wide.s32 %rd2, %r1, 16",
            "add.s64 %rd3, %rd0, %rd2",
            "ld.global.nc.v2.u8 {%r2, %r3}, [%rd3+2]",
            "add.s64 %rd4, %rd1, %rd2",
            "st.global.v2.u8 [%rd4+2], {%r3, %r2}",
            "ld.global.nc.v4.u8 {%r4, %r5, %r6, %r7}, [%rd3+4]",
            "st.global.v4.u8 [%rd4+4], {%r7, %r6, %r5, %r4}",
            "ld.global.nc.v2.b32 {%r16, %r17}, [%rd3+8]",
            "and.b32 %r8, %r16, 255",
            "shr.u32 %r18, %r16, 8",
            "and.b32 %r9, %r18, 255",
            "shr.u32 %r19, %r16, 16",
            "and.b32 %r10, %r19, 255",
            "shr.u32 %r20, %r16, 24",
            "and.b32 %r11, %r20, 255",
            "and.b32 %r12, %r17, 255",
            "shr.u32 %r21, %r17, 8",
            "and.b32 %r13, %r21, 255",
            "shr.u32 %r22, %r17, 16",
            "and.b32 %r14, %r22, 255",
            "shr.u32 %r23, %r17, 24",
            "and.b32 %r15, %r23, 255",
            "and.b32 %r24, %r15, 255",
            "and.b32 %r25, %r14, 255",
            "shl.b32 %r26, %r25, 8",
            "and.b32 %r27, %r13, 255",
            "shl.b32 %r28, %r27, 16",
            "and.b32 %r29, %r12, 255",
            "shl.b32 %r30, %r29, 24",
            "or.b32 %r31, %r24, %r26",
            "or.b32 %r32, %r31, %r28",
            "or.b32 %r33, %r32, %r30",
            "and.b32 %r34, %r11, 255",
            "and.b32 %r35, %r10, 255",
            "shl.b32 %r36, %r35, 8",
            "and.b32 %r37, %r9, 255",
            "shl.b32 %r38, %r37, 16",
            "and.b32 %r39, %r8, 255",
            "shl.b32 %r40, %r39, 24",
            "or.b32 %r41, %r34, %r36",
            "or.b32 %r42, %r41, %r38",
            "or.b32 %r43, %r42, %r40",
            "st.global.v2.b32 [%rd4+8], {%r33, %r43}",
            "cvt.s64.s32 %rd5, %r1",
            "add.s64 %rd6, %rd0, %rd5",
            "ld.global.nc.u8 %r44, [%rd6]",
            "mul.hi.u32 %r45, %r44, 2863311531",
            "shr.u32 %r46, %r45, 1",
            "add.s64 %rd7, %rd1, %rd5",
            "st.global.u8 [%rd7], %r46",
        ]
    );
}

/// The zeroing of another launch's accumulator is a loop of 32-bit stores from the thread's index in the grid,
/// by the grid's thread count, both held to the word count so the index stays in 32 bits. An atomic store is one
/// atomic add at the element's type, its result unread.
#[test]
fn an_atomic_store_adds_and_a_zeroing_strides_the_grid() {
    let module = row_module(
        "
        zero o words=1000
        f32 x = load a[r * 64 + t]
        f32 o[r] += x
        i32 one = t + 1
        i32 n[0] += one
        i64 wide = (i64)one
        i64 m[0] += wide
        ",
        SUMS,
        64,
        &[],
    );
    assert_eq!(
        row_code(&module),
        [
            "mov.u32 %r2, %ntid.x",
            "mul.lo.s32 %r3, %r0, %r2",
            "add.s32 %r4, %r3, %r1",
            "min.s32 %r5, %r4, 1000",
            "mov.u32 %r6, %nctaid.x",
            "mul.wide.s32 %rd4, %r2, %r6",
            "min.s64 %rd5, %rd4, 1000",
            "cvt.u32.u64 %r7, %rd5",
            "mov.s32 %r8, 0",
            "mov.u32 %r9, %r5",
            "setp.ge.s32 %p0, %r9, 1000",
            "@%p0 bra L1",
            "L2:",
            "mul.wide.s32 %rd6, %r9, 4",
            "add.s64 %rd7, %rd1, %rd6",
            "st.global.b32 [%rd7], %r8",
            "add.s32 %r9, %r9, %r7",
            "setp.lt.s32 %p1, %r9, 1000",
            "@%p1 bra L2",
            "L1:",
            "mul.wide.s32 %rd8, %r0, 256",
            "add.s64 %rd9, %rd0, %rd8",
            "mul.wide.s32 %rd10, %r1, 4",
            "add.s64 %rd11, %rd9, %rd10",
            "ld.global.nc.f32 %f0, [%rd11]",
            "mul.wide.s32 %rd12, %r0, 4",
            "add.s64 %rd13, %rd1, %rd12",
            "atom.global.add.f32 %f1, [%rd13], %f0",
            "add.s32 %r10, %r1, 1",
            "atom.global.add.s32 %r11, [%rd2], %r10",
            "cvt.s64.s32 %rd14, %r10",
            "atom.global.add.u64 %rd15, [%rd3], %rd14",
        ]
    );
}

/// A 16-byte copy goes past the first cache and a 4- or 8-byte one through it; the shared address is 32 bits
/// from the buffer's own symbol and the global one 64. Under a condition a 16-byte copy reads its bytes where
/// the condition holds and writes zeros where it does not. A commit closes the group and a wait names how many
/// groups may still be in flight.
#[test]
fn an_async_copy_is_one_instruction_between_two_aligned_addresses() {
    let module = row_module(
        "
        Smem(name=s, extents=(1024), dtype=f32, align=16)
        CpAsyncCopy(smem=s, smem_index=(t * 4), src=a, src_index=(r * 256 + t * 4), nbytes=16)
        CpAsyncCopy(smem=s, smem_index=(t * 4 + 256), src=a, src_index=(r * 256 + t * 4), nbytes=16, valid=t < 32)
        CpAsyncCopy(smem=s, smem_index=(t * 2 + 512), src=a, src_index=(t * 2), nbytes=8)
        CpAsyncCopy(smem=s, smem_index=(t + 768), src=a, src_index=(t), nbytes=4)
        CpAsyncCommit
        CpAsyncWait(group=1)
        CpAsyncWait
        ",
        A_O,
        64,
        &[],
    );
    assert_eq!(
        row_code(&module),
        [
            "mov.u32 %r2, k_shared_0",
            "shl.b32 %r3, %r1, 4",
            "add.s32 %r4, %r2, %r3",
            "mul.wide.s32 %rd2, %r0, 1024",
            "add.s64 %rd3, %rd0, %rd2",
            "mul.wide.s32 %rd4, %r1, 16",
            "add.s64 %rd5, %rd3, %rd4",
            "cp.async.cg.shared.global [%r4], [%rd5], 16",
            "setp.lt.s32 %p0, %r1, 32",
            "selp.u32 %r5, 16, 0, %p0",
            "cp.async.cg.shared.global [%r4+1024], [%rd5], 16, %r5",
            "shl.b32 %r6, %r1, 3",
            "add.s32 %r7, %r2, %r6",
            "mul.wide.s32 %rd6, %r1, 8",
            "add.s64 %rd7, %rd0, %rd6",
            "cp.async.ca.shared.global [%r7+2048], [%rd7], 8",
            "shl.b32 %r8, %r1, 2",
            "add.s32 %r9, %r2, %r8",
            "mul.wide.s32 %rd8, %r1, 4",
            "add.s64 %rd9, %rd0, %rd8",
            "cp.async.ca.shared.global [%r9+3072], [%rd9], 4",
            "cp.async.commit_group",
            "cp.async.wait_group 1",
            "cp.async.wait_group 0",
        ]
    );
}

/// A copy with a lane offset names the tile it fills by row and column and the thread's own row and column in the
/// tile: the element is the sum of the two, folded by the slab's extents, and the tile's constant part is the
/// address's offset.
#[test]
fn an_async_copy_with_a_lane_offset_adds_the_thread_place_to_its_tile() {
    let module = row_module(
        "
        Smem(name=s, extents=(16, 68), dtype=f32, align=16)
        CpAsyncCopy(smem=s, smem_index=(8, 0), src=a, src_index=(r * 256 + t * 4), nbytes=16, lane_index=(t / 16, 4 * (t % 16)), lane_rows=8)
        ",
        A_O,
        64,
        &[],
    );
    assert_eq!(
        row_code(&module),
        [
            "mov.u32 %r2, k_shared_0",
            "shr.s32 %r3, %r1, 4",
            "mul.lo.s32 %r4, %r3, 272",
            "add.s32 %r5, %r2, %r4",
            "and.b32 %r6, %r1, 15",
            "shl.b32 %r7, %r6, 4",
            "add.s32 %r8, %r5, %r7",
            "mul.wide.s32 %rd2, %r0, 1024",
            "add.s64 %rd3, %rd0, %rd2",
            "mul.wide.s32 %rd4, %r1, 16",
            "add.s64 %rd5, %rd3, %rd4",
            "cp.async.cg.shared.global [%r8+2176], [%rd5], 16",
        ]
    );
}

/// Each cell of a carried state is the register a read sees and the register a step defines, both seeded ahead
/// of the loop that carries it: with a constant, or from the cell's element of a buffer. A read copies the
/// first, a definition moves into the second, and the end of every trip moves the second into the first, so a
/// cell a trip skipped keeps its value. A read after the loop sees the last step's.
#[test]
fn a_carried_state_is_two_registers_a_cell() {
    let module = row_module(
        "
        for k in 0..8:
            f32 x = load a[t * 8 + k]
            f32 p = pre s[0]
            f32 q = pre u[1]
            f32 w = p + x
            f32 v = q + w
            f32 s[0] <- w seed 0.25
            if k < 4:
                f32 u[1] <- v seed s0
        f32 last = pre s[0]
        f32 kept = pre u[1]
        f32 both = last + kept
        f32 o[t] = both
        ",
        SEEDED,
        64,
        &[],
    );
    assert_eq!(
        row_code(&module),
        [
            "mov.f32 %f0, 0f3E800000",
            "mov.f32 %f1, %f0",
            "ld.global.nc.f32 %f2, [%rd2+4]",
            "mov.f32 %f3, %f2",
            "mul.wide.s32 %rd3, %r1, 32",
            "add.s64 %rd4, %rd0, %rd3",
            "mov.u64 %rd5, 0",
            "mov.u32 %r2, 0",
            "L1:",
            "shl.b64 %rd6, %rd5, 2",
            "add.s64 %rd7, %rd4, %rd6",
            "ld.global.nc.f32 %f4, [%rd7]",
            "mov.f32 %f5, %f0",
            "mov.f32 %f6, %f2",
            "add.rn.f32 %f7, %f5, %f4",
            "add.rn.f32 %f8, %f6, %f7",
            "mov.f32 %f1, %f7",
            "setp.lt.s32 %p0, %r2, 4",
            "@!%p0 bra L2",
            "mov.f32 %f3, %f8",
            "L2:",
            "mov.f32 %f0, %f1",
            "mov.f32 %f2, %f3",
            "add.s64 %rd5, %rd5, 1",
            "add.s32 %r2, %r2, 1",
            "setp.lt.s32 %p1, %r2, 8",
            "@%p1 bra L1",
            "mov.f32 %f9, %f0",
            "mov.f32 %f10, %f2",
            "add.rn.f32 %f11, %f9, %f10",
            "mul.wide.s32 %rd8, %r1, 4",
            "add.s64 %rd9, %rd1, %rd8",
            "st.global.f32 [%rd9], %f11",
        ]
    );
}

/// A load marked as a staged copy defines no name: it is the same vector access into the registers the first
/// load of those names made, so a read on either side of it reads the copy staged last.
#[test]
fn a_staged_copy_is_loaded_again_into_the_registers_its_first_load_declared() {
    let module = row_module(
        "
        f32 x0, x1 = load a[t * 2]
        for k in 0..4:
            f32 s <- add(s, x0) over k
            f32 x0, x1 = load a[k * 128 + t * 2 + 128] carried=load
        f32 v = s + x1
        f32 o[t] = v
        ",
        A_O,
        64,
        &[],
    );
    assert_eq!(
        row_code(&module),
        [
            "mul.wide.s32 %rd2, %r1, 8",
            "add.s64 %rd3, %rd0, %rd2",
            "ld.global.nc.v2.f32 {%f0, %f1}, [%rd3]",
            "mov.f32 %f2, 0f00000000",
            "mov.u64 %rd4, 0",
            "mov.u32 %r2, 0",
            "L1:",
            "add.rn.f32 %f2, %f2, %f0",
            "shl.b64 %rd5, %rd4, 9",
            "add.s64 %rd6, %rd3, %rd5",
            "ld.global.nc.v2.f32 {%f0, %f1}, [%rd6+512]",
            "add.s64 %rd4, %rd4, 1",
            "add.s32 %r2, %r2, 1",
            "setp.lt.s32 %p0, %r2, 4",
            "@%p0 bra L1",
            "add.rn.f32 %f3, %f2, %f1",
            "mul.wide.s32 %rd7, %r1, 4",
            "add.s64 %rd8, %rd1, %rd7",
            "st.global.f32 [%rd8], %f3",
        ]
    );
}

/// Sixteen bytes are one access of four words, in and out.
#[test]
fn sixteen_bytes_are_four_words() {
    let module = row_module(
        "
        u8 z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12, z13, z14, z15 = load c[t * 16]
        u8 d[t * 16] = (z15, z14, z13, z12, z11, z10, z9, z8, z7, z6, z5, z4, z3, z2, z1, z0)
        ",
        BYTES,
        64,
        &[],
    );
    let code = row_code(&module);
    let accesses: Vec<&String> = code
        .iter()
        .filter(|line| line.starts_with("ld.") || line.starts_with("st."))
        .collect();
    assert_eq!(
        accesses,
        [
            "ld.global.nc.v4.b32 {%r18, %r19, %r20, %r21}, [%rd3]",
            "st.global.v4.b32 [%rd4], {%r43, %r53, %r63, %r73}",
        ]
    );
}

/// An offset computed in 64 bits and kept as its low half stays a multiple of the powers of two that divided it,
/// so eight halves at it are still one aligned access.
#[test]
fn an_offset_narrowed_to_32_bits_keeps_its_alignment() {
    let module = row_module(
        "
        i64 e = load q[t]
        let i32 off = 8 * e
        f16 w0, w1, w2, w3, w4, w5, w6, w7 = load h[off]
        f16 g[off] = (w7, w6, w5, w4, w3, w2, w1, w0)
        ",
        VECTORS,
        64,
        &[],
    );
    assert_eq!(
        row_code(&module),
        [
            "mul.wide.s32 %rd5, %r1, 8",
            "add.s64 %rd6, %rd4, %rd5",
            "ld.global.nc.s64 %rd7, [%rd6]",
            "shl.b64 %rd8, %rd7, 3",
            "cvt.u32.u64 %r2, %rd8",
            "mul.wide.s32 %rd9, %r2, 2",
            "add.s64 %rd10, %rd2, %rd9",
            "ld.global.nc.v4.b32 {%r3, %r4, %r5, %r6}, [%rd10]",
            "mov.b32 {%h0, %h1}, %r3",
            "mov.b32 {%h2, %h3}, %r4",
            "mov.b32 {%h4, %h5}, %r5",
            "mov.b32 {%h6, %h7}, %r6",
            "add.s64 %rd11, %rd3, %rd9",
            "mov.b32 %r7, {%h7, %h6}",
            "mov.b32 %r8, {%h5, %h4}",
            "mov.b32 %r9, {%h3, %h2}",
            "mov.b32 %r10, {%h1, %h0}",
            "st.global.v4.b32 [%rd11], {%r7, %r8, %r9, %r10}",
        ]
    );
}

/// The cooperative forms assemble for the older targets that have them, and an async copy is refused by name
/// on a target that has none.
#[test]
fn the_cooperative_forms_follow_the_target() {
    let statements = "
        Smem(name=s, extents=(1024), dtype=f32, align=16)
        f32 acc = load a[t]
        CpAsyncCopy(smem=s, smem_index=(t * 4), src=a, src_index=(t * 4), nbytes=16)
        CpAsyncCommit
        CpAsyncWait
        Sync
        WarpShuffle(state=(acc), state_b=(acc__o), length=32):
            combine_states:
                f32 acc = acc + acc__o
        f32 o[t] += acc
        ";
    for sm in [80, 90] {
        row_kernel(statements, A_O, 64, &[], Target { sm }).unwrap();
    }
    let message = row_kernel(statements, A_O, 64, &[], Target { sm: 75 })
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("no PTX rule for an async copy on sm_75: `CpAsyncCopy(smem=s"),
        "{message}"
    );
}

/// A barrier is refused where the conditions and loops around it decide that the threads meant to meet at it do
/// not all reach it: under a condition over the thread index that holds for some threads only, on either side of
/// it, in a loop whose trips differ between threads, and at a named barrier whose count is not the threads that
/// reach it. ptxas takes a named barrier of whole warps only.
#[test]
fn a_barrier_some_threads_do_not_reach_is_refused() {
    for (statements, expected) in [
        (
            "
        let i32 lane = t % 32
        if lane == 0:
            Sync
        f32 x = load a[t]
            ",
            "a barrier of the whole block is reached by 2 of its 64 threads: `Sync`",
        ),
        (
            "
        if t < 64:
            f32 x = load a[t]
        else:
            Sync
        f32 y = load a[t]
            ",
            "a barrier of the whole block is reached by 0 of its 64 threads",
        ),
        (
            "
        f32 x = load a[t]
        if thread_idx.x < 48:
            if t < 2:
                Sync
            ",
            "a barrier of the whole block is reached by 2 of its 48 threads",
        ),
        (
            "
        for k in t..64 step 8:
            Sync
            ",
            "a barrier stands in a loop whose trips differ between the threads of a block",
        ),
        (
            "
        Smem(name=slab, extents=(64), dtype=f32)
        f32 acc = load a[t]
        if t < 32:
            TreeHalve(bufs=(slab), state=(acc), state_b=(acc__o), length=2, tid_var=t):
                combine_states:
                    f32 acc = acc + acc__o
        f32 o[t] = acc
            ",
            "a barrier of the whole block is reached by 32 of its 64 threads: `TreeHalve(",
        ),
        (
            "
        if t < 32:
            Sync(barrier_id=1, count=64)
        f32 x = load a[t]
            ",
            "a barrier of 64 threads is reached by 32",
        ),
        (
            "
        if thread_idx.x < 48:
            Sync(barrier_id=1, count=48)
            ",
            "a barrier of 48 threads: ptxas takes whole warps",
        ),
        (
            "
        Sync(barrier_id=16, count=64)
            ",
            "a named barrier is one of 1 to 15, with a thread count",
        ),
        (
            "
        Sync(barrier_id=2)
            ",
            "a named barrier is one of 1 to 15, with a thread count",
        ),
        (
            "
        Sync(barrier_id=2, warp=true)
            ",
            "a barrier of a warp names no barrier",
        ),
    ] {
        let message = row_refusal(statements, A_O, &[]);
        assert!(message.contains(expected), "{statements}: {message}");
    }
}

/// What the cooperative rules refuse, each by the statement's name: a plain load of an element only a barrier or a
/// tensor core reads, a shared buffer the interface gives no place,
/// a state that is none or of another type, a tree reduction whose slots the thread index does not decide or
/// threads share other than by warp, an access with no vector form or no alignment proven (a shared array is
/// aligned to its elements unless its declaration says more), an atomic add or an async copy with no instruction,
/// and a cell of a carried state at an index known only where the kernel runs.
#[test]
fn a_cooperative_statement_outside_its_rule_is_an_error_that_names_it() {
    let slab = "
        Smem(name=slab, extents=(64), dtype=f32)
        Smem(name=halves, extents=(64), dtype=f16)
        Smem(name=s, extents=(1024), dtype=f32, align=16)
        let i32 pair = t / 2
        let i32 warp = t / 32
        f32 acc = load a[t]
        f16 h = (f16)acc
        i64 wide = (i64)t";
    for (statements, expected) in [
        (
            "Smem(name=odd, extents=(4), dtype=f32, align=24)",
            "a shared array has a size and a power of two as alignment",
        ),
        (
            "Smem(name=b0, extents=(8192), dtype=f32)\n        Smem(name=b1, extents=(8192), dtype=f32)",
            "70016 bytes of shared arrays pass the 49152 a module may declare, and the interface gives no pool",
        ),
        (
            "WarpShuffle(state=(h), state_b=(h__o), length=4):\n            combine_states:\n                f32 h = h + h__o",
            "h is f16, updated at f32",
        ),
        (
            "WarpShuffle(state=(acc), state_b=(acc__o), length=6):\n            combine_states:\n                f32 acc = acc + acc__o",
            "6 lanes are no power of two up to a warp",
        ),
        (
            "WarpShuffle(state=(later), state_b=(later__o), length=4):\n            combine_states:\n                f32 later = later + later__o",
            "later is updated before it is defined",
        ),
        (
            "WarpBroadcast(state=(wide), length=4)",
            "no PTX rule for a shuffle of i64",
        ),
        (
            "TreeHalve(bufs=(slab), state=(acc), state_b=(acc__o), length=32, tid_var=pair):\n            combine_states:\n                f32 acc = acc + acc__o",
            "no PTX rule for a tree reduction where threads share the slot pair other than by warp",
        ),
        (
            "TreeHalve(bufs=(slab), state=(acc), state_b=(acc__o), length=32, tid_var=r):\n            combine_states:\n                f32 acc = acc + acc__o",
            "no PTX rule for a tree reduction whose slot, r, the thread index alone does not decide",
        ),
        (
            "TreeHalve(bufs=(halves), state=(acc), state_b=(acc__o), length=32, tid_var=t):\n            combine_states:\n                f32 acc = acc + acc__o",
            "halves holds f16, read at f32",
        ),
        (
            "TreeHalve(bufs=(slab), state=(acc), state_b=(acc__o), length=2, tid_var=warp, barrier_id=1, barrier_count=64):\n            combine_states:\n                f32 acc = acc + acc__o",
            "no PTX rule for a tree reduction over slots a warp shares, past a warp of them or at a named barrier",
        ),
        (
            "f32 x0, x1, x2 = load a[t * 4]",
            "no PTX rule for a load of 3 elements of f32",
        ),
        (
            "f32 o[t * 8] = (acc, acc, acc, acc, acc, acc, acc, acc)",
            "no PTX rule for a store of 8 elements of f32",
        ),
        (
            "f32 x0, x1, x2, x3 = load a[t * 2]",
            "`ld.global.nc.v4.f32 {%f1, %f2, %f3, %f4}, [%rd6]` is a 16-byte access with no alignment proven",
        ),
        (
            "f32 o[t * 2 + 1] = (acc, acc)",
            "is a 8-byte access with no alignment proven of its address",
        ),
        (
            "f32 x0, x1 = load slab[t * 2]",
            "`ld.shared.v2.f32 {%f1, %f2}, [%r8]` is a 8-byte access with no alignment proven",
        ),
        (
            "f32 o[t * 2] = (acc, acc) swizzle=B16",
            "no PTX rule for a slab of o laid out as B16",
        ),
        (
            "f32 x0, x1 = load a[t * 2] carried=load",
            "x0 is updated before it is defined",
        ),
        ("f16 halves[t] += h", "no PTX rule for an atomic add of f16"),
        ("zero a words=64", "a is no output"),
        (
            "zero o words=2000000000",
            "no PTX rule for a zeroing of 2000000000 words",
        ),
        (
            "CpAsyncCopy(smem=s, smem_index=(t * 4), src=a, src_index=(t * 4), nbytes=12)",
            "an async copy of 12 bytes",
        ),
        (
            "CpAsyncCopy(smem=s, smem_index=(t * 2), src=a, src_index=(t * 4), nbytes=16)",
            "is a 16-byte access with no alignment proven of its address",
        ),
        (
            "CpAsyncCopy(smem=slab, smem_index=(t * 4), src=a, src_index=(t * 4), nbytes=16)",
            "is a 16-byte access with no alignment proven of its address",
        ),
        (
            "CpAsyncCopy(smem=o, smem_index=(t * 4), src=a, src_index=(t * 4), nbytes=16)",
            "an async copy goes from a parameter to a shared buffer",
        ),
        (
            "CpAsyncCopy(smem=s, smem_index=(t * 4), src=a, src_index=(t * 4), nbytes=8, valid=t < 2)",
            "no PTX rule for an async copy of 8 bytes under a condition",
        ),
        (
            "CpAsyncCopy(smem=s, smem_index=(t, 0), src=a, src_index=(t * 4), nbytes=16)",
            "an index of 2 coordinates into s is not flat",
        ),
        (
            "CpAsyncCopy(smem=s, smem_index=(t * 4), src=a, src_index=(t * 4), nbytes=16, swizzle=V70A)",
            "no PTX rule for a slab of s laid out as V70A",
        ),
        (
            "for k in 0..8:\n            for j in 0..2:\n                f32 st[j] <- acc seed 0",
            "no PTX rule for a cell of the carried state st at an index known only where the kernel runs",
        ),
        (
            "f32 p = pre st[0]",
            "st is no carried state of a loop around",
        ),
        (
            "for k in 0..8:\n            f16 st[0] <- acc seed 0",
            "acc is f32, not f16",
        ),
        (
            "for k in 0..8:\n            f32 st[0] <- acc seed halves",
            "halves holds f16",
        ),
    ] {
        let message = row_refusal(&format!("{slab}\n        {statements}"), A_O, &[]);
        assert!(message.contains(expected), "{statements}: {message}");
    }
    for (pool, expected) in [
        (("s", 0), "slab has no place in the interface's pool"),
        (
            ("slab", 8),
            "slab stands at byte 8 of the pool, off its alignment of 16",
        ),
    ] {
        let statements = "Smem(name=slab, extents=(64), dtype=f32, align=16)";
        let message = row_refusal(&format!("        {statements}"), A_O, &[pool]);
        assert!(message.contains(expected), "{message}");
    }
}

/// A selection held at a half beside a reduction at f32 combines at the f32, as the renderer's merge did: the half
/// moves through a 32-bit register at its own width and is read widened, the program runs at f32, and the update
/// of the half is narrowed back. A tree reduction whose slabs hold f32 gives the half its root narrowed.
#[test]
fn states_held_at_two_types_combine_at_the_combines_own() {
    let module = row_module(
        "
        f16 m = load h[r * 64 + t]
        let f32 l = 1.0
        WarpShuffle(state=(m, l), state_b=(m__o, l__o), length=2):
            combine_states:
                big = maximum(m, m__o)
                d = m - big
                l = l * d + l__o
                m = big
        Smem(name=ms, extents=(64), dtype=f32)
        Smem(name=ls, extents=(64), dtype=f32)
        f32 mw = (f32)m
        f32 ms[t] = mw
        f32 ls[t] = l
        Sync
        TreeHalve(bufs=(ms, ls), state=(m, l), state_b=(m__o, l__o), length=2, tid_var=t):
            combine_states:
                big = maximum(m, m__o)
                d = m - big
                l = l * d + l__o
                m = big
        f32 mo = (f32)m
        f32 v = mo + l
        f32 o[r * 64 + t] = v
        ",
        VECTORS,
        64,
        &[],
    );
    assert_eq!(
        row_code(&module),
        [
            "mul.wide.s32 %rd5, %r0, 128",
            "add.s64 %rd6, %rd2, %rd5",
            "mul.wide.s32 %rd7, %r1, 2",
            "add.s64 %rd8, %rd6, %rd7",
            "ld.global.nc.b16 %h0, [%rd8]",
            "mov.f32 %f0, 0f3F800000",
            "activemask.b32 %r2",
            "cvt.u32.u16 %r3, %h0",
            "shfl.sync.bfly.b32 %r4, %r3, 1, 31, %r2",
            "cvt.u16.u32 %h1, %r4",
            "cvt.f32.f16 %f1, %h1",
            "mov.b32 %r5, %f0",
            "shfl.sync.bfly.b32 %r6, %r5, 1, 31, %r2",
            "mov.b32 %f2, %r6",
            "cvt.f32.f16 %f3, %h0",
            "mov.f32 %f4, %f3",
            "max.f32 %f5, %f4, %f1",
            "sub.rn.f32 %f6, %f4, %f5",
            "mul.rn.f32 %f7, %f0, %f6",
            "add.rn.f32 %f8, %f7, %f2",
            "mov.f32 %f0, %f8",
            "cvt.rn.f16.f32 %h2, %f5",
            "mov.b16 %h0, %h2",
            "cvt.f32.f16 %f9, %h0",
            "mov.f32 %f10, %f9",
            "mov.u32 %r7, k_shared_0",
            "mov.u32 %r8, k_shared_1",
            "cvt.f32.f16 %f11, %h0",
            "shl.b32 %r9, %r1, 2",
            "add.s32 %r10, %r7, %r9",
            "st.shared.f32 [%r10], %f11",
            "add.s32 %r11, %r8, %r9",
            "st.shared.f32 [%r11], %f0",
            "bar.sync 0",
            "setp.lt.s32 %p0, %r1, 1",
            "@!%p0 bra L1",
            "ld.shared.f32 %f12, [%r10]",
            "ld.shared.f32 %f13, [%r10+4]",
            "ld.shared.f32 %f14, [%r11]",
            "ld.shared.f32 %f15, [%r11+4]",
            "max.f32 %f16, %f12, %f13",
            "sub.rn.f32 %f17, %f12, %f16",
            "mul.rn.f32 %f18, %f14, %f17",
            "add.rn.f32 %f14, %f18, %f15",
            "mov.f32 %f12, %f16",
            "st.shared.f32 [%r10], %f12",
            "st.shared.f32 [%r11], %f14",
            "L1:",
            "bar.sync 0",
            "ld.shared.f32 %f20, [%r7]",
            "cvt.rn.f16.f32 %h3, %f20",
            "mov.b16 %h0, %h3",
            "ld.shared.f32 %f0, [%r8]",
            "cvt.f32.f16 %f21, %h0",
            "add.rn.f32 %f22, %f21, %f0",
            "mul.wide.s32 %rd9, %r0, 256",
            "add.s64 %rd10, %rd1, %rd9",
            "mul.wide.s32 %rd11, %r1, 4",
            "add.s64 %rd12, %rd10, %rd11",
            "st.global.f32 [%rd12], %f22",
        ]
    );
}
