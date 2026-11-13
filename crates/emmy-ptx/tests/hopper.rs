//! One test per Hopper rule: `wgmma` with its descriptors and its groups, the tensor copy with the barriers its
//! bytes arrive at, and the limit on a warp's registers. Each compares the instruction list a body in the text
//! form gives, and ptxas assembles the printed kernel at `-O0` and at its default level for the kernel's own
//! target wherever it is installed.

mod common;

use emmy_ir::dtype::DType;
use emmy_ptx::inst::Module;
use emmy_ptx::print::print;
use emmy_ptx::{Interface, ParamKind, Target};

use DType::{F16, F32};
use ParamKind::{Descriptor, Input, Output};
use common::{EXACT, Params, listing, lowered_in};

const PARAMS: Params = &[
    ("a", Input, Some(F16)),
    ("o", Output, Some(F32)),
    ("x_desc", Descriptor, None),
];

/// A kernel of four warps a block, one warp group, with the statements of a test, written eight spaces deep,
/// under its grid.
fn group(statements: &str, sm: u32, arch_specific: bool) -> emmy_ir::Result<Module> {
    let grid = "Grid(axes=(r in 0..4, t in 0..128), decode=(block_idx.x, thread_idx.x), block_threads=128):";
    let interface = Interface {
        arch_specific,
        ..common::interface(PARAMS, 128)
    };
    let body = format!("{grid}\n{}", common::indented(statements));
    lowered_in(&body, &interface, Target { sm }, EXACT)
}

fn refusal(statements: &str, sm: u32, arch_specific: bool) -> String {
    group(statements, sm, arch_specific)
        .unwrap_err()
        .to_string()
}

/// The instructions of `module` that are one of `opcodes`, by the start of their text.
fn only(module: &Module, opcodes: &[&str]) -> Vec<String> {
    let all = listing(module);
    let wanted = |line: &&String| opcodes.iter().any(|opcode| line.starts_with(opcode));
    all.iter().filter(wanted).cloned().collect()
}

const SLABS: &str = "
        Smem(name=sa, extents=(64, 64), dtype=f16, align=1024)
        Smem(name=sb, extents=(16, 64), dtype=f16, align=1024)
        RegFragment(name=c0, role=c, dtype=f32)
        RegFragment(name=c1, role=c, dtype=f32)
";

/// `wgmma` reads an operand through a descriptor: the slab's address in shared memory and the two byte offsets
/// between its 8x8 matrices, in units of 16 bytes at bits 0, 16 and 32, and the slab's swizzle in the top two
/// bits. The product of a 64 x 16 tile adds into the two accumulators a warp holds side by side; a predicate says
/// whether it keeps what they held, and the transpose bit of each descriptor follows the two scales, which are
/// one. A kernel that holds it waits for PTX 8.0 and the `sm_90a` target.
#[test]
fn wgmma_reads_its_operands_through_descriptors() {
    let body = format!(
        "{SLABS}
        WgmmaFence
        WgmmaDescriptor(name=da, smem=sa, smem_index=r * 1024 + 16, swizzle=B128, lbo_bytes=16, sbo_bytes=1024)
        WgmmaDescriptor(name=db, smem=sb, smem_index=none, swizzle=B32, lbo_bytes=2048, sbo_bytes=1024)
        WgmmaMma(c_frags=(c0, c1), b_desc=db, shape=(64, 16, 16), a_desc=da, trans_b=1)
        WgmmaCommit
        WgmmaWait(group=0)
        "
    );
    let module = group(&body, 90, true).unwrap();
    assert_eq!((module.version, &*module.target), ((8, 0), "sm_90a"));
    assert_eq!(
        listing(&module)[12..],
        [
            "wgmma.fence.sync.aligned",
            "shl.b32 %r4, %r0, 11",
            "add.s32 %r5, %r2, %r4",
            "add.s32 %r6, %r5, 32",
            "shr.u32 %r7, %r6, 4",
            "and.b32 %r8, %r7, 16383",
            "cvt.u64.u32 %rd3, %r8",
            "or.b64 %rd4, %rd3, 4611686293305360384",
            "shr.u32 %r9, %r3, 4",
            "and.b32 %r10, %r9, 16383",
            "cvt.u64.u32 %rd5, %r10",
            "or.b64 %rd6, %rd5, -4611685743541092352",
            "setp.ne.b32 %p0, 1, 0",
            "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%f0, %f1, %f2, %f3, %f4, %f5, %f6, %f7}, %rd4, \
             %rd6, %p0, 1, 1, 0, 1",
            "wgmma.commit_group.sync.aligned",
            "wgmma.wait_group.sync.aligned 0",
        ]
    );
}

/// The second form takes A from the four registers of an A fragment, and one transpose bit, B's. A product whose
/// accumulator starts anew says so with a predicate that does not hold. The tile is 64 by any multiple of eight
/// columns up to 256, an accumulator fragment every eight.
#[test]
fn wgmma_takes_a_from_registers_and_any_width_of_tile() {
    let body = format!(
        "{SLABS}
        RegFragment(name=fa, role=a, dtype=bf16)
        WgmmaDescriptor(name=db, smem=sb, smem_index=64, swizzle=NONE, lbo_bytes=16, sbo_bytes=1024)
        LdmatrixLoad(frag=fa, src_buffer=sa, src_index=(0, 0), role=a, ldm=64)
        WgmmaMma(c_frags=(c0, c1), b_desc=db, shape=(64, 16, 16), ab_dtype=bf16, a_frag=fa, scale_d=0)
        "
    );
    let module = group(&body, 90, true).unwrap();
    assert_eq!(
        only(&module, &["setp", "wgmma", "or.b64"]),
        [
            "or.b64 %rd4, %rd3, 274877972480",
            "setp.ne.b32 %p0, 0, 0",
            "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%f0, %f1, %f2, %f3, %f4, %f5, %f6, %f7}, \
             {%r4, %r5, %r6, %r7}, %rd4, %p0, 1, 1, 0",
        ]
    );
    for n in [8, 64, 256] {
        let fragments: Vec<String> = (0..n / 8).map(|j| format!("w{j}")).collect();
        let declared: Vec<String> = fragments
            .iter()
            .map(|name| format!("        RegFragment(name={name}, role=c, dtype=f32)\n"))
            .collect();
        let body = format!(
            "{SLABS}{}
        WgmmaDescriptor(name=da, smem=sa, smem_index=none, swizzle=B128, lbo_bytes=16, sbo_bytes=1024)
        WgmmaDescriptor(name=db, smem=sb, smem_index=none, swizzle=B128, lbo_bytes=16, sbo_bytes=1024)
        WgmmaMma(c_frags=({}), b_desc=db, shape=(64, {n}, 16), a_desc=da)
        ",
            declared.concat(),
            fragments.join(", ")
        );
        let module = group(&body, 90, true).unwrap();
        let products = only(&module, &["wgmma"]);
        let form = format!("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.f16.f16 ");
        assert!(products[0].starts_with(&form), "{}", products[0]);
        assert_eq!(products[0].matches("%f").count(), n / 2);
    }
}

/// `wgmma` is assembled for `sm_90a` alone: a kernel whose interface does not ask for the arch-specific
/// instruction set, a later card, a type the rule has no form for, a slab laid out as no swizzle mode and a
/// descriptor of a slab with no 16-byte alignment proven are each refused. The statement checks its tile itself.
#[test]
fn wgmma_is_refused_off_sm_90a_and_out_of_its_forms() {
    let fence = "WgmmaFence";
    assert!(refusal(fence, 90, false).contains(
        "`wgmma` needs the arch-specific instruction set, sm_90a, which the interface does not ask for"
    ));
    for sm in [80, 100, 120] {
        let message = refusal(fence, sm, true);
        let expected = format!("no PTX rule for `wgmma` on sm_{sm}, which needs sm_90a");
        assert!(message.contains(&expected), "{message}");
    }
    let with = |rest: &str| {
        let descriptors = "
        WgmmaDescriptor(name=da, smem=sa, smem_index=none, swizzle=B128, lbo_bytes=16, sbo_bytes=1024)";
        refusal(
            &format!("{}{descriptors}\n        {rest}", SLABS.trim_end()),
            90,
            true,
        )
    };
    assert!(
        with("WgmmaMma(c_frags=(c0, c1), b_desc=da, shape=(64, 16, 16), a_desc=da, ab_dtype=e4m3)")
            .contains("no PTX rule for a `wgmma` at e4m3")
    );
    assert!(
        with("WgmmaDescriptor(name=dx, smem=sa, smem_index=t, swizzle=B128, lbo_bytes=16, sbo_bytes=1024)")
            .contains("a descriptor holds its address in units of 16 bytes")
    );
    assert!(
        with("WgmmaDescriptor(name=dx, smem=sa, smem_index=none, swizzle=V70A, lbo_bytes=16, sbo_bytes=1024)")
            .contains("no PTX rule for a slab laid out as V70A")
    );
}

/// A tensor copy moves one box of a tensor into a shared slab by the copy engine, through the descriptor the
/// launch passes as a pointer. One thread starts each barrier; the same thread states the bytes a phase waits for,
/// behind a fence that orders the block's earlier shared-memory accesses before the copy engine's, and issues the
/// copy, whose coordinates the instruction takes innermost first. A wait asks the barrier again until its phase
/// is over. The module states PTX 8.6, which names the copy's destination as the kernels spell it.
#[test]
fn a_tensor_copy_arrives_at_a_barrier_every_thread_waits_on() {
    let body = "
        Smem(name=tile, extents=(2, 16, 16), dtype=f32, align=128)
        Smem(name=bar, extents=(2), dtype=`unsigned long long`)
        TmaDescriptor(name=x_desc, src_buf=x, src_shape=(64, 16), box_extents=(16, 16))
        if t == 0:
            MbarrierInit(mbar=bar, count=1, slot=0)
            MbarrierInit(mbar=bar, count=1, slot=1)
        Sync
        let i32 slot = r % 2
        if t == 0:
            MbarrierArriveExpectTx(mbar=bar, bytes_=1024, slot=slot)
            TmaLoad(smem=tile, smem_index=(slot, 0, 0), desc=x_desc, coords=(r * 16, 0), mbar=bar, mbar_slot=slot)
        MbarrierWait(mbar=bar, phase=0, slot=slot)
        MbarrierArrive(mbar=bar, slot=1)
        MbarrierWait(mbar=bar, phase=r % 2, slot=1)
    ";
    for (sm, version) in [(90, (8, 6)), (120, (8, 7))] {
        let module = group(body, sm, false).unwrap();
        assert_eq!(module.version, version);
        let text = print(&module);
        assert!(text.contains("\t.param .u64 .ptr .align 64 k_param_2\n"));
        assert!(text.contains("\tld.param.u64 %rd2, [k_param_2];\n"));
        assert!(text.contains(".shared .align 8 .b8 k_shared_1[16];\n"));
        assert_eq!(
            listing(&module)[2..],
            [
                "mov.u32 %r2, k_shared_0",
                "mov.u32 %r3, k_shared_1",
                "setp.eq.s32 %p0, %r1, 0",
                "@!%p0 bra L1",
                "mbarrier.init.shared.b64 [%r3], 1",
                "mbarrier.init.shared.b64 [%r3+8], 1",
                "L1:",
                "bar.sync 0",
                "and.b32 %r4, %r0, 1",
                "@!%p0 bra L2",
                "shl.b32 %r5, %r4, 3",
                "add.s32 %r6, %r3, %r5",
                "fence.proxy.async.shared::cta",
                "mbarrier.arrive.expect_tx.shared.b64 %rd3, [%r6], 1024",
                "mov.s32 %r7, 0",
                "shl.b32 %r8, %r0, 4",
                "shl.b32 %r9, %r4, 10",
                "add.s32 %r10, %r2, %r9",
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%r10], [%rd2, {%r7, %r8}], \
                 [%r6]",
                "L2:",
                "shl.b32 %r11, %r4, 3",
                "add.s32 %r12, %r3, %r11",
                "L3:",
                "mbarrier.try_wait.parity.shared.b64 %p1, [%r12], 0",
                "@!%p1 bra L3",
                "fence.proxy.async.shared::cta",
                "mbarrier.arrive.shared.b64 %rd4, [%r3+8]",
                "L4:",
                "mbarrier.try_wait.parity.shared.b64 %p2, [%r3+8], %r4",
                "@!%p2 bra L4",
            ]
        );
    }
}

/// A box has one to five coordinates. A descriptor the launch does not pass, a card with no copy engine and a
/// destination with no 16-byte alignment proven are refused.
#[test]
fn a_tensor_copy_has_one_to_five_coordinates_and_an_aligned_destination() {
    let copy = |align: u32, descriptor: &str, coords: &str| {
        format!(
            "
        Smem(name=tile, extents=(16, 16), dtype=f32, align={align})
        Smem(name=bar, extents=(1), dtype=u64)
        TmaDescriptor(name=x_desc, src_buf=x, src_shape=(16, 16), box_extents=(16, 16))
        TmaLoad(smem=tile, smem_index=(0), desc={descriptor}, coords=({coords}), mbar=bar)
        "
        )
    };
    for rank in 1..=5 {
        let coords = vec!["r"; rank].join(", ");
        let module = group(&copy(16, "x_desc", &coords), 90, false).unwrap();
        let form =
            format!("cp.async.bulk.tensor.{rank}d.shared::cta.global.mbarrier::complete_tx::bytes");
        assert_eq!(only(&module, &[&form]).len(), 1);
    }
    let six = "r, r, r, r, r, r";
    assert!(
        refusal(&copy(16, "x_desc", six), 90, false)
            .contains("no PTX rule for a tensor copy of 6 coordinates")
    );
    assert!(
        refusal(&copy(16, "other", "r, r"), 90, false)
            .contains("other is no descriptor the launch passes")
    );
    assert!(
        refusal(&copy(16, "x_desc", "r, r"), 80, false).contains(
            "no PTX rule for `cp.async.bulk.tensor` on sm_80, which needs sm_90 or later"
        )
    );
    assert!(
        refusal(&copy(4, "x_desc", "r, r"), 90, false)
            .contains("is a 16-byte access with no alignment proven of its address")
    );
    // A barrier is started on Ampere; waiting on one and arriving behind the fence are Hopper's.
    let barrier = |rest: &str| format!("Smem(name=bar, extents=(1), dtype=u64)\n        {rest}");
    let started = group(&barrier("MbarrierInit(mbar=bar, count=1)"), 80, false).unwrap();
    assert_eq!(
        only(&started, &["mbarrier"]),
        ["mbarrier.init.shared.b64 [%r2], 1"]
    );
    assert!(
        refusal(&barrier("MbarrierInit(mbar=bar, count=1)"), 75, false)
            .contains("no PTX rule for `mbarrier.init` on sm_75, which needs sm_80 or later")
    );
    assert!(
        refusal(&barrier("MbarrierWait(mbar=bar, phase=0)"), 89, false)
            .contains("no PTX rule for `mbarrier.try_wait` on sm_89, which needs sm_90 or later")
    );
    assert!(refusal(&barrier("MbarrierArrive(mbar=bar)"), 89, false).contains(
        "no PTX rule for an arrival at an `mbarrier` behind `fence.proxy.async` on sm_89, which needs sm_90 or later"
    ));
}

/// A warp gives back the registers it will not use, or claims more, by one instruction with the count. Only the
/// arch-specific instruction set has it, on Hopper and on every later card.
#[test]
fn a_warp_s_register_limit_is_one_instruction_of_the_arch_specific_set() {
    let body = "
        SetMaxNReg(count=40, direction=dec)
        SetMaxNReg(count=232, direction=inc)
    ";
    for sm in [90, 100, 120] {
        let module = group(body, sm, true).unwrap();
        assert_eq!(
            listing(&module)[2..],
            [
                "setmaxnreg.dec.sync.aligned.u32 40",
                "setmaxnreg.inc.sync.aligned.u32 232"
            ]
        );
        assert_eq!(module.target, format!("sm_{sm}a"));
    }
    assert!(refusal(body, 90, false).contains(
        "`setmaxnreg` needs the arch-specific instruction set, sm_90a, which the interface does not ask for"
    ));
    assert!(
        refusal(body, 89, true)
            .contains("no PTX rule for `setmaxnreg` on sm_89, which needs sm_90a or later")
    );
    let sideways = "SetMaxNReg(count=40, direction=keep)";
    assert!(refusal(sideways, 90, true).contains("a register limit moved keep"));
}
