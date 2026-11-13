//! One test per tensor-core rule: fragments as registers, `ldmatrix` in each form with its swizzled address,
//! `mma.sync` in each form on each target that has it, and the statements that stay in registers between two
//! products. Each compares the instruction list a body in the text form gives, and ptxas assembles the printed
//! kernel at `-O0` and at its default level for the kernel's own target wherever it is installed. The toolkit's
//! ptxas no longer takes `sm_70`: a Volta form is listed for `sm_70` and assembled for `sm_80`, which has it too.

mod common;

use emmy_ir::dtype::DType;
use emmy_ptx::inst::Module;
use emmy_ptx::{Interface, Kernel, ParamKind, Target, lower};

use DType::{F16, F32, U8};
use ParamKind::{Input, Output};
use common::{EXACT, Params, listing, lowered_in};

const PARAMS: Params = &[
    ("a", Input, Some(F16)),
    ("o", Output, Some(F32)),
    ("s", Input, Some(U8)),
];

const GRID: &str =
    "Grid(axes=(r in 0..4, t in 0..32), decode=(block_idx.x, thread_idx.x), block_threads=32):";

fn interface(arch_specific: bool) -> Interface {
    Interface {
        arch_specific,
        ..common::interface(PARAMS, 32)
    }
}

/// A kernel of one warp a block with the statements of a test, written eight spaces deep, under its grid,
/// assembled for its target.
fn warp(statements: &str, sm: u32, arch_specific: bool) -> emmy_ir::Result<Module> {
    let body = format!("{GRID}\n{}", common::indented(statements));
    lowered_in(&body, &interface(arch_specific), Target { sm }, EXACT)
}

/// The same kernel lowered and not assembled: for a target the installed ptxas does not take.
fn warp_unassembled(statements: &str, sm: u32) -> Module {
    let body = common::parse(&format!("{GRID}\n{}", common::indented(statements)));
    let interface = interface(false);
    let kernel = Kernel {
        name: "k",
        body: &body,
        interface: &interface,
    };
    lower(&kernel, Target { sm }, EXACT).unwrap()
}

/// The code of such a kernel after the decode of `r` and `t`, which are `%r0` and `%r1`.
fn code(statements: &str, sm: u32) -> Vec<String> {
    listing(&warp(statements, sm, false).unwrap())[2..].to_vec()
}

/// The instructions of `module` that are one of `opcodes`, by the start of their text.
fn only(module: &Module, opcodes: &[&str]) -> Vec<String> {
    let all = listing(module);
    let wanted = |line: &&String| opcodes.iter().any(|opcode| line.starts_with(opcode));
    all.iter().filter(wanted).cloned().collect()
}

fn refusal(statements: &str, sm: u32, arch_specific: bool) -> String {
    warp(statements, sm, arch_specific).unwrap_err().to_string()
}

/// The slabs and the fragments of one m16n8k16 tile.
const TILE: &str = "
        Smem(name=sa, extents=(16, 64), dtype=f16, align=1024)
        Smem(name=sb, extents=(64, 8), dtype=f16, align=16)
        RegFragment(name=fa, role=a, dtype=f16)
        RegFragment(name=fb, role=b, dtype=f16)
        RegFragment(name=fc, role=c, dtype=f32)
";
const LOADS: &str = "
        LdmatrixLoad(frag=fa, src_buffer=sa, src_index=(0, 16), role=a, ldm=64)
        LdmatrixLoad(frag=fb, src_buffer=sb, src_index=(16, 0), role=b, ldm=8)
";

fn tile(rest: &str) -> String {
    format!("{TILE}{rest}")
}

/// A fragment is its registers. An operand's hold 32 bits of packed elements each and nothing until a statement
/// loads them; an accumulator's are floats, or packed halves, and start at zero. A `Load` or a `Write` that names
/// a fragment names its registers by a constant: a load copies one, since the register is defined again by the
/// next product, and a store moves a value into one.
#[test]
fn a_fragment_is_its_registers_and_an_accumulator_starts_at_zero() {
    let lines = code(
        "
        RegFragment(name=fa, role=a, dtype=f16)
        RegFragment(name=fc, role=c, dtype=f32)
        RegFragment(name=fh, role=c, dtype=f16)
        f32 x = load fc[1]
        f32 fc[2] = x
        u32 w = load fh[0]
        u32 fa[3] = w
        f32 o[t] = x
        ",
        80,
    );
    assert_eq!(
        lines,
        [
            "mov.f32 %f0, 0f00000000",
            "mov.f32 %f1, 0f00000000",
            "mov.f32 %f2, 0f00000000",
            "mov.f32 %f3, 0f00000000",
            "mov.u32 %r6, 0",
            "mov.u32 %r7, 0",
            "mov.f32 %f4, %f1",
            "mov.f32 %f2, %f4",
            "mov.u32 %r8, %r6",
            "mov.u32 %r5, %r8",
            "mul.wide.s32 %rd3, %r1, 4",
            "add.s64 %rd4, %rd1, %rd3",
            "st.global.f32 [%rd4], %f4",
        ]
    );
}

/// What a fragment's registers are read as must be what they hold, a register is named by a constant the
/// fragment has, a fragment is not read as one value, and a declaration of several fragments in one, which asks
/// for a loop over them, is refused.
#[test]
fn a_fragment_is_read_by_its_registers_at_their_type() {
    let declared = "
        RegFragment(name=fa, role=a, dtype=f16)
        RegFragment(name=fc, role=c, dtype=f32)
    ";
    let with = |rest: &str| {
        let body = format!("{}\n        {rest}", declared.trim_end());
        refusal(&body, 80, false)
    };
    assert!(with("f32 x = load fa[0]").contains("a register of fa holds u32"));
    assert!(with("f32 x = load fc[4]").contains("is named by a constant below 4"));
    assert!(with("f32 x = load fc[t]").contains("is named by a constant below 4"));
    assert!(with("f32 x = fc + 1.0").contains("fc is a fragment, read one register at a time"));
    assert!(
        with("MmaSyncPtx(c_frag=fa, a_frag=fa, b_frag=fa, shape=(16, 8, 16))")
            .contains("the fragment fa holds 4 registers of u32, read as 2 of u32")
    );
}

/// `ldmatrix` reads 8x8 matrices of halves from a shared slab, each lane giving the address of one row. A is four
/// matrices: lane `l` reads row `l % 16` at the column block `l / 16`. B from a slab with K along its rows is two
/// transposed matrices, lane `l` reading row `l % 16`; its row stride is the slab's own where the statement
/// names none. The lane's part of both addresses is computed once.
#[test]
fn ldmatrix_gives_each_lane_the_address_of_a_row() {
    let lines = code(&tile(LOADS), 80);
    assert_eq!(
        lines[6..],
        [
            "and.b32 %r10, %r1, 31",
            "and.b32 %r11, %r10, 15",
            "shl.b32 %r12, %r11, 7",
            "add.s32 %r13, %r2, %r12",
            "shr.s32 %r14, %r10, 4",
            "shl.b32 %r15, %r14, 4",
            "add.s32 %r16, %r13, %r15",
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r4, %r5, %r6, %r7}, [%r16+32]",
            "shl.b32 %r17, %r11, 4",
            "add.s32 %r18, %r3, %r17",
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%r8, %r9}, [%r18+256]",
        ]
    );
}

/// The other forms. A B slab with N along its rows (a transposed B) is read by two plain matrices of eight rows.
/// Two B fragments side by side in their slab are one load of four matrices into the registers of both:
/// transposed where K runs along the rows, plain where N does.
#[test]
fn ldmatrix_reads_a_transposed_b_plainly_and_two_fragments_at_once() {
    let forms = "
        RegFragment(name=fb2, role=b, dtype=f16)
        LdmatrixLoad(frag=fb, src_buffer=sa, src_index=(8, 16), role=b, ldm=64, b_trans=true)
        LdmatrixLoad(frag=fb, src_buffer=sb, src_index=(16, 0), role=b, ldm=8, pair_frag=fb2)
        LdmatrixLoad(frag=fb, src_buffer=sa, src_index=(0, 16), role=b, ldm=64, b_trans=true, pair_frag=fb2)
    ";
    for sm in [80, 89, 90, 120] {
        let module = warp(&tile(forms), sm, false).unwrap();
        let loads = only(&module, &["ldmatrix"]);
        assert_eq!(loads.len(), 3);
        assert!(loads[0].starts_with("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%r8, %r9}, "));
        assert!(loads[1].starts_with(
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r8, %r9, %r10, %r11}, "
        ));
        assert!(
            loads[2]
                .starts_with("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r8, %r9, %r10, %r11}, ")
        );
    }
    // A transposed B: lane `l` reads row `l % 8` at the column block `l / 8 % 2`.
    let lines = code(&tile(forms), 80);
    assert_eq!(
        lines[6..15],
        [
            "and.b32 %r12, %r1, 31",
            "and.b32 %r13, %r12, 7",
            "shl.b32 %r14, %r13, 7",
            "add.s32 %r15, %r2, %r14",
            "shr.s32 %r16, %r12, 3",
            "and.b32 %r17, %r16, 1",
            "shl.b32 %r18, %r17, 4",
            "add.s32 %r19, %r15, %r18",
            "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%r8, %r9}, [%r19+1056]",
        ]
    );
}

/// A swizzled slab holds its 16-byte chunks exchanged within a row group: element `e` stands at
/// `e ^ (((e >> 6) & mask) << 3)`. Where the tile's row and column are multiples of what a lane's offset spans,
/// the lane's own part is swizzled alone and shared by every load, the column's part is a constant, and the row
/// is added after. The second load reuses the first's row term.
#[test]
fn a_swizzled_address_exchanges_the_lane_s_part_once() {
    let lines = code(
        &tile(
            "
        LdmatrixLoad(frag=fa, src_buffer=sa, src_index=(0, 16), role=a, ldm=64, swizzle=B128)
        LdmatrixLoad(frag=fb, src_buffer=sa, src_index=(0, 48), role=b, ldm=64, swizzle=B128)
        ",
        ),
        80,
    );
    assert_eq!(
        lines[6..],
        [
            "and.b32 %r10, %r1, 31",
            "and.b32 %r11, %r10, 15",
            "shl.b32 %r12, %r11, 6",
            "shr.s32 %r13, %r10, 4",
            "shl.b32 %r14, %r13, 3",
            "add.s32 %r15, %r12, %r14",
            "shr.s32 %r16, %r15, 6",
            "and.b32 %r17, %r16, 7",
            "shl.b32 %r18, %r17, 3",
            "xor.b32 %r19, %r15, %r18",
            "xor.b32 %r20, %r19, 16",
            "shl.b32 %r21, %r20, 1",
            "add.s32 %r22, %r2, %r21",
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r4, %r5, %r6, %r7}, [%r22]",
            "shr.s32 %r23, %r12, 6",
            "and.b32 %r24, %r23, 7",
            "shl.b32 %r25, %r24, 3",
            "xor.b32 %r26, %r12, %r25",
            "xor.b32 %r27, %r26, 48",
            "shl.b32 %r28, %r27, 1",
            "add.s32 %r29, %r2, %r28",
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%r8, %r9}, [%r29]",
        ]
    );
}

/// A tile whose index does not split, here one flat index, is swizzled whole. A mode's row mask is its own, and
/// `<mode>@<shift>` states the shift of a slab whose row is not one swizzle row. A layout that is no exchange of
/// chunks has no rule.
#[test]
fn a_swizzle_mode_gives_its_mask_and_its_shift() {
    let swizzled = |index: &str, mode: &str| {
        let load = format!(
            "LdmatrixLoad(frag=fa, src_buffer=sa, src_index=({index}), role=a, ldm=64, swizzle={mode})"
        );
        only(
            &warp(&tile(&load), 80, false).unwrap(),
            &["shr", "and", "xor"],
        )
    };
    // The whole index, the tile's part and the lane's, is shifted and exchanged.
    let whole = swizzled("t / 32 * 8 + 16", "B64");
    assert_eq!(
        whole[whole.len() - 3..],
        [
            "shr.s32 %r20, %r19, 6",
            "and.b32 %r21, %r20, 3",
            "xor.b32 %r23, %r19, %r22"
        ]
    );
    assert!(swizzled("0, 0", "B32").contains(&"and.b32 %r17, %r16, 1".to_string()));
    assert!(swizzled("0, 0", "`B128@7`").contains(&"shr.s32 %r16, %r15, 7".to_string()));
    let volta =
        "LdmatrixLoad(frag=fa, src_buffer=sa, src_index=(0, 0), role=a, ldm=64, swizzle=V70A)";
    assert!(refusal(&tile(volta), 80, false).contains("no PTX rule for a slab laid out as V70A"));
}

/// Every form of `mma.sync` the kernels hold, on every target it is assembled for: the accumulator's registers
/// are defined again from A's, B's and their own. The module states the PTX version its instructions need.
#[test]
fn mma_sync_has_a_form_per_tile_and_type_pair() {
    let product = |rest: &str| format!("MmaSyncPtx(c_frag=fc, a_frag=fa, b_frag=fb, {rest})");
    let fragments = |dtype: &str, acc: &str| {
        format!(
            "
        Smem(name=sa, extents=(16, 64), dtype=f16, align=16)
        RegFragment(name=fa, role=a, dtype={dtype})
        RegFragment(name=fb, role=b, dtype={dtype})
        RegFragment(name=fc, role=c, dtype={acc})
        LdmatrixLoad(frag=fa, src_buffer=sa, src_index=(0, 0), role=a, ldm=64)
        LdmatrixLoad(frag=fb, src_buffer=sa, src_index=(0, 0), role=b, ldm=64)
        "
        )
    };
    let floats = "{%f0, %f1, %f2, %f3}, {%r3, %r4, %r5, %r6}, {%r7, %r8}, {%f0, %f1, %f2, %f3}";
    let halves = "{%r9, %r10}, {%r3, %r4, %r5, %r6}, {%r7, %r8}, {%r9, %r10}";
    let forms = [
        (
            "f16",
            "f32",
            "shape=(16, 8, 16)",
            "m16n8k16.row.col.f32.f16.f16.f32",
            floats,
            vec![80, 89, 90, 120],
            (7, 8),
        ),
        (
            "bf16",
            "f32",
            "shape=(16, 8, 16), ab_dtype=bf16",
            "m16n8k16.row.col.f32.bf16.bf16.f32",
            floats,
            vec![80, 89, 90, 120],
            (7, 8),
        ),
        (
            "f16",
            "f16",
            "shape=(16, 8, 16), c_dtype=f16",
            "m16n8k16.row.col.f16.f16.f16.f16",
            halves,
            vec![80, 89, 90, 120],
            (7, 8),
        ),
        (
            "f8e4m3",
            "f32",
            "shape=(16, 8, 32), ab_dtype=e4m3",
            "m16n8k32.row.col.f32.e4m3.e4m3.f32",
            floats,
            vec![89, 90, 120],
            (8, 4),
        ),
        (
            "f8e5m2",
            "f32",
            "shape=(16, 8, 32), ab_dtype=e5m2",
            "m16n8k32.row.col.f32.e5m2.e5m2.f32",
            floats,
            vec![89, 90, 120],
            (8, 4),
        ),
    ];
    for (dtype, acc, fields, form, operands, targets, version) in forms {
        for sm in targets {
            let body = format!("{}{}", fragments(dtype, acc), product(fields));
            let module = warp(&body, sm, false).unwrap();
            assert_eq!(
                only(&module, &["mma"]),
                [format!("mma.sync.aligned.{form} {operands}")]
            );
            let stated = if sm >= 100 { (8, 7) } else { version };
            assert_eq!(
                (module.version, &*module.target),
                (stated, &*format!("sm_{sm}"))
            );
        }
    }
}

/// The block-scaled product of 4-bit floats takes, after the accumulator, each side's register of scales with
/// the byte it starts at and the selector of the lanes that supply it. It is assembled for `sm_120a` alone, so
/// the interface must ask for the arch-specific instruction set, and the module's target carries the suffix.
#[test]
fn the_block_scaled_product_takes_a_scale_register_a_side() {
    let body = "
        RegFragment(name=fa, role=a, dtype=f4e2m1x2)
        RegFragment(name=fb, role=b, dtype=f4e2m1x2)
        RegFragment(name=fc, role=c, dtype=f32)
        RegFragment(name=sfa, role=sfa, dtype=u8)
        RegFragment(name=sfb, role=sfb, dtype=u8)
        u32 w = load fc[0]
    ";
    let filled = "
        RegFragment(name=fa, role=a, dtype=f4e2m1x2)
        RegFragment(name=fb, role=b, dtype=f4e2m1x2)
        RegFragment(name=fc, role=c, dtype=f32)
        RegFragment(name=fh, role=c, dtype=f16)
        RegFragment(name=sfa, role=sfa, dtype=u8)
        RegFragment(name=sfb, role=sfb, dtype=u8)
        u32 w = load fh[0]
        u32 fa[0] = w
        u32 fa[1] = w
        u32 fa[2] = w
        u32 fa[3] = w
        u32 fb[0] = w
        u32 fb[1] = w
        BlockScaleLoad(frag=sfa, src_buffer=s, src_index=(r * 64), role=a, ldm=4)
        BlockScaleLoad(frag=sfb, src_buffer=s, src_index=(r * 32), role=b, ldm=4)
        MmaSyncPtx(c_frag=fc, a_frag=fa, b_frag=fb, shape=(16, 8, 64), ab_dtype=e2m1, sfa_frag=sfa, sfb_frag=sfb)
    ";
    assert!(refusal(body, 120, true).contains("a register of fc holds f32"));
    let module = warp(filled, 120, true).unwrap();
    assert_eq!((module.version, &*module.target), ((8, 7), "sm_120a"));
    assert_eq!(
        only(&module, &["mma"]),
        [
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 \
             {%f0, %f1, %f2, %f3}, {%r2, %r3, %r4, %r5}, {%r6, %r7}, {%f0, %f1, %f2, %f3}, \
             {%r10}, {0, 0}, {%r11}, {0, 1}"
        ]
    );
    assert!(refusal(filled, 120, false).contains(
        "the block-scaled `mma.sync` m16n8k64 needs the arch-specific instruction set, sm_120a, which the interface \
         does not ask for"
    ));
    assert!(refusal(filled, 100, true).contains(
        "no PTX rule for the block-scaled `mma.sync` m16n8k64 on sm_100, which needs sm_120a or later"
    ));
}

/// A lane's register of block scales is the four scales of its row of A or of its column of B, a byte each, as
/// they stand in the slab. Only the lanes the instruction's selector names supply one: for A those whose place in
/// their group of four is 0 or 1, the second eight rows on; for B those whose place is 1. Every other lane
/// holds zero and reads nothing.
#[test]
fn block_scales_are_loaded_by_the_lanes_that_supply_them() {
    let module = warp(
        "
        RegFragment(name=sfb, role=sfb, dtype=u8)
        BlockScaleLoad(frag=sfb, src_buffer=s, src_index=(r * 32), role=b, ldm=4)
        ",
        120,
        true,
    )
    .unwrap();
    assert_eq!(
        listing(&module)[2..],
        [
            "mov.b32 %r2, 0",
            "and.b32 %r3, %r1, 31",
            "and.b32 %r4, %r3, 3",
            "setp.eq.s32 %p0, %r4, 1",
            "@!%p0 bra L1",
            "shr.s32 %r5, %r3, 2",
            "mul.wide.s32 %rd3, %r5, 4",
            "add.s64 %rd4, %rd2, %rd3",
            "mul.wide.s32 %rd5, %r0, 32",
            "add.s64 %rd6, %rd4, %rd5",
            "ld.global.nc.u8 %r6, [%rd6]",
            "ld.global.nc.u8 %r7, [%rd6+1]",
            "shl.b32 %r8, %r7, 8",
            "or.b32 %r9, %r6, %r8",
            "ld.global.nc.u8 %r10, [%rd6+2]",
            "shl.b32 %r11, %r10, 16",
            "or.b32 %r12, %r9, %r11",
            "ld.global.nc.u8 %r13, [%rd6+3]",
            "shl.b32 %r14, %r13, 24",
            "or.b32 %r2, %r12, %r14",
            "L1:",
        ]
    );
    let a = warp(
        "
        RegFragment(name=sfa, role=sfa, dtype=u8)
        BlockScaleLoad(frag=sfa, src_buffer=s, src_index=(r * 64), role=a, ldm=4)
        ",
        120,
        true,
    )
    .unwrap();
    assert_eq!(
        listing(&a)[2..10],
        [
            "mov.b32 %r2, 0",
            "and.b32 %r3, %r1, 31",
            "and.b32 %r4, %r3, 3",
            "shr.s32 %r5, %r4, 1",
            "setp.eq.s32 %p0, %r5, 0",
            "@!%p0 bra L1",
            "shr.s32 %r6, %r3, 2",
            "mul.wide.s32 %rd3, %r6, 4",
        ]
    );
}

/// Volta's product is four 8 x 8 products a warp: two registers an operand, eight floats or four packed registers
/// of accumulator, and a second form that reads B by rows. `sm_70` is listed; the same kernel is assembled for
/// `sm_80`.
#[test]
fn volta_s_product_has_a_form_that_reads_b_by_rows() {
    let body = "
        RegFragment(name=fa, role=a, dtype=f16, fragment_layout=m8n8k4)
        RegFragment(name=fb, role=b, dtype=f16, fragment_layout=m8n8k4)
        RegFragment(name=fh, role=c, dtype=f16, fragment_layout=m8n8k4)
        RegFragment(name=fc, role=c, dtype=f32, fragment_layout=m8n8k4)
        u32 w = load fh[0]
        u32 fa[0] = w
        u32 fa[1] = w
        u32 fb[0] = w
        u32 fb[1] = w
        MmaSyncPtx(c_frag=fc, a_frag=fa, b_frag=fb, shape=(8, 8, 4))
        MmaSyncPtx(c_frag=fc, a_frag=fa, b_frag=fb, shape=(8, 8, 4), b_row_major=true)
        MmaSyncPtx(c_frag=fh, a_frag=fa, b_frag=fb, shape=(8, 8, 4), c_dtype=f16)
        MmaSyncPtx(c_frag=fh, a_frag=fa, b_frag=fb, shape=(8, 8, 4), c_dtype=f16, b_row_major=true)
        FragmentPromote(dst=fc, src=fh, fragment_layout=m8n8k4)
    ";
    let floats = "{%f0, %f1, %f2, %f3, %f4, %f5, %f6, %f7}";
    let halves = "{%r6, %r7, %r8, %r9}";
    let operands = "{%r2, %r3}, {%r4, %r5}";
    let expected = [
        format!("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {floats}, {operands}, {floats}"),
        format!("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 {floats}, {operands}, {floats}"),
        format!("mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16 {halves}, {operands}, {halves}"),
        format!("mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16 {halves}, {operands}, {halves}"),
    ];
    let volta = warp_unassembled(body, 70);
    assert_eq!(only(&volta, &["mma"]), expected);
    assert_eq!((volta.version, &*volta.target), ((7, 8), "sm_70"));
    let ampere = warp(body, 80, false).unwrap();
    assert_eq!(listing(&ampere), listing(&volta));
    // The promote: a lane's halves are one row of the tile and its floats two, so each pair is taken from the
    // lane that holds it, the lower or the upper of the lane's own two registers by its place in the group.
    let promote = listing(&volta);
    let start = promote
        .iter()
        .position(|line| line.starts_with("and.b32"))
        .unwrap();
    assert_eq!(
        promote[start..start + 11],
        [
            "and.b32 %r11, %r1, 31",
            "and.b32 %r12, %r11, 2",
            "setp.ne.s32 %p0, %r12, 0",
            "and.b32 %r13, %r11, 29",
            "shfl.sync.idx.b32 %r14, %r6, %r13, 31, -1",
            "shfl.sync.idx.b32 %r15, %r7, %r13, 31, -1",
            "selp.b32 %r16, %r15, %r14, %p0",
            "mov.b32 {%h0, %h1}, %r16",
            "cvt.f32.f16 %f8, %h0",
            "add.rn.f32 %f0, %f0, %f8",
            "cvt.f32.f16 %f9, %h1",
        ]
    );
    assert_eq!(
        promote[promote.len() - 4..],
        [
            "mov.u32 %r6, 0",
            "mov.u32 %r7, 0",
            "mov.u32 %r8, 0",
            "mov.u32 %r9, 0"
        ]
    );
}

/// An accumulator of halves is folded into its accumulator of floats and zeroed: each register is split into its
/// two halves, each widened and added to its float.
#[test]
fn a_promote_adds_the_halves_into_the_floats_and_zeroes_them() {
    let lines = code(
        "
        RegFragment(name=fc, role=c, dtype=f32)
        RegFragment(name=fh, role=c, dtype=f16)
        FragmentPromote(dst=fc, src=fh)
        ",
        80,
    );
    assert_eq!(
        lines[6..],
        [
            "mov.b32 {%h0, %h1}, %r2",
            "cvt.f32.f16 %f4, %h0",
            "add.rn.f32 %f0, %f0, %f4",
            "cvt.f32.f16 %f5, %h1",
            "add.rn.f32 %f1, %f1, %f5",
            "mov.b32 {%h2, %h3}, %r3",
            "cvt.f32.f16 %f6, %h2",
            "add.rn.f32 %f2, %f2, %f6",
            "cvt.f32.f16 %f7, %h3",
            "add.rn.f32 %f3, %f3, %f7",
            "mov.u32 %r2, 0",
            "mov.u32 %r3, 0",
        ]
    );
}

/// Two accumulators side by side along K are one A fragment: a lane's floats stand where A's halves do, so each
/// register is one packed conversion, rounded to nearest, the first float the low half. The fragment is the one of
/// its name in scope, else a new one.
#[test]
fn two_accumulators_are_one_a_fragment_by_packed_conversions() {
    let lines = code(
        "
        RegFragment(name=c0, role=c, dtype=f32)
        RegFragment(name=c1, role=c, dtype=f32)
        RegFragment(name=fa, role=a, dtype=f16)
        FragmentRepack(frag=fa, srcs=(c0, c1))
        FragmentRepack(frag=fz, srcs=(c0, c1), ab_dtype=bf16)
        ",
        80,
    );
    assert_eq!(
        lines[8..],
        [
            "cvt.rn.f16x2.f32 %r2, %f1, %f0",
            "cvt.rn.f16x2.f32 %r3, %f3, %f2",
            "cvt.rn.f16x2.f32 %r4, %f5, %f4",
            "cvt.rn.f16x2.f32 %r5, %f7, %f6",
            "cvt.rn.bf16x2.f32 %r6, %f1, %f0",
            "cvt.rn.bf16x2.f32 %r7, %f3, %f2",
            "cvt.rn.bf16x2.f32 %r8, %f5, %f4",
            "cvt.rn.bf16x2.f32 %r9, %f7, %f6",
        ]
    );
}

/// One accumulator as a B fragment: a lane's packed pair holds two columns of a row, and B wants two rows of a
/// column. Lane `l`, of group `g = l / 4` and place `q = l % 4`, takes the pairs of lanes `8 * q + g / 2` and four
/// on, which hold its column's rows, and keeps the half of each that `g % 2` names. Every lane of the warp takes
/// part in the exchange.
#[test]
fn an_accumulator_is_a_b_fragment_by_an_exchange_of_packed_pairs() {
    let lines = code(
        "
        RegFragment(name=c0, role=c, dtype=f32)
        FragmentRepack(frag=fb, srcs=(c0), role=b)
        ",
        80,
    );
    assert_eq!(
        lines[4..],
        [
            "and.b32 %r4, %r1, 31",
            "shr.s32 %r5, %r4, 2",
            "and.b32 %r6, %r5, 1",
            "shl.b32 %r7, %r6, 4",
            "and.b32 %r8, %r4, 3",
            "shl.b32 %r9, %r8, 3",
            "shr.s32 %r10, %r5, 1",
            "add.s32 %r11, %r9, %r10",
            "add.s32 %r12, %r11, 4",
            "cvt.rn.f16x2.f32 %r13, %f1, %f0",
            "shfl.sync.idx.b32 %r14, %r13, %r11, 31, -1",
            "shr.u32 %r15, %r14, %r7",
            "and.b32 %r16, %r15, 65535",
            "shfl.sync.idx.b32 %r17, %r13, %r12, 31, -1",
            "shr.u32 %r18, %r17, %r7",
            "and.b32 %r19, %r18, 65535",
            "shl.b32 %r20, %r19, 16",
            "or.b32 %r2, %r16, %r20",
            "cvt.rn.f16x2.f32 %r21, %f3, %f2",
            "shfl.sync.idx.b32 %r22, %r21, %r11, 31, -1",
            "shr.u32 %r23, %r22, %r7",
            "and.b32 %r24, %r23, 65535",
            "shfl.sync.idx.b32 %r25, %r21, %r12, 31, -1",
            "shr.u32 %r26, %r25, %r7",
            "and.b32 %r27, %r26, 65535",
            "shl.b32 %r28, %r27, 16",
            "or.b32 %r3, %r24, %r28",
        ]
    );
}

/// Volta's accumulator covers 16 by 16; `part` picks four of its columns as an A fragment or four of its rows as
/// a B fragment. `sm_70` has no packed conversion, so each pair is two conversions joined by a move. Each part of
/// each role is listed for `sm_70` and assembled for `sm_80`.
#[test]
fn volta_s_accumulator_gives_a_part_of_itself_as_an_operand() {
    for (role, shuffles, conversions) in [("a", 4, 4), ("b", 8, 8)] {
        for part in 0..4 {
            let body = format!(
                "
        RegFragment(name=fc, role=c, dtype=f32, fragment_layout=m8n8k4)
        FragmentRepack(frag=fx, srcs=(fc), fragment_layout=m8n8k4, part={part}, role={role})
        "
            );
            let volta = warp_unassembled(&body, 70);
            assert_eq!(listing(&warp(&body, 80, false).unwrap()), listing(&volta));
            assert_eq!(only(&volta, &["shfl.sync.idx.b32"]).len(), shuffles);
            assert_eq!(only(&volta, &["cvt.rn.f16.f32"]).len(), conversions);
            assert!(only(&volta, &["cvt.rn.f16x2"]).is_empty());
        }
    }
    // A's columns 12 to 15: the lanes of the group that holds them for the lane's rows, the lane's own row.
    let body = "
        RegFragment(name=fc, role=c, dtype=f32, fragment_layout=m8n8k4)
        FragmentRepack(frag=fx, srcs=(fc), fragment_layout=m8n8k4, part=3, role=a)
    ";
    let lines = listing(&warp_unassembled(body, 70));
    assert_eq!(
        lines[10..],
        [
            "and.b32 %r4, %r1, 31",
            "and.b32 %r5, %r4, 16",
            "and.b32 %r6, %r4, 15",
            "shr.s32 %r7, %r6, 2",
            "and.b32 %r8, %r7, 2",
            "or.b32 %r9, %r8, 1",
            "shl.b32 %r10, %r9, 2",
            "or.b32 %r11, %r5, %r10",
            "and.b32 %r12, %r4, 1",
            "or.b32 %r13, %r11, %r12",
            "add.s32 %r14, %r13, 2",
            "and.b32 %r15, %r4, 2",
            "setp.ne.s32 %p0, %r15, 0",
            "cvt.rn.f16.f32 %h0, %f4",
            "cvt.rn.f16.f32 %h1, %f5",
            "mov.b32 %r16, {%h0, %h1}",
            "cvt.rn.f16.f32 %h2, %f6",
            "cvt.rn.f16.f32 %h3, %f7",
            "mov.b32 %r17, {%h2, %h3}",
            "shfl.sync.idx.b32 %r18, %r16, %r13, 31, -1",
            "shfl.sync.idx.b32 %r19, %r17, %r13, 31, -1",
            "selp.b32 %r2, %r19, %r18, %p0",
            "shfl.sync.idx.b32 %r20, %r16, %r14, 31, -1",
            "shfl.sync.idx.b32 %r21, %r17, %r14, 31, -1",
            "selp.b32 %r3, %r21, %r20, %p0",
        ]
    );
}

/// Each rule refuses a target that lacks its instruction, and names both.
#[test]
fn a_target_without_the_instruction_is_refused_by_name() {
    let product = "MmaSyncPtx(c_frag=fc, a_frag=fa, b_frag=fb, shape=(16, 8, 16))";
    let message = warp_refusal_unassembled(&tile(&format!("{LOADS}{product}")), 75);
    assert!(
        message
            .contains("no PTX rule for `mma.sync` m16n8k16 on sm_75, which needs sm_80 or later")
    );
    let message = warp_refusal_unassembled(&tile(LOADS), 70);
    assert!(message.contains("no PTX rule for `ldmatrix` on sm_70, which needs sm_75 or later"));
    let bytes = "
        RegFragment(name=fa, role=a, dtype=f8e4m3)
        RegFragment(name=fb, role=b, dtype=f8e4m3)
        RegFragment(name=fc, role=c, dtype=f32)
        MmaSyncPtx(c_frag=fc, a_frag=fa, b_frag=fb, shape=(16, 8, 32), ab_dtype=e4m3)
    ";
    assert!(refusal(bytes, 80, false).contains(
        "no PTX rule for `mma.sync` m16n8k32 of 8-bit floats on sm_80, which needs sm_89 or later"
    ));
    let repack = "
        RegFragment(name=c0, role=c, dtype=f32)
        RegFragment(name=c1, role=c, dtype=f32)
        FragmentRepack(frag=fa, srcs=(c0, c1))
    ";
    assert!(
        warp_refusal_unassembled(repack, 75)
            .contains("(`cvt.f16x2.f32`) on sm_75, which needs sm_80 or later")
    );
    let shapes = "MmaSyncPtx(c_frag=fc, a_frag=fa, b_frag=fb, shape=(16, 8, 16), ab_dtype=e4m3)";
    assert!(
        refusal(&tile(&format!("{LOADS}{shapes}")), 120, false)
            .contains("no PTX rule for an `mma.sync` of m16n8k16 at e4m3 into f32")
    );
    let by_rows =
        "MmaSyncPtx(c_frag=fc, a_frag=fa, b_frag=fb, shape=(16, 8, 16), b_row_major=true)";
    assert!(
        refusal(&tile(&format!("{LOADS}{by_rows}")), 120, false)
            .contains("an `mma.sync` of m16n8k16 at f16 into f32 with B by rows")
    );
}

fn warp_refusal_unassembled(statements: &str, sm: u32) -> String {
    let body = common::parse(&format!("{GRID}\n{}", common::indented(statements)));
    let interface = interface(false);
    let kernel = Kernel {
        name: "k",
        body: &body,
        interface: &interface,
    };
    lower(&kernel, Target { sm }, EXACT)
        .unwrap_err()
        .to_string()
}

/// The verifier holds what ptxas does not check. An operand fragment no statement loaded is read undefined by the
/// product, and a load under a condition defines its fragment on some paths only. Each row `ldmatrix` reads is 16
/// bytes at once, so the slab must be proven aligned to that.
#[test]
fn the_verifier_refuses_an_unloaded_fragment_and_an_unaligned_row() {
    let product = "MmaSyncPtx(c_frag=fc, a_frag=fa, b_frag=fb, shape=(16, 8, 16))";
    let message = refusal(&tile(product), 80, false);
    assert!(
        message.contains("PTX verifier") && message.contains("on a path that does not define it")
    );
    let guarded = format!(
        "
        if t < 16:
            LdmatrixLoad(frag=fa, src_buffer=sa, src_index=(0, 16), role=a, ldm=64)
        LdmatrixLoad(frag=fb, src_buffer=sb, src_index=(16, 0), role=b, ldm=8)
        {product}
        "
    );
    let message = refusal(&tile(&guarded), 80, false);
    assert!(message.contains("reads %r4 on a path that does not define it"));
    let unaligned = "
        Smem(name=sa, extents=(16, 64), dtype=f16)
        RegFragment(name=fa, role=a, dtype=f16)
        LdmatrixLoad(frag=fa, src_buffer=sa, src_index=(0, 16), role=a, ldm=64)
    ";
    let message = refusal(unaligned, 80, false);
    assert!(message.contains("is a 16-byte access with no alignment proven of its address"));
    let odd_rows = "
        Smem(name=sa, extents=(16, 20), dtype=f16, align=16)
        RegFragment(name=fa, role=a, dtype=f16)
        LdmatrixLoad(frag=fa, src_buffer=sa, src_index=(0, 0), role=a, ldm=20)
    ";
    let message = refusal(odd_rows, 80, false);
    assert!(message.contains("is a 16-byte access with no alignment proven of its address"));
}
