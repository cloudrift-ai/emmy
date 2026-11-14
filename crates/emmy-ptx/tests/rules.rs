//! One test per rule, each from a body in the text form: the instruction list the rule writes, and that ptxas
//! assembles the printed kernel at `-O0` and at its default level wherever it is installed.

mod common;

use std::collections::HashMap;

use emmy_ir::dtype::DType;
use emmy_ptx::inst::{
    Bank, Base, Cache, Cmp, Inst, Item, Label, Module, Opcode, Operand, ParamDecl, Reg, Shared,
    Space, Ty,
};
use emmy_ptx::print::print;
use emmy_ptx::verify::verify;
use emmy_ptx::{Precision, Target};

use DType::{BF16, Bool, F16, F32, I16, I32, U16, U32};
use common::{A_O, EXACT, FUSED, Params, SM_120, assemble, code, lowered, refusal};
use emmy_ptx::ParamKind::{Input, Int, Output};

/// One thread per element of 4,096, with no tail to guard.
const GRID: &str =
    "Grid(axes=(i in 0..4096), decode=(block_idx.x * 256 + thread_idx.x), block_threads=256):";

/// What `GRID` decodes `i` with: `%r3`.
const DECODE: [&str; 4] = [
    "mov.u32 %r0, %ctaid.x",
    "shl.b32 %r1, %r0, 8",
    "mov.u32 %r2, %tid.x",
    "add.s32 %r3, %r1, %r2",
];

/// The code of the statements `body` holds under `GRID`, after the decode.
fn under_grid(body: &str, params: Params, precision: Precision) -> Vec<String> {
    let body = format!("{GRID}\n{}", common::indented(body));
    let all = code(&body, params, precision);
    assert_eq!(all[..DECODE.len()], DECODE);
    all[DECODE.len()..].to_vec()
}

// --- the grid --------------------------------------------------------------------------------------------

/// A grid binds each axis to its decode over the built-ins: a division and a remainder by a power of two are a
/// shift and a mask, and the guard over the launch's tail is a branch past the body.
#[test]
fn a_grid_decodes_its_axes_and_guards_its_tail() {
    let got = code(
        "
Grid(axes=(r in 0..100, c in 0..4), decode=((block_idx.x * 256 + thread_idx.x) / 4, thread_idx.x % 4), block_threads=256):
    if block_idx.x * 256 + thread_idx.x < 400:
        f32 x = load a[r * 4 + c]
        f32 o[r * 4 + c] = x
",
        A_O,
        EXACT,
    );
    assert_eq!(
        got,
        [
            "mov.u32 %r0, %ctaid.x",
            "shl.b32 %r1, %r0, 8",
            "mov.u32 %r2, %tid.x",
            "add.s32 %r3, %r1, %r2",
            "shr.s32 %r4, %r3, 2",
            "and.b32 %r5, %r2, 3",
            "setp.lt.s32 %p0, %r3, 400",
            "@!%p0 bra L1",
            "mul.wide.s32 %rd2, %r4, 16",
            "add.s64 %rd3, %rd0, %rd2",
            "mul.wide.s32 %rd4, %r5, 4",
            "add.s64 %rd5, %rd3, %rd4",
            "ld.global.nc.f32 %f0, [%rd5]",
            "add.s64 %rd6, %rd1, %rd2",
            "add.s64 %rd7, %rd6, %rd4",
            "st.global.f32 [%rd7], %f0",
            "L1:",
        ]
    );
}

/// The directives of a module: the PTX version the target needs, the thread bound, and one pointer parameter per
/// buffer with the alignment of its elements, a 32-bit one per symbolic size. A launch of one wave says that one
/// block on each multiprocessor is enough.
#[test]
fn a_module_states_its_target_its_thread_bound_and_its_parameters() {
    let body = format!("{GRID}\n    f16 x = load a[i]\n    f16 o[i] = x");
    let params: Params = &[
        ("a", Input, Some(F16)),
        ("o", Output, Some(F16)),
        ("n", Int, None),
    ];
    let module = lowered(&body, params, SM_120, EXACT).unwrap();
    let text = print(&module);
    let head = text.split('{').next().unwrap();
    assert_eq!(
        head,
        ".version 8.7\n.target sm_120\n.address_size 64\n\n.visible .entry k(\n\
         \t.param .u64 .ptr .align 16 k_param_0,\n\t.param .u64 .ptr .align 16 k_param_1,\n\
         \t.param .u32 k_param_2\n)\n.maxntid 256, 1, 1\n"
    );
    assert!(text.contains("\t// f16 x = load a[i]\n"), "{text}");
    assert!(text.is_ascii());
    let one_wave = Module {
        single_wave: true,
        ..module.clone()
    };
    assert!(print(&one_wave).contains(".maxntid 256, 1, 1\n.minnctapersm 1\n{"));
    let older = lowered(&body, params, Target { sm: 80 }, EXACT).unwrap();
    assert_eq!((older.version, older.target.as_str()), ((7, 8), "sm_80"));
    // A kernel waits for the grid launched ahead of it, first of all, on the targets that have the instruction.
    let waits = |module: &Module| {
        let first = module.instructions().next().unwrap();
        first.opcode == Opcode::GridWait
    };
    let hopper = lowered(&body, params, Target { sm: 90 }, EXACT).unwrap();
    assert!(waits(&module) && waits(&hopper) && !waits(&older));
    assert!(text.contains("\tgriddepcontrol.wait;\n\tld.param.u64 %rd0, [k_param_0];\n"));
}

/// A parameter aligned to less than one of its elements, or to no power of two, is refused.
#[test]
fn a_buffer_is_aligned_to_its_elements_at_least() {
    let body = common::parse(&format!("{GRID}\n    f32 x = load a[i]\n    f32 o[i] = x"));
    for align in [2, 24] {
        let mut interface = common::interface(A_O, 256);
        interface.params[0].align = align;
        let kernel = emmy_ptx::Kernel {
            name: "k",
            body: &body,
            interface: &interface,
        };
        let message = emmy_ptx::lower(&kernel, SM_120, EXACT)
            .unwrap_err()
            .to_string();
        assert!(
            message.contains("which is no alignment of its elements"),
            "{message}"
        );
    }
}

// --- addresses -------------------------------------------------------------------------------------------

/// The variable part of a flat index is scaled to bytes and widened by one multiply, shared by every buffer read
/// at it; the constant part is the offset of the address.
#[test]
fn an_address_is_a_base_a_scaled_index_and_a_constant() {
    let got = under_grid(
        "
        f32 x = load a[i]
        f32 y = load a[i + 3]
        f32 z = load a[i - 1]
        f32 v = x + y
        f32 w = v + z
        f32 o[i * 2 + 1] = w
        ",
        A_O,
        EXACT,
    );
    assert_eq!(
        got,
        [
            "mul.wide.s32 %rd2, %r3, 4",
            "add.s64 %rd3, %rd0, %rd2",
            "ld.global.nc.f32 %f0, [%rd3]",
            "ld.global.nc.f32 %f1, [%rd3+12]",
            "ld.global.nc.f32 %f2, [%rd3+-4]",
            "add.rn.f32 %f3, %f0, %f1",
            "add.rn.f32 %f4, %f3, %f2",
            "mul.wide.s32 %rd4, %r3, 8",
            "add.s64 %rd5, %rd1, %rd4",
            "st.global.f32 [%rd5+4], %f4",
        ]
    );
}

/// Each term of the index is widened by itself and added in 64 bits, the terms bound in outer bodies first: the
/// part a loop does not change is one register for all its trips, which is the form ptxas folds when it unrolls.
#[test]
fn the_terms_of_an_index_are_added_in_64_bits_outermost_first() {
    let got = under_grid(
        "
        for j in 0..8:
            f32 x = load a[j + i * 8]
            f32 s <- add(s, x) over j
        f32 o[i] = s
        ",
        A_O,
        EXACT,
    );
    assert_eq!(
        got,
        [
            "mov.f32 %f0, 0f00000000",
            "mul.wide.s32 %rd2, %r3, 32",
            "add.s64 %rd3, %rd0, %rd2",
            "mov.u64 %rd4, 0",
            "mov.u32 %r4, 0",
            "L1:",
            "shl.b64 %rd5, %rd4, 2",
            "add.s64 %rd6, %rd3, %rd5",
            "ld.global.nc.f32 %f1, [%rd6]",
            "add.rn.f32 %f0, %f0, %f1",
            "add.s64 %rd4, %rd4, 1",
            "add.s32 %r4, %r4, 1",
            "setp.lt.s32 %p0, %r4, 8",
            "@%p0 bra L1",
            "mul.wide.s32 %rd7, %r3, 4",
            "add.s64 %rd8, %rd1, %rd7",
            "st.global.f32 [%rd8], %f0",
        ]
    );
}

/// An index past the 32-bit range is computed in 64 bits from the term that leaves it: the product of two 32-bit
/// values is the widening multiply, and a 64-bit value is scaled by a shift.
#[test]
fn an_index_past_32_bits_is_widened_once() {
    let got = code(
        "
Grid(axes=(i in 0..1048576), decode=(block_idx.x * 256 + thread_idx.x), block_threads=256):
    i64 big = i * 1048576
    f32 x = load a[big + 5]
    f32 o[i] = x
",
        A_O,
        EXACT,
    );
    assert_eq!(
        got[4..],
        [
            "mul.wide.s32 %rd2, %r3, 1048576",
            "shl.b64 %rd3, %rd2, 2",
            "add.s64 %rd4, %rd0, %rd3",
            "ld.global.nc.f32 %f0, [%rd4+20]",
            "mul.wide.s32 %rd5, %r3, 4",
            "add.s64 %rd6, %rd1, %rd5",
            "st.global.f32 [%rd6], %f0",
        ]
    );
}

// --- integers --------------------------------------------------------------------------------------------

/// A division by a constant that is no power of two, over a dividend the ranges prove non-negative, is a
/// multiply-high and a shift; its remainder subtracts the quotient's multiple.
#[test]
fn a_division_by_a_constant_over_a_non_negative_index_is_a_multiply_high() {
    let got = under_grid(
        "
        i32 q = i / 7
        i32 m = i % 7
        i32 s = q + m
        f32 v = (f32)s
        f32 o[i] = v
        ",
        A_O,
        EXACT,
    );
    assert_eq!(
        got[..5],
        [
            "mul.hi.u32 %r4, %r3, 2454267027",
            "shr.u32 %r5, %r4, 2",
            "mul.lo.s32 %r6, %r5, 7",
            "sub.s32 %r7, %r3, %r6",
            "add.s32 %r8, %r5, %r7",
        ]
    );
}

/// A dividend that may be negative keeps the floor: an arithmetic shift and a mask by a power of two, and for any
/// other constant the multiply-high over the dividend folded by its sign, the quotient folded back.
#[test]
fn a_division_of_a_possibly_negative_value_floors() {
    let got = under_grid(
        "
        i32 d = (i - 2000) / 7
        i32 m = (i - 2000) % 7
        i32 e = (i - 2000) % 8
        i32 f = (i - 2000) / 8
        i32 s = d + m + e + f
        f32 v = (f32)s
        f32 o[i] = v
        ",
        A_O,
        EXACT,
    );
    assert_eq!(
        got[..10],
        [
            "sub.s32 %r4, %r3, 2000",
            "shr.s32 %r5, %r4, 31",
            "xor.b32 %r6, %r4, %r5",
            "mul.hi.u32 %r7, %r6, 2454267027",
            "shr.u32 %r8, %r7, 2",
            "xor.b32 %r9, %r8, %r5",
            "mul.lo.s32 %r10, %r9, 7",
            "sub.s32 %r11, %r4, %r10",
            "and.b32 %r12, %r4, 7",
            "shr.s32 %r13, %r4, 3",
        ]
    );
}

/// A division by a value known only where it runs is the divide instruction: unsigned where the ranges prove both
/// sides non-negative (a symbolic size never is negative) or the operands are unsigned, else lowered by one where the
/// remainder is nonzero and its sign is not the divisor's, the remainder raised by one divisor.
#[test]
fn a_division_by_a_variable_is_the_divide_instruction() {
    let got = under_grid(
        "
        u32 u = load b[i]
        i32 q = i / (i + 1)
        u32 m = u % u
        u32 r = u >> 3
        i32 back = i - 2000
        i32 fq = i / back
        i32 fr = i % back
        u32 s = q + m + r + fq + fr
        u32 o[i] = s
        ",
        &[("b", Input, Some(U32)), ("o", Output, Some(U32))],
        EXACT,
    );
    assert_eq!(
        got[3..18],
        [
            "add.s32 %r5, %r3, 1",
            "div.u32 %r6, %r3, %r5",
            "rem.u32 %r7, %r4, %r4",
            "shr.u32 %r8, %r4, 3",
            "sub.s32 %r9, %r3, 2000",
            "rem.s32 %r10, %r3, %r9",
            "xor.b32 %r11, %r10, %r9",
            "setp.lt.s32 %p0, %r11, 0",
            "setp.ne.s32 %p1, %r10, 0",
            "and.pred %p2, %p0, %p1",
            "div.s32 %r12, %r3, %r9",
            "selp.s32 %r13, -1, 0, %p2",
            "add.s32 %r14, %r12, %r13",
            "selp.s32 %r15, %r9, 0, %p2",
            "add.s32 %r16, %r10, %r15",
        ]
    );
    let sized = code(
        "
Grid(axes=(i in 0..4096), decode=(block_idx.x * 256 + thread_idx.x), block_threads=256):
    i32 w = (i + 1) % n
    i32 o[i] = w
",
        &[("o", Output, Some(I32)), ("n", Int, None)],
        EXACT,
    );
    assert_eq!(
        sized[4..6],
        ["add.s32 %r5, %r4, 1", "rem.u32 %r6, %r5, %r0"]
    );
}

/// A grid decode over a symbolic size in 64 bits: the built-ins, the size, their sums, products and casts are never
/// negative, so a quotient is the unsigned instruction with no floor correction, behind cicc's branch to the 32-bit
/// one where both operands fit; a remainder of two 32-bit values widened is the 32-bit instruction; and a remainder
/// of the 64-bit quotient by a constant is a multiply-high. A quotient computed twice is computed once, and a
/// remainder beside it subtracts its multiple.
#[test]
fn a_decode_over_a_symbolic_size_divides_unsigned() {
    let got = code(
        "
Grid(axes=(a0 in 0..4, a1 in 0..6, a3 in 0..n, c in 0..256), decode=((256 * (long long)block_idx.x + (long long)thread_idx.x) // (1536 * (long long)n), (256 * (long long)block_idx.x + (long long)thread_idx.x) // (256 * (long long)n) % 6, (long long)block_idx.x % (long long)n, thread_idx.x), block_threads=256):
    i64 q = (256 * (long long)block_idx.x + (long long)thread_idx.x) // (1536 * (long long)n)
    i64 r = (256 * (long long)block_idx.x + (long long)thread_idx.x) % (1536 * (long long)n)
    i64 s = q + r + a0 + a1 + a3
    i64 o[c] = s
",
        &[("o", Output, Some(DType::I64)), ("n", Int, None)],
        EXACT,
    );
    let divisions: Vec<&str> = got
        .iter()
        .map(String::as_str)
        .filter(|line| {
            ["div.", "rem.", "mul.hi", "mul.lo.s64", "sub.s64"]
                .iter()
                .any(|op| line.contains(op))
        })
        .collect();
    assert_eq!(
        divisions,
        [
            "mul.lo.s64 %rd6, 1536, %rd5",
            "div.u32 %r5, %r3, %r4",
            "div.u64 %rd9, %rd4, %rd6",
            "div.u32 %r8, %r6, %r7",
            "div.u64 %rd14, %rd4, %rd11",
            "mul.hi.u64 %rd16, %rd15, -6148914691236517205",
            "rem.u32 %r13, %r1, %r0",
            "mul.lo.s64 %rd18, %rd10, %rd6",
            "sub.s64 %rd19, %rd4, %rd18",
        ]
    );
    assert!(
        !got.iter().any(|line| line.contains(".s64 %p")),
        "no floor correction"
    );
}

/// The integer operations that are one instruction each, at the width and the signedness of their operands.
#[test]
fn the_integer_operations_are_one_instruction_each() {
    let got = under_grid(
        "
        i32 x = load b[i]
        i32 a1 = x * 6
        i32 a2 = x & 255
        i32 a3 = x | i
        i32 a4 = x ^ i
        i32 a5 = x << 2
        i32 a6 = x >> 1
        i32 a7 = minimum(x, i)
        i32 a8 = maximum(x, 3)
        i32 a9 = -x
        i32 b1 = abs(x)
        i32 b2 = bitwise_count(x)
        bool b3__bool = x < i
        i32 b3 = (i32)b3__bool
        i32 b4 = x == 4 ? a1 : a2
        i64 b5 = (i64)x
        i64 b6 = b5 * 3
        i32 b7 = (i32)b6
        i32 o[i] = b7
        ",
        &[("b", Input, Some(I32)), ("o", Output, Some(I32))],
        EXACT,
    );
    assert_eq!(
        got[3..21],
        [
            "mul.lo.s32 %r5, %r4, 6",
            "and.b32 %r6, %r4, 255",
            "or.b32 %r7, %r4, %r3",
            "xor.b32 %r8, %r4, %r3",
            "shl.b32 %r9, %r4, 2",
            "shr.s32 %r10, %r4, 1",
            "min.s32 %r11, %r4, %r3",
            "max.s32 %r12, %r4, 3",
            "neg.s32 %r13, %r4",
            "abs.s32 %r14, %r4",
            "popc.b32 %r15, %r4",
            "setp.lt.s32 %p0, %r4, %r3",
            "selp.s32 %r16, 1, 0, %p0",
            "setp.eq.s32 %p1, %r4, 4",
            "selp.s32 %r17, %r5, %r6, %p1",
            "cvt.s64.s32 %rd4, %r4",
            "mul.lo.s64 %rd5, %rd4, 3",
            "cvt.u32.u64 %r18, %rd5",
        ]
    );
}

/// The same operations over 64-bit and unsigned values: the instruction's type follows the operands, a quotient of
/// a 64-bit value is the divide instruction with its floor correction, an unsigned index is widened without its
/// sign, and a 64-bit state steps in place.
#[test]
fn the_64_bit_and_unsigned_forms_follow_their_operands() {
    let got = under_grid(
        "
        u32 u = load b[i]
        i64 w = load c[i]
        init i64 total = 0
        u32 a1 = minimum(u, 7)
        u32 a2 = maximum(u, a1)
        u32 a3 = a2 / 3
        i64 b1 = w + 5
        i64 b2 = w - b1
        i64 b3 = minimum(w, b2)
        i64 b4 = maximum(b3, 9)
        i64 b5 = b4 / 3
        i64 b6 = w >> 2
        i64 b7 = w & 255
        i64 b8 = -w
        i64 b9 = abs(w)
        i32 c1 = bitwise_count(w)
        i64 c2 = w <= b1 ? w : b1
        i64 c3 = (i64)u
        i64 total <- add(total, c3)
        i64 total <- maximum(total, b9)
        let i32 c4 = i * 4
        c5 = select:
            c1 when u != 3 || w >= 0
            c4 when i >= 0
        f32 x = load a[u]
        i64 o[i] = total
        ",
        &[
            ("a", Input, Some(F32)),
            ("b", Input, Some(U32)),
            ("c", Input, Some(DType::I64)),
            ("o", Output, Some(DType::I64)),
        ],
        EXACT,
    );
    let other: Vec<&str> = got
        .iter()
        .map(String::as_str)
        .filter(|line| !line.starts_with("add.s64 %rd") || line.contains(", 5"))
        .collect();
    assert_eq!(
        other,
        [
            "mul.wide.s32 %rd4, %r3, 4",
            "ld.global.nc.u32 %r4, [%rd5]",
            "mul.wide.s32 %rd6, %r3, 8",
            "ld.global.nc.s64 %rd8, [%rd7]",
            "mov.s64 %rd9, 0",
            "min.u32 %r5, %r4, 7",
            "max.u32 %r6, %r4, %r5",
            "div.u32 %r7, %r6, 3",
            "add.s64 %rd10, %rd8, 5",
            "sub.s64 %rd11, %rd8, %rd10",
            "min.s64 %rd12, %rd8, %rd11",
            "max.s64 %rd13, %rd12, 9",
            "rem.s64 %rd14, %rd13, 3",
            "xor.b64 %rd15, %rd14, 3",
            "setp.lt.s64 %p0, %rd15, 0",
            "setp.ne.s64 %p1, %rd14, 0",
            "and.pred %p2, %p0, %p1",
            "div.s64 %rd16, %rd13, 3",
            "selp.s64 %rd17, -1, 0, %p2",
            "shr.s64 %rd19, %rd8, 2",
            "and.b64 %rd20, %rd8, 255",
            "neg.s64 %rd21, %rd8",
            "abs.s64 %rd22, %rd8",
            "popc.b64 %r8, %rd8",
            "setp.le.s64 %p3, %rd8, %rd10",
            "selp.s64 %rd23, %rd8, %rd10, %p3",
            "cvt.u64.u32 %rd24, %r4",
            "max.s64 %rd9, %rd9, %rd22",
            "shl.b32 %r9, %r3, 2",
            "setp.ne.u32 %p4, %r4, 3",
            "setp.ge.s64 %p5, %rd8, 0",
            "or.pred %p6, %p4, %p5",
            "selp.s32 %r10, %r8, %r9, %p6",
            "mul.wide.u32 %rd25, %r4, 4",
            "ld.global.nc.f32 %f0, [%rd26]",
            "st.global.s64 [%rd27], %rd9",
        ]
    );
}

// --- floats ----------------------------------------------------------------------------------------------

/// Every float instruction that rounds says so, and with the fusion decision off a product and the sum that reads
/// it stay two instructions.
#[test]
fn a_product_and_its_sum_stay_apart_without_the_fusion_decision() {
    let got = under_grid(
        "
        f32 x = load a[i]
        f32 t = x * x
        f32 v = t + x
        f32 w = 2 * v - x
        f32 o[i] = w
        ",
        A_O,
        EXACT,
    );
    assert_eq!(
        got[3..7],
        [
            "mul.rn.f32 %f1, %f0, %f0",
            "add.rn.f32 %f2, %f1, %f0",
            "mul.rn.f32 %f3, 0f40000000, %f2",
            "sub.rn.f32 %f4, %f3, %f0",
        ]
    );
}

/// With the fusion decision on, a product read once by a sum or a difference in its own body is one fused
/// multiply-add with it: named or written in the reader, on either side, a difference through an exact negation.
#[test]
fn a_product_read_once_by_a_sum_fuses_under_the_fusion_decision() {
    let got = under_grid(
        "
        f32 x = load a[i]
        f32 t = x * x
        f32 v = t + x
        f32 w = 2 * v - x
        f32 u = x - 3 * w
        f32 p = w * u
        f32 q = 1.5 - p
        f32 o[i] = q
        ",
        A_O,
        FUSED,
    );
    assert_eq!(
        got[3..10],
        [
            "fma.rn.f32 %f1, %f0, %f0, %f0",
            "neg.f32 %f2, %f0",
            "fma.rn.f32 %f3, 0f40000000, %f1, %f2",
            "fma.rn.f32 %f4, 0fC0400000, %f3, %f0",
            "neg.f32 %f5, %f3",
            "fma.rn.f32 %f6, %f5, %f4, 0f3FC00000",
            "add.s64 %rd4, %rd1, %rd2",
        ]
    );
}

/// A product with two readers, or one read in another body, is computed once where it stands and fuses with
/// neither.
#[test]
fn a_product_read_twice_or_in_another_body_does_not_fuse() {
    let got = under_grid(
        "
        f32 x = load a[i]
        f32 t = x * x
        f32 v = t + x
        f32 w = t + v
        f32 p = w * x
        if i < 5:
            f32 q = p + x
            f32 o[i] = q
        f32 o[i + 1] = w
        ",
        A_O,
        FUSED,
    );
    assert_eq!(
        got[3..10],
        [
            "mul.rn.f32 %f1, %f0, %f0",
            "add.rn.f32 %f2, %f1, %f0",
            "add.rn.f32 %f3, %f1, %f2",
            "mul.rn.f32 %f4, %f3, %f0",
            "setp.lt.s32 %p0, %r3, 5",
            "@!%p0 bra L1",
            "add.rn.f32 %f5, %f4, %f0",
        ]
    );
}

/// The body's own `fma` is the instruction whatever the fusion decision.
#[test]
fn an_fma_the_body_writes_is_an_fma() {
    let got = under_grid(
        "
        f32 x = load a[i]
        f32 v = fma(x, x, 0.5)
        f32 o[i] = v
        ",
        A_O,
        EXACT,
    );
    assert_eq!(got[3], "fma.rn.f32 %f1, %f0, %f0, 0f3F000000");
}

/// The float operations that are one instruction each.
#[test]
fn the_float_operations_are_one_instruction_each() {
    let got = under_grid(
        "
        f32 x = load a[i]
        f32 a1 = x / 3.0
        f32 a2 = true_divide(x, a1)
        f32 a3 = maximum(x, a2)
        f32 a4 = minimum(x, a3)
        f32 a5 = sqrt(a4)
        f32 a6 = reciprocal(a5)
        f32 a7 = floor(a6)
        f32 a8 = ceil(a7)
        f32 a9 = trunc(a8)
        f32 b1 = rint(a9)
        f32 b2 = abs(b1)
        f32 b3 = -b2
        f32 b4 = relu(b3)
        f32 b5 = square(b4)
        f32 b6 = where(b5 > x, b5, x)
        f32 b7 = b6 != x ? b6 : 0.25
        f32 o[i] = b7
        ",
        A_O,
        EXACT,
    );
    assert_eq!(
        got[3..21],
        [
            "div.rn.f32 %f1, %f0, 0f40400000",
            "div.rn.f32 %f2, %f0, %f1",
            "max.f32 %f3, %f0, %f2",
            "min.f32 %f4, %f0, %f3",
            "sqrt.rn.f32 %f5, %f4",
            "rcp.rn.f32 %f6, %f5",
            "cvt.rmi.f32.f32 %f7, %f6",
            "cvt.rpi.f32.f32 %f8, %f7",
            "cvt.rzi.f32.f32 %f9, %f8",
            "cvt.rni.f32.f32 %f10, %f9",
            "abs.f32 %f11, %f10",
            "neg.f32 %f12, %f11",
            "max.f32 %f13, %f12, 0f00000000",
            "mul.rn.f32 %f14, %f13, %f13",
            "setp.gt.f32 %p0, %f14, %f0",
            "selp.f32 %f15, %f14, %f0, %p0",
            "setp.neu.f32 %p1, %f15, %f0",
            "selp.f32 %f16, %f15, 0f3E800000, %p1",
        ]
    );
}

/// With the flush-to-zero decision on, every f32 instruction that takes the flag carries it.
#[test]
fn the_flush_to_zero_decision_marks_every_f32_instruction() {
    let ftz = Precision {
        fuse: true,
        ftz: true,
    };
    let got = under_grid(
        "
        f32 x = load a[i]
        f32 t = x * x
        f32 v = t + x
        f32 w = sqrt(v) / x
        f32 u = floor(w)
        i32 n = (i32)u
        f32 y = maximum(u, x)
        f32 z = y < x ? -y : abs(x)
        f32 r = z + (f32)n
        f32 o[i] = r
        ",
        A_O,
        ftz,
    );
    assert_eq!(
        got[3..15],
        [
            "fma.rn.ftz.f32 %f1, %f0, %f0, %f0",
            "sqrt.rn.ftz.f32 %f2, %f1",
            "div.rn.ftz.f32 %f3, %f2, %f0",
            "cvt.rmi.ftz.f32.f32 %f4, %f3",
            "cvt.rzi.ftz.s32.f32 %r4, %f4",
            "max.ftz.f32 %f5, %f4, %f0",
            "setp.lt.ftz.f32 %p0, %f5, %f0",
            "neg.ftz.f32 %f6, %f5",
            "abs.ftz.f32 %f7, %f0",
            "selp.f32 %f8, %f6, %f7, %p0",
            "cvt.rn.f32.s32 %f9, %r4",
            "add.rn.ftz.f32 %f10, %f8, %f9",
        ]
    );
}

/// Half precision runs natively where the body says so: its values live in 16-bit registers, its instructions take
/// no immediate, so a constant is moved into a register by its bits. A half product is never fused with its sum:
/// each operation rounds, as eager's do.
#[test]
fn half_precision_runs_in_16_bit_registers() {
    let got = under_grid(
        "
        f16 x = load a[i]
        f16 y = x * x + x
        f16 z = maximum(y, -x) - 1.0
        f16 w = abs(z) < x ? z : x
        f16 o[i] = w
        ",
        &[("a", Input, Some(F16)), ("o", Output, Some(F16))],
        FUSED,
    );
    assert_eq!(
        got,
        [
            "mul.wide.s32 %rd2, %r3, 2",
            "add.s64 %rd3, %rd0, %rd2",
            "ld.global.nc.b16 %h0, [%rd3]",
            "mul.rn.f16 %h1, %h0, %h0",
            "add.rn.f16 %h2, %h1, %h0",
            "neg.f16 %h3, %h0",
            "max.f16 %h4, %h2, %h3",
            "mov.b16 %h5, 15360",
            "sub.rn.f16 %h6, %h4, %h5",
            "abs.f16 %h7, %h6",
            "setp.lt.f16 %p0, %h7, %h0",
            "selp.b16 %h8, %h6, %h0, %p0",
            "add.s64 %rd4, %rd1, %rd2",
            "st.global.b16 [%rd4], %h8",
        ]
    );
}

/// A bfloat16 accumulator of a maximum or a minimum steps at f32 between two conversions, which are exact: bf16 is
/// never computed in.
#[test]
fn a_bfloat16_selection_accumulator_steps_at_f32() {
    let got = under_grid(
        "
        bf16 x = load a[i]
        init bf16 m = 0
        bf16 m <- maximum(m, x)
        bf16 o[i] = m
        ",
        &[("a", Input, Some(BF16)), ("o", Output, Some(BF16))],
        EXACT,
    );
    let steps: Vec<&str> = got
        .iter()
        .map(String::as_str)
        .filter(|line| line.contains("f32") || line.contains("bf16"))
        .collect();
    assert_eq!(
        steps,
        [
            "cvt.f32.bf16 %f0, %h1",
            "cvt.f32.bf16 %f1, %h0",
            "max.f32 %f2, %f0, %f1",
            "cvt.rn.bf16.f32 %h1, %f2",
        ]
    );
}

/// The half forms the other tests leave out: a product and a sum apart, a minimum, the truth of a half against a
/// zero in a register, and the conversions between a half and the integers.
#[test]
fn the_other_half_forms() {
    let got = under_grid(
        "
        f16 x = load a[i]
        bf16 b = load c[i]
        f16 p = x * x
        f16 s = p + x
        f16 m = minimum(s, x)
        bool t = (bool)m
        bool e = (m >= x) ^ (m == s)
        u32 u = (u32)m
        i64 l = (i64)m
        f16 h1 = (f16)u
        f16 h2 = (f16)l
        i32 n = (i32)b
        bf16 b2 = (bf16)n
        f16 h3 = t ? h1 : h2
        f16 h4 = e ? h3 : m
        f16 o[i] = h4
        bf16 q[i] = b2
        ",
        &[
            ("a", Input, Some(F16)),
            ("c", Input, Some(BF16)),
            ("o", Output, Some(F16)),
            ("q", Output, Some(BF16)),
        ],
        EXACT,
    );
    assert_eq!(
        got[5..22],
        [
            "mul.rn.f16 %h2, %h0, %h0",
            "add.rn.f16 %h3, %h2, %h0",
            "min.f16 %h4, %h3, %h0",
            "mov.b16 %h5, 0",
            "setp.neu.f16 %p0, %h4, %h5",
            "setp.ge.f16 %p1, %h4, %h0",
            "setp.eq.f16 %p2, %h4, %h3",
            "xor.pred %p3, %p1, %p2",
            "cvt.rzi.u32.f16 %r4, %h4",
            "cvt.rzi.s64.f16 %rd7, %h4",
            "cvt.rn.f16.u32 %h6, %r4",
            "cvt.rn.f16.s64 %h7, %rd7",
            "cvt.rzi.s32.bf16 %r5, %h1",
            "cvt.rn.bf16.s32 %h8, %r5",
            "selp.b16 %h9, %h6, %h7, %p0",
            "selp.b16 %h10, %h9, %h4, %p3",
            "add.s64 %rd8, %rd2, %rd4",
        ]
    );
}

/// A cast is one conversion: exact from a half to f32, rounded to nearest into a float, toward zero into an
/// integer, and through a predicate to and from a bool.
#[test]
fn a_cast_is_one_conversion() {
    let got = under_grid(
        "
        f16 h = load a[i]
        bf16 b = load c[i]
        bool m = load d[i]
        f32 x = (f32)h
        f32 y = (f32)b
        f32 z = x + y
        f16 h2 = (f16)z
        bf16 b2 = (bf16)z
        i32 n = (i32)z
        u32 u = (u32)z
        i64 l = (i64)z
        f32 f1 = (f32)n + (f32)u
        f32 f2 = f1 + (f32)l
        f32 f3 = f2 + (f32)m
        i32 k = (i32)h
        f16 h3 = (f16)k
        bool t = (bool)f3
        bool t2 = t && m
        i32 e = (i32)t2
        f32 o[i] = f3
        f16 p[i] = h2
        bf16 q[i] = b2
        bool r[i] = t2
        i32 s[i] = e
        f16 p[i + 1] = h3
        ",
        &[
            ("a", Input, Some(F16)),
            ("c", Input, Some(BF16)),
            ("d", Input, Some(Bool)),
            ("o", Output, Some(F32)),
            ("p", Output, Some(F16)),
            ("q", Output, Some(BF16)),
            ("r", Output, Some(Bool)),
            ("s", Output, Some(I32)),
        ],
        EXACT,
    );
    let conversions: Vec<&str> = got
        .iter()
        .map(String::as_str)
        .filter(|line| {
            !line.contains("mul.wide")
                && !line.starts_with("add.s64")
                && !line.starts_with("cvt.s64")
        })
        .collect();
    assert_eq!(
        conversions,
        [
            "ld.global.nc.b16 %h0, [%rd9]",
            "ld.global.nc.b16 %h1, [%rd10]",
            "ld.global.nc.u8 %r4, [%rd12]",
            "setp.ne.u32 %p0, %r4, 0",
            "cvt.f32.f16 %f0, %h0",
            "cvt.f32.bf16 %f1, %h1",
            "add.rn.f32 %f2, %f0, %f1",
            "cvt.rn.f16.f32 %h2, %f2",
            "cvt.rn.bf16.f32 %h3, %f2",
            "cvt.rzi.s32.f32 %r5, %f2",
            "cvt.rzi.u32.f32 %r6, %f2",
            "cvt.rzi.s64.f32 %rd13, %f2",
            "cvt.rn.f32.s32 %f3, %r5",
            "cvt.rn.f32.u32 %f4, %r6",
            "add.rn.f32 %f5, %f3, %f4",
            "cvt.rn.f32.s64 %f6, %rd13",
            "add.rn.f32 %f7, %f5, %f6",
            "selp.f32 %f8, 0f3F800000, 0f00000000, %p0",
            "add.rn.f32 %f9, %f7, %f8",
            "cvt.rzi.s32.f16 %r7, %h0",
            "cvt.rn.f16.s32 %h4, %r7",
            "setp.neu.f32 %p1, %f9, 0f00000000",
            "and.pred %p2, %p1, %p0",
            "selp.s32 %r8, 1, 0, %p2",
            "st.global.f32 [%rd15], %f9",
            "st.global.b16 [%rd16], %h2",
            "st.global.b16 [%rd17], %h3",
            "st.global.u8 [%rd18], %r8",
            "st.global.s32 [%rd19], %r8",
            "st.global.b16 [%rd16+2], %h4",
        ]
    );
}

// --- loops, conditions and states ------------------------------------------------------------------------

/// A loop over a static axis is a counter from zero to a constant with its test at the bottom, and the accumulator
/// it folds is one register: seeded ahead of the loop, stepped in place, read after it.
#[test]
fn a_loop_counts_from_zero_and_steps_its_accumulator_in_place() {
    let got = under_grid(
        "
        for j in 0..8:
            f32 x = load a[j]
            f32 t = x * x
            f32 s <- add(s, t) over j
            f32 m <- maximum(m, x) over j
        f32 v = s + m
        f32 o[i] = v
        ",
        A_O,
        FUSED,
    );
    assert_eq!(
        got[..13],
        [
            "mov.f32 %f0, 0f00000000",
            "mov.f32 %f1, 0fF149F2CA",
            "mov.u64 %rd2, 0",
            "mov.u32 %r4, 0",
            "L1:",
            "shl.b64 %rd3, %rd2, 2",
            "add.s64 %rd4, %rd0, %rd3",
            "ld.global.nc.f32 %f2, [%rd4]",
            "fma.rn.f32 %f0, %f2, %f2, %f0",
            "max.f32 %f1, %f1, %f2",
            "add.s64 %rd2, %rd2, 1",
            "add.s32 %r4, %r4, 1",
            "setp.lt.s32 %p0, %r4, 8",
        ]
    );
}

/// A loop over a symbolic extent may run no trip, so its test also stands ahead of it.
#[test]
fn a_loop_over_a_symbolic_extent_is_tested_ahead() {
    let got = code(
        &format!(
            "{GRID}\n    for j in 0..n:\n        f32 x = load a[j]\n        f32 s <- add(s, x) over j\n    f32 o[i] = s"
        ),
        &[
            ("a", Input, Some(F32)),
            ("o", Output, Some(F32)),
            ("n", Int, None),
        ],
        EXACT,
    );
    assert_eq!(
        got[4..16],
        [
            "mov.f32 %f0, 0f00000000",
            "mov.u64 %rd2, 0",
            "mov.u32 %r5, 0",
            "setp.ge.s32 %p0, %r5, %r0",
            "@%p0 bra L1",
            "L2:",
            "shl.b64 %rd3, %rd2, 2",
            "add.s64 %rd4, %rd0, %rd3",
            "ld.global.nc.f32 %f1, [%rd4]",
            "add.rn.f32 %f0, %f0, %f1",
            "add.s64 %rd2, %rd2, 1",
            "add.s32 %r5, %r5, 1",
        ]
    );
}

/// A loop whose copies the kernel lowering wrote is kept as it is: ptxas is told not to unroll it again, whether it
/// counts its trips or steps its axis, unless it is static and runs at most four trips. A kept loop's address reads
/// a multiple of its counter stepped by the term's bytes, not the counter shifted in the trip.
#[test]
fn an_unrolled_loop_is_kept_as_it_is() {
    let params = [
        ("a", Input, Some(F32)),
        ("o", Output, Some(F32)),
        ("n", Int, None),
    ];
    for head in [
        "for j in 0..64 step 2 unrolled=true:",
        "for j in 0..n step 2 until n - 1 unrolled=true:",
    ] {
        let got = code(
            &format!(
                "{GRID}\n    {head}\n        f32 x = load a[j]\n        f32 y = load a[j + 1]\n        f32 s <- add(s, x) over j\n        f32 s <- add(s, y) over j\n    f32 o[i] = s"
            ),
            &params,
            EXACT,
        );
        let at = got.iter().position(|line| line == ".pragma \"nounroll\"");
        let head = got
            .iter()
            .position(|line| line.starts_with('L') && line.ends_with(':'));
        assert!(at.is_some() && head < at, "{got:#?}");
    }
    let body = |head: &str| {
        format!(
            "{GRID}\n    {head}\n        f32 x = load a[j]\n        f32 y = load a[j + 1]\n        f32 s <- add(s, x) over j\n        f32 s <- add(s, y) over j\n    f32 o[i] = s"
        )
    };
    let kept = code(
        &body("for j in 0..64 step 2 unrolled=true:"),
        &params,
        EXACT,
    );
    assert!(
        kept.iter()
            .any(|line| line.starts_with("add.s64 %rd") && line.ends_with(", 8")),
        "{kept:#?}"
    );
    assert!(
        !kept.iter().any(|line| line.starts_with("shl.b64")),
        "{kept:#?}"
    );
    let short = code(&body("for j in 0..8 step 2 unrolled=true:"), &params, EXACT);
    assert!(
        !short.iter().any(|line| line.contains("nounroll")),
        "{short:#?}"
    );
    let rolled = code(
        &format!(
            "{GRID}\n    for j in 0..n:\n        f32 x = load a[j]\n        f32 s <- add(s, x) over j\n    f32 o[i] = s"
        ),
        &params,
        EXACT,
    );
    assert!(!rolled.iter().any(|line| line.contains("nounroll")));
}

/// A strided loop whose step from the lowest start already passes the highest end runs at most one trip: it is
/// the condition that its start is below its end, with the axis its start.
#[test]
fn a_strided_loop_of_at_most_one_trip_is_a_condition() {
    let got = under_grid(
        "
        for k in thread_idx.x..32 step 256:
            f32 x = load a[k]
            f32 o[k] = x
        ",
        A_O,
        EXACT,
    );
    assert_eq!(
        got,
        [
            "setp.lt.s32 %p0, %r2, 32",
            "@!%p0 bra L1",
            "mul.wide.s32 %rd2, %r2, 4",
            "add.s64 %rd3, %rd0, %rd2",
            "ld.global.nc.f32 %f0, [%rd3]",
            "add.s64 %rd4, %rd1, %rd2",
            "st.global.f32 [%rd4], %f0",
            "L1:",
        ]
    );
}

/// A strided loop with a constant start and step runs the same trips in every thread: a counter from zero to the
/// trip count, the axis computed from it. One that starts at a thread's own value steps the axis itself.
#[test]
fn a_strided_loop_counts_its_trips_where_they_are_the_same_for_every_thread() {
    let got = under_grid(
        "
        for k in 4..64 step 8:
            f32 x = load a[k]
            f32 s <- add(s, x) over k
        for k in thread_idx.x..64 step 32 seed=false:
            f32 y = load a[k]
            f32 s <- add(s, y) over k
        f32 o[i] = s
        ",
        A_O,
        EXACT,
    );
    assert_eq!(
        got[..20],
        [
            "mov.f32 %f0, 0f00000000",
            "mov.u64 %rd2, 0",
            "mov.u32 %r4, 0",
            "L1:",
            "shl.b32 %r5, %r4, 3",
            "add.s32 %r6, %r5, 4",
            "shl.b64 %rd3, %rd2, 5",
            "add.s64 %rd4, %rd0, %rd3",
            "ld.global.nc.f32 %f1, [%rd4+16]",
            "add.rn.f32 %f0, %f0, %f1",
            "add.s64 %rd2, %rd2, 1",
            "add.s32 %r4, %r4, 1",
            "setp.lt.s32 %p0, %r4, 8",
            "@%p0 bra L1",
            "mov.u32 %r7, %r2",
            "setp.ge.s32 %p1, %r7, 64",
            "@%p1 bra L2",
            "L3:",
            "mul.wide.s32 %rd5, %r7, 4",
            "add.s64 %rd6, %rd0, %rd5",
        ]
    );
}

/// A condition is a branch past its body, and with an else side a branch over that; each side makes again what
/// the other made, since a thread runs one of them. A select is the value of its first branch whose predicate
/// holds, the last where none does.
#[test]
fn a_condition_branches_and_a_select_picks() {
    let got = under_grid(
        "
        f32 x = load a[i]
        f32 y = -x
        f32 z = abs(x)
        v = select:
            x when i < 8
            y when i < 16 && x > 0.0
            z when i >= 16
        if v > x:
            f32 o[i] = v
        else:
            f32 o[i] = x
        ",
        A_O,
        EXACT,
    );
    assert_eq!(
        got[3..],
        [
            "neg.f32 %f1, %f0",
            "abs.f32 %f2, %f0",
            "setp.lt.s32 %p0, %r3, 16",
            "setp.gt.f32 %p1, %f0, 0f00000000",
            "and.pred %p2, %p0, %p1",
            "selp.f32 %f3, %f1, %f2, %p2",
            "setp.lt.s32 %p3, %r3, 8",
            "selp.f32 %f4, %f0, %f3, %p3",
            "setp.gt.f32 %p4, %f4, %f0",
            "@!%p4 bra L1",
            "add.s64 %rd4, %rd1, %rd2",
            "st.global.f32 [%rd4], %f4",
            "bra L2",
            "L1:",
            "add.s64 %rd5, %rd1, %rd2",
            "st.global.f32 [%rd5], %f0",
            "L2:",
        ]
    );
}

/// A bool converted to a number is its predicate's 1 or 0: the truth of that number is the predicate again, a
/// logical operation over two such numbers combines their predicates, and the number converted to another number
/// type selects that type's 1 or 0. No comparison with zero and no integer conversion stands between them.
#[test]
fn a_bool_converted_to_a_number_keeps_its_predicate() {
    let got = under_grid(
        "
        f32 x = load a[i]
        bool b__bool = x < 2.0
        f32 b = (f32)b__bool
        bool c__bool = x == 3.0
        f32 c = (f32)c__bool
        f32 both = b && c
        bool d = (bool)both
        i64 k = (i64)d
        f32 h = (f32)k
        f32 o[i] = h
        ",
        A_O,
        EXACT,
    );
    let selp = got.iter().filter(|line| line.starts_with("selp")).count();
    assert!(
        got.iter().all(|line| !line.starts_with("setp.neu")),
        "{got:#?}"
    );
    assert!(
        got.iter().all(|line| !line.starts_with("cvt.rn.f32.s64")),
        "{got:#?}"
    );
    assert!(
        got.iter().any(|line| line.starts_with("and.pred")),
        "{got:#?}"
    );
    assert!(selp >= 1, "{got:#?}");
}

/// A bool equals a literal 0 where it does not hold, a literal 1 where it does, and another bool where their
/// exclusive or does not hold: the chunk bounds write `0 == i < n` for a negation.
#[test]
fn a_comparison_of_bools_is_a_predicate_or_its_negation() {
    let got = under_grid(
        "
        f32 x = load a[i]
        if 0 == i < 8 || (i < 16) == (x > 0.0) || 1 != i < 32:
            f32 o[i] = x
        ",
        A_O,
        EXACT,
    );
    assert_eq!(
        got[3..11],
        [
            "setp.lt.s32 %p0, %r3, 8",
            "not.pred %p1, %p0",
            "setp.lt.s32 %p2, %r3, 16",
            "setp.gt.f32 %p3, %f0, 0f00000000",
            "xor.pred %p4, %p2, %p3",
            "not.pred %p5, %p4",
            "or.pred %p6, %p1, %p5",
            "setp.lt.s32 %p7, %r3, 32",
        ]
    );
}

/// A state seeded by the body is a register an accumulator steps under a condition, from another base, or at an
/// integer type; a name bound to a state is a copy, since the state steps on.
#[test]
fn a_seeded_state_steps_in_place_and_is_copied_where_it_is_named() {
    let got = under_grid(
        "
        init f32 best = -inf
        init i32 count = 0
        f32 x = load a[i]
        f32 before = best
        if x > 1.0:
            f32 best <- maximum(best, x)
            i32 count <- add(count, 1)
        f32 scaled = x * 0.5
        f32 best <- add(scaled, before)
        f32 n = (f32)count
        f32 v = best + n
        f32 o[i] = v
        ",
        A_O,
        EXACT,
    );
    assert_eq!(
        got,
        [
            "mov.f32 %f0, 0fFF800000",
            "mov.s32 %r4, 0",
            "mul.wide.s32 %rd2, %r3, 4",
            "add.s64 %rd3, %rd0, %rd2",
            "ld.global.nc.f32 %f1, [%rd3]",
            "mov.f32 %f2, %f0",
            "setp.gt.f32 %p0, %f1, 0f3F800000",
            "@!%p0 bra L1",
            "max.f32 %f0, %f0, %f1",
            "add.s32 %r4, %r4, 1",
            "L1:",
            "mul.rn.f32 %f3, %f1, 0f3F000000",
            "add.rn.f32 %f0, %f3, %f2",
            "cvt.rn.f32.s32 %f4, %r4",
            "add.rn.f32 %f5, %f0, %f4",
            "add.s64 %rd4, %rd1, %rd2",
            "st.global.f32 [%rd4], %f5",
        ]
    );
}

/// The finite stand-in of a maximum's or a minimum's unbounded neutral element seeds an integer state with its
/// type's own bound: the lowest value for a maximum, the highest for a minimum, by the body's seed or the loop's.
#[test]
fn an_integer_maximum_or_minimum_is_seeded_with_its_types_bound() {
    let got = under_grid(
        "
        init i64 low = -1e+30
        init i32 high = 1e+30
        for j in 0..4:
            i32 m <- maximum(m, j)
        i64 low <- maximum(low, m)
        i32 high <- minimum(high, m)
        i64 o[i] = low
        ",
        &[("o", Output, Some(DType::I64))],
        EXACT,
    );
    let seeds: Vec<&String> = got
        .iter()
        .filter(|line| line.starts_with("mov.s"))
        .collect();
    assert_eq!(
        seeds,
        [
            "mov.s64 %rd1, -9223372036854775808",
            "mov.s32 %r4, 2147483647",
            "mov.s32 %r5, -2147483648",
        ],
        "{got:?}"
    );
}

// --- the value table -------------------------------------------------------------------------------------

/// A value the compiler makes twice in one scope is made once; what a body made is forgotten when the body closes,
/// since not every thread ran it; and nothing computed from a state is kept, since the state steps on.
#[test]
fn a_value_made_twice_is_made_once_within_its_scope() {
    let got = under_grid(
        "
        init f32 s = 0.0
        if i < 8:
            f32 x = load a[i * 2]
            f32 y = load a[i * 2 + 1]
            f32 s <- add(s, x)
            f32 d = s * 2.0
            f32 s <- add(s, y)
            f32 e = s * 2.0
            f32 f = d + e
            f32 s <- add(s, f)
        f32 z = load a[i * 2]
        f32 v = s + z
        f32 o[i] = v
        ",
        A_O,
        EXACT,
    );
    assert_eq!(
        got,
        [
            "mov.f32 %f0, 0f00000000",
            "setp.lt.s32 %p0, %r3, 8",
            "@!%p0 bra L1",
            "mul.wide.s32 %rd2, %r3, 8",
            "add.s64 %rd3, %rd0, %rd2",
            "ld.global.nc.f32 %f1, [%rd3]",
            "ld.global.nc.f32 %f2, [%rd3+4]",
            "add.rn.f32 %f0, %f0, %f1",
            "mul.rn.f32 %f3, %f0, 0f40000000",
            "add.rn.f32 %f0, %f0, %f2",
            "mul.rn.f32 %f4, %f0, 0f40000000",
            "add.rn.f32 %f5, %f3, %f4",
            "add.rn.f32 %f0, %f0, %f5",
            "L1:",
            "mul.wide.s32 %rd4, %r3, 8",
            "add.s64 %rd5, %rd0, %rd4",
            "ld.global.nc.f32 %f6, [%rd5]",
            "add.rn.f32 %f7, %f0, %f6",
            "mul.wide.s32 %rd6, %r3, 4",
            "add.s64 %rd7, %rd1, %rd6",
            "st.global.f32 [%rd7], %f7",
        ]
    );
}

// --- what is refused -------------------------------------------------------------------------------------

/// A body that is not typed and flat is refused by the statement that is not; a statement or an operation with no
/// rule is an error that names it. Nothing is skipped.
#[test]
fn a_statement_outside_the_rules_is_an_error_that_names_it() {
    let under = |statements: &str| format!("{GRID}\n    f32 x = load a[i]\n{statements}");
    for (statements, expected) in [
        (
            "    f32 o[i, 0] = x",
            "an index of 2 coordinates into o is not flat: `f32 o[i, 0] = x`",
        ),
        (
            "    f32 v = log2(x)",
            "no PTX rule for the operation log2 at f32: `f32 v = log2(x)`",
        ),
        (
            "    WgmmaFence",
            "no PTX rule for `wgmma` on sm_120, which needs sm_90a: `WgmmaFence`",
        ),
        (
            "    RegStore(dst_buffer=o, dst_index=(i), frag=c, shape=(16, 8, 16))",
            "the fragment c is read before it is declared",
        ),
        (
            "    y = load a[i]",
            "a value carries no dtype: `y = load a[i]`",
        ),
        (
            "    f16 v = x + x",
            "x is f32, read as f16: `f16 v = x + x`",
        ),
        (
            "    f16 v = (f16)x\n    f32 o[i] = v",
            "o holds f32, v is f16",
        ),
        ("    f32 a[i] = x", "a is an input"),
        (
            "    f32 y = load missing[i]",
            "missing is no buffer of the kernel",
        ),
        ("    f32 o[i] = later", "later is read before it is defined"),
        (
            "    f32 s <- add(s, x)",
            "the accumulator s has no seed in scope",
        ),
        (
            "    for j in 0..4 unroll=true:\n        f32 o[j] = x",
            "a loop asks to be unrolled",
        ),
        (
            "    f64 w = (f64)x\n    f64 y = exp(w)",
            "no PTX rule for the operation exp at f64",
        ),
        (
            "    init i32 m = 0.5",
            "no PTX rule for a state of i32 seeded 0.5",
        ),
        (
            "    bf16 v = (bf16)x\n    bf16 w = v + v",
            "no PTX rule for Add at bf16 on sm_120",
        ),
    ] {
        let message = refusal(&under(statements), A_O);
        assert!(message.contains(expected), "{statements}: {message}");
    }
    // A half maximum before `sm_80`, which has no half instruction for it, computes in f32 and rounds once.
    let half = "    f16 h = (f16)x\n    f16 m = maximum(h, h)";
    let module = lowered(&under(half), A_O, Target { sm: 75 }, EXACT).unwrap();
    let lines = common::listing(&module);
    assert!(
        lines.iter().any(|line| line.starts_with("max.f32")),
        "{lines:?}"
    );
    assert!(
        !lines.iter().any(|line| line.starts_with("max.f16")),
        "{lines:?}"
    );
}

// --- the printer and the verifier, on lists built by hand -----------------------------------------------

fn reg(bank: Bank, number: u32) -> Reg {
    Reg { bank, number }
}

/// A module around `items`: one pointer parameter loaded into `%rd0`, with every bank four registers wide.
fn module(items: Vec<Item>, alignment: HashMap<Reg, u32>) -> Module {
    let load = Inst::new(
        Opcode::Ld(Space::Param),
        Ty::U64,
        vec![reg(Bank::B64, 0)],
        vec![Operand::Addr(Base::Param(0), 0)],
    );
    let ret = Inst::new(Opcode::Ret, Ty::Pred, Vec::new(), Vec::new());
    Module {
        version: (8, 7),
        target: "sm_120".into(),
        entry: "k".into(),
        params: vec![ParamDecl::Pointer { align: 4 }],
        shared: Vec::new(),
        max_threads: 64,
        single_wave: false,
        registers: Bank::ALL.iter().map(|bank| (*bank, 4)).collect(),
        items: [vec![Item::Inst(load)], items, vec![Item::Inst(ret)]].concat(),
        alignment,
    }
}

fn inst(opcode: Opcode, ty: Ty, result: Option<Reg>, operands: Vec<Operand>) -> Item {
    Item::Inst(Inst::new(
        opcode,
        ty,
        result.into_iter().collect(),
        operands,
    ))
}

fn branch(to: u32, pred: Reg) -> Item {
    let jump = Inst::new(
        Opcode::Bra,
        Ty::Pred,
        Vec::new(),
        vec![Operand::Label(Label(to))],
    );
    Item::Inst(jump.guarded(pred, false))
}

/// `%p0 = %f0 < 0`.
fn test_f0() -> Item {
    let operands = vec![reg(Bank::F32, 0).into(), Operand::f32(0.0)];
    inst(
        Opcode::Setp(Cmp::Lt),
        Ty::F32,
        Some(reg(Bank::Pred, 0)),
        operands,
    )
}

fn mov(to: Reg, from: Operand) -> Item {
    inst(Opcode::Mov, Ty::F32, Some(to), vec![from])
}

/// A list built by hand prints as ptxas reads it: a shared array and the external pool, an instruction under a
/// guard, and a vector access over a register tuple.
#[test]
fn the_printer_spells_shared_declarations_guards_and_tuples() {
    let (f, p, rd) = (Bank::F32, Bank::Pred, Bank::B64);
    let address = Operand::Addr(Base::Reg(reg(rd, 0)), 16);
    let pair = vec![reg(f, 0), reg(f, 1)];
    let load = Inst::new(
        Opcode::Ld(Space::Global),
        Ty::F32,
        pair.clone(),
        vec![address.clone()],
    );
    let store = Inst::new(
        Opcode::St(Space::Global),
        Ty::F32,
        Vec::new(),
        vec![address, Operand::Tuple(pair)],
    );
    let shared = Operand::Addr(Base::Shared(0), 8);
    let guarded = Inst::new(
        Opcode::St(Space::Shared),
        Ty::F32,
        Vec::new(),
        vec![shared, reg(f, 0).into()],
    );
    let items = vec![
        Item::Inst(load),
        test_f0(),
        Item::Inst(guarded.guarded(reg(p, 0), true)),
        Item::Inst(store),
    ];
    let mut module = module(items, HashMap::from([(reg(rd, 0), 16)]));
    module.shared = vec![
        Shared {
            align: 16,
            bytes: Some(64),
        },
        Shared {
            align: 16,
            bytes: None,
        },
    ];
    verify(&module).unwrap();
    let text = print(&module);
    for line in [
        ".shared .align 16 .b8 k_shared_0[64];\n",
        ".extern .shared .align 16 .b8 k_shared_1[];\n",
        "\tld.global.v2.f32 {%f0, %f1}, [%rd0+16];\n",
        "\t@!%p0 st.shared.f32 [k_shared_0+8], %f0;\n",
        "\tst.global.v2.f32 [%rd0+16], {%f0, %f1};\n",
    ] {
        assert!(text.contains(line), "{line}: {text}");
    }
    assemble(&text, "sm_120");
}

/// A register read on a path that does not define it is refused: one defined in one side of a condition only, one
/// defined under a guard only, one never defined.
#[test]
fn the_verifier_refuses_a_register_not_defined_on_every_path() {
    let (f, p) = (Bank::F32, Bank::Pred);
    let zero = Operand::f32(0.0);
    let one_side = vec![
        mov(reg(f, 0), zero.clone()),
        test_f0(),
        branch(1, reg(p, 0)),
        mov(reg(f, 1), zero.clone()),
        Item::Label(Label(1)),
        mov(reg(f, 2), reg(f, 1).into()),
    ];
    let message = verify(&module(one_side, HashMap::new()))
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("`mov.f32 %f2, %f1` reads %f1 on a path that does not define it"),
        "{message}"
    );
    let both_sides = vec![
        mov(reg(f, 0), zero.clone()),
        mov(reg(f, 1), zero.clone()),
        test_f0(),
        branch(1, reg(p, 0)),
        mov(reg(f, 1), reg(f, 0).into()),
        Item::Label(Label(1)),
        mov(reg(f, 2), reg(f, 1).into()),
    ];
    verify(&module(both_sides, HashMap::new())).unwrap();
    let guarded = Inst::new(Opcode::Mov, Ty::F32, vec![reg(f, 1)], vec![zero.clone()]);
    let under_guard = vec![
        mov(reg(f, 0), zero),
        test_f0(),
        Item::Inst(guarded.guarded(reg(p, 0), false)),
        mov(reg(f, 2), reg(f, 1).into()),
    ];
    assert!(verify(&module(under_guard, HashMap::new())).is_err());
    let never = vec![mov(reg(f, 2), reg(f, 3).into())];
    assert!(verify(&module(never, HashMap::new())).is_err());
}

/// A register a loop defines in its body and reads at the top of the next trip is not defined on the path that
/// enters the loop.
#[test]
fn the_verifier_follows_a_loop_back_to_its_head() {
    let (f, p) = (Bank::F32, Bank::Pred);
    let zero = Operand::f32(0.0);
    let looped = |seeded: bool| {
        let mut items = vec![mov(reg(f, 0), zero.clone())];
        if seeded {
            items.push(mov(reg(f, 1), zero.clone()));
        }
        let sum = vec![reg(f, 1).into(), reg(f, 0).into()];
        items.extend([
            Item::Label(Label(1)),
            inst(Opcode::Add, Ty::F32, Some(reg(f, 2)), sum),
            mov(reg(f, 1), reg(f, 2).into()),
            test_f0(),
            branch(1, reg(p, 0)),
        ]);
        module(items, HashMap::new())
    };
    verify(&looped(true)).unwrap();
    assert!(verify(&looped(false)).is_err());
}

/// A branch to a label nothing binds, and a label bound twice, are refused.
#[test]
fn the_verifier_refuses_a_label_not_bound_once() {
    let jump = inst(Opcode::Bra, Ty::Pred, None, vec![Operand::Label(Label(7))]);
    let message = verify(&module(vec![jump], HashMap::new()))
        .unwrap_err()
        .to_string();
    assert!(message.contains("the label L7 is never bound"), "{message}");
    let twice = vec![Item::Label(Label(1)), Item::Label(Label(1))];
    let message = verify(&module(twice, HashMap::new()))
        .unwrap_err()
        .to_string();
    assert!(message.contains("the label L1 is bound twice"), "{message}");
}

/// An access of several elements needs its address proven aligned to the whole of it, the constant offset
/// included: a load of two registers, and an async copy at both of its addresses.
#[test]
fn the_verifier_refuses_an_access_of_several_elements_with_no_alignment_fact() {
    let (f, r, rd) = (Bank::F32, Bank::B32, Bank::B64);
    let load = |offset: i64| {
        let address = Operand::Addr(Base::Reg(reg(rd, 0)), offset);
        let pair = vec![reg(f, 0), reg(f, 1)];
        let load = Inst::new(Opcode::LdConst, Ty::F32, pair, vec![address]);
        vec![Item::Inst(load)]
    };
    let copy = |offset: i64| {
        let to = Operand::Addr(Base::Reg(reg(r, 0)), 0);
        let from = Operand::Addr(Base::Reg(reg(rd, 0)), offset);
        let operands = vec![to, from, Operand::Int(8)];
        let copy = Inst::new(Opcode::CpAsync(Cache::All), Ty::B32, Vec::new(), operands);
        let symbol = vec![Operand::Shared(0)];
        vec![
            inst(Opcode::Mov, Ty::U32, Some(reg(r, 0)), symbol),
            Item::Inst(copy),
        ]
    };
    let aligned =
        |bytes: u32, shared: u32| HashMap::from([(reg(rd, 0), bytes), (reg(r, 0), shared)]);
    verify(&module(load(8), aligned(8, 1))).unwrap();
    verify(&module(copy(8), aligned(8, 8))).unwrap();
    for (items, proven) in [
        (load(8), HashMap::new()),
        (load(8), aligned(4, 1)),
        (load(4), aligned(8, 1)),
        (copy(8), aligned(8, 4)),
        (copy(8), aligned(4, 8)),
        (copy(4), aligned(8, 8)),
    ] {
        let message = verify(&module(items, proven)).unwrap_err().to_string();
        assert!(
            message.contains("is a 8-byte access with no alignment proven"),
            "{message}"
        );
    }
}

// --- what the pipeline's kernels hold --------------------------------------------------------------------

/// A comparison kept as a number is its truth value converted to a float: one where it holds and zero where it
/// does not; a selection among floats computes at its values' type, whatever its condition compares.
#[test]
fn a_truth_value_converted_to_a_float_selects_one_or_zero() {
    let got = under_grid(
        "
        f32 x = load a[i]
        bool same__bool = 0.0 == x
        f32 same = (f32)same__bool
        f32 kept = where(x == 0.0, same, x)
        f32 o[i] = kept
        ",
        A_O,
        EXACT,
    );
    assert_eq!(
        got[3..7],
        [
            "setp.eq.f32 %p0, 0f00000000, %f0",
            "selp.f32 %f1, 0f3F800000, 0f00000000, %p0",
            "setp.eq.f32 %p1, %f0, 0f00000000",
            "selp.f32 %f2, %f1, %f0, %p1",
        ]
    );
}

/// The 64-bit widening an index term carries where its buffer passes the 32-bit range is a conversion among
/// integers: the product is computed at 64 bits.
#[test]
fn an_index_term_widened_to_64_bits_is_an_integer() {
    let params: Params = &[
        ("a", Input, Some(F32)),
        ("o", Output, Some(F32)),
        ("n", Int, None),
    ];
    let body = format!("{GRID}\n    f32 x = load a[(long long)i * (long long)n]\n    f32 o[i] = x");
    let got = code(&body, params, EXACT);
    assert_eq!(
        got[4..9],
        [
            "cvt.s64.s32 %rd2, %r4",
            "cvt.s64.s32 %rd3, %r0",
            "mul.lo.s64 %rd4, %rd2, %rd3",
            "shl.b64 %rd5, %rd4, 2",
            "add.s64 %rd6, %rd0, %rd5",
        ]
    );
}

/// A condition the body's optimization decided is no branch: its one side stands in its place.
#[test]
fn a_decided_condition_is_its_one_side() {
    let got = under_grid(
        "
        f32 x = load a[i]
        if 1:
            f32 o[i] = x
        if 0:
            f32 o[i + 1] = x
        ",
        A_O,
        EXACT,
    );
    assert_eq!(
        got,
        [
            "mul.wide.s32 %rd2, %r3, 4",
            "add.s64 %rd3, %rd0, %rd2",
            "ld.global.nc.f32 %f0, [%rd3]",
            "add.s64 %rd4, %rd1, %rd2",
            "st.global.f32 [%rd4], %f0",
        ]
    );
}

/// A selection between two indices is a multiple of what both are, so a vector access at it keeps its alignment
/// proof: the verifier would refuse the four-wide load otherwise.
#[test]
fn a_selected_index_keeps_the_multiple_of_its_two_values() {
    let got = code(
        "
Grid(axes=(i in 0..1024), decode=(block_idx.x * 256 + thread_idx.x), block_threads=256):
    for k in 0..384 step 128:
        f32 x0, x1, x2, x3 = load a[(k + 256 < 384 ? k + 256 : 256) + 512 * i]
        f32 o[k + 512 * i] = x0
",
        A_O,
        EXACT,
    );
    assert_eq!(
        got[11..17],
        [
            "add.s32 %r6, %r5, 256",
            "setp.lt.s32 %p0, %r6, 384",
            "selp.s32 %r7, %r6, 256, %p0",
            "mul.wide.s32 %rd4, %r7, 4",
            "add.s64 %rd5, %rd3, %rd4",
            "ld.global.nc.v4.f32 {%f0, %f1, %f2, %f3}, [%rd5]",
        ]
    );
}

/// A store of a Volta copy precomputes where its element stands in the slab: the layout's index of a logical row and
/// column, computed once like any integer.
#[test]
fn a_volta_layout_index_is_its_integer_expression() {
    let params: Params = &[("o", Output, Some(I32))];
    for function in ["emmy_volta_crosswise", "emmy_volta_b_congruous"] {
        let got = under_grid(
            &format!("let i32 at = {function}(i / 4, 8 * (i % 4), 128)\ni32 o[i] = at"),
            params,
            EXACT,
        );
        assert!(
            got.iter().any(|line| line.starts_with("xor.b32")),
            "{function}: {got:#?}"
        );
    }
}

/// A 16-bit integer is held extended in a 32-bit register, by its sign or by zeros, and a 64-bit unsigned one shifts
/// right by zeros; a bitcast moves the bits between the 16-bit integers and a half through the low half.
#[test]
fn sixteen_bit_and_unsigned_64_bit_integers_keep_their_bits() {
    let params: Params = &[
        ("a", Input, Some(I16)),
        ("o", Output, Some(U16)),
        ("h", Output, Some(F16)),
    ];
    let got = under_grid(
        "i16 x = load a[i]
u16 y = bitcast(x)
u32 z = (u32)y
u64 wide = (u64)z
u64 up = wide << 47
u64 down = up >> 40
u32 low = (u32)down
u16 code = (u16)low
f16 half = bitcast(code)
u16 o[i] = code
f16 h[i] = half",
        params,
        EXACT,
    );
    let has = |prefix: &str| got.iter().any(|line| line.starts_with(prefix));
    assert!(has("ld.global.nc.s16"), "{got:#?}");
    assert!(
        has("and.b32") && has("shr.u64") && has("cvt.u16.u32"),
        "{got:#?}"
    );
    assert!(has("st.global.u16") && has("st.global.b16"), "{got:#?}");
    assert!(!has("shr.s64"), "{got:#?}");
}

/// A double is a value converted through and back: exactly from a half or an f32, rounded to nearest into an f32.
#[test]
fn a_double_converts_exactly_out_and_rounds_back() {
    let params: Params = &[("a", Input, Some(F16)), ("o", Output, Some(F32))];
    let got = under_grid(
        "f16 x = load a[i]\nf64 wide = (f64)x\nf32 back = (f32)wide\nf32 o[i] = back",
        params,
        EXACT,
    );
    let has = |prefix: &str| got.iter().any(|line| line.starts_with(prefix));
    assert!(has("cvt.f64.f32") && has("cvt.rn.f32.f64"), "{got:#?}");
}
