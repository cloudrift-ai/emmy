//! The math functions: the sequence each is written as, and that ptxas assembles it at `-O0` and at its default
//! level with the flush-to-zero decision off and on. What a sequence computes is measured on a card, by the sweeps
//! of `tests/compiler/ptx/test_math.py`.

mod common;

use emmy_ir::dtype::DType::{F16, F32};
use emmy_ptx::ParamKind::{Input, Output};
use emmy_ptx::{Precision, Target};

use common::{A_O, EXACT, Params, code, lowered, refusal};

const GRID: &str =
    "Grid(axes=(i in 0..4096), decode=(block_idx.x * 256 + thread_idx.x), block_threads=256):";

const FLUSHING: Precision = Precision {
    fuse: false,
    ftz: true,
};

/// Two f32 inputs and one f32 output.
const A_B_O: Params = &[
    ("a", Input, Some(F32)),
    ("b", Input, Some(F32)),
    ("o", Output, Some(F32)),
];

/// The kernel that stores `function` of each element of `a`, and of `b` beside it where the function takes two.
fn kernel(function: &str, operands: usize) -> String {
    let second = if operands == 2 {
        "\n    f32 y = load b[i]"
    } else {
        ""
    };
    let args = if operands == 2 { "x, y" } else { "x" };
    format!(
        "{GRID}\n    f32 x = load a[i]{second}\n    f32 v = {function}({args})\n    f32 o[i] = v"
    )
}

/// The instructions `function` is written as: what stands between the load of its operands and the store.
fn sequence(function: &str, operands: usize, precision: Precision) -> Vec<String> {
    let params = if operands == 2 { A_B_O } else { A_O };
    let all = code(&kernel(function, operands), params, precision);
    let first = all
        .iter()
        .rposition(|line| line.starts_with("ld."))
        .unwrap()
        + 1;
    let last = all.iter().position(|line| line.starts_with("st.")).unwrap();
    // The store's address is formed ahead of it.
    all[first..last]
        .iter()
        .filter(|line| !line.contains("%rd"))
        .cloned()
        .collect()
}

/// The exponential: the floor of the base-two exponent held to a normal float's, the remainder of the product
/// taken with the rest of the constant, the hardware's power of two in its flushing form, and the scale through
/// its bits.
#[test]
fn the_exponential_reduces_to_the_hardware_power_of_two() {
    assert_eq!(
        sequence("exp", 1, EXACT),
        [
            "mul.rn.f32 %f1, %f0, 0f3FB8AA3B",
            "cvt.rmi.f32.f32 %f2, %f1",
            "max.f32 %f3, %f2, 0fC2FC0000",
            "min.f32 %f4, %f3, 0f42FE0000",
            "neg.f32 %f5, %f4",
            "fma.rn.f32 %f6, %f0, 0f3FB8AA3B, %f5",
            "fma.rn.f32 %f7, %f0, 0f32A57060, %f6",
            "ex2.approx.ftz.f32 %f8, %f7",
            "cvt.rzi.s32.f32 %r4, %f4",
            "add.s32 %r5, %r4, 127",
            "shl.b32 %r6, %r5, 23",
            "mov.b32 %f9, %r6",
            "mul.rn.f32 %f10, %f8, %f9",
        ]
    );
}

/// The logarithm: the exponent and a mantissa in `[2/3, 4/3)` through the float's bits, a polynomial of the
/// mantissa less one, and the hardware logarithm's value for the operands that have no finite one.
#[test]
fn the_logarithm_splits_the_exponent_off_through_the_bits() {
    let got = sequence("log", 1, EXACT);
    assert_eq!(
        got[..12],
        [
            "setp.lt.f32 %p0, %f0, 0f00800000",
            "mul.rn.f32 %f1, %f0, 0f4B000000",
            "selp.f32 %f2, %f1, %f0, %p0",
            "selp.b32 %r4, 1252698795, 1059760811, %p0",
            "mov.b32 %r5, %f2",
            "sub.s32 %r6, %r5, %r4",
            "and.b32 %r7, %r6, -8388608",
            "sub.s32 %r8, %r6, %r7",
            "add.s32 %r9, %r8, 1059760811",
            "mov.b32 %f3, %r9",
            "cvt.rn.f32.s32 %f4, %r7",
            "add.rn.f32 %f5, %f3, 0fBF800000",
        ]
    );
    assert_eq!(
        got[got.len() - 5..],
        [
            "fma.rn.f32 %f16, %f4, 0f33B17218, %f15",
            "sub.s32 %r10, %r5, 1",
            "setp.ge.u32 %p1, %r10, 2139095039",
            "lg2.approx.ftz.f32 %f17, %f2",
            "selp.f32 %f18, %f17, %f16, %p1",
        ]
    );
    assert_eq!(got.len(), 27);
}

/// The hyperbolic tangent and the error function: a polynomial below the split, the hardware's power of two from
/// it on, and one select between them.
#[test]
fn the_hyperbolic_tangent_and_the_error_function_select_between_two_forms() {
    let tanh = sequence("tanh", 1, EXACT);
    assert_eq!(
        tanh[..7],
        [
            "abs.f32 %f1, %f0",
            "mul.rn.f32 %f2, %f1, 0f4038AA3B",
            "ex2.approx.ftz.f32 %f3, %f2",
            "add.rn.f32 %f4, %f3, 0f3F800000",
            "rcp.approx.ftz.f32 %f5, %f4",
            "fma.rn.f32 %f6, %f5, 0fC0000000, 0f3F800000",
            "copysign.f32 %f7, %f0, %f6",
        ]
    );
    assert_eq!(
        tanh[tanh.len() - 2..],
        [
            "setp.ge.f32 %p0, %f1, 0f3F800000",
            "selp.f32 %f17, %f7, %f16, %p0"
        ]
    );
    assert_eq!(tanh.len(), 18);
    // The series times the square has a zero added, which keeps the sign of a zero operand.
    assert_eq!(tanh[14], "fma.rn.f32 %f15, %f14, %f8, 0f00000000");
    let erf = sequence("erf", 1, EXACT);
    assert_eq!(erf.len(), 21);
    let count = |mnemonic: &str| erf.iter().filter(|line| line.starts_with(mnemonic)).count();
    // Seven steps of the exponent's polynomial, six of the small form's and the one that closes it.
    assert_eq!((count("fma.rn.f32"), count("ex2.approx.ftz.f32")), (14, 1));
}

/// The power: the logarithm of the base as a pair, through the hardware's reciprocal with what it leaves computed
/// back; the product with the exponent as a pair; a polynomial of its fraction with the scale in two factors; and
/// the selects for what the operands' kinds decide.
#[test]
fn the_power_holds_its_logarithm_and_its_product_as_pairs() {
    let got = sequence("pow", 2, EXACT);
    let count = |mnemonic: &str| got.iter().filter(|line| line.starts_with(mnemonic)).count();
    assert_eq!(got.len(), 103);
    // The one hardware approximation is the reciprocal, whose error the residual takes back out.
    assert_eq!(
        (count("rcp.approx.ftz"), count("ex2"), count("lg2")),
        (1, 0, 0)
    );
    let quotient = got.iter().position(|line| line.starts_with("rcp")).unwrap();
    assert_eq!(
        got[quotient - 2..quotient + 8],
        [
            "add.rn.f32 %f8, %f5, 0fBF800000",
            "add.rn.f32 %f9, %f5, 0f3F800000",
            "rcp.approx.ftz.f32 %f10, %f9",
            "add.rn.f32 %f11, %f8, %f8",
            "mul.rn.f32 %f12, %f11, %f10",
            "sub.rn.f32 %f13, %f8, %f12",
            "add.rn.f32 %f14, %f13, %f13",
            "neg.f32 %f15, %f12",
            "fma.rn.f32 %f16, %f15, %f8, %f14",
            "mul.rn.f32 %f17, %f16, %f10",
        ]
    );
    // A NaN operand or a negative finite base under a fractional exponent is a NaN, and a zero exponent or a
    // base of one is one whatever the other operand is: the last select wins.
    let mnemonics: Vec<&str> = got[got.len() - 13..]
        .iter()
        .map(|line| line.split(' ').next().unwrap())
        .collect();
    assert_eq!(
        mnemonics,
        [
            "setp.nan.f32",
            "cvt.rmi.f32.f32",
            "setp.neu.f32",
            "setp.lt.f32",
            "setp.gt.f32",
            "and.pred",
            "and.pred",
            "or.pred",
            "selp.f32",
            "setp.eq.f32",
            "setp.eq.f32",
            "or.pred",
            "selp.f32",
        ]
    );
    assert!(got[got.len() - 5].contains("0f7FC00000") && got[got.len() - 1].contains("0f3F800000"));
    assert_eq!(sequence("power", 2, EXACT), got);
    assert_eq!(sequence("pow", 2, FLUSHING).len(), got.len());
}

/// The reciprocal square root is the hardware's, as `rsqrtf` is, and flushes when the kernel's decision says. So
/// does every instruction of a sequence that takes the modifier.
#[test]
fn the_flush_to_zero_decision_reaches_every_instruction_that_takes_it() {
    assert_eq!(sequence("rsqrt", 1, EXACT), ["rsqrt.approx.f32 %f1, %f0"]);
    assert_eq!(
        sequence("rsqrt", 1, FLUSHING),
        ["rsqrt.approx.ftz.f32 %f1, %f0"]
    );
    let exp = sequence("exp", 1, FLUSHING);
    assert_eq!(
        exp[..5],
        [
            "mul.rn.ftz.f32 %f1, %f0, 0f3FB8AA3B",
            "cvt.rmi.ftz.f32.f32 %f2, %f1",
            "max.ftz.f32 %f3, %f2, 0fC2FC0000",
            "min.ftz.f32 %f4, %f3, 0f42FE0000",
            "neg.ftz.f32 %f5, %f4",
        ]
    );
    for function in ["log", "tanh", "erf"] {
        let flushing = sequence(function, 1, FLUSHING);
        let takes = |line: &&String| {
            ["mul", "add", "sub", "fma", "abs", "setp"]
                .iter()
                .any(|mnemonic| line.starts_with(mnemonic) && line.contains(".f32 "))
        };
        assert!(
            flushing
                .iter()
                .filter(takes)
                .all(|line| line.contains(".ftz.")),
            "{flushing:?}"
        );
    }
}

/// The fast exponential is the hardware's power of two of the scaled operand. With the flush-to-zero decision off
/// it is the form that keeps subnormals, which ptxas writes by halving the operand and squaring the result.
#[test]
fn the_fast_exponential_is_one_hardware_instruction_on_a_scaled_operand() {
    assert_eq!(
        sequence("exp_fast", 1, EXACT),
        ["mul.rn.f32 %f1, %f0, 0f3FB8AA3B", "ex2.approx.f32 %f2, %f1"]
    );
    assert_eq!(
        sequence("exp_fast", 1, FLUSHING),
        [
            "mul.rn.ftz.f32 %f1, %f0, 0f3FB8AA3B",
            "ex2.approx.ftz.f32 %f2, %f1"
        ]
    );
}

/// Each approximate operation is the hardware instruction of its row, under the name the row gives it.
#[test]
fn an_approximate_operation_is_the_hardware_instruction_of_its_row() {
    let one = |function: &str| sequence(function, 1, FLUSHING);
    assert_eq!(
        one("log_fast"),
        [
            "lg2.approx.ftz.f32 %f1, %f0",
            "mul.rn.ftz.f32 %f2, %f1, 0f3F317218"
        ]
    );
    assert_eq!(one("sin_fast"), ["sin.approx.ftz.f32 %f1, %f0"]);
    assert_eq!(one("cos_fast"), ["cos.approx.ftz.f32 %f1, %f0"]);
    assert_eq!(one("sqrt_fast"), ["sqrt.approx.ftz.f32 %f1, %f0"]);
    assert_eq!(one("rsqrt_fast"), ["rsqrt.approx.ftz.f32 %f1, %f0"]);
    assert_eq!(one("reciprocal_fast"), ["rcp.approx.ftz.f32 %f1, %f0"]);
    // The hardware's hyperbolic tangent has no flushing form.
    assert_eq!(one("tanh_fast"), ["tanh.approx.f32 %f1, %f0"]);
    assert_eq!(
        sequence("divide_fast", 2, FLUSHING),
        ["div.approx.ftz.f32 %f2, %f0, %f1"]
    );
    assert_eq!(
        sequence("pow_fast", 2, EXACT),
        [
            "lg2.approx.f32 %f2, %f0",
            "mul.rn.f32 %f3, %f1, %f2",
            "ex2.approx.f32 %f4, %f3"
        ]
    );
    let turing = lowered(&kernel("tanh_fast", 1), A_O, Target { sm: 75 }, EXACT);
    assert!(turing.is_ok());
    let volta = lowered(&kernel("tanh_fast", 1), A_O, Target { sm: 70 }, EXACT);
    let message = volta.unwrap_err().to_string();
    assert!(
        message.contains("no PTX rule for tanh_fast on sm_70"),
        "{message}"
    );
}

/// Two calls of one function on one operand are one sequence: the value table shares every instruction.
#[test]
fn a_function_computed_twice_is_written_once() {
    let body = format!(
        "{GRID}\n    f32 x = load a[i]\n    f32 v = exp(x)\n    f32 w = exp(x)\n    f32 s = v + w\n    f32 o[i] = s"
    );
    let got = code(&body, A_O, EXACT);
    let count = |mnemonic: &str| got.iter().filter(|line| line.starts_with(mnemonic)).count();
    assert_eq!((count("ex2.approx"), count("cvt.rmi")), (1, 1));
}

/// A half value has no sequence, and a function no kernel holds has no rule: each is refused by its name.
#[test]
fn a_half_operand_and_an_unwritten_function_are_refused_by_name() {
    let halves: Params = &[("a", Input, Some(F16)), ("o", Output, Some(F16))];
    let body = format!("{GRID}\n    f16 x = load a[i]\n    f16 v = exp(x)\n    f16 o[i] = v");
    let message = refusal(&body, halves);
    assert!(
        message.contains("exp at f16: a math function computes in f32"),
        "{message}"
    );
    let body = format!("{GRID}\n    f16 x = load a[i]\n    f16 v = log_fast(x)\n    f16 o[i] = v");
    let message = refusal(&body, halves);
    assert!(message.contains("log_fast at f16"), "{message}");
    for function in ["sin", "cos", "sigmoid", "log2", "expm1"] {
        let message = refusal(&kernel(function, 1), A_O);
        let expected = format!("no PTX rule for the operation {function} at f32");
        assert!(message.contains(&expected), "{message}");
    }
    let message = refusal(&kernel("cbrt_fast", 1), A_O);
    assert!(
        message.contains("no PTX rule for the operation cbrt_fast"),
        "{message}"
    );
}
