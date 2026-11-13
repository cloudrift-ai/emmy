//! The math functions. Each is a sequence of f32 instructions written through the builder, so the value table shares
//! what two of them compute alike. Every instruction that rounds carries its mode, a fused multiply-add stands only
//! where a sequence writes one, and nothing here reads the kernel's fusion decision: a sequence's accuracy is
//! measured with the instructions it has. A half value has no sequence. A math function on one computes in f32 and
//! converts once, which the body says with its casts.
//!
//! The accurate functions aim at the bound NVIDIA's CUDA math documentation states for the same function, and give
//! the special values the CUDA function gives. The polynomial coefficients are minimax fits made for these
//! sequences; the reductions are the textbook ones (an exponent split off through the float's bits as fdlibm and
//! musl do for the logarithm, a base-two exponent for the exponential). `ARCHITECTURE.md` holds the measured error
//! of each.

use emmy_ir::Result;
use emmy_ir::elementwise::Operation;
use emmy_ir::expr::{Expression, SimplifyCtx};

use super::{Lowerer, SM_TURING, VTy};
use crate::inst::{Bank, Cmp, Inst, Opcode, Operand, Reg, Round, Ty};

/// What an approximate operation computes: one hardware instruction, on its operands as they stand or scaled.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) enum Fast {
    /// The instruction alone.
    Inst(Opcode),
    /// The instruction on the operand times a constant: `e^x` is `2^(x * log2(e))`.
    OfScaled(Opcode, f32),
    /// The instruction's result times a constant: `ln(x)` is `log2(x) * ln(2)`.
    Scaled(Opcode, f32),
    /// `x^y` as `2^(y * log2(x))`.
    Power,
}

/// The approximate operations: what `--use_fast_math` and the fast intrinsics gave a CUDA kernel, each the hardware
/// instruction of its row. An instruction flushes to zero when the kernel's decision says. The reciprocal square
/// root has no row: `rsqrt` is the hardware's instruction already. `sigmoid_fast` has none either: it is a
/// composite, compiled through its definition in the operations table.
const FAST: [(Operation, Fast); 9] = [
    (Operation::ExpFast, Fast::OfScaled(Opcode::Ex2, LOG2_E)),
    (
        Operation::LogFast,
        Fast::Scaled(Opcode::Lg2, std::f32::consts::LN_2),
    ),
    (Operation::SinFast, Fast::Inst(Opcode::Sin)),
    (Operation::CosFast, Fast::Inst(Opcode::Cos)),
    (Operation::TanhFast, Fast::Inst(Opcode::Tanh)),
    (Operation::PowFast, Fast::Power),
    (Operation::DivideFast, Fast::Inst(Opcode::Div)),
    (Operation::ReciprocalFast, Fast::Inst(Opcode::Rcp)),
    (Operation::SqrtFast, Fast::Inst(Opcode::Sqrt)),
];

/// The row of the approximate operation `op`, when it is one.
pub(super) fn fast(op: Operation) -> Option<Fast> {
    FAST.iter()
        .find(|(operation, _)| *operation == op)
        .map(|(_, row)| *row)
}

/// `log2(e)` as the nearest f32, and what that leaves of it.
const LOG2_E: f32 = std::f32::consts::LOG2_E;
const LOG2_E_REST: f32 = (std::f64::consts::LOG2_E - LOG2_E as f64) as f32;
/// `log2(e) / 252`, one step below the nearest f32, as cicc writes it.
const EXP_COARSE: f32 = f32::from_bits(0x3bbb_989d);
/// `1.5 * 2^23 + 1`: a float whose low bits hold an integer added to it, plus one, which with the unbias below is
/// the exponent's bias of 127 less the 126 steps the clamped product starts below zero.
const EXP_MAGIC: f32 = 12582913.0;
/// Takes the magic constant and the bias back out: `-(1.5 * 2^23 + 127)`.
const EXP_UNBIAS: f32 = -12583039.0;

/// The smallest normal f32.
const MIN_NORMAL: f32 = f32::MIN_POSITIVE;

/// The bits of `2/3`: the logarithm takes its mantissa from `[2/3, 4/3)`, so the value it adds to the exponent's
/// share is near zero only where that share is zero.
const TWO_THIRDS_BITS: i64 = 0x3f2a_aaab;

/// The bits of the square root of a half: the power takes its mantissa from there to the square root of two, the
/// range that is even in the variable its logarithm is a series of.
const ROOT_HALF_BITS: i64 = 0x3f35_04f3;

/// The exponent field of an f32, where it sits, and what takes a value out of it.
const EXPONENT_MASK: i64 = !0x007f_ffff;
const EXPONENT_UNIT: f32 = 1.0 / 8388608.0;

/// `ln(1 + f) = f - f^2/2 + f^3 * P(f)` over `|f| <= 1/3`, `P` by ascending power, as every table here is.
const LOG: [f32; 8] = [
    3.3333215e-1,
    -2.4999797e-1,
    2.0010406e-1,
    -1.6680487e-1,
    1.4005862e-1,
    -1.2184278e-1,
    1.3984467e-1,
    -1.2891601e-1,
];

/// Where the hyperbolic tangent and the error function change from their polynomial to their exponential form.
const TANH_SPLIT: f32 = 1.0;
const ERF_SPLIT: f32 = 1.0;

/// `tanh(x) = x + x * s * P(s)` with `s = x^2`, over `|x| <= 1`.
const TANH: [f32; 7] = [
    -3.3333296e-1,
    1.3332345e-1,
    -5.38798e-2,
    2.1486651e-2,
    -7.946096e-3,
    2.3013577e-3,
    -3.5845002e-4,
];

/// `erf(x) = x + x * P(s)` with `s = x^2`, over `|x| <= 1`.
const ERF: [f32; 7] = [
    1.2837917e-1,
    -3.7612626e-1,
    1.12835854e-1,
    -2.6853811e-2,
    5.1883277e-3,
    -8.0101925e-4,
    7.853858e-5,
];

/// `erf(a) = 1 - 2^G(a)` over `a >= 1`: `G` is `log2(erfc(a))`, fit with the error weighted by what it moves of
/// `erf(a)`, so it is tight at 1 and loose where the complement is small. Its leading coefficient is negative, so
/// past the fit `G` keeps falling and the result is 1.
const ERFC_LOG2: [f32; 8] = [
    2.712812e-4,
    -1.6296371e0,
    -9.135432e-1,
    -1.5641491e-1,
    3.6367808e-2,
    -6.0658944e-3,
    6.367126e-4,
    -3.1287414e-5,
];

/// `log2(m) = (u + u * z * Q(z)) * log2(e)` with `u = 2f / (2 + f)`, `f = m - 1` and `z = u^2`, over a mantissa
/// from the square root of a half to the square root of two: the series of `2 * atanh(u / 2)` past its first term.
/// The table is `Q * log2(e)`.
const ATANH_LOG2: [f32; 4] = [1.2022459e-1, 1.8033713e-2, 3.218181e-3, 6.5721624e-4];

/// `2^g = 1 + g * P(g)` over `|g| <= 1/2`.
const EXP2: [f32; 6] = [
    6.931472e-1,
    2.4022646e-1,
    5.550352e-2,
    9.6186325e-3,
    1.339523e-3,
    1.5306543e-4,
];

/// An exponent's magnitude is held to this: past it every power of a base other than one is zero or infinite.
const EXPONENT_LIMIT: f32 = 1.2676506e30;

/// A product of exponent and logarithm is held to this: past it the power is zero or infinite.
const PRODUCT_LIMIT: f32 = 160.0;

/// A value of a sequence: a register or a constant.
#[derive(Clone, Copy, Debug)]
enum V {
    Reg(Reg),
    F32(f32),
    Int(i64),
}

impl From<f32> for V {
    fn from(value: f32) -> V {
        V::F32(value)
    }
}

impl From<i64> for V {
    fn from(value: i64) -> V {
        V::Int(value)
    }
}

impl From<V> for Operand {
    fn from(value: V) -> Operand {
        match value {
            V::Reg(reg) => reg.into(),
            V::F32(value) => Operand::f32(value),
            V::Int(value) => Operand::Int(value),
        }
    }
}

/// A sequence being written. Every instruction goes through the builder's value table.
struct Seq<'a>(&'a mut Lowerer);

impl Seq<'_> {
    fn value(&mut self, inst: Inst, bank: Bank) -> V {
        V::Reg(self.0.b.value(inst, bank))
    }

    /// An f32 instruction as every float rule writes one: rounded to nearest where it rounds, flushing to zero
    /// where the kernel's decision says.
    fn float<const N: usize>(&mut self, opcode: Opcode, operands: [V; N]) -> V {
        let operands = operands.into_iter().map(Operand::from).collect();
        let inst = self.0.float_inst(opcode, VTy::F32, operands);
        self.value(inst, Bank::F32)
    }

    fn add(&mut self, a: impl Into<V>, b: impl Into<V>) -> V {
        self.float(Opcode::Add, [a.into(), b.into()])
    }

    fn sub(&mut self, a: impl Into<V>, b: impl Into<V>) -> V {
        self.float(Opcode::Sub, [a.into(), b.into()])
    }

    fn mul(&mut self, a: impl Into<V>, b: impl Into<V>) -> V {
        self.float(Opcode::Mul, [a.into(), b.into()])
    }

    fn fma(&mut self, a: impl Into<V>, b: impl Into<V>, c: impl Into<V>) -> V {
        self.float(Opcode::Fma, [a.into(), b.into(), c.into()])
    }

    fn min(&mut self, a: impl Into<V>, b: impl Into<V>) -> V {
        self.float(Opcode::Min, [a.into(), b.into()])
    }

    fn max(&mut self, a: impl Into<V>, b: impl Into<V>) -> V {
        self.float(Opcode::Max, [a.into(), b.into()])
    }

    fn neg(&mut self, a: V) -> V {
        self.float(Opcode::Neg, [a])
    }

    fn abs(&mut self, a: V) -> V {
        self.float(Opcode::Abs, [a])
    }

    /// `magnitude` with the sign of `sign`.
    fn copysign(&mut self, sign: V, magnitude: V) -> V {
        let operands = vec![sign.into(), magnitude.into()];
        let inst = Inst::new(Opcode::Copysign, Ty::F32, Vec::new(), operands);
        self.value(inst, Bank::F32)
    }

    /// `a * b + c` with the result clamped to `[0, 1]`.
    fn saturated_fma(&mut self, a: V, b: impl Into<V>, c: impl Into<V>) -> V {
        let operands = vec![a.into(), b.into().into(), c.into().into()];
        let inst = self
            .0
            .float_inst(Opcode::Fma, VTy::F32, operands)
            .saturating();
        self.value(inst, Bank::F32)
    }

    /// `a * b + c` rounded towards minus infinity.
    fn fma_down(&mut self, a: V, b: impl Into<V>, c: impl Into<V>) -> V {
        let operands = vec![a.into(), b.into().into(), c.into().into()];
        let inst = self
            .0
            .float_inst(Opcode::Fma, VTy::F32, operands)
            .rounded(Round::Rm);
        self.value(inst, Bank::F32)
    }

    /// The hardware's approximation of `opcode`, flushing to zero when `ftz` says.
    fn approx<const N: usize>(&mut self, opcode: Opcode, operands: [V; N], ftz: bool) -> V {
        let operands = operands.into_iter().map(Operand::from).collect();
        let inst = Inst::new(opcode, Ty::F32, Vec::new(), operands)
            .rounded(Round::Approx)
            .flushing(ftz);
        self.value(inst, Bank::F32)
    }

    /// The hardware's approximation inside an accurate sequence, in its flushing form: no sequence hands one a
    /// subnormal whose value matters or reads a subnormal from one, and ptxas writes three more instructions
    /// around the form that keeps them.
    fn hardware(&mut self, opcode: Opcode, a: V) -> V {
        self.approx(opcode, [a], true)
    }

    /// `a` rounded to an integral value, still a float.
    fn integral(&mut self, a: V, round: Round) -> V {
        let inst = Inst::new(Opcode::Cvt(Ty::F32), Ty::F32, Vec::new(), vec![a.into()])
            .rounded(round)
            .flushing(self.0.precision.ftz);
        self.value(inst, Bank::F32)
    }

    /// The integer an integral float is.
    fn truncated(&mut self, a: V) -> V {
        let inst = Inst::new(Opcode::Cvt(Ty::F32), Ty::S32, Vec::new(), vec![a.into()])
            .rounded(Round::Rzi)
            .flushing(self.0.precision.ftz);
        self.value(inst, Bank::B32)
    }

    fn converted(&mut self, a: V) -> V {
        let inst =
            Inst::new(Opcode::Cvt(Ty::S32), Ty::F32, Vec::new(), vec![a.into()]).rounded(Round::Rn);
        self.value(inst, Bank::F32)
    }

    /// The 32 bits of a float as an integer.
    fn bits(&mut self, a: V) -> V {
        let inst = Inst::new(Opcode::Mov, Ty::B32, Vec::new(), vec![a.into()]);
        self.value(inst, Bank::B32)
    }

    /// The float an integer holds the bits of.
    fn of_bits(&mut self, a: V) -> V {
        let inst = Inst::new(Opcode::Mov, Ty::B32, Vec::new(), vec![a.into()]);
        self.value(inst, Bank::F32)
    }

    fn int(&mut self, opcode: Opcode, a: V, b: impl Into<V>) -> V {
        let ty = match opcode {
            Opcode::Add | Opcode::Sub | Opcode::Shr => Ty::S32,
            _ => Ty::B32,
        };
        let inst = Inst::new(opcode, ty, Vec::new(), vec![a.into(), b.into().into()]);
        self.value(inst, Bank::B32)
    }

    /// A comparison of two floats, or at `ty` of two integers.
    fn compare(&mut self, cmp: Cmp, ty: Ty, a: V, b: impl Into<V>) -> V {
        let operands = vec![a.into(), b.into().into()];
        let inst = Inst::new(Opcode::Setp(cmp), ty, Vec::new(), operands)
            .flushing(self.0.precision.ftz && ty == Ty::F32);
        self.value(inst, Bank::Pred)
    }

    /// A logical operation over two predicates.
    fn logic(&mut self, opcode: Opcode, a: V, b: V) -> V {
        let inst = Inst::new(opcode, Ty::Pred, Vec::new(), vec![a.into(), b.into()]);
        self.value(inst, Bank::Pred)
    }

    /// `a` where `pred` holds, else `b`, at `ty`.
    fn select(&mut self, ty: Ty, a: impl Into<V>, b: impl Into<V>, pred: V) -> V {
        let operands = vec![a.into().into(), b.into().into(), pred.into()];
        let inst = Inst::new(Opcode::Selp, ty, Vec::new(), operands);
        self.value(inst, ty.bank())
    }

    /// The polynomial of `coefficients`, by ascending power, at `x`: one fused multiply-add a coefficient.
    fn horner(&mut self, coefficients: &[f32], x: V) -> V {
        let (last, rest) = coefficients.split_last().expect("a polynomial has a term");
        rest.iter()
            .rev()
            .fold(V::F32(*last), |sum, c| self.fma(sum, x, *c))
    }

    /// `2^k` for an integer `k` a normal float has as its exponent, through its bits.
    fn power_of_two(&mut self, k: V) -> V {
        let biased = self.int(Opcode::Add, k, 127);
        let bits = self.int(Opcode::Shl, biased, 23);
        self.of_bits(bits)
    }

    /// A positive float as its exponent and a mantissa in `[fold, 2 * fold)`, `fold` given by its bits. A
    /// subnormal is scaled into the normals first, and the offset takes the scaling back out of the exponent. The
    /// bits less those of `fold` hold the exponent in their exponent field; taking that field back out leaves the
    /// mantissa. Returns the float as it was split, its bits, the exponent still in the exponent's place, and the
    /// mantissa.
    fn split(&mut self, x: V, fold: i64) -> (V, V, V, V) {
        let subnormal = self.compare(Cmp::Lt, Ty::F32, x, MIN_NORMAL);
        let scaled = self.mul(x, 8388608.0);
        let x = self.select(Ty::F32, scaled, x, subnormal);
        let offset = self.select(Ty::B32, fold + (23 << 23), fold, subnormal);
        let bits = self.bits(x);
        let shifted = self.int(Opcode::Sub, bits, offset);
        let exponent = self.int(Opcode::And, shifted, EXPONENT_MASK);
        let mantissa = self.int(Opcode::Sub, shifted, exponent);
        let mantissa = self.int(Opcode::Add, mantissa, fold);
        let mantissa = self.of_bits(mantissa);
        (x, bits, exponent, mantissa)
    }

    /// Whether the float of `bits` is anything but positive and finite: zero, a negative, an infinity or a NaN,
    /// by one unsigned comparison.
    fn outside_the_positive(&mut self, bits: V) -> V {
        let below = self.int(Opcode::Sub, bits, 1);
        self.compare(Cmp::Ge, Ty::U32, below, 0x7f7f_ffff_i64)
    }

    /// `e^x`, by the sequence cicc writes for CUDA's `expf`. The exponent `k` is the floor of `x * log2(e)`, taken
    /// as `x * log2(e) / 252 + 1/2` clamped to `[0, 1]` and scaled back by 252, so it is held to the exponents a
    /// normal f32 has without a comparison; its integer is formed in the low bits of a float by rounding down into a
    /// magic constant, which also holds the bias, so `2^k` is those bits shifted into the exponent. What is left of the product, taken
    /// with the rest of the constant, goes through the hardware's power of two, which carries the result past
    /// either end of `k` to infinity, through the subnormals, and to zero. A NaN clamps to zero and comes back
    /// through the fused multiply-adds.
    fn exp(&mut self, x: V) -> V {
        let coarse = self.saturated_fma(x, EXP_COARSE, 0.5);
        let k = self.fma_down(coarse, 252.0, EXP_MAGIC);
        let unbiased = self.add(k, EXP_UNBIAS);
        let minus_k = self.neg(unbiased);
        let fraction = self.fma(x, LOG2_E, minus_k);
        let fraction = self.fma(x, LOG2_E_REST, fraction);
        let bits = self.bits(k);
        let scale = self.int(Opcode::Shl, bits, 23);
        let scale = self.of_bits(scale);
        let power = self.hardware(Opcode::Ex2, fraction);
        self.mul(power, scale)
    }

    /// `ln(x)`. With the exponent `k` and a mantissa `m` in `[2/3, 4/3)`, and `f = m - 1` exact, the result is
    /// `k * ln(2) + ln(1 + f)`: over that range `ln(m)` is near zero only where `k` is zero. Zero, a negative, an
    /// infinity and a NaN take the hardware logarithm's value, which is the right one for each.
    fn log(&mut self, x: V) -> V {
        let (x, bits, exponent, m) = self.split(x, TWO_THIRDS_BITS);
        let k = self.converted(exponent);
        let f = self.add(m, -1.0);
        let series = self.horner(&LOG, f);
        let series = self.fma(series, f, -0.5);
        let series = self.mul(series, f);
        let log_m = self.fma(series, f, f);
        // `k` is still in the exponent's place: the constant takes it out.
        let finite = self.fma(k, std::f32::consts::LN_2 * EXPONENT_UNIT, log_m);
        let special = self.outside_the_positive(bits);
        let hardware = self.hardware(Opcode::Lg2, x);
        self.select(Ty::F32, hardware, finite, special)
    }

    /// `tanh(x)`: an odd polynomial below the split, and `1 - 2 / (e^(2|x|) + 1)` with the sign of `x` from it
    /// on, through the hardware's power of two and reciprocal. Past 9 the quotient is below half a step of 1.
    /// A NaN fails the comparison and comes back through the polynomial.
    fn tanh(&mut self, x: V) -> V {
        let magnitude = self.abs(x);
        let doubled = self.mul(magnitude, 2.0 * LOG2_E);
        let power = self.hardware(Opcode::Ex2, doubled);
        let denominator = self.add(power, 1.0);
        let reciprocal = self.hardware(Opcode::Rcp, denominator);
        let large = self.fma(reciprocal, -2.0, 1.0);
        let large = self.copysign(x, large);
        let square = self.mul(x, x);
        let series = self.horner(&TANH, square);
        // The product with a zero added: at a zero operand it is `+0`, not the `-0` the negative first coefficient
        // would leave, so `-0` times it plus `-0` keeps its sign.
        let series = self.fma(series, square, 0.0);
        let small = self.fma(series, x, x);
        let is_large = self.compare(Cmp::Ge, Ty::F32, magnitude, TANH_SPLIT);
        self.select(Ty::F32, large, small, is_large)
    }

    /// `erf(x)`: an odd polynomial below the split, and `1 - 2^G(|x|)` with the sign of `x` from it on. A NaN
    /// fails the comparison and comes back through the polynomial.
    fn erf(&mut self, x: V) -> V {
        let magnitude = self.abs(x);
        let exponent = self.horner(&ERFC_LOG2, magnitude);
        let complement = self.hardware(Opcode::Ex2, exponent);
        let large = self.sub(1.0, complement);
        let large = self.copysign(x, large);
        let square = self.mul(x, x);
        let series = self.horner(&ERF, square);
        let small = self.fma(series, x, x);
        let is_large = self.compare(Cmp::Ge, Ty::F32, magnitude, ERF_SPLIT);
        self.select(Ty::F32, large, small, is_large)
    }

    /// `x^y` as `2^(y * log2|x|)`, with the logarithm and the product each held as a pair of floats whose sum is
    /// the value: the product reaches 150, and its error is the result's relative error.
    ///
    /// The logarithm: with the exponent `k`, a mantissa `m` and `f = m - 1`, `log2(m)` is a series in
    /// `u = 2f / (2 + f)`. The quotient is taken through the hardware's reciprocal and what it leaves of `u` is
    /// computed back from `2f - u * (2 + f)`, so the reciprocal's own error drops out. The first term `u * log2(e)`
    /// is split exactly by a fused multiply-add, `k` is added with its rounding kept, and the series joins the low
    /// parts with one rounding.
    ///
    /// The power: the product less its nearest integer `n`, with the product's low part, goes through a
    /// polynomial, and `2^n` is applied in two factors, since `n` runs past both ends of a normal float's exponent.
    ///
    /// Then what the operands' kinds decide: a zero or an infinite base, the sign of a negative base under an odd
    /// exponent, a NaN (an operand that is one, or a negative finite base with a fractional exponent), and the
    /// two cases that are one whatever the other operand is, a zero exponent and a base of one.
    fn pow(&mut self, x: V, y: V) -> V {
        let magnitude = self.abs(x);
        let (_, bits, exponent, m) = self.split(magnitude, ROOT_HALF_BITS);
        let k = self.converted(exponent);
        let k = self.mul(k, EXPONENT_UNIT);
        let f = self.add(m, -1.0);
        let sum = self.add(m, 1.0);
        let reciprocal = self.hardware(Opcode::Rcp, sum);
        let twice = self.add(f, f);
        let u = self.mul(twice, reciprocal);
        // `2f - u * (2 + f)` is `2 * (f - u) - u * f`, and `f - u` is exact.
        let gap = self.sub(f, u);
        let gap = self.add(gap, gap);
        let minus_u = self.neg(u);
        let residual = self.fma(minus_u, f, gap);
        let u_rest = self.mul(residual, reciprocal);
        let z = self.mul(u, u);
        let series = self.horner(&ATANH_LOG2, z);
        let cube = self.mul(u, z);
        let head = self.mul(u, LOG2_E);
        let high = self.add(k, head);
        let rest = self.sub(k, high);
        let rest = self.add(rest, head);
        let minus_head = self.neg(head);
        let low = self.fma(u, LOG2_E, minus_head);
        let low = self.fma(u, LOG2_E_REST, low);
        let low = self.add(low, rest);
        // What the rest of `u` moves: the first term, and the cubic one by `3 * u^2 / 12` of it.
        let slope = self.fma(z, 0.25 * LOG2_E, LOG2_E);
        let low = self.fma(u_rest, slope, low);
        let low = self.fma(cube, series, low);
        let log_high = self.add(high, low);
        let log_low = self.sub(high, log_high);
        let log_low = self.add(log_low, low);

        // A NaN exponent leaves the clamp as a number; the selects below bring the NaN back.
        let exponent = self.max(y, -EXPONENT_LIMIT);
        let exponent = self.min(exponent, EXPONENT_LIMIT);
        let product = self.mul(exponent, log_high);
        let minus_product = self.neg(product);
        let product_rest = self.fma(exponent, log_high, minus_product);
        let product_rest = self.fma(exponent, log_low, product_rest);

        let held = self.max(product, -PRODUCT_LIMIT);
        let held = self.min(held, PRODUCT_LIMIT);
        let whole = self.integral(held, Round::Rni);
        let fraction = self.sub(held, whole);
        let fraction = self.add(fraction, product_rest);
        // Past the limit the low part is no fraction: any value in the polynomial's reach will do.
        let fraction = self.max(fraction, -1.0);
        let fraction = self.min(fraction, 1.0);
        let series = self.horner(&EXP2, fraction);
        let power = self.fma(series, fraction, 1.0);
        let whole = self.truncated(whole);
        let first = self.int(Opcode::Shr, whole, 1);
        let second = self.int(Opcode::Sub, whole, first);
        let scale = self.power_of_two(first);
        let power = self.mul(power, scale);
        let scale = self.power_of_two(second);
        let power = self.mul(power, scale);

        // A zero or an infinite base gives zero or infinity, by which it is and the exponent's sign.
        let edge = self.outside_the_positive(bits);
        let zero_base = self.compare(Cmp::Eq, Ty::F32, magnitude, 0.0);
        let falling = self.compare(Cmp::Lt, Ty::F32, y, 0.0);
        let vanishes = self.logic(Opcode::Xor, zero_base, falling);
        let at_edge = self.select(Ty::F32, 0.0, f32::INFINITY, vanishes);
        let power = self.select(Ty::F32, at_edge, power, edge);
        // A negative base under an odd exponent gives a negative power.
        let halved = self.mul(y, 0.5);
        let halved = self.integral(halved, Round::Rzi);
        let even = self.add(halved, halved);
        let parity = self.sub(y, even);
        let parity = self.abs(parity);
        let odd = self.compare(Cmp::Eq, Ty::F32, parity, 1.0);
        let base_bits = self.bits(x);
        let signed = self.compare(Cmp::Lt, Ty::S32, base_bits, 0_i64);
        let flips = self.logic(Opcode::And, signed, odd);
        let flipped = self.neg(power);
        let power = self.select(Ty::F32, flipped, power, flips);
        let either = self.compare(Cmp::Nan, Ty::F32, x, y);
        let floor = self.integral(y, Round::Rmi);
        let fractional = self.compare(Cmp::Neu, Ty::F32, floor, y);
        let below_zero = self.compare(Cmp::Lt, Ty::F32, x, 0.0);
        let finite = self.compare(Cmp::Gt, Ty::F32, x, f32::NEG_INFINITY);
        let negative = self.logic(Opcode::And, below_zero, finite);
        let rootless = self.logic(Opcode::And, negative, fractional);
        let nan = self.logic(Opcode::Or, either, rootless);
        let power = self.select(Ty::F32, f32::NAN, power, nan);
        let zero_exponent = self.compare(Cmp::Eq, Ty::F32, y, 0.0);
        let unit_base = self.compare(Cmp::Eq, Ty::F32, x, 1.0);
        let one = self.logic(Opcode::Or, zero_exponent, unit_base);
        self.select(Ty::F32, 1.0, power, one)
    }
}

impl Lowerer {
    /// The operands of a math function, each an f32 value. A half has no rule: its function computes in f32 and
    /// converts once, and the body says where.
    fn math_operands<const N: usize>(
        &mut self,
        name: &str,
        args: &[Expression],
        ty: VTy,
        ctx: &SimplifyCtx,
    ) -> Result<[V; N]> {
        if ty != VTy::F32 {
            let ty = ty.text();
            return Err(self.no_rule(format_args!(
                "{name} at {ty}: a math function computes in f32, and the body casts its operand and its result"
            )));
        }
        if args.len() != N {
            return Err(self.refuse(format_args!("{name} takes {N} operands")));
        }
        let mut operands = [V::Int(0); N];
        for (operand, arg) in operands.iter_mut().zip(args) {
            *operand = match self.float(arg, ty, ctx)? {
                Operand::Reg(reg) => V::Reg(reg),
                Operand::F32(bits) => V::F32(f32::from_bits(bits)),
                _ => return Err(self.no_rule(format_args!("the operand {arg} of {name}"))),
            };
        }
        Ok(operands)
    }

    /// The value of the math function `op` over `args`: `None` where `op` is no math function.
    pub(super) fn math(
        &mut self,
        op: Operation,
        args: &[Expression],
        ty: VTy,
        ctx: &SimplifyCtx,
    ) -> Result<Option<Operand>> {
        use Operation as O;
        if matches!(op, O::Pow | O::Power) {
            let [x, y] = self.math_operands(op.text(), args, ty, ctx)?;
            return Ok(Some(Seq(self).pow(x, y).into()));
        }
        if !matches!(op, O::Exp | O::Log | O::Tanh | O::Erf | O::Rsqrt) {
            return Ok(None);
        }
        let [x] = self.math_operands(op.text(), args, ty, ctx)?;
        let ftz = self.precision.ftz;
        let mut seq = Seq(self);
        let value = match op {
            O::Exp => seq.exp(x),
            O::Log => seq.log(x),
            O::Tanh => seq.tanh(x),
            O::Erf => seq.erf(x),
            // The reciprocal square root is the hardware's, as `rsqrtf` is.
            _ => seq.approx(Opcode::Rsqrt, [x], ftz),
        };
        Ok(Some(value.into()))
    }

    /// The value of the approximate operation `row` over `args`, spelled `name`.
    pub(super) fn fast(
        &mut self,
        name: &str,
        row: Fast,
        args: &[Expression],
        ty: VTy,
        ctx: &SimplifyCtx,
    ) -> Result<Operand> {
        let ftz = self.precision.ftz;
        let value = match row {
            Fast::Inst(Opcode::Div) => {
                let operands: [V; 2] = self.math_operands(name, args, ty, ctx)?;
                Seq(self).approx(Opcode::Div, operands, ftz)
            }
            Fast::Inst(Opcode::Tanh) => {
                self.needs(SM_TURING, name)?;
                let operands: [V; 1] = self.math_operands(name, args, ty, ctx)?;
                // The one approximate instruction with no flushing form.
                Seq(self).approx(Opcode::Tanh, operands, false)
            }
            Fast::Inst(opcode) => {
                let operands: [V; 1] = self.math_operands(name, args, ty, ctx)?;
                Seq(self).approx(opcode, operands, ftz)
            }
            Fast::OfScaled(opcode, scale) => {
                let [x] = self.math_operands(name, args, ty, ctx)?;
                let mut seq = Seq(self);
                let scaled = seq.mul(x, scale);
                seq.approx(opcode, [scaled], ftz)
            }
            Fast::Scaled(opcode, scale) => {
                let operands: [V; 1] = self.math_operands(name, args, ty, ctx)?;
                let mut seq = Seq(self);
                let value = seq.approx(opcode, operands, ftz);
                seq.mul(value, scale)
            }
            Fast::Power => {
                let [x, y] = self.math_operands(name, args, ty, ctx)?;
                let mut seq = Seq(self);
                let log = seq.approx(Opcode::Lg2, [x], ftz);
                let product = seq.mul(y, log);
                seq.approx(Opcode::Ex2, [product], ftz)
            }
        };
        Ok(value.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The polynomial of `coefficients` at `x`, in f64.
    fn poly(coefficients: &[f32], x: f64) -> f64 {
        coefficients
            .iter()
            .rev()
            .fold(0.0, |sum, c| sum * x + f64::from(*c))
    }

    /// 2,001 points across `[low, high]`.
    fn grid(low: f64, high: f64) -> impl Iterator<Item = f64> {
        (0..=2000).map(move |step| low + (high - low) * f64::from(step) / 2000.0)
    }

    /// Each table of coefficients, evaluated exactly, is the function it is fit to within a sixth of an f32's
    /// step, so what a sweep on a card measures is mostly the rounding of the sequence, not the fit.
    #[test]
    fn the_coefficients_are_the_functions_they_are_fit_to() {
        let close = |got: f64, want: f64, x: f64| {
            let error = ((got - want) / want).abs();
            assert!(want == 0.0 || error < 1e-8, "{error:e} at {x}");
        };
        for f in grid(-1.0 / 3.0, 1.0 / 3.0) {
            close(f - f * f / 2.0 + f * f * f * poly(&LOG, f), f.ln_1p(), f);
        }
        for x in grid(-f64::from(TANH_SPLIT), f64::from(TANH_SPLIT)) {
            close(x + x * x * x * poly(&TANH, x * x), x.tanh(), x);
        }
        for x in grid(-f64::from(ERF_SPLIT), f64::from(ERF_SPLIT)) {
            close(x + x * poly(&ERF, x * x), libm::erf(x), x);
        }
        // Past the split the fit is weighted by the complement, so the error is taken of the result.
        for a in grid(f64::from(ERF_SPLIT), 6.0) {
            let error = (1.0 - poly(&ERFC_LOG2, a).exp2() - libm::erf(a)).abs();
            assert!(error < 1e-8, "{error:e} at {a}");
        }
        assert!(ERFC_LOG2.last().unwrap().is_sign_negative());
        // The power's two series are held tighter: their error is multiplied by the exponent.
        let fold = 2.0 * (2f64.sqrt() - 1.0) / (2f64.sqrt() + 1.0);
        for u in grid(-fold, fold) {
            let z = u * u;
            let got = u * std::f64::consts::LOG2_E + u * z * poly(&ATANH_LOG2, z);
            let want = ((1.0 + u / 2.0) / (1.0 - u / 2.0)).log2();
            assert!(want == 0.0 || ((got - want) / want).abs() < 1e-9, "at {u}");
        }
        for g in grid(-0.5, 0.5) {
            close(1.0 + g * poly(&EXP2, g), g.exp2(), g);
        }
    }

    /// The two parts of `log2(e)` are the constant to twice an f32's precision.
    #[test]
    fn the_split_constant_adds_up() {
        let sum = f64::from(LOG2_E) + f64::from(LOG2_E_REST);
        assert!((sum - std::f64::consts::LOG2_E).abs() < 1e-15);
    }

    /// Every row of the approximate operations has its own operation.
    #[test]
    fn an_approximate_operation_has_one_row() {
        for (index, (op, row)) in FAST.iter().enumerate() {
            assert_eq!(fast(*op), Some(*row));
            assert!(FAST[..index].iter().all(|(earlier, _)| earlier != op));
        }
        assert_eq!(fast(Operation::Exp), None);
    }
}
