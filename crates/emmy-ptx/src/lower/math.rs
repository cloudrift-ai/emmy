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

/// The approximate operations: what `--use_fast_math` and the fast intrinsics give a CUDA kernel, each the hardware
/// instruction of its row. An instruction flushes to zero when the kernel's decision says.
///
/// PROVISIONAL NAMES: only `exp_fast` is an operation the IR declares. The other rows stand under the names this
/// table expects the pass that picks the approximate forms to give them; they are matched by spelling and are
/// reconciled with the IR's declarations when that pass lands.
const FAST: [(&str, Fast); 10] = [
    ("exp_fast", Fast::OfScaled(Opcode::Ex2, LOG2_E)),
    (
        "log_fast",
        Fast::Scaled(Opcode::Lg2, std::f32::consts::LN_2),
    ),
    ("sin_fast", Fast::Inst(Opcode::Sin)),
    ("cos_fast", Fast::Inst(Opcode::Cos)),
    ("tanh_fast", Fast::Inst(Opcode::Tanh)),
    ("pow_fast", Fast::Power),
    ("divide_fast", Fast::Inst(Opcode::Div)),
    ("reciprocal_fast", Fast::Inst(Opcode::Rcp)),
    ("sqrt_fast", Fast::Inst(Opcode::Sqrt)),
    ("rsqrt_fast", Fast::Inst(Opcode::Rsqrt)),
];

/// The approximate operation `name` spells.
pub(super) fn fast(name: &str) -> Option<Fast> {
    FAST.iter()
        .find(|(spelling, _)| *spelling == name)
        .map(|(_, row)| *row)
}

/// `log2(e)` as the nearest f32, and what that leaves of it.
const LOG2_E: f32 = std::f32::consts::LOG2_E;
const LOG2_E_REST: f32 = (std::f64::consts::LOG2_E - LOG2_E as f64) as f32;

/// The smallest normal f32.
const MIN_NORMAL: f32 = f32::MIN_POSITIVE;

/// The bits of `2/3`: the logarithm takes its mantissa from `[2/3, 4/3)`, so the value it adds to the exponent's
/// share is near zero only where that share is zero.
const TWO_THIRDS_BITS: i64 = 0x3f2a_aaab;

/// The exponent field of an f32, where it sits.
const EXPONENT_MASK: i64 = !0x007f_ffff;

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
            Opcode::Add | Opcode::Sub => Ty::S32,
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

    /// `e^x`. The exponent `k` is the floor of `x * log2(e)`, held to the exponents a normal f32 has; what is
    /// left of the product, taken with the rest of the constant, goes through the hardware's power of two, which
    /// is most accurate over `[0, 1)`; and `2^k` is written through its bits. Past either end `k` stays at its
    /// bound and the hardware instruction carries the result to infinity, through the subnormals, and to zero. A
    /// NaN leaves the clamp as a number and comes back through the fused multiply-add.
    fn exp(&mut self, x: V) -> V {
        let scaled = self.mul(x, LOG2_E);
        let k = self.integral(scaled, Round::Rmi);
        let k = self.max(k, -126.0);
        let k = self.min(k, 127.0);
        let minus_k = self.neg(k);
        let fraction = self.fma(x, LOG2_E, minus_k);
        let fraction = self.fma(x, LOG2_E_REST, fraction);
        let power = self.hardware(Opcode::Ex2, fraction);
        let exponent = self.truncated(k);
        let biased = self.int(Opcode::Add, exponent, 127);
        let bits = self.int(Opcode::Shl, biased, 23);
        let scale = self.of_bits(bits);
        self.mul(power, scale)
    }

    /// `ln(x)`. A subnormal is scaled into the normals first. The bits less those of `2/3` hold the exponent `k`
    /// in their exponent field, and taking that field back out leaves a mantissa `m` in `[2/3, 4/3)`. With
    /// `f = m - 1`, exact, the result is `k * ln(2) + ln(1 + f)`. Zero, a negative, an infinity and a NaN take
    /// the hardware logarithm's value, which is the right one for each; one unsigned comparison finds them all.
    fn log(&mut self, x: V) -> V {
        let subnormal = self.compare(Cmp::Lt, Ty::F32, x, MIN_NORMAL);
        let scaled = self.mul(x, 8388608.0);
        let x = self.select(Ty::F32, scaled, x, subnormal);
        // The scaling by 2^23 is taken back out of the exponent with the offset.
        let offset = self.select(
            Ty::B32,
            TWO_THIRDS_BITS + (23 << 23),
            TWO_THIRDS_BITS,
            subnormal,
        );
        let bits = self.bits(x);
        let shifted = self.int(Opcode::Sub, bits, offset);
        let exponent = self.int(Opcode::And, shifted, EXPONENT_MASK);
        let mantissa = self.int(Opcode::Sub, shifted, exponent);
        let mantissa = self.int(Opcode::Add, mantissa, TWO_THIRDS_BITS);
        let m = self.of_bits(mantissa);
        let k = self.converted(exponent);
        let f = self.add(m, -1.0);
        let series = self.horner(&LOG, f);
        let series = self.fma(series, f, -0.5);
        let series = self.mul(series, f);
        let log_m = self.fma(series, f, f);
        // `k` is still shifted into the exponent's place: the constant takes the shift back.
        let finite = self.fma(k, std::f32::consts::LN_2 / 8388608.0, log_m);
        let below = self.int(Opcode::Sub, bits, 1);
        let special = self.compare(Cmp::Ge, Ty::U32, below, 0x7f7f_ffff_i64);
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
}

impl Lowerer {
    /// The operands of a math function, each an f32 value. A half is refused: its function computes in f32 and
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
            return Err(self.refuse(format_args!(
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
    }

    /// The two parts of `log2(e)` are the constant to twice an f32's precision.
    #[test]
    fn the_split_constant_adds_up() {
        let sum = f64::from(LOG2_E) + f64::from(LOG2_E_REST);
        assert!((sum - std::f64::consts::LOG2_E).abs() < 1e-15);
    }

    /// Every row of the approximate operations has its own name.
    #[test]
    fn an_approximate_operation_has_one_row() {
        for (index, (name, row)) in FAST.iter().enumerate() {
            assert_eq!(fast(name), Some(*row));
            assert!(FAST[..index].iter().all(|(earlier, _)| earlier != name));
        }
        assert_eq!(fast("exp"), None);
    }
}
