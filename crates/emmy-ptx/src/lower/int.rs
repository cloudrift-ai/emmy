//! Integers and addresses. An index is computed in 32 bits where the range analysis says it fits and widened once;
//! a division by a constant is a shift, a mask or a multiply-high; an address is its base plus each term of the
//! flat index scaled and widened by itself, with the constant in the address.

use emmy_ir::Result;
use emmy_ir::elementwise::{ElementwiseOp, Operation};
use emmy_ir::expr::{Builtin, Expr, Expression, Interval, Num, SimplifyCtx, apply_binop};
use emmy_ir::fragment::volta_slab_index;

use super::memory::Buffer;
use super::{Lowerer, VTy, Val};
use crate::inst::{Bank, Base, Cmp, Inst, Opcode, Operand, Space, Special, Ty};

/// The lanes of a warp.
const WARP_SIZE: i64 = 32;

/// Whether every value of `range` is one a register of `ty` holds.
fn holds(ty: VTy, range: Interval) -> bool {
    match ty {
        VTy::I32 => range.lo >= i128::from(i32::MIN) && range.hi <= i128::from(i32::MAX),
        VTy::U32 => range.lo >= 0 && range.hi <= i128::from(u32::MAX),
        _ => true,
    }
}

/// Whether `value` is an integer a state of the integer type `ty` holds.
pub(super) fn holds_seed(ty: VTy, value: f64) -> bool {
    let whole = value as i128;
    let fits = i128::from(i64::MIN) <= whole && whole <= i128::from(i64::MAX);
    value.fract() == 0.0 && fits && holds(ty, Interval::new(whole, whole))
}

/// The narrowest integer type that holds `range`: 32 bits where it fits, signed first.
fn narrowest(range: Interval) -> VTy {
    [VTy::I32, VTy::U32]
        .into_iter()
        .find(|ty| holds(*ty, range))
        .unwrap_or(VTy::I64)
}

/// The type C computes two integers at, for operands with no range: 64 bits when either is, else unsigned when
/// either is.
fn wider(a: VTy, b: VTy) -> VTy {
    [VTy::U64, VTy::I64, VTy::U32]
        .into_iter()
        .find(|ty| a == *ty || b == *ty)
        .unwrap_or(VTy::I32)
}

/// The type of the bitwise instructions over a value of `ty`.
fn bits(ty: VTy) -> Ty {
    if ty.is_wide() { Ty::B64 } else { Ty::B32 }
}

pub(super) fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.saturating_abs()
}

/// The largest power of two `value` is a multiple of, held to 2^16; zero is a multiple of every one.
fn low_power(value: i64) -> i64 {
    1 << value.trailing_zeros().min(16)
}

/// The exponent of `value` when it is a power of two above one.
fn power_of_two(value: i64) -> Option<i64> {
    (value > 1 && value & (value - 1) == 0).then(|| i64::from(value.trailing_zeros()))
}

/// `(m, s)` with `x / d == ((x * m) >> 32) >> s` for every `0 <= x < 2^31`, for a divisor from 3 up that is no
/// power of two: `m` is `2^(31 + l) / d` rounded up, with `l` the bits of `d`, which fits 32 bits and overshoots
/// the true quotient by less than `1 / d`.
fn magic(d: u32) -> (u32, u32) {
    let l = 32 - (d - 1).leading_zeros();
    let m = ((1u64 << (31 + l)) + u64::from(d) - 1) / u64::from(d);
    (m as u32, l - 1)
}

/// `expression` as a sum of integer multiples of its terms plus a constant; `None` where a coefficient overflows.
pub(super) fn linear(
    expression: &Expression,
    scale: i64,
    terms: &mut Vec<(i64, Expression)>,
    constant: &mut i64,
) -> Option<()> {
    match expression.operation() {
        Some((Operation::Add, [a, b])) => {
            linear(a, scale, terms, constant)?;
            linear(b, scale, terms, constant)
        }
        Some((Operation::Subtract, [a, b])) => {
            linear(a, scale, terms, constant)?;
            linear(b, scale.checked_neg()?, terms, constant)
        }
        Some((Operation::Negative, [a])) => linear(a, scale.checked_neg()?, terms, constant),
        Some((Operation::Multiply, [a, b]))
            if let Some((factor, other)) = [(a, b), (b, a)]
                .into_iter()
                .find_map(|(k, other)| k.int_literal().map(|k| (k, other))) =>
        {
            linear(other, scale.checked_mul(factor)?, terms, constant)
        }
        _ => {
            if let Some(value) = expression.int_literal() {
                *constant = constant.checked_add(scale.checked_mul(value)?)?;
            } else if let Some(term) = terms.iter_mut().find(|(_, e)| e == expression) {
                term.0 = term.0.checked_add(scale)?;
            } else {
                terms.push((scale, expression.clone()));
            }
            Some(())
        }
    }
}

/// The constant `a - b` is, where every term of the two cancels.
pub(super) fn constant_difference(a: &Expression, b: &Expression) -> Option<i64> {
    let (mut terms, mut constant) = (Vec::new(), 0);
    linear(a, 1, &mut terms, &mut constant)?;
    linear(b, -1, &mut terms, &mut constant)?;
    terms
        .iter()
        .all(|(coefficient, _)| *coefficient == 0)
        .then_some(constant)
}

/// The floor of `expression / by`, for a positive `by`: spelled with no division where every coefficient of the
/// expression's terms is a multiple of `by`.
pub(super) fn divided(expression: &Expression, by: i64) -> Expression {
    let (mut terms, mut constant) = (Vec::new(), 0);
    let exact = linear(expression, 1, &mut terms, &mut constant).is_some()
        && terms.iter().all(|(coefficient, _)| coefficient % by == 0);
    if !exact {
        return Expression::binary(
            Operation::FloorDivide,
            expression.clone(),
            Expression::int(by),
        );
    }
    let mut quotient = Expression::int(constant.div_euclid(by));
    for (coefficient, term) in terms {
        let part = Expression::binary(Operation::Multiply, term, Expression::int(coefficient / by));
        quotient = Expression::binary(Operation::Add, quotient, part);
    }
    quotient
}

impl Val {
    pub(super) fn int(value: i64) -> Val {
        let range = Interval::new(i128::from(value), i128::from(value));
        Val {
            op: Operand::Int(value),
            ty: narrowest(range),
        }
    }
}

impl Lowerer {
    /// A number `operand` is a multiple of: a constant's magnitude, a register's known multiple.
    pub(super) fn multiple(&self, operand: &Operand) -> i64 {
        match operand {
            Operand::Int(value) => value.saturating_abs(),
            Operand::Reg(reg) => self.b.multiple(*reg),
            _ => 1,
        }
    }

    /// The register of a pure integer instruction at `ty`, known a multiple of `multiple`.
    fn int_value(
        &mut self,
        opcode: Opcode,
        at: Ty,
        ty: VTy,
        operands: Vec<Operand>,
        multiple: i64,
    ) -> Val {
        let reg = self
            .b
            .value(Inst::new(opcode, at, Vec::new(), operands), ty.bank());
        self.b.set_multiple(reg, multiple);
        Val::reg(reg, ty)
    }

    /// `val` as an operand of an instruction at `to`: a 32-bit value is widened by its own sign, a 64-bit one keeps
    /// its low half, a 16-bit one keeps its low 16 bits extended by its own sign, and a constant is the constant.
    pub(super) fn as_int(&mut self, val: &Val, to: VTy) -> Operand {
        let wide = |ty: VTy| ty.is_wide();
        if matches!(to, VTy::I16 | VTy::U16) && val.ty != to {
            return self.narrowed_to_16(val, to);
        }
        match &val.op {
            Operand::Int(value) => Operand::Int(match to {
                VTy::I32 => i64::from(*value as i32),
                VTy::U32 => i64::from(*value as u32),
                _ => *value,
            }),
            operand if wide(val.ty) == wide(to) => operand.clone(),
            operand => {
                let (at, from) = match (val.ty, to) {
                    (VTy::U32 | VTy::U16, _) => (Ty::U64, Ty::U32),
                    (VTy::I64 | VTy::U64, _) => (Ty::U32, Ty::U64),
                    _ => (Ty::S64, Ty::S32),
                };
                // Widening keeps the value; keeping the low half keeps it modulo 2^32, so a power of two below that
                // still divides it.
                let multiple = self.multiple(operand);
                let multiple = if wide(to) {
                    multiple
                } else {
                    low_power(multiple)
                };
                self.int_value(Opcode::Cvt(from), at, to, vec![operand.clone()], multiple)
                    .op
            }
        }
    }

    /// The low 16 bits of `val`, extended to 32 by their sign for an `i16` and by zeros for a `u16`.
    fn narrowed_to_16(&mut self, val: &Val, to: VTy) -> Operand {
        if let Some(value) = val.op.int() {
            return Operand::Int(if to == VTy::I16 {
                i64::from(value as i16)
            } else {
                i64::from(value as u16)
            });
        }
        let low = self.as_int(val, VTy::U32);
        let low = self.int_value(
            Opcode::And,
            Ty::B32,
            VTy::U32,
            vec![low, Operand::Int(0xffff)],
            1,
        );
        if to == VTy::U16 {
            return low.op;
        }
        let high = self.int_value(
            Opcode::Shl,
            Ty::B32,
            VTy::U32,
            vec![low.op, Operand::Int(16)],
            1,
        );
        self.int_value(
            Opcode::Shr,
            Ty::S32,
            VTy::I32,
            vec![high.op, Operand::Int(16)],
            1,
        )
        .op
    }

    /// The first of the three integer types that holds both operands as they are: what a division, a shift right,
    /// a minimum or a comparison runs at, since those read every bit.
    fn common(
        &self,
        a: &Val,
        a_range: Option<Interval>,
        b: &Val,
        b_range: Option<Interval>,
    ) -> VTy {
        let fits = |val: &Val, range: Option<Interval>, ty: VTy| match val.op.int() {
            Some(value) => holds(ty, Interval::new(i128::from(value), i128::from(value))),
            None => val.ty == ty || range.is_some_and(|range| holds(ty, range)),
        };
        let wide = if a.ty == VTy::U64 || b.ty == VTy::U64 {
            VTy::U64
        } else {
            VTy::I64
        };
        [VTy::I32, VTy::U32]
            .into_iter()
            .find(|ty| fits(a, a_range, *ty) && fits(b, b_range, *ty))
            .unwrap_or(wide)
    }

    fn builtin(&mut self, builtin: Builtin) -> Val {
        let special = match builtin {
            Builtin::ThreadIdxX => Special::TidX,
            Builtin::ThreadIdxY => Special::TidY,
            Builtin::ThreadIdxZ => Special::TidZ,
            Builtin::BlockIdxX => Special::CtaidX,
            Builtin::BlockIdxY => Special::CtaidY,
            Builtin::BlockIdxZ => Special::CtaidZ,
            Builtin::BlockDimX => Special::NtidX,
            Builtin::BlockDimY => Special::NtidY,
            Builtin::BlockDimZ => Special::NtidZ,
            Builtin::GridDimX => Special::NctaidX,
            Builtin::GridDimY => Special::NctaidY,
            Builtin::GridDimZ => Special::NctaidZ,
            Builtin::WarpSize => return Val::int(WARP_SIZE),
        };
        self.int_value(
            Opcode::Mov,
            Ty::U32,
            VTy::I32,
            vec![Operand::Special(special)],
            1,
        )
    }

    /// The value of an integer expression, at the narrowest type its range allows.
    pub(super) fn int(&mut self, expression: &Expression, ctx: &SimplifyCtx) -> Result<Val> {
        match &**expression {
            Expr::Literal(lit) => match lit.value {
                Num::Float(_) => Err(self.refuse("a float literal is read as an integer")),
                value => Ok(Val::int(value.as_int()?)),
            },
            Expr::Var(name) => {
                let val = self.lookup(*name)?;
                match val.ty {
                    ty if ty.is_int() => Ok(val),
                    VTy::Bool => self.flag(&val),
                    // The bits of an 8-bit float or a pair of codes are a byte.
                    ty if ty.is_byte_float() => Ok(Val {
                        op: val.op,
                        ty: VTy::U32,
                    }),
                    ty => {
                        Err(self
                            .refuse(format_args!("{name} is {}, read as an integer", ty.text())))
                    }
                }
            }
            Expr::Builtin(builtin) => Ok(self.builtin(*builtin)),
            Expr::Cast(target, inner) => {
                let to = self.cast_target(target)?;
                if !to.is_int() {
                    return Err(self.refuse(format_args!(
                        "a cast to {} is read as an integer",
                        to.text()
                    )));
                }
                self.cast(inner, to, ctx)
            }
            Expr::Ternary(cond, if_true, if_false) => {
                let pred = self.pred(cond, ctx)?;
                let (a, b) = (self.int(if_true, ctx)?, self.int(if_false, ctx)?);
                let ty = self.common(&a, if_true.range(ctx), &b, if_false.range(ctx));
                let multiple = gcd(self.multiple(&a.op), self.multiple(&b.op));
                let operands = vec![self.as_int(&a, ty), self.as_int(&b, ty), pred.into()];
                Ok(self.int_value(Opcode::Selp, ty.arith(), ty, operands, multiple))
            }
            Expr::Apply(ElementwiseOp::Known(op), args) => {
                self.int_apply(expression, *op, args, ctx)
            }
            Expr::Apply(ElementwiseOp::Other(name), args) if &**name == super::value::TO_E2M1 => {
                let code = self.e2m1_code(args, ctx)?;
                Ok(Val::reg(code, VTy::U32))
            }
            // Where a Volta slab layout puts the element at a logical row and column, the slab's rows or its row
            // stride the extent: the index a store of the copy precomputes once.
            Expr::Apply(ElementwiseOp::Other(name), args) if args.len() == 3 => {
                let layout = match &**name {
                    "emmy_volta_crosswise" => "V70A",
                    "emmy_volta_b_congruous" => "V70B",
                    _ => return Err(self.no_rule(format_args!("the expression {expression}"))),
                };
                let index = volta_slab_index(layout, &args[0], &args[1], &args[2])
                    .expect("a Volta layout has an index");
                self.int(&index, ctx)
            }
            _ => Err(self.no_rule(format_args!("the expression {expression}"))),
        }
    }

    fn int_apply(
        &mut self,
        expression: &Expression,
        op: Operation,
        args: &[Expression],
        ctx: &SimplifyCtx,
    ) -> Result<Val> {
        use Operation as O;
        match (op, args) {
            (
                O::Add | O::Subtract | O::Multiply | O::BitwiseAnd | O::BitwiseOr | O::BitwiseXor,
                [x, y],
            ) => self.wrapping(expression, op, x, y, ctx),
            (O::LeftShift | O::RightShift, [x, y]) => self.shift(expression, op, x, y, ctx),
            (O::Divide | O::FloorDivide, [x, y]) => self.quotient(x, y, false, ctx),
            (O::Remainder | O::Mod, [x, y]) => self.quotient(x, y, true, ctx),
            (O::Minimum | O::Maximum, [x, y]) => {
                let (a, b) = (self.int(x, ctx)?, self.int(y, ctx)?);
                if let Some(folded) = self.folded(op, &a, &b)? {
                    return Ok(folded);
                }
                let ty = self.common(&a, x.range(ctx), &b, y.range(ctx));
                let operands = vec![self.as_int(&a, ty), self.as_int(&b, ty)];
                let opcode = if op == O::Minimum {
                    Opcode::Min
                } else {
                    Opcode::Max
                };
                Ok(self.int_value(opcode, ty.arith(), ty, operands, 1))
            }
            (O::Negative, [x]) => {
                let a = self.int(x, ctx)?;
                if let Some(value) = a.op.int() {
                    return Ok(Val::int(value.wrapping_neg()));
                }
                let ty = expression.range(ctx).map_or(a.ty, narrowest);
                let (operand, multiple) = (self.as_int(&a, ty), self.multiple(&a.op));
                let at = if ty.is_wide() { Ty::S64 } else { Ty::S32 };
                Ok(self.int_value(Opcode::Neg, at, ty, vec![operand], multiple))
            }
            (O::Abs | O::Absolute, [x]) => {
                let a = self.int(x, ctx)?;
                match (a.op.int(), a.ty) {
                    (Some(value), _) => Ok(Val::int(value.wrapping_abs())),
                    (None, VTy::U32) => Ok(a),
                    (None, ty) => {
                        let multiple = self.multiple(&a.op);
                        Ok(self.int_value(Opcode::Abs, ty.arith(), ty, vec![a.op], multiple))
                    }
                }
            }
            (O::BitwiseCount, [x]) => {
                let a = self.int(x, ctx)?;
                let operand = self.register(&a).into();
                Ok(self.int_value(Opcode::Popc, bits(a.ty), VTy::I32, vec![operand], 1))
            }
            (O::Copy | O::Positive, [x]) => self.int(x, ctx),
            _ => Err(self.no_rule(format_args!("the integer operation {op}"))),
        }
    }

    /// The constant `op` gives two constants, as the simplifier folds it.
    fn folded(&self, op: Operation, a: &Val, b: &Val) -> Result<Option<Val>> {
        let (Some(a), Some(b)) = (a.op.int(), b.op.int()) else {
            return Ok(None);
        };
        let value = match op {
            Operation::Minimum => a.min(b),
            Operation::Maximum => a.max(b),
            op => apply_binop(op, Num::Int(a), Num::Int(b))?.as_int()?,
        };
        Ok(Some(Val::int(value)))
    }

    /// A sum, a difference, a product or a bitwise operation: the low bits of the result depend on the low bits
    /// of the operands alone, so it runs at the type its own range needs, whatever its operands were computed at.
    fn wrapping(
        &mut self,
        expression: &Expression,
        op: Operation,
        x: &Expression,
        y: &Expression,
        ctx: &SimplifyCtx,
    ) -> Result<Val> {
        let (a, b) = (self.int(x, ctx)?, self.int(y, ctx)?);
        if let Some(folded) = self.folded(op, &a, &b)? {
            return Ok(folded);
        }
        let ty = expression
            .range(ctx)
            .map_or_else(|| wider(a.ty, b.ty), narrowest);
        let (ma, mb) = (self.multiple(&a.op), self.multiple(&b.op));
        if op == Operation::Multiply {
            return Ok(self.product_of(&a, &b, ty, ma.saturating_mul(mb)));
        }
        let operands = vec![self.as_int(&a, ty), self.as_int(&b, ty)];
        Ok(match op {
            Operation::Add => self.int_value(Opcode::Add, ty.arith(), ty, operands, gcd(ma, mb)),
            Operation::Subtract => {
                self.int_value(Opcode::Sub, ty.arith(), ty, operands, gcd(ma, mb))
            }
            // The low zero bits both operands share stay zero under any bitwise operation; a mask keeps the
            // zero bits of either side.
            Operation::BitwiseAnd => {
                let multiple = low_power(ma).max(low_power(mb));
                self.int_value(Opcode::And, bits(ty), ty, operands, multiple)
            }
            Operation::BitwiseOr => {
                self.int_value(Opcode::Or, bits(ty), ty, operands, low_power(gcd(ma, mb)))
            }
            _ => self.int_value(Opcode::Xor, bits(ty), ty, operands, low_power(gcd(ma, mb))),
        })
    }

    /// A product at `ty`: a shift where one side is a power of two, the 32-by-32 multiply into 64 bits where two
    /// 32-bit operands of one signedness give a 64-bit product, else the low half.
    fn product_of(&mut self, a: &Val, b: &Val, ty: VTy, multiple: i64) -> Val {
        for (value, constant) in [(a, b), (b, a)] {
            if let Some(shift) = constant.op.int().and_then(power_of_two)
                && (!ty.is_wide() || value.ty.is_wide())
            {
                let operands = vec![self.as_int(value, ty), Operand::Int(shift)];
                return self.int_value(Opcode::Shl, bits(ty), ty, operands, multiple);
            }
        }
        if ty.is_wide() && !a.ty.is_wide() && !b.ty.is_wide() {
            let constant = |val: &Val, ty| {
                val.op
                    .int()
                    .is_some_and(|v| holds(ty, Interval::new(v.into(), v.into())))
            };
            for half in [VTy::I32, VTy::U32] {
                let fits =
                    |val: &Val| val.op.int().is_none() && val.ty == half || constant(val, half);
                if fits(a) && fits(b) {
                    let operands = vec![a.op.clone(), b.op.clone()];
                    return self.int_value(Opcode::MulWide, half.arith(), ty, operands, multiple);
                }
            }
        }
        let operands = vec![self.as_int(a, ty), self.as_int(b, ty)];
        self.int_value(Opcode::Mul, ty.arith(), ty, operands, multiple)
    }

    /// A shift left keeps the low bits like a product; a shift right reads every bit, so it runs at its operand's
    /// type, arithmetic for a signed one.
    fn shift(
        &mut self,
        expression: &Expression,
        op: Operation,
        x: &Expression,
        y: &Expression,
        ctx: &SimplifyCtx,
    ) -> Result<Val> {
        let (a, b) = (self.int(x, ctx)?, self.int(y, ctx)?);
        if let Some(folded) = self.folded(op, &a, &b)? {
            return Ok(folded);
        }
        let count = self.as_int(&b, VTy::I32);
        if op == Operation::LeftShift {
            let ty = expression.range(ctx).map_or(a.ty, narrowest);
            let multiple = match b.op.int() {
                Some(shift @ 0..=62) => self.multiple(&a.op).saturating_mul(1 << shift),
                _ => 1,
            };
            let operands = vec![self.as_int(&a, ty), count];
            return Ok(self.int_value(Opcode::Shl, bits(ty), ty, operands, multiple));
        }
        let value = self.register(&a).into();
        Ok(self.int_value(Opcode::Shr, a.ty.arith(), a.ty, vec![value, count], 1))
    }

    /// A floor quotient or its remainder. By a constant power of two it is an arithmetic shift or a mask, which
    /// floor whatever the sign; by another constant under a 32-bit dividend it is a multiply-high; else it is the
    /// divide instruction, corrected to the floor where a sign may differ.
    fn quotient(
        &mut self,
        x: &Expression,
        y: &Expression,
        remainder: bool,
        ctx: &SimplifyCtx,
    ) -> Result<Val> {
        // Two values no wider than 32 bits, neither negative, widened to 64: their quotient and remainder are the
        // 32-bit ones, with no 64-bit divide.
        let signs = (self.non_negative(x, ctx), self.non_negative(y, ctx));
        let (x, y) = if signs == (true, true) {
            (self.unwidened(x), self.unwidened(y))
        } else {
            (x, y)
        };
        let (a, b) = (self.int(x, ctx)?, self.int(y, ctx)?);
        let op = if remainder {
            Operation::Remainder
        } else {
            Operation::FloorDivide
        };
        if let Some(folded) = self.folded(op, &a, &b)? {
            return Ok(folded);
        }
        let (a_range, b_range) = (x.range(ctx), y.range(ctx));
        let multiple = self.multiple(&a.op);
        if let Some(divisor) = b.op.int().filter(|divisor| *divisor > 0) {
            if divisor == 1 || (remainder && multiple % divisor == 0) {
                return Ok(if remainder { Val::int(0) } else { a });
            }
            if let Some(shift) = power_of_two(divisor) {
                let value = self.register(&a).into();
                // A mask keeps the low zero bits of its operand; an exact quotient is the multiple over the divisor.
                return Ok(if remainder {
                    let mask = Operand::Int(divisor - 1);
                    let kept = gcd(multiple, divisor);
                    self.int_value(Opcode::And, bits(a.ty), a.ty, vec![value, mask], kept)
                } else {
                    let operands = vec![value, Operand::Int(shift)];
                    let left = if multiple % divisor == 0 {
                        multiple / divisor
                    } else {
                        1
                    };
                    self.int_value(Opcode::Shr, a.ty.arith(), a.ty, operands, left)
                });
            }
            let small = a_range.is_some_and(|range| holds(VTy::I32, range) && range.lo >= 0);
            if (small || a.ty == VTy::I32) && divisor <= i64::from(i32::MAX) {
                let quotient = self.multiply_high_quotient(&a, small || signs.0, divisor);
                if !remainder {
                    return Ok(quotient);
                }
                let operands = vec![quotient.op, Operand::Int(divisor)];
                let whole = self.int_value(Opcode::Mul, Ty::S32, VTy::I32, operands, divisor);
                let operands = vec![self.as_int(&a, VTy::I32), whole.op];
                let kept = gcd(multiple, divisor);
                return Ok(self.int_value(Opcode::Sub, Ty::S32, VTy::I32, operands, kept));
            }
            if a.ty.is_wide() && signs.0 {
                return Ok(self.wide_multiply_high(&a, divisor, remainder, multiple));
            }
        }
        // A remainder by a constant is a multiple of what the dividend and the divisor share.
        let kept = match b.op.int() {
            Some(divisor) if remainder && divisor != 0 => gcd(multiple, divisor),
            _ => 1,
        };
        let ty = self.common(&a, a_range, &b, b_range);
        let at = ty.arith();
        let (left, right) = (self.as_int(&a, ty), self.as_int(&b, ty));
        // Truncation is the floor where neither side is negative, and a division by zero has no value in either
        // form. The unsigned instruction is then the same value, and the cheaper one.
        let exact = ty == VTy::U32 || signs == (true, true);
        let operands = vec![left, right.clone()];
        if exact {
            let opcode = if remainder { Opcode::Rem } else { Opcode::Div };
            let at = match at {
                Ty::S32 => Ty::U32,
                Ty::S64 => Ty::U64,
                at => at,
            };
            // A remainder beside the quotient of the same operands is the dividend less the quotient's multiple.
            let quotient = Inst::new(Opcode::Div, at, Vec::new(), operands.clone());
            if remainder && let Some(quotient) = self.b.known(&quotient) {
                let at = if ty.is_wide() { Ty::S64 } else { Ty::S32 };
                let product = vec![quotient.into(), operands[1].clone()];
                let whole = self.int_value(Opcode::Mul, at, ty, product, 1);
                let operands = vec![operands[0].clone(), whole.op];
                return Ok(self.int_value(Opcode::Sub, at, ty, operands, kept));
            }
            if at == Ty::U64 && operands.iter().all(|operand| operand.int().is_none()) {
                return Ok(self.bypassed(opcode, operands, ty, kept));
            }
            return Ok(self.int_value(opcode, at, ty, operands, kept));
        }
        // The instruction rounds toward zero: where the remainder is nonzero and its sign is not the divisor's,
        // the floor is one lower and its remainder one divisor higher.
        let rest = self.int_value(Opcode::Rem, at, ty, operands.clone(), kept);
        let signs = self.int_value(
            Opcode::Xor,
            bits(ty),
            ty,
            vec![rest.op.clone(), right.clone()],
            1,
        );
        let differ = Inst::new(
            Opcode::Setp(Cmp::Lt),
            at,
            Vec::new(),
            vec![signs.op, Operand::Int(0)],
        );
        let differ = self.b.value(differ, Bank::Pred);
        let nonzero = Inst::new(
            Opcode::Setp(Cmp::Ne),
            at,
            Vec::new(),
            vec![rest.op.clone(), Operand::Int(0)],
        );
        let nonzero = self.b.value(nonzero, Bank::Pred);
        let both = Inst::new(
            Opcode::And,
            Ty::Pred,
            Vec::new(),
            vec![differ.into(), nonzero.into()],
        );
        let lower = self.b.value(both, Bank::Pred);
        let (base, step) = if remainder {
            (rest, right)
        } else {
            (
                self.int_value(Opcode::Div, at, ty, operands, 1),
                Operand::Int(-1),
            )
        };
        let correction = self.int_value(
            Opcode::Selp,
            at,
            ty,
            vec![step, Operand::Int(0), lower.into()],
            kept,
        );
        Ok(self.int_value(Opcode::Add, at, ty, vec![base.op, correction.op], kept))
    }

    /// Whether no value of `expression` is negative: its range says so, or it is a built-in, a symbolic size or a
    /// literal from zero up, or it is made of such values by sums, products, quotients, remainders by them,
    /// minima, maxima, selections, masks, shifts and casts. A sum or a product of them that wrapped past its
    /// type is a value C's own index arithmetic assumes away, as cicc does.
    pub(super) fn non_negative(&self, expression: &Expression, ctx: &SimplifyCtx) -> bool {
        use Operation as O;
        if expression.range(ctx).is_some_and(|range| range.lo >= 0) {
            return true;
        }
        let each = |args: &[Expression]| args.iter().all(|arg| self.non_negative(arg, ctx));
        match &**expression {
            Expr::Var(name) => self.sizes.contains(name),
            Expr::Builtin(_) => true,
            Expr::Literal(_) => expression.int_literal().is_some_and(|value| value >= 0),
            Expr::Ternary(_, a, b) => self.non_negative(a, ctx) && self.non_negative(b, ctx),
            // A cast to fewer than 64 bits keeps a value from zero up only where it fits the 32-bit range: a size
            // and a built-in always do.
            Expr::Cast(target, inner) => {
                let fits = |to: VTy| {
                    to.is_wide()
                        || matches!(&**inner, Expr::Builtin(_))
                        || matches!(&**inner, Expr::Var(name) if self.sizes.contains(name))
                        || inner.range(ctx).is_some_and(|range| holds(VTy::I32, range))
                };
                self.cast_target(target)
                    .is_ok_and(|to| to.is_int() && fits(to))
                    && self.non_negative(inner, ctx)
            }
            _ => match expression.operation() {
                Some((
                    O::Add | O::Multiply | O::Divide | O::FloorDivide | O::Minimum | O::BitwiseOr,
                    args,
                )) => each(args),
                Some((O::Maximum | O::BitwiseAnd, args)) => {
                    args.iter().any(|arg| self.non_negative(arg, ctx))
                }
                // A floor remainder takes the divisor's sign.
                Some((O::Remainder | O::Mod, [_, divisor])) => self.non_negative(divisor, ctx),
                Some((O::RightShift | O::LeftShift, [value, _])) => self.non_negative(value, ctx),
                Some((O::Copy | O::Positive, [value])) => self.non_negative(value, ctx),
                _ => false,
            },
        }
    }

    /// The value inside a cast to 64 bits of a value C computes at 32 bits or fewer; `expression` itself otherwise.
    fn unwidened<'e>(&self, expression: &'e Expression) -> &'e Expression {
        match &**expression {
            Expr::Cast(target, inner)
                if self.cast_target(target).is_ok_and(VTy::is_wide)
                    && matches!(
                        self.infer(inner),
                        Some(VTy::I32 | VTy::U32 | VTy::I16 | VTy::U16)
                    ) =>
            {
                inner
            }
            _ => expression,
        }
    }

    /// The unsigned 64-bit quotient or remainder of two registers, with cicc's bypass: where both operands fit 32
    /// bits when the kernel runs, the 32-bit instruction, a fraction of the 64-bit one's cost, behind a branch.
    fn bypassed(&mut self, opcode: Opcode, operands: Vec<Operand>, ty: VTy, kept: i64) -> Val {
        let whole = Inst::new(opcode, Ty::U64, Vec::new(), operands.clone());
        if let Some(reg) = self.b.known(&whole) {
            return Val::reg(reg, ty);
        }
        let either = self.int_value(Opcode::Or, Ty::B64, VTy::U64, operands.clone(), 1);
        let mask = Operand::Int(-1 << 32);
        let high = self.int_value(Opcode::And, Ty::B64, VTy::U64, vec![either.op, mask], 1);
        let narrow = Inst::new(
            Opcode::Setp(Cmp::Eq),
            Ty::U64,
            Vec::new(),
            vec![high.op, Operand::Int(0)],
        );
        let narrow = self.b.value(narrow, Bank::Pred);
        let result = self.b.state(Bank::B64);
        let mut open = self.b.open_if(narrow);
        let halves: Vec<Operand> = operands
            .iter()
            .map(|operand| {
                let low = Inst::new(
                    Opcode::Cvt(Ty::U64),
                    Ty::U32,
                    Vec::new(),
                    vec![operand.clone()],
                );
                self.b.value(low, Bank::B32).into()
            })
            .collect();
        let short = self
            .b
            .value(Inst::new(opcode, Ty::U32, Vec::new(), halves), Bank::B32);
        let widened = Inst::new(
            Opcode::Cvt(Ty::U32),
            Ty::U64,
            vec![result],
            vec![short.into()],
        );
        self.b.emit(widened);
        self.b
            .else_(&mut open)
            .expect("the condition just opened is innermost");
        self.b.emit(Inst {
            results: vec![result],
            ..whole.clone()
        });
        self.b
            .close_if(open)
            .expect("the condition just opened is innermost");
        let value = self.b.reg(Bank::B64);
        self.b.emit(Inst::new(
            Opcode::Mov,
            Ty::B64,
            vec![value],
            vec![result.into()],
        ));
        self.b.set_multiple(value, kept);
        self.b.remember(&whole, value);
        Val::reg(value, ty)
    }

    /// The floor quotient or remainder of a 64-bit value from zero up by a constant from 3 up that is no power of
    /// two, as a multiply-high: `m` is `2^(63 + l) / d` rounded up, `l` the bits of `d`, exact below `2^63` as the
    /// 32-bit form is below `2^31`. A remainder by a 32-bit divisor is computed from the low halves.
    fn wide_multiply_high(&mut self, a: &Val, divisor: i64, remainder: bool, multiple: i64) -> Val {
        let d = divisor as u64;
        let l = 64 - (d - 1).leading_zeros();
        let m = (1u128 << (63 + l)).div_ceil(u128::from(d)) as u64;
        let value = self.as_int(a, VTy::U64);
        let operands = vec![value.clone(), Operand::Int(m as i64)];
        let high = self.int_value(Opcode::MulHi, Ty::U64, VTy::U64, operands, 1);
        let shift = Operand::Int(i64::from(l - 1));
        let quotient = self.int_value(Opcode::Shr, Ty::U64, VTy::U64, vec![high.op, shift], 1);
        if !remainder {
            return quotient;
        }
        let kept = gcd(multiple, divisor);
        if divisor <= i64::from(i32::MAX) {
            let low = |this: &mut Lowerer, operand: Operand| {
                this.int_value(Opcode::Cvt(Ty::U64), Ty::U32, VTy::U32, vec![operand], 1)
                    .op
            };
            let (value, quotient) = (low(self, value), low(self, quotient.op));
            let operands = vec![quotient, Operand::Int(divisor)];
            let whole = self.int_value(Opcode::Mul, Ty::S32, VTy::I32, operands, divisor);
            return self.int_value(Opcode::Sub, Ty::S32, VTy::I32, vec![value, whole.op], kept);
        }
        let operands = vec![quotient.op, Operand::Int(divisor)];
        let whole = self.int_value(Opcode::Mul, Ty::S64, VTy::I64, operands, divisor);
        self.int_value(Opcode::Sub, Ty::S64, VTy::I64, vec![value, whole.op], kept)
    }

    /// The floor quotient of a 32-bit value by a constant from 3 up that is no power of two, as a multiply-high.
    /// The multiply is exact over `0..2^31`; a dividend that may be negative is folded into that range by its
    /// sign and the quotient folded back, since `~x` is `-x - 1` and the floor of a negative `x / d` is
    /// `~(~x / d)`.
    fn multiply_high_quotient(&mut self, a: &Val, non_negative: bool, divisor: i64) -> Val {
        let value = self.as_int(a, VTy::I32);
        let mut int = |opcode, at, left: Operand, right: Operand| {
            self.int_value(opcode, at, VTy::I32, vec![left, right], 1)
                .op
        };
        let sign =
            (!non_negative).then(|| int(Opcode::Shr, Ty::S32, value.clone(), Operand::Int(31)));
        let folded = match &sign {
            Some(sign) => int(Opcode::Xor, Ty::B32, value, sign.clone()),
            None => value,
        };
        let (m, shift) = magic(divisor as u32);
        let high = int(Opcode::MulHi, Ty::U32, folded, Operand::Int(m.into()));
        let quotient = int(Opcode::Shr, Ty::U32, high, Operand::Int(shift.into()));
        let quotient = match sign {
            Some(sign) => int(Opcode::Xor, Ty::B32, quotient, sign),
            None => quotient,
        };
        Val {
            op: quotient,
            ty: VTy::I32,
        }
    }

    /// The predicate of an integer comparison, at the first type that holds both sides. A constant stands on the
    /// right.
    pub(super) fn int_compare(
        &mut self,
        cmp: Cmp,
        x: &Expression,
        y: &Expression,
        ctx: &SimplifyCtx,
    ) -> Result<crate::inst::Reg> {
        let (a, b) = (self.int(x, ctx)?, self.int(y, ctx)?);
        let ty = self.common(&a, x.range(ctx), &b, y.range(ctx));
        let (left, right) = (self.as_int(&a, ty), self.as_int(&b, ty));
        let (cmp, left, right) = match (&left, &right) {
            // Both sides known where the kernel is compiled (an unrolled copy's axis): ptxas folds the comparison
            // once one side stands in a register.
            (Operand::Int(_), Operand::Int(_)) => {
                let held = Inst::new(Opcode::Mov, ty.arith(), Vec::new(), vec![left.clone()]);
                (cmp, Operand::Reg(self.b.value(held, ty.bank())), right)
            }
            (Operand::Int(_), _) => (cmp.mirrored(), right, left),
            _ => (cmp, left, right),
        };
        let inst = Inst::new(Opcode::Setp(cmp), ty.arith(), Vec::new(), vec![left, right]);
        Ok(self.b.value(inst, Bank::Pred))
    }

    /// The axis of a loop that counts its trips from zero: `counter * step + start`.
    pub(super) fn scaled_counter(
        &mut self,
        counter: crate::inst::Reg,
        step: i64,
        start: i64,
    ) -> Val {
        let counter = Val::reg(counter, VTy::I32);
        let scaled = if step == 1 {
            counter
        } else {
            self.product_of(&counter, &Val::int(step), VTy::I32, step)
        };
        if start == 0 {
            return scaled;
        }
        let operands = vec![scaled.op, Operand::Int(start)];
        self.int_value(Opcode::Add, Ty::S32, VTy::I32, operands, gcd(step, start))
    }

    /// How many bodies deep the innermost name `expression` reads was bound: where its value is first known.
    fn depth(&self, expression: &Expression) -> usize {
        expression
            .free_vars()
            .iter()
            .filter_map(|name| self.find(*name).map(|(depth, _)| depth))
            .max()
            .unwrap_or(0)
    }

    /// `val` times `scale` in 64 bits: a 32-bit value is widened by the multiply itself.
    pub(super) fn scaled(&mut self, val: &Val, scale: i64) -> Operand {
        let multiple = self
            .multiple(&val.op)
            .saturating_mul(scale.saturating_abs());
        let narrow = |ty| holds(ty, Interval::new(scale.into(), scale.into()));
        match val.ty {
            _ if scale == 1 => self.as_int(val, VTy::I64),
            ty @ (VTy::I32 | VTy::U32) if narrow(ty) => {
                let operands = vec![val.op.clone(), Operand::Int(scale)];
                self.int_value(Opcode::MulWide, ty.arith(), VTy::I64, operands, multiple)
                    .op
            }
            _ => {
                let wide = Val {
                    op: self.as_int(val, VTy::I64),
                    ty: VTy::I64,
                };
                self.product_of(&wide, &Val::int(scale), VTy::I64, multiple)
                    .op
            }
        }
    }

    /// The address of one element of `buffer`: the base plus each term of the flat index, scaled to bytes and
    /// widened by itself, the terms bound in outer bodies first, and the constant in the address. A term added in
    /// 64 bits is one ptxas folds across the trips of a loop it unrolls, which a 32-bit sum is not. A shared
    /// buffer's address is 32 bits, and so are its terms.
    pub(super) fn address(
        &mut self,
        name: emmy_ir::Name,
        index: &[Expression],
        ctx: &SimplifyCtx,
    ) -> Result<Operand> {
        let buffer = self.buffer(name)?;
        if self.pages.contains_key(&name) {
            let (page, within) = self.page_of(name, buffer, index, ctx)?;
            return self.address_in(page, &within, ctx);
        }
        let folded;
        let flat = match index {
            [] => return Ok(Operand::Addr(Base::Reg(buffer.base), buffer.offset)),
            [flat] => flat,
            // One coordinate a dimension: what a fragment statement and the loads of its epilogue hold.
            _ => {
                folded = self.flat_index(name, index)?;
                &folded
            }
        };
        self.address_in(buffer, flat, ctx)
    }

    /// The address of element `flat` of `buffer`.
    pub(super) fn address_in(
        &mut self,
        buffer: Buffer,
        flat: &Expression,
        ctx: &SimplifyCtx,
    ) -> Result<Operand> {
        let (bytes, wide) = (buffer.elem.bytes(), buffer.space != Space::Shared);
        let (mut terms, mut constant) = (Vec::new(), 0);
        if linear(flat, 1, &mut terms, &mut constant).is_none() {
            (terms, constant) = (vec![(1, flat.clone())], 0);
        }
        terms.retain(|(coefficient, _)| *coefficient != 0);
        terms.sort_by_key(|(_, term)| self.depth(term));
        let overflow = self.refuse("an address past 64 bits");
        let mut offset = constant
            .checked_mul(bytes)
            .and_then(|constant| constant.checked_add(buffer.offset))
            .ok_or_else(|| overflow.clone())?;
        let mut address = buffer.base;
        for (coefficient, term) in &terms {
            let scale = coefficient
                .checked_mul(bytes)
                .ok_or_else(|| overflow.clone())?;
            let val = self.int(term, ctx)?;
            if let Some(value) = val.op.int() {
                let term = value.checked_mul(scale).ok_or_else(|| overflow.clone())?;
                offset = offset.checked_add(term).ok_or_else(|| overflow.clone())?;
                continue;
            }
            let counted = val.op.reg().and_then(|reg| self.counted.get(&reg).copied());
            let mut counter = None;
            if let Some((var, step, start)) = counted {
                // The axis is `step * counter + start`: the start's bytes go into the constant.
                let bytes = start.checked_mul(scale).ok_or_else(|| overflow.clone())?;
                let factor = step.checked_mul(scale).ok_or_else(|| overflow.clone())?;
                // A loop ptxas keeps reads a multiple stepped by the term's bytes; one it may unroll reads a
                // 64-bit copy of the counter scaled in the trip, which it folds into each copy's offset.
                if self.b.kept_rolled(var) {
                    let bank = if wide { Bank::B64 } else { Bank::B32 };
                    counter = self
                        .b
                        .counter_multiple(var, factor, bank)
                        .map(|reg| (reg.into(), bytes));
                } else if wide && let Some(reg) = self.b.counter_multiple(var, 1, Bank::B64) {
                    let copy = Val::reg(reg, VTy::I64);
                    let multiple = factor.saturating_abs();
                    let scaled = self
                        .product_of(&copy, &Val::int(factor), VTy::I64, multiple)
                        .op;
                    counter = Some((scaled, bytes));
                }
            }
            let (scaled, at) = if let Some((counter, bytes)) = counter {
                offset = offset.checked_add(bytes).ok_or_else(|| overflow.clone())?;
                (counter, if wide { Ty::S64 } else { Ty::S32 })
            } else if wide {
                (self.scaled(&val, scale), Ty::S64)
            } else {
                let multiple = self
                    .multiple(&val.op)
                    .saturating_mul(scale.saturating_abs());
                let product = self.product_of(&val, &Val::int(scale), VTy::I32, multiple);
                (product.op, Ty::S32)
            };
            let aligned = gcd(i64::from(self.b.alignment(address)), self.multiple(&scaled));
            let sum = Inst::new(Opcode::Add, at, Vec::new(), vec![address.into(), scaled]);
            address = self.b.value(sum, at.bank());
            self.b
                .set_alignment(address, 1 << aligned.trailing_zeros().min(16));
        }
        Ok(Operand::Addr(Base::Reg(address), offset))
    }
}

#[cfg(test)]
mod tests {
    use emmy_ir::expr::parse;

    use super::*;

    /// The multiply-high quotient is the true one at every edge of every divisor: around each multiple, and at the
    /// top of the range the rule admits. So is the floor of a negative dividend folded into that range.
    #[test]
    fn the_multiply_high_quotient_is_exact_below_two_to_the_31() {
        let top = i32::MAX as u64;
        let divisors = [
            3u32,
            5,
            6,
            7,
            9,
            10,
            12,
            25,
            100,
            641,
            1000,
            65537,
            1 << 30 | 1,
            i32::MAX as u32,
        ];
        for d in divisors {
            let (m, shift) = magic(d);
            let d = u64::from(d);
            let multiples = [1, 2, 3, top / d / 2, top / d];
            let multiples = multiples.iter().filter(|k| **k > 0);
            let edges = multiples.flat_map(|k| [k * d - 1, k * d, k * d + 1]);
            for x in edges.chain([0, 1, top - 1, top]).filter(|x| *x <= top) {
                let quotient = |y: u64| ((y * u64::from(m)) >> 32) >> shift;
                assert_eq!(quotient(x), x / d, "{x} / {d}");
                // A negative dividend folded by its sign: `~x` is `-x - 1`, which is `x` here.
                let negative = -(x as i64) - 1;
                assert_eq!(
                    !(quotient(x) as i64),
                    negative.div_euclid(d as i64),
                    "{negative} / {d}"
                );
            }
        }
    }

    /// An index is its terms with their coefficients and its constant, through sums, differences, negations and
    /// literal factors; a term that is no sum is kept whole.
    #[test]
    fn an_index_splits_into_scaled_terms_and_a_constant() {
        let index = parse("r * 32 + (k - 1) * 8 + j - 2 * r + 3 + k / 4").unwrap();
        let (mut terms, mut constant) = (Vec::new(), 0);
        linear(&index, 1, &mut terms, &mut constant).unwrap();
        let terms: Vec<(i64, String)> = terms.iter().map(|(k, e)| (*k, e.text())).collect();
        let expected =
            [(30, "r"), (8, "k"), (1, "j"), (1, "k / 4")].map(|(k, e)| (k, e.to_string()));
        assert_eq!((terms, constant), (expected.to_vec(), -5));
    }

    #[test]
    fn the_narrowest_type_holds_the_range() {
        let ty = |lo: i128, hi: i128| narrowest(Interval::new(lo, hi));
        assert_eq!(ty(-5, i128::from(i32::MAX)), VTy::I32);
        assert_eq!(ty(0, i128::from(u32::MAX)), VTy::U32);
        assert_eq!(ty(-1, i128::from(u32::MAX)), VTy::I64);
        assert_eq!(ty(0, 1 << 40), VTy::I64);
    }
}
