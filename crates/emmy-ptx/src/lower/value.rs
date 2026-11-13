//! Floats, predicates and casts. Every float instruction that rounds carries its rounding mode, so ptxas fuses
//! nothing on its own; a fused multiply-add is written where the body says `fma`, or where the kernel's fusion
//! decision meets a product read once by a sum.

use emmy_ir::elementwise::{ElementwiseOp, Operation};
use emmy_ir::expr::{CastTarget, Expr, Expression, LitType, SimplifyCtx};
use emmy_ir::{Name, Result};

use super::math::fast;
use super::{Lowerer, SM_AMPERE, VTy, Val};
use crate::inst::{Bank, Cmp, Inst, Opcode, Operand, Reg, Round, Ty};

/// The sign bit of an `f32`.
const F32_SIGN: u32 = 1 << 31;

/// The 16 bits of `value` in a half-precision format, rounded to nearest even from the `f64`: `f16` has 5 exponent
/// bits and 10 of mantissa, `bf16` has 8 and 7.
pub(super) fn half_bits(ty: VTy, value: f64) -> u16 {
    let (exponent_bits, mantissa_bits) = if ty == VTy::Bf16 { (8, 7) } else { (5, 10) };
    let sign = u16::from(value.is_sign_negative()) << 15;
    let top = ((1u16 << exponent_bits) - 1) << mantissa_bits;
    if value.is_nan() {
        return sign | top | 1 << (mantissa_bits - 1);
    }
    let magnitude = value.abs();
    if magnitude == 0.0 {
        return sign;
    }
    let bias = (1i32 << (exponent_bits - 1)) - 1;
    // The exponent of the result's quantum: the value's own, or the format's smallest for a subnormal result.
    let exponent = (((magnitude.to_bits() >> 52) & 0x7ff) as i32 - 1023).max(1 - bias);
    if exponent > bias {
        return sign | top;
    }
    let implicit = 1u64 << mantissa_bits;
    let scaled = (magnitude / 2f64.powi(exponent - mantissa_bits)).round_ties_even() as u64;
    let bits = if scaled < implicit {
        scaled
    } else {
        (((exponent + bias) as u64) << mantissa_bits) + (scaled - implicit)
    };
    sign | bits.min(u64::from(top)) as u16
}

/// The cast `ir/address.py` writes on an index term of a buffer past the 32-bit range.
const WIDE_INDEX: &str = "long long";

fn comparison(op: Operation) -> Option<Cmp> {
    Some(match op {
        Operation::Less => Cmp::Lt,
        Operation::LessEqual => Cmp::Le,
        Operation::Greater => Cmp::Gt,
        Operation::GreaterEqual => Cmp::Ge,
        Operation::Equal => Cmp::Eq,
        Operation::NotEqual => Cmp::Ne,
        _ => return None,
    })
}

fn logical(op: Operation) -> Option<Opcode> {
    Some(match op {
        Operation::LogicalAnd | Operation::BitwiseAnd => Opcode::And,
        Operation::LogicalOr | Operation::BitwiseOr => Opcode::Or,
        Operation::BitwiseXor => Opcode::Xor,
        _ => return None,
    })
}

impl Lowerer {
    /// The type a cast names: a dtype, or one of the three literal types.
    pub(super) fn cast_target(&self, target: &CastTarget) -> Result<VTy> {
        match target {
            CastTarget::Known(LitType::Int) => Ok(VTy::I32),
            CastTarget::Known(LitType::Float) => Ok(VTy::F32),
            CastTarget::Known(LitType::Bool) => Ok(VTy::Bool),
            // The widening an index takes where its buffer passes the 32-bit range.
            CastTarget::Other(name) if name == WIDE_INDEX => Ok(VTy::I64),
            CastTarget::Other(name) => name
                .parse()
                .ok()
                .and_then(VTy::of)
                .ok_or_else(|| self.no_rule(format_args!("a cast to {name}"))),
        }
    }

    /// The type `expression` computes at, where its operands say: a literal alone takes its reader's.
    pub(super) fn infer(&self, expression: &Expression) -> Option<VTy> {
        match &**expression {
            Expr::Var(name) => self
                .products
                .get(name)
                .map(|(ty, ..)| *ty)
                .or_else(|| self.find(*name).map(|(_, val)| val.ty)),
            Expr::Builtin(_) => Some(VTy::I32),
            Expr::Literal(lit) => (lit.dtype == LitType::Bool).then_some(VTy::Bool),
            Expr::Cast(target, _) => self.cast_target(target).ok(),
            Expr::Ternary(_, a, b) => self.infer(a).or_else(|| self.infer(b)),
            Expr::Apply(ElementwiseOp::Known(op), args) => {
                let truth = comparison(*op).is_some()
                    || matches!(op, Operation::LogicalAnd | Operation::LogicalOr);
                // A selection computes at its two values' type, not at its condition's.
                let values = if *op == Operation::Where { 1 } else { 0 };
                if truth {
                    Some(VTy::Bool)
                } else {
                    args.iter().skip(values).find_map(|arg| self.infer(arg))
                }
            }
            _ => None,
        }
    }

    /// The type of an operand with no reader to say it: its own, else `f32` for a float literal and `i32` for an
    /// integer one.
    fn operand_type(&self, expression: &Expression) -> VTy {
        self.infer(expression).unwrap_or_else(|| {
            let float = expression
                .subterms()
                .iter()
                .any(|term| matches!(&**term, Expr::Literal(lit) if lit.dtype == LitType::Float));
            if float { VTy::F32 } else { VTy::I32 }
        })
    }

    /// A float instruction of `opcode` at `ty`, without its result: rounded to nearest where the operation rounds,
    /// and flushing to zero where the kernel's decision says and the type is `f32`.
    pub(super) fn float_inst(&self, opcode: Opcode, ty: VTy, operands: Vec<Operand>) -> Inst {
        let rounds = matches!(
            opcode,
            Opcode::Add
                | Opcode::Sub
                | Opcode::Mul
                | Opcode::Fma
                | Opcode::Div
                | Opcode::Rcp
                | Opcode::Sqrt
        );
        let inst = Inst::new(opcode, ty.arith(), Vec::new(), operands)
            .flushing(self.precision.ftz && ty == VTy::F32);
        if rounds {
            inst.rounded(Round::Rn)
        } else {
            inst
        }
    }

    /// An error where the target has no instruction of `opcode` at the float type `ty`.
    pub(super) fn float_supported(&self, opcode: Opcode, ty: VTy) -> Result<()> {
        let native = match (ty, opcode) {
            (VTy::F32, _) | (_, Opcode::Selp | Opcode::Mov) => true,
            (VTy::F16, Opcode::Add | Opcode::Sub | Opcode::Mul | Opcode::Fma) => true,
            (VTy::F16, Opcode::Neg | Opcode::Abs) => true,
            (VTy::F16, Opcode::Min | Opcode::Max) => self.target.sm >= SM_AMPERE,
            _ => false,
        };
        if native {
            Ok(())
        } else {
            let (ty, sm) = (ty.text(), self.target.sm);
            Err(self.no_rule(format_args!("{opcode:?} at {ty} on sm_{sm}")))
        }
    }

    fn float_value(&mut self, opcode: Opcode, ty: VTy, operands: Vec<Operand>) -> Result<Operand> {
        self.float_supported(opcode, ty)?;
        let inst = self.float_inst(opcode, ty, operands);
        Ok(self.b.value(inst, ty.bank()).into())
    }

    /// A float constant as an operand: an immediate at `f32`, a register at a half type, whose instructions take
    /// no immediate.
    fn float_constant(&mut self, ty: VTy, value: f64) -> Operand {
        match ty {
            VTy::F32 => Operand::f32(value as f32),
            _ => {
                let bits = Operand::Int(i64::from(half_bits(ty, value)));
                let inst = Inst::new(Opcode::Mov, Ty::B16, Vec::new(), vec![bits]);
                self.b.value(inst, Bank::B16).into()
            }
        }
    }

    /// The value of a float expression at `ty`.
    pub(super) fn float(
        &mut self,
        expression: &Expression,
        ty: VTy,
        ctx: &SimplifyCtx,
    ) -> Result<Operand> {
        match &**expression {
            Expr::Literal(lit) => Ok(self.float_constant(ty, lit.value.as_f64())),
            Expr::Var(name) => {
                let val = self.lookup(*name)?;
                if val.ty != ty {
                    let (held, read) = (val.ty.text(), ty.text());
                    return Err(self.refuse(format_args!("{name} is {held}, read as {read}")));
                }
                Ok(val.op)
            }
            Expr::Cast(target, inner) => {
                let to = self.cast_target(target)?;
                if to != ty {
                    let (to, read) = (to.text(), ty.text());
                    return Err(self.refuse(format_args!("a cast to {to} is read as {read}")));
                }
                Ok(self.cast(inner, to, ctx)?.op)
            }
            Expr::Ternary(cond, if_true, if_false) => {
                self.float_select(cond, if_true, if_false, ty, ctx)
            }
            Expr::Apply(ElementwiseOp::Known(op), args) => {
                if let Some(row) = fast(*op) {
                    return self.fast(&op.to_string(), row, args, ty, ctx);
                }
                self.float_apply(*op, args, ty, ctx)
            }
            Expr::Apply(ElementwiseOp::Other(name), _) => {
                Err(self.no_rule(format_args!("the operation {name}")))
            }
            _ => Err(self.no_rule(format_args!("the expression {expression}"))),
        }
    }

    fn float_select(
        &mut self,
        cond: &Expression,
        if_true: &Expression,
        if_false: &Expression,
        ty: VTy,
        ctx: &SimplifyCtx,
    ) -> Result<Operand> {
        let pred = self.pred(cond, ctx)?;
        let operands = vec![
            self.float(if_true, ty, ctx)?,
            self.float(if_false, ty, ctx)?,
            pred.into(),
        ];
        let inst = Inst::new(Opcode::Selp, ty.held(), Vec::new(), operands);
        Ok(self.b.value(inst, ty.bank()).into())
    }

    fn float_apply(
        &mut self,
        op: Operation,
        args: &[Expression],
        ty: VTy,
        ctx: &SimplifyCtx,
    ) -> Result<Operand> {
        use Operation as O;
        match (op, args) {
            (O::Add | O::Subtract, [x, y]) => {
                let subtract = op == O::Subtract;
                if let Some(fused) = self.fused(subtract, x, y, ty, ctx)? {
                    return Ok(fused);
                }
                let operands = vec![self.float(x, ty, ctx)?, self.float(y, ty, ctx)?];
                let opcode = if subtract { Opcode::Sub } else { Opcode::Add };
                self.float_value(opcode, ty, operands)
            }
            (O::Multiply, [x, y]) => {
                let operands = vec![self.float(x, ty, ctx)?, self.float(y, ty, ctx)?];
                self.float_value(Opcode::Mul, ty, operands)
            }
            (O::Divide | O::TrueDivide | O::DivideRnF32, [x, y]) => {
                let operands = vec![self.float(x, ty, ctx)?, self.float(y, ty, ctx)?];
                self.float_value(Opcode::Div, ty, operands)
            }
            (O::Maximum | O::Fmax, [x, y]) => {
                let operands = vec![self.float(x, ty, ctx)?, self.float(y, ty, ctx)?];
                self.float_value(Opcode::Max, ty, operands)
            }
            (O::Minimum | O::Fmin, [x, y]) => {
                let operands = vec![self.float(x, ty, ctx)?, self.float(y, ty, ctx)?];
                self.float_value(Opcode::Min, ty, operands)
            }
            (O::Fma, [x, y, z]) => {
                let operands = vec![
                    self.float(x, ty, ctx)?,
                    self.float(y, ty, ctx)?,
                    self.float(z, ty, ctx)?,
                ];
                self.float_value(Opcode::Fma, ty, operands)
            }
            (O::Square, [x]) => {
                let x = self.float(x, ty, ctx)?;
                self.float_value(Opcode::Mul, ty, vec![x.clone(), x])
            }
            (O::Relu, [x]) => {
                let operands = vec![self.float(x, ty, ctx)?, self.float_constant(ty, 0.0)];
                self.float_value(Opcode::Max, ty, operands)
            }
            (O::Negative, [x]) => {
                let x = self.float(x, ty, ctx)?;
                self.float_value(Opcode::Neg, ty, vec![x])
            }
            (O::Abs | O::Absolute | O::Fabs, [x]) => {
                let x = self.float(x, ty, ctx)?;
                self.float_value(Opcode::Abs, ty, vec![x])
            }
            (O::Sqrt, [x]) => {
                let x = self.float(x, ty, ctx)?;
                self.float_value(Opcode::Sqrt, ty, vec![x])
            }
            (O::Reciprocal, [x]) => {
                let x = self.float(x, ty, ctx)?;
                self.float_value(Opcode::Rcp, ty, vec![x])
            }
            (O::Floor | O::Ceil | O::Trunc | O::Rint | O::Round, [x]) if ty == VTy::F32 => {
                let mode = match op {
                    O::Floor => Round::Rmi,
                    O::Ceil => Round::Rpi,
                    O::Trunc => Round::Rzi,
                    _ => Round::Rni,
                };
                let x = self.float(x, ty, ctx)?;
                let inst = Inst::new(Opcode::Cvt(Ty::F32), Ty::F32, Vec::new(), vec![x])
                    .rounded(mode)
                    .flushing(self.precision.ftz);
                Ok(self.b.value(inst, Bank::F32).into())
            }
            (O::Where, [cond, if_true, if_false]) => {
                self.float_select(cond, if_true, if_false, ty, ctx)
            }
            (O::Copy | O::Positive, [x]) => self.float(x, ty, ctx),
            // A comparison read as a float is one where it holds and zero where it does not.
            (op, [x, y]) if comparison(op).is_some() => {
                let pred = self.compare(comparison(op).expect("a comparison"), x, y, ctx)?;
                let (one, zero) = (self.float_constant(ty, 1.0), self.float_constant(ty, 0.0));
                let inst = Inst::new(
                    Opcode::Selp,
                    ty.held(),
                    Vec::new(),
                    vec![one, zero, pred.into()],
                );
                Ok(self.b.value(inst, ty.bank()).into())
            }
            _ => {
                if let Some(value) = self.math(op, args, ty, ctx)? {
                    return Ok(value);
                }
                // A composite a kernel pass left as it was is its definition in primitive operations.
                if let [operand] = args
                    && let Some(definition) = op.definition(operand)?
                {
                    return self.float(&definition, ty, ctx);
                }
                Err(self.no_rule(format_args!("the operation {op} at {}", ty.text())))
            }
        }
    }

    /// The operands of the product `expression` is, where the kernel's fusion decision is on: a product held back
    /// for this reader, or a multiply written in the reader's own expression.
    pub(super) fn product(
        &mut self,
        expression: &Expression,
        ty: VTy,
        ctx: &SimplifyCtx,
    ) -> Result<Option<(Operand, Operand)>> {
        if !self.precision.fuse || !matches!(ty, VTy::F32 | VTy::F16) {
            return Ok(None);
        }
        if let Some(name) = expression.as_var() {
            return Ok(self.held_product(*name, ty));
        }
        match expression.operation() {
            Some((Operation::Multiply, [x, y])) => {
                Ok(Some((self.float(x, ty, ctx)?, self.float(y, ty, ctx)?)))
            }
            _ => Ok(None),
        }
    }

    fn held_product(&mut self, name: Name, ty: VTy) -> Option<(Operand, Operand)> {
        let held = self
            .products
            .get(&name)
            .is_some_and(|(held, ..)| *held == ty);
        held.then(|| self.products.remove(&name))
            .flatten()
            .map(|(_, a, b)| (a, b))
    }

    fn negated(&mut self, operand: Operand, ty: VTy) -> Result<Operand> {
        match operand {
            Operand::F32(bits) => Ok(Operand::F32(bits ^ F32_SIGN)),
            other => self.float_value(Opcode::Neg, ty, vec![other]),
        }
    }

    /// `x + y` or `x - y` as one fused multiply-add, where one side is a product the fusion decision takes. A
    /// negation is exact, so a difference is the fused sum over a negated operand.
    fn fused(
        &mut self,
        subtract: bool,
        x: &Expression,
        y: &Expression,
        ty: VTy,
        ctx: &SimplifyCtx,
    ) -> Result<Option<Operand>> {
        if let Some((a, b)) = self.product(x, ty, ctx)? {
            let addend = self.float(y, ty, ctx)?;
            let addend = if subtract {
                self.negated(addend, ty)?
            } else {
                addend
            };
            return self
                .float_value(Opcode::Fma, ty, vec![a, b, addend])
                .map(Some);
        }
        if let Some((a, b)) = self.product(y, ty, ctx)? {
            let addend = self.float(x, ty, ctx)?;
            let a = if subtract { self.negated(a, ty)? } else { a };
            return self
                .float_value(Opcode::Fma, ty, vec![a, b, addend])
                .map(Some);
        }
        Ok(None)
    }

    /// The predicate that holds where `val` is nonzero: a NaN is nonzero.
    pub(super) fn truth(&mut self, val: &Val) -> Result<Reg> {
        let (cmp, zero) = match val.ty {
            VTy::Bool => {
                return val
                    .op
                    .reg()
                    .ok_or_else(|| self.no_rule("a constant condition"));
            }
            VTy::Bf16 => return Err(self.no_rule("the truth of a bf16 value")),
            ty if ty.is_float() => (Cmp::Neu, self.float_constant(ty, 0.0)),
            _ => (Cmp::Ne, Operand::Int(0)),
        };
        if !matches!(val.op, Operand::Reg(_)) {
            return Err(self.no_rule("a constant condition"));
        }
        let inst = Inst::new(
            Opcode::Setp(cmp),
            val.ty.arith(),
            Vec::new(),
            vec![val.op.clone(), zero],
        )
        .flushing(self.precision.ftz && val.ty == VTy::F32);
        Ok(self.b.value(inst, Bank::Pred))
    }

    /// A bool as the integer 1 or 0.
    pub(super) fn flag(&mut self, val: &Val) -> Result<Val> {
        let pred = self.truth(val)?;
        let operands = vec![Operand::Int(1), Operand::Int(0), pred.into()];
        let inst = Inst::new(Opcode::Selp, Ty::S32, Vec::new(), operands);
        Ok(Val::reg(self.b.value(inst, Bank::B32), VTy::I32))
    }

    /// The predicate `expression` is: a comparison, a logical operation over predicates, or the truth of a value.
    pub(super) fn pred(&mut self, expression: &Expression, ctx: &SimplifyCtx) -> Result<Reg> {
        if let Some((op, [x, y])) = expression.operation() {
            if let Some(cmp) = comparison(op) {
                return self.compare(cmp, x, y, ctx);
            }
            if let Some(opcode) = logical(op)
                && self.infer(expression) == Some(VTy::Bool)
            {
                let operands = vec![self.pred(x, ctx)?.into(), self.pred(y, ctx)?.into()];
                let inst = Inst::new(opcode, Ty::Pred, Vec::new(), operands);
                return Ok(self.b.value(inst, Bank::Pred));
            }
        }
        if let Expr::Cast(target, inner) = &**expression
            && self.cast_target(target)? == VTy::Bool
        {
            return self.pred(inner, ctx);
        }
        let ty = self.operand_type(expression);
        let val = match &**expression {
            Expr::Var(name) => self.lookup(*name)?,
            _ if ty == VTy::Bool => {
                return Err(self.no_rule(format_args!("the condition {expression}")));
            }
            _ => self.typed(expression, ty, ctx)?,
        };
        self.truth(&val)
    }

    fn compare(
        &mut self,
        cmp: Cmp,
        x: &Expression,
        y: &Expression,
        ctx: &SimplifyCtx,
    ) -> Result<Reg> {
        let ty = self.infer(x).or_else(|| self.infer(y));
        let ty = ty.unwrap_or_else(|| self.operand_type(x).max_float(self.operand_type(y)));
        if ty == VTy::Bf16 {
            return Err(self.no_rule("a comparison at bf16"));
        }
        if ty.is_float() {
            // A float differs from another where they are unordered too, as `!=` has it.
            let cmp = if cmp == Cmp::Ne { Cmp::Neu } else { cmp };
            let operands = vec![self.float(x, ty, ctx)?, self.float(y, ty, ctx)?];
            let inst = Inst::new(Opcode::Setp(cmp), ty.arith(), Vec::new(), operands)
                .flushing(self.precision.ftz && ty == VTy::F32);
            return Ok(self.b.value(inst, Bank::Pred));
        }
        if ty == VTy::Bool {
            return Err(self.no_rule("a comparison of bools"));
        }
        self.int_compare(cmp, x, y, ctx)
    }

    /// The value of `inner` converted to `to`.
    pub(super) fn cast(&mut self, inner: &Expression, to: VTy, ctx: &SimplifyCtx) -> Result<Val> {
        let from = self.operand_type(inner);
        let converted =
            |this: &mut Lowerer, from_ty: Ty, operand: Operand, round: Option<Round>, ftz: bool| {
                let inst = Inst::new(Opcode::Cvt(from_ty), to.arith(), Vec::new(), vec![operand])
                    .flushing(ftz);
                let inst = match round {
                    Some(round) => inst.rounded(round),
                    None => inst,
                };
                Val::reg(this.b.value(inst, to.bank()), to)
            };
        match (from, to) {
            _ if from == to || (from.is_int() && to.is_int()) => self.typed(inner, to, ctx),
            (_, VTy::Bool) => Ok(Val::reg(self.pred(inner, ctx)?, VTy::Bool)),
            (VTy::Bool, _) => {
                let pred = self.pred(inner, ctx)?;
                let (one, zero) = if to.is_float() {
                    (self.float_constant(to, 1.0), self.float_constant(to, 0.0))
                } else {
                    (Operand::Int(1), Operand::Int(0))
                };
                let inst = Inst::new(
                    Opcode::Selp,
                    to.held(),
                    Vec::new(),
                    vec![one, zero, pred.into()],
                );
                Ok(Val::reg(self.b.value(inst, to.bank()), to))
            }
            // A half widens exactly; a narrowing rounds to nearest.
            (VTy::F16 | VTy::Bf16, VTy::F32) => {
                let operand = self.float(inner, from, ctx)?;
                Ok(converted(self, from.arith(), operand, None, false))
            }
            (VTy::F32, VTy::F16 | VTy::Bf16) => {
                let operand = self.float(inner, from, ctx)?;
                Ok(converted(self, Ty::F32, operand, Some(Round::Rn), false))
            }
            (_, _) if from.is_float() && to.is_float() => {
                Err(self.no_rule(format_args!("a cast from {} to {}", from.text(), to.text())))
            }
            // A float truncates toward zero into an integer, as a C cast does.
            (_, _) if from.is_float() => {
                let operand = self.float(inner, from, ctx)?;
                let ftz = self.precision.ftz && from == VTy::F32;
                Ok(converted(
                    self,
                    from.arith(),
                    operand,
                    Some(Round::Rzi),
                    ftz,
                ))
            }
            _ => {
                let val = self.int(inner, ctx)?;
                match val.op.int() {
                    Some(constant) => Ok(Val {
                        op: self.float_constant(to, constant as f64),
                        ty: to,
                    }),
                    None => Ok(converted(
                        self,
                        val.ty.arith(),
                        val.op,
                        Some(Round::Rn),
                        false,
                    )),
                }
            }
        }
    }
}

impl VTy {
    /// The float type of the two, when one is a float: what a comparison of a float with a literal runs at.
    fn max_float(self, other: VTy) -> VTy {
        if self.is_float() { self } else { other }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A half constant is the nearest value of its format, ties to even, with the format's subnormals, infinities
    /// and signed zero.
    #[test]
    fn a_half_constant_rounds_to_nearest_even() {
        let f16 = |value: f64| half_bits(VTy::F16, value);
        assert_eq!(f16(1.0), 0x3c00);
        assert_eq!(f16(-2.0), 0xc000);
        assert_eq!(f16(0.0), 0x0000);
        assert_eq!(f16(-0.0), 0x8000);
        assert_eq!(f16(65504.0), 0x7bff);
        assert_eq!(f16(65519.9), 0x7bff);
        assert_eq!(f16(65520.0), 0x7c00);
        assert_eq!(f16(-1e30), 0xfc00);
        assert_eq!(f16(f64::INFINITY), 0x7c00);
        assert_eq!(f16(2f64.powi(-24)), 0x0001);
        assert_eq!(f16(2f64.powi(-25)), 0x0000);
        assert_eq!(f16(1.5 * 2f64.powi(-24)), 0x0002);
        assert_eq!(f16(2f64.powi(-14)), 0x0400);
        // 1 + 2^-11 is halfway between two halves: it rounds to the even one, and just above it rounds up.
        assert_eq!(f16(1.0 + 2f64.powi(-11)), 0x3c00);
        assert_eq!(f16(1.0 + 2f64.powi(-11) + 2f64.powi(-30)), 0x3c01);
        assert_eq!(f16(f64::NAN) & 0x7e00, 0x7e00);
        let bf16 = |value: f64| half_bits(VTy::Bf16, value);
        assert_eq!(bf16(1.0), 0x3f80);
        assert_eq!(bf16(-0.5), 0xbf00);
        assert_eq!(bf16(1e30), (1e30f32.to_bits() >> 16) as u16 + 1);
        assert_eq!(bf16(f64::INFINITY), 0x7f80);
    }
}
