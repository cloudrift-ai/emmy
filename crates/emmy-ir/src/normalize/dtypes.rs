//! The dtype a body's names compute at, read from the body alone: a load's declared dtype (the op types every load
//! from its input at construction), a typed definition, else the table's promotion of the operands a definition
//! reads. A name no rule reaches is unknown, and a rule that needs it does the conservative thing: the spelling of a
//! literal in a sum, and whether an identity element folds away, are decided by the sum's operands' types where
//! they are known, and by the literal's own kind where they are not.

use std::collections::HashMap;

use crate::Name;
use crate::body::Body;
use crate::dtype::{DType, promote};
use crate::elementwise::Operation;
use crate::expr::{CastTarget, Expr, Expression, LitType};
use crate::statement::Statement;

/// What is known about the type a value computes at.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Typed {
    Known(DType),
    /// A value of literals alone: it takes the type of whatever it meets, as a literal does.
    Literal(LitType),
    Unknown,
}

/// The kind a sum computes in, from its operands: integer arithmetic stays integer, a float operand makes it float,
/// and an operand nobody typed leaves the question to the literals.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SumKind {
    Integer,
    Float,
    Unknown,
}

#[derive(Default)]
pub struct DTypes(HashMap<Name, DType>);

impl DTypes {
    pub fn of(&self, name: &Name) -> Option<DType> {
        self.0.get(name).copied()
    }

    /// The type `e` computes at: a name's, a literal's kind, or the table's promotion of an operation's typed
    /// operands — a float literal over integer operands alone makes the operation a float one, as the render
    /// computes it.
    pub fn value(&self, e: &Expression) -> Typed {
        match &**e {
            Expr::Var(name) => self.of(name).map_or(Typed::Unknown, Typed::Known),
            Expr::Literal(lit) => Typed::Literal(lit.dtype),
            Expr::Cast(target, _) => Typed::Known(match target {
                CastTarget::Known(LitType::Int) => DType::I32,
                CastTarget::Known(_) => DType::F32,
                CastTarget::Other(name) => match name.parse::<DType>() {
                    Ok(dtype) => dtype,
                    Err(_) => return Typed::Unknown,
                },
            }),
            Expr::Apply(op, args) => self.promoted(op.known(), args),
            Expr::Ternary(_, if_true, if_false) => {
                self.promoted(Some(Operation::Add), &[if_true.clone(), if_false.clone()])
            }
            Expr::Builtin(_) | Expr::FlatIndex(..) => Typed::Unknown,
        }
    }

    fn promoted(&self, op: Option<Operation>, args: &[Expression]) -> Typed {
        let mut known = Vec::new();
        let mut float_literal = false;
        for arg in args {
            match self.value(arg) {
                Typed::Unknown => return Typed::Unknown,
                Typed::Known(dtype) => known.push(dtype),
                Typed::Literal(kind) => float_literal |= kind == LitType::Float,
            }
        }
        if known.is_empty() {
            return Typed::Literal(if float_literal {
                LitType::Float
            } else {
                LitType::Int
            });
        }
        if float_literal && known.iter().all(|d| d.is_integer()) {
            return Typed::Known(DType::F32);
        }
        Typed::Known(promote(op, &known))
    }

    /// The kind a sum over `operands` computes in.
    pub fn sum_kind(&self, operands: &[Expression]) -> SumKind {
        let mut kind = SumKind::Unknown;
        for operand in operands {
            match self.value(operand) {
                Typed::Known(dtype) if dtype.is_integer() || dtype == DType::Bool => {
                    if kind == SumKind::Unknown {
                        kind = SumKind::Integer;
                    }
                }
                Typed::Known(_) => return SumKind::Float,
                Typed::Literal(LitType::Float) => return SumKind::Float,
                Typed::Literal(_) => {}
                Typed::Unknown => return SumKind::Unknown,
            }
        }
        kind
    }
}

/// The dtypes of every name `body` defines, in any scope, as far as the body says: passes until nothing new is
/// learned, since a carry reads a value its scope defines later.
pub fn dtypes_of(body: &Body) -> DTypes {
    let mut known = DTypes::default();
    for _ in 0..4 {
        let before = known.0.len();
        body.walk(&mut |here| {
            if let Some((name, dtype)) = defined(here.stmt(), &known) {
                known.0.insert(name, dtype);
            }
        });
        if known.0.len() == before {
            break;
        }
    }
    known
}

fn defined(stmt: &Statement, known: &DTypes) -> Option<(Name, DType)> {
    let of_value = |value: &Expression| match known.value(value) {
        Typed::Known(dtype) => Some(dtype),
        Typed::Literal(LitType::Float) => Some(DType::F32),
        Typed::Literal(LitType::Int) => Some(DType::I32),
        Typed::Literal(LitType::Bool) => Some(DType::Bool),
        Typed::Unknown => None,
    };
    match stmt {
        Statement::Load(l) => l.dtype.map(|d| (l.names[0], d)),
        Statement::Assign(a) => a.dtype.or_else(|| of_value(&a.value)).map(|d| (a.name, d)),
        Statement::Let(l) => l.dtype.or_else(|| of_value(&l.value)).map(|d| (l.name, d)),
        Statement::Accum(a) => a.dtype.or_else(|| of_value(&a.value)).map(|d| (a.name, d)),
        Statement::Carry(c) => c.dtype.or_else(|| known.of(&c.value)).map(|d| (c.name, d)),
        Statement::Pre(p) => known.of(&p.carrier).map(|d| (p.name, d)),
        Statement::Init(i) => Some((i.name, i.dtype)),
        Statement::Select(s) => {
            let branches: Option<Vec<DType>> =
                s.branches.iter().map(|b| known.of(&b.value)).collect();
            branches
                .filter(|b| !b.is_empty())
                .map(|b| (s.name, promote(Some(Operation::Add), &b)))
        }
        _ => None,
    }
}
