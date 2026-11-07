//! Expressions: the coordinate and predicate sublanguage every IR layer indexes and guards with.
//!
//! An expression is a shared, immutable tree ([`Expression`] is an `Arc`). Its text is its one form: on the wire,
//! in every listing and as its structural identity ([`Expression::text`] and [`parse`] are exact inverses). The
//! simplifier ([`Expression::simplify`]) folds literals, drops algebraic identities and decides static comparisons
//! under the ranges a [`SimplifyCtx`] carries; the affine form ([`affine_form`]) reads an index as anchor plus
//! integer coefficients.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::{Error, Name, Result};

// --- numbers --------------------------------------------------------------------------------------

/// The value a literal holds, as Python holds it: an int, a float or a bool. Two numbers are equal when their values
/// are (`2 == 2.0 == True + True`), as in Python.
#[derive(Clone, Copy, Debug)]
pub enum Num {
    Int(i64),
    Float(f64),
    Bool(bool),
}

impl Num {
    pub fn as_f64(self) -> f64 {
        match self {
            Num::Int(v) => v as f64,
            Num::Float(v) => v,
            Num::Bool(v) => v as i64 as f64,
        }
    }

    /// Python's `int(value)`: a float truncates toward zero.
    pub fn as_int(self) -> Result<i64> {
        match self {
            Num::Int(v) => Ok(v),
            Num::Bool(v) => Ok(v as i64),
            Num::Float(v) if v.is_finite() => Ok(v.trunc() as i64),
            Num::Float(v) => Err(Error::Value(format!("cannot convert float {v} to integer"))),
        }
    }

    /// Python's `isinstance(value, int)`: an int or a bool.
    pub fn is_integral(self) -> bool {
        !matches!(self, Num::Float(_))
    }

    /// An int that is no bool: what gives a literal a range.
    pub fn int(self) -> Option<i64> {
        match self {
            Num::Int(v) => Some(v),
            _ => None,
        }
    }

    pub fn truthy(self) -> bool {
        self.as_f64() != 0.0
    }
}

impl PartialEq for Num {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Num::Int(a), Num::Int(b)) => a == b,
            (Num::Float(a), Num::Float(b)) => a == b,
            (Num::Bool(a), Num::Bool(b)) => a == b,
            _ => self.as_f64() == other.as_f64(),
        }
    }
}

impl Eq for Num {}

impl Hash for Num {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_f64(self.as_f64(), state);
    }
}

/// Hash a float the way its equality compares: `0.0` and `-0.0` alike, every NaN alike.
pub fn hash_f64<H: Hasher>(value: f64, state: &mut H) {
    let value = if value == 0.0 { 0.0 } else { value };
    if value.is_nan() {
        u64::MAX.hash(state)
    } else {
        value.to_bits().hash(state)
    }
}

/// Python's `repr(float)`: the shortest digits that read back, positional for exponents in `-4..16`, else `d.de±XX`.
pub fn float_repr(value: f64) -> String {
    if value.is_nan() {
        return "nan".into();
    }
    if value.is_infinite() {
        return if value < 0.0 {
            "-inf".into()
        } else {
            "inf".into()
        };
    }
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    let sci = format!("{value:e}");
    let (mantissa, exponent) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let negative = mantissa.starts_with('-');
    let digits: String = mantissa.chars().filter(|c| c.is_ascii_digit()).collect();
    let mut out = String::new();
    if negative {
        out.push('-');
    }
    if (-4..16).contains(&exponent) {
        if exponent >= 0 {
            let point = exponent as usize + 1;
            if digits.len() <= point {
                out.push_str(&digits);
                out.extend(std::iter::repeat_n('0', point - digits.len()));
                out.push_str(".0");
            } else {
                out.push_str(&digits[..point]);
                out.push('.');
                out.push_str(&digits[point..]);
            }
        } else {
            out.push_str("0.");
            out.extend(std::iter::repeat_n('0', (-exponent - 1) as usize));
            out.push_str(&digits);
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let _ = write!(
            out,
            "e{}{:02}",
            if exponent < 0 { '-' } else { '+' },
            exponent.abs()
        );
    }
    out
}

// --- operators ------------------------------------------------------------------------------------

/// A binary operator and how tightly it binds (higher binds tighter).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BinOp {
    Or,
    And,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    Xor,
    BitAnd,
    BitOr,
    Shl,
    Shr,
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Mod,
}

impl BinOp {
    pub const ALL: [BinOp; 19] = [
        BinOp::Or,
        BinOp::And,
        BinOp::Eq,
        BinOp::Ne,
        BinOp::Lt,
        BinOp::Gt,
        BinOp::Le,
        BinOp::Ge,
        BinOp::Xor,
        BinOp::BitAnd,
        BinOp::BitOr,
        BinOp::Shl,
        BinOp::Shr,
        BinOp::Add,
        BinOp::Sub,
        BinOp::Mul,
        BinOp::Div,
        BinOp::FloorDiv,
        BinOp::Mod,
    ];

    pub fn text(self) -> &'static str {
        match self {
            BinOp::Or => "||",
            BinOp::And => "&&",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::Lt => "<",
            BinOp::Gt => ">",
            BinOp::Le => "<=",
            BinOp::Ge => ">=",
            BinOp::Xor => "^",
            BinOp::BitAnd => "&",
            BinOp::BitOr => "|",
            BinOp::Shl => "<<",
            BinOp::Shr => ">>",
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::FloorDiv => "//",
            BinOp::Mod => "%",
        }
    }

    pub fn parse(text: &str) -> Option<BinOp> {
        BinOp::ALL.into_iter().find(|op| op.text() == text)
    }

    pub fn precedence(self) -> u8 {
        match self {
            BinOp::Or => 1,
            BinOp::And => 2,
            BinOp::Eq | BinOp::Ne => 3,
            BinOp::Lt
            | BinOp::Gt
            | BinOp::Le
            | BinOp::Ge
            | BinOp::Xor
            | BinOp::BitAnd
            | BinOp::BitOr
            | BinOp::Shl
            | BinOp::Shr => 4,
            BinOp::Add | BinOp::Sub => 5,
            BinOp::Mul | BinOp::Div | BinOp::FloorDiv | BinOp::Mod => 6,
        }
    }

    pub fn is_division(self) -> bool {
        matches!(self, BinOp::Div | BinOp::FloorDiv)
    }

    pub fn is_comparison(self) -> bool {
        matches!(
            self,
            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq
        )
    }
}

/// Python's `apply_binop` over two evaluated numbers: integer floor division for `/` and `//`, logical `&&` / `||`
/// on truth values, a bool for every comparison.
pub fn apply_binop(op: BinOp, left: Num, right: Num) -> Result<Num> {
    fn arith(
        left: Num,
        right: Num,
        int: impl Fn(i64, i64) -> Option<i64>,
        float: impl Fn(f64, f64) -> f64,
    ) -> Result<Num> {
        if left.is_integral() && right.is_integral() {
            let (a, b) = (left.as_int()?, right.as_int()?);
            return int(a, b)
                .map(Num::Int)
                .ok_or_else(|| Error::Value("integer overflow".into()));
        }
        Ok(Num::Float(float(left.as_f64(), right.as_f64())))
    }
    fn ints(left: Num, right: Num) -> Result<(i64, i64)> {
        if !(left.is_integral() && right.is_integral()) {
            return Err(Error::Type(
                "unsupported operand type(s) for a bit operation: float".into(),
            ));
        }
        Ok((left.as_int()?, right.as_int()?))
    }
    fn bits(
        left: Num,
        right: Num,
        bools: impl Fn(bool, bool) -> bool,
        ints_fn: impl Fn(i64, i64) -> Option<i64>,
    ) -> Result<Num> {
        if let (Num::Bool(a), Num::Bool(b)) = (left, right) {
            return Ok(Num::Bool(bools(a, b)));
        }
        let (a, b) = ints(left, right)?;
        ints_fn(a, b)
            .map(Num::Int)
            .ok_or_else(|| Error::Value("integer overflow".into()))
    }
    let floor_div = |a: i64, b: i64| -> Result<i64> {
        if b == 0 {
            return Err(Error::ZeroDivision);
        }
        let q = a / b;
        Ok(if a % b != 0 && ((a < 0) != (b < 0)) {
            q - 1
        } else {
            q
        })
    };
    let py_mod = |a: i64, b: i64| -> Result<i64> {
        if b == 0 {
            return Err(Error::ZeroDivision);
        }
        let r = a % b;
        Ok(if r != 0 && ((r < 0) != (b < 0)) {
            r + b
        } else {
            r
        })
    };
    match op {
        BinOp::Add => arith(left, right, |a, b| a.checked_add(b), |a, b| a + b),
        BinOp::Sub => arith(left, right, |a, b| a.checked_sub(b), |a, b| a - b),
        BinOp::Mul => arith(left, right, |a, b| a.checked_mul(b), |a, b| a * b),
        BinOp::Div | BinOp::FloorDiv => Ok(Num::Int(floor_div(left.as_int()?, right.as_int()?)?)),
        BinOp::Mod => Ok(Num::Int(py_mod(left.as_int()?, right.as_int()?)?)),
        BinOp::Lt => Ok(Num::Bool(left.as_f64() < right.as_f64())),
        BinOp::Le => Ok(Num::Bool(left.as_f64() <= right.as_f64())),
        BinOp::Gt => Ok(Num::Bool(left.as_f64() > right.as_f64())),
        BinOp::Ge => Ok(Num::Bool(left.as_f64() >= right.as_f64())),
        BinOp::Eq => Ok(Num::Bool(left.as_f64() == right.as_f64())),
        BinOp::And => Ok(Num::Bool(left.truthy() && right.truthy())),
        BinOp::Or => Ok(Num::Bool(left.truthy() || right.truthy())),
        BinOp::BitAnd => bits(left, right, |a, b| a & b, |a, b| Some(a & b)),
        BinOp::BitOr => bits(left, right, |a, b| a | b, |a, b| Some(a | b)),
        BinOp::Xor => bits(left, right, |a, b| a ^ b, |a, b| Some(a ^ b)),
        BinOp::Shl => {
            let (a, b) = ints(left, right)?;
            if !(0..64).contains(&b) {
                return Err(Error::Value("shift out of range".into()));
            }
            a.checked_shl(b as u32)
                .map(Num::Int)
                .ok_or_else(|| Error::Value("integer overflow".into()))
        }
        BinOp::Shr => {
            let (a, b) = ints(left, right)?;
            if b < 0 {
                return Err(Error::Value("negative shift count".into()));
            }
            Ok(Num::Int(if b >= 64 {
                if a < 0 { -1 } else { 0 }
            } else {
                a >> b
            }))
        }
        BinOp::Ne => Err(Error::Value("Unknown BinOp: !=".into())),
    }
}

// --- the tree -------------------------------------------------------------------------------------

/// A numeric constant: its value and the dtype it is spelled by (`int`, `float` or `bool`).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Lit {
    pub value: Num,
    pub dtype: Name,
}

/// One node of an expression tree.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Expr {
    Var(Name),
    Literal(Lit),
    Builtin(Name),
    Binary(BinOp, Expression, Expression),
    Call(Name, Arc<[Expression]>),
    Ternary(Expression, Expression, Expression),
    Cast(Name, Expression),
    FlatIndex(Name, Arc<[Expression]>),
}

/// A shared, immutable expression.
#[derive(Clone, Debug)]
pub struct Expression(pub Arc<Expr>);

impl PartialEq for Expression {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0) || *self.0 == *other.0
    }
}

impl Eq for Expression {}

impl Hash for Expression {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl std::ops::Deref for Expression {
    type Target = Expr;

    fn deref(&self) -> &Expr {
        &self.0
    }
}

impl std::fmt::Display for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.text())
    }
}

/// Every GPU built-in's name. The reader tells a built-in from a variable by this set, so no variable takes one.
pub const BUILTINS: [&str; 13] = [
    "thread_idx.x",
    "thread_idx.y",
    "thread_idx.z",
    "block_idx.x",
    "block_idx.y",
    "block_idx.z",
    "block_dim.x",
    "block_dim.y",
    "block_dim.z",
    "grid_dim.x",
    "grid_dim.y",
    "grid_dim.z",
    "warp_size",
];

pub fn is_builtin(name: &str) -> bool {
    BUILTINS.contains(&name)
}

/// Names the reader gives a meaning of their own, so a variable of that name is written quoted.
const RESERVED: [&str; 4] = ["true", "false", "inf", "nan"];

pub const PLACEHOLDER_PREFIX: &str = "out_coord_";

impl Expression {
    pub fn new(expr: Expr) -> Expression {
        Expression(Arc::new(expr))
    }

    pub fn var(name: &str) -> Expression {
        Expression::new(Expr::Var(name.into()))
    }

    pub fn int(value: i64) -> Expression {
        Expression::new(Expr::Literal(Lit {
            value: Num::Int(value),
            dtype: "int".into(),
        }))
    }

    pub fn float(value: f64) -> Expression {
        Expression::new(Expr::Literal(Lit {
            value: Num::Float(value),
            dtype: "float".into(),
        }))
    }

    pub fn literal(value: Num, dtype: &str) -> Expression {
        Expression::new(Expr::Literal(Lit {
            value,
            dtype: dtype.into(),
        }))
    }

    pub fn binary(op: BinOp, left: Expression, right: Expression) -> Expression {
        Expression::new(Expr::Binary(op, left, right))
    }

    /// The placeholder variable for output coordinate axis `d`.
    pub fn placeholder(d: usize) -> Expression {
        Expression::var(&format!("{PLACEHOLDER_PREFIX}{d}"))
    }

    pub fn is_placeholder(&self, d: Option<usize>) -> bool {
        match &**self {
            Expr::Var(name) => match d {
                None => name.starts_with(PLACEHOLDER_PREFIX),
                Some(d) => **name == *format!("{PLACEHOLDER_PREFIX}{d}"),
            },
            _ => false,
        }
    }

    pub fn as_literal(&self) -> Option<&Lit> {
        match &**self {
            Expr::Literal(lit) => Some(lit),
            _ => None,
        }
    }

    pub fn as_var(&self) -> Option<&Name> {
        match &**self {
            Expr::Var(name) => Some(name),
            _ => None,
        }
    }

    /// The literal's value when the expression is one.
    pub fn literal_value(&self) -> Option<Num> {
        self.as_literal().map(|lit| lit.value)
    }

    /// A positive int literal of dtype `int` (what the simplifier takes a divisor or a factor from).
    pub fn int_literal(&self) -> Option<i64> {
        match &**self {
            Expr::Literal(Lit { value, dtype }) if &**dtype == "int" && value.is_integral() => {
                value.as_int().ok()
            }
            _ => None,
        }
    }

    pub fn is_zero(&self) -> bool {
        self.literal_value().is_some_and(|v| v.as_f64() == 0.0)
    }

    pub fn is_one(&self) -> bool {
        self.literal_value().is_some_and(|v| v.as_f64() == 1.0)
    }

    pub fn is_truthy(&self) -> bool {
        self.literal_value().is_some_and(|v| v.truthy())
    }

    pub fn is_falsy(&self) -> bool {
        self.literal_value().is_some_and(|v| !v.truthy())
    }

    // --- the text ---------------------------------------------------------------------------------

    /// The expression's text: `parse`'s inverse, what it is on the wire and in every listing. An expression with no
    /// text (an operator with no spelling, a name holding a backtick) renders as its debug form.
    pub fn text(&self) -> String {
        self.try_text().unwrap_or_else(|_| format!("{:?}", self.0))
    }

    /// The text, or the reason the expression has none (an operator with no spelling, a name with a backtick).
    pub fn try_text(&self) -> Result<String> {
        let mut out = String::new();
        self.spell(0, &mut out)?;
        Ok(out)
    }

    fn spell(&self, parent: u8, out: &mut String) -> Result<()> {
        match &**self {
            Expr::Var(name) => out.push_str(&spell_name(name, &[])?),
            Expr::Builtin(name) => {
                if !is_builtin(name) {
                    return Err(Error::Value(format!("{name:?} is no GPU built-in")));
                }
                out.push_str(name);
            }
            Expr::Literal(lit) => {
                let text = literal_text(lit)?;
                if text.starts_with('-') && parent == ATOM {
                    let _ = write!(out, "({text})");
                } else {
                    out.push_str(&text);
                }
            }
            Expr::Binary(op, left, right) => {
                let prec = op.precedence();
                let paren = prec < parent;
                if paren {
                    out.push('(');
                }
                left.spell(prec, out)?;
                let _ = write!(out, " {} ", op.text());
                right.spell(prec + 1, out)?;
                if paren {
                    out.push(')');
                }
            }
            Expr::Call(name, args) => {
                let _ = write!(out, "{name}(");
                for (index, arg) in args.iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    arg.spell(0, out)?;
                }
                out.push(')');
            }
            Expr::Ternary(cond, if_true, if_false) => {
                if parent > 0 {
                    out.push('(');
                }
                cond.spell(1, out)?;
                out.push_str(" ? ");
                if_true.spell(0, out)?;
                out.push_str(" : ");
                if_false.spell(0, out)?;
                if parent > 0 {
                    out.push(')');
                }
            }
            Expr::Cast(dtype, inner) => {
                let _ = write!(out, "({dtype})");
                inner.spell(ATOM, out)?;
            }
            Expr::FlatIndex(buffer, index) => {
                out.push_str(&spell_name(buffer, &[])?);
                out.push('[');
                for (position, e) in index.iter().enumerate() {
                    if position > 0 {
                        out.push_str(", ");
                    }
                    e.spell(0, out)?;
                }
                out.push(']');
            }
        }
        Ok(())
    }

    // --- walks ------------------------------------------------------------------------------------

    /// The expression's children, in field order.
    pub fn children(&self) -> Vec<Expression> {
        match &**self {
            Expr::Var(_) | Expr::Literal(_) | Expr::Builtin(_) => vec![],
            Expr::Binary(_, left, right) => vec![left.clone(), right.clone()],
            Expr::Call(_, args) | Expr::FlatIndex(_, args) => args.to_vec(),
            Expr::Ternary(a, b, c) => vec![a.clone(), b.clone(), c.clone()],
            Expr::Cast(_, inner) => vec![inner.clone()],
        }
    }

    /// This node over `children` (as many as `children` returned), itself when none changed.
    pub fn with_children(&self, children: Vec<Expression>) -> Result<Expression> {
        let same = |items: &[Expression]| {
            items
                .iter()
                .zip(&children)
                .all(|(old, new)| Arc::ptr_eq(&old.0, &new.0))
        };
        let count = self.children().len();
        if children.len() != count {
            return Err(Error::Value(format!(
                "an expression with {count} children cannot take {}",
                children.len()
            )));
        }
        Ok(match &**self {
            Expr::Var(_) | Expr::Literal(_) | Expr::Builtin(_) => self.clone(),
            Expr::Binary(op, l, r) if same(&[l.clone(), r.clone()]) => self.clone(),
            Expr::Binary(op, _, _) => {
                Expression::new(Expr::Binary(*op, children[0].clone(), children[1].clone()))
            }
            Expr::Call(name, args) => {
                if same(args) {
                    self.clone()
                } else {
                    Expression::new(Expr::Call(name.clone(), children.into()))
                }
            }
            Expr::FlatIndex(name, args) => {
                if same(args) {
                    self.clone()
                } else {
                    Expression::new(Expr::FlatIndex(name.clone(), children.into()))
                }
            }
            Expr::Ternary(a, b, c) if same(&[a.clone(), b.clone(), c.clone()]) => self.clone(),
            Expr::Ternary(..) => Expression::new(Expr::Ternary(
                children[0].clone(),
                children[1].clone(),
                children[2].clone(),
            )),
            Expr::Cast(dtype, inner) => {
                if same(std::slice::from_ref(inner)) {
                    self.clone()
                } else {
                    Expression::new(Expr::Cast(dtype.clone(), children[0].clone()))
                }
            }
        })
    }

    /// This node with `f` applied to each child; itself when every child comes back as it was.
    pub fn map_children(&self, f: &mut dyn FnMut(&Expression) -> Expression) -> Expression {
        let same = |old: &Expression, new: &Expression| Arc::ptr_eq(&old.0, &new.0);
        match &**self {
            Expr::Var(_) | Expr::Literal(_) | Expr::Builtin(_) => self.clone(),
            Expr::Binary(op, l, r) => {
                let (nl, nr) = (f(l), f(r));
                if same(l, &nl) && same(r, &nr) {
                    self.clone()
                } else {
                    Expression::new(Expr::Binary(*op, nl, nr))
                }
            }
            Expr::Call(name, args) => {
                let new: Vec<Expression> = args.iter().map(&mut *f).collect();
                if new.iter().zip(args.iter()).all(|(n, o)| same(o, n)) {
                    self.clone()
                } else {
                    Expression::new(Expr::Call(name.clone(), new.into()))
                }
            }
            Expr::FlatIndex(name, args) => {
                let new: Vec<Expression> = args.iter().map(&mut *f).collect();
                if new.iter().zip(args.iter()).all(|(n, o)| same(o, n)) {
                    self.clone()
                } else {
                    Expression::new(Expr::FlatIndex(name.clone(), new.into()))
                }
            }
            Expr::Ternary(a, b, c) => {
                let (na, nb, nc) = (f(a), f(b), f(c));
                if same(a, &na) && same(b, &nb) && same(c, &nc) {
                    self.clone()
                } else {
                    Expression::new(Expr::Ternary(na, nb, nc))
                }
            }
            Expr::Cast(dtype, inner) => {
                let new = f(inner);
                if same(inner, &new) {
                    self.clone()
                } else {
                    Expression::new(Expr::Cast(dtype.clone(), new))
                }
            }
        }
    }

    /// Pre-order walk over the tree, itself first.
    pub fn subterms(&self) -> Vec<Expression> {
        let mut out = Vec::new();
        fn walk(e: &Expression, out: &mut Vec<Expression>) {
            out.push(e.clone());
            match &**e {
                Expr::Var(_) | Expr::Literal(_) | Expr::Builtin(_) => {}
                Expr::Binary(_, l, r) => {
                    walk(l, out);
                    walk(r, out);
                }
                Expr::Call(_, args) | Expr::FlatIndex(_, args) => {
                    args.iter().for_each(|a| walk(a, out))
                }
                Expr::Ternary(a, b, c) => {
                    walk(a, out);
                    walk(b, out);
                    walk(c, out);
                }
                Expr::Cast(_, inner) => walk(inner, out),
            }
        }
        walk(self, &mut out);
        out
    }

    /// Every variable name the expression reads, each once, in first appearance.
    pub fn free_vars(&self) -> Vec<Name> {
        let mut out = Vec::new();
        self.collect_free_vars(&mut out);
        out
    }

    pub fn collect_free_vars(&self, out: &mut Vec<Name>) {
        match &**self {
            Expr::Var(name) => {
                if !out.iter().any(|n| n == name) {
                    out.push(name.clone());
                }
            }
            Expr::Literal(_) | Expr::Builtin(_) => {}
            Expr::Binary(_, l, r) => {
                l.collect_free_vars(out);
                r.collect_free_vars(out);
            }
            Expr::Call(_, args) | Expr::FlatIndex(_, args) => {
                args.iter().for_each(|a| a.collect_free_vars(out))
            }
            Expr::Ternary(a, b, c) => {
                a.collect_free_vars(out);
                b.collect_free_vars(out);
                c.collect_free_vars(out);
            }
            Expr::Cast(_, inner) => inner.collect_free_vars(out),
        }
    }

    pub fn reads(&self, name: &str) -> bool {
        match &**self {
            Expr::Var(n) => &**n == name,
            Expr::Literal(_) | Expr::Builtin(_) => false,
            Expr::Binary(_, l, r) => l.reads(name) || r.reads(name),
            Expr::Call(_, args) | Expr::FlatIndex(_, args) => args.iter().any(|a| a.reads(name)),
            Expr::Ternary(a, b, c) => a.reads(name) || b.reads(name) || c.reads(name),
            Expr::Cast(_, inner) => inner.reads(name),
        }
    }

    /// `Var(name)` subterms replaced by `mapping[name]`; names the mapping lacks pass through.
    pub fn substitute(&self, mapping: &HashMap<Name, Expression>) -> Expression {
        if mapping.is_empty() {
            return self.clone();
        }
        match &**self {
            Expr::Var(name) => mapping.get(name).cloned().unwrap_or_else(|| self.clone()),
            Expr::Literal(_) | Expr::Builtin(_) => self.clone(),
            _ => self.map_children(&mut |c| c.substitute(mapping)),
        }
    }

    /// Every free variable's name mapped through `rename`; a name the function keeps passes through.
    pub fn rename(&self, rename: &mut dyn FnMut(&str) -> Option<Name>) -> Expression {
        match &**self {
            Expr::Var(name) => match rename(name) {
                Some(new) if new != *name => Expression::new(Expr::Var(new)),
                _ => self.clone(),
            },
            Expr::Literal(_) | Expr::Builtin(_) => self.clone(),
            _ => self.map_children(&mut |c| c.rename(rename)),
        }
    }

    /// `rename` by a mapping: a name absent from it stays.
    pub fn rename_by(&self, mapping: &HashMap<Name, Name>) -> Expression {
        if mapping.is_empty() {
            return self.clone();
        }
        self.rename(&mut |name| mapping.get(name).cloned())
    }

    /// Bottom-up map: `f` applied to every subterm once its children are rebuilt.
    pub fn rebuild(&self, f: &mut dyn FnMut(Expression) -> Expression) -> Expression {
        let rebuilt = self.map_children(&mut |c| c.rebuild(f));
        f(rebuilt)
    }

    /// The buffers the flat indices inside this expression name, each once.
    pub fn flat_buffers(&self) -> Vec<Name> {
        let mut out = Vec::new();
        for term in self.subterms() {
            if let Expr::FlatIndex(buffer, _) = &*term
                && !out.contains(buffer)
            {
                out.push(buffer.clone());
            }
        }
        out
    }

    /// Evaluate a closed expression (no variables, no built-ins).
    pub fn eval(&self, env: &HashMap<Name, Num>) -> Result<Num> {
        match &**self {
            Expr::Var(name) => env
                .get(name)
                .copied()
                .ok_or_else(|| Error::Key(name.to_string())),
            Expr::Literal(lit) => Ok(lit.value),
            Expr::Builtin(name) => Err(Error::NotImplemented(format!(
                "Builtin {name:?} is GPU-only; cannot eval"
            ))),
            Expr::Binary(op, l, r) => apply_binop(*op, l.eval(env)?, r.eval(env)?),
            Expr::Call(name, _) => Err(Error::NotImplemented(format!(
                "no host evaluation of {name}"
            ))),
            Expr::Ternary(c, t, f) => {
                if c.eval(env)?.truthy() {
                    t.eval(env)
                } else {
                    f.eval(env)
                }
            }
            Expr::Cast(dtype, inner) => {
                let value = inner.eval(env)?;
                Ok(if &**dtype == "int" {
                    Num::Int(value.as_int()?)
                } else {
                    value
                })
            }
            Expr::FlatIndex(..) => Err(Error::Type("FlatIndex has no host evaluation".into())),
        }
    }

    // --- simplification ---------------------------------------------------------------------------

    /// Conservative integer range, `None` when unknown.
    pub fn range(&self, ctx: &SimplifyCtx) -> Option<Interval> {
        match &**self {
            Expr::Var(name) => ctx.ranges.get(name).copied(),
            Expr::Literal(lit) => lit.value.int().map(|v| Interval::new(v as i128, v as i128)),
            Expr::Binary(op, left, right) => {
                let (la, lb) = (left.range(ctx), right.range(ctx));
                let logical = matches!(
                    op,
                    BinOp::Lt
                        | BinOp::Le
                        | BinOp::Gt
                        | BinOp::Ge
                        | BinOp::Eq
                        | BinOp::And
                        | BinOp::Or
                );
                let (Some(la), Some(lb)) = (la, lb) else {
                    return logical.then_some(Interval::new(0, 1));
                };
                match op {
                    BinOp::Add => Some(Interval::new(la.lo + lb.lo, la.hi + lb.hi)),
                    BinOp::Sub => Some(Interval::new(la.lo - lb.hi, la.hi - lb.lo)),
                    BinOp::Mul => {
                        let prods = [la.lo * lb.lo, la.lo * lb.hi, la.hi * lb.lo, la.hi * lb.hi];
                        Some(Interval::new(
                            prods.iter().copied().fold(i128::MAX, i128::min),
                            prods.iter().copied().fold(i128::MIN, i128::max),
                        ))
                    }
                    BinOp::Div | BinOp::FloorDiv => (lb.lo == lb.hi && lb.lo > 0)
                        .then(|| Interval::new(la.lo.div_euclid(lb.lo), la.hi.div_euclid(lb.lo))),
                    BinOp::Mod => {
                        (lb.lo == lb.hi && lb.lo > 0).then(|| Interval::new(0, lb.lo - 1))
                    }
                    _ if logical => Some(Interval::new(0, 1)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Bottom-up rewrite: constant-fold literals, drop algebraic identities, collapse static comparisons. Pure and
    /// idempotent.
    pub fn simplify(&self, ctx: &SimplifyCtx) -> Result<Expression> {
        match &**self {
            Expr::Var(_) | Expr::Literal(_) | Expr::Builtin(_) => Ok(self.clone()),
            Expr::Binary(op, left, right) => simplify_binary(self, *op, left, right, ctx),
            Expr::Call(name, args) => {
                let new: Result<Vec<Expression>> = args.iter().map(|a| a.simplify(ctx)).collect();
                let new = new?;
                if new
                    .iter()
                    .zip(args.iter())
                    .all(|(a, b)| Arc::ptr_eq(&a.0, &b.0))
                {
                    Ok(self.clone())
                } else {
                    Ok(Expression::new(Expr::Call(name.clone(), new.into())))
                }
            }
            Expr::Ternary(cond, if_true, if_false) => {
                let cond = cond.simplify(ctx)?;
                if cond.is_truthy() {
                    return if_true.simplify(ctx);
                }
                if cond.is_falsy() {
                    return if_false.simplify(ctx);
                }
                let a = if_true.simplify(ctx)?;
                let b = if_false.simplify(ctx)?;
                if a == b {
                    return Ok(a);
                }
                Ok(Expression::new(Expr::Ternary(cond, a, b)))
            }
            Expr::Cast(dtype, inner) => {
                let new = inner.simplify(ctx)?;
                if let Some(lit) = new.as_literal()
                    && &**dtype == "int"
                {
                    return Ok(Expression::int(lit.value.as_int()?));
                }
                if Arc::ptr_eq(&new.0, &inner.0) {
                    return Ok(self.clone());
                }
                Ok(Expression::new(Expr::Cast(dtype.clone(), new)))
            }
            Expr::FlatIndex(buffer, index) => {
                let new: Result<Vec<Expression>> = index.iter().map(|e| e.simplify(ctx)).collect();
                Ok(Expression::new(Expr::FlatIndex(
                    buffer.clone(),
                    new?.into(),
                )))
            }
        }
    }
}

/// How tightly an atom binds: a cast's operand is written at this level.
const ATOM: u8 = 100;

fn is_identifier(name: &str) -> bool {
    let mut parts = name.split('.');
    let first = parts.next().unwrap_or("");
    let word = |part: &str| {
        let mut chars = part.chars();
        matches!(chars.next(), Some(c) if c.is_alphabetic() || c == '_')
            && chars.all(|c| c.is_alphanumeric() || c == '_')
    };
    match (word(first), parts.next(), parts.next()) {
        (true, None, _) => true,
        (true, Some(second), None) => word(second),
        _ => false,
    }
}

/// `name` as the reader reads a name back: bare when it is an identifier with no other meaning, else between
/// backticks (a `$0` port, a name that is also a built-in or a keyword).
pub fn spell_name(name: &str, reserved: &[&str]) -> Result<String> {
    if is_identifier(name)
        && !is_builtin(name)
        && !RESERVED.contains(&name)
        && !reserved.contains(&name)
    {
        return Ok(name.to_string());
    }
    if name.contains('`') || name.contains('\n') {
        return Err(Error::Value(format!(
            "the name {} has no spelling",
            python_repr(name)
        )));
    }
    Ok(format!("`{name}`"))
}

/// A literal written by its dtype: `3`, `0.5` / `2.0` / `inf`, `true`.
pub fn literal_text(lit: &Lit) -> Result<String> {
    match &*lit.dtype {
        "bool" => Ok(if lit.value.truthy() { "true" } else { "false" }.into()),
        "int" if lit.value.as_f64().fract() == 0.0 && lit.value.as_f64().is_finite() => {
            Ok(lit.value.as_int()?.to_string())
        }
        "float" => Ok(float_repr(lit.value.as_f64())),
        dtype => Err(Error::Value(format!(
            "the literal {} of dtype {} has no spelling",
            num_repr(lit.value),
            python_repr(dtype)
        ))),
    }
}

/// Python's `repr` of a number.
pub fn num_repr(value: Num) -> String {
    match value {
        Num::Int(v) => v.to_string(),
        Num::Float(v) => float_repr(v),
        Num::Bool(v) => if v { "True" } else { "False" }.into(),
    }
}

/// Python's `repr` of a string.
pub fn python_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::new();
    out.push(quote);
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

// --- the simplifier's helpers ---------------------------------------------------------------------

fn fold_binop_literals(op: BinOp, left: &Lit, right: &Lit) -> Result<Expression> {
    let folded = apply_binop(op, left.value, right.value)?;
    if let Num::Bool(flag) = folded {
        return Ok(Expression::int(flag as i64));
    }
    let both_int = &*left.dtype == "int" && &*right.dtype == "int";
    if let Num::Int(v) = folded
        && both_int
    {
        return Ok(Expression::int(v));
    }
    Ok(Expression::float(folded.as_f64()))
}

fn simplify_binary(
    this: &Expression,
    op: BinOp,
    left: &Expression,
    right: &Expression,
    ctx: &SimplifyCtx,
) -> Result<Expression> {
    let (orig_left, orig_right) = (left, right);
    let left = left.simplify(ctx)?;
    let right = right.simplify(ctx)?;
    if let (Some(l), Some(r)) = (left.as_literal(), right.as_literal()) {
        return fold_binop_literals(op, l, r);
    }
    match op {
        BinOp::Add => {
            if left.is_zero() {
                return Ok(right);
            }
            if right.is_zero() {
                return Ok(left);
            }
            if let Some(recomposed) = fold_divmod_recompose(&left, &right, ctx)? {
                return Ok(recomposed);
            }
        }
        BinOp::Sub => {
            if right.is_zero() {
                return Ok(left);
            }
            if left == right {
                return Ok(Expression::int(0));
            }
        }
        BinOp::Mul => {
            if left.is_zero() || right.is_zero() {
                return Ok(Expression::int(0));
            }
            if left.is_one() {
                return Ok(right);
            }
            if right.is_one() {
                return Ok(left);
            }
        }
        BinOp::Div | BinOp::FloorDiv => {
            if right.is_one() {
                return Ok(left);
            }
            if left.is_zero() {
                return Ok(Expression::int(0));
            }
            if let Some(n) = right.int_literal().filter(|n| *n > 1) {
                if let Expr::Binary(inner_op, inner_left, inner_right) = &*left
                    && inner_op.is_division()
                    && let Some(Lit { value, dtype }) = inner_right.as_literal()
                    && &**dtype == "int"
                    && value.as_f64() > 0.0
                    && inner_left.range(ctx).is_some_and(|bounds| bounds.lo >= 0)
                {
                    // Positive constant quotient chains have one denominator in index normal form.
                    let product = apply_binop(BinOp::Mul, *value, Num::Int(n))?;
                    return Expression::binary(
                        BinOp::Div,
                        inner_left.clone(),
                        Expression::literal(product, "int"),
                    )
                    .simplify(ctx);
                }
                if let Some((q, _)) = div_mod_decompose(&left, n, ctx)? {
                    return Ok(q);
                }
                if op == BinOp::FloorDiv && left.range(ctx).is_some_and(|bounds| bounds.lo >= 0) {
                    return Ok(Expression::binary(BinOp::Div, left, right));
                }
            }
            if let Some(cancelled) = cancel_common_factors(op, &left, &right, ctx) {
                return Ok(cancelled);
            }
        }
        BinOp::Mod => {
            if right.is_one() || left.is_zero() {
                return Ok(Expression::int(0));
            }
            if let Some(n) = right.int_literal().filter(|n| *n > 1) {
                if let Some((_, r)) = div_mod_decompose(&left, n, ctx)? {
                    return Ok(r);
                }
            } else if let Some(below) = mod_below_divisor(&left, &right, ctx)? {
                return Ok(below);
            }
            if let Some(cancelled) = cancel_common_factors(op, &left, &right, ctx) {
                return Ok(cancelled);
            }
        }
        BinOp::And => {
            if left.is_truthy() {
                return Ok(right);
            }
            if right.is_truthy() {
                return Ok(left);
            }
            if left.is_falsy() || right.is_falsy() {
                return Ok(Expression::int(0));
            }
        }
        BinOp::Or => {
            if left.is_falsy() {
                return Ok(right);
            }
            if right.is_falsy() {
                return Ok(left);
            }
            if left.is_truthy() || right.is_truthy() {
                return Ok(Expression::int(1));
            }
        }
        BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq => {
            if let (Some(la), Some(lb)) = (left.range(ctx), right.range(ctx))
                && let Some(decided) = static_cmp(op, la, lb)
            {
                return Ok(Expression::int(decided as i64));
            }
        }
        _ => {}
    }
    if Arc::ptr_eq(&left.0, &orig_left.0) && Arc::ptr_eq(&right.0, &orig_right.0) {
        return Ok(this.clone());
    }
    Ok(Expression::binary(op, left, right))
}

/// Flatten a tree of `*` into its non-int factors and its int coefficient.
fn multiplicative_factors(expression: &Expression) -> (Vec<Expression>, i64) {
    if let Some(Lit { value, dtype }) = expression.as_literal()
        && &**dtype == "int"
        && value.is_integral()
    {
        return (vec![], value.as_int().unwrap_or(0));
    }
    if let Expr::Binary(BinOp::Mul, left, right) = &**expression {
        let (mut lf, lk) = multiplicative_factors(left);
        let (rf, rk) = multiplicative_factors(right);
        lf.extend(rf);
        return (lf, lk.wrapping_mul(rk));
    }
    (vec![expression.clone()], 1)
}

fn is_positive(expression: &Expression, ctx: &SimplifyCtx) -> bool {
    if let Some(Lit { value, .. }) = expression.as_literal()
        && value.is_integral()
    {
        return value.as_int().is_ok_and(|v| v >= 1);
    }
    expression.range(ctx).is_some_and(|rng| rng.lo >= 1)
}

fn rebuild_product(factors: &[Expression], k: i64) -> Expression {
    if k == 0 {
        return Expression::int(0);
    }
    if factors.is_empty() {
        return Expression::int(k);
    }
    let mut out = factors[0].clone();
    for f in &factors[1..] {
        out = Expression::binary(BinOp::Mul, out, f.clone());
    }
    if k != 1 {
        out = Expression::binary(BinOp::Mul, out, Expression::int(k));
    }
    out
}

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.abs()
}

fn cancel_common_factors(
    op: BinOp,
    left: &Expression,
    right: &Expression,
    ctx: &SimplifyCtx,
) -> Option<Expression> {
    let (l_factors, mut l_k) = multiplicative_factors(left);
    let (r_factors, mut r_k) = multiplicative_factors(right);
    let mut cancelled = false;
    let mut r_remaining = r_factors;
    let mut l_remaining = Vec::new();
    let mut common_factors = Vec::new();
    let mut common_int = 1;
    for f in l_factors {
        match r_remaining
            .iter()
            .position(|rf| f == *rf && is_positive(&f, ctx))
        {
            Some(index) => {
                r_remaining.remove(index);
                cancelled = true;
                common_factors.push(f);
            }
            None => l_remaining.push(f),
        }
    }
    if l_k > 0 && r_k > 0 {
        let g = gcd(l_k, r_k);
        if g > 1 {
            l_k /= g;
            r_k /= g;
            cancelled = true;
            common_int = g;
        }
    }
    if !cancelled {
        return None;
    }
    let new_left = rebuild_product(&l_remaining, l_k);
    let new_right = rebuild_product(&r_remaining, r_k);
    if op.is_division() {
        if new_right.is_one() {
            return Some(new_left);
        }
        if new_left.is_zero() {
            return Some(Expression::int(0));
        }
    } else if op == BinOp::Mod {
        if new_right.is_one() || new_left.is_zero() {
            return Some(Expression::int(0));
        }
        let common = rebuild_product(&common_factors, common_int);
        let modulo = Expression::binary(BinOp::Mod, new_left, new_right);
        return Some(if common.is_one() {
            modulo
        } else {
            Expression::binary(BinOp::Mul, common, modulo)
        });
    }
    Some(Expression::binary(op, new_left, new_right))
}

fn addend_chain(e: &Expression, out: &mut Vec<Expression>) {
    if let Expr::Binary(BinOp::Add, l, r) = &**e {
        addend_chain(l, out);
        addend_chain(r, out);
    } else {
        out.push(e.clone());
    }
}

/// `e` as `(x <op> c) * a` (either `*` operand order, bare form `a = 1`), `op` among `ops`, `c` and `a` positive int
/// literals: `(x, c, a)`.
fn scaled_term(e: &Expression, ops: &[BinOp]) -> Option<(Expression, i64, i64)> {
    let mut a = 1;
    let mut e = e.clone();
    if let Expr::Binary(BinOp::Mul, left, right) = &*e.clone() {
        for (lit_side, other) in [(right, left), (left, right)] {
            if let Some(value) = lit_side.int_literal().filter(|v| *v > 0) {
                a = value;
                e = other.clone();
                break;
            }
        }
    }
    if let Expr::Binary(op, left, right) = &*e
        && ops.contains(op)
        && let Some(c) = right.int_literal().filter(|c| *c > 1)
    {
        return Some((left.clone(), c, a));
    }
    None
}

/// Fold `(x / c) * (c·k) + (x % c) * k → x * k` inside an additive chain.
fn fold_divmod_recompose(
    left: &Expression,
    right: &Expression,
    ctx: &SimplifyCtx,
) -> Result<Option<Expression>> {
    let mut terms = Vec::new();
    addend_chain(left, &mut terms);
    addend_chain(right, &mut terms);
    for (i, t1) in terms.iter().enumerate() {
        let Some((x, c, a)) = scaled_term(t1, &[BinOp::Div, BinOp::FloorDiv]) else {
            continue;
        };
        if a % c != 0 {
            continue;
        }
        let k = a / c;
        for (j, t2) in terms.iter().enumerate() {
            if j == i {
                continue;
            }
            match scaled_term(t2, &[BinOp::Mod]) {
                Some((rx, rc, rk)) if rx == x && rc == c && rk == k => {}
                _ => continue,
            }
            let mut out = if k == 1 {
                x.clone()
            } else {
                Expression::binary(BinOp::Mul, x.clone(), Expression::int(k))
            };
            for (pos, t) in terms.iter().enumerate() {
                if pos != i && pos != j {
                    out = Expression::binary(BinOp::Add, out, t.clone());
                }
            }
            return Ok(Some(out.simplify(ctx)?));
        }
    }
    Ok(None)
}

/// `expression` as `n * q + r` with `0 <= r < n`, both proven from operand ranges.
fn div_mod_decompose(
    expression: &Expression,
    n: i64,
    ctx: &SimplifyCtx,
) -> Result<Option<(Expression, Expression)>> {
    if let Some(value) = expression.int_literal().filter(|v| *v >= 0) {
        return Ok(Some((
            Expression::int(value.div_euclid(n)),
            Expression::int(value.rem_euclid(n)),
        )));
    }
    if let Expr::Binary(BinOp::Mul, left, right) = &**expression {
        for (k_side, other_side) in [(right, left), (left, right)] {
            let Some(k) = k_side.int_literal().filter(|k| *k > 0) else {
                continue;
            };
            if other_side.range(ctx).is_none_or(|rng| rng.lo < 0) {
                continue;
            }
            if k % n == 0 {
                let q = Expression::binary(BinOp::Mul, other_side.clone(), Expression::int(k / n))
                    .simplify(ctx)?;
                return Ok(Some((q, Expression::int(0))));
            }
            if n % k == 0 {
                let d = Expression::int(n / k);
                let q =
                    Expression::binary(BinOp::Div, other_side.clone(), d.clone()).simplify(ctx)?;
                let r = Expression::binary(
                    BinOp::Mul,
                    Expression::binary(BinOp::Mod, other_side.clone(), d),
                    Expression::int(k),
                )
                .simplify(ctx)?;
                return Ok(Some((q, r)));
            }
        }
    }
    if let Expr::Binary(BinOp::Add, left, right) = &**expression {
        let mut l = div_mod_decompose(left, n, ctx)?;
        let mut r = div_mod_decompose(right, n, ctx)?;
        if let Some((_, rl)) = &l
            && r.is_none()
            && rl.is_zero()
        {
            r = split_addend(right, n, ctx)?;
        } else if let Some((_, rr)) = &r
            && l.is_none()
            && rr.is_zero()
        {
            l = split_addend(left, n, ctx)?;
        }
        let (Some((ql, rl)), Some((qr, rr))) = (l, r) else {
            return Ok(None);
        };
        let r_sum = Expression::binary(BinOp::Add, rl, rr).simplify(ctx)?;
        let Some(rng) = r_sum.range(ctx) else {
            return Ok(None);
        };
        if rng.lo < 0 || rng.hi >= n as i128 {
            return Ok(None);
        }
        let q = Expression::binary(BinOp::Add, ql, qr).simplify(ctx)?;
        return Ok(Some((q, r_sum)));
    }
    if let Some(rng) = expression.range(ctx)
        && rng.lo >= 0
        && rng.hi < n as i128
    {
        return Ok(Some((Expression::int(0), expression.clone())));
    }
    if let Expr::Binary(op, left, divisor) = &**expression
        && op.is_division()
        && let Some(d) = divisor.int_literal().filter(|d| *d > 0)
        && let Some((q_inner, r_inner)) = div_mod_decompose(left, n * d, ctx)?
    {
        return Ok(Some((
            q_inner,
            Expression::binary(*op, r_inner, divisor.clone()).simplify(ctx)?,
        )));
    }
    Ok(None)
}

fn split_addend(
    expression: &Expression,
    n: i64,
    ctx: &SimplifyCtx,
) -> Result<Option<(Expression, Expression)>> {
    if expression.range(ctx).is_none_or(|rng| rng.lo < 0) {
        return Ok(None);
    }
    let lit = Expression::int(n);
    Ok(Some((
        Expression::binary(BinOp::Div, expression.clone(), lit.clone()).simplify(ctx)?,
        Expression::binary(BinOp::Mod, expression.clone(), lit).simplify(ctx)?,
    )))
}

fn mod_below_divisor(
    left: &Expression,
    right: &Expression,
    ctx: &SimplifyCtx,
) -> Result<Option<Expression>> {
    let Expr::Var(name) = &**left else {
        return Ok(None);
    };
    let Some(bound) = ctx.bounds.get(name) else {
        return Ok(None);
    };
    if bound.simplify(ctx)? != *right {
        return Ok(None);
    }
    Ok(left
        .range(ctx)
        .filter(|rng| rng.lo >= 0)
        .map(|_| left.clone()))
}

fn static_cmp(op: BinOp, la: Interval, lb: Interval) -> Option<bool> {
    match op {
        BinOp::Lt => (la.hi < lb.lo)
            .then_some(true)
            .or((la.lo >= lb.hi).then_some(false)),
        BinOp::Le => (la.hi <= lb.lo)
            .then_some(true)
            .or((la.lo > lb.hi).then_some(false)),
        BinOp::Gt => (la.lo > lb.hi)
            .then_some(true)
            .or((la.hi <= lb.lo).then_some(false)),
        BinOp::Ge => (la.lo >= lb.hi)
            .then_some(true)
            .or((la.hi < lb.lo).then_some(false)),
        BinOp::Eq => {
            if la.hi < lb.lo || la.lo > lb.hi {
                Some(false)
            } else if la.lo == la.hi && la.hi == lb.lo && lb.lo == lb.hi {
                Some(true)
            } else {
                None
            }
        }
        _ => None,
    }
}

// --- the simplification context -------------------------------------------------------------------

/// A closed integer interval `[lo, hi]`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Interval {
    pub lo: i128,
    pub hi: i128,
}

impl Interval {
    pub fn new(lo: i128, hi: i128) -> Interval {
        Interval { lo, hi }
    }
}

/// The ranges in scope: an integer interval per name, and a symbolic exclusive upper bound per name where one is
/// known (`i < seq_len`), so a modulo by a non-literal divisor folds.
#[derive(Clone, Default, PartialEq, Debug)]
pub struct SimplifyCtx {
    pub ranges: HashMap<Name, Interval>,
    pub bounds: HashMap<Name, Expression>,
}

impl SimplifyCtx {
    pub fn empty() -> SimplifyCtx {
        SimplifyCtx::default()
    }

    pub fn extend(&self, name: &str, interval: Interval, bound: Option<Expression>) -> SimplifyCtx {
        let mut out = self.clone();
        out.ranges.insert(name.into(), interval);
        if let Some(bound) = bound {
            out.bounds.insert(name.into(), bound);
        }
        out
    }
}

// --- the affine form ------------------------------------------------------------------------------

/// `expression` as `anchor + Σ coeffs[v]·v` over `vars`; `None` when it is not affine in them.
pub fn affine_form(
    expression: &Expression,
    vars: &dyn Fn(&str) -> bool,
) -> Option<(Expression, BTreeMap<Name, i64>)> {
    let uses = |e: &Expression| e.free_vars().into_iter().any(|name| vars(&name));
    if !uses(expression) {
        return Some((expression.clone(), BTreeMap::new()));
    }
    match &**expression {
        Expr::Var(name) => Some((Expression::int(0), BTreeMap::from([(name.clone(), 1)]))),
        Expr::Binary(op @ (BinOp::Add | BinOp::Sub), left, right) => {
            let (l_anchor, l_coeffs) = affine_form(left, vars)?;
            let (r_anchor, r_coeffs) = affine_form(right, vars)?;
            let sign = if *op == BinOp::Add { 1 } else { -1 };
            let anchor = Expression::binary(*op, l_anchor, r_anchor);
            let mut coeffs = l_coeffs;
            for (v, c) in r_coeffs {
                *coeffs.entry(v).or_insert(0) += sign * c;
            }
            coeffs.retain(|_, c| *c != 0);
            Some((anchor, coeffs))
        }
        Expr::Binary(BinOp::Mul, left, right) => {
            let (l_uses, r_uses) = (uses(left), uses(right));
            if l_uses && r_uses {
                return None;
            }
            let (side, mult) = if l_uses { (left, right) } else { (right, left) };
            let Lit { value, .. } = mult.as_literal()?;
            if !value.is_integral() {
                return None;
            }
            let factor = value.as_int().ok()?;
            let (sub_anchor, sub_coeffs) = affine_form(side, vars)?;
            let anchor = if l_uses {
                Expression::binary(BinOp::Mul, sub_anchor, mult.clone())
            } else {
                Expression::binary(BinOp::Mul, mult.clone(), sub_anchor)
            };
            Some((
                anchor,
                sub_coeffs
                    .into_iter()
                    .map(|(v, c)| (v, c * factor))
                    .collect(),
            ))
        }
        _ => None,
    }
}

/// `expression` as `(its integer anchor, the rest)`, or `None` when it is not affine with a literal anchor.
pub fn split_anchor(expression: &Expression) -> Option<(i64, Expression)> {
    let free = expression.free_vars();
    let (anchor, coeffs) = affine_form(expression, &|name| free.iter().any(|n| &**n == name))?;
    let anchor = anchor.eval(&HashMap::new()).ok()?.as_int().ok()?;
    let mut rest = Expression::int(0);
    for (name, coeff) in coeffs {
        let variable = Expression::new(Expr::Var(name));
        let term = if coeff == 1 {
            variable
        } else {
            Expression::binary(BinOp::Mul, variable, Expression::int(coeff))
        };
        rest = Expression::binary(BinOp::Add, rest, term);
    }
    Some((anchor, rest))
}

/// An upper bound on the number of distinct coordinate tuples `expressions` take over `var_extents`.
pub fn index_set_size(
    expressions: &[Expression],
    var_extents: &HashMap<Name, i64>,
) -> Option<i128> {
    let mut total: i128 = 1;
    for e in expressions {
        let (_, coeffs) = affine_form(e, &|name| var_extents.contains_key(name))?;
        let mut size: i128 = 1;
        for (v, c) in coeffs {
            size += (c as i128).abs() * (var_extents[&v] as i128 - 1);
        }
        total *= size;
    }
    Some(total)
}

// --- the reader -----------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TokenKind {
    Number,
    Name,
    Quoted,
    Op,
}

#[derive(Clone, Debug)]
pub struct Token {
    pub kind: TokenKind,
    pub text: String,
    pub position: usize,
}

/// Operators and punctuation an expression's text uses.
pub const PUNCTUATION: [&str; 26] = [
    "||", "&&", "==", "!=", "<=", ">=", "<<", ">>", "//", "-", "+", "*", "/", "%", "<", ">", "^",
    "&", "|", "?", ":", "(", ")", ",", "[", "]",
];

/// The reader of an expression's text. Text that holds expressions among other things (a statement) is read by a
/// parser built with more punctuation, which takes each expression with [`Parser::expression`].
pub struct Parser {
    pub text: String,
    pub tokens: Vec<Token>,
    pub index: usize,
}

impl Parser {
    /// Tokenize `text` with the expression punctuation plus `extra` (longest first where one starts another).
    pub fn new(text: &str, extra: &[&str]) -> Result<Parser> {
        let mut punctuation: Vec<&str> = PUNCTUATION.to_vec();
        punctuation.extend(extra.iter().copied());
        punctuation.sort_by_key(|p| std::cmp::Reverse(p.len()));
        let mut tokens = Vec::new();
        let bytes = text.as_bytes();
        let mut position = 0;
        while position < bytes.len() {
            let Some(c) = text[position..].chars().next() else {
                break;
            };
            if c.is_whitespace() {
                position += c.len_utf8();
                continue;
            }
            if c.is_ascii_digit() {
                let start = position;
                let mut end = position;
                let rest = &text[position..];
                let digits = |s: &str| s.chars().take_while(|c| c.is_ascii_digit()).count();
                end += digits(rest);
                // A number needs a digit after its point, so ``0..4`` is a number, two dots and a number.
                if text[end..].starts_with('.')
                    && text[end + 1..]
                        .chars()
                        .next()
                        .is_some_and(|c| c.is_ascii_digit())
                {
                    end += 1 + digits(&text[end + 1..]);
                }
                if let Some(e) = text[end..]
                    .chars()
                    .next()
                    .filter(|c| *c == 'e' || *c == 'E')
                {
                    let mut probe = end + e.len_utf8();
                    if text[probe..].starts_with('+') || text[probe..].starts_with('-') {
                        probe += 1;
                    }
                    let count = digits(&text[probe..]);
                    if count > 0 {
                        end = probe + count;
                    }
                }
                tokens.push(Token {
                    kind: TokenKind::Number,
                    text: text[start..end].to_string(),
                    position: start,
                });
                position = end;
                continue;
            }
            if c.is_alphabetic() || c == '_' {
                let start = position;
                let word = |s: &str| {
                    s.chars()
                        .take_while(|c| c.is_alphanumeric() || *c == '_')
                        .map(char::len_utf8)
                        .sum::<usize>()
                };
                let mut end = position + word(&text[position..]);
                if text[end..].starts_with('.')
                    && text[end + 1..]
                        .chars()
                        .next()
                        .is_some_and(|c| c.is_alphabetic() || c == '_')
                {
                    end += 1 + word(&text[end + 1..]);
                }
                tokens.push(Token {
                    kind: TokenKind::Name,
                    text: text[start..end].to_string(),
                    position: start,
                });
                position = end;
                continue;
            }
            if c == '`' {
                let start = position;
                let inner = &text[position + 1..];
                let Some(close) = inner.find('`') else {
                    return Err(Parser::error_at("unexpected character", start, text));
                };
                let quoted = &inner[..close];
                if quoted.is_empty() || quoted.contains('\n') {
                    return Err(Parser::error_at("unexpected character", start, text));
                }
                tokens.push(Token {
                    kind: TokenKind::Quoted,
                    text: quoted.to_string(),
                    position: start + 1,
                });
                position += close + 2;
                continue;
            }
            match punctuation
                .iter()
                .find(|p| text[position..].starts_with(**p))
            {
                Some(p) => {
                    tokens.push(Token {
                        kind: TokenKind::Op,
                        text: p.to_string(),
                        position,
                    });
                    position += p.len();
                }
                None => return Err(Parser::error_at("unexpected character", position, text)),
            }
        }
        Ok(Parser {
            text: text.to_string(),
            tokens,
            index: 0,
        })
    }

    fn error_at(message: &str, position: usize, text: &str) -> Error {
        Error::Value(format!(
            "{message} at position {position} of {}",
            python_repr(text)
        ))
    }

    pub fn error(&self, message: &str) -> Error {
        let position = self
            .tokens
            .get(self.index)
            .map_or(self.text.len(), |t| t.position);
        Parser::error_at(message, position, &self.text)
    }

    pub fn peek(&self, ahead: usize) -> Option<&Token> {
        self.tokens.get(self.index + ahead)
    }

    pub fn take(&mut self) -> Result<Token> {
        let token = self
            .peek(0)
            .cloned()
            .ok_or_else(|| self.error("unexpected end"))?;
        self.index += 1;
        Ok(token)
    }

    pub fn at(&self, op: &str, ahead: usize) -> bool {
        self.peek(ahead)
            .is_some_and(|t| t.kind == TokenKind::Op && t.text == op)
    }

    pub fn at_word(&self, word: &str, ahead: usize) -> bool {
        self.peek(ahead)
            .is_some_and(|t| t.kind == TokenKind::Name && t.text == word)
    }

    pub fn accept(&mut self, op: &str) -> bool {
        if self.at(op, 0) {
            self.index += 1;
            return true;
        }
        false
    }

    pub fn require(&mut self, op: &str) -> Result<()> {
        if !self.accept(op) {
            return Err(self.error(&format!("expected {}", python_repr(op))));
        }
        Ok(())
    }

    pub fn done(&self) -> bool {
        self.index >= self.tokens.len()
    }

    pub fn expression(&mut self) -> Result<Expression> {
        let cond = self.binary(1)?;
        if !self.accept("?") {
            return Ok(cond);
        }
        let if_true = self.expression()?;
        self.require(":")?;
        let if_false = self.expression()?;
        Ok(Expression::new(Expr::Ternary(cond, if_true, if_false)))
    }

    /// A comma-separated list of expressions up to `close`, which is consumed.
    pub fn expressions(&mut self, close: &str) -> Result<Vec<Expression>> {
        let mut items = Vec::new();
        while !self.accept(close) {
            if !items.is_empty() {
                self.require(",")?;
            }
            items.push(self.expression()?);
        }
        Ok(items)
    }

    fn binary(&mut self, least: u8) -> Result<Expression> {
        let mut left = self.operand()?;
        while let Some(token) = self.peek(0) {
            if token.kind != TokenKind::Op {
                break;
            }
            let Some(op) = BinOp::parse(&token.text) else {
                break;
            };
            if op.precedence() < least {
                break;
            }
            self.index += 1;
            let right = self.binary(op.precedence() + 1)?;
            left = Expression::binary(op, left, right);
        }
        Ok(left)
    }

    /// The number next in the text, signed or not, or `None` when what is next is no number.
    pub fn number(&mut self) -> Result<Option<Lit>> {
        let sign = self.at("-", 0);
        let ahead = sign as usize;
        let Some(token) = self.peek(ahead) else {
            return Ok(None);
        };
        let literal = match token.kind {
            TokenKind::Number => {
                let text = format!("{}{}", if sign { "-" } else { "" }, token.text);
                if token.text.contains(['.', 'e', 'E']) {
                    let value: f64 = text.parse().map_err(|_| self.error("not a number"))?;
                    Lit {
                        value: Num::Float(value),
                        dtype: "float".into(),
                    }
                } else {
                    let value: i64 = text
                        .parse()
                        .map_err(|_| self.error("integer out of range"))?;
                    Lit {
                        value: Num::Int(value),
                        dtype: "int".into(),
                    }
                }
            }
            TokenKind::Name if token.text == "inf" || token.text == "nan" => {
                let value = if token.text == "inf" {
                    f64::INFINITY
                } else {
                    f64::NAN
                };
                Lit {
                    value: Num::Float(if sign { -value } else { value }),
                    dtype: "float".into(),
                }
            }
            _ => return Ok(None),
        };
        self.index += ahead + 1;
        Ok(Some(literal))
    }

    fn operand(&mut self) -> Result<Expression> {
        if let Some(literal) = self.number()? {
            return Ok(Expression::new(Expr::Literal(literal)));
        }
        let token = self.take()?;
        match token.kind {
            TokenKind::Quoted => {
                if self.accept("[") {
                    let index = self.expressions("]")?;
                    return Ok(Expression::new(Expr::FlatIndex(
                        token.text.into(),
                        index.into(),
                    )));
                }
                Ok(Expression::new(Expr::Var(token.text.into())))
            }
            TokenKind::Name => {
                if token.text == "true" || token.text == "false" {
                    return Ok(Expression::literal(Num::Bool(token.text == "true"), "bool"));
                }
                if self.accept("(") {
                    let args = self.expressions(")")?;
                    return Ok(Expression::new(Expr::Call(token.text.into(), args.into())));
                }
                if self.accept("[") {
                    let index = self.expressions("]")?;
                    return Ok(Expression::new(Expr::FlatIndex(
                        token.text.into(),
                        index.into(),
                    )));
                }
                if is_builtin(&token.text) {
                    return Ok(Expression::new(Expr::Builtin(token.text.into())));
                }
                Ok(Expression::new(Expr::Var(token.text.into())))
            }
            TokenKind::Op if token.text == "(" => {
                // ``(dtype)`` followed by an operand is a cast; the writer never parenthesizes a lone name otherwise.
                if let Some(next) = self.peek(0)
                    && next.kind == TokenKind::Name
                    && self.at(")", 1)
                    && self.starts_operand(2)
                {
                    let dtype: Name = next.text.clone().into();
                    self.index += 2;
                    let inner = self.operand()?;
                    return Ok(Expression::new(Expr::Cast(dtype, inner)));
                }
                let inner = self.expression()?;
                self.require(")")?;
                Ok(inner)
            }
            _ => {
                self.index -= 1;
                Err(self.error("expected an operand"))
            }
        }
    }

    fn starts_operand(&self, ahead: usize) -> bool {
        self.peek(ahead).is_some_and(|t| {
            matches!(
                t.kind,
                TokenKind::Number | TokenKind::Name | TokenKind::Quoted
            ) || t.text == "("
        })
    }
}

/// The expression `text` spells: the inverse of [`Expression::text`].
pub fn parse(text: &str) -> Result<Expression> {
    let mut parser = Parser::new(text, &[])?;
    let expression = parser.expression()?;
    if !parser.done() {
        return Err(parser.error("unexpected text"));
    }
    Ok(expression)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_print_as_python_does() {
        for (value, text) in [
            (0.125, "0.125"),
            (2.0, "2.0"),
            (1e-6, "1e-06"),
            (1e30, "1e+30"),
            (1.5e16, "1.5e+16"),
            (123456789012345.0, "123456789012345.0"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (-0.5, "-0.5"),
            (1e16, "1e+16"),
            (f64::INFINITY, "inf"),
            (-1e30, "-1e+30"),
            (100.0, "100.0"),
            (1.0 / 3.0, "0.3333333333333333"),
        ] {
            assert_eq!(float_repr(value), text, "{value}");
        }
    }

    #[test]
    fn text_reads_back() {
        for text in [
            "a5 / 128 * 128 + a6",
            "(a0 + 1) * 4",
            "x < 2 ? y : z",
            "(int)in0",
            "(int)(a + 1)",
            "`model.layers.0.weight`[a0, a1]",
            "exp(a0 - 1) + -1",
            "-inf",
            "true",
            "thread_idx.x * 2",
            "(float)(-1)",
        ] {
            let e = parse(text).unwrap_or_else(|err| panic!("{text}: {err}"));
            assert_eq!(e.text(), text);
        }
    }

    #[test]
    fn literals_fold() {
        let ctx = SimplifyCtx::empty();
        let e = parse("(3 + 4) * 2 - 1").unwrap().simplify(&ctx).unwrap();
        assert_eq!(e.text(), "13");
        let e = parse("7 / 2").unwrap().simplify(&ctx).unwrap();
        assert_eq!(e.text(), "3");
        let e = parse("-7 / 2").unwrap().simplify(&ctx).unwrap();
        assert_eq!(e.text(), "-4");
        let e = parse("1.5 + 1").unwrap().simplify(&ctx).unwrap();
        assert_eq!(e.text(), "2.5");
        let e = parse("3 < 4").unwrap().simplify(&ctx).unwrap();
        assert_eq!(e.text(), "1");
    }

    #[test]
    fn ranges_decide_comparisons_and_divisions() {
        let ctx = SimplifyCtx::empty()
            .extend("i", Interval::new(0, 7), None)
            .extend("k", Interval::new(0, 1), None);
        let e = parse("i < 8").unwrap().simplify(&ctx).unwrap();
        assert_eq!(e.text(), "1");
        let e = parse("(i * 2 + k) / 2").unwrap().simplify(&ctx).unwrap();
        assert_eq!(e.text(), "i");
        let e = parse("(i * 2 + k) % 2").unwrap().simplify(&ctx).unwrap();
        assert_eq!(e.text(), "k");
        let e = parse("i / 2 * 2 + i % 2").unwrap().simplify(&ctx).unwrap();
        assert_eq!(e.text(), "i");
    }
}
