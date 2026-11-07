//! Expressions: the coordinate and predicate sublanguage every IR layer indexes and guards with.
//!
//! An expression is a shared, immutable tree ([`Expression`] is an `Arc`). Its text is its one form: on the wire,
//! in every listing and as its structural identity ([`Expression::text`] and [`parse`] are exact inverses). The
//! simplifier ([`Expression::simplify`]) folds literals, drops algebraic identities and decides static comparisons
//! under the ranges a [`SimplifyCtx`] carries; the affine form ([`affine_form`]) reads an index as anchor plus
//! integer coefficients.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};

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
        let (q, r) = (a.checked_div(b), a.checked_rem(b));
        let (Some(q), Some(r)) = (q, r) else {
            return Err(Error::Value("integer overflow".into()));
        };
        Ok(if r != 0 && ((a < 0) != (b < 0)) {
            q - 1
        } else {
            q
        })
    };
    let py_mod = |a: i64, b: i64| -> Result<i64> {
        if b == 0 {
            return Err(Error::ZeroDivision);
        }
        let Some(r) = a.checked_rem(b) else {
            return Err(Error::Value("integer overflow".into()));
        };
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
        BinOp::Ne => Ok(Num::Bool(left.as_f64() != right.as_f64())),
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
                .filter(|shifted| shifted >> b == a)
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
    }
}

// --- the tree -------------------------------------------------------------------------------------

/// A numeric constant: its value and the dtype it is spelled by (`int`, `float` or `bool`).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Lit {
    pub value: Num,
    pub dtype: LitType,
}

/// The type a literal is spelled in.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum LitType {
    Int,
    Float,
    Bool,
}

impl LitType {
    pub fn name(self) -> &'static str {
        match self {
            LitType::Int => "int",
            LitType::Float => "float",
            LitType::Bool => "bool",
        }
    }
}

impl std::fmt::Display for LitType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for LitType {
    type Err = Error;

    fn from_str(name: &str) -> Result<LitType> {
        match name {
            "int" => Ok(LitType::Int),
            "float" => Ok(LitType::Float),
            "bool" => Ok(LitType::Bool),
            _ => Err(Error::Value(format!("no literal type {name}"))),
        }
    }
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
pub struct Expression(pub Arc<Node>);

/// An expression's node: its tree and the tree's height, so what refuses a deep expression reads one number.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Node {
    pub expr: Expr,
    pub height: u32,
}

thread_local! {
    static DROPPING: Cell<usize> = const { Cell::new(0) };
    static DEFERRED: RefCell<Vec<Expression>> = const { RefCell::new(Vec::new()) };
}

/// How many levels a drop unwinds before the rest of a chain is set aside for the outermost drop: the nesting the
/// reader and the constructors admit, so an expression they built drops in one descent.
const DROP_DEPTH: usize = MAX_NESTING;

/// The leaf a taken child leaves behind: one shared literal.
fn leaf() -> Expression {
    static LEAF: OnceLock<Expression> = OnceLock::new();
    LEAF.get_or_init(|| Expression::int(0)).clone()
}

impl Expr {
    fn children_mut(&mut self, f: &mut dyn FnMut(&mut Expression)) {
        match self {
            Expr::Var(_) | Expr::Literal(_) | Expr::Builtin(_) => {}
            Expr::Binary(_, left, right) => {
                f(left);
                f(right);
            }
            Expr::Ternary(a, b, c) => {
                f(a);
                f(b);
                f(c);
            }
            Expr::Cast(_, inner) => f(inner),
            Expr::Call(_, args) | Expr::FlatIndex(_, args) => {
                if let Some(items) = Arc::get_mut(args) {
                    items.iter_mut().for_each(f);
                }
            }
        }
    }
}

impl Drop for Expression {
    /// A chain of nested expressions drops level by level: a node dropped `DROP_DEPTH` levels down sets its
    /// children aside, and the outermost drop takes them up once it has unwound, so no chain overflows the stack.
    fn drop(&mut self) {
        if Arc::strong_count(&self.0) != 1 {
            return;
        }
        let Some(node) = Arc::get_mut(&mut self.0) else {
            return;
        };
        let node = &mut node.expr;
        if matches!(node, Expr::Var(_) | Expr::Literal(_) | Expr::Builtin(_)) {
            return;
        }
        let depth = DROPPING.with(Cell::get);
        if depth >= DROP_DEPTH {
            DEFERRED.with(|deferred| {
                let mut deferred = deferred.borrow_mut();
                node.children_mut(&mut |child| deferred.push(std::mem::replace(child, leaf())));
            });
            return;
        }
        DROPPING.with(|d| d.set(depth + 1));
        node.children_mut(&mut |child| drop(std::mem::replace(child, leaf())));
        DROPPING.with(|d| d.set(depth));
        if depth == 0 {
            while let Some(expression) = DEFERRED.with(|deferred| deferred.borrow_mut().pop()) {
                drop(expression);
            }
        }
    }
}

impl PartialEq for Expression {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0) || self.0.expr == other.0.expr
    }
}

impl Eq for Expression {}

impl Hash for Expression {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.expr.hash(state);
    }
}

impl std::ops::Deref for Expression {
    type Target = Expr;

    fn deref(&self) -> &Expr {
        &self.0.expr
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
        let below = match &expr {
            Expr::Var(_) | Expr::Literal(_) | Expr::Builtin(_) => 0,
            Expr::Binary(_, left, right) => left.height().max(right.height()),
            Expr::Ternary(a, b, c) => a.height().max(b.height()).max(c.height()),
            Expr::Cast(_, inner) => inner.height(),
            Expr::Call(_, args) | Expr::FlatIndex(_, args) => {
                args.iter().map(Expression::height).max().unwrap_or(0)
            }
        };
        Expression(Arc::new(Node {
            expr,
            height: below + 1,
        }))
    }

    /// How many levels this tree nests: one for a leaf.
    pub fn height(&self) -> u32 {
        self.0.height
    }

    /// This expression when it nests no deeper than the compiler walks, else the error the reader and the
    /// constructors refuse it with.
    pub fn bounded(self) -> Result<Expression> {
        if self.height() as usize > MAX_NESTING {
            return Err(Error::Recursion(format!(
                "the expression nests {} levels deep; the limit is {MAX_NESTING}",
                self.height()
            )));
        }
        Ok(self)
    }

    pub fn var(name: &str) -> Expression {
        Expression::new(Expr::Var(name.into()))
    }

    pub fn int(value: i64) -> Expression {
        Expression::new(Expr::Literal(Lit {
            value: Num::Int(value),
            dtype: LitType::Int,
        }))
    }

    pub fn float(value: f64) -> Expression {
        Expression::new(Expr::Literal(Lit {
            value: Num::Float(value),
            dtype: LitType::Float,
        }))
    }

    pub fn literal(value: Num, dtype: LitType) -> Expression {
        Expression::new(Expr::Literal(Lit { value, dtype }))
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
            Expr::Literal(Lit { value, dtype })
                if *dtype == LitType::Int && value.is_integral() =>
            {
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
                    BinOp::Add => Some(Interval::new(
                        la.lo.checked_add(lb.lo)?,
                        la.hi.checked_add(lb.hi)?,
                    )),
                    BinOp::Sub => Some(Interval::new(
                        la.lo.checked_sub(lb.hi)?,
                        la.hi.checked_sub(lb.lo)?,
                    )),
                    BinOp::Mul => {
                        let prods = [
                            la.lo.checked_mul(lb.lo)?,
                            la.lo.checked_mul(lb.hi)?,
                            la.hi.checked_mul(lb.lo)?,
                            la.hi.checked_mul(lb.hi)?,
                        ];
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
                if [(&cond, if_true), (&a, if_true), (&b, if_false)]
                    .iter()
                    .all(|(new, old)| Arc::ptr_eq(&new.0, &old.0))
                {
                    return Ok(self.clone());
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
                let new = new?;
                if new
                    .iter()
                    .zip(index.iter())
                    .all(|(new, old)| Arc::ptr_eq(&new.0, &old.0))
                {
                    return Ok(self.clone());
                }
                Ok(Expression::new(Expr::FlatIndex(buffer.clone(), new.into())))
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
    match lit.dtype {
        LitType::Bool => Ok(if lit.value.truthy() { "true" } else { "false" }.into()),
        LitType::Int if lit.value.as_f64().fract() == 0.0 && lit.value.as_f64().is_finite() => {
            Ok(lit.value.as_int()?.to_string())
        }
        LitType::Float => Ok(float_repr(lit.value.as_f64())),
        LitType::Int => Err(Error::Value(format!(
            "the literal {} of dtype int has no spelling",
            num_repr(lit.value)
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

pub mod parse;
pub mod simplify;

pub use parse::*;
pub use simplify::*;

#[cfg(test)]
mod tests {
    use super::*;

    fn chain(depth: usize) -> Expression {
        let mut chain = Expression::var("a");
        for _ in 0..depth {
            chain = Expression::binary(BinOp::Add, chain, Expression::int(1));
        }
        chain
    }

    #[test]
    fn a_chain_too_deep_is_refused_by_its_height() {
        assert!(matches!(chain(200_000).bounded(), Err(Error::Recursion(_))));
        assert!(chain(MAX_NESTING - 1).bounded().is_ok());
        assert_eq!(chain(3).height(), 4);
    }

    #[test]
    fn a_deep_chain_drops_without_recursion() {
        drop(chain(200_000));
    }

    #[test]
    fn a_text_nesting_too_deep_is_refused() {
        let nested = format!("{}a{}", "(".repeat(5000), ")".repeat(5000));
        assert!(parse(&nested).is_err());
        assert!(parse(&format!("{}a{}", "(".repeat(50), ")".repeat(50))).is_ok());
    }

    fn int(text: &str) -> Expression {
        parse(text).unwrap()
    }

    fn eval_int(e: &Expression, env: &[(&str, i64)]) -> i64 {
        let env: HashMap<Name, Num> = env
            .iter()
            .map(|(name, value)| (Name::from(*name), Num::Int(*value)))
            .collect();
        e.eval(&env).unwrap().as_int().unwrap()
    }

    /// `n*K + k` over the packed-weight ranges: the unpacked element offset the packed address is built from.
    const K: i64 = 4096;
    const N: i64 = 12288;
    const BLOCK: i64 = 16;

    fn packed_ctx(k_hi: i64) -> SimplifyCtx {
        SimplifyCtx::empty()
            .extend("n", Interval::new(0, i128::from(N) - 1), None)
            .extend("k", Interval::new(0, i128::from(k_hi)), None)
    }

    fn same_value(a: &Expression, b: &Expression, what: &str) {
        for nv in [0, 1, 7, 1023, N - 1] {
            for kv in [0, 1, 2, 15, 16, 4095] {
                let env = [("n", nv), ("k", kv)];
                assert_eq!(
                    eval_int(a, &env),
                    eval_int(b, &env),
                    "{what}: mismatch at n={nv} k={kv}"
                );
            }
        }
    }

    #[test]
    fn constant_quotient_chains_have_one_normal_form() {
        let ctx = SimplifyCtx::empty().extend("k", Interval::new(0, 16383), None);
        let direct = int("k / 256").simplify(&ctx).unwrap();
        for (outer, inner) in [("/", "/"), ("//", "/"), ("/", "//"), ("//", "//")] {
            let nested = int(&format!("(k {inner} 128) {outer} 2"));
            assert_eq!(nested.simplify(&ctx).unwrap(), direct);
            for k in [0, 127, 128, 255, 256, 257, 16383] {
                assert_eq!(
                    eval_int(&nested, &[("k", k)]),
                    eval_int(&direct, &[("k", k)])
                );
            }
        }
    }

    #[test]
    fn a_quotient_chain_does_not_assume_unknown_operands_are_integer_indices() {
        let nested = int("(value / 128) / 2");
        assert_eq!(nested.simplify(&SimplifyCtx::empty()).unwrap(), nested);
    }

    #[test]
    fn the_packed_modulo_drops_the_row_axis() {
        // `((n*K + k) / 2) % (K/2)` is `k/2`: the row axis leaves the expression entirely.
        let e = int(&format!("((n * {K} + k) / 2) % {}", K / 2));
        let s = e.simplify(&packed_ctx(K - 1)).unwrap();
        assert!(
            !s.free_vars().iter().any(|v| &**v == "n"),
            "row axis survived: {}",
            s.text()
        );
        same_value(&e, &s, "packed modulo");
    }

    #[test]
    fn the_packed_quotient_is_the_row_axis() {
        let e = int(&format!("((n * {K} + k) / 2) / {}", K / 2));
        assert_eq!(
            e.simplify(&packed_ctx(K - 1)).unwrap(),
            Expression::var("n")
        );
    }

    #[test]
    fn the_scale_block_index_drops_the_row_axis() {
        let e = int(&format!("((n * {K} + k) / {BLOCK}) % {}", K / BLOCK));
        let s = e.simplify(&packed_ctx(K - 1)).unwrap();
        assert!(
            !s.free_vars().iter().any(|v| &**v == "n"),
            "row axis survived: {}",
            s.text()
        );
        same_value(&e, &s, "block-scale index");
    }

    #[test]
    fn a_row_major_flatten_of_the_packed_load_loses_its_residue() {
        // `n*(K/2) + ((n*K + k)/2) % (K/2)`: no division holds the row axis afterwards.
        let flat = int(&format!("n * {} + ((n * {K} + k) / 2) % {}", K / 2, K / 2));
        let s = flat.simplify(&packed_ctx(K - 1)).unwrap();
        assert!(!s.text().contains('%'), "residue survived: {}", s.text());
        same_value(&flat, &s, "row-major flatten");
    }

    #[test]
    fn no_range_does_not_fold_the_packed_address() {
        // Without a range the operands are not provably non-negative: the fold declines rather than commit to a
        // rounding direction it cannot justify.
        let e = int(&format!("((n * {K} + k) / 2) % {}", K / 2));
        assert_eq!(e.simplify(&SimplifyCtx::empty()).unwrap(), e);
    }

    #[test]
    fn a_signed_row_axis_does_not_fold() {
        let e = int(&format!("((n * {K} + k) / 2) % {}", K / 2));
        let signed = SimplifyCtx::empty()
            .extend("n", Interval::new(-1, i128::from(N) - 1), None)
            .extend("k", Interval::new(0, i128::from(K) - 1), None);
        assert_eq!(e.simplify(&signed).unwrap(), e);
    }

    #[test]
    fn a_wider_reduce_range_stays_value_exact() {
        let e = int(&format!("((n * {K} + k) / 2) % {}", K / 2));
        for k_hi in [K - 1, K, K + 7, 3 * K] {
            let s = e.simplify(&packed_ctx(k_hi)).unwrap();
            for nv in [0, 1, 2, 37] {
                for kv in [0, 1, (K - 1).min(k_hi), k_hi] {
                    let env = [("n", nv), ("k", kv)];
                    assert_eq!(
                        eval_int(&e, &env),
                        eval_int(&s, &env),
                        "k_hi={k_hi}: n={nv} k={kv}: {}",
                        s.text()
                    );
                }
            }
        }
    }

    #[test]
    fn a_bare_divmod_pair_recomposes() {
        let e = int("(f // 512) * 512 + f % 512");
        assert_eq!(
            e.simplify(&SimplifyCtx::empty()).unwrap(),
            Expression::var("f")
        );
    }

    #[test]
    fn a_scaled_divmod_pair_preserves_its_value() {
        // `(f/c)*(c*k) + (f%c)*k` folds to `f*k`.
        for (c, k) in [(512, 1), (64, 2), (24, 512), (8, 8)] {
            let rem = if k == 1 {
                format!("f % {c}")
            } else {
                format!("(f % {c}) * {k}")
            };
            let e = int(&format!("(f // {c}) * {} + {rem}", c * k));
            let s = e.simplify(&SimplifyCtx::empty()).unwrap();
            let spelled = s.text();
            assert!(
                !spelled.contains('%') && !spelled.contains('/'),
                "{spelled}"
            );
            for v in 0..(4 * c + 3) {
                assert_eq!(
                    eval_int(&e, &[("f", v)]),
                    eval_int(&s, &[("f", v)]),
                    "mismatch at f={v}: {spelled}"
                );
            }
        }
    }

    #[test]
    fn a_divmod_pair_inside_a_longer_chain_recomposes() {
        let e = int("row * 12288 + (f // 512) * 512 + f % 512");
        let s = e.simplify(&SimplifyCtx::empty()).unwrap();
        assert!(!s.text().contains('%'));
        for (fv, rv) in [(0, 0), (511, 1), (513, 3), (12287, 7)] {
            let env = [("f", fv), ("row", rv)];
            assert_eq!(eval_int(&e, &env), eval_int(&s, &env));
        }
    }

    #[test]
    fn mismatched_divmod_pairs_do_not_recompose() {
        for (text, values) in [
            ("(f // 512) * 512 + f % 256", vec![0, 255, 256, 511, 513]),
            ("(f // 512) * 500 + f % 512", vec![0, 511, 512, 1025]),
        ] {
            let e = int(text);
            let s = e.simplify(&SimplifyCtx::empty()).unwrap();
            for v in values {
                assert_eq!(eval_int(&e, &[("f", v)]), eval_int(&s, &[("f", v)]));
            }
            assert!(
                s.text().contains('%'),
                "{text} must not recompose: {}",
                s.text()
            );
        }
        let s = int("(f // 64) * 64 + g % 64")
            .simplify(&SimplifyCtx::empty())
            .unwrap();
        assert!(s.text().contains('%'));
    }

    #[test]
    fn the_constant_fold_refuses_what_overflows_an_integer() {
        let least = Num::Int(i64::MIN);
        assert!(apply_binop(BinOp::Div, least, Num::Int(-1)).is_err());
        assert!(apply_binop(BinOp::Mod, least, Num::Int(-1)).is_err());
        assert!(apply_binop(BinOp::Shl, Num::Int(3), Num::Int(62)).is_err());
        assert_eq!(
            apply_binop(BinOp::Shl, Num::Int(3), Num::Int(2)).ok(),
            Some(Num::Int(12))
        );
        assert_eq!(
            apply_binop(BinOp::Mod, Num::Int(-7), Num::Int(3)).ok(),
            Some(Num::Int(2))
        );
    }

    #[test]
    fn integer_rewrites_decline_rather_than_overflow() {
        // 2^62 * 4 leaves i64: the product is not folded, the affine form is refused, and evaluation reports it.
        let huge = parse("(a * 4611686018427387904) * 4").unwrap();
        let simplified = huge.simplify(&SimplifyCtx::empty()).unwrap();
        assert!(simplified.as_literal().is_none(), "{}", simplified.text());
        assert!(
            simplified.text().contains("4611686018427387904"),
            "{}",
            simplified.text()
        );
        assert!(affine_form(&huge, &|name| name == "a").is_none());
        let env = HashMap::from([(Name::from("a"), Num::Int(1))]);
        assert!(simplified.eval(&env).is_err());
    }

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
