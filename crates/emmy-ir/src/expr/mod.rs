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

use crate::closed_set;
use crate::elementwise::Operation;
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
            (Num::Float(a), Num::Float(b)) => same_f64(*a, *b),
            (Num::Bool(a), Num::Bool(b)) => a == b,
            _ => same_f64(self.as_f64(), other.as_f64()),
        }
    }
}

/// Float equality that is an equivalence, as `Hash` needs: `0.0 == -0.0` as Python has it, and every NaN alike.
pub fn same_f64(a: f64, b: f64) -> bool {
    a == b || (a.is_nan() && b.is_nan())
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

/// Python's `apply_binop` over two evaluated numbers: integer floor division for `/` and `//`, logical `&&` / `||`
/// on truth values, a bool for every comparison.
pub fn apply_binop(op: Operation, left: Num, right: Num) -> Result<Num> {
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
        Operation::Add => arith(left, right, |a, b| a.checked_add(b), |a, b| a + b),
        Operation::Subtract => arith(left, right, |a, b| a.checked_sub(b), |a, b| a - b),
        Operation::Multiply => arith(left, right, |a, b| a.checked_mul(b), |a, b| a * b),
        Operation::Divide | Operation::FloorDivide => {
            Ok(Num::Int(floor_div(left.as_int()?, right.as_int()?)?))
        }
        Operation::Remainder => Ok(Num::Int(py_mod(left.as_int()?, right.as_int()?)?)),
        Operation::Less => Ok(Num::Bool(left.as_f64() < right.as_f64())),
        Operation::LessEqual => Ok(Num::Bool(left.as_f64() <= right.as_f64())),
        Operation::Greater => Ok(Num::Bool(left.as_f64() > right.as_f64())),
        Operation::GreaterEqual => Ok(Num::Bool(left.as_f64() >= right.as_f64())),
        Operation::Equal => Ok(Num::Bool(left.as_f64() == right.as_f64())),
        Operation::NotEqual => Ok(Num::Bool(left.as_f64() != right.as_f64())),
        Operation::LogicalAnd => Ok(Num::Bool(left.truthy() && right.truthy())),
        Operation::LogicalOr => Ok(Num::Bool(left.truthy() || right.truthy())),
        Operation::BitwiseAnd => bits(left, right, |a, b| a & b, |a, b| Some(a & b)),
        Operation::BitwiseOr => bits(left, right, |a, b| a | b, |a, b| Some(a | b)),
        Operation::BitwiseXor => bits(left, right, |a, b| a ^ b, |a, b| Some(a ^ b)),
        Operation::LeftShift => {
            let (a, b) = ints(left, right)?;
            if !(0..64).contains(&b) {
                return Err(Error::Value("shift out of range".into()));
            }
            a.checked_shl(b as u32)
                .filter(|shifted| shifted >> b == a)
                .map(Num::Int)
                .ok_or_else(|| Error::Value("integer overflow".into()))
        }
        Operation::RightShift => {
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
        _ => Err(Error::Value(format!("{op} has no infix form to evaluate"))),
    }
}

// --- the tree -------------------------------------------------------------------------------------

/// A numeric constant: its value and the dtype it is spelled by (`int`, `float` or `bool`).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Lit {
    pub value: Num,
    pub dtype: LitType,
}

closed_set! {
    /// The two kinds of scalar arithmetic the IR has, as predicates over the operation table that every node of an
    /// expression must satisfy, one inside the other. `Value` admits every operation of the table over any operand:
    /// SSA names, axes, GPU built-ins, literals of any type, select, cast and a buffer's flat index. `Index` is its
    /// restriction to integer arithmetic over integer operands: the affine operations, division and remainder,
    /// minimum and maximum, the comparisons and logical operators, the bitwise operators and shifts, over axes,
    /// built-ins, names and integer literals. Every expression field declares the class it admits beside its roles.
    pub enum ExprClass ("expression class") { Index = "Index", Value = "Value", }
}

impl ExprClass {
    /// `expression` as a member of this class, or why it is not.
    pub fn check(self, expression: &Expression) -> Result<()> {
        self.admits(expression).map_err(|why| {
            Error::Value(format!(
                "{} is not an {self} expression: {why}",
                expression
                    .try_text()
                    .unwrap_or_else(|_| format!("{expression:?}"))
            ))
        })
    }

    fn admits(self, expression: &Expression) -> std::result::Result<(), String> {
        if self == ExprClass::Value {
            return Ok(());
        }
        match (self, &**expression) {
            (_, Expr::Var(_)) => Ok(()),
            (ExprClass::Index, Expr::Literal(lit)) if !matches!(lit.dtype, LitType::Float) => {
                Ok(())
            }
            (ExprClass::Index, Expr::Literal(lit)) => Err(format!("a {} literal", lit.dtype)),
            (ExprClass::Value, Expr::Literal(_)) => Ok(()),
            (ExprClass::Index, Expr::Builtin(_)) => Ok(()),
            (ExprClass::Value, Expr::Builtin(builtin)) => Err(format!("the built-in {builtin}")),
            (ExprClass::Index, Expr::FlatIndex(_, indices)) => {
                indices.iter().try_for_each(|e| self.admits(e))
            }
            (ExprClass::Value, Expr::FlatIndex(buffer, _)) => Err(format!("an index of {buffer}")),
            (_, Expr::Binary(op, left, right)) => {
                if !self.admits_operation(*op) {
                    return Err(format!("the operation {op}"));
                }
                self.admits(left)?;
                self.admits(right)
            }
            (_, Expr::Call(name, args)) => match crate::elementwise::ElementwiseOp::new(name) {
                crate::elementwise::ElementwiseOp::Known(op) if self.admits_operation(op) => {
                    args.iter().try_for_each(|e| self.admits(e))
                }
                _ => Err(format!("a call of {name}")),
            },
            (_, Expr::Ternary(cond, if_true, if_false)) => {
                self.admits(cond)?;
                self.admits(if_true)?;
                self.admits(if_false)
            }
            (_, Expr::Cast(_, inner)) => self.admits(inner),
        }
    }

    fn admits_operation(self, op: Operation) -> bool {
        match self {
            ExprClass::Value => true,
            ExprClass::Index => {
                op.info().affine.is_some()
                    || op.is_division()
                    || op.is_comparison()
                    || op.info().integer
                    || matches!(
                        op,
                        Operation::Minimum
                            | Operation::Maximum
                            | Operation::NotEqual
                            | Operation::LogicalAnd
                            | Operation::LogicalOr
                    )
            }
        }
    }
}

closed_set! {
    /// The type a literal is spelled in.
    pub enum LitType ("literal type") { Int = "int", Float = "float", Bool = "bool", }
}

/// A cast's target: a literal type the simplifier knows (a cast to `int` of a literal folds), or any other C type
/// the renderer spells.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum CastTarget {
    Known(LitType),
    Other(Name),
}

impl CastTarget {
    pub fn is(&self, lit: LitType) -> bool {
        *self == CastTarget::Known(lit)
    }
}

impl std::fmt::Display for CastTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CastTarget::Known(lit) => f.write_str(lit.text()),
            CastTarget::Other(name) => f.write_str(name),
        }
    }
}

impl std::str::FromStr for CastTarget {
    type Err = Error;

    fn from_str(text: &str) -> Result<CastTarget> {
        Ok(text
            .parse()
            .map_or_else(|_| CastTarget::Other(text.into()), CastTarget::Known))
    }
}

/// Every expression type, declared once: the variant, the Python class it is (its struct and its name) and its
/// fields in order. `$callback` receives the list and generates from it: the enum and its walks here, the classes in
/// the extension. The digest spells a node by the variant's name, so this list is also the digest's.
#[macro_export]
macro_rules! with_exprs {
    ($callback:ident) => {
        $callback! {
            /// A variable: an SSA value, an axis or a free name.
            Var as PyVar "Var" { name: Name },
            /// A number with the type it is spelled in.
            Literal as PyLiteral "Literal" { lit: Lit },
            /// A GPU built-in with no host form.
            Builtin as PyBuiltin "Builtin" { name: Builtin },
            Binary as PyBinaryExpression "BinaryExpression" { op: Operation, left: Expression, right: Expression },
            /// A call of a named function over its arguments.
            Call as PyFunctionCallExpression "FunctionCallExpression" { name: Name, args: Arc<[Expression]> },
            Ternary as PyTernaryExpression "TernaryExpression" { cond: Expression, if_true: Expression, if_false: Expression },
            Cast as PyCastExpression "CastExpression" { dtype: CastTarget, expression: Expression },
            /// A buffer read at a flat index: one coordinate per dimension.
            FlatIndex as PyFlatIndex "FlatIndex" { buffer: Name, index: Arc<[Expression]> },
        }
    };
}

/// What a field of an expression node is to the walks: a child expression, several, or a leaf datum.
pub trait ExprField: Clone {
    fn each_child(&self, _f: &mut dyn FnMut(&Expression)) {}
    fn each_child_mut(&mut self, _f: &mut dyn FnMut(&mut Expression)) {}
    fn height(&self) -> u32 {
        0
    }
    /// This field with `f` on each child; `None` when every child came back the same object.
    fn mapped(&self, _f: &mut dyn FnMut(&Expression) -> Expression) -> Option<Self> {
        None
    }
    /// This field over the next children of `items`; `None` when they are the ones it holds.
    fn taken(&self, _items: &mut dyn Iterator<Item = Expression>) -> Option<Self> {
        None
    }
}

impl ExprField for Name {}
impl ExprField for Lit {}
impl ExprField for Operation {}
impl ExprField for Builtin {}
impl ExprField for CastTarget {}

impl ExprField for Expression {
    fn each_child(&self, f: &mut dyn FnMut(&Expression)) {
        f(self);
    }
    fn each_child_mut(&mut self, f: &mut dyn FnMut(&mut Expression)) {
        f(self);
    }
    fn height(&self) -> u32 {
        Expression::height(self)
    }
    fn mapped(&self, f: &mut dyn FnMut(&Expression) -> Expression) -> Option<Self> {
        let new = f(self);
        (!Arc::ptr_eq(&new.0, &self.0)).then_some(new)
    }
    fn taken(&self, items: &mut dyn Iterator<Item = Expression>) -> Option<Self> {
        items.next().filter(|new| !Arc::ptr_eq(&new.0, &self.0))
    }
}

impl ExprField for Arc<[Expression]> {
    fn each_child(&self, f: &mut dyn FnMut(&Expression)) {
        self.iter().for_each(f);
    }
    fn each_child_mut(&mut self, f: &mut dyn FnMut(&mut Expression)) {
        if let Some(items) = Arc::get_mut(self) {
            items.iter_mut().for_each(f);
        }
    }
    fn height(&self) -> u32 {
        self.iter().map(Expression::height).max().unwrap_or(0)
    }
    fn mapped(&self, f: &mut dyn FnMut(&Expression) -> Expression) -> Option<Self> {
        let new: Vec<Expression> = self.iter().map(&mut *f).collect();
        let same = new
            .iter()
            .zip(self.iter())
            .all(|(n, o)| Arc::ptr_eq(&n.0, &o.0));
        (!same).then(|| new.into())
    }
    fn taken(&self, items: &mut dyn Iterator<Item = Expression>) -> Option<Self> {
        let new: Vec<Expression> = items.take(self.len()).collect();
        let same = new.len() == self.len()
            && new
                .iter()
                .zip(self.iter())
                .all(|(n, o)| Arc::ptr_eq(&n.0, &o.0));
        (!same).then(|| new.into())
    }
}

/// The core's callback: the `Expr` enum and the walks every expression type answers through its fields.
macro_rules! define_exprs {
    ($( $(#[doc = $doc:literal])* $Variant:ident as $Py:ident $name:literal { $( $field:ident : $ty:ty ),* } ,)*) => {
        /// One node of an expression tree.
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        pub enum Expr {
            $( $(#[doc = $doc])* $Variant( $($ty),* ), )*
        }

        /// Every expression type by name: the tag the digest spells a node by (`ty().text()`).
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
        pub enum ExprType { $( $Variant, )* }

        impl ExprType {
            pub const fn text(self) -> &'static str {
                match self { $( ExprType::$Variant => stringify!($Variant), )* }
            }
        }

        #[allow(unused_variables, unused_mut, unused_assignments)]
        impl Expr {
            pub fn ty(&self) -> ExprType {
                match self { $( Expr::$Variant(..) => ExprType::$Variant, )* }
            }

            /// `f` on each child, in field order.
            pub fn each_child(&self, f: &mut dyn FnMut(&Expression)) {
                match self { $( Expr::$Variant($($field),*) => { $( ExprField::each_child($field, f); )* } )* }
            }

            fn each_child_mut(&mut self, f: &mut dyn FnMut(&mut Expression)) {
                match self { $( Expr::$Variant($($field),*) => { $( ExprField::each_child_mut($field, f); )* } )* }
            }

            /// The height of the tree under this node: the tallest child's, zero for a leaf.
            fn below(&self) -> u32 {
                match self { $( Expr::$Variant($($field),*) => 0 $(.max(ExprField::height($field)))*, )* }
            }
        }

        #[allow(unused_variables, unused_mut, unused_assignments)]
        impl Expression {
            /// This node with `f` applied to each child; itself when every child comes back as it was.
            pub fn map_children(&self, f: &mut dyn FnMut(&Expression) -> Expression) -> Expression {
                match &**self { $( Expr::$Variant($($field),*) => {
                    let mut changed = false;
                    $( let $field = match ExprField::mapped($field, f) { Some(new) => { changed = true; new } None => $field.clone() }; )*
                    if changed { Expression::new(Expr::$Variant($($field),*)) } else { self.clone() }
                } )* }
            }

            /// This node over `children` (as many as `children` returned), itself when none changed.
            pub fn with_children(&self, children: Vec<Expression>) -> Result<Expression> {
                let count = self.children().len();
                if children.len() != count {
                    return Err(Error::Value(format!(
                        "an expression with {count} children cannot take {}",
                        children.len()
                    )));
                }
                let mut items = children.into_iter();
                Ok(match &**self { $( Expr::$Variant($($field),*) => {
                    let mut changed = false;
                    $( let $field = match ExprField::taken($field, &mut items) { Some(new) => { changed = true; new } None => $field.clone() }; )*
                    if changed { Expression::new(Expr::$Variant($($field),*)) } else { self.clone() }
                } )* })
            }
        }
    };
}

with_exprs!(define_exprs);

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
                node.each_child_mut(&mut |child| deferred.push(std::mem::replace(child, leaf())));
            });
            return;
        }
        DROPPING.with(|d| d.set(depth + 1));
        node.each_child_mut(&mut |child| drop(std::mem::replace(child, leaf())));
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

closed_set! {
    /// Every GPU built-in. The reader tells a built-in from a variable by this set, so no variable takes one of its
    /// names.
    pub enum Builtin ("GPU built-in") {
        ThreadIdxX = "thread_idx.x", ThreadIdxY = "thread_idx.y", ThreadIdxZ = "thread_idx.z",
        BlockIdxX = "block_idx.x", BlockIdxY = "block_idx.y", BlockIdxZ = "block_idx.z",
        BlockDimX = "block_dim.x", BlockDimY = "block_dim.y", BlockDimZ = "block_dim.z",
        GridDimX = "grid_dim.x", GridDimY = "grid_dim.y", GridDimZ = "grid_dim.z",
        WarpSize = "warp_size",
    }
}

pub fn is_builtin(name: &str) -> bool {
    name.parse::<Builtin>().is_ok()
}

/// Names the reader gives a meaning of their own, so a variable of that name is written quoted.
const RESERVED: [&str; 4] = ["true", "false", "inf", "nan"];

pub const PLACEHOLDER_PREFIX: &str = "out_coord_";

impl Expression {
    pub fn new(expr: Expr) -> Expression {
        let height = expr.below() + 1;
        Expression(Arc::new(Node { expr, height }))
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

    pub fn binary(op: Operation, left: Expression, right: Expression) -> Expression {
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
            Expr::Builtin(builtin) => out.push_str(builtin.text()),
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
                let _ = write!(out, " {} ", op.infix().unwrap_or(op.text()));
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
        let mut out = Vec::new();
        (**self).each_child(&mut |child| out.push(child.clone()));
        out
    }

    /// Pre-order walk over the tree, itself first.
    pub fn subterms(&self) -> Vec<Expression> {
        let mut out = Vec::new();
        fn walk(e: &Expression, out: &mut Vec<Expression>) {
            out.push(e.clone());
            (**e).each_child(&mut |child| walk(child, out));
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
        if let Expr::Var(name) = &**self {
            if !out.iter().any(|n| n == name) {
                out.push(*name);
            }
            return;
        }
        (**self).each_child(&mut |child| child.collect_free_vars(out));
    }

    pub fn reads(&self, name: Name) -> bool {
        if let Expr::Var(n) = &**self {
            return *n == name;
        }
        let mut found = false;
        (**self).each_child(&mut |child| found |= child.reads(name));
        found
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
    pub fn rename(&self, rename: &mut dyn FnMut(Name) -> Option<Name>) -> Expression {
        match &**self {
            Expr::Var(name) => match rename(*name) {
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
        self.rename(&mut |name| mapping.get(&name).copied())
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
                out.push(*buffer);
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
            Expr::Builtin(builtin) => Err(Error::NotImplemented(format!(
                "Builtin {builtin} is GPU-only; cannot eval"
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
                Ok(if dtype.is(LitType::Int) {
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
                    Operation::Less
                        | Operation::LessEqual
                        | Operation::Greater
                        | Operation::GreaterEqual
                        | Operation::Equal
                        | Operation::LogicalAnd
                        | Operation::LogicalOr
                );
                let (Some(la), Some(lb)) = (la, lb) else {
                    return logical.then_some(Interval::new(0, 1));
                };
                match op {
                    Operation::Add => Some(Interval::new(
                        la.lo.checked_add(lb.lo)?,
                        la.hi.checked_add(lb.hi)?,
                    )),
                    Operation::Subtract => Some(Interval::new(
                        la.lo.checked_sub(lb.hi)?,
                        la.hi.checked_sub(lb.lo)?,
                    )),
                    Operation::Multiply => {
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
                    Operation::Divide | Operation::FloorDivide => (lb.lo == lb.hi && lb.lo > 0)
                        .then(|| Interval::new(la.lo.div_euclid(lb.lo), la.hi.div_euclid(lb.lo))),
                    Operation::Remainder => {
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
                    Ok(Expression::new(Expr::Call(*name, new.into())))
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
                    && dtype.is(LitType::Int)
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
                Ok(Expression::new(Expr::FlatIndex(*buffer, new.into())))
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
    use crate::normalize::testing::eval_int;

    #[test]
    fn the_built_ins_and_operators_spell_as_they_always_did() {
        let builtins: Vec<&str> = Builtin::ALL.iter().map(|b| b.text()).collect();
        assert_eq!(
            builtins,
            [
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
            ]
        );
        let operators: Vec<&str> = Operation::ALL.iter().filter_map(|op| op.infix()).collect();
        assert_eq!(
            operators,
            [
                "+", "-", "*", "/", "//", "%", "<<", ">>", "&", "|", "^", "<", ">", "<=", ">=",
                "==", "!=", "&&", "||"
            ]
        );
        let literal_types: Vec<&str> = LitType::ALL.iter().map(|t| t.text()).collect();
        assert_eq!(literal_types, ["int", "float", "bool"]);
        let types: Vec<&str> = [
            Expression::var("x"),
            Expression::int(1),
            Expression::new(Expr::Builtin(Builtin::WarpSize)),
            Expression::binary(Operation::Add, Expression::var("x"), Expression::int(1)),
            Expression::new(Expr::Call("f".into(), Arc::from([]))),
            Expression::new(Expr::Ternary(
                Expression::var("c"),
                Expression::int(1),
                Expression::int(2),
            )),
            Expression::new(Expr::Cast(
                CastTarget::Known(LitType::Int),
                Expression::var("x"),
            )),
            Expression::new(Expr::FlatIndex("b".into(), Arc::from([]))),
        ]
        .iter()
        .map(|e| e.ty().text())
        .collect();
        assert_eq!(
            types,
            [
                "Var",
                "Literal",
                "Builtin",
                "Binary",
                "Call",
                "Ternary",
                "Cast",
                "FlatIndex"
            ]
        );
    }

    fn chain(depth: usize) -> Expression {
        let mut chain = Expression::var("a");
        for _ in 0..depth {
            chain = Expression::binary(Operation::Add, chain, Expression::int(1));
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
        assert!(apply_binop(Operation::Divide, least, Num::Int(-1)).is_err());
        assert!(apply_binop(Operation::Remainder, least, Num::Int(-1)).is_err());
        assert!(apply_binop(Operation::LeftShift, Num::Int(3), Num::Int(62)).is_err());
        assert_eq!(
            apply_binop(Operation::LeftShift, Num::Int(3), Num::Int(2)).ok(),
            Some(Num::Int(12))
        );
        assert_eq!(
            apply_binop(Operation::Remainder, Num::Int(-7), Num::Int(3)).ok(),
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

    #[test]
    fn a_name_holding_a_backtick_has_no_text() {
        let err = Expression::var("a`b").try_text().unwrap_err();
        assert!(
            matches!(&err, Error::Value(message) if message.contains("no spelling")),
            "{err}"
        );
    }

    fn var_name(e: &Expression) -> Option<&str> {
        e.as_var().map(|name| &**name)
    }

    /// The mapping that sends the placeholder of axis `d` to the variable `to`.
    fn placeholder_to(d: usize, to: &str) -> HashMap<Name, Expression> {
        HashMap::from([(
            Name::from(format!("{PLACEHOLDER_PREFIX}{d}")),
            Expression::var(to),
        )])
    }

    #[test]
    fn a_placeholder_is_named_by_the_prefix() {
        assert_eq!(
            var_name(&Expression::placeholder(0)),
            Some(format!("{PLACEHOLDER_PREFIX}0").as_str())
        );
        assert_eq!(
            var_name(&Expression::placeholder(3)),
            Some(format!("{PLACEHOLDER_PREFIX}3").as_str())
        );
    }

    #[test]
    fn with_no_axis_every_placeholder_is_one_and_another_name_is_not() {
        assert!(Expression::placeholder(0).is_placeholder(None));
        assert!(Expression::placeholder(7).is_placeholder(None));
        assert!(!Expression::var("row").is_placeholder(None));
    }

    #[test]
    fn with_an_axis_only_that_axis_placeholder_is_one() {
        assert!(Expression::placeholder(2).is_placeholder(Some(2)));
        assert!(!Expression::placeholder(2).is_placeholder(Some(3)));
    }

    #[test]
    fn substitute_replaces_a_variable() {
        let result = Expression::placeholder(0).substitute(&placeholder_to(0, "row"));
        assert_eq!(var_name(&result), Some("row"));
    }

    #[test]
    fn substitute_passes_an_unmapped_variable() {
        let result = Expression::placeholder(1).substitute(&placeholder_to(0, "row"));
        assert_eq!(
            var_name(&result),
            Some(format!("{PLACEHOLDER_PREFIX}1").as_str())
        );
    }

    #[test]
    fn substitute_leaves_a_literal_alone() {
        let e = Expression::int(5);
        assert_eq!(e.substitute(&HashMap::new()), e);
    }

    #[test]
    fn substitute_rewrites_a_binary_expression() {
        let e = Expression::binary(
            Operation::Add,
            Expression::placeholder(0),
            Expression::int(7),
        );
        let result = e.substitute(&placeholder_to(0, "row"));
        let Expr::Binary(op, left, right) = &*result else {
            panic!("not a binary expression: {result}");
        };
        assert_eq!(*op, Operation::Add);
        assert_eq!(var_name(left), Some("row"));
        assert_eq!(right.literal_value(), Some(Num::Int(7)));
    }

    #[test]
    fn substitute_rewrites_a_ternary() {
        let p0 = Expression::placeholder(0);
        let cond = Expression::binary(Operation::Less, p0.clone(), Expression::int(64));
        let wrapped = Expression::binary(Operation::Subtract, p0.clone(), Expression::int(64));
        let e = Expression::new(Expr::Ternary(cond, p0, wrapped));
        let result = e.substitute(&placeholder_to(0, "col"));
        let Expr::Ternary(_, if_true, if_false) = &*result else {
            panic!("not a ternary: {result}");
        };
        assert_eq!(var_name(if_true), Some("col"));
        assert!(
            matches!(&**if_false, Expr::Binary(Operation::Subtract, ..)),
            "{if_false}"
        );
    }
}
