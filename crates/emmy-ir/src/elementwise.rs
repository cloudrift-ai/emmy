//! What a statement knows of an elementwise operation: its name, whether its operands commute, its neutral element,
//! the compute-unit cluster it belongs to, and its value on literal operands. The callable behind a name lives in
//! Python (`ElementwiseImpl`); these properties are the one source, which that module reads back through the
//! extension's tables.

use std::fmt;
use std::sync::Arc;

use crate::{Name, closed_set};

/// How `fold_affine` reads an operation: as a term of a sum.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AffineRole {
    /// `a + b`: both operands with the scale.
    Sum,
    /// `a - b`: the second operand with the negated scale.
    Difference,
    /// `-a`: the operand with the negated scale.
    Negation,
    /// `a * b` with one literal operand: the other operand scaled by it.
    Scale,
    /// `a / b` with a literal divisor: the dividend scaled by its reciprocal.
    Quotient,
}

impl AffineRole {
    pub fn arity(self) -> usize {
        match self {
            AffineRole::Negation => 1,
            _ => 2,
        }
    }
}

/// How an operation computes on host floats, for constant folding; `None` where it has no host form here (it stays
/// as written).
#[derive(Clone, Copy)]
pub enum Eval {
    None,
    Unary(fn(f32) -> f32),
    Binary(fn(f32, f32) -> f32),
    Ternary(fn(f32, f32, f32) -> f32),
}

/// Everything the compiler knows of a known operation, in one row of the table `OPS` indexes by `Operation`.
#[derive(Clone, Copy)]
pub struct OperationInfo {
    /// A binary combine where `op(a, b) == op(b, a)`.
    pub commutative: bool,
    /// A binary combine where `op(op(a, b), c) == op(a, op(b, c))`: the reassociable reduce combines.
    pub associative: bool,
    /// The reducer's neutral element; `seed` is what a kernel initializes with.
    pub neutral: Option<f64>,
    /// The compute-unit cluster representative: the clustered identity collapses operations sharing a GPU unit.
    pub cluster: Option<Operation>,
    /// Arithmetic on integer operands stays in the integer family (the dtype promotion rule).
    pub integer: bool,
    /// The copy alias: the one operation the alias reads of normalization (placement, the ties, the constant fold)
    /// read through. `positive` and `pad` also return their operand and are not aliases: marking them would move
    /// identities.
    pub copies: bool,
    /// The operation chooses one of its operands (the max/min family) rather than accumulate a magnitude, so an
    /// accumulation over it may stay in the input dtype.
    pub selecting: bool,
    /// How `fold_affine` reads it, when it is a term of a sum.
    pub affine: Option<AffineRole>,
    /// Its value on `f32` operands, as NumPy computes it for `np.float32` scalars (a NaN operand gives NaN, which
    /// no fold takes).
    pub eval: Eval,
    /// The infix spelling an expression gives it, and how tightly that binds (higher binds tighter).
    pub infix: Option<(&'static str, u8)>,
}

impl OperationInfo {
    const DEFAULT: OperationInfo = OperationInfo {
        commutative: false,
        associative: false,
        neutral: None,
        cluster: None,
        integer: false,
        copies: false,
        selecting: false,
        affine: None,
        eval: Eval::None,
        infix: None,
    };
}

/// The finite stand-in a kernel initializes an unbounded neutral element with: the CUDA source spells `-1e+30f`,
/// and two recipe goldens' bodies spell `-1e+30` (the hardware goldens and the corpus seed no maximum). A bounded
/// neutral element is itself.
const SEED_BOUND: f64 = 1e30;

/// The known operations: one list declares each variant, its spelling and its properties, and generates the enum,
/// its `Display`/`FromStr` pair and the table. A property left out takes `OperationInfo::DEFAULT`'s value.
macro_rules! operations {
    ($( $Variant:ident = $text:literal { $( $field:ident : $value:expr ),* $(,)? }, )*) => {
        closed_set! {
            /// The operations the compiler knows a property of, by spelling; a spelling outside the set is an
            /// [`ElementwiseOp::Other`].
            pub enum Operation ("elementwise operation") { $( $Variant = $text, )* }
        }

        /// The properties of every known operation, in `Operation` order.
        pub static OPS: [OperationInfo; Operation::ALL.len()] = [
            $( OperationInfo { $( $field: $value, )* ..OperationInfo::DEFAULT }, )*
        ];
    };
}

use Operation as E;

operations! {
    Add = "add" { commutative: true, associative: true, neutral: Some(0.0), cluster: Some(E::Multiply), integer: true, affine: Some(AffineRole::Sum), eval: Eval::Binary(|a, b| a + b), infix: Some(("+", 5)) },
    Sum = "sum" { commutative: true, associative: true, neutral: Some(0.0), cluster: Some(E::Multiply) },
    Subtract = "subtract" { cluster: Some(E::Multiply), integer: true, affine: Some(AffineRole::Difference), eval: Eval::Binary(|a, b| a - b), infix: Some(("-", 5)) },
    Sub = "sub" { cluster: Some(E::Multiply) },
    Negative = "negative" { cluster: Some(E::Multiply), integer: true, affine: Some(AffineRole::Negation), eval: Eval::Unary(|x| -x) },
    Multiply = "multiply" { commutative: true, associative: true, neutral: Some(1.0), cluster: Some(E::Multiply), integer: true, affine: Some(AffineRole::Scale), eval: Eval::Binary(|a, b| a * b), infix: Some(("*", 6)) },
    Prod = "prod" { commutative: true, associative: true, neutral: Some(1.0), cluster: Some(E::Multiply) },
    Fma = "fma" { cluster: Some(E::Multiply) },
    Square = "square" { cluster: Some(E::Multiply), eval: Eval::Unary(|x| x * x) },
    Abs = "abs" { cluster: Some(E::Multiply), eval: Eval::Unary(f32::abs) },
    Absolute = "absolute" { eval: Eval::Unary(f32::abs) },
    Fabs = "fabs" { eval: Eval::Unary(f32::abs) },
    Maximum = "maximum" { commutative: true, associative: true, neutral: Some(f64::NEG_INFINITY), cluster: Some(E::Maximum), selecting: true, eval: Eval::Binary(|a, b| if a.is_nan() || b.is_nan() { f32::NAN } else { a.max(b) }) },
    Minimum = "minimum" { commutative: true, associative: true, neutral: Some(f64::INFINITY), cluster: Some(E::Maximum), selecting: true, eval: Eval::Binary(|a, b| if a.is_nan() || b.is_nan() { f32::NAN } else { a.min(b) }) },
    Amax = "amax" { commutative: true, associative: true, neutral: Some(f64::NEG_INFINITY), cluster: Some(E::Maximum), selecting: true },
    Fmax = "fmax" { cluster: Some(E::Maximum), eval: Eval::Binary(f32::max) },
    Fmin = "fmin" { cluster: Some(E::Maximum), eval: Eval::Binary(f32::min) },
    Sign = "sign" { cluster: Some(E::Maximum), eval: Eval::Unary(|x| if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }) },
    Divide = "divide" { cluster: Some(E::Divide), affine: Some(AffineRole::Quotient), eval: Eval::Binary(|a, b| a / b), infix: Some(("/", 6)) },
    DivideRnF32 = "divide_rn_f32" { cluster: Some(E::Divide), eval: Eval::Binary(|a, b| a / b) },
    TrueDivide = "true_divide" { cluster: Some(E::Divide), eval: Eval::Binary(|a, b| a / b) },
    FloorDivide = "floor_divide" { cluster: Some(E::Divide), integer: true, eval: Eval::Binary(|a, b| (a / b).floor()), infix: Some(("//", 6)) },
    Remainder = "remainder" { cluster: Some(E::Divide), integer: true, eval: Eval::Binary(|a, b| a - (a / b).floor() * b), infix: Some(("%", 6)) },
    Mod = "mod" { cluster: Some(E::Divide), integer: true, eval: Eval::Binary(|a, b| a - (a / b).floor() * b) },
    Reciprocal = "reciprocal" { cluster: Some(E::Divide), eval: Eval::Unary(|x| 1.0 / x) },
    Sqrt = "sqrt" { cluster: Some(E::Exp), eval: Eval::Unary(f32::sqrt) },
    Rsqrt = "rsqrt" { cluster: Some(E::Exp), eval: Eval::Unary(|x| 1.0 / x.sqrt()) },
    Exp = "exp" { cluster: Some(E::Exp), eval: Eval::Unary(f32::exp) },
    ExpFast = "exp_fast" { cluster: Some(E::Exp), eval: Eval::Unary(f32::exp) },
    Exp2 = "exp2" { eval: Eval::Unary(f32::exp2) },
    Expm1 = "expm1" { eval: Eval::Unary(f32::exp_m1) },
    Log = "log" { cluster: Some(E::Exp), eval: Eval::Unary(f32::ln) },
    Log2 = "log2" { cluster: Some(E::Exp), eval: Eval::Unary(f32::log2) },
    Log10 = "log10" { cluster: Some(E::Exp), eval: Eval::Unary(f32::log10) },
    Log1p = "log1p" { eval: Eval::Unary(f32::ln_1p) },
    Sin = "sin" { cluster: Some(E::Exp), eval: Eval::Unary(f32::sin) },
    Cos = "cos" { cluster: Some(E::Exp), eval: Eval::Unary(f32::cos) },
    Tan = "tan" { cluster: Some(E::Exp), eval: Eval::Unary(f32::tan) },
    Tanh = "tanh" { cluster: Some(E::Exp), eval: Eval::Unary(f32::tanh) },
    Sinh = "sinh" { eval: Eval::Unary(f32::sinh) },
    Cosh = "cosh" { eval: Eval::Unary(f32::cosh) },
    Arctan = "arctan" { eval: Eval::Unary(f32::atan) },
    Arcsin = "arcsin" { eval: Eval::Unary(f32::asin) },
    Arccos = "arccos" { eval: Eval::Unary(f32::acos) },
    Floor = "floor" { eval: Eval::Unary(f32::floor) },
    Ceil = "ceil" { eval: Eval::Unary(f32::ceil) },
    Trunc = "trunc" { eval: Eval::Unary(f32::trunc) },
    Rint = "rint" { eval: Eval::Unary(libm::rintf) },
    Round = "round" { eval: Eval::Unary(libm::rintf) },
    Sigmoid = "sigmoid" { cluster: Some(E::Exp), eval: Eval::Unary(|x| 1.0 / (1.0 + (-x).exp())) },
    Silu = "silu" { cluster: Some(E::Exp), eval: Eval::Unary(|x| x / (1.0 + (-x).exp())) },
    // np.logaddexp(0, x) = max(0, x) + log1p(exp(-|x|))
    Softplus = "softplus" { cluster: Some(E::Exp), eval: Eval::Unary(|x| x.max(0.0) + (-x.abs()).exp().ln_1p()) },
    Erf = "erf" { cluster: Some(E::Exp), eval: Eval::Unary(libm::erff) },
    Gelu = "gelu" { cluster: Some(E::Exp), eval: Eval::Unary(|x| 0.5 * x * (1.0 + libm::erff(x / 2f32.sqrt()))) },
    GeluTanh = "gelu_tanh" { cluster: Some(E::Exp), eval: Eval::Unary(|x| 0.5 * x * (1.0 + ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())) },
    Pow = "pow" { cluster: Some(E::Exp), eval: Eval::Binary(f32::powf) },
    Power = "power" { cluster: Some(E::Exp), eval: Eval::Binary(f32::powf) },
    Relu = "relu" { cluster: Some(E::Exp), eval: Eval::Unary(|x| x.max(0.0)) },
    Where = "where" { eval: Eval::Ternary(|c, a, b| if c != 0.0 { a } else { b }) },
    Copy = "copy" { cluster: Some(E::Copy), copies: true, eval: Eval::Unary(|x| x) },
    Pad = "pad" { eval: Eval::Unary(|x| x) },
    Positive = "positive" { eval: Eval::Unary(|x| x) },
    Bitcast = "bitcast" { eval: Eval::Unary(|x| x) },
    LeftShift = "left_shift" { integer: true, infix: Some(("<<", 4)) },
    RightShift = "right_shift" { integer: true, infix: Some((">>", 4)) },
    BitwiseAnd = "bitwise_and" { integer: true, infix: Some(("&", 4)) },
    BitwiseOr = "bitwise_or" { integer: true, infix: Some(("|", 4)) },
    BitwiseXor = "bitwise_xor" { integer: true, infix: Some(("^", 4)) },
    Less = "less" { infix: Some(("<", 4)) },
    Greater = "greater" { infix: Some((">", 4)) },
    LessEqual = "less_equal" { infix: Some(("<=", 4)) },
    GreaterEqual = "greater_equal" { infix: Some((">=", 4)) },
    Equal = "equal" { commutative: true, infix: Some(("==", 3)) },
    NotEqual = "not_equal" { commutative: true, infix: Some(("!=", 3)) },
    LogicalAnd = "logical_and" { commutative: true, associative: true, infix: Some(("&&", 2)) },
    LogicalOr = "logical_or" { commutative: true, associative: true, infix: Some(("||", 1)) },
    BitwiseCount = "bitwise_count" { integer: true },
}

/// Spellings that mean an operation the renderers know under another name, normalized at construction.
pub const ALIASES: [(&str, Operation); 4] = [
    ("__rshift__", Operation::RightShift),
    ("bitwise_right_shift", Operation::RightShift),
    ("min", Operation::Minimum),
    ("max", Operation::Maximum),
];

impl Operation {
    /// The infix spelling an expression gives this operation, when it has one (`+`, `<`, `&&`, …).
    pub fn infix(self) -> Option<&'static str> {
        self.info().infix.map(|(text, _)| text)
    }

    /// How tightly the infix binds: higher binds tighter; an operation with no infix binds like a call.
    pub fn precedence(self) -> u8 {
        self.info().infix.map_or(7, |(_, precedence)| precedence)
    }

    /// The operation an infix spelling names.
    pub fn from_infix(text: &str) -> Option<Operation> {
        Operation::ALL
            .iter()
            .copied()
            .find(|op| op.infix() == Some(text))
    }

    pub fn is_division(self) -> bool {
        matches!(self, Operation::Divide | Operation::FloorDivide)
    }

    /// An ordering comparison or equality: the operators a predicate's two sides meet through.
    pub fn is_comparison(self) -> bool {
        matches!(
            self,
            Operation::Less
                | Operation::LessEqual
                | Operation::Greater
                | Operation::GreaterEqual
                | Operation::Equal
        )
    }

    /// The row of the table: every property in one place.
    pub fn info(self) -> &'static OperationInfo {
        &OPS[self as usize]
    }

    /// The neutral element as a kernel initializes it: the bounded stand-in of an unbounded one.
    pub fn seed(self) -> Option<f64> {
        self.info()
            .neutral
            .map(|neutral| neutral.clamp(-SEED_BOUND, SEED_BOUND))
    }

    /// The op's value on `f32` operands, as NumPy computes it for `np.float32` scalars; `None` for an op with no host
    /// form here (it stays as written), an arity that does not fit, or a result that is not finite.
    pub fn evaluate_f32(self, args: &[f32]) -> Option<f32> {
        let value = match (self.info().eval, args) {
            (Eval::Unary(f), [x]) => f(*x),
            (Eval::Binary(f), [a, b]) => f(*a, *b),
            (Eval::Ternary(f), [a, b, c]) => f(*a, *b, *c),
            _ => return None,
        };
        value.is_finite().then_some(value)
    }
}

/// An elementwise operation as a statement holds it: one the compiler knows by property, or any other spelling a
/// renderer knows. Built through [`ElementwiseOp::new`] or from an [`Operation`]: `Other` is for a spelling the enum lacks,
/// and an `Other` built by hand with a known spelling would never equal its `Known` twin.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum ElementwiseOp {
    Known(Operation),
    Other(Name),
}

impl ElementwiseOp {
    pub fn new(name: &str) -> ElementwiseOp {
        let name = ALIASES
            .iter()
            .find(|(alias, _)| *alias == name)
            .map_or(name, |(_, canonical)| canonical.text());
        name.parse()
            .map_or_else(|_| ElementwiseOp::Other(name.into()), ElementwiseOp::Known)
    }

    pub fn name(&self) -> &str {
        match self {
            ElementwiseOp::Known(op) => op.text(),
            ElementwiseOp::Other(name) => name,
        }
    }

    pub fn known(&self) -> Option<Operation> {
        match self {
            ElementwiseOp::Known(op) => Some(*op),
            ElementwiseOp::Other(_) => None,
        }
    }

    /// The known operation's properties; `Other` has none.
    pub fn info(&self) -> Option<&'static OperationInfo> {
        self.known().map(Operation::info)
    }

    pub fn commutative(&self) -> bool {
        self.info().is_some_and(|info| info.commutative)
    }

    pub fn associative(&self) -> bool {
        self.info().is_some_and(|info| info.associative)
    }

    /// The reducer's neutral element.
    pub fn neutral(&self) -> Option<f64> {
        self.info().and_then(|info| info.neutral)
    }

    /// The neutral element as a kernel initializes it.
    pub fn seed(&self) -> Option<f64> {
        self.known().and_then(Operation::seed)
    }

    /// The operation returns its one operand unchanged.
    pub fn copies(&self) -> bool {
        self.info().is_some_and(|info| info.copies)
    }

    /// How `fold_affine` reads the operation, when it is a term of a sum.
    pub fn affine(&self) -> Option<AffineRole> {
        self.info().and_then(|info| info.affine)
    }

    /// The op's compute-unit cluster representative; the op itself when no cluster lists it.
    pub fn cluster_representative(&self) -> ElementwiseOp {
        match self.info().and_then(|info| info.cluster) {
            Some(representative) => ElementwiseOp::Known(representative),
            None => self.clone(),
        }
    }
}

impl From<Operation> for ElementwiseOp {
    fn from(op: Operation) -> ElementwiseOp {
        ElementwiseOp::Known(op)
    }
}

impl From<&str> for ElementwiseOp {
    fn from(name: &str) -> ElementwiseOp {
        ElementwiseOp::new(name)
    }
}

impl fmt::Display for ElementwiseOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

pub type ElementwiseOps = Arc<[ElementwiseOp]>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_is_in_enum_order_and_spells_every_operation() {
        for op in Operation::ALL {
            assert_eq!(op.text().parse::<Operation>().unwrap(), *op);
        }
        assert!(Operation::Copy.info().copies && !Operation::Positive.info().copies);
        assert!(Operation::Maximum.info().selecting && !Operation::Add.info().selecting);
        assert_eq!(Operation::Maximum.evaluate_f32(&[1.0, f32::NAN]), None);
        let texts: Vec<&str> = Operation::ALL.iter().map(|op| op.text()).collect();
        assert_eq!(
            texts,
            [
                "add",
                "sum",
                "subtract",
                "sub",
                "negative",
                "multiply",
                "prod",
                "fma",
                "square",
                "abs",
                "absolute",
                "fabs",
                "maximum",
                "minimum",
                "amax",
                "fmax",
                "fmin",
                "sign",
                "divide",
                "divide_rn_f32",
                "true_divide",
                "floor_divide",
                "remainder",
                "mod",
                "reciprocal",
                "sqrt",
                "rsqrt",
                "exp",
                "exp_fast",
                "exp2",
                "expm1",
                "log",
                "log2",
                "log10",
                "log1p",
                "sin",
                "cos",
                "tan",
                "tanh",
                "sinh",
                "cosh",
                "arctan",
                "arcsin",
                "arccos",
                "floor",
                "ceil",
                "trunc",
                "rint",
                "round",
                "sigmoid",
                "silu",
                "softplus",
                "erf",
                "gelu",
                "gelu_tanh",
                "pow",
                "power",
                "relu",
                "where",
                "copy",
                "pad",
                "positive",
                "bitcast",
                "left_shift",
                "right_shift",
                "bitwise_and",
                "bitwise_or",
                "bitwise_xor",
                "less",
                "greater",
                "less_equal",
                "greater_equal",
                "equal",
                "not_equal",
                "logical_and",
                "logical_or",
                "bitwise_count",
            ]
        );
        assert_eq!(Operation::Maximum.info().neutral, Some(f64::NEG_INFINITY));
        assert_eq!(Operation::Maximum.seed(), Some(-1e30));
        assert_eq!(Operation::Minimum.seed(), Some(1e30));
        assert_eq!(Operation::Add.seed(), Some(0.0));
        assert_eq!(Operation::Add.evaluate_f32(&[1.5, 2.0]), Some(3.5));
        assert_eq!(Operation::Add.evaluate_f32(&[1.5]), None);
        assert_eq!(Operation::Where.evaluate_f32(&[0.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(
            ElementwiseOp::new("__rshift__"),
            ElementwiseOp::Known(Operation::RightShift)
        );
        assert!(ElementwiseOp::new("unknown_op").info().is_none());
    }
}
