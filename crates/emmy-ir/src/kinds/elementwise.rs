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

/// Everything the compiler knows of a known operation, in one row of the table `OPS` indexes by `Elementwise`.
#[derive(Clone, Copy)]
pub struct OpInfo {
    /// The spelling: how the operation prints and reads.
    pub text: &'static str,
    /// A binary combine where `op(a, b) == op(b, a)`.
    pub commutative: bool,
    /// A binary combine where `op(op(a, b), c) == op(a, op(b, c))`: the reassociable reduce combines.
    pub associative: bool,
    /// The reducer's neutral element; `seed` is what a kernel initializes with.
    pub neutral: Option<f64>,
    /// The compute-unit cluster representative: the clustered identity collapses operations sharing a GPU unit.
    pub cluster: Option<Elementwise>,
    /// Arithmetic on integer operands stays in the integer family (the dtype promotion rule).
    pub integer: bool,
    /// The operation returns its one operand unchanged: an alias normalization reads through.
    pub copies: bool,
    /// How `fold_affine` reads it, when it is a term of a sum.
    pub affine: Option<AffineRole>,
    /// Its value on `f32` operands, as NumPy computes it for `np.float32` scalars.
    pub eval: Eval,
}

impl OpInfo {
    const DEFAULT: OpInfo = OpInfo {
        text: "",
        commutative: false,
        associative: false,
        neutral: None,
        cluster: None,
        integer: false,
        copies: false,
        affine: None,
        eval: Eval::None,
    };
}

/// The finite stand-in a kernel initializes an unbounded neutral element with (the CUDA source and every stored
/// kernel spell `-1e+30f`); a bounded one is itself.
const SEED_BOUND: f64 = 1e30;

/// The known operations: one list declares each variant, its spelling and its properties, and generates the enum,
/// its `Display`/`FromStr` pair and the table. A property left out takes `OpInfo::DEFAULT`'s value.
macro_rules! operations {
    ($( $Variant:ident = $text:literal { $( $field:ident : $value:expr ),* $(,)? }, )*) => {
        closed_set! {
            /// The operations the compiler knows a property of, by spelling; a spelling outside the set is an
            /// [`Op::Other`].
            pub enum Elementwise ("elementwise operation") { $( $Variant = $text, )* }
        }

        /// The properties of every known operation, in `Elementwise` order.
        pub static OPS: [OpInfo; Elementwise::ALL.len()] = [
            $( OpInfo { text: $text, $( $field: $value, )* ..OpInfo::DEFAULT }, )*
        ];
    };
}

use Elementwise as E;

operations! {
    Add = "add" { commutative: true, associative: true, neutral: Some(0.0), cluster: Some(E::Multiply), integer: true, affine: Some(AffineRole::Sum), eval: Eval::Binary(|a, b| a + b) },
    Sum = "sum" { commutative: true, associative: true, neutral: Some(0.0), cluster: Some(E::Multiply) },
    Subtract = "subtract" { cluster: Some(E::Multiply), integer: true, affine: Some(AffineRole::Difference), eval: Eval::Binary(|a, b| a - b) },
    Sub = "sub" { cluster: Some(E::Multiply) },
    Negative = "negative" { cluster: Some(E::Multiply), integer: true, affine: Some(AffineRole::Negation), eval: Eval::Unary(|x| -x) },
    Multiply = "multiply" { commutative: true, associative: true, neutral: Some(1.0), cluster: Some(E::Multiply), integer: true, affine: Some(AffineRole::Scale), eval: Eval::Binary(|a, b| a * b) },
    Prod = "prod" { commutative: true, associative: true, neutral: Some(1.0), cluster: Some(E::Multiply) },
    Fma = "fma" { cluster: Some(E::Multiply) },
    Square = "square" { cluster: Some(E::Multiply), eval: Eval::Unary(|x| x * x) },
    Abs = "abs" { cluster: Some(E::Multiply), eval: Eval::Unary(f32::abs) },
    Absolute = "absolute" { eval: Eval::Unary(f32::abs) },
    Fabs = "fabs" { eval: Eval::Unary(f32::abs) },
    Maximum = "maximum" { commutative: true, associative: true, neutral: Some(f64::NEG_INFINITY), cluster: Some(E::Maximum), eval: Eval::Binary(f32::max) },
    Minimum = "minimum" { commutative: true, associative: true, neutral: Some(f64::INFINITY), cluster: Some(E::Maximum), eval: Eval::Binary(f32::min) },
    Amax = "amax" { commutative: true, associative: true, neutral: Some(f64::NEG_INFINITY), cluster: Some(E::Maximum) },
    Fmax = "fmax" { cluster: Some(E::Maximum), eval: Eval::Binary(f32::max) },
    Fmin = "fmin" { cluster: Some(E::Maximum), eval: Eval::Binary(f32::min) },
    Sign = "sign" { cluster: Some(E::Maximum), eval: Eval::Unary(|x| if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }) },
    Divide = "divide" { cluster: Some(E::Divide), affine: Some(AffineRole::Quotient), eval: Eval::Binary(|a, b| a / b) },
    DivideRnF32 = "divide_rn_f32" { cluster: Some(E::Divide), eval: Eval::Binary(|a, b| a / b) },
    TrueDivide = "true_divide" { cluster: Some(E::Divide), eval: Eval::Binary(|a, b| a / b) },
    FloorDivide = "floor_divide" { cluster: Some(E::Divide), integer: true, eval: Eval::Binary(|a, b| (a / b).floor()) },
    Remainder = "remainder" { cluster: Some(E::Divide), integer: true, eval: Eval::Binary(|a, b| a - (a / b).floor() * b) },
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
    LeftShift = "left_shift" { integer: true },
    RightShift = "right_shift" { integer: true },
    BitwiseAnd = "bitwise_and" { integer: true },
    BitwiseOr = "bitwise_or" { integer: true },
    BitwiseXor = "bitwise_xor" { integer: true },
    BitwiseCount = "bitwise_count" { integer: true },
}

/// Spellings that mean an operation the renderers know under another name, normalized at construction.
pub const ALIASES: [(&str, Elementwise); 2] = [
    ("__rshift__", Elementwise::RightShift),
    ("bitwise_right_shift", Elementwise::RightShift),
];

impl Elementwise {
    /// The row of the table: every property in one place.
    pub fn info(self) -> &'static OpInfo {
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
/// renderer knows. Built through [`Op::new`] or from an [`Elementwise`]: `Other` is for a spelling the enum lacks,
/// and an `Other` built by hand with a known spelling would never equal its `Known` twin.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Op {
    Known(Elementwise),
    Other(Name),
}

impl Op {
    pub fn new(name: &str) -> Op {
        let name = ALIASES
            .iter()
            .find(|(alias, _)| *alias == name)
            .map_or(name, |(_, canonical)| canonical.text());
        name.parse()
            .map_or_else(|_| Op::Other(name.into()), Op::Known)
    }

    pub fn name(&self) -> &str {
        match self {
            Op::Known(op) => op.text(),
            Op::Other(name) => name,
        }
    }

    pub fn known(&self) -> Option<Elementwise> {
        match self {
            Op::Known(op) => Some(*op),
            Op::Other(_) => None,
        }
    }

    pub fn is(&self, op: Elementwise) -> bool {
        *self == Op::Known(op)
    }

    /// The known operation's properties; `Other` has none.
    pub fn info(&self) -> Option<&'static OpInfo> {
        self.known().map(Elementwise::info)
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
        self.known().and_then(Elementwise::seed)
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
    pub fn cluster_representative(&self) -> Op {
        match self.info().and_then(|info| info.cluster) {
            Some(representative) => Op::Known(representative),
            None => self.clone(),
        }
    }
}

impl From<Elementwise> for Op {
    fn from(op: Elementwise) -> Op {
        Op::Known(op)
    }
}

impl From<&str> for Op {
    fn from(name: &str) -> Op {
        Op::new(name)
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

pub type Ops = Arc<[Op]>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_is_in_enum_order_and_spells_every_operation() {
        for (index, op) in Elementwise::ALL.iter().enumerate() {
            assert_eq!(OPS[index].text, op.text(), "{op:?}");
            assert_eq!(op.text().parse::<Elementwise>().unwrap(), *op);
        }
        let texts: Vec<&str> = Elementwise::ALL.iter().map(|op| op.text()).collect();
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
                "bitwise_count",
            ]
        );
        assert_eq!(Elementwise::Maximum.info().neutral, Some(f64::NEG_INFINITY));
        assert_eq!(Elementwise::Maximum.seed(), Some(-1e30));
        assert_eq!(Elementwise::Minimum.seed(), Some(1e30));
        assert_eq!(Elementwise::Add.seed(), Some(0.0));
        assert_eq!(Elementwise::Add.evaluate_f32(&[1.5, 2.0]), Some(3.5));
        assert_eq!(Elementwise::Add.evaluate_f32(&[1.5]), None);
        assert_eq!(Elementwise::Where.evaluate_f32(&[0.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(Op::new("__rshift__"), Op::Known(Elementwise::RightShift));
        assert!(Op::new("unknown_op").info().is_none());
    }
}
