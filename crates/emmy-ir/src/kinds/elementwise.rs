//! What a statement knows of an elementwise operation: its name, whether its operands commute, its neutral element,
//! the compute-unit cluster it belongs to, and its value on literal operands. The callable behind a name lives in
//! Python (`ElementwiseImpl`); these properties are the one source, which that module reads back through the
//! extension's tables.

use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use crate::{Error, Name, Result};

/// The operations the compiler knows a property of, each with its one spelling. `$text` is the name a statement's
/// text and the Python `ElementwiseImpl` use; a spelling outside the list is an [`Op::Other`].
macro_rules! elementwise {
    ($( $Variant:ident = $text:literal, )*) => {
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
        pub enum Elementwise { $( $Variant, )* }

        impl Elementwise {
            pub const ALL: &[Elementwise] = &[ $( Elementwise::$Variant, )* ];

            pub fn text(self) -> &'static str {
                match self { $( Elementwise::$Variant => $text, )* }
            }
        }

        impl FromStr for Elementwise {
            type Err = Error;

            fn from_str(name: &str) -> Result<Elementwise> {
                match name {
                    $( $text => Ok(Elementwise::$Variant), )*
                    _ => Err(Error::Value(format!("no elementwise operation {name}"))),
                }
            }
        }
    };
}

elementwise! {
    Add = "add", Sum = "sum", Subtract = "subtract", Sub = "sub", Negative = "negative", Multiply = "multiply",
    Prod = "prod", Fma = "fma", Square = "square", Abs = "abs", Absolute = "absolute", Fabs = "fabs",
    Maximum = "maximum", Minimum = "minimum", Amax = "amax", Fmax = "fmax", Fmin = "fmin", Sign = "sign",
    Divide = "divide", DivideRnF32 = "divide_rn_f32", TrueDivide = "true_divide", FloorDivide = "floor_divide",
    Remainder = "remainder", Mod = "mod", Reciprocal = "reciprocal",
    Sqrt = "sqrt", Rsqrt = "rsqrt", Exp = "exp", ExpFast = "exp_fast", Exp2 = "exp2", Expm1 = "expm1",
    Log = "log", Log2 = "log2", Log10 = "log10", Log1p = "log1p", Sin = "sin", Cos = "cos", Tan = "tan",
    Tanh = "tanh", Sinh = "sinh", Cosh = "cosh", Arctan = "arctan", Arcsin = "arcsin", Arccos = "arccos",
    Floor = "floor", Ceil = "ceil", Trunc = "trunc", Rint = "rint", Round = "round",
    Sigmoid = "sigmoid", Silu = "silu", Softplus = "softplus", Erf = "erf", Gelu = "gelu", GeluTanh = "gelu_tanh",
    Pow = "pow", Power = "power", Relu = "relu", Where = "where",
    Copy = "copy", Pad = "pad", Positive = "positive", Bitcast = "bitcast",
    LeftShift = "left_shift", RightShift = "right_shift", BitwiseAnd = "bitwise_and", BitwiseOr = "bitwise_or",
    BitwiseXor = "bitwise_xor", BitwiseCount = "bitwise_count",
}

impl fmt::Display for Elementwise {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.text())
    }
}

/// Spellings that mean an operation the renderers know under another name, normalized at construction.
pub const ALIASES: [(&str, Elementwise); 2] = [
    ("__rshift__", Elementwise::RightShift),
    ("bitwise_right_shift", Elementwise::RightShift),
];

impl Elementwise {
    /// A binary combine where `op(a, b) == op(b, a)`.
    pub fn commutative(self) -> bool {
        use Elementwise::*;
        matches!(self, Add | Multiply | Maximum | Minimum | Amax | Sum | Prod)
    }

    /// A binary combine where `op(op(a, b), c) == op(a, op(b, c))`: the reassociable reduce combines.
    pub fn associative(self) -> bool {
        self.commutative()
    }

    /// A reducer's neutral element.
    pub fn identity(self) -> Option<f64> {
        use Elementwise::*;
        match self {
            Add | Sum => Some(0.0),
            Multiply | Prod => Some(1.0),
            Maximum | Amax => Some(-1e30),
            Minimum => Some(1e30),
            _ => None,
        }
    }

    /// The compute-unit cluster representative: the clustered identity collapses operations that share a GPU
    /// functional unit. `None` for an operation no cluster lists.
    pub fn cluster(self) -> Option<Elementwise> {
        use Elementwise::*;
        Some(match self {
            Add | Sum | Subtract | Sub | Negative | Multiply | Prod | Abs | Fma | Square => {
                Multiply
            }
            Maximum | Minimum | Amax | Sign | Fmax | Fmin => Maximum,
            Divide | DivideRnF32 | TrueDivide | FloorDivide | Remainder | Mod | Reciprocal => {
                Divide
            }
            Sqrt | Rsqrt | Exp | Log | Log2 | Log10 | Sin | Cos | Tan | Tanh | Sigmoid | Silu
            | Softplus | Erf | Gelu | GeluTanh | Pow | Relu | Power | ExpFast => Exp,
            Copy => Copy,
            _ => return None,
        })
    }

    /// Arithmetic on integer operands stays in the integer family.
    pub fn integer(self) -> bool {
        use Elementwise::*;
        matches!(
            self,
            Add | Subtract
                | Negative
                | Multiply
                | FloorDivide
                | Remainder
                | Mod
                | LeftShift
                | RightShift
                | BitwiseAnd
                | BitwiseOr
                | BitwiseXor
                | BitwiseCount
        )
    }

    /// Whether `fold_affine` reads this operation as a term of a sum: a sum, a difference, a negation, or a product
    /// or quotient by a literal.
    pub fn affine(self) -> bool {
        use Elementwise::*;
        matches!(self, Add | Subtract | Negative | Multiply | Divide)
    }
}

/// An elementwise operation as a statement holds it: one the compiler knows by property, or any other spelling a
/// renderer knows.
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

    pub fn commutative(&self) -> bool {
        self.known().is_some_and(Elementwise::commutative)
    }

    pub fn associative(&self) -> bool {
        self.known().is_some_and(Elementwise::associative)
    }

    pub fn identity(&self) -> Option<f64> {
        self.known().and_then(Elementwise::identity)
    }

    /// The op's compute-unit cluster representative; the op itself when no cluster lists it.
    pub fn cluster_representative(&self) -> Op {
        match self.known().and_then(Elementwise::cluster) {
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

impl Elementwise {
    /// The op's value on `f32` operands, as NumPy computes it for `np.float32` scalars; `None` for an op with no host
    /// form here (it stays as written) or a result that is not finite.
    pub fn evaluate_f32(self, args: &[f32]) -> Option<f32> {
        use Elementwise::*;
        let one = || args.first().copied();
        let two = || {
            if args.len() == 2 {
                Some((args[0], args[1]))
            } else {
                None
            }
        };
        let erf = |x: f32| libm::erff(x);
        let value = match self {
            Copy | Pad | Positive | Bitcast => one()?,
            Negative => -one()?,
            Abs | Absolute | Fabs => one()?.abs(),
            Exp | ExpFast => one()?.exp(),
            Exp2 => one()?.exp2(),
            Expm1 => one()?.exp_m1(),
            Log => one()?.ln(),
            Log2 => one()?.log2(),
            Log10 => one()?.log10(),
            Log1p => one()?.ln_1p(),
            Sqrt => one()?.sqrt(),
            Rsqrt => 1.0 / one()?.sqrt(),
            Square => one()? * one()?,
            Reciprocal => 1.0 / one()?,
            Sin => one()?.sin(),
            Cos => one()?.cos(),
            Tan => one()?.tan(),
            Tanh => one()?.tanh(),
            Sinh => one()?.sinh(),
            Cosh => one()?.cosh(),
            Arctan => one()?.atan(),
            Arcsin => one()?.asin(),
            Arccos => one()?.acos(),
            Floor => one()?.floor(),
            Ceil => one()?.ceil(),
            Trunc => one()?.trunc(),
            Rint | Round => libm::rintf(one()?),
            Sign => {
                let x = one()?;
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            Relu => one()?.max(0.0),
            Sigmoid => 1.0 / (1.0 + (-one()?).exp()),
            Silu => one()? / (1.0 + (-one()?).exp()),
            Softplus => {
                let x = one()?;
                // np.logaddexp(0, x) = max(0, x) + log1p(exp(-|x|))
                x.max(0.0) + (-x.abs()).exp().ln_1p()
            }
            Erf => erf(one()?),
            Gelu => {
                let x = one()?;
                0.5 * x * (1.0 + erf(x / 2f32.sqrt()))
            }
            GeluTanh => {
                let x = one()?;
                0.5 * x
                    * (1.0
                        + ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x))
                            .tanh())
            }
            Add => two().map(|(a, b)| a + b)?,
            Subtract => two().map(|(a, b)| a - b)?,
            Multiply => two().map(|(a, b)| a * b)?,
            Divide | TrueDivide | DivideRnF32 => two().map(|(a, b)| a / b)?,
            FloorDivide => two().map(|(a, b)| (a / b).floor())?,
            Remainder | Mod => two().map(|(a, b)| a - (a / b).floor() * b)?,
            Maximum | Fmax => two().map(|(a, b)| a.max(b))?,
            Minimum | Fmin => two().map(|(a, b)| a.min(b))?,
            Power | Pow => two().map(|(a, b)| a.powf(b))?,
            Where if args.len() == 3 => {
                if args[0] != 0.0 {
                    args[1]
                } else {
                    args[2]
                }
            }
            _ => return None,
        };
        value.is_finite().then_some(value)
    }
}
