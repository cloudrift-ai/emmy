//! What a statement knows of an elementwise operation: its name, whether its operands commute, its neutral element,
//! the compute-unit cluster it belongs to, and its value on literal operands. The callable behind a name lives in
//! Python (`ElementwiseImpl`); these tables are the one source, which that module reads back.

use std::sync::Arc;

use crate::Name;

/// Binary combines where `op(a, b) == op(b, a)`.
pub const COMMUTATIVE: [&str; 7] = [
    "add", "multiply", "maximum", "minimum", "amax", "sum", "prod",
];
/// Binary combines where `op(op(a, b), c) == op(a, op(b, c))`: the reassociable reduce combines.
pub const ASSOCIATIVE: [&str; 7] = COMMUTATIVE;
/// Reducer neutral elements.
pub const IDENTITY: [(&str, f64); 7] = [
    ("add", 0.0),
    ("sum", 0.0),
    ("multiply", 1.0),
    ("prod", 1.0),
    ("maximum", -1e30),
    ("amax", -1e30),
    ("minimum", 1e30),
];
/// Spellings that mean an op the renderers know under another name, normalized at construction.
pub const ALIASES: [(&str, &str); 2] = [
    ("__rshift__", "right_shift"),
    ("bitwise_right_shift", "right_shift"),
];

/// Each op name's compute-unit cluster representative: the clustered identity collapses ops that share a GPU
/// functional unit.
pub const CLUSTERS: [(&str, &str); 44] = [
    ("add", "multiply"),
    ("sum", "multiply"),
    ("subtract", "multiply"),
    ("sub", "multiply"),
    ("negative", "multiply"),
    ("multiply", "multiply"),
    ("prod", "multiply"),
    ("abs", "multiply"),
    ("fma", "multiply"),
    ("maximum", "maximum"),
    ("minimum", "maximum"),
    ("amax", "maximum"),
    ("sign", "maximum"),
    ("divide", "divide"),
    ("divide_rn_f32", "divide"),
    ("true_divide", "divide"),
    ("floor_divide", "divide"),
    ("remainder", "divide"),
    ("mod", "divide"),
    ("reciprocal", "divide"),
    ("sqrt", "exp"),
    ("rsqrt", "exp"),
    ("exp", "exp"),
    ("log", "exp"),
    ("log2", "exp"),
    ("log10", "exp"),
    ("sin", "exp"),
    ("cos", "exp"),
    ("tan", "exp"),
    ("tanh", "exp"),
    ("sigmoid", "exp"),
    ("silu", "exp"),
    ("softplus", "exp"),
    ("erf", "exp"),
    ("gelu", "exp"),
    ("gelu_tanh", "exp"),
    ("pow", "exp"),
    ("relu", "exp"),
    ("copy", "copy"),
    ("power", "exp"),
    ("square", "multiply"),
    ("fmax", "maximum"),
    ("fmin", "maximum"),
    ("exp_fast", "exp"),
];

/// An elementwise operation as a statement holds it: its canonical name and the traits read of it.
#[derive(Clone, Debug)]
pub struct Op {
    pub name: Name,
    pub commutative: bool,
    pub identity: Option<f64>,
}

impl Op {
    pub fn new(name: &str) -> Op {
        let name = ALIASES
            .iter()
            .find(|(alias, _)| *alias == name)
            .map_or(name, |(_, canonical)| canonical);
        Op {
            name: name.into(),
            commutative: COMMUTATIVE.contains(&name),
            identity: IDENTITY
                .iter()
                .find(|(op, _)| *op == name)
                .map(|(_, value)| *value),
        }
    }

    pub fn associative(&self) -> bool {
        ASSOCIATIVE.contains(&&*self.name)
    }

    /// The op's compute-unit cluster representative; the op itself when no cluster lists it.
    pub fn cluster_representative(&self) -> Op {
        match CLUSTERS.iter().find(|(name, _)| *name == &*self.name) {
            Some((_, representative)) => Op::new(representative),
            None => self.clone(),
        }
    }
}

impl PartialEq for Op {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for Op {}

impl std::hash::Hash for Op {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl From<&str> for Op {
    fn from(name: &str) -> Op {
        Op::new(name)
    }
}

pub type Ops = Arc<[Op]>;

/// The op's value on `f32` operands, as NumPy computes it for `np.float32` scalars; `None` for an op with no host
/// form here (it stays as written) or a result that is not finite.
pub fn evaluate_f32(name: &str, args: &[f32]) -> Option<f32> {
    let one = || args.first().copied();
    let two = || {
        if args.len() == 2 {
            Some((args[0], args[1]))
        } else {
            None
        }
    };
    let erf = |x: f32| libm::erff(x);
    let value = match name {
        "copy" | "pad" | "positive" | "bitcast" => one()?,
        "negative" => -one()?,
        "abs" | "absolute" | "fabs" => one()?.abs(),
        "exp" | "exp_fast" => one()?.exp(),
        "exp2" => one()?.exp2(),
        "expm1" => one()?.exp_m1(),
        "log" => one()?.ln(),
        "log2" => one()?.log2(),
        "log10" => one()?.log10(),
        "log1p" => one()?.ln_1p(),
        "sqrt" => one()?.sqrt(),
        "rsqrt" => 1.0 / one()?.sqrt(),
        "square" => one()? * one()?,
        "reciprocal" => 1.0 / one()?,
        "sin" => one()?.sin(),
        "cos" => one()?.cos(),
        "tan" => one()?.tan(),
        "tanh" => one()?.tanh(),
        "sinh" => one()?.sinh(),
        "cosh" => one()?.cosh(),
        "arctan" => one()?.atan(),
        "arcsin" => one()?.asin(),
        "arccos" => one()?.acos(),
        "floor" => one()?.floor(),
        "ceil" => one()?.ceil(),
        "trunc" => one()?.trunc(),
        "rint" | "round" => libm::rintf(one()?),
        "sign" => {
            let x = one()?;
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        "relu" => one()?.max(0.0),
        "sigmoid" => 1.0 / (1.0 + (-one()?).exp()),
        "silu" => one()? / (1.0 + (-one()?).exp()),
        "softplus" => {
            let x = one()?;
            // np.logaddexp(0, x) = max(0, x) + log1p(exp(-|x|))
            x.max(0.0) + (-x.abs()).exp().ln_1p()
        }
        "erf" => erf(one()?),
        "gelu" => {
            let x = one()?;
            0.5 * x * (1.0 + erf(x / 2f32.sqrt()))
        }
        "gelu_tanh" => {
            let x = one()?;
            0.5 * x
                * (1.0
                    + ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
        }
        "add" => two().map(|(a, b)| a + b)?,
        "subtract" => two().map(|(a, b)| a - b)?,
        "multiply" => two().map(|(a, b)| a * b)?,
        "divide" | "true_divide" | "divide_rn_f32" => two().map(|(a, b)| a / b)?,
        "floor_divide" => two().map(|(a, b)| (a / b).floor())?,
        "remainder" | "mod" => two().map(|(a, b)| a - (a / b).floor() * b)?,
        "maximum" | "fmax" => two().map(|(a, b)| a.max(b))?,
        "minimum" | "fmin" => two().map(|(a, b)| a.min(b))?,
        "power" | "pow" => two().map(|(a, b)| a.powf(b))?,
        "where" if args.len() == 3 => {
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
