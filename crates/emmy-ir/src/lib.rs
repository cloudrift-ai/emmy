//! The compiler's statement and expression vocabulary, in Rust: the kinds every IR layer shares, bodies and their
//! text, value numbering, identity and normalization. Pure Rust; the Python extension beside it (`emmy-ir-py`) wraps
//! these types as the classes `emmy.compiler.ir.stmt` and `emmy.compiler.ir.expression` export.

use std::sync::Arc;

pub mod axis;
pub mod dtype;
pub mod elementwise;
pub mod expr;

/// A name: an SSA value, an axis, a buffer. Shared, so a rename that keeps a name copies a pointer.
pub type Name = Arc<str>;

/// What a function of this crate refuses: a bad value, a wrong type, a division by zero, a missing key, an operation
/// with no host form. The extension maps each onto the Python exception of the same name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Value(String),
    Type(String),
    Key(String),
    ZeroDivision,
    NotImplemented(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Value(m) | Error::Type(m) | Error::NotImplemented(m) => f.write_str(m),
            Error::Key(m) => write!(f, "{m:?}"),
            Error::ZeroDivision => f.write_str("integer division or modulo by zero"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub fn value_error(message: impl Into<String>) -> Error {
    Error::Value(message.into())
}
pub mod body;
pub mod field;
pub mod identity;
pub mod kinds;
pub mod protocol;
pub mod text;
pub mod order;
pub mod values;
pub mod normalize;
