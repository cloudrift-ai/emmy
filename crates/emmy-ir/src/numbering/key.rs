//! Digest material: the `Key` tree every number, parameter and tie is spelled in, and its hash.

use crate::Name;
use crate::closed_set;
use crate::expr::Num;
use crate::expr::{Expr, ExprType, Expression};

/// Digest material: a tree of primitives, hashed to a number and ordered totally.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum Key {
    None,
    Bool(bool),
    Int(i64),
    Float(u64),
    /// A name, or a short text of a closed vocabulary (a tag, a dtype, an operation): interned.
    Str(Name),
    /// A text of its own, hashed like a `Str`: a body's text, which no table should keep.
    Text(Box<str>),
    Num(u128),
    Tuple(Vec<Key>),
}

pub type Number = u128;

closed_set! {
    /// The tags digest material carries, each spelled as the Python digest spelled it: a tag's text is identity
    /// material and never changes.
    pub enum Tag ("digest tag") {
        Block = "block", Stmt = "stmt", Fixed = "fixed", Pick = "pick", Open = "open",
        Expr = "expr", Coordinate = "c", Operand = "v", Free = "free", State = "state", Buffer = "buffer",
        Bound = "bound", Parameter = "parameter",
        Load = "load", Assign = "assign", Term = "term", Affine = "affine", Init = "init", Reduce = "reduce",
        Select = "select", Store = "store", Carry = "carry", Pre = "pre", Loop = "loop", Cond = "cond", Other = "other",
        Composite = "composite",
    }
}

impl Key {
    pub fn str(text: &str) -> Key {
        Key::Str(Name::new(text))
    }

    pub fn text(text: String) -> Key {
        Key::Text(text.into())
    }

    pub fn tag(tag: Tag) -> Key {
        Key::Str(Name::new(tag.text()))
    }

    pub fn float(value: f64) -> Key {
        // The bits, so a signed zero keys apart from zero as its text does; every NaN alike.
        Key::Float(if value.is_nan() {
            f64::NAN.to_bits()
        } else {
            value.to_bits()
        })
    }

    pub fn tuple(parts: impl IntoIterator<Item = Key>) -> Key {
        Key::Tuple(parts.into_iter().collect())
    }

    pub fn num(value: Num) -> Key {
        match value {
            Num::Int(v) => Key::Int(v),
            Num::Float(v) => Key::float(v),
            Num::Bool(v) => Key::Bool(v),
        }
    }

    pub fn option(text: Option<&str>) -> Key {
        text.map_or(Key::None, Key::str)
    }

    fn feed(&self, hasher: &mut blake3::Hasher) {
        match self {
            Key::None => hasher.update(&[0]),
            Key::Bool(v) => hasher.update(&[1, *v as u8]),
            Key::Int(v) => {
                hasher.update(&[2]);
                hasher.update(&v.to_le_bytes())
            }
            Key::Float(v) => {
                hasher.update(&[3]);
                hasher.update(&v.to_le_bytes())
            }
            Key::Str(s) => Key::feed_text(s, hasher),
            Key::Text(s) => Key::feed_text(s, hasher),
            Key::Num(n) => {
                hasher.update(&[5]);
                hasher.update(&n.to_le_bytes())
            }
            Key::Tuple(parts) => {
                hasher.update(&[6]);
                hasher.update(&(parts.len() as u64).to_le_bytes());
                for part in parts {
                    part.feed(hasher);
                }
                hasher
            }
        };
    }

    fn feed_text<'h>(text: &str, hasher: &'h mut blake3::Hasher) -> &'h mut blake3::Hasher {
        hasher.update(&[4]);
        hasher.update(&(text.len() as u64).to_le_bytes());
        hasher.update(text.as_bytes())
    }

    pub fn hash(&self) -> Number {
        let mut hasher = blake3::Hasher::new();
        self.feed(&mut hasher);
        let bytes = hasher.finalize();
        let mut head = [0u8; 16];
        head.copy_from_slice(&bytes.as_bytes()[..16]);
        u128::from_le_bytes(head)
    }
}

/// The hash of a sequence of parts.
pub fn hash_parts(parts: impl IntoIterator<Item = Key>) -> Number {
    Key::tuple(parts).hash()
}

/// A parameter: the form of a coordinate-only expression, every axis spelled by its binding depth.
pub type Param = Key;

pub fn spelled_key(name: &Name) -> Key {
    Key::Str(*name)
}

/// A hex spelling of a number, for the identity key.
pub fn hex(number: Number) -> String {
    format!("{number:032x}")
}

/// An expression as digest material. `leaf` spells the subexpressions that have a form of their own in the caller's
/// eyes (a variable by its binding, a coordinate-only subexpression by its parameter index) and declines the rest,
/// which are spelled by their structure, the same way everywhere.
pub fn expression_key(e: &Expression, leaf: &mut dyn FnMut(&Expression) -> Option<Key>) -> Key {
    if let Some(key) = leaf(e) {
        return key;
    }
    match &**e {
        Expr::Var(name) => Key::tuple([Key::str(ExprType::Var.text()), Key::Str(*name)]),
        Expr::Literal(lit) => Key::tuple([
            Key::str(ExprType::Literal.text()),
            Key::num(lit.value),
            Key::str(lit.dtype.text()),
        ]),
        Expr::Builtin(builtin) => {
            Key::tuple([Key::str(ExprType::Builtin.text()), Key::str(builtin.text())])
        }
        Expr::Binary(op, l, r) => Key::tuple([
            Key::str(ExprType::Binary.text()),
            Key::str(op.infix().unwrap_or(op.text())),
            expression_key(l, leaf),
            expression_key(r, leaf),
        ]),
        Expr::Call(name, args) => Key::tuple([
            Key::str(ExprType::Call.text()),
            Key::Str(*name),
            Key::Tuple(args.iter().map(|a| expression_key(a, leaf)).collect()),
        ]),
        Expr::Ternary(a, b, c) => Key::tuple([
            Key::str(ExprType::Ternary.text()),
            expression_key(a, leaf),
            expression_key(b, leaf),
            expression_key(c, leaf),
        ]),
        Expr::Cast(dtype, inner) => Key::tuple([
            Key::str(ExprType::Cast.text()),
            Key::str(&dtype.to_string()),
            expression_key(inner, leaf),
        ]),
        Expr::FlatIndex(buffer, index) => Key::tuple([
            Key::str(ExprType::FlatIndex.text()),
            Key::Str(*buffer),
            Key::Tuple(index.iter().map(|a| expression_key(a, leaf)).collect()),
        ]),
    }
}

#[cfg(test)]
mod tests {
    use super::Tag;

    /// A tag's text is identity material: this list is what every stored identity was digested with.
    #[test]
    fn every_digest_tag_spells_as_it_always_did() {
        let texts: Vec<&str> = Tag::ALL.iter().map(|tag| tag.text()).collect();
        assert_eq!(
            texts,
            [
                "block",
                "stmt",
                "fixed",
                "pick",
                "open",
                "expr",
                "c",
                "v",
                "free",
                "state",
                "buffer",
                "bound",
                "parameter",
                "load",
                "assign",
                "term",
                "affine",
                "init",
                "reduce",
                "select",
                "store",
                "carry",
                "pre",
                "loop",
                "cond",
                "other",
                "composite",
            ]
        );
    }
}
