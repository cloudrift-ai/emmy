//! Digest material: the `Key` tree every number, parameter and tie is spelled in, and its hash.

use std::sync::Arc;

use crate::Name;
use crate::closed_set;
use crate::expr::Num;
use crate::expr::{Expr, Expression};

/// Digest material: a tree of primitives, hashed to a number and ordered totally.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum Key {
    None,
    Bool(bool),
    Int(i64),
    Float(u64),
    Str(Arc<str>),
    Num(u128),
    Tuple(Vec<Key>),
}

pub type Number = u128;

closed_set! {
    /// The tags digest material carries, each spelled as the Python digest spelled it: a tag's text is identity
    /// material and never changes.
    pub enum Tag ("digest tag") {
        Block = "block", Stmt = "stmt", Fixed = "fixed", Pick = "pick", Open = "open",
        Var = "Var", Literal = "Literal", Builtin = "Builtin", Binary = "Binary", Call = "Call", Ternary = "Ternary",
        Cast = "Cast", FlatIndex = "FlatIndex",
        Expr = "expr", Coordinate = "c", Operand = "v", Free = "free", State = "state", Buffer = "buffer",
        Bound = "bound", Parameter = "parameter",
        Load = "load", Assign = "assign", Term = "term", Affine = "affine", Init = "init", Reduce = "reduce",
        Select = "select", Store = "store", Carry = "carry", Pre = "pre", Loop = "loop", Cond = "cond", Other = "other",
    }
}

impl Key {
    pub fn str(text: &str) -> Key {
        Key::Str(text.into())
    }

    pub fn tag(tag: Tag) -> Key {
        Key::Str(tag.text().into())
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
            Key::Str(s) => {
                hasher.update(&[4]);
                hasher.update(&(s.len() as u64).to_le_bytes());
                hasher.update(s.as_bytes())
            }
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
    Key::Str(name.clone())
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
        Expr::Var(name) => Key::tuple([Key::tag(Tag::Var), Key::Str(name.clone())]),
        Expr::Literal(lit) => Key::tuple([
            Key::tag(Tag::Literal),
            Key::num(lit.value),
            Key::str(lit.dtype.name()),
        ]),
        Expr::Builtin(builtin) => Key::tuple([Key::tag(Tag::Builtin), Key::str(builtin.text())]),
        Expr::Binary(op, l, r) => Key::tuple([
            Key::tag(Tag::Binary),
            Key::str(op.text()),
            expression_key(l, leaf),
            expression_key(r, leaf),
        ]),
        Expr::Call(name, args) => Key::tuple([
            Key::tag(Tag::Call),
            Key::Str(name.clone()),
            Key::Tuple(args.iter().map(|a| expression_key(a, leaf)).collect()),
        ]),
        Expr::Ternary(a, b, c) => Key::tuple([
            Key::tag(Tag::Ternary),
            expression_key(a, leaf),
            expression_key(b, leaf),
            expression_key(c, leaf),
        ]),
        Expr::Cast(dtype, inner) => Key::tuple([
            Key::tag(Tag::Cast),
            Key::Str(dtype.clone()),
            expression_key(inner, leaf),
        ]),
        Expr::FlatIndex(buffer, index) => Key::tuple([
            Key::tag(Tag::FlatIndex),
            Key::Str(buffer.clone()),
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
                "Var",
                "Literal",
                "Builtin",
                "Binary",
                "Call",
                "Ternary",
                "Cast",
                "FlatIndex",
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
            ]
        );
    }
}
