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
        Load = "load", Assign = "assign", Affine = "affine", Init = "init", Reduce = "reduce",
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
        // Each tag's name is interned once: a key is built millions of times, and interning takes a global lock.
        static NAMES: std::sync::OnceLock<Vec<Name>> = std::sync::OnceLock::new();
        let names =
            NAMES.get_or_init(|| Tag::ALL.iter().map(|tag| Name::new(tag.text())).collect());
        Key::Str(names[tag as usize])
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

    /// The bytes the hash reads, appended to `out`: a byte tag per primitive, lengths as eight little-endian bytes.
    fn write(&self, out: &mut Vec<u8>) {
        match self {
            Key::None => out.push(0),
            Key::Bool(v) => out.extend_from_slice(&[1, *v as u8]),
            Key::Int(v) => {
                out.push(2);
                out.extend_from_slice(&v.to_le_bytes());
            }
            Key::Float(v) => {
                out.push(3);
                out.extend_from_slice(&v.to_le_bytes());
            }
            Key::Str(s) => Key::write_text(s, out),
            Key::Text(s) => Key::write_text(s, out),
            Key::Num(n) => {
                out.push(5);
                out.extend_from_slice(&n.to_le_bytes());
            }
            Key::Tuple(parts) => {
                out.push(6);
                out.extend_from_slice(&(parts.len() as u64).to_le_bytes());
                for part in parts {
                    part.write(out);
                }
            }
        }
    }

    fn write_text(text: &str, out: &mut Vec<u8>) {
        out.push(4);
        out.extend_from_slice(&(text.len() as u64).to_le_bytes());
        out.extend_from_slice(text.as_bytes());
    }

    /// The key's number: the first sixteen bytes of the BLAKE3 hash of its bytes, little-endian. The bytes are
    /// gathered in one buffer and hashed at once, which costs far less than feeding the hasher a primitive at a time.
    pub fn hash(&self) -> Number {
        thread_local! {
            static BYTES: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        let digest = BYTES.with(|bytes| {
            let mut bytes = bytes.borrow_mut();
            bytes.clear();
            self.write(&mut bytes);
            blake3::hash(&bytes)
        });
        let mut head = [0u8; 16];
        head.copy_from_slice(&digest.as_bytes()[..16]);
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
        Expr::Apply(op, args) => Key::tuple([
            Key::str(ExprType::Apply.text()),
            Key::str(op.name()),
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
