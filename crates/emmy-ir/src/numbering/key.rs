//! Digest material: the `Key` tree every number, parameter and tie is spelled in, and its hash.

use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::Name;
use crate::closed_set;
use crate::expr::Num;
use crate::expr::{Expr, ExprType, Expression};

/// Digest material: a tree of primitives, hashed to a number and ordered totally.
#[derive(Clone)]
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
    /// A parameter, shared rather than copied: it hashes, compares and orders as the key it holds.
    Param(Param),
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
            Key::Param(param) => param.key().write(out),
        }
    }

    /// The key a parameter holds, or the key itself.
    fn peeled(&self) -> &Key {
        match self {
            Key::Param(param) => param.key(),
            other => other,
        }
    }

    /// The variant's position, which orders two keys of different variants.
    fn rank(&self) -> u8 {
        match self {
            Key::None => 0,
            Key::Bool(_) => 1,
            Key::Int(_) => 2,
            Key::Float(_) => 3,
            Key::Str(_) => 4,
            Key::Text(_) => 5,
            Key::Num(_) => 6,
            Key::Tuple(_) => 7,
            Key::Param(_) => unreachable!("a peeled key is never a parameter"),
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

impl PartialEq for Key {
    fn eq(&self, other: &Key) -> bool {
        if let (Key::Param(a), Key::Param(b)) = (self, other) {
            return a == b;
        }
        match (self.peeled(), other.peeled()) {
            (Key::None, Key::None) => true,
            (Key::Bool(a), Key::Bool(b)) => a == b,
            (Key::Int(a), Key::Int(b)) => a == b,
            (Key::Float(a), Key::Float(b)) => a == b,
            (Key::Str(a), Key::Str(b)) => a == b,
            (Key::Text(a), Key::Text(b)) => a == b,
            (Key::Num(a), Key::Num(b)) => a == b,
            (Key::Tuple(a), Key::Tuple(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for Key {}

/// Spelled as derived, a parameter as the key it holds: an operand's spelling in a shell's text is this.
impl std::fmt::Debug for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Key::None => f.write_str("None"),
            Key::Bool(v) => f.debug_tuple("Bool").field(v).finish(),
            Key::Int(v) => f.debug_tuple("Int").field(v).finish(),
            Key::Float(v) => f.debug_tuple("Float").field(v).finish(),
            Key::Str(v) => f.debug_tuple("Str").field(v).finish(),
            Key::Text(v) => f.debug_tuple("Text").field(v).finish(),
            Key::Num(v) => f.debug_tuple("Num").field(v).finish(),
            Key::Tuple(v) => f.debug_tuple("Tuple").field(v).finish(),
            Key::Param(param) => param.key().fmt(f),
        }
    }
}

impl Ord for Key {
    fn cmp(&self, other: &Key) -> Ordering {
        if let (Key::Param(a), Key::Param(b)) = (self, other) {
            return a.cmp(b);
        }
        let (a, b) = (self.peeled(), other.peeled());
        match (a, b) {
            (Key::Bool(a), Key::Bool(b)) => a.cmp(b),
            (Key::Int(a), Key::Int(b)) => a.cmp(b),
            (Key::Float(a), Key::Float(b)) => a.cmp(b),
            (Key::Str(a), Key::Str(b)) => a.cmp(b),
            (Key::Text(a), Key::Text(b)) => a.cmp(b),
            (Key::Num(a), Key::Num(b)) => a.cmp(b),
            (Key::Tuple(a), Key::Tuple(b)) => a.cmp(b),
            _ => a.rank().cmp(&b.rank()),
        }
    }
}

impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Key) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Hash for Key {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let key = self.peeled();
        state.write_u8(key.rank());
        match key {
            Key::None | Key::Param(_) => {}
            Key::Bool(v) => v.hash(state),
            Key::Int(v) => v.hash(state),
            Key::Float(v) => v.hash(state),
            Key::Str(v) => v.hash(state),
            Key::Text(v) => v.hash(state),
            Key::Num(v) => v.hash(state),
            Key::Tuple(v) => v.hash(state),
        }
    }
}

/// A parameter: the form of a coordinate-only expression, every axis spelled by its binding depth. A value carries
/// its parameters to every statement that reads it, so a parameter is shared rather than copied, and keeps a
/// fingerprint so two parameters compare at once unless they are equal.
#[derive(Clone)]
pub struct Param(Arc<(u64, Key)>);

impl Param {
    pub fn new(key: Key) -> Param {
        let mut hasher = std::hash::DefaultHasher::new();
        Hash::hash(&key, &mut hasher);
        Param(Arc::new((hasher.finish(), key)))
    }

    pub fn key(&self) -> &Key {
        &self.0.1
    }
}

/// Parameters as one key: a tuple of them, each shared.
pub fn params_key(params: &[Param]) -> Key {
    Key::Tuple(params.iter().cloned().map(Key::Param).collect())
}

impl From<Param> for Key {
    fn from(param: Param) -> Key {
        Key::Param(param)
    }
}

impl PartialEq for Param {
    fn eq(&self, other: &Param) -> bool {
        Arc::ptr_eq(&self.0, &other.0) || (self.0.0 == other.0.0 && self.0.1 == other.0.1)
    }
}

impl Eq for Param {}

impl Ord for Param {
    fn cmp(&self, other: &Param) -> Ordering {
        if Arc::ptr_eq(&self.0, &other.0) {
            return Ordering::Equal;
        }
        self.key().cmp(other.key())
    }
}

impl PartialOrd for Param {
    fn partial_cmp(&self, other: &Param) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Hash for Param {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.0.0);
    }
}

impl std::fmt::Debug for Param {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.key().fmt(f)
    }
}

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
