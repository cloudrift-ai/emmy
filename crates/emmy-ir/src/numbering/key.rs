//! Digest material: the `Key` tree every number, parameter and tie is spelled in, and its hash.

use std::sync::Arc;

use crate::Name;
use crate::expr::Num;

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

impl Key {
    pub fn str(text: &str) -> Key {
        Key::Str(text.into())
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
