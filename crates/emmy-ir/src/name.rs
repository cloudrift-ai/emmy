//! Names, interned: an SSA value, an axis, a buffer. A name is a symbol into one process-wide table, so equality and
//! hashing are integer compares and a copy is a word; its text is resolved on read (`Deref<Target = str>`,
//! `Display`), and a name orders by its text, as the canonical order and the digests spell it. The table never frees:
//! a lowering mints names a body at a time, and the distinct names of a long compile stay in the thousands.

use std::collections::HashMap;
use std::fmt;
use std::ops::Deref;
use std::sync::{OnceLock, RwLock};

use crate::locked_read;
use crate::locked_write;

/// A name: an SSA value, an axis, a buffer.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Name(u32);

/// The texts of every name this process has seen, each once, and the symbol of each text.
struct Table {
    symbols: HashMap<&'static str, u32>,
    texts: Vec<&'static str>,
}

fn table() -> &'static RwLock<Table> {
    static TABLE: OnceLock<RwLock<Table>> = OnceLock::new();
    TABLE.get_or_init(|| {
        RwLock::new(Table {
            symbols: HashMap::new(),
            texts: Vec::new(),
        })
    })
}

impl Name {
    /// The name spelled `text`: the symbol it has, or the next one.
    pub fn new(text: &str) -> Name {
        if let Some(&symbol) = locked_read(table()).symbols.get(text) {
            return Name(symbol);
        }
        let mut table = locked_write(table());
        if let Some(&symbol) = table.symbols.get(text) {
            return Name(symbol);
        }
        let text: &'static str = Box::leak(text.into());
        let symbol = table.texts.len() as u32;
        table.texts.push(text);
        table.symbols.insert(text, symbol);
        Name(symbol)
    }

    /// The name's text.
    pub fn as_str(&self) -> &'static str {
        locked_read(table()).texts[self.0 as usize]
    }

    /// How many distinct names the process has interned.
    pub fn interned() -> usize {
        locked_read(table()).texts.len()
    }
}

impl Deref for Name {
    type Target = str;

    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<str> for Name {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl fmt::Debug for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_str(), f)
    }
}

impl PartialOrd for Name {
    fn partial_cmp(&self, other: &Name) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Name {
    fn cmp(&self, other: &Name) -> std::cmp::Ordering {
        if self.0 == other.0 {
            return std::cmp::Ordering::Equal;
        }
        self.as_str().cmp(other.as_str())
    }
}

impl PartialEq<str> for Name {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<&str> for Name {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl From<&str> for Name {
    fn from(text: &str) -> Name {
        Name::new(text)
    }
}

impl From<String> for Name {
    fn from(text: String) -> Name {
        Name::new(&text)
    }
}

impl From<&String> for Name {
    fn from(text: &String) -> Name {
        Name::new(text)
    }
}

impl From<Name> for String {
    fn from(name: Name) -> String {
        name.as_str().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_name_is_its_text_and_orders_by_it() {
        let (a, b, a2) = (
            Name::new("acc0"),
            Name::new("v1"),
            Name::from("acc0".to_string()),
        );
        assert_eq!(a, a2);
        assert_ne!(a, b);
        assert!(
            a < b
                && &*a == "acc0"
                && a == "acc0"
                && format!("{a}") == "acc0"
                && format!("{a:?}") == "\"acc0\""
        );
        assert!(Name::interned() >= 2);
    }
}
