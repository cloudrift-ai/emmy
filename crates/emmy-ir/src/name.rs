//! Names, interned: an SSA value, an axis, a buffer. A name is its text, kept once in one process-wide table and
//! never freed, so equality and hashing compare the pointer, a copy is a word and reading the text costs nothing. A
//! name orders by its text, as the canonical order sorts names and as every digest spells them. A lowering mints
//! names a body at a time; the A100 golden's fresh lowering and the corpus together intern some twenty thousand.

use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::{Mutex, OnceLock};

use crate::locked;

/// A name: an SSA value, an axis, a buffer.
#[derive(Clone, Copy)]
pub struct Name(&'static str);

fn table() -> &'static Mutex<HashSet<&'static str>> {
    static TABLE: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    TABLE.get_or_init(|| Mutex::new(HashSet::new()))
}

impl Name {
    /// The name spelled `text`: the one the table holds, or `text` kept from now on.
    pub fn new(text: &str) -> Name {
        let mut table = locked(table());
        if let Some(&kept) = table.get(text) {
            return Name(kept);
        }
        let kept: &'static str = Box::leak(text.into());
        table.insert(kept);
        Name(kept)
    }

    /// The name's text.
    pub fn as_str(&self) -> &'static str {
        self.0
    }

    /// How many distinct names the process has interned.
    pub fn interned() -> usize {
        locked(table()).len()
    }
}

impl PartialEq for Name {
    fn eq(&self, other: &Name) -> bool {
        std::ptr::eq(self.0, other.0)
    }
}

impl Eq for Name {}

impl Hash for Name {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.0.as_ptr() as usize).hash(state);
    }
}

impl Deref for Name {
    type Target = str;

    fn deref(&self) -> &str {
        self.0
    }
}

impl AsRef<str> for Name {
    fn as_ref(&self) -> &str {
        self.0
    }
}

impl fmt::Display for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl fmt::Debug for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.0, f)
    }
}

impl PartialOrd for Name {
    fn partial_cmp(&self, other: &Name) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Name {
    fn cmp(&self, other: &Name) -> std::cmp::Ordering {
        if *self == *other {
            return std::cmp::Ordering::Equal;
        }
        self.0.cmp(other.0)
    }
}

impl PartialEq<str> for Name {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

impl PartialEq<&str> for Name {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
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
        name.0.to_string()
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
