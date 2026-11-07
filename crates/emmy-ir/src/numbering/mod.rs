//! Value numbering of a body with coordinates abstracted: the numbers the canonical order and the identity read.
//!
//! A statement's NUMBER is its kind and payload over its operands' numbers, with every maximal coordinate-only index
//! expression a PARAMETER of the statement, numbered by first appearance. A reduce binds the parameters that mention
//! its axis. Commutative operands order by number. A body's identity is the hash of its SCOPE TREE: every block a
//! node described by its kind and its extent or predicate, every leaf an instance with its coordinates spelled by
//! binding depth, every scope's members sorted. The external buffers and carried states it reads through are keyed
//! by USE, never by spelling, refined where two tie by individualizing each in turn.
//!
//! Statements are numbered by position in the pre-order walk, so no statement is keyed by object identity.

pub mod digest;
pub mod key;
pub mod values;

pub use digest::*;
pub use key::*;
pub use values::*;
