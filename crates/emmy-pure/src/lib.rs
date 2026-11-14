//! The compiler's pure terms, over the `emmy-ir` statement vocabulary: the `Lambda` every IR level binds with. Pure
//! Rust; the Python extension (`emmy-ir-py`) wraps them as the classes `emmy.compiler.ir.pure` exports. A term binds
//! and never executes: it reaches a statement position only by being rendered to statements.

pub mod lambda;

pub use lambda::Lambda;
