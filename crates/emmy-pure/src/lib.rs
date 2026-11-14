//! The compiler's pure terms, over the `emmy-ir` statement vocabulary: the `Lambda` every IR level binds with, and the Tile IR `Fold` term with the `Twist` recipe that conjugates its monoid. Pure
//! Rust; the Python extension (`emmy-ir-py`) wraps them as the classes `emmy.compiler.ir.pure` exports. A term binds
//! and never executes: it reaches a statement position only by being rendered to statements.

pub mod fold;
pub mod lambda;
pub mod twist;

pub use fold::Fold;
pub use lambda::Lambda;
pub use twist::{Channel, Recipe, Twist};
