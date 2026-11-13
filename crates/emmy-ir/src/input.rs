//! The input check of a code generator: what a kernel body says itself, so that whatever emits its code decides
//! nothing.
//!
//! - **Typed.** Every value carries its dtype and every conversion is a definition (`types::check_types`).
//! - **No requests.** No loop carries the `unroll` flag: a loop in the body is a loop in the code. The kernel
//!   lowering's unroll pass spends the flag.
//!
//! The error names the statement.

use std::collections::HashMap;

use crate::body::Body;
use crate::dtype::DType;
use crate::statement::*;
use crate::text::print::statement_text;
use crate::types::check_types;
use crate::{Name, Result, value_error};

/// Check that `body` is a code generator's input. `buffers` gives the element dtype of the buffers the body reads
/// and writes by name.
pub fn check_input(body: &Body, buffers: &HashMap<Name, DType>) -> Result<()> {
    check_types(body, buffers)?;
    structure(body)
}

fn structure(body: &Body) -> Result<()> {
    for stmt in body.stmts() {
        let request = match &**stmt {
            Statement::Loop(l) => l.unroll,
            Statement::StridedLoop(l) => l.unroll,
            _ => false,
        };
        if request {
            let text = statement_text(stmt).unwrap_or_else(|_| format!("{stmt:?}"));
            let head = text.lines().next().unwrap_or_default().trim();
            return Err(value_error(format!(
                "`{head}` asks for unrolling: a loop of a code generator's input is a loop in the code"
            )));
        }
        for nested in stmt.nested() {
            structure(&nested)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::parse::parse_body;

    fn check(source: &str) -> Result<()> {
        check_input(&parse_body(source).unwrap(), &HashMap::new())
    }

    #[test]
    fn a_loop_that_asks_for_unrolling_is_refused() {
        check("for k in 0..4:\n    f32 x = load A[k]\n    f32 O[k] = x").unwrap();
        let flagged = check("if k < 2:\n    for k in 0..4 unroll=true:\n        f32 x = load A[k]\n        f32 O[k] = x")
            .unwrap_err()
            .to_string();
        assert!(
            flagged.contains("`for k in 0..4 unroll=true:` asks for unrolling"),
            "{flagged}"
        );
    }
}
