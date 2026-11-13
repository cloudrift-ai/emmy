//! The input check of a code generator: what a kernel body says itself, so that whatever emits its code decides
//! nothing.
//!
//! - **Typed.** Every value carries its dtype and every conversion is a definition (`types::check_types`).
//! - **Flat.** A `Load`, a `Write` and a `CpAsyncCopy` hold one element index into their buffer. The accesses their
//!   dimensions define keep them: one of a paged buffer (its index along the paged axis picks the page), a `Write`
//!   under a Volta slab layout (a function of row and column), and the shared-memory side of a copy with a lane
//!   offset (its swizzle is split over the tile's row and column).
//! - **No requests.** No loop carries the `unroll` flag: a loop in the body is a loop in the code. The kernel
//!   lowering's unroll pass spends the flag.
//!
//! The error names the statement.

use std::collections::{HashMap, HashSet};

use crate::body::Body;
use crate::dtype::DType;
use crate::statement::*;
use crate::text::print::statement_text;
use crate::types::check_types;
use crate::{Name, Result, value_error};

/// The slab layouts that are a function of a store's logical row and column (the Kernel IR's Volta layouts).
const TWO_DIMENSIONAL_LAYOUTS: [&str; 2] = ["V70A", "V70B"];

/// Check that `body` is a code generator's input. `buffers` gives the element dtype of the buffers the body reads
/// and writes by name, `paged` the buffers held as a table of pages.
pub fn check_input(
    body: &Body,
    buffers: &HashMap<Name, DType>,
    paged: &HashSet<Name>,
) -> Result<()> {
    check_types(body, buffers)?;
    structure(body, paged)
}

fn structure(body: &Body, paged: &HashSet<Name>) -> Result<()> {
    for stmt in body.stmts() {
        let fault = match &**stmt {
            Statement::Loop(l) if l.unroll => Some(UNROLL),
            Statement::StridedLoop(l) if l.unroll => Some(UNROLL),
            Statement::Load(load) if !paged.contains(&load.input) && load.index.len() != 1 => {
                Some(FLAT)
            }
            Statement::Write(write)
                if !paged.contains(&write.output)
                    && !TWO_DIMENSIONAL_LAYOUTS.contains(&write.swizzle.as_str())
                    && write.index.len() != 1 =>
            {
                Some(FLAT)
            }
            Statement::CpAsyncCopy(copy)
                if copy.src_index.len() != 1
                    || (copy.lane_index.is_none() && copy.smem_index.len() != 1) =>
            {
                Some(FLAT)
            }
            _ => None,
        };
        if let Some(fault) = fault {
            let text = statement_text(stmt).unwrap_or_else(|_| format!("{stmt:?}"));
            let head = text.lines().next().unwrap_or_default().trim();
            return Err(value_error(format!("`{head}` {fault}")));
        }
        for nested in stmt.nested() {
            structure(&nested, paged)?;
        }
    }
    Ok(())
}

const UNROLL: &str = "asks for unrolling: a loop of a code generator's input is a loop in the code";
const FLAT: &str =
    "holds one index per dimension: an access of a code generator's input holds one flat index";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::parse::parse_body;

    fn check(source: &str) -> Result<()> {
        let paged = HashSet::from([Name::new("K")]);
        check_input(&parse_body(source).unwrap(), &HashMap::new(), &paged)
    }

    #[test]
    fn a_loop_that_asks_for_unrolling_is_refused() {
        check("for k in 0..4:\n    f32 x = load A[k]\n    f32 O[k] = x").unwrap();
        let flagged = check(
            "if k < 2:\n    for k in 0..4 unroll=true:\n        f32 x = load A[k]\n        f32 O[k] = x",
        )
        .unwrap_err()
        .to_string();
        assert!(
            flagged.contains("`for k in 0..4 unroll=true:` asks for unrolling"),
            "{flagged}"
        );
    }

    #[test]
    fn an_access_holds_one_index_unless_its_dimensions_define_it() {
        check("f32 x = load A[4 * i + j]\nf32 O[4 * i + j] = x").unwrap();
        // A paged buffer, a Volta layout and a copy with a lane offset keep their dimensions.
        check("f32 x = load K[i, j]\nf32 K[i, j] = x").unwrap();
        check("f32 x = load A[i]\nf32 s[i, j] = x swizzle=V70A").unwrap();
        check("CpAsyncCopy(smem=s, smem_index=(i, j), src=a, src_index=(i), lane_index=(i, j))")
            .unwrap();
        for (source, statement) in [
            (
                "f32 x = load A[i, j]\nf32 O[i] = x",
                "`f32 x = load A[i, j]`",
            ),
            ("f32 x = load A[i]\nf32 O[i, j] = x", "`f32 O[i, j] = x`"),
            (
                "f32 x = load A[i]\nf32 s[i, j] = x swizzle=B128",
                "`f32 s[i, j] = x swizzle=B128`",
            ),
            (
                "CpAsyncCopy(smem=s, smem_index=(i, j), src=a, src_index=(i))",
                "`CpAsyncCopy(smem=s, smem_index=(i, j), src=a, src_index=(i))`",
            ),
            (
                "CpAsyncCopy(smem=s, smem_index=(i), src=a, src_index=(i, j), lane_index=(i, j))",
                "lane_index=(i, j))`",
            ),
        ] {
            let refused = check(source).unwrap_err().to_string();
            assert!(
                refused.contains(statement) && refused.contains("holds one index per dimension"),
                "{refused}"
            );
        }
    }
}
