//! The range step: every range a loop states takes its canonical spelling.
//!
//! A reduce loop that folds only part of its axis states the part as its range (`for a4 in 0..a0 + 2 of 62`): the
//! split writes each block so, the roll each step of a loop that grows, and causal attention the keys a row folds.
//! [`canonical_ranges`] simplifies the bounds under the loops around them, drops a range that is the whole axis,
//! and shortens a static extent no trip reaches.

use std::sync::Arc;

use super::inner_ctx;
use crate::Result;
use crate::axis::{Axis, Dim, Range};
use crate::body::Body;
use crate::expr::SimplifyCtx;
use crate::statement::*;

/// `l` over the canonical spelling of `range` under `ctx`: no range when it is the whole axis, and the extent the
/// least bound the analysis proves of the end, where that is below what the extent states.
fn over(l: &Loop, range: Range, ctx: &SimplifyCtx) -> Result<Loop> {
    let range = Range {
        start: range.start.simplify(ctx)?,
        end: range.end.simplify(ctx)?,
    };
    let whole = range.start.int_literal() == Some(0) && range.end == l.axis.extent.expression;
    let mut axis = l.axis.clone();
    let reached = range.interval(ctx).hi + 1;
    if !whole
        && let Some(stated) = l.axis.extent.as_static()
        && reached >= 1
        && reached < i128::from(stated)
    {
        axis = Arc::new(Axis {
            name: l.axis.name,
            extent: Dim::static_(reached as i64),
            window: l.axis.window.clone(),
        });
    }
    Ok(Loop {
        axis,
        range: (!whole).then_some((range.start, range.end)),
        ..l.clone()
    })
}

/// `body` with every range in its canonical spelling. A loop that states no range is left as it is: its extent is
/// its spelling.
pub fn canonical_ranges(body: &Body) -> Result<Body> {
    scope(body, &SimplifyCtx::empty())
}

fn scope(body: &Body, ctx: &SimplifyCtx) -> Result<Body> {
    let mut out: Vec<Arc<Statement>> = Vec::with_capacity(body.len());
    for stmt in body.stmts() {
        let mut stmt = stmt.clone();
        if let Statement::Loop(l) = &*stmt
            && l.range.is_some()
        {
            let ranged = over(l, l.walk(), ctx)?;
            if ranged != *l {
                stmt = Arc::new(Statement::Loop(ranged));
            }
        }
        let nested = stmt.nested();
        if !nested.is_empty() {
            let inner = inner_ctx(ctx, &stmt);
            let bodies: Result<Vec<Body>> =
                nested.iter().map(|child| scope(child, &inner)).collect();
            stmt = super::with_bodies(&stmt, bodies?);
        }
        out.push(stmt);
    }
    Ok(body.rebuilt(out))
}

#[cfg(test)]
mod tests {
    use crate::normalize::normalize_body;
    use crate::normalize::testing::*;

    fn normal(source: &str) -> String {
        text(&normalize_body(&body(source)).unwrap())
    }

    #[test]
    fn a_ranged_loop_reads_back_and_settles() {
        let given = "for a0 in 0..8:\n  for a1 in 0..a0 + 1 of 8:\n    in0 = load X[a0, a1]\n    acc0 <- add(acc0, in0) over a1\n  O[a0] = acc0";
        assert_eq!(normal(given), given);
    }

    #[test]
    fn a_range_that_is_the_whole_axis_is_no_range() {
        let given = "
      for i in 0..8:
        for k in 0..8 of 8:
          x = load X[i, k]
          s <- add(s, x) over k
        O[i] = s
            ";
        assert!(
            normal(given).contains("for a1 in 0..8:"),
            "{}",
            normal(given)
        );
    }

    #[test]
    fn a_static_end_narrows_the_extent_too() {
        let given = "
      for i in 0..8:
        for k in 2..6 of 8:
          x = load X[i, k]
          s <- add(s, x) over k
        O[i] = s
            ";
        assert!(
            normal(given).contains("for a1 in 2..6:"),
            "{}",
            normal(given)
        );
    }
}
