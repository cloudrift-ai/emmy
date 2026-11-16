//! Strength reduction: the target's spelling of an index division, a pass mechanic an emitter runs last.

use std::sync::Arc;

use crate::Result;
use crate::body::Body;
use crate::elementwise::Operation;
use crate::expr::{ExprClass, Expression, SimplifyCtx};
use crate::normalize::{inner_ctx, launch_ctx, with_bodies};
use crate::statement::Statement;

/// `body` with every integer division and remainder by a power of two whose dividend the ranges prove non-negative
/// (the ranges of the axes around it, and of the built-ins under a `Grid`'s launch) spelled as a right shift and a
/// mask: the target's spelling of `/` and `%`. A signed division rounds toward zero, so a target spells it in a few
/// instructions unless the dividend is known non-negative; the ranges know it. The simplifier reads `/` and `%`, not
/// shifts, so this is no step of the normal form: an emitter runs it on the body it is about to spell.
pub fn reduce_strength(body: &Body) -> Result<Body> {
    reduce_in(body, &SimplifyCtx::empty())
}

fn reduce_in(body: &Body, ctx: &SimplifyCtx) -> Result<Body> {
    let stmts: Result<Vec<Arc<Statement>>> = body
        .stmts()
        .iter()
        .map(|stmt| {
            let launch = launch_ctx(ctx, stmt);
            let ctx = launch.as_ref().unwrap_or(ctx);
            let reduced = Statement::try_map_classed_expressions(stmt, &mut |class, e| {
                Ok(reduced(e, ctx, class))
            })?;
            let nested = reduced.nested();
            if nested.is_empty() {
                return Ok(reduced);
            }
            let inner = inner_ctx(ctx, &reduced);
            let bodies: Result<Vec<Body>> = nested.iter().map(|b| reduce_in(b, &inner)).collect();
            Ok(with_bodies(&reduced, bodies?))
        })
        .collect();
    Ok(body.rebuilt(stmts?))
}

/// `e` with its divisions and remainders by a power of two over a non-negative dividend as shifts and masks, decided
/// top-down on the expression as written: a shift's range is unknown to the analysis, so a reduced operand would
/// hide its parent's dividend. Only an index: a value's division may render in its float type.
fn reduced(e: &Expression, ctx: &SimplifyCtx, class: ExprClass) -> Expression {
    if class != ExprClass::Index {
        return e.clone();
    }
    if let Some((
        op @ (Operation::Divide | Operation::FloorDivide | Operation::Remainder),
        [dividend, divisor],
    )) = e.operation()
        && let Some(power) = divisor.as_literal().and_then(|lit| lit.value.int())
        && power > 1
        && power & (power - 1) == 0
        && dividend.range(ctx).is_some_and(|span| span.lo >= 0)
    {
        let dividend = reduced(dividend, ctx, class);
        return match op {
            Operation::Divide | Operation::FloorDivide => Expression::binary(
                Operation::RightShift,
                dividend,
                Expression::int(i64::from(power.trailing_zeros())),
            ),
            _ => Expression::binary(Operation::BitwiseAnd, dividend, Expression::int(power - 1)),
        };
    }
    let children = e.children();
    if children.is_empty() {
        return e.clone();
    }
    let children = children.iter().map(|c| reduced(c, ctx, class)).collect();
    e.with_children(children).unwrap_or_else(|_| e.clone())
}

#[cfg(test)]
mod tests {
    use super::reduce_strength;
    use crate::normalize::testing::{body, text};

    /// A division and a remainder by a power of two over a non-negative index are a shift and a mask; one whose
    /// dividend may be negative, or by another divisor, stays.
    #[test]
    fn a_power_of_two_division_over_a_non_negative_index_is_a_shift() {
        let given = body(
            "
            for k in 0..64:
                x = load X[k / 8, k % 8, (k - 32) / 8, k / 3]
                O[k] = x
            ",
        );
        let out = text(&reduce_strength(&given).unwrap());
        assert!(
            out.contains("load X[k >> 3, k & 7, (k - 32) / 8, k / 3]"),
            "{out}"
        );
    }

    /// A built-in is non-negative on any launch, and a `Grid` bounds it for its decode and its body.
    #[test]
    fn a_grid_decode_over_the_built_ins_is_shifts_and_masks() {
        let given = body(
            "
            Grid(axes=(a in 0..16, b in 0..32), decode=(block_idx.x / 2, thread_idx.x % 32), block_threads=64):
                O[a, b] = a
            ",
        );
        let out = text(&reduce_strength(&given).unwrap());
        assert!(
            out.contains("decode=(block_idx.x >> 1, thread_idx.x & 31)"),
            "{out}"
        );
    }
}
