//! The Kernel IR optimizer: the exact rewrites a kernel body takes after lowering, each removing work and adding none,
//! so none is a decision and none answers to a knob. It is a pipeline pass's mechanics, not normalization: a kernel
//! body is no identity's input, and normalization's steps run inside `normalize_body` alone. It reads the crate's
//! simplifier, numbering and placement as they are; kernel statements need no arm of their own there, since every
//! one is read through the protocol (its fields' roles, its effects).

use std::sync::Arc;

use crate::body::Body;
use crate::elementwise::Operation;
use crate::expr::{ExprClass, Expression, SimplifyCtx};
use crate::normalize::{
    drop_unread_values, extend_simplify_ctx, merge_congruent_states, place_values_keyed,
    share_values, simplify_in, value_canonical, with_bodies,
};
use crate::numbering::{Key, congruent_states, group_keys};
use crate::statement::Statement;
use crate::{Name, Result};

/// `body` with every expression simplified under `ctx` (the launch's ranges of the built-ins) and the ranges of the
/// axes around it (a `Tile`'s included), every pure value a scope computes twice computed once, every group of
/// congruent carried states one state, and every pure value nothing reads dropped, repeated until a round changes
/// nothing. Nothing moves to another scope: moving a value out of a loop lengthens its live range, which is a choice
/// for evidence.
pub fn optimize_kernel_body(body: &Body, ctx: &SimplifyCtx) -> Result<Body> {
    rounds(body, ctx, false)
}

/// [`optimize_kernel_body`] with motion: every value moves out of each loop its operands do not vary in (placement's,
/// as normalization places a body), so a loop computes once what each of its trips computed alike. Nothing leaves a
/// branch, a tile or a grid, which guard what they hold. It lengthens the value's live range across the loop, a trade
/// of registers for work, so a knob decides it and evidence decides the knob.
pub fn hoist_invariants(body: &Body, ctx: &SimplifyCtx) -> Result<Body> {
    rounds(body, ctx, true)
}

fn rounds(body: &Body, ctx: &SimplifyCtx, motion: bool) -> Result<Body> {
    let mut stmts = body.clone();
    for _ in 0..8 {
        let simplified = canonical_indices(&simplify_in(&stmts, ctx)?)?;
        let groups = congruent_states(&simplified);
        let keys = group_keys(&groups);
        let shared = if motion {
            place_values_keyed(&simplified, &keys)?
        } else {
            share_values(&simplified, &keys)?
        };
        let next = drop_unread_values(&merge_congruent_states(&shared, &groups));
        if next == stmts {
            return Ok(next);
        }
        stmts = next;
    }
    Ok(stmts)
}

/// `body` with every index in its affine spelling (`value_canonical` under the names as they stand): a sum's literal
/// terms fold and like terms merge, so two spellings of one address are one value to share. Only an index: a value's
/// sum is float arithmetic, which its order rounds.
fn canonical_indices(body: &Body) -> Result<Body> {
    let rank = |name: &Name| Key::Str(*name);
    let stmts: Result<Vec<Arc<Statement>>> = body
        .stmts()
        .iter()
        .map(|stmt| {
            let stmt = Statement::try_map_classed_expressions(stmt, &mut |class, e| {
                Ok(if class == ExprClass::Index {
                    value_canonical(e, &rank, class)
                } else {
                    e.clone()
                })
            })?;
            let nested = stmt.nested();
            if nested.is_empty() {
                return Ok(stmt);
            }
            let bodies: Result<Vec<Body>> = nested.iter().map(canonical_indices).collect();
            Ok(with_bodies(&stmt, bodies?))
        })
        .collect();
    Ok(body.rebuilt(stmts?))
}

/// `body` with every integer division and remainder by a power of two whose dividend the ranges prove non-negative
/// (under `ctx` and the axes around it) spelled as a right shift and a mask: the target's spelling of `/` and `%`,
/// which the simplifier does not read, so it runs after the optimizer. A signed division rounds toward zero, so a
/// target spells it in a few instructions unless the dividend is known non-negative; the ranges know it.
pub fn reduce_strength(body: &Body, ctx: &SimplifyCtx) -> Result<Body> {
    let stmts: Result<Vec<Arc<Statement>>> = body
        .stmts()
        .iter()
        .map(|stmt| {
            let stmt = Statement::try_map_classed_expressions(stmt, &mut |class, e| {
                Ok(reduced(e, ctx, class))
            })?;
            let nested = stmt.nested();
            if nested.is_empty() {
                return Ok(stmt);
            }
            let inner = stmt
                .bound_axes()
                .iter()
                .fold(ctx.clone(), |inner, axis| extend_simplify_ctx(&inner, axis));
            let bodies: Result<Vec<Body>> =
                nested.iter().map(|b| reduce_strength(b, &inner)).collect();
            Ok(with_bodies(&stmt, bodies?))
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
    use super::{hoist_invariants, optimize_kernel_body, reduce_strength};
    use crate::expr::{Interval, SimplifyCtx};
    use crate::normalize::testing::{body, text};

    /// The text of `given` after the optimizer.
    fn optimized(given: &str) -> String {
        text(&optimize_kernel_body(&body(given), &SimplifyCtx::empty()).unwrap())
    }

    #[test]
    fn a_value_two_cells_compute_is_computed_once() {
        let out = optimized(
            "
            Tile(axes=(m in 0..4, n in 0..8)):
                x0 = load X[m]
                y0 = multiply(x0, 2.0)
                x1 = load X[m]
                y1 = multiply(x1, 2.0)
                O[m, 2 * n] = y0
                O[m, 2 * n + 1] = y1
            ",
        );
        assert_eq!(out.matches("load X").count(), 1, "{out}");
        assert_eq!(out.matches("x0 * 2.0").count(), 1, "{out}");
        assert!(!out.contains("y1"), "{out}");
    }

    #[test]
    fn an_index_simplifies_under_the_tile_axes() {
        let out = optimized(
            "
            Tile(axes=(m in 0..4, _lane in 0..32)):
                x = load X[(32 * m + _lane) / 32, (32 * m + _lane) % 32]
                O[m, _lane] = x
            ",
        );
        assert!(out.contains("load X[m, _lane]"), "{out}");
    }

    /// A value a loop's every step computes alike stays in the loop: moving it out lengthens its live range, a
    /// choice for evidence.
    #[test]
    fn nothing_moves_to_another_scope() {
        let given = "
            Tile(axes=(m in 0..4)):
                for k in 0..16:
                    s = load S[m]
                    t = load T[m, k]
                    u = multiply(s, t)
                    O[m, k] = u
            ";
        assert_eq!(optimized(given), text(&body(given)));
    }

    /// With motion, the value every trip computes alike leaves the loop; nothing leaves the tile.
    #[test]
    fn a_hoist_moves_an_invariant_out_of_the_loop() {
        let out = text(
            &hoist_invariants(
                &body(
                    "
                    Tile(axes=(m in 0..4)):
                        for k in 0..16:
                            s = load S[m]
                            t = load T[m, k]
                            u = multiply(s, t)
                            O[m, k] = u
                    ",
                ),
                &SimplifyCtx::empty(),
            )
            .unwrap(),
        );
        let (tile, loop_) = out.split_once("for k").unwrap();
        assert!(
            tile.contains("load S[m]") && !loop_.contains("load S[m]"),
            "{out}"
        );
        assert!(out.starts_with("Tile("), "{out}");
    }

    /// A barrier orders what the body writes: a read of a buffer nothing writes is one value on both sides of it, a
    /// read of the shared slab the body fills is two.
    #[test]
    fn a_barrier_orders_only_what_the_body_writes() {
        let out = optimized(
            "
            Tile(axes=(m in 0..4, _lane in 0..32)):
                x = load X[m, _lane]
                s = load S[_lane]
                S[_lane] = x
                Sync
                y = load X[m, _lane]
                t = load S[_lane]
                O[m, _lane] = y
                P[m, _lane] = t
                Q[m, _lane] = s
            ",
        );
        assert_eq!(out.matches("load X").count(), 1, "{out}");
        assert_eq!(out.matches("load S").count(), 2, "{out}");
    }

    #[test]
    fn a_barrier_stays_where_it_is() {
        let given = "
            Tile(axes=(m in 0..4)):
                x = load X[m]
                Sync
                y = load X[m]
                O[m] = y
                P[m] = x
            ";
        let out = optimized(given);
        assert_eq!(out.matches("Sync").count(), 1, "{out}");
    }

    /// Two accumulators of one value are congruent, but a combine that updates each in place after the loop holds
    /// them as two states: the merge leaves both.
    #[test]
    fn a_state_a_combine_updates_in_place_stays() {
        let out = optimized(
            "
            Tile(axes=(m in 0..4, _lane in 0..32)):
                for k in 0..16:
                    x = load X[m, k]
                    a <- add(a, x) over k
                    y = load X[m, k]
                    b <- add(b, y) over k
                WarpShuffle(state=(a, b), state_b=(a__o, b__o), length=32):
                    combine_states:
                        a = a + a__o
                        b = b + b__o
                O[m, 0] = a
                O[m, 1] = b
            ",
        );
        assert_eq!(out.matches(" <- add").count(), 2, "{out}");
        assert!(out.contains("state=(a, b)"), "{out}");
    }

    /// A grid's decode of its flat thread id splits into the block's and the thread's coordinates under the launch's
    /// ranges of the built-ins (32 CTAs of 128 threads over 4096 cells), and the body reads the axes' ranges.
    #[test]
    fn a_decode_splits_under_the_launch() {
        let launch = SimplifyCtx::empty()
            .extend("block_idx.x", Interval::new(0, 31), None)
            .extend("thread_idx.x", Interval::new(0, 127), None);
        let given = body(
            "
            Grid(axes=(a in 0..16, b in 0..2, c in 0..4, d in 0..32), decode=((block_idx.x * 128 + thread_idx.x) / 256, (block_idx.x * 128 + thread_idx.x) / 128 % 2, (block_idx.x * 128 + thread_idx.x) / 32 % 4, (block_idx.x * 128 + thread_idx.x) % 32), block_threads=128):
                x = load X[(32 * c + d) / 32, (32 * c + d) % 32]
                O[a, b, c, d] = x
            ",
        );
        let out = text(&optimize_kernel_body(&given, &launch).unwrap());
        assert!(
            out.contains(
                "decode=(block_idx.x / 2, block_idx.x % 2, thread_idx.x / 32, thread_idx.x % 32)"
            ),
            "{out}"
        );
        assert!(out.contains("load X[c, d]"), "{out}");
    }

    /// A one-CTA launch pins the CTA index to 0.
    #[test]
    fn a_builtin_the_launch_pins_is_its_value() {
        let launch = SimplifyCtx::empty()
            .extend("block_idx.x", Interval::new(0, 0), None)
            .extend("thread_idx.x", Interval::new(0, 255), None);
        let given = body(
            "
            Grid(axes=(a in 0..100), decode=(block_idx.x * 256 + thread_idx.x), block_threads=256):
                if block_idx.x * 256 + thread_idx.x < 100:
                    O[a] = a
            ",
        );
        let out = text(&optimize_kernel_body(&given, &launch).unwrap());
        assert!(out.contains("decode=(thread_idx.x)"), "{out}");
        assert!(out.contains("if thread_idx.x < 100:"), "{out}");
    }

    /// With no launch in the context a built-in takes what the hardware allows: a thread index is below 1024.
    #[test]
    fn a_builtin_takes_the_hardware_range() {
        let given = body(
            "
            Grid(axes=(a in 0..4096), decode=(block_idx.x * 128 + thread_idx.x % 1024), block_threads=128):
                O[a] = a
            ",
        );
        let out = text(&optimize_kernel_body(&given, &SimplifyCtx::empty()).unwrap());
        assert!(
            out.contains("decode=(128 * block_idx.x + thread_idx.x)"),
            "{out}"
        );
    }

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
        let out = text(&reduce_strength(&given, &SimplifyCtx::empty()).unwrap());
        assert!(
            out.contains("load X[k >> 3, k & 7, (k - 32) / 8, k / 3]"),
            "{out}"
        );
    }
}
