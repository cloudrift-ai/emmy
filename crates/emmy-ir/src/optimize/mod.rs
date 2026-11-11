//! The Kernel IR optimizer: the exact rewrites a kernel body takes after lowering, each removing work and adding none,
//! so none is a decision and none answers to a knob. It is a pipeline pass's mechanics, not normalization: a kernel
//! body is no identity's input, and normalization's steps run inside `normalize_body` alone. It reads the crate's
//! simplifier, numbering and placement as they are; kernel statements need no arm of their own there, since every
//! one is read through the protocol (its fields' roles, its effects).

use crate::Result;
use crate::body::Body;
use crate::normalize::{drop_unread_values, merge_congruent_states, share_values, simplify_body};
use crate::numbering::{congruent_states, group_keys};

/// `body` with every expression simplified under the ranges of the axes around it (a `Tile`'s included), every pure
/// value a scope computes twice computed once, every group of congruent carried states one state, and every pure
/// value nothing reads dropped, repeated until a round changes nothing. Nothing moves to another scope: moving a
/// value out of a loop lengthens its live range, which is a choice for evidence.
pub fn optimize_kernel_body(body: &Body) -> Result<Body> {
    let mut stmts = body.clone();
    for _ in 0..8 {
        let simplified = simplify_body(&stmts)?;
        let groups = congruent_states(&simplified);
        let shared = share_values(&simplified, &group_keys(&groups))?;
        let next = drop_unread_values(&merge_congruent_states(&shared, &groups));
        if next == stmts {
            return Ok(next);
        }
        stmts = next;
    }
    Ok(stmts)
}

#[cfg(test)]
mod tests {
    use super::optimize_kernel_body;
    use crate::normalize::testing::{body, text};

    /// The text of `given` after the optimizer.
    fn optimized(given: &str) -> String {
        text(&optimize_kernel_body(&body(given)).unwrap())
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
}
