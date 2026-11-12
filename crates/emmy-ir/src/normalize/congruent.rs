//! Congruent carried states made one: of each group `congruent_states` finds, the first member stays and every other
//! member whose reads allow it goes, its readers reading the first.

use std::collections::HashMap;
use std::sync::Arc;

use super::with_bodies;
use crate::Name;
use crate::body::Body;
use crate::statement::Statement;

/// `body` with every congruent state that can go gone: its update dropped and its reads reading the group's first
/// member. Congruent states hold one value at every step, so a read is the first member's wherever its value is the
/// same version. A carried state is read through `pre`, the step before wherever it sits, so every read can go. An
/// accumulator read in its own scope reads the old value ahead of its update and the new one behind it, which the
/// first member's update may not match; such a member stays until placement, numbering the group alike, has merged
/// those reads into the first member's, and its reads after the loop then read the first member's final value.
pub fn merge_congruent_states(body: &Body, groups: &[Vec<Name>]) -> Body {
    let mut alias: HashMap<Name, Name> = HashMap::new();
    for group in groups {
        for member in &group[1..] {
            if can_go(body, *member) {
                alias.insert(*member, group[0]);
            }
        }
    }
    if alias.is_empty() {
        return body.clone();
    }
    rebuilt(body, &alias)
}

fn updates(stmt: &Statement, name: Name) -> bool {
    matches!(stmt, Statement::Accum(a) if a.name == name)
        || matches!(stmt, Statement::Carry(c) if c.name == name)
}

/// Whether `stmt`, or a statement nested in it, reads `name`.
fn reads(stmt: &Arc<Statement>, name: Name) -> bool {
    let mut found = false;
    Body::new(vec![stmt.clone()]).walk(&mut |here| {
        let stmt = here.stmt();
        found |= !updates(stmt, name)
            && (stmt.deps().contains(&name) || stmt.expressions().iter().any(|e| e.reads(name)));
    });
    found
}

/// Whether the congruent state `name` can go now: a carried state is read only through `pre`; an accumulator is read
/// nowhere in the scope that updates it but by its update.
fn can_go(body: &Body, name: Name) -> bool {
    let mut carried = false;
    body.walk(&mut |here| {
        carried |= matches!(&**here.stmt(), Statement::Carry(c) if c.name == name)
    });
    if carried {
        let mut plain = false;
        body.walk(&mut |here| {
            let stmt = here.stmt();
            plain |= !matches!(&**stmt, Statement::Pre(_))
                && !updates(stmt, name)
                && (stmt.deps().contains(&name)
                    || stmt.expressions().iter().any(|e| e.reads(name)));
        });
        return !plain;
    }
    scope_of(body, name).is_some_and(|scope| !scope.stmts().iter().any(|stmt| reads(stmt, name)))
}

/// The body whose own statements hold the update of the accumulator `name`.
fn scope_of(body: &Body, name: Name) -> Option<Body> {
    if body.stmts().iter().any(|stmt| updates(stmt, name)) {
        return Some(body.clone());
    }
    body.stmts()
        .iter()
        .flat_map(|stmt| stmt.nested())
        .find_map(|child| scope_of(&child, name))
}

/// `body` without the updates of the states `alias` names, every read of one reading what it maps to.
fn rebuilt(body: &Body, alias: &HashMap<Name, Name>) -> Body {
    let rename = |_: crate::statement::Roles, name: Name| alias.get(&name).copied();
    let empty = HashMap::new();
    let stmts = body
        .stmts()
        .iter()
        .filter(|stmt| !alias.keys().any(|name| updates(stmt, *name)))
        .map(|stmt| {
            let nested = stmt.nested();
            let shell = if nested.is_empty() {
                stmt.clone()
            } else {
                with_bodies(stmt, nested.iter().map(|_| Body::empty()).collect())
            };
            let renamed = Statement::rewrite(
                &shell,
                &crate::protocol::Rewrite {
                    rename: &rename,
                    sigma: &empty,
                    axis_fn: None,
                },
            );
            if nested.is_empty() {
                renamed
            } else {
                with_bodies(
                    &renamed,
                    nested.iter().map(|child| rebuilt(child, alias)).collect(),
                )
            }
        })
        .collect();
    body.rebuilt(stmts)
}

#[cfg(test)]
mod tests {
    use crate::normalize::normalize_uncached;
    use crate::normalize::testing::*;
    use crate::statement::Statement;

    /// The accumulators and the carried states of the normal form of `given`, and its text.
    fn states(given: &str) -> (usize, usize, String) {
        let normal = normalize_uncached(&body(given)).unwrap();
        let (mut accums, mut carries) = (0, 0);
        normal.walk(&mut |here| match &**here.stmt() {
            Statement::Accum(_) => accums += 1,
            Statement::Carry(_) => carries += 1,
            _ => {}
        });
        (accums, carries, text(&normal))
    }

    #[test]
    fn two_seeded_reductions_of_one_value_are_one_read_where_the_first_is() {
        let (accums, _, listing) = states(
            "
            for a in 0..4:
                for b in 0..4:
                    x = load X[a, b]
                    acc1 <- add(acc1, x)
                    y = load X[a, b]
                    acc2 <- add(acc2, y)
                v = add(acc1, acc2)
                Y[a] = v
            ",
        );
        assert_eq!(accums, 1, "{listing}");
        assert!(listing.contains("v0 = 2 * acc0"), "{listing}");
    }

    #[test]
    fn twin_online_softmax_states_are_one_pair() {
        let (accums, _, listing) = states(
            "
            for r in 0..4:
                for k in 0..16:
                    s = load S[r, k]
                    g1 = maximum(m1, s)
                    d1 = subtract(m1, g1)
                    b1 = exp(d1)
                    e1 = subtract(s, g1)
                    p1 = exp(e1)
                    lb1 = multiply(l1, b1)
                    l1 <- add(lb1, p1) over k
                    m1 <- maximum(m1, s) over k
                    t = load S[r, k]
                    g2 = maximum(m2, t)
                    d2 = subtract(m2, g2)
                    b2 = exp(d2)
                    e2 = subtract(t, g2)
                    p2 = exp(e2)
                    lb2 = multiply(l2, b2)
                    l2 <- add(lb2, p2) over k
                    m2 <- maximum(m2, t) over k
                Y[r] = l1
                Z[r] = l2
                M[r] = m1
                N[r] = m2
            ",
        );
        assert_eq!(accums, 2, "{listing}");
        assert_eq!(listing.matches("exp(").count(), 2, "{listing}");
    }

    #[test]
    fn twin_pivots_read_inside_the_loop_are_one() {
        let (accums, _, listing) = states(
            "
            for r in 0..4:
                for k in 0..16:
                    s = load S[r, k]
                    g1 = maximum(m1, s)
                    W1[r, k] = g1
                    m1 <- maximum(m1, s) over k
                    t = load S[r, k]
                    g2 = maximum(m2, t)
                    W2[r, k] = g2
                    m2 <- maximum(m2, t) over k
                Y[r] = m1
                Z[r] = m2
            ",
        );
        assert_eq!(accums, 1, "{listing}");
    }

    const TWIN_CARRIES: &str = "
        for t in 0..8:
            for i in 0..4:
                u = load U[t, i]
                h1p = pre h1[i]
                n1 = add(h1p, u)
                h1[i] <- n1 seed 0
                w = load U[t, i]
                h2p = pre h2[i]
                n2 = add(h2p, w)
                h2[i] <- n2 seed SEED
        for j in 0..4:
            a = pre h1[j]
            Y[j] = a
            b = pre h2[j]
            Z[j] = b
        ";

    #[test]
    fn twin_carried_states_are_one() {
        let (_, carries, listing) = states(&TWIN_CARRIES.replace("SEED", "0"));
        assert_eq!(carries, 1, "{listing}");
    }

    #[test]
    fn carried_states_with_two_seeds_stay_two() {
        let (_, carries, listing) = states(&TWIN_CARRIES.replace("SEED", "1"));
        assert_eq!(carries, 2, "{listing}");
    }

    #[test]
    fn reductions_of_two_values_stay_two() {
        let (accums, _, listing) = states(
            "
            for r in 0..4:
                for k in 0..16:
                    s = load S[r, k]
                    m1 <- maximum(m1, s) over k
                    t = load T[r, k]
                    m2 <- maximum(m2, t) over k
                Y[r] = m1
                Z[r] = m2
            ",
        );
        assert_eq!(accums, 2, "{listing}");
    }

    /// An unseeded loop's accumulators start from whatever they held before it, which need not agree.
    #[test]
    fn unseeded_accumulators_stay_two() {
        let (accums, _, listing) = states(
            "
            for k in 0..8 seed=false:
                x = load X[k]
                left <- add(left, x) over k
                right <- add(right, x) over k
            O[0] = left
            P[0] = right
            ",
        );
        assert_eq!(accums, 2, "{listing}");
    }

    /// A member read ahead of its update after the first member's update has no equal value placement can merge it
    /// into, so it stays; what it computes is still the old value.
    #[test]
    fn a_member_read_after_the_first_update_stays() {
        let (accums, _, listing) = states(
            "
            for r in 0..4:
                for k in 0..16:
                    s = load S[r, k]
                    m1 <- maximum(m1, s) over k
                    g2 = maximum(m2, s)
                    W[r, k] = g2
                    m2 <- maximum(m2, s) over k
                Y[r] = m1
                Z[r] = m2
            ",
        );
        assert_eq!(accums, 2, "{listing}");
    }
}
