//! The folds: a select whose branches bind one value, a literal into the expressions that read it, a value nothing
//! reads, and a value read once into its reader's expression.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::{assign, with_bodies};
use crate::Name;
use crate::body::Body;
use crate::expr::{Expr, Expression};
use crate::protocol::literal_binding;
use crate::statement::*;

// --- folds ----------------------------------------------------------------------------------------------

/// A `Select` whose branches all bind one value is a copy of it.
pub fn fold_trivial_selects(body: &Body) -> Body {
    body.map_stmts(&mut |stmt| {
        if let Statement::Select(s) = &**stmt
            && s.branches.iter().all(|b| b.value == s.branches[0].value)
        {
            return Some(assign(&s.name, Expression::var(&s.branches[0].value), None));
        }
        Some(stmt.clone())
    })
}

/// Every `Let` of a float literal at the default precision spelled into the expressions that read it, in its scope
/// and the scopes under it: a constant is a literal, not a binding. The `Let` stays until nothing reads it.
pub fn inline_literals(body: &Body) -> Body {
    fn scope(body: &Body, env: &HashMap<Name, Expression>) -> Body {
        let mut env = env.clone();
        let mut out = Vec::new();
        for stmt in body.stmts() {
            let mut stmt = if env.is_empty() {
                stmt.clone()
            } else {
                Statement::substitute(stmt, &env)
            };
            let nested = stmt.nested();
            if !nested.is_empty() {
                stmt = with_bodies(
                    &stmt,
                    nested.iter().map(|child| scope(child, &env)).collect(),
                );
            }
            for name in stmt.defines() {
                env.remove(&name);
            }
            if let Some((name, value)) = literal_binding(&stmt) {
                env.insert(*name, Expression::float(value));
            }
            out.push(stmt);
        }
        body.rebuilt(out)
    }
    scope(body, &HashMap::new())
}

/// Every name read anywhere in `body`, counted: SSA reads and expression variables, less a statement's own definitions.
pub(crate) fn read_counts(body: &Body, exposed: &HashSet<Name>) -> HashMap<Name, usize> {
    let mut counts: HashMap<Name, usize> = exposed.iter().map(|name| (*name, 1)).collect();
    body.walk(&mut |here| {
        let stmt = here.stmt();
        // Every occurrence counts: a value an expression reads twice is read twice. A name a statement reads through
        // no expression (a store's value, a reduction's own state) counts once.
        let own: HashSet<Name> = stmt.defines().into_iter().collect();
        let mut here: HashMap<Name, usize> = HashMap::new();
        fn occurrences(e: &Expression, out: &mut HashMap<Name, usize>) {
            match &**e {
                Expr::Var(name) => *out.entry(*name).or_default() += 1,
                _ => (**e).each_child(&mut |child| occurrences(child, out)),
            }
        }
        for expression in stmt.expressions() {
            occurrences(&expression, &mut here);
        }
        for name in stmt.deps() {
            here.entry(name).or_insert(1);
        }
        for (name, count) in here {
            if !own.contains(&name) {
                *counts.entry(name).or_default() += count;
            }
        }
    });
    counts
}

pub(crate) fn is_droppable_pure(stmt: &Statement) -> bool {
    match stmt {
        Statement::Load(l) => l.carried.is_empty(),
        Statement::Accum(_)
        | Statement::Assign(_)
        | Statement::Init(_)
        | Statement::Let(_)
        | Statement::Select(_) => true,
        _ => false,
    }
}

/// Drop every pure statement whose values nothing reads, and a loop or branch left with no statement. A body that
/// stores nothing is a fragment whose values are all its reader's, so nothing is dropped from it.
pub fn drop_unread_values(body: &Body) -> Body {
    if body.writes().is_empty() {
        return body.clone();
    }
    let mut stmts = body.clone();
    loop {
        let reads = read_counts(&stmts, &HashSet::new());
        let mut dropped = false;
        stmts = stmts.map_stmts(&mut |stmt| {
            let unread = is_droppable_pure(stmt)
                && !stmt
                    .defines()
                    .iter()
                    .any(|name| reads.get(name).is_some_and(|c| *c > 0));
            let nested = stmt.nested();
            if unread || (!nested.is_empty() && nested.iter().all(|b| b.is_empty())) {
                dropped = true;
                return None;
            }
            Some(stmt.clone())
        });
        if !dropped {
            return stmts;
        }
    }
}

/// A pure value read once folds into its reader's expression, chains deep, until the one reader reads it by name (a
/// store, a select branch, a carry) or from another scope, a dtype stamps it, or the reader's class or the depth bound
/// refuses the folded expression. A value read twice stays a statement: sharing is what a statement expresses.
/// `exposed` names are read once more, from outside the body.
pub fn fold_single_readers(body: &Body, exposed: &HashSet<Name>) -> Body {
    fn scope(body: &Body, counts: &HashMap<Name, usize>) -> Body {
        let mut stmts: Vec<Arc<Statement>> = body.stmts().to_vec();
        for stmt in &mut stmts {
            let nested = stmt.nested();
            if !nested.is_empty() {
                *stmt = with_bodies(
                    stmt,
                    nested.iter().map(|child| scope(child, counts)).collect(),
                );
            }
        }
        // Two definitions of one value are the numbering's to merge into a value read twice, not the fold's to spell
        // twice.
        let mut seen: HashMap<Expression, usize> = HashMap::new();
        for stmt in &stmts {
            if let Statement::Assign(a) = &**stmt {
                *seen.entry(a.value.clone()).or_default() += 1;
            }
        }
        let mut index = 0;
        while index < stmts.len() {
            let Statement::Assign(assign) = &*stmts[index] else {
                index += 1;
                continue;
            };
            if assign.dtype.is_some()
                || counts.get(&assign.name).copied().unwrap_or(0) != 1
                || seen.get(&assign.value).copied().unwrap_or(0) > 1
            {
                index += 1;
                continue;
            }
            let reader = stmts
                .iter()
                .enumerate()
                .skip(index + 1)
                .find(|(_, stmt)| stmt.expressions().iter().any(|e| e.reads(assign.name)));
            let Some((at, reader)) = reader else {
                index += 1;
                continue;
            };
            // A statement between the definition and its reader that redefines a name the value reads (an
            // accumulator's update after a read of its previous step) would change what the reader computes.
            if stmts[index + 1..at]
                .iter()
                .any(|stmt| stmt.defines().iter().any(|name| assign.value.reads(*name)))
            {
                index += 1;
                continue;
            }
            let folded = Statement::substitute(
                reader,
                &HashMap::from([(assign.name, assign.value.clone())]),
            );
            let fits = folded
                .expressions()
                .iter()
                .all(|e| e.clone().bounded().is_ok())
                && folded.check().is_ok();
            if !fits {
                index += 1;
                continue;
            }
            stmts[at] = folded;
            stmts.remove(index);
            // The reader may itself be a value read once; it is looked at when the walk reaches it.
        }
        body.rebuilt(stmts)
    }
    scope(body, &read_counts(body, exposed))
}

#[cfg(test)]
mod tests {
    use super::drop_unread_values;
    use crate::identity::canonicalize_identity;
    use crate::normalize::normalize_body;
    use crate::normalize::testing::*;

    /// A value read once folds into its reader only while nothing between them redefines a name it reads: a read of
    /// an accumulator between two updates observes the first, and folding it past the second would make it observe
    /// both.
    #[test]
    fn a_read_of_an_accumulator_does_not_fold_past_the_next_update() {
        let observed = |reader_after_second_update: bool| {
            let read = "v = total * 2.0";
            let reader = "w = v + 1.0";
            let second = "total <- add(total, second)";
            let order = if reader_after_second_update {
                [read, second, reader]
            } else {
                [read, reader, second]
            };
            body(&format!(
                "init f32 total = 0\nlet first = 1.0\nlet second = 2.0\ntotal <- add(total, first)\n{}\nO[] = w",
                order.join("\n")
            ))
        };
        let kept = text(&normalize_body(&observed(true)).unwrap());
        let lines: Vec<&str> = kept.lines().collect();
        let read = lines
            .iter()
            .position(|line| line.contains("= 2 * acc0"))
            .unwrap();
        let second = lines
            .iter()
            .rposition(|line| line.contains("<- add("))
            .unwrap();
        assert!(read < second && !kept.contains("2 * acc0 + 1"), "{kept}");
        let folded = text(&normalize_body(&observed(false)).unwrap());
        assert!(folded.contains("2 * acc0 + 1"), "{folded}");
    }

    #[test]
    fn unread_values_leave_with_everything_only_they_read() {
        // A dropped value leaves its operands unread in turn: one call drops the whole chain.
        let chain = ["y = load Y[i]", "a = exp(y)", "b = exp(a)", "c = exp(b)"];
        let kept = ["x = load X[i]", "O[i] = x"];
        let looped =
            |parts: &[&str]| body(&format!("for i in 0..8:\n{}", indented(&parts.join("\n"))));
        assert_eq!(
            drop_unread_values(&looped(&[chain.as_slice(), kept.as_slice()].concat())),
            looped(&kept)
        );
    }

    /// The normal form of a loop that reads `v` from `x` and `s` from `y`, then runs `lines`.
    fn normal_of(lines: &[&str]) -> String {
        let mut spelled = vec![
            "for i in 0..8:".to_string(),
            "    v = load x[i]".to_string(),
            "    s = load y[i]".to_string(),
        ];
        spelled.extend(lines.iter().map(|line| format!("    {line}")));
        text(&normalize_body(&body(&spelled.join("\n"))).unwrap())
    }

    #[test]
    fn one_key_softmax_weight_folds_to_nothing() {
        // `v * (exp(s - s) / exp(s - s))` is `v`: the difference is zero, its exponential one, the quotient one.
        assert_eq!(
            normal_of(&[
                "z = subtract(s, s)",
                "u = exp(z)",
                "w = divide(u, u)",
                "o = multiply(v, w)",
                "O[i] = o"
            ]),
            "for a0 in 0..8:\n    in0 = load x[a0]\n    O[a0] = in0"
        );
    }

    #[test]
    fn a_constant_read_by_a_write_keeps_its_statement() {
        assert_eq!(
            normal_of(&["z = subtract(s, s)", "u = exp(z)", "O[i] = u"]),
            "let f32 v0 = 1.0\nfor a0 in 0..8:\n    O[a0] = v0"
        );
    }

    #[test]
    fn a_typed_constant_is_not_read_through() {
        // A stamped dtype is a rounding the program performed: the literal keeps it, and nothing computes past it.
        assert_eq!(
            normal_of(&[
                "z = subtract(s, s)",
                "u = exp(z)",
                "f16 h = copy(u)",
                "o = multiply(v, h)",
                "O[i] = o"
            ]),
            "let f16 v0 = 1.0\nfor a0 in 0..8:\n    in0 = load x[a0]\n    v1 = in0 * v0\n    O[a0] = v1"
        );
    }

    #[test]
    fn a_score_nothing_reads_takes_its_dot_product_with_it() {
        // A one-key softmax weight is `1` whatever the score, so the scaled score, the dot product that fed it and
        // the loop that summed it reach no output: normalization drops them all, and keeps the stores.
        let normal = normalize_body(&body(
            "
            for i in 0..8:
                for k in 0..4:
                    q = load Q[i, k]
                    kk = load K[i, k]
                    qk = multiply(q, kk)
                    acc <- add(acc, qk) over k
                let f32 scale = 0.125
                s = multiply(acc, scale)
                z = subtract(s, s)
                w = exp(z)
                v = load V[i]
                o = multiply(v, w)
                O[i] = o
            ",
        ))
        .unwrap();
        assert_eq!(
            text(&normal),
            "for a0 in 0..8:\n    in0 = load V[a0]\n    O[a0] = in0"
        );
    }

    /// A loop that reads `x` from `X`, runs `lines` and stores `o`.
    fn affine_body(lines: &[&str]) -> crate::body::Body {
        let mut spelled = vec![
            "for i in 0..8:".to_string(),
            "    x = load X[i]".to_string(),
        ];
        spelled.extend(lines.iter().map(|line| format!("    {line}")));
        spelled.push("    O[i] = o".to_string());
        body(&spelled.join("\n"))
    }

    fn affine_normal(lines: &[&str]) -> String {
        text(&normalize_body(&affine_body(lines)).unwrap())
    }

    /// The normal form of `O[i] = coefficient * X[i]`.
    fn scaled(coefficient: &str) -> String {
        format!(
            "for a0 in 0..8:\n    in0 = load X[a0]\n    v0 = {coefficient} * in0\n    O[a0] = v0"
        )
    }

    const COPY: &str = "for a0 in 0..8:\n    in0 = load X[a0]\n    O[a0] = in0";

    #[test]
    fn every_spelling_of_a_scale_is_one_body() {
        let twice = affine_body(&["o = add(x, x)"]);
        let scaled_by_two = affine_body(&["let f32 c = 2.0", "o = multiply(x, c)"]);
        let halved = affine_body(&["let f32 c = 0.5", "o = divide(x, c)"]);
        // One affine statement, and the literal the fold took is dropped.
        for spelling in [&twice, &scaled_by_two, &halved] {
            assert_eq!(text(&normalize_body(spelling).unwrap()), scaled("2"));
        }
        let key = |body: &crate::body::Body| canonicalize_identity(body, false, None).unwrap().key;
        assert_eq!(key(&twice), key(&scaled_by_two));
        assert_eq!(key(&twice), key(&halved));
        assert_ne!(
            key(&twice),
            key(&affine_body(&["let f32 c = 3.0", "o = multiply(x, c)"]))
        );
    }

    #[test]
    fn a_division_by_a_literal_is_the_product_by_its_reciprocal() {
        assert_eq!(
            affine_normal(&["let f32 c = 4.0", "o = divide(x, c)"]),
            scaled("0.25")
        );
        assert_eq!(
            affine_normal(&["let f32 c = 0.25", "o = multiply(x, c)"]),
            scaled("0.25")
        );
        assert_eq!(
            affine_normal(&["let f32 c = 1.0", "o = divide(x, c)"]),
            COPY,
            "a division by one is a copy"
        );
    }

    #[test]
    fn like_terms_sum_and_cancel() {
        assert_eq!(
            affine_normal(&[
                "let f32 c = 3.0",
                "t = multiply(x, c)",
                "s = add(t, x)",
                "o = subtract(s, x)"
            ]),
            scaled("3")
        );
        // Every term cancels: the value is the constant zero, which the store reads, and nothing loads `X`.
        assert_eq!(
            affine_normal(&["o = subtract(x, x)"]),
            "let f32 v0 = 0.0\nfor a0 in 0..8:\n    O[a0] = v0"
        );
    }

    #[test]
    fn a_select_whose_branches_bind_one_value_is_that_value() {
        assert_eq!(
            affine_normal(&["o = select:", "    x when i < i", "    x when i < i"]),
            COPY
        );
    }
}
