//! One operation per definition: the granularity every definition of the normal form has.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::fold::read_counts;
use super::{assign, with_bodies};
use crate::Name;
use crate::body::Body;
use crate::expr::{Expr, ExprClass, Expression};
use crate::normalize::canonical::affine_node;
use crate::statement::*;

/// Every definition one operation over names and literals: a nested operation node of an `Assign`'s value is a
/// definition of its own ahead of its reader, and an `Accum` sums a name or a literal. An affine region (a sum with its
/// differences, negations and literal scales — the affine form) is one node, as a sum is one operation, and an affine
/// definition read exactly once, inside an affine reader, joins the reader's region. `exposed` names are read from
/// outside the body and keep their statements. The step is a canonical form: it decides nothing, and a chain of named
/// temps and one tree are one body up to the single-reader guard — a value two statements read stays a definition,
/// whatever its readers are.
pub fn one_operation_per_definition(body: &Body, exposed: &HashSet<Name>) -> Body {
    let mut taken: HashSet<Name> = body.ssa_uses().clone();
    body.walk(&mut |here| taken.extend(here.stmt().defines()));
    let split = scope(body, &mut taken);
    let counts = read_counts(&split, exposed);
    merge_affine(&split, &counts)
}

/// Every affine definition read exactly once, by an expression of a later `Assign` of the same scope at a place every
/// node above which is affine, joined into that reader's region; nothing between the two may redefine a name the
/// definition reads (a read of an accumulator between two updates observes the first).
fn merge_affine(body: &Body, counts: &HashMap<Name, usize>) -> Body {
    let mut stmts: Vec<Arc<Statement>> = body
        .stmts()
        .iter()
        .map(|stmt| {
            let nested = stmt.nested();
            if nested.is_empty() {
                stmt.clone()
            } else {
                with_bodies(
                    stmt,
                    nested
                        .iter()
                        .map(|child| merge_affine(child, counts))
                        .collect(),
                )
            }
        })
        .collect();
    let mut index = 0;
    while index < stmts.len() {
        let Statement::Assign(definition) = &*stmts[index] else {
            index += 1;
            continue;
        };
        let joins = definition.dtype.is_none()
            && counts.get(&definition.name).copied() == Some(1)
            && affine_node_of(&definition.value);
        let reader = stmts
            .iter()
            .enumerate()
            .skip(index + 1)
            .find(|(_, stmt)| stmt.expressions().iter().any(|e| e.reads(definition.name)));
        let Some((at, reader)) = reader.filter(|_| joins) else {
            index += 1;
            continue;
        };
        let Statement::Assign(target) = &**reader else {
            index += 1;
            continue;
        };
        let redefined = stmts[index + 1..at].iter().any(|stmt| {
            stmt.defines()
                .iter()
                .any(|name| definition.value.reads(*name))
        });
        if redefined || !affine_path(&target.value, definition.name) {
            index += 1;
            continue;
        }
        let joined = Statement::substitute(
            reader,
            &HashMap::from([(definition.name, definition.value.clone())]),
        );
        stmts[at] = joined;
        stmts.remove(index);
    }
    body.rebuilt(stmts)
}

/// Whether every node of `value` above an occurrence of `name` is an affine node, the root included: the occurrence
/// sits inside the value's affine region.
fn affine_path(value: &Expression, name: Name) -> bool {
    if matches!(&**value, Expr::Var(n) if *n == name) {
        return true;
    }
    if !value.reads(name) {
        return false;
    }
    affine_node_of(value)
        && value
            .children()
            .iter()
            .filter(|child| child.reads(name))
            .all(|child| affine_path(child, name))
}

fn scope(body: &Body, taken: &mut HashSet<Name>) -> Body {
    let mut out: Vec<Arc<Statement>> = Vec::new();
    for stmt in body.stmts() {
        let nested = stmt.nested();
        let stmt = if nested.is_empty() {
            stmt.clone()
        } else {
            with_bodies(
                stmt,
                nested.iter().map(|child| scope(child, taken)).collect(),
            )
        };
        match &*stmt {
            Statement::Assign(a) => {
                let value = split(&a.value, &a.name, affine_node_of(&a.value), &mut out, taken);
                out.push(if value == a.value {
                    stmt.clone()
                } else {
                    assign(&a.name, value, a.dtype)
                });
            }
            Statement::Accum(acc) => {
                // An accumulation sums a name or a literal: its value's operation is a definition ahead of it.
                let value = part(&acc.value, &acc.name, false, &mut out, taken);
                out.push(if value == acc.value {
                    stmt.clone()
                } else {
                    Arc::new(Statement::Accum(Accum {
                        value,
                        ..acc.clone()
                    }))
                });
            }
            _ => out.push(stmt.clone()),
        }
    }
    body.rebuilt(out)
}

/// `value` with every node below its root defined ahead of it, except the nodes of the root's own affine region when
/// the root is one (`in_affine`).
fn split(
    value: &Expression,
    base: &Name,
    in_affine: bool,
    out: &mut Vec<Arc<Statement>>,
    taken: &mut HashSet<Name>,
) -> Expression {
    value.map_children(&mut |child| part(child, base, in_affine, out, taken))
}

/// `node` with the nodes below it defined, itself defined too unless it is a name, a literal or a node of the
/// enclosing affine region.
fn part(
    node: &Expression,
    base: &Name,
    in_affine: bool,
    out: &mut Vec<Arc<Statement>>,
    taken: &mut HashSet<Name>,
) -> Expression {
    let affine = affine_node_of(node);
    let rebuilt = split(node, base, affine, out, taken);
    if matches!(&*rebuilt, Expr::Var(_) | Expr::Literal(_)) || (in_affine && affine) {
        return rebuilt;
    }
    let name = fresh(base, taken);
    out.push(assign(&name, rebuilt, None));
    Expression::var(&name)
}

fn affine_node_of(value: &Expression) -> bool {
    value
        .operation()
        .is_some_and(|(op, args)| affine_node(op, args, ExprClass::Value))
}

fn fresh(base: &Name, taken: &mut HashSet<Name>) -> Name {
    let mut index = 0;
    loop {
        let name = Name::new(&format!("{base}__e{index}"));
        if taken.insert(name) {
            return name;
        }
        index += 1;
    }
}

#[cfg(test)]
mod tests {
    use crate::normalize::normalize_body;
    use crate::normalize::testing::*;

    /// A chain of named temps and one tree are one normal form: every operation a definition, an affine region one.
    #[test]
    fn a_tree_and_its_chain_spelling_are_one_body() {
        let chain = body(
            "for i in 0..8:
    x = load X[i]
    s = load S[i]
    e = exp(x)
    t = add(e, s)
    u = multiply(t, 2.0)
    v = subtract(u, x)
    w = rsqrt(v)
    O[i] = w",
        );
        let tree = body(
            "for i in 0..8:
    x = load X[i]
    s = load S[i]
    w = rsqrt(2.0 * (exp(x) + s) - x)
    O[i] = w",
        );
        let normal = text(&normalize_body(&tree).unwrap());
        assert_eq!(normal, text(&normalize_body(&chain).unwrap()));
        assert_eq!(
            normal,
            "for a0 in 0..8:
    in0 = load S[a0]
    in1 = load X[a0]
    v0 = exp(in1)
    v1 = 2 * in0 - in1 + 2 * v0
    v2 = rsqrt(v1)
    O[a0] = v2"
        );
    }

    /// An accumulation sums a name; what it sums is defined ahead of it, one operation at a time.
    #[test]
    fn an_accumulation_sums_one_operation() {
        let normal = text(
            &normalize_body(&body(
                "for i in 0..8:
    for k in 0..4:
        x = load X[i, k]
        s = load S[k]
        acc <- add(acc, exp(x) * s) over k
    O[i] = acc",
            ))
            .unwrap(),
        );
        assert!(
            normal.contains("v0 = exp(in0)\n") && normal.contains("acc0 <- add(acc0, v1) over a1"),
            "{normal}"
        );
        assert!(
            normal.contains("v1 = in1 * v0\n") || normal.contains("v1 = v0 * in1\n"),
            "{normal}"
        );
    }

    /// A nested select or cast is a definition of its own, like any node that is no name, literal or affine node;
    /// so is the comparison a select reads.
    #[test]
    fn a_nested_select_or_cast_is_a_definition_of_its_own() {
        let loads = "for i in 0..8:\n    x = load X[i]\n    s = load S[i]\n";
        let selected = text(
            &normalize_body(&body(&format!(
                "{loads}    y = exp(x < s ? x * s : s)\n    O[i] = y"
            )))
            .unwrap(),
        );
        let chain = text(
            &normalize_body(&body(&format!(
                "{loads}    p = x * s\n    t = x < s ? p : s\n    y = exp(t)\n    O[i] = y"
            )))
            .unwrap(),
        );
        assert_eq!(selected, chain);
        assert!(
            selected.contains("v0 = in0 * in1\n")
                && selected.contains("v1 = in1 < in0\n")
                && selected.contains("v2 = v1 ? v0 : in0\n")
                && selected.contains("v3 = exp(v2)\n"),
            "{selected}"
        );
        let cast = text(
            &normalize_body(&body(&format!(
                "{loads}    y = exp((f16)(x * s))\n    O[i] = y"
            )))
            .unwrap(),
        );
        assert_eq!(
            cast,
            text(
                &normalize_body(&body(&format!(
                    "{loads}    p = x * s\n    c = (f16)(p)\n    y = exp(c)\n    O[i] = y"
                )))
                .unwrap()
            )
        );
        assert!(
            cast.contains("v1 = (f16)v0\n") && cast.contains("v2 = exp(v1)\n"),
            "{cast}"
        );
    }

    /// An accumulation sums a name: a select it summed is a definition ahead of it.
    #[test]
    fn an_accumulation_sums_a_select_by_name() {
        let loads = "for i in 0..8:\n    for k in 0..4:\n        x = load X[i, k]\n";
        let selected = text(
            &normalize_body(&body(&format!(
                "{loads}        acc <- add(acc, k < i ? x : 0.0) over k\n    O[i] = acc"
            )))
            .unwrap(),
        );
        let chain = text(
            &normalize_body(&body(&format!(
                "{loads}        m = k < i ? x : 0.0\n        acc <- add(acc, m) over k\n    O[i] = acc"
            )))
            .unwrap(),
        );
        assert_eq!(selected, chain);
        assert!(
            selected.contains("v1 = v0 ? in0 : 0.0\n")
                && selected.contains("acc0 <- add(acc0, v1) over a1"),
            "{selected}"
        );
    }

    /// The single-reader guard: an affine value two statements read stays a definition, and so does one a
    /// non-affine node reads, however its reader spells the same sum again.
    #[test]
    fn a_shared_affine_value_stays_a_definition() {
        let loads = "for i in 0..8:\n    x = load X[i]\n    s = load S[i]\n    y = load Y[i]\n    z = load Z[i]\n";
        let shared = text(
            &normalize_body(&body(&format!(
                "{loads}    t = add(x, s)\n    u = add(t, y)\n    v = subtract(t, z)\n    O[i] = u\n    P[i] = v"
            )))
            .unwrap(),
        );
        assert!(
            shared.contains("v0 = in0 + in1\n")
                && shared.contains("v1 = in2 + v0\n")
                && shared.contains("v2 = -in3 + v0\n"),
            "{shared}"
        );
        let twice = text(
            &normalize_body(&body(
                "for i in 0..8:
    x = load X[i]
    s = load S[i]
    w = (x + s) + exp(x + s)
    O[i] = w",
            ))
            .unwrap(),
        );
        assert_eq!(
            twice,
            "for a0 in 0..8:
    in0 = load S[a0]
    in1 = load X[a0]
    v0 = in0 + in1
    v1 = exp(v0)
    v2 = in0 + in1 + v1
    O[a0] = v2"
        );
    }

    /// An alias names nothing of its own and a literal is a literal: placement and the literal inlining read both
    /// through, so the split meets neither.
    #[test]
    fn an_alias_or_a_literal_definition_is_read_through() {
        let normal = text(
            &normalize_body(&body(
                "for i in 0..8:
    x = load X[i]
    y = copy(x)
    c = 2.0
    z = multiply(y, c)
    O[i] = z
    P[i] = c",
            ))
            .unwrap(),
        );
        assert_eq!(
            normal,
            "let f32 v0 = 2.0\nfor a0 in 0..8:\n    in0 = load X[a0]\n    v1 = 2 * in0\n    O[a0] = v1\n    P[a0] = v0"
        );
    }

    /// Like terms sum and cancel inside one affine region, whichever statements spelled them.
    #[test]
    fn like_terms_sum_and_cancel() {
        let affine = |lines: &[&str]| {
            let mut given = String::from("for i in 0..8:\n    x = load X[i]\n");
            for line in lines {
                given.push_str(&format!("    {line}\n"));
            }
            given.push_str("    O[i] = o");
            text(&normalize_body(&body(&given)).unwrap())
        };
        assert_eq!(
            affine(&[
                "let f32 c = 3.0",
                "t = multiply(x, c)",
                "s = add(t, x)",
                "o = subtract(s, x)"
            ]),
            "for a0 in 0..8:\n    in0 = load X[a0]\n    v0 = 3 * in0\n    O[a0] = v0"
        );
        // Every term cancels: the value is the constant zero, which the store reads, and nothing loads `X`.
        assert_eq!(
            affine(&["o = subtract(x, x)"]),
            "let f32 v0 = 0.0\nfor a0 in 0..8:\n    O[a0] = v0"
        );
    }
}
