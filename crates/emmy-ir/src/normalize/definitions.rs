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
/// definition of its own ahead of its reader, an `Accum` sums a name or a literal, and an affine region (a sum with its
/// differences, negations and literal scales — the affine form) is one node, as a sum is one operation, so an affine
/// definition read once inside an affine reader joins the reader's region. A definition that is a bare name is an
/// alias, and its readers read the name; one that is a literal spells the literal where an expression reads it, and
/// stays while a statement reads it by name. `exposed` names are read from outside the body and keep their statements.
/// The step is a canonical form: it decides nothing, and two spellings of one program — a chain of named temps or one
/// tree — take the same one.
pub fn one_operation_per_definition(body: &Body, exposed: &HashSet<Name>) -> Body {
    let mut taken: HashSet<Name> = body.ssa_uses().clone();
    body.walk(&mut |here| taken.extend(here.stmt().defines()));
    let split = scope(body, exposed, &mut taken);
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

fn scope(body: &Body, exposed: &HashSet<Name>, taken: &mut HashSet<Name>) -> Body {
    let mut out: Vec<Arc<Statement>> = Vec::new();
    let mut aliases: HashMap<Name, Name> = HashMap::new();
    let mut literals: HashMap<Name, Expression> = HashMap::new();
    for stmt in body.stmts() {
        let mut stmt = stmt.clone();
        if !aliases.is_empty() {
            stmt = Statement::rename_with(&stmt, &|name| aliases.get(&name).copied());
        }
        if !literals.is_empty() {
            stmt = Statement::substitute(&stmt, &literals);
        }
        let nested = stmt.nested();
        if !nested.is_empty() {
            stmt = with_bodies(
                &stmt,
                nested
                    .iter()
                    .map(|child| scope(child, exposed, taken))
                    .collect(),
            );
        }
        match &*stmt {
            Statement::Assign(a) if a.dtype.is_none() && !exposed.contains(&a.name) => {
                match &*a.value {
                    Expr::Var(target) => {
                        aliases.insert(a.name, *target);
                        continue;
                    }
                    Expr::Literal(_) => {
                        literals.insert(a.name, a.value.clone());
                        out.push(stmt.clone());
                        continue;
                    }
                    _ => {}
                }
            }
            _ => {}
        }
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

/// `value` with every operation node below its root defined ahead of it, except the nodes of the root's own affine
/// region when the root is one (`in_affine`).
fn split(
    value: &Expression,
    base: &Name,
    in_affine: bool,
    out: &mut Vec<Arc<Statement>>,
    taken: &mut HashSet<Name>,
) -> Expression {
    let children: Vec<Expression> = value
        .children()
        .iter()
        .map(|child| part(child, base, in_affine, out, taken))
        .collect();
    value
        .with_children(children)
        .unwrap_or_else(|_| value.clone())
}

/// `node` with the operation nodes below it defined, itself defined too unless it is a node of the enclosing affine
/// region or no operation at all.
fn part(
    node: &Expression,
    base: &Name,
    in_affine: bool,
    out: &mut Vec<Arc<Statement>>,
    taken: &mut HashSet<Name>,
) -> Expression {
    let affine = affine_node_of(node);
    let rebuilt = split(node, base, affine, out, taken);
    if rebuilt.operation().is_none() || (in_affine && affine) {
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
    let name = (0..)
        .map(|index| Name::new(&format!("{base}__e{index}")))
        .find(|name| !taken.contains(name))
        .unwrap_or_else(|| Name::new(&format!("{base}__e")));
    taken.insert(name);
    name
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

    /// An alias names nothing of its own: its readers read the name. A literal is spelled where an expression reads
    /// it, and a statement that reads it by name keeps the definition.
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
}
