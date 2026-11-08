//! The axis steps: a free loop of extent one is inlined, a canonical extent-one reduction is its single update, and an
//! outer free-loop chain takes row-major output storage order.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use super::{assign, canonicalize_expressions, rename_ssa_sequential, substituted, with_bodies};
use crate::body::{Body, free_ssa};
use crate::expr::{Expr, Expression, affine_form};
use crate::field::AxisRef;
use crate::numbering::Key;
use crate::statement::*;
use crate::text::structural_body_text;
use crate::{Name, Result};

// --- size-one axes ----------------------------------------------------------------------------------------

/// Inline every free loop of extent one, its axis a literal zero.
pub fn drop_size_one_free_axes(body: &Body) -> Body {
    body.map(&mut |stmt| {
        if let Statement::Loop(l) = &**stmt
            && l.axis.extent.as_static() == Some(1)
            && !l.is_reduce()
        {
            return Some(
                l.body
                    .stmts()
                    .iter()
                    .map(|c| substituted(c, &l.axis.name, Expression::int(0)))
                    .collect(),
            );
        }
        Some(vec![stmt.clone()])
    })
}

/// Inline a canonical extent-one reduction as its single update. The update folds the seed a kernel starts the
/// reduction from into its value, as the loop did (a base names the step's own running value instead), and a read of
/// the accumulator ahead of the update reads the step before, which the one step has none of: the seed, bound by a
/// `Let` the reads ahead are renamed to.
pub fn drop_size_one_reduce_axes(body: &Body) -> Body {
    body.map(&mut |stmt| {
        let Statement::Loop(l) = &**stmt else {
            return Some(vec![stmt.clone()]);
        };
        if !(l.is_reduce() && l.axis.extent.as_static() == Some(1)) {
            return Some(vec![stmt.clone()]);
        }
        let accums: Vec<&Accum> = l
            .body
            .stmts()
            .iter()
            .filter_map(|m| match &**m {
                Statement::Accum(a) => Some(a),
                _ => None,
            })
            .collect();
        let names: HashSet<&Name> = accums.iter().map(|a| &a.name).collect();
        if accums.is_empty()
            || names.len() != accums.len()
            || l.body
                .stmts()
                .iter()
                .any(|m| !(m.is_pure() || matches!(**m, Statement::Accum(_))))
        {
            return Some(vec![stmt.clone()]);
        }
        let mut taken: HashSet<Name> = l.body.ssa_uses().clone();
        l.body.walk(&mut |here| taken.extend(here.stmt().defines()));
        let mut seeds: HashMap<Name, Name> = HashMap::new();
        let mut out: Vec<Arc<Statement>> = Vec::new();
        for a in &accums {
            let read = l.body.stmts().iter().any(|m| {
                !matches!(&**m, Statement::Accum(x) if x.name == a.name)
                    && free_ssa(m).contains(&a.name)
            });
            if !(read || a.base.is_none()) {
                continue;
            }
            let Some(seed) = a.op.seed() else {
                return Some(vec![stmt.clone()]);
            };
            let mut index = 0;
            let bound = loop {
                let candidate = Name::new(&format!("{}__seed{index}", a.name));
                if taken.insert(candidate) {
                    break candidate;
                }
                index += 1;
            };
            seeds.insert(a.name, bound);
            out.push(Arc::new(Statement::Let(Let {
                name: bound,
                value: Expression::float(seed),
                dtype: a.dtype,
            })));
        }
        for member in l.body.stmts() {
            let member = substituted(member, &l.axis.name, Expression::int(0));
            let seed = match &*member {
                Statement::Accum(a) => seeds.remove(&a.name),
                _ => None,
            };
            let member = if seeds.is_empty() {
                member
            } else {
                Statement::rename_with(&member, &|name| seeds.get(&name).copied())
            };
            match &*member {
                // The one update folds the value it started from — its base as renamed (the step's own running value,
                // or the seed of a sibling accumulator updated later in the step) or its own bound seed: a sum's zero
                // and a product's one fold away, a maximum keeps the one operation the loop ran.
                Statement::Accum(a) => {
                    let Some(op) = a.op.known() else {
                        return Some(vec![stmt.clone()]);
                    };
                    let start = match (&a.base, seed) {
                        (Some(base), _) => Expression::var(base),
                        (None, Some(seed)) => Expression::var(&seed),
                        // The first pass bound a seed for every accumulator without a base.
                        (None, None) => return Some(vec![stmt.clone()]),
                    };
                    let combined = Expression::apply(op, vec![start, a.value.clone()]);
                    out.push(assign(&a.name, combined, a.dtype))
                }
                _ => out.push(member),
            }
        }
        Some(out)
    })
}

// --- the free-axis order ----------------------------------------------------------------------------------

/// The innermost row-major output-coordinate depth carrying one free axis.
pub(crate) fn output_storage_depth(stmts: &Body, axis: &Name) -> Option<usize> {
    let mut depths: Vec<usize> = Vec::new();
    for write in stmts.writes() {
        let Statement::Write(w) = &**write.stmt() else {
            continue;
        };
        let mut positions = Vec::new();
        for (position, expression) in w.index.iter().enumerate() {
            if !expression.reads(*axis) {
                continue;
            }
            if let Some((_, coeffs)) = affine_form(expression, &|name| name == **axis)
                && coeffs.get(axis).copied().unwrap_or(0) != 1
            {
                return None;
            }
            positions.push(position);
        }
        if let Some(last) = positions.last() {
            depths.push(w.index.len() - last - 1);
        }
    }
    match depths.first() {
        Some(first) if depths.iter().all(|d| d == first) => Some(*first),
        _ => None,
    }
}

pub(crate) fn loop_over(axis: AxisRef, body: Body, unroll: bool, seed: bool) -> Body {
    Body::new(vec![Arc::new(Statement::Loop(Loop {
        axis,
        body,
        unroll,
        seed,
    }))])
}

/// Permute distinct items once modulo transpositions proven to preserve the whole form.
pub(crate) fn orders_modulo_transpositions<T: Clone>(
    items: &[T],
    interchangeable: &dyn Fn(&T, &T) -> bool,
) -> Vec<Vec<T>> {
    let mut classes: Vec<Vec<T>> = Vec::new();
    for item in items {
        match classes
            .iter_mut()
            .find(|group| interchangeable(item, &group[0]))
        {
            Some(group) => group.push(item.clone()),
            None => classes.push(vec![item.clone()]),
        }
    }
    let mut out = Vec::new();
    fn walk<T: Clone>(
        classes: &[Vec<T>],
        remaining: &mut Vec<usize>,
        positions: &mut Vec<usize>,
        out: &mut Vec<Vec<T>>,
    ) {
        if remaining.iter().all(|count| *count == 0) {
            let mut offsets = vec![0; classes.len()];
            let mut order = Vec::new();
            for &class in positions.iter() {
                order.push(classes[class][offsets[class]].clone());
                offsets[class] += 1;
            }
            out.push(order);
            return;
        }
        for class in 0..classes.len() {
            if remaining[class] == 0 {
                continue;
            }
            remaining[class] -= 1;
            positions.push(class);
            walk(classes, remaining, positions, out);
            positions.pop();
            remaining[class] += 1;
        }
    }
    let mut remaining: Vec<usize> = classes.iter().map(|group| group.len()).collect();
    walk(&classes, &mut remaining, &mut Vec::new(), &mut out);
    out
}

pub(crate) fn rename_axes(body: &Body, mapping: &HashMap<Name, Name>) -> Body {
    let sigma: HashMap<Name, Expression> = mapping
        .iter()
        .map(|(name, replacement)| (*name, Expression::new(Expr::Var(*replacement))))
        .collect();
    body.map_each(&mut |stmt| Statement::substitute(stmt, &sigma))
}

pub(crate) fn text_key(body: &Body) -> Key {
    Key::text(structural_body_text(body))
}

/// Sort an outer free-loop chain by row-major output storage order; where the writes do not order it, by the least
/// renamed structural form.
pub fn canonicalize_free_axis_order(body: &Body) -> Result<Body> {
    let mut chain: Vec<Loop> = Vec::new();
    let mut current = body.clone();
    while current.len() == 1
        && let Statement::Loop(l) = &*current.stmts()[0]
    {
        if l.is_reduce() {
            break;
        }
        chain.push(l.clone());
        current = l.body.clone();
    }
    let terminal: Vec<Arc<Statement>> = current
        .stmts()
        .iter()
        .map(|stmt| {
            let nested = stmt.nested();
            if nested.is_empty() {
                return Ok(stmt.clone());
            }
            Ok(with_bodies(
                stmt,
                nested
                    .iter()
                    .map(canonicalize_free_axis_order)
                    .collect::<Result<Vec<_>>>()?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let terminal = current.rebuilt(terminal);
    let depths: Vec<Option<usize>> = chain
        .iter()
        .map(|l| output_storage_depth(&terminal, &l.axis.name))
        .collect();
    let distinct_depths: HashSet<Option<usize>> = depths.iter().copied().collect();
    let sorted: Vec<Loop> = if depths.iter().all(|d| d.is_some())
        && distinct_depths.len() == depths.len()
    {
        let mut pairs: Vec<(usize, Loop)> = depths
            .iter()
            .flatten()
            .copied()
            .zip(chain.iter().cloned())
            .collect();
        pairs.sort_by_key(|(depth, _)| std::cmp::Reverse(*depth));
        pairs.into_iter().map(|(_, l)| l).collect()
    } else {
        let axis_metadata = |l: &Loop| -> Result<Key> {
            let mut mapping: HashMap<Name, Name> =
                HashMap::from([(l.axis.name, Name::from("__axis__"))]);
            for (depth, source) in l.axis.sources().iter().enumerate() {
                mapping
                    .entry(source.name)
                    .or_insert_with(|| format!("__parent{depth}__").into());
            }
            let renamed = Statement::rename(&Arc::new(Statement::Loop(l.clone())), &mapping);
            let Statement::Loop(renamed) = &*renamed else {
                return Ok(Key::None);
            };
            Ok(Key::tuple([
                Key::text(crate::axis::spell_axis(&renamed.axis, None)?),
                Key::Bool(renamed.unroll),
                Key::Bool(renamed.seed),
            ]))
        };
        let metadata: HashMap<Name, Key> = chain
            .iter()
            .map(|l| Ok((l.axis.name, axis_metadata(l)?)))
            .collect::<Result<_>>()?;
        let mut source_counts: HashMap<Name, usize> = HashMap::new();
        for l in &chain {
            if let Some(source) = l.axis.source_axis() {
                *source_counts.entry(source.name).or_default() += 1;
            }
        }
        let mut role_of: HashMap<Name, Key> = HashMap::new();
        for (focus, depth) in chain.iter().zip(&depths) {
            let mapping: HashMap<Name, Name> = chain
                .iter()
                .map(|l| {
                    (
                        l.axis.name,
                        Name::from(if l.axis.name == focus.axis.name {
                            "__self__"
                        } else {
                            "__other__"
                        }),
                    )
                })
                .collect();
            // The current nest orders affine sums too; compare roles independently of that order to avoid cycles.
            let focused = canonicalize_expressions(
                &rename_axes(&terminal, &mapping),
                &[Name::from("__self__"), Name::from("__other__")],
            )?;
            let source_arity = focus.axis.source_axis().map_or(1, |source| {
                source_counts.get(&source.name).copied().unwrap_or(1)
            });
            // A coordinate carried by a consistent output position has a fixed row-major role; an unresolved one
            // stays outside the known output suffix, then its structural role breaks ties.
            let storage_order = match depth {
                None => Key::tuple([Key::Int(0), Key::Int(0)]),
                Some(d) => Key::tuple([Key::Int(1), Key::Int(-(*d as i64))]),
            };
            let role = Key::tuple([
                storage_order,
                Key::tuple([
                    metadata[&focus.axis.name].clone(),
                    Key::Int(source_arity as i64),
                    text_key(&focused),
                ]),
            ]);
            role_of.insert(focus.axis.name, role);
        }
        let mut groups: BTreeMap<Key, Vec<Loop>> = BTreeMap::new();
        for l in &chain {
            groups
                // Every loop of the chain took its role in the loop above.
                .entry(role_of.get(&l.axis.name).cloned().unwrap_or(Key::None))
                .or_default()
                .push(l.clone());
        }
        let interchangeable = |left: &Loop, right: &Loop| -> bool {
            if metadata[&left.axis.name] != metadata[&right.axis.name] {
                return false;
            }
            if let (Some(ls), Some(rs)) = (left.axis.source_axis(), right.axis.source_axis())
                && ls.name != rs.name
                && (source_counts.get(&ls.name).copied() != Some(1)
                    || source_counts.get(&rs.name).copied() != Some(1))
            {
                return false;
            }
            let swapped = rename_axes(
                &terminal,
                &HashMap::from([
                    (left.axis.name, right.axis.name),
                    (right.axis.name, left.axis.name),
                ]),
            );
            swapped == terminal
        };
        let candidate = |order: &[Loop]| -> Key {
            let mut result = terminal.clone();
            for l in order.iter().rev() {
                result = loop_over(l.axis.clone(), result, l.unroll, l.seed);
            }
            text_key(&rename_ssa_sequential(&result))
        };
        let per_group: Vec<Vec<Vec<Loop>>> = groups
            .values()
            .map(|group| orders_modulo_transpositions(group, &interchangeable))
            .collect();
        let mut best: Option<(Key, Vec<Loop>)> = None;
        fn product(groups: &[Vec<Vec<Loop>>], chosen: &mut Vec<Loop>, f: &mut dyn FnMut(&[Loop])) {
            if groups.is_empty() {
                f(chosen);
                return;
            }
            for choice in &groups[0] {
                let len = chosen.len();
                chosen.extend(choice.iter().cloned());
                product(&groups[1..], chosen, f);
                chosen.truncate(len);
            }
        }
        product(&per_group, &mut Vec::new(), &mut |order| {
            let key = candidate(order);
            if best.as_ref().is_none_or(|(best_key, _)| key < *best_key) {
                best = Some((key, order.to_vec()));
            }
        });
        best.map_or_else(Vec::new, |(_, order)| order)
    };
    if terminal.ptr_eq(&current) && sorted.iter().zip(&chain).all(|(a, b)| a.same(b)) {
        return Ok(body.clone());
    }
    let mut result = terminal;
    for l in sorted.iter().rev() {
        result = loop_over(l.axis.clone(), result, l.unroll, l.seed);
    }
    Ok(result)
}
