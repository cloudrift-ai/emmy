//! The range steps: a reduce loop whose every accumulator is masked to one part of its axis walks that part, and a
//! range takes its canonical spelling.
//!
//! A reduction over a triangle is written `select(v when a4 < a0 + 2, 0 when true)` under a loop over the whole axis:
//! every trip is taken and the masked ones fold the neutral element. [`range_masked_reductions`] moves the bound into
//! the loop's range. Under the range the simplifier decides the comparison (`expr/implied.rs`), the select folds to
//! its value, and a clamp of an index the range already bounds goes with it. Each rule removes work and adds none.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::inner_ctx;
use crate::axis::{Axis, Dim, Range};
use crate::body::{Body, free_ssa};
use crate::elementwise::Operation;
use crate::expr::{Expr, Expression, Linear, SimplifyCtx};
use crate::statement::*;
use crate::{Name, Result};

/// The bounds a mask states of one axis: every lower bound and every exclusive upper bound, each sorted by its text.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
struct Bounds {
    lower: Vec<Expression>,
    upper: Vec<Expression>,
}

impl Bounds {
    fn is_empty(&self) -> bool {
        self.lower.is_empty() && self.upper.is_empty()
    }
}

/// `form` without `axis`, as an index expression.
fn rest_of(form: &Linear, axis: Name, sign: i64, shift: i64) -> Option<Expression> {
    let mut out = Expression::int(form.constant.checked_mul(sign)?.checked_add(shift)?);
    for (name, coefficient) in &form.coefficients {
        if *name == axis {
            continue;
        }
        let term = Expression::binary(
            Operation::Multiply,
            Expression::int(coefficient.checked_mul(sign)?),
            Expression::var(name),
        );
        out = Expression::binary(Operation::Add, out, term);
    }
    out.simplify(&SimplifyCtx::empty()).ok()
}

/// The bound one comparison states of `axis`, added to `bounds`: `false` when it is no bound of the axis by the
/// names `outer` allows (the coordinates around the loop and the symbolic dims).
fn bound_of(
    op: Operation,
    left: &Expression,
    right: &Expression,
    axis: Name,
    outer: &dyn Fn(Name) -> bool,
    bounds: &mut Bounds,
) -> bool {
    let Some(difference) = Linear::at_least(left, right) else {
        return false;
    };
    let coefficient = difference.coefficients.get(&axis).copied().unwrap_or(0);
    if coefficient.abs() != 1
        || !difference
            .coefficients
            .keys()
            .all(|name| *name == axis || outer(*name))
    {
        return false;
    }
    // With `left - right = c·axis + rest`: the least value the axis may take, and one past the most.
    let from = |shift: i64| rest_of(&difference, axis, -coefficient, shift);
    let (lower, upper) = match (op, coefficient) {
        (Operation::Less, 1) => (None, from(0)),
        (Operation::Less, _) => (from(1), None),
        (Operation::LessEqual, 1) => (None, from(1)),
        (Operation::LessEqual, _) => (from(0), None),
        (Operation::Greater, 1) => (from(1), None),
        (Operation::Greater, _) => (None, from(0)),
        (Operation::GreaterEqual, 1) => (from(0), None),
        (Operation::GreaterEqual, _) => (None, from(1)),
        (Operation::Equal, _) => (from(0), from(1)),
        _ => return false,
    };
    if lower.is_none() && upper.is_none() {
        return false;
    }
    bounds.lower.extend(lower);
    bounds.upper.extend(upper);
    true
}

/// The bounds a condition states of `axis`: those of each comparison of a conjunction that bounds it. A conjunct
/// that does not bound the axis stays the select's.
fn bounds_of(cond: &Expression, axis: Name, outer: &dyn Fn(Name) -> bool, bounds: &mut Bounds) {
    match cond.as_binary() {
        Some((Operation::LogicalAnd, left, right)) => {
            bounds_of(left, axis, outer, bounds);
            bounds_of(right, axis, outer, bounds);
        }
        Some((op, left, right)) => {
            bound_of(op, left, right, axis, outer, bounds);
        }
        None => {}
    }
}

/// The comparison that holds exactly where `op` does not.
fn negation(op: Operation) -> Option<Operation> {
    Some(match op {
        Operation::Less => Operation::GreaterEqual,
        Operation::LessEqual => Operation::Greater,
        Operation::Greater => Operation::LessEqual,
        Operation::GreaterEqual => Operation::Less,
        _ => return None,
    })
}

/// The part of its axis the loop's accumulators are all masked to, or `None`: the loop is a reduction with no other
/// effect, no statement of it reads an accumulator, and every accumulator folds a select between a value and its
/// operation's neutral element whose condition bounds the axis alike.
fn masked_bounds(
    l: &Loop,
    literals: &HashMap<Name, f64>,
    outer: &dyn Fn(Name) -> bool,
) -> Option<Bounds> {
    if !l.carries().is_empty() {
        return None;
    }
    let stmts = l.body.stmts();
    let mut accums: Vec<&Accum> = Vec::new();
    for stmt in stmts {
        match &**stmt {
            Statement::Accum(a) if a.base.is_none() => accums.push(a),
            Statement::Accum(_) | Statement::Carry(_) => return None,
            other if other.has_side_effects() => return None,
            _ => {}
        }
    }
    if accums.is_empty() {
        return None;
    }
    let selects: HashMap<Name, &Select> = stmts
        .iter()
        .filter_map(|stmt| match &**stmt {
            Statement::Select(s) => Some((s.name, s)),
            _ => None,
        })
        .collect();
    let axis = l.axis.name;
    let mut found: Option<Bounds> = None;
    for accum in &accums {
        if stmts.iter().any(|stmt| {
            !matches!(&**stmt, Statement::Accum(a) if a.name == accum.name)
                && free_ssa(stmt).contains(&accum.name)
        }) {
            return None;
        }
        // A kernel seeds the accumulator with the operation's seed (a maximum's is the bounded stand-in of its
        // neutral element), so folding either leaves the accumulator as the trip found it.
        let op = accum.op.known()?;
        let neutral = op.info().neutral?;
        let seed = op.seed();
        let select = selects.get(accum.value.as_var()?)?;
        let [first, last] = &select.branches[..] else {
            return None;
        };
        if !last.select.is_truthy() {
            return None;
        }
        let is_neutral = |name: &Name| {
            literals
                .get(name)
                .is_some_and(|value| *value == neutral || Some(*value) == seed)
        };
        let mut bounds = Bounds::default();
        if is_neutral(&last.value) && !is_neutral(&first.value) {
            bounds_of(&first.select, axis, outer, &mut bounds);
        } else if is_neutral(&first.value)
            && let Some((op, left, right)) = first.select.as_binary()
            && let Some(kept) = negation(op)
        {
            // The neutral element where the comparison holds: the value where its negation does.
            bound_of(kept, left, right, axis, outer, &mut bounds);
        }
        if bounds.is_empty() {
            return None;
        }
        bounds.lower.sort_by_key(|e| e.text());
        bounds.lower.dedup();
        bounds.upper.sort_by_key(|e| e.text());
        bounds.upper.dedup();
        match &found {
            None => found = Some(bounds),
            Some(shared) if *shared == bounds => {}
            Some(_) => return None,
        }
    }
    found
}

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

/// Every reduce loop whose accumulators are all masked to one part of its axis walks that part, and every range
/// stands in its canonical spelling. The masks stay where they are: under the range their comparisons are decided,
/// and the simplification that follows folds them away.
pub fn range_masked_reductions(body: &Body) -> Result<Body> {
    let values: HashSet<Name> = body.definitions().keys().copied().collect();
    scope(body, &SimplifyCtx::empty(), &HashMap::new(), &values, &[])
}

fn scope(
    body: &Body,
    ctx: &SimplifyCtx,
    literals: &HashMap<Name, f64>,
    values: &HashSet<Name>,
    around: &[Name],
) -> Result<Body> {
    let mut literals = literals.clone();
    let mut out: Vec<Arc<Statement>> = Vec::with_capacity(body.len());
    for stmt in body.stmts() {
        if let Statement::Let(binding) = &**stmt
            && let Expr::Literal(literal) = &*binding.value
        {
            literals.insert(binding.name, literal.value.as_f64());
        }
        let mut stmt = stmt.clone();
        if let Statement::Loop(l) = &*stmt {
            // A bound reads the coordinates around the loop and the symbolic dims: no value the body computes.
            let outer = |name: Name| around.contains(&name) || !values.contains(&name);
            let mut range = l.walk();
            let masked = masked_bounds(l, &literals, &outer);
            // A loop that states no range and gains none is left as it is: its extent is its spelling.
            if l.range.is_some() || masked.is_some() {
                if let Some(bounds) = masked {
                    range.start = bounds.lower.into_iter().fold(range.start, |start, bound| {
                        Expression::apply(Operation::Maximum, vec![start, bound])
                    });
                    range.end = bounds.upper.into_iter().fold(range.end, |end, bound| {
                        Expression::apply(Operation::Minimum, vec![end, bound])
                    });
                }
                let ranged = over(l, range, ctx)?;
                if ranged != *l {
                    stmt = Arc::new(Statement::Loop(ranged));
                }
            }
        }
        let nested = stmt.nested();
        if !nested.is_empty() {
            let inner = inner_ctx(ctx, &stmt);
            let mut deeper = around.to_vec();
            deeper.extend(stmt.binds_axes());
            let bodies: Result<Vec<Body>> = nested
                .iter()
                .map(|child| scope(child, &inner, &literals, values, &deeper))
                .collect();
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
    fn a_triangle_masked_to_the_neutral_element_becomes_the_range() {
        let given = "
      for i in 0..8:
        for k in 0..8:
          x = load X[i, k]
          let zero = 0.0
          m = select:
            x when k < i + 1
            zero when true
          s <- add(s, m) over k
        O[i] = s
            ";
        assert_eq!(
            normal(given),
            "for a0 in 0..8:\n  for a1 in 0..a0 + 1 of 8:\n    in0 = load X[a0, a1]\n    acc0 <- add(acc0, in0) over a1\n  O[a0] = acc0"
        );
    }

    #[test]
    fn the_range_folds_the_clamp_of_an_index_it_bounds() {
        let given = "
      for i in 0..8:
        for k in 0..8:
          x = load X[i, minimum(k, i)]
          let zero = 0.0
          m = select:
            x when k <= i
            zero when true
          s <- add(s, m) over k
        O[i] = s
            ";
        assert!(
            normal(given).contains("in0 = load X[a0, a1]"),
            "{}",
            normal(given)
        );
    }

    #[test]
    fn a_static_bound_narrows_the_extent_too() {
        let given = "
      for i in 0..8:
        for k in 0..8:
          x = load X[i, k]
          let zero = 0.0
          m = select:
            x when 2 <= k && k < 6
            zero when true
          s <- add(s, m) over k
        O[i] = s
            ";
        assert!(
            normal(given).contains("for a1 in 2..6 of 6:"),
            "{}",
            normal(given)
        );
    }

    #[test]
    fn a_mask_that_is_not_the_neutral_element_stays_a_select() {
        let masked = |fill: &str, op: &str| {
            normal(&format!(
                "
      for i in 0..8:
        for k in 0..8:
          x = load X[i, k]
          let fill = {fill}
          m = select:
            x when k < i + 1
            fill when true
          s <- {op}(s, m) over k
        O[i] = s
            "
            ))
        };
        assert!(
            masked("1.0", "add").contains("for a1 in 0..8:"),
            "a sum's neutral element is zero"
        );
        assert!(
            masked("0.0", "maximum").contains("for a1 in 0..8:"),
            "a maximum's is not"
        );
        assert!(
            masked("1.0", "multiply").contains(" of 8:"),
            "a product's is one"
        );
    }

    #[test]
    fn a_condition_on_data_or_on_another_axis_stays_a_select() {
        let given = "
      for i in 0..8:
        for j in 0..8:
          for k in 0..8:
            x = load X[i, k]
            let zero = 0.0
            m = select:
              x when j < i + 1
              zero when true
            s <- add(s, m) over k
          O[i, j] = s
            ";
        assert!(
            normal(given).contains("for a2 in 0..8:"),
            "{}",
            normal(given)
        );
    }

    #[test]
    fn two_accumulators_masked_apart_keep_the_whole_axis() {
        let given = "
      for i in 0..8:
        for k in 0..8:
          x = load X[i, k]
          let zero = 0.0
          m = select:
            x when k < i + 1
            zero when true
          s <- add(s, m) over k
          t <- add(t, x) over k
        O[i] = s
        P[i] = t
            ";
        assert!(
            normal(given).contains("for a1 in 0..8:"),
            "{}",
            normal(given)
        );
    }

    #[test]
    fn a_ranged_loop_reads_back_and_settles() {
        let given = "for a0 in 0..8:\n  for a1 in 0..a0 + 1 of 8:\n    in0 = load X[a0, a1]\n    acc0 <- add(acc0, in0) over a1\n  O[a0] = acc0";
        assert_eq!(normal(given), given);
    }
}
