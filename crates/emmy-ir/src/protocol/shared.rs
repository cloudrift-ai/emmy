//! What the shared statement types answer beyond their fields: the states a loop carries, a reduction's seed, a
//! sum's terms and its binary form, the literal a statement binds.

use crate::Name;
use crate::axis::Range;
use crate::body::Body;
use crate::expr::Expression;
use crate::field::{AxisRef, NumOr};
use crate::statement::*;
use std::collections::HashMap;

/// The seeded accumulators a reduce loop declares in its enclosing scope: the `Accum` targets of its immediate body.
pub fn seeded_accumulators(stmt: &Statement) -> Vec<Name> {
    let (body, seed) = match stmt {
        Statement::Loop(l) => (&l.body, l.seed),
        _ => return vec![],
    };
    if !seed || !body.carries().is_empty() {
        return vec![];
    }
    body.stmts()
        .iter()
        .filter_map(|s| match &**s {
            Statement::Accum(a) => Some(a.name),
            _ => None,
        })
        .collect()
}

/// The carried states the loop over `axis` carries, `body` its body, each with its cell shape: an axis per
/// position of the cell, `None` where normalization dropped a size-one axis. A state is carried by the loop its
/// first coordinate names.
pub fn carried_cells(axis: &Name, body: &Body) -> Vec<(Name, Vec<Option<AxisRef>>)> {
    if body.carries().is_empty() {
        return vec![];
    }
    let mut out: Vec<(Name, Vec<Option<AxisRef>>)> = Vec::new();
    fn walk(
        axis: &Name,
        body: &Body,
        inner: &HashMap<Name, AxisRef>,
        out: &mut Vec<(Name, Vec<Option<AxisRef>>)>,
    ) {
        for stmt in body.stmts() {
            if let Statement::Carry(carry) = &**stmt
                && carry.step() == Some(axis)
            {
                let shape = carry
                    .cell()
                    .iter()
                    .map(|e| e.as_var().and_then(|name| inner.get(name).cloned()))
                    .collect();
                match out.iter_mut().find(|(name, _)| *name == carry.name) {
                    Some(entry) => entry.1 = shape,
                    None => out.push((carry.name, shape)),
                }
            }
            for child in stmt.nested() {
                if let Statement::Loop(l) = &**stmt {
                    let mut deeper = inner.clone();
                    deeper.insert(l.axis.name, l.axis.clone());
                    walk(axis, &child, &deeper, out);
                } else {
                    walk(axis, &child, inner, out);
                }
            }
        }
    }
    walk(axis, body, &HashMap::new(), &mut out);
    out
}

impl Loop {
    /// The carried states this loop carries.
    pub fn carries(&self) -> Vec<(Name, Vec<Option<AxisRef>>)> {
        carried_cells(&self.axis.name, &self.body)
    }

    /// Where the axis walks: its range, `0..extent` when the loop states none.
    pub fn walk(&self) -> Range {
        match &self.range {
            Some((start, end)) => Range {
                start: start.clone(),
                end: end.clone(),
            },
            None => Range::whole(&self.axis),
        }
    }

    /// How far the axis moves each trip: the step, one when the loop states none.
    pub fn stride(&self) -> Expression {
        self.step.clone().unwrap_or_else(|| Expression::int(1))
    }

    /// The one value the axis takes when the loop is known to take exactly one trip: `0` for a loop of extent one
    /// without a range, the start of a range whose end is one past it. `None` for any other loop.
    pub fn one_trip(&self) -> Option<Expression> {
        match &self.range {
            None => (self.axis.extent.as_static() == Some(1)).then(|| Expression::int(0)),
            Some((start, end)) => Range {
                start: start.clone(),
                end: end.clone(),
            }
            .trips()
            .filter(|trips| *trips == 1)
            .map(|_| start.clone()),
        }
    }

    /// A loop is a reduce loop iff it carries something: an `Accum` in its immediate body, or a state.
    pub fn is_reduce(&self) -> bool {
        self.body
            .stmts()
            .iter()
            .any(|s| matches!(**s, Statement::Accum(_)))
            || !self.carries().is_empty()
    }
}

impl Carry {
    /// The axis of the loop that carries the state: the first coordinate, the step.
    pub fn step(&self) -> Option<&Name> {
        self.index.first().and_then(|e| e.as_var())
    }

    /// The cell this statement defines: the coordinates past the step.
    pub fn cell(&self) -> &[Expression] {
        self.index.get(1..).unwrap_or(&[])
    }

    /// The axes of the cell this statement defines: the entries of the cell that are variables.
    pub fn cells(&self) -> Vec<&Name> {
        self.cell().iter().filter_map(|e| e.as_var()).collect()
    }
}

impl Accum {
    /// The value the accumulator starts from: the op's neutral element as a kernel seeds it, `0.0` when it has none.
    pub fn init(&self) -> Expression {
        Expression::float(self.op.seed().unwrap_or(0.0))
    }
}

/// The axis a plain Loop IR coordinate expression names, when it is one variable.
pub fn axis_of(expression: &Expression) -> Option<&Name> {
    expression.as_var()
}

/// A `NumOr` value's number, when it is one.
pub fn number_of<T>(value: &NumOr<T>) -> Option<f64> {
    match value {
        NumOr::Num(n) => Some(n.as_f64()),
        NumOr::Value(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize::normalize_body;
    use crate::normalize::testing::{body, indented, text};

    /// `S_c = decay_c * S_{c-1} + W @ S_{c-1} + U_c` from zero, storing each step the state it read: a step reads
    /// other cells of its own state.
    const STEP: &str = "
    for c in 0..3:
      decay = load D[c]
      for i in 0..4:
        for j in 0..4:
          for k in 0..4:
            w = load W[i, k]
            other = load S[c - 1, k, j]
            p = multiply(w, other)
            mixed <- add(mixed, p)
          u = load U[c, i, j]
          own = load S[c - 1, i, j]
          kept = multiply(own, decay)
          moved = add(kept, mixed)
          next = add(moved, u)
          S[c, i, j] <- next seed 0.0
          out[c, i, j] = own
    ";

    #[test]
    fn a_loop_over_the_cells_carries_nothing() {
        // A scalar fold is carried by its own loop, and a state by the loop its first coordinate names, outside
        // its cells.
        let step = body(STEP);
        let Statement::Loop(outer) = &*step.stmts()[0] else {
            panic!("not a loop");
        };
        let loops: Vec<Loop> = outer
            .body
            .iter_stmts()
            .iter()
            .filter_map(|stmt| match &**stmt {
                Statement::Loop(inner) => Some(inner.clone()),
                _ => None,
            })
            .collect();
        let [over_i, over_j, over_k] = &loops[..] else {
            panic!("three loops under the steps");
        };
        assert_eq!(
            outer.carries(),
            [(
                Name::from("S"),
                vec![Some(over_i.axis.clone()), Some(over_j.axis.clone())]
            )]
        );
        assert!(!over_i.is_reduce() && !over_j.is_reduce());
        assert!(over_k.is_reduce() && over_k.carries().is_empty());
    }

    #[test]
    fn a_carried_state_prints_as_its_statements() {
        // The listing is the body's text: the state's update names its seed, and a read of it is a load: one step
        // back inside the loop, the last step after it.
        let listing = |source: &str| indented(&text(&normalize_body(&body(source)).unwrap()));
        let fold_then_read = "
      for c in 0..3:
        for i in 0..4:
          u = load U[c, i]
          own = load S[c - 1, i]
          next = add(own, u)
          S[c, i] <- next seed 0
      for r in 0..4:
        last = load S[2, r]
        out[r] = last
        ";
        assert_eq!(
            listing(fold_then_read),
            "  for a0 in 0..3:\n    for a1 in 0..4:\n      in0 = load acc0[a0 - 1, a1]\n      in1 = load U[a0, a1]\n      v0 = in0 + in1\n      acc0[a0, a1] <- v0 seed 0\n  for a2 in 0..4:\n    in2 = load acc0[2, a2]\n    out[a2] = in2"
        );
        // The same shape with a scalar fold in place of the carried state.
        let plain_fold = "
      for c in 0..3:
        for i in 0..4:
          u = load U[c, i]
          total <- add(total, u)
        out[c] = total
        ";
        assert_eq!(
            listing(plain_fold),
            "  for a0 in 0..3:\n    for a1 in 0..4:\n      in0 = load U[a0, a1]\n      acc0 <- add(acc0, in0)\n    out[a0] = acc0"
        );
    }
}
