//! What the shared statement types answer beyond their fields: the states a loop carries, a reduction's seed, a
//! sum's terms and its binary form, a select's mask, the literal a statement binds.

use crate::Name;
use crate::body::Body;
use crate::dtype::DType;
use crate::elementwise::Operation;
use crate::expr::{Expr, Expression, LitType};
use crate::field::{AxisRef, NumOr};
use crate::statement::*;
use std::collections::{HashMap, HashSet};

/// The seeded accumulators a reduce loop declares in its enclosing scope: the `Accum` targets of its immediate body.
pub fn seeded_accumulators(stmt: &Statement) -> Vec<Name> {
    let (body, seed) = match stmt {
        Statement::Loop(l) => (&l.body, l.seed),
        Statement::StridedLoop(l) => (&l.body, l.seed),
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

/// The carried states a loop over `body` carries, each with its cell shape: an axis per index position, `None`
/// where normalization dropped a size-one axis.
pub fn carried_cells(body: &Body) -> Vec<(Name, Vec<Option<AxisRef>>)> {
    if body.carries().is_empty() {
        return vec![];
    }
    let mut out: Vec<(Name, Vec<Option<AxisRef>>)> = Vec::new();
    fn walk(
        body: &Body,
        inner: &HashMap<Name, AxisRef>,
        out: &mut Vec<(Name, Vec<Option<AxisRef>>)>,
    ) {
        for stmt in body.stmts() {
            if let Statement::Carry(carry) = &**stmt {
                let cells: HashSet<&Name> = carry.cells().into_iter().collect();
                let keys: HashSet<&Name> = inner.keys().collect();
                if cells == keys {
                    let shape = carry
                        .index
                        .iter()
                        .map(|e| e.as_var().and_then(|name| inner.get(name).cloned()))
                        .collect();
                    match out.iter_mut().find(|(name, _)| *name == carry.name) {
                        Some(entry) => entry.1 = shape,
                        None => out.push((carry.name, shape)),
                    }
                }
            }
            for child in stmt.nested() {
                if let Statement::Loop(l) = &**stmt {
                    let mut deeper = inner.clone();
                    deeper.insert(l.axis.name, l.axis.clone());
                    walk(&child, &deeper, out);
                } else {
                    walk(&child, inner, out);
                }
            }
        }
    }
    walk(body, &HashMap::new(), &mut out);
    out
}

impl Loop {
    /// The carried states this loop carries.
    pub fn carries(&self) -> Vec<(Name, Vec<Option<AxisRef>>)> {
        carried_cells(&self.body)
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

impl StridedLoop {
    pub fn is_reduce(&self) -> bool {
        self.body
            .stmts()
            .iter()
            .any(|s| matches!(**s, Statement::Accum(_)))
    }
}

impl Carry {
    /// The axes of the cell this statement defines: the index entries that are variables.
    pub fn cells(&self) -> Vec<&Name> {
        self.index.iter().filter_map(|e| e.as_var()).collect()
    }
}

impl Accum {
    /// The value the accumulator starts from: the op's neutral element as a kernel seeds it, `0.0` when it has none.
    pub fn init(&self) -> Expression {
        Expression::float(self.op.seed().unwrap_or(0.0))
    }
}

/// The mask branch of an ordered two-way comparison, or `None` for another `Select` form.
pub fn mask_select_predicate(select: &Select) -> Option<Expression> {
    if select.branches.len() != 2 {
        return None;
    }
    let (keep, mask) = (&select.branches[0], &select.branches[1]);
    let (
        Expr::Binary(keep_op, keep_left, keep_right),
        Expr::Binary(mask_op, mask_left, mask_right),
    ) = (&*keep.select, &*mask.select)
    else {
        return None;
    };
    let (mut op, mut left, mut right) = (*mask_op, mask_left.clone(), mask_right.clone());
    if *keep_left == right && *keep_right == left {
        op = op.mirror()?;
        std::mem::swap(&mut left, &mut right);
    }
    if *keep_left != left || *keep_right != right {
        return None;
    }
    use Operation::{Greater, GreaterEqual, Less, LessEqual};
    if !matches!(
        (keep_op, op),
        (Less, GreaterEqual) | (LessEqual, Greater) | (Greater, LessEqual) | (GreaterEqual, Less)
    ) {
        return None;
    }
    Some(mask.select.clone())
}

/// A `Let` of a float literal at the default precision: the number it binds its name to, else `None`.
pub fn literal_of(stmt: &Statement) -> Option<f64> {
    literal_binding(stmt).map(|(_, value)| value)
}

/// `literal_of` with the name it binds.
pub fn literal_binding(stmt: &Statement) -> Option<(&Name, f64)> {
    if let Statement::Let(let_) = stmt
        && let Expr::Literal(lit) = &*let_.value
        && lit.dtype == LitType::Float
        && let_.dtype.is_none_or(|d| d == DType::F32)
    {
        return Some((&let_.name, lit.value.as_f64()));
    }
    None
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
                        other = pre S[k, j]
                        p = multiply(w, other)
                        mixed <- add(mixed, p)
                    u = load U[c, i, j]
                    own = pre S[i, j]
                    kept = multiply(own, decay)
                    moved = add(kept, mixed)
                    next = add(moved, u)
                    S[i, j] <- next seed 0.0
                    out[c, i, j] = own
    ";

    #[test]
    fn a_loop_over_the_cells_carries_nothing() {
        // One rule places every carrier, the nearest enclosing loop its index does not read: a scalar fold is
        // carried by its own loop and a state by the loop outside its cells.
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
        // The listing is the body's text: the state's update names its seed, and a read of it is a `pre`.
        let listing = |source: &str| indented(&text(&normalize_body(&body(source)).unwrap()));
        let fold_then_read = "
            for c in 0..3:
                for i in 0..4:
                    u = load U[c, i]
                    own = pre S[i]
                    next = add(own, u)
                    S[i] <- next seed 0
            for r in 0..4:
                last = pre S[r]
                out[r] = last
        ";
        assert_eq!(
            listing(fold_then_read),
            "    for a0 in 0..3:\n        for a1 in 0..4:\n            in0 = load U[a0, a1]\n            v0 = pre acc0[a1]\n            v1 = in0 + v0\n            acc0[a1] <- v1 seed 0\n    for a2 in 0..4:\n        v2 = pre acc0[a2]\n        out[a2] = v2"
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
            "    for a0 in 0..3:\n        for a1 in 0..4:\n            in0 = load U[a0, a1]\n            acc0 <- add(acc0, in0)\n        out[a0] = acc0"
        );
    }
}
