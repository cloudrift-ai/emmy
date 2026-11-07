//! What the shared kinds answer beyond their fields: the states a loop carries, a reduction's seed, a sum's terms
//! and its binary form, a select's mask, the literal a statement binds.

use crate::body::Body;
use crate::expr::{Expr, Expression, LitType};
use crate::kinds::elementwise::Op;
use crate::kinds::field::{AxisRef, Names, NumOr};
use crate::kinds::*;
use crate::{Error, Name, Result};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

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
            Statement::Accum(a) => Some(a.name.clone()),
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
                        None => out.push((carry.name.clone(), shape)),
                    }
                }
            }
            for child in stmt.nested() {
                if let Statement::Loop(l) = &**stmt {
                    let mut deeper = inner.clone();
                    deeper.insert(l.axis.name.clone(), l.axis.clone());
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
    /// The identity value of the accumulator: the op's neutral element, `0.0` when it has none.
    pub fn init(&self) -> Expression {
        Expression::float(self.op.identity.unwrap_or(0.0))
    }
}

impl Affine {
    pub fn terms(&self) -> impl Iterator<Item = (f64, &Name)> {
        self.coefficients
            .iter()
            .zip(self.args.iter())
            .map(|(c, arg)| (c.0, arg))
    }

    /// Python's checks: equal lengths, at least one term, no zero coefficient.
    pub fn validate(&self) -> Result<()> {
        let name = &self.name;
        if self.args.len() != self.coefficients.len() {
            return Err(Error::Value(format!(
                "Affine {name:?}: {} args and {} coefficients",
                self.args.len(),
                self.coefficients.len()
            )));
        }
        if self.args.is_empty() {
            return Err(Error::Value(format!(
                "Affine {name:?} has no term: a constant is a Let"
            )));
        }
        if self.coefficients.iter().any(|c| c.0 == 0.0) {
            let shown: Vec<String> = self
                .coefficients
                .iter()
                .map(|c| crate::kinds::field::spell_number(c.0))
                .collect();
            return Err(Error::Value(format!(
                "Affine {name:?} carries a zero coefficient: ({})",
                shown.join(", ")
            )));
        }
        Ok(())
    }

    /// The same value as binary `Assign`s over `Let`-bound literals, the last one named `name`.
    pub fn as_assigns(&self) -> Vec<Arc<Statement>> {
        struct Chain<'a> {
            affine: &'a Affine,
            out: Vec<Arc<Statement>>,
            counter: usize,
        }
        impl Chain<'_> {
            fn fresh(&mut self) -> Name {
                let name: Name = format!("{}__t{}", self.affine.name, self.counter).into();
                self.counter += 1;
                name
            }
            fn bound(&mut self, value: f64) -> Name {
                let name = self.fresh();
                let let_ = Let {
                    name: name.clone(),
                    value: Expression::float(value),
                    dtype: self.affine.dtype.clone(),
                };
                self.out.push(Arc::new(Statement::Let(let_)));
                name
            }
            fn step(&mut self, op: &str, args: Vec<Name>) -> Name {
                let name = self.fresh();
                let assign = Assign {
                    name: name.clone(),
                    op: Op::new(op),
                    args: args.into(),
                    dtype: self.affine.dtype.clone(),
                };
                self.out.push(Arc::new(Statement::Assign(assign)));
                name
            }
        }
        let mut chain = Chain {
            affine: self,
            out: Vec::new(),
            counter: 0,
        };
        let mut total: Option<Name> = None;
        // A positive term leads when there is one, so the chain opens with a value rather than a negation.
        let mut terms: Vec<(f64, &Name)> = self.terms().collect();
        terms.sort_by_key(|(c, _)| *c < 0.0);
        for (coefficient, arg) in terms {
            let term = if coefficient.abs() == 1.0 {
                arg.clone()
            } else {
                let literal = chain.bound(coefficient.abs());
                chain.step("multiply", vec![arg.clone(), literal])
            };
            total = Some(match total {
                None if coefficient > 0.0 => term,
                None => chain.step("negative", vec![term]),
                Some(current) => chain.step(
                    if coefficient > 0.0 { "add" } else { "subtract" },
                    vec![current, term],
                ),
            });
        }
        // An affine has a term (its constructor refuses none), so the chain holds a total here.
        let Some(mut total) = total else {
            return chain.out;
        };
        if self.offset.0 != 0.0 {
            let literal = chain.bound(self.offset.0.abs());
            total = chain.step(
                if self.offset.0 > 0.0 {
                    "add"
                } else {
                    "subtract"
                },
                vec![total, literal],
            );
        }
        let mut out = chain.out;
        match out.last_mut() {
            Some(last) if last.name() == Some(&total) => {
                if let Statement::Assign(a) = &**last {
                    *last = Arc::new(Statement::Assign(Assign {
                        name: self.name.clone(),
                        ..a.clone()
                    }));
                }
            }
            _ => out.push(Arc::new(Statement::Assign(Assign {
                name: self.name.clone(),
                op: Op::new("copy"),
                args: Names::from([total]),
                dtype: self.dtype.clone(),
            }))),
        }
        out
    }
}

/// `stmts` with every `Affine` spelled as its binary `Assign`s; the other statements pass through.
pub fn binary(stmts: &[Arc<Statement>]) -> Vec<Arc<Statement>> {
    stmts
        .iter()
        .flat_map(|stmt| match &**stmt {
            Statement::Affine(affine) => affine.as_assigns(),
            _ => vec![stmt.clone()],
        })
        .collect()
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
        use crate::expr::BinOp::*;
        op = match op {
            Lt => Gt,
            Le => Ge,
            Gt => Lt,
            Ge => Le,
            _ => return None,
        };
        std::mem::swap(&mut left, &mut right);
    }
    if *keep_left != left || *keep_right != right {
        return None;
    }
    use crate::expr::BinOp::*;
    if !matches!((keep_op, op), (Lt, Ge) | (Le, Gt) | (Gt, Le) | (Ge, Lt)) {
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
        && let_.dtype.as_ref().is_none_or(|d| d.name() == "f32")
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
