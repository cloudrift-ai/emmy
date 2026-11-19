//! `Fold::masked`: a tree with every range spelled as a kernel tier reads it.
//!
//! A tensor-core block is a rectangle and a cooperative lane a strided walk, so a tier covers an axis from zero to
//! its extent and masks what the range leaves out. This is the one place that turns a range into that form: the
//! axis walks `0..extent`, every read of it is `start + axis`, and each value the term folds is selected against the
//! fold's neutral element under `start + axis < end`. A bound the extent already holds states no mask. Loop IR's
//! normal form reads the result back as the range it came from, so the two forms are one kernel.

use std::collections::HashMap;
use std::sync::Arc;

use emmy_ir::axis::Axis;
use emmy_ir::body::Body;
use emmy_ir::elementwise::Operation;
use emmy_ir::expr::{Expression, LitType, Num, SimplifyCtx};
use emmy_ir::statement::{Let, Select, SelectBranch, Statement};
use emmy_ir::{Error, Name, Result};

use crate::fold::{Fold, Parts};
use crate::lambda::Lambda;
use crate::rewrite::TermRewrite;

/// The operands of a chain of `op`, in order.
fn chain(e: &Expression, op: Operation, out: &mut Vec<Expression>) {
    match e.operation() {
        Some((found, [a, b])) if found == op => {
            chain(a, op, out);
            chain(b, op, out);
        }
        _ => out.push(e.clone()),
    }
}

/// A name of `stem` that `taken` does not hold yet, then held.
fn fresh(stem: &str, taken: &mut Vec<Name>) -> Name {
    let mut index = 0;
    loop {
        let candidate = Name::new(&format!("{stem}{index}"));
        if !taken.contains(&candidate) {
            taken.push(candidate);
            return candidate;
        }
        index += 1;
    }
}

impl Fold {
    /// This tree with no range left: every term that walks a range walks its whole extent from zero, reads its axis
    /// as `start + axis`, and folds each value selected against the neutral element where `start + axis` is past
    /// the end. `axes` is the kernel's axis table, whose extents say which bounds need no mask. A term without a
    /// range, over operands without one, is the term itself, so sharing is kept.
    pub fn masked(&self, axes: &[Arc<Axis>]) -> Result<Fold> {
        let mut taken = self.names();
        self.masked_in(axes, &mut taken, &mut HashMap::new())
    }

    fn masked_in(
        &self,
        axes: &[Arc<Axis>],
        taken: &mut Vec<Name>,
        done: &mut HashMap<usize, Fold>,
    ) -> Result<Fold> {
        if let Some(found) = done.get(&self.address()) {
            return Ok(found.clone());
        }
        let operands = self
            .operands()
            .iter()
            .map(|edge| edge.masked_in(axes, taken, done))
            .collect::<Result<Vec<_>>>()?;
        let same = operands
            .iter()
            .zip(self.operands())
            .all(|(new, old)| new.ptr_eq(old));
        let out = match (self.range(), same) {
            (None, true) => self.clone(),
            (None, false) => {
                let mut parts = self.parts();
                parts.operands = operands;
                Fold::formed(parts)
            }
            (Some((start, end)), _) => self.masking(operands, start, end, axes, taken)?,
        };
        done.insert(self.address(), out.clone());
        Ok(out)
    }

    fn masking(
        &self,
        operands: Vec<Fold>,
        start: &Expression,
        end: &Expression,
        axes: &[Arc<Axis>],
        taken: &mut Vec<Name>,
    ) -> Result<Fold> {
        let axis = self
            .axis()
            .ok_or_else(|| Error::Value("a range is a reduce axis's".into()))?;
        if self.twist().is_some() || self.observe().is_some() || self.carries() {
            return Err(Error::NotImplemented(format!(
                "no tier masks the range of '{axis}': the term is twisted, observed or carries a state"
            )));
        }
        let ops = self.ops().unwrap_or_default();
        let lift = self.lift();
        // The axis as the body reads it once the loop starts at zero.
        let rebased = start.int_literal() != Some(0);
        let coordinate = if rebased {
            Expression::binary(Operation::Add, start.clone(), Expression::var(&axis))
        } else {
            Expression::var(&axis)
        };
        let (mut operands, mut body) = (operands, lift.body().stmts().to_vec());
        if rebased {
            let sigma = HashMap::from([(axis, coordinate.clone())]);
            let rewrite = TermRewrite {
                sigma: sigma.clone(),
                ..TermRewrite::default()
            };
            operands = operands
                .iter()
                .map(|edge| edge.rewrite(&rewrite))
                .collect::<Result<_>>()?;
            body = body
                .iter()
                .map(|stmt| Statement::substitute(stmt, &sigma))
                .collect();
        }
        // One comparison per operand of the end's `min`, less the ones the extent holds: `start + axis < bound`
        // for every `axis` below the extent when `bound - start` is at least the extent.
        let extent = axes
            .iter()
            .find(|candidate| candidate.name == axis)
            .and_then(|found| found.extent.as_static());
        let mut bounds = Vec::new();
        chain(end, Operation::Minimum, &mut bounds);
        let mut condition: Option<Expression> = None;
        for bound in bounds {
            let room = Expression::binary(Operation::Subtract, bound.clone(), start.clone())
                .simplify(&SimplifyCtx::empty())?
                .int_literal();
            if let (Some(room), Some(extent)) = (room, extent)
                && room >= extent
            {
                continue;
            }
            let holds = Expression::binary(Operation::Less, coordinate.clone(), bound);
            condition = Some(match condition {
                None => holds,
                Some(so_far) => Expression::binary(Operation::LogicalAnd, so_far, holds),
            });
        }
        let lead = 1 + self.bindings().len();
        let mut params = lift.params()[..lead].to_vec();
        let mut results = lift.results().to_vec();
        if let Some(condition) = condition {
            let mut neutrals: Vec<(u64, Name)> = Vec::new();
            for (position, value) in lift.results().iter().enumerate() {
                let neutral = ops
                    .get(position)
                    .and_then(|op| op.known())
                    .and_then(|op| op.info().neutral)
                    .ok_or_else(|| {
                        Error::Value(format!(
                            "the fold over '{axis}' walks a range and its operation has no neutral element"
                        ))
                    })?;
                let literal = match neutrals.iter().find(|(bits, _)| *bits == neutral.to_bits()) {
                    Some((_, name)) => *name,
                    None => {
                        let name = fresh(&format!("{axis}__neutral"), taken);
                        neutrals.push((neutral.to_bits(), name));
                        let bound = Arc::new(Statement::Let(Let {
                            name,
                            value: Expression::float(neutral),
                            dtype: None,
                        }));
                        operands.push(Fold::new(Parts {
                            lift: Some(Lambda::closing(
                                Vec::new(),
                                &Body::new(vec![bound]),
                                vec![name],
                            )?),
                            ..Parts::default()
                        })?);
                        params.push(name);
                        name
                    }
                };
                let masked = fresh(&format!("{value}__in"), taken);
                body.push(Arc::new(Statement::Select(Select {
                    name: masked,
                    branches: Arc::from(vec![
                        SelectBranch {
                            value: *value,
                            select: condition.clone(),
                        },
                        SelectBranch {
                            value: literal,
                            select: Expression::literal(Num::Bool(true), LitType::Bool),
                        },
                    ]),
                })));
                results[position] = masked;
            }
        }
        let mut parts = self.parts();
        parts.operands = operands;
        parts.lift = Some(Lambda::closing(params, &Body::new(body), results)?);
        parts.range = None;
        Fold::new(parts)
    }
}
