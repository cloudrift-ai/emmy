//! `Fold::masked`: a tree with every range spelled as a kernel tier reads it.
//!
//! A tensor-core block is a rectangle and a cooperative lane a strided walk, so a tier covers an axis from zero to
//! a number of trips and masks what the range leaves out. This is the one place that turns a range into that form.
//! Each value the term folds is selected against the fold's seed where the position is outside the range, and a
//! bound the trips already hold states no mask. A block of a split (an axis whose window is a partition) is walked
//! from the block's start, the start some end lies a fixed way past: the axis takes that distance as its trips,
//! every read of it is `start + axis`, and a read below the term clamps onto the range, so no masked trip reads
//! past a buffer. Any other range is walked over the whole axis, and the position is the axis. A twisted carrier has no value to select against:
//! its score takes the pivot's seed added through the `where` a program's own mask spells, which is the coordinate
//! mask the chunk tier lands. Loop IR's normal form reads a mask back as the range it came from.

use std::collections::HashMap;
use std::sync::Arc;

use emmy_ir::axis::{Axis, Dim};
use emmy_ir::body::Body;
use emmy_ir::elementwise::Operation;
use emmy_ir::expr::{Expr, ExprClass, Expression, LitType, Num, SimplifyCtx};
use emmy_ir::normalize::ties::unranked;
use emmy_ir::normalize::{affine_sum, affine_terms};
use emmy_ir::statement::{Assign, Let, Select, SelectBranch, Statement};
use emmy_ir::{Error, Name, Result};

use crate::fold::{Fold, Parts, Seed};
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
    if !taken.iter().any(|name| &**name == stem) {
        let name = Name::new(stem);
        taken.push(name);
        return name;
    }
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

fn bind(name: Name, value: Expression) -> Arc<Statement> {
    Arc::new(Statement::Let(Let {
        name,
        value,
        dtype: None,
    }))
}

fn assign(name: Name, value: Expression) -> Arc<Statement> {
    Arc::new(Statement::Assign(Assign {
        name,
        value,
        dtype: None,
    }))
}

/// What a term's range comes to once its axis walks from zero.
struct Walk {
    /// The axis as the body reads it: `start + axis`, or the axis itself where the walk leaves from zero.
    position: Expression,
    /// The starts the walk does not leave from, each a bound `position` stays at or past.
    starts: Vec<Expression>,
    /// The ends the trips do not already hold, each a bound `position` stays below.
    ends: Vec<Expression>,
    /// The trips, where the range fixes them apart from the extent the axis table states.
    trips: Option<Dim>,
}

impl Walk {
    /// A range starts at the greatest of its starts and ends at the least of its ends. A block of a partition
    /// (the axis's window says so) leaves from the start some end lies a fixed way past, the shortest way first:
    /// the block's start, and it takes the block's length, whatever range the fold walked before the split. Any
    /// other walk covers the axis from zero, and each start is a mask like each end.
    fn of(axis: Name, start: &Expression, end: &Expression, axes: &[Arc<Axis>]) -> Result<Walk> {
        let (mut lows, mut highs) = (Vec::new(), Vec::new());
        chain(start, Operation::Maximum, &mut lows);
        chain(end, Operation::Minimum, &mut highs);
        let coordinate = |name: &Name| axes.iter().any(|known| known.name == *name);
        // How far `high` lies past `low`, like terms merged: a block's end less its start is the block's length.
        let room = |low: &Expression, high: &Expression| {
            let apart = Expression::binary(Operation::Subtract, high.clone(), low.clone());
            affine_sum(affine_terms(&apart, ExprClass::Index), &|_| unranked())
                .simplify(&SimplifyCtx::empty())
        };
        let stated = axes.iter().find(|known| known.name == axis);
        // Only a block of a partition leaves from its own start: the block coordinate is one value for a whole
        // launch block, where a start that moves with a row would give every row of a tile its own positions.
        let block = stated.is_some_and(|known| known.window.as_ref().is_some_and(|w| w.partition));
        let mut from: Option<(usize, Option<i64>)> = None;
        for (index, low) in lows.iter().enumerate().filter(|_| block) {
            for high in &highs {
                let room = room(low, high)?;
                if room.free_vars().iter().any(coordinate) {
                    continue;
                }
                let shorter = match (&from, room.int_literal()) {
                    (None, _) => true,
                    (Some((_, held)), Some(new)) => held.is_none_or(|held| new < held),
                    (Some(_), None) => false,
                };
                if shorter {
                    from = Some((index, room.int_literal()));
                }
            }
        }
        // No end a fixed way past any start: the walk covers the axis from zero and every start is a mask.
        let base = match from {
            Some((index, _)) => lows.remove(index),
            None => Expression::int(0),
        };
        lows.retain(|low| low.int_literal() != Some(0) || base.int_literal() != Some(0));
        let rebased = base.int_literal() != Some(0);
        let position = if rebased {
            Expression::binary(Operation::Add, base.clone(), Expression::var(&axis))
        } else {
            Expression::var(&axis)
        };
        // How far each end lies past the start: the trips are the least of the ones no coordinate moves.
        let rooms: Vec<Expression> = highs
            .iter()
            .map(|high| room(&base, high))
            .collect::<Result<_>>()?;
        let fixed: Vec<&Expression> = rooms
            .iter()
            .filter(|room| !room.free_vars().iter().any(coordinate))
            .collect();
        let trips: Option<Expression> = if !rebased {
            stated.map(|known| known.extent.expression.clone())
        } else {
            match fixed.iter().filter_map(|room| room.int_literal()).min() {
                Some(least) => Some(Expression::int(least)),
                None => fixed.first().map(|room| (*room).clone()),
            }
        };
        let ends = highs
            .into_iter()
            .zip(&rooms)
            .filter(|(_, room)| match (&trips, room.int_literal()) {
                (Some(trips), Some(room)) => trips.int_literal().is_none_or(|trips| room < trips),
                (Some(trips), None) => *room != trips,
                (None, _) => true,
            })
            .map(|(bound, _)| bound)
            .collect();
        Ok(Walk {
            position,
            starts: lows,
            ends,
            trips: trips.filter(|_| rebased).map(|trips| Dim::new(trips, None)),
        })
    }

    /// `position` at or past every start the walk does not leave from and below every end the trips do not hold,
    /// or `None` when the walk holds them all.
    fn inside(&self) -> Option<Expression> {
        let past = self.starts.iter().map(|start| {
            Expression::binary(Operation::LessEqual, start.clone(), self.position.clone())
        });
        let below = self
            .ends
            .iter()
            .map(|end| Expression::binary(Operation::Less, self.position.clone(), end.clone()));
        past.chain(below)
            .reduce(|so_far, holds| Expression::binary(Operation::LogicalAnd, so_far, holds))
    }

    /// The position a read below the term takes: clamped onto the first or the last position in range where a
    /// trip may leave it, so the read stays inside its buffer and the fold masks what it folds there.
    fn read(&self) -> Expression {
        let least = |a, b| Expression::apply(Operation::Minimum, vec![a, b]);
        let greatest = |a, b| Expression::apply(Operation::Maximum, vec![a, b]);
        let below = match self.ends.iter().cloned().reduce(least) {
            None => self.position.clone(),
            Some(end) => Expression::new(Expr::Ternary(
                Expression::binary(Operation::Less, self.position.clone(), end.clone()),
                self.position.clone(),
                Expression::binary(Operation::Subtract, end, Expression::int(1)),
            )),
        };
        match self.starts.iter().cloned().reduce(greatest) {
            None => below,
            Some(start) => Expression::new(Expr::Ternary(
                Expression::binary(Operation::Less, self.position.clone(), start.clone()),
                start,
                below,
            )),
        }
    }
}

impl Fold {
    /// This tree with no range left, and the axis table it then reads: every term that walks a range walks from
    /// zero, reads its axis as `start + axis`, and folds each value selected against its seed where `start + axis`
    /// is past an end. `axes` is the kernel's axis table; an axis whose range fixes its trips takes them as its
    /// extent in the table returned. A term without a range, over operands without one, is the term itself, so
    /// sharing is kept.
    pub fn masked(&self, axes: &[Arc<Axis>]) -> Result<(Fold, Vec<Arc<Axis>>)> {
        let mut taken = self.names();
        let mut table = axes.to_vec();
        let masked = self.masked_in(axes, &mut table, &mut taken, &mut HashMap::new())?;
        Ok((masked, table))
    }

    fn masked_in(
        &self,
        axes: &[Arc<Axis>],
        table: &mut Vec<Arc<Axis>>,
        taken: &mut Vec<Name>,
        done: &mut HashMap<usize, Fold>,
    ) -> Result<Fold> {
        if let Some(found) = done.get(&self.address()) {
            return Ok(found.clone());
        }
        let operands = self
            .operands()
            .iter()
            .map(|edge| edge.masked_in(axes, table, taken, done))
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
            (Some((start, end)), _) => self.masking(operands, start, end, axes, table, taken)?,
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
        table: &mut [Arc<Axis>],
        taken: &mut Vec<Name>,
    ) -> Result<Fold> {
        let axis = self
            .axis()
            .ok_or_else(|| Error::Value("a range is a reduce axis's".into()))?;
        if self.observe().is_some() || self.carries() {
            return Err(Error::NotImplemented(format!(
                "no tier masks the range of '{axis}': the term is observed or carries a state"
            )));
        }
        let walk = Walk::of(axis, start, end, axes)?;
        if let Some(trips) = &walk.trips
            && let Some(slot) = table.iter_mut().find(|known| known.name == axis)
        {
            *slot = Arc::new(Axis {
                name: axis,
                extent: trips.clone(),
                window: slot.window.clone(),
            });
        }
        let lift = self.lift();
        let (mut operands, mut body) = (operands, lift.body().stmts().to_vec());
        if walk.position.as_var().is_none() {
            let reads = TermRewrite {
                sigma: HashMap::from([(axis, walk.read())]),
                ..TermRewrite::default()
            };
            // An operand that does not read the axis is the term it was, and one reached twice is rewritten once:
            // which term stands at an operand position is how a tree says a value is shared.
            let mut moved: HashMap<usize, Fold> = HashMap::new();
            operands = operands
                .iter()
                .map(|edge| {
                    if !edge.free_axes().contains(&axis) {
                        return Ok(edge.clone());
                    }
                    if let Some(found) = moved.get(&edge.address()) {
                        return Ok(found.clone());
                    }
                    let rewritten = edge.rewrite(&reads)?;
                    moved.insert(edge.address(), rewritten.clone());
                    Ok(rewritten)
                })
                .collect::<Result<_>>()?;
            let sigma = HashMap::from([(axis, walk.position.clone())]);
            body = body
                .iter()
                .map(|stmt| Statement::substitute(stmt, &sigma))
                .collect();
        }
        let lead = 1 + self.bindings().len();
        let mut params = lift.params()[..lead].to_vec();
        let mut results = lift.results().to_vec();
        let seed = |position: usize| {
            self.init()
                .get(position)
                .and_then(|seed| match seed {
                    Seed::Int(value) => Some(*value as f64),
                    Seed::Float(value) => Some(*value),
                    Seed::Buffer(_) => None,
                })
                .ok_or_else(|| {
                    Error::Value(format!(
                        "the fold over '{axis}' walks a range and has no seed to fold past its end"
                    ))
                })
        };
        match walk.inside() {
            None => {}
            Some(_) if self.twist().is_some() => {
                // The score, less the pivot's seed past the end: `score + where(key < end, 0, seed)`.
                let score = results[0];
                let name =
                    |part: &str, taken: &mut Vec<Name>| fresh(&format!("{score}__{part}"), taken);
                let (key, end_, keep, fill) = (
                    name("key", taken),
                    name("end", taken),
                    name("keep", taken),
                    name("fill", taken),
                );
                let (inside, mask, masked) = (
                    name("inside", taken),
                    name("mask", taken),
                    name("in", taken),
                );
                let end = walk
                    .ends
                    .iter()
                    .cloned()
                    .reduce(|a, b| Expression::apply(Operation::Minimum, vec![a, b]))
                    .unwrap_or_else(|| end.clone());
                let var = |name: &Name| Expression::var(name);
                body.extend([
                    bind(key, walk.position.clone()),
                    bind(end_, end),
                    bind(keep, Expression::float(0.0)),
                    bind(fill, Expression::float(seed(0)?)),
                    assign(
                        inside,
                        Expression::binary(Operation::Less, var(&key), var(&end_)),
                    ),
                    assign(
                        mask,
                        Expression::apply(
                            Operation::Where,
                            vec![var(&inside), var(&keep), var(&fill)],
                        ),
                    ),
                    assign(
                        masked,
                        Expression::binary(Operation::Add, var(&score), var(&mask)),
                    ),
                ]);
                results[0] = masked;
            }
            Some(condition) => {
                let mut seeds: Vec<(u64, Name)> = Vec::new();
                for (position, value) in lift.results().iter().enumerate() {
                    let seed = seed(position)?;
                    let literal = match seeds.iter().find(|(bits, _)| *bits == seed.to_bits()) {
                        Some((_, name)) => *name,
                        None => {
                            let name = fresh(&format!("{value}__seed"), taken);
                            seeds.push((seed.to_bits(), name));
                            operands.push(Fold::new(Parts {
                                lift: Some(Lambda::closing(
                                    Vec::new(),
                                    &Body::new(vec![bind(name, Expression::float(seed))]),
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
        }
        let mut parts = self.parts();
        parts.operands = operands;
        parts.lift = Some(Lambda::closing(params, &Body::new(body), results)?);
        parts.range = None;
        Fold::new(parts)
    }
}
