//! `Fold::scheduled`: a tree as the schedule addresses it and a kernel tier lowers it.
//!
//! A tier reads a term's range itself: a scalar walk and a cooperative reduce as the bounds of their loop, the chunk
//! tier as the chunks it walks and the mask of the block a bound cuts. So a term keeps its range, and the one thing
//! this form changes is a block of a split (an axis whose window is a partition). Such a block is walked from its own
//! start, the start some end lies a fixed way past: the axis takes that distance as its trips, every read of it is
//! `start + axis`, and the range the term keeps is what its own bounds leave of the block. The last block of a
//! symbolic stream may run past the stream's end, so a read below such a term clamps onto the last position.

use std::collections::HashMap;
use std::sync::Arc;

use emmy_ir::axis::{Axis, Dim};
use emmy_ir::body::Body;
use emmy_ir::elementwise::Operation;
use emmy_ir::expr::{Expr, ExprClass, Expression, SimplifyCtx};
use emmy_ir::normalize::ties::unranked;
use emmy_ir::normalize::{affine_sum, affine_terms};
use emmy_ir::statement::Statement;
use emmy_ir::{Error, Name, Result};

use crate::fold::Fold;
use crate::lambda::Lambda;
use crate::rewrite::TermRewrite;

/// How far `high` lies past `low`, like terms merged: a block's end less its start is the block's length.
fn room(low: &Expression, high: &Expression) -> Result<Expression> {
    let apart = Expression::binary(Operation::Subtract, high.clone(), low.clone());
    affine_sum(affine_terms(&apart, ExprClass::Index), &|_| unranked())
        .simplify(&SimplifyCtx::empty())
}

/// What a term's range comes to once its axis walks from zero.
struct Walk {
    /// The start the walk leaves from, where that is not zero: the start of a block of a partition.
    base: Option<Expression>,
    /// The axis as the body reads it: `start + axis`, or the axis itself where the walk leaves from zero.
    position: Expression,
    /// The starts the walk does not leave from, each a bound `position` stays at or past.
    starts: Vec<Expression>,
    /// The ends the trips do not already hold, each a bound `position` stays below.
    ends: Vec<Expression>,
    /// The trips, where the range fixes them apart from the extent the axis table states.
    trips: Option<Dim>,
    /// The axis's extent, where a walk that leaves from a block's start may run past it.
    overrun: Option<Expression>,
}

impl Walk {
    /// A range starts at the greatest of its starts and ends at the least of its ends. A block of a partition
    /// (the axis's window says so) leaves from the start some end lies a fixed way past, the shortest way first:
    /// the block's start, and it takes the block's length, whatever range the fold walked before the split. Any
    /// other walk covers the axis from zero, and each start is a mask like each end.
    fn of(axis: Name, start: &Expression, end: &Expression, axes: &[Arc<Axis>]) -> Result<Walk> {
        let mut lows = start.chain(Operation::Maximum);
        let highs = end.chain(Operation::Minimum);
        let coordinate = |name: &Name| axes.iter().any(|known| known.name == *name);
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
        let ends: Vec<Expression> = ends;
        let overrun = stated
            .map(|known| known.extent.expression.clone())
            .filter(|extent| rebased && ends.contains(extent));
        Ok(Walk {
            base: rebased.then_some(base),
            position,
            starts: lows,
            ends,
            trips: trips.filter(|_| rebased).map(|trips| Dim::new(trips, None)),
            overrun,
        })
    }

    /// The position a read below the term takes. A walk that leaves from a block's start can run past the end
    /// of a symbolic stream, whose last block is short: there the read clamps onto the stream's last position, so
    /// it stays inside its buffer and the fold masks what it folds. No other bound moves a read: a block of a
    /// static stream lies inside it, and a walk from zero stays inside the axis.
    fn read(&self) -> Expression {
        match &self.overrun {
            None => self.position.clone(),
            Some(extent) => Expression::new(Expr::Ternary(
                Expression::binary(Operation::Less, self.position.clone(), extent.clone()),
                self.position.clone(),
                Expression::binary(Operation::Subtract, extent.clone(), Expression::int(1)),
            )),
        }
    }
}

impl Fold {
    /// This tree as the schedule and the kernel tiers take it, and the axis table it then reads. Every term keeps
    /// its range, stated of the axis as a tier walks it: a block of a partition walks from zero, takes the block's
    /// length as its extent in the table returned, reads its axis as `start + axis`, and keeps the range its own
    /// bounds leave of the block. A term that walks no such block, over operands that walk none, is the term
    /// itself, so sharing is kept.
    pub fn scheduled(&self, axes: &[Arc<Axis>]) -> Result<(Fold, Vec<Arc<Axis>>)> {
        let mut table = axes.to_vec();
        let mut done = HashMap::new();
        let walked = self.walked(axes, &mut table, &mut done)?;
        Ok((walked, table))
    }

    /// The start a block of a partition leaves from, for a term that walks such a block ([`Walk::of`]); `None`
    /// for any other term. The one place that says which coordinate moves a block.
    pub fn block_start(&self, axes: &[Arc<Axis>]) -> Result<Option<Expression>> {
        match (self.axis(), self.range()) {
            (Some(axis), Some((start, end))) => Ok(Walk::of(axis, start, end, axes)?.base),
            _ => Ok(None),
        }
    }

    fn walked(
        &self,
        axes: &[Arc<Axis>],
        table: &mut Vec<Arc<Axis>>,
        done: &mut HashMap<usize, Fold>,
    ) -> Result<Fold> {
        if let Some(found) = done.get(&self.address()) {
            return Ok(found.clone());
        }
        let operands = self
            .operands()
            .iter()
            .map(|edge| edge.walked(axes, table, done))
            .collect::<Result<Vec<_>>>()?;
        let same = operands
            .iter()
            .zip(self.operands())
            .all(|(new, old)| new.ptr_eq(old));
        let block = match (self.axis(), self.range()) {
            (Some(axis), Some((start, end))) => {
                let walk = Walk::of(axis, start, end, axes)?;
                walk.base.is_some().then_some((axis, walk))
            }
            _ => None,
        };
        let out = match block {
            Some((axis, walk)) => self.rebased(axis, &walk, operands, table)?,
            None if same => self.clone(),
            None => {
                let mut parts = self.parts();
                parts.operands = operands;
                Fold::formed(parts)
            }
        };
        done.insert(self.address(), out.clone());
        Ok(out)
    }

    /// This term, a block of a partition, walked from zero: the axis takes the block's length in `table`, the lift
    /// and the operands read it as `start + axis`, and the range is what the term's own bounds leave of the block.
    fn rebased(
        &self,
        axis: Name,
        walk: &Walk,
        operands: Vec<Fold>,
        table: &mut [Arc<Axis>],
    ) -> Result<Fold> {
        let (Some(base), Some(trips)) = (&walk.base, &walk.trips) else {
            return Err(Error::Value(
                "a block of a partition has a start and a length".into(),
            ));
        };
        if let Some(slot) = table.iter_mut().find(|known| known.name == axis) {
            *slot = Arc::new(Axis {
                name: axis,
                extent: trips.clone(),
                window: slot.window.clone(),
            });
        }
        let reads = TermRewrite {
            sigma: HashMap::from([(axis, walk.read())]),
            ..TermRewrite::default()
        };
        // An operand that does not read the axis is the term it was, and one reached twice is rewritten once:
        // which term stands at an operand position is how a tree says a value is shared.
        let mut moved: HashMap<usize, Fold> = HashMap::new();
        let operands = operands
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
        let lift = self.lift();
        let sigma = HashMap::from([(axis, walk.position.clone())]);
        let body: Vec<Arc<Statement>> = lift
            .body()
            .stmts()
            .iter()
            .map(|stmt| Statement::substitute(stmt, &sigma))
            .collect();
        // Inside the block on both sides: a bound past the block's end, or ahead of its start, cuts nothing.
        let least = |a, b| Expression::apply(Operation::Minimum, vec![a, b]);
        let greatest = |a, b| Expression::apply(Operation::Maximum, vec![a, b]);
        let from: Vec<Expression> = walk
            .starts
            .iter()
            .map(|start| room(base, start))
            .collect::<Result<_>>()?;
        let to: Vec<Expression> = walk
            .ends
            .iter()
            .map(|end| room(base, end))
            .collect::<Result<_>>()?;
        let mut parts = self.parts();
        parts.operands = operands;
        parts.lift = Some(Lambda::closing(
            lift.params()[..1 + self.bindings().len()].to_vec(),
            &Body::new(body),
            lift.results().to_vec(),
        )?);
        parts.range = (!from.is_empty() || !to.is_empty()).then(|| {
            (
                from.into_iter().fold(Expression::int(0), greatest),
                to.into_iter().fold(trips.expression.clone(), least),
            )
        });
        Fold::new(parts)
    }
}
