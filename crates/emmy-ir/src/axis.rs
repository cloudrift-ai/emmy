//! Axes, their extents and their windows, as the statements hold them. The compiler's own `Axis`, `Dim` and
//! `Window` classes live in Python; a statement built from one keeps what a loop header spells of it.

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::elementwise::Operation;
use crate::expr::{Expr, Expression, Interval, Linear, Num, SimplifyCtx, spell_name};
use crate::{Name, Result};

/// A name as a statement's text spells it: bare, or between backticks when it reads as a keyword or is no identifier.
pub fn spell_stmt_name(name: &str) -> Result<String> {
    spell_name(name)
}

/// The expected size of a symbolic dim when none is supplied.
pub const DEFAULT_SEQ_HINT: i64 = 512;

/// One extent: a shape expression and, for a symbolic one, the size the planner pretends it has. The hint is no part
/// of equality.
#[derive(Clone, Debug)]
pub struct Dim {
    pub expression: Expression,
    pub hint: Option<i64>,
}

impl Dim {
    /// A dim as Python's constructor builds it: an atomic symbolic dim with no hint takes the default.
    pub fn new(expression: Expression, hint: Option<i64>) -> Dim {
        let hint = match hint {
            None if matches!(*expression, Expr::Var(_)) => Some(DEFAULT_SEQ_HINT),
            other => other,
        };
        Dim { expression, hint }
    }

    pub fn static_(value: i64) -> Dim {
        Dim {
            expression: Expression::int(value),
            hint: None,
        }
    }

    pub fn is_static(&self) -> bool {
        self.as_static().is_some()
    }

    pub fn as_static(&self) -> Option<i64> {
        match &*self.expression {
            Expr::Literal(lit) => match lit.value {
                Num::Int(v) => Some(v),
                _ => None,
            },
            _ => None,
        }
    }

    /// The hint the default rule gives this expression: what the text leaves unwritten.
    pub fn default_hint(&self) -> Option<i64> {
        matches!(*self.expression, Expr::Var(_)).then_some(DEFAULT_SEQ_HINT)
    }
}

impl PartialEq for Dim {
    fn eq(&self, other: &Self) -> bool {
        self.expression == other.expression
    }
}

impl Eq for Dim {}

impl Hash for Dim {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.expression.hash(state);
    }
}

/// The slice of a parent axis an axis walks: its provenance and whether a cross-CTA split produced it.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct Window {
    pub parent: Option<Arc<Axis>>,
    pub base: Option<Expression>,
    pub bound: Option<Expression>,
    pub partition: bool,
}

/// One named iteration variable. Equal to another by name and extent: the window is provenance.
#[derive(Clone, Debug)]
pub struct Axis {
    pub name: Name,
    pub extent: Dim,
    pub window: Option<Window>,
}

impl Axis {
    pub fn new(name: &str, extent: Dim, window: Option<Window>) -> Arc<Axis> {
        Arc::new(Axis {
            name: name.into(),
            extent,
            window,
        })
    }

    pub fn source_axis(&self) -> Option<&Arc<Axis>> {
        self.window.as_ref().and_then(|w| w.parent.as_ref())
    }

    /// The source-axis chain, nearest first; a cycle ends it.
    pub fn sources(&self) -> Vec<Arc<Axis>> {
        let mut out: Vec<Arc<Axis>> = Vec::new();
        let mut parent = self.source_axis().cloned();
        while let Some(axis) = parent {
            if out.iter().any(|seen| Arc::ptr_eq(seen, &axis)) {
                break;
            }
            parent = axis.source_axis().cloned();
            out.push(axis);
        }
        out
    }

    /// The static extent, or the hint when symbolic.
    pub fn hint_extent(&self) -> i64 {
        self.extent
            .as_static()
            .or(self.extent.hint)
            .unwrap_or(DEFAULT_SEQ_HINT)
    }

    /// This axis with its name and every name its window spells mapped through `rename`.
    pub fn renamed(&self, rename: &mut dyn FnMut(Name) -> Option<Name>) -> Arc<Axis> {
        fn tree(axis: &Axis, rename: &mut dyn FnMut(Name) -> Option<Name>, depth: usize) -> Axis {
            let window = axis.window.as_ref().map(|window| {
                let parent = window.parent.as_ref().map(|parent| {
                    if depth > 32 {
                        Arc::new(Axis {
                            name: rename(parent.name).unwrap_or(parent.name),
                            ..(**parent).clone()
                        })
                    } else {
                        Arc::new(tree(parent, rename, depth + 1))
                    }
                });
                Window {
                    parent,
                    base: window.base.as_ref().map(|e| e.rename(rename)),
                    bound: window.bound.as_ref().map(|e| e.rename(rename)),
                    partition: window.partition,
                }
            });
            Axis {
                name: rename(axis.name).unwrap_or(axis.name),
                extent: axis.extent.clone(),
                window,
            }
        }
        Arc::new(tree(self, rename, 0))
    }
}

impl PartialEq for Axis {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.extent == other.extent
    }
}

impl Eq for Axis {}

impl Hash for Axis {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.extent.hash(state);
    }
}

impl Axis {
    /// Equal with the hint and the window too, through every parent: what a memo keys on, since both are identity
    /// material though neither is part of equality.
    pub fn exact_eq(&self, other: &Axis) -> bool {
        self == other
            && self.extent.hint == other.extent.hint
            && match (&self.window, &other.window) {
                (None, None) => true,
                (Some(a), Some(b)) => {
                    a.base == b.base
                        && a.bound == b.bound
                        && a.partition == b.partition
                        && match (&a.parent, &b.parent) {
                            (None, None) => true,
                            (Some(p), Some(q)) => p.exact_eq(q),
                            _ => false,
                        }
                }
                _ => false,
            }
    }

    /// The hash `exact_eq` agrees with.
    pub fn exact_hash<H: Hasher>(&self, state: &mut H) {
        self.hash(state);
        self.extent.hint.hash(state);
        match &self.window {
            None => false.hash(state),
            Some(window) => {
                true.hash(state);
                window.base.hash(state);
                window.bound.hash(state);
                window.partition.hash(state);
                match &window.parent {
                    None => false.hash(state),
                    Some(parent) => {
                        true.hash(state);
                        parent.exact_hash(state);
                    }
                }
            }
        }
    }
}

/// The largest value a coordinate is taken to reach where nothing bounds it: a symbolic extent's stand-in.
pub const SYMBOLIC_AXIS_HI: i128 = 1 << 30;

/// Where an axis walks: from `start` up to `end`, exclusive, by one. Each is an index over the coordinates around
/// the loop and the symbolic dims, affine or a `min` / `max` of affine forms. It bounds one variable and is no
/// polyhedron: every IR that says "from where to where" says it with this type, and reads it through the few
/// questions below.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Range {
    pub start: Expression,
    pub end: Expression,
}

impl Range {
    /// The range of an axis that states none: `0..extent`.
    pub fn whole(axis: &Axis) -> Range {
        Range {
            start: Expression::int(0),
            end: axis.extent.expression.clone(),
        }
    }

    /// The trip count where the range fixes it: `end - start` when that is a literal. An empty range has none.
    pub fn trips(&self) -> Option<i64> {
        let trips = Expression::binary(Operation::Subtract, self.end.clone(), self.start.clone())
            .simplify(&SimplifyCtx::empty())
            .ok()?
            .int_literal()?;
        (trips > 0).then_some(trips)
    }

    /// The comparisons the range states of `axis`, each an affine form that is at least zero: `start <= axis` and
    /// `axis < end`. A `max` in the start and a `min` in the end state one per operand; a bound that is not affine
    /// states nothing.
    pub fn relations(&self, axis: Name) -> Vec<Linear> {
        fn operands(e: &Expression, op: Operation, out: &mut Vec<Expression>) {
            match e.operation() {
                Some((found, [a, b])) if found == op => {
                    operands(a, op, out);
                    operands(b, op, out);
                }
                _ => out.push(e.clone()),
            }
        }
        let axis = Expression::var(&axis);
        let (mut lower, mut upper) = (Vec::new(), Vec::new());
        operands(&self.start, Operation::Maximum, &mut lower);
        operands(&self.end, Operation::Minimum, &mut upper);
        let one = Expression::int(1);
        lower
            .iter()
            .filter_map(|bound| Linear::at_least(&axis, bound))
            .chain(upper.iter().filter_map(|bound| {
                let last = Expression::binary(Operation::Subtract, bound.clone(), one.clone());
                Linear::at_least(&last, &axis)
            }))
            .collect()
    }

    /// The interval the axis stays in under `ctx`: from the least value the start takes to one below the most the
    /// end takes, each side open (the stand-in for a symbolic extent) where the analysis cannot bound it.
    pub fn interval(&self, ctx: &SimplifyCtx) -> Interval {
        /// The bound of `e` on one side: its least value (`low`) or its most. A `max` is at least each operand and a
        /// `min` at most each, so one bounded operand is enough there; the other way round needs both.
        fn side(e: &Expression, ctx: &SimplifyCtx, low: bool) -> Option<i128> {
            let pick = |range: Interval| if low { range.lo } else { range.hi };
            match e.operation() {
                Some((op @ (Operation::Minimum | Operation::Maximum), [a, b])) => {
                    let (a, b) = (side(a, ctx, low), side(b, ctx, low));
                    let one_is_enough = (op == Operation::Maximum) == low;
                    match (a, b) {
                        (Some(a), Some(b)) if op == Operation::Maximum => Some(a.max(b)),
                        (Some(a), Some(b)) => Some(a.min(b)),
                        (Some(only), None) | (None, Some(only)) if one_is_enough => Some(only),
                        _ => None,
                    }
                }
                _ => e.range(ctx).map(pick),
            }
        }
        Interval::new(
            side(&self.start, ctx, true).unwrap_or(-SYMBOLIC_AXIS_HI),
            side(&self.end, ctx, false).map_or(SYMBOLIC_AXIS_HI, |end| end - 1),
        )
    }

    /// `ctx` inside a loop that walks `axis` over this range: the axis's interval, the comparisons the range
    /// states, and the end as the axis's symbolic bound where it is one expression.
    pub fn extend(&self, ctx: &SimplifyCtx, axis: Name) -> SimplifyCtx {
        let clamps = matches!(
            self.end.operation(),
            Some((Operation::Minimum | Operation::Maximum, _))
        );
        let mut out = ctx.extend(
            &axis,
            self.interval(ctx),
            (!clamps).then(|| self.end.clone()),
        );
        if clamps {
            out.bounds.remove(&axis);
        }
        out.relations.extend(self.relations(axis));
        out
    }
}

/// An extent as a loop header spells it: the expression, with the hint when it is not the default.
pub fn spell_extent(extent: &Dim) -> Result<String> {
    let mut out = extent.expression.try_text()?;
    if extent.hint != extent.default_hint()
        && let Some(hint) = extent.hint
    {
        out.push_str(&format!("~{hint}"));
    }
    Ok(out)
}

/// `name in start..extent`, with the hint when it is not the default, and the window.
pub fn spell_axis(axis: &Axis, start: Option<&str>) -> Result<String> {
    let mut out = format!(
        "{} in {}..{}",
        spell_stmt_name(&axis.name)?,
        start.unwrap_or("0"),
        spell_extent(&axis.extent)?
    );
    out.push_str(&spell_window(axis.window.as_ref())?);
    Ok(out)
}

/// `name in start..end of extent`, the header of a loop that walks a range: the extent is the most trips it takes.
pub fn spell_ranged_axis(axis: &Axis, range: &Range) -> Result<String> {
    Ok(format!(
        "{} in {}..{} of {}{}",
        spell_stmt_name(&axis.name)?,
        range.start.try_text()?,
        range.end.try_text()?,
        spell_extent(&axis.extent)?,
        spell_window(axis.window.as_ref())?
    ))
}

pub fn spell_window(window: Option<&Window>) -> Result<String> {
    let Some(window) = window else {
        return Ok(String::new());
    };
    let mut parts: Vec<String> = Vec::new();
    if let Some(parent) = &window.parent {
        parts.push(format!("parent={}", spell_axis(parent, None)?));
    }
    if let Some(base) = &window.base {
        parts.push(format!("base={}", base.try_text()?));
    }
    if let Some(bound) = &window.bound {
        parts.push(format!("bound={}", bound.try_text()?));
    }
    if window.partition {
        parts.push("partition=true".into());
    }
    Ok(format!(" window({})", parts.join(", ")))
}
