//! Axes, their extents and their windows, as the statements hold them. The compiler's own `Axis`, `Dim` and
//! `Window` classes live in Python; a statement built from one keeps what a loop header spells of it.

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::expr::{Expr, Expression, Num, spell_name};
use crate::{Name, Result};

/// Words the reader gives a meaning, so a name spelled like one is written between backticks.
pub const KEYWORDS: [&str; 25] = [
    "for", "in", "if", "else", "let", "init", "zero", "load", "pre", "select", "when", "over",
    "seed", "step", "until", "window", "carried", "swizzle", "words", "unroll", "sub", "call",
    "true", "false", "none",
];

/// A name as a statement's text spells it: bare, or between backticks when it reads as a keyword or is no identifier.
pub fn spell_stmt_name(name: &str) -> Result<String> {
    spell_name(name, &KEYWORDS)
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
    pub fn renamed(&self, rename: &mut dyn FnMut(&str) -> Option<Name>) -> Arc<Axis> {
        fn tree(axis: &Axis, rename: &mut dyn FnMut(&str) -> Option<Name>, depth: usize) -> Axis {
            let window = axis.window.as_ref().map(|window| {
                let parent = window.parent.as_ref().map(|parent| {
                    if depth > 32 {
                        Arc::new(Axis {
                            name: rename(&parent.name).unwrap_or_else(|| parent.name),
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
                name: rename(&axis.name).unwrap_or(axis.name),
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

/// `name in start..extent`, with the hint when it is not the default, and the window.
pub fn spell_axis(axis: &Axis, start: Option<&str>) -> Result<String> {
    let mut extent = axis.extent.expression.try_text()?;
    if axis.extent.hint != axis.extent.default_hint()
        && let Some(hint) = axis.extent.hint
    {
        extent.push_str(&format!("~{hint}"));
    }
    let mut out = format!(
        "{} in {}..{}",
        spell_stmt_name(&axis.name)?,
        start.unwrap_or("0"),
        extent
    );
    out.push_str(&spell_window(axis.window.as_ref())?);
    Ok(out)
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
