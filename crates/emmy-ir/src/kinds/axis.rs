//! Axes, their extents and their windows, as the statements hold them. The compiler's own `Axis`, `Dim` and
//! `Window` classes live in Python; a statement built from one keeps what a loop header spells of it.

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::Name;
use crate::expr::{Expr, Expression, Num};

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
                            name: rename(&parent.name).unwrap_or_else(|| parent.name.clone()),
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
                name: rename(&axis.name).unwrap_or_else(|| axis.name.clone()),
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
