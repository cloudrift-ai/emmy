//! `Fold::rewrite`: a term renamed, substituted and axis-mapped, the way `Statement::rewrite` rewrites a statement.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use emmy_ir::body::Body;
use emmy_ir::expr::{Expression, SimplifyCtx};
use emmy_ir::field::AxisRef;
use emmy_ir::protocol::Rewrite;
use emmy_ir::statement::{Roles, Statement};
use emmy_ir::{Name, Result};

use crate::fold::Fold;
use crate::lambda::Lambda;

/// What a rewrite does to a term: every SSA name through `names` (a name it does not hold stays), every expression
/// through `sigma`, folded under `folding` once the substitution has landed when one is given, and every axis a
/// statement binds through the `axes` pairs.
#[derive(Default)]
pub struct TermRewrite {
    pub names: HashMap<Name, Name>,
    pub sigma: HashMap<Name, Expression>,
    pub folding: Option<SimplifyCtx>,
    pub axes: Vec<(AxisRef, AxisRef)>,
}

/// `stmt` with every expression of its subtree folded under `ctx`.
fn folded(stmt: &Arc<Statement>, ctx: &SimplifyCtx) -> Result<Arc<Statement>> {
    let expressions = stmt
        .expressions()
        .iter()
        .map(|expression| expression.simplify(ctx))
        .collect::<Result<Vec<_>>>()?;
    let stmt = Statement::with_expressions(stmt, expressions);
    let nested = stmt.nested();
    if nested.is_empty() {
        return Ok(stmt);
    }
    let bodies = nested
        .iter()
        .map(|body| body.try_map_each(&mut |child| folded(child, ctx)))
        .collect::<Result<Vec<_>>>()?;
    Statement::with_bodies(&stmt, bodies)
}

impl TermRewrite {
    fn name(&self, name: Name) -> Name {
        *self.names.get(&name).unwrap_or(&name)
    }

    /// A coordinate param: an axis name renames through the substitution, as every use of it in the body does.
    fn coordinate(&self, sigma: &HashMap<Name, Expression>, name: Name) -> Name {
        match sigma.get(&name).and_then(|mapped| mapped.as_var()) {
            Some(var) => *var,
            None => self.name(name),
        }
    }

    fn body(&self, sigma: &HashMap<Name, Expression>, body: &Body) -> Result<Body> {
        let rename = |_: Roles, name: Name| self.names.get(&name).copied();
        let axis_fn = |axis: &AxisRef| {
            self.axes
                .iter()
                .find(|(old, _)| **old == **axis && old.window == axis.window)
                .map_or_else(|| axis.clone(), |(_, new)| new.clone())
        };
        let rewrite = Rewrite {
            rename: &rename,
            sigma,
            axis_fn: if self.axes.is_empty() {
                None
            } else {
                Some(&axis_fn)
            },
        };
        body.try_map_each(&mut |stmt| {
            let rewritten = Statement::rewrite(stmt, &rewrite);
            match &self.folding {
                Some(ctx) => folded(&rewritten, ctx),
                None => Ok(rewritten),
            }
        })
    }

    fn term(
        &self,
        fold: &Fold,
        sigma: &HashMap<Name, Expression>,
        done: &mut HashMap<(usize, Vec<Name>), Fold>,
    ) -> Result<Fold> {
        // The reduce axis this term binds is another variable than a substitution key spelled alike: its mapping is
        // dropped for the subtree.
        let restricted: HashMap<Name, Expression>;
        let sigma = match fold.axis() {
            Some(axis) if sigma.contains_key(&axis) => {
                restricted = sigma
                    .iter()
                    .filter(|(name, _)| **name != axis)
                    .map(|(name, value)| (*name, value.clone()))
                    .collect();
                &restricted
            }
            _ => sigma,
        };
        let mut live: Vec<Name> = sigma.keys().copied().collect();
        live.sort();
        let key = (fold.address(), live);
        if let Some(found) = done.get(&key) {
            return Ok(found.clone());
        }
        let operands = fold
            .operands()
            .iter()
            .map(|edge| self.term(edge, sigma, done))
            .collect::<Result<Vec<_>>>()?;
        let lift = fold.lift();
        let lead = usize::from(fold.axis().is_some());
        let bound = lead + fold.bindings().len();
        let trailing = lift.params().get(bound..).unwrap_or(&[]);
        // A coordinate substituted by an expression leaves the param list, unless the expression still reads it.
        let substituted: HashSet<Name> = trailing
            .iter()
            .filter(|p| sigma.get(p).is_some_and(|mapped| mapped.as_var().is_none()))
            .copied()
            .collect();
        let needed: HashSet<Name> = substituted
            .iter()
            .flat_map(|p| sigma[p].free_vars())
            .collect();
        let mut params: Vec<Name> = lift.params()[..lead]
            .iter()
            .map(|p| self.name(*p))
            .collect();
        params.extend(
            lift.params()[lead..bound.min(lift.params().len())]
                .iter()
                .map(|p| self.coordinate(sigma, *p)),
        );
        params.extend(
            trailing
                .iter()
                .filter(|p| !substituted.contains(p) || needed.contains(p))
                .map(|p| self.coordinate(sigma, *p)),
        );
        let body = self.body(sigma, lift.body())?;
        let results: Vec<Name> = lift.results().iter().map(|r| self.name(*r)).collect();
        // Only an expression substitution can bring in a coordinate the tail did not hold.
        let lift = if substituted.is_empty() {
            Lambda::new(params, &body, results)?
        } else {
            Lambda::closing(params, &body, results)?
        };
        let mut parts = fold.parts();
        parts.operands = operands;
        parts.base = fold
            .base()
            .map(|base| base.rename(&self.names))
            .transpose()?;
        // The observer renames in lockstep: its first param tracks the axis, the rest the carried state.
        parts.observe = fold
            .observe()
            .map(|observe| {
                Lambda::new(
                    observe.params().iter().map(|p| self.name(*p)).collect(),
                    &self.body(sigma, observe.body())?,
                    observe.results().iter().map(|r| self.name(*r)).collect(),
                )
            })
            .transpose()?;
        parts.lift = Some(lift);
        // The range reads the coordinates around the term: it renames and substitutes as an index of the body does.
        parts.range = fold
            .range()
            .map(|(start, end)| {
                let rewritten = |e: &Expression| {
                    let moved = e
                        .substitute(sigma)
                        .rename(&mut |name| self.names.get(&name).copied());
                    match &self.folding {
                        Some(ctx) => moved.simplify(ctx),
                        None => Ok(moved),
                    }
                };
                Ok::<_, emmy_ir::Error>((rewritten(start)?, rewritten(end)?))
            })
            .transpose()?;
        let rewritten = Fold::new(parts)?;
        done.insert(key, rewritten.clone());
        Ok(rewritten)
    }
}

impl Fold {
    /// This term rewritten: the lift and the monoid rename in lockstep, the substitution is hygienic about the axis
    /// each term binds, and operands rewrite through the same entry.
    pub fn rewrite(&self, rewrite: &TermRewrite) -> Result<Fold> {
        rewrite.term(self, &rewrite.sigma, &mut HashMap::new())
    }

    /// Every SSA and axis name this tree spells: what a rename by a function is asked about.
    pub fn names(&self) -> Vec<Name> {
        let mut names: Vec<Name> = Vec::new();
        let mut seen: HashSet<usize> = HashSet::new();
        let mut pending = vec![self.clone()];
        while let Some(term) = pending.pop() {
            if !seen.insert(term.address()) {
                continue;
            }
            for lambda in [Some(term.lift()), term.base(), term.observe()]
                .into_iter()
                .flatten()
            {
                names.extend(lambda.params());
                names.extend(lambda.results());
                for stmt in lambda.body().stmts() {
                    names.extend(stmt.names());
                }
            }
            pending.extend(term.operands().iter().cloned());
        }
        let mut kept = HashSet::new();
        names.retain(|name| kept.insert(*name));
        names
    }

    /// Every axis a statement of this tree binds: what an axis map by a function is asked about.
    pub fn bound_axes(&self) -> Vec<AxisRef> {
        let mut axes: Vec<AxisRef> = Vec::new();
        let mut seen: HashSet<usize> = HashSet::new();
        let mut pending = vec![self.clone()];
        while let Some(term) = pending.pop() {
            if !seen.insert(term.address()) {
                continue;
            }
            for lambda in [Some(term.lift()), term.observe()].into_iter().flatten() {
                for here in lambda.body().iter_refs() {
                    axes.extend(here.stmt().bound_axes());
                }
            }
            pending.extend(term.operands().iter().cloned());
        }
        axes
    }
}
