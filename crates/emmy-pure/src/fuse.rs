//! `Fold::fuse`: a dependent reduce fused onto the reduce it reads, by a recipe. The ONE generic algorithm over the
//! declarative recipes.
//!
//! The pivot is found among the term's operands: a reduce folding the recipe's pivot ⊕, or a fold this recipe already
//! fused. The pivot's state is the lift param bound to it; the score is the sub-cone of the lift alpha-equal to the
//! pivot's own per-element map, operand for operand; and what remains of the lift with that cone cut out, its params
//! in role order, must equal a channel's pattern by canonical form. A click gives the role-to-name map, and the fused
//! fold is the recipe instantiated by renaming. Which recipe to try, and where, is the pass's decision.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use emmy_ir::axis::Axis;
use emmy_ir::body::Body;
use emmy_ir::elementwise::ElementwiseOp;
use emmy_ir::expr::{Expr, Expression};
use emmy_ir::identity::canonicalize_identity;
use emmy_ir::protocol::Rewrite;
use emmy_ir::statement::{Roles, Statement};
use emmy_ir::{Error, Name, Result};

use crate::fold::{Fold, Parts, Seed};
use crate::lambda::Lambda;
use crate::readings::reduce_canon;
use crate::rewrite::TermRewrite;
use crate::twist::{Recipe, Twist};

/// The closed cone defining one name of a lift, and the value each of its params binds.
struct Cone {
    lambda: Lambda,
    values: Vec<Value>,
}

/// What a cone param binds: a coordinate, a reduce's state (that term), or a projection's result (its own cone).
enum Value {
    Coordinate,
    Term(Fold),
    Cone(Cone),
}

/// The kernel's axes by name, and the symbolic extents the launch binds: what a comparison of two values lowers
/// them under.
pub struct Scope<'a> {
    pub axes: &'a [Arc<Axis>],
    pub extents: &'a HashSet<Name>,
}

impl Scope<'_> {
    fn axis(&self, name: Name) -> Result<&Arc<Axis>> {
        self.axes
            .iter()
            .rev()
            .find(|axis| axis.name == name)
            .ok_or_else(|| Error::Key(format!("'{name}'")))
    }
}

fn cone(fold: &Fold, name: Name) -> Result<Cone> {
    // A param this term only passes through is the edge's own cone: the value is the same whether the statements sit
    // in the lift or in the operand a factoring moved them to.
    for binding in fold.bindings() {
        let edge = &fold.operands()[binding.operand];
        if binding.param == name && edge.axis().is_none() {
            return cone(edge, edge.exposes()[binding.component]);
        }
    }
    let lambda = fold.lift().cone(name)?;
    let values = lambda
        .params()
        .iter()
        .map(|param| {
            let bound = fold.bindings().iter().rfind(|b| b.param == *param);
            Ok(match bound.map(|b| &fold.operands()[b.operand]) {
                None => Value::Coordinate,
                Some(edge) if edge.axis().is_some() => Value::Term(edge.clone()),
                Some(edge) => Value::Cone(cone(edge, *param)?),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Cone { lambda, values })
}

fn renamed(stmt: &Arc<Statement>, names: &HashMap<Name, Name>) -> Arc<Statement> {
    let rename = |_: Roles, name: Name| names.get(&name).copied();
    let rewrite = Rewrite {
        rename: &rename,
        sigma: &HashMap::new(),
        axis_fn: None,
    };
    Statement::rewrite(stmt, &rewrite)
}

fn assigned(stmt: &Statement) -> Option<(Name, &Expression)> {
    match stmt {
        Statement::Assign(assign) => Some((assign.name, &assign.value)),
        _ => None,
    }
}

/// `params` cut into one group per edge: the positional binding, one param per result component.
fn slots(params: &[Name], edges: &[Fold]) -> Vec<Vec<Name>> {
    let mut cursor = 0;
    edges
        .iter()
        .map(|edge| {
            let width = edge.exposes().len();
            let slot = params[cursor..cursor + width].to_vec();
            cursor += width;
            slot
        })
        .collect()
}

struct Fusing<'a> {
    this: &'a Fold,
    pivot: &'a Fold,
    scope: &'a Scope<'a>,
}

impl Fusing<'_> {
    /// Two terms compared as values under their actual coordinate scopes, whatever order a lift takes its coordinates
    /// in and whatever the axes its operands bind are called.
    fn alike(&self, x: &Fold, y: &Fold) -> Result<bool> {
        let (old, new) = (self.this.axis(), self.pivot.axis());
        let x = match (old, new) {
            (Some(old), Some(new)) if old != new => x.rewrite(&TermRewrite {
                sigma: HashMap::from([(old, Expression::new(Expr::Var(new)))]),
                ..TermRewrite::default()
            })?,
            _ => x.clone(),
        };
        let identity = |term: &Fold| {
            let body = term.lower(None, &[], self.scope.axes, self.scope.extents)?;
            canonicalize_identity(&body, false, None)
        };
        Ok(identity(&x)? == identity(y)?)
    }

    fn same(&self, a: &Cone, b: &Cone) -> Result<bool> {
        if a.lambda.canonical()? != b.lambda.canonical()? || a.values.len() != b.values.len() {
            return Ok(false);
        }
        for pair in a.values.iter().zip(&b.values) {
            let equal = match pair {
                (Value::Coordinate, Value::Coordinate) => true,
                (Value::Term(x), Value::Term(y)) => self.alike(x, y)?,
                (Value::Cone(x), Value::Cone(y)) => self.same(x, y)?,
                _ => false,
            };
            if !equal {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

impl Fold {
    /// This reduce fused onto the reduce it reads, by `recipe`: one fold carrying both states under the recipe's
    /// twisted ⊕, or `None` when no channel of the recipe clicks.
    pub fn fuse(&self, recipe: &Recipe, scope: &Scope) -> Result<Option<Fold>> {
        let Some(view) = self.as_reduction()? else {
            return Ok(None);
        };
        if self.observe().is_some() || view.ops.is_none() || view.states.len() != 1 {
            return Ok(None);
        }
        for pivot in self.operands() {
            let Some(pview) = pivot.as_reduction()? else {
                continue;
            };
            if pivot.observe().is_some() {
                continue;
            }
            match &pview.ops {
                Some(ops) => {
                    if pview.states.len() != 1
                        || reduce_canon(ops[0].name()) != recipe.pivot().name()
                    {
                        continue;
                    }
                }
                // A twisted pivot must already carry THIS recipe: another one's carrier is not its pivot.
                None => {
                    if pivot.twist().is_none_or(|twist| twist.recipe != *recipe) {
                        continue;
                    }
                }
            }
            let (mine, theirs) = (scope.axis(view.axis)?, scope.axis(pview.axis)?);
            if mine.extent != theirs.extent || mine.window != theirs.window {
                continue;
            }
            // The carrier walks one axis once: two folds fuse where they walk the same part of it. A range reads
            // the coordinates around its fold, never the axis, so the two spell it alike.
            if self.range() != pivot.range() {
                continue;
            }
            if let Some(fused) = self.twisted(pivot, recipe, scope)? {
                return Ok(Some(fused));
            }
        }
        Ok(None)
    }

    fn twisted(&self, pivot: &Fold, recipe: &Recipe, scope: &Scope) -> Result<Option<Fold>> {
        let missing = || Error::Value("a fused pair is two reduces".into());
        let view = self.as_reduction()?.ok_or_else(missing)?;
        let pview = pivot.as_reduction()?.ok_or_else(missing)?;
        let (old, new) = (view.axis, pview.axis);
        let fusing = Fusing {
            this: self,
            pivot,
            scope,
        };
        let lift = self.lift();
        let bound: Vec<(Name, &Fold)> = self
            .bindings()
            .iter()
            .map(|b| (b.param, &self.operands()[b.operand]))
            .collect();
        let edge_of = |param: &Name| bound.iter().rfind(|(p, _)| p == param).map(|(_, e)| *e);
        let pivot_params: Vec<Name> = bound
            .iter()
            .filter(|(_, edge)| edge.ptr_eq(pivot))
            .map(|(p, _)| *p)
            .collect();
        let g = pivot_params[0];
        let score = cone(pivot, pivot.lift().results()[0])?;
        let candidates: Vec<Name> = lift.params()[1..]
            .iter()
            .filter(|p| !pivot_params.contains(p))
            .copied()
            .chain(lift.body().stmts().iter().flat_map(|stmt| stmt.defines()))
            .collect();
        for x in candidates {
            // The cone may read coordinates free: this fold's own axis is the pivot's once fused, so it is compared
            // under the pivot's name, while the residual below is cut by the cone's own statements.
            let Cone {
                lambda: found,
                values,
            } = cone(self, x)?;
            let spelled = Cone {
                lambda: if old == new {
                    found.clone()
                } else {
                    found.rename(&HashMap::from([(old, new)]))?
                },
                values,
            };
            if !fusing.same(&spelled, &score)? {
                continue;
            }
            // What remains of the lift with the score cut out, closed over what it still reads: `(score, pivot,
            // *extras)` in role order, the extras operand-bound values.
            let cut = found.body().stmts();
            let residual: Vec<Arc<Statement>> = lift
                .body()
                .stmts()
                .iter()
                .filter(|stmt| !cut.contains(stmt))
                .cloned()
                .collect();
            let rest = Lambda::closing(vec![x, g], &Body::new(residual), lift.results().to_vec())?;
            let extras = &rest.params()[2..];
            if extras
                .iter()
                .any(|p| edge_of(p).is_none() || pivot_params.contains(p))
            {
                continue;
            }
            // A channel whose base ⊕ is this fold's and whose pattern is what remains.
            let plus = reduce_canon(view.ops.as_ref().ok_or_else(missing)?[0].name());
            let mut matched = None;
            for (index, channel) in recipe.channels.iter().enumerate() {
                if let Some(pattern) = &channel.pattern
                    && recipe.base[1 + index].name() == plus
                    && rest.canonical()? == pattern.canonical()?
                {
                    matched = Some((index, pattern));
                    break;
                }
            }
            let Some((matched, pattern)) = matched else {
                continue;
            };
            // Instantiate: every operand an extra binds joins, its axis spelled as the pivot's.
            let mut extra_edges: Vec<Fold> = Vec::new();
            for (param, edge) in &bound {
                if extras.contains(param) && !extra_edges.contains(*edge) {
                    extra_edges.push((*edge).clone());
                }
            }
            let extra_params: Vec<Name> = extra_edges
                .iter()
                .flat_map(|edge| {
                    bound
                        .iter()
                        .filter(move |(_, e)| e.ptr_eq(edge))
                        .map(|(p, _)| *p)
                })
                .collect();
            if old != new {
                for edge in &mut extra_edges {
                    let axes = edge
                        .bound_axes()
                        .into_iter()
                        .filter(|axis| axis.name == old)
                        .map(|axis| {
                            let spelled = Arc::new(Axis {
                                name: new,
                                ..(*axis).clone()
                            });
                            (axis, spelled)
                        })
                        .collect();
                    *edge = edge.rewrite(&TermRewrite {
                        sigma: HashMap::from([(old, Expression::new(Expr::Var(new)))]),
                        axes,
                        ..TermRewrite::default()
                    })?;
                }
            }
            // The carrier's states: the pivot's, then every channel in recipe order — the matched one is this fold's
            // own state, one without a pattern a state the recipe adds. The contribution stored is the recipe's
            // authored INJECTION: the singleton in the carrier's own state space. Seeds are the carrier's.
            let state = view.states[0];
            let score_name = pivot.lift().results()[0];
            let mut added: Vec<(Name, usize, Seed)> = recipe
                .channels
                .iter()
                .enumerate()
                .filter(|(index, channel)| *index == matched || channel.pattern.is_none())
                .map(|(index, channel)| {
                    if index == matched {
                        (state, index, self.init()[0].clone())
                    } else {
                        (
                            Name::new(&format!("{state}__{}", channel.name)),
                            index,
                            Seed::Float(channel.init),
                        )
                    }
                })
                .collect();
            let mut roles: HashMap<Name, Name> =
                HashMap::from([(recipe.lift.params()[0], score_name)]);
            if pattern.params().len() - 2 != extras.len() {
                return Err(Error::Value(
                    "a channel's pattern binds one role per extra".into(),
                ));
            }
            roles.extend(
                pattern.params()[2..]
                    .iter()
                    .copied()
                    .zip(extras.iter().copied()),
            );
            let arity: usize = pivot.operands().iter().map(|e| e.exposes().len()).sum();
            let mut held: Vec<(Fold, Vec<Name>)> = pivot
                .operands()
                .iter()
                .cloned()
                .zip(slots(
                    &pivot.lift().params()[1..1 + arity],
                    pivot.operands(),
                ))
                .chain(
                    extra_edges
                        .iter()
                        .cloned()
                        .zip(slots(&extra_params, &extra_edges)),
                )
                .collect();
            let mut body: Vec<Arc<Statement>> = pivot.lift().body().stmts().to_vec();
            let mut results: Vec<Name> = pivot.lift().results().to_vec();
            let mut inits: Vec<Seed> = pivot.init().to_vec();
            for (name, index, init) in added.clone() {
                let mut taken: HashSet<Name> =
                    body.iter().flat_map(|stmt| stmt.defines()).collect();
                taken.extend(held.iter().flat_map(|(_, slot)| slot.iter().copied()));
                let injection = &recipe.channels[index].injection;
                let mut names: HashMap<Name, Name> = HashMap::new();
                for param in injection.params() {
                    let role = roles
                        .get(param)
                        .ok_or_else(|| Error::Key(format!("'{param}'")))?;
                    names.insert(*param, *role);
                }
                // Statement by statement, so a channel BINDS what the body already computes rather than spelling it
                // again: softmax's denominator and its expectation share the whole weight prefix.
                let mut have: HashMap<Expression, Name> = body
                    .iter()
                    .filter_map(|stmt| assigned(stmt))
                    .map(|(name, value)| (value.clone(), name))
                    .collect();
                for stmt in injection.body().stmts() {
                    let own = *stmt.name().ok_or_else(|| {
                        Error::Value("an injection statement defines one name".into())
                    })?;
                    let spelled = renamed(stmt, &names);
                    if let Some((_, value)) = assigned(&spelled)
                        && let Some(prior) = have.get(value)
                    {
                        names.insert(own, *prior);
                        continue;
                    }
                    let fresh = if taken.contains(&own) {
                        Name::new(&format!("{name}__{own}"))
                    } else {
                        own
                    };
                    names.insert(own, fresh);
                    let spelled = renamed(stmt, &names);
                    if let Some((_, value)) = assigned(&spelled) {
                        have.insert(value.clone(), fresh);
                    }
                    body.push(spelled);
                    taken.insert(fresh);
                }
                if let Some(slot) = pview.states.iter().position(|s| *s == name) {
                    let mut bound_roles = roles.clone();
                    bound_roles.insert(recipe.lift.params()[0], x);
                    let spelling: HashMap<Name, Name> = injection
                        .names()
                        .into_iter()
                        .map(|read| {
                            let to = bound_roles
                                .get(&read)
                                .copied()
                                .unwrap_or_else(|| Name::new(&format!("{name}__{read}")));
                            (read, to)
                        })
                        .collect();
                    let injected = injection.rename(&spelling)?;
                    let mut stmts = found.body().stmts().to_vec();
                    stmts.extend(injected.body().stmts().iter().cloned());
                    let mut parts = self.parts();
                    parts.lift = Some(Lambda::closing(
                        lift.params().to_vec(),
                        &Body::new(stmts),
                        injected.results().to_vec(),
                    )?);
                    let mut contribution = cone(&Fold::new(parts)?, injected.results()[0])?;
                    if old != new {
                        contribution.lambda =
                            contribution.lambda.rename(&HashMap::from([(old, new)]))?;
                    }
                    // A shared dependent can reach the pivot again through another score cone. Reuse its state only
                    // when both copies inject the same value into the same channel.
                    let agrees = slot != 0
                        && pivot
                            .twist()
                            .is_some_and(|twist| twist.channels[slot - 1] == index)
                        && pivot.init()[slot] == init
                        && fusing
                            .same(&cone(pivot, pivot.lift().results()[slot])?, &contribution)?;
                    if !agrees {
                        return Err(Error::Value(format!(
                            "conflicting definitions of twisted state '{name}'"
                        )));
                    }
                    added.retain(|(kept, at, _)| !(*kept == name && *at == index));
                    continue;
                }
                results.push(names[&injection.results()[0]]);
                inits.push(init);
            }
            if added.is_empty() {
                return Ok(Some(pivot.clone()));
            }
            // An operand the lift no longer names is nobody's: the recipe's own contribution replaced the statements
            // that read it.
            let mut read: HashSet<Name> = body.iter().flat_map(|stmt| stmt.deps()).collect();
            read.extend(results.iter().copied());
            held.retain(|(_, slot)| slot.iter().any(|p| read.contains(p)));
            // B SECOND. A leads by canonical form; a bilinear channel's other factor must follow it, because staging
            // and both atom tiers take B from `operands[1]`. An operand with no free coordinates is no channel's
            // factor at all, so it sorts behind the streamed value.
            if held.is_empty() {
                return Err(Error::Value("a fused carrier reads no operand".into()));
            }
            held[1..].sort_by_key(|(edge, _)| edge.free_axes().is_empty());
            // The carrier is the RECIPE's own vector: the pivot, then the channels in the order the recipe declares
            // them, whatever order the tree happened to fuse them in.
            let channels: Vec<usize> = pivot
                .twist()
                .map(|twist| twist.channels.clone())
                .unwrap_or_default()
                .into_iter()
                .chain(added.iter().map(|(_, index, _)| *index))
                .collect();
            let mut order: Vec<usize> = (0..channels.len()).collect();
            order.sort_by_key(|slot| channels[*slot]);
            fn by_channel<T: Clone>(order: &[usize], all: &[T]) -> Vec<T> {
                std::iter::once(all[0].clone())
                    .chain(order.iter().map(|slot| all[1 + slot].clone()))
                    .collect()
            }
            let states: Vec<Name> = pview
                .states
                .iter()
                .copied()
                .chain(added.iter().map(|(name, _, _)| *name))
                .collect();
            let ops: Vec<ElementwiseOp> = pivot
                .ops()
                .ok_or_else(missing)?
                .into_iter()
                .chain(
                    added
                        .iter()
                        .map(|(_, index, _)| recipe.base[1 + index].clone()),
                )
                .collect();
            let channelled = by_channel(&order, &results);
            let mut wanted = channelled.clone();
            let mut kept = HashSet::new();
            wanted.retain(|name| kept.insert(*name));
            let members: Vec<Arc<Statement>> = Body::new(body)
                .backward_cone(&wanted)
                .0
                .iter()
                .map(|here| here.stmt().clone())
                .collect();
            let params: Vec<Name> = std::iter::once(new)
                .chain(held.iter().flat_map(|(_, slot)| slot.iter().copied()))
                .chain(pivot.lift().params()[1 + arity..].iter().copied())
                .collect();
            return Fold::new(Parts {
                operands: held.into_iter().map(|(edge, _)| edge).collect(),
                lift: Some(Lambda::new(params, &Body::new(members), channelled)?),
                init: by_channel(&order, &inits),
                base: Some(Lambda::componentwise(
                    &by_channel(&order, &ops),
                    &by_channel(&order, &states),
                )?),
                twist: Some(Twist {
                    recipe: recipe.clone(),
                    channels: order.iter().map(|slot| channels[*slot]).collect(),
                }),
                range: pivot.range().cloned(),
                ..Parts::default()
            })
            .map(Some);
        }
        Ok(None)
    }
}
