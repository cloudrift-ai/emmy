//! The derived readings of a [`Fold`]: its restriction to some results, the lift seen in base coordinates, the
//! recipe roles it spells, its bilinear channels and the contraction they read as, whether a tier folds it whole, the
//! term per carried state, its reduction and affine readings, and the statements its step and merge render to. Every
//! one is read off the stored term, never stored on it, and computed once per term.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};

use emmy_ir::body::Body;
use emmy_ir::elementwise::ElementwiseOp;
use emmy_ir::expr::{Expr, ExprClass, Expression, Num};
use emmy_ir::field::NumOr;
use emmy_ir::normalize::{AffineTerms, affine_sum, affine_terms, ties};
use emmy_ir::statement::{Accum, Assign, Carry, PURE, Statement};
use emmy_ir::{Error, Name, Result};

use crate::fold::{Fold, Parts, Seed};
use crate::lambda::Lambda;

/// The memo table of a term's readings: each filled the first time it is asked.
#[derive(Default)]
pub struct Memos {
    exposing: Mutex<HashMap<Vec<Name>, Fold>>,
    based: OnceLock<Lambda>,
    roles: OnceLock<Vec<Name>>,
    channels: OnceLock<Vec<ChannelOperand>>,
    contraction: OnceLock<Option<Contraction>>,
    chunked: OnceLock<bool>,
    per_state: OnceLock<Option<Fold>>,
    reduction: OnceLock<Option<Reduction>>,
    affine: OnceLock<Option<Affine>>,
    merge: Mutex<HashMap<Vec<Name>, Body>>,
    step: OnceLock<Body>,
}

/// One bilinear channel: the carried state it is, the operand it multiplies `operands[0]` by, and which result of
/// that operand the product reads.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelOperand {
    pub state: usize,
    pub operand: usize,
    pub name: Name,
}

/// A term's BILINEAR reading as geometry: the reduction its operand pair shares, the free axes each side carries
/// alone and the ones they share besides, the semiring instance, whether B is stored N-major, and which carried state
/// the products fold into.
#[derive(Clone, Debug, PartialEq)]
pub struct Contraction {
    pub axis: Name,
    pub left_axes: HashSet<Name>,
    pub right_axes: HashSet<Name>,
    pub shared_axes: HashSet<Name>,
    pub product: ElementwiseOp,
    pub plus: ElementwiseOp,
    pub b_trans: bool,
    pub channel: usize,
}

/// A term's reading as a monoid fold: its carried states, the names the combine binds the second operand to, the
/// injected singleton's results, and the componentwise ops (`None` for a twisted carrier).
#[derive(Clone, Debug, PartialEq)]
pub struct Reduction {
    pub axis: Name,
    pub states: Vec<Name>,
    pub other: Vec<Name>,
    pub terms: Vec<Name>,
    pub ops: Option<Vec<ElementwiseOp>>,
}

/// A carried state's step read as affine in the state and column-wise: the position the carrier reads mix, and the
/// own position a probe packs basis vectors along (both `None` for a diagonal map).
#[derive(Clone, Debug, PartialEq)]
pub struct Affine {
    pub state: Name,
    pub mixed: Option<usize>,
    pub kept: Option<usize>,
}

/// A reduce op's canonical combine: `sum` is `add`, `amax` is `maximum`; any other name is itself.
pub fn reduce_canon(name: &str) -> &str {
    match name {
        "sum" => "add",
        "prod" => "multiply",
        "amax" | "fmax" => "maximum",
        "fmin" => "minimum",
        other => other,
    }
}

/// Whether `product` distributes over the combine `plus`: the semirings a contraction is read over.
fn distributes_over(product: &ElementwiseOp, plus: &ElementwiseOp) -> bool {
    reduce_canon(plus.name()) == "add" && product.name() == "multiply"
}

/// One carried state's cone and the `Assign` that ENDS it — the product a bilinear channel contributes — or `None`
/// when the state is a pass-through or its cone ends in something else.
fn channel_product(lift: &Lambda, result: Name) -> Result<(Lambda, Option<Arc<Statement>>)> {
    let cone = lift.cone(result)?;
    let stmt = cone
        .body()
        .stmts()
        .iter()
        .find(|s| s.defines().contains(&result))
        .cloned();
    let product = stmt.filter(|s| matches!(&**s, Statement::Assign(_)));
    Ok((cone, product))
}

fn assign_parts(stmt: &Statement) -> Option<(&Name, &Expression)> {
    match stmt {
        Statement::Assign(Assign { name, value, .. }) => Some((name, value)),
        _ => None,
    }
}

fn operation(value: &Expression) -> Option<(&ElementwiseOp, &[Expression])> {
    match &**value {
        Expr::Apply(op, args) => Some((op, args)),
        _ => None,
    }
}

/// Whether `operand` is the A factor of `cone`'s product: a component `operands[0]` binds, or a value the cone
/// computes from those components, A's coordinates and uniform values.
fn over_a(
    operand: &Expression,
    cone: &Lambda,
    a_names: &HashSet<Name>,
    available: &HashSet<Name>,
) -> Result<bool> {
    let mut reads: HashSet<Name> = HashSet::new();
    for name in operand.free_vars() {
        reads.extend(cone.cone(name)?.params().iter().copied());
    }
    Ok(reads.iter().any(|r| a_names.contains(r))
        && reads
            .iter()
            .all(|r| a_names.contains(r) || available.contains(r)))
}

fn last_index_reads(load: &Statement, axis: Name) -> Result<bool> {
    let Statement::Load(load) = load else {
        return Ok(false);
    };
    let last = load
        .index
        .last()
        .ok_or_else(|| Error::Value("a slab with no index has no last coordinate".into()))?;
    Ok(last.free_vars().contains(&axis))
}

fn name(text: &str) -> Name {
    Name::new(text)
}

impl Fold {
    fn memos(&self) -> &Memos {
        self.readings()
    }

    /// This term restricted to the results `names` (exposed names), same operands and params, body cut to what those
    /// results need. Twisted, observed and carried states stay whole: their components depend on each other.
    pub fn exposing(&self, names: &[Name]) -> Result<Fold> {
        if self.exposes() == names
            || self.twist().is_some()
            || self.observe().is_some()
            || self.carries()
        {
            return Ok(self.clone());
        }
        if let Some(found) = self
            .memos()
            .exposing
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(names)
        {
            return Ok(found.clone());
        }
        let positions: Vec<usize> = names
            .iter()
            .map(|n| {
                self.exposes()
                    .iter()
                    .position(|e| e == n)
                    .ok_or_else(|| Error::Value(format!("'{n}' is not in list")))
            })
            .collect::<Result<_>>()?;
        let lift = self.lift();
        let keep: Vec<Name> = positions.iter().map(|&i| lift.results()[i]).collect();
        let members: Vec<Arc<Statement>> = lift
            .body()
            .backward_cone(&keep)
            .0
            .iter()
            .map(|here| here.stmt().clone())
            .collect();
        let lead =
            lift.params()[..usize::from(self.axis().is_some()) + self.bindings().len()].to_vec();
        let narrowed = Lambda::closing(lead, &Body::new(members), keep)?;
        let mut parts = self.parts();
        parts.lift = Some(narrowed);
        if self.axis().is_some() {
            let ops = self.ops().unwrap_or_default();
            parts.init = positions.iter().map(|&i| self.init()[i].clone()).collect();
            let chosen: Vec<ElementwiseOp> = positions.iter().map(|&i| ops[i].clone()).collect();
            parts.base = Some(Lambda::componentwise(&chosen, names)?);
        }
        let fold = Fold::new(parts)?;
        self.memos()
            .exposing
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(names.to_vec(), fold.clone());
        Ok(fold)
    }

    /// The singleton the ⊕ folds: the applied lift.
    pub fn injected(&self) -> Result<&Lambda> {
        self.applied()
    }

    /// The lift in BASE coordinates — `psi_inv ∘ lift` — for recognition only: bilinearity lives in base coordinates.
    pub fn based(&self) -> Result<Lambda> {
        if let Some(found) = self.memos().based.get() {
            return Ok(found.clone());
        }
        let applied = self.applied()?.clone();
        let based = match self.twist() {
            Some(twist) if !twist.channels.is_empty() => {
                let psi_inv = &twist.recipe.psi_inv;
                let slots: Vec<usize> = std::iter::once(0)
                    .chain(twist.channels.iter().map(|c| 1 + c))
                    .collect();
                let wanted: Vec<Name> = slots.iter().map(|&s| psi_inv.results()[s]).collect();
                let members: Vec<Arc<Statement>> = psi_inv
                    .body()
                    .backward_cone(&wanted)
                    .0
                    .iter()
                    .map(|h| h.stmt().clone())
                    .collect();
                let restricted = Lambda::new(
                    slots.iter().map(|&s| psi_inv.params()[s]).collect(),
                    &Body::new(members.clone()),
                    wanted,
                )?;
                let mut taken: HashSet<Name> = applied
                    .body()
                    .stmts()
                    .iter()
                    .flat_map(|s| s.defines())
                    .collect();
                taken.extend(applied.params().iter().copied());
                if restricted.params().len() != applied.results().len() {
                    return Err(Error::Value(
                        "psi_inv's restriction binds one param per carried state".into(),
                    ));
                }
                let mut names: HashMap<Name, Name> = restricted
                    .params()
                    .iter()
                    .copied()
                    .zip(applied.results().iter().copied())
                    .collect();
                for stmt in &members {
                    if let Some(own) = stmt.name() {
                        let spelled = if taken.contains(own) {
                            name(&format!("_b_{own}"))
                        } else {
                            *own
                        };
                        names.insert(*own, spelled);
                    }
                }
                let instance = restricted.rename(&names)?;
                let mut body = applied.body().stmts().to_vec();
                body.extend(instance.body().stmts().iter().cloned());
                Lambda::new(
                    applied.params().to_vec(),
                    &Body::new(body),
                    instance.results().to_vec(),
                )?
            }
            _ => applied,
        };
        Ok(self.memos().based.get_or_init(|| based).clone())
    }

    /// This term's spelling of the recipe's roles, `(score, *extras)`: the pivot's contribution, then each operand
    /// component a channel reads that is neither derived from the score nor carried.
    pub fn roles(&self) -> Result<Vec<Name>> {
        if let Some(found) = self.memos().roles.get() {
            return Ok(found.clone());
        }
        let bound: HashMap<Name, (usize, usize)> = self
            .bindings()
            .iter()
            .map(|b| (b.param, (b.operand, b.component)))
            .collect();
        let lift = self.lift();
        let defined: HashSet<Name> = lift
            .body()
            .stmts()
            .iter()
            .flat_map(|s| s.defines())
            .collect();
        let carried: HashSet<Name> = lift
            .results()
            .iter()
            .copied()
            .filter(|r| defined.contains(r))
            .collect();
        let mut extras: Vec<Name> = Vec::new();
        for result in &lift.results()[1..] {
            for param in lift.cone(*result)?.params() {
                let Some(&(operand, component)) = bound.get(param) else {
                    continue;
                };
                let exposed = self.operands()[operand].exposes()[component];
                if carried.contains(param) || extras.contains(&exposed) {
                    continue;
                }
                extras.push(exposed);
            }
        }
        let mut roles = vec![self.applied()?.results()[0]];
        roles.extend(extras);
        Ok(self.memos().roles.get_or_init(|| roles).clone())
    }

    /// Whether this operand varies over nothing the term it feeds does not evaluate over: no free axis, or a leaf over
    /// `axes` alone whose body is loads indexed by no coordinate and values computed from them, those and literals.
    pub fn uniform_leaf(&self, axes: &HashSet<Name>) -> bool {
        if self.free_axes().is_empty() {
            return true;
        }
        if !self.operands().is_empty() || self.base().is_some() || !self.free_axes().is_subset(axes)
        {
            return false;
        }
        let mut known: HashSet<Name> = axes.clone();
        for stmt in self.lift().body().stmts() {
            match &**stmt {
                Statement::Load(load) if load.index.iter().all(|i| i.free_vars().is_empty()) => {
                    known.insert(load.names[0]);
                }
                Statement::Let(l) if l.value.free_vars().iter().all(|n| known.contains(n)) => {
                    known.insert(l.name);
                }
                Statement::Assign(a) if a.value.free_vars().iter().all(|n| known.contains(n)) => {
                    known.insert(a.name);
                }
                _ => return false,
            }
        }
        true
    }

    /// Every carried state whose per-element contribution is ONE product of `operands[0]` by another operand edge, in
    /// carrier order: the state, the streamed operand, and the result of it the product multiplies.
    pub fn channel_operands(&self) -> Result<Vec<ChannelOperand>> {
        if let Some(found) = self.memos().channels.get() {
            return Ok(found.clone());
        }
        let out = self.find_channel_operands()?;
        Ok(self.memos().channels.get_or_init(|| out).clone())
    }

    fn find_channel_operands(&self) -> Result<Vec<ChannelOperand>> {
        let mut out = Vec::new();
        if self.axis().is_none() || self.base().is_none() || self.operands().len() < 2 {
            return Ok(out);
        }
        let Some(pluses) = self.ops() else {
            return Ok(out);
        };
        let operands = self.operands();
        let a_edge = &operands[0];
        let mut by_name: HashMap<Name, usize> = HashMap::new();
        let mut a_names: HashSet<Name> = HashSet::new();
        let mut uniform: HashSet<Name> = HashSet::new();
        for b in self.bindings() {
            let edge = &operands[b.operand];
            let exposed = edge.exposes()[b.component];
            by_name.insert(exposed, b.operand);
            if edge.ptr_eq(a_edge) {
                a_names.insert(exposed);
            } else if edge.uniform_leaf(a_edge.free_axes()) {
                uniform.insert(exposed);
            }
        }
        let mut available: HashSet<Name> = uniform;
        available.extend(a_edge.free_axes().iter().copied());
        let based = self.based()?;
        for (index, result) in based.results().iter().enumerate() {
            let (cone, product) = channel_product(&based, *result)?;
            let Some(product) = product else { continue };
            let Some((_, value)) = assign_parts(&product) else {
                continue;
            };
            let Some((op, args)) = operation(value) else {
                continue;
            };
            if args.len() != 2 {
                continue;
            }
            let plus = &pluses[index];
            if !(plus.associative() && plus.commutative() && plus.seed().is_some()) {
                continue;
            }
            if self.twist().is_none() {
                let identity = plus.seed();
                let seeded = self.init()[index].clone();
                if identity.is_none_or(|identity| seeded != Seed::Float(identity)) {
                    continue;
                }
            }
            if !distributes_over(op, plus) {
                continue;
            }
            let mut left = Vec::new();
            for arg in args.iter() {
                if over_a(arg, &cone, &a_names, &available)? {
                    left.push(arg.clone());
                }
            }
            let right: Vec<Name> = args
                .iter()
                .filter_map(|arg| arg.as_var().copied())
                .filter(|n| {
                    by_name.get(n).is_some_and(|&e| {
                        !operands[e].ptr_eq(a_edge) && !operands[e].free_axes().is_empty()
                    })
                })
                .collect();
            if left.len() != 1 || right.len() != 1 || left[0] == Expression::var(&right[0]) {
                continue;
            }
            out.push(ChannelOperand {
                state: index,
                operand: by_name[&right[0]],
                name: right[0],
            });
        }
        Ok(out)
    }

    /// The bilinear channels as `(state, streamed operand)`.
    pub fn bilinear_channels(&self) -> Result<Vec<(usize, usize)>> {
        Ok(self
            .channel_operands()?
            .iter()
            .map(|c| (c.state, c.operand))
            .collect())
    }

    /// The contraction one channel of this term reads as, or `None` when no channel is bilinear or the pair carries
    /// no output role.
    pub fn as_contraction(&self) -> Result<Option<Contraction>> {
        if let Some(found) = self.memos().contraction.get() {
            return Ok(found.clone());
        }
        let found = self.find_contraction()?;
        Ok(self.memos().contraction.get_or_init(|| found).clone())
    }

    fn find_contraction(&self) -> Result<Option<Contraction>> {
        let channels = self.bilinear_channels()?;
        let Some(&(channel, b_index)) = channels.first() else {
            return Ok(None);
        };
        let axis = self
            .axis()
            .ok_or_else(|| Error::Value("a contraction reduces".into()))?;
        let based = self.based()?;
        let (_, product) = channel_product(&based, based.results()[channel])?;
        let product_op = product
            .as_deref()
            .and_then(assign_parts)
            .and_then(|(_, value)| operation(value))
            .map(|(op, _)| op.clone())
            .ok_or_else(|| Error::Value("a bilinear channel ends in its product".into()))?;
        let plus = self
            .ops()
            .ok_or_else(|| Error::Value("a bilinear term is planar in base coordinates".into()))?
            [channel]
            .clone();
        let operands = self.operands();
        let (a_space, b_space) = (operands[0].free_axes(), operands[b_index].free_axes());
        if !(a_space.contains(&axis) && b_space.contains(&axis)) {
            return Ok(None);
        }
        let left: HashSet<Name> = a_space.difference(b_space).copied().collect();
        let right: HashSet<Name> = b_space.difference(a_space).copied().collect();
        if left.is_empty() && right.is_empty() {
            return Ok(None);
        }
        let transposed = |edge: &Fold| -> Result<bool> {
            match edge.as_slab() {
                Some(load) => last_index_reads(load, axis),
                None => Ok(false),
            }
        };
        let b_trans = transposed(&operands[b_index])?;
        for &(_, other) in &channels[1..] {
            if operands[other].free_axes() != b_space || transposed(&operands[other])? != b_trans {
                return Ok(None);
            }
        }
        let mut shared: HashSet<Name> = a_space.intersection(b_space).copied().collect();
        shared.remove(&axis);
        Ok(Some(Contraction {
            axis,
            left_axes: left,
            right_axes: right,
            shared_axes: shared,
            product: product_op,
            plus,
            b_trans,
            channel,
        }))
    }

    /// Whether the recipe this twisted carrier names lets a tier fold it chunk by chunk.
    pub fn chunked(&self) -> Result<bool> {
        if let Some(found) = self.memos().chunked.get() {
            return Ok(*found);
        }
        let found = (|| -> Result<bool> {
            let (Some(twist), Some(base)) = (self.twist(), self.base()) else {
                return Ok(false);
            };
            if twist.channels.is_empty() {
                return Ok(false);
            }
            let recipe = &twist.recipe;
            if recipe.advance.is_none() || recipe.scale.is_none() {
                return Ok(false);
            }
            if twist.channels.len() != base.results().len() - 1 {
                return Ok(false);
            }
            if twist
                .channels
                .iter()
                .any(|&i| recipe.channels[i].pattern.is_none())
            {
                return Ok(false);
            }
            let channels = self.bilinear_channels()?;
            if channels.len() != 1 {
                return Ok(false);
            }
            let channel = &recipe.channels[twist.channels[channels[0].0 - 1]];
            let Some(pattern) = &channel.pattern else {
                return Ok(false);
            };
            let Some(last) = pattern.body().stmts().last() else {
                return Ok(false);
            };
            Ok(match assign_parts(last) {
                Some((own, value)) => {
                    operation(value).is_some_and(|(_, args)| args.len() == 2)
                        && *own == pattern.results()[0]
                }
                None => false,
            })
        })()?;
        Ok(*self.memos().chunked.get_or_init(|| found))
    }

    /// Whether an atom tier could fold this term WHOLE.
    pub fn tiles_whole(&self) -> Result<bool> {
        if self.as_contraction()?.is_none() {
            return Ok(false);
        }
        if self.twist().is_some() {
            return self.chunked();
        }
        Ok(self.bilinear_channels()?.len() == self.base().map_or(0, |b| b.results().len()))
    }

    /// The operand a tier streams along this carrier's axis.
    pub fn streamed(&self) -> Result<usize> {
        if self.twist().is_some()
            && let Some(&(_, operand)) = self.bilinear_channels()?.first()
        {
            return Ok(operand);
        }
        Ok(1)
    }

    /// This carrier as ONE fold per carried state under a projection re-exposing the state tuple, or `None` where
    /// taking it apart buys nothing.
    pub fn per_state(&self) -> Result<Option<Fold>> {
        if let Some(found) = self.memos().per_state.get() {
            return Ok(found.clone());
        }
        let found = self.find_per_state()?;
        Ok(self.memos().per_state.get_or_init(|| found).clone())
    }

    fn find_per_state(&self) -> Result<Option<Fold>> {
        let Some(axis) = self.axis() else {
            return Ok(None);
        };
        let Some(base) = self.base() else {
            return Ok(None);
        };
        if self.twist().is_some() || self.observe().is_some() || self.carries() {
            return Ok(None);
        }
        let Some(pluses) = base.components() else {
            return Ok(None);
        };
        if base.results().len() < 2 || self.tiles_whole()? {
            return Ok(None);
        }
        let mut children: Vec<Fold> = Vec::new();
        for (index, result) in self.lift().results().iter().enumerate() {
            let members: Vec<Arc<Statement>> = self
                .lift()
                .body()
                .backward_cone(&[*result])
                .0
                .iter()
                .map(|h| h.stmt().clone())
                .collect();
            let body = Body::new(members);
            let mut read: HashSet<Name> = body.ssa_uses().clone();
            read.insert(*result);
            let mut operands = Vec::new();
            let mut params = vec![axis];
            for edge in self.operands() {
                let slots: Vec<Name> = self
                    .bindings()
                    .iter()
                    .filter(|b| self.operands()[b.operand].ptr_eq(edge))
                    .map(|b| b.param)
                    .collect();
                if slots.iter().any(|p| read.contains(p)) {
                    operands.push(edge.clone());
                    params.extend(slots);
                }
            }
            let mut parts = self.parts();
            parts.operands = operands;
            parts.lift = Some(Lambda::closing(params, &body, vec![*result])?);
            parts.init = vec![self.init()[index].clone()];
            parts.base = Some(Lambda::componentwise(
                &[pluses[index].clone()],
                &[base.results()[index]],
            )?);
            children.push(Fold::new(parts)?);
        }
        let mut whole = false;
        for child in &children {
            whole |= child.tiles_whole()?;
        }
        let spaces: Vec<Vec<Name>> = children
            .iter()
            .map(|c| {
                let mut v: Vec<Name> = c.free_axes().iter().copied().collect();
                v.sort();
                v
            })
            .collect();
        if !whole && spaces.windows(2).all(|w| w[0] == w[1]) {
            return Ok(None);
        }
        let exposes = self.exposes().to_vec();
        Ok(Some(Fold::new(Parts {
            operands: children,
            lift: Some(Lambda::closing(exposes.clone(), &Body::empty(), exposes)?),
            ..Parts::default()
        })?))
    }

    /// This term's reading as a monoid fold, or `None` for a term without a combine.
    pub fn as_reduction(&self) -> Result<Option<Reduction>> {
        if let Some(found) = self.memos().reduction.get() {
            return Ok(found.clone());
        }
        let found = match self.combine()? {
            None => None,
            Some(combine) => {
                let states = combine.results().to_vec();
                Some(Reduction {
                    axis: self
                        .axis()
                        .ok_or_else(|| Error::Value("a reduction binds an axis".into()))?,
                    other: combine.params()[states.len()..].to_vec(),
                    terms: self.injected()?.results().to_vec(),
                    ops: combine.components(),
                    states,
                })
            }
        };
        Ok(self.memos().reduction.get_or_init(|| found).clone())
    }

    /// Whether this term is ONE value for the whole kernel: no operands, no axis, no free coordinates.
    pub fn scalar(&self) -> bool {
        self.axis().is_none()
            && self.operands().is_empty()
            && self.free_axes().is_empty()
            && !self.lift().body().is_empty()
    }

    /// One gmem read as a term: a lift of one `Load` over the coordinates it indexes.
    pub fn slab(load: Arc<Statement>) -> Result<Fold> {
        let Statement::Load(l) = &*load else {
            return Err(Error::Type("a slab reads one Load".into()));
        };
        let names = l.names.to_vec();
        Fold::new(Parts {
            lift: Some(Lambda::closing(
                Vec::new(),
                &Body::new(vec![load.clone()]),
                names,
            )?),
            ..Parts::default()
        })
    }

    /// A read of a carried state as a term: a lift of one `Load` of the state one step back, binding the state, the
    /// carrying axis, and the cell coordinates its index reads.
    pub fn carrier_read(read: Arc<Statement>, axis: Name) -> Result<Fold> {
        let Statement::Load(load) = &*read else {
            return Err(Error::Type("a carrier read is one Load".into()));
        };
        let (carrier, own) = (load.input, load.names[0]);
        Fold::new(Parts {
            lift: Some(Lambda::closing(
                vec![carrier, axis],
                &Body::new(vec![read.clone()]),
                vec![own],
            )?),
            ..Parts::default()
        })
    }

    /// The affine reading of this carried state, or `None`.
    pub fn affine(&self) -> Result<Option<Affine>> {
        if let Some(found) = self.memos().affine.get() {
            return Ok(found.clone());
        }
        let found = self.find_affine()?;
        Ok(self.memos().affine.get_or_init(|| found).clone())
    }

    fn find_affine(&self) -> Result<Option<Affine>> {
        let Some(base) = self.base() else {
            return Ok(None);
        };
        if !self.carries() || base.results().len() != 1 {
            return Ok(None);
        }
        let state = base.results()[0];
        let mut reads: Vec<Arc<Statement>> = Vec::new();
        let classes = affine_classes(self, state, &mut reads, &mut HashMap::new(), None)?;
        let kind = classes.get(&state).map_or(Kind::Free, |c| c.0);
        if !matches!(kind, Kind::Affine | Kind::Free) || reads.is_empty() {
            return Ok(None);
        }
        let own: Vec<Expression> = self.cells().iter().map(|c| Expression::var(c)).collect();
        let mut mixed: HashSet<usize> = HashSet::new();
        for read in &reads {
            let Statement::Load(load) = &**read else {
                continue;
            };
            for (position, expression) in load.index.iter().skip(1).enumerate() {
                if own.get(position) != Some(expression) {
                    mixed.insert(position);
                }
            }
        }
        if mixed.is_empty() {
            return Ok(Some(Affine {
                state,
                mixed: None,
                kept: None,
            }));
        }
        if mixed.len() != 1 || self.cells().len() < 2 {
            return Ok(None);
        }
        let position = *mixed.iter().next().expect("one mixed position");
        let kept = (0..self.cells().len())
            .filter(|&p| p != position)
            .max()
            .expect("two cells");
        let varies = affine_classes(
            self,
            state,
            &mut Vec::new(),
            &mut HashMap::new(),
            Some((self.cells()[kept], kept)),
        )?;
        if varies.get(&state).is_some_and(|c| c.1) {
            return Ok(None);
        }
        Ok(Some(Affine {
            state,
            mixed: Some(position),
            kept: Some(kept),
        }))
    }

    /// The combine APPLIED at `(state, other)` — a second, fully reduced state named `other` — as loop-IR statements:
    /// the combine's temps renamed onto `other`'s spelling, then one in-place `Accum` per state component (a rescaled
    /// old state its `base`).
    pub fn merge(&self, other: &[Name]) -> Result<Body> {
        if let Some(found) = self
            .memos()
            .merge
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(other)
        {
            return Ok(found.clone());
        }
        let body = self.render_merge(other)?;
        self.memos()
            .merge
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(other.to_vec(), body.clone());
        Ok(body)
    }

    fn render_merge(&self, other: &[Name]) -> Result<Body> {
        let view = self
            .as_reduction()?
            .ok_or_else(|| Error::Value("merge applies a combine".into()))?;
        let combine = self
            .combine()?
            .ok_or_else(|| Error::Value("merge applies a combine".into()))?;
        if view.other.len() != other.len() {
            return Err(Error::Value("one merged name per carried state".into()));
        }
        let prefix = format!("{}__", view.other[0]);
        let states: HashSet<Name> = view.states.iter().copied().collect();
        let mut named: HashMap<Name, Name> = view
            .other
            .iter()
            .copied()
            .zip(other.iter().copied())
            .collect();
        for stmt in combine.body().stmts() {
            if let Some(own) = stmt.name()
                && !states.contains(own)
            {
                let stripped = own.strip_prefix(prefix.as_str()).unwrap_or(own);
                named.insert(*own, name(&format!("{}__{stripped}", other[0])));
            }
        }
        let applied = combine.rename(&named)?;
        let definitions: HashMap<Name, Arc<Statement>> = applied
            .body()
            .stmts()
            .iter()
            .filter_map(|s| {
                s.name()
                    .filter(|n| !states.contains(n))
                    .map(|n| (*n, s.clone()))
            })
            .collect();
        fn reads(
            expression: &Expression,
            state: Name,
            definitions: &HashMap<Name, Arc<Statement>>,
        ) -> bool {
            expression.free_vars().into_iter().any(|n| {
                n == state
                    || definitions
                        .get(&n)
                        .and_then(|s| assign_parts(s))
                        .is_some_and(|(_, value)| reads(value, state, definitions))
            })
        }
        let mut out: Vec<Arc<Statement>> = Vec::new();
        for stmt in applied.body().stmts() {
            let Some(own) = stmt.name().copied() else {
                out.push(stmt.clone());
                continue;
            };
            if !states.contains(&own) {
                out.push(stmt.clone());
                continue;
            }
            let state = own;
            let Some((_, value)) = assign_parts(stmt) else {
                return Err(Error::Value(
                    "a combine writes each state as an Assign".into(),
                ));
            };
            let mut value = value.clone();
            let state_var = Expression::var(&state);
            if let Some(var) = value.as_var().copied()
                && let Some(pivot) = definitions.get(&var)
                && let Some((_, pivot_value)) = assign_parts(pivot)
                && operation(pivot_value).is_some_and(|(_, args)| args.contains(&state_var))
            {
                value = pivot_value.clone();
            }
            let Some((op, operands)) = operation(&value).map(|(o, a)| (o.clone(), a.to_vec()))
            else {
                return Err(Error::Value("a combine's state is an operation".into()));
            };
            if op.name() == "add" {
                let form = affine_terms(&value, ExprClass::Value);
                if let Some(position) = form
                    .terms
                    .iter()
                    .position(|(c, t)| *c == 1.0 && *t == state_var)
                {
                    let mut terms = form.terms.clone();
                    terms.remove(position);
                    let rest = if !terms.is_empty() || form.offset != 0.0 {
                        affine_sum(
                            AffineTerms {
                                terms,
                                offset: form.offset,
                                float: form.float,
                            },
                            &|_| ties::unranked(),
                        )
                    } else {
                        state_var.clone()
                    };
                    out.push(accum(state, rest, op, None, None));
                    continue;
                }
            } else if operands.contains(&state_var) {
                let terms: Vec<Expression> = operands
                    .iter()
                    .filter(|o| **o != state_var)
                    .cloned()
                    .collect();
                let rest = match terms.len() {
                    0 => state_var.clone(),
                    1 => terms[0].clone(),
                    _ => Expression::call(op.name(), terms),
                };
                out.push(accum(state, rest, op, None, None));
                continue;
            }
            if operands.len() != 2 {
                return Err(Error::Value(
                    "a rescaled state combines two operands".into(),
                ));
            }
            let (mut old, folded) = if reads(&operands[0], state, &definitions) {
                (operands[0].clone(), operands[1].clone())
            } else {
                (operands[1].clone(), operands[0].clone())
            };
            if old.as_var().is_none() {
                let bound = name(&format!("{}__{state}_base", other[0]));
                out.push(Arc::new(Statement::Assign(Assign {
                    name: bound,
                    value: old,
                    dtype: None,
                })));
                old = Expression::var(&bound);
            }
            out.push(accum(
                state,
                folded,
                op,
                Some(*old.as_var().expect("a named base")),
                None,
            ));
        }
        Ok(Body::new(out))
    }

    /// The per-step statements this fold derives: the injected lift's body, then the combine applied at the singleton
    /// (each `Accum` over the reduce axis) or a carried state's `Carry`s, then an observer's tap. A map's step is its
    /// lift body.
    pub fn step(&self) -> Result<Body> {
        if let Some(found) = self.memos().step.get() {
            return Ok(found.clone());
        }
        let lift = self.injected()?.clone();
        let body = match self.combine()? {
            None => lift.body().clone(),
            Some(_) if self.carries() => {
                let axis = self.axis().expect("a carried state binds its step");
                let index: Vec<Expression> = std::iter::once(&axis)
                    .chain(self.cells())
                    .map(|c| Expression::var(c))
                    .collect();
                let base = self.base().expect("a carried state has a base");
                let mut stmts = lift.body().stmts().to_vec();
                for ((state, value), seed) in
                    base.results().iter().zip(lift.results()).zip(self.init())
                {
                    let seed = match seed {
                        Seed::Int(v) => NumOr::Num(Num::Int(*v)),
                        Seed::Float(v) => NumOr::Num(Num::Float(*v)),
                        Seed::Buffer(n) => NumOr::Value(*n),
                    };
                    stmts.push(Arc::new(Statement::Carry(Carry {
                        name: *state,
                        value: *value,
                        index: index.clone().into(),
                        seed,
                        dtype: None,
                    })));
                }
                Body::new(stmts)
            }
            Some(_) => {
                let axis = self.axis().expect("a combine binds an axis");
                let mut stmts = lift.body().stmts().to_vec();
                for stmt in self.merge(lift.results())?.stmts() {
                    match &**stmt {
                        Statement::Accum(a) => {
                            stmts.push(accum(
                                a.name,
                                a.value.clone(),
                                a.op.clone(),
                                a.base,
                                Some(axis),
                            ));
                        }
                        _ => stmts.push(stmt.clone()),
                    }
                }
                if let Some(observe) = self.observe() {
                    stmts.extend(observe.body().stmts().iter().cloned());
                }
                Body::new(stmts)
            }
        };
        Ok(self.memos().step.get_or_init(|| body).clone())
    }
}

fn accum(
    name: Name,
    value: Expression,
    op: ElementwiseOp,
    base: Option<Name>,
    axis: Option<Name>,
) -> Arc<Statement> {
    Arc::new(Statement::Accum(Accum {
        name,
        value,
        op,
        dtype: None,
        axes: axis.into_iter().collect::<Vec<_>>().into(),
        base,
    }))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Free,
    Affine,
    Non,
}

type Classes = HashMap<Name, (Kind, bool)>;

/// How every value `term` exposes depends on the carried `state` — free, affine or neither — beside whether it varies
/// with the kept cell coordinate `kept` (its name and position). The carrier reads of the state met are collected.
fn affine_classes(
    term: &Fold,
    state: Name,
    reads: &mut Vec<Arc<Statement>>,
    memo: &mut HashMap<usize, Classes>,
    kept: Option<(Name, usize)>,
) -> Result<Classes> {
    if let Some(found) = memo.get(&term.address()) {
        return Ok(found.clone());
    }
    let (kept_name, position) = match kept {
        Some((n, p)) => (Some(n), Some(p)),
        None => (None, None),
    };
    if let Some(read) = term.as_carrier_read() {
        let Statement::Load(load) = &**read else {
            unreachable!("a carrier read is a Load")
        };
        if load.input == state {
            reads.push(read.clone());
        }
        let varies = kept.is_some()
            && load.index.iter().skip(1).enumerate().any(|(i, e)| {
                Some(i) != position && kept_name.is_some_and(|n| e.free_vars().contains(&n))
            });
        let out: Classes = HashMap::from([(
            load.names[0],
            (
                if load.input == state {
                    Kind::Affine
                } else {
                    Kind::Free
                },
                varies,
            ),
        )]);
        memo.insert(term.address(), out.clone());
        return Ok(out);
    }
    let mut env: Classes = HashMap::new();
    for b in term.bindings() {
        let edge = &term.operands()[b.operand];
        let classes = affine_classes(edge, state, reads, memo, kept)?;
        env.insert(
            b.param,
            classes
                .get(&edge.exposes()[b.component])
                .copied()
                .unwrap_or((Kind::Free, false)),
        );
    }
    let of = |env: &Classes, value: &Name| -> (Kind, bool) {
        env.get(value)
            .copied()
            .unwrap_or((Kind::Free, Some(*value) == kept_name))
    };
    fn combine(op: &str, args: &[(Kind, bool)]) -> (Kind, bool) {
        let kinds: Vec<Kind> = args.iter().map(|a| a.0).collect();
        if kinds.iter().all(|k| *k == Kind::Free) {
            return (Kind::Free, args.iter().any(|a| a.1));
        }
        if kinds.contains(&Kind::Non) {
            return (Kind::Non, false);
        }
        let affine_varies = args.iter().any(|a| a.0 == Kind::Affine && a.1);
        match op {
            "add" | "subtract" | "copy" | "negative" => (Kind::Affine, affine_varies),
            "multiply" if kinds.iter().filter(|k| **k == Kind::Affine).count() == 1 => {
                (Kind::Affine, args.iter().any(|a| a.1))
            }
            "divide" if kinds == [Kind::Affine, Kind::Free] => {
                (Kind::Affine, args.iter().any(|a| a.1))
            }
            _ => (Kind::Non, false),
        }
    }
    fn classify(
        env: &Classes,
        expression: &Expression,
        of: &dyn Fn(&Classes, &Name) -> (Kind, bool),
    ) -> (Kind, bool) {
        match &**expression {
            Expr::Var(n) => of(env, n),
            Expr::Literal(_) => (Kind::Free, false),
            Expr::Apply(op, args) => {
                let classes: Vec<(Kind, bool)> =
                    args.iter().map(|a| classify(env, a, of)).collect();
                combine(op.name(), &classes)
            }
            _ => (Kind::Non, false),
        }
    }
    for stmt in term.lift().body().stmts() {
        match &**stmt {
            Statement::Assign(a) => {
                let class = classify(&env, &a.value, &of);
                env.insert(a.name, class);
            }
            Statement::Load(load) => {
                let coordinates: Vec<(Kind, bool)> = load
                    .index
                    .iter()
                    .flat_map(|e| e.free_vars())
                    .map(|v| of(&env, &v))
                    .collect();
                let class = if coordinates.iter().any(|c| c.0 != Kind::Free) {
                    (Kind::Non, false)
                } else {
                    (Kind::Free, coordinates.iter().any(|c| c.1))
                };
                env.insert(load.names[0], class);
            }
            Statement::Select(select) => {
                let branches: Vec<(Kind, bool)> =
                    select.branches.iter().map(|b| of(&env, &b.value)).collect();
                let predicates: Vec<(Kind, bool)> = select
                    .branches
                    .iter()
                    .flat_map(|b| b.select.free_vars())
                    .map(|v| of(&env, &v))
                    .collect();
                let class = if predicates.iter().any(|c| c.0 != Kind::Free) {
                    (Kind::Non, false)
                } else {
                    let (kind, varies) = combine("add", &branches);
                    (kind, varies || predicates.iter().any(|c| c.1))
                };
                env.insert(select.name, class);
            }
            other => {
                let class = if other.flags().contains(PURE) && other.deps().is_empty() {
                    (Kind::Free, false)
                } else {
                    (Kind::Non, false)
                };
                for defined in other.defines() {
                    env.insert(defined, class);
                }
            }
        }
    }
    let mut out: Classes = HashMap::new();
    match term.base() {
        None => {
            for (exposed, own) in term.exposes().iter().zip(term.lift().results()) {
                out.insert(*exposed, of(&env, own));
            }
        }
        Some(base) => {
            let ops = base.components();
            for (index, (exposed, own)) in
                base.results().iter().zip(term.lift().results()).enumerate()
            {
                let (kind, varies) = of(&env, own);
                let folded = kind == Kind::Free
                    || (kind == Kind::Affine
                        && (term.carries()
                            || ops.as_ref().is_some_and(|ops| ops[index].name() == "add")));
                out.insert(
                    *exposed,
                    if folded {
                        (kind, varies)
                    } else {
                        (Kind::Non, false)
                    },
                );
            }
            let n = base.results().len();
            for (exposed, own) in term
                .exposes()
                .iter()
                .skip(n)
                .zip(term.lift().results().iter().skip(n))
            {
                out.insert(*exposed, of(&env, own));
            }
        }
    }
    memo.insert(term.address(), out.clone());
    Ok(out)
}
