//! Twist recipes: the streaming carriers a two-pass reduce pair fuses into, as data.
//!
//! A recipe is a TWISTED MONOID: a componentwise monoid `base` (one ⊕ per state, `lift` the per-element contribution
//! to each) conjugated by a bijection `psi` on the carrier. What the tree is matched on and what the kernel runs are
//! data beside that definition, because stability is not preserved by conjugation: the `channels` say which
//! per-element map a dependent's lift must spell and what each state is at the singleton, and the fused ⊕ is spelled
//! in its stable form, as an `advance` and a per-channel `scale` (one recipe for any channel count) or as one
//! `combine` lambda. The recipes themselves are defined in Python (`emmy.compiler.ir.pure.twist`); this module holds
//! their type and the two programs a fold derives from one.

use std::collections::HashMap;
use std::sync::Arc;

use emmy_ir::body::Body;
use emmy_ir::elementwise::ElementwiseOp;
use emmy_ir::expr::Expression;
use emmy_ir::statement::{Assign, Statement};
use emmy_ir::{Error, Name, Result};

use crate::lambda::Lambda;

/// One state of the fused carrier beyond the pivot's. `injection` is the state at the injected singleton over
/// `(score, *extras)`; with a `pattern` the state is a dependent fold's own, the per-element map over
/// `(score, pivot, *extras)` that fold's lift must spell; without one it is a state the two-pass form never had,
/// suffixed `name` and seeded `init`.
#[derive(Clone, Debug, PartialEq)]
pub struct Channel {
    pub injection: Lambda,
    pub pattern: Option<Lambda>,
    pub name: String,
    pub init: f64,
}

/// A twisted monoid as data. Shared: a clone is the same recipe, and a fold names its recipe by it.
#[derive(Clone, Debug)]
pub struct Recipe(Arc<RecipeData>);

#[derive(Debug, PartialEq)]
pub struct RecipeData {
    pub name: String,
    /// The base ⊕ of every carrier state: the pivot's, then one per channel in recipe order.
    pub base: Vec<ElementwiseOp>,
    pub lift: Lambda,
    pub psi: Lambda,
    pub psi_inv: Lambda,
    pub channels: Vec<Channel>,
    pub advance: Option<Lambda>,
    pub scale: Option<Lambda>,
    pub combine: Option<Lambda>,
}

impl PartialEq for Recipe {
    fn eq(&self, other: &Recipe) -> bool {
        Arc::ptr_eq(&self.0, &other.0) || self.0 == other.0
    }
}

impl std::ops::Deref for Recipe {
    type Target = RecipeData;
    fn deref(&self) -> &RecipeData {
        &self.0
    }
}

fn name(text: &str) -> Name {
    Name::new(text)
}

fn renamed(
    lambda: &Lambda,
    names: &HashMap<Name, Name>,
    fallback: impl Fn(&Name) -> Name,
) -> Result<Lambda> {
    let mapping: HashMap<Name, Name> = lambda
        .names()
        .into_iter()
        .map(|n| (n, names.get(&n).copied().unwrap_or_else(|| fallback(&n))))
        .collect();
    lambda.rename(&mapping)
}

impl Recipe {
    pub fn new(data: RecipeData) -> Recipe {
        Recipe(Arc::new(data))
    }

    /// The shared data: what a table keyed per recipe holds a weak handle on.
    pub fn arc(&self) -> &Arc<RecipeData> {
        &self.0
    }

    pub fn ptr_eq(&self, other: &Recipe) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// The pivot's ⊕: the base monoid's first component.
    pub fn pivot(&self) -> &ElementwiseOp {
        &self.0.base[0]
    }

    /// The fused ⊕ over these state names, `S × S → S` with the second operand `<n>__o`. Temps are namespaced on the
    /// second pivot's name, so two merges into one state never collide. The advance/scale spelling: the advance over
    /// the pivot pair, each channel's two sides scaled by their factors and joined by the channel's base ⊕, the pivot
    /// written last (the channels read the old pivot through the factors).
    pub fn program(&self, states: &[Name]) -> Result<Lambda> {
        let other: Vec<Name> = states.iter().map(|s| name(&format!("{s}__o"))).collect();
        let key = other[0];
        if let Some(combine) = &self.0.combine {
            let roles = combine.results();
            let mut names: HashMap<Name, Name> = HashMap::new();
            let second: Vec<Name> = roles.iter().map(|r| name(&format!("{r}__o"))).collect();
            if roles.len() + second.len() != states.len() + other.len() {
                return Err(Error::Value(
                    "recipe combine arity differs from the carrier".into(),
                ));
            }
            for (role, state) in roles.iter().chain(&second).zip(states.iter().chain(&other)) {
                names.insert(*role, *state);
            }
            return renamed(combine, &names, |n| name(&format!("{key}__{n}")));
        }
        let (Some(advance), Some(scale)) = (&self.0.advance, &self.0.scale) else {
            return Err(Error::Value(format!(
                "recipe {} spells no fused ⊕",
                self.0.name
            )));
        };
        let roles: HashMap<Name, Name> = advance
            .params()
            .iter()
            .copied()
            .zip([states[0], key])
            .collect();
        let advance = renamed(advance, &roles, |n| name(&format!("{key}__{n}")))?;
        let pivot = advance.results()[0];
        let factors = &advance.results()[1..];
        let mut body: Vec<Arc<Statement>> = advance.body().stmts().to_vec();
        let ops: std::collections::BTreeSet<&str> =
            self.0.base[1..].iter().map(|op| op.name()).collect();
        if ops.len() != 1 {
            return Err(Error::Value(
                "the open-count spelling takes one channel ⊕".into(),
            ));
        }
        let op = self.0.base[1].clone();
        for (state, second) in states[1..].iter().zip(&other[1..]) {
            let mut sides = Vec::new();
            for (side, factor, tag) in [(*state, factors[0], "sa"), (*second, factors[1], "sb")] {
                let mut names: HashMap<Name, Name> =
                    scale.params().iter().copied().zip([side, factor]).collect();
                let result = name(&format!("{key}__{state}_{tag}"));
                names.insert(scale.results()[0], result);
                let instance = renamed(scale, &names, |n| name(&format!("{key}__{state}_{n}")))?;
                body.extend(instance.body().stmts().iter().cloned());
                sides.push(result);
            }
            body.push(Arc::new(Statement::Assign(Assign {
                name: *state,
                value: Expression::call(
                    op.name(),
                    sides.iter().map(|s| Expression::var(s)).collect(),
                ),
                dtype: None,
            })));
        }
        body.push(Arc::new(Statement::Assign(Assign {
            name: states[0],
            value: Expression::var(&pivot),
            dtype: None,
        })));
        let mut params = states.to_vec();
        params.extend(other);
        Lambda::new(params, &Body::new(body), states.to_vec())
    }
}

/// One recipe INSTANTIATED on a term: the schema plus which recipe channel each carried state past the pivot is, in
/// the order the tree fused them. Empty `channels` says the term folds no per-element contribution at all (a split's
/// partial merge of finished carrier states).
#[derive(Clone, Debug, PartialEq)]
pub struct Twist {
    pub recipe: Recipe,
    pub channels: Vec<usize>,
}

impl Twist {
    pub fn name(&self) -> &str {
        &self.recipe.name
    }

    /// `combine` over these state names, in the stable spelling the recipe authored.
    pub fn program(&self, states: &[Name]) -> Result<Lambda> {
        self.recipe.program(states)
    }

    /// `psi ∘ lift` at the singleton: the score, then one channel injection per carried state past the pivot, over
    /// `roles` (this term's spelling of the recipe lift's params in order; a role the term never bound is absent, and
    /// an injection that reads one is refused). Written out by the recipe's AUTHORED injections, which state the
    /// simplification at the singleton that evaluating `psi` apart would overflow on.
    pub fn inject(&self, roles: &[Name], states: &[Name]) -> Result<Lambda> {
        let spelled: HashMap<Name, Name> = self
            .recipe
            .lift
            .params()
            .iter()
            .copied()
            .zip(roles.iter().copied())
            .collect();
        let mut body: Vec<Arc<Statement>> = Vec::new();
        let mut results = vec![roles[0]];
        if states.len().saturating_sub(1) != self.channels.len() {
            return Err(Error::Value(
                "one channel per carried state past the pivot".into(),
            ));
        }
        for (state, index) in states[1..].iter().zip(&self.channels) {
            let injection = &self.recipe.channels[*index].injection;
            let mut names: HashMap<Name, Name> = HashMap::new();
            for param in injection.params() {
                let role = spelled
                    .get(param)
                    .ok_or_else(|| Error::Key(format!("'{param}'")))?;
                names.insert(*param, *role);
            }
            for stmt in injection.body().stmts() {
                if let Some(defined) = stmt.name() {
                    names.insert(*defined, name(&format!("{state}__{defined}")));
                }
            }
            let instance = injection.rename(&names)?;
            body.extend(instance.body().stmts().iter().cloned());
            results.extend(instance.results().iter().copied());
        }
        Lambda::new(roles.to_vec(), &Body::new(body), results)
    }
}
