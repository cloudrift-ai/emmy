//! `Fold`: the one reduce term of Tile IR, `reduce(⊕) ∘ map(f)` in the λ-foldMap spelling.
//!
//! A pure term: a tuple of `operands` (each a term, bound POSITIONALLY to the lift's params), a pure `lift` lambda
//! whose first param is the iteration var when the term reduces, the monoid's `(init, base)` pair, the optional
//! `twist` recipe that conjugates it, a carried state's `cells` and an optional per-step `observe`r. Every reading is
//! derived from those and computed once per term ([`Fold::exposes`], [`Fold::bindings`], [`Fold::applied`],
//! [`Fold::free_axes`], [`Fold::combine`], [`Fold::canonical`]); the term is immutable, so a structural copy that
//! keeps a subtree keeps its answers. Construction ([`Fold::new`]) states the formation rules: it drops a shadowed
//! operand of a zero-axis term, checks the positional binding, and orients a bilinear term A-first.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};

use emmy_ir::body::Body;
use emmy_ir::elementwise::ElementwiseOp;
use emmy_ir::expr::Expr;
use emmy_ir::statement::Statement;
use emmy_ir::{Error, Name, Result};

use crate::lambda::Lambda;
use crate::twist::Twist;

/// A ⊕ seed: an op identity, or for a carried state the value it holds before the first step — a constant, or the
/// name of a buffer of the state's shape.
#[derive(Clone, Debug)]
pub enum Seed {
    Int(i64),
    Float(f64),
    Buffer(Name),
}

impl Seed {
    fn number(&self) -> Option<f64> {
        match self {
            Seed::Int(value) => Some(*value as f64),
            Seed::Float(value) => Some(*value),
            Seed::Buffer(_) => None,
        }
    }
}

impl PartialEq for Seed {
    /// Python's equality: a number equals the same number of the other kind, a buffer name only itself.
    fn eq(&self, other: &Seed) -> bool {
        match (self, other) {
            (Seed::Buffer(a), Seed::Buffer(b)) => a == b,
            (Seed::Buffer(_), _) | (_, Seed::Buffer(_)) => false,
            _ => self.number() == other.number(),
        }
    }
}

impl Hash for Seed {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Seed::Buffer(name) => name.hash(state),
            _ => {
                let value = self.number().unwrap_or_default();
                (if value == 0.0 { 0.0f64 } else { value })
                    .to_bits()
                    .hash(state);
            }
        }
    }
}

/// One reduce term. Immutable and shared: a clone is the same term, and sharing (one term reached through several
/// operand positions) is meaningful: the placement reads one value where it sees one term.
#[derive(Clone)]
pub struct Fold(Arc<Data>);

/// One operand-bound lift param: the param, the operand it binds (an index into `operands`) and which of that
/// operand's exposed components.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Binding {
    pub param: Name,
    pub operand: usize,
    pub component: usize,
}

pub struct Data {
    operands: Vec<Fold>,
    lift: Lambda,
    init: Vec<Seed>,
    cells: Vec<Name>,
    base: Option<Lambda>,
    twist: Option<Twist>,
    observe: Option<Lambda>,
    hash: OnceLock<u64>,
    exposes: OnceLock<Vec<Name>>,
    bindings: OnceLock<Vec<Binding>>,
    applied: OnceLock<Lambda>,
    carries: OnceLock<bool>,
    combine: OnceLock<Option<Lambda>>,
    free_axes: OnceLock<HashSet<Name>>,
    canonical: OnceLock<Fold>,
}

/// The stored parts of a term, what [`Fold::new`] forms one from.
#[derive(Clone, Default)]
pub struct Parts {
    pub operands: Vec<Fold>,
    pub lift: Option<Lambda>,
    pub init: Vec<Seed>,
    pub cells: Vec<Name>,
    pub base: Option<Lambda>,
    pub twist: Option<Twist>,
    pub observe: Option<Lambda>,
}

impl PartialEq for Fold {
    fn eq(&self, other: &Fold) -> bool {
        if Arc::ptr_eq(&self.0, &other.0) {
            return true;
        }
        let (a, b) = (&*self.0, &*other.0);
        self.structural_hash() == other.structural_hash()
            && a.lift == b.lift
            && a.init == b.init
            && a.cells == b.cells
            && a.base == b.base
            && a.twist == b.twist
            && a.observe == b.observe
            && a.operands == b.operands
    }
}

impl Eq for Fold {}

impl Hash for Fold {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.structural_hash().hash(state);
    }
}

impl std::fmt::Debug for Fold {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fold")
            .field("lift", &self.0.lift)
            .field("operands", &self.0.operands.len())
            .finish_non_exhaustive()
    }
}

/// The names an operation's operands read directly: every operand that is a name.
fn operand_names(stmt: &Statement) -> Vec<Name> {
    match stmt {
        Statement::Assign(assign) => match &*assign.value {
            Expr::Apply(_, args) => args
                .iter()
                .filter_map(|arg| arg.as_var().copied())
                .collect(),
            _ => Vec::new(),
        },
        _ => Vec::new(),
    }
}

/// Whether a statement is an `Assign` whose value is an operation over exactly two operands.
fn binary_assign(stmt: &Statement) -> bool {
    matches!(stmt, Statement::Assign(assign) if matches!(&*assign.value, Expr::Apply(_, args) if args.len() == 2))
}

impl Fold {
    /// The term over `parts`, formed: a shadowed operand of a zero-axis term leaves with its params, the positional
    /// binding and the monoid arity are checked, a carried state's carrier reads index every cell axis, a bilinear term
    /// takes its A operand first, and an observer binds the iteration var then the carried state.
    pub fn new(parts: Parts) -> Result<Fold> {
        let Parts {
            mut operands,
            lift,
            init,
            cells,
            base,
            twist,
            observe,
        } = parts;
        let mut lift = lift.ok_or_else(|| Error::Type("a Fold takes a lift".into()))?;
        let Some(base) = base else {
            if !init.is_empty() || twist.is_some() {
                return Err(Error::Value(
                    "a zero-axis Fold carries no monoid to seed or twist".into(),
                ));
            }
            if observe.is_some() {
                return Err(Error::Value(
                    "a zero-axis Fold carries no per-step state to observe".into(),
                ));
            }
            // An operand another operand is built OVER and passes through whole is that operand's shadow: the maximal
            // one stays, its params go with it, and the body reads the pass-through by the same name.
            let mut slots: Vec<Vec<Name>> = Vec::new();
            let mut cursor = 0;
            for edge in &operands {
                let width = edge.exposes().len();
                slots.push(
                    lift.params()
                        .get(cursor..cursor + width)
                        .unwrap_or(&[])
                        .to_vec(),
                );
                cursor += width;
            }
            let shadowed: HashSet<usize> = (0..operands.len())
                .filter(|&index| {
                    let edge = &operands[index];
                    slots[index] == edge.exposes()
                        && operands.iter().enumerate().any(|(position, other)| {
                            !other.ptr_eq(edge)
                                && other.0.operands.iter().any(|held| held.ptr_eq(edge))
                                && {
                                    let mine: HashSet<&Name> = edge.exposes().iter().collect();
                                    let theirs: HashSet<&Name> = other.exposes().iter().collect();
                                    mine.len() < theirs.len() && mine.is_subset(&theirs)
                                }
                                && slots[position] == other.exposes()
                        })
                })
                .collect();
            if !shadowed.is_empty() {
                let mut kept = Vec::new();
                let mut params = Vec::new();
                let mut cursor = 0;
                for (index, edge) in operands.iter().enumerate() {
                    let width = edge.exposes().len();
                    if !shadowed.contains(&index) {
                        kept.push(edge.clone());
                        params.extend_from_slice(&lift.params()[cursor..cursor + width]);
                    }
                    cursor += width;
                }
                params.extend_from_slice(&lift.params()[cursor..]);
                operands = kept;
                lift = Lambda::new(params, lift.body(), lift.results().to_vec())?;
            }
            let arity: usize = operands.iter().map(|edge| edge.exposes().len()).sum();
            if lift.params().len() < arity {
                return Err(Error::Value(format!(
                    "lift binds {} params for {arity} operand result components",
                    lift.params().len()
                )));
            }
            return Ok(Fold::formed(Parts {
                operands,
                lift: Some(lift),
                init,
                cells,
                base: None,
                twist,
                observe,
            }));
        };
        let n = init.len();
        if base.params().len() != 2 * n || base.results().len() != n || base.components().is_none()
        {
            return Err(Error::Value(format!(
                "Fold base must be the componentwise ⊕ at arity {n}: params={:?} results={:?}",
                base.params(),
                base.results()
            )));
        }
        if lift.params().is_empty() {
            return Err(Error::Value(
                "a reducing lift binds its iteration var first".into(),
            ));
        }
        let arity: usize = operands.iter().map(|edge| edge.exposes().len()).sum();
        if lift.params().len() < 1 + arity {
            return Err(Error::Value(format!(
                "lift binds {} params after the axis for {arity} operand result components",
                lift.params().len() - 1
            )));
        }
        let ops = base.components().unwrap_or_default();
        let carries = !ops.is_empty() && ops.iter().all(|op| op.name() == "next");
        if carries {
            if lift.results().len() < n {
                return Err(Error::Value(
                    "a carried state's lift defines every state's next value first".into(),
                ));
            }
            if twist.is_some() || observe.is_some() {
                return Err(Error::Value(
                    "a carried state streams its step's own results; it is neither twisted nor observed".into(),
                ));
            }
            let states: HashSet<Name> = base.results().iter().copied().collect();
            let mut pending: Vec<Fold> = operands.clone();
            let mut seen: HashSet<usize> = HashSet::new();
            while let Some(edge) = pending.pop() {
                if !seen.insert(edge.address()) {
                    continue;
                }
                if let Some(read) = edge.as_carrier_read()
                    && let Statement::Pre(pre) = &**read
                    && states.contains(&pre.carrier)
                    && pre.index.len() != cells.len()
                {
                    return Err(Error::Value(format!(
                        "carrier read {}[{}] against a state with {} cell axes",
                        pre.carrier,
                        pre.index.len(),
                        cells.len()
                    )));
                }
                pending.extend(edge.0.operands.iter().cloned());
            }
        } else {
            if lift.results().len() != n {
                return Err(Error::Value("one lift result per monoid component".into()));
            }
            if !cells.is_empty() {
                return Err(Error::Value(
                    "cell axes belong to a carried state: a fold whose ⊕ is `next`".into(),
                ));
            }
        }
        let axis = lift.params()[0];
        // CANONICAL ORIENTATION of a bilinear term: A, the operand every product multiplies, comes first. With several
        // channels A is the argument the products share; with one product the layout decides (A reads `[…, k]`).
        if operands.len() >= 2
            && !lift.body().is_empty()
            && lift.body().stmts().iter().all(|stmt| binary_assign(stmt))
        {
            let bindings = bindings_of(&operands, lift.params(), true);
            let by_name: HashMap<Name, usize> =
                bindings.iter().map(|b| (b.param, b.operand)).collect();
            let arguments: Vec<Vec<Name>> = lift
                .body()
                .stmts()
                .iter()
                .map(|stmt| operand_names(stmt))
                .collect();
            let a_edge: Option<usize> = if arguments.len() > 1 {
                let shared: Vec<usize> = arguments[0]
                    .iter()
                    .filter(|name| {
                        by_name.contains_key(name)
                            && arguments[1..].iter().all(|args| args.contains(name))
                    })
                    .map(|name| by_name[name])
                    .collect();
                if shared.len() == 1 {
                    Some(shared[0])
                } else {
                    None
                }
            } else {
                let pair: Vec<usize> = arguments[0]
                    .iter()
                    .filter_map(|name| by_name.get(name).copied())
                    .collect();
                let k_last_of = |edge: &Fold| -> Result<bool> {
                    let Some(load) = edge.as_slab() else {
                        return Ok(false);
                    };
                    let Statement::Load(load) = &**load else {
                        return Ok(false);
                    };
                    let last = load.index.last().ok_or_else(|| {
                        Error::Value("a slab with no index has no last coordinate".into())
                    })?;
                    Ok(last.free_vars().contains(&axis))
                };
                let slabs =
                    pair.len() == 2 && pair.iter().all(|&e| operands[e].as_slab().is_some());
                let mut k_last: Vec<usize> = Vec::new();
                if slabs {
                    for &e in &pair {
                        if k_last_of(&operands[e])? {
                            k_last.push(e);
                        }
                    }
                }
                let rows: HashSet<Name> = if slabs {
                    let (a, b) = (operands[pair[0]].free_axes(), operands[pair[1]].free_axes());
                    a.symmetric_difference(b).copied().collect()
                } else {
                    HashSet::new()
                };
                let key = |e: &usize| -> String {
                    operands[*e]
                        .free_axes()
                        .iter()
                        .filter(|name| rows.contains(name))
                        .map(|name| name.trim_start_matches('_').to_string())
                        .min()
                        .unwrap_or_default()
                };
                k_last.sort_by_key(key);
                let mut a_edge = if pair.len() == 2 && !k_last.is_empty() {
                    Some(k_last[0])
                } else {
                    None
                };
                let loaded: Vec<usize> = pair
                    .iter()
                    .copied()
                    .filter(|&e| operands[e].as_slab().is_some())
                    .collect();
                if pair.len() == 2 && loaded.len() == 1 && !k_last_of(&operands[loaded[0]])? {
                    a_edge = pair
                        .iter()
                        .copied()
                        .find(|&e| !operands[e].ptr_eq(&operands[loaded[0]]));
                }
                a_edge
            };
            if let Some(a) = a_edge
                && !operands[0].ptr_eq(&operands[a])
            {
                let a_term = operands[a].clone();
                let mut reordered = vec![a_term.clone()];
                reordered.extend(
                    operands
                        .iter()
                        .filter(|edge| !edge.ptr_eq(&a_term))
                        .cloned(),
                );
                let mut bound = vec![axis];
                for edge in &reordered {
                    for binding in &bindings {
                        if operands[binding.operand].ptr_eq(edge) {
                            bound.push(binding.param);
                        }
                    }
                }
                bound.extend_from_slice(&lift.params()[1 + arity..]);
                operands = reordered;
                lift = Lambda::new(bound, lift.body(), lift.results().to_vec())?;
            }
        }
        let fold = Fold::formed(Parts {
            operands,
            lift: Some(lift),
            init,
            cells,
            base: Some(base),
            twist,
            observe,
        });
        if let Some(observe) = &fold.0.observe {
            let combine = fold
                .combine()?
                .ok_or_else(|| Error::Value("an observed fold has a combine".into()))?;
            if combine.components().is_none() {
                return Err(Error::Value(
                    "a twisted carrier does not support a per-step observer: its state is rescaled, never a stream of \
                     partials"
                        .into(),
                ));
            }
            let mut expected = vec![axis];
            expected.extend_from_slice(combine.results());
            if observe.params() != expected.as_slice() {
                return Err(Error::Value(format!(
                    "observer params {:?} must bind the iteration var then the carried state {expected:?} positionally",
                    observe.params()
                )));
            }
            let defined: HashSet<Name> = observe
                .body()
                .stmts()
                .iter()
                .flat_map(|stmt| stmt.defines())
                .collect();
            if !observe
                .results()
                .iter()
                .all(|result| defined.contains(result))
            {
                return Err(Error::Value(
                    "observer results must be FRESH names its body defines — never the carried state itself".into(),
                ));
            }
        }
        Ok(fold)
    }

    /// The term over `parts` taken as formed: what a rewrite that keeps every formation fact builds.
    pub fn formed(parts: Parts) -> Fold {
        let Parts {
            operands,
            lift,
            init,
            cells,
            base,
            twist,
            observe,
        } = parts;
        Fold(Arc::new(Data {
            operands,
            lift: lift.expect("a formed Fold has a lift"),
            init,
            cells,
            base,
            twist,
            observe,
            hash: OnceLock::new(),
            exposes: OnceLock::new(),
            bindings: OnceLock::new(),
            applied: OnceLock::new(),
            carries: OnceLock::new(),
            combine: OnceLock::new(),
            free_axes: OnceLock::new(),
            canonical: OnceLock::new(),
        }))
    }

    /// The parts this term was formed from, for a rebuild that changes some of them.
    pub fn parts(&self) -> Parts {
        Parts {
            operands: self.0.operands.clone(),
            lift: Some(self.0.lift.clone()),
            init: self.0.init.clone(),
            cells: self.0.cells.clone(),
            base: self.0.base.clone(),
            twist: self.0.twist.clone(),
            observe: self.0.observe.clone(),
        }
    }

    /// The shared data: what a table keyed per term holds a weak handle on.
    pub fn arc(&self) -> &Arc<Data> {
        &self.0
    }

    pub fn ptr_eq(&self, other: &Fold) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// The address of this term's shared data: what a walk keys the terms it has seen by.
    pub fn address(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }

    /// A weak handle on this term, for a table keyed by [`Fold::address`] to tell a live entry from a dead one.
    pub fn downgrade(&self) -> std::sync::Weak<Data> {
        Arc::downgrade(&self.0)
    }

    pub fn operands(&self) -> &[Fold] {
        &self.0.operands
    }

    pub fn lift(&self) -> &Lambda {
        &self.0.lift
    }

    pub fn init(&self) -> &[Seed] {
        &self.0.init
    }

    pub fn cells(&self) -> &[Name] {
        &self.0.cells
    }

    pub fn base(&self) -> Option<&Lambda> {
        self.0.base.as_ref()
    }

    pub fn twist(&self) -> Option<&Twist> {
        self.0.twist.as_ref()
    }

    pub fn observe(&self) -> Option<&Lambda> {
        self.0.observe.as_ref()
    }

    fn structural_hash(&self) -> u64 {
        *self.0.hash.get_or_init(|| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            self.0.lift.hash(&mut hasher);
            self.0.init.hash(&mut hasher);
            self.0.cells.hash(&mut hasher);
            self.0.base.hash(&mut hasher);
            self.0
                .twist
                .as_ref()
                .map(|t| (t.recipe.name.clone(), t.channels.clone()))
                .hash(&mut hasher);
            self.0.observe.hash(&mut hasher);
            for edge in &self.0.operands {
                edge.structural_hash().hash(&mut hasher);
            }
            hasher.finish()
        })
    }

    /// The name of the axis this term BINDS — its lift's iteration var — or `None` for a zero-axis term.
    pub fn axis(&self) -> Option<Name> {
        self.0.base.as_ref().map(|_| self.0.lift.params()[0])
    }

    /// The positional binding of the lift to the operands: for every operand-bound param in order, the param, the
    /// operand and the component of it the param binds.
    pub fn bindings(&self) -> &[Binding] {
        self.0.bindings.get_or_init(|| {
            bindings_of(
                &self.0.operands,
                self.0.lift.params(),
                self.0.base.is_some(),
            )
        })
    }

    /// The lift APPLIED to the operands: every operand-bound param spelled as the operand result it binds — the one
    /// reading a renderer of this term's statements takes.
    pub fn applied(&self) -> Result<&Lambda> {
        if let Some(found) = self.0.applied.get() {
            return Ok(found);
        }
        let mut mapping: HashMap<Name, Name> = HashMap::new();
        for binding in self.bindings() {
            mapping.insert(
                binding.param,
                self.0.operands[binding.operand].exposes()[binding.component],
            );
        }
        let applied = self.0.lift.rename(&mapping)?;
        Ok(self.0.applied.get_or_init(|| applied))
    }

    /// The result names this term defines for its consumer, one per produced component: a reducing term its carried
    /// state (an observed one its observer's values beside it, a carried one its per-step outputs), a zero-axis term
    /// its applied lift's results.
    pub fn exposes(&self) -> &[Name] {
        self.0.exposes.get_or_init(|| match &self.0.base {
            None => self
                .applied()
                .map(|a| a.results().to_vec())
                .unwrap_or_default(),
            Some(base) => {
                let state = base.results();
                if self.carries() {
                    let mut out = state.to_vec();
                    if let Ok(applied) = self.applied() {
                        out.extend_from_slice(applied.results().get(state.len()..).unwrap_or(&[]));
                    }
                    out
                } else if let Some(observe) = &self.0.observe {
                    let mut out = state.to_vec();
                    out.extend_from_slice(observe.results());
                    out
                } else {
                    state.to_vec()
                }
            }
        })
    }

    /// Whether this fold CARRIES A STATE: its ⊕ is the action `next` on every component.
    pub fn carries(&self) -> bool {
        *self.0.carries.get_or_init(|| {
            let ops = self
                .0
                .base
                .as_ref()
                .and_then(|base| base.components())
                .unwrap_or_default();
            !ops.is_empty() && ops.iter().all(|op| op.name() == "next")
        })
    }

    /// THE ⊕ this fold folds with, `None` at zero axes: the base itself for a planar fold, the recipe's stable
    /// conjugate over the same states for a twisted one.
    pub fn combine(&self) -> Result<Option<Lambda>> {
        if let Some(found) = self.0.combine.get() {
            return Ok(found.clone());
        }
        let combine = match (&self.0.base, &self.0.twist) {
            (Some(base), Some(twist)) => Some(twist.program(base.results())?),
            (base, _) => base.clone(),
        };
        Ok(self.0.combine.get_or_init(|| combine).clone())
    }

    /// The one `Load` of a SLAB — no operands, no monoid, a body of one load — or `None`.
    pub fn as_slab(&self) -> Option<&Arc<Statement>> {
        let body = self.0.lift.body();
        if !self.0.operands.is_empty() || self.0.base.is_some() || body.len() != 1 {
            return None;
        }
        matches!(&*body.stmts()[0], Statement::Load(_)).then(|| &body.stmts()[0])
    }

    /// The one `Pre` of a CARRIER READ — a slab over an enclosing fold's carried state at some cell — or `None`.
    pub fn as_carrier_read(&self) -> Option<&Arc<Statement>> {
        let body = self.0.lift.body();
        if !self.0.operands.is_empty() || self.0.base.is_some() || body.len() != 1 {
            return None;
        }
        matches!(&*body.stmts()[0], Statement::Pre(_)).then(|| &body.stmts()[0])
    }

    /// The coordinates this term is evaluated over as seen from outside: the lift's trailing params (past the axis
    /// and the operand binding) and its operands' free coordinates, less what the operands produce, the axis it binds
    /// and a carried state's cells. A carried state's operands are its own loops' business and add nothing.
    pub fn free_axes(&self) -> &HashSet<Name> {
        self.0.free_axes.get_or_init(|| {
            let lead = usize::from(self.axis().is_some()) + self.bindings().len();
            let mut space: HashSet<Name> = self
                .0
                .lift
                .params()
                .get(lead..)
                .unwrap_or(&[])
                .iter()
                .copied()
                .collect();
            if let Some(read) = self.as_carrier_read()
                && let Statement::Pre(pre) = &**read
            {
                space.remove(&pre.carrier);
            }
            if !self.carries() {
                for edge in &self.0.operands {
                    space.extend(edge.free_axes().iter().copied());
                }
            }
            for edge in &self.0.operands {
                for name in edge.exposes() {
                    space.remove(name);
                }
            }
            if let Some(axis) = self.axis() {
                space.remove(&axis);
            }
            for cell in &self.0.cells {
                space.remove(cell);
            }
            space
        })
    }

    /// The α-canonical form of this TERM, a `Fold`: every bound name renumbered positionally (the axis `_a0`, the
    /// trailing coordinates `_a<i>`, what it exposes `_r<i>`, what it binds from each operand `_e<i>`, its own
    /// definitions `_v<i>`, the ⊕'s second operand `_c<i>`), each operand canonicalized first. Free names pass through,
    /// so equal canonical forms mean equal value under the same environment. Computed once.
    pub fn canonical(&self) -> Result<Fold> {
        if let Some(found) = self.0.canonical.get() {
            return Ok(found.clone());
        }
        let lift = &self.0.lift;
        let lead: Vec<Name> = self.axis().into_iter().collect();
        let mut mapping: HashMap<Name, Name> = HashMap::new();
        let trailing = lift
            .params()
            .get(lead.len() + self.bindings().len()..)
            .unwrap_or(&[]);
        for (index, name) in lead.iter().chain(trailing).enumerate() {
            mapping.insert(*name, Name::new(&format!("_a{index}")));
        }
        let own: Vec<Name> = if self.0.base.is_none() {
            lift.results().to_vec()
        } else {
            self.exposes().to_vec()
        };
        for (index, name) in own.iter().enumerate() {
            mapping.insert(*name, Name::new(&format!("_r{index}")));
        }
        for (index, binding) in self.bindings().iter().enumerate() {
            mapping
                .entry(binding.param)
                .or_insert_with(|| Name::new(&format!("_e{index}")));
        }
        let mut counter = 0;
        for here in lift.body().iter_refs() {
            for name in here.stmt().defines() {
                if !mapping.contains_key(&name) {
                    mapping.insert(name, Name::new(&format!("_v{counter}")));
                    counter += 1;
                }
            }
        }
        let base = match &self.0.base {
            None => None,
            Some(base) => {
                let mut own = mapping.clone();
                for name in base.params() {
                    let next = Name::new(&format!("_c{}", own.len()));
                    own.entry(*name).or_insert(next);
                }
                Some(base.rename(&own)?)
            }
        };
        let operands = self
            .0
            .operands
            .iter()
            .map(|edge| edge.canonical())
            .collect::<Result<Vec<_>>>()?;
        let observe = self
            .0
            .observe
            .as_ref()
            .map(|o| o.rename(&mapping))
            .transpose()?;
        let canonical = Fold::new(Parts {
            operands,
            lift: Some(lift.rename(&mapping)?),
            init: self.0.init.clone(),
            cells: self.0.cells.clone(),
            base,
            twist: self.0.twist.clone(),
            observe,
        })?;
        Ok(self.0.canonical.get_or_init(|| canonical).clone())
    }

    /// The base ⊕ components, when the base is componentwise.
    pub fn ops(&self) -> Option<Vec<ElementwiseOp>> {
        self.0.base.as_ref().and_then(|base| base.components())
    }

    /// The parts with the body of the lift replaced: a convenience for rebuilds that keep everything else.
    pub fn with_lift_body(&self, body: &Body) -> Result<Fold> {
        let mut parts = self.parts();
        parts.lift = Some(Lambda::new(
            self.0.lift.params().to_vec(),
            body,
            self.0.lift.results().to_vec(),
        )?);
        Fold::new(parts)
    }
}

/// The positional binding of `params` to `operands`: past the iteration var when the term reduces, one param per
/// operand result component in order.
fn bindings_of(operands: &[Fold], params: &[Name], reduces: bool) -> Vec<Binding> {
    let params = &params[usize::from(reduces).min(params.len())..];
    let mut out = Vec::new();
    for (operand, edge) in operands.iter().enumerate() {
        for component in 0..edge.exposes().len() {
            if let Some(param) = params.get(out.len()) {
                out.push(Binding {
                    param: *param,
                    operand,
                    component,
                });
            }
        }
    }
    out
}
