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
use std::sync::{Arc, Mutex, OnceLock};

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
    runs_on: bool,
    hash: u64,
    exposes: OnceLock<Vec<Name>>,
    bindings: OnceLock<Vec<Binding>>,
    applied: OnceLock<Lambda>,
    carries: OnceLock<bool>,
    combine: OnceLock<Option<Lambda>>,
    free_axes: OnceLock<HashSet<Name>>,
    canonical: OnceLock<Fold>,
    readings: crate::readings::Memos,
    normal: Mutex<Vec<Vec<Name>>>,
    lowered: Mutex<Vec<(crate::lower::Request, emmy_ir::body::Body)>>,
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
    /// A fold that carries a state and does not seed it: a further level of the steps of the fold it sits in, the
    /// state running on through it from where that fold's step left it. `Loop.seed = false`, said of the term.
    pub runs_on: bool,
}

impl PartialEq for Fold {
    fn eq(&self, other: &Fold) -> bool {
        if Arc::ptr_eq(&self.0, &other.0) {
            return true;
        }
        let (a, b) = (&*self.0, &*other.0);
        a.hash == b.hash
            && a.lift == b.lift
            && a.init == b.init
            && a.cells == b.cells
            && a.base == b.base
            && a.twist == b.twist
            && a.observe == b.observe
            && a.runs_on == b.runs_on
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

/// Whether a componentwise ⊕ is the action `next` on every component: what a carried state folds with.
fn is_action(ops: &[ElementwiseOp]) -> bool {
    !ops.is_empty() && ops.iter().all(|op| op.name() == "next")
}

/// The lift params one operand each, in order: the positional binding past `lead` params.
fn operand_slots<'a>(operands: &[Fold], params: &'a [Name]) -> Vec<&'a [Name]> {
    let mut cursor = 0;
    operands
        .iter()
        .map(|edge| {
            let width = edge.exposes().len();
            let slot = params.get(cursor..cursor + width).unwrap_or(&[]);
            cursor += width;
            slot
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------------------------
// The canonical rewrites formation makes: the same term whichever spelling it was built from.
// ---------------------------------------------------------------------------------------------------------------

/// A zero-axis term without its SHADOWED operands. An operand another operand is built OVER and passes through
/// whole is that operand's shadow: the maximal one stays, its params go with it, and the body reads the pass-through
/// by the same name.
fn without_shadowed(operands: Vec<Fold>, lift: Lambda) -> Result<(Vec<Fold>, Lambda)> {
    let slots = operand_slots(&operands, lift.params());
    let whole = |index: usize| slots[index] == operands[index].exposes();
    let shadowed: Vec<bool> = operands
        .iter()
        .enumerate()
        .map(|(index, edge)| {
            whole(index)
                && operands.iter().enumerate().any(|(position, other)| {
                    !other.ptr_eq(edge)
                        && other.operands().iter().any(|held| held.ptr_eq(edge))
                        && other.exposes().len() > edge.exposes().len()
                        && edge
                            .exposes()
                            .iter()
                            .all(|name| other.exposes().contains(name))
                        && whole(position)
                })
        })
        .collect();
    if !shadowed.contains(&true) {
        return Ok((operands, lift));
    }
    let bound: usize = slots.iter().map(|slot| slot.len()).sum();
    let params: Vec<Name> = slots
        .iter()
        .zip(&shadowed)
        .filter(|(_, shadowed)| !**shadowed)
        .flat_map(|(slot, _)| slot.iter().copied())
        .chain(lift.params()[bound..].iter().copied())
        .collect();
    let lift = Lambda::new(params, lift.body(), lift.results().to_vec())?;
    let kept = operands
        .into_iter()
        .zip(shadowed)
        .filter(|(_, shadowed)| !shadowed)
        .map(|(edge, _)| edge)
        .collect();
    Ok((kept, lift))
}

/// Whether a slab's last index coordinate reads `axis`.
fn k_last(edge: &Fold, axis: Name) -> Result<bool> {
    let Some(load) = edge.as_slab() else {
        return Ok(false);
    };
    let Statement::Load(load) = &**load else {
        return Ok(false);
    };
    let last = load
        .index
        .last()
        .ok_or_else(|| Error::Value("a slab with no index has no last coordinate".into()))?;
    Ok(last.free_vars().contains(&axis))
}

/// The operand of a bilinear lift that is A, the one every product multiplies, or `None` when the lift names none.
/// With several channels A is the argument the products share; with one product the layout decides (A reads
/// `[…, k]`).
fn a_operand(operands: &[Fold], lift: &Lambda) -> Result<Option<usize>> {
    let axis = lift.params()[0];
    let by_name: HashMap<Name, usize> = bindings_of(operands, lift.params(), true)
        .iter()
        .map(|b| (b.param, b.operand))
        .collect();
    let arguments: Vec<Vec<Name>> = lift
        .body()
        .stmts()
        .iter()
        .map(|stmt| operand_names(stmt))
        .collect();
    if arguments.len() > 1 {
        let shared: Vec<usize> = arguments[0]
            .iter()
            .filter(|name| arguments[1..].iter().all(|args| args.contains(name)))
            .filter_map(|name| by_name.get(name).copied())
            .collect();
        return Ok((shared.len() == 1).then(|| shared[0]));
    }
    let pair: Vec<usize> = arguments[0]
        .iter()
        .filter_map(|name| by_name.get(name).copied())
        .collect();
    if pair.len() != 2 {
        return Ok(None);
    }
    let loaded: Vec<usize> = pair
        .iter()
        .copied()
        .filter(|&e| operands[e].as_slab().is_some())
        .collect();
    // One slab, and it does not read k last: the computed operand is A.
    if loaded.len() == 1 && !k_last(&operands[loaded[0]], axis)? {
        return Ok(pair
            .iter()
            .copied()
            .find(|&e| !operands[e].ptr_eq(&operands[loaded[0]])));
    }
    if loaded.len() != 2 {
        return Ok(None);
    }
    let mut candidates = Vec::new();
    for &e in &pair {
        if k_last(&operands[e], axis)? {
            candidates.push(e);
        }
    }
    // Both slabs read k last: A is the one whose own row is named first.
    let (a, b) = (operands[pair[0]].free_axes(), operands[pair[1]].free_axes());
    let row = |e: &usize| -> String {
        operands[*e]
            .free_axes()
            .iter()
            .filter(|name| a.contains(*name) != b.contains(*name))
            .map(|name| name.trim_start_matches('_').to_string())
            .min()
            .unwrap_or_default()
    };
    candidates.sort_by_key(row);
    Ok(candidates.first().copied())
}

/// A bilinear term with its A operand FIRST: the canonical orientation. A term whose lift is not all binary
/// operations over two operands or more is left as it stands.
fn a_first(operands: Vec<Fold>, lift: Lambda) -> Result<(Vec<Fold>, Lambda)> {
    let bilinear = operands.len() >= 2
        && !lift.body().is_empty()
        && lift.body().stmts().iter().all(|stmt| binary_assign(stmt));
    let a = match bilinear {
        true => a_operand(&operands, &lift)?,
        false => None,
    };
    let Some(a) = a.filter(|a| !operands[*a].ptr_eq(&operands[0])) else {
        return Ok((operands, lift));
    };
    let a_term = operands[a].clone();
    let reordered: Vec<Fold> = std::iter::once(a_term.clone())
        .chain(
            operands
                .iter()
                .filter(|edge| !edge.ptr_eq(&a_term))
                .cloned(),
        )
        .collect();
    let bindings = bindings_of(&operands, lift.params(), true);
    let arity: usize = operands.iter().map(|edge| edge.exposes().len()).sum();
    let params: Vec<Name> = std::iter::once(lift.params()[0])
        .chain(reordered.iter().flat_map(|edge| {
            bindings
                .iter()
                .filter(|b| operands[b.operand].ptr_eq(edge))
                .map(|b| b.param)
        }))
        .chain(lift.params()[1 + arity..].iter().copied())
        .collect();
    let lift = Lambda::new(params, lift.body(), lift.results().to_vec())?;
    Ok((reordered, lift))
}

// ---------------------------------------------------------------------------------------------------------------
// The formation checks: what a term must hold. They refuse; they never repair.
// ---------------------------------------------------------------------------------------------------------------

fn check_zero_axis(parts: &Parts, lift: &Lambda) -> Result<()> {
    if !parts.init.is_empty() || parts.twist.is_some() {
        return Err(Error::Value(
            "a zero-axis Fold carries no monoid to seed or twist".into(),
        ));
    }
    if parts.observe.is_some() {
        return Err(Error::Value(
            "a zero-axis Fold carries no per-step state to observe".into(),
        ));
    }
    let arity: usize = parts.operands.iter().map(|edge| edge.exposes().len()).sum();
    if lift.params().len() < arity {
        return Err(Error::Value(format!(
            "lift binds {} params for {arity} operand result components",
            lift.params().len()
        )));
    }
    Ok(())
}

fn check_reduce(parts: &Parts, lift: &Lambda, base: &Lambda) -> Result<()> {
    let n = parts.init.len();
    let Some(ops) = base
        .components()
        .filter(|_| base.params().len() == 2 * n && base.results().len() == n)
    else {
        return Err(Error::Value(format!(
            "Fold base must be the componentwise ⊕ at arity {n}: params={:?} results={:?}",
            base.params(),
            base.results()
        )));
    };
    if lift.params().is_empty() {
        return Err(Error::Value(
            "a reducing lift binds its iteration var first".into(),
        ));
    }
    let arity: usize = parts.operands.iter().map(|edge| edge.exposes().len()).sum();
    if lift.params().len() < 1 + arity {
        return Err(Error::Value(format!(
            "lift binds {} params after the axis for {arity} operand result components",
            lift.params().len() - 1
        )));
    }
    if !is_action(&ops) {
        if lift.results().len() != n {
            return Err(Error::Value("one lift result per monoid component".into()));
        }
        if !parts.cells.is_empty() {
            return Err(Error::Value(
                "cell axes belong to a carried state: a fold whose ⊕ is `next`".into(),
            ));
        }
        return Ok(());
    }
    if lift.results().len() < n {
        return Err(Error::Value(
            "a carried state's lift defines every state's next value first".into(),
        ));
    }
    if parts.twist.is_some() || parts.observe.is_some() {
        return Err(Error::Value(
            "a carried state streams its step's own results; it is neither twisted nor observed"
                .into(),
        ));
    }
    // Every read of the carried state, anywhere under it, indexes every cell axis.
    let mut pending: Vec<Fold> = parts.operands.clone();
    let mut seen: HashSet<usize> = HashSet::new();
    while let Some(edge) = pending.pop() {
        if !seen.insert(edge.address()) {
            continue;
        }
        if let Some(read) = edge.as_carrier_read()
            && let Statement::Pre(pre) = &**read
            && base.results().contains(&pre.carrier)
            && pre.index.len() != parts.cells.len()
        {
            return Err(Error::Value(format!(
                "carrier read {}[{}] against a state with {} cell axes",
                pre.carrier,
                pre.index.len(),
                parts.cells.len()
            )));
        }
        pending.extend(edge.operands().iter().cloned());
    }
    Ok(())
}

impl Fold {
    /// An observer binds the iteration var then the carried state, and defines fresh names.
    fn check_observer(&self, observe: &Lambda) -> Result<()> {
        let combine = self
            .combine()?
            .ok_or_else(|| Error::Value("an observed fold has a combine".into()))?;
        if combine.components().is_none() {
            return Err(Error::Value(
                "a twisted carrier does not support a per-step observer: its state is rescaled, never a stream of \
                 partials"
                    .into(),
            ));
        }
        let expected: Vec<Name> = self
            .axis()
            .into_iter()
            .chain(combine.results().iter().copied())
            .collect();
        if observe.params() != expected.as_slice() {
            return Err(Error::Value(format!(
                "observer params {:?} must bind the iteration var then the carried state {expected:?} positionally",
                observe.params()
            )));
        }
        if !observe
            .results()
            .iter()
            .all(|result| observe.body().defined().contains(result))
        {
            return Err(Error::Value(
                "observer results must be FRESH names its body defines — never the carried state itself".into(),
            ));
        }
        Ok(())
    }

    /// The term over `parts`: put in its canonical spelling (a zero-axis term loses its shadowed operands, a
    /// bilinear one takes its A operand first), then checked. The checks refuse rather than repair.
    pub fn new(parts: Parts) -> Result<Fold> {
        let mut parts = parts;
        let lift = parts
            .lift
            .take()
            .ok_or_else(|| Error::Type("a Fold takes a lift".into()))?;
        let operands = std::mem::take(&mut parts.operands);
        let (operands, lift) = match &parts.base {
            None => without_shadowed(operands, lift)?,
            Some(base) => {
                // The orientation reads the binding, so the checks that make it readable come first.
                parts.operands = operands;
                check_reduce(&parts, &lift, base)?;
                a_first(std::mem::take(&mut parts.operands), lift)?
            }
        };
        parts.operands = operands;
        if parts.base.is_none() {
            check_zero_axis(&parts, &lift)?;
        }
        parts.lift = Some(lift);
        let fold = Fold::formed(parts);
        // What the term exposes is read off its applied lift: a lift that does not apply is refused here.
        fold.applied()?;
        if fold.runs_on() && !fold.carries() {
            return Err(Error::Value(
                "only a fold that carries a state can run on without seeding it".into(),
            ));
        }
        if let Some(observe) = fold.observe() {
            fold.check_observer(observe)?;
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
            runs_on,
        } = parts;
        let lift = lift.expect("a formed Fold has a lift");
        let hash = {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            lift.hash(&mut hasher);
            init.hash(&mut hasher);
            cells.hash(&mut hasher);
            base.hash(&mut hasher);
            twist
                .as_ref()
                .map(|t| (t.recipe.name.clone(), t.channels.clone()))
                .hash(&mut hasher);
            observe.hash(&mut hasher);
            runs_on.hash(&mut hasher);
            for edge in &operands {
                edge.0.hash.hash(&mut hasher);
            }
            hasher.finish()
        };
        let data = Data {
            operands,
            lift,
            init,
            cells,
            base,
            twist,
            observe,
            runs_on,
            hash,
            exposes: OnceLock::new(),
            bindings: OnceLock::new(),
            applied: OnceLock::new(),
            carries: OnceLock::new(),
            combine: OnceLock::new(),
            free_axes: OnceLock::new(),
            canonical: OnceLock::new(),
            readings: crate::readings::Memos::default(),
            normal: Mutex::new(Vec::new()),
            lowered: Mutex::new(Vec::new()),
        };
        Fold(Arc::new(data))
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
            runs_on: self.0.runs_on,
        }
    }

    /// Whether this fold carries its state on without seeding it: a further level of the steps of the fold outside.
    pub fn runs_on(&self) -> bool {
        self.0.runs_on
    }

    /// The shared data: what a table keyed per term holds a weak handle on.
    pub fn arc(&self) -> &Arc<Data> {
        &self.0
    }

    /// Whether this tree is known to be its own normal form under the stored names `key` (sorted).
    pub(crate) fn normal_under(&self, key: &[Name]) -> bool {
        self.0
            .normal
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .any(|k| k == key)
    }

    /// Remember that this tree is its own normal form under the stored names `key` (sorted).
    pub(crate) fn remember_normal(&self, key: Vec<Name>) {
        let mut normal = self.0.normal.lock().unwrap_or_else(|e| e.into_inner());
        if !normal.contains(&key) {
            normal.push(key);
        }
    }

    /// The lowerings this term has answered, by request.
    pub(crate) fn lowered(&self) -> &Mutex<Vec<(crate::lower::Request, emmy_ir::body::Body)>> {
        &self.0.lowered
    }

    /// The memo table of the derived readings.
    pub(crate) fn readings(&self) -> &crate::readings::Memos {
        &self.0.readings
    }

    pub fn ptr_eq(&self, other: &Fold) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// The address of this term's shared data: what a walk keys the terms it has seen by.
    pub fn address(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
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
        self.0.hash
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

    /// The applied lift of a formed term: formation refused a lift that does not apply.
    fn applied_checked(&self) -> &Lambda {
        self.applied().expect("formation applied the lift")
    }

    /// The result names this term defines for its consumer, one per produced component: a reducing term its carried
    /// state (an observed one its observer's values beside it, a carried one its per-step outputs), a zero-axis term
    /// its applied lift's results.
    pub fn exposes(&self) -> &[Name] {
        self.0.exposes.get_or_init(|| match &self.0.base {
            None => self.applied_checked().results().to_vec(),
            Some(base) => {
                let state = base.results();
                if self.carries() {
                    let mut out = state.to_vec();
                    out.extend_from_slice(&self.applied_checked().results()[state.len()..]);
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
        *self
            .0
            .carries
            .get_or_init(|| is_action(&self.ops().unwrap_or_default()))
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
                if let std::collections::hash_map::Entry::Vacant(slot) = mapping.entry(name) {
                    slot.insert(Name::new(&format!("_v{counter}")));
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
            runs_on: self.0.runs_on,
        })?;
        Ok(self.0.canonical.get_or_init(|| canonical).clone())
    }

    /// The base ⊕ components, when the base is componentwise.
    pub fn ops(&self) -> Option<Vec<ElementwiseOp>> {
        self.0.base.as_ref().and_then(|base| base.components())
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
