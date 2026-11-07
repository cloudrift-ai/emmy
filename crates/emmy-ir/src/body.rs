//! `Body`: a shared, immutable sequence of statements with the def-use readings, walks and orders every analysis
//! phrases itself over, each computed once on first use and kept with the body. And `Subroutine`, the shared
//! reduction body fusion builds a region with.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

use crate::expr::{Expression, Num};
use crate::kinds::field::{Axes, AxisRef, Names, Visitor};
use crate::kinds::{Kind, Roles, Statement};
use crate::numbering::{Key, expression_key};
use crate::{Error, Name, Result, infallible, locked};

/// What `Body::map` applies: a statement to the statements that replace it, or `None` to drop it.
pub type MapFn<'a> = dyn FnMut(&Arc<Statement>) -> Option<Vec<Arc<Statement>>> + 'a;
pub type TryMapEachFn<'a, E> =
    dyn FnMut(&Arc<Statement>) -> std::result::Result<Arc<Statement>, E> + 'a;
pub type TryMapStmtFn<'a, E> =
    dyn FnMut(&Arc<Statement>) -> std::result::Result<Option<Arc<Statement>>, E> + 'a;
pub type TryMapFn<'a, E> =
    dyn FnMut(&Arc<Statement>) -> std::result::Result<Option<Vec<Arc<Statement>>>, E> + 'a;

/// One statement inside a body: the body that holds it directly and its position there. It holds the statement and
/// only a weak reference to the body, so a body may keep references to its own statements without keeping itself
/// alive.
#[derive(Clone)]
pub struct StmtRef {
    stmt: Arc<Statement>,
    body: std::sync::Weak<BodyData>,
    holder: usize,
    pub index: usize,
}

impl StmtRef {
    pub fn stmt(&self) -> &Arc<Statement> {
        &self.stmt
    }

    /// The body that holds the statement, while it lives.
    pub fn body(&self) -> Option<Body> {
        self.body.upgrade().map(Body)
    }
}

impl std::ops::Deref for StmtRef {
    type Target = Statement;

    fn deref(&self) -> &Statement {
        self.stmt()
    }
}

impl std::fmt::Debug for StmtRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StmtRef({:#x}, {})", self.holder, self.index)
    }
}

impl PartialEq for StmtRef {
    fn eq(&self, other: &Self) -> bool {
        self.holder == other.holder && self.index == other.index
    }
}

impl Eq for StmtRef {}

impl Hash for StmtRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.holder.hash(state);
        self.index.hash(state);
    }
}

pub struct BodyData {
    stmts: Vec<Arc<Statement>>,
    hash: u64,
    side_effects: bool,
    carried_names: OnceLock<Vec<Name>>,
    local_defs: OnceLock<HashSet<Name>>,
    free_ssa: OnceLock<HashSet<Name>>,
    definitions: OnceLock<HashMap<Name, StmtRef>>,
    axis_names: OnceLock<HashSet<Name>>,
    ssa_uses: OnceLock<HashSet<Name>>,
    by_kind: Mutex<HashMap<Vec<Kind>, Vec<StmtRef>>>,
    exact: OnceLock<u64>,
    normalized: OnceLock<Normalized>,
}

/// What a body knows of its normal form once it was taken: that it is its own, or which body it is. Never the body's
/// own `Arc`, which would keep it alive forever.
#[derive(Clone)]
enum Normalized {
    Itself,
    Other(Body),
}

/// A body as a memo key: equal to another when the statements are equal and so is everything body equality leaves
/// out, the exact material: every axis's hint and window (provenance to the body, identity material to what the
/// memos keep), every expression with its literals by bits, and every number a field holds by bits (a signed zero
/// keys apart from zero, every NaN alike, as the digest keys them). The hash is computed once and kept with the body,
/// so no key is a printed text.
#[derive(Clone, Debug)]
pub struct MemoKey(pub Body);

impl PartialEq for MemoKey {
    fn eq(&self, other: &MemoKey) -> bool {
        self.0.ptr_eq(&other.0)
            || (self.0.exact_hash() == other.0.exact_hash()
                && self.0 == other.0
                && self.0.exact_material() == other.0.exact_material())
    }
}

impl Eq for MemoKey {}

impl Hash for MemoKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.exact_hash().hash(state);
    }
}

/// What body equality leaves out, in walk order.
#[derive(Default, PartialEq, Eq, Hash)]
struct Material {
    axes: Vec<ExactAxis>,
    expressions: Vec<Key>,
    numbers: Vec<Key>,
}

/// An axis compared and hashed exactly, hint and window included.
struct ExactAxis(AxisRef);

impl PartialEq for ExactAxis {
    fn eq(&self, other: &ExactAxis) -> bool {
        Arc::ptr_eq(&self.0, &other.0) || self.0.exact_eq(&other.0)
    }
}

impl Eq for ExactAxis {}

impl Hash for ExactAxis {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.exact_hash(state);
    }
}

impl Visitor for Material {
    fn expression(&mut self, _roles: Roles, expression: &Expression) {
        self.expressions
            .push(expression_key(expression, &mut |_| None));
    }
    fn axis(&mut self, _roles: Roles, axis: &AxisRef) {
        self.axes.push(ExactAxis(axis.clone()));
    }
    fn number(&mut self, value: Num) {
        self.numbers.push(Key::num(value));
    }
    fn body(&mut self, _roles: Roles, body: &Body) {
        for stmt in body.stmts() {
            stmt.visit(self);
        }
    }
    fn carried(&mut self, _roles: Roles, stmt: &Arc<Statement>) {
        stmt.visit(self);
    }
}

/// An immutable sequence of statements. Two bodies with equal statements are equal and hash alike; a slice or a
/// concatenation is a body.
#[derive(Clone)]
pub struct Body(pub Arc<BodyData>);

impl std::fmt::Debug for Body {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.stmts().iter()).finish()
    }
}

impl PartialEq for Body {
    fn eq(&self, other: &Self) -> bool {
        if Arc::ptr_eq(&self.0, &other.0) {
            return true;
        }
        self.0.hash == other.0.hash
            && self.len() == other.len()
            && self
                .stmts()
                .iter()
                .zip(other.stmts())
                .all(|(a, b)| Arc::ptr_eq(a, b) || a == b)
    }
}

impl Eq for Body {}

impl Hash for Body {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash.hash(state);
    }
}

impl Default for Body {
    fn default() -> Body {
        Body::empty()
    }
}

impl Body {
    pub fn new(stmts: Vec<Arc<Statement>>) -> Body {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        stmts.len().hash(&mut hasher);
        for stmt in &stmts {
            stmt.hash(&mut hasher);
        }
        let side_effects = stmts.iter().any(|stmt| stmt.has_side_effects());
        Body(Arc::new(BodyData {
            stmts,
            hash: hasher.finish(),
            side_effects,
            carried_names: OnceLock::new(),
            local_defs: OnceLock::new(),
            free_ssa: OnceLock::new(),
            definitions: OnceLock::new(),
            axis_names: OnceLock::new(),
            ssa_uses: OnceLock::new(),
            by_kind: Mutex::new(HashMap::new()),
            exact: OnceLock::new(),
            normalized: OnceLock::new(),
        }))
    }

    /// What the memo key compares beyond equality, collected from the whole subtree.
    fn exact_material(&self) -> Material {
        let mut material = Material::default();
        for stmt in self.stmts() {
            stmt.visit(&mut material);
        }
        material
    }

    /// The hash of the exact form: the content hash with the exact material, computed once.
    fn exact_hash(&self) -> u64 {
        *self.0.exact.get_or_init(|| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            self.0.hash.hash(&mut hasher);
            self.exact_material().hash(&mut hasher);
            hasher.finish()
        })
    }

    /// The normal form this body already knows: itself, or the body normalization gave it.
    pub(crate) fn normal_form(&self) -> Option<Body> {
        self.0.normalized.get().map(|normal| match normal {
            Normalized::Itself => self.clone(),
            Normalized::Other(normal) => normal.clone(),
        })
    }

    /// Remember `normal` as this body's normal form, and `normal` as its own.
    pub(crate) fn remember_normal_form(&self, normal: &Body) {
        normal.mark_normal();
        if !normal.ptr_eq(self) {
            let _ = self.0.normalized.set(Normalized::Other(normal.clone()));
        }
    }

    /// Whether this body is known to be its own normal form.
    pub(crate) fn is_normal(&self) -> bool {
        matches!(self.0.normalized.get(), Some(Normalized::Itself))
    }

    /// Record that this body is its own normal form.
    pub(crate) fn mark_normal(&self) {
        let _ = self.0.normalized.set(Normalized::Itself);
    }

    pub fn from_statements(stmts: impl IntoIterator<Item = Statement>) -> Body {
        Body::new(stmts.into_iter().map(Arc::new).collect())
    }

    pub fn empty() -> Body {
        static EMPTY: OnceLock<Body> = OnceLock::new();
        EMPTY.get_or_init(|| Body::new(Vec::new())).clone()
    }

    pub fn ptr_eq(&self, other: &Body) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    pub fn stmts(&self) -> &[Arc<Statement>] {
        &self.0.stmts
    }

    pub fn len(&self) -> usize {
        self.0.stmts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.stmts.is_empty()
    }

    pub fn content_hash(&self) -> u64 {
        self.0.hash
    }

    pub fn at(&self, index: usize) -> StmtRef {
        StmtRef {
            stmt: self.stmts()[index].clone(),
            body: Arc::downgrade(&self.0),
            holder: Arc::as_ptr(&self.0) as usize,
            index,
        }
    }

    pub fn slice(&self, start: usize, end: usize) -> Body {
        Body::new(self.0.stmts[start..end].to_vec())
    }

    pub fn concat(&self, other: &[Arc<Statement>]) -> Body {
        let mut stmts = self.0.stmts.clone();
        stmts.extend(other.iter().cloned());
        Body::new(stmts)
    }

    /// True iff a statement of this body, at any depth, writes a buffer.
    pub fn has_side_effects(&self) -> bool {
        self.0.side_effects
    }

    /// The statement positions of this body, then of every nested body, pre-order.
    pub fn walk(&self, f: &mut dyn FnMut(&StmtRef)) {
        for index in 0..self.len() {
            let here = self.at(index);
            f(&here);
            for child in here.stmt().nested() {
                child.walk(f);
            }
        }
    }

    pub fn iter_refs(&self) -> Vec<StmtRef> {
        let mut out = Vec::new();
        self.walk(&mut |here| out.push(here.clone()));
        out
    }

    /// Every statement at any depth, pre-order.
    pub fn iter_stmts(&self) -> Vec<Arc<Statement>> {
        let mut out = Vec::new();
        self.walk(&mut |here| out.push(here.stmt().clone()));
        out
    }

    /// `f` applied to each statement; this body itself when every statement comes back as it was.
    pub fn map_each(&self, f: &mut dyn FnMut(&Arc<Statement>) -> Arc<Statement>) -> Body {
        infallible(self.try_map_each(&mut |stmt| Ok::<_, std::convert::Infallible>(f(stmt))))
    }

    /// `f` on each statement of this body, none descended into; an error ends the walk.
    pub fn try_map_each<E>(&self, f: &mut TryMapEachFn<'_, E>) -> std::result::Result<Body, E> {
        let stmts: std::result::Result<Vec<Arc<Statement>>, E> =
            self.stmts().iter().map(f).collect();
        Ok(self.rebuilt(stmts?))
    }

    /// This body when `stmts` are its own statements, else a body of `stmts`: a step that changed nothing keeps the
    /// body and the readings it holds.
    pub fn rebuilt(&self, stmts: Vec<Arc<Statement>>) -> Body {
        if stmts.len() == self.len()
            && stmts
                .iter()
                .zip(self.stmts())
                .all(|(new, old)| Arc::ptr_eq(new, old))
        {
            self.clone()
        } else {
            Body::new(stmts)
        }
    }

    /// The recursive 1:N transformer: each block's bodies are mapped first, then `f` on the children-rewritten
    /// statement, which it keeps, drops (`None`) or replaces by several. An error ends the walk.
    pub fn try_map<E>(&self, f: &mut TryMapFn<'_, E>) -> std::result::Result<Body, E> {
        let mut out = Vec::new();
        for stmt in self.stmts() {
            let nested = stmt.nested();
            let descended = if nested.is_empty() {
                stmt.clone()
            } else {
                let bodies: std::result::Result<Vec<Body>, E> =
                    nested.iter().map(|body| body.try_map(f)).collect();
                Statement::rebuild_bodies(stmt, bodies?)
            };
            if let Some(replacement) = f(&descended)? {
                out.extend(replacement);
            }
        }
        Ok(self.rebuilt(out))
    }

    /// `try_map` with a function that cannot fail.
    pub fn map(&self, f: &mut MapFn<'_>) -> Body {
        infallible(self.try_map(&mut |stmt| Ok::<_, std::convert::Infallible>(f(stmt))))
    }

    /// `try_map` whose function keeps, drops or replaces one statement by one.
    pub fn try_map_stmts<E>(&self, f: &mut TryMapStmtFn<'_, E>) -> std::result::Result<Body, E> {
        let mut out = Vec::new();
        for stmt in self.stmts() {
            let nested = stmt.nested();
            let descended = if nested.is_empty() {
                stmt.clone()
            } else {
                let bodies: std::result::Result<Vec<Body>, E> =
                    nested.iter().map(|body| body.try_map_stmts(f)).collect();
                Statement::rebuild_bodies(stmt, bodies?)
            };
            if let Some(kept) = f(&descended)? {
                out.push(kept);
            }
        }
        Ok(self.rebuilt(out))
    }

    /// `try_map_stmts` with a function that cannot fail.
    pub fn map_stmts(&self, f: &mut dyn FnMut(&Arc<Statement>) -> Option<Arc<Statement>>) -> Body {
        infallible(self.try_map_stmts(&mut |stmt| Ok::<_, std::convert::Infallible>(f(stmt))))
    }

    // --- def-use readings -----------------------------------------------------------------------

    /// Accumulator and carried-state names, each once, in structural order.
    pub fn carried_names(&self) -> &[Name] {
        self.0.carried_names.get_or_init(|| {
            let mut out = Vec::new();
            self.walk(&mut |here| {
                if let Statement::Accum(_) | Statement::Carry(_) = &**here.stmt() {
                    for name in here.stmt().defines() {
                        if !out.contains(&name) {
                            out.push(name);
                        }
                    }
                }
            });
            out
        })
    }

    /// Bindings at this scope, including states exported by nested statements.
    pub fn local_defs(&self) -> &HashSet<Name> {
        self.0.local_defs.get_or_init(|| {
            self.stmts()
                .iter()
                .flat_map(|stmt| ordered_sibling_defs(stmt))
                .collect()
        })
    }

    /// SSA reads from the enclosing scope, excluding this scope's definitions and exported states.
    pub fn free_ssa(&self) -> &HashSet<Name> {
        self.0.free_ssa.get_or_init(|| {
            let mut reads: HashSet<Name> = HashSet::new();
            for stmt in self.stmts() {
                reads.extend(free_ssa(stmt));
            }
            for local in self.local_defs() {
                reads.remove(local);
            }
            reads
        })
    }

    /// Every SSA name produced anywhere inside this body to its defining statement (the last definer).
    pub fn definitions(&self) -> &HashMap<Name, StmtRef> {
        self.0.definitions.get_or_init(|| {
            let mut out = HashMap::new();
            self.walk(&mut |here| {
                for name in here.stmt().defines() {
                    out.insert(name, here.clone());
                }
            });
            out
        })
    }

    /// Every axis name bound anywhere inside this body.
    pub fn axis_names(&self) -> &HashSet<Name> {
        self.0.axis_names.get_or_init(|| {
            let mut out = HashSet::new();
            self.walk(&mut |here| out.extend(here.stmt().binds_axes()));
            out
        })
    }

    /// Every name a statement of this subtree reads through `deps`.
    pub fn ssa_uses(&self) -> &HashSet<Name> {
        self.0.ssa_uses.get_or_init(|| {
            let mut out = HashSet::new();
            self.walk(&mut |here| out.extend(here.stmt().deps()));
            out
        })
    }

    /// Each SSA definition to the axes its value depends on.
    pub fn axis_dependencies(&self) -> HashMap<Name, HashSet<Name>> {
        let axis_names = self.axis_names();
        let mut dependencies: HashMap<Name, HashSet<Name>> = HashMap::new();
        fn walk(
            body: &Body,
            axis_names: &HashSet<Name>,
            dependencies: &mut HashMap<Name, HashSet<Name>>,
        ) {
            for stmt in body.stmts() {
                for child in stmt.nested() {
                    walk(&child, axis_names, dependencies);
                }
                match &**stmt {
                    Statement::Loop(l) => {
                        for child in l.body.stmts() {
                            if let Statement::Accum(a) = &**child {
                                let mut axes =
                                    dependencies.get(&a.value).cloned().unwrap_or_default();
                                axes.remove(&l.axis.name);
                                dependencies.insert(a.name.clone(), axes);
                            }
                        }
                        continue;
                    }
                    Statement::StridedLoop(l) => {
                        for child in l.body.stmts() {
                            if let Statement::Accum(a) = &**child {
                                let axes = dependencies.get(&a.value).cloned().unwrap_or_default();
                                dependencies.insert(a.name.clone(), axes);
                            }
                        }
                        continue;
                    }
                    _ => {}
                }
                let mut reads: HashSet<Name> = stmt.deps().into_iter().collect();
                for expression in stmt.expressions() {
                    reads.extend(expression.free_vars());
                }
                let mut axes: HashSet<Name> = reads
                    .iter()
                    .filter(|name| axis_names.contains(*name))
                    .cloned()
                    .collect();
                for name in &reads {
                    if let Some(found) = dependencies.get(name) {
                        axes.extend(found.iter().cloned());
                    }
                }
                for name in stmt.defines() {
                    dependencies.insert(name, axes.clone());
                }
            }
        }
        walk(self, axis_names, &mut dependencies);
        dependencies
    }

    /// The statements of the given kinds, pre-order, cached per kind set.
    pub fn of_kinds(&self, kinds: &[Kind]) -> Vec<StmtRef> {
        let key = kinds.to_vec();
        if let Some(found) = locked(&self.0.by_kind).get(&key) {
            return found.clone();
        }
        let mut out = Vec::new();
        self.walk(&mut |here| {
            if kinds.contains(&here.stmt().kind()) {
                out.push(here.clone());
            }
        });
        locked(&self.0.by_kind).insert(key, out.clone());
        out
    }

    pub fn loads(&self) -> Vec<StmtRef> {
        self.of_kinds(&[Kind::Load])
    }

    pub fn writes(&self) -> Vec<StmtRef> {
        self.of_kinds(&[Kind::Write])
    }

    pub fn carries(&self) -> Vec<StmtRef> {
        self.of_kinds(&[Kind::Carry])
    }

    pub fn accums(&self) -> Vec<StmtRef> {
        self.of_kinds(&[Kind::Accum])
    }

    pub fn loops(&self) -> Vec<StmtRef> {
        self.of_kinds(&[Kind::Loop])
    }

    // --- cones and orders -------------------------------------------------------------------------

    /// The backward dependence cone of `roots` over this body's immediate statements: the members, in body order,
    /// and the names resolving to no member here.
    pub fn backward_cone(&self, roots: &[Name]) -> (Vec<StmtRef>, Vec<Name>) {
        let mut by_name: HashMap<Name, usize> = HashMap::new();
        for (index, stmt) in self.stmts().iter().enumerate() {
            for name in exposed_defines(stmt) {
                by_name.insert(name, index);
            }
        }
        let mut members: HashSet<usize> = HashSet::new();
        let mut external: Vec<Name> = Vec::new();
        let mut pending: Vec<Name> = roots.to_vec();
        let mut seen: HashSet<Name> = HashSet::new();
        while let Some(name) = pending.pop() {
            if !seen.insert(name.clone()) {
                continue;
            }
            let Some(&index) = by_name.get(&name) else {
                external.push(name);
                continue;
            };
            if !members.insert(index) {
                continue;
            }
            pending.extend(free_names(&self.stmts()[index]));
        }
        external.sort();
        (
            (0..self.len())
                .filter(|index| members.contains(index))
                .map(|index| self.at(index))
                .collect(),
            external,
        )
    }

    /// Kahn's order over indexed predecessor sets, as source indices; source order breaks ties, a priority may rank
    /// first. Constraints that hold a cycle (a carried value read ahead of its update inside one scope, as Kernel IR
    /// spells a loop-carried state) leave the source order: no sort reorders what it cannot order.
    pub fn topological_permutation<P: Ord>(
        &self,
        incoming: &[HashSet<usize>],
        priority: Option<&dyn Fn(usize) -> P>,
    ) -> Vec<usize> {
        let n = self.len();
        let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut degree: Vec<usize> = incoming.iter().map(|sources| sources.len()).collect();
        for (target, sources) in incoming.iter().enumerate() {
            for &source in sources {
                successors[source].push(target);
            }
        }
        let key = |index: usize| -> (Option<P>, usize) { (priority.map(|p| p(index)), index) };
        let mut ready: std::collections::BinaryHeap<std::cmp::Reverse<(Option<P>, usize)>> = (0..n)
            .filter(|&i| degree[i] == 0)
            .map(|i| std::cmp::Reverse(key(i)))
            .collect();
        let mut ordered = Vec::with_capacity(n);
        while let Some(std::cmp::Reverse((_, selected))) = ready.pop() {
            ordered.push(selected);
            for &target in &successors[selected] {
                degree[target] -= 1;
                if degree[target] == 0 {
                    ready.push(std::cmp::Reverse(key(target)));
                }
            }
        }
        if ordered.len() == n {
            ordered
        } else {
            (0..n).collect()
        }
    }

    /// This body in an order `incoming` allows, source order breaking ties.
    pub fn topological_order(&self, incoming: &[HashSet<usize>]) -> Body {
        let order = self.topological_permutation::<()>(incoming, None);
        self.rebuilt(
            order
                .into_iter()
                .map(|index| self.stmts()[index].clone())
                .collect(),
        )
    }
}

pub fn loop_parts(stmt: &Statement) -> Option<(bool, &Body)> {
    match stmt {
        Statement::Loop(l) => Some((l.is_reduce(), &l.body)),
        Statement::StridedLoop(l) => Some((l.is_reduce(), &l.body)),
        _ => None,
    }
}

/// Names each listed once, in first appearance.
pub fn unique(names: impl IntoIterator<Item = Name>) -> Vec<Name> {
    let mut out: Vec<Name> = Vec::new();
    for name in names {
        if !out.contains(&name) {
            out.push(name);
        }
    }
    out
}

/// Names visible to siblings in structural body order: a leaf's own definitions, a block's exported states.
pub fn ordered_sibling_defs(stmt: &Statement) -> Vec<Name> {
    let children = stmt.nested();
    if children.is_empty() {
        return stmt.defines();
    }
    unique(
        children
            .iter()
            .flat_map(|child| child.carried_names().iter().cloned()),
    )
}

/// SSA names `stmt` reads from the scope around it.
pub fn free_ssa(stmt: &Statement) -> HashSet<Name> {
    let children = stmt.nested();
    let mut reads: HashSet<Name> = stmt.deps().into_iter().collect();
    if children.is_empty() || stmt.deps_deep() {
        return reads;
    }
    for child in children {
        reads.extend(child.free_ssa().iter().cloned());
    }
    reads
}

/// SSA names `stmt` makes visible at its own scope level: its own and every nested definition.
pub fn exposed_defines(stmt: &Statement) -> Vec<Name> {
    let mut out = stmt.defines();
    for body in stmt.nested() {
        body.walk(&mut |here| out.extend(here.stmt().defines()));
    }
    unique(out)
}

/// Every name the whole subtree of `stmt` reads from its enclosing scope: SSA reads and index coordinates, less
/// what the subtree defines or binds.
pub fn free_names(stmt: &Statement) -> Vec<Name> {
    let mut reads: Vec<Name> = Vec::new();
    let mut defs: HashSet<Name> = HashSet::new();
    fn walk(
        stmt: &Statement,
        bound: &HashSet<Name>,
        reads: &mut Vec<Name>,
        defs: &mut HashSet<Name>,
    ) {
        for name in stmt.deps() {
            if !bound.contains(&name) && !reads.contains(&name) {
                reads.push(name);
            }
        }
        for expression in stmt.expressions() {
            for name in expression.free_vars() {
                if !bound.contains(&name) && !reads.contains(&name) {
                    reads.push(name);
                }
            }
        }
        defs.extend(stmt.defines());
        if stmt.deps_deep() {
            return;
        }
        let mut inner = bound.clone();
        inner.extend(stmt.binds_axes());
        for body in stmt.nested() {
            for child in body.stmts() {
                walk(child, &inner, reads, defs);
            }
        }
    }
    walk(stmt, &HashSet::new(), &mut reads, &mut defs);
    reads.retain(|name| !defs.contains(name));
    reads
}

/// One shared, read-only reduction cone with explicit coordinate parameters. Compared by object.
pub struct Subroutine {
    pub name: Name,
    pub axes: Axes,
    pub body: Body,
    pub result: Name,
    /// The buffers the body reads, each once, in first appearance.
    pub buffers: Names,
}

impl std::fmt::Debug for Subroutine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Subroutine({:?})", self.name)
    }
}

impl Subroutine {
    pub fn new(name: Name, axes: Axes, body: Body, result: Name) -> Result<Arc<Subroutine>> {
        if !body.carries().is_empty() || body.stmts().iter().any(|stmt| stmt.has_side_effects()) {
            return Err(Error::Value("a fusion subroutine must be read-only".into()));
        }
        if !body.definitions().contains_key(&result) {
            return Err(Error::Value(
                "a fusion subroutine must define its result".into(),
            ));
        }
        let buffers: Names = unique(
            body.iter_stmts()
                .iter()
                .flat_map(|stmt| stmt.external_reads()),
        )
        .into();
        Ok(Arc::new(Subroutine {
            name,
            axes,
            body,
            result,
            buffers,
        }))
    }

    pub fn params(&self) -> Vec<Name> {
        self.axes.iter().map(|axis| axis.name.clone()).collect()
    }

    /// Rebind the captured buffers once for all calls sharing this definition.
    pub fn with_buffers(&self, buffers: Names) -> Arc<Subroutine> {
        let mapping: HashMap<Name, Name> = self
            .buffers
            .iter()
            .cloned()
            .zip(buffers.iter().cloned())
            .collect();
        let body = self.body.rename_buffers(&mapping);
        Arc::new(Subroutine {
            name: self.name.clone(),
            axes: self.axes.clone(),
            body,
            result: self.result.clone(),
            buffers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinds::Affine;
    use crate::kinds::field::{Float, Floats};
    use crate::normalize::testing::{accum_names, body, indented, inner, strs};
    use crate::text::parse_body;

    fn names(list: &[&str]) -> HashSet<Name> {
        list.iter().map(|name| Name::from(*name)).collect()
    }

    fn roots(list: &[&str]) -> Vec<Name> {
        list.iter().map(|name| Name::from(*name)).collect()
    }

    /// A demoted-matmul cell: `p = cone(x) * w` with a plain B load.
    fn cell() -> Body {
        body(
            "
            a0 = load x[m, k]
            a1 = negative(a0)
            b = load w[k, n]
            p = multiply(a1, b)
            acc <- add(acc, p)
            ",
        )
    }

    #[test]
    fn a_topological_order_keeps_an_order_the_constraints_allow() {
        let given = body(
            "
            right = exp(x)
            left = abs(x)
            ",
        );
        assert_eq!(
            given.topological_order(&[HashSet::new(), HashSet::new()]),
            given
        );
    }

    #[test]
    fn a_scope_keeps_the_reads_above_a_name_it_shadows() {
        let given = body(
            "
            use = abs(outer)
            for i in 0..4:
                outer = exp(source)
            ",
        );
        assert_eq!(*inner(&given.stmts()[1]).free_ssa(), names(&["source"]));
        assert_eq!(*given.free_ssa(), names(&["outer", "source"]));
    }

    #[test]
    fn kind_queries_share_children_and_keep_a_mixed_order() {
        let given = body(
            "
            for i in 0..4:
                state[] <- value seed 0
                sum <- add(sum, value)
            state[] <- value seed 0
            ",
        );
        let child = inner(&given.stmts()[0]);
        let (carry, accum) = (child.at(0), child.at(1));
        assert_eq!(child.carries(), std::slice::from_ref(&carry));
        assert_eq!(child.accums(), std::slice::from_ref(&accum));
        assert_eq!(
            given.of_kinds(&[Kind::Accum, Kind::Carry]),
            [carry, accum, given.at(1)]
        );
        assert_eq!(child.of_kinds(&[Kind::Carry]), child.carries());
        assert_eq!(strs(given.carried_names()), ["state", "sum"]);
    }

    #[test]
    fn coalesce_keeps_the_loop_of_a_reduction_that_reads_a_finalized_one() {
        for dependent in [false, true] {
            for strided in [false, true] {
                let header = format!("for k in 0..8{}:", if strided { " step 1" } else { "" });
                let mut second = vec!["v = load x[k]".to_string()];
                if dependent {
                    second.push("shifted = subtract(v, maximum)".to_string());
                }
                second.push(format!(
                    "sum <- add(sum, {})",
                    if dependent { "shifted" } else { "v" }
                ));
                let coalesced = body(&format!(
                    "{header}\n    v = load x[k]\n    maximum <- maximum(maximum, v)\n{header}\n{}",
                    indented(&second.join("\n"))
                ))
                .coalesce();
                let expected = if dependent { 2 } else { 1 };
                assert_eq!(
                    coalesced.len(),
                    expected,
                    "dependent={dependent} strided={strided}"
                );
                assert_eq!(coalesced.loads().len(), expected);
                assert_eq!(strs(coalesced.carried_names()), ["maximum", "sum"]);
            }
        }
    }

    #[test]
    fn coalesce_declares_a_shared_fold_once_when_one_lowering_fuses_it() {
        // One cone edge lowers the shared fold `a` alone; another fuses `a` and its sibling `b` into one pass over
        // the same axis. In either order the scope declares `a` once and still folds `b` (the GDN in_proj
        // projections, read through two edges of one computed-A cone).
        for fused_first in [false, true] {
            for strided in [false, true] {
                let head = if strided {
                    "for k in 0..8 step 1:"
                } else {
                    "for k in 0..8:"
                };
                let alone = format!(
                    "{head}\n    x = load xs[k]\n    wa = load was[k]\n    pa = multiply(x, wa)\n    a <- add(a, pa)"
                );
                let fused = format!(
                    "{alone}\n    wb = load wbs[k]\n    pb = multiply(x, wb)\n    b <- add(b, pb)"
                );
                let (first, second) = if fused_first {
                    (&fused, &alone)
                } else {
                    (&alone, &fused)
                };
                // A statement between the two lowerings, as in the cone: the loops stay apart.
                let coalesced = body(&format!("{first}\nbetween = exp(a)\n{second}")).coalesce();
                let loops: Vec<&Body> = coalesced
                    .stmts()
                    .iter()
                    .filter_map(|stmt| loop_parts(stmt).map(|(_, body)| body))
                    .collect();
                let mut accumulators: Vec<String> =
                    loops.iter().flat_map(|body| accum_names(body)).collect();
                accumulators.sort();
                assert_eq!(
                    accumulators,
                    ["a", "b"],
                    "fused_first={fused_first} strided={strided}"
                );
                // The pass that keeps `b` alone drops `a`'s own reads and keeps the one `b` shares.
                let folding_b: Vec<&&Body> = loops
                    .iter()
                    .filter(|body| body.definitions().contains_key("b"))
                    .collect();
                let [last] = folding_b[..] else {
                    panic!("one loop folds b: {}", folding_b.len())
                };
                assert_eq!(last.definitions().contains_key("wa"), fused_first);
                assert!(last.definitions().contains_key("x"));
            }
        }
    }

    #[test]
    fn axis_dependencies_stay_compact_through_a_long_ssa_chain() {
        // Axis dependence grows with definitions times axes, not transitive SSA edges.
        let chain_length = 4096;
        let mut lines = vec![
            "for a in 0..4:".to_string(),
            "    v0 = load x[a]".to_string(),
        ];
        lines.extend((1..=chain_length).map(|i| format!("    v{i} = negative(v{})", i - 1)));
        let dependencies = body(&lines.join("\n")).axis_dependencies();
        assert_eq!(dependencies.len(), chain_length + 1);
        assert_eq!(
            dependencies.values().map(HashSet::len).sum::<usize>(),
            chain_length + 1
        );
        assert!(dependencies.values().all(|axes| *axes == names(&["a"])));
    }

    #[test]
    fn axis_dependencies_remove_only_the_closed_reduce_axis() {
        let dependencies = body(
            "
            for a in 0..8:
                for k in 0..32:
                    value = load x[a, k]
                    acc <- add(acc, value)
            ",
        )
        .axis_dependencies();
        assert_eq!(dependencies["value"], names(&["a", "k"]));
        assert_eq!(dependencies["acc"], names(&["a"]));
    }

    #[test]
    fn axis_dependencies_keep_a_strided_partial_axis() {
        let dependencies = body(
            "
            for k in start..32 step step:
                value = load x[k]
                partial <- add(partial, value)
            ",
        )
        .axis_dependencies();
        assert_eq!(dependencies["partial"], names(&["k"]));
    }

    #[test]
    fn a_backward_cone_lists_its_members_in_body_order() {
        let (members, _) = cell().backward_cone(&roots(&["a1"]));
        let defined: Vec<Name> = members
            .iter()
            .map(|member| member.defines()[0].clone())
            .collect();
        assert_eq!(strs(&defined), ["a0", "a1"]);
    }

    #[test]
    fn a_backward_cone_reads_its_axes_from_outside_and_resolves_its_chain_inside() {
        // The cone reads its index axes from the enclosing scope; the resolved SSA chain (a0) is internal, the
        // untouched B side absent entirely.
        let (_, external) = cell().backward_cone(&roots(&["a1"]));
        assert_eq!(strs(&external), ["k", "m"]);
    }

    #[test]
    fn a_backward_cone_root_defined_elsewhere_is_external() {
        // A root from another scope level yields no members and surfaces as an external read: the chaining signal,
        // not an error.
        let (members, external) = cell().backward_cone(&roots(&["stat"]));
        assert!(members.is_empty());
        assert_eq!(strs(&external), ["stat"]);
    }

    #[test]
    fn a_reduce_loop_joins_a_backward_cone_whole() {
        // A reduce loop resolves through its exposed accumulator and joins as one member; its own loop axis is
        // internal, the row axis it reads is not.
        let given = body(
            "
            for r in 0..32:
                s0 = load x[m, r]
                stat <- maximum(stat, s0)
            e = exp(stat)
            ",
        );
        let (members, external) = given.backward_cone(&roots(&["e"]));
        assert_eq!(members, [given.at(0), given.at(1)]);
        assert!(!strs(&external).contains(&"r"));
        assert!(strs(&external).contains(&"m"));
    }

    #[test]
    fn an_axis_a_backward_cone_reads_only_in_a_select_predicate_is_external() {
        // An axis reaching the cone only through a select predicate counts as read: materializing without it would
        // drop a real dependence (the rotate-half rotary shape).
        let given = body(
            "
            a0 = load x[k]
            a1 = load x[m]
            v = select:
                a0 when n < m
                a1 when m
            ",
        );
        let (_, external) = given.backward_cone(&roots(&["v"]));
        assert!(strs(&external).contains(&"n"));
    }

    #[test]
    fn the_exact_key_tells_apart_what_body_equality_does_not() {
        let plain = parse_body("for i in 0..n:\n    x = load X[i]\n    O[i] = x").unwrap();
        let hinted = parse_body("for i in 0..n~128:\n    x = load X[i]\n    O[i] = x").unwrap();
        let windowed =
            parse_body("for i in 0..n window(parent=j in 0..m):\n    x = load X[i]\n    O[i] = x")
                .unwrap();
        let again = parse_body("for i in 0..n:\n    x = load X[i]\n    O[i] = x").unwrap();
        assert_eq!(plain, hinted, "a hint is no part of body equality");
        assert_eq!(plain, windowed, "a window is no part of body equality");
        assert_eq!(MemoKey(plain.clone()), MemoKey(again.clone()));
        assert_eq!(plain.exact_hash(), again.exact_hash());
        assert_ne!(MemoKey(plain.clone()), MemoKey(hinted.clone()));
        assert_ne!(plain.exact_hash(), hinted.exact_hash());
        assert_ne!(MemoKey(plain.clone()), MemoKey(windowed.clone()));
        assert_ne!(plain.exact_hash(), windowed.exact_hash());
    }

    #[test]
    fn the_memo_key_reads_numbers_by_their_bits() {
        let zero = parse_body("let x = 0.0\nO[] = x").unwrap();
        let negative_zero = parse_body("let x = -0.0\nO[] = x").unwrap();
        assert_eq!(
            zero, negative_zero,
            "a literal compares as Python's float does"
        );
        assert_ne!(MemoKey(zero.clone()), MemoKey(negative_zero.clone()));
        assert_ne!(zero.exact_hash(), negative_zero.exact_hash());
        let nan = parse_body("init f32 acc = nan\nO[] = acc").unwrap();
        let again = parse_body("init f32 acc = nan\nO[] = acc").unwrap();
        assert_eq!(
            MemoKey(nan.clone()),
            MemoKey(again.clone()),
            "every NaN keys alike"
        );
        assert_eq!(nan.exact_hash(), again.exact_hash());
        let seeded = parse_body("state[i] <- v seed -0.0").unwrap();
        let other_seed = parse_body("state[i] <- v seed 0.0").unwrap();
        assert_ne!(MemoKey(seeded), MemoKey(other_seed));
        let nan_literal = parse_body("let x = nan\nO[] = x").unwrap();
        let again = parse_body("let x = nan\nO[] = x").unwrap();
        assert_eq!(
            MemoKey(nan_literal.clone()),
            MemoKey(again),
            "a NaN literal keys like its copy"
        );
        let coefficient = |c: f64| {
            Body::from_statements([Statement::Affine(Affine {
                name: "v".into(),
                args: Names::from(["x".into()]),
                coefficients: Floats::from([Float(c)]),
                offset: Float(0.0),
                dtype: None,
            })])
        };
        assert_eq!(
            coefficient(0.0),
            coefficient(-0.0),
            "a coefficient compares as a float"
        );
        assert_ne!(MemoKey(coefficient(0.0)), MemoKey(coefficient(-0.0)));
    }

    #[test]
    fn a_failing_walk_returns_its_error_and_builds_nothing() {
        let body =
            parse_body("x = load X[i]\nfor k in 0..4:\n    y = load Y[k]\n    O[k] = y").unwrap();
        let mut visited = 0;
        let failed = body.try_map_stmts(&mut |stmt| {
            visited += 1;
            match &**stmt {
                Statement::Load(load) if &*load.input == "Y" => Err("no Y"),
                _ => Ok(Some(stmt.clone())),
            }
        });
        assert_eq!(failed, Err("no Y"));
        assert_eq!(visited, 2, "the walk stops at the error");
        let kept = body.try_map(&mut |stmt| Ok::<_, ()>(Some(vec![stmt.clone()])));
        assert!(
            kept.unwrap().ptr_eq(&body),
            "a walk that keeps every statement keeps the body"
        );
    }
}
