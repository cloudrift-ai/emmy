//! `Body`: a shared, immutable sequence of statements with the def-use readings, walks and orders every analysis
//! phrases itself over, each computed once on first use and kept with the body. And `Subroutine`, the shared
//! reduction body fusion builds a region with.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

use crate::kinds::field::{Axes, Names};
use crate::kinds::{Kind, Statement};
use crate::{Error, Name, Result, locked};

/// What `Body::map` applies: a statement to the statements that replace it, or `None` to drop it.
pub type MapFn<'a> = dyn FnMut(&Arc<Statement>) -> Option<Vec<Arc<Statement>>> + 'a;

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
    pub(crate) normalized: OnceLock<Normalized>,
    pub(crate) identities: Mutex<Vec<(crate::identity::IdentityKey, crate::identity::Identity)>>,
}

/// What a body knows of its normal form once it was taken: that it is its own, or which body it is. Never the body's
/// own `Arc`, which would keep it alive forever.
#[derive(Clone)]
pub(crate) enum Normalized {
    Itself,
    Other(Body),
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
            normalized: OnceLock::new(),
            identities: Mutex::new(Vec::new()),
        }))
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
        let stmts: Vec<Arc<Statement>> = self.stmts().iter().map(&mut *f).collect();
        if stmts
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
    /// statement, which it keeps, drops (`None`) or replaces by several.
    pub fn map(&self, f: &mut MapFn<'_>) -> Body {
        let mut out = Vec::new();
        let mut changed = false;
        for stmt in self.stmts() {
            let nested = stmt.nested();
            let descended = if nested.is_empty() {
                stmt.clone()
            } else {
                let bodies: Vec<Body> = nested.iter().map(|body| body.map(f)).collect();
                Statement::rebuild_bodies(stmt, bodies)
            };
            match f(&descended) {
                None => changed = true,
                Some(replacement) => {
                    if !(replacement.len() == 1 && Arc::ptr_eq(&replacement[0], stmt)) {
                        changed = true;
                    }
                    out.extend(replacement);
                }
            }
        }
        if changed {
            Body::new(out)
        } else {
            self.clone()
        }
    }

    /// `map` whose function keeps, drops or replaces one statement by one.
    pub fn map_stmts(&self, f: &mut dyn FnMut(&Arc<Statement>) -> Option<Arc<Statement>>) -> Body {
        let mut out = Vec::new();
        let mut same = true;
        for stmt in self.stmts() {
            let nested = stmt.nested();
            let descended = if nested.is_empty() {
                stmt.clone()
            } else {
                let bodies: Vec<Body> = nested.iter().map(|body| body.map_stmts(f)).collect();
                Statement::rebuild_bodies(stmt, bodies)
            };
            match f(&descended) {
                Some(kept) => {
                    if !Arc::ptr_eq(&kept, stmt) {
                        same = false;
                    }
                    out.push(kept);
                }
                None => same = false,
            }
        }
        if same { self.clone() } else { Body::new(out) }
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
    /// first. A cycle leaves the order unchanged.
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

    pub fn topological_order(&self, incoming: &[HashSet<usize>]) -> Body {
        let order = self.topological_permutation::<()>(incoming, None);
        if order
            .iter()
            .enumerate()
            .all(|(position, &index)| position == index)
        {
            return self.clone();
        }
        Body::new(
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
