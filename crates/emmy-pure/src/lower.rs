//! The one lowering of a [`Fold`] tree to the Loop IR nest the materializer expands, with the kernel's boundary
//! stores placed in it.
//!
//! One rule places every term: a term is materialized at the SHALLOWEST scope on its path that binds all of its free
//! coordinates (a value of the tree it reads by name standing for the coordinates of the term that defines it). The
//! scopes are the plain loops opened for the free coordinates left unbound — outermost the coordinate the most terms
//! are evaluated over, ties in declaration order — and the reduce loop of each term on the way down. A boundary store
//! follows the term that defines the value it writes, at that term's scope. The free loops form a tree: a term's
//! operands sit on its path, so what a term reads is always in scope, and a shared term defines its names once per
//! scope. Kernel identity digests what this returns.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use emmy_ir::axis::Axis;
use emmy_ir::body::{Body, free_names};
use emmy_ir::expr::Expression;
use emmy_ir::statement::{Cond, Loop, Statement};
use emmy_ir::{Error, Name, Result};

use crate::fold::Fold;

/// One output write at the kernel boundary: the write, the output loop nest it rode in the source, and its guard.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Store {
    pub write: Arc<Statement>,
    pub sweep: Vec<Arc<Axis>>,
    pub when: Option<Expression>,
}

/// One lowering asked of a term: what its memo keys the result by.
pub struct Request {
    bound: Option<Vec<Name>>,
    stores: Vec<Store>,
    axes: Vec<Arc<Axis>>,
    extents: Vec<Name>,
}

impl Request {
    /// Equal with every axis's hint and window too: both reach the lowered nest, and neither is part of an axis's
    /// equality.
    fn same(&self, other: &Request) -> bool {
        let axes = |a: &[Arc<Axis>], b: &[Arc<Axis>]| {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.exact_eq(b))
        };
        self.bound == other.bound
            && self.extents == other.extents
            && axes(&self.axes, &other.axes)
            && self.stores.len() == other.stores.len()
            && self
                .stores
                .iter()
                .zip(&other.stores)
                .all(|(a, b)| a == b && axes(&a.sweep, &b.sweep))
    }
}

impl Store {
    fn values(&self) -> &[Name] {
        match &*self.write {
            Statement::Write(write) => &write.values,
            _ => &[],
        }
    }

    /// The expressions the store reads: its coordinates and its guard.
    fn reads(&self) -> Vec<Expression> {
        let mut out: Vec<Expression> = match &*self.write {
            Statement::Write(write) => write.index.to_vec(),
            _ => Vec::new(),
        };
        out.extend(self.when.iter().cloned());
        out
    }

    fn read_names(&self) -> HashSet<Name> {
        self.reads().iter().flat_map(|e| e.free_vars()).collect()
    }

    /// The store as it stands in a statement stream: the write, under its guard when it has one.
    fn stmt(&self) -> Arc<Statement> {
        match &self.when {
            None => self.write.clone(),
            Some(when) => Arc::new(Statement::Cond(Cond {
                cond: when.clone(),
                body: Body::new(vec![self.write.clone()]),
                else_body: Body::empty(),
            })),
        }
    }

    /// The store spelled through `mapping` (names only): what the term's own vocabulary calls the values it writes.
    fn renamed(&self, mapping: &HashMap<Name, Name>) -> Store {
        if mapping.is_empty() {
            return self.clone();
        }
        let empty = HashMap::new();
        let by_role = |_: emmy_ir::statement::Roles, name: Name| mapping.get(&name).copied();
        let rewrite = emmy_ir::protocol::Rewrite {
            rename: &by_role,
            sigma: &empty,
            axis_fn: None,
        };
        Store {
            write: Statement::rewrite(&self.write, &rewrite),
            sweep: self.sweep.clone(),
            when: self
                .when
                .as_ref()
                .map(|w| w.rename(&mut |n| mapping.get(&n).copied())),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kind {
    Step,
    State,
    Observed,
}

/// One scope's tree of free loops: the coordinates it may open, what it already binds, and the statement list at
/// each path (an index into the lowering's lists).
struct Nest {
    opened: Vec<Name>,
    bound: HashSet<Name>,
    paths: HashMap<Vec<Name>, usize>,
}

struct Lowering<'a> {
    coordinates: Vec<(Name, Arc<Axis>)>,
    bound: HashSet<Name>,
    internal: HashSet<Name>,
    declared: Vec<Name>,
    origin: HashMap<Name, (usize, Kind)>,
    terms: HashMap<usize, Fold>,
    indexed: HashSet<Name>,
    stored: HashSet<Name>,
    taken: HashMap<usize, Vec<Name>>,
    owned: HashMap<(usize, Kind), Vec<usize>>,
    stores: &'a [Store],
    lists: Vec<Vec<Arc<Statement>>>,
    nests: Vec<Nest>,
    emitted: HashSet<(usize, usize)>,
    retained: Vec<Fold>,
    writes: HashMap<usize, bool>,
}

type Loops = Vec<(Name, HashSet<Name>, usize)>;

impl Lowering<'_> {
    fn coordinate(&self, name: &Name) -> Option<&Arc<Axis>> {
        self.coordinates
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, a)| a)
    }

    fn has_coordinate(&self, name: &Name) -> bool {
        self.coordinates.iter().any(|(n, _)| n == name)
    }

    fn new_list(&mut self) -> usize {
        self.lists.push(Vec::new());
        self.lists.len() - 1
    }

    fn new_nest(&mut self, opened: Vec<Name>, bound: HashSet<Name>) -> usize {
        let root = self.new_list();
        self.nests.push(Nest {
            opened,
            bound,
            paths: HashMap::from([(Vec::new(), root)]),
        });
        self.nests.len() - 1
    }

    /// Whether any term of `term`'s subtree defines a value the boundary stores.
    fn writes_under(&mut self, term: &Fold) -> Result<bool> {
        if let Some(found) = self.writes.get(&term.address()) {
            return Ok(*found);
        }
        let mut pending = vec![term.clone()];
        let mut seen = HashSet::new();
        let mut found = false;
        while let Some(node) = pending.pop() {
            if !seen.insert(node.address()) {
                continue;
            }
            let step = node.step()?;
            if node
                .exposes()
                .iter()
                .chain(step.defined())
                .any(|n| self.stored.contains(n))
            {
                found = true;
                break;
            }
            pending.extend(node.operands().iter().cloned());
        }
        self.writes.insert(term.address(), found);
        Ok(found)
    }

    fn taken_of(&self, term: &Fold) -> Result<&Vec<Name>> {
        self.taken
            .get(&term.address())
            .ok_or_else(|| Error::Key("a term the demand walk did not reach".into()))
    }

    /// The operands placed for `term`, each narrowed to what its readers take (an operand under an interior store is
    /// kept whole), and the operands they stand for.
    fn placed(&mut self, term: &Fold) -> Result<(Vec<Fold>, Vec<Fold>)> {
        let mut placed = Vec::new();
        let mut kept = Vec::new();
        for edge in term.operands() {
            let writes = self.writes_under(edge)?;
            let taken = self.taken_of(edge)?.clone();
            if taken.is_empty() && !writes {
                continue;
            }
            placed.push(if writes {
                edge.clone()
            } else {
                edge.exposing(&taken)?
            });
            kept.push(edge.clone());
        }
        Ok((placed, kept))
    }

    /// The coordinates `names` are evaluated over: a coordinate is its own, and a value a term of the tree computes
    /// that is no coordinate stands for the coordinates of that term.
    fn evaluated_over(&self, names: &HashSet<Name>) -> HashSet<Name> {
        let mut out = HashSet::new();
        for name in names {
            let value = self.origin.contains_key(name)
                && !self.has_coordinate(name)
                && !self.bound.contains(name)
                && !self.internal.contains(name);
            if value {
                let term = &self.terms[&self.origin[name].0];
                out.extend(self.evaluated_over(term.free_axes()));
            } else {
                out.insert(*name);
            }
        }
        out
    }

    fn path_of(
        &self,
        nest: usize,
        free: &HashSet<Name>,
        path: Option<&[Name]>,
    ) -> Result<Vec<Name>> {
        let nest = &self.nests[nest];
        let needed: HashSet<Name> = nest
            .opened
            .iter()
            .filter(|n| free.contains(n))
            .copied()
            .collect();
        match path {
            None => Ok(nest
                .opened
                .iter()
                .filter(|n| needed.contains(n))
                .copied()
                .collect()),
            Some(path) => (0..=path.len())
                .find(|&depth| needed.iter().all(|n| path[..depth].contains(n)))
                .map(|depth| path[..depth].to_vec())
                .ok_or_else(|| {
                    Error::Value("lower: no prefix of the reader's path binds the operand".into())
                }),
        }
    }

    fn sink(&mut self, nest: usize, path: &[Name]) -> usize {
        for depth in 0..=path.len() {
            if !self.nests[nest].paths.contains_key(&path[..depth]) {
                let list = self.new_list();
                self.nests[nest].paths.insert(path[..depth].to_vec(), list);
            }
        }
        self.nests[nest].paths[path]
    }

    fn assemble(&self, nest: usize, path: &[Name]) -> Result<Body> {
        let here = &self.nests[nest];
        let mut body = self.lists[here.paths[path]].clone();
        let start = match path.last() {
            Some(last) => here
                .opened
                .iter()
                .position(|n| n == last)
                .map_or(here.opened.len(), |i| i + 1),
            None => 0,
        };
        for name in &here.opened[start..] {
            let mut deeper = path.to_vec();
            deeper.push(*name);
            if here.paths.contains_key(&deeper) {
                let axis = self
                    .coordinate(name)
                    .ok_or_else(|| Error::Value(format!("lower: no extent for {name}")))?;
                body.push(Arc::new(Statement::Loop(Loop {
                    axis: axis.clone(),
                    body: self.assemble(nest, &deeper)?,
                    unroll: false,
                    seed: true,
                    unrolled: false,
                    range: None,
                    step: None,
                })));
            }
        }
        Ok(Body::new(body).coalesce())
    }

    /// The stores over what `term` defines as `kind`, right after it in `target`; `computed` selects the stores whose
    /// coordinates name a value some term computes, the others, or every store.
    #[allow(clippy::too_many_arguments)]
    fn attach(
        &mut self,
        term: &Fold,
        kind: Kind,
        target: usize,
        node: &[Name],
        scope: &HashSet<Name>,
        nest: usize,
        computed: Option<bool>,
    ) -> Result<()> {
        let Some(specs) = self.owned.get(&(term.address(), kind)).cloned() else {
            return Ok(());
        };
        for index in specs {
            let spec = &self.stores[index];
            let read = spec.read_names();
            if let Some(computed) = computed
                && computed != read.iter().any(|n| self.indexed.contains(n))
            {
                continue;
            }
            let extra: HashSet<Name> = self
                .evaluated_over(&read)
                .difference(scope)
                .copied()
                .collect();
            let stmt = spec.stmt();
            if extra.is_empty() {
                self.lists[target].push(stmt);
                continue;
            }
            let opened = self.nests[nest].opened.clone();
            if !extra.iter().all(|n| opened.contains(n)) {
                let mut missing: Vec<&Name> =
                    extra.iter().filter(|n| !opened.contains(n)).collect();
                missing.sort();
                return Err(Error::Value(format!(
                    "a store reads coordinates {missing:?} no term declares and no sweep names"
                )));
            }
            let mut path = node.to_vec();
            path.extend(opened.iter().filter(|n| extra.contains(n)).copied());
            let list = self.sink(nest, &path);
            self.lists[list].push(stmt);
        }
        Ok(())
    }

    /// The coordinates a map's step is evaluated over: what it reads itself and what the operands it reads are
    /// evaluated over, not those of an operand it only carries to a store.
    fn needs(&self, term: &Fold) -> Result<HashSet<Name>> {
        let step = term.step()?;
        let reads: HashSet<Name> = step.stmts().iter().flat_map(|s| free_names(s)).collect();
        let mut needed = reads.clone();
        for b in term.bindings() {
            let edge = &term.operands()[b.operand];
            if reads.contains(&b.param) || reads.contains(&edge.exposes()[b.component]) {
                needed.extend(edge.free_axes().iter().copied());
            }
        }
        Ok(term.free_axes().intersection(&needed).copied().collect())
    }

    fn place(
        &mut self,
        term: &Fold,
        loops: &Loops,
        path: Option<&[Name]>,
        nest: usize,
    ) -> Result<()> {
        let evaluated_free = self.evaluated_over(term.free_axes());
        // `node` is the path the term sits on: under a reduce loop, the reader's path as it was given (which may be
        // unconstrained); otherwise the shallowest prefix that binds it.
        let (scope, stmts, loops, node): (HashSet<Name>, Option<usize>, Loops, Option<Vec<Name>>) =
            if loops
                .iter()
                .any(|(axis, _, _)| term.free_axes().contains(axis))
            {
                let depth = loops
                    .iter()
                    .position(|(_, scope, _)| evaluated_free.is_subset(scope))
                    .ok_or_else(|| {
                        Error::Value("lower: no enclosing reduce loop binds the term".into())
                    })?;
                let (_, scope, stmts) = loops[depth].clone();
                (
                    scope,
                    Some(stmts),
                    loops[..=depth].to_vec(),
                    path.map(<[Name]>::to_vec),
                )
            } else {
                let free = if term.axis().is_some() {
                    evaluated_free.clone()
                } else {
                    self.evaluated_over(&self.needs(term)?)
                };
                let node = self.path_of(nest, &free, path)?;
                let mut scope = self.nests[nest].bound.clone();
                scope.extend(node.iter().copied());
                (scope, None, Vec::new(), Some(node))
            };
        let step = term.step()?;
        if term.axis().is_some() || !step.is_empty() {
            let target = match stmts {
                Some(list) => list,
                None => self.sink(nest, node.as_deref().unwrap_or_default()),
            };
            if !self.emitted.insert((term.address(), target)) {
                return Ok(());
            }
            self.retained.push(term.clone());
        }
        let Some(axis) = term.axis() else {
            return self.place_map(term, &loops, path, nest, node, scope, stmts, step);
        };
        let here = node.clone().unwrap_or_default();
        if term.carries() {
            return self.place_carried(term, axis, nest, &here, &scope, stmts);
        }
        let inner = self.new_list();
        let mut inner_scope = scope.clone();
        inner_scope.insert(axis);
        let (placed, _) = self.placed(term)?;
        for edge in placed {
            let mut deeper = loops.clone();
            deeper.push((axis, inner_scope.clone(), inner));
            self.place(&edge, &deeper, node.as_deref(), nest)?;
        }
        self.lists[inner].extend(step.stmts().iter().cloned());
        self.attach(term, Kind::Observed, inner, &here, &inner_scope, nest, None)?;
        let target = match stmts {
            Some(list) => list,
            None => self.sink(nest, &here),
        };
        let coordinate = self.coordinate(&axis).cloned().ok_or_else(|| {
            Error::Value(format!(
                "lower: no extent for reduce axis '{axis}' — the kernel's axis table names it"
            ))
        })?;
        let body = Body::new(self.lists[inner].clone()).coalesce();
        self.lists[target].push(Arc::new(Statement::Loop(Loop {
            axis: coordinate,
            body,
            unroll: false,
            seed: true,
            unrolled: false,
            range: term.range().cloned(),
            step: None,
        })));
        self.attach(term, Kind::State, target, &here, &scope, nest, None)
    }

    #[allow(clippy::too_many_arguments)]
    fn place_map(
        &mut self,
        term: &Fold,
        loops: &Loops,
        path: Option<&[Name]>,
        nest: usize,
        mut node: Option<Vec<Name>>,
        scope: HashSet<Name>,
        stmts: Option<usize>,
        step: Body,
    ) -> Result<()> {
        let defined: HashSet<Name> = step.stmts().iter().flat_map(|s| s.defines()).collect();
        let reads: HashSet<Name> = step
            .stmts()
            .iter()
            .flat_map(|s| free_names(s))
            .filter(|n| !defined.contains(n))
            .collect();
        let early = !step.is_empty()
            && stmts.is_none()
            && !reads
                .iter()
                .any(|n| term.lift().params().contains(n) || self.origin.contains_key(n));
        let mut step_live = !step.is_empty();
        if early {
            let early_node = self.path_of(nest, &HashSet::new(), path)?;
            let list = self.sink(nest, &early_node);
            self.lists[list].extend(step.stmts().iter().cloned());
            let mut early_scope = self.nests[nest].bound.clone();
            early_scope.extend(early_node.iter().copied());
            self.attach(
                term,
                Kind::Step,
                list,
                &early_node,
                &early_scope,
                nest,
                Some(false),
            )?;
            node = Some(early_node);
            step_live = false;
        }
        let read: HashSet<usize> = term
            .bindings()
            .iter()
            .filter(|b| {
                reads.contains(&b.param)
                    || reads.contains(&term.operands()[b.operand].exposes()[b.component])
            })
            .map(|b| term.operands()[b.operand].address())
            .collect();
        let (placed, kept) = self.placed(term)?;
        for (edge, operand) in placed.iter().zip(&kept) {
            let reader_path = if step_live && read.contains(&operand.address()) {
                node.as_deref()
            } else {
                path
            };
            let reader_path = reader_path.map(<[Name]>::to_vec);
            self.place(edge, loops, reader_path.as_deref(), nest)?;
        }
        let here = node.unwrap_or_default();
        if early {
            let list = self.sink(nest, &here);
            let mut early_scope = self.nests[nest].bound.clone();
            early_scope.extend(here.iter().copied());
            self.attach(
                term,
                Kind::Step,
                list,
                &here,
                &early_scope,
                nest,
                Some(true),
            )?;
        }
        if step_live {
            let target = match stmts {
                Some(list) => list,
                None => self.sink(nest, &here),
            };
            self.lists[target].extend(step.stmts().iter().cloned());
            self.attach(term, Kind::Step, target, &here, &scope, nest, None)?;
        }
        Ok(())
    }

    fn place_carried(
        &mut self,
        term: &Fold,
        axis: Name,
        nest: usize,
        node: &[Name],
        scope: &HashSet<Name>,
        stmts: Option<usize>,
    ) -> Result<()> {
        if self
            .owned
            .get(&(term.address(), Kind::State))
            .is_some_and(|s| !s.is_empty())
        {
            return Err(Error::Value(
                "a carried state is stored per step, never after its loop".into(),
            ));
        }
        let mut inside = scope.clone();
        inside.insert(axis);
        let reads: HashSet<Name> = term
            .operands()
            .iter()
            .flat_map(|e| e.free_axes().iter().copied())
            .collect();
        let cells: Vec<Name> = term
            .cells()
            .iter()
            .filter(|c| !inside.contains(c))
            .copied()
            .collect();
        let extras: Vec<Name> = self
            .declared
            .iter()
            .filter(|n| reads.contains(n) && !inside.contains(n) && !term.cells().contains(n))
            .copied()
            .collect();
        let missing: Vec<Name> = std::iter::once(axis)
            .chain(cells.iter().copied())
            .chain(extras.iter().copied())
            .filter(|n| !self.has_coordinate(n) && !self.bound.contains(n))
            .collect();
        if !missing.is_empty() {
            let listed: Vec<String> = missing.iter().map(|n| format!("'{n}'")).collect();
            return Err(Error::Value(format!(
                "lower: no extent for [{}] — the kernel's axis table names a carried state's axis and cells",
                listed.join(", ")
            )));
        }
        let mut opened = cells.clone();
        opened.extend(extras);
        let local = self.new_nest(opened, inside.clone());
        let (placed, _) = self.placed(term)?;
        let mut within = inside.clone();
        within.extend(cells.iter().copied());
        for edge in placed {
            let path = if edge.free_axes().is_subset(&within) {
                Some(cells.clone())
            } else {
                None
            };
            self.place(&edge, &Vec::new(), path.as_deref(), local)?;
        }
        let target = self.sink(local, &cells);
        let step = term.step()?;
        self.lists[target].extend(step.stmts().iter().cloned());
        self.attach(term, Kind::Observed, target, &cells, &within, local, None)?;
        let body = self.assemble(local, &[])?;
        let outer = match stmts {
            Some(list) => list,
            None => self.sink(nest, node),
        };
        if self.bound.contains(&axis) {
            self.lists[outer].extend(body.stmts().iter().cloned());
        } else {
            let coordinate = self.coordinate(&axis).cloned().expect("checked above");
            self.lists[outer].push(Arc::new(Statement::Loop(Loop {
                axis: coordinate,
                body,
                unroll: false,
                seed: true,
                unrolled: false,
                range: None,
                step: None,
            })));
        }
        Ok(())
    }
}

impl Fold {
    /// The components each term in this tree supplies to all its readers and boundary stores, keyed by term address:
    /// demand propagated through narrowed lifts, a shared term keeping the union of its readers' requests.
    pub fn read_components(&self, stored: &HashSet<Name>) -> Result<HashMap<usize, Vec<Name>>> {
        let mut taken: HashMap<usize, Vec<Name>> = HashMap::new();
        let mut writes: HashMap<usize, bool> = HashMap::new();
        let mut pending: Vec<(Fold, Vec<Name>)> = vec![(self.clone(), self.exposes().to_vec())];
        let mut retained: Vec<Fold> = Vec::new();
        while let Some((term, asked)) = pending.pop() {
            let mut wanted: HashSet<Name> = taken
                .get(&term.address())
                .map(|t| t.iter().copied().collect())
                .unwrap_or_default();
            wanted.extend(asked);
            let names: Vec<Name> = term
                .exposes()
                .iter()
                .filter(|n| wanted.contains(n))
                .copied()
                .collect();
            if taken.get(&term.address()) == Some(&names) {
                continue;
            }
            taken.insert(term.address(), names.clone());
            retained.push(term.clone());
            let under = match writes.get(&term.address()) {
                Some(found) => *found,
                None => {
                    let mut lowering_writes = HashMap::new();
                    let found = writes_under(&term, stored, &mut lowering_writes)?;
                    writes.insert(term.address(), found);
                    found
                }
            };
            let narrowed = if under {
                Some(term.clone())
            } else if !names.is_empty() {
                Some(term.exposing(&names)?)
            } else {
                None
            };
            let read: HashSet<Name> = match &narrowed {
                Some(narrowed) => {
                    let mut read = narrowed.step()?.ssa_uses().clone();
                    read.extend(narrowed.exposes().iter().copied());
                    read
                }
                None => HashSet::new(),
            };
            for edge in term.operands() {
                let keep: Vec<Name> = edge
                    .exposes()
                    .iter()
                    .filter(|n| read.contains(n) || stored.contains(n))
                    .copied()
                    .collect();
                pending.push((edge.clone(), keep));
            }
        }
        drop(retained);
        Ok(taken)
    }

    /// Flatten this term to the Loop IR nest the materializer expands, the boundary `stores` placed in it. `bound`
    /// names the coordinates the caller binds (`None`: every free coordinate), `extents` the symbolic extents the
    /// launch binds, `axes` the kernel's axis table.
    pub fn lower(
        &self,
        bound: Option<&HashSet<Name>>,
        stores: &[Store],
        axes: &[Arc<Axis>],
        extents: &HashSet<Name>,
    ) -> Result<Body> {
        let sorted = |names: &HashSet<Name>| {
            let mut names: Vec<Name> = names.iter().copied().collect();
            names.sort();
            names
        };
        let request = Request {
            bound: bound.map(sorted),
            stores: stores.to_vec(),
            axes: axes.to_vec(),
            extents: sorted(extents),
        };
        if let Some((_, body)) = emmy_ir::locked(self.lowered())
            .iter()
            .find(|(asked, _)| asked.same(&request))
        {
            return Ok(body.clone());
        }
        let body = self.lower_fresh(bound, stores, axes, extents)?;
        emmy_ir::locked(self.lowered()).push((request, body.clone()));
        Ok(body)
    }

    /// For each axis a term of this tree walks as a block of a partition, the names the block's start reads
    /// ([`Fold::block_start`]): the block's coordinate.
    fn range_origins(&self, axes: &[Arc<Axis>]) -> Result<HashMap<Name, HashSet<Name>>> {
        let mut origins: HashMap<Name, HashSet<Name>> = HashMap::new();
        let mut seen: HashSet<usize> = HashSet::new();
        let mut pending = vec![self.clone()];
        while let Some(term) = pending.pop() {
            if !seen.insert(term.address()) {
                continue;
            }
            if let (Some(axis), Some(start)) = (term.axis(), term.block_start(axes)?) {
                origins.entry(axis).or_default().extend(start.free_vars());
            }
            pending.extend(term.operands().iter().cloned());
        }
        Ok(origins)
    }

    fn lower_fresh(
        &self,
        bound: Option<&HashSet<Name>>,
        stores: &[Store],
        axes: &[Arc<Axis>],
        extents: &HashSet<Name>,
    ) -> Result<Body> {
        let mut bound: HashSet<Name> = bound.cloned().unwrap_or_else(|| self.free_axes().clone());
        bound.extend(extents.iter().copied());
        let applied = self.applied()?;
        let spelled: HashMap<Name, Name> = self
            .lift()
            .params()
            .iter()
            .zip(applied.params())
            .filter(|(a, b)| a != b)
            .map(|(a, b)| (*a, *b))
            .collect();
        let stores: Vec<Store> = stores.iter().map(|s| s.renamed(&spelled)).collect();
        let mut coordinates: Vec<(Name, Arc<Axis>)> = Vec::new();
        for axis in axes {
            match coordinates.iter_mut().find(|(n, _)| *n == axis.name) {
                Some(slot) => slot.1 = axis.clone(),
                None => coordinates.push((axis.name, axis.clone())),
            }
        }
        let mut declared: Vec<Name> = Vec::new();
        let mut internal: HashSet<Name> = HashSet::new();
        let mut readers: HashMap<Name, usize> = HashMap::new();
        let mut origin: HashMap<Name, (usize, Kind)> = HashMap::new();
        let mut terms: HashMap<usize, Fold> = HashMap::new();
        let origins = self.range_origins(axes)?;
        let mut seen: HashSet<usize> = HashSet::new();
        let mut pending = vec![self.clone()];
        while let Some(term) = pending.pop() {
            if !seen.insert(term.address()) {
                continue;
            }
            terms.insert(term.address(), term.clone());
            let lead = usize::from(term.axis().is_some()) + term.bindings().len();
            for name in &term.lift().params()[lead..] {
                if !declared.contains(name) {
                    declared.push(*name);
                }
            }
            if let Some(axis) = term.axis() {
                internal.insert(axis);
            }
            if term.carries() {
                internal.extend(term.cells().iter().copied());
                for edge in term.operands() {
                    internal.extend(edge.free_axes().iter().copied());
                }
            }
            // A term read along a block of a partition is evaluated at `start + trip`: over the block's coordinate
            // too, as it would be were the start written into its index.
            let mut over: HashSet<Name> = term.free_axes().clone();
            for axis in term.free_axes() {
                over.extend(origins.get(axis).into_iter().flatten().copied());
            }
            for name in over {
                *readers.entry(name).or_default() += 1;
            }
            match term.base() {
                None => {
                    for stmt in term.lift().body().stmts() {
                        for name in stmt.defines() {
                            origin.insert(name, (term.address(), Kind::Step));
                        }
                    }
                }
                Some(base) => {
                    for name in base.results() {
                        origin.insert(*name, (term.address(), Kind::State));
                    }
                    if let Some(observe) = term.observe() {
                        for name in observe.results() {
                            origin.insert(*name, (term.address(), Kind::Observed));
                        }
                    }
                    if term.carries() {
                        for name in &term.exposes()[base.results().len()..] {
                            origin.insert(*name, (term.address(), Kind::Observed));
                        }
                    }
                }
            }
            pending.extend(term.operands().iter().rev().cloned());
        }
        let mut placed_at: HashMap<usize, usize> = HashMap::new();
        fn rank(term: &Fold, placed_at: &mut HashMap<usize, usize>) {
            if placed_at.contains_key(&term.address()) {
                return;
            }
            for edge in term.operands() {
                rank(edge, placed_at);
            }
            let next = placed_at.len();
            placed_at.insert(term.address(), next);
        }
        rank(self, &mut placed_at);
        let has_coordinate = |name: &Name, coordinates: &[(Name, Arc<Axis>)]| {
            coordinates.iter().any(|(n, _)| n == name)
        };
        let indexed: HashSet<Name> = stores
            .iter()
            .flat_map(|s| s.read_names())
            .filter(|n| {
                origin.contains_key(n)
                    && !has_coordinate(n, &coordinates)
                    && !bound.contains(n)
                    && !internal.contains(n)
            })
            .collect();
        let mut stored: HashSet<Name> = stores
            .iter()
            .flat_map(|s| s.values().iter().copied())
            .collect();
        stored.extend(indexed.iter().copied());
        let taken = self.read_components(&stored)?;
        let mut owned: HashMap<(usize, Kind), Vec<usize>> = HashMap::new();
        for (index, spec) in stores.iter().enumerate() {
            let first = spec
                .values()
                .first()
                .ok_or_else(|| Error::Value("a store writes a value".into()))?;
            let key = origin.get(first).copied();
            let Some(key) = key.filter(|k| spec.values().iter().all(|n| origin.get(n) == Some(k)))
            else {
                return Err(Error::Value(format!(
                    "a store over {:?} writes values no one term defines",
                    spec.values()
                )));
            };
            let mut rows: Vec<(usize, Kind)> = spec
                .read_names()
                .iter()
                .filter(|n| indexed.contains(n))
                .map(|n| origin[n])
                .collect();
            rows.sort_by_key(|k| (placed_at[&k.0], format!("{:?}", k.1)));
            rows.dedup();
            let mut best = key;
            for row in rows {
                if placed_at[&row.0] > placed_at[&best.0] {
                    best = row;
                }
            }
            owned.entry(best).or_default().push(index);
        }
        for spec in &stores {
            for axis in &spec.sweep {
                if !has_coordinate(&axis.name, &coordinates) {
                    coordinates.push((axis.name, axis.clone()));
                }
            }
        }
        for (name, _) in &coordinates {
            if !declared.contains(name) {
                declared.push(*name);
            }
        }
        let mut read: HashSet<Name> = self.free_axes().clone();
        for spec in &stores {
            read.extend(
                spec.read_names()
                    .into_iter()
                    .filter(|n| !indexed.contains(n)),
            );
        }
        let mut missing: Vec<Name> = read
            .iter()
            .filter(|n| {
                !bound.contains(n) && !internal.contains(n) && !has_coordinate(n, &coordinates)
            })
            .copied()
            .collect();
        if !missing.is_empty() {
            missing.sort();
            let listed: Vec<String> = missing.iter().map(|n| format!("'{n}'")).collect();
            return Err(Error::Value(format!(
                "lower: no extent for coordinates [{}] — the closed program takes them as axes",
                listed.join(", ")
            )));
        }
        let mut opened: Vec<Name> = coordinates
            .iter()
            .map(|(n, _)| *n)
            .filter(|n| {
                read.contains(n)
                    && !bound.contains(n)
                    && (!internal.contains(n) || self.free_axes().contains(n))
            })
            .collect();
        opened.sort_by_key(|n| {
            (
                std::cmp::Reverse(readers.get(n).copied().unwrap_or(0)),
                declared.iter().position(|d| d == n).unwrap_or(0),
            )
        });
        let mut lowering = Lowering {
            coordinates,
            bound: bound.clone(),
            internal,
            declared,
            origin,
            terms,
            indexed,
            stored,
            taken,
            owned,
            stores: &stores,
            lists: Vec::new(),
            nests: Vec::new(),
            emitted: HashSet::new(),
            retained: Vec::new(),
            writes: HashMap::new(),
        };
        let root = lowering.new_nest(opened, bound);
        lowering.place(self, &Vec::new(), None, root)?;
        lowering.assemble(root, &[])
    }
}

fn writes_under(
    term: &Fold,
    stored: &HashSet<Name>,
    memo: &mut HashMap<usize, bool>,
) -> Result<bool> {
    if let Some(found) = memo.get(&term.address()) {
        return Ok(*found);
    }
    let mut pending = vec![term.clone()];
    let mut seen = HashSet::new();
    let mut found = false;
    while let Some(node) = pending.pop() {
        if !seen.insert(node.address()) {
            continue;
        }
        let step = node.step()?;
        if node
            .exposes()
            .iter()
            .chain(step.defined())
            .any(|n| stored.contains(n))
        {
            found = true;
            break;
        }
        pending.extend(node.operands().iter().cloned());
    }
    memo.insert(term.address(), found);
    Ok(found)
}
