//! Placement: every pure value once, at the shallowest scope on its consumers' path, and the free-name aliasing it
//! rests on.

use super::*;

// --- placement -----------------------------------------------------------------------------------------------

/// `rewrite` under an alias map, made hygienic: the rename stops at a nested scope that re-binds an aliased name.
pub fn rename_free(stmt: &Arc<Statement>, alias: &HashMap<Name, Name>) -> Arc<Statement> {
    if alias.is_empty() {
        return stmt.clone();
    }
    let bodies = stmt.nested();
    let shell = if bodies.is_empty() {
        stmt.clone()
    } else {
        with_bodies(stmt, bodies.iter().map(|_| Body::empty()).collect())
    };
    let rename = |name: &str| alias.get(name).cloned();
    let empty = HashMap::new();
    let renamed = Statement::rewrite(
        &shell,
        &crate::protocol::Rewrite {
            rename: &rename,
            sigma: &empty,
            axis_fn: None,
        },
    );
    if bodies.is_empty() {
        return renamed;
    }
    let binds: HashSet<Name> = stmt.binds_axes().into_iter().collect();
    let mut inner = Vec::new();
    for b in bodies {
        let bound = b.local_defs();
        let pruned: HashMap<Name, Name> = alias
            .iter()
            .filter(|(k, _)| !bound.contains(*k) && !binds.contains(*k))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        // The surviving value may itself be shadowed here: rename the local binding before substituting.
        let mut captured: Vec<Name> = bound
            .iter()
            .filter(|name| pruned.values().any(|v| v == *name))
            .cloned()
            .collect();
        let mut b = b.clone();
        if !captured.is_empty() {
            captured.sort();
            let mut used: HashSet<Name> = b.definitions().keys().cloned().collect();
            used.extend(b.ssa_uses().iter().cloned());
            used.extend(b.axis_names().iter().cloned());
            used.extend(alias.keys().cloned());
            used.extend(alias.values().cloned());
            let mut renames: HashMap<Name, Name> = HashMap::new();
            for name in captured {
                let mut fresh = format!("{name}_local");
                while used.contains(&*fresh) {
                    fresh.push_str("_local");
                }
                let fresh: Name = fresh.into();
                used.insert(fresh.clone());
                renames.insert(name, fresh);
            }
            b = b.map_each(&mut |member| Statement::rename(member, &renames));
        }
        inner.push(b.map_each(&mut |c| rename_free(c, &pruned)));
    }
    with_bodies(&renamed, inner)
}

pub(crate) type Marks = BTreeSet<(Name, u64)>;
/// What tells two values apart for placement: the number applied to its coordinates, under the versions it read.
pub(crate) type PlaceKey = (Number, Vec<Param>, Marks);

/// Place every pure value once, at the shallowest loop scope on its consumers' path that binds its coordinates and
/// holds its operands; a second computation of a value a scope can already see reads the first instead.
pub fn place_values(body: &Body) -> Result<Body> {
    let body = distinct(body);
    let numbering = value_numbers(&body, &crate::numbering::spelled_key);
    let mut written: HashSet<Name> = HashSet::new();
    body.walk(&mut |here| {
        written.extend(here.stmt().external_writes());
        if let Statement::Carry(c) = &**here.stmt() {
            written.insert(c.name.clone());
        }
    });
    struct Placer<'a> {
        numbering: &'a Numbering,
        written: HashSet<Name>,
        alias: HashMap<Name, Name>,
        defined: HashMap<Name, usize>,
        taint: HashMap<Name, Marks>,
        version: HashMap<Name, u64>,
        version_depth: HashMap<(Name, u64), usize>,
        binders: Vec<HashSet<Name>>,
        available: Vec<HashMap<PlaceKey, Vec<Name>>>,
        floor: Vec<usize>,
        key_of: HashMap<usize, PlaceKey>,
        seeds: HashSet<Name>,
    }
    type Walked = (Vec<Arc<Statement>>, Vec<(usize, Arc<Statement>)>, Marks);
    /// A block placed: the statement where it stays (or `None` when it moved up), what moved up, its marks.
    type Placed = (Option<Arc<Statement>>, Vec<(usize, Arc<Statement>)>, Marks);
    impl Placer<'_> {
        fn bump(&mut self, name: &Name, depth: usize) {
            let v = self.version.entry(name.clone()).or_insert(0);
            *v += 1;
            self.version_depth.insert((name.clone(), *v), depth);
        }
        fn read_of(&self, name: &Name) -> Marks {
            if self.written.contains(name) {
                BTreeSet::from([(name.clone(), self.version.get(name).copied().unwrap_or(0))])
            } else {
                BTreeSet::new()
            }
        }
        fn depth_of(&self, name: &Name) -> usize {
            let name = self.alias.get(name).unwrap_or(name);
            if let Some(depth) = self.defined.get(name) {
                return *depth;
            }
            (0..self.binders.len())
                .rev()
                .find(|depth| self.binders[*depth].contains(name))
                .unwrap_or(0)
        }
        fn pure(stmt: &Statement) -> bool {
            (stmt.is_pure() || matches!(stmt, Statement::Call(_)))
                && stmt.nested().is_empty()
                && !matches!(stmt, Statement::Load(l) if !l.carried.is_empty())
        }
        fn reads(stmt: &Statement) -> HashSet<Name> {
            let mut out: HashSet<Name> = stmt.deps().into_iter().collect();
            for e in stmt.expressions() {
                out.extend(e.free_vars());
            }
            out
        }
        fn own_taint(&self, stmt: &Statement) -> Marks {
            let mut out: Marks = BTreeSet::new();
            for name in stmt.deps() {
                let name = self.alias.get(&name).unwrap_or(&name);
                if let Some(marks) = self.taint.get(name) {
                    out.extend(marks.iter().cloned());
                }
            }
            if let Some(v) = self.version.get("") {
                out.insert((Name::from(""), *v));
            }
            for name in stmt.external_reads() {
                out.extend(self.read_of(&name));
            }
            if let Statement::Pre(p) = stmt {
                out.extend(self.read_of(&p.carrier));
            }
            out
        }
        fn settle(&mut self, names: &[Name], depth: usize, marks: &Marks) {
            for name in names {
                self.defined.insert(name.clone(), depth);
                self.taint.insert(name.clone(), marks.clone());
            }
        }
        fn target(&self, depth: usize, names: &HashSet<Name>, marks: &Marks) -> usize {
            let mut out = self.floor[depth];
            for name in names {
                out = out.max(self.depth_of(name));
            }
            for mark in marks {
                out = out.max(self.version_depth.get(mark).copied().unwrap_or(0));
            }
            out
        }
        fn lookup(&self, key: &PlaceKey, depth: usize) -> Option<Vec<Name>> {
            self.available[..=depth]
                .iter()
                .find_map(|scope| scope.get(key).cloned())
        }
        fn walk(&mut self, body: &Body, depth: usize, seeded: bool) -> Result<Walked> {
            let mut kept: Vec<Arc<Statement>> = Vec::new();
            let mut escaped: Vec<(usize, Arc<Statement>)> = Vec::new();
            let mut marks_here: Marks = BTreeSet::new();
            let mut counts: HashMap<Name, usize> = HashMap::new();
            for stmt in body.stmts() {
                for name in stmt.defines() {
                    *counts.entry(name).or_default() += 1;
                }
            }
            let uses = body.ssa_uses();
            // A carried state read ahead of its update in this scope is the previous step's: it lives here.
            for stmt in body.stmts() {
                match &**stmt {
                    Statement::Accum(a) => {
                        self.defined.entry(a.name.clone()).or_insert(depth);
                    }
                    Statement::Carry(c) => {
                        self.defined.entry(c.name.clone()).or_insert(depth);
                    }
                    _ => {}
                }
            }
            for stmt in body.stmts() {
                for name in stmt.defines() {
                    self.alias.remove(&name);
                }
                if let Statement::Assign(a) = &**stmt
                    && &*a.op.name == "copy"
                    && a.args.len() == 1
                    && a.dtype.is_none()
                {
                    let source = self
                        .alias
                        .get(&a.args[0])
                        .cloned()
                        .unwrap_or_else(|| a.args[0].clone());
                    self.alias.insert(a.name.clone(), source.clone());
                    let (depth_of, marks) = (
                        self.depth_of(&source),
                        self.taint.get(&source).cloned().unwrap_or_default(),
                    );
                    self.settle(std::slice::from_ref(&a.name), depth_of, &marks);
                    continue;
                }
                if !stmt.nested().is_empty() {
                    let (built, up, marks) = self.block(stmt, depth)?;
                    for (where_, member) in up {
                        if where_ == depth {
                            kept.push(member);
                        } else {
                            escaped.push((where_, member));
                        }
                    }
                    marks_here.extend(marks);
                    if let Some(built) = built {
                        kept.push(built);
                    }
                    continue;
                }
                let marks = self.own_taint(stmt);
                marks_here.extend(marks.iter().cloned());
                let (number, params) = self.numbering.numbers[self.numbering.ordinal(stmt)]
                    .as_ref()
                    .map_or((0, Vec::new()), |v| (v.0, v.1.clone()));
                let key: PlaceKey = (number, params, marks.clone());
                let defines = stmt.defines();
                if Placer::pure(stmt) {
                    if let Some(names) = self.lookup(&key, depth) {
                        // A repeated copy of one binding changes nothing; another binding of the value reads the first,
                        // unless that one is rebound later.
                        if names == defines
                            || names
                                .iter()
                                .all(|name| counts.get(name).copied().unwrap_or(0) <= 1)
                        {
                            if names != defines {
                                for (own, other) in defines.iter().zip(&names) {
                                    self.alias.insert(own.clone(), other.clone());
                                }
                            }
                            let where_ = self.defined[&names[0]];
                            self.settle(&defines, where_, &marks);
                            continue;
                        }
                    }
                    let where_ = self.target(depth, &Placer::reads(stmt), &marks);
                    self.settle(&defines, where_, &marks);
                    self.available[where_]
                        .entry(key.clone())
                        .or_insert_with(|| defines.clone());
                    let emitted = rename_free(stmt, &self.alias);
                    self.key_of.insert(ptr(&emitted), key);
                    if where_ == depth {
                        kept.push(emitted);
                    } else {
                        escaped.push((where_, emitted));
                    }
                    continue;
                }
                if let Statement::Accum(a) = &**stmt {
                    let reduction = seeded
                        && counts.get(&a.name).copied().unwrap_or(0) == 1
                        && !uses.contains(&a.name)
                        && !self.seeds.contains(&a.name);
                    if reduction && let Some(names) = self.available[depth].get(&key).cloned() {
                        self.alias.insert(a.name.clone(), names[0].clone());
                        self.settle(std::slice::from_ref(&a.name), depth, &marks);
                        continue;
                    }
                    if reduction {
                        self.available[depth].insert(key.clone(), vec![a.name.clone()]);
                    }
                }
                if let Statement::Init(i) = &**stmt {
                    self.seeds.insert(i.name.clone());
                }
                for name in stmt.external_writes() {
                    self.bump(&name, depth);
                }
                if let Statement::Carry(c) = &**stmt {
                    self.bump(&c.name, depth);
                }
                if is_protocol(stmt) {
                    self.bump(&Name::from(""), depth);
                }
                let settled: Vec<Name> = defines.into_iter().chain(stmt.local_decls()).collect();
                self.settle(&settled, depth, &marks);
                kept.push(rename_free(stmt, &self.alias));
            }
            Ok((kept, escaped, marks_here))
        }
        fn block(&mut self, stmt: &Arc<Statement>, depth: usize) -> Result<Placed> {
            let nested = stmt.nested();
            let members: Vec<Arc<Statement>> =
                nested.iter().flat_map(|child| child.iter_stmts()).collect();
            let mut writes: HashSet<Name> =
                members.iter().flat_map(|m| m.external_writes()).collect();
            for member in &members {
                if let Statement::Carry(c) = &**member {
                    writes.insert(c.name.clone());
                }
                if is_protocol(member) {
                    writes.insert(Name::from(""));
                }
            }
            let is_cond = matches!(**stmt, Statement::Cond(_));
            if !is_cond {
                // A loop's write may precede a read on a back edge; a branch has no back edge.
                let mut sorted: Vec<&Name> = writes.iter().collect();
                sorted.sort();
                for name in sorted {
                    self.bump(name, depth + 1);
                }
            }
            let (scope, seed) = match &**stmt {
                Statement::Loop(l) => (true, l.seed),
                Statement::StridedLoop(l) => (true, l.seed),
                _ => (false, false),
            };
            self.binders.push(stmt.binds_axes().into_iter().collect());
            self.floor
                .push(if scope { self.floor[depth] } else { depth + 1 });
            let seeded = scope && seed;
            let entered = self.version.clone();
            let mut children: Vec<Walked> = Vec::new();
            for child in &nested {
                // Each child is its own scope, and each branch of a condition starts from the versions ahead of it.
                self.version = entered.clone();
                self.available.push(HashMap::new());
                let walked = self.walk(child, depth + 1, seeded)?;
                children.push(walked);
                self.available.pop();
            }
            self.version = entered;
            let mut up: Vec<(usize, Arc<Statement>)> = children
                .iter()
                .flat_map(|(_, escaped, _)| escaped.iter().cloned())
                .collect();
            if is_cond && children.iter().all(|(kept, _, _)| !kept.is_empty()) {
                up.extend(self.common(&mut children, depth));
            }
            self.binders.pop();
            self.floor.pop();
            let mut sorted: Vec<&Name> = writes.iter().collect();
            sorted.sort();
            for name in sorted {
                self.bump(name, depth);
            }
            let mut marks: Marks = BTreeSet::new();
            for (_, _, child_marks) in &children {
                marks.extend(child_marks.iter().cloned());
            }
            let shell = rename_free(
                &with_bodies(stmt, nested.iter().map(|_| Body::empty()).collect()),
                &self.alias,
            );
            let built = with_bodies(
                &shell,
                children
                    .iter()
                    .map(|(kept, _, _)| Body::new(kept.clone()))
                    .collect(),
            );
            // The states the block still holds: a nested block that escaped settled its own at the scope it moved to.
            let exported: Vec<Name> = built
                .nested()
                .iter()
                .flat_map(|child| child.carried_names().to_vec())
                .collect();
            let mut where_ = depth;
            if !stmt.has_side_effects()
                && !members.iter().any(|m| matches!(**m, Statement::Carry(_)))
            {
                where_ = self.target(depth, &free_names(&built).into_iter().collect(), &marks);
            }
            self.settle(&exported, where_, &marks);
            if where_ < depth {
                up.push((where_, built));
                return Ok((None, up, marks));
            }
            Ok((Some(built), up, marks))
        }
        /// Values both branches of a condition compute, with their operands held ahead of it, move ahead of it.
        fn common(
            &mut self,
            children: &mut [Walked],
            depth: usize,
        ) -> Vec<(usize, Arc<Statement>)> {
            let else_kept: Vec<Arc<Statement>> = children[1].0.clone();
            let mut in_else: HashMap<PlaceKey, usize> = HashMap::new();
            for (index, stmt) in else_kept.iter().enumerate() {
                if let Some(key) = self.key_of.get(&ptr(stmt)) {
                    in_else.entry(key.clone()).or_insert(index);
                }
            }
            let mut hoisted = Vec::new();
            let mut taken: HashSet<usize> = HashSet::new();
            let mut remaining = Vec::new();
            for stmt in children[0].0.clone() {
                let key = self.key_of.get(&ptr(&stmt)).cloned();
                let index = key.as_ref().and_then(|k| in_else.get(k).copied());
                if let (Some(key), Some(index)) = (key, index)
                    && !taken.contains(&index)
                {
                    let where_ = self.target(depth, &Placer::reads(&stmt), &key.2);
                    if where_ <= depth {
                        taken.insert(index);
                        for (other, own) in else_kept[index].defines().iter().zip(stmt.defines()) {
                            self.alias.insert(other.clone(), own);
                        }
                        let defines = stmt.defines();
                        self.settle(&defines, where_, &key.2);
                        self.available[where_].insert(key, defines);
                        hoisted.push((where_, stmt));
                        continue;
                    }
                }
                remaining.push(stmt);
            }
            children[0].0 = remaining;
            children[1].0 = else_kept
                .iter()
                .enumerate()
                .filter(|(index, _)| !taken.contains(index))
                .map(|(_, stmt)| rename_free(stmt, &self.alias))
                .collect();
            hoisted
        }
    }
    let mut placer = Placer {
        numbering: &numbering,
        written,
        alias: HashMap::new(),
        defined: HashMap::new(),
        taint: HashMap::new(),
        version: HashMap::new(),
        version_depth: HashMap::new(),
        binders: vec![HashSet::new()],
        available: vec![HashMap::new()],
        floor: vec![0],
        key_of: HashMap::new(),
        seeds: HashSet::new(),
    };
    let (kept, escaped, _) = placer.walk(&body, 0, false)?;
    debug_assert!(escaped.is_empty());
    Ok(Body::new(kept))
}
