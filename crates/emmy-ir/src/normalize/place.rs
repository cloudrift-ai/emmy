//! Placement: every pure value once, at the shallowest scope on its consumers' path, and the free-name aliasing it
//! rests on.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use super::with_bodies;
use crate::body::{Body, free_names};
use crate::expr::Expression;
use crate::numbering::{Key, Number, Numbering, Param, value_numbers};
use crate::order::{Site, is_protocol, site_kinds};
use crate::statement::*;
use crate::{Error, Name, Result};

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
    let rename = |_: Roles, name: Name| alias.get(&name).copied();
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
            .map(|(k, v)| (*k, *v))
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
                while used.contains(&Name::new(&fresh)) {
                    fresh.push_str("_local");
                }
                let fresh: Name = fresh.into();
                used.insert(fresh);
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

/// The state of one placement walk.
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
    /// Whether a value may move to another scope (placement), or only reuse one its own scope computed already.
    motion: bool,
}
/// The statements a scope keeps, each pure value with the key it is available under.
type Kept = Vec<(Arc<Statement>, Option<PlaceKey>)>;
type Walked = (Kept, Vec<(usize, Arc<Statement>)>, Marks);
/// A block placed: the statement where it stays (or `None` when it moved up), what moved up, its marks.
type Placed = (Option<Arc<Statement>>, Vec<(usize, Arc<Statement>)>, Marks);
impl Placer<'_> {
    fn bump(&mut self, name: &Name, depth: usize) {
        let v = self.version.entry(*name).or_insert(0);
        *v += 1;
        self.version_depth.insert((*name, *v), depth);
    }
    fn read_of(&self, name: &Name) -> Marks {
        if self.written.contains(name) {
            BTreeSet::from([(*name, self.version.get(name).copied().unwrap_or(0))])
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
        if let Some(v) = self.version.get(&Name::new("")) {
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
            self.defined.insert(*name, depth);
            self.taint.insert(*name, marks.clone());
        }
    }
    /// The shallowest scope at or above `depth` that holds `names` and every version `marks` read. A version bumped in
    /// a deeper scope (a barrier inside a loop or a branch, whose scope has closed) bounds nothing below `depth`.
    fn target(&self, depth: usize, names: &HashSet<Name>, marks: &Marks) -> usize {
        if !self.motion {
            return depth;
        }
        let mut out = self.floor[depth];
        for name in names {
            out = out.max(self.depth_of(name));
        }
        for mark in marks {
            out = out.max(self.version_depth.get(mark).copied().unwrap_or(0));
        }
        out.min(depth)
    }
    fn lookup(&self, key: &PlaceKey, depth: usize) -> Option<Vec<Name>> {
        let from = if self.motion { 0 } else { depth };
        self.available[from..=depth]
            .iter()
            .find_map(|scope| scope.get(key).cloned())
    }
    fn walk(&mut self, body: &Body, depth: usize) -> Result<Walked> {
        let mut kept: Kept = Vec::new();
        let mut escaped: Vec<(usize, Arc<Statement>)> = Vec::new();
        let mut marks_here: Marks = BTreeSet::new();
        let mut counts: HashMap<Name, usize> = HashMap::new();
        for stmt in body.stmts() {
            for name in stmt.defines() {
                *counts.entry(name).or_default() += 1;
            }
        }
        // A carried state read ahead of its update in this scope is the previous step's: it lives here. The order's
        // reading of an update site is the one reading: a leaf, or a block holding the update under a condition or
        // inside a map loop of the step.
        for stmt in body.stmts() {
            for (name, kind) in site_kinds(stmt) {
                if kind != Site::Binding {
                    self.defined.entry(name).or_insert(depth);
                }
            }
        }
        for index in 0..body.len() {
            let here = body.at(index);
            let stmt = here.stmt();
            for name in stmt.defines() {
                self.alias.remove(&name);
            }
            if let Statement::Assign(a) = &**stmt
                && a.dtype.is_none()
                && let Some(read) = alias_read(&a.value)
            {
                let source = self.alias.get(&read).copied().unwrap_or(read);
                self.alias.insert(a.name, source);
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
                        kept.push((member, None));
                    } else {
                        escaped.push((where_, member));
                    }
                }
                marks_here.extend(marks);
                if let Some(built) = built {
                    kept.push((built, None));
                }
                continue;
            }
            let marks = self.own_taint(stmt);
            marks_here.extend(marks.iter().cloned());
            let (number, params) = self.numbering.numbers[self.numbering.ordinal(&here)]
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
                                self.alias.insert(*own, *other);
                            }
                        }
                        let Some(&where_) = self.defined.get(&names[0]) else {
                            return Err(Error::Value(
                                "a value available for reuse was never placed".into(),
                            ));
                        };
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
                if where_ == depth {
                    kept.push((emitted, Some(key)));
                } else {
                    escaped.push((where_, emitted));
                }
                continue;
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
            kept.push((rename_free(stmt, &self.alias), None));
        }
        Ok((kept, escaped, marks_here))
    }
    fn block(&mut self, stmt: &Arc<Statement>, depth: usize) -> Result<Placed> {
        let nested = stmt.nested();
        let members: Vec<Arc<Statement>> =
            nested.iter().flat_map(|child| child.iter_stmts()).collect();
        let mut writes: HashSet<Name> = members.iter().flat_map(|m| m.external_writes()).collect();
        for member in &members {
            if let Statement::Carry(c) = &**member {
                writes.insert(c.name);
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
        let scope = matches!(&**stmt, Statement::Loop(_) | Statement::StridedLoop(_));
        self.binders.push(stmt.binds_axes().into_iter().collect());
        self.floor
            .push(if scope { self.floor[depth] } else { depth + 1 });
        let entered = self.version.clone();
        let mut children: Vec<Walked> = Vec::new();
        for child in &nested {
            // Each child is its own scope, and each branch of a condition starts from the versions ahead of it.
            self.version = entered.clone();
            self.available.push(HashMap::new());
            let walked = self.walk(child, depth + 1)?;
            children.push(walked);
            self.available.pop();
        }
        self.version = entered;
        let mut up: Vec<(usize, Arc<Statement>)> = children
            .iter()
            .flat_map(|(_, escaped, _)| escaped.iter().cloned())
            .collect();
        if self.motion && is_cond && children.iter().all(|(kept, _, _)| !kept.is_empty()) {
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
                .zip(&nested)
                .map(|((kept, _, _), child)| {
                    child.rebuilt(kept.iter().map(|(stmt, _)| stmt.clone()).collect())
                })
                .collect(),
        );
        // The states the block still holds: a nested block that escaped settled its own at the scope it moved to.
        let exported: Vec<Name> = built
            .nested()
            .iter()
            .flat_map(|child| child.defined_states().to_vec())
            .collect();
        // A block moves up only when nothing in it is an effect: a write, a carried update, or an ordered protocol
        // (a barrier) that every thread must reach as often as it did.
        let mut where_ = depth;
        if !stmt.has_side_effects()
            && !members
                .iter()
                .any(|m| matches!(**m, Statement::Carry(_)) || is_protocol(m))
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
    fn common(&mut self, children: &mut [Walked], depth: usize) -> Vec<(usize, Arc<Statement>)> {
        let else_kept: Kept = children[1].0.clone();
        let mut in_else: HashMap<PlaceKey, usize> = HashMap::new();
        for (index, (_, key)) in else_kept.iter().enumerate() {
            if let Some(key) = key {
                in_else.entry(key.clone()).or_insert(index);
            }
        }
        let mut hoisted = Vec::new();
        let mut taken: HashSet<usize> = HashSet::new();
        let mut remaining = Vec::new();
        for (stmt, key) in children[0].0.clone() {
            let index = key.as_ref().and_then(|k| in_else.get(k).copied());
            if let (Some(key), Some(index)) = (&key, index)
                && !taken.contains(&index)
            {
                let where_ = self.target(depth, &Placer::reads(&stmt), &key.2);
                if where_ <= depth {
                    taken.insert(index);
                    for (other, own) in else_kept[index].0.defines().iter().zip(stmt.defines()) {
                        self.alias.insert(*other, own);
                    }
                    let defines = stmt.defines();
                    self.settle(&defines, where_, &key.2);
                    self.available[where_].insert(key.clone(), defines);
                    hoisted.push((where_, stmt));
                    continue;
                }
            }
            remaining.push((stmt, key));
        }
        children[0].0 = remaining;
        children[1].0 = else_kept
            .iter()
            .enumerate()
            .filter(|(index, _)| !taken.contains(index))
            .map(|(_, (stmt, key))| (rename_free(stmt, &self.alias), key.clone()))
            .collect();
        hoisted
    }
}

/// Place every pure value once, at the shallowest loop scope on its consumers' path that binds its coordinates and
/// holds its operands; a second computation of a value a scope can already see reads the first instead.
pub fn place_values(body: &Body) -> Result<Body> {
    place_values_keyed(body, &HashMap::new())
}

/// [`place_values`] with the carried states `states` names numbered under its keys rather than their names: a read of
/// any member of a group of congruent states (`congruent_states`) is the same value, so placement merges the cones
/// the members compute.
pub fn place_values_keyed(body: &Body, states: &HashMap<Name, Key>) -> Result<Body> {
    placed(body, states, true)
}

/// Every pure value a scope computes twice computed once, the second reading the first: placement's numbering
/// and availability with no motion, so nothing moves to another scope and no live range reaches past the scope
/// that already held the value. The states `states` names are numbered under its keys, as in
/// [`place_values_keyed`].
pub fn share_values(body: &Body, states: &HashMap<Name, Key>) -> Result<Body> {
    placed(body, states, false)
}

fn placed(body: &Body, states: &HashMap<Name, Key>, motion: bool) -> Result<Body> {
    let numbering = value_numbers(
        body,
        &|name| {
            states
                .get(name)
                .cloned()
                .unwrap_or_else(|| crate::numbering::spelled_key(name))
        },
        crate::numbering::Chains::Flat,
    );
    let mut written: HashSet<Name> = HashSet::new();
    body.walk(&mut |here| {
        written.extend(here.stmt().external_writes());
        if let Statement::Carry(c) = &**here.stmt() {
            written.insert(c.name);
        }
    });
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
        motion,
    };
    let (kept, escaped, _) = placer.walk(body, 0)?;
    debug_assert!(escaped.is_empty());
    Ok(body.rebuilt(kept.into_iter().map(|(stmt, _)| stmt).collect()))
}

/// The name a value is an alias of: the name itself, or the one operand of a copy.
fn alias_read(value: &Expression) -> Option<Name> {
    if let Some(name) = value.as_var() {
        return Some(*name);
    }
    match value.operation() {
        Some((op, args)) if op.info().copies && args.len() == 1 => args[0].as_var().copied(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elementwise::Operation;
    use crate::normalize::normalize_body;
    use crate::normalize::testing::*;

    /// The names a value reads, sorted.
    fn reads(value: &Expression) -> Vec<String> {
        let mut out: Vec<String> = value.free_vars().iter().map(|n| n.to_string()).collect();
        out.sort();
        out
    }

    /// Whether `value` applies `op` at its root.
    fn applies(value: &Expression, op: Operation) -> bool {
        value.operation().is_some_and(|(applied, _)| applied == op)
    }

    fn placed(text: &str) -> Body {
        place_values(&body(text)).unwrap()
    }

    fn unchanged(text: &str) {
        let given = body(text);
        assert_eq!(place_values(&given).unwrap(), given, "{text}");
    }

    /// How many statements of `out`, at any depth, `matches` holds.
    fn count(out: &Body, matches: &dyn Fn(&Statement) -> bool) -> usize {
        let mut found = 0;
        out.walk(&mut |here| found += matches(here.stmt()) as usize);
        found
    }

    /// A vector load's lanes are four values: the products of each lane stay four.
    #[test]
    fn the_lanes_of_one_load_are_values_apart() {
        let out = placed(
            "
            for i in 0..4:
                x0, x1, x2, x3 = load X[i]
                y0 = multiply(x0, 2.0)
                y1 = multiply(x1, 2.0)
                y2 = multiply(x2, 2.0)
                y3 = multiply(x3, 2.0)
                O[i, 0] = y0
                O[i, 1] = y1
                O[i, 2] = y2
                O[i, 3] = y3
            ",
        );
        assert_eq!(
            count(&out, &|s| matches!(s, Statement::Assign(_))),
            4,
            "{}",
            text(&out)
        );
    }

    /// Two accumulators seeded alike ahead of an unseeded loop hold two values inside it from the second step on:
    /// what reads them there reads the step before, not the seed.
    #[test]
    fn a_seeded_state_is_read_inside_its_loop_as_the_step_before() {
        let out = placed(
            "
            for r in 0..4:
                init f32 m1 = -1e+30
                init f32 m2 = -1e+30
                for k in 0..16 seed=false:
                    s = load S[r, k]
                    t = load T[r, k]
                    g1 = maximum(m1, s)
                    g2 = maximum(m2, s)
                    W1[r, k] = g1
                    W2[r, k] = g2
                    m1 <- maximum(m1, s) over k
                    m2 <- maximum(m2, t) over k
            ",
        );
        assert_eq!(
            count(&out, &|s| matches!(s, Statement::Assign(_))),
            2,
            "{}",
            text(&out)
        );
    }

    /// A barrier pins the loop that holds it, and a value read after that loop is placed where it is read.
    #[test]
    fn a_loop_that_holds_a_barrier_stays_and_bounds_nothing_below_it() {
        let given = "
            for r in 0..4:
                init f32 acc = 0
                for k in 0..16 seed=false:
                    x = load X[k]
                    acc <- add(acc, x) over k
                    Sync
                y = add(acc, 1.0)
                O[r] = y
            ";
        let out = placed(given);
        assert_eq!(loop_axes(&out), ["r"], "{}", text(&out));
        assert_eq!(count(&out, &|s| matches!(s, Statement::Sync(_))), 1);
    }

    /// A statement that updates a name in place reads the value it held: two MMAs of one product into two
    /// accumulators that held two values are two values.
    #[test]
    fn a_name_updated_in_place_keys_the_value_it_read() {
        let given = body(
            "
            RegFragment(name=c1, role=c, shape=(16, 8, 16), dtype=f32)
            RegFragment(name=c2, role=c, shape=(16, 8, 16), dtype=f32)
            LdmatrixLoad(frag=a, src_buffer=A, src_index=(0, 0), role=a)
            LdmatrixLoad(frag=b, src_buffer=B, src_index=(0, 0), role=b)
            LdmatrixLoad(frag=b2, src_buffer=B2, src_index=(0, 0), role=b)
            MmaSyncPtx(c_frag=c2, a_frag=a, b_frag=b2, shape=(16, 8, 16))
            MmaSyncPtx(c_frag=c1, a_frag=a, b_frag=b, shape=(16, 8, 16))
            MmaSyncPtx(c_frag=c2, a_frag=a, b_frag=b, shape=(16, 8, 16))
            ",
        );
        let numbering = crate::numbering::value_numbers(
            &given,
            &crate::numbering::spelled_key,
            crate::numbering::Chains::Flat,
        );
        let number = |index: usize| {
            numbering.numbers[numbering.ordinal(&given.at(index))]
                .clone()
                .map(|v| v.0)
        };
        assert_ne!(number(6), number(7));
    }

    /// What the whole driver leaves of `text`.
    fn normalized(text: &str) -> Body {
        normalize_body(&body(text)).unwrap()
    }

    /// The normal form of a fresh body of `out`'s statements, which no cache on `out` answers.
    fn renormalized(out: &Body) -> Body {
        normalize_body(&Body::new(out.stmts().to_vec())).unwrap()
    }

    /// The one value a scalar write stores.
    fn written(stmt: &Statement) -> &Name {
        match stmt {
            Statement::Write(write) if write.values.len() == 1 => &write.values[0],
            _ => panic!("not a scalar write: {}", stmt.ty()),
        }
    }

    /// The first name of every load in the tree, pre-order.
    fn load_names(body: &Body) -> Vec<String> {
        body.loads()
            .iter()
            .filter_map(|here| match &**here.stmt() {
                Statement::Load(l) => Some(l.names[0].to_string()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn typed_copy_is_a_cast_not_an_alias() {
        unchanged("f32 wide = copy(narrow)");
    }

    #[test]
    fn copy_aliases_do_not_leak_between_sibling_scopes() {
        let out = placed(
            "
            for i in 0..4:
                x = load A[i]
                y = copy(x)
                OA[i] = y
            for i in 0..4:
                x = load B[i]
                y = exp(x)
                OB[i] = y
            ",
        );
        assert_eq!(stmt_text(&inner(&out.stmts()[1]).stmts()[1]), "y = exp(x)");
    }

    #[test]
    fn rename_free_and_placement_reach_nested_statements() {
        for depth in [2, 24] {
            let mut spelled = String::from(
                "if outer:\n    shadow = exp(outer)\n    left[] = shadow\nelse:\n    right[] = outer",
            );
            for level in 0..depth {
                spelled = format!("for i{level} in 0..4:\n{}", indented(&spelled));
            }
            let alias = HashMap::from([
                (Name::from("outer"), Name::from("renamed")),
                (Name::from("shadow"), Name::from("wrong")),
            ]);
            let renamed = rename_free(&body(&spelled).stmts()[0], &alias);
            let eliminated = placed(&format!(
                "outer = copy(renamed)\nshadow = copy(wrong)\n{spelled}"
            ))
            .stmts()[0]
                .clone();
            for result in [renamed, eliminated] {
                let nest = Body::new(vec![result]);
                let branches = nest.of_types(&[StatementType::Cond]);
                let Statement::Cond(branch) = &**branches[0].stmt() else {
                    unreachable!()
                };
                assert_eq!(branch.cond.text(), "renamed");
                assert_eq!(text(&branch.body), "shadow = exp(renamed)\nleft[] = shadow");
                assert_eq!(text(&branch.else_body), "right[] = renamed");
            }
        }
    }

    #[test]
    fn cse_shares_selections_and_their_downstream_cones() {
        let out = placed(
            "
            let `zero` = 0
            let other_zero = 0
            left = select:
                x when p
                `zero` when 1
            right = select:
                x when p
                other_zero when 1
            a = exp(left)
            b = exp(right)
            out[0] = b
            ",
        );
        assert_eq!(statement_types(&out), ["Let", "Select", "Assign", "Write"]);
        let Statement::Write(write) = &*out.stmts()[3] else {
            unreachable!()
        };
        assert_eq!(strs(&write.values), ["a"]);
    }

    #[test]
    fn cse_selection_predicates_observe_rebound_coordinates() {
        unchanged(
            "
            a = select:
                x when k
                y when 1
            for k in 0..4:
                b = select:
                    x when k
                    y when 1
                out[k] = b
            ",
        );
    }

    #[test]
    fn cse_does_not_drop_repeated_accumulator_updates() {
        unchanged(
            "
            for k in 0..8:
                sum <- add(sum, x) over k
                sum <- add(sum, x) over k
            ",
        );
    }

    #[test]
    fn cse_does_not_alias_distinct_unseeded_accumulators() {
        unchanged(
            "
            for k in 0..8 seed=false:
                left <- add(left, x) over k
                right <- add(right, x) over k
            ",
        );
    }

    #[test]
    fn cse_partial_state_changes_invalidate_dependent_values() {
        unchanged(
            "
            for k in 0..8:
                before = exp(sum)
                sum <- add(sum, x) over k
                after = exp(sum)
                out[k] = after
            ",
        );
    }

    #[test]
    fn cse_nested_state_changes_invalidate_enclosing_values() {
        unchanged(
            "
            before = exp(sum)
            for k in 0..4 seed=false:
                sum <- add(sum, x) over k
            after = exp(sum)
            out[0] = after
            ",
        );
    }

    #[test]
    fn cse_does_not_reuse_a_staged_load_assignment() {
        unchanged("value = load x[0]\nvalue = load x[0] carried=load");
    }

    #[test]
    fn cse_removes_repeated_copies_of_one_binding() {
        let out = placed(
            "
            value = load x[0]
            first[0] = value
            value = load x[0]
            second[0] = value
            ",
        );
        assert_eq!(
            text(&out),
            "value = load x[0]\nfirst[0] = value\nsecond[0] = value"
        );
    }

    #[test]
    fn cse_never_aliases_a_value_to_an_overwritten_representative() {
        unchanged(
            "
            value = load x[0]
            copy = load x[0]
            value = load y[0]
            out[0] = copy
            ",
        );
        unchanged("value = exp(value)\nvalue = exp(value)");
    }

    #[test]
    fn cse_does_not_merge_reductions_with_distinct_explicit_seeds() {
        unchanged(
            "
            init f32 left = 0
            init f32 right = 1
            for k in 0..8:
                left <- add(left, x)
                right <- add(right, x)
            ",
        );
    }

    #[test]
    fn common_branch_load_cannot_cross_a_write_on_either_path() {
        let read = "    value = load x[0]\n    out[0] = value\n";
        let write = "    x[0] = replacement\n";
        for (left, right) in [
            (format!("{write}{read}"), read.to_string()),
            (read.to_string(), format!("{write}{read}")),
        ] {
            unchanged(&format!("if predicate:\n{left}else:\n{right}"));
        }
    }

    #[test]
    fn dedup_loads_closes_commutative_chains_after_aliasing() {
        // An alias that reverses sorted operands must not hide the rest of a duplicate chain.
        let out = placed(
            "
            z = load X[0]
            a = load X[0]
            m = load Y[0]
            z1 = multiply(m, z)
            a1 = multiply(a, m)
            z2 = add(m, z1)
            a2 = add(a1, m)
            left = subtract(m, z1)
            right = subtract(a1, m)
            O[0] = a2
            L[0] = left
            R[0] = right
            ",
        );
        let assigned: Vec<&str> = out
            .stmts()
            .iter()
            .filter_map(|stmt| match &**stmt {
                Statement::Assign(a) => Some(&*a.name),
                _ => None,
            })
            .collect();
        assert_eq!(assigned, ["z1", "z2", "left", "right"]);
        let n = out.len();
        assert_eq!(stmt_text(&out.stmts()[n - 4]), "right = z1 - m");
        assert_eq!(stmt_text(&out.stmts()[n - 3]), "O[0] = z2");
        assert_eq!(place_values(&out).unwrap(), out);
    }

    #[test]
    fn dedup_loads_still_rewires_an_inner_use_of_the_dropped_name() {
        // The loop only reads the dropped name, so the alias must reach inside.
        let out = placed(
            "
            in0 = load const[0]
            in1 = load const[0]
            for a in 0..4:
                v = add(in0, in1)
            ",
        );
        assert_eq!(load_names(&out), ["in0"]);
        let assigns: Vec<String> = out
            .iter_stmts()
            .iter()
            .filter(|stmt| matches!(&***stmt, Statement::Assign(_)))
            .map(stmt_text)
            .collect();
        assert_eq!(assigns, ["v = in0 + in0"]);
    }

    #[test]
    fn cse_reuses_a_dominating_read_until_the_write_on_that_branch() {
        let out = placed(
            "
            old = load x[0]
            if p:
                before = load x[0]
                out[0] = before
                x[0] = replacement
                after = load x[0]
                new[0] = after
            ",
        );
        assert_eq!(
            stmt_text(&inner(&out.stmts()[1]).stmts()[0]),
            "out[0] = old"
        );
        assert_eq!(load_names(&out), ["old", "after"]);
    }

    #[test]
    fn dedup_loads_rewires_every_vector_lane() {
        let out = placed(
            "
            x0, x1 = load X[0]
            y0, y1 = load X[0]
            sum = add(y0, y1)
            ",
        );
        assert_eq!(text(&out), "x0, x1 = load X[0]\nsum = x0 + x1");
    }

    #[test]
    fn dedup_loads_invalidates_a_read_after_writing_its_buffer() {
        let out = placed(
            "
            old = load B[0]
            replacement = load R[0]
            B[0] = replacement
            new = load B[0]
            O[0] = new
            ",
        );
        let from_b: Vec<&str> = out
            .stmts()
            .iter()
            .filter_map(|stmt| match &**stmt {
                Statement::Load(l) if &*l.input == "B" => Some(&*l.names[0]),
                _ => None,
            })
            .collect();
        assert_eq!(from_b, ["old", "new"]);
    }

    #[test]
    fn dedup_loads_does_not_reuse_a_read_across_a_loop_that_writes_its_buffer() {
        let out = placed(
            "
            before = load B[0]
            for a in 0..4:
                current = load B[0]
                B[0] = current
            ",
        );
        assert_eq!(
            stmt_text(&inner(&out.stmts()[1]).stmts()[0]),
            "current = load B[0]"
        );
    }

    #[test]
    fn hoists_invariant_load_above_loop() {
        let out = placed(
            "
            for a in 0..4:
                x = load X[]
                o[a] = x
            ",
        );
        assert_eq!(statement_types(&out), ["Load", "Loop"]);
        assert_eq!(load_names(&out), ["x"]);
        assert_eq!(axis(&out.stmts()[1]), "a");
        assert_eq!(statement_types(&inner(&out.stmts()[1])), ["Write"]);
    }

    #[test]
    fn hoists_inner_loop_and_consumer_together() {
        // An inner reduce loop invariant in the outer axis and the statement that consumes its accumulator move
        // above the outer loop together: hoisting only the consumer would leave it reading an accumulator still
        // defined inside the outer loop's body.
        let out = placed(
            "
            for a in 0..4:
                for b in 0..8:
                    xb = load X[b]
                    acc <- add(acc, xb)
                v = exp(acc)
                out[a] = v
            ",
        );
        assert_eq!(
            statement_types(&out),
            ["Loop", "Assign", "Loop"],
            "{}",
            text(&out)
        );
        assert_eq!(axis(&out.stmts()[0]), "b");
        assert_eq!(stmt_text(&out.stmts()[1]), "v = exp(acc)");
        assert_eq!(axis(&out.stmts()[2]), "a");
        assert_eq!(statement_types(&inner(&out.stmts()[2])), ["Write"]);
        assert_eq!(accum_names(&inner(&out.stmts()[0])), ["acc"]);
    }

    #[test]
    fn keeps_inner_loop_when_it_references_outer_axis() {
        let out = placed(
            "
            for a in 0..4:
                for b in 0..8:
                    xab = load X[a, b]
                    acc <- add(acc, xab)
                out[a] = acc
            ",
        );
        assert_eq!(loop_axes(&out), ["a"]);
        assert_eq!(loop_axes(&inner(&out.stmts()[0])), ["b"]);
    }

    #[test]
    fn hoists_invariant_load_above_strided_loop() {
        // A strided loop binds a sequential axis the way a loop does, so it partitions its body the same way.
        let out = placed(
            "
            for a in 0..4 step 1:
                x = load X[]
                o[a] = x
            ",
        );
        assert_eq!(statement_types(&out), ["Load", "StridedLoop"]);
        assert_eq!(load_names(&out), ["x"]);
        assert_eq!(axis(&out.stmts()[1]), "a");
        assert_eq!(statement_types(&inner(&out.stmts()[1])), ["Write"]);
    }

    #[test]
    fn hoists_inner_strided_loop_and_consumer_together() {
        let out = placed(
            "
            for a in 0..4:
                for b in 0..8 step 1:
                    xb = load X[b]
                    acc <- add(acc, xb)
                v = exp(acc)
                out[a] = v
            ",
        );
        assert_eq!(
            statement_types(&out),
            ["StridedLoop", "Assign", "Loop"],
            "{}",
            text(&out)
        );
        assert_eq!(axis(&out.stmts()[0]), "b");
        assert_eq!(stmt_text(&out.stmts()[1]), "v = exp(acc)");
        assert_eq!(axis(&out.stmts()[2]), "a");
        assert_eq!(statement_types(&inner(&out.stmts()[2])), ["Write"]);
    }

    #[test]
    fn does_not_hoist_block_containing_write() {
        // The inner loop reads nothing of the outer axis but holds a write: hoisting it would change how often the
        // write happens, so it stays.
        let out = placed(
            "
            for a in 0..4:
                for b in 0..8:
                    xb = load X[b]
                    scratch[b] = xb
                out[a] = xb
            ",
        );
        assert_eq!(loop_axes(&out), ["a"]);
        assert_eq!(loop_axes(&inner(&out.stmts()[0])), ["b"]);
    }

    #[test]
    fn hoist_keeps_a_read_of_a_buffer_the_loop_writes() {
        // Axis-invariance alone earns no hoist: a read behind a write to its own buffer, or behind an ordered
        // execution protocol, follows the scope's ordering constraints and stays.
        let out = placed(
            "
            for i in 0..4:
                O[0] = one
                back = load O[0]
                Sync
                staged = load S[0]
                P[i] = back
                Q[i] = staged
            ",
        );
        assert_eq!(statement_types(&out), ["Loop"]);
        assert_eq!(
            statement_types(&inner(&out.stmts()[0])),
            ["Write", "Load", "Sync", "Load", "Write", "Write"]
        );
    }

    #[test]
    fn escaped_reduce_loop_keeps_its_state_at_the_scope_it_moved_to() {
        // An invariant reduce loop two scopes down moves to the top. The loops it left no longer hold its state,
        // so they must not settle it at their own scope: the read of `accc` inside `b` would otherwise pin the `a`
        // loop below its own scope and the read of `acca` after it would index a scope that does not exist. The
        // shape is a GDN decoder layer's fused normalization (Qwen3.5-9B, layer 0).
        let out = placed(
            "
            z = load Z[]
            for a in 0..4:
                for b in 0..4:
                    for c in 0..4:
                        accc <- add(accc, z)
                    x = load X[a, b]
                    v1 = add(accc, x)
                    accb <- add(accb, v1)
                acca <- add(acca, accb)
            v3 = exp(acca)
            Y[] = v3
            ",
        );
        assert_eq!(
            statement_types(&out),
            ["Load", "Loop", "Loop", "Assign", "Write"],
            "{}",
            text(&out)
        );
        assert_eq!(axis(&out.stmts()[1]), "c");
        assert_eq!(axis(&out.stmts()[2]), "a");
        let inner_b = inner(&inner(&out.stmts()[2]).stmts()[0]);
        assert!(inner_b.stmts().iter().any(
            |stmt| matches!(&**stmt, Statement::Assign(a) if reads(&a.value) == ["accc", "x"])
        ));
    }

    #[test]
    fn normalization_dedups_loads_and_rewires_gather_indices() {
        let out = normalized(
            "
            for a in 0..4:
                idx0 = load indices[a]
                idx1 = load indices[a]
                x0 = load values[idx0]
                x1 = load values[idx1]
                sum = add(x0, x1)
                out[a] = sum
            ",
        );
        assert_eq!(out.len(), 1, "{}", text(&out));
        let looped = inner(&out.stmts()[0]);
        let loads: Vec<String> = looped
            .stmts()
            .iter()
            .filter(|stmt| matches!(&***stmt, Statement::Load(_)))
            .map(stmt_text)
            .collect();
        assert_eq!(loads, ["in0 = load indices[a0]", "in1 = load values[in0]"]);
        assert_eq!(stmt_text(&looped.stmts()[looped.len() - 2]), "v0 = 2 * in1");
    }

    #[test]
    fn normalization_keeps_the_values_of_hoisted_shadowed_loads() {
        // Motion preserves lexical bindings before CSE merges equal loads.
        for inner_input in ["x", "y"] {
            let out = normalized(&format!(
                "
                value = load x[0]
                for i in 0..4:
                    value = load {inner_input}[0]
                    inner[i] = value
                outer[0] = value
                "
            ));
            let loads: HashMap<Name, Name> = out
                .stmts()
                .iter()
                .filter_map(|stmt| match &**stmt {
                    Statement::Load(load) => Some((load.names[0], load.input)),
                    _ => None,
                })
                .collect();
            let writes: HashMap<Name, Name> = out
                .writes()
                .iter()
                .filter_map(|here| match &**here.stmt() {
                    Statement::Write(write) => Some((write.output, *written(here))),
                    _ => None,
                })
                .collect();
            assert_eq!(&*loads[&writes[&Name::new("outer")]], "x");
            assert_eq!(&*loads[&writes[&Name::new("inner")]], inner_input);
            assert_eq!(loads.len(), if inner_input == "x" { 1 } else { 2 });
            assert_eq!(renormalized(&out), out);
        }
    }

    #[test]
    fn normalization_closes_simplification_cse_and_invariant_motion() {
        // An equal gather index exposes a constant predicate, then an invariant shared cone.
        let out = normalized(
            "
            x = load x[0]
            for i in 0..4:
                a = load indices[i]
                b = load indices[i]
                s = select:
                    x when a - b < 1
                    a when 1
                v = exp(s)
                out[i] = v
            ",
        );
        assert!(
            out.stmts().iter().any(
                |stmt| matches!(&**stmt, Statement::Assign(a) if applies(&a.value, Operation::Exp))
            ),
            "{}",
            text(&out)
        );
        assert_eq!(renormalized(&out), out);
    }

    #[test]
    fn cse_factors_common_branch_cones_without_speculation() {
        let branch = |prefix: &str, extra: &str| {
            indented(&format!(
                "{prefix}i = load indices[0]\n{prefix}x = load x[{prefix}i]\n{prefix}v = exp({prefix}x)\n\
                 {prefix}only = load {extra}[0]\n{prefix}out = add({prefix}v, {prefix}only)\nout[0] = {prefix}out"
            ))
        };
        let out = normalized(&format!(
            "if predicate:\n{}\nelse:\n{}",
            branch("a", "left"),
            branch("b", "right")
        ));
        assert_eq!(
            statement_types(&out),
            ["Load", "Load", "Assign", "Cond"],
            "{}",
            text(&out)
        );
        let Statement::Assign(shared) = &*out.stmts()[2] else {
            unreachable!()
        };
        assert!(applies(&shared.value, Operation::Exp));
        for arm in out.stmts()[3].nested() {
            assert_eq!(statement_types(&arm), ["Load", "Assign", "Write"]);
            let Statement::Assign(sum) = &*arm.stmts()[1] else {
                unreachable!()
            };
            assert!(sum.value.free_vars().contains(&shared.name));
        }
        assert_eq!(renormalized(&out), out);
    }

    #[test]
    fn normalized_quotient_addresses_fuse_and_share_the_whole_reduction() {
        let reduction = |axis: &str, name: &str, nested: bool| {
            let index = if nested {
                format!("{axis} / 128 / 2")
            } else {
                format!("{axis} / 256")
            };
            format!(
                "for {axis} in 0..1024:\n    {name}x = load x[{index}]\n    {name}v = exp({name}x)\n    \
                 {name} <- add({name}, {name}v) over {axis}"
            )
        };
        let out = normalized(&format!(
            "{}\n{}\nout[0] = right\nother[0] = left",
            reduction("i", "left", true),
            reduction("j", "right", false)
        ));
        assert_eq!(out.loads().len(), 1, "{}", text(&out));
        assert_eq!(out.accums().len(), 1);
        assert_eq!(out.loops().len(), 1);
        let n = out.len();
        assert_eq!(written(&out.stmts()[n - 1]), written(&out.stmts()[n - 2]));
        assert_eq!(renormalized(&out), out);
    }

    #[test]
    fn normalization_dedups_identical_accumulations() {
        // Two consumers of one fused value carry two copies of its accumulation in one reduce loop after the splice;
        // the second is the first under another name.
        let out = normalized(
            "
            for m in 0..4:
                for k in 0..8:
                    x0 = load x[m, k]
                    w0 = load w[k]
                    p0 = multiply(x0, w0)
                    acc0 <- add(acc0, p0) over k
                    x1 = load x[m, k]
                    w1 = load w[k]
                    p1 = multiply(x1, w1)
                    acc1 <- add(acc1, p1) over k
                gate = exp(acc0)
                up = exp(acc1)
                out = multiply(gate, up)
                out[m] = out
            ",
        );
        assert_eq!(out.len(), 1, "{}", text(&out));
        let outer = inner(&out.stmts()[0]);
        let reduces: Vec<&Arc<Statement>> = outer
            .stmts()
            .iter()
            .filter(|stmt| matches!(&***stmt, Statement::Loop(_)))
            .collect();
        let [reduce] = reduces[..] else {
            panic!("one reduce loop: {}", text(&out))
        };
        assert_eq!(
            statement_types(&inner(reduce)),
            ["Load", "Load", "Assign", "Accum"]
        );
        let product = outer
            .stmts()
            .iter()
            .filter_map(|stmt| match &**stmt {
                Statement::Assign(assign) => Some(assign),
                _ => None,
            })
            .next_back()
            .unwrap();
        assert_eq!(product.value.text(), "v1 * v1");
    }
}
