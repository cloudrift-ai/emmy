//! The folds: a select whose branches bind one value, an operation over literals, a value nothing reads, and a sum
//! into one `Affine` statement.

use super::*;

// --- folds ----------------------------------------------------------------------------------------------

/// A `Select` whose branches all bind one value is a copy of it.
pub fn fold_trivial_selects(body: &Body) -> Body {
    body.map_stmts(&mut |stmt| {
        if let Statement::Select(s) = &**stmt
            && s.branches.iter().all(|b| b.value == s.branches[0].value)
        {
            return Some(assign(
                &s.name,
                "copy",
                vec![s.branches[0].value.clone()],
                None,
            ));
        }
        Some(stmt.clone())
    })
}

pub(crate) fn let_literal(name: &Name, value: f64, dtype: Option<DType>) -> Arc<Statement> {
    Arc::new(Statement::Let(Let {
        name: name.clone(),
        value: Expression::float(value),
        dtype: Some(dtype.unwrap_or_else(|| DType::new("f32"))),
    }))
}

/// Evaluate every operation whose operands are all literals into the literal it computes.
pub fn fold_constants(body: &Body) -> Body {
    fn evaluated(stmt: &Assign, env: &HashMap<Name, f64>) -> Option<f64> {
        if stmt.args.is_empty()
            || stmt.args.iter().any(|arg| !env.contains_key(arg))
            || stmt.dtype.as_ref().is_some_and(|d| !is_float(d.name()))
        {
            return None;
        }
        let values: Vec<f64> = stmt.args.iter().map(|arg| env[arg]).collect();
        if &*stmt.op.name == "copy" {
            return Some(values[0]);
        }
        let args: Vec<f32> = values.iter().map(|v| *v as f32).collect();
        evaluate_f32(&stmt.op.name, &args).map(|v| v as f64)
    }
    fn scope(body: &Body, env: &HashMap<Name, f64>) -> Body {
        let mut env = env.clone();
        let mut out = Vec::new();
        for stmt in body.stmts() {
            let mut stmt = stmt.clone();
            let nested = stmt.nested();
            if !nested.is_empty() {
                stmt = with_bodies(
                    &stmt,
                    nested.iter().map(|child| scope(child, &env)).collect(),
                );
            }
            let folded = match &*stmt {
                Statement::Assign(a) => {
                    evaluated(a, &env).map(|value| (a.name.clone(), value, a.dtype.clone()))
                }
                Statement::Affine(a) if a.args.iter().all(|arg| env.contains_key(arg)) => {
                    let mut sum = 0.0;
                    for (coefficient, arg) in a.terms() {
                        sum += coefficient * env[arg];
                    }
                    Some((a.name.clone(), a.offset.0 + sum, a.dtype.clone()))
                }
                _ => None,
            };
            if let Some((name, value, dtype)) = folded {
                stmt = let_literal(&name, value, dtype);
            }
            for name in stmt.defines() {
                env.remove(&name);
            }
            if let Some((name, value)) = literal_binding(&stmt) {
                env.insert(name.clone(), value);
            }
            out.push(stmt);
        }
        body.rebuilt(out)
    }
    scope(body, &HashMap::new())
}

/// Every name read anywhere in `body`, counted: SSA reads and expression variables, less a statement's own definitions.
pub(crate) fn read_counts(body: &Body, exposed: &HashSet<Name>) -> HashMap<Name, usize> {
    let mut counts: HashMap<Name, usize> = exposed.iter().map(|name| (name.clone(), 1)).collect();
    body.walk(&mut |here| {
        let stmt = here.stmt();
        let own: HashSet<Name> = stmt.defines().into_iter().collect();
        for name in stmt.deps() {
            if !own.contains(&name) {
                *counts.entry(name).or_default() += 1;
            }
        }
        for expression in stmt.expressions() {
            for name in expression.free_vars() {
                if !own.contains(&name) {
                    *counts.entry(name).or_default() += 1;
                }
            }
        }
    });
    counts
}

pub(crate) fn is_droppable_pure(stmt: &Statement) -> bool {
    match stmt {
        Statement::Load(l) => l.carried.is_empty(),
        Statement::Accum(_)
        | Statement::Affine(_)
        | Statement::Assign(_)
        | Statement::Init(_)
        | Statement::Let(_)
        | Statement::Select(_) => true,
        _ => false,
    }
}

/// Drop every pure statement whose values nothing reads, and a loop or branch left with no statement. A body that
/// stores nothing is a fragment whose values are all its reader's, so nothing is dropped from it.
pub fn drop_unread_values(body: &Body) -> Body {
    if body.writes().is_empty() {
        return body.clone();
    }
    let mut stmts = body.clone();
    loop {
        let reads = read_counts(&stmts, &HashSet::new());
        let mut dropped = false;
        stmts = stmts.map_stmts(&mut |stmt| {
            let unread = is_droppable_pure(stmt)
                && !stmt
                    .defines()
                    .iter()
                    .any(|name| reads.get(name).is_some_and(|c| *c > 0));
            let nested = stmt.nested();
            if unread || (!nested.is_empty() && nested.iter().all(|b| b.is_empty())) {
                dropped = true;
                return None;
            }
            Some(stmt.clone())
        });
        if !dropped {
            return stmts;
        }
    }
}

pub(crate) const AFFINE_OPS: [&str; 5] = ["add", "subtract", "negative", "multiply", "divide"];

/// Fold every maximal single-reader cone of `add`, `subtract`, `negative`, literal `multiply`, literal `divide` and
/// `Affine` into one `Affine`.
pub fn fold_affine(body: &Body, exposed: &HashSet<Name>) -> Body {
    let counts = read_counts(body, exposed);

    fn dtype_of(stmt: &Statement, dtypes: &HashMap<Name, Name>) -> Name {
        if let Some(stamped) = stmt.dtype() {
            return stamped.0.clone();
        }
        let of = |args: &Names| -> Vec<&str> {
            args.iter()
                .map(|arg| dtypes.get(arg).map_or("f32", |d| &**d))
                .collect()
        };
        match stmt {
            Statement::Assign(a) => promote(&a.op.name, &of(&a.args)),
            Statement::Affine(a) => promote("add", &of(&a.args)),
            _ => "f32".into(),
        }
    }

    fn emitted(
        root: &Statement,
        name: &Name,
        terms: &[(Name, f64)],
        offset: f64,
    ) -> Arc<Statement> {
        let live: Vec<(Name, f64)> = terms.iter().filter(|(_, c)| *c != 0.0).cloned().collect();
        let name = name.clone();
        if live.is_empty() {
            return let_literal(&name, offset, root.dtype().cloned());
        }
        if live.len() == 1 && live[0].1 == 1.0 && offset == 0.0 {
            return assign(
                &name,
                "copy",
                vec![live[0].0.clone()],
                root.dtype().cloned(),
            );
        }
        Arc::new(Statement::Affine(Affine {
            name,
            args: live.iter().map(|(n, _)| n.clone()).collect(),
            coefficients: live.iter().map(|(_, c)| Float(*c)).collect(),
            offset: Float(offset),
            dtype: root.dtype().cloned(),
        }))
    }

    /// Equal up to the order of the terms, which the canonical order decides.
    fn same(a: &Statement, b: &Statement) -> bool {
        match (a, b) {
            (Statement::Affine(a), Statement::Affine(b)) => {
                let sorted = |x: &Affine| {
                    let mut terms: Vec<(Key, Name)> = x
                        .terms()
                        .map(|(c, arg)| (Key::float(c), arg.clone()))
                        .collect();
                    terms.sort();
                    terms
                };
                sorted(a) == sorted(b)
                    && a.offset == b.offset
                    && a.dtype == b.dtype
                    && a.name == b.name
            }
            _ => a == b,
        }
    }

    fn scope(
        body: &Body,
        dtypes: &HashMap<Name, Name>,
        literals: &HashMap<Name, f64>,
        counts: &HashMap<Name, usize>,
    ) -> Body {
        let stmts = body.stmts();
        let mut dtypes = dtypes.clone();
        let mut literals = literals.clone();
        // Each statement's arguments resolved to what bound them at that point: a statement, or the number a Let
        // binds. A body may rebind a name, so a cone is a set of positions, never of names.
        let mut current: HashMap<Name, usize> = HashMap::new();
        let mut binds: Vec<HashMap<Name, usize>> = Vec::new();
        let mut values: Vec<HashMap<Name, f64>> = Vec::new();
        for (index, stmt) in stmts.iter().enumerate() {
            let reads: Vec<Name> = match &**stmt {
                Statement::Assign(_) | Statement::Affine(_) => stmt.deps(),
                _ => vec![],
            };
            binds.push(
                reads
                    .iter()
                    .filter_map(|arg| current.get(arg).map(|i| (arg.clone(), *i)))
                    .collect(),
            );
            values.push(
                reads
                    .iter()
                    .filter_map(|arg| literals.get(arg).map(|v| (arg.clone(), *v)))
                    .collect(),
            );
            for name in stmt.defines() {
                current.insert(name.clone(), index);
                dtypes.insert(name.clone(), dtype_of(stmt, &dtypes));
                literals.remove(&name);
            }
            if let Some((name, number)) = literal_binding(stmt) {
                literals.insert(name.clone(), number);
            }
        }
        // The one name a cone member defines: an assign's or an affine's.
        let value_name = |index: usize| -> Option<&Name> {
            match &*stmts[index] {
                Statement::Assign(a) => Some(&a.name),
                Statement::Affine(a) => Some(&a.name),
                _ => None,
            }
        };
        let affine = |index: usize| -> bool {
            match &*stmts[index] {
                Statement::Affine(_) => true,
                Statement::Assign(a) => {
                    let op = &*a.op.name;
                    if !AFFINE_OPS.contains(&op)
                        || a.args.len() != if op == "negative" { 1 } else { 2 }
                    {
                        return false;
                    }
                    if op == "divide" {
                        return values[index].get(&a.args[1]).is_some_and(|v| *v != 0.0);
                    }
                    op != "multiply" || a.args.iter().any(|arg| values[index].contains_key(arg))
                }
                _ => false,
            }
        };
        let cone: HashSet<usize> = (0..stmts.len())
            .filter(|&i| {
                affine(i)
                    && value_name(i)
                        .is_some_and(|name| dtypes.get(name).is_some_and(|d| is_float(d)))
            })
            .collect();
        let mut read_by: HashMap<usize, usize> = HashMap::new();
        let mut reader: HashMap<usize, usize> = HashMap::new();
        for &index in &cone {
            for &bound in binds[index].values() {
                *read_by.entry(bound).or_default() += 1;
                reader.insert(bound, index);
            }
        }
        let absorbed: HashSet<usize> = cone
            .iter()
            .copied()
            .filter(|&index| {
                let (Some(name), Some(consumer)) = (
                    value_name(index),
                    reader.get(&index).and_then(|r| value_name(*r)),
                ) else {
                    return false;
                };
                counts.get(name).copied().unwrap_or(0) == 1
                    && read_by.get(&index).copied().unwrap_or(0) == 1
                    && dtypes.get(name) == dtypes.get(consumer)
            })
            .collect();

        struct Fold<'a> {
            stmts: &'a [Arc<Statement>],
            binds: &'a [HashMap<Name, usize>],
            values: &'a [HashMap<Name, f64>],
            absorbed: &'a HashSet<usize>,
        }
        impl Fold<'_> {
            fn collect(
                &self,
                index: usize,
                arg: &Name,
                scale: f64,
                terms: &mut Vec<(Name, f64)>,
                offset: &mut f64,
            ) {
                if let Some(value) = self.values[index].get(arg) {
                    *offset += scale * value;
                } else if let Some(bound) = self.binds[index]
                    .get(arg)
                    .filter(|b| self.absorbed.contains(b))
                {
                    self.expand(*bound, scale, terms, offset);
                } else {
                    match terms.iter_mut().find(|(name, _)| name == arg) {
                        Some(term) => term.1 += scale,
                        None => terms.push((arg.clone(), scale)),
                    }
                }
            }
            fn expand(
                &self,
                index: usize,
                scale: f64,
                terms: &mut Vec<(Name, f64)>,
                offset: &mut f64,
            ) {
                match &*self.stmts[index] {
                    Statement::Affine(a) => {
                        *offset += scale * a.offset.0;
                        for (coefficient, arg) in a.terms() {
                            self.collect(index, arg, scale * coefficient, terms, offset);
                        }
                    }
                    Statement::Assign(a) => match (&*a.op.name, &a.args[..]) {
                        ("negative", [x]) => self.collect(index, x, -scale, terms, offset),
                        ("divide", [x, d]) => {
                            self.collect(index, x, scale / self.values[index][d], terms, offset)
                        }
                        ("multiply", [left, right]) => {
                            if let Some(value) = self.values[index].get(left) {
                                self.collect(index, right, scale * value, terms, offset);
                            } else {
                                self.collect(
                                    index,
                                    left,
                                    scale * self.values[index][right],
                                    terms,
                                    offset,
                                );
                            }
                        }
                        (op, [a0, a1]) => {
                            self.collect(index, a0, scale, terms, offset);
                            self.collect(
                                index,
                                a1,
                                if op == "subtract" { -scale } else { scale },
                                terms,
                                offset,
                            );
                        }
                        _ => {} // a cone member is a unary negation or a binary operation
                    },
                    _ => {} // only assigns and affines are in a cone
                }
            }
            fn members(&self, root: usize) -> HashSet<usize> {
                let mut found = HashSet::new();
                let mut pending: Vec<usize> = self.binds[root]
                    .values()
                    .copied()
                    .filter(|b| self.absorbed.contains(b))
                    .collect();
                while let Some(index) = pending.pop() {
                    if found.insert(index) {
                        pending.extend(
                            self.binds[index]
                                .values()
                                .copied()
                                .filter(|b| self.absorbed.contains(b)),
                        );
                    }
                }
                found
            }
        }
        let fold = Fold {
            stmts,
            binds: &binds,
            values: &values,
            absorbed: &absorbed,
        };
        let mut rewritten: HashMap<usize, Arc<Statement>> = HashMap::new();
        let mut dropped: HashSet<usize> = HashSet::new();
        let mut roots: Vec<usize> = cone.difference(&absorbed).copied().collect();
        roots.sort_unstable();
        for root in roots {
            let Some(name) = value_name(root) else {
                continue;
            };
            let mut terms: Vec<(Name, f64)> = Vec::new();
            let mut offset = 0.0;
            fold.expand(root, 1.0, &mut terms, &mut offset);
            let folded = emitted(&stmts[root], name, &terms, offset);
            if !same(&folded, &stmts[root]) {
                rewritten.insert(root, folded);
                dropped.extend(fold.members(root));
            }
        }
        let mut out = Vec::new();
        for (index, stmt) in stmts.iter().enumerate() {
            let mut stmt = stmt.clone();
            let nested = stmt.nested();
            if !nested.is_empty() {
                stmt = with_bodies(
                    &stmt,
                    nested
                        .iter()
                        .map(|child| scope(child, &dtypes, &literals, counts))
                        .collect(),
                );
            }
            if !dropped.contains(&index) {
                out.push(rewritten.get(&index).cloned().unwrap_or(stmt));
            }
        }
        body.rebuilt(out)
    }

    let folded = scope(body, &HashMap::new(), &HashMap::new(), &counts);
    let reads = read_counts(&folded, exposed);
    folded.map_stmts(&mut |stmt| match literal_binding(stmt) {
        Some((name, _)) if reads.get(name).copied().unwrap_or(0) == 0 => None,
        _ => Some(stmt.clone()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize::testing::*;

    #[test]
    fn unread_values_leave_with_everything_only_they_read() {
        // A dropped value leaves its operands unread in turn: one call drops the whole chain.
        let chain = ["y = load Y[i]", "a = exp(y)", "b = exp(a)", "c = exp(b)"];
        let kept = ["x = load X[i]", "O[i] = x"];
        let looped =
            |parts: &[&str]| body(&format!("for i in 0..8:\n{}", indented(&parts.join("\n"))));
        assert_eq!(
            drop_unread_values(&looped(&[chain.as_slice(), kept.as_slice()].concat())),
            looped(&kept)
        );
    }
}
