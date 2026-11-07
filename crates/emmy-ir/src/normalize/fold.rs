//! The folds: a select whose branches bind one value, an operation over literals, a value nothing reads, and a sum
//! into one `Affine` statement.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::{assign, with_bodies};
use crate::Name;
use crate::body::Body;
use crate::dtype::promote;
use crate::elementwise::{AffineRole, Elementwise, Op};
use crate::expr::Expression;
use crate::field::{DType, Float, Names};
use crate::numbering::Key;
use crate::protocol::literal_binding;
use crate::statement::*;

// --- folds ----------------------------------------------------------------------------------------------

/// A `Select` whose branches all bind one value is a copy of it.
pub fn fold_trivial_selects(body: &Body) -> Body {
    body.map_stmts(&mut |stmt| {
        if let Statement::Select(s) = &**stmt
            && s.branches.iter().all(|b| b.value == s.branches[0].value)
        {
            return Some(assign(
                &s.name,
                Op::Known(Elementwise::Copy),
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
        dtype: Some(dtype.unwrap_or(DType::F32)),
    }))
}

/// Evaluate every operation whose operands are all literals into the literal it computes.
pub fn fold_constants(body: &Body) -> Body {
    fn evaluated(stmt: &Assign, env: &HashMap<Name, f64>) -> Option<f64> {
        if stmt.args.is_empty()
            || stmt.args.iter().any(|arg| !env.contains_key(arg))
            || stmt.dtype.is_some_and(|d| !d.is_float())
        {
            return None;
        }
        let values: Vec<f64> = stmt.args.iter().map(|arg| env[arg]).collect();
        if stmt.op.copies() {
            return Some(values[0]);
        }
        let args: Vec<f32> = values.iter().map(|v| *v as f32).collect();
        stmt.op.known()?.evaluate_f32(&args).map(|v| v as f64)
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
                    evaluated(a, &env).map(|value| (a.name.clone(), value, a.dtype))
                }
                Statement::Affine(a) if a.args.iter().all(|arg| env.contains_key(arg)) => {
                    let mut sum = 0.0;
                    for (coefficient, arg) in a.terms() {
                        sum += coefficient * env[arg];
                    }
                    Some((a.name.clone(), a.offset.0 + sum, a.dtype))
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

/// Fold every maximal single-reader cone of `add`, `subtract`, `negative`, literal `multiply`, literal `divide` and
/// `Affine` into one `Affine`.
pub fn fold_affine(body: &Body, exposed: &HashSet<Name>) -> Body {
    let counts = read_counts(body, exposed);

    fn dtype_of(stmt: &Statement, dtypes: &HashMap<Name, DType>) -> DType {
        if let Some(stamped) = stmt.dtype() {
            return *stamped;
        }
        let of = |args: &Names| -> Vec<DType> {
            args.iter()
                .map(|arg| dtypes.get(arg).copied().unwrap_or(DType::F32))
                .collect()
        };
        match stmt {
            Statement::Assign(a) => promote(a.op.known(), &of(&a.args)),
            Statement::Affine(a) => promote(Some(Elementwise::Add), &of(&a.args)),
            _ => DType::F32,
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
                Op::Known(Elementwise::Copy),
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
        dtypes: &HashMap<Name, DType>,
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
                    let Some(role) = a.op.affine() else {
                        return false;
                    };
                    if a.args.len() != role.arity() {
                        return false;
                    }
                    match role {
                        AffineRole::Quotient => {
                            values[index].get(&a.args[1]).is_some_and(|v| *v != 0.0)
                        }
                        AffineRole::Scale => {
                            a.args.iter().any(|arg| values[index].contains_key(arg))
                        }
                        _ => true,
                    }
                }
                _ => false,
            }
        };
        let cone: HashSet<usize> = (0..stmts.len())
            .filter(|&i| {
                affine(i)
                    && value_name(i)
                        .is_some_and(|name| dtypes.get(name).is_some_and(|d| d.is_float()))
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
                    Statement::Assign(a) => match (a.op.affine(), &a.args[..]) {
                        (Some(AffineRole::Negation), [x]) => {
                            self.collect(index, x, -scale, terms, offset)
                        }
                        (Some(AffineRole::Quotient), [x, d]) => {
                            self.collect(index, x, scale / self.values[index][d], terms, offset)
                        }
                        (Some(AffineRole::Scale), [left, right]) => {
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
                        (role, [a0, a1]) => {
                            self.collect(index, a0, scale, terms, offset);
                            self.collect(
                                index,
                                a1,
                                if role == Some(AffineRole::Difference) {
                                    -scale
                                } else {
                                    scale
                                },
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
    use super::drop_unread_values;
    use crate::identity::canonicalize_identity;
    use crate::normalize::normalize_body;
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

    /// The normal form of a loop that reads `v` from `x` and `s` from `y`, then runs `lines`.
    fn normal_of(lines: &[&str]) -> String {
        let mut spelled = vec![
            "for i in 0..8:".to_string(),
            "    v = load x[i]".to_string(),
            "    s = load y[i]".to_string(),
        ];
        spelled.extend(lines.iter().map(|line| format!("    {line}")));
        text(&normalize_body(&body(&spelled.join("\n"))).unwrap())
    }

    #[test]
    fn one_key_softmax_weight_folds_to_nothing() {
        // `v * (exp(s - s) / exp(s - s))` is `v`: the difference is zero, its exponential one, the quotient one.
        assert_eq!(
            normal_of(&[
                "z = subtract(s, s)",
                "u = exp(z)",
                "w = divide(u, u)",
                "o = multiply(v, w)",
                "O[i] = o"
            ]),
            "for a0 in 0..8:\n    in0 = load x[a0]\n    O[a0] = in0"
        );
    }

    #[test]
    fn a_constant_read_by_a_write_keeps_its_statement() {
        assert_eq!(
            normal_of(&["z = subtract(s, s)", "u = exp(z)", "O[i] = u"]),
            "let f32 v0 = 1.0\nfor a0 in 0..8:\n    O[a0] = v0"
        );
    }

    #[test]
    fn a_typed_constant_is_not_read_through() {
        // A stamped dtype is a rounding the program performed: the literal keeps it, and nothing computes past it.
        assert_eq!(
            normal_of(&[
                "z = subtract(s, s)",
                "u = exp(z)",
                "f16 h = copy(u)",
                "o = multiply(v, h)",
                "O[i] = o"
            ]),
            "let f16 v0 = 1.0\nfor a0 in 0..8:\n    in0 = load x[a0]\n    v1 = multiply(in0, v0)\n    O[a0] = v1"
        );
    }

    #[test]
    fn a_score_nothing_reads_takes_its_dot_product_with_it() {
        // A one-key softmax weight is `1` whatever the score, so the scaled score, the dot product that fed it and
        // the loop that summed it reach no output: normalization drops them all, and keeps the stores.
        let normal = normalize_body(&body(
            "
            for i in 0..8:
                for k in 0..4:
                    q = load Q[i, k]
                    kk = load K[i, k]
                    qk = multiply(q, kk)
                    acc <- add(acc, qk) over k
                let f32 scale = 0.125
                s = multiply(acc, scale)
                z = subtract(s, s)
                w = exp(z)
                v = load V[i]
                o = multiply(v, w)
                O[i] = o
            ",
        ))
        .unwrap();
        assert_eq!(
            text(&normal),
            "for a0 in 0..8:\n    in0 = load V[a0]\n    O[a0] = in0"
        );
    }

    /// A loop that reads `x` from `X`, runs `lines` and stores `o`.
    fn affine_body(lines: &[&str]) -> crate::body::Body {
        let mut spelled = vec![
            "for i in 0..8:".to_string(),
            "    x = load X[i]".to_string(),
        ];
        spelled.extend(lines.iter().map(|line| format!("    {line}")));
        spelled.push("    O[i] = o".to_string());
        body(&spelled.join("\n"))
    }

    fn affine_normal(lines: &[&str]) -> String {
        text(&normalize_body(&affine_body(lines)).unwrap())
    }

    /// The normal form of `O[i] = coefficient * X[i]`.
    fn scaled(coefficient: &str) -> String {
        format!("for a0 in 0..8:\n    in0 = load X[a0]\n    v0 = {coefficient}*in0\n    O[a0] = v0")
    }

    const COPY: &str = "for a0 in 0..8:\n    in0 = load X[a0]\n    O[a0] = in0";

    #[test]
    fn every_spelling_of_a_scale_is_one_body() {
        let twice = affine_body(&["o = add(x, x)"]);
        let scaled_by_two = affine_body(&["let f32 c = 2.0", "o = multiply(x, c)"]);
        let halved = affine_body(&["let f32 c = 0.5", "o = divide(x, c)"]);
        // One affine statement, and the literal the fold took is dropped.
        for spelling in [&twice, &scaled_by_two, &halved] {
            assert_eq!(text(&normalize_body(spelling).unwrap()), scaled("2"));
        }
        let key = |body: &crate::body::Body| canonicalize_identity(body, false, None).unwrap().key;
        assert_eq!(key(&twice), key(&scaled_by_two));
        assert_eq!(key(&twice), key(&halved));
        assert_ne!(
            key(&twice),
            key(&affine_body(&["let f32 c = 3.0", "o = multiply(x, c)"]))
        );
    }

    #[test]
    fn a_division_by_a_literal_is_the_product_by_its_reciprocal() {
        assert_eq!(
            affine_normal(&["let f32 c = 4.0", "o = divide(x, c)"]),
            scaled("0.25")
        );
        assert_eq!(
            affine_normal(&["let f32 c = 0.25", "o = multiply(x, c)"]),
            scaled("0.25")
        );
        assert_eq!(
            affine_normal(&["let f32 c = 1.0", "o = divide(x, c)"]),
            COPY,
            "a division by one is a copy"
        );
    }

    #[test]
    fn like_terms_sum_and_cancel() {
        assert_eq!(
            affine_normal(&[
                "let f32 c = 3.0",
                "t = multiply(x, c)",
                "s = add(t, x)",
                "o = subtract(s, x)"
            ]),
            scaled("3")
        );
        // Every term cancels: the value is the constant zero, which the store reads, and nothing loads `X`.
        assert_eq!(
            affine_normal(&["o = subtract(x, x)"]),
            "let f32 v0 = 0.0\nfor a0 in 0..8:\n    O[a0] = v0"
        );
    }

    #[test]
    fn a_select_whose_branches_bind_one_value_is_that_value() {
        assert_eq!(
            affine_normal(&["o = select:", "    x when i < i", "    x when i < i"]),
            COPY
        );
    }
}
