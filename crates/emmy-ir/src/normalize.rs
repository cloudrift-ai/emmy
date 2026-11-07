//! Body normalization: the canonical form every Loop IR body takes at construction and inside every identity.
//! Pure `Body → Body` steps and the one driver, `normalize_body`, memoized by the body's content. Normalization
//! answers to no knob, pin or evidence; a transform something decides is a pipeline pass.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};

use crate::axis::{Axis, Dim};
use crate::body::{Body, exposed_defines, free_names, free_ssa, ordered_sibling_defs};
use crate::dtype::{is_float, promote};
use crate::elementwise::{Op, evaluate_f32};
use crate::expr::{BinOp, Expr, Expression, Interval, Lit, Num, SimplifyCtx, affine_form};
use crate::field::{AxisRef, DType, Float, Names, NumOr};
use crate::kinds::*;
use crate::order::{is_protocol, ordering_constraints, topological_sort};
use crate::protocol::{literal_binding, unique};
use crate::text::{structural_body_text, structural_text};
use crate::values::{Key, Number, Numbering, distinct, ptr, roles, scope_tree, value_numbers};
use crate::{Error, Name, Result, locked};

const SYMBOLIC_AXIS_HI: i128 = 1 << 30;

/// Push an iteration axis `[0, extent)` into a simplification context so index expressions over it fold.
pub fn extend_simplify_ctx(ctx: &SimplifyCtx, axis: &Axis) -> SimplifyCtx {
    match axis.extent.as_static() {
        Some(n) => ctx.extend(
            &axis.name,
            Interval::new(0, n as i128 - 1),
            Some(Expression::int(n)),
        ),
        None => ctx.extend(
            &axis.name,
            Interval::new(0, SYMBOLIC_AXIS_HI),
            Some(axis.extent.expression.clone()),
        ),
    }
}

// --- the memo -------------------------------------------------------------------------------------------

/// A bounded memo keyed by a body's content: the normal forms of the bodies met in this process.
struct Memo {
    maxsize: usize,
    store: HashMap<u64, Vec<(Body, Body)>>,
    order: std::collections::VecDeque<(u64, Body)>,
}

impl Memo {
    fn new(maxsize: usize) -> Memo {
        Memo {
            maxsize,
            store: HashMap::new(),
            order: std::collections::VecDeque::new(),
        }
    }

    fn get(&self, body: &Body) -> Option<Body> {
        self.store
            .get(&body.content_hash())?
            .iter()
            .find(|(key, _)| key == body)
            .map(|(_, value)| value.clone())
    }

    fn put(&mut self, body: &Body, value: Body) {
        self.store
            .entry(body.content_hash())
            .or_default()
            .push((body.clone(), value));
        self.order.push_back((body.content_hash(), body.clone()));
        while self.order.len() > self.maxsize {
            let Some((hash, oldest)) = self.order.pop_front() else {
                break;
            };
            if let Some(entries) = self.store.get_mut(&hash) {
                entries.retain(|(key, _)| !key.ptr_eq(&oldest));
                if entries.is_empty() {
                    self.store.remove(&hash);
                }
            }
        }
    }

    fn clear(&mut self) {
        self.store.clear();
        self.order.clear();
    }
}

fn normal_forms() -> &'static Mutex<Memo> {
    static MEMO: OnceLock<Mutex<Memo>> = OnceLock::new();
    MEMO.get_or_init(|| Mutex::new(Memo::new(16384)))
}

/// Forget every memoized normal form (the tests of the memo use it).
pub fn clear_memo() {
    locked(normal_forms()).clear();
}

/// The executable normal form of `body`, cached on the body and its fixed point, and memoized by content.
pub fn normalize_body(body: &Body) -> Result<Body> {
    if let Some(normal) = body.0.normalized.get() {
        return Ok(normal.clone());
    }
    let found = locked(normal_forms()).get(body);
    let normal = match found {
        Some(found) => found,
        None => {
            let computed = normalize_uncached(body)?;
            locked(normal_forms()).put(body, computed.clone());
            computed
        }
    };
    let _ = normal.0.normalized.set(normal.clone());
    let _ = body.0.normalized.set(normal.clone());
    Ok(normal)
}

/// The normal form, computed afresh.
pub fn normalize_uncached(body: &Body) -> Result<Body> {
    // The normal form holds every operation: a call is storage sharing inside fusion, which expands it.
    if !body.of_kinds(&["Call"]).is_empty() {
        return Err(Error::Value(
            "a body that holds a subroutine call has no normal form; fusion expands its calls"
                .into(),
        ));
    }
    // Motion can put two lexical bindings in one scope. Give them distinct names first.
    let mut stmts = rename_ssa_sequential(&topological_sort(body));
    // The folds feed the steps ahead of them, so the steps run again after a fold that changed the body. The order
    // permutes and names, and one step reads names: a body the order left an unread value in goes through the steps
    // again; any other is final, since the order is final on its own output.
    for _ in 0..8 {
        let prepared = prepare_body(&stmts)?;
        stmts = fold_affine(&fold_trivial_selects(&prepared), &HashSet::new());
        if stmts != prepared {
            continue;
        }
        let ordered = canonical_order(&prepared)?;
        stmts = drop_unread_values(&ordered);
        if stmts == ordered {
            return Ok(ordered);
        }
    }
    let mut lines = Vec::new();
    crate::text::body_lines(&stmts, "  ", &mut lines)?;
    Err(Error::Value(format!(
        "the normal form did not settle in eight rounds:\n{}",
        lines.join("\n")
    )))
}

/// The exact simplifications, then every value placed once; the three run until the first two leave a placed body
/// alone.
pub fn prepare_body(body: &Body) -> Result<Body> {
    let mut stmts = topological_sort(body);
    stmts = drop_size_one_free_axes(&stmts);
    stmts = drop_size_one_reduce_axes(&stmts);
    stmts = canonicalize_free_axis_order(&stmts)?;
    stmts = fold_trivial_selects(&stmts);
    stmts = fold_constants(&stmts);
    stmts = drop_unread_values(&stmts);
    let mut placed: Option<Body> = None;
    loop {
        let simplified =
            canonicalize_expressions(&simplify_body(&merge_sibling_reduce_loops(&stmts))?, &[])?;
        if placed.as_ref() == Some(&simplified) {
            return Ok(simplified);
        }
        let next = place_values(&simplified)?;
        stmts = next.clone();
        placed = Some(next);
    }
}

fn with_bodies(stmt: &Arc<Statement>, bodies: Vec<Body>) -> Arc<Statement> {
    Statement::rebuild_bodies(stmt, bodies)
}

fn substituted(stmt: &Arc<Statement>, name: &Name, value: Expression) -> Arc<Statement> {
    Statement::substitute(stmt, &HashMap::from([(name.clone(), value)]))
}

fn assign(name: &Name, op: &str, args: Vec<Name>, dtype: Option<DType>) -> Arc<Statement> {
    Arc::new(Statement::Assign(Assign {
        name: name.clone(),
        op: Op::new(op),
        args: args.into(),
        dtype,
    }))
}

// --- size-one axes ----------------------------------------------------------------------------------------

/// Inline every free loop of extent one, its axis a literal zero.
pub fn drop_size_one_free_axes(body: &Body) -> Body {
    body.map(&mut |stmt| {
        if let Statement::Loop(l) = &**stmt
            && l.axis.extent.as_static() == Some(1)
            && !l.is_reduce()
        {
            return Some(
                l.body
                    .stmts()
                    .iter()
                    .map(|c| substituted(c, &l.axis.name, Expression::int(0)))
                    .collect(),
            );
        }
        Some(vec![stmt.clone()])
    })
}

/// Inline a canonical extent-one reduction as its single update.
pub fn drop_size_one_reduce_axes(body: &Body) -> Body {
    body.map(&mut |stmt| {
        let Statement::Loop(l) = &**stmt else {
            return Some(vec![stmt.clone()]);
        };
        if !(l.is_reduce() && l.axis.extent.as_static() == Some(1)) {
            return Some(vec![stmt.clone()]);
        }
        let accums: Vec<&Accum> = l
            .body
            .stmts()
            .iter()
            .filter_map(|m| match &**m {
                Statement::Accum(a) => Some(a),
                _ => None,
            })
            .collect();
        let names: HashSet<&Name> = accums.iter().map(|a| &a.name).collect();
        if accums.is_empty()
            || names.len() != accums.len()
            || l.body
                .stmts()
                .iter()
                .any(|m| !(m.is_pure() || matches!(**m, Statement::Accum(_))))
        {
            return Some(vec![stmt.clone()]);
        }
        let mut out = Vec::new();
        for member in l.body.stmts() {
            let member = substituted(member, &l.axis.name, Expression::int(0));
            match &*member {
                Statement::Accum(a) => match &a.base {
                    None => {
                        if a.op.identity.is_none() {
                            return Some(vec![stmt.clone()]);
                        }
                        out.push(assign(
                            &a.name,
                            "copy",
                            vec![a.value.clone()],
                            a.dtype.clone(),
                        ));
                    }
                    Some(base) => out.push(assign(
                        &a.name,
                        &a.op.name,
                        vec![base.clone(), a.value.clone()],
                        a.dtype.clone(),
                    )),
                },
                _ => out.push(member),
            }
        }
        Some(out)
    })
}

// --- the free-axis order ----------------------------------------------------------------------------------

/// The innermost row-major output-coordinate depth carrying one free axis.
fn output_storage_depth(stmts: &Body, axis: &Name) -> Option<usize> {
    let mut depths: Vec<usize> = Vec::new();
    for write in stmts.writes() {
        let Statement::Write(w) = &**write.stmt() else {
            continue;
        };
        let mut positions = Vec::new();
        for (position, expression) in w.index.iter().enumerate() {
            if !expression.reads(axis) {
                continue;
            }
            if let Some((_, coeffs)) = affine_form(expression, &|name| name == &**axis)
                && coeffs.get(axis).copied().unwrap_or(0) != 1
            {
                return None;
            }
            positions.push(position);
        }
        if let Some(last) = positions.last() {
            depths.push(w.index.len() - last - 1);
        }
    }
    match depths.first() {
        Some(first) if depths.iter().all(|d| d == first) => Some(*first),
        _ => None,
    }
}

fn loop_over(axis: AxisRef, body: Body, unroll: bool, seed: bool) -> Body {
    Body::new(vec![Arc::new(Statement::Loop(Loop {
        axis,
        body,
        unroll,
        seed,
    }))])
}

/// Permute distinct items once modulo transpositions proven to preserve the whole form.
fn orders_modulo_transpositions<T: Clone>(
    items: &[T],
    interchangeable: &dyn Fn(&T, &T) -> bool,
) -> Vec<Vec<T>> {
    let mut classes: Vec<Vec<T>> = Vec::new();
    for item in items {
        match classes
            .iter_mut()
            .find(|group| interchangeable(item, &group[0]))
        {
            Some(group) => group.push(item.clone()),
            None => classes.push(vec![item.clone()]),
        }
    }
    let mut out = Vec::new();
    fn walk<T: Clone>(
        classes: &[Vec<T>],
        remaining: &mut Vec<usize>,
        positions: &mut Vec<usize>,
        out: &mut Vec<Vec<T>>,
    ) {
        if remaining.iter().all(|count| *count == 0) {
            let mut offsets = vec![0; classes.len()];
            let mut order = Vec::new();
            for &class in positions.iter() {
                order.push(classes[class][offsets[class]].clone());
                offsets[class] += 1;
            }
            out.push(order);
            return;
        }
        for class in 0..classes.len() {
            if remaining[class] == 0 {
                continue;
            }
            remaining[class] -= 1;
            positions.push(class);
            walk(classes, remaining, positions, out);
            positions.pop();
            remaining[class] += 1;
        }
    }
    let mut remaining: Vec<usize> = classes.iter().map(|group| group.len()).collect();
    walk(&classes, &mut remaining, &mut Vec::new(), &mut out);
    out
}

fn rename_axes(body: &Body, mapping: &HashMap<Name, Name>) -> Body {
    let sigma: HashMap<Name, Expression> = mapping
        .iter()
        .map(|(name, replacement)| {
            (
                name.clone(),
                Expression::new(Expr::Var(replacement.clone())),
            )
        })
        .collect();
    body.map_each(&mut |stmt| Statement::substitute(stmt, &sigma))
}

fn text_key(body: &Body) -> Key {
    Key::str(&structural_body_text(body))
}

/// Sort an outer free-loop chain by row-major output storage order; where the writes do not order it, by the least
/// renamed structural form.
pub fn canonicalize_free_axis_order(body: &Body) -> Result<Body> {
    let mut chain: Vec<Loop> = Vec::new();
    let mut current = body.clone();
    while current.len() == 1
        && let Statement::Loop(l) = &*current.stmts()[0]
    {
        if l.is_reduce() {
            break;
        }
        chain.push(l.clone());
        current = l.body.clone();
    }
    let terminal: Vec<Arc<Statement>> = current
        .stmts()
        .iter()
        .map(|stmt| {
            let nested = stmt.nested();
            if nested.is_empty() {
                return Ok(stmt.clone());
            }
            Ok(with_bodies(
                stmt,
                nested
                    .iter()
                    .map(canonicalize_free_axis_order)
                    .collect::<Result<Vec<_>>>()?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let terminal = Body::new(terminal);
    let depths: Vec<Option<usize>> = chain
        .iter()
        .map(|l| output_storage_depth(&terminal, &l.axis.name))
        .collect();
    let distinct_depths: HashSet<Option<usize>> = depths.iter().copied().collect();
    let sorted: Vec<Loop> = if depths.iter().all(|d| d.is_some())
        && distinct_depths.len() == depths.len()
    {
        let mut pairs: Vec<(usize, Loop)> = depths
            .iter()
            .flatten()
            .copied()
            .zip(chain.iter().cloned())
            .collect();
        pairs.sort_by_key(|(depth, _)| std::cmp::Reverse(*depth));
        pairs.into_iter().map(|(_, l)| l).collect()
    } else {
        let axis_metadata = |l: &Loop| -> Key {
            let mut mapping: HashMap<Name, Name> =
                HashMap::from([(l.axis.name.clone(), Name::from("__axis__"))]);
            for (depth, source) in l.axis.sources().iter().enumerate() {
                mapping
                    .entry(source.name.clone())
                    .or_insert_with(|| format!("__parent{depth}__").into());
            }
            let renamed = Statement::rename(&Arc::new(Statement::Loop(l.clone())), &mapping);
            let Statement::Loop(renamed) = &*renamed else {
                return Key::None;
            };
            Key::tuple([
                Key::str(&crate::text::spell_axis(&renamed.axis, None).unwrap_or_default()),
                Key::Bool(renamed.unroll),
                Key::Bool(renamed.seed),
            ])
        };
        let mut source_counts: HashMap<Name, usize> = HashMap::new();
        for l in &chain {
            if let Some(source) = l.axis.source_axis() {
                *source_counts.entry(source.name.clone()).or_default() += 1;
            }
        }
        let mut role_of: HashMap<Name, Key> = HashMap::new();
        for (focus, depth) in chain.iter().zip(&depths) {
            let mapping: HashMap<Name, Name> = chain
                .iter()
                .map(|l| {
                    (
                        l.axis.name.clone(),
                        Name::from(if l.axis.name == focus.axis.name {
                            "__self__"
                        } else {
                            "__other__"
                        }),
                    )
                })
                .collect();
            // The current nest orders affine sums too; compare roles independently of that order to avoid cycles.
            let focused = canonicalize_expressions(
                &rename_axes(&terminal, &mapping),
                &[Name::from("__self__"), Name::from("__other__")],
            )?;
            let source_arity = focus
                .axis
                .source_axis()
                .map_or(1, |source| source_counts[&source.name]);
            // A coordinate carried by a consistent output position has a fixed row-major role; an unresolved one
            // stays outside the known output suffix, then its structural role breaks ties.
            let storage_order = match depth {
                None => Key::tuple([Key::Int(0), Key::Int(0)]),
                Some(d) => Key::tuple([Key::Int(1), Key::Int(-(*d as i64))]),
            };
            let role = Key::tuple([
                storage_order,
                Key::tuple([
                    axis_metadata(focus),
                    Key::Int(source_arity as i64),
                    text_key(&focused),
                ]),
            ]);
            role_of.insert(focus.axis.name.clone(), role);
        }
        let mut groups: BTreeMap<Key, Vec<Loop>> = BTreeMap::new();
        for l in &chain {
            groups
                .entry(role_of[&l.axis.name].clone())
                .or_default()
                .push(l.clone());
        }
        let interchangeable = |left: &Loop, right: &Loop| -> bool {
            if axis_metadata(left) != axis_metadata(right) {
                return false;
            }
            if let (Some(ls), Some(rs)) = (left.axis.source_axis(), right.axis.source_axis())
                && ls.name != rs.name
                && (source_counts[&ls.name] != 1 || source_counts[&rs.name] != 1)
            {
                return false;
            }
            let swapped = rename_axes(
                &terminal,
                &HashMap::from([
                    (left.axis.name.clone(), right.axis.name.clone()),
                    (right.axis.name.clone(), left.axis.name.clone()),
                ]),
            );
            swapped == terminal
        };
        let candidate = |order: &[Loop]| -> Key {
            let mut result = terminal.clone();
            for l in order.iter().rev() {
                result = loop_over(l.axis.clone(), result, l.unroll, l.seed);
            }
            text_key(&rename_ssa_sequential(&result))
        };
        let per_group: Vec<Vec<Vec<Loop>>> = groups
            .values()
            .map(|group| orders_modulo_transpositions(group, &interchangeable))
            .collect();
        let mut best: Option<(Key, Vec<Loop>)> = None;
        fn product(groups: &[Vec<Vec<Loop>>], chosen: &mut Vec<Loop>, f: &mut dyn FnMut(&[Loop])) {
            if groups.is_empty() {
                f(chosen);
                return;
            }
            for choice in &groups[0] {
                let len = chosen.len();
                chosen.extend(choice.iter().cloned());
                product(&groups[1..], chosen, f);
                chosen.truncate(len);
            }
        }
        product(&per_group, &mut Vec::new(), &mut |order| {
            let key = candidate(order);
            if best.as_ref().is_none_or(|(best_key, _)| key < *best_key) {
                best = Some((key, order.to_vec()));
            }
        });
        best.map_or_else(Vec::new, |(_, order)| order)
    };
    let mut result = terminal;
    for l in sorted.iter().rev() {
        result = loop_over(l.axis.clone(), result, l.unroll, l.seed);
    }
    Ok(result)
}

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

fn let_literal(name: &Name, value: f64, dtype: Option<DType>) -> Arc<Statement> {
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
        Body::new(out)
    }
    scope(body, &HashMap::new())
}

/// Every name read anywhere in `body`, counted: SSA reads and expression variables, less a statement's own definitions.
fn read_counts(body: &Body, exposed: &HashSet<Name>) -> HashMap<Name, usize> {
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

fn is_droppable_pure(stmt: &Statement) -> bool {
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

const AFFINE_OPS: [&str; 5] = ["add", "subtract", "negative", "multiply", "divide"];

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
        Body::new(out)
    }

    let folded = scope(body, &HashMap::new(), &HashMap::new(), &counts);
    let reads = read_counts(&folded, exposed);
    folded.map_stmts(&mut |stmt| match literal_binding(stmt) {
        Some((name, _)) if reads.get(name).copied().unwrap_or(0) == 0 => None,
        _ => Some(stmt.clone()),
    })
}

// --- sibling reduce loops ---------------------------------------------------------------------------------

const REDUCE_AXIS_PLACEHOLDER: &str = "__reduce_axis__";

/// The positions where a Load index within `body` walks the reduce axis, each keyed by how it walks the dimension.
fn reduce_axis_source_positions(body: &Body, axis: &Name) -> BTreeSet<Key> {
    let mut out = BTreeSet::new();
    for here in body.loads() {
        let Statement::Load(load) = &**here.stmt() else {
            continue;
        };
        for (dim, e) in load.index.iter().enumerate() {
            match affine_form(e, &|name| name == &**axis) {
                Some((anchor, coeffs)) => {
                    let coeff = coeffs.get(axis).copied().unwrap_or(0);
                    if coeff != 0 {
                        out.insert(Key::tuple([
                            Key::str("affine"),
                            Key::Str(load.input.clone()),
                            Key::Int(dim as i64),
                            Key::str(&anchor.text()),
                            Key::Int(coeff),
                        ]));
                    }
                }
                None if e.reads(axis) => {
                    let spelled = e
                        .substitute(&HashMap::from([(
                            axis.clone(),
                            Expression::var(REDUCE_AXIS_PLACEHOLDER),
                        )]))
                        .simplify(&SimplifyCtx::empty())
                        .unwrap_or_else(|_| e.clone());
                    out.insert(Key::tuple([
                        Key::str("composite"),
                        Key::Str(load.input.clone()),
                        Key::Int(dim as i64),
                        Key::str(&spelled.text()),
                    ]));
                }
                None => {}
            }
        }
    }
    out
}

/// Rename reduce-axis vars across sibling reduce loops whose Load positions overlap, so they share one axis name.
pub fn unify_siblings(body: &Body) -> Body {
    let mut stmts: Vec<Arc<Statement>> = body.stmts().to_vec();
    let mut entries: Vec<(usize, Name, Expression, BTreeSet<Key>)> = Vec::new();
    for (i, s) in stmts.iter().enumerate() {
        if let Statement::Loop(l) = &**s
            && l.is_reduce()
        {
            let positions = reduce_axis_source_positions(&l.body, &l.axis.name);
            if !positions.is_empty() {
                entries.push((
                    i,
                    l.axis.name.clone(),
                    l.axis.extent.expression.clone(),
                    positions,
                ));
            }
        }
    }
    if entries.len() < 2 {
        return body.clone();
    }
    let mut parent: Vec<usize> = (0..entries.len()).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for a in 0..entries.len() {
        for b in a + 1..entries.len() {
            if entries[a].2 != entries[b].2 {
                continue;
            }
            if !entries[a].3.is_disjoint(&entries[b].3) {
                let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                if ra != rb {
                    parent[ra.max(rb)] = ra.min(rb);
                }
            }
        }
    }
    let mut changed = false;
    for k in 0..entries.len() {
        let root = find(&mut parent, k);
        let canonical = entries[root].1.clone();
        let (index, axis_name, extent, _) = &entries[k];
        if canonical == *axis_name {
            continue;
        }
        let Statement::Loop(l) = &*stmts[*index] else {
            continue;
        };
        // A RENAME, binders and uses through one map, which is why it may pass through the very scopes it renames.
        let new_axis = Arc::new(Axis {
            name: canonical.clone(),
            extent: Dim {
                expression: extent.clone(),
                hint: l.axis.extent.hint,
            },
            window: l.axis.window.clone(),
        });
        let mapping = HashMap::from([(l.axis.name.clone(), canonical.clone())]);
        let renamed = l.body.map_each(&mut |s| Statement::rename(s, &mapping));
        stmts[*index] = Arc::new(Statement::Loop(Loop {
            axis: new_axis,
            body: renamed,
            unroll: l.unroll,
            seed: l.seed,
        }));
        changed = true;
    }
    if changed {
        Body::new(stmts)
    } else {
        body.clone()
    }
}

/// The names a loop over `body` still binds after it closes.
fn carried_out(body: &Body) -> HashSet<Name> {
    let mut out: HashSet<Name> = body
        .stmts()
        .iter()
        .filter_map(|s| match &**s {
            Statement::Accum(a) => Some(a.name.clone()),
            _ => None,
        })
        .collect();
    for stmt in body.stmts() {
        match &**stmt {
            Statement::Loop(l) if !l.seed => out.extend(carried_out(&l.body)),
            Statement::StridedLoop(l) if !l.seed => out.extend(carried_out(&l.body)),
            Statement::Cond(c) => {
                out.extend(carried_out(&c.body));
                out.extend(carried_out(&c.else_body));
            }
            _ => {}
        }
    }
    out
}

/// `body` with each name in `clashing` renamed to one neither side spells.
fn rename_apart(body: &Body, clashing: &HashSet<Name>, taken: &HashSet<Name>) -> Body {
    let mut used: HashSet<Name> = body.definitions().keys().cloned().collect();
    used.extend(body.ssa_uses().iter().cloned());
    used.extend(taken.iter().cloned());
    let mut mapping: HashMap<Name, Name> = HashMap::new();
    let mut sorted: Vec<&Name> = clashing.iter().collect();
    sorted.sort();
    for name in sorted {
        let fresh = (1..)
            .map(|n| Name::from(format!("{name}__m{n}")))
            .find(|candidate| !used.contains(candidate))
            .unwrap_or_else(|| name.clone());
        used.insert(fresh.clone());
        mapping.insert(name.clone(), fresh);
    }
    body.map_each(&mut |s| Statement::rename(s, &mapping))
}

fn merge_sibling_reduce_loops_here(body: &Body) -> Body {
    let items = body.stmts();
    if items.len() < 2 {
        return body.clone();
    }
    let mut out: Vec<Arc<Statement>> = Vec::new();
    let mut consumed: HashSet<usize> = HashSet::new();
    let mut changed = false;
    for (i, s) in items.iter().enumerate() {
        if consumed.contains(&i) {
            continue;
        }
        let Statement::Loop(first) = &**s else {
            out.push(s.clone());
            continue;
        };
        if !first.is_reduce() {
            out.push(s.clone());
            continue;
        }
        let mut merged = first.clone();
        for j in i + 1..items.len() {
            if consumed.contains(&j) {
                continue;
            }
            let Statement::Loop(t) = &*items[j] else {
                continue;
            };
            if !(t.is_reduce()
                && t.axis.name == merged.axis.name
                && t.axis.extent == merged.axis.extent
                && t.unroll == merged.unroll
                && t.seed == merged.seed)
            {
                continue;
            }
            // ONE reading of what the incoming loop needs from around it: the names it reads and does not bind.
            let reads: HashSet<Name> = free_names(&items[j]).into_iter().collect();
            let merged_defs: HashSet<Name> = merged.body.definitions().keys().cloned().collect();
            if !merged_defs.is_disjoint(&reads) {
                continue;
            }
            let incoming = t.body.clone();
            // What is left of the shared spellings is a COLLISION, not a dependence: rename the incoming copy apart,
            // unless the loop still binds the name after it closes.
            let clashing: HashSet<Name> = merged_defs
                .intersection(&incoming.definitions().keys().cloned().collect())
                .cloned()
                .collect();
            if !clashing.is_disjoint(&carried_out(&incoming)) {
                continue;
            }
            let mut between_defs: HashSet<Name> = HashSet::new();
            for (k, between) in items.iter().enumerate().take(j).skip(i + 1) {
                if !consumed.contains(&k) {
                    between_defs.extend(exposed_defines(between));
                }
            }
            if !between_defs.is_disjoint(&reads) {
                continue;
            }
            let renamed = rename_apart(&incoming, &clashing, &merged_defs);
            merged = Loop {
                axis: merged.axis.clone(),
                body: merged.body.concat(renamed.stmts()),
                unroll: merged.unroll,
                seed: merged.seed,
            };
            consumed.insert(j);
            changed = true;
        }
        out.push(if consumed.is_empty() && !changed {
            s.clone()
        } else {
            Arc::new(Statement::Loop(merged))
        });
    }
    if changed {
        Body::new(out)
    } else {
        body.clone()
    }
}

/// Merge sibling reduce loops with one axis into one loop whose body is the concatenation; parents before children.
pub fn merge_sibling_reduce_loops(body: &Body) -> Body {
    let merged = merge_sibling_reduce_loops_here(&unify_siblings(body));
    merged.map_each(&mut |s| {
        let nested = s.nested();
        if nested.is_empty() {
            return s.clone();
        }
        with_bodies(s, nested.iter().map(merge_sibling_reduce_loops).collect())
    })
}

// --- expression simplification ----------------------------------------------------------------------------

/// Simplify every expression inside a body, each under the value ranges of the loop axes around it.
pub fn simplify_body(body: &Body) -> Result<Body> {
    let ctx = SimplifyCtx::empty();
    let stmts: Result<Vec<Arc<Statement>>> = body
        .stmts()
        .iter()
        .map(|s| simplify_stmt(s, &ctx))
        .collect();
    Ok(Body::new(stmts?))
}

fn simplify_stmt(stmt: &Arc<Statement>, ctx: &SimplifyCtx) -> Result<Arc<Statement>> {
    let mut failure: Option<Error> = None;
    let stmt = Statement::map_expressions(stmt, &mut |e| match e.simplify(ctx) {
        Ok(simplified) => simplified,
        Err(err) => {
            failure = Some(err);
            e.clone()
        }
    });
    if let Some(err) = failure {
        return Err(err);
    }
    match &*stmt {
        Statement::Select(s) => Ok(Arc::new(Statement::Select(decided(s)))),
        Statement::Loop(l) => {
            let inner = extend_simplify_ctx(ctx, &l.axis);
            Ok(with_bodies(&stmt, vec![simplify_in(&l.body, &inner)?]))
        }
        Statement::StridedLoop(l) => {
            let inner = extend_simplify_ctx(ctx, &l.axis);
            Ok(with_bodies(&stmt, vec![simplify_in(&l.body, &inner)?]))
        }
        Statement::Cond(c) => Ok(with_bodies(
            &stmt,
            vec![simplify_in(&c.body, ctx)?, simplify_in(&c.else_body, ctx)?],
        )),
        _ => Ok(stmt),
    }
}

fn simplify_in(body: &Body, ctx: &SimplifyCtx) -> Result<Body> {
    let stmts: Result<Vec<Arc<Statement>>> =
        body.stmts().iter().map(|s| simplify_stmt(s, ctx)).collect();
    Ok(Body::new(stmts?))
}

/// `s` without the branches its simplified predicates decide.
fn decided(s: &Select) -> Select {
    let mut kept: Vec<SelectBranch> = Vec::new();
    let Some((last, rest)) = s.branches.split_last() else {
        return s.clone();
    };
    for branch in rest {
        match branch.select.literal_value() {
            None => kept.push(branch.clone()),
            Some(value) if value.truthy() => {
                kept.push(branch.clone());
                return Select {
                    name: s.name.clone(),
                    branches: kept.into(),
                };
            }
            Some(_) => {}
        }
    }
    kept.push(last.clone());
    Select {
        name: s.name.clone(),
        branches: kept.into(),
    }
}

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

type Marks = BTreeSet<(Name, u64)>;
type PlaceKey = (Number, Marks);

/// Place every pure value once, at the shallowest loop scope on its consumers' path that binds its coordinates and
/// holds its operands; a second computation of a value a scope can already see reads the first instead.
pub fn place_values(body: &Body) -> Result<Body> {
    let body = distinct(body);
    let numbering = value_numbers(&body, &crate::values::spelled_key);
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
                let number = self.numbering.numbers[self.numbering.ordinal(stmt)]
                    .as_ref()
                    .map_or(0, |value| value.0);
                let key: PlaceKey = (number, marks.clone());
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
                    let where_ = self.target(depth, &Placer::reads(&stmt), &key.1);
                    if where_ <= depth {
                        taken.insert(index);
                        for (other, own) in else_kept[index].defines().iter().zip(stmt.defines()) {
                            self.alias.insert(other.clone(), own);
                        }
                        let defines = stmt.defines();
                        self.settle(&defines, where_, &key.1);
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

// --- sequential names -----------------------------------------------------------------------------------

fn ssa_prefix(stmt: &Statement) -> &'static str {
    match stmt {
        Statement::Load(_) => "in",
        Statement::Accum(_) | Statement::Init(_) => "acc",
        _ => "v",
    }
}

struct Counters {
    counts: HashMap<&'static str, usize>,
    reserved: HashSet<Name>,
}

impl Counters {
    fn fresh(&mut self, kind: &'static str) -> Name {
        loop {
            let count = self.counts.entry(kind).or_insert(0);
            let name: Name = format!("{kind}{count}").into();
            *count += 1;
            if !self.reserved.contains(&name) {
                return name;
            }
        }
    }
}

/// Mutable state for one lexical scope of sequential renaming.
struct SequentialScope {
    ssa: HashMap<Name, Name>,
    sources: HashMap<Name, Name>,
    inherited_axes: HashMap<Name, Name>,
    owned: HashSet<Name>,
    fixed: HashSet<Name>,
}

impl SequentialScope {
    fn allocate(&mut self, counters: &mut Counters, old: &Name, kind: &'static str) {
        if self.owned.contains(old) || self.fixed.contains(old) {
            return;
        }
        self.ssa.insert(old.clone(), counters.fresh(kind));
        self.owned.insert(old.clone());
    }

    /// Rename one next statement and advance this scope's allocation state.
    fn step(&mut self, counters: &mut Counters, stmt: &Arc<Statement>) -> Arc<Statement> {
        let children = stmt.nested();
        // A scalar accumulator is declared beside its own reduce loop, not beside every ancestor loop.
        let mut exported: Vec<Name> = Vec::new();
        for child in &children {
            let scalar = carried_out(child);
            for here in child.of_kinds(&["Accum", "Carry"]) {
                match &**here.stmt() {
                    Statement::Carry(c) => exported.push(c.name.clone()),
                    Statement::Accum(a) if scalar.contains(&a.name) => {
                        exported.push(a.name.clone())
                    }
                    _ => {}
                }
            }
        }
        let exported = unique(exported);
        if !children.is_empty() {
            for name in &exported {
                self.allocate(counters, name, "acc");
            }
        } else {
            for name in stmt.defines() {
                self.allocate(counters, &name, ssa_prefix(stmt));
            }
        }
        let mut axes = self.inherited_axes.clone();
        for old in stmt.binds_axes() {
            axes.insert(old, counters.fresh("a"));
        }
        for axis in stmt.bound_axes() {
            for source in axis.sources() {
                if !self.sources.contains_key(&source.name) {
                    self.sources
                        .insert(source.name.clone(), counters.fresh("p"));
                }
            }
        }
        let mut names = self.ssa.clone();
        names.extend(self.sources.iter().map(|(k, v)| (k.clone(), v.clone())));
        names.extend(axes.iter().map(|(k, v)| (k.clone(), v.clone())));
        let shell = if children.is_empty() {
            stmt.clone()
        } else {
            with_bodies(stmt, children.iter().map(|_| Body::empty()).collect())
        };
        let renamed = Statement::rename(&shell, &names);
        if children.is_empty() {
            return renamed;
        }
        let mut renamed_children = Vec::new();
        for child in &children {
            let mut scope = SequentialScope {
                ssa: self.ssa.clone(),
                sources: self.sources.clone(),
                inherited_axes: axes.clone(),
                owned: HashSet::new(),
                fixed: exported.iter().cloned().collect(),
            };
            let members: Vec<Arc<Statement>> = child
                .stmts()
                .iter()
                .map(|member| scope.step(counters, member))
                .collect();
            self.sources.extend(scope.sources);
            renamed_children.push(Body::new(members));
        }
        with_bodies(&renamed, renamed_children)
    }
}

/// Canonicalize names in a fused body: axes to `a0, a1, …`, window parents to `p0, …`, loads to `in0, …`,
/// accumulators to `acc0, …`, every other definition to `v0, …`, in definition order. Each nested body is a lexical
/// scope, while its assigned names remain globally unique.
pub fn rename_ssa_sequential(body: &Body) -> Body {
    let mut reserved: HashSet<Name> = HashSet::new();
    for stmt in body.stmts() {
        reserved.extend(free_names(stmt));
    }
    for name in body.definitions().keys() {
        reserved.remove(name);
    }
    let mut counters = Counters {
        counts: HashMap::new(),
        reserved,
    };
    let mut scope = SequentialScope {
        ssa: HashMap::new(),
        sources: HashMap::new(),
        inherited_axes: HashMap::new(),
        owned: HashSet::new(),
        fixed: HashSet::new(),
    };
    Body::new(
        body.stmts()
            .iter()
            .map(|stmt| scope.step(&mut counters, stmt))
            .collect(),
    )
}

// --- the canonical order --------------------------------------------------------------------------------

/// A sum with its coefficients sorted: the order lists a sum's terms itself, so what it sorts by reads none of that.
fn unlisted(stmt: &Arc<Statement>) -> Arc<Statement> {
    if let Statement::Affine(a) = &**stmt {
        let mut coefficients: Vec<Float> = a.coefficients.to_vec();
        coefficients.sort_by(|x, y| x.0.total_cmp(&y.0));
        return Arc::new(Statement::Affine(Affine {
            coefficients: coefficients.into(),
            ..a.clone()
        }));
    }
    stmt.clone()
}

/// Dependency-valid statement order, commutative operand order and names, all from the value numbering.
pub fn canonical_order(body: &Body) -> Result<Body> {
    let stmts = distinct(&canonicalize_expressions(body, &[])?);
    let role = roles(&stmts, None);
    let numbering = value_numbers(&stmts, &|name| {
        role.get(name)
            .map_or_else(|| Key::Str(name.clone()), |r| Key::Int(*r as i64))
    });
    let slot = |_: &str| Some(Name::from("__slot__"));
    let slots: HashMap<Name, Name> = HashMap::new();
    let _ = &slots;

    // A statement's own shape with every name and buffer one placeholder.
    let shape = |stmt: &Arc<Statement>| -> Key {
        let children = stmt.nested();
        let shell = if children.is_empty() {
            stmt.clone()
        } else {
            with_bodies(stmt, children.iter().map(|_| Body::empty()).collect())
        };
        let renamed = Statement::rename_with(&shell, &slot);
        let mut buffers: HashMap<Name, Name> = HashMap::new();
        for name in renamed
            .external_reads()
            .into_iter()
            .chain(renamed.external_writes())
        {
            buffers.insert(name, Name::from("__slot__"));
        }
        let shell = unlisted(&Statement::rename_buffers(&renamed, &buffers));
        Key::str(&structural_text(&shell))
    };
    // The whole subtree with names abstracted and buffers spelled: the tie-break between siblings of one shape.
    let spelling = |stmt: &Arc<Statement>| -> Key {
        let renamed = Statement::rename_with(stmt, &|_| Some(Name::from("__name__")));
        let body = Body::new(vec![renamed]).map_stmts(&mut |s| Some(unlisted(s)));
        text_key(&body)
    };
    fn ordered(
        body: &Body,
        numbering: &Numbering,
        shape: &dyn Fn(&Arc<Statement>) -> Key,
        spelling: &dyn Fn(&Arc<Statement>) -> Key,
    ) -> Body {
        let rebuilt = body.map_each(&mut |stmt| {
            let nested = stmt.nested();
            if nested.is_empty() {
                return stmt.clone();
            }
            with_bodies(
                stmt,
                nested
                    .iter()
                    .map(|child| ordered(child, numbering, shape, spelling))
                    .collect(),
            )
        });
        let shapes: Vec<Key> = body.stmts().iter().map(shape).collect();
        let mut ties: HashMap<&Key, usize> = HashMap::new();
        for own in &shapes {
            *ties.entry(own).or_default() += 1;
        }
        let mut effects = HashMap::new();
        let priorities: Vec<Key> = body
            .stmts()
            .iter()
            .zip(rebuilt.stmts())
            .zip(&shapes)
            .map(|((stmt, built), own)| {
                let children = stmt.nested();
                let category = if !children.is_empty() && stmt.has_side_effects() {
                    2
                } else {
                    children.is_empty() as i64
                };
                let spelled = if ties[own] > 1 {
                    spelling(built)
                } else {
                    Key::str("")
                };
                if !children.is_empty() {
                    let tree = scope_tree(numbering, &Body::new(vec![stmt.clone()]), &mut effects);
                    Key::tuple([Key::Int(category), own.clone(), spelled, Key::Num(tree)])
                } else {
                    let (number, params) = numbering.numbers[numbering.ordinal(stmt)]
                        .as_ref()
                        .map_or((Key::None, Vec::new()), |(n, p)| (Key::Num(*n), p.clone()));
                    Key::tuple([
                        Key::Int(category),
                        own.clone(),
                        spelled,
                        number,
                        Key::Tuple(params),
                    ])
                }
            })
            .collect();
        let incoming = ordering_constraints(&rebuilt, true, true);
        let order =
            rebuilt.topological_permutation(&incoming, Some(&|index| priorities[index].clone()));
        Body::new(
            order
                .into_iter()
                .map(|index| rebuilt.stmts()[index].clone())
                .collect(),
        )
    }
    let placed = ordered(&stmts, &numbering, &shape, &spelling);
    let result = rename_ssa_sequential(&placed);
    // The operands of a commutative operation list accumulators, then loads, then computed values, then by value,
    // and two operands of one value by the names the order just gave them.
    let mut operand: HashMap<Name, Key> = HashMap::new();
    for (before, after) in placed.iter_stmts().iter().zip(result.iter_stmts()) {
        if let Some(ordinal) = numbering.ordinal_of(before)
            && let Some((number, params)) = &numbering.numbers[ordinal]
        {
            let rank = match &*after {
                Statement::Accum(_) | Statement::Init(_) => 0,
                Statement::Load(_) => 1,
                _ => 2,
            };
            for name in after.defines() {
                operand.insert(
                    name,
                    Key::tuple([
                        Key::Int(rank),
                        Key::Num(*number),
                        Key::Tuple(params.clone()),
                    ]),
                );
            }
        }
    }
    let operand_key = |name: &Name| {
        operand
            .get(name)
            .cloned()
            .unwrap_or_else(|| Key::tuple([Key::Int(3), Key::str("")]))
    };
    let listed = result.map_stmts(&mut |stmt| match &**stmt {
        Statement::Assign(a) if a.op.commutative && a.args.len() > 1 => {
            let mut args: Vec<Name> = a.args.to_vec();
            args.sort_by_cached_key(|name| (operand_key(name), name.clone()));
            Some(Arc::new(Statement::Assign(Assign {
                args: args.into(),
                ..a.clone()
            })))
        }
        Statement::Affine(a) if a.args.len() > 1 => {
            let mut terms: Vec<(f64, Name)> = a.terms().map(|(c, arg)| (c, arg.clone())).collect();
            terms.sort_by_cached_key(|(c, name)| (operand_key(name), name.clone(), Key::float(*c)));
            Some(Arc::new(Statement::Affine(Affine {
                args: terms.iter().map(|(_, n)| n.clone()).collect(),
                coefficients: terms.iter().map(|(c, _)| Float(*c)).collect(),
                ..a.clone()
            })))
        }
        _ => Some(stmt.clone()),
    });
    // Sequential naming separates lexical binders. Restore shared reduction dimensions without merging dependent
    // loops or changing the order.
    let unified = listed.map_stmts(&mut |stmt| {
        let nested = stmt.nested();
        if nested.is_empty() {
            return Some(stmt.clone());
        }
        Some(with_bodies(
            stmt,
            nested.iter().map(unify_siblings).collect(),
        ))
    });
    Ok(unify_siblings(&unified))
}

/// Canonicalize integer expressions using lexical binding order, never axis spelling.
pub fn canonicalize_expressions(body: &Body, axes: &[Name]) -> Result<Body> {
    let axis_order: HashMap<Name, usize> = axes
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), i))
        .collect();
    let commutative = |op: BinOp| {
        matches!(
            op,
            BinOp::Add
                | BinOp::Mul
                | BinOp::Eq
                | BinOp::Ne
                | BinOp::And
                | BinOp::Or
                | BinOp::BitAnd
                | BinOp::BitOr
                | BinOp::Xor
        )
    };
    fn unspelled(e: &Expression, axis_order: &HashMap<Name, usize>) -> Key {
        match &**e {
            Expr::Var(name) => match axis_order.get(name) {
                Some(index) => Key::tuple([Key::str("Var"), Key::Int(*index as i64)]),
                None => Key::tuple([Key::str("Var"), Key::Str(name.clone())]),
            },
            Expr::Literal(Lit { value, dtype }) => Key::tuple([
                Key::str("Literal"),
                Key::num(*value),
                Key::Str(dtype.clone()),
            ]),
            Expr::Builtin(name) => Key::tuple([Key::str("Builtin"), Key::Str(name.clone())]),
            Expr::Binary(op, l, r) => Key::tuple([
                Key::str("Binary"),
                Key::str(op.text()),
                unspelled(l, axis_order),
                unspelled(r, axis_order),
            ]),
            Expr::Call(name, args) => Key::tuple([
                Key::str("Call"),
                Key::Str(name.clone()),
                Key::tuple(args.iter().map(|a| unspelled(a, axis_order))),
            ]),
            Expr::Ternary(a, b, c) => Key::tuple([
                Key::str("Ternary"),
                unspelled(a, axis_order),
                unspelled(b, axis_order),
                unspelled(c, axis_order),
            ]),
            Expr::Cast(dtype, inner) => Key::tuple([
                Key::str("Cast"),
                Key::Str(dtype.clone()),
                unspelled(inner, axis_order),
            ]),
            Expr::FlatIndex(buffer, index) => Key::tuple([
                Key::str("FlatIndex"),
                Key::Str(buffer.clone()),
                Key::tuple(index.iter().map(|a| unspelled(a, axis_order))),
            ]),
        }
    }
    let affine = |expression: Expression| -> Result<Expression> {
        let variables = expression.free_vars();
        // Reassociation and coefficient folding are exact for integer coordinates; an SSA value may be floating point.
        if variables.is_empty() || !variables.iter().all(|v| axis_order.contains_key(v)) {
            return Ok(expression);
        }
        let Some((anchor, coefficients)) =
            affine_form(&expression, &|name| variables.iter().any(|v| &**v == name))
        else {
            return Ok(expression);
        };
        let anchor = anchor.simplify(&SimplifyCtx::empty())?;
        let mut sorted: Vec<(&Name, &i64)> = coefficients.iter().collect();
        sorted.sort_by_key(|(name, _)| axis_order[*name]);
        let mut terms: Vec<Expression> = Vec::new();
        for (name, coefficient) in sorted {
            let variable = Expression::new(Expr::Var(name.clone()));
            terms.push(if *coefficient == 1 {
                variable
            } else {
                Expression::binary(BinOp::Mul, Expression::int(*coefficient), variable)
            });
        }
        if !anchor.is_zero() {
            terms.push(anchor);
        }
        let mut terms = terms.into_iter();
        let Some(mut result) = terms.next() else {
            return Ok(Expression::int(0));
        };
        for term in terms {
            result = Expression::binary(BinOp::Add, result, term);
        }
        Ok(result)
    };
    fn rewrite(
        e: &Expression,
        affine: &dyn Fn(Expression) -> Result<Expression>,
        commutative: &dyn Fn(BinOp) -> bool,
        axis_order: &HashMap<Name, usize>,
    ) -> Result<Expression> {
        Ok(match &**e {
            Expr::Binary(op, left, right) => {
                let (mut left, mut right) = (
                    rewrite(left, affine, commutative, axis_order)?,
                    rewrite(right, affine, commutative, axis_order)?,
                );
                let mut op = *op;
                match op {
                    BinOp::Gt => {
                        op = BinOp::Lt;
                        std::mem::swap(&mut left, &mut right);
                    }
                    BinOp::Ge => {
                        op = BinOp::Le;
                        std::mem::swap(&mut left, &mut right);
                    }
                    _ => {}
                }
                if commutative(op) && unspelled(&right, axis_order) < unspelled(&left, axis_order) {
                    std::mem::swap(&mut left, &mut right);
                }
                let result = Expression::binary(op, left, right);
                if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul) {
                    affine(result)?
                } else {
                    result
                }
            }
            Expr::Call(name, args) => {
                let args: Result<Vec<Expression>> = args
                    .iter()
                    .map(|a| rewrite(a, affine, commutative, axis_order))
                    .collect();
                Expression::new(Expr::Call(name.clone(), args?.into()))
            }
            Expr::Ternary(a, b, c) => Expression::new(Expr::Ternary(
                rewrite(a, affine, commutative, axis_order)?,
                rewrite(b, affine, commutative, axis_order)?,
                rewrite(c, affine, commutative, axis_order)?,
            )),
            Expr::Cast(dtype, inner) => Expression::new(Expr::Cast(
                dtype.clone(),
                rewrite(inner, affine, commutative, axis_order)?,
            )),
            _ => e.clone(),
        })
    }
    let mut failure: Option<Error> = None;
    let out: Vec<Arc<Statement>> = body
        .stmts()
        .iter()
        .map(|stmt| {
            let rewritten = Statement::map_expressions(stmt, &mut |e| match rewrite(
                e,
                &affine,
                &commutative,
                &axis_order,
            ) {
                Ok(done) => done,
                Err(err) => {
                    failure = Some(err);
                    e.clone()
                }
            });
            let mut bound: Vec<Name> = axes.to_vec();
            bound.extend(stmt.bound_axes().iter().map(|axis| axis.name.clone()));
            let nested = rewritten.nested();
            if nested.is_empty() {
                return Ok(rewritten);
            }
            let bodies: Result<Vec<Body>> = nested
                .iter()
                .map(|child| canonicalize_expressions(child, &bound))
                .collect();
            Ok(with_bodies(&rewritten, bodies?))
        })
        .collect::<Result<_>>()?;
    if let Some(err) = failure {
        return Err(err);
    }
    Ok(Body::new(out))
}

/// Every `Body` reading the normalization holds of a statement for the `coalesce` step and the readers.
pub fn body_from_free(stmt: &Arc<Statement>) -> HashSet<Name> {
    free_ssa(stmt)
}

pub fn sibling_defs(stmt: &Statement) -> Vec<Name> {
    ordered_sibling_defs(stmt)
}

pub fn carried_names_out(body: &Body) -> HashSet<Name> {
    carried_out(body)
}

pub fn placeholder_num(value: f64) -> Num {
    Num::Float(value)
}

pub fn step_of(step: &NumOr<Expression>) -> Option<&Expression> {
    match step {
        NumOr::Value(e) => Some(e),
        NumOr::Num(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::{body_text, parse_body};

    fn normal(text: &str) -> String {
        let body = parse_body(text).unwrap();
        let normal = normalize_body(&body).unwrap_or_else(|err| panic!("{err}"));
        assert_eq!(
            normalize_uncached(&normal).unwrap(),
            normal,
            "the normal form is final"
        );
        body_text(&normal).unwrap()
    }

    #[test]
    fn a_body_normalizes_and_settles() {
        let text = normal(
            "for i in 0..4:\n    for k in 0..8:\n        x = load a[i, k]\n        y = exp(x)\n        s <- add(s, y) over k\n    out[i] = s",
        );
        assert!(text.contains("acc0 <- add(acc0, v0) over a1"), "{text}");
        let folded = normal(
            "for i in 0..4:\n    x = load a[i]\n    let c = 2.0\n    y = multiply(x, c)\n    z = add(y, x)\n    out[i] = z",
        );
        assert!(folded.contains("v0 = 3*in0"), "{folded}");
    }

    #[test]
    fn renamed_bodies_share_an_identity() {
        let a = parse_body("for i in 0..4:\n    x = load p[i]\n    y = exp(x)\n    out[i] = y")
            .unwrap();
        let b = parse_body("for j in 0..4:\n    q = load r[j]\n    w = exp(q)\n    res[j] = w")
            .unwrap();
        let ia = crate::identity::canonicalize_identity(&a, false, None).unwrap();
        let ib = crate::identity::canonicalize_identity(&b, false, None).unwrap();
        assert_eq!(ia.key, ib.key);
        assert_eq!(ia.arguments, vec![Name::from("p"), Name::from("out")]);
        let c = parse_body("for j in 0..4:\n    q = load r[j]\n    w = log(q)\n    res[j] = w")
            .unwrap();
        assert_ne!(
            crate::identity::canonicalize_identity(&c, false, None)
                .unwrap()
                .key,
            ia.key
        );
    }
}
