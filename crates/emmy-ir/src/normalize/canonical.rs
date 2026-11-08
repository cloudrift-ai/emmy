//! The canonical order: sequential names, a dependency-valid statement order with commutative operand order from the
//! value numbering, and integer expressions canonicalized by lexical binding order.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::ties;
use super::{carried_out, unify_siblings, with_bodies};
use crate::body::{Body, free_names, unique};
use crate::elementwise::{AffineRole, Operation};
use crate::expr::{Expr, Expression, LitType, SimplifyCtx, affine_form};
use crate::numbering::{Chains, Key, Numbering, roles, statement_tree, value_numbers};
use crate::order::ordering_constraints;
use crate::statement::*;
use crate::{Name, Result};

// --- sequential names -----------------------------------------------------------------------------------

pub(crate) fn ssa_prefix(stmt: &Statement) -> &'static str {
    match stmt {
        Statement::Load(_) => "in",
        Statement::Accum(_) | Statement::Init(_) => "acc",
        _ => "v",
    }
}

pub(crate) struct Counters {
    counts: HashMap<&'static str, usize>,
    reserved: HashSet<Name>,
}

impl Counters {
    fn fresh(&mut self, prefix: &'static str) -> Name {
        loop {
            let count = self.counts.entry(prefix).or_insert(0);
            let name: Name = format!("{prefix}{count}").into();
            *count += 1;
            if !self.reserved.contains(&name) {
                return name;
            }
        }
    }
}

/// Mutable state for one lexical scope of sequential renaming.
pub(crate) struct SequentialScope {
    ssa: HashMap<Name, Name>,
    sources: HashMap<Name, Name>,
    inherited_axes: HashMap<Name, Name>,
    owned: HashSet<Name>,
    fixed: HashSet<Name>,
}

impl SequentialScope {
    fn allocate(&mut self, counters: &mut Counters, old: &Name, prefix: &'static str) {
        if self.owned.contains(old) || self.fixed.contains(old) {
            return;
        }
        self.ssa.insert(*old, counters.fresh(prefix));
        self.owned.insert(*old);
    }

    /// Rename one next statement and advance this scope's allocation state.
    fn step(&mut self, counters: &mut Counters, stmt: &Arc<Statement>) -> Arc<Statement> {
        let children = stmt.nested();
        // A scalar accumulator is declared beside its own reduce loop, not beside every ancestor loop.
        let mut exported: Vec<Name> = Vec::new();
        for child in &children {
            let scalar = carried_out(child);
            for here in child.of_types(&[StatementType::Accum, StatementType::Carry]) {
                match &**here.stmt() {
                    Statement::Carry(c) => exported.push(c.name),
                    Statement::Accum(a) if scalar.contains(&a.name) => exported.push(a.name),
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
                self.sources
                    .entry(source.name)
                    .or_insert_with(|| counters.fresh("p"));
            }
        }
        let mut names = self.ssa.clone();
        names.extend(self.sources.iter().map(|(k, v)| (*k, *v)));
        names.extend(axes.iter().map(|(k, v)| (*k, *v)));
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
            renamed_children.push(child.rebuilt(members));
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
    body.rebuilt(
        body.stmts()
            .iter()
            .map(|stmt| scope.step(&mut counters, stmt))
            .collect(),
    )
}

// --- the canonical order --------------------------------------------------------------------------------

/// `body` with every scope's members in the canonical order: dependency-valid, then by the sibling's category, its
/// shape, its spelling where shapes tie, and its number or scope tree.
fn ordered(body: &Body, numbering: &Numbering) -> Result<Body> {
    let rebuilt = body.try_map_each(&mut |stmt| {
        let nested = stmt.nested();
        if nested.is_empty() {
            return Ok(stmt.clone());
        }
        let bodies: Result<Vec<Body>> = nested
            .iter()
            .map(|child| ordered(child, numbering))
            .collect();
        Ok(with_bodies(stmt, bodies?))
    })?;
    let shapes: Vec<Key> = body
        .stmts()
        .iter()
        .map(ties::sibling_shape)
        .collect::<Result<_>>()?;
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
        .enumerate()
        .map(|(index, ((stmt, built), own))| {
            let children = stmt.nested();
            let category = ties::sibling_category(stmt);
            let spelled = if ties[own] > 1 {
                ties::sibling_spelling(built)
            } else {
                Key::str("")
            };
            if !children.is_empty() {
                let tree = statement_tree(numbering, &body.at(index), &mut effects);
                Key::tuple([Key::Int(category), own.clone(), spelled, Key::Num(tree)])
            } else {
                let (number, params) = numbering.numbers[numbering.ordinal(&body.at(index))]
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
    Ok(rebuilt.rebuilt(
        order
            .into_iter()
            .map(|index| rebuilt.stmts()[index].clone())
            .collect(),
    ))
}

pub fn canonical_order(body: &Body) -> Result<Body> {
    let stmts = canonicalize_expressions(body, &[])?;
    let role = roles(&stmts, None);
    let numbering = value_numbers(
        &stmts,
        &|name| {
            role.get(name)
                .map_or_else(|| Key::Str(*name), |r| Key::Int(*r as i64))
        },
        Chains::Flat,
    );
    let placed = ordered(&stmts, &numbering)?;
    let result = rename_ssa_sequential(&placed);
    let operand = ties::operand_ranks(&placed, &result, &numbering);
    let operand_key = |name: &Name| operand.get(name).cloned().unwrap_or_else(ties::unranked);
    let rank = |name: &Name| operand_key(name);
    let listed = result.map_stmts(&mut |stmt| Some(value_fields(stmt, &rank)));
    // Sequential naming separates lexical binders. Restore shared reduction dimensions without merging dependent
    // loops or changing the order.
    let unified = listed.try_map_stmts(&mut |stmt| {
        let nested = stmt.nested();
        if nested.is_empty() {
            return Ok(Some(stmt.clone()));
        }
        let bodies: Result<Vec<Body>> = nested.iter().map(unify_siblings).collect();
        Ok(Some(with_bodies(stmt, bodies?)))
    })?;
    unify_siblings(&unified)
}

/// A sum over bound axes with its terms in binding order and its anchor last.
fn affine(expression: Expression, axis_order: &HashMap<Name, usize>) -> Result<Expression> {
    let variables = expression.free_vars();
    // Reassociation and coefficient folding are exact for integer coordinates; an SSA value may be floating point.
    if variables.is_empty() || !variables.iter().all(|v| axis_order.contains_key(v)) {
        return Ok(expression);
    }
    let Some((anchor, coefficients)) = affine_form(&expression, &|name| variables.contains(&name))
    else {
        return Ok(expression);
    };
    let anchor = anchor.simplify(&SimplifyCtx::empty())?;
    let mut sorted: Vec<(&Name, &i64)> = coefficients.iter().collect();
    sorted.sort_by_key(|(name, _)| axis_order[*name]);
    let mut terms: Vec<Expression> = Vec::new();
    for (name, coefficient) in sorted {
        let variable = Expression::new(Expr::Var(*name));
        terms.push(if *coefficient == 1 {
            variable
        } else {
            Expression::binary(Operation::Multiply, Expression::int(*coefficient), variable)
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
        result = Expression::binary(Operation::Add, result, term);
    }
    Ok(result)
}

/// `e` canonicalized: comparisons as `<` and `<=`, commutative operands ordered, sums in affine form.
fn rewrite(e: &Expression, axis_order: &HashMap<Name, usize>) -> Result<Expression> {
    Ok(match &**e {
        Expr::Binary(op, left, right) => {
            let (mut left, mut right) = (rewrite(left, axis_order)?, rewrite(right, axis_order)?);
            let mut op = *op;
            if matches!(op, Operation::Greater | Operation::GreaterEqual)
                && let Some(mirrored) = op.mirror()
            {
                op = mirrored;
                std::mem::swap(&mut left, &mut right);
            }
            if op.info().commutative
                && ties::unspelled(&right, axis_order) < ties::unspelled(&left, axis_order)
            {
                std::mem::swap(&mut left, &mut right);
            }
            let result = Expression::binary(op, left, right);
            if matches!(
                op.info().affine,
                Some(AffineRole::Sum | AffineRole::Difference | AffineRole::Scale)
            ) {
                affine(result, axis_order)?
            } else {
                result
            }
        }
        Expr::Call(name, args) => {
            let args: Result<Vec<Expression>> =
                args.iter().map(|a| rewrite(a, axis_order)).collect();
            Expression::new(Expr::Call(*name, args?.into()))
        }
        Expr::Ternary(a, b, c) => Expression::new(Expr::Ternary(
            rewrite(a, axis_order)?,
            rewrite(b, axis_order)?,
            rewrite(c, axis_order)?,
        )),
        Expr::Cast(dtype, inner) => {
            Expression::new(Expr::Cast(dtype.clone(), rewrite(inner, axis_order)?))
        }
        _ => e.clone(),
    })
}

/// Canonicalize integer expressions using lexical binding order, never axis spelling.
/// `stmt` with every Value-class field in its canonical form under `rank` (see [`value_canonical`]): the value of
/// an assign, an accum or a let, a condition, a select's predicates. Index fields keep the index rules.
fn value_fields(stmt: &Arc<Statement>, rank: &dyn Fn(&Name) -> Key) -> Arc<Statement> {
    match &**stmt {
        Statement::Assign(a) => Arc::new(Statement::Assign(Assign {
            value: value_canonical(&a.value, &rank),
            ..a.clone()
        })),
        Statement::Accum(a) => Arc::new(Statement::Accum(Accum {
            value: value_canonical(&a.value, &rank),
            ..a.clone()
        })),
        Statement::Let(l) => Arc::new(Statement::Let(Let {
            value: value_canonical(&l.value, &rank),
            ..l.clone()
        })),
        Statement::Cond(c) => Arc::new(Statement::Cond(Cond {
            cond: value_canonical(&c.cond, &rank),
            ..c.clone()
        })),
        Statement::Select(sel) => Arc::new(Statement::Select(Select {
            branches: sel
                .branches
                .iter()
                .map(|b| SelectBranch {
                    value: b.value,
                    select: value_canonical(&b.select, &rank),
                })
                .collect(),
            ..sel.clone()
        })),
        _ => stmt.clone(),
    }
}

/// A value's canonical form: comparisons as `<` and `<=`, a commutative node's operands by rank, and every sum in
/// affine form — its terms `coefficient * operand` collected through additions, subtractions, negations and literal
/// factors or divisors, like terms merged, zero terms dropped, the terms by the rank of their operands and the
/// literal offset last. The reassociation of a sum and the rounding of a literal reciprocal are the exactness every
/// reduce schedule and fast-math division already take.
pub(crate) fn value_canonical(e: &Expression, rank: &dyn Fn(&Name) -> Key) -> Expression {
    let e = e.map_children(&mut |child| value_canonical(child, rank));
    let Some((op, args)) = e.operation() else {
        return e;
    };
    if affine_node(op, &args) {
        return affine_value(&e, rank);
    }
    if let [left, right] = &args[..] {
        let (mut op, mut left, mut right) = (op, left.clone(), right.clone());
        if matches!(op, Operation::Greater | Operation::GreaterEqual)
            && let Some(mirrored) = op.mirror()
        {
            op = mirrored;
            std::mem::swap(&mut left, &mut right);
        }
        if op.info().commutative && ranked(&right, rank) < ranked(&left, rank) {
            std::mem::swap(&mut left, &mut right);
        }
        return Expression::apply(op, vec![left, right]);
    }
    e
}

/// The rank a value sub-expression sorts by: a name's operand rank, then literals, built-ins and nodes by their text.
fn ranked(e: &Expression, rank: &dyn Fn(&Name) -> Key) -> Key {
    match &**e {
        // Two operands of one rank (two loads of buffers the numbering keys alike) order by the names the order gave
        // them; a name the order did not rank (an axis) keeps its place, which the binding order decided.
        Expr::Var(name) => {
            let key = rank(name);
            if key == ties::unranked() {
                Key::tuple([key, Key::None])
            } else {
                Key::tuple([key, Key::Str(*name)])
            }
        }
        Expr::Literal(lit) => Key::tuple([Key::Int(4), Key::num(lit.value)]),
        Expr::Builtin(builtin) => Key::tuple([Key::Int(5), Key::str(builtin.text())]),
        _ => {
            let head = e.operation().map_or_else(
                || e.ty().text().to_string(),
                |(op, _)| op.text().to_string(),
            );
            Key::tuple([
                Key::Int(6),
                Key::str(&head),
                Key::Tuple(
                    e.children()
                        .iter()
                        .map(|child| ranked(child, rank))
                        .collect(),
                ),
            ])
        }
    }
}

/// The terms of an affine tree, `coefficient * operand` each, and its literal offset. A quotient by an integer
/// literal stays a term, since integer division floors; only a float literal divisor folds into a reciprocal scale.
pub struct AffineTerms {
    pub terms: Vec<(f64, Expression)>,
    pub offset: f64,
}

/// The affine form's terms: collected through additions, subtractions, negations and literal factors or divisors.
/// Whether an operation node is a node of the affine form: a sum, a difference, a negation, a product with a literal
/// factor or a quotient by a literal. A product of two values or a quotient by a value is an operation like any other.
pub fn affine_node(op: Operation, args: &[Expression]) -> bool {
    let literal = |index: usize| matches!(args.get(index).map(|a| &**a), Some(Expr::Literal(_)));
    match op.info().affine {
        Some(AffineRole::Sum | AffineRole::Difference | AffineRole::Negation) => true,
        Some(AffineRole::Scale) => literal(0) || literal(1),
        Some(AffineRole::Quotient) => literal(1),
        None => false,
    }
}

pub fn affine_terms(e: &Expression) -> AffineTerms {
    fn collect(e: &Expression, scale: f64, out: &mut AffineTerms) {
        if let Expr::Literal(lit) = &**e
            && lit.dtype != LitType::Bool
        {
            out.offset += scale * lit.value.as_f64();
            return;
        }
        if let Some((op, args)) = e.operation() {
            match (op.info().affine, &args[..]) {
                (Some(AffineRole::Sum), [a, b]) => {
                    collect(a, scale, out);
                    collect(b, scale, out);
                    return;
                }
                (Some(AffineRole::Difference), [a, b]) => {
                    collect(a, scale, out);
                    collect(b, -scale, out);
                    return;
                }
                (Some(AffineRole::Negation), [a]) => {
                    collect(a, -scale, out);
                    return;
                }
                (Some(AffineRole::Scale), [a, b]) => {
                    let factor =
                        [(a, b), (b, a)].into_iter().find_map(
                            |(operand, literal)| match &**literal {
                                Expr::Literal(lit) if lit.dtype != LitType::Bool => {
                                    Some((operand, lit))
                                }
                                _ => None,
                            },
                        );
                    if let Some((operand, lit)) = factor {
                        collect(operand, scale * lit.value.as_f64(), out);
                        return;
                    }
                }
                (Some(AffineRole::Quotient), [a, b]) => {
                    if let Expr::Literal(lit) = &**b
                        && lit.dtype == LitType::Float
                        && lit.value.as_f64() != 0.0
                    {
                        collect(a, scale / lit.value.as_f64(), out);
                        return;
                    }
                }
                _ => {}
            }
        }
        out.terms.push((scale, e.clone()));
    }
    let mut out = AffineTerms {
        terms: Vec::new(),
        offset: 0.0,
    };
    collect(e, 1.0, &mut out);
    out
}

/// `e`, an affine node, as a sum in affine form: see [`value_canonical`].
fn affine_value(e: &Expression, rank: &dyn Fn(&Name) -> Key) -> Expression {
    affine_sum(affine_terms(e), rank)
}

/// The sum an affine form spells, its terms ordered by `rank`: the one spelling of a sum in the normal form, and the
/// one every reader that takes a sum apart and puts it back uses.
pub fn affine_sum(form: AffineTerms, rank: &dyn Fn(&Name) -> Key) -> Expression {
    let AffineTerms { terms, offset } = form;
    // A whole-number coefficient or offset is spelled as the integer: `x + x`, `2.0 * x` and `x / 0.5` are one
    // form, and integer arithmetic over coordinates stays integer. A value promotes an integer literal as it does
    // any operand.
    let literal = |value: f64| {
        if value.fract() == 0.0 && value.abs() < 9007199254740992.0 {
            Expression::int(value as i64)
        } else {
            Expression::float(value)
        }
    };
    // Like terms merge; a coefficient that cancels drops its term.
    let mut merged: Vec<(f64, Expression)> = Vec::new();
    for (coefficient, term) in terms {
        match merged.iter_mut().find(|(_, t)| *t == term) {
            Some(entry) => entry.0 += coefficient,
            None => merged.push((coefficient, term)),
        }
    }
    merged.retain(|(c, _)| *c != 0.0);
    merged.sort_by_cached_key(|(c, term)| (ranked(term, rank), Key::float(*c)));
    let mut out: Option<Expression> = None;
    for (coefficient, term) in merged {
        let magnitude = if coefficient.abs() == 1.0 {
            term
        } else {
            Expression::apply(Operation::Multiply, vec![literal(coefficient.abs()), term])
        };
        out = Some(match out {
            None if coefficient < 0.0 => Expression::apply(Operation::Negative, vec![magnitude]),
            None => magnitude,
            Some(sum) if coefficient < 0.0 => {
                Expression::apply(Operation::Subtract, vec![sum, magnitude])
            }
            Some(sum) => Expression::apply(Operation::Add, vec![sum, magnitude]),
        });
    }
    match out {
        None => literal(offset),
        Some(sum) if offset == 0.0 => sum,
        Some(sum) if offset < 0.0 => {
            Expression::apply(Operation::Subtract, vec![sum, literal(-offset)])
        }
        Some(sum) => Expression::apply(Operation::Add, vec![sum, literal(offset)]),
    }
}

/// Every value expression of `body`'s statements in its canonical spelling under the names as they stand (see
/// [`value_canonical`]): what a hand-written program and a derived one compare by once their names agree.
pub fn canonical_values(body: &Body) -> Body {
    Body::new(
        body.stmts()
            .iter()
            .map(|stmt| value_fields(stmt, &|name: &Name| Key::Str(*name)))
            .collect(),
    )
}

pub fn canonicalize_expressions(body: &Body, axes: &[Name]) -> Result<Body> {
    let axis_order: HashMap<Name, usize> = axes
        .iter()
        .enumerate()
        .map(|(i, name)| (*name, i))
        .collect();
    let out: Vec<Arc<Statement>> = body
        .stmts()
        .iter()
        .map(|stmt| {
            let rewritten = Statement::try_map_expressions(stmt, &mut |e| rewrite(e, &axis_order))?;
            // A bound axis ranks by its binding, as the index rules order it; a value by its name until the order ranks it.
            let rewritten = value_fields(&rewritten, &|name: &Name| match axis_order.get(name) {
                Some(index) => Key::tuple([Key::Int(0), Key::Int(*index as i64)]),
                None => Key::tuple([Key::Int(1), Key::Str(*name)]),
            });
            let mut bound: Vec<Name> = axes.to_vec();
            bound.extend(stmt.bound_axes().iter().map(|axis| axis.name));
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
    Ok(body.rebuilt(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize::normalize_uncached;
    use crate::normalize::testing::*;
    use crate::statement::Load;

    fn find_load(body: &Body, input: &str) -> Load {
        body.loads()
            .into_iter()
            .find_map(|here| match &**here.stmt() {
                Statement::Load(l) if &*l.input == input => Some(l.clone()),
                _ => None,
            })
            .unwrap_or_else(|| panic!("no load from {input}"))
    }

    #[test]
    fn gather_index_survives_rename_chain_collision() {
        // Definition order fixes the renumber: the ids loads take in0 and in1, the gather in2, then a load
        // literally named `in1` (a surviving old name) in3. The gather's index names the second ids load, whose
        // new name `in1` collides with that surviving `in1`: the shape of the embedding + RMSNorm-weight kernel
        // whose gather a rename published into both the name and the axis channel once wired to the wrong row.
        let out = rename_ssa_sequential(&body(
            "
            idx0 = load ids[a0]
            idx1 = load ids[a0]
            gather = load w[idx1, a0]
            in1 = load weight[a0]
            ",
        ));
        assert_eq!(strs(&find_load(&out, "ids").names), ["in0"]);
        assert_eq!(strs(&find_load(&out, "weight").names), ["in3"]);
        assert_eq!(find_load(&out, "w").index[0].text(), "in1");
    }

    #[test]
    fn canonical_names_do_not_capture_free_arguments() {
        let renamed = rename_ssa_sequential(&body(
            "
            for k in 0..4:
                value = load x[a0, k, in0]
                total <- add(total, value)
            ",
        ));
        let bound = axis(&renamed.stmts()[0]).to_string();
        let load = find_load(&renamed, "x");
        assert_ne!(bound, "a0");
        assert_ne!(&*load.names[0], "in0");
        let index: Vec<String> = load.index.iter().map(|e| e.text()).collect();
        assert_eq!(index, ["a0", bound.as_str(), "in0"]);
        assert_eq!(rename_ssa_sequential(&renamed), renamed);
    }

    #[test]
    fn nested_loop_updates_keep_their_enclosing_seed() {
        for strided in [false, true] {
            for seed in [false, true] {
                for conditional in [false, true] {
                    let inner_loop = format!(
                        "for k in 0..4{}{}:",
                        if strided { " step 1" } else { "" },
                        if seed { "" } else { " seed=false" }
                    );
                    let wrapper = if conditional {
                        "if enabled:"
                    } else {
                        "for i in 0..2:"
                    };
                    let renamed = rename_ssa_sequential(&body(&format!(
                        "
                        init f32 total = 0
                        {wrapper}
                            {inner_loop}
                                value = load x[k]
                                total <- add(total, value) over k
                        out[] = total
                        "
                    )));
                    let [ref initial] = renamed.of_types(&[StatementType::Init])[..] else {
                        panic!("one init")
                    };
                    let [ref update] = renamed.of_types(&[StatementType::Accum])[..] else {
                        panic!("one update")
                    };
                    let [ref output] = renamed.of_types(&[StatementType::Write])[..] else {
                        panic!("one write")
                    };
                    let (
                        Statement::Init(initial),
                        Statement::Accum(update),
                        Statement::Write(output),
                    ) = (&**initial.stmt(), &**update.stmt(), &**output.stmt())
                    else {
                        unreachable!()
                    };
                    assert_eq!(strs(&output.values), [&*initial.name]);
                    assert_eq!(
                        update.name == initial.name,
                        !seed,
                        "strided={strided} seed={seed} conditional={conditional}"
                    );
                    assert_eq!(rename_ssa_sequential(&renamed), renamed);
                }
            }
        }
    }

    #[test]
    fn a_commutative_comparison_orders_its_axes_by_binding_not_spelling() {
        // Ordered by spelling, a body re-spelled by its own normalization, as a golden's stored kernel is when it
        // is read back, normalized again to another body, and the kernel's identity moved.
        let condition = |outer: &str, inner_axis: &str| {
            let out = canonicalize_expressions(
                &body(&format!(
                    "
                    for {outer} in 0..4:
                        for {inner_axis} in 0..4:
                            x = load X[{outer}, {inner_axis}]
                            if {inner_axis} == {outer}:
                                O[{outer}, {inner_axis}] = x
                    "
                )),
                &[],
            )
            .unwrap();
            let scope = inner(&inner(&out.stmts()[0]).stmts()[0]);
            let Statement::Cond(cond) = &*scope.stmts()[1] else {
                unreachable!()
            };
            cond.cond.text()
        };
        assert_eq!(condition("a10", "a5"), "a10 == a5");
        assert_eq!(condition("a5", "a10"), "a5 == a10");
    }

    #[test]
    fn a_product_of_two_values_orders_its_operands_by_rank_and_settles() {
        // A product with no literal factor is an operation like any other: its operands take the order's ranks, so
        // the normal form of the normal form is itself whatever order the loads came in.
        for spelling in ["in0 * in1", "in1 * in0"] {
            let given = body(&format!(
                "for a0 in 0..2:\n    for a2 in 0..4:\n        in0 = load A[a2]\n        in1 = load B[a2, a0]\n        f32 v0 = {spelling}\n        acc0 <- add(acc0, v0) over a2\n    O[a0] = acc0"
            ));
            let once = normalize_uncached(&given).unwrap();
            assert_eq!(normalize_uncached(&once).unwrap(), once, "{}", text(&once));
        }
    }
}
