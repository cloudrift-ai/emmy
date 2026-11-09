//! The canonical order: sequential names, a dependency-valid statement order with commutative operand order from the
//! value numbering, and integer expressions canonicalized by lexical binding order.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::ties;
use super::{carried_out, unify_siblings, with_bodies};
use crate::body::{Body, free_names, unique};
use crate::elementwise::{AffineRole, Operation};
use crate::expr::{Expr, ExprClass, Expression, LitType};
use crate::field::Rewriter;
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

    /// Every name in scope to its sequential name: the values, the window parents and the axes.
    fn names(&self, axes: &HashMap<Name, Name>) -> HashMap<Name, Name> {
        let mut names = self.ssa.clone();
        names.extend(self.sources.iter().map(|(k, v)| (*k, *v)));
        names.extend(axes.iter().map(|(k, v)| (*k, *v)));
        names
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
        if children.is_empty() {
            // The reads take the names in scope before the statement, the definitions the names after it: a second
            // binding of a name this scope owns (motion can put two lexical bindings in one scope) is a value of its
            // own, and a first binding that reads an outer name of its spelling reads the outer value. One rename reads
            // the role of each name, so a definition spelled like another's sequential name is no clash.
            let names = self.names(&self.inherited_axes);
            let mut defined: HashMap<Name, Name> = HashMap::new();
            for name in stmt.defines() {
                if stmt.is_pure() && self.owned.contains(&name) {
                    self.ssa.insert(name, counters.fresh(ssa_prefix(stmt)));
                } else {
                    self.allocate(counters, &name, ssa_prefix(stmt));
                }
                if let Some(new) = self.ssa.get(&name) {
                    defined.insert(name, *new);
                }
            }
            return Statement::rename_roles(stmt, &|roles, name| {
                let definition = if roles.contains_any(DEFINES) {
                    defined.get(&name).copied()
                } else {
                    None
                };
                definition.or_else(|| names.get(&name).copied())
            });
        }
        for name in &exported {
            self.allocate(counters, name, "acc");
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
        let names = self.names(&axes);
        let shell = with_bodies(stmt, children.iter().map(|_| Body::empty()).collect());
        let renamed = Statement::rename(&shell, &names);
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
    let incoming = ordering_constraints(&rebuilt, true);
    let order =
        rebuilt.topological_permutation(&incoming, Some(&|index| priorities[index].clone()))?;
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
    let listed = result.map_stmts(&mut |stmt| Some(canonical_fields(stmt, &rank)));
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

/// Every expression of `stmt` in canonical form, each in the coefficient domain its field's class declares
/// (`Index`: integer coefficients, exact over coordinates; `Value`: float), its operands ranked by `rank`.
fn canonical_fields(stmt: &Arc<Statement>, rank: &dyn Fn(&Name) -> Key) -> Arc<Statement> {
    struct Canonical<'a>(&'a dyn Fn(&Name) -> Key);
    impl Rewriter for Canonical<'_> {
        fn expression(&mut self, roles: Roles, expression: &Expression) -> Expression {
            value_canonical(
                expression,
                self.0,
                class_of(roles).unwrap_or(ExprClass::Value),
            )
        }
    }
    let new = stmt.rewrite_fields(&mut Canonical(rank));
    if new.same(stmt) {
        stmt.clone()
    } else {
        Arc::new(new)
    }
}

/// A value's canonical form: comparisons as `<` and `<=`, a commutative node's operands by rank, and every sum in
/// affine form — its terms `coefficient * operand` collected through additions, subtractions, negations and literal
/// factors or divisors, like terms merged, zero terms dropped, the terms by the rank of their operands and the
/// literal offset last. The reassociation of a sum and the rounding of a literal reciprocal are the exactness every
/// reduce schedule and fast-math division already take.
pub(crate) fn value_canonical(
    e: &Expression,
    rank: &dyn Fn(&Name) -> Key,
    class: ExprClass,
) -> Expression {
    let e = e.map_children(&mut |child| value_canonical(child, rank, class));
    let Some((op, args)) = e.operation() else {
        return e;
    };
    // A quotient of two integer literals that does not divide exactly is the class's to fold, ahead of the affine
    // form that would carry it as a term: an index floors it, a value computes it as the float the promotion makes
    // it (`(x - x + 3) / 2` over f32 is `1.5`).
    if op == Operation::Divide
        && let [left, right] = args
        && let (Some(l), Some(r)) = (left.int_literal(), right.int_literal())
        && r != 0
    {
        return match class {
            ExprClass::Index => Expression::int((l as f64 / r as f64).floor() as i64),
            ExprClass::Value => Expression::float(l as f64 / r as f64),
        };
    }
    if affine_node(op, args, class) {
        return affine_sum(affine_terms(&e, class), rank);
    }
    if let [left, right] = args {
        // A remainder by one is zero and a floor division by one is its dividend for integers — an index is one; a
        // value's operand may be a float, whose `x % 1` is its fraction and `x // 1` its floor, so a value keeps both
        // (the simplifier folds them for a dividend under loop ranges; this is the index read through a bare name).
        if class == ExprClass::Index && right.int_literal() == Some(1) {
            match op {
                Operation::Remainder => return Expression::int(0),
                Operation::FloorDivide => return left.clone(),
                _ => {}
            }
        }
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
    /// Whether a float literal fed a coefficient or the offset: then every whole-number coefficient and offset the
    /// sum spells stays a float literal, and the operation keeps the float dtype the literal gave it.
    pub float: bool,
}

/// Whether an operation node is a node of the affine form in `class`'s coefficient domain: a sum, a difference, a
/// negation, a product with a literal factor, and for a value a quotient by a literal. An `Index` folds integer
/// literals alone and no quotient, since integer division floors; a product of two values or a quotient by a value is
/// an operation like any other.
pub fn affine_node(op: Operation, args: &[Expression], class: ExprClass) -> bool {
    let literal = |index: usize| {
        matches!(args.get(index).map(|a| &**a), Some(Expr::Literal(lit))
            if lit.dtype != LitType::Bool && (class == ExprClass::Value || lit.dtype == LitType::Int))
    };
    match op.info().affine {
        Some(AffineRole::Sum | AffineRole::Difference | AffineRole::Negation) => true,
        Some(AffineRole::Scale) => literal(0) || literal(1),
        Some(AffineRole::Quotient) => class == ExprClass::Value && literal(1),
        None => false,
    }
}

/// The affine form's terms: collected through additions, subtractions, negations and literal factors or divisors.
pub fn affine_terms(e: &Expression, class: ExprClass) -> AffineTerms {
    fn collect(e: &Expression, scale: f64, class: ExprClass, out: &mut AffineTerms) {
        if let Expr::Literal(lit) = &**e
            && lit.dtype != LitType::Bool
            && (class == ExprClass::Value || lit.dtype == LitType::Int)
        {
            out.offset += scale * lit.value.as_f64();
            out.float |= lit.dtype == LitType::Float;
            return;
        }
        if let Some((op, args)) = e.operation()
            && affine_node(op, args, class)
        {
            match (op.info().affine, args) {
                (Some(AffineRole::Sum), [a, b]) => {
                    collect(a, scale, class, out);
                    collect(b, scale, class, out);
                    return;
                }
                (Some(AffineRole::Difference), [a, b]) => {
                    collect(a, scale, class, out);
                    collect(b, -scale, class, out);
                    return;
                }
                (Some(AffineRole::Negation), [a]) => {
                    collect(a, -scale, class, out);
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
                        out.float |= lit.dtype == LitType::Float;
                        collect(operand, scale * lit.value.as_f64(), class, out);
                        return;
                    }
                }
                (Some(AffineRole::Quotient), [a, b]) => {
                    if let Expr::Literal(lit) = &**b
                        && lit.dtype == LitType::Float
                        && lit.value.as_f64() != 0.0
                    {
                        out.float = true;
                        collect(a, scale / lit.value.as_f64(), class, out);
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
        float: false,
    };
    collect(e, 1.0, class, &mut out);
    out
}

/// The sum an affine form spells, its terms ordered by `rank`: the one spelling of a sum in the normal form, and the
/// one every reader that takes a sum apart and puts it back uses.
pub fn affine_sum(form: AffineTerms, rank: &dyn Fn(&Name) -> Key) -> Expression {
    let AffineTerms {
        terms,
        offset,
        float,
    } = form;
    // A whole-number coefficient or offset is spelled as the integer when no float literal fed the sum: `x + x` is
    // `2 * x`, and integer arithmetic over coordinates stays integer. Once a float literal fed it, every literal the
    // sum spells is a float (`2.0 * x`, `x - 8.0`, `2.0 * x + 2.0 * y` for `x + x + 2.0 * y`): the literal is what
    // makes an operation over integer operands a float one, and an integer in its place would compute `code - 8` at
    // the code's width. A sum that would spell no literal at all (`1.0 * x`, `x + 0.0`, `x - 8.0 + 8.0`) keeps one on
    // its first term, `1.0 * x`, for the same reason.
    let literal = |value: f64| {
        if !float && value.fract() == 0.0 && value.abs() < 9007199254740992.0 {
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
    let spells_none = offset == 0.0 && merged.iter().all(|(c, _)| c.abs() == 1.0);
    let mut out: Option<Expression> = None;
    for (index, (coefficient, term)) in merged.into_iter().enumerate() {
        let magnitude = if coefficient.abs() == 1.0 && !(float && spells_none && index == 0) {
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
            .map(|stmt| canonical_fields(stmt, &|name: &Name| Key::Str(*name)))
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
            // A bound axis ranks by its binding, as the index rules order it; a value by its name until the order ranks it.
            let rewritten = canonical_fields(stmt, &|name: &Name| match axis_order.get(name) {
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
    use crate::normalize::testing::*;
    use crate::normalize::{normalize_body, normalize_uncached};
    use crate::statement::Load;

    /// Two lexical bindings of one name in one scope are two values: the reads between them read the one before,
    /// and a binding in a deeper scope shadows neither.
    #[test]
    fn two_lexical_bindings_of_one_name_are_two_values() {
        let normal = text(
            &normalize_body(&body(
                "for i in 0..4:
    x = load X[i]
    y = exp(x)
    y = negative(y)
    for j in 0..8:
        y = load Z[i, j]
        w = sqrt(y)
        O[i, j] = w
    P[i] = y",
            ))
            .unwrap(),
        );
        assert_eq!(
            normal,
            "for a0 in 0..4:
    in0 = load X[a0]
    v0 = exp(in0)
    v1 = -v0
    P[a0] = v1
    for a1 in 0..8:
        in1 = load Z[a0, a1]
        v2 = sqrt(in1)
        O[a0, a1] = v2"
        );
    }

    /// A read ahead of every binding of a twice-bound name (a scope is order-free) binds the first binding and stays
    /// ahead of the second, whatever else it waits on.
    #[test]
    fn an_order_free_read_of_a_rebound_name_binds_the_first_binding() {
        let normal = text(
            &normalize_body(&body(
                "w = add(y, z)
y = load A[]
y = negative(y)
z = load B[]
O[] = w
Q[] = y",
            ))
            .unwrap(),
        );
        assert_eq!(
            normal,
            "in0 = load A[]\nv0 = -in0\nin1 = load B[]\nv1 = in0 + in1\nO[] = v1\nQ[] = v0"
        );
    }

    /// A statement that binds a name again beside one it binds for the first time (a multi-name load) gives each its
    /// own sequential name.
    #[test]
    fn a_first_binding_beside_a_rebind_takes_a_sequential_name() {
        let normal = text(
            &normalize_body(&body(
                "x = load A[0]\nP[] = x\nx, y = load B[0]\nz = add(x, y)\nO[] = z",
            ))
            .unwrap(),
        );
        assert_eq!(
            normal,
            "in0, in1 = load B[0]\nv0 = in0 + in1\nin2 = load A[0]\nO[] = v0\nP[] = in2"
        );
    }

    /// A first binding that reads an outer name of its own spelling reads the outer value.
    #[test]
    fn a_first_binding_reading_an_outer_name_of_its_spelling_reads_the_outer_value() {
        let normal = text(
            &normalize_body(&body(
                "y = load A[]\nfor i in 0..4:\n    y = negative(y)\n    O[i] = y",
            ))
            .unwrap(),
        );
        assert_eq!(
            normal,
            "in0 = load A[]\nv0 = -in0\nfor a0 in 0..4:\n    O[a0] = v0"
        );
    }

    /// The bindings of one name keep their source order: a binding that reads forward still sits where it was
    /// written, and the readers of a later binding read that binding.
    #[test]
    fn the_bindings_of_one_name_keep_their_source_order() {
        let normal =
            text(&normalize_body(&body("y = negative(y)\ny = load A[]\nO[] = y")).unwrap());
        assert_eq!(normal, "in0 = load A[]\nO[] = in0");
    }

    /// A scope whose bindings and reads wait on each other is an error, not a body in source order.
    #[test]
    fn a_cycle_through_the_bindings_is_an_error() {
        let cyclic = body("w = add(y, z)\ny = load A[]\ny = negative(y)\nz = exp(y)\nO[] = w");
        assert!(normalize_body(&cyclic).is_err());
    }

    /// A carried state orders nothing within its step: the update and the reads of the step before take the canonical
    /// order whichever side of each other they were written on.
    #[test]
    fn a_carried_state_orders_nothing_within_its_step() {
        let step = |lines: &str| {
            text(
                &normalize_body(&body(&format!(
                    "for c in 0..3:\n    for i in 0..4:\n        u = load U[c, i]\n{lines}\nfor r in 0..4:\n    last = pre S[r]\n    out[r] = last"
                )))
                .unwrap(),
            )
        };
        let read_first =
            step("        own = pre S[i]\n        next = add(own, u)\n        S[i] <- next seed 0");
        let update_first =
            step("        S[i] <- next seed 0\n        own = pre S[i]\n        next = add(own, u)");
        assert_eq!(read_first, update_first);
        assert!(read_first.contains("v0 = pre acc0[a1]\n"), "{read_first}");
    }

    /// An accumulator is read where it stands: a read ahead of its update takes the step before and stays ahead of
    /// the update, a read behind it the updated value — the online softmax's rescale reads the old maximum and the new
    /// one in one step, and its sum's update reads what reads the sum, which is no cycle.
    #[test]
    fn an_accumulator_is_read_where_it_stands() {
        let normal = text(
            &normalize_body(&body(
                "for i in 0..8:
    for k in 0..4:
        x = load X[i, k]
        m_new = maximum(m, x)
        scale = exp(m - m_new)
        p = exp(x - m_new)
        m <- maximum(m, x) over k
        s_old = s * scale
        s <- add(s_old, p) over k
    O[i] = s",
            ))
            .unwrap(),
        );
        let lines: Vec<&str> = normal.lines().map(str::trim).collect();
        let at = |needle: &str| {
            lines
                .iter()
                .position(|l| l.starts_with(needle))
                .unwrap_or_else(|| panic!("{needle}: {normal}"))
        };
        assert!(
            at("v0 = maximum(acc0, in0)") < at("acc0 <- maximum(acc0, in0) over a1"),
            "{normal}"
        );
        assert!(
            at("v1 = acc0 - v0") < at("acc0 <- maximum(acc0, in0) over a1"),
            "{normal}"
        );
        assert!(normal.contains("acc1 * "), "{normal}");
    }

    /// A one-step reduction is its update, and a read of its accumulator ahead of the update reads the seed: the
    /// online softmax's first step rescales by `exp(-inf)` and starts its sum from zero.
    #[test]
    fn a_one_step_reduction_reads_its_seed_ahead_of_its_update() {
        let normal = normalize_body(&body(
            "for i in 0..8:
    for k in 0..1:
        x = load X[i, k]
        m_new = maximum(m, x)
        scale = exp(m - m_new)
        p = exp(x - m_new)
        m <- maximum(m, x) over k
        s_old = s * scale
        s <- add(s_old, p) over k
    O[i] = s",
        ))
        .unwrap();
        let normal = text(&normal);
        assert!(
            !normal.contains("acc") && !normal.contains("<-"),
            "{normal}"
        );
    }

    /// An accumulator updated under a condition is still read where it stands: the reads ahead of the update stay
    /// ahead of it.
    #[test]
    fn an_accumulator_updated_under_a_condition_is_read_where_it_stands() {
        let normal = text(
            &normalize_body(&body(
                "for i in 0..8:
    for k in 0..4:
        x = load X[i, k]
        m_new = maximum(m, x)
        scale = exp(m - m_new)
        if k < 3:
            m <- maximum(m, x) over k
        s_old = s * scale
        s <- add(s_old, m_new) over k
    O[i] = s",
            ))
            .unwrap(),
        );
        let lines: Vec<&str> = normal.lines().map(str::trim).collect();
        let at = |needle: &str| {
            lines
                .iter()
                .position(|l| l.starts_with(needle))
                .unwrap_or_else(|| panic!("{needle}: {normal}"))
        };
        assert!(at("v0 = maximum(acc0, in0)") < at("if "), "{normal}");
        assert!(at("v1 = acc0 - v0") < at("if "), "{normal}");
    }

    /// A carried state's cell loop under a condition is still the step's own update: the body has a reading.
    #[test]
    fn a_carried_states_cell_loop_under_a_condition_orders_nothing_within_its_step() {
        let normal = normalize_body(&body(
            "for c in 0..3:
    if c < 2:
        for i in 0..4:
            u = load U[c, i]
            own = pre S[i]
            next = add(own, u)
            S[i] <- next seed 0
for r in 0..4:
    last = pre S[r]
    out[r] = last",
        ));
        assert!(normal.is_ok(), "{normal:?}");
    }

    /// A statement that rebinds one name and binds another spelled like the first's sequential name gives each its own
    /// name: the definitions map through the statement's own names, the reads through the scope's.
    #[test]
    fn a_rebind_beside_a_name_spelled_like_a_sequential_one_takes_two_names() {
        let normal = text(
            &normalize_body(&body(
                "x = load A[0]\nP[] = x\nx, in0 = load B[0]\nz = add(x, in0)\nO[] = z",
            ))
            .unwrap(),
        );
        assert_eq!(
            normal,
            "in0, in1 = load B[0]\nv0 = in0 + in1\nin2 = load A[0]\nO[] = v0\nP[] = in2"
        );
    }

    /// A one-step reduction's accumulator read ahead of the update through a name field (a select branch) reads the
    /// seed too, and a masked element (`-inf`) meets the seed a kernel starts from, not the neutral element: no `NaN`.
    #[test]
    fn a_one_step_reductions_seed_is_the_kernels_and_reaches_a_name_field() {
        let normal = text(
            &normalize_body(&body(
                "for i in 0..8:
    for k in 0..1:
        x = load X[i, k]
        m_new = maximum(m, x)
        scale = exp(m - m_new)
        y = select:
            m when i < 4
            x when true
        m <- maximum(m, x) over k
        s_old = s * scale
        s <- add(s_old, y) over k
    O[i] = s",
            ))
            .unwrap(),
        );
        assert!(
            !normal.contains("inf") && !normal.contains("acc"),
            "{normal}"
        );
        assert!(
            normal.contains("e+30") || normal.contains("e30"),
            "{normal}"
        );
    }

    /// A one-step update with no base folds the seed its loop started from, as the loop did: a masked `-inf` meets
    /// `-1e30`, and the written maximum is the loop's, read ahead and behind the update alike.
    #[test]
    fn a_one_step_update_folds_the_kernels_seed() {
        let normal = text(
            &normalize_body(&body(
                "for i in 0..8:
    for k in 0..1:
        x = load X[i, k]
        a = m * 2
        m <- maximum(m, x) over k
        b = m * 3
        t = a + b
        s <- add(s, t) over k
    M[i] = m
    S[i] = s",
            ))
            .unwrap(),
        );
        assert!(
            normal.contains("maximum(") && !normal.contains("inf") && !normal.contains("<-"),
            "{normal}"
        );
        assert!(
            normal.contains("e+30") || normal.contains("e30"),
            "{normal}"
        );
    }

    /// A one-step accumulator whose base names a sibling accumulator updated later in the step reads that sibling's
    /// seed, as the loop's first step did: the product with a zero-seeded sum's start is zero, not a free name.
    #[test]
    fn a_one_step_base_naming_a_later_sibling_reads_its_seed() {
        let normal = text(
            &normalize_body(&body(
                "for i in 0..8:
    for k in 0..1:
        x = load X[i, k]
        y = load Y[i, k]
        s <- multiply(t, x) over k
        t <- add(t, y) over k
    S[i] = s
    T[i] = t",
            ))
            .unwrap(),
        );
        assert!(
            normal.contains("= 0.0") && !normal.contains("acc") && !normal.contains("<-"),
            "{normal}"
        );
    }

    /// A sum's zero and a product's one fold away: a one-step sum is its one addend.
    #[test]
    fn a_one_step_sums_seed_folds_away() {
        let normal = text(
            &normalize_body(&body(
                "for i in 0..8:
    for k in 0..1:
        x = load X[i, k]
        s <- add(s, x) over k
    S[i] = s",
            ))
            .unwrap(),
        );
        assert_eq!(
            normal, "for a0 in 0..8:\n    in0 = load X[a0, 0]\n    S[a0] = in0",
            "{normal}"
        );
    }

    /// The loop that folds an accumulator produces it wherever it sits in the block: under a condition (the state
    /// seeded ahead of the block, which is what its siblings may read), or as the loop itself when the accumulator
    /// names no axes. A reader written ahead of the block follows the block.
    #[test]
    fn a_reader_ahead_of_the_block_holding_the_folding_loop_follows_it() {
        for src in [
            "for i in 0..8:
    O[i] = m
    if i < 4:
        for k in 0..4 seed=false:
            x = load X[i, k]
            m <- maximum(m, x) over k",
            "for i in 0..8:
    O[i] = m
    for k in 0..4:
        x = load X[i, k]
        m <- maximum(m, x)",
            "for i in 0..8:
    O[i] = m
    if i < 4:
        for k in 0..4 seed=false:
            for j in 0..2 seed=false:
                x = load X[i, k, j]
                m <- maximum(m, x) over k, j",
        ] {
            let normal = text(&normalize_body(&body(src)).unwrap());
            let update = normal.find("<-").expect(&normal);
            let write = normal.find("O[a0]").expect(&normal);
            assert!(update < write, "{normal}");
        }
    }

    /// The normal form of `src` keeps the read spelled `read` inside the step (behind the inner loop's head) and ahead
    /// of the update spelled `update`: a read of the step before, neither hoisted out of the step nor moved behind the
    /// update.
    fn read_stays_ahead(src: &str, read: &str, update: &str) {
        let normal = text(&normalize_body(&body(src)).unwrap());
        let head = normal.find("for a1").expect(&normal);
        let at = normal.find(read).expect(&normal);
        let updated = normal.find(update).expect(&normal);
        assert!(head < at && at < updated, "{normal}");
    }

    /// An accumulator folded by the enclosing loop and updated inside a map loop of the step (the P·V accumulation of
    /// an attention step) is still an update site: a read of the step before stays ahead of the map loop, whether it
    /// also reads the step's own values or the accumulator alone (which placement alone could hoist).
    #[test]
    fn an_update_inside_a_map_loop_of_the_step_is_read_where_it_stands() {
        for (src, read) in [
            (
                "for i in 0..8:
    for k in 0..4:
        x = load X[i, k]
        r = acc * x
        R[i, k] = r
        for d in 0..2 seed=false:
            v = load V[k, d]
            w = v * x
            acc <- add(acc, w) over k
    O[i] = acc",
                "acc0 * in0",
            ),
            (
                "for i in 0..8:
    for k in 0..4:
        x = load X[i, k]
        r = acc * 2
        R[i, k] = r
        for d in 0..2 seed=false:
            v = load V[k, d]
            w = v * x
            acc <- add(acc, w) over k
    O[i] = acc",
                "2 * acc0",
            ),
        ] {
            read_stays_ahead(src, read, "<- add");
        }
    }

    /// A fold over two axes whose outer axis is the step's: the block holding the inner loop updates the accumulator
    /// in the step's scope, bare or under a condition, and a read ahead of it stays the step before's.
    #[test]
    fn a_fold_whose_outer_axis_is_the_steps_is_an_update_of_the_step() {
        read_stays_ahead(
            "for i in 0..8:
    for k in 0..4:
        x = load X[i, k]
        r = acc * x
        R[i, k] = r
        for j in 0..2 seed=false:
            v = load V[i, k, j]
            acc <- add(acc, v) over k, j
    O[i] = acc",
            "acc0 * in0",
            "<- add",
        );
        read_stays_ahead(
            "for i in 0..8:
    for k in 0..4:
        x = load X[i, k]
        r = m * 2
        R[i, k] = r
        if k < 2:
            for j in 0..2 seed=false:
                y = load Y[i, k, j]
                m <- maximum(m, y) over k, j
    O[i] = m",
            "2 * acc0",
            "<- maximum",
        );
    }

    /// An update under a condition, alone or beside a second one, is read where it stands: a read of the accumulator
    /// alone ahead of the condition stays in the step (placement reads the update site as the order does).
    #[test]
    fn an_update_under_a_condition_keeps_a_read_of_the_accumulator_alone_in_the_step() {
        read_stays_ahead(
            "for i in 0..8:
    for k in 0..4:
        x = load X[i, k]
        r = m * 2
        R[i, k] = r
        if k < 2:
            m <- maximum(m, x) over k
    O[i] = m",
            "2 * acc0",
            "<- maximum",
        );
        read_stays_ahead(
            "for i in 0..8:
    for k in 0..4:
        x = load X[i, k]
        y = load Y[i, k]
        r = m * 2
        R[i, k] = r
        if k < 2:
            m <- maximum(m, x) over k
            m <- maximum(m, y) over k
    O[i] = m",
            "2 * acc0",
            "<- maximum",
        );
    }

    /// A block holding a bare update and a two-axis fold of one name reads as an update whichever comes first (the two
    /// updates keep their source order, as the sites of one name do): the read stays ahead in both spellings.
    #[test]
    fn a_blocks_site_kind_does_not_depend_on_the_order_inside_it() {
        read_stays_ahead(
            "for i in 0..8:
    for k in 0..4:
        x = load X[i, k]
        r = m * 2
        R[i, k] = r
        if k < 2:
            m <- maximum(m, x) over k
            for j in 0..2 seed=false:
                y = load Y[i, k, j]
                m <- maximum(m, y) over k, j
    O[i] = m",
            "2 * acc0",
            "<- maximum",
        );
        read_stays_ahead(
            "for i in 0..8:
    for k in 0..4:
        x = load X[i, k]
        r = m * 2
        R[i, k] = r
        if k < 2:
            for j in 0..2 seed=false:
                y = load Y[i, k, j]
                m <- maximum(m, y) over k, j
            m <- maximum(m, x) over k
    O[i] = m",
            "2 * acc0",
            "<- maximum",
        );
    }

    /// One affine form, in the coefficient domain of the field's class: an index folds integer literals alone and
    /// keeps a quotient a term, a value folds any literal and a quotient by a float literal into its reciprocal.
    #[test]
    fn the_affine_form_takes_its_coefficient_domain_from_the_class() {
        let normal = normalize_uncached(&body(
            "for a0 in 0..8:
    in0 = load x[a0 + a0 * 2 - 1, a0 / 2 + 1]
    v = in0 / 2.0 + in0 + 0.5 * in0
    out[a0] = v",
        ))
        .unwrap();
        let listing = text(&normal);
        assert!(
            listing.contains("load x[3 * a0 - 1, a0 / 2 + 1]"),
            "{listing}"
        );
        assert!(listing.contains("= 2.0 * in0"), "{listing}");
    }

    /// A literal's integer kind reaches only what an integer may: an inexact quotient of integer literals is the
    /// class's to fold (an index floors it, a value is the float the promotion makes it), an operation an index may
    /// hold stays over integer literals, and `% 1` and `// 1` fold for an index or a ranged dividend while a value
    /// keeps them, since its operand may be a float.
    #[test]
    fn a_literals_integer_kind_reaches_only_what_an_integer_may() {
        let normal = |program: &str| text(&normalize_uncached(&body(program)).unwrap());
        let cases: &[(&str, &str)] = &[
            (
                "f32 x = load X[a]\nd = subtract(x, x)\ne = add(d, 3)\nw = divide(e, 2)\nO[a] = w",
                "let f32 v0 = 1.5\n",
            ),
            (
                "f32 x = load X[a]\nd = subtract(x, x)\ne = subtract(d, 1)\nw = divide(e, 2)\nO[a] = w",
                "let f32 v0 = -0.5\n",
            ),
            (
                "for a0 in 0..8:\n    x = load X[(a0 - a0 + 3) / 2]\n    O[a0] = x",
                "load X[1]",
            ),
            (
                "for a0 in 0..8:\n    x = load X[(a0 - a0 - 7) / 2]\n    O[a0] = x",
                "load X[-4]",
            ),
            (
                "for a0 in 0..8:\n    x = load X[maximum(1, 2) * a0]\n    O[a0] = x",
                "load X[2 * a0]",
            ),
            (
                "for a0 in 0..8:\n    x = load X[a0 + minimum(3, 1)]\n    O[a0] = x",
                "load X[a0 + 1]",
            ),
            (
                "f32 x = load X[a]\nv = remainder(x, 1)\nO[a] = v",
                "v0 = in0 % 1\n",
            ),
            (
                "f32 x = load X[a]\nv = floor_divide(x, 1)\nO[a] = v",
                "v0 = in0 // 1\n",
            ),
            (
                "for h in 0..8:\n    for j in 0..1:\n        x = load X[(h + j) % 1, h // 1]\n        O[h, j] = x",
                "load X[0, a0]",
            ),
            (
                "for a0 in 0..8:\n    i = load I[a0]\n    x = load X[i % 1, i // 1]\n    O[a0] = x",
                "load X[0, in0]",
            ),
        ];
        for (program, expected) in cases {
            let listing = normal(program);
            assert!(listing.contains(expected), "{program}\n--\n{listing}");
        }
    }

    /// A whole-number coefficient or offset is an integer only when no float literal fed the sum: a float literal
    /// keeps its spelling, since it is what makes an operation over integer operands a float one, and a sum that
    /// would spell none keeps one on its first term.
    #[test]
    fn the_affine_form_keeps_a_float_literal_over_integer_operands() {
        let listing = |program: &str| text(&normalize_uncached(&body(program)).unwrap());
        let doubled = listing(
            "for a0 in 0..8:
    u8 code = load x[a0]
    v = code + code
    out[a0] = v",
        );
        assert!(doubled.contains("= 2 * in0"), "{doubled}");
        let scaled = listing(
            "for a0 in 0..8:
    u8 code = load x[a0]
    v = multiply(2.0, code)
    out[a0] = v",
        );
        assert!(scaled.contains("= 2.0 * in0"), "{scaled}");
        let shifted = listing(
            "for a0 in 0..8:
    u8 code = load x[a0]
    f32 eight = 8.0
    v = subtract(code, eight)
    out[a0] = v",
        );
        assert!(shifted.contains("= in0 - 8.0"), "{shifted}");
        let collapsed = listing(
            "for a0 in 0..8:
    u8 code = load x[a0]
    v = multiply(code, 1.0)
    w = multiply(v, v)
    out[a0] = w",
        );
        assert!(
            collapsed.contains("v0 = 1.0 * in0\n") && collapsed.contains("v1 = v0 * v0"),
            "{collapsed}"
        );
        let cancelled = listing(
            "for a0 in 0..8:
    u8 code = load x[a0]
    v = subtract(code, 8.0)
    w = add(v, 8.0)
    out[a0] = w",
        );
        assert!(cancelled.contains("= 1.0 * in0"), "{cancelled}");
        let mixed = listing(
            "for a0 in 0..8:
    u8 code = load x[a0]
    u8 other = load y[a0]
    v = add(code, code)
    w = multiply(2.0, other)
    o = add(v, w)
    out[a0] = o",
        );
        assert!(
            mixed.contains("2.0 * in0") && mixed.contains("2.0 * in1"),
            "{mixed}"
        );
    }

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
