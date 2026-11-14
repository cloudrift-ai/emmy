//! The canonical order: sequential names, a dependency-valid statement order with commutative operand order from the
//! value numbering, and integer expressions canonicalized by lexical binding order.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::ties;
use super::{refilled, unify_siblings, with_bodies};
use crate::body::{Body, free_names, unique};
use crate::elementwise::{AffineRole, Operation};
use crate::expr::{Expr, ExprClass, Expression, LitType};
use crate::field::Rewriter;
use crate::numbering::{Chains, Key, Numbering, roles, statement_tree, value_numbers};
use crate::order::ordering_constraints_within;
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
    counts: HashMap<Name, usize>,
    reserved: HashSet<Name>,
    /// The spellings taken so far where binders keep their names ([`distinct_names`]); `None` names sequentially.
    kept: Option<HashSet<Name>>,
}

impl Counters {
    /// The name of a new binder spelled `old`: the next sequential name under `prefix`, or, where binders keep their
    /// names, `old` itself for the first binder of the spelling and `old_1`, `old_2`, … for the later ones.
    fn fresh(&mut self, old: &Name, prefix: &'static str) -> Name {
        let Some(taken) = &mut self.kept else {
            loop {
                let count = self.counts.entry(Name::from(prefix)).or_insert(0);
                let name: Name = format!("{prefix}{count}").into();
                *count += 1;
                if !self.reserved.contains(&name) {
                    return name;
                }
            }
        };
        if !self.reserved.contains(old) && taken.insert(*old) {
            return *old;
        }
        loop {
            let count = self.counts.entry(*old).or_insert(0);
            *count += 1;
            let name: Name = format!("{old}_{count}").into();
            if !self.reserved.contains(&name) && taken.insert(name) {
                return name;
            }
        }
    }

    /// The name of an axis a statement binds: sequential, or, where binders keep their names, its own unless a
    /// statement around it or a value in scope holds the spelling. Sibling loops over one axis keep the one name.
    fn fresh_axis(&mut self, old: &Name, shadows: bool) -> Name {
        match &mut self.kept {
            Some(taken) if !shadows && !self.reserved.contains(old) => {
                taken.insert(*old);
                *old
            }
            _ => self.fresh(old, "a"),
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
        self.ssa.insert(*old, counters.fresh(old, prefix));
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
            let scalar = child.escaping_states();
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
            let updates = stmt.updates();
            for name in stmt.defines() {
                // Storage updated in place keeps the name it has in scope: its field reads and defines it. Where
                // binders keep their names, so does storage no statement in scope declares (one element of a
                // fragment family a loop indexes): an update binds nothing.
                if updates.contains(&name)
                    && (self.ssa.contains_key(&name) || counters.kept.is_some())
                {
                    continue;
                }
                if stmt.is_pure() && self.owned.contains(&name) {
                    self.ssa
                        .insert(name, counters.fresh(&name, ssa_prefix(stmt)));
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
            let shadows = axes.contains_key(&old) || self.ssa.values().any(|name| *name == old);
            axes.insert(old, counters.fresh_axis(&old, shadows));
        }
        for axis in stmt.bound_axes() {
            for source in axis.sources() {
                self.sources
                    .entry(source.name)
                    .or_insert_with(|| counters.fresh(&source.name, "p"));
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
        refilled(stmt, &renamed, renamed_children)
    }
}

/// Canonicalize names in a fused body: axes to `a0, a1, …`, window parents to `p0, …`, loads to `in0, …`,
/// accumulators to `acc0, …`, every other definition to `v0, …`, in definition order. Each nested body is a lexical
/// scope, while its assigned names remain globally unique.
pub fn rename_ssa_sequential(body: &Body) -> Body {
    renamed(body, None)
}

/// `body` with every binding under a name of its own, the names as they stand wherever they already are: the first
/// binder of a spelling keeps it and a later one takes it with a number. Motion needs it (two lexical bindings of one
/// name can land in one scope) and it is all the naming a body no identity reads takes.
pub fn distinct_names(body: &Body) -> Body {
    renamed(body, Some(HashSet::new()))
}

fn renamed(body: &Body, kept: Option<HashSet<Name>>) -> Body {
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
        kept,
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
    ordered_within(body, numbering, &HashSet::new())
}

/// [`ordered`] for a scope inside one that binds `outer` (`ordering_constraints_within`).
fn ordered_within(body: &Body, numbering: &Numbering, outer: &HashSet<Name>) -> Result<Body> {
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
    let incoming = ordering_constraints_within(&rebuilt, true, outer);
    let order =
        rebuilt.topological_permutation(&incoming, Some(&|index| priorities[index].clone()))?;
    Ok(rebuilt.rebuilt(
        order
            .into_iter()
            .map(|index| rebuilt.stmts()[index].clone())
            .collect(),
    ))
}

/// A fragment's statements in the canonical order, its names as they stand: dependency-valid, then as the canonical
/// order ranks siblings. Nothing is renamed or dropped, so a pure fragment with named results (a lambda's body) keeps
/// them; its buffers are keyed by name, since a fragment compares its free names as spelled. `outer` are the names
/// the fragment's reader binds (a lambda's parameters): a read of one ahead of its first binding in the fragment reads
/// the outer one. A body whose statements wait on each other has no order, and is an error.
pub fn fragment_order(body: &Body, outer: &HashSet<Name>) -> Result<Body> {
    if body.len() < 2 {
        return Ok(body.clone());
    }
    let numbering = value_numbers(body, &crate::numbering::spelled_key, Chains::Flat);
    ordered_within(body, &numbering, outer)
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
/// literal offset last. The reassociation of a sum is the exactness every reduce schedule already takes; a literal
/// divisor becomes its reciprocal only where that is exact, since a division's rounding is the division knob's.
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

/// The terms of an affine tree, `coefficient * operand` each, and its literal offset. A quotient folds into a
/// reciprocal scale only by a float literal whose reciprocal is exact (`affine_node`); any other stays a term.
pub struct AffineTerms {
    pub terms: Vec<(f64, Expression)>,
    pub offset: f64,
    /// Whether a float literal fed a coefficient or the offset: then every whole-number coefficient and offset the
    /// sum spells stays a float literal, and the operation keeps the float dtype the literal gave it.
    pub float: bool,
}

/// Whether an operation node is a node of the affine form in `class`'s coefficient domain: a sum, a difference, a
/// negation, a product with a literal factor, and for a value a quotient by a float literal whose reciprocal is exact
/// (a power of two). An `Index` folds integer literals alone and no quotient, since integer division floors; any
/// other quotient by a literal keeps its own rounding, which the division knob decides, and a product of two values
/// or a quotient by a value is an operation like any other.
pub fn affine_node(op: Operation, args: &[Expression], class: ExprClass) -> bool {
    let literal = |index: usize| {
        matches!(args.get(index).map(|a| &**a), Some(Expr::Literal(lit))
            if lit.dtype != LitType::Bool && (class == ExprClass::Value || lit.dtype == LitType::Int))
    };
    match op.info().affine {
        Some(AffineRole::Sum | AffineRole::Difference | AffineRole::Negation) => true,
        Some(AffineRole::Scale) => literal(0) || literal(1),
        Some(AffineRole::Quotient) => {
            class == ExprClass::Value && args.get(1).is_some_and(exact_reciprocal)
        }
        None => false,
    }
}

/// Whether `divisor` is a float literal whose reciprocal is exact and normal in `f32`: `±2^k` for `|k| <= 126`.
fn exact_reciprocal(divisor: &Expression) -> bool {
    let Expr::Literal(lit) = &**divisor else {
        return false;
    };
    let value = lit.value.as_f64();
    lit.dtype == LitType::Float
        && value != 0.0
        && value.is_finite()
        && value.abs().log2().fract() == 0.0
        && value.abs().log2().abs() <= 126.0
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
                    if let Expr::Literal(lit) = &**b {
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
    use crate::normalize::{normalize_body, normalize_uncached, optimize_uncached};
    use crate::statement::Load;

    /// The names of `given`'s statements in the fragment order, `outer` bound by its reader.
    fn fragment_names(given: &str, outer: &[&str]) -> Result<Vec<String>> {
        let outer: HashSet<Name> = outer.iter().map(|name| Name::from(*name)).collect();
        Ok(fragment_order(&body(given), &outer)?
            .stmts()
            .iter()
            .map(|stmt| stmt.defines()[0].to_string())
            .collect())
    }

    #[test]
    fn a_fragment_orders_alike_whatever_order_it_was_written_in() {
        let loads = "a = load x[k]\nb = load w[k]\nc = load y[k]";
        let one = fragment_names(
            &format!("{loads}\nab = multiply(a, b)\nbc = multiply(b, c)"),
            &[],
        )
        .unwrap();
        let two = fragment_names(
            &format!("{loads}\nbc = multiply(b, c)\nab = multiply(a, b)"),
            &[],
        )
        .unwrap();
        assert_eq!(one, two);
    }

    #[test]
    fn a_fragment_reads_an_outer_name_it_rebinds_ahead_of_the_rebinding() {
        let given = "e = subtract(d, m)\nm = add(m, e)";
        assert_eq!(fragment_names(given, &["m", "d"]).unwrap(), ["e", "m"]);
        assert!(
            fragment_names(given, &[]).is_err(),
            "inside an order-free scope the read is the rebinding's"
        );
    }

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
    /// keeps a quotient a term, a value folds any literal and a quotient by a power-of-two float literal into its
    /// reciprocal.
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

    /// A cross-thread combine updates its states in place: a read behind it reads the combined value, and stays
    /// behind it.
    #[test]
    fn a_read_of_a_state_stays_behind_the_combine_that_updates_it() {
        let normal = text(
            &normalize_uncached(&body(
                "
                Tile(axes=(m in 0..8, c in 0..128), block_threads=128):
                    init f32 s = 0
                    f32 x = load X[m, c]
                    f32 q = multiply(x, x)
                    f32 s <- add(s, q) over c
                    WarpShuffle(state=(s), state_b=(s__o), length=32):
                        combine_states:
                            s = s + s__o
                    Smem(name=s_smem, extents=(4))
                    if lane == 0:
                        f32 s_smem[warp] = s
                    Sync
                    TreeHalve(bufs=(s_smem), state=(s), state_b=(s__o), length=4, tid_var=warp):
                        combine_states:
                            s = s + s__o
                    f32 r = rsqrt(s)
                    f32 y = multiply(r, x)
                    f32 O[m, c] = y
                ",
            ))
            .unwrap(),
        );
        let at = |what: &str| {
            normal
                .find(what)
                .unwrap_or_else(|| panic!("{what}\n{normal}"))
        };
        assert!(at("TreeHalve") < at("rsqrt"), "{normal}");
        assert!(at("WarpShuffle") < at("Smem"), "{normal}");
    }

    /// A register fragment is storage: the statements that update it and the statements that read it name it alike,
    /// in whatever scope they sit.
    #[test]
    fn a_fragment_keeps_one_name_through_its_updates() {
        let normal = text(
            &optimize_uncached(&body(
                "
                Tile(axes=(m in 0..2, n in 0..16, _lane in 0..32), block_threads=32):
                    RegFragment(name=_a0, role=a, dtype=f16)
                    RegFragment(name=_b0, role=b, dtype=f16)
                    RegFragment(name=_c0, role=c, dtype=f32)
                    for k in 0..128 step 16:
                        LdmatrixLoad(frag=_a0, src_buffer=A, src_index=(16 * m, k), role=a, ldm=128)
                        LdmatrixLoad(frag=_b0, src_buffer=B, src_index=(k, 8 * n), role=b, ldm=8)
                        MmaSyncPtx(c_frag=_c0, a_frag=_a0, b_frag=_b0, shape=(16, 8, 16))
                    RegStore(dst_buffer=O, dst_index=(16 * m, 8 * n), frag=_c0, shape=(16, 8, 16), ldm=128, ldn=1, row_dim=0, col_dim=1)
                ",
            ))
            .unwrap(),
        );
        let field = |line: &str, key: &str| -> String {
            let line = normal
                .lines()
                .find(|l| l.contains(line))
                .unwrap_or_else(|| panic!("{line}\n{normal}"));
            let rest = &line[line.find(key).unwrap() + key.len()..];
            rest[..rest.find([',', ')']).unwrap()].to_string()
        };
        assert_eq!(
            field("role=a, dtype", "name="),
            field("MmaSyncPtx", "a_frag="),
            "{normal}"
        );
        assert_eq!(
            field("role=b, dtype", "name="),
            field("MmaSyncPtx", "b_frag="),
            "{normal}"
        );
        assert_eq!(
            field("role=c, dtype", "name="),
            field("RegStore", "frag="),
            "{normal}"
        );
        assert_eq!(
            field("role=a, ldm", "frag="),
            field("MmaSyncPtx", "a_frag="),
            "{normal}"
        );
    }

    /// The text of the form a kernel body takes: the optimizing half of the normal form, and final.
    fn kernel_normal(given: &str) -> String {
        let optimized = optimize_uncached(&body(given)).unwrap();
        assert_eq!(optimize_uncached(&optimized).unwrap(), optimized);
        text(&optimized)
    }

    /// The optimized form moves a value and keeps what it does not move: the statements stand in the order their
    /// builder wrote them and under its names.
    #[test]
    fn an_optimized_body_keeps_its_names_and_its_order() {
        let out = kernel_normal(
            "
            Tile(axes=(m in 0..4, _lane in 0..32)):
                for k in 0..16:
                    _scale = load S[m]
                    zeta = load T[m, k]
                    alpha = load U[m, k]
                    prod = multiply(zeta, alpha)
                    out = multiply(prod, _scale)
                    O[m, k] = out
            ",
        );
        let at = |needle: &str| {
            out.find(needle)
                .unwrap_or_else(|| panic!("{needle}\n{out}"))
        };
        assert!(at("_scale = load S[m]") < at("for k in"), "{out}");
        assert!(at("zeta = load T") < at("alpha = load U"), "{out}");
        assert!(at("alpha = load U") < at("prod = "), "{out}");
    }

    /// Two sibling loops each bind `x`; motion puts both bindings in the scope around them, where the later one
    /// takes a number. A body whose names are already distinct is left alone.
    #[test]
    fn a_later_binder_of_a_spelling_takes_a_number() {
        let given = body(
            "
            Tile(axes=(m in 0..4)):
                for k in 0..16:
                    x = load X[m]
                    t = load T[m, k]
                    u = multiply(x, t)
                    O[m, k] = u
                for k in 0..16:
                    x = load Y[m]
                    t = load T[m, k]
                    u = add(x, t)
                    P[m, k] = u
            ",
        );
        let distinct = distinct_names(&given);
        let out = text(&distinct);
        for name in [
            "x = load X",
            "x_1 = load Y",
            "t_1 = load T",
            "P[m, k] = u_1",
        ] {
            assert!(out.contains(name), "{name}\n{out}");
        }
        assert_eq!(
            out.matches("for k in").count(),
            2,
            "sibling loops keep the one axis name\n{out}"
        );
        assert_eq!(distinct_names(&distinct), distinct);
        let optimized = kernel_normal(&text(&given));
        let at = |needle: &str| {
            optimized
                .find(needle)
                .unwrap_or_else(|| panic!("{needle}\n{optimized}"))
        };
        assert!(at("x = load X[m]") < at("for k in"), "{optimized}");
        assert!(
            at("x_1 = load Y[m]") < optimized.rfind("for k in").unwrap(),
            "{optimized}"
        );
    }

    #[test]
    fn a_value_two_cells_of_a_tile_compute_is_computed_once() {
        let out = kernel_normal(
            "
            Tile(axes=(m in 0..4, n in 0..8)):
                x0 = load X[m]
                y0 = multiply(x0, 2.0)
                x1 = load X[m]
                y1 = multiply(x1, 2.0)
                O[m, 2 * n] = y0
                O[m, 2 * n + 1] = y1
            ",
        );
        assert_eq!(out.matches("load X").count(), 1, "{out}");
        assert_eq!(out.matches("2.0 * ").count(), 1, "{out}");
    }

    #[test]
    fn an_index_simplifies_under_the_tile_axes() {
        let out = kernel_normal(
            "
            Tile(axes=(m in 0..4, _lane in 0..32)):
                x = load X[(32 * m + _lane) / 32, (32 * m + _lane) % 32]
                O[m, _lane] = x
            ",
        );
        assert!(out.contains("load X[m, _lane]"), "{out}");
    }

    #[test]
    fn a_barrier_stays_where_it_is() {
        let out = kernel_normal(
            "
            Tile(axes=(m in 0..4)):
                x = load X[m]
                Sync
                y = load X[m]
                O[m] = y
                P[m] = x
            ",
        );
        assert_eq!(out.matches("Sync").count(), 1, "{out}");
    }

    /// Two accumulators of one value are congruent, but a combine that updates each in place after the loop holds
    /// them as two states: the merge leaves both.
    #[test]
    fn a_state_a_combine_updates_in_place_stays() {
        let out = kernel_normal(
            "
            Tile(axes=(m in 0..4, _lane in 0..32)):
                for k in 0..16:
                    x = load X[m, k]
                    a <- add(a, x) over k
                    y = load X[m, k]
                    b <- add(b, y) over k
                WarpShuffle(state=(a, b), state_b=(a__o, b__o), length=32):
                    combine_states:
                        a = a + a__o
                        b = b + b__o
                O[m, 0] = a
                O[m, 1] = b
            ",
        );
        assert_eq!(out.matches(" <- ").count(), 2, "{out}");
    }

    /// A tree combine reads the coordinates that index its slab: they follow the axes they name.
    #[test]
    fn a_tree_combine_follows_the_axes_that_index_its_slab() {
        let normal = text(
            &normalize_uncached(&body(
                "
                Tile(axes=(b in 0..2, k_co in 0..4, n_ln in 0..32), block_threads=128):
                    for k in k_co..256 step 4:
                        f32 x = load X[32 * b + n_ln, k]
                        f32 s <- add(s, x) over k
                    Smem(name=s_smem, extents=(128))
                    f32 s_smem[k_co * 32 + n_ln] = s
                    Sync
                    TreeHalve(bufs=(s_smem), state=(s), state_b=(s__o), length=4, tid_var=k_co, inner=(n_ln, 32)):
                        combine_states:
                            s = s + s__o
                    if k_co == 0:
                        f32 O[32 * b + n_ln] = s
                ",
            ))
            .unwrap(),
        );
        assert!(normal.contains("tid_var=a1, inner=(a2, 32)"), "{normal}");
    }

    /// A block-scale load fills a declared fragment, and a pointwise rewrite in place rewrites one: both keep its
    /// name.
    #[test]
    fn a_fragment_keeps_one_name_through_a_scale_load_and_a_rewrite_in_place() {
        let normal = text(
            &normalize_uncached(&body(
                "
                Tile(axes=(m in 0..2, n in 0..16, _lane in 0..32), block_threads=32):
                    RegFragment(name=_sf, role=sfa, dtype=f4e2m1x2)
                    RegFragment(name=_c, role=c, dtype=f32)
                    f32 w = load W[m]
                    for k in 0..128 step 64:
                        BlockScaleLoad(frag=_sf, src_buffer=S, src_index=(16 * m, k / 16), role=a, ldm=16)
                        FragmentApply(out=_c, op=multiply, args=(FragmentArg(kind=frag, names=(_c)), FragmentArg(kind=uniform, names=(w))), in_place=true)
                    RegStore(dst_buffer=O, dst_index=(16 * m, 8 * n), frag=_c, shape=(16, 8, 16), ldm=128, ldn=1, row_dim=0, col_dim=1)
                ",
            ))
            .unwrap(),
        );
        let named = |line: &str, key: &str| -> String {
            let line = normal
                .lines()
                .find(|l| l.contains(line))
                .unwrap_or_else(|| panic!("{line}\n{normal}"));
            let rest = &line[line.find(key).unwrap() + key.len()..];
            rest[..rest.find([',', ')']).unwrap()].to_string()
        };
        assert_eq!(
            named("role=sfa", "name="),
            named("BlockScaleLoad", "frag="),
            "{normal}"
        );
        assert_eq!(
            named("dtype=f32", "name="),
            named("FragmentApply", "out="),
            "{normal}"
        );
        assert_eq!(
            named("dtype=f32", "name="),
            named("RegStore", "frag="),
            "{normal}"
        );
    }

    /// A staged copy re-assigns, inside the chunk loop, the registers a load declared ahead of it: the three keep
    /// one set of names.
    #[test]
    fn a_staged_copy_keeps_the_names_its_first_load_declared() {
        let normal = text(
            &normalize_uncached(&body(
                "
                Tile(axes=(m in 0..4), block_threads=32):
                    Smem(name=_s, extents=(2, 2))
                    f32 p, q = load X[m, 0]
                    for k in 0..8 step 2:
                        f32 p, q = load X[0] carried=unpack
                        f32 _s[k / 2 % 2, 0] = (p, q)
                        Sync
                        f32 p, q = load X[m, k + 2 < 8 ? k + 2 : 0] carried=load
                ",
            ))
            .unwrap(),
        );
        let names: Vec<&str> = normal
            .lines()
            .filter(|line| line.contains("= load X"))
            .map(|line| line.trim().split(" = ").next().unwrap())
            .collect();
        assert_eq!(names.len(), 3, "{normal}");
        assert!(names.iter().all(|name| *name == names[0]), "{normal}");
    }

    /// A read of a state ahead of every update of it in its scope reads the value the scope began with, and stays
    /// ahead of them, a combine's in-place update among them.
    #[test]
    fn a_read_of_the_previous_state_stays_ahead_of_its_updates() {
        let normal = text(
            &normalize_uncached(&body(
                "
                Tile(axes=(m in 0..9, c in 0..32), block_threads=128):
                    init f32 s = 0
                    for chunk in 0..64 step 32 unroll=true seed=false:
                        f32 previous = s
                        f32 x = load X[m, chunk + c]
                        init f32 zero = 0
                        f32 s <- add(zero, x) over chunk
                        WarpShuffle(state=(s), state_b=(s__o), length=32, prefix=true):
                            combine_states:
                                s = s + s__o
                        f32 s <- add(s, previous)
                        f32 O[m, chunk + c] = s
                        WarpBroadcast(state=(s), length=32)
                ",
            ))
            .unwrap(),
        );
        let lines: Vec<&str> = normal.lines().map(str::trim).collect();
        let at = |what: &str| {
            lines
                .iter()
                .position(|l| l.contains(what))
                .unwrap_or_else(|| panic!("{what}\n{normal}"))
        };
        assert!(at("for ") < at("= acc0"), "{normal}");
        assert!(at("= acc0") < at("acc0 <- add"), "{normal}");
    }
}
