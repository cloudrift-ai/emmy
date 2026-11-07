//! The canonical order: sequential names, a dependency-valid statement order with commutative operand order from the
//! value numbering, and integer expressions canonicalized by lexical binding order.

use super::*;

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
pub(crate) struct SequentialScope {
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
            for here in child.of_kinds(&[Kind::Accum, Kind::Carry]) {
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
pub(crate) fn unlisted(stmt: &Arc<Statement>) -> Arc<Statement> {
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

/// `body` with every scope's members in the canonical order: dependency-valid, then by the sibling's category, its
/// shape, its spelling where shapes tie, and its number or scope tree.
fn ordered(body: &Body, numbering: &Numbering) -> Body {
    let rebuilt = body.map_each(&mut |stmt| {
        let nested = stmt.nested();
        if nested.is_empty() {
            return stmt.clone();
        }
        with_bodies(
            stmt,
            nested
                .iter()
                .map(|child| ordered(child, numbering))
                .collect(),
        )
    });
    let shapes: Vec<Key> = body.stmts().iter().map(ties::sibling_shape).collect();
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
            let category = if !children.is_empty() && stmt.has_side_effects() {
                2
            } else {
                children.is_empty() as i64
            };
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
    Body::new(
        order
            .into_iter()
            .map(|index| rebuilt.stmts()[index].clone())
            .collect(),
    )
}

pub fn canonical_order(body: &Body) -> Result<Body> {
    let stmts = canonicalize_expressions(body, &[])?;
    let role = roles(&stmts, None);
    let numbering = value_numbers(&stmts, &|name| {
        role.get(name)
            .map_or_else(|| Key::Str(name.clone()), |r| Key::Int(*r as i64))
    });
    let placed = ordered(&stmts, &numbering);
    let result = rename_ssa_sequential(&placed);
    let operand = ties::operand_ranks(&placed, &result, &numbering);
    let operand_key = |name: &Name| operand.get(name).cloned().unwrap_or_else(ties::unranked);
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

/// Whether `op`'s operands may trade places.
fn commutative(op: BinOp) -> bool {
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
}

/// An expression's form with every bound axis spelled by its binding position: the order of two operands.
fn unspelled(e: &Expression, axis_order: &HashMap<Name, usize>) -> Key {
    match &**e {
        Expr::Var(name) => match axis_order.get(name) {
            Some(index) => Key::tuple([Key::str("Var"), Key::Int(*index as i64)]),
            None => Key::tuple([Key::str("Var"), Key::Str(name.clone())]),
        },
        Expr::Literal(Lit { value, dtype }) => Key::tuple([
            Key::str("Literal"),
            Key::num(*value),
            Key::str(dtype.name()),
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

/// A sum over bound axes with its terms in binding order and its anchor last.
fn affine(expression: Expression, axis_order: &HashMap<Name, usize>) -> Result<Expression> {
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
}

/// `e` canonicalized: comparisons as `<` and `<=`, commutative operands ordered, sums in affine form.
fn rewrite(e: &Expression, axis_order: &HashMap<Name, usize>) -> Result<Expression> {
    Ok(match &**e {
        Expr::Binary(op, left, right) => {
            let (mut left, mut right) = (rewrite(left, axis_order)?, rewrite(right, axis_order)?);
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
                affine(result, axis_order)?
            } else {
                result
            }
        }
        Expr::Call(name, args) => {
            let args: Result<Vec<Expression>> =
                args.iter().map(|a| rewrite(a, axis_order)).collect();
            Expression::new(Expr::Call(name.clone(), args?.into()))
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
pub fn canonicalize_expressions(body: &Body, axes: &[Name]) -> Result<Body> {
    let axis_order: HashMap<Name, usize> = axes
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), i))
        .collect();
    let mut failure: Option<Error> = None;
    let out: Vec<Arc<Statement>> = body
        .stmts()
        .iter()
        .map(|stmt| {
            let rewritten =
                Statement::map_expressions(stmt, &mut |e| match rewrite(e, &axis_order) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinds::Load;
    use crate::normalize::testing::*;

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
                    let [ref initial] = renamed.of_kinds(&[Kind::Init])[..] else {
                        panic!("one init")
                    };
                    let [ref update] = renamed.of_kinds(&[Kind::Accum])[..] else {
                        panic!("one update")
                    };
                    let [ref output] = renamed.of_kinds(&[Kind::Write])[..] else {
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
}
