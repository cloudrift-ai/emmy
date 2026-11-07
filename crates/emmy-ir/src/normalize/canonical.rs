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

/// Dependency-valid statement order, commutative operand order and names, all from the value numbering.
/// The digest the Python order keyed a loop's extent by: SHA-256 of the Python `repr` of the extent's structural
/// form in its old spelling, `(('Literal', 4096, 'int'),)`. Any stable total order over extents would do; this is the
/// one the stored goldens' decisions are addressed in.
pub(crate) fn extent_digest(expression: &Expression) -> String {
    use sha2::Digest;

    fn form(expression: &Expression) -> String {
        fn tuple(items: &[Expression]) -> String {
            match items {
                [one] => format!("({},)", form(one)),
                _ => format!(
                    "({})",
                    items.iter().map(form).collect::<Vec<_>>().join(", ")
                ),
            }
        }
        match &**expression {
            Expr::Var(name) => format!("('Var', {})", python_repr(name)),
            Expr::Literal(lit) => {
                let value = match lit.value {
                    Num::Int(v) => v.to_string(),
                    Num::Float(v) => crate::expr::float_repr(v),
                    Num::Bool(v) => if v { "True" } else { "False" }.to_string(),
                };
                format!("('Literal', {value}, {})", python_repr(&lit.dtype))
            }
            Expr::Builtin(name) => format!("('Builtin', {})", python_repr(name)),
            Expr::Binary(op, left, right) => {
                format!(
                    "('BinaryExpr', {}, {}, {})",
                    python_repr(op.text()),
                    form(left),
                    form(right)
                )
            }
            Expr::Call(name, args) => {
                format!("('FuncCallExpr', {}, {})", python_repr(name), tuple(args))
            }
            Expr::Ternary(cond, yes, no) => format!(
                "('TernaryExpr', {}, {}, {})",
                form(cond),
                form(yes),
                form(no)
            ),
            Expr::Cast(dtype, inner) => {
                format!("('CastExpr', {}, {})", python_repr(dtype), form(inner))
            }
            Expr::FlatIndex(buffer, index) => {
                format!("('FlatIndex', {}, {})", python_repr(buffer), tuple(index))
            }
        }
    }
    let digest = sha2::Sha256::digest(format!("({},)", form(expression)).as_bytes());
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// A header field's text as an order key. A tuple of one element sorts after every longer tuple, as the Python order
/// had it from the spelling of a one-element tuple, so a one-coordinate load follows the two-coordinate loads beside
/// it; everything else compares as text.
fn field_key(text: &str) -> Key {
    if text.starts_with('(') && text.ends_with(')') {
        let mut depth = 0i32;
        let mut top_level_comma = false;
        for ch in text.chars() {
            match ch {
                '(' | '[' => depth += 1,
                ')' | ']' => depth -= 1,
                ',' if depth == 1 => top_level_comma = true,
                _ => {}
            }
        }
        return Key::tuple([Key::Int(i64::from(!top_level_comma)), Key::str(text)]);
    }
    Key::str(text)
}

/// How many statements of its own scope each value is computed through: a leaf counts itself and, once each, the
/// leaves of this scope it reads through any chain; a name defined in a nested scope is a leaf's read, not a
/// statement of this one. The order's depth-first tie among computed operands.
pub(crate) fn cone_sizes(body: &Body, out: &mut HashMap<Name, usize>) {
    let mut sets: Vec<HashSet<usize>> = Vec::with_capacity(body.len());
    let mut defined: HashMap<Name, usize> = HashMap::new();
    for (index, stmt) in body.stmts().iter().enumerate() {
        let mut set: HashSet<usize> = HashSet::from([index]);
        if stmt.nested().is_empty() {
            for name in stmt.deps() {
                if let Some(&at) = defined.get(&name) {
                    set.extend(sets[at].iter().copied());
                }
            }
            for name in stmt.defines() {
                defined.insert(name.clone(), index);
                out.insert(name, set.len());
            }
        } else {
            for child in stmt.nested() {
                cone_sizes(&child, out);
            }
        }
        sets.push(set);
    }
}

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

    // A statement's own shape with every name and buffer one placeholder: its kind, then its fields in declaration
    // order. Among the leaves a scope may place in either order, kinds sort by name, which puts every store after the
    // values it reads, and one kind's statements by their first differing field, so an `Assign`'s operation decides
    // before its dtype. Sibling loops order by the digest of their extent, the order the stored goldens were spelled
    // in: the lift reads sibling regions in this order, so the order is part of every recorded decision's address.
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
        let extent = match &*shell {
            Statement::Loop(l) => Some(&l.axis.extent.expression),
            Statement::StridedLoop(l) => Some(&l.axis.extent.expression),
            _ => None,
        };
        let fields = shell
            .header_fields()
            .map(|fields| {
                fields
                    .into_iter()
                    .map(|(name, text)| match extent {
                        Some(expression) if name == "axis" => {
                            Key::Str(extent_digest(expression).into())
                        }
                        _ => field_key(&text),
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|_| vec![Key::str(&structural_text(&shell))]);
        Key::tuple([Key::str(stmt.kind_name()), Key::Tuple(fields)])
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
    // The operands of a commutative operation list accumulators, then loads and the copies that widen them — the
    // lift reads a product's slab operands ahead of its computed ones, and a slab is a load or a cast of one — then
    // computed values, the deeper cone first: the statements of this scope a value is computed through, so an
    // activation chain precedes the dequantized weight it multiplies and attention's probabilities precede the copy
    // of the values they weight. Then by value, and two operands of one value by the names the order just gave them.
    let loaded: HashSet<Name> = result
        .iter_stmts()
        .iter()
        .filter(|stmt| matches!(&***stmt, Statement::Load(_)))
        .flat_map(|stmt| stmt.defines())
        .collect();
    let mut cone = HashMap::new();
    cone_sizes(&result, &mut cone);
    let mut operand: HashMap<Name, Key> = HashMap::new();
    for (before, after) in placed.iter_stmts().iter().zip(result.iter_stmts()) {
        if let Some(ordinal) = numbering.ordinal_of(before)
            && let Some((number, params)) = &numbering.numbers[ordinal]
        {
            let rank = match &*after {
                Statement::Accum(_) | Statement::Init(_) => 0,
                Statement::Load(_) => 1,
                Statement::Assign(a)
                    if &*a.op.name == "copy"
                        && a.args.len() == 1
                        && loaded.contains(&a.args[0]) =>
                {
                    1
                }
                _ => 2,
            };
            for name in after.defines() {
                let depth = cone.get(&name).copied().unwrap_or(0) as i64;
                operand.insert(
                    name,
                    Key::tuple([
                        Key::Int(rank),
                        Key::Int(-depth),
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
