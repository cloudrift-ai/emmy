//! Where the canonical order breaks a tie the digest leaves open, and why each rule is the one it is.
//!
//! The order among siblings a scope may place either way, and the operand order of a commutative operation, reach no
//! identity, but they reach the lift: sibling regions are addressed by position in every recorded cut decision, and a
//! product's operand order is its orientation where no output axis decides. So the ties are broken the way the stored
//! goldens were spelled, and each rule below states what it reproduces. Changing one is a restamp decision.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::with_bodies;
use crate::Name;
use crate::Result;
use crate::body::Body;
use crate::expr::{BinOp, Expr, ExprType, Expression, Num, python_repr};
use crate::field::{FieldType, Float};
use crate::numbering::{Key, Numbering, expression_key};
use crate::statement::{Affine, Statement};

/// A statement's own shape with every name and buffer one placeholder: its kind, then its fields in declaration order.
/// Kinds sort by name, which puts every store after the values it reads, as the Python order's structural form had
/// it, and one kind's statements sort by their first differing field, so an `Assign`'s operation decides before its
/// dtype. A loop's axis field is its extent's digest ([`extent_digest`]).
pub(crate) fn sibling_shape(stmt: &Arc<Statement>) -> Result<Key> {
    let children = stmt.nested();
    let shell = if children.is_empty() {
        stmt.clone()
    } else {
        with_bodies(stmt, children.iter().map(|_| Body::empty()).collect())
    };
    let renamed = Statement::rename_with(&shell, &|_| Some(Name::from("__slot__")));
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
        .header_fields()?
        .into_iter()
        .map(|(name, text)| match extent {
            Some(expression) if name == "axis" => Key::Str(extent_digest(expression).into()),
            _ => field_key(shell.ty().field_ty(name), &text),
        })
        .collect::<Vec<_>>();
    Ok(Key::tuple([Key::str(stmt.ty().name()), Key::Tuple(fields)]))
}

/// The whole subtree with names abstracted and buffers spelled: the tie-break between siblings of one shape, so
/// symmetric siblings order by the buffers they touch.
pub(crate) fn sibling_spelling(stmt: &Arc<Statement>) -> Key {
    let renamed = Statement::rename_with(stmt, &|_| Some(Name::from("__name__")));
    let body = Body::new(vec![renamed]).map_stmts(&mut |s| Some(unlisted(s)));
    Key::str(&crate::text::structural_body_text(&body))
}

/// A header field's key: its text, and for a sequence field whether it holds one element, which sorts after longer
/// sequences, as the Python order's text of a one-element tuple did.
fn field_key(ty: Option<FieldType>, text: &str) -> Key {
    if matches!(ty, Some(FieldType::Seq(_) | FieldType::Tuple(_)))
        && text.starts_with('(')
        && text.ends_with(')')
    {
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
                format!("('Literal', {value}, {})", python_repr(lit.dtype.text()))
            }
            Expr::Builtin(builtin) => format!("('Builtin', {})", python_repr(builtin.text())),
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
                format!(
                    "('CastExpr', {}, {})",
                    python_repr(&dtype.to_string()),
                    form(inner)
                )
            }
            Expr::FlatIndex(buffer, index) => {
                format!("('FlatIndex', {}, {})", python_repr(buffer), tuple(index))
            }
        }
    }
    let digest = sha2::Sha256::digest(format!("({},)", form(expression)).as_bytes());
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
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

/// The key a commutative operation's operands sort by: accumulators, then loads and the copies that widen them — the
/// lift reads a product's slab operands ahead of its computed ones, and a slab is a load or a cast of one — then
/// computed values, the deeper cone first ([`cone_sizes`]): an activation chain precedes the dequantized weight it
/// multiplies and attention's probabilities precede the copy of the values they weight. Then by value, and two
/// operands of one value by the names the order gave them.
pub(crate) fn operand_ranks(
    placed: &Body,
    result: &Body,
    numbering: &Numbering,
) -> HashMap<Name, Key> {
    let loaded: HashSet<Name> = result
        .iter_stmts()
        .iter()
        .filter(|stmt| matches!(&***stmt, Statement::Load(_)))
        .flat_map(|stmt| stmt.defines())
        .collect();
    let mut cone = HashMap::new();
    cone_sizes(result, &mut cone);
    // The order permuted the numbered body's statements without rebuilding them, so a placed statement is the
    // numbered one it shares its object with, while both bodies live.
    let numbered: HashMap<*const Statement, usize> = numbering
        .refs
        .iter()
        .enumerate()
        .map(|(ordinal, here)| (Arc::as_ptr(here.stmt()), ordinal))
        .collect();
    let mut operand: HashMap<Name, Key> = HashMap::new();
    for (before, after) in placed.iter_stmts().iter().zip(result.iter_stmts()) {
        if let Some(&ordinal) = numbered.get(&Arc::as_ptr(before))
            && let Some((number, params)) = &numbering.numbers[ordinal]
        {
            let rank = match &*after {
                Statement::Accum(_) | Statement::Init(_) => 0,
                Statement::Load(_) => 1,
                Statement::Assign(a)
                    if a.op.copies() && a.args.len() == 1 && loaded.contains(&a.args[0]) =>
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
    operand
}

/// The key of an operand no statement of the body defines: a free name, after every defined one.
pub(crate) fn unranked() -> Key {
    Key::tuple([Key::Int(3), Key::str("")])
}

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

/// Where a sibling sorts first: leaves, then blocks, then blocks with side effects.
pub(crate) fn sibling_category(stmt: &Statement) -> i64 {
    let block = !stmt.nested().is_empty();
    if block && stmt.has_side_effects() {
        2
    } else {
        i64::from(!block)
    }
}

/// The key a sum's terms sort by: the operand rank, the name, the coefficient.
pub(crate) fn term_key(operand: Key, name: &Name, coefficient: f64) -> (Key, Name, Key) {
    (operand, name.clone(), Key::float(coefficient))
}

/// Whether `op`'s operands may trade places.
pub(crate) fn commutative(op: BinOp) -> bool {
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
pub(crate) fn unspelled(e: &Expression, axis_order: &HashMap<Name, usize>) -> Key {
    expression_key(e, &mut |e| match &**e {
        Expr::Var(name) => axis_order
            .get(name)
            .map(|index| Key::tuple([Key::str(ExprType::Var.text()), Key::Int(*index as i64)])),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::extent_digest;
    use crate::expr::{Builtin, CastTarget, Expr, Expression, LitType, parse};

    /// The extent digest spells an expression in the Python order's old form (`BinaryExpr`, `FuncCallExpr`, ...);
    /// the stored goldens' decisions are addressed in it, so each kind's spelling is pinned through its digest.
    #[test]
    fn the_extent_digest_spells_every_expression_type_as_it_always_did() {
        let samples = [
            Expression::var("n"),
            Expression::int(4096),
            Expression::new(Expr::Builtin(Builtin::WarpSize)),
            parse("n * 2").unwrap(),
            Expression::new(Expr::Call("f".into(), Arc::from([Expression::var("n")]))),
            parse("c ? 1 : 2").unwrap(),
            Expression::new(Expr::Cast(
                CastTarget::Known(LitType::Int),
                Expression::var("n"),
            )),
            Expression::new(Expr::FlatIndex(
                "b".into(),
                Arc::from([Expression::var("i")]),
            )),
        ];
        let digests: Vec<String> = samples.iter().map(extent_digest).collect();
        assert_eq!(
            digests,
            [
                "1d853204c55fe10c3b631736260de7badefe031a5e6643b76acf77089fce6640",
                "d1e873473019195ecd14c66351b0fad712808041e6e157c9980d3e92a40d4ec7",
                "a29b45df8d2919533196cea4899182ce3b44db5ce0d67f1b054fd41c1eac14c2",
                "6851899934cf8768ae8bf8a5c9db2ae378549bbc2b0d720788738ed14a01466d",
                "d932b5b6d247bd35b90c67c2fc9ad90cb4cb23d8e933dc280ca6088c8fe4166d",
                "58f9b6c7f305258af077cca47134711d029c5ef10dfd3417a162c82fbcb3a1e8",
                "d6c4a7737917bac409b76ef7cdf22a68faaeddc557b884eb9de246aea0a91268",
                "2dbaed08deb340643afa3c16fc6ebe1c45f91906a928714cd4c3498947dcae3e",
            ]
            .map(String::from)
        );
    }
}
