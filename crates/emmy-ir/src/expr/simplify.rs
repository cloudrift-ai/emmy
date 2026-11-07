//! The simplifier: folds literals, drops algebraic identities and decides static comparisons under the ranges a
//! [`SimplifyCtx`] carries; the affine form reads an index as anchor plus integer coefficients.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;

use super::{BinOp, Expr, Expression, Lit, LitType, Num, apply_binop};
use crate::{Name, Result};

use std::collections::BTreeMap;

// --- the simplifier's helpers ---------------------------------------------------------------------

pub(super) fn fold_binop_literals(op: BinOp, left: &Lit, right: &Lit) -> Result<Expression> {
    let folded = apply_binop(op, left.value, right.value)?;
    if let Num::Bool(flag) = folded {
        return Ok(Expression::int(flag as i64));
    }
    let both_int = left.dtype == LitType::Int && right.dtype == LitType::Int;
    if let Num::Int(v) = folded
        && both_int
    {
        return Ok(Expression::int(v));
    }
    Ok(Expression::float(folded.as_f64()))
}

pub(super) fn simplify_binary(
    this: &Expression,
    op: BinOp,
    left: &Expression,
    right: &Expression,
    ctx: &SimplifyCtx,
) -> Result<Expression> {
    let (orig_left, orig_right) = (left, right);
    let left = left.simplify(ctx)?;
    let right = right.simplify(ctx)?;
    if let (Some(l), Some(r)) = (left.as_literal(), right.as_literal()) {
        return fold_binop_literals(op, l, r);
    }
    match op {
        BinOp::Add => {
            if left.is_zero() {
                return Ok(right);
            }
            if right.is_zero() {
                return Ok(left);
            }
            if let Some(recomposed) = fold_divmod_recompose(&left, &right, ctx)? {
                return Ok(recomposed);
            }
        }
        BinOp::Sub => {
            if right.is_zero() {
                return Ok(left);
            }
            if left == right {
                return Ok(Expression::int(0));
            }
        }
        BinOp::Mul => {
            if left.is_zero() || right.is_zero() {
                return Ok(Expression::int(0));
            }
            if left.is_one() {
                return Ok(right);
            }
            if right.is_one() {
                return Ok(left);
            }
        }
        BinOp::Div | BinOp::FloorDiv => {
            if right.is_one() {
                return Ok(left);
            }
            if left.is_zero() {
                return Ok(Expression::int(0));
            }
            if let Some(n) = right.int_literal().filter(|n| *n > 1) {
                if let Expr::Binary(inner_op, inner_left, inner_right) = &*left
                    && inner_op.is_division()
                    && let Some(Lit { value, dtype }) = inner_right.as_literal()
                    && *dtype == LitType::Int
                    && value.as_f64() > 0.0
                    && inner_left.range(ctx).is_some_and(|bounds| bounds.lo >= 0)
                    // Positive constant quotient chains have one denominator in index normal form.
                    && let Ok(product) = apply_binop(BinOp::Mul, *value, Num::Int(n))
                {
                    return Expression::binary(
                        BinOp::Div,
                        inner_left.clone(),
                        Expression::literal(product, LitType::Int),
                    )
                    .simplify(ctx);
                }
                if let Some((q, _)) = div_mod_decompose(&left, n, ctx)? {
                    return Ok(q);
                }
                if op == BinOp::FloorDiv && left.range(ctx).is_some_and(|bounds| bounds.lo >= 0) {
                    return Ok(Expression::binary(BinOp::Div, left, right));
                }
            }
            if let Some(cancelled) = cancel_common_factors(op, &left, &right, ctx) {
                return Ok(cancelled);
            }
        }
        BinOp::Mod => {
            if right.is_one() || left.is_zero() {
                return Ok(Expression::int(0));
            }
            if let Some(n) = right.int_literal().filter(|n| *n > 1) {
                if let Some((_, r)) = div_mod_decompose(&left, n, ctx)? {
                    return Ok(r);
                }
            } else if let Some(below) = mod_below_divisor(&left, &right, ctx)? {
                return Ok(below);
            }
            if let Some(cancelled) = cancel_common_factors(op, &left, &right, ctx) {
                return Ok(cancelled);
            }
        }
        BinOp::And => {
            if left.is_truthy() {
                return Ok(right);
            }
            if right.is_truthy() {
                return Ok(left);
            }
            if left.is_falsy() || right.is_falsy() {
                return Ok(Expression::int(0));
            }
        }
        BinOp::Or => {
            if left.is_falsy() {
                return Ok(right);
            }
            if right.is_falsy() {
                return Ok(left);
            }
            if left.is_truthy() || right.is_truthy() {
                return Ok(Expression::int(1));
            }
        }
        BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq => {
            if let (Some(la), Some(lb)) = (left.range(ctx), right.range(ctx))
                && let Some(decided) = static_cmp(op, la, lb)
            {
                return Ok(Expression::int(decided as i64));
            }
        }
        _ => {}
    }
    if Arc::ptr_eq(&left.0, &orig_left.0) && Arc::ptr_eq(&right.0, &orig_right.0) {
        return Ok(this.clone());
    }
    Ok(Expression::binary(op, left, right))
}

/// Flatten a tree of `*` into its non-int factors and its int coefficient.
pub(super) fn multiplicative_factors(expression: &Expression) -> (Vec<Expression>, i64) {
    if let Some(Lit { value, dtype }) = expression.as_literal()
        && *dtype == LitType::Int
        && value.is_integral()
    {
        return (vec![], value.as_int().unwrap_or(0));
    }
    if let Expr::Binary(BinOp::Mul, left, right) = &**expression {
        let (mut lf, lk) = multiplicative_factors(left);
        let (rf, rk) = multiplicative_factors(right);
        if let Some(k) = lk.checked_mul(rk) {
            lf.extend(rf);
            return (lf, k);
        }
    }
    (vec![expression.clone()], 1)
}

pub(super) fn is_positive(expression: &Expression, ctx: &SimplifyCtx) -> bool {
    if let Some(Lit { value, .. }) = expression.as_literal()
        && value.is_integral()
    {
        return value.as_int().is_ok_and(|v| v >= 1);
    }
    expression.range(ctx).is_some_and(|rng| rng.lo >= 1)
}

pub(super) fn rebuild_product(factors: &[Expression], k: i64) -> Expression {
    if k == 0 {
        return Expression::int(0);
    }
    if factors.is_empty() {
        return Expression::int(k);
    }
    let mut out = factors[0].clone();
    for f in &factors[1..] {
        out = Expression::binary(BinOp::Mul, out, f.clone());
    }
    if k != 1 {
        out = Expression::binary(BinOp::Mul, out, Expression::int(k));
    }
    out
}

pub(super) fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.abs()
}

pub(super) fn cancel_common_factors(
    op: BinOp,
    left: &Expression,
    right: &Expression,
    ctx: &SimplifyCtx,
) -> Option<Expression> {
    let (l_factors, mut l_k) = multiplicative_factors(left);
    let (r_factors, mut r_k) = multiplicative_factors(right);
    let mut cancelled = false;
    let mut r_remaining = r_factors;
    let mut l_remaining = Vec::new();
    let mut common_factors = Vec::new();
    let mut common_int = 1;
    for f in l_factors {
        match r_remaining
            .iter()
            .position(|rf| f == *rf && is_positive(&f, ctx))
        {
            Some(index) => {
                r_remaining.remove(index);
                cancelled = true;
                common_factors.push(f);
            }
            None => l_remaining.push(f),
        }
    }
    if l_k > 0 && r_k > 0 {
        let g = gcd(l_k, r_k);
        if g > 1 {
            l_k /= g;
            r_k /= g;
            cancelled = true;
            common_int = g;
        }
    }
    if !cancelled {
        return None;
    }
    let new_left = rebuild_product(&l_remaining, l_k);
    let new_right = rebuild_product(&r_remaining, r_k);
    if op.is_division() {
        if new_right.is_one() {
            return Some(new_left);
        }
        if new_left.is_zero() {
            return Some(Expression::int(0));
        }
    } else if op == BinOp::Mod {
        if new_right.is_one() || new_left.is_zero() {
            return Some(Expression::int(0));
        }
        let common = rebuild_product(&common_factors, common_int);
        let modulo = Expression::binary(BinOp::Mod, new_left, new_right);
        return Some(if common.is_one() {
            modulo
        } else {
            Expression::binary(BinOp::Mul, common, modulo)
        });
    }
    Some(Expression::binary(op, new_left, new_right))
}

pub(super) fn addend_chain(e: &Expression, out: &mut Vec<Expression>) {
    if let Expr::Binary(BinOp::Add, l, r) = &**e {
        addend_chain(l, out);
        addend_chain(r, out);
    } else {
        out.push(e.clone());
    }
}

/// `e` as `(x <op> c) * a` (either `*` operand order, bare form `a = 1`), `op` among `ops`, `c` and `a` positive int
/// literals: `(x, c, a)`.
pub(super) fn scaled_term(e: &Expression, ops: &[BinOp]) -> Option<(Expression, i64, i64)> {
    let mut a = 1;
    let mut e = e.clone();
    if let Expr::Binary(BinOp::Mul, left, right) = &*e.clone() {
        for (lit_side, other) in [(right, left), (left, right)] {
            if let Some(value) = lit_side.int_literal().filter(|v| *v > 0) {
                a = value;
                e = other.clone();
                break;
            }
        }
    }
    if let Expr::Binary(op, left, right) = &*e
        && ops.contains(op)
        && let Some(c) = right.int_literal().filter(|c| *c > 1)
    {
        return Some((left.clone(), c, a));
    }
    None
}

/// Fold `(x / c) * (c·k) + (x % c) * k → x * k` inside an additive chain.
pub(super) fn fold_divmod_recompose(
    left: &Expression,
    right: &Expression,
    ctx: &SimplifyCtx,
) -> Result<Option<Expression>> {
    let mut terms = Vec::new();
    addend_chain(left, &mut terms);
    addend_chain(right, &mut terms);
    for (i, t1) in terms.iter().enumerate() {
        let Some((x, c, a)) = scaled_term(t1, &[BinOp::Div, BinOp::FloorDiv]) else {
            continue;
        };
        if a % c != 0 {
            continue;
        }
        let k = a / c;
        for (j, t2) in terms.iter().enumerate() {
            if j == i {
                continue;
            }
            match scaled_term(t2, &[BinOp::Mod]) {
                Some((rx, rc, rk)) if rx == x && rc == c && rk == k => {}
                _ => continue,
            }
            let mut out = if k == 1 {
                x.clone()
            } else {
                Expression::binary(BinOp::Mul, x.clone(), Expression::int(k))
            };
            for (pos, t) in terms.iter().enumerate() {
                if pos != i && pos != j {
                    out = Expression::binary(BinOp::Add, out, t.clone());
                }
            }
            return Ok(Some(out.simplify(ctx)?));
        }
    }
    Ok(None)
}

/// `expression` as `n * q + r` with `0 <= r < n`, both proven from operand ranges.
pub(super) fn div_mod_decompose(
    expression: &Expression,
    n: i64,
    ctx: &SimplifyCtx,
) -> Result<Option<(Expression, Expression)>> {
    if let Some(value) = expression.int_literal().filter(|v| *v >= 0) {
        return Ok(Some((
            Expression::int(value.div_euclid(n)),
            Expression::int(value.rem_euclid(n)),
        )));
    }
    if let Expr::Binary(BinOp::Mul, left, right) = &**expression {
        for (k_side, other_side) in [(right, left), (left, right)] {
            let Some(k) = k_side.int_literal().filter(|k| *k > 0) else {
                continue;
            };
            if other_side.range(ctx).is_none_or(|rng| rng.lo < 0) {
                continue;
            }
            if k % n == 0 {
                let q = Expression::binary(BinOp::Mul, other_side.clone(), Expression::int(k / n))
                    .simplify(ctx)?;
                return Ok(Some((q, Expression::int(0))));
            }
            if n % k == 0 {
                let d = Expression::int(n / k);
                let q =
                    Expression::binary(BinOp::Div, other_side.clone(), d.clone()).simplify(ctx)?;
                let r = Expression::binary(
                    BinOp::Mul,
                    Expression::binary(BinOp::Mod, other_side.clone(), d),
                    Expression::int(k),
                )
                .simplify(ctx)?;
                return Ok(Some((q, r)));
            }
        }
    }
    if let Expr::Binary(BinOp::Add, left, right) = &**expression {
        let mut l = div_mod_decompose(left, n, ctx)?;
        let mut r = div_mod_decompose(right, n, ctx)?;
        if let Some((_, rl)) = &l
            && r.is_none()
            && rl.is_zero()
        {
            r = split_addend(right, n, ctx)?;
        } else if let Some((_, rr)) = &r
            && l.is_none()
            && rr.is_zero()
        {
            l = split_addend(left, n, ctx)?;
        }
        let (Some((ql, rl)), Some((qr, rr))) = (l, r) else {
            return Ok(None);
        };
        let r_sum = Expression::binary(BinOp::Add, rl, rr).simplify(ctx)?;
        let Some(rng) = r_sum.range(ctx) else {
            return Ok(None);
        };
        if rng.lo < 0 || rng.hi >= n as i128 {
            return Ok(None);
        }
        let q = Expression::binary(BinOp::Add, ql, qr).simplify(ctx)?;
        return Ok(Some((q, r_sum)));
    }
    if let Some(rng) = expression.range(ctx)
        && rng.lo >= 0
        && rng.hi < n as i128
    {
        return Ok(Some((Expression::int(0), expression.clone())));
    }
    if let Expr::Binary(op, left, divisor) = &**expression
        && op.is_division()
        && let Some(d) = divisor.int_literal().filter(|d| *d > 0)
        && let Some(nd) = n.checked_mul(d)
        && let Some((q_inner, r_inner)) = div_mod_decompose(left, nd, ctx)?
    {
        return Ok(Some((
            q_inner,
            Expression::binary(*op, r_inner, divisor.clone()).simplify(ctx)?,
        )));
    }
    Ok(None)
}

pub(super) fn split_addend(
    expression: &Expression,
    n: i64,
    ctx: &SimplifyCtx,
) -> Result<Option<(Expression, Expression)>> {
    if expression.range(ctx).is_none_or(|rng| rng.lo < 0) {
        return Ok(None);
    }
    let lit = Expression::int(n);
    Ok(Some((
        Expression::binary(BinOp::Div, expression.clone(), lit.clone()).simplify(ctx)?,
        Expression::binary(BinOp::Mod, expression.clone(), lit).simplify(ctx)?,
    )))
}

pub(super) fn mod_below_divisor(
    left: &Expression,
    right: &Expression,
    ctx: &SimplifyCtx,
) -> Result<Option<Expression>> {
    let Expr::Var(name) = &**left else {
        return Ok(None);
    };
    let Some(bound) = ctx.bounds.get(name) else {
        return Ok(None);
    };
    if bound.simplify(ctx)? != *right {
        return Ok(None);
    }
    Ok(left
        .range(ctx)
        .filter(|rng| rng.lo >= 0)
        .map(|_| left.clone()))
}

pub(super) fn static_cmp(op: BinOp, la: Interval, lb: Interval) -> Option<bool> {
    match op {
        BinOp::Lt => (la.hi < lb.lo)
            .then_some(true)
            .or((la.lo >= lb.hi).then_some(false)),
        BinOp::Le => (la.hi <= lb.lo)
            .then_some(true)
            .or((la.lo > lb.hi).then_some(false)),
        BinOp::Gt => (la.lo > lb.hi)
            .then_some(true)
            .or((la.hi <= lb.lo).then_some(false)),
        BinOp::Ge => (la.lo >= lb.hi)
            .then_some(true)
            .or((la.hi < lb.lo).then_some(false)),
        BinOp::Eq => {
            if la.hi < lb.lo || la.lo > lb.hi {
                Some(false)
            } else if la.lo == la.hi && la.hi == lb.lo && lb.lo == lb.hi {
                Some(true)
            } else {
                None
            }
        }
        _ => None,
    }
}

// --- the simplification context -------------------------------------------------------------------

/// A closed integer interval `[lo, hi]`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Interval {
    pub lo: i128,
    pub hi: i128,
}

impl Interval {
    pub fn new(lo: i128, hi: i128) -> Interval {
        Interval { lo, hi }
    }
}

/// The ranges in scope: an integer interval per name, and a symbolic exclusive upper bound per name where one is
/// known (`i < seq_len`), so a modulo by a non-literal divisor folds.
#[derive(Clone, Default, PartialEq, Debug)]
pub struct SimplifyCtx {
    pub ranges: HashMap<Name, Interval>,
    pub bounds: HashMap<Name, Expression>,
}

impl SimplifyCtx {
    pub fn empty() -> SimplifyCtx {
        SimplifyCtx::default()
    }

    pub fn extend(&self, name: &str, interval: Interval, bound: Option<Expression>) -> SimplifyCtx {
        let mut out = self.clone();
        out.ranges.insert(name.into(), interval);
        if let Some(bound) = bound {
            out.bounds.insert(name.into(), bound);
        }
        out
    }
}

// --- the affine form ------------------------------------------------------------------------------

/// `expression` as `anchor + Σ coeffs[v]·v` over `vars`; `None` when it is not affine in them.
pub fn affine_form(
    expression: &Expression,
    vars: &dyn Fn(&str) -> bool,
) -> Option<(Expression, BTreeMap<Name, i64>)> {
    let uses = |e: &Expression| e.free_vars().into_iter().any(|name| vars(&name));
    if !uses(expression) {
        return Some((expression.clone(), BTreeMap::new()));
    }
    match &**expression {
        Expr::Var(name) => Some((Expression::int(0), BTreeMap::from([(*name, 1)]))),
        Expr::Binary(op @ (BinOp::Add | BinOp::Sub), left, right) => {
            let (l_anchor, l_coeffs) = affine_form(left, vars)?;
            let (r_anchor, r_coeffs) = affine_form(right, vars)?;
            let sign = if *op == BinOp::Add { 1 } else { -1 };
            let anchor = Expression::binary(*op, l_anchor, r_anchor);
            let mut coeffs = l_coeffs;
            for (v, c) in r_coeffs {
                let sum = coeffs
                    .get(&v)
                    .copied()
                    .unwrap_or(0)
                    .checked_add(c.checked_mul(sign)?)?;
                coeffs.insert(v, sum);
            }
            coeffs.retain(|_, c| *c != 0);
            Some((anchor, coeffs))
        }
        Expr::Binary(BinOp::Mul, left, right) => {
            let (l_uses, r_uses) = (uses(left), uses(right));
            if l_uses && r_uses {
                return None;
            }
            let (side, mult) = if l_uses { (left, right) } else { (right, left) };
            let Lit { value, .. } = mult.as_literal()?;
            if !value.is_integral() {
                return None;
            }
            let factor = value.as_int().ok()?;
            let (sub_anchor, sub_coeffs) = affine_form(side, vars)?;
            let anchor = if l_uses {
                Expression::binary(BinOp::Mul, sub_anchor, mult.clone())
            } else {
                Expression::binary(BinOp::Mul, mult.clone(), sub_anchor)
            };
            let coeffs = sub_coeffs
                .into_iter()
                .map(|(v, c)| Some((v, c.checked_mul(factor)?)))
                .collect::<Option<_>>()?;
            Some((anchor, coeffs))
        }
        _ => None,
    }
}

/// `expression` as `(its integer anchor, the rest)`, or `None` when it is not affine with a literal anchor.
pub fn split_anchor(expression: &Expression) -> Option<(i64, Expression)> {
    let free = expression.free_vars();
    let (anchor, coeffs) = affine_form(expression, &|name| free.iter().any(|n| &**n == name))?;
    let anchor = anchor.eval(&HashMap::new()).ok()?.as_int().ok()?;
    let mut rest = Expression::int(0);
    for (name, coeff) in coeffs {
        let variable = Expression::new(Expr::Var(name));
        let term = if coeff == 1 {
            variable
        } else {
            Expression::binary(BinOp::Mul, variable, Expression::int(coeff))
        };
        rest = Expression::binary(BinOp::Add, rest, term);
    }
    Some((anchor, rest))
}

/// An upper bound on the number of distinct coordinate tuples `expressions` take over `var_extents`.
pub fn index_set_size(
    expressions: &[Expression],
    var_extents: &HashMap<Name, i64>,
) -> Option<i128> {
    let mut total: i128 = 1;
    for e in expressions {
        let (_, coeffs) = affine_form(e, &|name| var_extents.contains_key(&Name::new(name)))?;
        let mut size: i128 = 1;
        for (v, c) in coeffs {
            let span = (c as i128)
                .abs()
                .checked_mul(i128::from(*var_extents.get(&v)?) - 1)?;
            size = size.checked_add(span)?;
        }
        total = total.checked_mul(size)?;
    }
    Some(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(f * coeff) <op> divisor`.
    fn scaled(op: BinOp, coeff: i64, divisor: i64) -> Expression {
        let product = Expression::binary(BinOp::Mul, Expression::var("f"), Expression::int(coeff));
        Expression::binary(op, product, Expression::int(divisor))
    }

    fn at(e: &Expression, f: i64) -> Num {
        e.eval(&HashMap::from([(Name::from("f"), Num::Int(f))]))
            .unwrap()
    }

    /// `(c·x) % (c·y)` is `c·(x % y)`: the common factor scales the remainder, it does not vanish.
    #[test]
    fn the_modulo_factor_cancel_keeps_the_value() {
        for (coeff, modulus) in [(8, 64), (2, 4), (3, 6), (4, 16), (16, 128), (2, 8), (5, 10)] {
            let e = scaled(BinOp::Mod, coeff, modulus);
            let s = e.simplify(&SimplifyCtx::empty()).unwrap();
            for v in 0..256 {
                assert_eq!(at(&e, v), at(&s, v), "mismatch at f={v}: {e} vs {s}");
            }
        }
    }

    /// The shape that broke vectorized cp.async: `(f*8) % 64` is `8 * (f % 8)`, not `f % 8`.
    #[test]
    fn the_modulo_factor_cancel_keeps_the_common_factor() {
        let s = scaled(BinOp::Mod, 8, 64)
            .simplify(&SimplifyCtx::empty())
            .unwrap();
        assert_eq!(at(&s, 1), Num::Int(8));
        assert_eq!(at(&s, 9), Num::Int(8));
    }

    /// Division cancels the factor outright: `(f*8) / 64 == f / 8`.
    #[test]
    fn the_division_factor_cancel_keeps_the_value() {
        let e = scaled(BinOp::Div, 8, 64);
        let s = e.simplify(&SimplifyCtx::empty()).unwrap();
        for v in 0..256 {
            assert_eq!(at(&e, v), at(&s, v), "mismatch at f={v}: {e} vs {s}");
        }
    }
}
