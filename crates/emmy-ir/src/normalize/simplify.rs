//! Expression simplification over a body, each expression under the ranges of the axes in scope.

use std::sync::Arc;

use super::{extend_simplify_ctx, with_bodies};
use crate::Result;
use crate::body::Body;
use crate::dtype::DType;
use crate::expr::{Expr, Expression, LitType, SimplifyCtx};
use crate::statement::*;

// --- expression simplification ----------------------------------------------------------------------------

/// Simplify every expression inside a body, each under the value ranges of the loop axes around it.
pub fn simplify_body(body: &Body) -> Result<Body> {
    let ctx = SimplifyCtx::empty();
    let stmts: Result<Vec<Arc<Statement>>> = body
        .stmts()
        .iter()
        .map(|s| simplify_stmt(s, &ctx))
        .collect();
    Ok(body.rebuilt(stmts?))
}

/// `stmt` with every expression simplified as its field's class (an index under the integer rules, a value without
/// them) under `ctx`, and every child body under the ranges of the axes the statement binds for it.
pub(crate) fn simplify_stmt(stmt: &Arc<Statement>, ctx: &SimplifyCtx) -> Result<Arc<Statement>> {
    let stmt =
        Statement::try_map_classed_expressions(stmt, &mut |class, e| e.simplify_as(ctx, class))?;
    match &*stmt {
        // A value that is a literal is a constant: a `Let`, typed as declared, else `f32` for a float literal and
        // untyped for an integer one (`x - x` over integers is the integer zero, not a float), its literal a float
        // where the type is.
        Statement::Assign(a) if let Expr::Literal(lit) = &*a.value => {
            let dtype = a
                .dtype
                .or((lit.dtype == LitType::Float).then_some(DType::F32));
            let value = if dtype.is_some_and(DType::is_float) && lit.dtype != LitType::Float {
                Expression::float(lit.value.as_f64())
            } else {
                a.value.clone()
            };
            Ok(Arc::new(Statement::Let(Let {
                name: a.name,
                value,
                dtype,
            })))
        }
        Statement::Select(s) => Ok(Arc::new(Statement::Select(decided(s)))),
        _ => {
            let nested = stmt.nested();
            if nested.is_empty() {
                return Ok(stmt);
            }
            let inner = stmt
                .bound_axes()
                .iter()
                .fold(ctx.clone(), |inner, axis| extend_simplify_ctx(&inner, axis));
            let bodies: Result<Vec<Body>> = nested.iter().map(|b| simplify_in(b, &inner)).collect();
            Ok(with_bodies(&stmt, bodies?))
        }
    }
}

pub(crate) fn simplify_in(body: &Body, ctx: &SimplifyCtx) -> Result<Body> {
    let stmts: Result<Vec<Arc<Statement>>> =
        body.stmts().iter().map(|s| simplify_stmt(s, ctx)).collect();
    Ok(body.rebuilt(stmts?))
}

/// `s` without the branches its simplified predicates decide.
pub(crate) fn decided(s: &Select) -> Select {
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
                    name: s.name,
                    branches: kept.into(),
                };
            }
            Some(_) => {}
        }
    }
    kept.push(last.clone());
    Select {
        name: s.name,
        branches: kept.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::simplify_body;
    use crate::normalize::testing::*;

    /// The branches simplification leaves of one scatter read: the near value under `predicate`, the base buffer
    /// otherwise. Nothing but the select reads `cone` or `far`, so a branch that left released its value.
    fn branches(predicate: &str) -> Vec<String> {
        let simplified = simplify_body(&body(&format!(
            "
            for a in 0..4:
                near = load scattered[a]
                far = load base[a]
                cone = multiply(near, near)
                v = select:
                    cone when {predicate}
                    far when true
                out[a] = v
            "
        )))
        .unwrap();
        text(&simplified)
            .lines()
            .filter(|line| line.contains(" when "))
            .map(|line| line.trim().to_string())
            .collect()
    }

    #[test]
    fn a_branch_that_can_never_hold_leaves_and_releases_its_cone() {
        assert_eq!(branches("a < 0"), ["far when true"]);
    }

    #[test]
    fn a_branch_that_always_holds_becomes_the_else() {
        let values: Vec<String> = branches("a < 4")
            .iter()
            .map(|branch| {
                branch
                    .split(" when ")
                    .next()
                    .unwrap_or_default()
                    .to_string()
            })
            .collect();
        assert_eq!(values, ["cone"]);
    }

    #[test]
    fn an_undecided_branch_is_kept_whole() {
        assert_eq!(branches("a < 2"), ["cone when a < 2", "far when true"]);
    }

    /// The simplified line of `body` that holds `marker`.
    fn line(body: &str, marker: &str) -> String {
        let simplified = text(&simplify_body(&super::super::testing::body(body)).unwrap());
        simplified
            .lines()
            .find(|line| line.contains(marker))
            .unwrap_or_else(|| panic!("no line with {marker}:\n{simplified}"))
            .trim()
            .to_string()
    }

    #[test]
    fn a_value_quotient_is_a_true_division_and_keeps_its_digits() {
        let body = "
            for a in 0..4:
                for b in 0..8:
                    x = load X[(8 * a + b) / 8]
                    c = (8 * a + b) / 8
                    O[a, b] = c
            ";
        assert_eq!(line(body, "load X"), "x = load X[a]");
        assert_eq!(line(body, "c ="), "c = (8 * a + b) / 8");
    }

    #[test]
    fn a_value_quotient_keeps_its_common_factor() {
        let body = "
            for i in 0..8:
                x = load X[i]
                y = load Y[i]
                c = divide(3 * x, 3 * y)
                O[i] = c
            ";
        assert_eq!(line(body, "c ="), "c = 3 * x / (3 * y)");
    }

    #[test]
    fn a_tile_binds_its_axes_for_its_body() {
        let body = "
            Tile(axes=(m in 0..4, n in 0..8)):
                x = load X[(8 * m + n) / 8]
                O[m, n] = x
            ";
        assert_eq!(line(body, "load X"), "x = load X[m]");
    }
}
