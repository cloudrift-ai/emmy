//! Expression simplification over a body, each expression under the ranges of the axes in scope.

use std::sync::Arc;

use super::{extend_simplify_ctx, with_bodies};
use crate::Result;
use crate::body::Body;
use crate::expr::SimplifyCtx;
use crate::kinds::*;

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

pub(crate) fn simplify_stmt(stmt: &Arc<Statement>, ctx: &SimplifyCtx) -> Result<Arc<Statement>> {
    let stmt = Statement::try_map_expressions(stmt, &mut |e| e.simplify(ctx))?;
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
}
