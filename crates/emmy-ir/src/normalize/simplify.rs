//! Expression simplification over a body, each expression under the ranges of the axes in scope.

use super::*;

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
