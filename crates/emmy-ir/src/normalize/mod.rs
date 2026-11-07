//! Body normalization: the canonical form every Loop IR body takes at construction and inside every identity.
//! Pure `Body → Body` steps and the one driver, `normalize_body`, memoized by the body's content. Normalization
//! answers to no knob, pin or evidence; a transform something decides is a pipeline pass.

pub mod order;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};

use crate::body::{Body, Normalized, exposed_defines, free_names, free_ssa, ordered_sibling_defs};
use crate::expr::{BinOp, Expr, Expression, Interval, Lit, Num, SimplifyCtx, affine_form};
use crate::kinds::axis::{Axis, Dim};
use crate::kinds::dtype::{is_float, promote};
use crate::kinds::elementwise::{Op, evaluate_f32};
use crate::kinds::field::{AxisRef, DType, Float, Names, NumOr};
use crate::kinds::*;
use crate::memo::Memo;
use crate::normalize::order::{is_protocol, ordering_constraints, topological_sort};
use crate::numbering::{
    Key, Number, Numbering, Param, distinct, ptr, roles, scope_tree, value_numbers,
};
use crate::protocol::{literal_binding, unique};
use crate::text::{body_text, structural_body_text};
use crate::{Error, Name, Result, locked};

const SYMBOLIC_AXIS_HI: i128 = 1 << 30;

/// Push an iteration axis `[0, extent)` into a simplification context so index expressions over it fold.
pub fn extend_simplify_ctx(ctx: &SimplifyCtx, axis: &Axis) -> SimplifyCtx {
    match axis.extent.as_static() {
        Some(n) => ctx.extend(
            &axis.name,
            Interval::new(0, n as i128 - 1),
            Some(Expression::int(n)),
        ),
        None => ctx.extend(
            &axis.name,
            Interval::new(0, SYMBOLIC_AXIS_HI),
            Some(axis.extent.expression.clone()),
        ),
    }
}

// --- the memo -------------------------------------------------------------------------------------------

/// The normal forms of the bodies met in this process.
fn normal_forms() -> &'static Mutex<Memo<u64, Body>> {
    static MEMO: OnceLock<Mutex<Memo<u64, Body>>> = OnceLock::new();
    MEMO.get_or_init(|| Mutex::new(Memo::new(16384)))
}

/// Forget every memoized normal form (the tests of the memo use it).
pub fn clear_memo() {
    locked(normal_forms()).clear();
}

/// The executable normal form of `body`, cached on the body and its fixed point, and memoized by text.
pub fn normalize_body(body: &Body) -> Result<Body> {
    if let Some(normal) = body.0.normalized.get() {
        return Ok(match normal {
            Normalized::Itself => body.clone(),
            Normalized::Other(normal) => normal.clone(),
        });
    }
    let text = body_text(body).ok();
    let found = text
        .as_deref()
        .and_then(|text| locked(normal_forms()).get(&body.content_hash(), text));
    let normal = match found {
        Some(found) => found,
        None => {
            let computed = normalize_uncached(body)?;
            if let Some(text) = text {
                locked(normal_forms()).put(body.content_hash(), text, computed.clone());
            }
            computed
        }
    };
    let _ = normal.0.normalized.set(Normalized::Itself);
    if !normal.ptr_eq(body) {
        let _ = body.0.normalized.set(Normalized::Other(normal.clone()));
    }
    Ok(normal)
}

/// The normal form, computed afresh.
pub fn normalize_uncached(body: &Body) -> Result<Body> {
    // The normal form holds every operation: a call is storage sharing inside fusion, which expands it.
    if !body.of_kinds(&["Call"]).is_empty() {
        return Err(Error::Value(
            "a body that holds a subroutine call has no normal form; fusion expands its calls"
                .into(),
        ));
    }
    // Motion can put two lexical bindings in one scope. Give them distinct names first.
    let mut stmts = rename_ssa_sequential(&topological_sort(body));
    // The folds feed the steps ahead of them, so the steps run again after a fold that changed the body. The order
    // permutes and names, and one step reads names: a body the order left an unread value in goes through the steps
    // again; any other is final, since the order is final on its own output.
    for _ in 0..8 {
        let prepared = prepare_body(&stmts)?;
        stmts = fold_affine(&fold_trivial_selects(&prepared), &HashSet::new());
        if stmts != prepared {
            continue;
        }
        let ordered = canonical_order(&prepared)?;
        stmts = drop_unread_values(&ordered);
        if stmts == ordered {
            return Ok(ordered);
        }
    }
    let mut lines = Vec::new();
    crate::text::body_lines(&stmts, "  ", &mut lines)?;
    Err(Error::Value(format!(
        "the normal form did not settle in eight rounds:\n{}",
        lines.join("\n")
    )))
}

/// The exact simplifications, then every value placed once; the three run until the first two leave a placed body
/// alone.
pub fn prepare_body(body: &Body) -> Result<Body> {
    let mut stmts = topological_sort(body);
    stmts = drop_size_one_free_axes(&stmts);
    stmts = drop_size_one_reduce_axes(&stmts);
    stmts = canonicalize_free_axis_order(&stmts)?;
    stmts = fold_trivial_selects(&stmts);
    stmts = fold_constants(&stmts);
    stmts = drop_unread_values(&stmts);
    let mut placed: Option<Body> = None;
    loop {
        let simplified =
            canonicalize_expressions(&simplify_body(&merge_sibling_reduce_loops(&stmts))?, &[])?;
        if placed.as_ref() == Some(&simplified) {
            return Ok(simplified);
        }
        let next = place_values(&simplified)?;
        stmts = next.clone();
        placed = Some(next);
    }
}

fn with_bodies(stmt: &Arc<Statement>, bodies: Vec<Body>) -> Arc<Statement> {
    Statement::rebuild_bodies(stmt, bodies)
}

fn substituted(stmt: &Arc<Statement>, name: &Name, value: Expression) -> Arc<Statement> {
    Statement::substitute(stmt, &HashMap::from([(name.clone(), value)]))
}

fn assign(name: &Name, op: &str, args: Vec<Name>, dtype: Option<DType>) -> Arc<Statement> {
    Arc::new(Statement::Assign(Assign {
        name: name.clone(),
        op: Op::new(op),
        args: args.into(),
        dtype,
    }))
}

pub mod axes;
pub mod canonical;
pub mod fold;
pub mod place;
pub mod siblings;
pub mod simplify;
pub mod ties;

pub use axes::*;
pub use canonical::*;
pub use fold::*;
pub use place::*;
pub use siblings::*;
pub use simplify::*;

// --- helpers the steps share -------------------------------------------------------------------------------

/// Every `Body` reading the normalization holds of a statement for the `coalesce` step and the readers.
pub fn body_from_free(stmt: &Arc<Statement>) -> HashSet<Name> {
    free_ssa(stmt)
}

pub fn sibling_defs(stmt: &Statement) -> Vec<Name> {
    ordered_sibling_defs(stmt)
}

pub fn carried_names_out(body: &Body) -> HashSet<Name> {
    carried_out(body)
}

pub fn placeholder_num(value: f64) -> Num {
    Num::Float(value)
}

pub fn step_of(step: &NumOr<Expression>) -> Option<&Expression> {
    match step {
        NumOr::Value(e) => Some(e),
        NumOr::Num(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::{body_text, parse_body};

    #[test]
    fn a_normal_form_is_freed_with_its_body() {
        let body = parse_body("x = load X[a]\nO[a] = x").unwrap();
        let normal = normalize_body(&body).unwrap();
        assert!(
            normalize_body(&normal).unwrap().ptr_eq(&normal),
            "the normal form is final"
        );
        let weak = Arc::downgrade(&normal.0);
        drop(normal);
        drop(body);
        clear_memo();
        assert!(weak.upgrade().is_none(), "a body holds no Arc to itself");
    }

    fn normal(text: &str) -> String {
        let body = parse_body(text).unwrap();
        let normal = normalize_body(&body).unwrap_or_else(|err| panic!("{err}"));
        assert_eq!(
            normalize_uncached(&normal).unwrap(),
            normal,
            "the normal form is final"
        );
        body_text(&normal).unwrap()
    }

    #[test]
    fn a_body_normalizes_and_settles() {
        let text = normal(
            "for i in 0..4:\n    for k in 0..8:\n        x = load a[i, k]\n        y = exp(x)\n        s <- add(s, y) over k\n    out[i] = s",
        );
        assert!(text.contains("acc0 <- add(acc0, v0) over a1"), "{text}");
        let folded = normal(
            "for i in 0..4:\n    x = load a[i]\n    let c = 2.0\n    y = multiply(x, c)\n    z = add(y, x)\n    out[i] = z",
        );
        assert!(folded.contains("v0 = 3*in0"), "{folded}");
    }

    #[test]
    fn renamed_bodies_share_an_identity() {
        let a = parse_body("for i in 0..4:\n    x = load p[i]\n    y = exp(x)\n    out[i] = y")
            .unwrap();
        let b = parse_body("for j in 0..4:\n    q = load r[j]\n    w = exp(q)\n    res[j] = w")
            .unwrap();
        let ia = crate::numbering::identity::canonicalize_identity(&a, false, None).unwrap();
        let ib = crate::numbering::identity::canonicalize_identity(&b, false, None).unwrap();
        assert_eq!(ia.key, ib.key);
        assert_eq!(ia.arguments, vec![Name::from("p"), Name::from("out")]);
        let c = parse_body("for j in 0..4:\n    q = load r[j]\n    w = log(q)\n    res[j] = w")
            .unwrap();
        assert_ne!(
            crate::numbering::identity::canonicalize_identity(&c, false, None)
                .unwrap()
                .key,
            ia.key
        );
    }

    #[test]
    fn atomic_partition_axis_order_converges_independently_of_affine_spelling() {
        // An axis absent from an atomic output must not trade places with a scaled output axis every round.
        let mut forms: HashSet<String> = HashSet::new();
        for (row, part) in [("row", "part"), ("z", "a")] {
            for reverse_loops in [false, true] {
                for reverse_sum in [false, true] {
                    let terms = [format!("{row} * 256"), format!("{part} * 64")];
                    let index = if reverse_sum {
                        format!("{} + {}", terms[1], terms[0])
                    } else {
                        format!("{} + {}", terms[0], terms[1])
                    };
                    let mut spelled = format!("x = load X[{index}]\nO[{row} * 2] += x");
                    let wrapping = if reverse_loops {
                        [row, part]
                    } else {
                        [part, row]
                    };
                    for name in wrapping {
                        spelled = format!("for {name} in 0..2:\n{}", testing::indented(&spelled));
                    }
                    let given = parse_body(&spelled).unwrap();
                    let mut prepared = rename_ssa_sequential(&given);
                    // A bounded number of rounds before the fixed-point driver: the regression used to cycle.
                    for _ in 0..4 {
                        prepared = prepare_body(&prepared).unwrap();
                    }
                    assert_eq!(prepare_body(&prepared).unwrap(), prepared);
                    let normalized = normalize_body(&given).unwrap();
                    let reread = parse_body(&body_text(&normalized).unwrap()).unwrap();
                    assert_eq!(normalize_body(&reread).unwrap(), normalized);
                    forms.insert(body_text(&normalized).unwrap());
                }
            }
        }
        assert_eq!(forms.len(), 1);
    }

    #[test]
    fn normal_form_is_a_fixed_point_when_the_order_breaks_a_tie_by_spelling() {
        // Two loop-invariant values of one shape tie in the canonical order, which breaks the tie by their
        // spelling: the spelling reads nothing the order decides, so normalizing the normal form changes nothing.
        let once = normalize_uncached(&testing::body(
            "
            for a0 in 0..48:
                for a2 in 0..512:
                    v36 = multiply(v35, v3)
                    v37 = pad(v36)
                    reshape_9[0, a0, a2 / 64, a2 % 64] = v37
                    for a7 in 0..128:
                        v146 = pad(v145)
                        reshape_5[0, a0, a2 / 64, a2 % 64, a7] = v146
            ",
        ))
        .unwrap();
        let again = normalize_uncached(&once).unwrap();
        assert_eq!(body_text(&again).unwrap(), body_text(&once).unwrap());
        assert_eq!(again, once);
    }

    /// `v * (exp(s - s) / exp(s - s))`, which is `v`.
    fn one_key_softmax() -> Body {
        testing::body(
            "
            for i in 0..8:
                v = load x[i]
                s = load y[i]
                z = subtract(s, s)
                u = exp(z)
                w = divide(u, u)
                o = multiply(v, w)
                O[i] = o
            ",
        )
    }

    /// Two reads of one index make a predicate constant, the select it decided a copy, and its reader invariant.
    fn equal_gather_index() -> Body {
        testing::body(
            "
            x = load x[0]
            for i in 0..4:
                a = load indices[i]
                b = load indices[i]
                s = select:
                    x when a - b < 1
                    a when 1
                v = exp(s)
                out[i] = v
            ",
        )
    }

    #[test]
    fn no_step_changes_a_prepared_body_the_folds_leave_alone() {
        // The driver runs the steps again only after a fold changed the body, so a prepared body the folds leave
        // alone has to be one every step leaves alone.
        for given in [one_key_softmax(), equal_gather_index()] {
            let mut prepared = rename_ssa_sequential(&given);
            let mut settled = false;
            for _ in 0..8 {
                prepared = prepare_body(&prepared).unwrap();
                let folded = fold_affine(&fold_trivial_selects(&prepared), &HashSet::new());
                if folded == prepared {
                    settled = true;
                    break;
                }
                prepared = folded;
            }
            assert!(settled, "the folds did not settle");
            assert_eq!(prepare_body(&prepared).unwrap(), prepared);
        }
    }

    #[test]
    fn a_reduction_only_another_bindings_readers_kept_alive_is_dropped() {
        // A cut can splice one cone twice under the same names: a statistic, then a contraction that reads it,
        // twice. Placement keeps the first statistic for both readers and the merge folds the second into a
        // contraction, where nothing reads it. It shares the first one's name, and the unread-value drop reads by
        // name, so only the names the order gives tell the two apart: the normal form holds one statistic, and
        // normalization leaves it alone.
        let cone = |weights: &str, out: &str| {
            format!(
                "
            for k in 0..4:
                x = load X[i, k]
                sq = multiply(x, x)
                s <- add(s, sq)
            r = rsqrt(s)
            for m in 0..4:
                y = load X[i, m]
                scaled = multiply(y, r)
                w_{out} = load {weights}[m, j]
                p_{out} = multiply(w_{out}, scaled)
                {out} <- add({out}, p_{out})"
            )
        };
        let writes = "
            O1[i, j] = o1
            O2[i, j] = o2";
        let given = testing::body(&format!(
            "for i in 0..8:\n    for j in 0..3:{}{}{writes}",
            cone("W1", "o1"),
            cone("W2", "o2")
        ));
        let normalized = normalize_body(&given).unwrap();
        assert_eq!(
            normalized.accums().len(),
            3,
            "{}",
            body_text(&normalized).unwrap()
        );
        assert_eq!(normalize_uncached(&normalized).unwrap(), normalized);
    }
}

/// What the tests of the steps share: a body from its text, texts back, and the fields they assert on.
#[cfg(test)]
pub(crate) mod testing {
    use std::sync::Arc;

    use crate::Name;
    use crate::body::Body;
    use crate::kinds::Statement;
    use crate::text::{body_text, parse_body, statement_text};

    pub(crate) fn body(text: &str) -> Body {
        parse_body(text).unwrap_or_else(|err| panic!("{err}\n{text}"))
    }

    pub(crate) fn text(body: &Body) -> String {
        body_text(body).unwrap()
    }

    pub(crate) fn stmt_text(stmt: &Arc<Statement>) -> String {
        statement_text(stmt).unwrap()
    }

    pub(crate) fn kinds(body: &Body) -> Vec<&'static str> {
        body.stmts().iter().map(|stmt| stmt.kind_name()).collect()
    }

    /// The body a loop holds, or a branch's taken arm.
    pub(crate) fn inner(stmt: &Statement) -> Body {
        match stmt.nested().as_slice() {
            [first, ..] => first.clone(),
            [] => panic!("not a statement over a body: {}", stmt.kind_name()),
        }
    }

    pub(crate) fn axis(stmt: &Statement) -> &str {
        match stmt {
            Statement::Loop(l) => &l.axis.name,
            Statement::StridedLoop(l) => &l.axis.name,
            _ => panic!("not a loop: {}", stmt.kind_name()),
        }
    }

    /// The axes of the top-level loops, in order.
    pub(crate) fn loop_axes(body: &Body) -> Vec<String> {
        body.stmts()
            .iter()
            .filter(|stmt| matches!(&***stmt, Statement::Loop(_)))
            .map(|stmt| axis(stmt).to_string())
            .collect()
    }

    /// The names the top-level accumulations update, in order.
    pub(crate) fn accum_names(body: &Body) -> Vec<String> {
        body.stmts()
            .iter()
            .filter_map(|stmt| match &**stmt {
                Statement::Accum(a) => Some(a.name.to_string()),
                _ => None,
            })
            .collect()
    }

    pub(crate) fn strs(names: &[Name]) -> Vec<&str> {
        names.iter().map(|name| &**name).collect()
    }

    /// `text` one level further in.
    pub(crate) fn indented(text: &str) -> String {
        text.lines()
            .map(|line| format!("    {line}"))
            .collect::<Vec<_>>()
            .join("\n")
    }
}
