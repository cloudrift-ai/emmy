//! Body normalization: the canonical form every Loop IR body takes at construction and inside every identity.
//! Pure `Body → Body` steps and the one driver, `normalize_body`, memoized by the body's exact form. Normalization
//! answers to no knob, pin or evidence; a transform something decides is a pipeline pass.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};

use crate::axis::Axis;
use crate::body::{Body, MemoKey};
use crate::dtype::DType;
use crate::expr::{Expression, Interval, SimplifyCtx};
use crate::memo::{BODY_MEMO_WEIGHT, Memo};
use crate::numbering::{congruent_states, group_keys};
use crate::order::topological_sort;
use crate::statement::*;
use crate::{Error, Name, Result};

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

/// `ctx` under the ranges `stmt` states for the built-ins, which hold in its own expressions and in its child bodies
/// (a `Grid`'s launch); `None` where it states none.
pub fn launch_ctx(ctx: &SimplifyCtx, stmt: &Statement) -> Option<SimplifyCtx> {
    let ranges = stmt.launch_ranges();
    (!ranges.is_empty()).then(|| {
        ranges
            .into_iter()
            .fold(ctx.clone(), |ctx, (builtin, range)| {
                ctx.extend(builtin.text(), range, None)
            })
    })
}

/// `ctx` inside `stmt`'s child bodies: under the ranges of the axes it binds for them.
pub fn inner_ctx(ctx: &SimplifyCtx, stmt: &Statement) -> SimplifyCtx {
    stmt.bound_axes()
        .iter()
        .fold(ctx.clone(), |inner, axis| extend_simplify_ctx(&inner, axis))
}

// --- the memo -------------------------------------------------------------------------------------------

/// The normal forms of the bodies met in this process, by their exact form; each weighs the statements of the body it
/// keys and the normal form it holds.
pub fn normal_forms() -> &'static Memo<MemoKey, Body> {
    static MEMO: OnceLock<Memo<MemoKey, Body>> = OnceLock::new();
    MEMO.get_or_init(|| {
        Memo::weighted(BODY_MEMO_WEIGHT, |key, normal| {
            (key.0.statement_count() + normal.statement_count()) as u64
        })
    })
}

/// The executable normal form of `body`: what the body already knows, else the memo's, else computed and kept.
pub fn normalize_body(body: &Body) -> Result<Body> {
    if let Some(normal) = body.normal_form() {
        return Ok(normal);
    }
    let normal =
        normal_forms().get_or_try_insert(&MemoKey(body.clone()), || normalize_uncached(body))?;
    body.remember_normal_form(&normal);
    Ok(normal)
}

/// The optimized forms of the bodies met in this process, by their exact form.
pub fn optimized_forms() -> &'static Memo<MemoKey, Body> {
    static MEMO: OnceLock<Memo<MemoKey, Body>> = OnceLock::new();
    MEMO.get_or_init(|| {
        Memo::weighted(BODY_MEMO_WEIGHT, |key, optimized| {
            (key.0.statement_count() + optimized.statement_count()) as u64
        })
    })
}

/// The optimizing half of the normal form: every exact simplification and every value computed once at the
/// shallowest scope that can hold it, with the statement order and the names as they stand wherever nothing moved.
/// It is what a body no identity reads takes (a Kernel IR body): [`normalize_body`] is this and then the canonical
/// order and names.
pub fn optimize_body(body: &Body) -> Result<Body> {
    optimized_forms().get_or_try_insert(&MemoKey(body.clone()), || optimize_uncached(body))
}

/// The optimized form, computed afresh.
pub fn optimize_uncached(body: &Body) -> Result<Body> {
    let mut stmts = distinct_names(&topological_sort(body)?);
    for _ in 0..8 {
        let prepared = prepared(&stmts, false)?;
        let next = one_operation_per_definition(&fold_trivial_selects(&prepared), &HashSet::new());
        if next == stmts {
            return Ok(next);
        }
        stmts = next;
    }
    unsettled(&stmts, "the optimized form")
}

/// The normal form, computed afresh.
pub fn normalize_uncached(body: &Body) -> Result<Body> {
    // The normal form holds every operation: a call is storage sharing inside fusion, which expands it.
    if !body.of_types(&[StatementType::Call]).is_empty() {
        return Err(Error::Value(
            "a body that holds a subroutine call has no normal form; fusion expands its calls"
                .into(),
        ));
    }
    // Motion can put two lexical bindings of one name in one scope; the order keeps their lexical reading and the
    // sequential names tell them apart.
    let mut stmts = rename_ssa_sequential(&topological_sort(body)?);
    // The split feeds the steps ahead of it, so the steps run again after a split that changed the body. The order
    // permutes and names, and the operand order inside a value reads the ranks the order gives, which the order in
    // turn spells by: the form is final when one whole round leaves the body alone.
    for _ in 0..8 {
        let prepared = prepare_body(&stmts)?;
        let split = one_operation_per_definition(&fold_trivial_selects(&prepared), &HashSet::new());
        if split != prepared {
            stmts = split;
            continue;
        }
        let next = drop_unread_values(&canonical_order(&prepared)?);
        if next == stmts {
            return Ok(next);
        }
        stmts = next;
    }
    unsettled(&stmts, "the normal form")
}

/// The exact simplifications, then every value placed once; the three run until the first two leave a placed body
/// alone.
pub fn prepare_body(body: &Body) -> Result<Body> {
    prepared(body, true)
}

/// [`prepare_body`] with its canonical loop steps (size-one loops dropped, free axes ordered, sibling reduce loops
/// over one axis merged into one loop) or without them. They are Loop IR's: the merge makes the four loops of four
/// rows one body. A kernel's loops are its schedule, and stay as their builder wrote them.
fn prepared(body: &Body, canonical_loops: bool) -> Result<Body> {
    let mut stmts = topological_sort(body)?;
    if canonical_loops {
        stmts = drop_size_one_free_axes(&stmts);
        stmts = drop_size_one_reduce_axes(&stmts);
        stmts = canonicalize_free_axis_order(&stmts)?;
    }
    stmts = fold_trivial_selects(&stmts);
    stmts = inline_literals(&stmts);
    stmts = drop_unread_values(&stmts);
    let mut placed: Option<Body> = None;
    for _ in 0..8 {
        let merged = if canonical_loops {
            merge_sibling_reduce_loops(&stmts)?
        } else {
            stmts.clone()
        };
        let simplified = canonicalize_expressions(
            &drop_unread_values(&simplify_body(&inline_literals(&merged))?),
            &[],
        )?;
        if placed.as_ref() == Some(&simplified) {
            return Ok(simplified);
        }
        let groups = congruent_states(&simplified);
        let next = merge_congruent_states(
            &place_values_keyed(&simplified, &group_keys(&groups))?,
            &groups,
        );
        stmts = next.clone();
        placed = Some(next);
    }
    unsettled(&stmts, "the merge, simplification and placement")
}

/// The error of a fixed point that did not settle in eight rounds, with the body it reached.
fn unsettled(stmts: &Body, steps: &str) -> Result<Body> {
    let mut lines = Vec::new();
    crate::text::body_lines(stmts, "  ", &mut lines)?;
    Err(Error::Value(format!(
        "{steps} did not settle in eight rounds:\n{}",
        lines.join("\n")
    )))
}

pub(crate) fn with_bodies(stmt: &Arc<Statement>, bodies: Vec<Body>) -> Arc<Statement> {
    Statement::rebuild_bodies(stmt, bodies)
}

fn substituted(stmt: &Arc<Statement>, name: &Name, value: Expression) -> Arc<Statement> {
    Statement::substitute(stmt, &HashMap::from([(*name, value)]))
}

fn assign(name: &Name, value: Expression, dtype: Option<DType>) -> Arc<Statement> {
    Arc::new(Statement::Assign(Assign {
        name: *name,
        value,
        dtype,
    }))
}

pub mod axes;
pub mod canonical;
pub mod congruent;
pub mod definitions;
pub mod fold;
pub mod place;
pub mod siblings;
pub mod simplify;
pub mod ties;

pub use axes::*;
pub use canonical::*;
pub use congruent::*;
pub use definitions::*;
pub use fold::*;
pub use place::*;
pub use siblings::*;
pub use simplify::*;

#[cfg(test)]
mod tests {
    use super::testing::eval_int;
    use super::*;
    use crate::axis::Dim;
    use crate::expr::parse as parse_expression;
    use crate::text::{body_text, parse_body};

    #[test]
    fn a_body_that_holds_a_call_has_no_normal_form() {
        let given = parse_body(
            "sub project(row in 0..3, col in 0..8) -> sum:
    for k in 0..16:
        xv = load x[row, k]
        wv = load weight[col, k]
        product = multiply(xv, wv)
        sum <- add(sum, product) over k
for k in 0..3:
    for c in 0..8:
        left = call project[x, weight](k, c)
        right = call project[x, weight](k, (c + 4) % 8)
        value = add(left, right)
        out[k, c] = value",
        )
        .unwrap();
        let error = normalize_body(&given).unwrap_err().to_string();
        assert!(error.contains("no normal form"), "{error}");
    }

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
        normal_forms().clear();
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
        assert!(
            text.contains("v0 = exp(in0)\n") && text.contains("acc0 <- add(acc0, v0) over a1"),
            "{text}"
        );
        let folded = normal(
            "for i in 0..4:\n    x = load a[i]\n    let c = 2.0\n    y = multiply(x, c)\n    z = add(y, x)\n    out[i] = z",
        );
        assert!(folded.contains("v0 = 3.0 * in0"), "{folded}");
    }

    #[test]
    fn renamed_bodies_share_an_identity() {
        let a = parse_body("for i in 0..4:\n    x = load p[i]\n    y = exp(x)\n    out[i] = y")
            .unwrap();
        let b = parse_body("for j in 0..4:\n    q = load r[j]\n    w = exp(q)\n    res[j] = w")
            .unwrap();
        let ia = crate::identity::canonicalize_identity(&a, false, None).unwrap();
        let ib = crate::identity::canonicalize_identity(&b, false, None).unwrap();
        assert_eq!(ia.key, ib.key);
        assert_eq!(ia.arguments, vec![Name::from("p"), Name::from("out")]);
        let c = parse_body("for j in 0..4:\n    q = load r[j]\n    w = log(q)\n    res[j] = w")
            .unwrap();
        assert_ne!(
            crate::identity::canonicalize_identity(&c, false, None)
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
    fn a_carry_sorts_after_every_pure_leaf() {
        // A `Pre` reads the step before and a `Carry` writes the step's state; neither orders the other by data flow,
        // so where the carry's value is a load the order alone decides. Load, carry, pre, write is what the lift
        // refuses (the state's store ahead of its load); the carry sorts after every pure leaf.
        let normal = crate::text::body_text(
            &normalize_uncached(&testing::body(
                "
                for a0 in 0..8:
                    for a1 in 0..4:
                        in0 = load U[a0, a1]
                        acc0[a1] <- in0 seed 0
                        v2 = pre acc0[a1]
                        out[a0, a1] = v2
                ",
            ))
            .unwrap(),
        )
        .unwrap();
        let lines: Vec<&str> = normal.lines().map(str::trim).collect();
        assert!(lines.last().unwrap().starts_with("acc0[a1] <-"), "{normal}");
        assert!(
            lines.iter().position(|l| l.starts_with("out[")).unwrap() < lines.len() - 1,
            "{normal}"
        );
    }

    #[test]
    fn the_operand_order_of_a_normal_form_is_final() {
        // A gate/up pair: the order ranks a product's operands through the numbering of the body it permuted, and a
        // rank lost there flipped the operands on every second normalization.
        let normal = normalize_uncached(&testing::body(
            "
            let f32 v0 = 1.0
            for a0 in 0..128:
                for a1 in 0..512:
                    for a2 in 0..512:
                        in0 = load wg[a1, a2]
                        in1 = load wu[a1, a2]
                        in2 = load x[a0, a2]
                        f32 v1 = multiply(in0, in2)
                        acc0 <- add(acc0, v1) over a2
                        f32 v2 = multiply(in1, in2)
                        acc1 <- add(acc1, v2) over a2
                    f16 v3 = copy(acc1)
                    f16 v4 = copy(acc0)
                    f32 v5 = copy(v4)
                    v6 = negative(v5)
                    v7 = exp(v6)
                    v8 = add(v7, v0)
                    f16 v9 = divide(v5, v8)
                    v10 = multiply(v3, v9)
                    o[a0, a1] = v10
            ",
        ))
        .unwrap();
        let again = normalize_uncached(&normal).unwrap();
        assert_eq!(body_text(&again).unwrap(), body_text(&normal).unwrap());
    }

    fn symbolic_axis(name: &str, extent: &str) -> std::sync::Arc<Axis> {
        Axis::new(
            name,
            Dim::new(parse_expression(extent).unwrap(), None),
            None,
        )
    }

    /// `((a0 * stride + a2) // stride) % seq_len`: the delinearized sequence coordinate a collapsed reshape composed
    /// with a matmul read carries.
    fn flat_seq_coord(stride: i64) -> Expression {
        parse_expression(&format!("((a0 * {stride} + a2) // {stride}) % seq_len")).unwrap()
    }

    /// The context the loop simplifier builds: `a0` over the symbolic `seq_len`, `a2` over the static `stride`.
    fn loop_ctx(stride: i64) -> SimplifyCtx {
        let ctx = extend_simplify_ctx(&SimplifyCtx::empty(), &symbolic_axis("a0", "seq_len"));
        extend_simplify_ctx(&ctx, &Axis::new("a2", Dim::static_(stride), None))
    }

    #[test]
    fn a_bare_var_modulo_its_symbolic_extent_folds() {
        let ctx = extend_simplify_ctx(&SimplifyCtx::empty(), &symbolic_axis("a0", "seq_len"));
        assert_eq!(
            parse_expression("a0 % seq_len")
                .unwrap()
                .simplify(&ctx)
                .unwrap(),
            Expression::var("a0")
        );
    }

    #[test]
    fn a_collapsed_seq_coordinate_folds_to_its_row() {
        for stride in [1024, 2048] {
            let folded = flat_seq_coord(stride).simplify(&loop_ctx(stride)).unwrap();
            assert_eq!(
                folded,
                Expression::var("a0"),
                "stride={stride}: got {}",
                folded.text()
            );
        }
        // The fold is a strength reduction, not a change of value.
        let original = flat_seq_coord(2048);
        let folded = original.simplify(&loop_ctx(2048)).unwrap();
        for seq_len in [7, 64, 512] {
            for a0 in 0..seq_len {
                for a2 in [0, 1, 1024, 2047] {
                    let env = [("a0", a0), ("a2", a2), ("seq_len", seq_len)];
                    assert_eq!(eval_int(&original, &env), a0);
                    assert_eq!(eval_int(&folded, &env), a0);
                }
            }
        }
    }

    #[test]
    fn a_modulo_without_a_bound_or_with_another_divisor_stays() {
        let e = parse_expression("b % seq_len").unwrap();
        let ranged = SimplifyCtx::empty().extend("b", Interval::new(0, 1 << 30), None);
        assert_eq!(e.simplify(&ranged).unwrap(), e);
        let ctx = extend_simplify_ctx(&SimplifyCtx::empty(), &symbolic_axis("a0", "seq_len"));
        let other = parse_expression("a0 % other").unwrap();
        assert_eq!(other.simplify(&ctx).unwrap(), other);
    }

    #[test]
    fn a_symbolic_axis_registers_its_sign_and_its_bound() {
        let ctx = extend_simplify_ctx(&SimplifyCtx::empty(), &symbolic_axis("a0", "seq_len"));
        assert_eq!(ctx.ranges[&Name::from("a0")].lo, 0);
        assert_eq!(ctx.bounds[&Name::from("a0")], Expression::var("seq_len"));
        let static_ctx = extend_simplify_ctx(
            &SimplifyCtx::empty(),
            &Axis::new("a", Dim::static_(32), None),
        );
        assert_eq!(static_ctx.ranges[&Name::from("a")], Interval::new(0, 31));
        assert_eq!(
            parse_expression("a % 32")
                .unwrap()
                .simplify(&static_ctx)
                .unwrap(),
            Expression::var("a")
        );
    }

    /// Two loads of one cell spelled with different digits of one index are one load in the normal form.
    #[test]
    fn two_digit_spellings_of_one_index_load_once() {
        for (a, b) in [
            ("2 * (i / 4) + (i / 2) % 2", "i / 2"),
            ("(i % 8) / 2", "(i / 2) % 4"),
            ("(i % 16) % 8", "i % 8"),
            ("2 * ((i / 2) % 4) + i % 2", "i % 8"),
            ("i - i % 8", "8 * (i / 8)"),
            ("i + j - i % 8", "8 * (i / 8) + j"),
            ("minimum(i, 63)", "i"),
        ] {
            let normal = normalize_uncached(&parse_body(&format!(
                "for i in 0..64:\n    for j in 0..8:\n        a = load X[{a}]\n        b = load X[{b}]\n        \
                 c = add(a, b)\n        O[i, j] = c"
            ))
            .unwrap())
            .unwrap();
            let listing = body_text(&normal).unwrap();
            assert_eq!(
                listing.matches("load X").count(),
                1,
                "{a} and {b}:\n{listing}"
            );
        }
    }

    #[test]
    fn a_scaled_term_quotient_folds_through_a_larger_divisor() {
        // `(h*128 + c) / 1024` with `c < 128` is `h / 8`: the collapsed GQA-group index of a decode sweep.
        let ctx = extend_simplify_ctx(
            &extend_simplify_ctx(
                &SimplifyCtx::empty(),
                &Axis::new("h", Dim::static_(64), None),
            ),
            &Axis::new("c", Dim::static_(128), None),
        );
        let quotient = parse_expression("(h * 128 + c) / 1024")
            .unwrap()
            .simplify(&ctx)
            .unwrap();
        assert_eq!(quotient, parse_expression("h / 8").unwrap());
        let remainder = parse_expression("(h * 128 + c) % 1024")
            .unwrap()
            .simplify(&ctx)
            .unwrap();
        assert_eq!(remainder, parse_expression("h % 8 * 128 + c").unwrap());
    }

    #[test]
    fn no_step_changes_a_prepared_body_the_split_leaves_alone() {
        // The driver runs the steps again only after the split changed the body, so a prepared body the split leaves
        // alone has to be one every step leaves alone.
        for given in [one_key_softmax(), equal_gather_index()] {
            let mut prepared = rename_ssa_sequential(&given);
            let mut settled = false;
            for _ in 0..8 {
                prepared = prepare_body(&prepared).unwrap();
                let split =
                    one_operation_per_definition(&fold_trivial_selects(&prepared), &HashSet::new());
                if split == prepared {
                    settled = true;
                    break;
                }
                prepared = split;
            }
            assert!(settled, "the split did not settle");
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
    use crate::statement::Statement;
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

    pub(crate) fn statement_types(body: &Body) -> Vec<&'static str> {
        body.stmts().iter().map(|stmt| stmt.ty().name()).collect()
    }

    /// Every order of `items`.
    pub(crate) fn permutations<T: Clone>(items: &[T]) -> Vec<Vec<T>> {
        if items.is_empty() {
            return vec![Vec::new()];
        }
        (0..items.len())
            .flat_map(|index| {
                let mut rest = items.to_vec();
                let first = rest.remove(index);
                permutations(&rest).into_iter().map(move |mut tail| {
                    tail.insert(0, first.clone());
                    tail
                })
            })
            .collect()
    }

    /// `e` evaluated over integer bindings.
    pub(crate) fn eval_int(e: &crate::expr::Expression, env: &[(&str, i64)]) -> i64 {
        let env: std::collections::HashMap<Name, crate::expr::Num> = env
            .iter()
            .map(|(name, value)| (Name::from(*name), crate::expr::Num::Int(*value)))
            .collect();
        e.eval(&env).unwrap().as_int().unwrap()
    }

    /// The body a loop holds, or a branch's taken arm.
    pub(crate) fn inner(stmt: &Statement) -> Body {
        match stmt.nested().as_slice() {
            [first, ..] => first.clone(),
            [] => panic!("not a statement over a body: {}", stmt.ty()),
        }
    }

    pub(crate) fn axis(stmt: &Statement) -> &str {
        match stmt {
            Statement::Loop(l) => &l.axis.name,
            Statement::StridedLoop(l) => &l.axis.name,
            _ => panic!("not a loop: {}", stmt.ty()),
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
