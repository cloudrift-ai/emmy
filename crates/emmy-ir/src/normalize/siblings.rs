//! Sibling reduce loops: reduce-axis names unified where load positions overlap, and loops over one axis merged.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use super::with_bodies;
use crate::axis::{Axis, Dim};
use crate::body::{Body, exposed_defines, free_names};
use crate::expr::{Expression, SimplifyCtx, affine_form};
use crate::numbering::{Key, Tag};
use crate::statement::*;
use crate::{Name, Result};

// --- sibling reduce loops ---------------------------------------------------------------------------------

pub(crate) const REDUCE_AXIS_PLACEHOLDER: &str = "__reduce_axis__";

/// The positions where a Load index within `body` walks the reduce axis, each keyed by how it walks the dimension.
pub(crate) fn reduce_axis_source_positions(body: &Body, axis: &Name) -> Result<BTreeSet<Key>> {
    let mut out = BTreeSet::new();
    for here in body.loads() {
        let Statement::Load(load) = &**here.stmt() else {
            continue;
        };
        for (dim, e) in load.index.iter().enumerate() {
            match affine_form(e, &|name| name == **axis) {
                Some((anchor, coeffs)) => {
                    let coeff = coeffs.get(axis).copied().unwrap_or(0);
                    if coeff != 0 {
                        out.insert(Key::tuple([
                            Key::tag(Tag::Affine),
                            Key::Str(load.input),
                            Key::Int(dim as i64),
                            Key::str(&anchor.text()),
                            Key::Int(coeff),
                        ]));
                    }
                }
                None if e.reads(*axis) => {
                    let spelled = e
                        .substitute(&HashMap::from([(
                            *axis,
                            Expression::var(REDUCE_AXIS_PLACEHOLDER),
                        )]))
                        .simplify(&SimplifyCtx::empty())?;
                    out.insert(Key::tuple([
                        Key::tag(Tag::Composite),
                        Key::Str(load.input),
                        Key::Int(dim as i64),
                        Key::str(&spelled.text()),
                    ]));
                }
                None => {}
            }
        }
    }
    Ok(out)
}

/// Rename reduce-axis vars across sibling reduce loops whose Load positions overlap, so they share one axis name.
pub fn unify_siblings(body: &Body) -> Result<Body> {
    let mut stmts: Vec<Arc<Statement>> = body.stmts().to_vec();
    let mut entries: Vec<(usize, Name, Expression, BTreeSet<Key>)> = Vec::new();
    for (i, s) in stmts.iter().enumerate() {
        if let Statement::Loop(l) = &**s
            && l.is_reduce()
        {
            let positions = reduce_axis_source_positions(&l.body, &l.axis.name)?;
            if !positions.is_empty() {
                entries.push((i, l.axis.name, l.axis.extent.expression.clone(), positions));
            }
        }
    }
    if entries.len() < 2 {
        return Ok(body.clone());
    }
    let mut parent: Vec<usize> = (0..entries.len()).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for a in 0..entries.len() {
        for b in a + 1..entries.len() {
            if entries[a].2 != entries[b].2 {
                continue;
            }
            if !entries[a].3.is_disjoint(&entries[b].3) {
                let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                if ra != rb {
                    parent[ra.max(rb)] = ra.min(rb);
                }
            }
        }
    }
    for k in 0..entries.len() {
        let root = find(&mut parent, k);
        let canonical = entries[root].1;
        let (index, axis_name, extent, _) = &entries[k];
        if canonical == *axis_name {
            continue;
        }
        let Statement::Loop(l) = &*stmts[*index] else {
            continue;
        };
        // A RENAME, binders and uses through one map, which is why it may pass through the very scopes it renames.
        let new_axis = Arc::new(Axis {
            name: canonical,
            extent: Dim {
                expression: extent.clone(),
                hint: l.axis.extent.hint,
            },
            window: l.axis.window.clone(),
        });
        let mapping = HashMap::from([(l.axis.name, canonical)]);
        let renamed = l.body.map_each(&mut |s| Statement::rename(s, &mapping));
        stmts[*index] = Arc::new(Statement::Loop(Loop {
            axis: new_axis,
            body: renamed,
            unroll: l.unroll,
            seed: l.seed,
        }));
    }
    Ok(body.rebuilt(stmts))
}

/// The names a loop over `body` still binds after it closes.
pub(crate) fn carried_out(body: &Body) -> HashSet<Name> {
    let mut out: HashSet<Name> = body
        .stmts()
        .iter()
        .filter_map(|s| match &**s {
            Statement::Accum(a) => Some(a.name),
            _ => None,
        })
        .collect();
    for stmt in body.stmts() {
        match &**stmt {
            Statement::Loop(l) if !l.seed => out.extend(carried_out(&l.body)),
            Statement::StridedLoop(l) if !l.seed => out.extend(carried_out(&l.body)),
            Statement::Cond(c) => {
                out.extend(carried_out(&c.body));
                out.extend(carried_out(&c.else_body));
            }
            _ => {}
        }
    }
    out
}

/// `body` with each name in `clashing` renamed to one neither side spells.
pub(crate) fn rename_apart(body: &Body, clashing: &HashSet<Name>, taken: &HashSet<Name>) -> Body {
    let mut used: HashSet<Name> = body.definitions().keys().cloned().collect();
    used.extend(body.ssa_uses().iter().cloned());
    used.extend(taken.iter().cloned());
    let mut mapping: HashMap<Name, Name> = HashMap::new();
    let mut sorted: Vec<&Name> = clashing.iter().collect();
    sorted.sort();
    for name in sorted {
        let fresh = (1..)
            .map(|n| Name::from(format!("{name}__m{n}")))
            .find(|candidate| !used.contains(candidate))
            .unwrap_or(*name);
        used.insert(fresh);
        mapping.insert(*name, fresh);
    }
    body.map_each(&mut |s| Statement::rename(s, &mapping))
}

pub(crate) fn merge_sibling_reduce_loops_here(body: &Body) -> Body {
    let items = body.stmts();
    if items.len() < 2 {
        return body.clone();
    }
    let mut out: Vec<Arc<Statement>> = Vec::new();
    let mut consumed: HashSet<usize> = HashSet::new();
    for (i, s) in items.iter().enumerate() {
        if consumed.contains(&i) {
            continue;
        }
        let Statement::Loop(first) = &**s else {
            out.push(s.clone());
            continue;
        };
        if !first.is_reduce() {
            out.push(s.clone());
            continue;
        }
        let mut merged = first.clone();
        let mut absorbed = false;
        for j in i + 1..items.len() {
            if consumed.contains(&j) {
                continue;
            }
            let Statement::Loop(t) = &*items[j] else {
                continue;
            };
            if !(t.is_reduce()
                && t.axis.name == merged.axis.name
                && t.axis.extent == merged.axis.extent
                && t.unroll == merged.unroll
                && t.seed == merged.seed)
            {
                continue;
            }
            // ONE reading of what the incoming loop needs from around it: the names it reads and does not bind.
            let reads: HashSet<Name> = free_names(&items[j]).into_iter().collect();
            let merged_defs: HashSet<Name> = merged.body.definitions().keys().cloned().collect();
            if !merged_defs.is_disjoint(&reads) {
                continue;
            }
            let incoming = t.body.clone();
            // What is left of the shared spellings is a COLLISION, not a dependence: rename the incoming copy apart,
            // unless the loop still binds the name after it closes.
            let clashing: HashSet<Name> = merged_defs
                .intersection(&incoming.definitions().keys().cloned().collect())
                .cloned()
                .collect();
            if !clashing.is_disjoint(&carried_out(&incoming)) {
                continue;
            }
            let mut between_defs: HashSet<Name> = HashSet::new();
            for (k, between) in items.iter().enumerate().take(j).skip(i + 1) {
                if !consumed.contains(&k) {
                    between_defs.extend(exposed_defines(between));
                }
            }
            if !between_defs.is_disjoint(&reads) {
                continue;
            }
            let renamed = rename_apart(&incoming, &clashing, &merged_defs);
            merged = Loop {
                axis: merged.axis.clone(),
                body: merged.body.concat(renamed.stmts()),
                unroll: merged.unroll,
                seed: merged.seed,
            };
            consumed.insert(j);
            absorbed = true;
        }
        out.push(if absorbed {
            Arc::new(Statement::Loop(merged))
        } else {
            s.clone()
        });
    }
    body.rebuilt(out)
}

/// Merge sibling reduce loops with one axis into one loop whose body is the concatenation; parents before children.
pub fn merge_sibling_reduce_loops(body: &Body) -> Result<Body> {
    let merged = merge_sibling_reduce_loops_here(&unify_siblings(body)?);
    merged.try_map_each(&mut |s| {
        let nested = s.nested();
        if nested.is_empty() {
            return Ok(s.clone());
        }
        let bodies: Result<Vec<Body>> = nested.iter().map(merge_sibling_reduce_loops).collect();
        Ok(with_bodies(s, bodies?))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize::normalize_body;
    use crate::normalize::testing::*;

    fn lines<S: AsRef<str>>(parts: &[S]) -> String {
        parts
            .iter()
            .map(AsRef::as_ref)
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The text of a loop `header` over `parts`, each a statement or a nested loop's text.
    fn looped<S: AsRef<str>>(header: &str, parts: &[S]) -> String {
        let mut out = vec![header.to_string()];
        out.extend(parts.iter().map(|part| indented(part.as_ref())));
        out.join("\n")
    }

    /// One reduce loop `acc += W[k, n] * x[k]`, its names suffixed by `out_acc` so two compose with disjoint
    /// definitions.
    fn matmul_reduce(axis: &str, extent: usize, w_buffer: &str, out_acc: &str) -> String {
        looped(
            &format!("for {axis} in 0..{extent}:"),
            &[
                format!("w_{out_acc} = load {w_buffer}[{axis}, n]"),
                format!("x_{out_acc} = load x[{axis}]"),
                format!("m_{out_acc} = multiply(w_{out_acc}, x_{out_acc})"),
                format!("{out_acc} <- add({out_acc}, m_{out_acc})"),
            ],
        )
    }

    fn unified(text: &str) -> Body {
        unify_siblings(&body(text)).unwrap()
    }

    fn merged(text: &str) -> Body {
        merge_sibling_reduce_loops(&body(text)).unwrap()
    }

    #[test]
    fn unify_groups_loops_with_shared_load_position() {
        // Two reduce loops over different axis names, both indexing `x` at the same dimension, share one axis name.
        let out = unified(&lines(&[
            matmul_reduce("a2", 16, "Wg", "acc0"),
            matmul_reduce("a3", 16, "Wu", "acc1"),
        ]));
        assert_eq!(loop_axes(&out), ["a2", "a2"]);
    }

    #[test]
    fn unify_skips_when_extents_differ() {
        // Overlap on (source, dimension) is necessary, not sufficient: different extents are different iteration
        // spaces.
        let out = unified(&lines(&[
            matmul_reduce("a2", 16, "Wg", "acc0"),
            matmul_reduce("a3", 32, "Wu", "acc1"),
        ]));
        assert_eq!(loop_axes(&out), ["a2", "a3"]);
    }

    #[test]
    fn unify_skips_disjoint_load_positions() {
        let out = unified(
            "
            for a2 in 0..16:
                p = load A[a2]
                acc0 <- add(acc0, p)
            for a3 in 0..16:
                q = load B[a3]
                acc1 <- add(acc1, q)
            ",
        );
        assert_eq!(loop_axes(&out), ["a2", "a3"]);
    }

    #[test]
    fn unify_transitively_groups_three_loops() {
        // A overlaps B on (x, 0) and B overlaps C on (y, 0), so all three take one name.
        let out = unified(
            "
            for a in 0..8:
                p = load x[a]
                acc_a <- add(acc_a, p)
            for b in 0..8:
                q = load x[b]
                r = load y[b]
                s = multiply(q, r)
                acc_b <- add(acc_b, s)
            for c in 0..8:
                t = load y[c]
                acc_c <- add(acc_c, t)
            ",
        );
        assert_eq!(loop_axes(&out), ["a", "a", "a"]);
    }

    #[test]
    fn merge_concatenates_matching_reduce_bodies() {
        let out = merged(&lines(&[
            matmul_reduce("k", 16, "Wg", "acc0"),
            matmul_reduce("k", 16, "Wu", "acc1"),
            "out[n] = acc1".to_string(),
        ]));
        assert_eq!(loop_axes(&out).len(), 1, "{}", text(&out));
        assert_eq!(accum_names(&inner(&out.stmts()[0])), ["acc0", "acc1"]);
    }

    #[test]
    fn merge_skips_when_second_loop_reads_first_loops_accum() {
        // Softmax: the second reduce reads the first's finished accumulator inside its iteration. Merged, the read
        // would see the accumulator in flight.
        let out = merged(
            "
            for k in 0..16:
                x1 = load X[k]
                acc_max <- maximum(acc_max, x1)
            for k in 0..16:
                x2 = load X[k]
                d = subtract(x2, acc_max)
                e = exp(d)
                acc_sum <- add(acc_sum, e)
            ",
        );
        assert_eq!(loop_axes(&out).len(), 2, "sequential reduces stay distinct");
    }

    #[test]
    fn merge_renames_a_colliding_local_rather_than_refusing() {
        // Two bodies binding one name for unrelated values is a collision, not a dependence, and it is what
        // alpha-equal copies of a single cone always look like. The incoming body's local is renamed apart.
        let out = merged(
            "
            for k in 0..16:
                v = load A[k]
                acc0 <- add(acc0, v)
            for k in 0..16:
                v = load B[k]
                acc1 <- add(acc1, v)
            ",
        );
        assert_eq!(loop_axes(&out).len(), 1);
        let loop_body = inner(&out.stmts()[0]);
        let loads: HashMap<&str, &str> = loop_body
            .stmts()
            .iter()
            .filter_map(|stmt| match &**stmt {
                Statement::Load(l) => Some((&*l.input, &*l.names[0])),
                _ => None,
            })
            .collect();
        let accums: HashMap<&str, &str> = loop_body
            .stmts()
            .iter()
            .filter_map(|stmt| match &**stmt {
                Statement::Accum(a) => Some((&*a.name, a.value.as_var().map_or("", |n| &**n))),
                _ => None,
            })
            .collect();
        assert_ne!(loads["A"], loads["B"], "each half keeps its own binding");
        assert_eq!(accums["acc0"], loads["A"]);
        assert_eq!(accums["acc1"], loads["B"]);
    }

    #[test]
    fn merge_refuses_a_collision_on_a_name_the_loop_still_binds() {
        // A carrier of the incoming loop's own scope is declared ahead of that loop and read after it, so a rename
        // could not reach its readers.
        let out = merged(
            "
            for k in 0..16:
                p = load A[k]
                acc <- add(acc, p)
            for k in 0..16:
                q = load B[k]
                acc <- add(acc, q)
            out[n] = acc
            ",
        );
        assert_eq!(loop_axes(&out).len(), 2);
    }

    /// An outer reduce over `k` folding an inner reduce over `d` of `tag[k, d]` into `out_<tag>`.
    fn half(tag: &str) -> String {
        looped(
            "for k in 0..16:",
            &[
                looped(
                    "for d in 0..8:",
                    &[
                        format!("v = load {tag}[k, d]"),
                        "acc0 <- add(acc0, v)".to_string(),
                    ],
                ),
                format!("out_{tag} <- add(out_{tag}, acc0)"),
            ],
        )
    }

    #[test]
    fn merge_renames_an_accum_that_stays_inside() {
        // An accumulator bound in a nested loop crosses that loop's boundary but is consumed within this body:
        // nothing outside reads it, so it renames apart like any other local.
        let out = merged(&lines(&[half("A"), half("B")]));
        assert_eq!(loop_axes(&out).len(), 1);
    }

    #[test]
    fn unify_groups_loops_indexed_affinely_in_the_reduce_axis() {
        // A blocked reduce reads its stream at `outer * B + inner`; the axis still walks that dimension.
        let blocked: Vec<String> = ["i", "j"]
            .iter()
            .map(|name| {
                looped(
                    &format!("for {name} in 0..64:"),
                    &[
                        format!("v_{name} = load x[o * 64 + {name}]"),
                        format!("acc_{name} <- add(acc_{name}, v_{name})"),
                    ],
                )
            })
            .collect();
        let out = unified(&lines(&blocked));
        assert_eq!(loop_axes(&out), ["i", "i"]);
    }

    #[test]
    fn unify_keeps_apart_blocks_at_different_offsets() {
        // `o * B + i` and `o * B + 32 + j` walk different halves of one dimension; the anchor is on the key.
        let at = |name: &str, extra: usize| {
            let shifted = if extra == 0 {
                "o * 64".to_string()
            } else {
                format!("o * 64 + {extra}")
            };
            looped(
                &format!("for {name} in 0..32:"),
                &[
                    format!("v_{name} = load x[{shifted} + {name}]"),
                    format!("acc_{name} <- add(acc_{name}, v_{name})"),
                ],
            )
        };
        let out = unified(&lines(&[at("i", 0), at("j", 32)]));
        assert_eq!(loop_axes(&out), ["i", "j"]);
    }

    /// A reduce loop `acc += W[k] * bits[index(k)]` reading one byte-packed stream `bits` through `index` of its
    /// axis, the shape a 4-bit activation is read in. `w_buffer` names the weight, so two instances compute
    /// distinct values.
    fn packed_stream_reduce(name: &str, index: &dyn Fn(&str) -> String, w_buffer: &str) -> String {
        looped(
            &format!("for {name} in 0..256:"),
            &[
                format!("w_{name} = load {w_buffer}[{name}]"),
                format!("v_{name} = load bits[{}]", index(name)),
                format!("m_{name} = multiply(w_{name}, v_{name})"),
                format!("acc_{name} <- add(acc_{name}, m_{name})"),
            ],
        )
    }

    /// The byte holding logical element `k` of a stream packed two codes per byte in blocks of 16.
    fn two_codes_per_byte(k: &str) -> String {
        format!("{k} / 16 * 8 + {k} % 16 / 2")
    }

    #[test]
    fn unify_groups_loops_indexed_through_one_composite_expression() {
        // A packed stream is read through div and mod of the axis, which is not affine in it. Two siblings reading
        // one stream through the same expression of their own axis walk the same dimension.
        let out = unified(&lines(&[
            packed_stream_reduce("i", &two_codes_per_byte, "W"),
            packed_stream_reduce("j", &two_codes_per_byte, "W"),
        ]));
        assert_eq!(loop_axes(&out), ["i", "i"]);
    }

    #[test]
    fn unify_keeps_apart_different_composite_expressions() {
        // The composite key is the whole expression: `bits[k / 16]` and `bits[k % 16]` walk different things.
        let out = unified(&lines(&[
            packed_stream_reduce("i", &|k| format!("{k} / 16"), "Wg"),
            packed_stream_reduce("j", &|k| format!("{k} % 16"), "Wu"),
        ]));
        assert_eq!(loop_axes(&out), ["i", "j"]);
    }

    #[test]
    fn unify_then_merge_collapses_two_readers_of_one_packed_stream() {
        // The two-channel contraction a fused gate/up pair over a 4-bit activation lowers to.
        let out = normalize_body(&body(&lines(&[
            packed_stream_reduce("i", &two_codes_per_byte, "Wg"),
            packed_stream_reduce("j", &two_codes_per_byte, "Wu"),
        ])))
        .unwrap();
        assert_eq!(loop_axes(&out).len(), 1);
        let loop_body = inner(&out.stmts()[0]);
        assert_eq!(accum_names(&loop_body).len(), 2);
        let stream_reads = loop_body
            .stmts()
            .iter()
            .filter(|stmt| matches!(&***stmt, Statement::Load(l) if &*l.input == "bits"))
            .count();
        assert_eq!(stream_reads, 1, "the shared stream read is one load");
    }

    #[test]
    fn merge_skips_when_between_stmt_def_used_by_second_loop() {
        // A pure statement between the two loops defines a name the second loop's body reads.
        let out = merged(
            "
            for k in 0..16:
                p = load A[k]
                acc0 <- add(acc0, p)
            bias = exp(acc0)
            for k in 0..16:
                q = load B[k]
                m = multiply(q, bias)
                acc1 <- add(acc1, m)
            ",
        );
        assert_eq!(loop_axes(&out).len(), 2);
    }

    #[test]
    fn merge_keeps_intermediate_pure_stmts_in_place() {
        // A pure statement between the loops that reads only the first accumulator stays valid after the merge.
        let out = merged(&lines(&[
            matmul_reduce("k", 16, "Wg", "acc0"),
            "silu_val = silu(acc0)".to_string(),
            matmul_reduce("k", 16, "Wu", "acc1"),
            "result = multiply(silu_val, acc1)".to_string(),
            "out[n] = result".to_string(),
        ]));
        assert_eq!(loop_axes(&out).len(), 1);
        assert_eq!(accum_names(&inner(&out.stmts()[0])), ["acc0", "acc1"]);
    }

    #[test]
    fn merge_recurses_into_nested_loops() {
        let out = merged(&looped(
            "for n in 0..4:",
            &[
                matmul_reduce("k", 16, "Wg", "acc0"),
                matmul_reduce("k", 16, "Wu", "acc1"),
                "o[n] = acc1".to_string(),
            ],
        ));
        assert_eq!(loop_axes(&out), ["n"]);
        assert_eq!(loop_axes(&inner(&out.stmts()[0])).len(), 1);
    }

    #[test]
    fn unify_then_merge_collapses_gated_mlp_pattern() {
        // `silu(x @ Wg) * (x @ Wu)` enters with distinct axis names indexing the same `x` slot. One loop carries
        // both reductions and the multiply after it still reads both accumulators.
        let out = merged(&lines(&[
            matmul_reduce("a2", 16, "Wg", "acc0"),
            "silu_val = silu(acc0)".to_string(),
            matmul_reduce("a3", 16, "Wu", "acc1"),
            "result = multiply(silu_val, acc1)".to_string(),
            "out[n] = result".to_string(),
        ]));
        assert_eq!(loop_axes(&out).len(), 1);
        let loop_body = inner(&out.stmts()[0]);
        assert_eq!(accum_names(&loop_body), ["acc0", "acc1"]);
        let x_loads = loop_body
            .stmts()
            .iter()
            .filter(|stmt| matches!(&***stmt, Statement::Load(l) if &*l.input == "x"))
            .count();
        assert_eq!(
            x_loads, 2,
            "placement shares the loads; the merge alone leaves both"
        );
    }

    #[test]
    fn normalize_closes_reductions_exposed_by_hoisting() {
        // A reduction hoisted beside an existing sibling merges in the same normalization.
        let normalized = normalize_body(&body(
            "
            for i in 0..2:
                for k0 in 0..4:
                    x0 = load X[k0]
                    a <- add(a, x0)
                A[i] = a
            for k1 in 0..4:
                x1 = load X[k1]
                b <- add(b, x1)
            B[] = b
            ",
        ))
        .unwrap();
        let reduce_loops: Vec<&Arc<Statement>> = normalized
            .stmts()
            .iter()
            .filter(|stmt| matches!(&***stmt, Statement::Loop(l) if l.is_reduce()))
            .collect();
        assert_eq!(reduce_loops.len(), 1, "{}", text(&normalized));
        assert_eq!(inner(reduce_loops[0]).loads().len(), 1);
        assert_eq!(
            normalize_body(&Body::new(normalized.stmts().to_vec())).unwrap(),
            normalized
        );
    }

    #[test]
    fn merge_collapses_three_alpha_equal_siblings() {
        // Merging is pairwise, so a third copy folds into the pair already merged.
        let out = merged(&lines(&[half("A"), half("B"), half("C")]));
        assert_eq!(loop_axes(&out).len(), 1);
    }

    /// One inner fold of a blocked twisted carrier: re-derive the score over the block, weigh it against the block
    /// pivot, then accumulate this channel's own contribution. Every channel is an alpha-equal copy of one cone
    /// down to the spellings, and each recomputes the score, which is the recompute the merge exists to remove.
    fn blocked_channel(block: &str, tag: &str, tail: &[&str]) -> String {
        let mut parts = vec![
            looped(
                "for a3 in 0..32:",
                &[
                    "in1 = load x0[a3]".to_string(),
                    format!("in2 = load x1[a2_o * 64 + {block}, a3]"),
                    "v0 = multiply(in1, in2)".to_string(),
                    "acc0 <- add(acc0, v0)".to_string(),
                ],
            ),
            "v1 = multiply(acc0, scale)".to_string(),
            "acc1__o__gn = maximum(acc1__blk, v1)".to_string(),
            "acc1__o__dg_o = subtract(v1, acc1__o__gn)".to_string(),
            "acc1__o__beta = exp(acc1__o__dg_o)".to_string(),
        ];
        parts.extend(tail.iter().map(|line| line.to_string()));
        parts.push(format!("{tag}__blk <- add({tag}__blk, {tag}__blk__e)"));
        looped(&format!("for {block} in 0..64:"), &parts)
    }

    #[test]
    fn merge_collapses_the_channel_loops_of_a_blocked_twisted_carrier() {
        // The shape blocking attention's carrier produces: a block pivot, then one inner loop per channel, all
        // over the same block. The pivot stays apart, since the weight reads its finished accumulator, and the
        // channels collapse, which takes `Q·K` from three passes over the block to two. The enclosing scope
        // re-binds the very temps the channels use, so a merge gate that asks whether a name is read anywhere
        // around the pair, instead of what the loop itself still binds, refuses here.
        let pivot = looped(
            "for a2_p in 0..64:",
            &[
                looped(
                    "for a3 in 0..32:",
                    &[
                        "in1 = load x0[a3]",
                        "in2 = load x1[a2_o * 64 + a2_p, a3]",
                        "v0 = multiply(in1, in2)",
                        "acc0 <- add(acc0, v0)",
                    ],
                ),
                "acc1__blk__e = multiply(acc0, scale)".to_string(),
                "acc1__blk <- add(acc1__blk, acc1__blk__e)".to_string(),
            ],
        );
        let expectation = blocked_channel(
            "a2_c1",
            "acc5__sum",
            &[
                "in7 = load x2[a2_o * 64 + a2_p, a5]",
                "acc5__sum__blk__e = multiply(acc1__o__beta, in7)",
            ],
        );
        let denominator = blocked_channel("a2_c2", "acc3", &["acc3__blk__e = copy(acc1__o__beta)"]);
        let combine = [
            "acc1__o__gn = maximum(acc1, acc1__blk)",
            "acc1__o__dg_o = subtract(acc1__blk, acc1__o__gn)",
            "acc1__o__beta = exp(acc1__o__dg_o)",
            "acc5__sum <- add(acc5__sum, acc5__sum__blk)",
            "acc3 <- add(acc3, acc3__blk)",
            "acc1 <- add(acc1, acc1__o__gn)",
        ];
        let mut parts = vec![pivot, expectation, denominator];
        parts.extend(combine.iter().map(|line| line.to_string()));
        let out = merged(&lines(&parts));
        let loops: Vec<&Arc<Statement>> = out
            .stmts()
            .iter()
            .filter(|stmt| matches!(&***stmt, Statement::Loop(_)))
            .collect();
        assert_eq!(
            loops.len(),
            2,
            "the pivot stays apart; the two channels merge"
        );
        assert_eq!(accum_names(&inner(loops[0])), ["acc1__blk"]);
        assert_eq!(
            accum_names(&inner(loops[1])),
            ["acc5__sum__blk", "acc3__blk"]
        );
        let channel = inner(loops[1]);
        let score_loops: Vec<&Arc<Statement>> = channel
            .stmts()
            .iter()
            .filter(|stmt| matches!(&***stmt, Statement::Loop(_)))
            .collect();
        assert_eq!(score_loops.len(), 1);
        assert_eq!(
            inner(score_loops[0]).accums().len(),
            2,
            "both score states share one inner loop"
        );
    }
}
