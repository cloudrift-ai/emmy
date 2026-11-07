//! A body's identity: the digest of its value-numbered scope tree, with every external buffer keyed by its type and
//! its use instead of its spelling, and the roles its buffers fill.

use std::collections::HashMap;
use std::sync::Arc;

use crate::body::{Body, MemoKey};
use crate::kinds::elementwise::{Elementwise, Op};
use crate::kinds::{Assign, Statement};
use crate::memo::BODY_MEMO_CAPACITY;
use crate::normalize::normalize_body;
use crate::numbering::{digest, hex};
use crate::{Name, Result, memo};

/// What an identity was computed under: the clustered reading or the exact one, and each buffer's color.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct IdentityKey {
    pub cluster: bool,
    pub colors: Option<Vec<(Name, String)>>,
}

/// A body's identity material: its key and the external buffers in canonical rank order (`arguments[i]` fills role
/// `i`).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Identity {
    pub key: String,
    pub arguments: Vec<Name>,
}

/// Replace each statement's operation with its compute-unit cluster representative. The result is digest material
/// and must not be executed.
fn canonicalize_op_clusters(body: &Body) -> Body {
    body.map_stmts(&mut |stmt| {
        Some(match &**stmt {
            // A sum is FMA-class work; which coefficients it carries is not the cluster's business.
            Statement::Affine(a) => Arc::new(Statement::Assign(Assign {
                name: a.name.clone(),
                op: Op::Known(Elementwise::Add).cluster_representative(),
                args: a.args.clone(),
                dtype: a.dtype,
            })),
            other => match other.op() {
                Some(op) => {
                    let representative = op.cluster_representative();
                    if representative != *op {
                        Arc::new(other.with_op(representative))
                    } else {
                        stmt.clone()
                    }
                }
                None => stmt.clone(),
            },
        })
    })
}

memo! {
    /// The identities of the normal forms met in this process, per coloring: a cut's arms lower the same pieces
    /// again and again.
    pub fn identities() -> Memo<(MemoKey, IdentityKey), Identity> [capacity = BODY_MEMO_CAPACITY, policy = S3Fifo]
}

/// Key the normal form of `body`, optionally with operations collapsed to compute-unit clusters. `colors` types
/// each external buffer, so differently typed roles never share a rank.
pub fn canonicalize_identity(
    body: &Body,
    cluster: bool,
    colors: Option<&HashMap<Name, String>>,
) -> Result<Identity> {
    let mut stmts = normalize_body(body)?;
    if cluster {
        stmts = normalize_body(&canonicalize_op_clusters(&stmts))?;
    }
    let colors_key = colors.map(|c| {
        let mut sorted: Vec<(Name, String)> =
            c.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        sorted.sort();
        sorted
    });
    let key = IdentityKey {
        cluster,
        colors: colors_key,
    };
    identities().get_or_try_insert(&(MemoKey(stmts.clone()), key), || {
        let (tree, arguments) = digest(&stmts, colors);
        Ok(Identity {
            key: hex(tree),
            arguments,
        })
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::normalize::testing::*;

    fn identity_key(given: &Body, cluster: bool) -> String {
        canonicalize_identity(given, cluster, None).unwrap().key
    }

    /// The exact identity of the body `source` spells.
    fn key(source: &str) -> String {
        identity_key(&body(source), false)
    }

    /// The identity with operations collapsed to their compute-unit clusters.
    fn clustered_key(source: &str) -> String {
        identity_key(&body(source), true)
    }

    /// The text of the normal form of the body `source` spells.
    fn normal(source: &str) -> String {
        text(&normalize_body(&body(source)).unwrap())
    }

    /// The text of one loop over `blocks`, each of which may span several lines.
    fn looped<S: AsRef<str>>(axis: &str, extent: usize, blocks: &[S]) -> String {
        let lines: Vec<&str> = blocks.iter().map(AsRef::as_ref).collect();
        format!(
            "for {axis} in 0..{extent}:\n{}",
            indented(&lines.join("\n"))
        )
    }

    /// Every order of `items`.
    /// A tiny multiply-and-write body over the buffers named.
    fn matmul(input_x: &str, input_y: &str, output: &str) -> String {
        format!(
            "
            for a in 0..4:
                x = load {input_x}[a]
                y = load {input_y}[a]
                z = multiply(x, y)
                {output}[a] = z
            "
        )
    }

    /// One pointwise kernel with an authored load order over row and column arguments.
    fn pointwise(inputs: &[(&str, &str, &str)], op: &str, args: &[&str]) -> String {
        let mut lines: Vec<String> = inputs
            .iter()
            .map(|(value, buffer, index)| format!("{value} = load {buffer}[{index}]"))
            .collect();
        lines.push(format!("result = {op}({})", args.join(", ")));
        lines.push("output[row, column] = result".into());
        looped("row", 4, &[looped("column", 4, &lines)])
    }

    /// One operation over two loaded operands.
    fn binary(op: &str, (left, right): (&str, &str)) -> String {
        format!(
            "
            for a in 0..4:
                x = load X[a]
                y = load Y[a]
                z = {op}({left}, {right})
                O[a] = z
            "
        )
    }

    #[test]
    fn equivalent_affine_indices_and_repeated_loads_have_one_exact_identity() {
        let mut keys = HashSet::new();
        let mut normals = HashSet::new();
        for (i, j) in [("i", "j"), ("z", "q")] {
            let forms = [
                format!("{i} + 2 * {j} + 3"),
                format!("3 + {j} + ({i} + {j})"),
                format!("3 * ({i} + {j} + 1) - (2 * {i} + {j})"),
            ];
            for first in &forms {
                for second in &forms {
                    let source = format!(
                        "
                        for {i} in 0..4:
                            for {j} in 0..4:
                                a = load x[{first}]
                                b = load x[{second}]
                                left = exp(a)
                                right = exp(b)
                                sum = add(right, left)
                                out[{i}, {j}] = sum
                        "
                    );
                    keys.insert(key(&source));
                    normals.insert(normal(&source));
                }
            }
        }
        assert_eq!(keys.len(), 1);
        assert_eq!(normals.len(), 1);
        let only = normals.into_iter().next().unwrap();
        let lines: Vec<&str> = only.lines().map(str::trim).collect();
        let loads = lines
            .iter()
            .filter(|line| line.contains(" = load "))
            .count();
        let computed = lines
            .iter()
            .filter(|line| {
                !line.starts_with("for ") && !line.contains(" = load ") && !line.starts_with("out[")
            })
            .count();
        assert_eq!(loads, 1, "{only}");
        assert_eq!(computed, 2, "{only}");
        assert_eq!(normal(&only), only);
    }

    #[test]
    fn renamed_buffers_share_an_identity_but_not_a_normal_form() {
        let a = matmul("X", "Y", "O");
        let b = matmul("foo", "bar", "baz");
        assert_ne!(normal(&a), normal(&b));
        assert_eq!(clustered_key(&a), clustered_key(&b));
    }

    #[test]
    fn swapped_commutative_arguments_share_a_normal_form_and_an_identity() {
        let xy = "
            for a in 0..4:
                lx = load X[a]
                ly = load Y[a]
                z = add(lx, ly)
                O[a] = z
            ";
        let yx = "
            for a in 0..4:
                ly = load Y[a]
                lx = load X[a]
                z = add(ly, lx)
                O[a] = z
            ";
        assert_eq!(normal(xy), normal(yx));
        assert_eq!(clustered_key(xy), clustered_key(yx));
    }

    /// A Q·K-style product has one identity whether the trace encounters Q or K first.
    #[test]
    fn a_product_keys_alike_whichever_operand_and_load_comes_first() {
        let query_first = pointwise(
            &[
                ("query", "query_buffer", "row"),
                ("key", "key_buffer", "column"),
            ],
            "multiply",
            &["query", "key"],
        );
        let key_first = pointwise(
            &[
                ("renamed_key", "renamed_key_buffer", "column"),
                ("renamed_query", "renamed_query_buffer", "row"),
            ],
            "multiply",
            &["renamed_key", "renamed_query"],
        );
        assert_eq!(key(&query_first), key(&key_first));
    }

    #[test]
    fn every_valid_order_of_independent_operations_has_one_identity() {
        let load = "input = load input_buffer[element]";
        let absolute = "absolute = abs(input)";
        let negated = "negated = negative(input)";
        let combine = "result = add(absolute, negated)";
        let write = "output[element] = result";
        let at = |order: &[&str], line: &str| order.iter().position(|here| *here == line).unwrap();
        let valid: Vec<Vec<&str>> = permutations(&[load, absolute, negated, combine])
            .into_iter()
            .filter(|order| {
                at(order, load) < at(order, absolute)
                    && at(order, absolute) < at(order, combine)
                    && at(order, load) < at(order, negated)
                    && at(order, negated) < at(order, combine)
            })
            .collect();
        let bodies: Vec<String> = valid
            .iter()
            .map(|order| looped("element", 4, &[&order[..], &[write]].concat()))
            .collect();
        assert_eq!(valid.len(), 2);
        assert_eq!(
            bodies
                .iter()
                .map(|given| normal(given))
                .collect::<HashSet<_>>()
                .len(),
            1
        );
        assert_eq!(
            bodies
                .iter()
                .map(|given| key(given))
                .collect::<HashSet<_>>()
                .len(),
            1
        );
    }

    /// Argument roles, not same-shaped loads or their names, fix the canonical buffer order.
    #[test]
    fn permuted_same_shape_arguments_share_an_identity() {
        let entries = [("x", "X", "row"), ("y", "Y", "row"), ("z", "Z", "row")];
        let keys: HashSet<String> = permutations(&entries)
            .iter()
            .map(|order| key(&pointwise(order, "where", &["x", "y", "z"])))
            .collect();
        assert_eq!(keys.len(), 1);
    }

    /// Use roles tell equal producers apart without relying on which duplicate came first.
    #[test]
    fn reordered_duplicate_producers_share_an_identity() {
        let load = "input = load input_buffer[element]";
        let first = "first = abs(input)";
        let second = "second = abs(input)";
        let tail = [
            "exponential = exp(first)",
            "logarithm = log(second)",
            "result = subtract(exponential, logarithm)",
            "output[element] = result",
        ];
        assert_eq!(
            key(&looped(
                "element",
                4,
                &[&[load, first, second][..], &tail[..]].concat()
            )),
            key(&looped(
                "element",
                4,
                &[&[load, second, first][..], &tail[..]].concat()
            )),
        );
    }

    /// An unrelated write does not pin a pure definition to one side of it.
    #[test]
    fn pure_work_moved_across_an_unrelated_write_keeps_its_identity() {
        let load = "x = load X[element]";
        let absolute = "absolute = abs(x)";
        let exponential = "exponential = exp(x)";
        let write_a = "A[element] = absolute";
        let write_b = "B[element] = exponential";
        let one = looped(
            "element",
            4,
            &[load, absolute, write_a, exponential, write_b],
        );
        let two = looped(
            "element",
            4,
            &[load, absolute, exponential, write_a, write_b],
        );
        assert_eq!(key(&one), key(&two));
    }

    /// Buffer role discovery cannot invent aliases among unrelated chains.
    #[test]
    fn reordered_independent_read_write_chains_share_an_identity() {
        let chain_one = ["scalar = load Scalar[]", "Vector[element] = scalar"];
        let chain_two = [
            "element_value = load Input[element]",
            "ScalarOutput[] = element_value",
        ];
        assert_eq!(
            key(&looped("element", 4, &[chain_one, chain_two].concat())),
            key(&looped("element", 4, &[chain_two, chain_one].concat())),
        );
    }

    /// Writes to distinct buffers are dependency-independent effects.
    #[test]
    fn reordered_writes_to_distinct_buffers_share_an_identity() {
        let prefix = [
            "x = load X[element]",
            "absolute = abs(x)",
            "exponential = exp(x)",
        ];
        let write_a = "A[element] = absolute";
        let write_b = "B[element] = exponential";
        assert_eq!(
            key(&looped(
                "element",
                4,
                &[&prefix[..], &[write_a, write_b]].concat()
            )),
            key(&looped(
                "element",
                4,
                &[&prefix[..], &[write_b, write_a]].concat()
            )),
        );
    }

    /// Updates of distinct reduction states have no ordering edge.
    #[test]
    fn reordered_updates_of_distinct_accumulators_share_an_identity() {
        let loads = ["x = load X[reduce]", "y = load Y[reduce]"];
        let accum_x = "sum_x <- add(sum_x, x) over reduce";
        let accum_y = "sum_y <- add(sum_y, y) over reduce";
        let writes = "A[] = sum_x\nB[] = sum_y";
        let one = looped("reduce", 4, &[&loads[..], &[accum_x, accum_y]].concat()) + "\n" + writes;
        let two = looped("reduce", 4, &[&loads[..], &[accum_y, accum_x]].concat()) + "\n" + writes;
        assert_eq!(key(&one), key(&two));
    }

    /// Reduction-axis metadata is set-like, not an authored-order constraint.
    #[test]
    fn reduction_axes_are_a_set_but_which_axes_reduce_is_identity() {
        let make = |axes: &str| {
            format!(
                "
                for row in 0..4:
                    for column in 0..4:
                        x = load X[row, column]
                        total <- add(total, x) over {axes}
                O[] = total
                "
            )
        };
        let per_row = "
            for row in 0..4:
                for column in 0..4:
                    x = load X[row, column]
                    total <- add(total, x) over column
                O[row] = total
            ";
        assert_eq!(key(&make("row, column")), key(&make("column, row")));
        assert_ne!(key(&make("row, column")), key(per_row));
    }

    /// Axis spelling cannot decide a loop order when scalar output geometry does not.
    #[test]
    fn ambiguous_free_axes_key_alike_whatever_their_spelling() {
        let make = |outer: &str, inner: &str| {
            format!(
                "
                for {outer} in 0..2:
                    for {inner} in 0..3:
                        x = load X[{outer}, {inner}]
                        O[] = x
                "
            )
        };
        assert_eq!(normal(&make("z", "a")), normal(&make("a", "z")));
        assert_eq!(key(&make("z", "a")), key(&make("a", "z")));
    }

    /// A composed batch coordinate stays outside matrix rows and columns after alpha-renaming.
    #[test]
    fn a_batch_axis_stays_outside_the_output_rows_and_columns() {
        let make = |[batch, row, column]: [&str; 3], order: [usize; 3]| {
            let index = format!("{batch} // 3, {batch} % 3, {row}, {column}");
            let axes = [(batch, 6), (row, 128), (column, 128)];
            order.iter().rev().fold(
                format!("x = load X[{index}]\nO[{index}] = x"),
                |inner, &position| {
                    let (axis, extent) = axes[position];
                    looped(axis, extent, &[inner])
                },
            )
        };
        let first = normal(&make(["batch", "row", "column"], [0, 1, 2]));
        let second = normal(&make(["b", "m", "n"], [2, 0, 1]));
        assert_eq!(first, second);
        // The three loops nest in that order: each header one level deeper, the batch outermost.
        let headers: Vec<&str> = first.lines().take(3).collect();
        let depths: Vec<usize> = headers
            .iter()
            .map(|header| header.find("for ").unwrap())
            .collect();
        let extents: Vec<&str> = headers
            .iter()
            .map(|header| header.split(" in ").nth(1).unwrap())
            .collect();
        assert_eq!(depths, [0, 4, 8], "{first}");
        assert_eq!(extents, ["0..6:", "0..128:", "0..128:"], "{first}");
    }

    /// Axis and parent provenance names cannot choose the order of otherwise ambiguous loops.
    #[test]
    fn axis_and_parent_names_do_not_order_ambiguous_windowed_loops() {
        let make = |[first, second]: [&str; 2], [a, b]: [&str; 2], reverse: bool| {
            let loops = [
                format!("for {first} in 0..4 window(parent={a} in 0..8, base={a}, bound={a} + 4):"),
                format!(
                    "for {second} in 0..4 window(parent={b} in 0..8, base={b} + 4, bound={b} + 8, partition=true):"
                ),
            ];
            let (outer, inner) = if reverse {
                (&loops[1], &loops[0])
            } else {
                (&loops[0], &loops[1])
            };
            format!("{outer}\n    {inner}\n        x = load X[]\n        O[] = x")
        };
        assert_eq!(
            key(&make(["outer", "inner"], ["source_a", "source_b"], false)),
            key(&make(
                ["renamed_outer", "renamed_inner"],
                ["renamed_a", "renamed_b"],
                true
            )),
        );
    }

    /// The key includes codegen-relevant fields excluded from equality.
    #[test]
    fn axis_windows_key_apart_though_the_bodies_compare_equal() {
        let make = |window: &str| {
            body(&format!(
                "
                for element in 0..4 window({window}):
                    x = load X[element]
                    O[element] = x
                "
            ))
        };
        let first = make("parent=source in 0..8, base=0, bound=4");
        let second = make("parent=other_source in 0..8, base=4, bound=8, partition=true");
        assert_eq!(first, second);
        assert_ne!(identity_key(&first, false), identity_key(&second, false));
    }

    /// An expected size guides tuning but does not change the kernel's structural identity.
    #[test]
    fn a_symbolic_size_hint_does_not_change_the_identity() {
        let make = |hint: usize| {
            format!(
                "
                for element in 0..size~{hint}:
                    x = load X[element]
                    O[element] = x
                "
            )
        };
        assert_eq!(key(&make(32)), key(&make(64)));
    }

    /// Every axis reference follows its binder through conditions and reduction metadata.
    #[test]
    fn renamed_condition_and_reduction_axes_share_an_identity() {
        let make = |element: &str, reduce: &str, predicate: &str, value: &str, total: &str| {
            format!(
                "
                for {element} in 0..4:
                    {predicate} = load Mask[{element}]
                    for {reduce} in 0..4:
                        {value} = load X[{element}, {reduce}]
                        {total} <- add({total}, {value}) over {reduce}
                    if {predicate}:
                        O[{element}] = {total}
                "
            )
        };
        assert_eq!(
            key(&make("row", "k", "predicate", "value", "total")),
            key(&make(
                "renamed_row",
                "renamed_k",
                "renamed_predicate",
                "renamed_value",
                "renamed_total"
            )),
        );
    }

    /// Several exported states are allocated in body order, never by the order of their source names.
    #[test]
    fn renamed_exported_accumulators_share_an_identity() {
        let make = |total: &str, maximum: &str| {
            format!(
                "
                for reduce in 0..4:
                    x = load X[reduce]
                    y = load Y[reduce]
                    absolute = abs(x)
                    exponential = exp(y)
                    {total} <- add({total}, absolute) over reduce
                    {maximum} <- maximum({maximum}, exponential) over reduce
                Sum[] = {total}
                Max[] = {maximum}
                "
            )
        };
        assert_eq!(
            key(&make("total", "maximum")),
            key(&make("alpha_ssa_0", "alpha_ssa_1"))
        );
    }

    /// Identity canonicalizes commutative expressions and comparison duals.
    #[test]
    fn a_commuted_index_and_a_dual_comparison_share_a_normal_form_and_an_identity() {
        let make = |index: &str, condition: &str| {
            format!(
                "
                for element in 0..8:
                    x = load X[{index}]
                    if {condition}:
                        O[element] = x
                "
            )
        };
        let left = make("element + 1", "element < 4");
        let right = make("1 + element", "4 > element");
        assert_eq!(normal(&left), normal(&right));
        assert_eq!(key(&left), key(&right));
    }

    /// A loop that reads `X` at `index` and stores the value in a scalar.
    fn indexed(index: &str) -> String {
        format!(
            "
            for element in 0..8:
                x = load X[{index}]
                O[] = x
            "
        )
    }

    /// Constant folding and association do not split equivalent affine addresses.
    #[test]
    fn equivalent_affine_index_spellings_share_an_identity() {
        assert_eq!(
            key(&indexed("element + 1 + 2")),
            key(&indexed("element + 3"))
        );
        assert_eq!(
            key(&indexed("element * 2 * 3")),
            key(&indexed("element * 6"))
        );
        assert_eq!(key(&indexed("element - 1")), key(&indexed("element + -1")));
    }

    #[test]
    fn the_operands_of_a_subtraction_in_an_index_are_not_reordered() {
        assert_ne!(key(&indexed("element - 1")), key(&indexed("1 - element")));
    }

    /// Affine coordinate folding cannot collapse a floating-point SSA operation tree.
    #[test]
    fn floating_point_arithmetic_on_values_is_not_reassociated() {
        let make = |value: &str| {
            format!(
                "
                x = load X[]
                if {value} < 0.0:
                    O[] = x
                "
            )
        };
        assert_ne!(key(&make("x + 1.0 + 2.0")), key(&make("x + 3.0")));
    }

    /// Every statement that defines a name takes part in alpha-renaming.
    #[test]
    fn renamed_constants_and_inits_share_an_identity() {
        let make = |constant: &str, initial: &str| {
            format!(
                "
                let {constant} = 2.0
                init f32 {initial} = 0
                result = add({constant}, {initial})
                O[] = result
                "
            )
        };
        assert_eq!(
            key(&make("constant", "initial")),
            key(&make("renamed_constant", "renamed_initial"))
        );
    }

    /// The lanes a vector load binds are alpha-renamed together.
    #[test]
    fn renamed_vector_load_lanes_share_an_identity() {
        let make = |first: &str, second: &str| {
            format!(
                "
                {first}, {second} = load X[]
                result = add({first}, {second})
                O[] = result
                "
            )
        };
        assert_eq!(key(&make("x0", "x1")), key(&make("renamed0", "renamed1")));
    }

    /// A spelling reused in sibling scopes still denotes two lexical binders.
    #[test]
    fn sibling_scopes_that_reuse_local_names_key_as_distinct_binders() {
        let make = |axes: [&str; 2], values: [&str; 2], buffers: [&str; 2], outputs: [&str; 2]| {
            (0..2)
                .map(|side| {
                    let (axis, value, buffer, output) =
                        (axes[side], values[side], buffers[side], outputs[side]);
                    let op = ["abs", "exp"][side];
                    looped(
                        axis,
                        4,
                        &[
                            format!("{value} = load {buffer}[{axis}]"),
                            format!("{value}_result = {op}({value})"),
                            format!("{output}[{axis}] = {value}_result"),
                        ],
                    )
                })
                .collect::<Vec<_>>()
        };
        let reused = make(["i", "i"], ["x", "x"], ["A", "B"], ["OA", "OB"]);
        let distinct = make(
            ["left_axis", "right_axis"],
            ["left_value", "right_value"],
            ["left_input", "right_input"],
            ["left_output", "right_output"],
        );
        let mut renamed_and_reordered = make(
            ["renamed_left_axis", "renamed_right_axis"],
            ["renamed_left_value", "renamed_right_value"],
            ["renamed_left_input", "renamed_right_input"],
            ["renamed_left_output", "renamed_right_output"],
        );
        renamed_and_reordered.reverse();
        let keys: HashSet<String> = [reused, distinct, renamed_and_reordered]
            .iter()
            .map(|loops| key(&loops.join("\n")))
            .collect();
        assert_eq!(keys.len(), 1);
    }

    /// Two ambiguous child scopes do not multiply their order choices or leak alpha names.
    #[test]
    fn nested_scope_orders_are_canonicalized_independently() {
        let child = |axis: &str,
                     stem: &str,
                     source: &str,
                     outputs: [&str; 2],
                     operations: [&str; 2],
                     reverse: bool| {
            let value = format!("{stem}_input");
            let mut chains: Vec<[String; 2]> = (0..2)
                .map(|index| {
                    [
                        format!("{stem}_value_{index} = {}({value})", operations[index]),
                        format!("{}[{axis}] = {stem}_value_{index}", outputs[index]),
                    ]
                })
                .collect();
            if reverse {
                chains.reverse();
            }
            let lines: Vec<String> = std::iter::once(format!("{value} = load {source}[{axis}]"))
                .chain(chains.into_iter().flatten())
                .collect();
            format!("if {axis} < 4:\n{}", indented(&lines.join("\n")))
        };
        let make = |axis: &str, reverse_scopes: bool, reverse_left: bool, reverse_right: bool| {
            let mut scopes = [
                child(
                    axis,
                    "left",
                    "X",
                    ["A0", "A1"],
                    ["abs", "negative"],
                    reverse_left,
                ),
                child(
                    axis,
                    "right",
                    "Y",
                    ["B0", "B1"],
                    ["exp", "reciprocal"],
                    reverse_right,
                ),
            ];
            if reverse_scopes {
                scopes.reverse();
            }
            looped(axis, 4, &scopes)
        };
        let normalized: HashSet<String> = (0..16)
            .map(|bits| [8, 4, 2, 1].map(|bit| bits & bit != 0))
            .map(|[renamed, reverse_scopes, reverse_left, reverse_right]| {
                let axis = if renamed { "renamed_axis" } else { "element" };
                normal(&make(axis, reverse_scopes, reverse_left, reverse_right))
            })
            .collect();
        assert_eq!(normalized.len(), 1);
    }

    /// A nested order cannot depend on the source spelling of an outer SSA value.
    #[test]
    fn outer_captures_are_canonicalized_before_a_nested_order() {
        let make = |left: &str| {
            body(&format!(
                "
                {left} = load X[]
                right = load Y[]
                if 1:
                    left_abs = abs({left})
                    right_abs = abs(right)
                    result = add(left_abs, right_abs)
                    O[] = result
                "
            ))
        };
        let canonical = normalize_body(&make("x")).unwrap();
        assert_eq!(normalize_body(&make("renamed_x")).unwrap(), canonical);
        assert_eq!(normalize_body(&canonical).unwrap(), canonical);
        assert!(normalize_body(&canonical).unwrap().ptr_eq(&canonical));
    }

    #[test]
    fn a_copy_alias_ahead_of_a_sibling_scope_keys_alike_whatever_the_names() {
        let make = |reuse_names: bool| {
            let right_axis = if reuse_names { "i" } else { "j" };
            let right_input = if reuse_names { "x" } else { "right_input" };
            let right_result = if reuse_names { "y" } else { "right_result" };
            format!(
                "
                for i in 0..4:
                    x = load A[i]
                    y = copy(x)
                    OA[i] = y
                for {right_axis} in 0..4:
                    {right_input} = load B[{right_axis}]
                    {right_result} = exp({right_input})
                    OB[{right_axis}] = {right_result}
                "
            )
        };
        assert_eq!(key(&make(true)), key(&make(false)));
    }

    /// Exact symmetry does not trigger factorial buffer, producer, or axis searches.
    #[test]
    fn large_symmetric_partitions_key_without_a_factorial_search() {
        let names = |stem: &str| {
            (0..9)
                .map(|i| format!("{stem}{i}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let loads = (0..9).map(|i| format!("x{i} = load X{i}[element]"));
        let buffers = looped(
            "element",
            4,
            &loads
                .chain([
                    format!("result = add({})", names("x")),
                    "O[element] = result".into(),
                ])
                .collect::<Vec<_>>(),
        );
        let producers = (0..9).map(|i| format!("p{i} = abs(x)"));
        let duplicate_producers = looped(
            "element",
            4,
            &std::iter::once("x = load X[element]".to_string())
                .chain(producers)
                .chain([
                    format!("result = add({})", names("p")),
                    "O[element] = result".into(),
                ])
                .collect::<Vec<_>>(),
        );
        let nested = (0..9)
            .rev()
            .fold("x = load X[]\nO[] = x".to_string(), |inner, i| {
                looped(&format!("axis{i}"), 2, &[inner])
            });
        let keys: HashSet<String> = [buffers, duplicate_producers, nested]
            .iter()
            .map(|source| key(source))
            .collect();
        assert_eq!(keys.len(), 3);
    }

    /// Graph context splits a tie between local forms before the exact ordering fallback.
    #[test]
    fn a_large_asymmetric_sibling_partition_is_split_by_graph_context() {
        let make = |prefix: &str, reverse: bool| {
            let constants = (0..9).map(|index| format!("let {prefix}_source_{index} = {index}.0"));
            let mut consumers: Vec<String> = (0..9)
                .map(|index| {
                    format!("if 1:\n    {prefix}_value_{index} = abs({prefix}_source_{index})")
                })
                .collect();
            if reverse {
                consumers.reverse();
            }
            constants.chain(consumers).collect::<Vec<_>>().join("\n")
        };
        assert_eq!(
            normal(&make("first", false)),
            normal(&make("renamed", true))
        );
    }

    /// Ready blocks stay ahead of leaf epilogues so normalization does not widen scheduling.
    #[test]
    fn independent_sibling_loops_stay_ahead_of_their_epilogues() {
        let normal_text = normal(
            "
            for first_axis in 0..4:
                first_input = load X[first_axis]
                first_sum <- add(first_sum, first_input) over first_axis
            first_result = abs(first_sum)
            for second_axis in 0..4:
                second_input = load Y[second_axis]
                second_sum <- add(second_sum, second_input) over second_axis
            second_result = exp(second_sum)
            O0[] = first_result
            O1[] = second_result
            ",
        );
        let top: Vec<&str> = normal_text
            .lines()
            .filter(|line| !line.starts_with(' '))
            .collect();
        assert!(
            top[0].starts_with("for ") && top[1].starts_with("for "),
            "{normal_text}"
        );
    }

    /// Statement order is free; the operand order of an exact subtract remains identity.
    #[test]
    fn reordered_subtraction_operands_key_apart() {
        let inputs = [
            ("left", "left_buffer", "row"),
            ("right", "right_buffer", "column"),
        ];
        let left_minus_right = pointwise(&inputs, "subtract", &["left", "right"]);
        let right_minus_left = pointwise(&inputs, "subtract", &["right", "left"]);
        assert_ne!(key(&left_minus_right), key(&right_minus_left));
    }

    /// A pure statement spelled twice over the same names is one value: normalization keeps the first and aliases the
    /// second, so the two spellings are one kernel and one key.
    #[test]
    fn a_computation_spelled_twice_keys_as_one_value() {
        let repeated = "
            for element in 0..4:
                input = load input_buffer[element]
                first = abs(input)
                second = abs(input)
                result = add(first, second)
                output[element] = result
            ";
        let shared = "
            for element in 0..4:
                input = load input_buffer[element]
                first = abs(input)
                result = add(first, first)
                output[element] = result
            ";
        assert_eq!(key(repeated), key(shared));
    }

    /// Writes to the same buffer retain their order even when their operands are independent.
    #[test]
    fn writes_to_one_buffer_keep_their_order() {
        let prefix = [
            "x = load X[element]",
            "y = load Y[element]",
            "absolute = abs(x)",
            "exponential = exp(y)",
        ];
        let write_x = "output[element] = absolute";
        let write_y = "output[element] = exponential";
        assert_ne!(
            key(&looped(
                "element",
                4,
                &[&prefix[..], &[write_x, write_y]].concat()
            )),
            key(&looped(
                "element",
                4,
                &[&prefix[..], &[write_y, write_x]].concat()
            )),
        );
    }

    #[test]
    fn a_read_keeps_its_place_across_a_write_to_its_buffer() {
        let prefix = "old = load B[]\nlet replacement = 7.0\nP[] = old";
        let write = "B[] = replacement";
        let read = "new = load B[]";
        let observe = "O[] = new";
        let after_write = [prefix, write, read, observe].join("\n");
        let before_write = [prefix, read, write, observe].join("\n");
        assert_ne!(key(&after_write), key(&before_write));
    }

    /// Updates to one reduction state retain their execution order.
    #[test]
    fn updates_of_one_accumulator_keep_their_order() {
        let prefix = [
            "x = load X[reduce]",
            "y = load Y[reduce]",
            "absolute = abs(x)",
            "exponential = exp(y)",
        ];
        let absolute = "total <- add(total, absolute) over reduce";
        let exponential = "total <- add(total, exponential) over reduce";
        let one = looped(
            "reduce",
            4,
            &[&prefix[..], &[absolute, exponential]].concat(),
        ) + "\nO[] = total";
        let two = looped(
            "reduce",
            4,
            &[&prefix[..], &[exponential, absolute]].concat(),
        ) + "\nO[] = total";
        assert_ne!(key(&one), key(&two));
    }

    /// A state read remains after the update whose value it observes.
    #[test]
    fn a_read_of_an_accumulator_stays_after_the_update_it_observes() {
        let prefix =
            "init f32 total = 0\nlet first = 1.0\nlet second = 2.0\ntotal <- add(total, first)";
        let final_update = "total <- add(total, second)";
        let observe = "O[] = total";
        let after_both = [prefix, final_update, observe].join("\n");
        let between = [prefix, observe, final_update].join("\n");
        assert_ne!(key(&after_both), key(&between));
    }

    /// A reduction seed changes execution and must remain in exact identity.
    #[test]
    fn the_seed_policy_of_a_reduction_is_identity() {
        let make = |flags: &str| {
            format!(
                "
                for reduce in 0..4{flags}:
                    x = load X[reduce]
                    total <- add(total, x) over reduce
                O[] = total
                "
            )
        };
        assert_ne!(key(&make("")), key(&make(" seed=false")));
    }

    /// Two argument names and one argument used twice are different signatures.
    #[test]
    fn one_buffer_read_twice_keys_apart_from_two_buffers() {
        let separate = pointwise(
            &[("left", "X", "row"), ("right", "Y", "column")],
            "multiply",
            &["left", "right"],
        );
        let aliased = pointwise(
            &[("left", "X", "row"), ("right", "X", "column")],
            "multiply",
            &["left", "right"],
        );
        assert_ne!(key(&separate), key(&aliased));
    }

    /// The FMA cluster's representative is `add`: bodies that differ only in which FMA-issued operation sits at one
    /// position are equivalent for the autotune search.
    #[test]
    fn the_fma_operations_share_a_clustered_identity() {
        let keys: HashSet<String> = ["add", "subtract", "multiply", "negative"]
            .iter()
            .map(|op| clustered_key(&binary(op, ("x", "y"))))
            .collect();
        assert_eq!(keys.len(), 1);
    }

    #[test]
    fn the_division_operations_share_a_clustered_identity() {
        let keys: HashSet<String> = ["divide", "true_divide", "floor_divide", "remainder", "mod"]
            .iter()
            .map(|op| clustered_key(&binary(op, ("x", "y"))))
            .collect();
        assert_eq!(keys.len(), 1);
    }

    #[test]
    fn normalization_keeps_the_operation_that_clustering_collapses() {
        let add = binary("add", ("x", "y"));
        let subtract = binary("subtract", ("x", "y"));
        assert_ne!(normal(&add), normal(&subtract));
        assert_eq!(clustered_key(&add), clustered_key(&subtract));
    }

    /// Clustering does not collapse the FMA cluster into a special-function one.
    #[test]
    fn operations_of_different_clusters_key_apart() {
        let keys: HashSet<String> = ["add", "divide", "exp", "maximum"]
            .iter()
            .map(|op| clustered_key(&binary(op, ("x", "y"))))
            .collect();
        assert_eq!(keys.len(), 4);
    }

    /// Clustering folds the non-commutative `subtract` into the commutative `add`, so swapped operands sort to one
    /// form: the autotune search treats `x - y` and `y - x` as one kernel shape.
    #[test]
    fn a_clustered_subtraction_keys_alike_with_its_operands_swapped() {
        assert_eq!(
            clustered_key(&binary("subtract", ("x", "y"))),
            clustered_key(&binary("subtract", ("y", "x"))),
        );
    }

    /// Identity is a fixed point of normalization: the normal form keys as the body it came from.
    #[test]
    fn a_normal_form_has_the_identity_of_its_body() {
        let given = body(&matmul("X", "Y", "O"));
        let canonical = canonicalize_identity(&given, false, None).unwrap();
        let normal_form = normalize_body(&given).unwrap();
        assert_eq!(
            canonicalize_identity(&normal_form, false, None).unwrap(),
            canonical
        );
        let arguments: HashSet<&str> = canonical.arguments.iter().map(|name| &**name).collect();
        assert_eq!(arguments, HashSet::from(["X", "Y", "O"]));
    }

    /// Renaming axes across lexical order must not re-key a saved kernel's measured evidence.
    #[test]
    fn renamed_axes_keep_the_coordinate_order_of_the_identity() {
        let mut keys = HashSet::new();
        for (outer, inner) in [("a9", "a10"), ("outer", "inner"), ("i", "j")] {
            let given = body(&format!(
                "
                for {outer} in 0..40:
                    for {inner} in 0..128:
                        v = load X[128 * {outer} + {inner}]
                        Y[{outer}, {inner}] = v
                "
            ));
            let canonical = identity_key(&given, false);
            let normal_form = normalize_body(&given).unwrap();
            assert_eq!(identity_key(&normal_form, false), canonical);
            let normal_text = text(&normal_form);
            assert_eq!(normal(&normal_text), normal_text);
            keys.insert(canonical);
        }
        assert_eq!(keys.len(), 1);
    }
}
