//! Dependency order for statement bodies: the constraints every sibling order must respect (a definition ahead of
//! its reads, a buffer's write ahead of the reads and writes that follow it, a carried state's updates in sequence,
//! a barrier pinned relative to every sibling), and the stable dependency sort that runs ahead of normalization.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::Name;
use crate::Result;
use crate::body::{Body, free_ssa, loop_parts, ordered_sibling_defs};
use crate::protocol::carried_cells;
use crate::statement::Statement;

/// The buffers a statement reads and writes and the carried states it updates, over its whole subtree.
fn resources(stmt: &Statement) -> (HashSet<Name>, HashSet<Name>, HashSet<Name>) {
    let nested = stmt.nested();
    let members: Vec<Arc<Statement>> = if nested.is_empty() {
        vec![Arc::new(stmt.clone())]
    } else {
        nested.iter().flat_map(|body| body.iter_stmts()).collect()
    };
    let mut reads = HashSet::new();
    let mut writes = HashSet::new();
    let mut state = HashSet::new();
    for member in &members {
        reads.extend(member.external_reads());
        writes.extend(member.external_writes());
        match &**member {
            Statement::Accum(a) => {
                state.insert(a.name);
            }
            Statement::Carry(c) => {
                state.insert(c.name);
            }
            _ => {}
        }
    }
    if let Statement::Init(init) = stmt {
        state.insert(init.name);
    }
    (reads, writes, state)
}

/// A no-dataflow, non-pure leaf: an ordered execution protocol (a barrier, an async commit), pinned relative to
/// every sibling.
pub fn is_protocol(stmt: &Statement) -> bool {
    !stmt.is_pure()
        && stmt.nested().is_empty()
        && stmt.defines().is_empty()
        && stmt.deps().is_empty()
}

/// Dependency and, when requested, effect predecessors for one lexical scope. A name bound more than once in the
/// scope reads lexically: its bindings keep their source order, a read binds the definition before it, or the first one
/// when none precedes it (a scope is order-free), and precedes every later binding of the name. An accumulator is read
/// where it stands: ahead of its update the step before, behind it the updated value. A state carried through `Carry`
/// orders nothing in its scope: a `Pre` reads the step before wherever it sits.
pub fn ordering_constraints(body: &Body, effects: bool) -> Vec<HashSet<usize>> {
    let stmts = body.stmts();
    let defs_uses: Vec<(Vec<Name>, HashSet<Name>)> = stmts
        .iter()
        .map(|stmt| (ordered_sibling_defs(stmt), free_ssa(stmt)))
        .collect();
    let mut definitions: HashMap<&Name, Vec<usize>> = HashMap::new();
    for (index, (defines, _)) in defs_uses.iter().enumerate() {
        for name in defines {
            definitions.entry(name).or_default().push(index);
        }
    }
    // How each name's sites define it: every site an accumulator's update, every site a carried state's next value, or
    // bindings (a producer among them: the loop that folds or carries the name).
    let site_kinds: Vec<HashMap<Name, Site>> = stmts.iter().map(|stmt| site_kinds(stmt)).collect();
    let kinds: HashMap<&Name, Site> = definitions
        .iter()
        .map(|(name, sites)| {
            let kind_at = |site: &usize| {
                site_kinds[*site]
                    .get(*name)
                    .copied()
                    .unwrap_or(Site::Binding)
            };
            let first = kind_at(&sites[0]);
            let kind = if sites.iter().all(|site| kind_at(site) == first) {
                first
            } else {
                Site::Binding
            };
            (*name, kind)
        })
        .collect();
    // A carried state orders nothing in its scope: a `Pre` reads the step before, whichever side of the update it
    // sits on. The loop that carries the state produces it, and its readers follow that loop.
    let sites_of = |name: &Name| {
        if kinds.get(name) == Some(&Site::Carry) {
            return &[][..];
        }
        definitions.get(name).map_or(&[][..], |v| &v[..])
    };
    // A read binds the update or binding before it. Ahead of every one: an accumulator reads the step before, a binding
    // that reads its own name reads the name outside the scope, and any other read binds the first binding (a scope
    // is order-free).
    let defining_stmt = |name: &Name, consumer: usize| -> Option<usize> {
        let sites = sites_of(name);
        if let Some(preceding) = sites.iter().rev().find(|site| **site < consumer) {
            return Some(*preceding);
        }
        if kinds.get(name) == Some(&Site::Accumulator) || sites.contains(&consumer) {
            return None;
        }
        sites.first().copied()
    };
    let mut incoming: Vec<HashSet<usize>> = Vec::with_capacity(stmts.len());
    for (index, (_, uses)) in defs_uses.iter().enumerate() {
        incoming.push(
            uses.iter()
                .filter_map(|name| defining_stmt(name, index))
                .collect(),
        );
    }
    // The bindings of one name keep their source order: a binding that reads forward still sits where it was written.
    for name in definitions.keys() {
        for pair in sites_of(name).windows(2) {
            incoming[pair[1]].insert(pair[0]);
        }
    }
    // A read precedes every update or binding of its name after the one it binds: a value read between two stays
    // between them, a read ahead of every binding sits between the first and the second, and a read of an accumulator
    // ahead of its update stays ahead of it.
    for (reader, (_, uses)) in defs_uses.iter().enumerate() {
        for name in uses {
            let from = defining_stmt(name, reader).unwrap_or(reader);
            for &later in sites_of(name) {
                if later > from && later != reader {
                    incoming[later].insert(reader);
                }
            }
        }
    }
    if effects {
        let accesses: Vec<_> = stmts.iter().map(|stmt| resources(stmt)).collect();
        let mut last_write: HashMap<Name, usize> = HashMap::new();
        let mut readers: HashMap<Name, HashSet<usize>> = HashMap::new();
        let mut last_state: HashMap<Name, usize> = HashMap::new();
        for (index, (reads, writes, state)) in accesses.iter().enumerate() {
            for name in reads {
                if let Some(&writer) = last_write.get(name) {
                    incoming[index].insert(writer);
                }
                if !writes.contains(name) {
                    readers.entry(*name).or_default().insert(index);
                }
            }
            for name in writes {
                if let Some(&writer) = last_write.get(name) {
                    incoming[index].insert(writer);
                }
                if let Some(earlier) = readers.remove(name) {
                    incoming[index].extend(earlier);
                }
                last_write.insert(*name, index);
            }
            for name in state {
                if let Some(&previous) = last_state.get(name) {
                    incoming[index].insert(previous);
                }
                last_state.insert(*name, index);
            }
        }
        let protocol: HashSet<usize> = (0..stmts.len())
            .filter(|&index| is_protocol(&stmts[index]))
            .collect();
        let mut preceding: Vec<usize> = Vec::new();
        let mut previous_protocol: Option<usize> = None;
        for (index, sources) in incoming.iter_mut().enumerate() {
            if protocol.contains(&index) {
                sources.extend(preceding.iter().copied());
                if let Some(previous) = previous_protocol {
                    sources.insert(previous);
                }
                preceding.clear();
                previous_protocol = Some(index);
            } else {
                if let Some(previous) = previous_protocol {
                    sources.insert(previous);
                }
                preceding.push(index);
            }
        }
    }
    incoming
}

/// How a sibling statement defines a name: as a binding or a producer, which its readers follow; as an update of an
/// accumulator, read where it stands; or as the next value of a carried state within its step, which orders nothing.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Site {
    Binding,
    Accumulator,
    Carry,
}

/// How `stmt` defines each name it exposes, in one walk of its subtree. A leaf `Accum` or `Carry`, or a block holding
/// one, is the update it holds, unless the block holds the loops that fold the accumulator (the loops on the path to
/// it bind every one of its axes, or any loop when it names none) or carries the state (`carried_cells`) — that block
/// produces the value and its readers follow it. A fold whose outer axis is bound outside the block is an update of the
/// block's scope, whatever loops the block holds. A name the walk meets no update of is a binding.
pub(crate) fn site_kinds(stmt: &Statement) -> HashMap<Name, Site> {
    fn walk(
        stmt: &Statement,
        bound: &HashSet<Name>,
        under_loop: bool,
        kinds: &mut HashMap<Name, Site>,
    ) {
        match stmt {
            Statement::Accum(a) => {
                let folded = if a.axes.is_empty() {
                    under_loop
                } else {
                    a.axes.iter().all(|axis| bound.contains(axis))
                };
                kinds.insert(
                    a.name,
                    if folded {
                        Site::Binding
                    } else {
                        Site::Accumulator
                    },
                );
            }
            Statement::Carry(c) => {
                kinds.insert(c.name, Site::Carry);
            }
            _ => {
                let is_loop = loop_parts(stmt).is_some();
                let mut inner = bound.clone();
                inner.extend(stmt.binds_axes());
                for body in stmt.nested() {
                    for child in body.stmts() {
                        walk(child, &inner, under_loop || is_loop, kinds);
                    }
                }
                if let Some((_, body)) = loop_parts(stmt) {
                    for (state, _) in carried_cells(body) {
                        kinds.insert(state, Site::Binding);
                    }
                }
            }
        }
    }
    let mut kinds = HashMap::new();
    walk(stmt, &HashSet::new(), false, &mut kinds);
    kinds
}

/// Stable recursive dependency sort used before structural normalization. A name bound twice in one scope, which
/// motion can produce, keeps its lexical reading: the reads between the two bindings stay between them. A scope whose
/// bindings and reads wait on each other is an error.
pub fn topological_sort(body: &Body) -> Result<Body> {
    let sorted = body.try_map_each(&mut |stmt| {
        let nested = stmt.nested();
        if nested.is_empty() {
            return Ok(stmt.clone());
        }
        let bodies: Result<Vec<Body>> = nested.iter().map(topological_sort).collect();
        Ok(Statement::rebuild_bodies(stmt, bodies?))
    })?;
    sorted.topological_order(&ordering_constraints(&sorted, false))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::ordering_constraints;
    use crate::body::Body;
    use crate::identity::canonicalize_identity;
    use crate::normalize::testing::*;
    use crate::normalize::{normalize_body, normalize_uncached};

    /// Every order of `items`.
    #[test]
    fn every_order_of_a_small_body_normalizes_to_one_form() {
        let statements = body(
            "
            x = load X[]
            absolute = abs(x)
            y = load Y[]
            result = add(absolute, y)
            O[] = result
            ",
        );
        let forms: Vec<Body> = permutations(statements.stmts())
            .into_iter()
            .map(|order| normalize_body(&Body::new(order)).unwrap())
            .collect();
        assert_eq!(forms.len(), 120);
        assert!(forms.iter().all(|form| *form == forms[0]));
    }

    #[test]
    fn the_executable_order_ignores_the_iteration_order_of_a_statements_buffers() {
        // Two projections of one normalized input, symmetric but for their weight buffer: which one comes first must
        // not follow the order a set of buffer names iterates in, which every set's own hash keys decide. A fused
        // two-projection serving kernel re-spelled on every boot. Each normal form is taken afresh, so every one
        // runs over sets with keys of their own.
        let projection = |weight: &str| {
            let k = format!("k_{weight}");
            format!(
                "
        for {k} in 0..8:
            x_{weight} = load X[i, {k}]
            n_{weight} = load N[{k}]
            w_{weight} = load {weight}[{k}, j]
            s_{weight} = multiply(scale, x_{weight})
            t_{weight} = multiply(n_{weight}, s_{weight})
            p_{weight} = multiply(t_{weight}, w_{weight})
            acc_{weight} <- add(acc_{weight}, p_{weight}) over {k}"
            )
        };
        let weights = ["W_up", "W_gate"];
        let spelled = format!(
            "for i in 0..4:\n    for j in 0..4:{}{}",
            weights.map(projection).concat(),
            weights
                .map(|w| format!("\n        O_{w}[i, j] = acc_{w}"))
                .concat()
        );
        let forms: HashSet<String> = (0..8)
            .map(|_| text(&normalize_uncached(&body(&spelled)).unwrap()))
            .collect();
        assert_eq!(forms.len(), 1);
    }

    #[test]
    fn effect_constraints_keep_only_intervening_resource_hazards() {
        let writes = body(
            &(0..100)
                .map(|i| format!("X[] = value_{i}"))
                .collect::<Vec<_>>()
                .join("\n"),
        );
        let incoming = ordering_constraints(&writes, true);
        assert_eq!(incoming[0], HashSet::new());
        for (index, sources) in incoming.iter().enumerate().skip(1) {
            assert_eq!(*sources, HashSet::from([index - 1]));
        }
        let incoming = ordering_constraints(
            &body(
                "
                X[] = first
                left = load X[]
                right = load X[]
                X[] = second
                tail = load X[]
                ",
            ),
            true,
        );
        let expected: Vec<HashSet<usize>> = [vec![], vec![0], vec![0], vec![0, 1, 2], vec![3]]
            .into_iter()
            .map(HashSet::from_iter)
            .collect();
        assert_eq!(incoming, expected);
    }

    #[test]
    fn the_order_of_protocol_statements_is_identity() {
        let protocols = [
            "CpAsyncCommit\nCpAsyncWait\nSync",
            "CpAsyncWait\nCpAsyncCommit\nSync",
            "WgmmaFence\nWgmmaCommit\nWgmmaWait",
            "WgmmaCommit\nWgmmaFence\nWgmmaWait",
        ];
        let keys: HashSet<String> = protocols
            .iter()
            .map(|text| canonicalize_identity(&body(text), false, None).unwrap().key)
            .collect();
        assert_eq!(keys.len(), protocols.len());
    }

    #[test]
    fn the_identity_combines_a_buffer_rename_with_a_nested_reordering() {
        let make = |reverse: bool, renamed: bool| {
            let axis = if renamed { "renamed_axis" } else { "axis" };
            let names = if renamed {
                ["renamed_left", "renamed_right"]
            } else {
                ["left", "right"]
            };
            let buffers = if renamed {
                ["renamed_X", "renamed_Y"]
            } else {
                ["X", "Y"]
            };
            let mut chains: Vec<String> = names
                .iter()
                .zip(buffers)
                .map(|(name, buffer)| {
                    format!(
                        "\n    if {axis} < 4:\n        {name} = load {buffer}[{axis}]\n        {name}_value = abs({name})\n        {buffer}_output[{axis}] = {name}_value"
                    )
                })
                .collect();
            if reverse {
                chains.reverse();
            }
            body(&format!("for {axis} in 0..4:{}", chains.concat()))
        };
        let keys: HashSet<String> = [false, true]
            .into_iter()
            .flat_map(|reverse| {
                [false, true]
                    .into_iter()
                    .map(move |renamed| (reverse, renamed))
            })
            .map(|(reverse, renamed)| {
                canonicalize_identity(
                    &normalize_body(&make(reverse, renamed)).unwrap(),
                    false,
                    None,
                )
                .unwrap()
                .key
            })
            .collect();
        assert_eq!(keys.len(), 1);
    }

    fn structural_key(given: &Body) -> String {
        canonicalize_identity(given, true, None).unwrap().key
    }

    fn exact_key(given: &Body) -> String {
        canonicalize_identity(given, false, None).unwrap().key
    }

    #[test]
    fn a_repeated_definition_is_captured_from_its_nearest_predecessor() {
        let make = |name: &str, inner: &str, result: &str, late_before_capture: bool| {
            let capture = format!("if true:\n    {inner} = abs({name})\n    P[] = {inner}\n");
            let second = format!("{name} = load B[]\n");
            let middle = if late_before_capture {
                format!("{second}{capture}")
            } else {
                format!("{capture}{second}")
            };
            body(&format!(
                "{name} = load A[]\n{middle}{result} = exp({name})\nO[] = {result}"
            ))
        };
        let before = normalize_body(&make("value", "inside", "result", false)).unwrap();
        let renamed =
            normalize_body(&make("renamed", "renamed_inside", "renamed_result", false)).unwrap();
        let after = normalize_body(&make("value", "inside", "result", true)).unwrap();
        assert_eq!(text(&before), text(&renamed));
        assert_ne!(text(&before), text(&after));
    }

    #[test]
    fn a_shadowing_after_the_read_keeps_the_outer_dependency() {
        // A deeper scope's rebind of a name binds nothing read above it: the block still depends on the enclosing
        // definition, so it cannot sort above it.
        let given = body(
            "if true:\n    y = abs(x)\n    if true:\n        let x = 2.0\n        P[] = x\n    O[] = y\nx = load X[]",
        );
        let normal = text(&normalize_body(&given).unwrap());
        let mut lines = normal.lines();
        let (first, second) = (lines.next().unwrap(), lines.next().unwrap());
        assert!(first.ends_with(" = load X[]"), "{normal}");
        assert!(second.starts_with("if "), "{normal}");
    }

    /// A small deterministic generator, so the property runs the same every time.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self, bound: usize) -> usize {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) % bound as u64) as usize
        }
    }

    #[test]
    fn the_structural_key_is_invariant_under_random_renaming_and_reordering() {
        // Seeded property check: independent load/compute/write chains keyed under every interleaving and spelling.
        // The relation graph never reads a spelling or a source position.
        let chains = |seed: u64, shuffle: bool| -> Body {
            let mut rng = Lcg(seed);
            let mut loops: Vec<String> = (0..4)
                .map(|chain| {
                    let prefix = if shuffle {
                        format!("c{chain}_")
                    } else {
                        format!("r{}_", rng.next(1000))
                    };
                    let op = ["abs", "exp", "negative"][chain % 3];
                    format!(
                        "for {prefix}i in 0..4:\n    {prefix}x = load {prefix}X[{prefix}i]\n    {prefix}y = {op}({prefix}x)\n    {prefix}O[{prefix}i] = {prefix}y"
                    )
                })
                .collect();
            if shuffle {
                for index in (1..loops.len()).rev() {
                    loops.swap(index, rng.next(index + 1));
                }
            }
            body(&loops.join("\n"))
        };
        let keys: HashSet<String> = (0..12u64)
            .flat_map(|seed| [false, true].map(|shuffle| structural_key(&chains(seed, shuffle))))
            .collect();
        assert_eq!(keys.len(), 1);
    }

    #[test]
    fn the_identity_ignores_a_buffer_spelling_that_reorders_the_executable_body() {
        // The executable order breaks symmetric ties by buffer spelling; identity labels the same graph without it,
        // so a rename that flips the executable order keys the same.
        let make = |left: &str, right: &str| {
            body(&format!(
                "x = load {left}[]\ny = load {right}[]\nz = add(x, y)\nO[] = z"
            ))
        };
        let left_leads = |given: &Body, left: &str| {
            text(given)
                .lines()
                .next()
                .unwrap()
                .ends_with(&format!(" = load {left}[]"))
        };
        let first = normalize_body(&make("X", "Y")).unwrap();
        let second = normalize_body(&make("z_input", "a_input")).unwrap();
        assert_ne!(left_leads(&first, "X"), left_leads(&second, "z_input"));
        assert_eq!(structural_key(&first), structural_key(&second));
        assert_eq!(exact_key(&first), exact_key(&second));
    }
}
