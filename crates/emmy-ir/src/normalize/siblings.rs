//! Sibling reduce loops: reduce-axis names unified where load positions overlap, and loops over one axis merged.

use super::*;

// --- sibling reduce loops ---------------------------------------------------------------------------------

pub(crate) const REDUCE_AXIS_PLACEHOLDER: &str = "__reduce_axis__";

/// The positions where a Load index within `body` walks the reduce axis, each keyed by how it walks the dimension.
pub(crate) fn reduce_axis_source_positions(body: &Body, axis: &Name) -> BTreeSet<Key> {
    let mut out = BTreeSet::new();
    for here in body.loads() {
        let Statement::Load(load) = &**here.stmt() else {
            continue;
        };
        for (dim, e) in load.index.iter().enumerate() {
            match affine_form(e, &|name| name == &**axis) {
                Some((anchor, coeffs)) => {
                    let coeff = coeffs.get(axis).copied().unwrap_or(0);
                    if coeff != 0 {
                        out.insert(Key::tuple([
                            Key::str("affine"),
                            Key::Str(load.input.clone()),
                            Key::Int(dim as i64),
                            Key::str(&anchor.text()),
                            Key::Int(coeff),
                        ]));
                    }
                }
                None if e.reads(axis) => {
                    let spelled = e
                        .substitute(&HashMap::from([(
                            axis.clone(),
                            Expression::var(REDUCE_AXIS_PLACEHOLDER),
                        )]))
                        .simplify(&SimplifyCtx::empty())
                        .unwrap_or_else(|_| e.clone());
                    out.insert(Key::tuple([
                        Key::str("composite"),
                        Key::Str(load.input.clone()),
                        Key::Int(dim as i64),
                        Key::str(&spelled.text()),
                    ]));
                }
                None => {}
            }
        }
    }
    out
}

/// Rename reduce-axis vars across sibling reduce loops whose Load positions overlap, so they share one axis name.
pub fn unify_siblings(body: &Body) -> Body {
    let mut stmts: Vec<Arc<Statement>> = body.stmts().to_vec();
    let mut entries: Vec<(usize, Name, Expression, BTreeSet<Key>)> = Vec::new();
    for (i, s) in stmts.iter().enumerate() {
        if let Statement::Loop(l) = &**s
            && l.is_reduce()
        {
            let positions = reduce_axis_source_positions(&l.body, &l.axis.name);
            if !positions.is_empty() {
                entries.push((
                    i,
                    l.axis.name.clone(),
                    l.axis.extent.expression.clone(),
                    positions,
                ));
            }
        }
    }
    if entries.len() < 2 {
        return body.clone();
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
    let mut changed = false;
    for k in 0..entries.len() {
        let root = find(&mut parent, k);
        let canonical = entries[root].1.clone();
        let (index, axis_name, extent, _) = &entries[k];
        if canonical == *axis_name {
            continue;
        }
        let Statement::Loop(l) = &*stmts[*index] else {
            continue;
        };
        // A RENAME, binders and uses through one map, which is why it may pass through the very scopes it renames.
        let new_axis = Arc::new(Axis {
            name: canonical.clone(),
            extent: Dim {
                expression: extent.clone(),
                hint: l.axis.extent.hint,
            },
            window: l.axis.window.clone(),
        });
        let mapping = HashMap::from([(l.axis.name.clone(), canonical.clone())]);
        let renamed = l.body.map_each(&mut |s| Statement::rename(s, &mapping));
        stmts[*index] = Arc::new(Statement::Loop(Loop {
            axis: new_axis,
            body: renamed,
            unroll: l.unroll,
            seed: l.seed,
        }));
        changed = true;
    }
    if changed {
        Body::new(stmts)
    } else {
        body.clone()
    }
}

/// The names a loop over `body` still binds after it closes.
pub(crate) fn carried_out(body: &Body) -> HashSet<Name> {
    let mut out: HashSet<Name> = body
        .stmts()
        .iter()
        .filter_map(|s| match &**s {
            Statement::Accum(a) => Some(a.name.clone()),
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
            .unwrap_or_else(|| name.clone());
        used.insert(fresh.clone());
        mapping.insert(name.clone(), fresh);
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
    let mut changed = false;
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
            changed = true;
        }
        out.push(if consumed.is_empty() && !changed {
            s.clone()
        } else {
            Arc::new(Statement::Loop(merged))
        });
    }
    if changed {
        Body::new(out)
    } else {
        body.clone()
    }
}

/// Merge sibling reduce loops with one axis into one loop whose body is the concatenation; parents before children.
pub fn merge_sibling_reduce_loops(body: &Body) -> Body {
    let merged = merge_sibling_reduce_loops_here(&unify_siblings(body));
    merged.map_each(&mut |s| {
        let nested = s.nested();
        if nested.is_empty() {
            return s.clone();
        }
        with_bodies(s, nested.iter().map(merge_sibling_reduce_loops).collect())
    })
}
