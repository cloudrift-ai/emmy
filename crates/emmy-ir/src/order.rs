//! Dependency order for statement bodies: the constraints every sibling order must respect (a definition ahead of
//! its reads, a buffer's write ahead of the reads and writes that follow it, a carried state's updates in sequence,
//! a barrier pinned relative to every sibling), and the stable dependency sort that runs ahead of normalization.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::Name;
use crate::body::{Body, free_ssa, ordered_sibling_defs};
use crate::kinds::Statement;

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
                state.insert(a.name.clone());
            }
            Statement::Carry(c) => {
                state.insert(c.name.clone());
            }
            _ => {}
        }
    }
    if let Statement::Init(init) = stmt {
        state.insert(init.name.clone());
    }
    (reads, writes, state)
}

/// A no-dataflow, non-pure leaf: an ordered execution protocol (a barrier, an async commit), pinned relative to
/// every sibling.
pub fn is_protocol(stmt: &Statement) -> bool {
    !stmt.is_pure() && stmt.nested().is_empty() && stmt.defines().is_empty() && stmt.deps().is_empty()
}

/// Dependency and, when requested, effect predecessors for one lexical scope.
pub fn ordering_constraints(body: &Body, effects: bool, redefinitions: bool) -> Vec<HashSet<usize>> {
    let stmts = body.stmts();
    let defs_uses: Vec<(Vec<Name>, HashSet<Name>)> = stmts.iter().map(|stmt| (ordered_sibling_defs(stmt), free_ssa(stmt))).collect();
    let mut definitions: HashMap<&Name, Vec<usize>> = HashMap::new();
    for (index, (defines, _)) in defs_uses.iter().enumerate() {
        for name in defines {
            definitions.entry(name).or_default().push(index);
        }
    }
    let defining_stmt = |name: &Name, consumer: usize| -> Option<usize> {
        let sites = definitions.get(name).map_or(&[][..], |v| &v[..]);
        if !redefinitions {
            return sites.first().copied().filter(|site| *site != consumer);
        }
        if let Some(preceding) = sites.iter().rev().find(|site| **site < consumer) {
            return Some(*preceding);
        }
        sites.iter().copied().find(|site| *site != consumer)
    };
    let mut incoming: Vec<HashSet<usize>> = Vec::with_capacity(stmts.len());
    for (index, (_, uses)) in defs_uses.iter().enumerate() {
        incoming.push(uses.iter().filter_map(|name| defining_stmt(name, index)).collect());
    }
    if redefinitions {
        for (reader, (_, uses)) in defs_uses.iter().enumerate() {
            for name in uses {
                for &later in definitions.get(name).map_or(&[][..], |v| &v[..]) {
                    if later > reader {
                        incoming[later].insert(reader);
                    }
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
                    readers.entry(name.clone()).or_default().insert(index);
                }
            }
            for name in writes {
                if let Some(&writer) = last_write.get(name) {
                    incoming[index].insert(writer);
                }
                if let Some(earlier) = readers.remove(name) {
                    incoming[index].extend(earlier);
                }
                last_write.insert(name.clone(), index);
            }
            for name in state {
                if let Some(&previous) = last_state.get(name) {
                    incoming[index].insert(previous);
                }
                last_state.insert(name.clone(), index);
            }
        }
        let protocol: HashSet<usize> = (0..stmts.len()).filter(|&index| is_protocol(&stmts[index])).collect();
        let mut preceding: Vec<usize> = Vec::new();
        let mut previous_protocol: Option<usize> = None;
        for index in 0..stmts.len() {
            if protocol.contains(&index) {
                incoming[index].extend(preceding.iter().copied());
                if let Some(previous) = previous_protocol {
                    incoming[index].insert(previous);
                }
                preceding.clear();
                previous_protocol = Some(index);
            } else {
                if let Some(previous) = previous_protocol {
                    incoming[index].insert(previous);
                }
                preceding.push(index);
            }
        }
    }
    incoming
}

/// Stable recursive dependency sort used before structural normalization.
pub fn topological_sort(body: &Body) -> Body {
    let sorted = body.map_each(&mut |stmt| {
        let nested = stmt.nested();
        if nested.is_empty() {
            return stmt.clone();
        }
        Statement::with_bodies(stmt, nested.iter().map(topological_sort).collect()).expect("the same count")
    });
    sorted.topological_order(&ordering_constraints(&sorted, false, false))
}
