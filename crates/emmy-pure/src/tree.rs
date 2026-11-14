//! The tree-wide canonical forms of a Tile IR fold tree: what only the WHOLE tree can decide.
//!
//! An identity projection dissolves into the one operand it re-exposes, a carrier the fusion left holding independent
//! contractions becomes one term per state, every operand result component no reader reads is dropped (the boundary
//! stores counting as readers), and same-value cones become ONE shared term. Object identity is how the placement
//! recognizes that two consumption sites read one value, so the sharing restored last is part of the form.
//!
//! Normalization carries no knob: it runs the same on every tree. The reached fixpoint is remembered on the result
//! per set of stored names, so a reconstruction answers without walking again.

use std::collections::{HashMap, HashSet};

use emmy_ir::body::Body;
use emmy_ir::{Name, Result};

use crate::fold::{Fold, Parts};
use crate::lambda::Lambda;

/// The operands an identity projection merely re-exposes, or `None`.
fn passthrough(node: &Fold) -> Option<Vec<Fold>> {
    if node.axis().is_some() || !node.lift().body().is_empty() {
        return None;
    }
    let bound: Vec<Name> = node.bindings().iter().map(|b| b.param).collect();
    (node.lift().results() == bound.as_slice()).then(|| node.operands().to_vec())
}

fn with_operands(node: &Fold, operands: Vec<Fold>) -> Result<Fold> {
    let mut parts = node.parts();
    parts.operands = operands;
    Fold::new(parts)
}

fn with_lift(
    node: &Fold,
    operands: Option<Vec<Fold>>,
    params: Option<Vec<Name>>,
    emptied: bool,
) -> Result<Fold> {
    let mut parts: Parts = node.parts();
    if let Some(operands) = operands {
        parts.operands = operands;
    }
    let lift = node.lift();
    let lift = if emptied {
        Lambda::new(lift.params().to_vec(), &Body::empty(), Vec::new())?
    } else {
        Lambda::new(
            params.unwrap_or_else(|| lift.params().to_vec()),
            lift.body(),
            lift.results().to_vec(),
        )?
    };
    parts.lift = Some(lift);
    Fold::new(parts)
}

/// Dissolve pass-throughs and take carriers apart, bottom-up. No memo: a term shared by two positions is normalized
/// at each, as the sharing pass after it expects.
fn normalize_fold(fold: &Fold) -> Result<Fold> {
    let mut operands = Vec::new();
    for edge in fold.operands() {
        let child = normalize_fold(edge)?;
        match passthrough(&child) {
            Some(parts) if !parts.is_empty() => operands.extend(parts),
            _ => operands.push(child),
        }
    }
    let node = if operands != fold.operands() {
        with_operands(fold, operands)?
    } else {
        fold.clone()
    };
    if let Some(collapsed) = passthrough(&node)
        && collapsed.len() == 1
    {
        return Ok(collapsed[0].clone());
    }
    Ok(node.per_state()?.unwrap_or(node))
}

/// A node's lift params split the way the positional binding reads them: the iteration var, one slot per operand,
/// the trailing free coordinates.
fn binds(node: &Fold) -> (Vec<Name>, Vec<Vec<Name>>, Vec<Name>) {
    let params = node.lift().params();
    let lead_len = usize::from(node.axis().is_some());
    let mut slots = Vec::new();
    let mut cursor = lead_len;
    for edge in node.operands() {
        let width = edge.exposes().len();
        slots.push(params[cursor..cursor + width].to_vec());
        cursor += width;
    }
    (
        params[..lead_len].to_vec(),
        slots,
        params[cursor..].to_vec(),
    )
}

/// The lift params `node` reads: its body's reads and its results.
fn reads(node: &Fold) -> HashSet<Name> {
    let mut read: HashSet<Name> = node.lift().body().ssa_uses().clone();
    read.extend(node.lift().results().iter().copied());
    read
}

/// Drop every operand result component no reader reads; a union over readers, tree-wide.
fn prune_unread(root: &Fold, stored: &HashSet<Name>) -> Result<Fold> {
    let mut kept: HashMap<usize, HashSet<usize>> = HashMap::new();
    let mut seen: HashSet<usize> = HashSet::new();
    fn collect(
        node: &Fold,
        stored: &HashSet<Name>,
        kept: &mut HashMap<usize, HashSet<usize>>,
        seen: &mut HashSet<usize>,
    ) {
        if !seen.insert(node.address()) {
            return;
        }
        let read = reads(node);
        let (_, slots, _) = binds(node);
        for (slot, edge) in slots.iter().zip(node.operands()) {
            let want = kept.entry(edge.address()).or_default();
            want.extend(
                slot.iter()
                    .enumerate()
                    .filter(|(_, p)| read.contains(p))
                    .map(|(i, _)| i),
            );
            want.extend(
                edge.exposes()
                    .iter()
                    .enumerate()
                    .filter(|(_, n)| stored.contains(n))
                    .map(|(i, _)| i),
            );
            collect(edge, stored, kept, seen);
        }
    }
    collect(root, stored, &mut kept, &mut seen);
    let mut done: HashMap<usize, Fold> = HashMap::new();
    fn visit(
        node: &Fold,
        stored: &HashSet<Name>,
        kept: &HashMap<usize, HashSet<usize>>,
        done: &mut HashMap<usize, Fold>,
    ) -> Result<Fold> {
        if let Some(found) = done.get(&node.address()) {
            return Ok(found.clone());
        }
        let (lead, slot_table, trailing) = binds(node);
        let read = reads(node);
        let mut operands = Vec::new();
        let mut params = Vec::new();
        let mut same = true;
        for (slots, edge) in slot_table.iter().zip(node.operands()) {
            let read_here = slots.iter().any(|p| read.contains(p))
                || edge.exposes().iter().any(|n| stored.contains(n));
            if !read_here {
                same = false;
                continue;
            }
            let new = visit(edge, stored, kept, done)?;
            let empty = HashSet::new();
            let want = kept.get(&edge.address()).unwrap_or(&empty);
            same = same && new.ptr_eq(edge);
            if edge.twist().is_some()
                || edge.observe().is_some()
                || edge.carries()
                || want.len() == slots.len()
            {
                operands.push(new);
                params.extend(slots.iter().copied());
                continue;
            }
            same = false;
            if !want.is_empty() {
                let names: Vec<Name> = edge
                    .exposes()
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| want.contains(i))
                    .map(|(_, n)| *n)
                    .collect();
                operands.push(new.exposing(&names)?);
                let mut indices: Vec<usize> = want.iter().copied().collect();
                indices.sort_unstable();
                params.extend(indices.iter().map(|&i| slots[i]));
            }
        }
        let rebuilt = if same {
            node.clone()
        } else {
            let mut all = lead;
            all.extend(params);
            all.extend(trailing);
            with_lift(node, Some(operands), Some(all), false)?
        };
        done.insert(node.address(), rebuilt.clone());
        Ok(rebuilt)
    }
    let mut pruned = visit(root, stored, &kept, &mut done)?;
    // The root answers to the stores alone: a zero-axis root none of whose own components a store writes collapses
    // onto the operand carrying them.
    while pruned.axis().is_none()
        && !stored.is_empty()
        && !pruned.exposes().iter().any(|n| stored.contains(n))
    {
        let carriers: Vec<Fold> = pruned
            .operands()
            .iter()
            .filter(|edge| stored.iter().all(|n| edge.exposes().contains(n)))
            .cloned()
            .collect();
        if carriers.len() != 1 {
            let step = pruned.step()?;
            if !step.defined().iter().any(|n| stored.contains(n)) {
                pruned = with_lift(&pruned, None, None, true)?;
            }
            break;
        }
        pruned = carriers[0].clone();
    }
    Ok(pruned)
}

/// Every name a term of `root`'s tree exposes or its lift defines.
fn tree_names(root: &Fold) -> HashSet<Name> {
    let mut names = HashSet::new();
    let mut seen = HashSet::new();
    let mut pending = vec![root.clone()];
    while let Some(node) = pending.pop() {
        if !seen.insert(node.address()) {
            continue;
        }
        names.extend(node.exposes().iter().copied());
        for stmt in node.lift().body().stmts() {
            names.extend(stmt.defines());
        }
        pending.extend(node.operands().iter().cloned());
    }
    names
}

/// Restore object sharing between same-value cones: every term hash-consed bottom-up onto the first occurrence in
/// walk order with the same canonical form, the same exposed names and the same free coordinates by name.
// A term's memos sit behind interior mutability, but its hash and equality read only its stored parts.
#[allow(clippy::mutable_key_type)]
fn share_common_cones(root: &Fold) -> Result<Fold> {
    type Key = (Fold, Vec<Name>, Vec<Name>);
    let mut canon: HashMap<Key, Fold> = HashMap::new();
    let mut seen: HashMap<usize, Fold> = HashMap::new();
    fn visit(
        node: &Fold,
        canon: &mut HashMap<Key, Fold>,
        seen: &mut HashMap<usize, Fold>,
    ) -> Result<Fold> {
        if let Some(found) = seen.get(&node.address()) {
            return Ok(found.clone());
        }
        let operands: Vec<Fold> = node
            .operands()
            .iter()
            .map(|edge| visit(edge, canon, seen))
            .collect::<Result<_>>()?;
        let changed = operands
            .iter()
            .zip(node.operands())
            .any(|(new, old)| !new.ptr_eq(old));
        let current = if changed {
            with_operands(node, operands)?
        } else {
            node.clone()
        };
        let mut free: Vec<Name> = current.free_axes().iter().copied().collect();
        free.sort();
        let key = (current.canonical()?, current.exposes().to_vec(), free);
        let prior = canon.entry(key).or_insert(current).clone();
        seen.insert(node.address(), prior.clone());
        Ok(prior)
    }
    visit(root, &mut canon, &mut seen)
}

impl Fold {
    /// This tree with every operand result component no reader reads dropped, `stored` the names the boundary stores
    /// read.
    pub fn pruned(&self, stored: &[Name]) -> Result<Fold> {
        prune_unread(self, &stored.iter().copied().collect())
    }

    /// This complete tree normalized under the kernel's boundary stores: `values` the names they write, `reads` every
    /// name they read (of which only what some term of the tree defines or exposes is a value of the tree). The
    /// fixpoint is remembered on the result per stored set.
    pub fn normalized(&self, values: &[Name], reads: &[Name]) -> Result<Fold> {
        let mut stored: HashSet<Name> = values.iter().copied().collect();
        if reads.iter().any(|n| !stored.contains(n)) {
            let names = tree_names(self);
            stored.extend(reads.iter().copied().filter(|n| names.contains(n)));
        }
        let mut key: Vec<Name> = stored.iter().copied().collect();
        key.sort();
        if self.normal_under(&key) {
            return Ok(self.clone());
        }
        let mut normalized = share_common_cones(self)?;
        loop {
            let again = prune_unread(&normalize_fold(&normalized)?, &stored)?;
            if again == normalized {
                break;
            }
            normalized = again;
        }
        let result = if normalized == *self {
            self.clone()
        } else {
            normalized
        };
        let result = share_common_cones(&result)?;
        result.remember_normal(key);
        Ok(result)
    }
}
