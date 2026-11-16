//! Congruent carried states: two states whose seeds agree and whose updates compute one value once the states are
//! assumed equal hold one value at every step. The assumption is refined until it holds (optimistic value numbering):
//! groups start from structure and split by their updates' numbers until no group splits.

use std::collections::HashMap;

use super::{Chains, Key, Number, Numbering, Param, spelled_key, value_numbers};

/// A statement's number and parameters, as the numbering gives them.
type Value = (Number, Vec<Param>);
use crate::Name;
use crate::body::Body;
use crate::statement::Statement;

/// The groups of congruent states of `body`, each of two or more, every member in the order of its update and the
/// groups in the order of their first members. A candidate is a state with one defining statement in the whole body:
/// an accumulator updated directly in a seeded loop, so its seed is the operation's identity at that loop's entry, or
/// a carried state, whose seed its update names. Two candidates start in one group when their updates are statements
/// of one type, operation and dtype in one scope; a group splits by its members' update numbers, the numbering keying every grouped state
/// by its group, until no group splits. A body with no two candidates in one scope is not numbered at all. The
/// numbering that found the groups stable comes with them: it keys every member under [`group_keys`] and every other
/// name as spelled, which is what placement numbers under, so placement takes it instead of numbering again.
pub fn congruent_states(body: &Body) -> (Vec<Vec<Name>>, Option<Numbering>) {
    let mut sites = Sites::default();
    sites.walk(body, None, true);
    let mut start: Vec<(Key, Vec<Name>)> = Vec::new();
    for name in &sites.order {
        let site = &sites.sites[name];
        if site.definitions != 1 || !site.candidate || sites.defining.get(name) != Some(&1) {
            continue;
        }
        match start.iter_mut().find(|(key, _)| *key == site.key) {
            Some((_, members)) => members.push(*name),
            None => start.push((site.key.clone(), vec![*name])),
        }
    }
    let mut groups: Vec<Vec<Name>> = start
        .into_iter()
        .map(|(_, members)| members)
        .filter(|members| members.len() > 1)
        .collect();
    while !groups.is_empty() {
        let keys = group_keys(&groups);
        let numbering = value_numbers(
            body,
            &|name| keys.get(name).cloned().unwrap_or_else(|| spelled_key(name)),
            Chains::Flat,
        );
        let mut next: Vec<Vec<Name>> = Vec::new();
        for group in &groups {
            let mut parts: Vec<(&Option<Value>, Vec<Name>)> = Vec::new();
            for name in group {
                let number = &numbering.numbers[sites.sites[name].ordinal];
                match parts.iter_mut().find(|(seen, _)| *seen == number) {
                    Some((_, members)) => members.push(*name),
                    None => parts.push((number, vec![*name])),
                }
            }
            next.extend(
                parts
                    .into_iter()
                    .map(|(_, members)| members)
                    .filter(|members| members.len() > 1),
            );
        }
        if next == groups {
            return (groups, Some(numbering));
        }
        groups = next;
    }
    (groups, None)
}

/// The resource key every member of a group is numbered under: its group's position.
pub fn group_keys(groups: &[Vec<Name>]) -> HashMap<Name, Key> {
    groups
        .iter()
        .enumerate()
        .flat_map(|(index, group)| {
            group
                .iter()
                .map(move |name| (*name, Key::Int(index as i64)))
        })
        .collect()
}

/// Where a state is defined: its update's ordinal in the numbering's pre-order walk, the start key (the update's
/// statement type, operation and dtype, and the scope that holds it), how many of its updates and seeds define it, and
/// whether its seed is known.
struct Site {
    ordinal: usize,
    key: Key,
    definitions: usize,
    candidate: bool,
}

#[derive(Default)]
struct Sites {
    sites: HashMap<Name, Site>,
    order: Vec<Name>,
    next: usize,
    /// How many statements, of any type, define each name: a kernel statement may update a state in place (a warp
    /// shuffle combining into it), which makes it no candidate.
    defining: HashMap<Name, usize>,
}

impl Sites {
    /// Every statement in the numbering's pre-order; `scope` is the block's ordinal and the position of the body in
    /// it, `seeded` whether that body is a seeded loop's.
    fn walk(&mut self, body: &Body, scope: Option<(usize, usize)>, seeded: bool) {
        for stmt in body.stmts() {
            let ordinal = self.next;
            self.next += 1;
            for name in stmt.defines() {
                *self.defining.entry(name).or_default() += 1;
            }
            let candidate = match &**stmt {
                Statement::Accum(_) => Some(seeded && scope.is_some()),
                Statement::Carry(_) => Some(true),
                Statement::Init(_) => Some(false),
                _ => None,
            };
            if let Some(candidate) = candidate {
                for name in stmt.defines() {
                    match self.sites.get_mut(&name) {
                        Some(site) => {
                            site.definitions += 1;
                            site.candidate &= candidate;
                        }
                        None => {
                            self.order.push(name);
                            self.sites.insert(
                                name,
                                Site {
                                    ordinal,
                                    key: start_key(stmt, scope),
                                    definitions: 1,
                                    candidate,
                                },
                            );
                        }
                    }
                }
            }
            let inner_seeded = match &**stmt {
                Statement::Loop(l) => l.seed,
                Statement::StridedLoop(l) => l.seed,
                _ => false,
            };
            for (position, child) in stmt.nested().iter().enumerate() {
                self.walk(child, Some((ordinal, position)), inner_seeded);
            }
        }
    }
}

/// The statement type, operation and dtype of a state's update, and the scope that holds it.
fn start_key(stmt: &Statement, scope: Option<(usize, usize)>) -> Key {
    let (op, dtype) = match stmt {
        Statement::Accum(a) => (Key::str(a.op.name()), a.dtype),
        Statement::Carry(c) => (Key::None, c.dtype),
        _ => (Key::None, None),
    };
    Key::tuple([
        Key::str(stmt.ty().name()),
        op,
        Key::option(dtype.as_ref().map(|d| d.text())),
        match scope {
            Some((block, position)) => {
                Key::tuple([Key::Int(block as i64), Key::Int(position as i64)])
            }
            None => Key::None,
        },
    ])
}
