//! The identity: the scope-tree hash over a numbering, with the buffers and carried states keyed by use.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::{Chains, Key, Number, Numbering, Tag, hash_parts, value_numbers};
use crate::body::{Body, StmtRef};
use crate::order::ordering_constraints;
use crate::{Name, locked};

/// Per member of `body`, the members it must follow for an effect and not for a value.
pub fn effect_order(body: &Body) -> Vec<HashSet<usize>> {
    let ordered = ordering_constraints(body, true, true);
    let dataflow = ordering_constraints(body, false, true);
    ordered
        .into_iter()
        .zip(dataflow)
        .map(|(o, d)| o.difference(&d).copied().collect())
        .collect()
}

/// Memoized effect orders, by scope, across the numberings of one body.
pub type Effects = HashMap<usize, Vec<HashSet<usize>>>;

fn body_key(body: &Body) -> usize {
    Arc::as_ptr(&body.0) as usize
}

/// The hash of `body`'s scope tree: each block by its header and its children, each leaf by its number and its
/// coordinates, each scope's members sorted.
pub fn scope_tree(numbering: &Numbering, body: &Body, effects: &mut Effects) -> Number {
    let order = effects
        .entry(body_key(body))
        .or_insert_with(|| effect_order(body))
        .clone();
    let mut members: Vec<Number> = Vec::with_capacity(body.len());
    for (index, sources) in order.iter().enumerate() {
        let mut preceding: Vec<Key> = sources
            .iter()
            .map(|source| Key::Num(members[*source]))
            .collect();
        preceding.sort();
        members.push(member_tree(numbering, &body.at(index), preceding, effects));
    }
    members.sort_unstable();
    hash_parts(members.into_iter().map(Key::Num))
}

/// The scope tree of a body holding `here` alone.
pub fn statement_tree(numbering: &Numbering, here: &StmtRef, effects: &mut Effects) -> Number {
    hash_parts([Key::Num(member_tree(numbering, here, Vec::new(), effects))])
}

/// One member of a scope: a block by its header, its children's trees and the members it follows for an effect, a
/// leaf by its number, its coordinates and those members.
fn member_tree(
    numbering: &Numbering,
    here: &StmtRef,
    preceding: Vec<Key>,
    effects: &mut Effects,
) -> Number {
    let ordinal = numbering.ordinal(here);
    match &numbering.blocks[ordinal] {
        Some((header, params)) => {
            let children = locked(&numbering.trees).get(&ordinal).cloned();
            let children = match children {
                Some(children) => children,
                None => {
                    let children: Vec<Number> = here
                        .stmt()
                        .nested()
                        .iter()
                        .map(|child| scope_tree(numbering, child, effects))
                        .collect();
                    locked(&numbering.trees).insert(ordinal, children.clone());
                    children
                }
            };
            hash_parts([
                Key::tag(Tag::Block),
                header.clone(),
                Key::Tuple(params.clone()),
                Key::Tuple(children.into_iter().map(Key::Num).collect()),
                Key::Tuple(preceding),
            ])
        }
        None => {
            let (number, params) = numbering.numbers[ordinal]
                .as_ref()
                .map_or((Key::None, Vec::new()), |(n, p)| (Key::Num(*n), p.clone()));
            hash_parts([
                Key::tag(Tag::Stmt),
                number,
                Key::Tuple(params),
                Key::Tuple(preceding),
            ])
        }
    }
}

/// Every resource's forward cone at statement grain: the sorted numbers of every statement that reads, directly or
/// through others, a statement touching it.
fn downstream(numbering: &Numbering) -> HashMap<Name, Vec<Key>> {
    let mut consumers: HashMap<usize, Vec<usize>> = HashMap::new();
    for (stmt, sources) in numbering.sources.iter().enumerate() {
        for &source in sources {
            consumers.entry(source).or_default().push(stmt);
        }
    }
    let number = |stmt: usize| -> Key {
        match &numbering.numbers[stmt] {
            Some((number, _)) => Key::Num(*number),
            None => numbering.blocks[stmt]
                .as_ref()
                .map_or(Key::None, |(header, _)| header.clone()),
        }
    };
    let mut out = HashMap::new();
    for (name, statements) in &numbering.touching {
        let mut seen: HashSet<usize> = statements.iter().copied().collect();
        let mut stack: Vec<usize> = statements.clone();
        while let Some(current) = stack.pop() {
            for &consumer in consumers.get(&current).map_or(&[][..], |v| &v[..]) {
                if seen.insert(consumer) {
                    stack.push(consumer);
                }
            }
        }
        let mut numbers: Vec<Key> = seen.into_iter().map(number).collect();
        numbers.sort();
        out.insert(*name, numbers);
    }
    out
}

type Cells = HashMap<Name, Key>;

/// Each buffer's color (its type as the caller spells it), when the identity is typed.
pub type Colors<'a> = Option<&'a HashMap<Name, String>>;

fn paint(colors: Colors, name: &Name) -> Key {
    colors
        .and_then(|c| c.get(name))
        .map_or(Key::None, |text| Key::str(text))
}

/// How a buffer or state is spelled in a payload during one numbering: fixed by its rank, picked out, or open with
/// its cell of use.
struct Keyed<'a> {
    colors: Colors<'a>,
    rank: &'a HashMap<Name, usize>,
    cell: &'a Cells,
    pick: Option<&'a Name>,
}

impl Keyed<'_> {
    fn key(&self, name: &Name) -> Key {
        if let Some(r) = self.rank.get(name) {
            return Key::tuple([
                Key::tag(Tag::Fixed),
                Key::Int(*r as i64),
                paint(self.colors, name),
            ]);
        }
        let tag = if self.pick == Some(name) {
            Tag::Pick
        } else {
            Tag::Open
        };
        Key::tuple([
            Key::tag(tag),
            paint(self.colors, name),
            self.cell.get(name).cloned().unwrap_or(Key::None),
        ])
    }
}

fn partition(cells: &Cells) -> Vec<Vec<Name>> {
    let mut groups: HashMap<&Key, Vec<Name>> = HashMap::new();
    for (name, value) in cells {
        groups.entry(value).or_default().push(*name);
    }
    let mut out: Vec<Vec<Name>> = groups
        .into_values()
        .map(|mut group| {
            group.sort();
            group
        })
        .collect();
    out.sort();
    out
}

/// Number until the open resources' cells stop splitting: each is colored by its type and by everything downstream
/// of it. `cell` is updated in place.
fn cells(
    body: &Body,
    colors: Colors,
    rank: &HashMap<Name, usize>,
    cell: &mut Cells,
    chains: Chains,
) -> Numbering {
    let number = |cell: &Cells| {
        value_numbers(
            body,
            &|name| {
                Keyed {
                    colors,
                    rank,
                    cell,
                    pick: None,
                }
                .key(name)
            },
            chains,
        )
    };
    let mut numbering = number(cell);
    loop {
        let down = downstream(&numbering);
        let fresh: Cells = numbering
            .buffers
            .iter()
            .chain(numbering.states.iter())
            .filter(|name| !rank.contains_key(*name))
            .map(|name| {
                (
                    *name,
                    Key::tuple([
                        paint(colors, name),
                        Key::Tuple(down.get(name).cloned().unwrap_or_default()),
                    ]),
                )
            })
            .collect();
        if partition(&fresh) == partition(cell) {
            return numbering;
        }
        *cell = fresh;
        numbering = number(cell);
    }
}

/// Every buffer's and carried state's role at a glance: the rank of its cell of use among the cells.
pub fn roles(body: &Body, colors: Colors) -> HashMap<Name, usize> {
    let mut cell: Cells = HashMap::new();
    cells(body, colors, &HashMap::new(), &mut cell, Chains::Flat);
    let mut values: Vec<&Key> = cell.values().collect();
    values.sort();
    values.dedup();
    cell.iter()
        .map(|(name, value)| (*name, values.iter().position(|v| *v == value).unwrap_or(0)))
        .collect()
}

/// The body's identity and its external buffers in role order.
pub fn digest(body: &Body, colors: Colors, chains: Chains) -> (Number, Vec<Name>) {
    let mut rank: HashMap<Name, usize> = HashMap::new();
    let mut cell: Cells = HashMap::new();
    let mut effects: Effects = HashMap::new();
    let numbering = cells(body, colors, &rank, &mut cell, chains);
    let names: Vec<Name> = numbering
        .buffers
        .iter()
        .chain(numbering.states.iter())
        .cloned()
        .collect();
    while rank.len() < names.len() {
        // One round ranks every cell in cell order: a singleton outright, a tie by the certificates of its members
        // under this round's coloring; only a member interchangeable with another sends the round back to refine.
        let mut unranked: Vec<Name> = names
            .iter()
            .filter(|name| !rank.contains_key(*name))
            .cloned()
            .collect();
        unranked.sort_by(|a, b| cell[a].cmp(&cell[b]).then_with(|| a.cmp(b)));
        let mut position = 0;
        while position < unranked.len() {
            let value = cell[&unranked[position]].clone();
            let mut group: Vec<Name> = Vec::new();
            while position < unranked.len() && cell[&unranked[position]] == value {
                group.push(unranked[position]);
                position += 1;
            }
            if group.len() == 1 {
                let r = rank.len();
                rank.insert(group[0], r);
                continue;
            }
            let certificate: HashMap<Name, Number> = group
                .iter()
                .map(|name| {
                    let keyed = Keyed {
                        colors,
                        rank: &rank,
                        cell: &cell,
                        pick: Some(name),
                    };
                    let tree = scope_tree(
                        &value_numbers(body, &|n| keyed.key(n), chains),
                        body,
                        &mut effects,
                    );
                    (*name, tree)
                })
                .collect();
            group.sort_by(|a, b| certificate[a].cmp(&certificate[b]).then_with(|| a.cmp(b)));
            for name in &group {
                let r = rank.len();
                rank.insert(*name, r);
                if group
                    .iter()
                    .filter(|other| certificate[*other] == certificate[name])
                    .count()
                    > 1
                {
                    break; // interchangeable with another: fix this one and refine before looking at the rest
                }
            }
        }
        if rank.len() < names.len() {
            cells(body, colors, &rank, &mut cell, chains);
        }
    }
    let keyed = Keyed {
        colors,
        rank: &rank,
        cell: &cell,
        pick: None,
    };
    let final_numbering = value_numbers(body, &|n| keyed.key(n), chains);
    let tree = scope_tree(&final_numbering, body, &mut effects);
    let mut buffers = final_numbering.buffers.clone();
    buffers.sort_by_key(|name| rank[name]);
    (tree, buffers)
}
