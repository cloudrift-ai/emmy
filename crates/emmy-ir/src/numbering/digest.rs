//! The identity: the scope-tree hash over a numbering, with the buffers and carried states keyed by use.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::values::numbered;
use super::{Chains, Key, Number, Numbering, Param, Tag, hash_parts, offset_of, params_key};
use crate::body::{Body, StmtRef};
use crate::order::ordering_constraints;
use crate::{Name, locked};

/// Per member of `body`, the members it must follow for an effect and not for a value.
pub fn effect_order(body: &Body) -> Vec<HashSet<usize>> {
    let ordered = ordering_constraints(body, true);
    let dataflow = ordering_constraints(body, false);
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
                params_key(params),
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
                params_key(&params),
                Key::Tuple(preceding),
            ])
        }
    }
}

/// The scope tree of `body` under a numbering with its integers apart (`value_numbers_apart`), as [`scope_tree`]
/// builds it: each block by its header, its parameters, its children and the members it follows for an effect, each
/// leaf by its number and its parameters, each scope's members sorted. An integer taken apart is keyed as `integer`
/// keys it, and `sites` takes every one in the order of the sorted tree. Two members that key alike keep the order
/// they are written in.
pub fn scope_tree_apart(
    numbering: &Numbering,
    body: &Body,
    integer: &dyn Fn(i64, i64) -> Key,
    effects: &mut Effects,
    sites: &mut Vec<i64>,
) -> Number {
    let order = effects
        .entry(body_key(body))
        .or_insert_with(|| effect_order(body))
        .clone();
    let mut members: Vec<(Number, Vec<i64>)> = Vec::with_capacity(body.len());
    for (index, sources) in order.iter().enumerate() {
        let mut preceding: Vec<Key> = sources
            .iter()
            .map(|source| Key::Num(members[*source].0))
            .collect();
        preceding.sort();
        let here = body.at(index);
        let ordinal = numbering.ordinal(&here);
        let mut own = Vec::new();
        let keyed = |params: &[Param], own: &mut Vec<i64>| {
            Key::Tuple(
                params
                    .iter()
                    .map(|param| match offset_of(param) {
                        Some((value, site)) => {
                            own.push(site);
                            integer(site, value)
                        }
                        None => Key::Param(param.clone()),
                    })
                    .collect(),
            )
        };
        let number = match &numbering.blocks[ordinal] {
            Some((header, params)) => {
                let params = keyed(params, &mut own);
                let children: Vec<Key> = here
                    .stmt()
                    .nested()
                    .iter()
                    .map(|child| {
                        Key::Num(scope_tree_apart(
                            numbering, child, integer, effects, &mut own,
                        ))
                    })
                    .collect();
                hash_parts([
                    Key::tag(Tag::Block),
                    header.clone(),
                    params,
                    Key::Tuple(children),
                    Key::Tuple(preceding),
                ])
            }
            None => {
                let (number, params) = match &numbering.numbers[ordinal] {
                    Some((number, params)) => (Key::Num(*number), keyed(params, &mut own)),
                    None => (Key::None, Key::Tuple(Vec::new())),
                };
                hash_parts([Key::tag(Tag::Stmt), number, params, Key::Tuple(preceding)])
            }
        };
        members.push((number, own));
    }
    members.sort_by_key(|member| member.0);
    let mut numbers = Vec::with_capacity(members.len());
    for (number, own) in members {
        numbers.push(Key::Num(number));
        sites.extend(own);
    }
    hash_parts(numbers)
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
/// of it. `cell` is updated in place. `apart` numbers with the coordinate integers apart.
fn cells(
    body: &Body,
    colors: Colors,
    rank: &HashMap<Name, usize>,
    cell: &mut Cells,
    chains: Chains,
    apart: bool,
) -> Numbering {
    let number = |cell: &Cells| {
        numbered(
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
            apart,
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
    cells(
        body,
        colors,
        &HashMap::new(),
        &mut cell,
        Chains::Flat,
        false,
    );
    let mut values: Vec<&Key> = cell.values().collect();
    values.sort();
    values.dedup();
    cell.iter()
        .map(|(name, value)| (*name, values.iter().position(|v| *v == value).unwrap_or(0)))
        .collect()
}

/// The body's identity and its external buffers in role order.
pub fn digest(body: &Body, colors: Colors, chains: Chains) -> (Number, Vec<Name>) {
    ranked(body, colors, chains, false)
}

/// [`digest`] with the integer of every affine coordinate expression and every static loop extent left out
/// ([`value_numbers_apart`]): what the body shares with the same computation at other offsets, bounds and extents.
/// Bodies that share it are candidates for the copies of one step, which [`prove`](super::prove) decides.
pub fn digest_apart(body: &Body, colors: Colors, chains: Chains) -> (Number, Vec<Name>) {
    ranked(body, colors, chains, true)
}

fn ranked(body: &Body, colors: Colors, chains: Chains, apart: bool) -> (Number, Vec<Name>) {
    let mut rank: HashMap<Name, usize> = HashMap::new();
    let mut cell: Cells = HashMap::new();
    let mut effects: Effects = HashMap::new();
    let hashed = |numbering: &Numbering, effects: &mut Effects| {
        if apart {
            scope_tree_apart(numbering, body, &|_, _| Key::None, effects, &mut Vec::new())
        } else {
            scope_tree(numbering, body, effects)
        }
    };
    let numbering = cells(body, colors, &rank, &mut cell, chains, apart);
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
                    let numbering = numbered(body, &|n| keyed.key(n), chains, apart);
                    (*name, hashed(&numbering, &mut effects))
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
            cells(body, colors, &rank, &mut cell, chains, apart);
        }
    }
    let keyed = Keyed {
        colors,
        rank: &rank,
        cell: &cell,
        pick: None,
    };
    let final_numbering = numbered(body, &|n| keyed.key(n), chains, apart);
    let tree = hashed(&final_numbering, &mut effects);
    let mut buffers = final_numbering.buffers.clone();
    buffers.sort_by_key(|name| rank[name]);
    (tree, buffers)
}
