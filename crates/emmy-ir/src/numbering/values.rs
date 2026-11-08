//! The value numbering: every statement's number and parameters, by its ordinal in the pre-order walk.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::{Key, Number, Param, Tag, expression_key, hash_parts};
use crate::Name;
use crate::body::{Body, StmtRef};
use crate::elementwise::Operation;
use crate::expr::{Expr, ExprType, Expression};
use crate::field::NumOr;
use crate::statement::Statement;
use crate::text::structural_text;

/// One body's numbering, by statement ordinal in pre-order.
#[derive(Default, Debug)]
pub struct Numbering {
    /// Statement (by its position) → ordinal.
    pub index: HashMap<StmtRef, usize>,
    pub refs: Vec<StmtRef>,
    /// Leaf statement → (number, parameters).
    pub numbers: Vec<Option<(Number, Vec<Param>)>>,
    /// Block statement → (header, parameters).
    pub blocks: Vec<Option<(Key, Vec<Param>)>>,
    /// Statement → the statements defining the names it reads.
    pub sources: Vec<Vec<usize>>,
    pub stores: Vec<(Number, Vec<Param>)>,
    pub touch: HashMap<Name, Vec<Key>>,
    pub touching: HashMap<Name, Vec<usize>>,
    pub buffers: Vec<Name>,
    pub states: Vec<Name>,
    pub operands: HashMap<Number, Vec<Number>>,
    pub(super) trees: std::sync::Mutex<HashMap<usize, Vec<Number>>>,
}

impl Numbering {
    pub fn ordinal(&self, here: &StmtRef) -> usize {
        self.index[here]
    }

    pub fn ordinal_of(&self, here: &StmtRef) -> Option<usize> {
        self.index.get(here).copied()
    }

    /// The number applied to its coordinate expressions as spelled.
    pub fn instance(&self, ordinal: usize) -> Option<&(Number, Vec<Param>)> {
        self.numbers[ordinal].as_ref()
    }

    /// A leaf's number or a block's header value: what a resource's touch records.
    fn touch_key(&self, ordinal: usize) -> Key {
        match &self.numbers[ordinal] {
            Some((number, _)) => Key::Num(*number),
            None => match &self.blocks[ordinal] {
                Some((Key::Tuple(header), _)) => header[1].clone(),
                _ => Key::None,
            },
        }
    }
}

type Value = (Number, Vec<Param>);

/// How a chain of one associative commutative operation is keyed: as one node over every term (`Flat`: two
/// associations of one sum number alike, which the normal form's order needs, since it orders a sum's terms by the
/// ranks the numbering gives), or as the tree it is spelled as (`Kept`: the clustered identity spells every FMA-class
/// tree as a chain of one representative, and which operand each term was is the tree's shape).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chains {
    Flat,
    Kept,
}

struct Context<'a> {
    out: Numbering,
    resource_key: &'a dyn Fn(&Name) -> Key,
    chains: Chains,
    env: HashMap<Name, Value>,
    axes: HashMap<Name, Key>,
    loops: Vec<Name>,
    path: Vec<usize>,
    carried: HashSet<Name>,
    binding: Vec<Name>,
    defined_by: HashMap<Name, usize>,
    depths_of: HashMap<Param, Vec<usize>>,
}

fn descriptor(axis: &crate::axis::Axis, stmt: &Statement) -> Key {
    let extent = Key::str(&axis.extent.expression.text());
    let window = match &axis.window {
        None => Key::None,
        Some(window) => Key::tuple([
            Key::Bool(window.partition),
            match &window.parent {
                None => Key::None,
                Some(parent) => Key::str(&parent.extent.expression.text()),
            },
        ]),
    };
    let flags = match stmt {
        Statement::Loop(l) => Key::tuple([Key::Bool(l.unroll), Key::Bool(l.seed)]),
        Statement::StridedLoop(l) => Key::tuple([Key::Bool(l.unroll), Key::Bool(l.seed)]),
        other => Key::str(other.ty().name()),
    };
    Key::tuple([extent, window, flags])
}

fn depths(key: &Key, out: &mut Vec<usize>) {
    if let Key::Tuple(parts) = key {
        if let [Key::Str(tag), Key::Int(depth)] = &parts[..]
            && &**tag == ExprType::Var.text()
        {
            out.push(*depth as usize);
            return;
        }
        for part in parts {
            depths(part, out);
        }
    }
}

impl Context<'_> {
    fn depth_of(&self, name: Name) -> Option<usize> {
        self.binding.iter().rposition(|bound| *bound == name)
    }

    fn reads_axis(&self, e: &Expression) -> bool {
        match &**e {
            Expr::Var(name) => self.axes.contains_key(name),
            Expr::Literal(_) | Expr::Builtin(_) => false,
            _ => e.children().iter().any(|c| self.reads_axis(c)),
        }
    }

    fn coordinate_only(&self, e: &Expression) -> bool {
        match &**e {
            Expr::Var(name) => self.axes.contains_key(name),
            Expr::Literal(_) | Expr::Builtin(_) => true,
            _ => e.children().iter().all(|c| self.coordinate_only(c)),
        }
    }

    /// A rendered expression with every bound axis spelled by its binding depth.
    fn by_depth(&self, e: &Expression) -> Key {
        expression_key(e, &mut |e| match &**e {
            Expr::Var(name) => self
                .depth_of(*name)
                .map(|depth| Key::tuple([Key::str(ExprType::Var.text()), Key::Int(depth as i64)])),
            _ => None,
        })
    }

    /// A value expression as digest material: `abstract_`, with a commutative node's operands in the order of what
    /// they compute rather than how they are spelled, so two spellings of one value number alike. A chain of one
    /// associative commutative operation is one node over every term it sums, so two associations of one sum number
    /// alike too: the normal form orders a sum's terms by ranks the numbering gives, and a number that read the
    /// association would move with the order it decides.
    fn value_key(
        &mut self,
        e: &Expression,
        params: &mut Vec<Param>,
        ops: &mut Vec<Value>,
        touched: &mut Vec<Name>,
    ) -> Key {
        if let Some((op, args)) = e.operation()
            && op.info().commutative
            && args.len() == 2
        {
            let mut terms: Vec<Expression> = Vec::new();
            if op.info().associative && self.chains == Chains::Flat {
                associative_terms(e, op, &mut terms);
            } else {
                terms = args;
            }
            // The sides are keyed twice: once to learn their content order, on state that is then rolled back, and
            // once in that order, so a parameter or resource first met inside this node takes the index the content
            // order gives it and not the one the spelling would.
            let marks = (
                self.out.buffers.len(),
                self.out.states.len(),
                params.len(),
                touched.len(),
            );
            let mut sides: Vec<(Key, &Expression)> = Vec::new();
            for side in &terms {
                let mut own: Vec<Value> = Vec::new();
                let key = self.value_key(side, params, &mut own, touched);
                sides.push((by_content(&key, &own, params), side));
            }
            sides.sort_by(|a, b| a.0.cmp(&b.0));
            self.out.buffers.truncate(marks.0);
            self.out.states.truncate(marks.1);
            params.truncate(marks.2);
            touched.truncate(marks.3);
            let mut keys = Vec::new();
            for (_, side) in sides {
                let mut own: Vec<Value> = Vec::new();
                let key = self.value_key(side, params, &mut own, touched);
                keys.push(shifted(&key, ops.len() as i64));
                ops.extend(own);
            }
            return Key::tuple([
                Key::str(ExprType::Binary.text()),
                Key::str(op.spelling()),
                Key::Tuple(keys),
            ]);
        }
        match &**e {
            Expr::Binary(..)
            | Expr::Call(..)
            | Expr::Ternary(..)
            | Expr::Cast(..)
            | Expr::FlatIndex(..)
                if !(self.reads_axis(e) && self.coordinate_only(e)) =>
            {
                let children: Vec<Key> = e
                    .children()
                    .iter()
                    .map(|child| self.value_key(child, params, ops, touched))
                    .collect();
                let head = match &**e {
                    Expr::Binary(op, ..) => {
                        vec![Key::str(ExprType::Binary.text()), Key::str(op.spelling())]
                    }
                    Expr::Call(name, _) => vec![Key::str(ExprType::Call.text()), Key::Str(*name)],
                    Expr::Ternary(..) => vec![Key::str(ExprType::Ternary.text())],
                    Expr::Cast(target, _) => vec![
                        Key::str(ExprType::Cast.text()),
                        Key::str(&target.to_string()),
                    ],
                    Expr::FlatIndex(buffer, _) => {
                        vec![Key::str(ExprType::FlatIndex.text()), Key::Str(*buffer)]
                    }
                    _ => vec![],
                };
                Key::tuple(head.into_iter().chain([Key::Tuple(children)]))
            }
            _ => self.abstract_(e, params, ops, touched),
        }
    }

    /// The expression with every maximal coordinate-only subexpression a parameter and every SSA read an operand.
    fn abstract_(
        &mut self,
        e: &Expression,
        params: &mut Vec<Param>,
        ops: &mut Vec<Value>,
        touched: &mut Vec<Name>,
    ) -> Key {
        expression_key(e, &mut |e| {
            if self.reads_axis(e) && self.coordinate_only(e) {
                let key = Key::tuple([Key::tag(Tag::Expr), self.by_depth(e)]);
                let index = match params.iter().position(|p| *p == key) {
                    Some(index) => index,
                    None => {
                        params.push(key);
                        params.len() - 1
                    }
                };
                return Some(Key::tuple([
                    Key::tag(Tag::Coordinate),
                    Key::Int(index as i64),
                ]));
            }
            match &**e {
                Expr::Var(name) => Some(self.operand(name, ops, touched)),
                _ => None,
            }
        })
    }

    fn operand(&mut self, name: &Name, ops: &mut Vec<Value>, touched: &mut Vec<Name>) -> Key {
        if let Some(value) = self.env.get(name) {
            ops.push(value.clone());
            return Key::tuple([Key::tag(Tag::Operand), Key::Int(ops.len() as i64 - 1)]);
        }
        if self.carried.contains(name) {
            // A read of a carried state ahead of its update in the loop: the previous step's value.
            return self.resource(name, Tag::State, touched);
        }
        Key::tuple([Key::tag(Tag::Free), Key::Str(*name)])
    }

    fn resource(&mut self, name: &Name, tag: Tag, touched: &mut Vec<Name>) -> Key {
        let held = if tag == Tag::State {
            &mut self.out.states
        } else {
            &mut self.out.buffers
        };
        if !held.contains(name) {
            held.push(*name);
        }
        touched.push(*name);
        Key::tuple([Key::tag(tag), (self.resource_key)(name)])
    }

    fn param_depths(&mut self, param: &Param) -> Vec<usize> {
        if let Some(found) = self.depths_of.get(param) {
            return found.clone();
        }
        let mut found = Vec::new();
        if let Key::Tuple(parts) = param
            && parts.len() == 2
        {
            depths(&parts[1], &mut found);
        }
        found.sort_unstable();
        found.dedup();
        self.depths_of.insert(param.clone(), found.clone());
        found
    }

    /// The parent's parameters (`params` first, then each operand's in order) and each operand's map onto them.
    fn laid_out(ops: &[Value], params: &[Param]) -> (Vec<Param>, Vec<Key>) {
        let mut parent: Vec<Param> = params.to_vec();
        let mut mapped = Vec::with_capacity(ops.len());
        for (number, its) in ops {
            for param in its {
                if !parent.contains(param) {
                    parent.push(param.clone());
                }
            }
            let positions = its.iter().map(|param| {
                Key::Int(
                    parent
                        .iter()
                        .position(|p| p == param)
                        .map_or(-1, |p| p as i64),
                )
            });
            mapped.push(Key::tuple([Key::Num(*number), Key::tuple(positions)]));
        }
        (parent, mapped)
    }

    /// Number `tag`/`payload` over `ops`, unifying parameters by their expression; `bound` axes reduce.
    fn compose(
        &mut self,
        tag: Tag,
        payload: Key,
        mut ops: Vec<Value>,
        params: &[Param],
        bound: &[Name],
        commutative: bool,
    ) -> Value {
        if commutative {
            ops.sort_by_key(|op| op.0);
            let mut groups: Vec<Vec<Value>> = Vec::new();
            for value in ops {
                match groups.last_mut() {
                    Some(group) if group[0].0 == value.0 => group.push(value),
                    _ => groups.push(vec![value]),
                }
            }
            ops = least_layout(&groups, params);
        }
        let (parent, mapped) = Context::laid_out(&ops, params);
        let mut reduced = Key::Tuple(vec![]);
        let mut kept: Vec<Param> = parent.clone();
        if !bound.is_empty() {
            let mut bound_depths: Vec<usize> = bound
                .iter()
                .filter_map(|name| self.depth_of(*name))
                .collect();
            bound_depths.sort_unstable();
            bound_depths.dedup();
            let reads_bound = |this: &mut Context, param: &Param| {
                this.param_depths(param)
                    .iter()
                    .any(|d| bound_depths.contains(d))
            };
            let reduced_indices: Vec<usize> = (0..parent.len())
                .filter(|&i| reads_bound(self, &parent[i]))
                .collect();
            kept = parent
                .iter()
                .filter(|param| !reads_bound(self, param))
                .cloned()
                .collect();
            for param in &parent {
                if reads_bound(self, param) {
                    // A bound composite still reads its free coordinates: they stay as bare parameters of the reduce.
                    for depth in self.param_depths(param) {
                        if !bound_depths.contains(&depth) {
                            let bare = Key::tuple([
                                Key::tag(Tag::Expr),
                                Key::tuple([
                                    Key::str(ExprType::Var.text()),
                                    Key::Int(depth as i64),
                                ]),
                            ]);
                            if !kept.contains(&bare) {
                                kept.push(bare);
                            }
                        }
                    }
                }
            }
            fn bound_form(value: &Key, bound_depths: &[usize], kept: &[Param]) -> Key {
                if let Key::Tuple(parts) = value {
                    if let [Key::Str(tag), Key::Int(depth)] = &parts[..]
                        && &**tag == ExprType::Var.text()
                    {
                        let depth = *depth as usize;
                        if let Some(position) = bound_depths.iter().position(|d| *d == depth) {
                            return Key::tuple([Key::tag(Tag::Bound), Key::Int(position as i64)]);
                        }
                        let bare = Key::tuple([Key::tag(Tag::Expr), value.clone()]);
                        let position = kept
                            .iter()
                            .position(|p| *p == bare)
                            .map_or(-1, |p| p as i64);
                        return Key::tuple([Key::tag(Tag::Parameter), Key::Int(position)]);
                    }
                    return Key::Tuple(
                        parts
                            .iter()
                            .map(|part| bound_form(part, bound_depths, kept))
                            .collect(),
                    );
                }
                value.clone()
            }
            reduced = Key::Tuple(
                reduced_indices
                    .iter()
                    .map(|&index| {
                        let form = match &parent[index] {
                            Key::Tuple(parts) if parts.len() == 2 => &parts[1],
                            other => other,
                        };
                        Key::tuple([
                            Key::Int(index as i64),
                            bound_form(form, &bound_depths, &kept),
                        ])
                    })
                    .collect(),
            );
        }
        // A value is a function of its coordinates whatever their range; a store's sweep is its domain.
        let mut domain = Vec::new();
        if tag == Tag::Store {
            for param in &kept {
                for depth in self.param_depths(param) {
                    domain.push(match self.binding.get(depth) {
                        Some(name) => self.axes.get(name).cloned().unwrap_or(Key::None),
                        None => Key::None,
                    });
                }
            }
        }
        let number = hash_parts([
            Key::tag(tag),
            payload,
            Key::Tuple(mapped),
            reduced,
            Key::Tuple(domain),
        ]);
        self.out
            .operands
            .entry(number)
            .or_insert_with(|| ops.iter().map(|(n, _)| *n).collect());
        (number, kept)
    }

    fn define(&mut self, ordinal: usize, stmt: &Statement, names: &[Name], value: Value) {
        self.out.sources[ordinal] = stmt
            .deps()
            .iter()
            .filter_map(|name| self.defined_by.get(name).copied())
            .collect();
        for name in names {
            self.env.insert(*name, value.clone());
            self.defined_by.insert(*name, ordinal);
        }
        self.out.numbers[ordinal] = Some(value);
    }

    /// Any statement type: its shape with names abstracted, its expressions as parameters and operands.
    fn generic(
        &mut self,
        stmt: &Arc<Statement>,
        ops: &mut Vec<Value>,
        params: &mut Vec<Param>,
        touched: &mut Vec<Name>,
    ) -> Key {
        let expressions: Vec<Key> = stmt
            .expressions()
            .iter()
            .map(|e| self.abstract_(e, params, ops, touched))
            .collect();
        let mut names: HashMap<Name, Name> = HashMap::new();
        // An operand's spelling is the debug text of its value key, which holds a value number: the shell is printed
        // with a stand-in name per operand and the spelling put back in the text, so no such text is interned.
        let mut spellings: Vec<(Name, String)> = Vec::new();
        let defined: Vec<Name> = stmt
            .defines()
            .into_iter()
            .chain(stmt.local_decls())
            .collect();
        for (index, name) in defined.iter().enumerate() {
            names.insert(*name, format!("$d{index}").into());
        }
        for name in stmt.deps() {
            if names.contains_key(&name) {
                continue;
            }
            let key = self.operand(&name, ops, touched);
            names.insert(name, placeholder(&mut spellings, format!("{key:?}")));
        }
        for name in self.axes.keys() {
            names.entry(*name).or_insert_with(|| "$axis".into());
        }
        for (index, axis) in stmt.bound_axes().iter().enumerate() {
            names.insert(axis.name, format!("$b{index}").into());
        }
        for name in stmt
            .external_reads()
            .into_iter()
            .chain(stmt.external_writes())
        {
            if names.contains_key(&name) {
                continue;
            }
            let key = if self.env.contains_key(&name) {
                self.operand(&name, ops, touched)
            } else {
                self.resource(&name, Tag::Buffer, touched)
            };
            names.insert(name, placeholder(&mut spellings, format!("{key:?}")));
        }
        let nested = stmt.nested();
        let shell = if nested.is_empty() {
            stmt.clone()
        } else {
            Statement::rebuild_bodies(stmt, nested.iter().map(|_| Body::empty()).collect())
        };
        let marker = Expression::var("$expr");
        let shell = Statement::map_expressions(&shell, &mut |_| marker.clone());
        let shell = Statement::rename_buffers(&Statement::rename(&shell, &names), &names);
        let mut text = structural_text(&shell);
        for (stand_in, spelling) in spellings {
            text = text.replace(&format!("`{stand_in}`"), &format!("`{spelling}`"));
        }
        Key::tuple([Key::text(text), Key::Tuple(expressions)])
    }

    fn visit(&mut self, body: &Body) {
        self.carried.extend(body.carried_names().iter().cloned());
        for index in 0..body.len() {
            let here = body.at(index);
            let stmt = here.stmt().clone();
            let ordinal = self.out.refs.len();
            self.out.index.insert(here.clone(), ordinal);
            self.out.refs.push(here);
            self.out.numbers.push(None);
            self.out.blocks.push(None);
            self.out.sources.push(Vec::new());
            let mut touched: Vec<Name> = Vec::new();
            let mut params: Vec<Param> = Vec::new();
            let mut ops: Vec<Value> = Vec::new();
            match &*stmt {
                Statement::Load(s) => {
                    let index: Vec<Key> = s
                        .index
                        .iter()
                        .map(|e| self.abstract_(e, &mut params, &mut ops, &mut touched))
                        .collect();
                    let buffer = if self.env.contains_key(&s.input) {
                        self.operand(&s.input, &mut ops, &mut touched)
                    } else {
                        self.resource(&s.input, Tag::Buffer, &mut touched)
                    };
                    let payload = Key::tuple([
                        buffer,
                        Key::Tuple(index),
                        Key::option(s.dtype.as_ref().map(|d| d.text())),
                        Key::Str(s.carried),
                        Key::Int(s.names.len() as i64),
                    ]);
                    let value = self.compose(Tag::Load, payload, ops, &params, &[], false);
                    let names = s.names.to_vec();
                    self.define(ordinal, &stmt, &names, value);
                }
                Statement::Assign(s) => {
                    let payload = Key::tuple([
                        Key::option(s.dtype.as_ref().map(|d| d.text())),
                        self.value_key(&s.value, &mut params, &mut ops, &mut touched),
                    ]);
                    let value = self.compose(Tag::Assign, payload, ops, &params, &[], false);
                    self.define(ordinal, &stmt, std::slice::from_ref(&s.name), value);
                }
                Statement::Init(s) => {
                    let payload = Key::tuple([Key::float(s.identity.0), Key::str(s.dtype.text())]);
                    let value = self.compose(Tag::Init, payload, vec![], &[], &[], false);
                    self.define(ordinal, &stmt, std::slice::from_ref(&s.name), value);
                }
                Statement::Accum(s) => {
                    let redirected = s.base.as_ref().is_some_and(|base| *base != s.name);
                    let value_key = self.value_key(&s.value, &mut params, &mut ops, &mut touched);
                    let base_key = match &s.base {
                        Some(base) if redirected => self.operand(base, &mut ops, &mut touched),
                        _ => Key::None,
                    };
                    let bound: Vec<Name> = if s.axes.is_empty() {
                        self.loops.last().cloned().into_iter().collect()
                    } else {
                        s.axes.to_vec()
                    };
                    let descriptors: Vec<Key> = bound
                        .iter()
                        .map(|axis| self.axes.get(axis).cloned().unwrap_or(Key::None))
                        .collect();
                    let payload = Key::tuple([
                        Key::str(s.op.name()),
                        Key::option(s.dtype.as_ref().map(|d| d.text())),
                        Key::Tuple(descriptors),
                        Key::tuple([value_key, base_key]),
                    ]);
                    let value = self.compose(Tag::Reduce, payload, ops, &params, &bound, false);
                    self.define(ordinal, &stmt, std::slice::from_ref(&s.name), value);
                }
                Statement::Select(s) => {
                    let mut payload = Vec::new();
                    for branch in s.branches.iter() {
                        let value = self.operand(&branch.value, &mut ops, &mut touched);
                        let select =
                            self.abstract_(&branch.select, &mut params, &mut ops, &mut touched);
                        payload.push(Key::tuple([value, select]));
                    }
                    let value =
                        self.compose(Tag::Select, Key::Tuple(payload), ops, &params, &[], false);
                    self.define(ordinal, &stmt, std::slice::from_ref(&s.name), value);
                }
                Statement::Write(s) => {
                    let index: Vec<Key> = s
                        .index
                        .iter()
                        .map(|e| self.abstract_(e, &mut params, &mut ops, &mut touched))
                        .collect();
                    let values: Vec<Key> = s
                        .values
                        .iter()
                        .map(|name| self.operand(name, &mut ops, &mut touched))
                        .collect();
                    let buffer = if self.env.contains_key(&s.output) {
                        self.operand(&s.output, &mut ops, &mut touched)
                    } else {
                        self.resource(&s.output, Tag::Buffer, &mut touched)
                    };
                    let payload = Key::tuple([
                        buffer,
                        Key::Tuple(index),
                        Key::Tuple(values),
                        Key::option(s.value_dtype.as_ref().map(|d| d.text())),
                        Key::Bool(s.atomic),
                        Key::Str(s.swizzle),
                    ]);
                    let value = self.compose(Tag::Store, payload, ops, &params, &[], false);
                    self.define(ordinal, &stmt, &[], value.clone());
                    self.out.stores.push(value);
                }
                Statement::Carry(s) => {
                    let value_key = self.operand(&s.value, &mut ops, &mut touched);
                    let index: Vec<Key> = s
                        .index
                        .iter()
                        .map(|e| self.abstract_(e, &mut params, &mut ops, &mut touched))
                        .collect();
                    let seed = match &s.seed {
                        NumOr::Value(name) => self.resource(name, Tag::Buffer, &mut touched),
                        NumOr::Num(number) => Key::num(*number),
                    };
                    let state = self.resource(&s.name, Tag::State, &mut touched);
                    let payload = Key::tuple([
                        state,
                        Key::Tuple(index),
                        value_key,
                        seed,
                        Key::option(s.dtype.as_ref().map(|d| d.text())),
                    ]);
                    let value = self.compose(Tag::Carry, payload, ops, &params, &[], false);
                    self.define(ordinal, &stmt, std::slice::from_ref(&s.name), value);
                }
                Statement::Pre(s) => {
                    let index: Vec<Key> = s
                        .index
                        .iter()
                        .map(|e| self.abstract_(e, &mut params, &mut ops, &mut touched))
                        .collect();
                    let state = self.resource(&s.carrier, Tag::State, &mut touched);
                    let value = self.compose(
                        Tag::Pre,
                        Key::tuple([state, Key::Tuple(index)]),
                        ops,
                        &params,
                        &[],
                        false,
                    );
                    self.define(ordinal, &stmt, std::slice::from_ref(&s.name), value);
                }
                _ if !stmt.nested().is_empty() => {
                    let header = match &*stmt {
                        Statement::Loop(l) => {
                            Key::tuple([Key::tag(Tag::Loop), descriptor(&l.axis, &stmt)])
                        }
                        Statement::Cond(c) => {
                            let cond = self.abstract_(&c.cond, &mut params, &mut ops, &mut touched);
                            let (number, kept) = self.compose(
                                Tag::Cond,
                                Key::tuple([cond]),
                                ops,
                                &params,
                                &[],
                                false,
                            );
                            params = kept;
                            Key::tuple([Key::tag(Tag::Cond), Key::Num(number)])
                        }
                        _ => {
                            let shape = self.generic(&stmt, &mut ops, &mut params, &mut touched);
                            let (number, kept) =
                                self.compose(Tag::Block, shape, ops, &params, &[], false);
                            params = kept;
                            Key::tuple([Key::tag(Tag::Block), Key::Num(number)])
                        }
                    };
                    self.out.blocks[ordinal] = Some((header, params));
                    self.out.sources[ordinal] = stmt
                        .deps()
                        .iter()
                        .filter_map(|name| self.defined_by.get(name).copied())
                        .collect();
                    let bound = stmt.bound_axes();
                    let saved_env = self.env.clone();
                    let saved_axes: Vec<(Name, Option<Key>)> = bound
                        .iter()
                        .map(|axis| (axis.name, self.axes.get(&axis.name).cloned()))
                        .collect();
                    for axis in &bound {
                        self.axes.insert(axis.name, descriptor(axis, &stmt));
                        self.binding.push(axis.name);
                    }
                    let is_loop = matches!(&*stmt, Statement::Loop(_) | Statement::StridedLoop(_));
                    if is_loop {
                        self.loops.push(bound[0].name);
                    }
                    self.path.push(ordinal);
                    for child in stmt.nested() {
                        self.visit(&child);
                    }
                    self.path.pop();
                    let keep = self.binding.len() - bound.len();
                    self.binding.truncate(keep);
                    if is_loop {
                        self.loops.pop();
                    }
                    let mut exported: Vec<(Name, Value)> = Vec::new();
                    for child in stmt.nested() {
                        for name in child.carried_names() {
                            if let Some(value) = self.env.get(name) {
                                exported.push((*name, value.clone()));
                            }
                        }
                    }
                    self.env = saved_env;
                    self.env.extend(exported);
                    for (name, previous) in saved_axes {
                        match previous {
                            None => {
                                self.axes.remove(&name);
                            }
                            Some(key) => {
                                self.axes.insert(name, key);
                            }
                        }
                    }
                }
                _ => {
                    let shape = self.generic(&stmt, &mut ops, &mut params, &mut touched);
                    let defined: Vec<Name> = stmt
                        .defines()
                        .into_iter()
                        .chain(stmt.local_decls())
                        .collect();
                    let value = self.compose(Tag::Other, shape, ops, &params, &[], false);
                    self.define(ordinal, &stmt, &defined, value);
                }
            }
            for name in touched {
                let key = self.out.touch_key(ordinal);
                self.out.touch.entry(name).or_default().push(key);
                self.out.touching.entry(name).or_default().push(ordinal);
            }
        }
    }
}

/// How many orders of one operation's operands are tried for the least layout before the order is built position by
/// position.
const EXHAUSTIVE_LAYOUTS: usize = 5040;

/// The operands of a commutative operation, grouped by number, in the order whose layout key is least. Every order
/// is tried while the groups allow at most [`EXHAUSTIVE_LAYOUTS`] of them; a larger operation, an unrolled window
/// sum over eight or more shifted reads, takes the order built position by position: canonical and independent of the
/// input order like the full search's, but not always its minimum.
fn least_layout(groups: &[Vec<Value>], params: &[Param]) -> Vec<Value> {
    if groups.iter().all(|group| group.len() <= 1) {
        return groups.iter().flatten().cloned().collect();
    }
    let arrangements = groups
        .iter()
        .try_fold(1usize, |n, group| n.checked_mul(factorial(group.len())?));
    match arrangements {
        Some(n) if n <= EXHAUSTIVE_LAYOUTS => exhaustive_layout(groups, params),
        _ => greedy_layout(groups, params),
    }
}

fn factorial(n: usize) -> Option<usize> {
    (1..=n).try_fold(1usize, |acc, k| acc.checked_mul(k))
}

/// The least layout key over every order of every group.
fn exhaustive_layout(groups: &[Vec<Value>], params: &[Param]) -> Vec<Value> {
    let mut groups = groups.to_vec();
    let mut best: Option<(Vec<Key>, Vec<Value>)> = None;
    for_each_arrangement(&mut groups, 0, &mut |arranged| {
        let order: Vec<Value> = arranged.iter().flatten().cloned().collect();
        let (parent, mapped) = Context::laid_out(&order, params);
        let key = vec![Key::Tuple(mapped), Key::Tuple(parent)];
        if best.as_ref().is_none_or(|(best_key, _)| key < *best_key) {
            best = Some((key, order));
        }
    });
    best.map_or_else(
        || groups.iter().flatten().cloned().collect(),
        |(_, order)| order,
    )
}

/// Every order of every group from `at` on, each given to `f` in place: a group of k costs k! swaps (Heap's
/// algorithm), not k! copies.
fn for_each_arrangement(groups: &mut [Vec<Value>], at: usize, f: &mut dyn FnMut(&[Vec<Value>])) {
    if at == groups.len() {
        f(groups);
        return;
    }
    let n = groups[at].len();
    let mut counts = vec![0usize; n];
    for_each_arrangement(groups, at + 1, f);
    let mut i = 1;
    while i < n {
        if counts[i] < i {
            let j = if i.is_multiple_of(2) { 0 } else { counts[i] };
            groups[at].swap(j, i);
            for_each_arrangement(groups, at + 1, f);
            counts[i] += 1;
            i = 1;
        } else {
            counts[i] = 0;
            i += 1;
        }
    }
}

/// The order built position by position: each position takes the remaining operand of its group whose layout entry,
/// then whose new parameters, are least. Two operands that tie on both are equal and interchangeable.
fn greedy_layout(groups: &[Vec<Value>], params: &[Param]) -> Vec<Value> {
    let mut parent: Vec<Param> = params.to_vec();
    let mut order: Vec<Value> = Vec::new();
    for group in groups {
        let mut remaining: Vec<Value> = group.clone();
        while !remaining.is_empty() {
            let mut best: Option<(Vec<Key>, Vec<Param>, usize)> = None;
            for (index, (_, its)) in remaining.iter().enumerate() {
                let mut appended: Vec<Param> = Vec::new();
                let mut positions: Vec<Key> = Vec::with_capacity(its.len());
                for param in its {
                    let position = match parent.iter().position(|p| p == param) {
                        Some(p) => p,
                        None => match appended.iter().position(|p| p == param) {
                            Some(p) => parent.len() + p,
                            None => {
                                appended.push(param.clone());
                                parent.len() + appended.len() - 1
                            }
                        },
                    };
                    positions.push(Key::Int(position as i64));
                }
                if best
                    .as_ref()
                    .is_none_or(|(entry, added, _)| (&positions, &appended) < (entry, added))
                {
                    best = Some((positions, appended, index));
                }
            }
            let Some((_, appended, index)) = best else {
                break;
            };
            parent.extend(appended);
            order.push(remaining.remove(index));
        }
    }
    order
}

/// Number every statement of `body`; `resource_key` spells a buffer or carried state in a payload.
pub fn value_numbers(
    body: &Body,
    resource_key: &dyn Fn(&Name) -> Key,
    chains: Chains,
) -> Numbering {
    let mut context = Context {
        out: Numbering::default(),
        resource_key,
        chains,
        env: HashMap::new(),
        axes: HashMap::new(),
        loops: Vec::new(),
        path: Vec::new(),
        carried: HashSet::new(),
        binding: Vec::new(),
        defined_by: HashMap::new(),
        depths_of: HashMap::new(),
    };
    context.visit(body);
    context.out
}

/// The stand-in name for the `index`th spelling a shell replaces an operand's name with: a bounded set, which the
/// printed text spells between backticks like the spelling it stands for.
fn placeholder(spellings: &mut Vec<(Name, String)>, spelling: String) -> Name {
    let stand_in = Name::new(&format!("$o{}", spellings.len()));
    spellings.push((stand_in, spelling));
    stand_in
}

/// The terms a chain of the associative operation `op` combines, in spelled order: the operands of every nested
/// `op` node, left to right.
fn associative_terms(e: &Expression, op: Operation, out: &mut Vec<Expression>) {
    match e.operation() {
        Some((nested, args)) if nested == op && args.len() == 2 => {
            for arg in &args {
                associative_terms(arg, op, out);
            }
        }
        _ => out.push(e.clone()),
    }
}

/// `key` with every operand placeholder replaced by the number of the value it stands for, and every coordinate
/// placeholder by its parameter: what an operand computes, for ordering two operands of one commutative node.
fn by_content(key: &Key, ops: &[Value], params: &[Param]) -> Key {
    match key {
        Key::Tuple(parts) => match &parts[..] {
            // The whole value, its parameters with its number: two operands that tie on the number alone can still
            // differ in what they bind, and the material the digest keeps carries both.
            [tag, Key::Int(index)] if *tag == Key::tag(Tag::Operand) => {
                ops.get(*index as usize).map_or_else(
                    || key.clone(),
                    |value| Key::tuple([Key::Num(value.0), Key::Tuple(value.1.clone())]),
                )
            }
            [tag, Key::Int(index)] if *tag == Key::tag(Tag::Coordinate) => params
                .get(*index as usize)
                .cloned()
                .unwrap_or_else(|| key.clone()),
            _ => Key::Tuple(
                parts
                    .iter()
                    .map(|part| by_content(part, ops, params))
                    .collect(),
            ),
        },
        other => other.clone(),
    }
}

/// `key` with every operand placeholder moved `by` positions: a side's operands appended after the other side's.
fn shifted(key: &Key, by: i64) -> Key {
    match key {
        Key::Tuple(parts) => match &parts[..] {
            [tag, Key::Int(index)] if *tag == Key::tag(Tag::Operand) => {
                Key::tuple([tag.clone(), Key::Int(index + by)])
            }
            _ => Key::Tuple(parts.iter().map(|part| shifted(part, by)).collect()),
        },
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn operand(number: u128, params: &[i64]) -> Value {
        (
            number,
            params
                .iter()
                .map(|p| Key::tuple([Key::str(ExprType::Var.text()), Key::Int(*p)]))
                .collect(),
        )
    }

    #[test]
    fn the_greedy_layout_of_shifted_reads_is_the_one_every_order_gives() {
        // Six reads of one buffer at six coordinates, scrambled: the least layout reads them in coordinate order.
        let coordinates = [3, 1, 5, 0, 2, 4];
        let groups = vec![
            vec![operand(2, &[9])],
            coordinates.iter().map(|c| operand(7, &[*c])).collect(),
        ];
        let params = vec![Key::tuple([Key::str(ExprType::Var.text()), Key::Int(9)])];
        let greedy = greedy_layout(&groups, &params);
        assert_eq!(greedy, exhaustive_layout(&groups, &params));
        assert_eq!(greedy, least_layout(&groups, &params));
    }

    #[test]
    fn a_chain_of_one_associative_operation_numbers_alike_under_every_association() {
        // The normal form orders a sum's terms by ranks this numbering gives: a number that read the association
        // would move with the order it decides, and the form would never settle.
        let parse = |sum: &str| {
            crate::text::parse_body(&format!(
                "x = load p[0]\ny = load q[0]\nz = load r[0]\nv = {sum}\nout[0] = v"
            ))
            .unwrap()
        };
        let numbered = |sum: &str| {
            let body = parse(sum);
            let numbering = value_numbers(&body, &|name| Key::Str(*name), Chains::Flat);
            numbering.numbers[3].clone().unwrap()
        };
        let reference = numbered("(x + y) + z");
        for spelling in ["x + (y + z)", "(x + z) + y", "z + (y + x)", "y + x + z"] {
            assert_eq!(numbered(spelling), reference, "{spelling}");
        }
        assert_ne!(numbered("(x - y) + z"), reference);
        assert_ne!(numbered("(x + y) * z"), reference);
        // Kept, the two associations are two trees.
        let kept = |sum: &str| {
            value_numbers(&parse(sum), &|name| Key::Str(*name), Chains::Kept).numbers[3]
                .clone()
                .unwrap()
        };
        assert_ne!(kept("(x + y) + z"), kept("x + (y + z)"));
    }

    #[test]
    fn sixteen_operands_of_one_number_are_laid_out_at_once() {
        let group: Vec<Value> = (0..16).rev().map(|i| operand(7, &[i])).collect();
        let start = std::time::Instant::now();
        let order = least_layout(&[group], &[]);
        assert!(start.elapsed() < std::time::Duration::from_millis(100));
        let sorted: Vec<Value> = (0..16).map(|i| operand(7, &[i])).collect();
        assert_eq!(order, sorted);
    }
}
