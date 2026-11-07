//! Value numbering of a body with coordinates abstracted, and the identity it gives a body.
//!
//! A statement's NUMBER is its kind and payload over its operands' numbers, with every maximal coordinate-only index
//! expression a PARAMETER of the statement, numbered by first appearance. A reduce binds the parameters that mention
//! its axis. Commutative operands order by number. A body's identity is the hash of its SCOPE TREE: every block a
//! node described by its kind and its extent or predicate, every leaf an instance with its coordinates spelled by
//! binding depth, every scope's members sorted. The external buffers and carried states it reads through are keyed
//! by USE, never by spelling, refined where two tie by individualizing each in turn.
//!
//! Statements are numbered by position in the pre-order walk, so no statement is keyed by object identity.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::body::{Body, StmtRef};
use crate::expr::{Expr, Expression, Num};
use crate::field::NumOr;
use crate::kinds::Statement;
use crate::order::ordering_constraints;
use crate::text::{spell_axis, structural_text};
use crate::{Name, locked};

/// Digest material: a tree of primitives, hashed to a number and ordered totally.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum Key {
    None,
    Bool(bool),
    Int(i64),
    Float(u64),
    Str(Arc<str>),
    Num(u128),
    Tuple(Vec<Key>),
}

pub type Number = u128;

impl Key {
    pub fn str(text: &str) -> Key {
        Key::Str(text.into())
    }

    pub fn float(value: f64) -> Key {
        Key::Float((if value == 0.0 { 0.0 } else { value }).to_bits())
    }

    pub fn tuple(parts: impl IntoIterator<Item = Key>) -> Key {
        Key::Tuple(parts.into_iter().collect())
    }

    pub fn num(value: Num) -> Key {
        match value {
            Num::Int(v) => Key::Int(v),
            Num::Float(v) => Key::float(v),
            Num::Bool(v) => Key::Bool(v),
        }
    }

    pub fn option(text: Option<&str>) -> Key {
        text.map_or(Key::None, Key::str)
    }

    fn feed(&self, hasher: &mut blake3::Hasher) {
        match self {
            Key::None => hasher.update(&[0]),
            Key::Bool(v) => hasher.update(&[1, *v as u8]),
            Key::Int(v) => {
                hasher.update(&[2]);
                hasher.update(&v.to_le_bytes())
            }
            Key::Float(v) => {
                hasher.update(&[3]);
                hasher.update(&v.to_le_bytes())
            }
            Key::Str(s) => {
                hasher.update(&[4]);
                hasher.update(&(s.len() as u64).to_le_bytes());
                hasher.update(s.as_bytes())
            }
            Key::Num(n) => {
                hasher.update(&[5]);
                hasher.update(&n.to_le_bytes())
            }
            Key::Tuple(parts) => {
                hasher.update(&[6]);
                hasher.update(&(parts.len() as u64).to_le_bytes());
                for part in parts {
                    part.feed(hasher);
                }
                hasher
            }
        };
    }

    pub fn hash(&self) -> Number {
        let mut hasher = blake3::Hasher::new();
        self.feed(&mut hasher);
        let bytes = hasher.finalize();
        let mut head = [0u8; 16];
        head.copy_from_slice(&bytes.as_bytes()[..16]);
        u128::from_le_bytes(head)
    }
}

/// The hash of a sequence of parts.
pub fn hash_parts(parts: impl IntoIterator<Item = Key>) -> Number {
    Key::tuple(parts).hash()
}

/// A parameter: the form of a coordinate-only expression, every axis spelled by its binding depth.
pub type Param = Key;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StmtKind {
    Load,
    Assign,
    Init,
    Reduce,
    Select,
    Store,
    Carry,
    Pre,
    Other,
    Block,
}

/// One body's numbering, by statement ordinal in pre-order.
#[derive(Default, Debug)]
pub struct Numbering {
    /// Statement (by the address of its shared object) → ordinal.
    pub index: HashMap<usize, usize>,
    pub refs: Vec<StmtRef>,
    /// Leaf statement → (number, parameters).
    pub numbers: Vec<Option<(Number, Vec<Param>)>>,
    pub kind: Vec<StmtKind>,
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
    trees: std::sync::Mutex<HashMap<usize, Vec<Number>>>,
}

/// The address a numbering keys a statement by: that of its shared object, which `distinct` makes unique.
pub fn ptr(stmt: &Arc<Statement>) -> usize {
    Arc::as_ptr(stmt) as usize
}

impl Numbering {
    pub fn ordinal(&self, stmt: &Arc<Statement>) -> usize {
        self.index[&ptr(stmt)]
    }

    pub fn ordinal_of(&self, stmt: &Arc<Statement>) -> Option<usize> {
        self.index.get(&ptr(stmt)).copied()
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

struct Context<'a> {
    out: Numbering,
    resource_key: &'a dyn Fn(&Name) -> Key,
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
        other => Key::str(other.kind_name()),
    };
    Key::tuple([extent, window, flags])
}

fn depths(key: &Key, out: &mut Vec<usize>) {
    if let Key::Tuple(parts) = key {
        if let [Key::Str(tag), Key::Int(depth)] = &parts[..]
            && &**tag == "Var"
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
    fn depth_of(&self, name: &str) -> Option<usize> {
        self.binding.iter().rposition(|bound| &**bound == name)
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
        match &**e {
            Expr::Var(name) => match self.depth_of(name) {
                Some(depth) => Key::tuple([Key::str("Var"), Key::Int(depth as i64)]),
                None => Key::tuple([Key::str("Var"), Key::Str(name.clone())]),
            },
            Expr::Literal(lit) => Key::tuple([
                Key::str("Literal"),
                Key::num(lit.value),
                Key::Str(lit.dtype.clone()),
            ]),
            Expr::Builtin(name) => Key::tuple([Key::str("Builtin"), Key::Str(name.clone())]),
            Expr::Binary(op, l, r) => Key::tuple([
                Key::str("Binary"),
                Key::str(op.text()),
                self.by_depth(l),
                self.by_depth(r),
            ]),
            Expr::Call(name, args) => Key::tuple([
                Key::str("Call"),
                Key::Str(name.clone()),
                Key::tuple(args.iter().map(|a| self.by_depth(a))),
            ]),
            Expr::Ternary(a, b, c) => Key::tuple([
                Key::str("Ternary"),
                self.by_depth(a),
                self.by_depth(b),
                self.by_depth(c),
            ]),
            Expr::Cast(dtype, inner) => Key::tuple([
                Key::str("Cast"),
                Key::Str(dtype.clone()),
                self.by_depth(inner),
            ]),
            Expr::FlatIndex(buffer, index) => Key::tuple([
                Key::str("FlatIndex"),
                Key::Str(buffer.clone()),
                Key::tuple(index.iter().map(|a| self.by_depth(a))),
            ]),
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
        if self.reads_axis(e) && self.coordinate_only(e) {
            let key = Key::tuple([Key::str("expr"), self.by_depth(e)]);
            let index = match params.iter().position(|p| *p == key) {
                Some(index) => index,
                None => {
                    params.push(key);
                    params.len() - 1
                }
            };
            return Key::tuple([Key::str("c"), Key::Int(index as i64)]);
        }
        match &**e {
            Expr::Var(name) => self.operand(name, ops, touched),
            Expr::Literal(lit) => Key::tuple([
                Key::str("Literal"),
                Key::num(lit.value),
                Key::Str(lit.dtype.clone()),
            ]),
            Expr::Builtin(name) => Key::tuple([Key::str("Builtin"), Key::Str(name.clone())]),
            Expr::Binary(op, l, r) => {
                let (l, r) = (
                    self.abstract_(l, params, ops, touched),
                    self.abstract_(r, params, ops, touched),
                );
                Key::tuple([Key::str("Binary"), Key::str(op.text()), l, r])
            }
            Expr::Call(name, args) => {
                let args: Vec<Key> = args
                    .iter()
                    .map(|a| self.abstract_(a, params, ops, touched))
                    .collect();
                Key::tuple([Key::str("Call"), Key::Str(name.clone()), Key::Tuple(args)])
            }
            Expr::Ternary(a, b, c) => {
                let parts = [a, b, c].map(|x| self.abstract_(x, params, ops, touched));
                Key::tuple([
                    Key::str("Ternary"),
                    parts[0].clone(),
                    parts[1].clone(),
                    parts[2].clone(),
                ])
            }
            Expr::Cast(dtype, inner) => {
                let inner = self.abstract_(inner, params, ops, touched);
                Key::tuple([Key::str("Cast"), Key::Str(dtype.clone()), inner])
            }
            Expr::FlatIndex(buffer, index) => {
                let index: Vec<Key> = index
                    .iter()
                    .map(|a| self.abstract_(a, params, ops, touched))
                    .collect();
                Key::tuple([
                    Key::str("FlatIndex"),
                    Key::Str(buffer.clone()),
                    Key::Tuple(index),
                ])
            }
        }
    }

    fn operand(&mut self, name: &Name, ops: &mut Vec<Value>, touched: &mut Vec<Name>) -> Key {
        if let Some(value) = self.env.get(name) {
            ops.push(value.clone());
            return Key::tuple([Key::str("v"), Key::Int(ops.len() as i64 - 1)]);
        }
        if self.carried.contains(name) {
            // A read of a carried state ahead of its update in the loop: the previous step's value.
            return self.resource(name, "state", touched);
        }
        Key::tuple([Key::str("free"), Key::Str(name.clone())])
    }

    /// An operation's argument no statement ahead of it binds: a carried state read ahead of its update, keyed by
    /// use, or a name free in the body, spelled.
    fn unbound(&mut self, name: &Name, touched: &mut Vec<Name>) -> Key {
        if self.carried.contains(name) {
            self.resource(name, "state", touched)
        } else {
            Key::Str(name.clone())
        }
    }

    fn resource(&mut self, name: &Name, kind: &str, touched: &mut Vec<Name>) -> Key {
        let held = if kind == "state" {
            &mut self.out.states
        } else {
            &mut self.out.buffers
        };
        if !held.contains(name) {
            held.push(name.clone());
        }
        touched.push(name.clone());
        Key::tuple([Key::str(kind), (self.resource_key)(name)])
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

    /// Number `kind`/`payload` over `ops`, unifying parameters by their expression; `bound` axes reduce.
    fn compose(
        &mut self,
        kind: &str,
        payload: Key,
        mut ops: Vec<Value>,
        params: &[Param],
        bound: &[Name],
        commutative: bool,
    ) -> Value {
        if commutative {
            ops.sort_by(|a, b| a.0.cmp(&b.0));
            let mut groups: Vec<Vec<Value>> = Vec::new();
            for value in ops {
                match groups.last_mut() {
                    Some(group) if group[0].0 == value.0 => group.push(value),
                    _ => groups.push(vec![value]),
                }
            }
            let flat: Vec<Value> = groups.iter().flatten().cloned().collect();
            ops = if groups.iter().any(|group| group.len() > 1) {
                let arrangements: Vec<Vec<Vec<Value>>> =
                    groups.iter().map(|group| permutations(group)).collect();
                let mut best: Option<(Vec<Key>, Vec<Value>)> = None;
                for_each_product(&arrangements, &mut |choice| {
                    let order: Vec<Value> = choice
                        .iter()
                        .flat_map(|group| group.iter().cloned())
                        .collect();
                    let (parent, mapped) = Context::laid_out(&order, params);
                    let key = vec![Key::Tuple(mapped), Key::Tuple(parent)];
                    if best.as_ref().is_none_or(|(best_key, _)| key < *best_key) {
                        best = Some((key, order));
                    }
                });
                best.map_or(flat, |(_, order)| order)
            } else {
                flat
            };
        }
        let (parent, mapped) = Context::laid_out(&ops, params);
        let mut reduced = Key::Tuple(vec![]);
        let mut kept: Vec<Param> = parent.clone();
        if !bound.is_empty() {
            let mut bound_depths: Vec<usize> = bound
                .iter()
                .filter_map(|name| self.depth_of(name))
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
                                Key::str("expr"),
                                Key::tuple([Key::str("Var"), Key::Int(depth as i64)]),
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
                        && &**tag == "Var"
                    {
                        let depth = *depth as usize;
                        if let Some(position) = bound_depths.iter().position(|d| *d == depth) {
                            return Key::tuple([Key::str("bound"), Key::Int(position as i64)]);
                        }
                        let bare = Key::tuple([Key::str("expr"), value.clone()]);
                        let position = kept
                            .iter()
                            .position(|p| *p == bare)
                            .map_or(-1, |p| p as i64);
                        return Key::tuple([Key::str("parameter"), Key::Int(position)]);
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
        if kind == "store" {
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
            Key::str(kind),
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

    fn define(
        &mut self,
        ordinal: usize,
        stmt: &Statement,
        names: &[Name],
        value: Value,
        kind: StmtKind,
    ) {
        self.out.sources[ordinal] = stmt
            .deps()
            .iter()
            .filter_map(|name| self.defined_by.get(name).copied())
            .collect();
        for name in names {
            self.env.insert(name.clone(), value.clone());
            self.defined_by.insert(name.clone(), ordinal);
        }
        self.out.numbers[ordinal] = Some(value);
        self.out.kind[ordinal] = kind;
    }

    /// Any statement kind: its shape with names abstracted, its expressions as parameters and operands.
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
        let defined: Vec<Name> = stmt
            .defines()
            .into_iter()
            .chain(stmt.local_decls())
            .collect();
        for (index, name) in defined.iter().enumerate() {
            names.insert(name.clone(), format!("$d{index}").into());
        }
        for name in stmt.deps() {
            if names.contains_key(&name) {
                continue;
            }
            let key = self.operand(&name, ops, touched);
            names.insert(name, format!("{key:?}").into());
        }
        for name in self.axes.keys() {
            names.entry(name.clone()).or_insert_with(|| "$axis".into());
        }
        for (index, axis) in stmt.bound_axes().iter().enumerate() {
            names.insert(axis.name.clone(), format!("$b{index}").into());
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
                self.resource(&name, "buffer", touched)
            };
            names.insert(name, format!("{key:?}").into());
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
        Key::tuple([Key::str(&structural_text(&shell)), Key::Tuple(expressions)])
    }

    fn visit(&mut self, body: &Body) {
        self.carried.extend(body.carried_names().iter().cloned());
        for index in 0..body.len() {
            let here = body.at(index);
            let stmt = here.stmt().clone();
            let ordinal = self.out.refs.len();
            self.out.index.insert(ptr(&stmt), ordinal);
            self.out.refs.push(here);
            self.out.numbers.push(None);
            self.out.kind.push(StmtKind::Other);
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
                        self.resource(&s.input, "buffer", &mut touched)
                    };
                    let payload = Key::tuple([
                        buffer,
                        Key::Tuple(index),
                        Key::option(s.dtype.as_ref().map(|d| d.name())),
                        Key::Str(s.carried.clone()),
                        Key::Int(s.names.len() as i64),
                    ]);
                    let value = self.compose("load", payload, ops, &params, &[], false);
                    let names = s.names.to_vec();
                    self.define(ordinal, &stmt, &names, value, StmtKind::Load);
                }
                Statement::Assign(s) => {
                    let ops: Vec<Value> = s
                        .args
                        .iter()
                        .filter_map(|arg| self.env.get(arg).cloned())
                        .collect();
                    let commutative = s.op.commutative;
                    let mut frees: Vec<Key> = s
                        .args
                        .iter()
                        .filter(|arg| !self.env.contains_key(*arg))
                        .cloned()
                        .collect::<Vec<_>>()
                        .iter()
                        .map(|arg| self.unbound(arg, &mut touched))
                        .collect();
                    if commutative {
                        frees.sort();
                    }
                    let payload = Key::tuple([
                        Key::Str(s.op.name.clone()),
                        Key::option(s.dtype.as_ref().map(|d| d.name())),
                        Key::Tuple(frees),
                    ]);
                    let value = self.compose("assign", payload, ops, &[], &[], commutative);
                    self.define(
                        ordinal,
                        &stmt,
                        std::slice::from_ref(&s.name),
                        value,
                        StmtKind::Assign,
                    );
                }
                Statement::Affine(s) => {
                    // Each term numbers as its coefficient over its value, and the sum lists the terms by number.
                    let mut ops: Vec<Value> = Vec::new();
                    let mut frees: Vec<Key> = Vec::new();
                    for (coefficient, arg) in s.terms() {
                        match self.env.get(arg).cloned() {
                            Some(value) => ops.push(self.compose(
                                "term",
                                Key::tuple([Key::float(coefficient)]),
                                vec![value],
                                &[],
                                &[],
                                false,
                            )),
                            None => {
                                let key = self.unbound(arg, &mut touched);
                                frees.push(Key::tuple([Key::float(coefficient), key]));
                            }
                        }
                    }
                    frees.sort();
                    let payload = Key::tuple([
                        Key::float(s.offset.0),
                        Key::Tuple(frees),
                        Key::option(s.dtype.as_ref().map(|d| d.name())),
                    ]);
                    let value = self.compose("affine", payload, ops, &[], &[], true);
                    self.define(
                        ordinal,
                        &stmt,
                        std::slice::from_ref(&s.name),
                        value,
                        StmtKind::Assign,
                    );
                }
                Statement::Init(s) => {
                    let payload = Key::tuple([Key::float(s.identity.0), Key::str(s.dtype.name())]);
                    let value = self.compose("init", payload, vec![], &[], &[], false);
                    self.define(
                        ordinal,
                        &stmt,
                        std::slice::from_ref(&s.name),
                        value,
                        StmtKind::Init,
                    );
                }
                Statement::Accum(s) => {
                    let redirected = s.base.as_ref().is_some_and(|base| *base != s.name);
                    let value_key = self.operand(&s.value, &mut ops, &mut touched);
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
                        Key::Str(s.op.name.clone()),
                        Key::option(s.dtype.as_ref().map(|d| d.name())),
                        Key::Tuple(descriptors),
                        Key::tuple([value_key, base_key]),
                    ]);
                    let value = self.compose("reduce", payload, ops, &params, &bound, false);
                    self.define(
                        ordinal,
                        &stmt,
                        std::slice::from_ref(&s.name),
                        value,
                        StmtKind::Reduce,
                    );
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
                        self.compose("select", Key::Tuple(payload), ops, &params, &[], false);
                    self.define(
                        ordinal,
                        &stmt,
                        std::slice::from_ref(&s.name),
                        value,
                        StmtKind::Select,
                    );
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
                        self.resource(&s.output, "buffer", &mut touched)
                    };
                    let payload = Key::tuple([
                        buffer,
                        Key::Tuple(index),
                        Key::Tuple(values),
                        Key::option(s.value_dtype.as_ref().map(|d| d.name())),
                        Key::Bool(s.atomic),
                        Key::Str(s.swizzle.clone()),
                    ]);
                    let value = self.compose("store", payload, ops, &params, &[], false);
                    self.define(ordinal, &stmt, &[], value.clone(), StmtKind::Store);
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
                        NumOr::Value(name) => self.resource(name, "buffer", &mut touched),
                        NumOr::Num(number) => Key::num(*number),
                    };
                    let state = self.resource(&s.name, "state", &mut touched);
                    let payload = Key::tuple([
                        state,
                        Key::Tuple(index),
                        value_key,
                        seed,
                        Key::option(s.dtype.as_ref().map(|d| d.name())),
                    ]);
                    let value = self.compose("carry", payload, ops, &params, &[], false);
                    self.define(
                        ordinal,
                        &stmt,
                        std::slice::from_ref(&s.name),
                        value,
                        StmtKind::Carry,
                    );
                }
                Statement::Pre(s) => {
                    let index: Vec<Key> = s
                        .index
                        .iter()
                        .map(|e| self.abstract_(e, &mut params, &mut ops, &mut touched))
                        .collect();
                    let state = self.resource(&s.carrier, "state", &mut touched);
                    let value = self.compose(
                        "pre",
                        Key::tuple([state, Key::Tuple(index)]),
                        ops,
                        &params,
                        &[],
                        false,
                    );
                    self.define(
                        ordinal,
                        &stmt,
                        std::slice::from_ref(&s.name),
                        value,
                        StmtKind::Pre,
                    );
                }
                _ if !stmt.nested().is_empty() => {
                    let header = match &*stmt {
                        Statement::Loop(l) => {
                            Key::tuple([Key::str("loop"), descriptor(&l.axis, &stmt)])
                        }
                        Statement::Cond(c) => {
                            let cond = self.abstract_(&c.cond, &mut params, &mut ops, &mut touched);
                            let (number, kept) =
                                self.compose("cond", Key::tuple([cond]), ops, &params, &[], false);
                            params = kept;
                            Key::tuple([Key::str("cond"), Key::Num(number)])
                        }
                        _ => {
                            let shape = self.generic(&stmt, &mut ops, &mut params, &mut touched);
                            let (number, kept) =
                                self.compose("block", shape, ops, &params, &[], false);
                            params = kept;
                            Key::tuple([Key::str("block"), Key::Num(number)])
                        }
                    };
                    self.out.blocks[ordinal] = Some((header, params));
                    self.out.kind[ordinal] = StmtKind::Block;
                    self.out.sources[ordinal] = stmt
                        .deps()
                        .iter()
                        .filter_map(|name| self.defined_by.get(name).copied())
                        .collect();
                    let bound = stmt.bound_axes();
                    let saved_env = self.env.clone();
                    let saved_axes: Vec<(Name, Option<Key>)> = bound
                        .iter()
                        .map(|axis| (axis.name.clone(), self.axes.get(&axis.name).cloned()))
                        .collect();
                    for axis in &bound {
                        self.axes.insert(axis.name.clone(), descriptor(axis, &stmt));
                        self.binding.push(axis.name.clone());
                    }
                    let is_loop = matches!(&*stmt, Statement::Loop(_) | Statement::StridedLoop(_));
                    if is_loop {
                        self.loops.push(bound[0].name.clone());
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
                                exported.push((name.clone(), value.clone()));
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
                    let value = self.compose("other", shape, ops, &params, &[], false);
                    self.define(ordinal, &stmt, &defined, value, StmtKind::Other);
                }
            }
            for name in touched {
                let key = self.out.touch_key(ordinal);
                self.out.touch.entry(name.clone()).or_default().push(key);
                self.out.touching.entry(name).or_default().push(ordinal);
            }
        }
    }
}

fn permutations<T: Clone>(items: &[T]) -> Vec<Vec<T>> {
    if items.len() <= 1 {
        return vec![items.to_vec()];
    }
    let mut out = Vec::new();
    for (index, item) in items.iter().enumerate() {
        let mut rest = items.to_vec();
        rest.remove(index);
        for mut tail in permutations(&rest) {
            tail.insert(0, item.clone());
            out.push(tail);
        }
    }
    out
}

fn for_each_product<T: Clone>(groups: &[Vec<T>], f: &mut dyn FnMut(&[T])) {
    fn walk<T: Clone>(groups: &[Vec<T>], chosen: &mut Vec<T>, f: &mut dyn FnMut(&[T])) {
        if chosen.len() == groups.len() {
            f(chosen);
            return;
        }
        for item in &groups[chosen.len()] {
            chosen.push(item.clone());
            walk(groups, chosen, f);
            chosen.pop();
        }
    }
    walk(groups, &mut Vec::new(), f);
}

/// `body` with every statement its own object: a numbering is keyed by object, and a body may hold one shared
/// statement twice.
pub fn distinct(body: &Body) -> Body {
    let mut seen: HashSet<usize> = HashSet::new();
    let mut repeated = false;
    body.walk(&mut |here| {
        if !seen.insert(ptr(here.stmt())) {
            repeated = true;
        }
    });
    if !repeated {
        return body.clone();
    }
    fn copy(stmt: &Arc<Statement>) -> Arc<Statement> {
        let nested = stmt.nested();
        if nested.is_empty() {
            return Arc::new((**stmt).clone());
        }
        let bodies: Vec<Body> = nested
            .iter()
            .map(|body| Body::new(body.stmts().iter().map(copy).collect()))
            .collect();
        Arc::new((*Statement::rebuild_bodies(stmt, bodies)).clone())
    }
    Body::new(body.stmts().iter().map(copy).collect())
}

/// Number every statement of `body`, which holds every statement once (`distinct`); `resource_key` spells a buffer
/// or carried state in a payload.
pub fn value_numbers(body: &Body, resource_key: &dyn Fn(&Name) -> Key) -> Numbering {
    let mut context = Context {
        out: Numbering::default(),
        resource_key,
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

pub fn spelled_key(name: &Name) -> Key {
    Key::Str(name.clone())
}

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
        let here = body.at(index);
        let ordinal = numbering.ordinal(here.stmt());
        let mut preceding: Vec<Key> = sources
            .iter()
            .map(|source| Key::Num(members[*source]))
            .collect();
        preceding.sort();
        let member = match &numbering.blocks[ordinal] {
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
                    Key::str("block"),
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
                    Key::str("stmt"),
                    number,
                    Key::Tuple(params),
                    Key::Tuple(preceding),
                ])
            }
        };
        members.push(member);
    }
    members.sort_unstable();
    hash_parts(members.into_iter().map(Key::Num))
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
        out.insert(name.clone(), numbers);
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
                Key::str("fixed"),
                Key::Int(*r as i64),
                paint(self.colors, name),
            ]);
        }
        let tag = if self.pick == Some(name) {
            "pick"
        } else {
            "open"
        };
        Key::tuple([
            Key::str(tag),
            paint(self.colors, name),
            self.cell.get(name).cloned().unwrap_or(Key::None),
        ])
    }
}

fn partition(cells: &Cells) -> Vec<Vec<Name>> {
    let mut groups: HashMap<&Key, Vec<Name>> = HashMap::new();
    for (name, value) in cells {
        groups.entry(value).or_default().push(name.clone());
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
fn cells(body: &Body, colors: Colors, rank: &HashMap<Name, usize>, cell: &mut Cells) -> Numbering {
    let number = |cell: &Cells| {
        value_numbers(body, &|name| {
            Keyed {
                colors,
                rank,
                cell,
                pick: None,
            }
            .key(name)
        })
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
                    name.clone(),
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
    cells(body, colors, &HashMap::new(), &mut cell);
    let mut values: Vec<&Key> = cell.values().collect();
    values.sort();
    values.dedup();
    cell.iter()
        .map(|(name, value)| {
            (
                name.clone(),
                values.iter().position(|v| *v == value).unwrap_or(0),
            )
        })
        .collect()
}

/// The body's identity and its external buffers in role order.
pub fn digest(body: &Body, colors: Colors) -> (Number, Vec<Name>) {
    let mut rank: HashMap<Name, usize> = HashMap::new();
    let mut cell: Cells = HashMap::new();
    let mut effects: Effects = HashMap::new();
    let numbering = cells(body, colors, &rank, &mut cell);
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
                group.push(unranked[position].clone());
                position += 1;
            }
            if group.len() == 1 {
                let r = rank.len();
                rank.insert(group[0].clone(), r);
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
                    let tree =
                        scope_tree(&value_numbers(body, &|n| keyed.key(n)), body, &mut effects);
                    (name.clone(), tree)
                })
                .collect();
            group.sort_by(|a, b| certificate[a].cmp(&certificate[b]).then_with(|| a.cmp(b)));
            for name in &group {
                let r = rank.len();
                rank.insert(name.clone(), r);
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
            cells(body, colors, &rank, &mut cell);
        }
    }
    let keyed = Keyed {
        colors,
        rank: &rank,
        cell: &cell,
        pick: None,
    };
    let final_numbering = value_numbers(body, &|n| keyed.key(n));
    let tree = scope_tree(&final_numbering, body, &mut effects);
    let mut buffers = final_numbering.buffers.clone();
    buffers.sort_by_key(|name| rank[name]);
    (tree, buffers)
}

/// A hex spelling of a number, for the identity key.
pub fn hex(number: Number) -> String {
    format!("{number:032x}")
}

/// The text an axis spells, for a descriptor.
pub fn axis_text(axis: &crate::axis::Axis) -> String {
    spell_axis(axis, None).unwrap_or_default()
}
