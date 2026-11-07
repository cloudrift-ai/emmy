//! The statement protocol: what every kind answers, written once over the roles of its fields. Reads (`deps`,
//! `defines`, the buffers, the expressions, the axes, the child bodies) and their rebuilding halves (`with_bodies`,
//! `map_expressions`, `rename_buffers`, and `rewrite` behind `rename` and `substitute`). The few members no role
//! gives are stated here by kind: an `Accum`'s reduced axes follow a substitution that splits an axis, a `Let` and
//! a `Call` read buffers their fields do not name, a `StridedLoop` with an `end` binds one more name than its axis.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::body::{Body, Normalized, Subroutine, exposed_defines, free_names, loop_parts, unique};
use crate::expr::{Expr, Expression, LitType};
use crate::kinds::axis::Axis;
use crate::kinds::elementwise::Op;
use crate::kinds::field::{
    AxisRef, DType, Exprs, Float, Names, NumOr, Rewriter, SubroutineRef, Visitor,
};
use crate::kinds::*;
use crate::{Error, Name, Result};

/// What a rewrite does to a statement: every SSA name through `rename`, every expression through `sigma`, every
/// bound axis through `axis_fn`. A `None` from `rename` keeps the name.
pub struct Rewrite<'a> {
    pub rename: &'a dyn Fn(&str) -> Option<Name>,
    pub sigma: &'a HashMap<Name, Expression>,
    pub axis_fn: Option<&'a dyn Fn(&AxisRef) -> AxisRef>,
}

fn keep(_name: &str) -> Option<Name> {
    None
}

fn sorted_free_vars(expression: &Expression) -> Vec<Name> {
    let mut vars = expression.free_vars();
    vars.sort();
    vars
}

struct Deps {
    reads: Vec<Name>,
    locals: Vec<Name>,
    carried: Vec<Arc<Statement>>,
}

impl Visitor for Deps {
    fn name(&mut self, roles: u32, name: &Name) {
        if roles & READING != 0 {
            self.reads.push(name.clone());
        }
        if roles & LOCAL != 0 {
            self.locals.push(name.clone());
        }
    }
    fn expression(&mut self, roles: u32, expression: &Expression) {
        if roles & READING != 0 {
            // A field read through: the names its expressions mention, each listed once.
            for name in sorted_free_vars(expression) {
                if !self.reads.contains(&name) {
                    self.reads.push(name);
                }
            }
        }
    }
    fn carried(&mut self, _roles: u32, stmt: &Arc<Statement>) {
        self.carried.push(stmt.clone());
    }
}

struct NamesByRole {
    roles: u32,
    out: Vec<Name>,
}

impl Visitor for NamesByRole {
    fn name(&mut self, roles: u32, name: &Name) {
        if roles & self.roles != 0 {
            self.out.push(name.clone());
        }
    }
}

/// Collects a member over the statement's own fields and the statements it carries.
struct Carried<'a> {
    roles: u32,
    own: Vec<Name>,
    member: &'a dyn Fn(&Statement) -> Vec<Name>,
}

impl Visitor for Carried<'_> {
    fn name(&mut self, roles: u32, name: &Name) {
        if roles & self.roles != 0 {
            self.own.push(name.clone());
        }
    }
    fn body(&mut self, roles: u32, body: &Body) {
        if roles & TERM != 0 {
            for stmt in body.stmts() {
                self.own.extend((self.member)(stmt));
            }
        }
    }
    fn carried(&mut self, _roles: u32, stmt: &Arc<Statement>) {
        self.own.extend((self.member)(stmt));
    }
}

struct Expressions(Vec<Expression>);

impl Visitor for Expressions {
    fn expression(&mut self, roles: u32, expression: &Expression) {
        if roles & EXPRESSION != 0 {
            self.0.push(expression.clone());
        }
    }
    fn body(&mut self, roles: u32, body: &Body) {
        if roles & TERM != 0 {
            for stmt in body.stmts() {
                self.0.extend(stmt.expressions());
            }
        }
    }
    fn carried(&mut self, _roles: u32, stmt: &Arc<Statement>) {
        self.0.extend(stmt.expressions());
    }
}

struct BoundAxes(Vec<AxisRef>);

impl Visitor for BoundAxes {
    fn axis(&mut self, _roles: u32, axis: &AxisRef) {
        self.0.push(axis.clone());
    }
}

struct Nested(Vec<Body>);

impl Visitor for Nested {
    fn body(&mut self, roles: u32, body: &Body) {
        if roles & BODY != 0 {
            self.0.push(body.clone());
        }
    }
}

struct WithBodies {
    bodies: std::vec::IntoIter<Body>,
    short: bool,
}

impl Rewriter for WithBodies {
    fn body(&mut self, roles: u32, body: &Body) -> Body {
        if roles & BODY == 0 {
            return body.clone();
        }
        match self.bodies.next() {
            Some(next) => next,
            None => {
                self.short = true;
                body.clone()
            }
        }
    }
}

struct MapExpressions<'a>(&'a mut dyn FnMut(&Expression) -> Expression);

impl Rewriter for MapExpressions<'_> {
    fn expression(&mut self, _roles: u32, expression: &Expression) -> Expression {
        (self.0)(expression)
    }
    fn body(&mut self, roles: u32, body: &Body) -> Body {
        if roles & TERM != 0 {
            body.map_each(&mut |stmt| Statement::map_expressions(stmt, self.0))
        } else {
            body.clone()
        }
    }
    fn carried(&mut self, _roles: u32, stmt: &Arc<Statement>) -> Arc<Statement> {
        Statement::map_expressions(stmt, self.0)
    }
}

/// A buffer renaming in progress: the mapping, and each shared subroutine renamed once, however many calls name it.
pub struct Renaming<'a> {
    mapping: &'a HashMap<Name, Name>,
    subroutines: RefCell<HashMap<usize, Arc<Subroutine>>>,
}

impl<'a> Renaming<'a> {
    pub fn new(mapping: &'a HashMap<Name, Name>) -> Renaming<'a> {
        Renaming {
            mapping,
            subroutines: RefCell::new(HashMap::new()),
        }
    }

    fn rename(&self, name: &Name) -> Name {
        self.mapping
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.clone())
    }

    fn subroutine(&self, target: &Arc<Subroutine>) -> Arc<Subroutine> {
        let key = Arc::as_ptr(target) as usize;
        if let Some(renamed) = self.subroutines.borrow().get(&key) {
            return renamed.clone();
        }
        let buffers: Vec<Name> = target
            .buffers
            .iter()
            .map(|name| self.rename(name))
            .collect();
        let renamed = if buffers[..] == target.buffers[..] {
            target.clone()
        } else {
            target.with_buffers(buffers.into())
        };
        self.subroutines.borrow_mut().insert(key, renamed.clone());
        renamed
    }
}

struct RenameBuffers<'a, 'b>(&'b Renaming<'a>);

impl Rewriter for RenameBuffers<'_, '_> {
    fn name(&mut self, roles: u32, name: &Name) -> Name {
        if roles & BUFFERS != 0 {
            self.0.rename(name)
        } else {
            name.clone()
        }
    }
    fn body(&mut self, roles: u32, body: &Body) -> Body {
        if roles & TERM != 0 {
            body.map_each(&mut |stmt| Statement::rename_buffers_with(stmt, self.0))
        } else {
            body.clone()
        }
    }
    fn carried(&mut self, _roles: u32, stmt: &Arc<Statement>) -> Arc<Statement> {
        Statement::rename_buffers_with(stmt, self.0)
    }
}

struct RewriteFields<'a, 'b> {
    rewrite: &'b Rewrite<'a>,
    this: &'b Statement,
    axis_names: Option<HashMap<Name, Name>>,
}

impl Rewriter for RewriteFields<'_, '_> {
    fn name(&mut self, roles: u32, name: &Name) -> Name {
        if roles & NAMES != 0 {
            return (self.rewrite.rename)(name).unwrap_or_else(|| name.clone());
        }
        if roles & AXIS_NAME != 0 {
            let Some(axis_fn) = self.rewrite.axis_fn else {
                return name.clone();
            };
            let axes = self.axis_names.get_or_insert_with(|| {
                self.this
                    .bound_axes()
                    .into_iter()
                    .map(|axis| (axis.name.clone(), axis_fn(&axis).name.clone()))
                    .collect()
            });
            return axes.get(name).cloned().unwrap_or_else(|| name.clone());
        }
        name.clone()
    }
    fn expression(&mut self, _roles: u32, expression: &Expression) -> Expression {
        expression
            .substitute(self.rewrite.sigma)
            .rename(&mut |name| (self.rewrite.rename)(name))
    }
    fn axis(&mut self, _roles: u32, axis: &AxisRef) -> AxisRef {
        match self.rewrite.axis_fn {
            Some(axis_fn) => axis_fn(axis),
            None => axis.clone(),
        }
    }
    fn body(&mut self, roles: u32, body: &Body) -> Body {
        if roles & BODY != 0 {
            return body.map_each(&mut |stmt| Statement::rewrite(stmt, self.rewrite));
        }
        if roles & TERM != 0 {
            // Its statements' names are its own; a coordinate its expressions read is not.
            let defined: HashSet<Name> = body
                .stmts()
                .iter()
                .flat_map(|stmt| stmt.defines())
                .collect();
            let inner = Rewrite {
                rename: &keep,
                sigma: self.rewrite.sigma,
                axis_fn: self.rewrite.axis_fn,
            };
            let rename = self.rewrite.rename;
            return body.map_each(&mut |stmt| {
                let moved = Statement::rewrite(stmt, &inner);
                Statement::map_expressions(&moved, &mut |expression| {
                    expression.rename(&mut |name| {
                        if defined.contains(name) {
                            None
                        } else {
                            rename(name)
                        }
                    })
                })
            });
        }
        body.clone()
    }
    fn carried(&mut self, _roles: u32, stmt: &Arc<Statement>) -> Arc<Statement> {
        Statement::rewrite(stmt, self.rewrite)
    }
}

/// The names a substitution's replacement of `name` spells: `name` itself when it is not substituted, else the
/// sorted free variables of the replacement (a split axis reduces over its sub-axes).
fn rewrite_axis_name(name: &Name, sigma: &HashMap<Name, Expression>) -> Vec<Name> {
    match sigma.get(name) {
        None => vec![name.clone()],
        Some(replacement) => sorted_free_vars(replacement),
    }
}

impl Statement {
    pub fn is_pure(&self) -> bool {
        self.flags() & PURE != 0
    }

    pub fn deps_deep(&self) -> bool {
        self.flags() & DEPS_DEEP != 0
    }

    /// True iff executing this statement writes a buffer, directly or inside a child body.
    pub fn has_side_effects(&self) -> bool {
        self.flags() & SIDE_EFFECTS != 0 || self.nested().iter().any(|body| body.has_side_effects())
    }

    /// The SSA names this statement reads.
    pub fn deps(&self) -> Vec<Name> {
        let mut visitor = Deps {
            reads: Vec::new(),
            locals: Vec::new(),
            carried: Vec::new(),
        };
        self.visit(&mut visitor);
        let Deps {
            mut reads,
            locals,
            carried,
        } = visitor;
        if !carried.is_empty() {
            // The carried statements are a scope of their own: one's definition hides the next one's read of it.
            let mut hidden: HashSet<Name> = reads.iter().cloned().chain(locals).collect();
            for stmt in carried {
                let fresh: Vec<Name> = unique(stmt.deps())
                    .into_iter()
                    .filter(|dep| !hidden.contains(dep))
                    .collect();
                reads.extend(fresh);
                hidden.extend(reads.iter().cloned());
                hidden.extend(stmt.defines());
            }
        }
        reads
    }

    /// The SSA names this statement defines.
    pub fn defines(&self) -> Vec<Name> {
        let mut visitor = NamesByRole {
            roles: DEFINING,
            out: Vec::new(),
        };
        self.visit(&mut visitor);
        visitor.out
    }

    /// The external buffers this statement reads, each once.
    pub fn external_reads(&self) -> Vec<Name> {
        match self {
            Statement::Let(let_) => let_.value.flat_buffers(),
            Statement::Call(call) => call.target.buffers.to_vec(),
            _ => {
                let mut visitor = Carried {
                    roles: BUFFER_READ,
                    own: Vec::new(),
                    member: &Statement::external_reads,
                };
                self.visit(&mut visitor);
                unique(visitor.own)
            }
        }
    }

    /// The external buffers this statement writes, each once.
    pub fn external_writes(&self) -> Vec<Name> {
        let mut visitor = Carried {
            roles: BUFFER_WRITE,
            own: Vec::new(),
            member: &Statement::external_writes,
        };
        self.visit(&mut visitor);
        unique(visitor.own)
    }

    /// The buffers this statement declares kernel-local.
    pub fn local_decls(&self) -> Vec<Name> {
        let mut visitor = NamesByRole {
            roles: LOCAL_BUFFER,
            out: Vec::new(),
        };
        self.visit(&mut visitor);
        visitor.out
    }

    /// This statement's own expressions, in field order, the carried statements' included.
    pub fn expressions(&self) -> Vec<Expression> {
        let mut visitor = Expressions(Vec::new());
        self.visit(&mut visitor);
        visitor.0
    }

    /// The axes this statement binds for its child bodies, in field order.
    pub fn bound_axes(&self) -> Vec<AxisRef> {
        let mut visitor = BoundAxes(Vec::new());
        self.visit(&mut visitor);
        visitor.0
    }

    /// The names this statement brings into scope for its child bodies.
    pub fn binds_axes(&self) -> Vec<Name> {
        let mut names: Vec<Name> = self
            .bound_axes()
            .iter()
            .map(|axis| axis.name.clone())
            .collect();
        if let Statement::StridedLoop(loop_) = self
            && loop_.end.is_some()
        {
            names.push(format!("{}_end", loop_.axis.name).into());
        }
        names
    }

    /// The child bodies, in field order.
    pub fn nested(&self) -> Vec<Body> {
        let mut visitor = Nested(Vec::new());
        self.visit(&mut visitor);
        visitor.0
    }

    /// This statement over `bodies`, positionally matching `nested`; refused when the counts disagree.
    pub fn with_bodies(this: &Arc<Statement>, bodies: Vec<Body>) -> Result<Arc<Statement>> {
        let count = bodies.len();
        let mut rewriter = WithBodies {
            bodies: bodies.into_iter(),
            short: false,
        };
        let new = this.rewrite_fields(&mut rewriter);
        if rewriter.short || rewriter.bodies.len() > 0 {
            return Err(Error::Value(format!(
                "{}.with_bodies: {} bodies for {} child bodies",
                this.kind(),
                count,
                this.nested().len()
            )));
        }
        Ok(if new.same(this) {
            this.clone()
        } else {
            Arc::new(new)
        })
    }

    /// `with_bodies` for a caller that built one body per child body: a count that disagrees keeps the statement's
    /// own bodies for the positions it does not cover.
    pub fn rebuild_bodies(this: &Arc<Statement>, bodies: Vec<Body>) -> Arc<Statement> {
        let mut rewriter = WithBodies {
            bodies: bodies.into_iter(),
            short: false,
        };
        let new = this.rewrite_fields(&mut rewriter);
        if new.same(this) {
            this.clone()
        } else {
            Arc::new(new)
        }
    }

    /// This statement with `f` applied to each of its own expressions; itself when none changes.
    pub fn map_expressions(
        this: &Arc<Statement>,
        f: &mut dyn FnMut(&Expression) -> Expression,
    ) -> Arc<Statement> {
        let new = this.rewrite_fields(&mut MapExpressions(f));
        if new.same(this) {
            this.clone()
        } else {
            Arc::new(new)
        }
    }

    /// This statement with every buffer renamed through `mapping`; itself when none changes.
    pub fn rename_buffers(this: &Arc<Statement>, mapping: &HashMap<Name, Name>) -> Arc<Statement> {
        Statement::rename_buffers_with(this, &Renaming::new(mapping))
    }

    /// `rename_buffers` inside one renaming: a body renames every statement through the same one, so a subroutine
    /// two calls share stays one definition.
    pub fn rename_buffers_with(this: &Arc<Statement>, renaming: &Renaming<'_>) -> Arc<Statement> {
        let mapping = renaming.mapping;
        if mapping.is_empty() {
            return this.clone();
        }
        match &**this {
            Statement::Let(let_) => {
                let value = let_.value.rebuild(&mut |e| match &*e {
                    Expr::FlatIndex(buffer, index) => match mapping.get(buffer) {
                        Some(renamed) => {
                            Expression::new(Expr::FlatIndex(renamed.clone(), index.clone()))
                        }
                        None => e,
                    },
                    _ => e,
                });
                if value == let_.value {
                    this.clone()
                } else {
                    Arc::new(Statement::Let(Let {
                        value,
                        ..let_.clone()
                    }))
                }
            }
            Statement::Call(call) => {
                let target = renaming.subroutine(&call.target.0);
                if Arc::ptr_eq(&target, &call.target.0) {
                    return this.clone();
                }
                Arc::new(Statement::Call(Call {
                    target: SubroutineRef(target),
                    ..call.clone()
                }))
            }
            _ => {
                let new = this.rewrite_fields(&mut RenameBuffers(renaming));
                if new.same(this) {
                    this.clone()
                } else {
                    Arc::new(new)
                }
            }
        }
    }

    /// Rename, substitute and map axes over this subtree. The substitution is hygienic: a name this statement binds
    /// is dropped from it for the subtree. A rename passes through, renaming the binder itself.
    pub fn rewrite(this: &Arc<Statement>, rewrite: &Rewrite) -> Arc<Statement> {
        let shadowed: Vec<Name> = if rewrite.sigma.is_empty() {
            Vec::new()
        } else {
            this.binds_axes()
                .into_iter()
                .filter(|name| rewrite.sigma.contains_key(name))
                .collect()
        };
        let restricted: HashMap<Name, Expression>;
        let rewrite = if shadowed.is_empty() {
            rewrite
        } else {
            restricted = rewrite
                .sigma
                .iter()
                .filter(|(name, _)| !shadowed.contains(name))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            &Rewrite {
                rename: rewrite.rename,
                sigma: &restricted,
                axis_fn: rewrite.axis_fn,
            }
        };
        let mut rewriter = RewriteFields {
            rewrite,
            this,
            axis_names: None,
        };
        let mut new = this.rewrite_fields(&mut rewriter);
        if let Statement::Accum(accum) = &mut new
            && let Statement::Accum(old) = &**this
        {
            // The reduced axes are names a substitution can split, which no role says.
            let mut axes: Vec<Name> = old
                .axes
                .iter()
                .flat_map(|name| rewrite_axis_name(name, rewrite.sigma))
                .map(|name| (rewrite.rename)(&name).unwrap_or(name))
                .collect();
            axes.sort();
            axes.dedup();
            if axes[..] != old.axes[..] {
                accum.axes = axes.into();
            }
        }
        if new.same(this) {
            this.clone()
        } else {
            Arc::new(new)
        }
    }

    /// α-rename this subtree, binders and uses, SSA names and axis names, through one map.
    pub fn rename(this: &Arc<Statement>, mapping: &HashMap<Name, Name>) -> Arc<Statement> {
        if mapping.is_empty() {
            return this.clone();
        }
        Statement::rename_with(this, &|name| mapping.get(name).cloned())
    }

    /// `rename` through a function; `None` keeps a name.
    pub fn rename_with(
        this: &Arc<Statement>,
        rename: &dyn Fn(&str) -> Option<Name>,
    ) -> Arc<Statement> {
        let axis_fn = |axis: &AxisRef| axis.renamed(&mut |name| rename(name));
        let empty = HashMap::new();
        Statement::rewrite(
            this,
            &Rewrite {
                rename,
                sigma: &empty,
                axis_fn: Some(&axis_fn),
            },
        )
    }

    /// β-substitute free coordinates by expressions, stopping at a scope that re-binds a substituted name.
    pub fn substitute(this: &Arc<Statement>, sigma: &HashMap<Name, Expression>) -> Arc<Statement> {
        if sigma.is_empty() {
            return this.clone();
        }
        Statement::rewrite(
            this,
            &Rewrite {
                rename: &keep,
                sigma,
                axis_fn: None,
            },
        )
    }

    /// The elementwise operation a kind applies, held as `op`.
    pub fn op(&self) -> Option<&Op> {
        match self {
            Statement::Assign(s) => Some(&s.op),
            Statement::Accum(s) => Some(&s.op),
            Statement::FragmentApply(s) => Some(&s.op),
            Statement::FragmentRowReduce(s) => Some(&s.op),
            _ => None,
        }
    }

    /// This statement with another `op`, for a kind that holds one.
    pub fn with_op(&self, op: Op) -> Statement {
        match self {
            Statement::Assign(s) => Statement::Assign(Assign { op, ..s.clone() }),
            Statement::Accum(s) => Statement::Accum(Accum { op, ..s.clone() }),
            Statement::FragmentApply(s) => {
                Statement::FragmentApply(FragmentApply { op, ..s.clone() })
            }
            Statement::FragmentRowReduce(s) => {
                Statement::FragmentRowReduce(FragmentRowReduce { op, ..s.clone() })
            }
            _ => self.clone(),
        }
    }

    /// The dtype a value-defining kind stamps, when it has one.
    pub fn dtype(&self) -> Option<&DType> {
        match self {
            Statement::Load(s) => s.dtype.as_ref(),
            Statement::Assign(s) => s.dtype.as_ref(),
            Statement::Affine(s) => s.dtype.as_ref(),
            Statement::Accum(s) => s.dtype.as_ref(),
            Statement::Carry(s) => s.dtype.as_ref(),
            Statement::Let(s) => s.dtype.as_ref(),
            Statement::Init(s) => Some(&s.dtype),
            _ => None,
        }
    }

    /// The one name a single-value kind defines.
    pub fn name(&self) -> Option<&Name> {
        match self {
            Statement::Assign(s) => Some(&s.name),
            Statement::Affine(s) => Some(&s.name),
            Statement::Accum(s) => Some(&s.name),
            Statement::Carry(s) => Some(&s.name),
            Statement::Pre(s) => Some(&s.name),
            Statement::Init(s) => Some(&s.name),
            Statement::Let(s) => Some(&s.name),
            Statement::Select(s) => Some(&s.name),
            Statement::Call(s) => Some(&s.name),
            _ => None,
        }
    }

    pub fn is_block(&self) -> bool {
        matches!(
            self,
            Statement::Loop(_)
                | Statement::StridedLoop(_)
                | Statement::Cond(_)
                | Statement::Tile(_)
        )
    }
}

// --- the kinds' own readings ----------------------------------------------------------------------

/// The seeded accumulators a reduce loop declares in its enclosing scope: the `Accum` targets of its immediate body.
pub fn seeded_accumulators(stmt: &Statement) -> Vec<Name> {
    let (body, seed) = match stmt {
        Statement::Loop(l) => (&l.body, l.seed),
        Statement::StridedLoop(l) => (&l.body, l.seed),
        _ => return vec![],
    };
    if !seed || !body.carries().is_empty() {
        return vec![];
    }
    body.stmts()
        .iter()
        .filter_map(|s| match &**s {
            Statement::Accum(a) => Some(a.name.clone()),
            _ => None,
        })
        .collect()
}

/// The carried states a loop over `body` carries, each with its cell shape: an axis per index position, `None`
/// where normalization dropped a size-one axis.
pub fn carried_cells(body: &Body) -> Vec<(Name, Vec<Option<AxisRef>>)> {
    if body.carries().is_empty() {
        return vec![];
    }
    let mut out: Vec<(Name, Vec<Option<AxisRef>>)> = Vec::new();
    fn walk(
        body: &Body,
        inner: &HashMap<Name, AxisRef>,
        out: &mut Vec<(Name, Vec<Option<AxisRef>>)>,
    ) {
        for stmt in body.stmts() {
            if let Statement::Carry(carry) = &**stmt {
                let cells: HashSet<&Name> = carry.cells().into_iter().collect();
                let keys: HashSet<&Name> = inner.keys().collect();
                if cells == keys {
                    let shape = carry
                        .index
                        .iter()
                        .map(|e| e.as_var().and_then(|name| inner.get(name).cloned()))
                        .collect();
                    match out.iter_mut().find(|(name, _)| *name == carry.name) {
                        Some(entry) => entry.1 = shape,
                        None => out.push((carry.name.clone(), shape)),
                    }
                }
            }
            for child in stmt.nested() {
                if let Statement::Loop(l) = &**stmt {
                    let mut deeper = inner.clone();
                    deeper.insert(l.axis.name.clone(), l.axis.clone());
                    walk(&child, &deeper, out);
                } else {
                    walk(&child, inner, out);
                }
            }
        }
    }
    walk(body, &HashMap::new(), &mut out);
    out
}

impl Loop {
    /// The carried states this loop carries.
    pub fn carries(&self) -> Vec<(Name, Vec<Option<AxisRef>>)> {
        carried_cells(&self.body)
    }

    /// A loop is a reduce loop iff it carries something: an `Accum` in its immediate body, or a state.
    pub fn is_reduce(&self) -> bool {
        self.body
            .stmts()
            .iter()
            .any(|s| matches!(**s, Statement::Accum(_)))
            || !self.carries().is_empty()
    }
}

impl StridedLoop {
    pub fn is_reduce(&self) -> bool {
        self.body
            .stmts()
            .iter()
            .any(|s| matches!(**s, Statement::Accum(_)))
    }
}

impl Carry {
    /// The axes of the cell this statement defines: the index entries that are variables.
    pub fn cells(&self) -> Vec<&Name> {
        self.index.iter().filter_map(|e| e.as_var()).collect()
    }
}

impl Accum {
    /// The identity value of the accumulator: the op's neutral element, `0.0` when it has none.
    pub fn init(&self) -> Expression {
        Expression::float(self.op.identity.unwrap_or(0.0))
    }
}

impl Affine {
    pub fn terms(&self) -> impl Iterator<Item = (f64, &Name)> {
        self.coefficients
            .iter()
            .zip(self.args.iter())
            .map(|(c, arg)| (c.0, arg))
    }

    /// Construct with Python's checks: equal lengths, at least one term, no zero coefficient.
    pub fn checked(
        name: Name,
        args: Names,
        coefficients: Floats,
        offset: Float,
        dtype: Option<DType>,
    ) -> Result<Affine> {
        if args.len() != coefficients.len() {
            return Err(Error::Value(format!(
                "Affine {name:?}: {} args and {} coefficients",
                args.len(),
                coefficients.len()
            )));
        }
        if args.is_empty() {
            return Err(Error::Value(format!(
                "Affine {name:?} has no term: a constant is a Let"
            )));
        }
        if coefficients.iter().any(|c| c.0 == 0.0) {
            let shown: Vec<String> = coefficients
                .iter()
                .map(|c| crate::kinds::field::spell_number(c.0))
                .collect();
            return Err(Error::Value(format!(
                "Affine {name:?} carries a zero coefficient: ({})",
                shown.join(", ")
            )));
        }
        Ok(Affine {
            name,
            args,
            coefficients,
            offset,
            dtype,
        })
    }

    /// The same value as binary `Assign`s over `Let`-bound literals, the last one named `name`.
    pub fn as_assigns(&self) -> Vec<Arc<Statement>> {
        struct Chain<'a> {
            affine: &'a Affine,
            out: Vec<Arc<Statement>>,
            counter: usize,
        }
        impl Chain<'_> {
            fn fresh(&mut self) -> Name {
                let name: Name = format!("{}__t{}", self.affine.name, self.counter).into();
                self.counter += 1;
                name
            }
            fn bound(&mut self, value: f64) -> Name {
                let name = self.fresh();
                let let_ = Let {
                    name: name.clone(),
                    value: Expression::float(value),
                    dtype: self.affine.dtype.clone(),
                };
                self.out.push(Arc::new(Statement::Let(let_)));
                name
            }
            fn step(&mut self, op: &str, args: Vec<Name>) -> Name {
                let name = self.fresh();
                let assign = Assign {
                    name: name.clone(),
                    op: Op::new(op),
                    args: args.into(),
                    dtype: self.affine.dtype.clone(),
                };
                self.out.push(Arc::new(Statement::Assign(assign)));
                name
            }
        }
        let mut chain = Chain {
            affine: self,
            out: Vec::new(),
            counter: 0,
        };
        let mut total: Option<Name> = None;
        // A positive term leads when there is one, so the chain opens with a value rather than a negation.
        let mut terms: Vec<(f64, &Name)> = self.terms().collect();
        terms.sort_by_key(|(c, _)| *c < 0.0);
        for (coefficient, arg) in terms {
            let term = if coefficient.abs() == 1.0 {
                arg.clone()
            } else {
                let literal = chain.bound(coefficient.abs());
                chain.step("multiply", vec![arg.clone(), literal])
            };
            total = Some(match total {
                None if coefficient > 0.0 => term,
                None => chain.step("negative", vec![term]),
                Some(current) => chain.step(
                    if coefficient > 0.0 { "add" } else { "subtract" },
                    vec![current, term],
                ),
            });
        }
        // An affine has a term (its constructor refuses none), so the chain holds a total here.
        let Some(mut total) = total else {
            return chain.out;
        };
        if self.offset.0 != 0.0 {
            let literal = chain.bound(self.offset.0.abs());
            total = chain.step(
                if self.offset.0 > 0.0 {
                    "add"
                } else {
                    "subtract"
                },
                vec![total, literal],
            );
        }
        let mut out = chain.out;
        match out.last_mut() {
            Some(last) if last.name() == Some(&total) => {
                if let Statement::Assign(a) = &**last {
                    *last = Arc::new(Statement::Assign(Assign {
                        name: self.name.clone(),
                        ..a.clone()
                    }));
                }
            }
            _ => out.push(Arc::new(Statement::Assign(Assign {
                name: self.name.clone(),
                op: Op::new("copy"),
                args: Names::from([total]),
                dtype: self.dtype.clone(),
            }))),
        }
        out
    }
}

/// `stmts` with every `Affine` spelled as its binary `Assign`s; the other statements pass through.
pub fn binary(stmts: &[Arc<Statement>]) -> Vec<Arc<Statement>> {
    stmts
        .iter()
        .flat_map(|stmt| match &**stmt {
            Statement::Affine(affine) => affine.as_assigns(),
            _ => vec![stmt.clone()],
        })
        .collect()
}

/// The mask branch of an ordered two-way comparison, or `None` for another `Select` form.
pub fn mask_select_predicate(select: &Select) -> Option<Expression> {
    if select.branches.len() != 2 {
        return None;
    }
    let (keep, mask) = (&select.branches[0], &select.branches[1]);
    let (
        Expr::Binary(keep_op, keep_left, keep_right),
        Expr::Binary(mask_op, mask_left, mask_right),
    ) = (&*keep.select, &*mask.select)
    else {
        return None;
    };
    let (mut op, mut left, mut right) = (*mask_op, mask_left.clone(), mask_right.clone());
    if *keep_left == right && *keep_right == left {
        use crate::expr::BinOp::*;
        op = match op {
            Lt => Gt,
            Le => Ge,
            Gt => Lt,
            Ge => Le,
            _ => return None,
        };
        std::mem::swap(&mut left, &mut right);
    }
    if *keep_left != left || *keep_right != right {
        return None;
    }
    use crate::expr::BinOp::*;
    if !matches!((keep_op, op), (Lt, Ge) | (Le, Gt) | (Gt, Le) | (Ge, Lt)) {
        return None;
    }
    Some(mask.select.clone())
}

/// A `Let` of a float literal at the default precision: the number it binds its name to, else `None`.
pub fn literal_of(stmt: &Statement) -> Option<f64> {
    literal_binding(stmt).map(|(_, value)| value)
}

/// `literal_of` with the name it binds.
pub fn literal_binding(stmt: &Statement) -> Option<(&Name, f64)> {
    if let Statement::Let(let_) = stmt
        && let Expr::Literal(lit) = &*let_.value
        && lit.dtype == LitType::Float
        && let_.dtype.as_ref().is_none_or(|d| d.name() == "f32")
    {
        return Some((&let_.name, lit.value.as_f64()));
    }
    None
}

/// Constructors with the checks Python's `__post_init__` made.
impl Load {
    pub fn checked(
        names: Names,
        input: Name,
        index: Exprs,
        dtype: Option<DType>,
        carried: Name,
    ) -> Result<Load> {
        if names.is_empty() {
            return Err(Error::Value("Load.names must be non-empty".into()));
        }
        Ok(Load {
            names,
            input,
            index,
            dtype,
            carried,
        })
    }
}

impl Write {
    pub fn checked(
        output: Name,
        index: Exprs,
        values: Names,
        value_dtype: Option<DType>,
        atomic: bool,
        swizzle: Name,
    ) -> Result<Write> {
        if values.is_empty() {
            return Err(Error::Value("Write.values must be non-empty".into()));
        }
        if atomic && values.len() != 1 {
            return Err(Error::Value(
                "an atomic Write is scalar (one value per atomicAdd)".into(),
            ));
        }
        Ok(Write {
            output,
            index,
            values,
            value_dtype,
            atomic,
            swizzle,
        })
    }
}

impl Accum {
    pub fn checked(
        name: Name,
        value: Name,
        op: Op,
        dtype: Option<DType>,
        axes: Names,
        base: Option<Name>,
    ) -> Accum {
        // The ordinary self-fold has one spelling.
        let base = base.filter(|b| *b != name);
        Accum {
            name,
            value,
            op,
            dtype,
            axes,
            base,
        }
    }
}

impl Select {
    pub fn checked(name: Name, branches: crate::kinds::field::Branches) -> Result<Select> {
        if branches.is_empty() {
            return Err(Error::Value("Select.branches must be non-empty".into()));
        }
        Ok(Select { name, branches })
    }
}

impl Call {
    pub fn checked(name: Name, target: SubroutineRef, args: Exprs) -> Result<Call> {
        if args.len() != target.axes.len() {
            return Err(Error::Value(
                "subroutine arguments do not match its parameters".into(),
            ));
        }
        Ok(Call { name, target, args })
    }
}

impl Tile {
    pub fn check(&self) -> Result<()> {
        if self.aux_threads != 0 && self.block_threads.is_none() {
            return Err(Error::Value(
                "Tile.aux_threads requires a cooperative block_threads".into(),
            ));
        }
        Ok(())
    }
}

impl WgmmaMma {
    pub fn check(&self) -> Result<()> {
        let crate::kinds::field::Shape3(m, n, k) = self.shape;
        if (m, k) != (64, 16) || n % 8 != 0 || !(8..=256).contains(&n) {
            return Err(Error::Value(format!(
                "wgmma: the shape is m64n<8..256, a multiple of 8>k16, got ({m}, {n}, {k})"
            )));
        }
        if self.c_frags.len() as i64 != n / 8 {
            return Err(Error::Value(format!(
                "wgmma m64n{n}: the accumulator is {} m16n8 C fragments, got {}",
                n / 8,
                self.c_frags.len()
            )));
        }
        if self.a_desc.is_none() == self.a_frag.is_none() {
            return Err(Error::Value(
                "wgmma: exactly one of a_desc / a_frag names the A operand".into(),
            ));
        }
        if self.a_frag.is_some() && self.trans_a != 0 {
            return Err(Error::Value(
                "wgmma: the register-form A operand is K-major (no transpose bit)".into(),
            ));
        }
        Ok(())
    }
}

/// The checks a kind makes on construction, by kind: what Python's `__post_init__` refused.
pub fn check(stmt: &Statement) -> Result<()> {
    match stmt {
        Statement::Load(s) => Load::checked(
            s.names.clone(),
            s.input.clone(),
            s.index.clone(),
            s.dtype.clone(),
            s.carried.clone(),
        )
        .map(|_| ()),
        Statement::Write(s) => Write::checked(
            s.output.clone(),
            s.index.clone(),
            s.values.clone(),
            s.value_dtype.clone(),
            s.atomic,
            s.swizzle.clone(),
        )
        .map(|_| ()),
        Statement::Affine(s) => Affine::checked(
            s.name.clone(),
            s.args.clone(),
            s.coefficients.clone(),
            s.offset,
            s.dtype.clone(),
        )
        .map(|_| ()),
        Statement::Select(s) => Select::checked(s.name.clone(), s.branches.clone()).map(|_| ()),
        Statement::Call(s) => {
            Call::checked(s.name.clone(), s.target.clone(), s.args.clone()).map(|_| ())
        }
        Statement::Tile(s) => s.check(),
        Statement::WgmmaMma(s) => s.check(),
        Statement::TreeHalve(s) => carried_kind(&s.combine_states, Kind::Assign),
        Statement::WarpShuffle(s) => carried_kind(&s.combine_states, Kind::Assign),
        Statement::RegStore(s) => carried_kind(&s.run, Kind::RegStore),
        _ => Ok(()),
    }
}

fn carried_kind(stmts: &[Arc<Statement>], kind: Kind) -> Result<()> {
    match stmts.iter().find(|s| s.kind() != kind) {
        Some(other) => Err(Error::Type(format!(
            "expected {kind} statements, got a {}",
            other.kind()
        ))),
        None => Ok(()),
    }
}

/// A statement with its `Accum` base normalized and its carried numbers settled: what every constructor returns.
pub fn finished(stmt: Statement) -> Result<Statement> {
    check(&stmt)?;
    Ok(match stmt {
        Statement::Accum(a) => Statement::Accum(Accum::checked(
            a.name, a.value, a.op, a.dtype, a.axes, a.base,
        )),
        other => other,
    })
}

/// The axis a plain Loop IR coordinate expression names, when it is one variable.
pub fn axis_of(expression: &Expression) -> Option<&Name> {
    expression.as_var()
}

/// A `NumOr` value's number, when it is one.
pub fn number_of<T>(value: &NumOr<T>) -> Option<f64> {
    match value {
        NumOr::Num(n) => Some(n.as_f64()),
        NumOr::Value(_) => None,
    }
}

impl Statement {
    /// Every SSA and axis name this subtree spells: binders, reads, locals, axis names and the variables its
    /// expressions mention. What a rename by a function is asked about.
    pub fn names(&self) -> Vec<Name> {
        struct AllNames(Vec<Name>);
        impl Visitor for AllNames {
            fn name(&mut self, roles: u32, name: &Name) {
                if roles & (NAMES | AXIS_NAME) != 0 {
                    self.0.push(name.clone());
                }
            }
            fn expression(&mut self, _roles: u32, expression: &Expression) {
                expression.collect_free_vars(&mut self.0);
            }
            fn axis(&mut self, _roles: u32, axis: &AxisRef) {
                self.0.push(axis.name.clone());
                for source in axis.sources() {
                    self.0.push(source.name.clone());
                }
                if let Some(window) = &axis.window {
                    for e in [&window.base, &window.bound].into_iter().flatten() {
                        e.collect_free_vars(&mut self.0);
                    }
                }
            }
            fn body(&mut self, _roles: u32, body: &Body) {
                for stmt in body.stmts() {
                    self.0.extend(stmt.names());
                }
            }
            fn carried(&mut self, _roles: u32, stmt: &Arc<Statement>) {
                self.0.extend(stmt.names());
            }
        }
        let mut visitor = AllNames(Vec::new());
        self.visit(&mut visitor);
        unique(visitor.0)
    }
}

pub fn axis_named(axis: &Axis) -> &Name {
    &axis.name
}

impl Body {
    /// This body with every external buffer renamed through `mapping`.
    pub fn rename_buffers(&self, mapping: &HashMap<Name, Name>) -> Body {
        let renaming = Renaming::new(mapping);
        let result =
            self.map_stmts(&mut |stmt| Some(Statement::rename_buffers_with(stmt, &renaming)));
        if result.ptr_eq(self) {
            return self.clone();
        }
        // An injective spelling-only rename keeps the normal form a body is its own.
        if matches!(self.0.normalized.get(), Some(Normalized::Itself)) {
            let resources: Vec<Name> = unique(
                self.iter_stmts()
                    .iter()
                    .flat_map(|s| s.external_reads().into_iter().chain(s.external_writes())),
            );
            let renamed: HashSet<Name> = resources
                .iter()
                .map(|name| mapping.get(name).cloned().unwrap_or_else(|| name.clone()))
                .collect();
            if renamed.len() == resources.len() {
                let _ = result.0.normalized.set(Normalized::Itself);
            }
        }
        result
    }

    /// Shared SSA definitions assembled once and adjacent independent reduction loops merged.
    pub fn coalesce(&self) -> Body {
        let mut out: Vec<Arc<Statement>> = Vec::new();
        let mut seen: Vec<Arc<Statement>> = Vec::new();
        let mut folded: HashSet<Name> = HashSet::new();
        for stmt in self.stmts() {
            let mut stmt = stmt.clone();
            if stmt.has_side_effects() {
                seen.clear();
                folded.clear();
            } else if !exposed_defines(&stmt).is_empty() {
                if seen.iter().any(|s| **s == *stmt) {
                    continue;
                }
                seen.push(stmt.clone());
                let accumulators: HashSet<Name> = seeded_accumulators(&stmt).into_iter().collect();
                if !accumulators.is_disjoint(&folded) {
                    if accumulators.is_subset(&folded) {
                        continue;
                    }
                    let wanted: Vec<Name> = accumulators.difference(&folded).cloned().collect();
                    if let Some((_, body)) = loop_parts(&stmt) {
                        let (members, _) = body.backward_cone(&wanted);
                        let kept = Body::new(members.iter().map(|m| m.stmt().clone()).collect());
                        stmt = Statement::rebuild_bodies(&stmt, vec![kept]);
                    }
                }
            }
            if let Some(prior) = out.last()
                && let (Some((prior_reduce, prior_body)), Some((stmt_reduce, stmt_body))) =
                    (loop_parts(prior), loop_parts(&stmt))
                && prior_reduce
                && stmt_reduce
                && !(prior.has_side_effects() || stmt.has_side_effects())
                && *Statement::rebuild_bodies(prior, vec![stmt_body.clone()]) == *stmt
                && free_names(&stmt)
                    .iter()
                    .all(|name| !prior_body.definitions().contains_key(name))
            {
                let merged = prior_body.concat(stmt_body.stmts()).coalesce();
                stmt = Statement::rebuild_bodies(prior, vec![merged]);
                out.pop();
            }
            folded.extend(seeded_accumulators(&stmt));
            out.push(stmt);
        }
        Body::new(out)
    }
}
