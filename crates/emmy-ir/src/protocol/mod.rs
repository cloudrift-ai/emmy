//! The statement protocol: what every statement type answers, written once over the roles of its fields. Reads
//! (`deps`, `defines`, the buffers, the expressions, the axes, the child bodies) and their rebuilding halves
//! (`with_bodies`, `map_expressions`, `rename_buffers`, and `rewrite` behind `rename` and `substitute`). The few
//! members no role gives are stated here by statement type: an `Accum`'s reduced axes follow a substitution that
//! splits an axis, a `Let` and a `Call` read buffers their fields do not name, a `StridedLoop` with an `end` binds one
//! more name than its axis.

pub mod checks;
pub mod shared;

pub use checks::*;
pub use shared::*;

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::axis::Axis;
use crate::body::{Body, Subroutine, exposed_defines, free_names, loop_parts, unique};
use crate::expr::{Expr, ExprClass, Expression};
use crate::field::{AxisRef, Rewriter, SubroutineRef, Visitor};
use crate::statement::*;
use crate::{Error, Name, Result};

/// Inside a term, the coordinates its index expressions read follow the outer rename, less the names the term
/// defines; its value expressions read the term's own names and keep them, as its name fields do.
struct Coordinates<'a> {
    rename: &'a dyn Fn(Roles, Name) -> Option<Name>,
    defined: &'a HashSet<Name>,
}

impl Rewriter for Coordinates<'_> {
    fn expression(&mut self, roles: Roles, expression: &Expression) -> Expression {
        if roles.contains_any(VALUE) {
            return expression.clone();
        }
        expression.rename(&mut |name| {
            if self.defined.contains(&name) {
                None
            } else {
                (self.rename)(roles, name)
            }
        })
    }
}

/// What a rewrite does to a statement: every SSA name through `rename`, which sees the role the name has in its field,
/// every expression through `sigma`, every bound axis through `axis_fn`. A `None` from `rename` keeps the name.
pub struct Rewrite<'a> {
    pub rename: &'a dyn Fn(Roles, Name) -> Option<Name>,
    pub sigma: &'a HashMap<Name, Expression>,
    pub axis_fn: Option<&'a dyn Fn(&AxisRef) -> AxisRef>,
}

fn keep(_roles: Roles, _name: Name) -> Option<Name> {
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
    fn name(&mut self, roles: Roles, name: &Name) {
        if roles.contains_any(READING) {
            self.reads.push(*name);
        }
        if roles.contains_any(LOCAL) {
            self.locals.push(*name);
        }
    }
    fn expression(&mut self, roles: Roles, expression: &Expression) {
        if roles.contains_any(READING) {
            // A field read through: the names its expressions mention, each listed once.
            for name in sorted_free_vars(expression) {
                if !self.reads.contains(&name) {
                    self.reads.push(name);
                }
            }
        }
    }
    fn carried(&mut self, _roles: Roles, stmt: &Arc<Statement>) {
        self.carried.push(stmt.clone());
    }
}

struct NamesByRole {
    roles: Roles,
    out: Vec<Name>,
}

impl Visitor for NamesByRole {
    fn name(&mut self, roles: Roles, name: &Name) {
        if roles.contains_any(self.roles) {
            self.out.push(*name);
        }
    }
}

/// Collects a member over the statement's own fields and the statements it carries.
struct Carried<'a> {
    roles: Roles,
    own: Vec<Name>,
    member: &'a dyn Fn(&Statement) -> Vec<Name>,
}

impl Visitor for Carried<'_> {
    fn name(&mut self, roles: Roles, name: &Name) {
        if roles.contains_any(self.roles) {
            self.own.push(*name);
        }
    }
    fn body(&mut self, roles: Roles, body: &Body) {
        if roles.contains_any(TERM) {
            for stmt in body.stmts() {
                self.own.extend((self.member)(stmt));
            }
        }
    }
    fn carried(&mut self, _roles: Roles, stmt: &Arc<Statement>) {
        self.own.extend((self.member)(stmt));
    }
}

struct Expressions(Vec<Expression>);

impl Visitor for Expressions {
    fn expression(&mut self, roles: Roles, expression: &Expression) {
        if roles.contains_any(EXPRESSION) {
            self.0.push(expression.clone());
        }
    }
    fn body(&mut self, roles: Roles, body: &Body) {
        if roles.contains_any(TERM) {
            for stmt in body.stmts() {
                self.0.extend(stmt.expressions());
            }
        }
    }
    fn carried(&mut self, _roles: Roles, stmt: &Arc<Statement>) {
        self.0.extend(stmt.expressions());
    }
}

/// The expressions of a statement's own fields alone: what the statement reads itself, not what a program it carries
/// (a combine, a term) computes under its own binders.
struct OwnExpressions(Vec<Expression>);

impl Visitor for OwnExpressions {
    fn expression(&mut self, roles: Roles, expression: &Expression) {
        if roles.contains_any(EXPRESSION) {
            self.0.push(expression.clone());
        }
    }
}

struct BoundAxes(Vec<AxisRef>);

impl Visitor for BoundAxes {
    fn axis(&mut self, _roles: Roles, axis: &AxisRef) {
        self.0.push(axis.clone());
    }
}

struct Nested(Vec<Body>);

impl Visitor for Nested {
    fn body(&mut self, roles: Roles, body: &Body) {
        if roles.contains_any(BODY) {
            self.0.push(body.clone());
        }
    }
}

struct WithBodies {
    bodies: std::vec::IntoIter<Body>,
    short: bool,
}

impl Rewriter for WithBodies {
    fn body(&mut self, roles: Roles, body: &Body) -> Body {
        if !roles.contains_any(BODY) {
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
    fn expression(&mut self, _roles: Roles, expression: &Expression) -> Expression {
        (self.0)(expression)
    }
    fn body(&mut self, roles: Roles, body: &Body) -> Body {
        if roles.contains_any(TERM) {
            body.map_each(&mut |stmt| Statement::map_expressions(stmt, self.0))
        } else {
            body.clone()
        }
    }
    fn carried(&mut self, _roles: Roles, stmt: &Arc<Statement>) -> Arc<Statement> {
        Statement::map_expressions(stmt, self.0)
    }
}

/// `f` on every expression a statement holds, a term's and a carried statement's included, with the class its field
/// declares (`Value` where it declares none, as the canonical form reads it); the first error ends the walk.
struct MapClassed<'a, E> {
    f: &'a mut dyn FnMut(ExprClass, &Expression) -> std::result::Result<Expression, E>,
    error: Option<E>,
}

impl<E> MapClassed<'_, E> {
    fn statement(&mut self, stmt: &Arc<Statement>) -> Arc<Statement> {
        let new = stmt.rewrite_fields(self);
        if new.same(stmt) {
            stmt.clone()
        } else {
            Arc::new(new)
        }
    }
}

impl<E> Rewriter for MapClassed<'_, E> {
    fn expression(&mut self, roles: Roles, expression: &Expression) -> Expression {
        if self.error.is_some() || !roles.contains_any(EXPRESSION) {
            return expression.clone();
        }
        match (self.f)(class_of(roles).unwrap_or(ExprClass::Value), expression) {
            Ok(mapped) => mapped,
            Err(error) => {
                self.error = Some(error);
                expression.clone()
            }
        }
    }
    fn body(&mut self, roles: Roles, body: &Body) -> Body {
        if roles.contains_any(TERM) {
            body.map_each(&mut |stmt| self.statement(stmt))
        } else {
            body.clone()
        }
    }
    fn carried(&mut self, _roles: Roles, stmt: &Arc<Statement>) -> Arc<Statement> {
        self.statement(stmt)
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
        self.mapping.get(name).cloned().unwrap_or(*name)
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
    fn name(&mut self, roles: Roles, name: &Name) -> Name {
        if roles.contains_any(BUFFERS) {
            self.0.rename(name)
        } else {
            *name
        }
    }
    fn body(&mut self, roles: Roles, body: &Body) -> Body {
        if roles.contains_any(TERM) {
            body.map_each(&mut |stmt| Statement::rename_buffers_with(stmt, self.0))
        } else {
            body.clone()
        }
    }
    fn carried(&mut self, _roles: Roles, stmt: &Arc<Statement>) -> Arc<Statement> {
        Statement::rename_buffers_with(stmt, self.0)
    }
}

struct RewriteFields<'a, 'b> {
    rewrite: &'b Rewrite<'a>,
    this: &'b Statement,
    axis_names: Option<HashMap<Name, Name>>,
}

impl Rewriter for RewriteFields<'_, '_> {
    fn name(&mut self, roles: Roles, name: &Name) -> Name {
        if roles.contains_any(NAMES) {
            return (self.rewrite.rename)(roles, *name).unwrap_or(*name);
        }
        if roles.contains_any(AXIS_NAME) {
            let Some(axis_fn) = self.rewrite.axis_fn else {
                return *name;
            };
            let axes = self.axis_names.get_or_insert_with(|| {
                self.this
                    .bound_axes()
                    .into_iter()
                    .map(|axis| (axis.name, axis_fn(&axis).name))
                    .collect()
            });
            return axes.get(name).cloned().unwrap_or(*name);
        }
        *name
    }
    fn expression(&mut self, roles: Roles, expression: &Expression) -> Expression {
        expression
            .substitute(self.rewrite.sigma)
            .rename(&mut |name| (self.rewrite.rename)(roles, name))
    }
    fn axis(&mut self, _roles: Roles, axis: &AxisRef) -> AxisRef {
        match self.rewrite.axis_fn {
            Some(axis_fn) => axis_fn(axis),
            None => axis.clone(),
        }
    }
    fn body(&mut self, roles: Roles, body: &Body) -> Body {
        if roles.contains_any(BODY) {
            return body.map_each(&mut |stmt| Statement::rewrite(stmt, self.rewrite));
        }
        if roles.contains_any(TERM) {
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
                Arc::new(moved.rewrite_fields(&mut Coordinates {
                    rename,
                    defined: &defined,
                }))
            });
        }
        body.clone()
    }
    fn carried(&mut self, _roles: Roles, stmt: &Arc<Statement>) -> Arc<Statement> {
        Statement::rewrite(stmt, self.rewrite)
    }
}

/// The names a substitution's replacement of `name` spells: `name` itself when it is not substituted, else the
/// sorted free variables of the replacement (a split axis reduces over its sub-axes).
fn rewrite_axis_name(name: &Name, sigma: &HashMap<Name, Expression>) -> Vec<Name> {
    match sigma.get(name) {
        None => vec![*name],
        Some(replacement) => sorted_free_vars(replacement),
    }
}

impl Statement {
    /// A load of a staged copy (`carried`) re-assigns registers declared ahead of its loop: an update, not a value.
    pub fn is_pure(&self) -> bool {
        self.flags().contains(PURE)
            && !matches!(self, Statement::Load(load) if !load.carried.is_empty())
    }

    pub fn deps_deep(&self) -> bool {
        self.flags().contains(DEPS_DEEP)
    }

    /// True iff executing this statement writes a buffer, directly or inside a child body.
    pub fn has_side_effects(&self) -> bool {
        self.flags().contains(SIDE_EFFECTS)
            || self.nested().iter().any(|body| body.has_side_effects())
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

    /// The names this statement updates in place: storage it both reads and defines (a register fragment a load
    /// fills, a state a cross-thread combine merges into).
    pub fn updates(&self) -> Vec<Name> {
        let mut visitor = NamesByRole {
            roles: UPDATES,
            out: Vec::new(),
        };
        self.visit(&mut visitor);
        // A pointwise rewrite of a fragment in place names the fragment it rewrites as its output, and a load of a
        // staged copy the registers it re-assigns.
        match self {
            Statement::FragmentApply(apply) if apply.in_place => visitor.out.push(apply.out),
            Statement::Load(load) if !load.carried.is_empty() => {
                visitor.out.extend(load.names.iter().copied())
            }
            _ => {}
        }
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

    /// The expressions of this statement's own fields, a carried program's left to its own scope.
    pub fn own_expressions(&self) -> Vec<Expression> {
        let mut visitor = OwnExpressions(Vec::new());
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
        let mut names: Vec<Name> = self.bound_axes().iter().map(|axis| axis.name).collect();
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
                this.ty(),
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

    /// This statement with its expressions replaced positionally, as `expressions` lists them; itself when none
    /// changes.
    pub fn with_expressions(this: &Arc<Statement>, expressions: Vec<Expression>) -> Arc<Statement> {
        let mut items = expressions.into_iter();
        Statement::map_expressions(this, &mut |e| items.next().unwrap_or_else(|| e.clone()))
    }

    /// This statement with `f` on each of its expressions, in the order `expressions` lists them; itself when none
    /// changes. An error ends the walk.
    pub fn try_map_expressions<E>(
        this: &Arc<Statement>,
        f: &mut dyn FnMut(&Expression) -> std::result::Result<Expression, E>,
    ) -> std::result::Result<Arc<Statement>, E> {
        let mapped: std::result::Result<Vec<Expression>, E> =
            this.expressions().iter().map(&mut *f).collect();
        Ok(Statement::with_expressions(this, mapped?))
    }

    /// This statement with `f` on each of its expressions, given the class each one's field declares; itself when
    /// none changes. An error ends the walk.
    pub fn try_map_classed_expressions<E>(
        this: &Arc<Statement>,
        f: &mut dyn FnMut(ExprClass, &Expression) -> std::result::Result<Expression, E>,
    ) -> std::result::Result<Arc<Statement>, E> {
        let mut walk = MapClassed { f, error: None };
        let new = walk.statement(this);
        match walk.error {
            Some(error) => Err(error),
            None => Ok(new),
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
                        Some(renamed) => Expression::new(Expr::FlatIndex(*renamed, index.clone())),
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
                .map(|(k, v)| (*k, v.clone()))
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
                .map(|name| (rewrite.rename)(AXIS_NAME, name).unwrap_or(name))
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
        Statement::rename_with(this, &|name| mapping.get(&name).copied())
    }

    /// `rename` through a function; `None` keeps a name.
    pub fn rename_with(
        this: &Arc<Statement>,
        rename: &dyn Fn(Name) -> Option<Name>,
    ) -> Arc<Statement> {
        Statement::rename_roles(this, &|_, name| rename(name))
    }

    /// `rename` through a function that sees the role a name has in its field (a definition, a read, an axis), so one
    /// rewrite can give a statement's reads and its definitions different names; `None` keeps a name.
    pub fn rename_roles(
        this: &Arc<Statement>,
        rename: &dyn Fn(Roles, Name) -> Option<Name>,
    ) -> Arc<Statement> {
        let axis_fn = |axis: &AxisRef| axis.renamed(&mut |name| rename(AXIS_NAME, name));
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

    /// A statement with a body field: what the text indents statements under.
    pub fn is_block(&self) -> bool {
        self.fields().iter().any(|f| f.roles.contains_any(BODY))
    }
}

impl Statement {
    /// Every SSA and axis name this subtree spells: binders, reads, locals, axis names and the variables its
    /// expressions mention. What a rename by a function is asked about.
    pub fn names(&self) -> Vec<Name> {
        struct AllNames(Vec<Name>);
        impl Visitor for AllNames {
            fn name(&mut self, roles: Roles, name: &Name) {
                if roles.contains_any(NAMES.or(AXIS_NAME)) {
                    self.0.push(*name);
                }
            }
            fn expression(&mut self, _roles: Roles, expression: &Expression) {
                expression.collect_free_vars(&mut self.0);
            }
            fn axis(&mut self, _roles: Roles, axis: &AxisRef) {
                self.0.push(axis.name);
                for source in axis.sources() {
                    self.0.push(source.name);
                }
                if let Some(window) = &axis.window {
                    for e in [&window.base, &window.bound].into_iter().flatten() {
                        e.collect_free_vars(&mut self.0);
                    }
                }
            }
            fn body(&mut self, _roles: Roles, body: &Body) {
                for stmt in body.stmts() {
                    self.0.extend(stmt.names());
                }
            }
            fn carried(&mut self, _roles: Roles, stmt: &Arc<Statement>) {
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
        if self.is_normal() {
            let resources: Vec<Name> = unique(
                self.iter_stmts()
                    .iter()
                    .flat_map(|s| s.external_reads().into_iter().chain(s.external_writes())),
            );
            let renamed: HashSet<Name> = resources
                .iter()
                .map(|name| mapping.get(name).cloned().unwrap_or(*name))
                .collect();
            if renamed.len() == resources.len() {
                result.mark_normal();
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
                        let kept = body.rebuilt(members.iter().map(|m| m.stmt().clone()).collect());
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
        self.rebuilt(out)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    use super::*;
    use crate::expr::parse::parse;
    use crate::field::{Exprs, FieldType, Names, NumOr};
    use crate::identity::canonicalize_identity;
    use crate::normalize::testing::{body, stmt_text, strs, text};
    use crate::text::parse_statement;

    /// One statement of every statement type, in declaration order; a call is spelled after the subroutine it names.
    const EXAMPLES: &[&str] = &[
        "f32 x = load a[i]",
        "y = exp(x)",
        "s <- add(s, x)",
        "c[i] <- x seed 0",
        "p = pre c[i]",
        "init f32 s = 0",
        "let n = i + 1",
        "out[i] = x",
        "m = select:\n    x when i < 2\n    y when 1",
        "zero out words=16",
        "for k in 0..8:\n    x = load a[k]\n    s <- add(s, x)",
        "for k in 0..8 step 2 until n:\n    x = load a[k]",
        "if i < 2:\n    out[] = x\nelse:\n    out[] = y",
        "sub row_sum(row in 0..4) -> s:\n    for k in 0..8:\n        x = load a[row, k]\n        s <- add(s, x)\nr = call row_sum[a](i)",
        "Smem(name=s, extents=(16, 16))",
        "Sync",
        "Tile(axes=(m in 0..4, n in 0..8), raster_axes=(m, n), raster_group=2):\n    x = load a[m, n]",
        "CpAsyncCopy(smem=s, smem_index=(i, j), src=a, src_index=(i, j), lane_index=(i, j), valid=i < 2)",
        "CpAsyncCommit",
        "CpAsyncWait(group=1)",
        "TmaDescriptor(name=d, src_buf=a, src_shape=(64, 64), box_extents=(16, 16))",
        "TmaLoad(smem=s, smem_index=(i, j), desc=d, coords=(i, j), mbar=mb, mbar_slot=p)",
        "MbarrierInit(mbar=mb, slot=p)",
        "MbarrierArriveExpectTx(mbar=mb, bytes_=64, slot=p)",
        "MbarrierArrive(mbar=mb, slot=p)",
        "SetMaxNReg(count=64, direction=dec)",
        "MbarrierWait(mbar=mb, phase=ph, slot=p)",
        "TreeHalve(bufs=(b), state=(acc), state_b=(acc__o), length=8, tid_var=t):\n    combine_states:\n        t = acc * acc__o\n        acc = add(acc, t)",
        "WarpShuffle(state=(acc), state_b=(acc__o), length=8):\n    combine_states:\n        t = acc * acc__o\n        acc = add(acc, t)",
        "WarpBroadcast(state=(acc), length=8)",
        "FragmentApply(out=o, op=where, args=(FragmentArg(kind=coord, expressions=(__fcol > __frow)), FragmentArg(kind=uniform, names=(u)), FragmentArg(kind=row, names=(r0, r1)), FragmentArg(kind=frag, names=(f)), FragmentArg(kind=gmem, expressions=(i, j), buffer=mask)), row_base=q, col_base=kv)",
        "FragmentRowReduce(top=t0, bot=b0, frags=(f0, f1), op=maximum)",
        "RegFragment(name=f, role=c, shape=(16, 8, 16), dtype=f32)",
        "BlockScaleLoad(frag=sf, src_buffer=s, src_index=(i, j), role=a)",
        "LdmatrixLoad(frag=f, src_buffer=s, src_index=(i, j), role=b, gmem_guard=(i, j), k_zero=(i, j), pair_frag=g, scale_buffer=sc, scale_index=(i, j))",
        "MmaSyncPtx(c_frag=c, a_frag=a, b_frag=b, shape=(16, 8, 16), sfa_frag=sa, sfb_frag=sb)",
        "FragmentPromote(dst=c, src=h)",
        "FragmentRepack(frag=a, srcs=(c0, c1))",
        "WgmmaDescriptor(name=bD, smem=s, smem_index=i, swizzle=B128, lbo_bytes=16, sbo_bytes=1024)",
        "WgmmaMma(c_frags=(c0, c1, c2, c3, c4, c5, c6, c7), b_desc=bD, shape=(64, 64, 16), a_desc=aD)",
        "WgmmaFence",
        "WgmmaCommit",
        "WgmmaWait",
        "RegStore(dst_buffer=out, dst_index=(i, j), frag=c, shape=(16, 8, 16), epilogue_params=(acc), epilogue_result=y, extra_frags=(c2), m_guard=(i, j), n_guard=(i, j)):\n    epilogue:\n        r = load res[i, j]\n        y = add(acc, r)\n    run:\n        RegStore(dst_buffer=out, dst_index=(i, j), frag=d, shape=(16, 8, 16))",
    ];

    /// A wgmma cell over eight C fragments, its A operand a shared-memory descriptor.
    const SS: &str = "WgmmaMma(c_frags=(c0, c1, c2, c3, c4, c5, c6, c7), b_desc=bD, shape=(64, 64, 16), ab_dtype=bf16, a_desc=aD)";
    /// The same cell with its A operand a register fragment.
    const RS: &str = "WgmmaMma(c_frags=(c0, c1, c2, c3, c4, c5, c6, c7), b_desc=bD, shape=(64, 64, 16), ab_dtype=bf16, a_frag=a)";
    const FRAGS: [&str; 8] = ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7"];

    fn statement(text: &str) -> Arc<Statement> {
        parse_statement(text).unwrap_or_else(|err| panic!("{err}\n{text}"))
    }

    /// The examples, one per statement type.
    fn examples() -> Vec<Arc<Statement>> {
        let examples: Vec<Arc<Statement>> = EXAMPLES.iter().map(|text| statement(text)).collect();
        let statement_types: Vec<StatementType> = examples.iter().map(|stmt| stmt.ty()).collect();
        assert_eq!(
            statement_types,
            StatementType::ALL,
            "one example per statement type, in declaration order"
        );
        examples
    }

    fn kernel_examples() -> Vec<Arc<Statement>> {
        examples()
            .into_iter()
            .filter(|stmt| stmt.layer() == Layer::Kernel)
            .collect()
    }

    fn example(ty: StatementType) -> Arc<Statement> {
        examples().into_iter().find(|stmt| stmt.ty() == ty).unwrap()
    }

    fn mapping(pairs: &[(&str, &str)]) -> HashMap<Name, Name> {
        pairs
            .iter()
            .map(|(from, to)| (Name::from(*from), Name::from(*to)))
            .collect()
    }

    /// Every name `stmt` spells, sent through `f`: the mapping a rename by a function asks for.
    fn names_by(stmt: &Statement, f: impl Fn(&str) -> String) -> HashMap<Name, Name> {
        stmt.names()
            .into_iter()
            .map(|name| {
                let renamed = Name::from(f(&name));
                (name, renamed)
            })
            .collect()
    }

    /// `stmt` with every name through `names` and every expression through `sigma`, its axes as they are.
    fn rewritten(
        stmt: &Arc<Statement>,
        names: &HashMap<Name, Name>,
        sigma: &HashMap<Name, Expression>,
    ) -> Arc<Statement> {
        Statement::rewrite(
            stmt,
            &Rewrite {
                rename: &|_, name| names.get(&name).copied(),
                sigma,
                axis_fn: None,
            },
        )
    }

    /// `stmt` with every name it spells suffixed.
    fn suffixed(stmt: &Arc<Statement>, suffix: &str) -> Arc<Statement> {
        let names = names_by(stmt, |name| format!("{name}{suffix}"));
        rewritten(stmt, &names, &HashMap::new())
    }

    /// `stmt` rewritten with every name kept.
    fn kept(stmt: &Arc<Statement>) -> Arc<Statement> {
        rewritten(stmt, &names_by(stmt, str::to_string), &HashMap::new())
    }

    fn owned(names: &[Name]) -> Vec<String> {
        names.iter().map(|name| name.to_string()).collect()
    }

    fn suffix_all(names: &[Name], suffix: &str) -> Vec<String> {
        names.iter().map(|name| format!("{name}{suffix}")).collect()
    }

    fn expression(text: &str) -> Expression {
        parse(text).unwrap()
    }

    #[test]
    fn a_statement_type_holds_its_elementwise_operation_as_op() {
        // The clustered identity reads a statement's operation by the name `op`, in every layer.
        let is_op = |ty: &FieldType| match ty {
            FieldType::ElementwiseOp => true,
            FieldType::Opt(inner) => **inner == FieldType::ElementwiseOp,
            _ => false,
        };
        let held: Vec<(&str, &str)> = StatementType::ALL
            .iter()
            .flat_map(|ty| {
                ty.fields()
                    .iter()
                    .filter(|field| ty.field_ty(field.name).as_ref().is_some_and(is_op))
                    .map(|field| (ty.name(), field.name))
            })
            .collect();
        let types: HashSet<&str> = held.iter().map(|(ty, _)| *ty).collect();
        let names: HashSet<&str> = held.iter().map(|(_, name)| *name).collect();
        assert!(types.is_superset(&HashSet::from(["Accum"])), "{held:?}");
        assert_eq!(names, HashSet::from(["op"]));
    }

    #[test]
    fn a_number_in_a_role_field_is_plain_data() {
        // A literal seed is no buffer.
        let carry = statement("state[i] <- value seed 0.5");
        assert!(carry.external_reads().is_empty());
        let renamed = Statement::rename_buffers(&carry, &mapping(&[("state", "other")]));
        assert!(Arc::ptr_eq(&renamed, &carry));
        let Statement::Carry(fields) = &*carry else {
            panic!("not a carry");
        };
        let seeded = Statement::Carry(Carry {
            seed: NumOr::Value("init".into()),
            ..fields.clone()
        });
        assert_eq!(strs(&seeded.external_reads()), ["init"]);
    }

    #[test]
    fn a_field_read_through_lists_each_name_once() {
        // The names an index or a predicate mentions are reads, listed in field order beside the names a field holds.
        let write = statement("out[i, i + gather] = value");
        assert_eq!(strs(&write.deps()), ["i", "gather", "value"]);
        let select = statement("picked = select:\n    x when i < 2\n    y when i >= 2");
        assert_eq!(strs(&select.deps()), ["x", "i", "y"]);
        let carry = statement("state[i] <- value seed 0");
        assert_eq!(strs(&carry.deps()), ["value"]);
    }

    #[test]
    fn a_rename_reaches_every_name_and_no_buffer() {
        // What a declared statement reads and defines follows a rename; the buffers it reads and writes follow a buffer
        // rename, the ones a carried statement or an epilogue reads included.
        for stmt in kernel_examples() {
            let ty = stmt.ty();
            let renamed = suffixed(&stmt, "_");
            assert_eq!(
                owned(&renamed.deps()),
                suffix_all(&stmt.deps(), "_"),
                "{ty}"
            );
            assert_eq!(
                owned(&renamed.defines()),
                suffix_all(&stmt.defines(), "_"),
                "{ty}"
            );
            let defines = stmt.defines();
            let locals: Vec<String> = stmt
                .local_decls()
                .iter()
                .map(|name| {
                    if defines.contains(name) {
                        format!("{name}_")
                    } else {
                        name.to_string()
                    }
                })
                .collect();
            assert_eq!(
                (
                    renamed.external_reads(),
                    renamed.external_writes(),
                    owned(&renamed.local_decls())
                ),
                (stmt.external_reads(), stmt.external_writes(), locals),
                "{ty}"
            );
            let buffers: HashMap<Name, Name> = stmt
                .external_reads()
                .into_iter()
                .chain(stmt.external_writes())
                .map(|name| {
                    let renamed = Name::from(format!("{name}_"));
                    (name, renamed)
                })
                .collect();
            let moved = Statement::rename_buffers(&stmt, &buffers);
            let through = |names: Vec<Name>| -> Vec<Name> {
                names.iter().map(|name| buffers[name]).collect()
            };
            assert_eq!(
                moved.external_reads(),
                through(stmt.external_reads()),
                "{ty}"
            );
            assert_eq!(
                moved.external_writes(),
                through(stmt.external_writes()),
                "{ty}"
            );
            assert_eq!(
                (moved.deps(), moved.defines()),
                (stmt.deps(), stmt.defines()),
                "{ty}"
            );
        }
    }

    #[test]
    fn a_carried_program_hides_its_own_names() {
        // A combine a statement carries reads the state and its shuffled copy and binds temporaries; the statement
        // reads the state alone, and updates it in place.
        let halve = example(StatementType::TreeHalve);
        assert_eq!(strs(&halve.deps()), ["acc"]);
        assert_eq!(strs(&halve.defines()), ["acc"]);
        let Statement::TreeHalve(moved) = &*suffixed(&halve, "_") else {
            panic!("not a tree halve");
        };
        assert_eq!(*moved.combine_states[1], *statement("acc_ = add(acc_, t_)"));
    }

    #[test]
    fn a_term_keeps_its_names_and_follows_a_substitution() {
        // A store's epilogue names are its own: a rename leaves them and reaches the coordinates its indices read.
        let store = example(StatementType::RegStore);
        let sigma = HashMap::from([(Name::from("i"), expression("i + 4"))]);
        let names = names_by(&store, |name| format!("{name}_"));
        let Statement::RegStore(moved) = &*rewritten(&store, &names, &sigma) else {
            panic!("not a register store");
        };
        let epilogue = moved.epilogue.as_ref().unwrap().stmts();
        assert_eq!(*epilogue[1], *statement("y = add(acc, r)"));
        assert_eq!(strs(&moved.epilogue_params), ["acc"]);
        let load = Statement::Load(Load {
            names: Names::from([Name::from("r")]),
            input: "res".into(),
            index: moved.dst_index.clone(),
            dtype: None,
            carried: "".into(),
        });
        assert_eq!(*epilogue[0], load);
        assert_eq!(
            moved.dst_index[..],
            [expression("i_ + 4"), expression("j_")]
        );
        let Statement::RegStore(run) = &*moved.run[0] else {
            panic!("not a register store");
        };
        assert_eq!(
            (&*moved.frag, strs(&moved.extra_frags), &*run.frag),
            ("c_", vec!["c2_"], "d_")
        );
    }

    #[test]
    fn a_bound_axis_name_follows_its_axis() {
        let tile = statement(
            "Tile(axes=(m in 0..4, n in 0..8), raster_axes=(m, n), raster_group=2, spread=true):\n    x = load a[m, n]",
        );
        let renamed = Statement::rename(&tile, &mapping(&[("m", "m2"), ("x", "x2")]));
        let Statement::Tile(fields) = &*renamed else {
            panic!("not a tile");
        };
        let axes: Vec<&str> = fields.axes.iter().map(|axis| &*axis.name).collect();
        assert_eq!(axes, ["m2", "n"]);
        assert_eq!(
            fields.raster_axes,
            Some((Name::from("m2"), Name::from("n")))
        );
        assert_eq!(fields.body, body("x2 = load a[m2, n]"));
        assert!(fields.spread);
        let bound: HashSet<Name> = renamed.binds_axes().into_iter().collect();
        assert_eq!(bound, HashSet::from([Name::from("m2"), Name::from("n")]));
    }

    #[test]
    fn a_statement_rebuilds_as_itself() {
        // Every rebuilding member, given what its reading half returned, gives back an equal statement.
        for stmt in examples() {
            let ty = stmt.ty();
            assert_eq!(*kept(&stmt), *stmt, "{ty}");
            assert_eq!(
                *Statement::with_bodies(&stmt, stmt.nested()).unwrap(),
                *stmt,
                "{ty}"
            );
            let mapped = Statement::map_expressions(&stmt, &mut |expression| expression.clone());
            assert!(Arc::ptr_eq(&mapped, &stmt), "{ty}");
            assert_eq!(
                *Statement::rename_buffers(&stmt, &HashMap::new()),
                *stmt,
                "{ty}"
            );
        }
    }

    /// Whether a value of type `ty` is one `want` accepts, or holds one in an option, a tuple or a sequence.
    fn holds(ty: &FieldType, want: &dyn Fn(&FieldType) -> bool) -> bool {
        want(ty)
            || match ty {
                FieldType::Opt(inner) | FieldType::Seq(inner) | FieldType::NumOr(inner) => {
                    holds(inner, want)
                }
                FieldType::Tuple(items) => items.iter().any(|item| holds(item, want)),
                _ => false,
            }
    }

    #[test]
    fn an_expression_map_reaches_every_expression_a_statement_holds() {
        // `expressions` lists every expression of the statement and `map_expressions` reaches each one in that
        // order. A walk visits a field that plays a role, so every field that holds an expression plays EXPRESSION
        // and every field that holds statements plays BODY, TERM or STMTS: then the walk reaches every one.
        let declared = StatementType::ALL
            .iter()
            .flat_map(|ty| {
                ty.fields()
                    .iter()
                    .map(move |field| (ty.name(), field, ty.field_ty(field.name)))
            })
            .chain(SelectBranch::FIELDS.iter().map(|field| {
                (
                    SelectBranch::NAME,
                    field,
                    SelectBranch::field_ty(field.name),
                )
            }))
            .chain(
                FragmentArg::FIELDS
                    .iter()
                    .map(|field| (FragmentArg::NAME, field, FragmentArg::field_ty(field.name))),
            );
        for (owner, field, ty) in declared {
            let ty = ty.unwrap();
            if holds(&ty, &|ty| *ty == FieldType::Expr) {
                assert!(
                    field.roles.contains_any(EXPRESSION),
                    "{owner}.{}",
                    field.name
                );
            }
            if holds(&ty, &|ty| *ty == FieldType::Block) {
                assert!(
                    field.roles.contains_any(BODY.or(TERM).or(STMTS)),
                    "{owner}.{}",
                    field.name
                );
            }
        }
        for stmt in examples() {
            let ty = stmt.ty();
            let markers: Vec<Expression> = (0..stmt.expressions().len())
                .map(|index| Expression::var(&format!("e{index}")))
                .collect();
            let mut fresh = markers.iter();
            let mapped = Statement::map_expressions(&stmt, &mut |_| fresh.next().unwrap().clone());
            assert_eq!(mapped.expressions(), markers, "{ty}");
            let mut back = stmt.expressions().into_iter();
            let restored = Statement::map_expressions(&mapped, &mut |_| back.next().unwrap());
            assert_eq!(*restored, *stmt, "{ty}");
        }
    }

    #[test]
    fn a_rename_reaches_names_and_expressions() {
        let names = mapping(&[("x", "x2"), ("i", "j")]);
        let both = body("x = load a[i]\nout[i] = x");
        let [load, write] = both.stmts() else {
            panic!("two statements");
        };
        assert_eq!(
            stmt_text(&Statement::rename(load, &names)),
            "x2 = load a[j]"
        );
        assert_eq!(stmt_text(&Statement::rename(write, &names)), "out[j] = x2");
        let mapped =
            Statement::map_expressions(load, &mut |expression| expression.rename_by(&names));
        assert_eq!(stmt_text(&mapped), "x = load a[j]");
    }

    /// An accumulator buffer zeroed ahead of a reduction over `x`, the sum stored to `out`.
    fn zeroed_sum(x: &str, out: &str, zero: &str) -> Body {
        body(&format!(
            "zero {zero} words=16\nfor k in 0..8:\n    v = load {x}[k]\n    w = relu(v)\n    acc <- add(acc, w) over k\n{out}[] = acc"
        ))
    }

    #[test]
    fn a_body_rename_reaches_nested_leaves_and_every_buffer_field() {
        let renamed = zeroed_sum("x", "o", "acc_buf").rename_buffers(&mapping(&[
            ("x", "x9"),
            ("o", "o9"),
            ("acc_buf", "z9"),
        ]));
        assert_eq!(text(&renamed), text(&zeroed_sum("x9", "o9", "z9")));
    }

    #[test]
    fn a_rename_is_the_identity_off_the_mapping() {
        let sum = zeroed_sum("x", "o", "acc_buf");
        assert_eq!(sum.rename_buffers(&mapping(&[("unrelated", "name")])), sum);
    }

    #[test]
    fn the_identity_covers_buffer_fields_other_than_load_and_write() {
        // A body that differs only in the buffer its zero prologue clears keys as the other one does.
        let key = |zero: &str| {
            canonicalize_identity(&zeroed_sum("x", "o", zero), true, None)
                .unwrap()
                .key
        };
        assert_eq!(key("acc_buf"), key("differently_named"));
    }

    #[test]
    fn the_cell_reads_its_operands_and_defines_its_accumulator() {
        let ss = statement(SS);
        assert_eq!(strs(&ss.deps()), [&FRAGS[..], &["bD", "aD"]].concat());
        assert_eq!(strs(&ss.defines()), FRAGS);
        let rs = statement(RS);
        assert_eq!(strs(&rs.deps()), [&FRAGS[..], &["bD", "a"]].concat());
        assert_eq!(strs(&rs.defines()), FRAGS);
    }

    #[test]
    fn a_rewrite_renames_ssa_names_and_keeps_buffers() {
        let cells = body(
            &[
                "WgmmaDescriptor(name=bD, smem=K_s, smem_index=k0 * 64, swizzle=B128, lbo_bytes=16, sbo_bytes=1024)",
                "WgmmaFence",
                SS,
                RS,
                "WgmmaCommit",
                "WgmmaWait",
            ]
            .join("\n"),
        );
        let renamed = cells.map_each(&mut |stmt| suffixed(stmt, "_r"));
        let [desc, fence, ss, rs, commit, wait] = renamed.stmts() else {
            panic!("six statements");
        };
        assert_eq!(
            **desc,
            *statement(
                "WgmmaDescriptor(name=bD_r, smem=K_s, smem_index=k0_r * 64, swizzle=B128, lbo_bytes=16, sbo_bytes=1024)"
            )
        );
        let (Statement::WgmmaMma(ss), Statement::WgmmaMma(rs)) = (&**ss, &**rs) else {
            panic!("not two wgmma cells");
        };
        let frags: Vec<String> = FRAGS.iter().map(|frag| format!("{frag}_r")).collect();
        assert_eq!(owned(&ss.c_frags), frags);
        assert_eq!((ss.a_desc.as_deref(), &*ss.b_desc), (Some("aD_r"), "bD_r"));
        assert_eq!(
            (rs.a_frag.as_deref(), rs.a_desc.as_deref(), &*rs.b_desc),
            (Some("a_r"), None, "bD_r")
        );
        for (stmt, spelled) in [
            (fence, "WgmmaFence"),
            (commit, "WgmmaCommit"),
            (wait, "WgmmaWait"),
        ] {
            assert_eq!(**stmt, *statement(spelled));
        }
        assert_eq!(cells.map_each(&mut kept), cells);
    }

    #[test]
    fn a_rewrite_substitutes_bases_and_templates_but_not_the_reserved_coordinates() {
        let mask = statement(
            "FragmentApply(out=s, op=where, args=(FragmentArg(kind=coord, expressions=(__fcol > __frow)), FragmentArg(kind=uniform, expressions=(-1e+30)), FragmentArg(kind=frag, names=(s))), in_place=true, row_base=q, col_base=kv + 8)",
        );
        let sigma = HashMap::from([
            (Name::from("q"), Expression::var("a1")),
            (Name::from("kv"), Expression::var("a3")),
        ]);
        let names = names_by(&mask, |name| match name {
            "s" => "s9".to_string(),
            _ => name.to_string(),
        });
        let moved = rewritten(&mask, &names, &sigma);
        let Statement::FragmentApply(fields) = &*moved else {
            panic!("not a fragment apply");
        };
        assert_eq!(&*fields.out, "s9");
        let frag = FragmentArg {
            kind: "frag".into(),
            names: Names::from([Name::from("s9")]),
            expressions: Exprs::from([]),
            buffer: None,
        };
        assert_eq!(fields.args[2], frag);
        assert_eq!(fields.row_base, Some(Expression::var("a1")));
        // The reserved coordinates are not local axes.
        assert_eq!(
            fields.args[0].expressions[..],
            [expression("__fcol > __frow")]
        );
        // A predicate and a literal contribute no SSA reads.
        assert_eq!(strs(&moved.deps()), ["s9"]);
        assert_eq!(moved.expressions()[0], fields.args[0].expressions[0]);
    }

    #[test]
    fn a_rewrite_keeps_the_segment_layout() {
        // The transposed combine halves segment by segment; losing `inner` in a rewrite folds across outputs.
        let halve = statement(
            "TreeHalve(bufs=(b), state=(acc0), state_b=(acc0__o), length=4, tid_var=k_co, inner=(n_ln, 32)):\n    combine_states:\n        acc0 = add(acc0, acc0__o)",
        );
        let names = names_by(&halve, str::to_string);
        let same_axis = |axis: &AxisRef| axis.clone();
        let rewrite = Rewrite {
            rename: &|_, name| names.get(&name).copied(),
            sigma: &HashMap::new(),
            axis_fn: Some(&same_axis),
        };
        for moved in [Statement::rewrite(&halve, &rewrite), suffixed(&halve, "_")] {
            let Statement::TreeHalve(moved) = &*moved else {
                panic!("not a tree halve");
            };
            assert_eq!(moved.inner.as_ref().map(|(_, scale)| *scale), Some(32));
        }
    }
}
