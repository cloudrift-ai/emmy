//! What a declared kind's fields are made of, and the one trait every field type implements: how it is visited and
//! rewritten by role, spelled on a header and read back by its type.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::axis::Axis;
use crate::axis::spell_axis;
use crate::body::Body;
use crate::elementwise::ElementwiseOp;
use crate::expr::{Expression, Num, float_repr, hash_f64, same_f64, spell_name};
use crate::statement::{
    AXIS, AXIS_NAME, BODY, BUFFERS, EXPRESSION, FragmentArg, NAMES, Roles, STMTS, SelectBranch,
    Statement, TERM,
};
use crate::{Error, Name, Result};

/// Words the reader gives a meaning, so a name spelled like one is written between backticks.
pub const KEYWORDS: [&str; 25] = [
    "for", "in", "if", "else", "let", "init", "zero", "load", "pre", "select", "when", "over",
    "seed", "step", "until", "window", "carried", "swizzle", "words", "unroll", "sub", "call",
    "true", "false", "none",
];

/// A name as a statement's text spells it: bare, or between backticks when it reads as a keyword or is no identifier.
pub fn spell_stmt_name(name: &str) -> Result<String> {
    spell_name(name, &KEYWORDS)
}

pub type Names = Arc<[Name]>;
pub type Exprs = Arc<[Expression]>;
pub type AxisRef = Arc<Axis>;
pub type Axes = Arc<[AxisRef]>;
pub type Stmts = Arc<[Arc<Statement>]>;
pub type Branches = Arc<[SelectBranch]>;
pub type FragmentArgs = Arc<[FragmentArg]>;
pub type ElementwiseOps = Arc<[ElementwiseOp]>;
pub type Ints = Arc<[i64]>;
pub type Floats = Arc<[Float]>;

pub use crate::dtype::DType;

/// A float with the equality a Python float has (`0.0 == -0.0`), hashable.
#[derive(Clone, Copy, Debug)]
pub struct Float(pub f64);

impl PartialEq for Float {
    fn eq(&self, other: &Self) -> bool {
        same_f64(self.0, other.0)
    }
}

impl Eq for Float {}

impl Hash for Float {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_f64(self.0, state);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Shape3(pub i64, pub i64, pub i64);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Int2(pub i64, pub i64);

/// A field that holds a number where a name, a buffer or an expression is expected: the number is plain data.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum NumOr<T> {
    Num(Num),
    Value(T),
}

/// A shared reduction body a `Call` names: compared by object, like Python's.
#[derive(Clone, Debug)]
pub struct SubroutineRef(pub Arc<crate::body::Subroutine>);

impl PartialEq for SubroutineRef {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for SubroutineRef {}

impl Hash for SubroutineRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (Arc::as_ptr(&self.0) as usize).hash(state);
    }
}

impl std::ops::Deref for SubroutineRef {
    type Target = crate::body::Subroutine;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// --- walks ----------------------------------------------------------------------------------------

/// Reads every role-playing value of a statement, told the roles of the field it sits in.
pub trait Visitor {
    fn name(&mut self, _roles: Roles, _name: &Name) {}
    fn expression(&mut self, _roles: Roles, _expression: &Expression) {}
    fn axis(&mut self, _roles: Roles, _axis: &AxisRef) {}
    /// A number a field holds as plain data: a coefficient, an offset, a literal seed.
    fn number(&mut self, _value: Num) {}
    fn body(&mut self, _roles: Roles, _body: &Body) {}
    fn carried(&mut self, _roles: Roles, _stmt: &Arc<Statement>) {}
}

/// Rebuilds a statement value by value; the default keeps each.
pub trait Rewriter {
    fn name(&mut self, _roles: Roles, name: &Name) -> Name {
        name.clone()
    }
    fn expression(&mut self, _roles: Roles, expression: &Expression) -> Expression {
        expression.clone()
    }
    fn axis(&mut self, _roles: Roles, axis: &AxisRef) -> AxisRef {
        axis.clone()
    }
    fn body(&mut self, _roles: Roles, body: &Body) -> Body {
        body.clone()
    }
    fn carried(&mut self, _roles: Roles, stmt: &Arc<Statement>) -> Arc<Statement> {
        stmt.clone()
    }
}

// --- the text reader's values ---------------------------------------------------------------------

/// How a header field is read.
#[derive(Clone, Debug, PartialEq)]
pub enum FieldType {
    Name,
    Int,
    Float,
    Bool,
    DType,
    ElementwiseOp,
    Expr,
    Axis,
    Block,
    Subroutine,
    Opt(Box<FieldType>),
    Seq(Box<FieldType>),
    Tuple(Vec<FieldType>),
    Record(&'static str),
    NumOr(Box<FieldType>),
}

/// A value the reader collected, typed by what it read.
#[derive(Clone, Debug)]
pub enum Value {
    None,
    Bool(bool),
    Number(Num),
    Name(Name),
    Expr(Expression),
    Axis(AxisRef),
    Tuple(Vec<Value>),
    Record(String, Values),
    Body(Body),
    Stmts(Vec<Arc<Statement>>),
}

impl Value {
    pub(crate) fn mismatch(&self, wanted: &str) -> Error {
        Error::Value(format!("expected {wanted}, got {self:?}"))
    }
}

/// The field values a header and its blocks gave, by field name.
#[derive(Clone, Debug, Default)]
pub struct Values(pub HashMap<String, Value>);

impl Values {
    pub fn take(&mut self, field: &str) -> Option<Value> {
        self.0.remove(field)
    }

    pub fn set(&mut self, field: &str, value: Value) -> Result<()> {
        if self.0.insert(field.to_string(), value).is_some() {
            return Err(Error::Value(format!("{field} is given twice")));
        }
        Ok(())
    }

    pub fn finish(self, kind: &str) -> Result<()> {
        match self.0.keys().next() {
            Some(field) => Err(Error::Value(format!("a {kind} has no field {field:?}"))),
            None => Ok(()),
        }
    }
}

// --- the field trait ------------------------------------------------------------------------------

/// What every field type does: it is visited and rewritten by the roles of its field, spelled on a header and read
/// back by its type.
pub trait Field: Clone + PartialEq + Eq + Hash + std::fmt::Debug + Sized + 'static {
    fn ty() -> FieldType;
    fn visit(&self, roles: Roles, visitor: &mut dyn Visitor);
    fn rewrite(&self, roles: Roles, rewriter: &mut dyn Rewriter) -> Self;
    /// The same value, by identity where the value is shared: what tells a rebuild that changed nothing.
    fn same(&self, other: &Self) -> bool {
        self == other
    }
    fn equals(&self, other: &Self) -> bool {
        self == other
    }
    fn spell(&self, out: &mut String) -> Result<()>;
    fn from_value(value: Value) -> Result<Self>;
    fn is_block() -> bool {
        false
    }
    /// This field as an operation: `ElementwiseOp` answers.
    fn as_op(&self) -> Option<&ElementwiseOp> {
        None
    }
    /// This field with `op` in place of the operation it is; itself when it is none.
    fn with_op(&self, _op: &ElementwiseOp) -> Self {
        self.clone()
    }
    /// This field as a dtype: `DType` answers, and an `Option` holding one.
    fn as_dtype(&self) -> Option<&DType> {
        None
    }
    /// This field as a name: `Name` answers.
    fn as_name(&self) -> Option<&Name> {
        None
    }
    /// The statements a block field holds.
    fn block(&self) -> Vec<Arc<Statement>> {
        vec![]
    }
}

fn spell_seq<T: Field>(items: &[T], out: &mut String) -> Result<()> {
    out.push('(');
    for (index, item) in items.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        item.spell(out)?;
    }
    out.push(')');
    Ok(())
}

fn seq_from_value<T: Field>(value: Value) -> Result<Vec<T>> {
    match value {
        Value::Tuple(items) => items.into_iter().map(T::from_value).collect(),
        other => Err(other.mismatch("a tuple")),
    }
}

fn rewrite_seq<T: Field>(items: &Arc<[T]>, roles: Roles, rewriter: &mut dyn Rewriter) -> Arc<[T]> {
    let out: Vec<T> = items
        .iter()
        .map(|item| item.rewrite(roles, rewriter))
        .collect();
    if out.iter().zip(items.iter()).all(|(new, old)| new.same(old)) {
        items.clone()
    } else {
        out.into()
    }
}

impl Field for Name {
    fn ty() -> FieldType {
        FieldType::Name
    }
    fn visit(&self, roles: Roles, visitor: &mut dyn Visitor) {
        if roles != Roles::NONE {
            visitor.name(roles, self);
        }
    }
    fn rewrite(&self, roles: Roles, rewriter: &mut dyn Rewriter) -> Self {
        if roles.contains_any(NAMES.or(BUFFERS).or(AXIS_NAME)) {
            rewriter.name(roles, self)
        } else {
            self.clone()
        }
    }
    fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(self, other) || self == other
    }
    fn spell(&self, out: &mut String) -> Result<()> {
        out.push_str(&spell_stmt_name(self)?);
        Ok(())
    }
    fn from_value(value: Value) -> Result<Self> {
        match value {
            Value::Name(name) => Ok(name),
            other => Err(other.mismatch("a name")),
        }
    }
    fn as_name(&self) -> Option<&Name> {
        Some(self)
    }
}

impl<T: Field> Field for Arc<[T]> {
    fn ty() -> FieldType {
        FieldType::Seq(Box::new(T::ty()))
    }
    fn visit(&self, roles: Roles, visitor: &mut dyn Visitor) {
        for item in self.iter() {
            item.visit(roles, visitor);
        }
    }
    fn rewrite(&self, roles: Roles, rewriter: &mut dyn Rewriter) -> Self {
        rewrite_seq(self, roles, rewriter)
    }
    fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(self, other)
            || (self.len() == other.len() && self.iter().zip(other.iter()).all(|(a, b)| a.same(b)))
    }
    fn spell(&self, out: &mut String) -> Result<()> {
        spell_seq(self, out)
    }
    fn from_value(value: Value) -> Result<Self> {
        Ok(seq_from_value::<T>(value)?.into())
    }
    fn is_block() -> bool {
        T::is_block()
    }
    fn block(&self) -> Vec<Arc<Statement>> {
        self.iter().flat_map(|item| item.block()).collect()
    }
}

impl<T: Field> Field for Option<T> {
    fn ty() -> FieldType {
        FieldType::Opt(Box::new(T::ty()))
    }
    fn visit(&self, roles: Roles, visitor: &mut dyn Visitor) {
        if let Some(value) = self {
            value.visit(roles, visitor);
        }
    }
    fn rewrite(&self, roles: Roles, rewriter: &mut dyn Rewriter) -> Self {
        self.as_ref().map(|value| value.rewrite(roles, rewriter))
    }
    fn same(&self, other: &Self) -> bool {
        match (self, other) {
            (None, None) => true,
            (Some(a), Some(b)) => a.same(b),
            _ => false,
        }
    }
    fn spell(&self, out: &mut String) -> Result<()> {
        match self {
            None => {
                out.push_str("none");
                Ok(())
            }
            Some(value) => value.spell(out),
        }
    }
    fn from_value(value: Value) -> Result<Self> {
        match value {
            Value::None => Ok(None),
            other => Ok(Some(T::from_value(other)?)),
        }
    }
    fn is_block() -> bool {
        T::is_block()
    }
    fn as_dtype(&self) -> Option<&DType> {
        self.as_ref().and_then(Field::as_dtype)
    }
    fn block(&self) -> Vec<Arc<Statement>> {
        self.as_ref().map_or_else(Vec::new, |value| value.block())
    }
}

impl<A: Field, B: Field> Field for (A, B) {
    fn ty() -> FieldType {
        FieldType::Tuple(vec![A::ty(), B::ty()])
    }
    fn visit(&self, roles: Roles, visitor: &mut dyn Visitor) {
        self.0.visit(roles, visitor);
        self.1.visit(roles, visitor);
    }
    fn rewrite(&self, roles: Roles, rewriter: &mut dyn Rewriter) -> Self {
        (
            self.0.rewrite(roles, rewriter),
            self.1.rewrite(roles, rewriter),
        )
    }
    fn same(&self, other: &Self) -> bool {
        self.0.same(&other.0) && self.1.same(&other.1)
    }
    fn spell(&self, out: &mut String) -> Result<()> {
        out.push('(');
        self.0.spell(out)?;
        out.push_str(", ");
        self.1.spell(out)?;
        out.push(')');
        Ok(())
    }
    fn from_value(value: Value) -> Result<Self> {
        match value {
            Value::Tuple(items) if items.len() == 2 => {
                let [a, b]: [Value; 2] = items
                    .try_into()
                    .map_err(|_| Error::Value("expected a pair".into()))?;
                Ok((A::from_value(a)?, B::from_value(b)?))
            }
            other => Err(other.mismatch("a pair")),
        }
    }
}

impl Field for Expression {
    fn ty() -> FieldType {
        FieldType::Expr
    }
    fn visit(&self, roles: Roles, visitor: &mut dyn Visitor) {
        if roles != Roles::NONE {
            visitor.expression(roles, self);
        }
    }
    fn rewrite(&self, roles: Roles, rewriter: &mut dyn Rewriter) -> Self {
        if roles.contains_any(EXPRESSION) {
            rewriter.expression(roles, self)
        } else {
            self.clone()
        }
    }
    fn same(&self, other: &Self) -> bool {
        self == other
    }
    fn spell(&self, out: &mut String) -> Result<()> {
        out.push_str(&self.try_text()?);
        Ok(())
    }
    fn from_value(value: Value) -> Result<Self> {
        match value {
            Value::Expr(e) => Ok(e),
            other => Err(other.mismatch("an expression")),
        }
    }
}

impl Field for AxisRef {
    fn ty() -> FieldType {
        FieldType::Axis
    }
    fn visit(&self, roles: Roles, visitor: &mut dyn Visitor) {
        if roles.contains_any(AXIS) {
            visitor.axis(roles, self);
        }
    }
    fn rewrite(&self, roles: Roles, rewriter: &mut dyn Rewriter) -> Self {
        if roles.contains_any(AXIS) {
            rewriter.axis(roles, self)
        } else {
            self.clone()
        }
    }
    fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(self, other) || self == other
    }
    fn spell(&self, out: &mut String) -> Result<()> {
        out.push_str(&spell_axis(self, None)?);
        Ok(())
    }
    fn from_value(value: Value) -> Result<Self> {
        match value {
            Value::Axis(axis) => Ok(axis),
            other => Err(other.mismatch("an axis")),
        }
    }
}

impl Field for Body {
    fn ty() -> FieldType {
        FieldType::Block
    }
    fn visit(&self, roles: Roles, visitor: &mut dyn Visitor) {
        if roles.contains_any(BODY.or(TERM)) {
            visitor.body(roles, self);
        }
    }
    fn rewrite(&self, roles: Roles, rewriter: &mut dyn Rewriter) -> Self {
        if roles.contains_any(BODY.or(TERM)) {
            rewriter.body(roles, self)
        } else {
            self.clone()
        }
    }
    fn same(&self, other: &Self) -> bool {
        self.ptr_eq(other)
    }
    fn spell(&self, _out: &mut String) -> Result<()> {
        Err(Error::Value(
            "a body is spelled as a block, not on a header".into(),
        ))
    }
    fn from_value(value: Value) -> Result<Self> {
        match value {
            Value::Body(body) => Ok(body),
            Value::Stmts(stmts) => Ok(Body::new(stmts)),
            other => Err(other.mismatch("a body")),
        }
    }
    fn is_block() -> bool {
        true
    }
    fn block(&self) -> Vec<Arc<Statement>> {
        self.stmts().to_vec()
    }
}

impl Field for Stmts {
    fn ty() -> FieldType {
        FieldType::Block
    }
    fn visit(&self, roles: Roles, visitor: &mut dyn Visitor) {
        if roles.contains_any(STMTS) {
            for stmt in self.iter() {
                visitor.carried(roles, stmt);
            }
        }
    }
    fn rewrite(&self, roles: Roles, rewriter: &mut dyn Rewriter) -> Self {
        if !roles.contains_any(STMTS) {
            return self.clone();
        }
        let out: Vec<Arc<Statement>> = self
            .iter()
            .map(|stmt| rewriter.carried(roles, stmt))
            .collect();
        if out
            .iter()
            .zip(self.iter())
            .all(|(new, old)| Arc::ptr_eq(new, old))
        {
            self.clone()
        } else {
            out.into()
        }
    }
    fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(self, other)
            || (self.len() == other.len()
                && self
                    .iter()
                    .zip(other.iter())
                    .all(|(a, b)| Arc::ptr_eq(a, b) || a == b))
    }
    fn spell(&self, _out: &mut String) -> Result<()> {
        Err(Error::Value(
            "carried statements are spelled as a block, not on a header".into(),
        ))
    }
    fn from_value(value: Value) -> Result<Self> {
        match value {
            Value::Stmts(stmts) => Ok(stmts.into()),
            Value::Body(body) => Ok(body.stmts().to_vec().into()),
            other => Err(other.mismatch("statements")),
        }
    }
    fn is_block() -> bool {
        true
    }
    fn block(&self) -> Vec<Arc<Statement>> {
        self.to_vec()
    }
}

/// A field that plays no role: its type tag, its spelling, its reading, and any accessor it answers.
macro_rules! plain_field {
    ($ty:ty, $tag:expr, |$this:ident, $out:ident| $spell:expr, |$value:ident| $read:expr, fn visit $($visit:tt)*) => {
        plain_field!(@visit $ty, $tag, |$this, $out| $spell, |$value| $read, { fn visit $($visit)* });
    };
    ($ty:ty, $tag:expr, |$this:ident, $out:ident| $spell:expr, |$value:ident| $read:expr $(, $extra:item)*) => {
        plain_field!(@visit $ty, $tag, |$this, $out| $spell, |$value| $read, {
            fn visit(&self, _roles: Roles, _visitor: &mut dyn Visitor) {}
        } $(, $extra)*);
    };
    (@visit $ty:ty, $tag:expr, |$this:ident, $out:ident| $spell:expr, |$value:ident| $read:expr, { $visit:item } $(, $extra:item)*) => {
        impl Field for $ty {
            fn ty() -> FieldType {
                $tag
            }
            $visit
            fn rewrite(&self, _roles: Roles, _rewriter: &mut dyn Rewriter) -> Self {
                self.clone()
            }
            fn spell(&self, $out: &mut String) -> Result<()> {
                let $this = self;
                $spell;
                Ok(())
            }
            fn from_value($value: Value) -> Result<Self> {
                $read
            }
            $( $extra )*
        }
    };
}

plain_field!(
    i64,
    FieldType::Int,
    |this, out| out.push_str(&this.to_string()),
    |value| match value {
        Value::Number(number) => number.as_int(),
        other => Err(other.mismatch("an integer")),
    }
);

plain_field!(
    bool,
    FieldType::Bool,
    |this, out| out.push_str(if *this { "true" } else { "false" }),
    |value| match value {
        Value::Bool(flag) => Ok(flag),
        other => Err(other.mismatch("true or false")),
    }
);

plain_field!(
    Float,
    FieldType::Float,
    |this, out| out.push_str(&spell_number(this.0)),
    |value| match value {
        Value::Number(number) => Ok(Float(number.as_f64())),
        other => Err(other.mismatch("a number")),
    },
    fn visit(&self, _roles: Roles, visitor: &mut dyn Visitor) {
        visitor.number(Num::Float(self.0));
    }
);

/// A closed set as a field: spelled by its text, read back from a name.
macro_rules! vocabulary_field {
    ($ty:ty, $tag:expr $(, $extra:item)*) => {
        plain_field!(
            $ty,
            $tag,
            |this, out| out.push_str(&spell_stmt_name(&this.text())?),
            |value| match value {
                Value::Name(name) => name.parse(),
                other => Err(other.mismatch(concat!("a ", stringify!($ty)))),
            }
            $(, $extra)*
        );
    };
}

vocabulary_field!(
    DType,
    FieldType::DType,
    fn as_dtype(&self) -> Option<&DType> {
        Some(self)
    }
);

plain_field!(
    ElementwiseOp,
    FieldType::ElementwiseOp,
    |this, out| out.push_str(&spell_stmt_name(this.name())?),
    |value| match value {
        Value::Name(name) => Ok(ElementwiseOp::new(&name)),
        other => Err(other.mismatch("an elementwise op")),
    },
    fn as_op(&self) -> Option<&ElementwiseOp> {
        Some(self)
    },
    fn with_op(&self, op: &ElementwiseOp) -> Self {
        op.clone()
    }
);

plain_field!(
    Shape3,
    FieldType::Tuple(vec![FieldType::Int, FieldType::Int, FieldType::Int]),
    |this, out| out.push_str(&format!("({}, {}, {})", this.0, this.1, this.2)),
    |value| {
        let items: Vec<i64> = seq_from_value(value)?;
        match items[..] {
            [a, b, c] => Ok(Shape3(a, b, c)),
            _ => Err(Error::Value("a shape is three integers".into())),
        }
    }
);

plain_field!(
    Int2,
    FieldType::Tuple(vec![FieldType::Int, FieldType::Int]),
    |this, out| out.push_str(&format!("({}, {})", this.0, this.1)),
    |value| {
        let items: Vec<i64> = seq_from_value(value)?;
        match items[..] {
            [a, b] => Ok(Int2(a, b)),
            _ => Err(Error::Value("two integers".into())),
        }
    }
);

impl Field for SubroutineRef {
    fn ty() -> FieldType {
        FieldType::Subroutine
    }
    fn visit(&self, _roles: Roles, _visitor: &mut dyn Visitor) {}
    fn rewrite(&self, _roles: Roles, _rewriter: &mut dyn Rewriter) -> Self {
        self.clone()
    }
    fn spell(&self, _out: &mut String) -> Result<()> {
        Err(Error::Value(
            "a subroutine is defined ahead of its calls, not on a header".into(),
        ))
    }
    fn from_value(value: Value) -> Result<Self> {
        Err(value.mismatch("a subroutine"))
    }
}

impl<T: Field> Field for NumOr<T> {
    fn ty() -> FieldType {
        FieldType::NumOr(Box::new(T::ty()))
    }
    fn visit(&self, roles: Roles, visitor: &mut dyn Visitor) {
        match self {
            NumOr::Value(value) => value.visit(roles, visitor),
            NumOr::Num(number) => visitor.number(*number),
        }
    }
    fn rewrite(&self, roles: Roles, rewriter: &mut dyn Rewriter) -> Self {
        match self {
            NumOr::Num(_) => self.clone(),
            NumOr::Value(value) => NumOr::Value(value.rewrite(roles, rewriter)),
        }
    }
    fn same(&self, other: &Self) -> bool {
        match (self, other) {
            (NumOr::Num(a), NumOr::Num(b)) => a == b,
            (NumOr::Value(a), NumOr::Value(b)) => a.same(b),
            _ => false,
        }
    }
    fn spell(&self, out: &mut String) -> Result<()> {
        match self {
            NumOr::Num(number) => {
                out.push_str(&spell_number(number.as_f64()));
                Ok(())
            }
            NumOr::Value(value) => value.spell(out),
        }
    }
    fn from_value(value: Value) -> Result<Self> {
        match value {
            Value::Number(number) => Ok(NumOr::Num(number)),
            other => Ok(NumOr::Value(T::from_value(other)?)),
        }
    }
}

/// The one spelling of a number a statement holds: an integer without a point, anything else as Python writes it.
pub fn spell_number(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 9007199254740992.0 {
        (value as i64).to_string()
    } else {
        float_repr(value)
    }
}
