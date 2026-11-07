//! The statement classes: one base, `Statement`, holding the shared statement and the Python objects its fields hand
//! back, and one class per kind generated from the crate's declaration. A kind check is a native class check; a
//! field read hands back a held object.

use std::collections::HashMap;
use std::sync::Arc;

use emmy_ir::Name;
use emmy_ir::body::Body;
use emmy_ir::expr::Expression;
use emmy_ir::kinds::field::{Rewriter, Visitor};
use emmy_ir::kinds::{FieldInfo, Statement};
use emmy_ir::protocol::{Rewrite, finished};
use emmy_ir::text::{parse_statement, statement_text};
use pyo3::exceptions::{PyAttributeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PyString, PyTuple, PyType};

use crate::body::PyBody;
use crate::convert::{
    IntoPyErr, PyField, Slots, axis_from_py, axis_to_py, cached, expression_from_py, slots,
    statement_from_py, str_tuple,
};
use crate::expression::{PyExpression, mapping_from_py, renames_from_py};

/// The base class of every statement kind.
#[pyclass(subclass, frozen, module = "emmy.compiler.ir.stmt", name = "Statement")]
pub struct PyStatement {
    pub stmt: Arc<Statement>,
    slots: Slots,
}

impl PyStatement {
    pub fn new(stmt: Arc<Statement>, held: Vec<Option<Py<PyAny>>>) -> PyStatement {
        let slots = slots(stmt.fields().len());
        for (slot, value) in slots.iter().zip(held) {
            if let Some(value) = value {
                let _ = slot.set(value);
            }
        }
        PyStatement { stmt, slots }
    }

    /// The Python object of field `index`, made on first read.
    pub fn field<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.slots[index], || {
            field_to_py(&self.stmt, index, py)
        })
    }

    fn field_named<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        match self.stmt.fields().iter().position(|f| f.name == name) {
            Some(index) => self.field(py, index),
            None => Err(PyAttributeError::new_err(format!(
                "{} has no field {name}",
                self.stmt.kind_name()
            ))),
        }
    }

    /// The held object of field `index`, when a read or the constructor made one.
    pub fn held(&self, py: Python<'_>, index: usize) -> Option<Py<PyAny>> {
        self.slots[index].get().map(|value| value.clone_ref(py))
    }
}

/// The index of `name` in `names`, at compile time.
pub const fn position(names: &[&str], name: &str) -> usize {
    let mut index = 0;
    while index < names.len() {
        if str_eq(names[index], name) {
            return index;
        }
        index += 1;
    }
    usize::MAX
}

const fn str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

fn names_tuple<'py>(py: Python<'py>, names: &[Name]) -> PyResult<Bound<'py, PyTuple>> {
    str_tuple(py, names)
}

fn sigma_from_py(value: &Bound<'_, PyAny>) -> PyResult<HashMap<Name, Expression>> {
    if value.cast::<PyDict>().is_ok() {
        return mapping_from_py(value);
    }
    mapping_from_py(&value.getattr("mapping")?)
}

/// Wraps the Python object of a statement, the same for a statement reached again through its body.
pub fn wrap_statement(py: Python<'_>, stmt: &Arc<Statement>) -> PyResult<Py<PyAny>> {
    wrap_kind(py, stmt, Vec::new())
}

fn same_or_wrap<'py>(
    slf: &Bound<'py, PyStatement>,
    rebuilt: Arc<Statement>,
) -> PyResult<Bound<'py, PyAny>> {
    if Arc::ptr_eq(&rebuilt, &slf.get().stmt) {
        return Ok(slf.clone().into_any());
    }
    Ok(wrap_statement(slf.py(), &rebuilt)?.into_bound(slf.py()))
}

#[pymethods]
impl PyStatement {
    /// SSA names this statement reads.
    fn deps<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        names_tuple(py, &self.stmt.deps())
    }

    /// SSA names this statement defines.
    fn defines<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        names_tuple(py, &self.stmt.defines())
    }

    fn external_reads<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        names_tuple(py, &self.stmt.external_reads())
    }

    fn external_writes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        names_tuple(py, &self.stmt.external_writes())
    }

    fn local_decls<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        names_tuple(py, &self.stmt.local_decls())
    }

    /// This statement's own expressions, in field order.
    fn expressions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let items: PyResult<Vec<Bound<'py, PyAny>>> = self
            .stmt
            .expressions()
            .iter()
            .map(|e| PyExpression::wrap(py, e))
            .collect();
        PyTuple::new(py, items?)
    }

    /// The axes this statement binds for its child bodies, in field order.
    fn bound_axes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let items: PyResult<Vec<Bound<'py, PyAny>>> = self
            .stmt
            .bound_axes()
            .iter()
            .map(|axis| axis_to_py(py, axis))
            .collect();
        PyTuple::new(py, items?)
    }

    /// The names this statement brings into scope for its child bodies.
    fn binds_axes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
        PyFrozenSet::new(
            py,
            self.stmt
                .binds_axes()
                .iter()
                .map(|name| PyString::new(py, name)),
        )
    }

    /// The child bodies, in field order: the same objects the fields hand back.
    fn nested<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let mut items = Vec::new();
        for (index, field) in self.stmt.fields().iter().enumerate() {
            if field.roles & emmy_ir::kinds::BODY != 0 {
                items.push(self.field(py, index)?);
            }
        }
        PyTuple::new(py, items)
    }

    /// This statement over `bodies`, positionally matching `nested`.
    fn with_bodies<'py>(
        slf: &Bound<'py, Self>,
        bodies: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let bodies: PyResult<Vec<Body>> = bodies
            .try_iter()?
            .map(|item| Body::from_py(&item?).map(|(body, _)| body))
            .collect();
        let rebuilt = Statement::with_bodies(&slf.get().stmt, bodies?).py()?;
        same_or_wrap(slf, rebuilt)
    }

    /// This statement with its expressions replaced positionally, as `expressions` lists them.
    fn _with_expressions<'py>(
        slf: &Bound<'py, Self>,
        expressions: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let items: PyResult<Vec<Expression>> = expressions
            .try_iter()?
            .map(|item| expression_from_py(&item?))
            .collect();
        let mut items = items?.into_iter();
        let rebuilt = Statement::map_expressions(&slf.get().stmt, &mut |e| {
            items.next().unwrap_or_else(|| e.clone())
        });
        if items.next().is_some() {
            return Err(PyValueError::new_err(
                "more expressions than the statement holds",
            ));
        }
        same_or_wrap(slf, rebuilt)
    }

    /// This statement with every buffer renamed through `mapping`.
    fn rename_buffers<'py>(
        slf: &Bound<'py, Self>,
        mapping: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mapping = renames_from_py(mapping)?;
        same_or_wrap(slf, Statement::rename_buffers(&slf.get().stmt, &mapping))
    }

    /// α-rename this subtree, binders and uses, through `mapping`.
    fn _rename<'py>(
        slf: &Bound<'py, Self>,
        mapping: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mapping = renames_from_py(mapping)?;
        same_or_wrap(slf, Statement::rename(&slf.get().stmt, &mapping))
    }

    /// β-substitute free coordinates by expressions (a mapping, or a `Sigma`), stopping at a re-binding scope.
    fn substitute<'py>(
        slf: &Bound<'py, Self>,
        coords: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sigma = sigma_from_py(coords)?;
        same_or_wrap(slf, Statement::substitute(&slf.get().stmt, &sigma))
    }

    /// Rename, substitute and map axes in one rewrite: `names` maps SSA and axis names, `sigma` substitutes
    /// expressions, `axes` pairs each bound axis of the subtree with its replacement.
    #[pyo3(signature = (names = None, sigma = None, axes = None))]
    fn _rewrite<'py>(
        slf: &Bound<'py, Self>,
        names: Option<&Bound<'py, PyAny>>,
        sigma: Option<&Bound<'py, PyAny>>,
        axes: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let names = names
            .filter(|n| !n.is_none())
            .map(renames_from_py)
            .transpose()?
            .unwrap_or_default();
        let sigma = sigma
            .filter(|s| !s.is_none())
            .map(sigma_from_py)
            .transpose()?
            .unwrap_or_default();
        let mut pairs: Vec<(
            Arc<emmy_ir::kinds::axis::Axis>,
            Arc<emmy_ir::kinds::axis::Axis>,
        )> = Vec::new();
        if let Some(axes) = axes.filter(|a| !a.is_none()) {
            for pair in axes.try_iter()? {
                let (old, new): (Bound<'py, PyAny>, Bound<'py, PyAny>) = pair?.extract()?;
                pairs.push((axis_from_py(&old, 0)?, axis_from_py(&new, 0)?));
            }
        }
        let rename = |name: &str| names.get(name).cloned();
        let axis_fn = |axis: &Arc<emmy_ir::kinds::axis::Axis>| {
            pairs
                .iter()
                .find(|(old, _)| **old == **axis && old.window == axis.window)
                .map_or_else(|| axis.clone(), |(_, new)| new.clone())
        };
        let rewrite = Rewrite {
            rename: &rename,
            sigma: &sigma,
            axis_fn: if pairs.is_empty() {
                None
            } else {
                Some(&axis_fn)
            },
        };
        same_or_wrap(slf, Statement::rewrite(&slf.get().stmt, &rewrite))
    }

    /// Every SSA and axis name this subtree spells: what a rename by a function is asked about.
    fn _names<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        names_tuple(py, &self.stmt.names())
    }

    /// True iff executing this statement writes a buffer, directly or inside a child body.
    #[getter]
    fn has_side_effects(&self) -> bool {
        self.stmt.has_side_effects()
    }

    /// The structural form: the kind's name over its fields' forms, trailing defaults left out.
    fn structural_form<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let fields = self.stmt.fields();
        let mut parts: Vec<Bound<'py, PyAny>> =
            vec![PyString::new(py, self.stmt.kind_name()).into_any()];
        let header: HashMap<&str, String> = self.stmt.header_fields().py()?.into_iter().collect();
        let blocks: HashMap<&str, ()> = self
            .stmt
            .block_fields()
            .into_iter()
            .map(|(name, _)| (name, ()))
            .collect();
        // A field at its default says nothing about what the value is; trailing ones are left out.
        let mut kept = fields.len();
        while kept > 0 {
            let field = &fields[kept - 1];
            if !field.has_default
                || header.contains_key(field.name)
                || blocks.contains_key(field.name)
            {
                break;
            }
            kept -= 1;
        }
        for index in 0..kept {
            parts.push(self.field(py, index)?);
        }
        PyTuple::new(py, parts)
    }

    fn __str__(&self) -> PyResult<String> {
        statement_text(&self.stmt).py()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.cast::<PyStatement>().is_ok_and(|other| {
            Arc::ptr_eq(&other.get().stmt, &self.stmt) || *other.get().stmt == *self.stmt
        })
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.stmt.hash(&mut hasher);
        hasher.finish()
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, (String,))> {
        let module = slf.py().import("emmy.emmy_ir")?;
        Ok((
            module.getattr("_statement_from_text")?,
            (statement_text(&slf.get().stmt).py()?,),
        ))
    }

    /// The field `name` holds, by name.
    fn _field<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        self.field_named(py, name)
    }
}

/// A statement's text read back.
#[pyfunction]
pub fn _statement_from_text(py: Python<'_>, text: &str) -> PyResult<Py<PyAny>> {
    wrap_statement(py, &parse_statement(text).py()?)
}

fn too_many(kind: &str, count: usize) -> PyErr {
    PyTypeError::new_err(format!(
        "{kind}() takes at most {count} positional arguments"
    ))
}

fn missing(kind: &str, field: &str) -> PyErr {
    PyTypeError::new_err(format!("{kind}() missing a required argument: '{field}'"))
}

fn unexpected(kind: &str, key: &str) -> PyErr {
    PyTypeError::new_err(format!(
        "{kind}() got an unexpected keyword argument '{key}'"
    ))
}

/// The arguments a constructor received, matched to the kind's fields.
struct Arguments<'a, 'py> {
    kind: &'static str,
    fields: &'static [FieldInfo],
    args: &'a Bound<'py, PyTuple>,
    kwargs: Option<&'a Bound<'py, PyDict>>,
    positional: usize,
}

impl<'py> Arguments<'_, 'py> {
    /// The argument for `field`, by keyword or by position, `None` when it was not given.
    fn take(&mut self, field: &str, roles: u32) -> PyResult<Option<Bound<'py, PyAny>>> {
        if let Some(kwargs) = self.kwargs
            && let Some(value) = kwargs.get_item(field)?
        {
            return Ok(Some(value));
        }
        if roles & emmy_ir::kinds::KW_ONLY != 0 {
            return Ok(None);
        }
        if self.positional < self.args.len() {
            let value = self.args.get_item(self.positional)?;
            self.positional += 1;
            return Ok(Some(value));
        }
        Ok(None)
    }

    fn finish(&self) -> PyResult<()> {
        let allowed = self
            .fields
            .iter()
            .filter(|f| f.roles & emmy_ir::kinds::KW_ONLY == 0)
            .count();
        if self.args.len() > allowed {
            return Err(too_many(self.kind, allowed));
        }
        if let Some(kwargs) = self.kwargs {
            for key in kwargs.keys() {
                let key: String = key.extract()?;
                if !self.fields.iter().any(|f| f.name == key) {
                    return Err(unexpected(self.kind, &key));
                }
            }
        }
        Ok(())
    }
}

macro_rules! define_py_kinds {
    (
        kinds { $( $(#[doc = $doc:literal])* $layer:ident $Kind:ident {$($flag:tt)|*} { $( $field:ident : $ty:ty [$($role:ident)|*] $(= $default:expr)? ,)* } )* }
        records { $( $(#[doc = $rdoc:literal])* $Record:ident { $( $rfield:ident : $rty:ty [$($rrole:ident)|*] $(= $rdefault:expr)? ,)* } )* }
    ) => {
        $(
            define_py_kinds!(@class $layer $Kind $(#[doc = $doc])*);

            #[allow(non_upper_case_globals, unused_variables, unused_mut, clippy::too_many_arguments)]
            impl $Kind {
                const FIELD_NAMES: &'static [&'static str] = &[ $( stringify!($field), )* ];

                /// The statement the constructor arguments build, and the objects it keeps beside its fields.
                fn build(py: Python<'_>, args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<(Statement, Vec<Option<Py<PyAny>>>)> {
                    use emmy_ir::kinds as k;
                    let mut given = Arguments { kind: k::$Kind::NAME, fields: k::$Kind::FIELDS, args, kwargs, positional: 0 };
                    let mut held: Vec<Option<Py<PyAny>>> = Vec::with_capacity(k::$Kind::FIELDS.len());
                    $(
                        let $field: $ty = match given.take(stringify!($field), 0 $(| k::$role)*)? {
                            Some(value) => {
                                let (value, kept) = <$ty as PyField>::from_py(&value)
                                    .map_err(|err| PyTypeError::new_err(format!("{}.{}: {}", stringify!($Kind), stringify!($field), err)))?;
                                held.push(kept);
                                value
                            }
                            None => {
                                held.push(None);
                                define_py_kinds!(@default $Kind $field $(= $default)?)
                            }
                        };
                    )*
                    given.finish()?;
                    let stmt = finished(Statement::$Kind(k::$Kind { $( $field, )* })).py()?;
                    Ok((stmt, held))
                }

                /// This statement with the named fields changed.
                fn rebuild(base: &PyStatement, py: Python<'_>, changes: &Bound<'_, PyDict>) -> PyResult<(Statement, Vec<Option<Py<PyAny>>>)> {
                    use emmy_ir::kinds as k;
                    let Statement::$Kind(current) = &*base.stmt else { return Err(PyTypeError::new_err("not a ".to_owned() + stringify!($Kind))) };
                    let mut held: Vec<Option<Py<PyAny>>> = Vec::with_capacity(k::$Kind::FIELDS.len());
                    #[allow(unused_assignments)]
                    let mut index = 0usize;
                    $(
                        let $field: $ty = match changes.get_item(stringify!($field))? {
                            Some(value) => {
                                let (value, kept) = <$ty as PyField>::from_py(&value)
                                    .map_err(|err| PyTypeError::new_err(format!("{}.{}: {}", stringify!($Kind), stringify!($field), err)))?;
                                held.push(kept);
                                value
                            }
                            None => {
                                held.push(base.held(py, index));
                                current.$field.clone()
                            }
                        };
                        index += 1;
                    )*
                    for key in changes.keys() {
                        let key: String = key.extract()?;
                        if !k::$Kind::FIELDS.iter().any(|f| f.name == key) {
                            return Err(unexpected(stringify!($Kind), &key));
                        }
                    }
                    let stmt = finished(Statement::$Kind(k::$Kind { $( $field, )* })).py()?;
                    Ok((stmt, held))
                }
            }

            #[pymethods]
            impl $Kind {
                #[new]
                #[pyo3(signature = (*args, **kwargs))]
                fn new(py: Python<'_>, args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<PyClassInitializer<Self>> {
                    let (stmt, held) = Self::build(py, args, kwargs)?;
                    Ok(PyClassInitializer::from(PyStatement::new(Arc::new(stmt), held)).add_subclass($Kind))
                }

                /// A copy of this statement with the named fields changed: the one way a statement is edited.
                #[pyo3(signature = (**changes))]
                fn replace<'py>(slf: &Bound<'py, Self>, changes: Option<&Bound<'py, PyDict>>) -> PyResult<Bound<'py, PyAny>> {
                    let Some(changes) = changes.filter(|c| !c.is_empty()) else { return Ok(slf.clone().into_any()) };
                    let base = slf.as_super().get();
                    let (stmt, held) = Self::rebuild(base, slf.py(), changes)?;
                    Ok(Bound::new(slf.py(), PyClassInitializer::from(PyStatement::new(Arc::new(stmt), held)).add_subclass($Kind))?.into_any())
                }

                /// The names of this kind's fields, in declaration order.
                #[classattr]
                fn _fields<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
                    PyTuple::new(py, emmy_ir::kinds::$Kind::FIELDS.iter().map(|f| PyString::new(py, f.name)))
                }

                /// A pure value binding: a legal member of a stored term.
                #[classattr]
                fn pure() -> bool {
                    emmy_ir::kinds::$Kind::FLAGS & emmy_ir::kinds::PURE != 0
                }

                /// `deps` already reports the whole subtree's reads.
                #[classattr]
                fn deps_deep() -> bool {
                    emmy_ir::kinds::$Kind::FLAGS & emmy_ir::kinds::DEPS_DEEP != 0
                }

                $(
                    #[getter]
                    fn $field<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
                        const INDEX: usize = position($Kind::FIELD_NAMES, stringify!($field));
                        slf.as_super().get().field(slf.py(), INDEX)
                    }
                )*
            }
        )*

        $(
            $(#[doc = $rdoc])*
            #[pyclass(frozen, module = "emmy.compiler.ir.stmt")]
            pub struct $Record {
                pub record: emmy_ir::kinds::$Record,
                slots: Slots,
            }

            impl $Record {
                const FIELD_NAMES: &'static [&'static str] = &[ $( stringify!($rfield), )* ];

                pub fn new_from(record: emmy_ir::kinds::$Record) -> $Record {
                    $Record { slots: slots(emmy_ir::kinds::$Record::FIELDS.len()), record }
                }
            }

            #[allow(non_upper_case_globals, unused_variables, unused_mut)]
            #[pymethods]
            impl $Record {
                #[new]
                #[pyo3(signature = (*args, **kwargs))]
                fn new(py: Python<'_>, args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<$Record> {
                    use emmy_ir::kinds as k;
                    let mut given = Arguments { kind: k::$Record::NAME, fields: k::$Record::FIELDS, args, kwargs, positional: 0 };
                    let mut held: Vec<Option<Py<PyAny>>> = Vec::new();
                    $(
                        let $rfield: $rty = match given.take(stringify!($rfield), 0 $(| k::$rrole)*)? {
                            Some(value) => {
                                let (value, kept) = <$rty as PyField>::from_py(&value)
                                    .map_err(|err| PyTypeError::new_err(format!("{}.{}: {}", stringify!($Record), stringify!($rfield), err)))?;
                                held.push(kept);
                                value
                            }
                            None => {
                                held.push(None);
                                define_py_kinds!(@default $Record $rfield $(= $rdefault)?)
                            }
                        };
                    )*
                    given.finish()?;
                    let out = $Record::new_from(k::$Record { $( $rfield, )* });
                    for (slot, value) in out.slots.iter().zip(held) {
                        if let Some(value) = value {
                            let _ = slot.set(value);
                        }
                    }
                    Ok(out)
                }

                #[pyo3(signature = (**changes))]
                fn replace(slf: &Bound<'_, Self>, changes: Option<&Bound<'_, PyDict>>) -> PyResult<$Record> {
                    use emmy_ir::kinds as k;
                    let current = &slf.get().record;
                    let Some(changes) = changes else { return Ok($Record::new_from(current.clone())) };
                    $(
                        let $rfield: $rty = match changes.get_item(stringify!($rfield))? {
                            Some(value) => <$rty as PyField>::from_py(&value)
                                .map_err(|err| PyTypeError::new_err(format!("{}.{}: {}", stringify!($Record), stringify!($rfield), err)))?.0,
                            None => current.$rfield.clone(),
                        };
                    )*
                    for key in changes.keys() {
                        let key: String = key.extract()?;
                        if !k::$Record::FIELDS.iter().any(|f| f.name == key) {
                            return Err(unexpected(stringify!($Record), &key));
                        }
                    }
                    Ok($Record::new_from(k::$Record { $( $rfield, )* }))
                }

                #[classattr]
                fn _fields<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
                    PyTuple::new(py, emmy_ir::kinds::$Record::FIELDS.iter().map(|f| PyString::new(py, f.name)))
                }

                $(
                    #[getter]
                    fn $rfield<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
                        const INDEX: usize = position($Record::FIELD_NAMES, stringify!($rfield));
                        let this = slf.get();
                        cached(slf.py(), &this.slots[INDEX], || <$rty as PyField>::to_py(&this.record.$rfield, slf.py()))
                    }
                )*

                fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
                    other.cast::<$Record>().is_ok_and(|other| other.get().record == self.record)
                }

                fn __hash__(&self) -> u64 {
                    use std::hash::{Hash, Hasher};
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    self.record.hash(&mut hasher);
                    hasher.finish()
                }

                fn structural_form<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
                    let mut parts: Vec<Bound<'py, PyAny>> = vec![PyString::new(slf.py(), stringify!($Record)).into_any()];
                    $( parts.push(Self::$rfield(slf)?); )*
                    PyTuple::new(slf.py(), parts)
                }
            }
        )*

        /// The Python object of field `index` of `stmt`.
        pub fn field_to_py(stmt: &Arc<Statement>, index: usize, py: Python<'_>) -> PyResult<Py<PyAny>> {
            match &**stmt {
                $( Statement::$Kind(k) => {
                    #[allow(unused_mut, unused_variables, unused_assignments)]
                    let mut position = 0usize;
                    $(
                        if position == index {
                            return <$ty as PyField>::to_py(&k.$field, py);
                        }
                        position += 1;
                    )*
                    Err(PyAttributeError::new_err(format!("{} has no field {index}", stringify!($Kind))))
                } )*
            }
        }

        /// The Python object of `stmt`, an instance of its kind's class.
        pub fn wrap_kind(py: Python<'_>, stmt: &Arc<Statement>, held: Vec<Option<Py<PyAny>>>) -> PyResult<Py<PyAny>> {
            let base = PyClassInitializer::from(PyStatement::new(stmt.clone(), held));
            Ok(match &**stmt {
                $( Statement::$Kind(_) => Py::new(py, base.add_subclass($Kind))?.into_any(), )*
            })
        }

        /// The kind class of each kind name, for `iter_of_type`.
        pub fn kind_of(cls: &Bound<'_, PyType>) -> Option<&'static str> {
            $( if cls.is(&cls.py().get_type::<$Kind>()) { return Some(stringify!($Kind)); } )*
            None
        }

        pub fn register_kinds(m: &Bound<'_, PyModule>) -> PyResult<()> {
            m.add_class::<PyStatement>()?;
            $( m.add_class::<$Kind>()?; )*
            $( m.add_class::<$Record>()?; )*
            Ok(())
        }
    };
    (@class shared $Kind:ident $(#[doc = $doc:literal])*) => {
        $(#[doc = $doc])*
        #[pyclass(extends = PyStatement, frozen, module = "emmy.compiler.ir.stmt")]
        pub struct $Kind;
    };
    (@class kernel $Kind:ident $(#[doc = $doc:literal])*) => {
        $(#[doc = $doc])*
        #[pyclass(extends = PyStatement, frozen, module = "emmy.compiler.ir.kernel.ir")]
        pub struct $Kind;
    };
    (@default $Kind:ident $field:ident = $default:expr) => { $default };
    (@default $Kind:ident $field:ident) => { return Err(missing(stringify!($Kind), stringify!($field))) };
}

#[allow(
    unused_imports,
    unused_variables,
    unused_assignments,
    unused_mut,
    dead_code
)]
mod generated {
    use super::*;
    use emmy_ir::Name;
    use emmy_ir::body::Body;
    use emmy_ir::kinds::elementwise::Op;
    use emmy_ir::kinds::field::{
        Axes, AxisRef, Branches, DType, Exprs, Float, Floats, FragmentArgs, Int2, Ints, Names,
        NumOr, Ops, Shape3, Stmts, SubroutineRef,
    };

    emmy_ir::with_kinds!(define_py_kinds);
}

pub use generated::*;

// --- what the shared kinds know beyond their fields -----------------------------------------------

fn load_of(slf: &Bound<'_, Load>) -> PyResult<emmy_ir::kinds::Load> {
    match &*slf.as_super().get().stmt {
        Statement::Load(load) => Ok(load.clone()),
        _ => Err(PyTypeError::new_err("not a Load")),
    }
}

#[pymethods]
impl Load {
    /// Scalar-only shorthand for `names[0]`.
    #[getter]
    fn name<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let load = load_of(slf)?;
        if load.names.len() != 1 {
            return Err(pyo3::exceptions::PyAssertionError::new_err(format!(
                "Load has {} names — use .names for vector Loads",
                load.names.len()
            )));
        }
        Ok(PyString::new(slf.py(), &load.names[0]).into_any())
    }

    #[getter]
    fn width(slf: &Bound<'_, Self>) -> PyResult<usize> {
        Ok(load_of(slf)?.names.len())
    }

    #[getter]
    fn is_vector(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(load_of(slf)?.names.len() > 1)
    }

    #[getter]
    fn is_scalar(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(load_of(slf)?.names.len() == 1)
    }
}

fn write_of(slf: &Bound<'_, Write>) -> PyResult<emmy_ir::kinds::Write> {
    match &*slf.as_super().get().stmt {
        Statement::Write(write) => Ok(write.clone()),
        _ => Err(PyTypeError::new_err("not a Write")),
    }
}

#[pymethods]
impl Write {
    /// Scalar-only shorthand for `values[0]`.
    #[getter]
    fn value<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let write = write_of(slf)?;
        if write.values.len() != 1 {
            return Err(pyo3::exceptions::PyAssertionError::new_err(format!(
                "Write has {} values — use .values for vector Writes",
                write.values.len()
            )));
        }
        Ok(PyString::new(slf.py(), &write.values[0]).into_any())
    }

    #[getter]
    fn width(slf: &Bound<'_, Self>) -> PyResult<usize> {
        Ok(write_of(slf)?.values.len())
    }

    #[getter]
    fn is_vector(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(write_of(slf)?.values.len() > 1)
    }

    #[getter]
    fn is_scalar(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(write_of(slf)?.values.len() == 1)
    }
}

#[pymethods]
impl Affine {
    /// `(coefficient, arg)` per term.
    #[getter]
    fn terms<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let Statement::Affine(affine) = &*slf.as_super().get().stmt else {
            return Err(PyTypeError::new_err("not an Affine"));
        };
        let items: PyResult<Vec<Bound<'py, PyTuple>>> = affine
            .terms()
            .map(|(c, arg)| {
                PyTuple::new(
                    slf.py(),
                    [
                        c.into_pyobject(slf.py())?.into_any(),
                        PyString::new(slf.py(), arg).into_any(),
                    ],
                )
            })
            .collect();
        PyTuple::new(slf.py(), items?)
    }

    /// The same value as binary `Assign`s over `Let`-bound literals, the last one named `name`.
    fn as_assigns<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyList>> {
        let Statement::Affine(affine) = &*slf.as_super().get().stmt else {
            return Err(PyTypeError::new_err("not an Affine"));
        };
        let items: PyResult<Vec<Py<PyAny>>> = affine
            .as_assigns()
            .iter()
            .map(|stmt| wrap_statement(slf.py(), stmt))
            .collect();
        pyo3::types::PyList::new(slf.py(), items?)
    }
}

#[pymethods]
impl Accum {
    /// The identity value of the accumulator.
    #[getter]
    fn init<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let Statement::Accum(accum) = &*slf.as_super().get().stmt else {
            return Err(PyTypeError::new_err("not an Accum"));
        };
        PyExpression::wrap(slf.py(), &accum.init())
    }

    #[getter]
    fn associative(slf: &Bound<'_, Self>) -> PyResult<bool> {
        let Statement::Accum(accum) = &*slf.as_super().get().stmt else {
            return Err(PyTypeError::new_err("not an Accum"));
        };
        Ok(accum.op.associative())
    }

    #[getter]
    fn commutative(slf: &Bound<'_, Self>) -> PyResult<bool> {
        let Statement::Accum(accum) = &*slf.as_super().get().stmt else {
            return Err(PyTypeError::new_err("not an Accum"));
        };
        Ok(accum.op.commutative)
    }

    #[getter]
    fn has_identity(slf: &Bound<'_, Self>) -> PyResult<bool> {
        let Statement::Accum(accum) = &*slf.as_super().get().stmt else {
            return Err(PyTypeError::new_err("not an Accum"));
        };
        Ok(accum.op.identity.is_some())
    }
}

#[pymethods]
impl Carry {
    /// The axes of the cell this statement defines.
    #[getter]
    fn cells<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let Statement::Carry(carry) = &*slf.as_super().get().stmt else {
            return Err(PyTypeError::new_err("not a Carry"));
        };
        PyTuple::new(
            slf.py(),
            carry
                .cells()
                .iter()
                .map(|name| PyString::new(slf.py(), name)),
        )
    }
}

fn carries_dict<'py>(
    py: Python<'py>,
    cells: Vec<(Name, Vec<Option<Arc<emmy_ir::kinds::axis::Axis>>>)>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for (name, shape) in cells {
        let entries: PyResult<Vec<Bound<'py, PyAny>>> = shape
            .iter()
            .map(|cell| match cell {
                Some(axis) => axis_to_py(py, axis),
                None => Ok(1i64.into_pyobject(py)?.into_any()),
            })
            .collect();
        out.set_item(&*name, PyTuple::new(py, entries?)?)?;
    }
    Ok(out)
}

#[pymethods]
impl Loop {
    /// The carried states this loop carries, each with its cell shape.
    #[getter]
    fn carries<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDict>> {
        let Statement::Loop(l) = &*slf.as_super().get().stmt else {
            return Err(PyTypeError::new_err("not a Loop"));
        };
        carries_dict(slf.py(), l.carries())
    }

    /// A loop is a reduce loop iff it carries something.
    #[getter]
    fn is_reduce(slf: &Bound<'_, Self>) -> PyResult<bool> {
        let Statement::Loop(l) = &*slf.as_super().get().stmt else {
            return Err(PyTypeError::new_err("not a Loop"));
        };
        Ok(l.is_reduce())
    }
}

#[pymethods]
impl StridedLoop {
    #[getter]
    fn is_reduce(slf: &Bound<'_, Self>) -> PyResult<bool> {
        let Statement::StridedLoop(l) = &*slf.as_super().get().stmt else {
            return Err(PyTypeError::new_err("not a StridedLoop"));
        };
        Ok(l.is_reduce())
    }
}

/// The names of every kind, for the module's listing.
pub fn kind_names() -> &'static [&'static str] {
    emmy_ir::kinds::KIND_NAMES
}

/// Every statement a Python iterable holds.
pub fn statements_from_py(value: &Bound<'_, PyAny>) -> PyResult<Vec<Arc<Statement>>> {
    if let Ok(body) = value.cast::<PyBody>() {
        return Ok(body.get().body.stmts().to_vec());
    }
    value
        .try_iter()?
        .map(|item| statement_from_py(&item?))
        .collect()
}

/// Visitors and rewriters are the crate's; this module only wraps.
#[allow(dead_code)]
fn _traits_in_scope(_v: &dyn Visitor, _r: &dyn Rewriter) {}
