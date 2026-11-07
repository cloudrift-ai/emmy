//! The statement classes: one base, `Statement`, holding the shared statement and the Python objects its fields hand
//! back, and one class per statement type generated from the crate's declaration. An `isinstance` check on a
//! statement type is a native class check; a field read hands back a held object.

use std::collections::HashMap;
use std::sync::Arc;

use emmy_ir::Name;
use emmy_ir::body::Body;
use emmy_ir::expr::Expression;
use emmy_ir::protocol::{Rewrite, finished};
use emmy_ir::statement::{FieldInfo, Statement};
use emmy_ir::text::{parse_statement, statement_text};
use pyo3::exceptions::{PyAttributeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PyString, PyTuple, PyType};

use crate::body::PyBody;
use crate::convert::{
    IntoPyErr, PyField, STATEMENT_FROM_TEXT, Slots, axis_from_py, axis_to_py, cached,
    expression_from_py, located, slots, statement_from_py, str_tuple,
};
use crate::expression::{PyExpression, mapping_from_py, renames_from_py};

/// The base class of every statement type.
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
                self.stmt.ty()
            ))),
        }
    }

    /// The held object of field `index`, when a read or the constructor made one.
    pub fn held(&self, py: Python<'_>, index: usize) -> Option<Py<PyAny>> {
        self.slots[index].get().map(|value| value.clone_ref(py))
    }
}

/// The index of `name` in `names`, at compile time.
pub const fn position(fields: &[FieldInfo], name: &str) -> usize {
    let mut index = 0;
    while index < fields.len() {
        if str_eq(fields[index].name, name) {
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
            if field.roles.contains_any(emmy_ir::statement::BODY) {
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
        let items = items?;
        if items.len() > slf.get().stmt.expressions().len() {
            return Err(PyValueError::new_err(
                "more expressions than the statement holds",
            ));
        }
        same_or_wrap(slf, Statement::with_expressions(&slf.get().stmt, items))
    }

    /// This statement with `function` applied to each of its own expressions, in the order `expressions` lists
    /// them; the statement itself when no expression changes.
    fn map_expressions<'py>(
        slf: &Bound<'py, Self>,
        function: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let rebuilt = Statement::try_map_expressions(&slf.get().stmt, &mut |e| {
            expression_from_py(&function.call1((PyExpression::wrap(py, e)?,))?)
        })?;
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
        let mut pairs: Vec<(Arc<emmy_ir::axis::Axis>, Arc<emmy_ir::axis::Axis>)> = Vec::new();
        if let Some(axes) = axes.filter(|a| !a.is_none()) {
            for pair in axes.try_iter()? {
                let (old, new): (Bound<'py, PyAny>, Bound<'py, PyAny>) = pair?.extract()?;
                pairs.push((axis_from_py(&old, 0)?, axis_from_py(&new, 0)?));
            }
        }
        let rename = |name: &str| names.get(name).cloned();
        let axis_fn = |axis: &Arc<emmy_ir::axis::Axis>| {
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

    /// The structural form: the statement type's name over its fields' forms, trailing defaults left out.
    fn structural_form<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let fields = self.stmt.fields();
        let mut parts: Vec<Bound<'py, PyAny>> =
            vec![PyString::new(py, self.stmt.ty().name()).into_any()];
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
        Ok((
            STATEMENT_FROM_TEXT.get(slf.py())?.clone(),
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

fn too_many(type_name: &str, count: usize) -> PyErr {
    PyTypeError::new_err(format!(
        "{type_name}() takes at most {count} positional arguments"
    ))
}

fn missing(type_name: &str, field: &str) -> PyErr {
    PyTypeError::new_err(format!(
        "{type_name}() missing a required argument: '{field}'"
    ))
}

fn unexpected(type_name: &str, key: &str) -> PyErr {
    PyTypeError::new_err(format!(
        "{type_name}() got an unexpected keyword argument '{key}'"
    ))
}

/// The arguments a constructor received, matched to the statement type's fields.
struct Arguments<'a, 'py> {
    type_name: &'static str,
    fields: &'static [FieldInfo],
    args: &'a Bound<'py, PyTuple>,
    kwargs: Option<&'a Bound<'py, PyDict>>,
    positional: usize,
}

impl<'py> Arguments<'_, 'py> {
    /// The argument for `field`, by keyword or by position, `None` when it was not given.
    fn take(&mut self, field: &str, kw_only: bool) -> PyResult<Option<Bound<'py, PyAny>>> {
        if let Some(kwargs) = self.kwargs
            && let Some(value) = kwargs.get_item(field)?
        {
            return Ok(Some(value));
        }
        if kw_only {
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
        let allowed = self.fields.iter().filter(|f| !f.kw_only).count();
        if self.args.len() > allowed {
            return Err(too_many(self.type_name, allowed));
        }
        if let Some(kwargs) = self.kwargs {
            for key in kwargs.keys() {
                let key: String = key.extract()?;
                if !self.fields.iter().any(|f| f.name == key) {
                    return Err(unexpected(self.type_name, &key));
                }
            }
        }
        Ok(())
    }
}

macro_rules! define_py_statement_types {
    (
        statement_types { $( $(#[doc = $doc:literal])* $layer:ident $StatementType:ident {$($flag:tt)|*} { $( $field:ident : $ty:ty [$($role:ident)|*] $(= $default:expr)? ,)* } )* }
        records { $( $(#[doc = $rdoc:literal])* $rlayer:ident $Record:ident { $( $rfield:ident : $rty:ty [$($rrole:ident)|*] $(= $rdefault:expr)? ,)* } )* }
    ) => {
        $(
            define_py_statement_types!(@class $layer $StatementType $(#[doc = $doc])*);

            #[allow(non_upper_case_globals, unused_variables, unused_mut)]
            impl $StatementType {
                /// The statement the constructor arguments build, and the objects it keeps beside its fields.
                fn build(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<(Statement, Held)> {
                    let mut held = Held::new();
                    let stmt = define_py_statement_types!(@given $StatementType args kwargs held { $( $field : $ty [$($role)|*] $(= $default)? ,)* });
                    Ok((finished(Statement::$StatementType(stmt)).py()?, held))
                }

                /// This statement with the named fields changed.
                fn rebuild(base: &PyStatement, py: Python<'_>, changes: &Bound<'_, PyDict>) -> PyResult<(Statement, Held)> {
                    let Statement::$StatementType(current) = &*base.stmt else { return Err(PyTypeError::new_err("not a ".to_owned() + stringify!($StatementType))) };
                    let mut held = Held::new();
                    let stmt = define_py_statement_types!(@changed $StatementType current changes held (|index| base.held(py, index)) { $( $field : $ty [$($role)|*] $(= $default)? ,)* });
                    Ok((finished(Statement::$StatementType(stmt)).py()?, held))
                }
            }

            #[pymethods]
            impl $StatementType {
                #[new]
                #[pyo3(signature = (*args, **kwargs))]
                fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<PyClassInitializer<Self>> {
                    let (stmt, held) = Self::build(args, kwargs)?;
                    Ok(PyClassInitializer::from(PyStatement::new(Arc::new(stmt), held)).add_subclass($StatementType))
                }

                /// A copy of this statement with the named fields changed: the one way a statement is edited.
                #[pyo3(signature = (**changes))]
                fn replace<'py>(slf: &Bound<'py, Self>, changes: Option<&Bound<'py, PyDict>>) -> PyResult<Bound<'py, PyAny>> {
                    let Some(changes) = changes.filter(|c| !c.is_empty()) else { return Ok(slf.clone().into_any()) };
                    let base = slf.as_super().get();
                    let (stmt, held) = Self::rebuild(base, slf.py(), changes)?;
                    Ok(Bound::new(slf.py(), PyClassInitializer::from(PyStatement::new(Arc::new(stmt), held)).add_subclass($StatementType))?.into_any())
                }

                /// The names of this statement type's fields, in declaration order.
                #[classattr]
                fn _fields<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
                    PyTuple::new(py, emmy_ir::statement::$StatementType::FIELDS.iter().map(|f| PyString::new(py, f.name)))
                }

                /// A pure value binding: a legal member of a stored term.
                #[classattr]
                fn pure() -> bool {
                    emmy_ir::statement::$StatementType::FLAGS.contains(emmy_ir::statement::PURE)
                }

                /// `deps` already reports the whole subtree's reads.
                #[classattr]
                fn deps_deep() -> bool {
                    emmy_ir::statement::$StatementType::FLAGS.contains(emmy_ir::statement::DEPS_DEEP)
                }

                $(
                    #[getter]
                    fn $field<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
                        const INDEX: usize = position(emmy_ir::statement::$StatementType::FIELDS, stringify!($field));
                        slf.as_super().get().field(slf.py(), INDEX)
                    }
                )*
            }
        )*

        $(
            define_py_statement_types!(@record $rlayer $Record $(#[doc = $rdoc])*);

            impl $Record {
                pub fn new_from(record: emmy_ir::statement::$Record) -> $Record {
                    $Record { slots: slots(emmy_ir::statement::$Record::FIELDS.len()), record }
                }

                /// The record with the Python objects its fields came from kept beside them.
                fn with_held(record: emmy_ir::statement::$Record, held: Held) -> $Record {
                    let out = $Record::new_from(record);
                    for (slot, value) in out.slots.iter().zip(held) {
                        if let Some(value) = value {
                            let _ = slot.set(value);
                        }
                    }
                    out
                }
            }

            #[allow(non_upper_case_globals, unused_variables, unused_mut)]
            #[pymethods]
            impl $Record {
                #[new]
                #[pyo3(signature = (*args, **kwargs))]
                fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<$Record> {
                    let mut held = Held::new();
                    let record = define_py_statement_types!(@given $Record args kwargs held { $( $rfield : $rty [$($rrole)|*] $(= $rdefault)? ,)* });
                    Ok($Record::with_held(record, held))
                }

                /// A copy with the named fields changed; itself when nothing changes.
                #[pyo3(signature = (**changes))]
                fn replace<'py>(slf: &Bound<'py, Self>, changes: Option<&Bound<'py, PyDict>>) -> PyResult<Bound<'py, Self>> {
                    let Some(changes) = changes.filter(|c| !c.is_empty()) else { return Ok(slf.clone()) };
                    let this = slf.get();
                    let current = &this.record;
                    let mut held = Held::new();
                    let record = define_py_statement_types!(@changed $Record current changes held (|index: usize| this.slots[index].get().map(|value| value.clone_ref(slf.py()))) { $( $rfield : $rty [$($rrole)|*] $(= $rdefault)? ,)* });
                    Bound::new(slf.py(), $Record::with_held(record, held))
                }

                #[classattr]
                fn _fields<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
                    PyTuple::new(py, emmy_ir::statement::$Record::FIELDS.iter().map(|f| PyString::new(py, f.name)))
                }

                $(
                    #[getter]
                    fn $rfield<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
                        const INDEX: usize = position(emmy_ir::statement::$Record::FIELDS, stringify!($rfield));
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
                $( Statement::$StatementType(k) => {
                    #[allow(unused_mut, unused_variables, unused_assignments)]
                    let mut position = 0usize;
                    $(
                        if position == index {
                            return <$ty as PyField>::to_py(&k.$field, py);
                        }
                        position += 1;
                    )*
                    Err(PyAttributeError::new_err(format!("{} has no field {index}", stringify!($StatementType))))
                } )*
            }
        }

        /// The Python object of `stmt`, an instance of its statement type's class.
        pub fn wrap_statement(py: Python<'_>, stmt: &Arc<Statement>) -> PyResult<Py<PyAny>> {
            let base = PyClassInitializer::from(PyStatement::new(stmt.clone(), Vec::new()));
            Ok(match &**stmt {
                $( Statement::$StatementType(_) => Py::new(py, base.add_subclass($StatementType))?.into_any(), )*
            })
        }

        /// The statement type a class stands for, for `iter_of_type`.
        pub fn statement_type_of(cls: &Bound<'_, PyType>) -> Option<emmy_ir::statement::StatementType> {
            $( if cls.is(&cls.py().get_type::<$StatementType>()) { return Some(emmy_ir::statement::StatementType::$StatementType); } )*
            None
        }

        pub fn register_statement_types(m: &Bound<'_, PyModule>) -> PyResult<()> {
            m.add_class::<PyStatement>()?;
            $( m.add_class::<$StatementType>()?; )*
            $( m.add_class::<$Record>()?; )*
            Ok(())
        }
    };
    (@class shared $StatementType:ident $(#[doc = $doc:literal])*) => {
        $(#[doc = $doc])*
        #[pyclass(extends = PyStatement, frozen, module = "emmy.compiler.ir.stmt")]
        pub struct $StatementType;
    };
    (@class kernel $StatementType:ident $(#[doc = $doc:literal])*) => {
        $(#[doc = $doc])*
        #[pyclass(extends = PyStatement, frozen, module = "emmy.compiler.ir.kernel.ir")]
        pub struct $StatementType;
    };
    (@record shared $Record:ident $(#[doc = $doc:literal])*) => {
        $(#[doc = $doc])*
        #[pyclass(frozen, module = "emmy.compiler.ir.stmt")]
        pub struct $Record {
            pub record: emmy_ir::statement::$Record,
            slots: Slots,
        }
    };
    (@record kernel $Record:ident $(#[doc = $doc:literal])*) => {
        $(#[doc = $doc])*
        #[pyclass(frozen, module = "emmy.compiler.ir.kernel.ir")]
        pub struct $Record {
            pub record: emmy_ir::statement::$Record,
            slots: Slots,
        }
    };
    // The constructor's fields: each by keyword or position, converted and kept, or its default.
    (@given $Type:ident $args:ident $kwargs:ident $held:ident { $( $field:ident : $ty:ty [$($role:ident)|*] $(= $default:expr)? ,)* }) => {{
        use emmy_ir::statement as k;
        let mut given = Arguments { type_name: k::$Type::NAME, fields: k::$Type::FIELDS, args: $args, kwargs: $kwargs, positional: 0 };
        let out = k::$Type {
            $( $field: match given.take(stringify!($field), define_py_statement_types!(@kw_only $($role)|*))? {
                Some(value) => define_py_statement_types!(@convert $Type $field [$ty] value $held),
                None => {
                    $held.push(None);
                    define_py_statement_types!(@default $Type $field $(= $default)?)
                }
            }, )*
        };
        given.finish()?;
        out
    }};
    // A copy's fields: each from `changes`, converted and kept, or the current value with the object kept so far.
    (@changed $Type:ident $current:ident $changes:ident $held:ident ($kept:expr) { $( $field:ident : $ty:ty [$($role:ident)|*] $(= $default:expr)? ,)* }) => {{
        use emmy_ir::statement as k;
        only_fields(k::$Type::NAME, k::$Type::FIELDS, $changes)?;
        let kept = $kept;
        #[allow(unused_assignments)]
        let mut index = 0usize;
        k::$Type {
            $( $field: {
                let value = match $changes.get_item(stringify!($field))? {
                    Some(value) => define_py_statement_types!(@convert $Type $field [$ty] value $held),
                    None => {
                        $held.push(kept(index));
                        $current.$field.clone()
                    }
                };
                index += 1;
                value
            }, )*
        }
    }};
    (@convert $Type:ident $field:ident [$ty:ty] $value:ident $held:ident) => {{
        let (value, kept) = <$ty as PyField>::from_py(&$value)
            .map_err(|err| located($value.py(), &format!("{}.{}", stringify!($Type), stringify!($field)), err))?;
        $held.push(kept);
        value
    }};
    (@kw_only) => { false };
    (@kw_only KW_ONLY $(| $rest:ident)*) => { true };
    (@kw_only $role:ident $(| $rest:ident)*) => { define_py_statement_types!(@kw_only $($rest)|*) };
    (@default $StatementType:ident $field:ident = $default:expr) => { $default };
    (@default $StatementType:ident $field:ident) => { return Err(missing(stringify!($StatementType), stringify!($field))) };
}

/// The Python objects a statement's fields came from, one slot per field.
type Held = Vec<Option<Py<PyAny>>>;

/// Every key of `changes` names a field of the statement type.
fn only_fields(type_name: &str, fields: &[FieldInfo], changes: &Bound<'_, PyDict>) -> PyResult<()> {
    for key in changes.keys() {
        let key: String = key.extract()?;
        if !fields.iter().any(|f| f.name == key) {
            return Err(unexpected(type_name, &key));
        }
    }
    Ok(())
}

/// The generated classes: a statement type with no field leaves the per-field locals of its constructors unused.
#[allow(unused_variables, unused_assignments, unused_mut)]
mod generated {
    use super::*;
    use emmy_ir::Name;
    use emmy_ir::body::Body;
    use emmy_ir::dtype::DType;
    use emmy_ir::elementwise::ElementwiseOp;
    use emmy_ir::field::*;

    emmy_ir::with_statement_types!(define_py_statement_types);
}

pub use generated::*;

// --- what the shared statement types know beyond their fields -------------------------------------

/// The statement behind a statement type's class, as that type's struct.
fn inner<'a, K>(base: &'a PyStatement) -> PyResult<&'a K>
where
    &'a K: TryFrom<&'a Statement, Error = emmy_ir::Error>,
{
    <&K>::try_from(&*base.stmt).py()
}

#[pymethods]
impl Load {
    /// Scalar-only shorthand for `names[0]`.
    #[getter]
    fn name<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let load = inner::<emmy_ir::statement::Load>(slf.as_super().get())?;
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
        Ok(inner::<emmy_ir::statement::Load>(slf.as_super().get())?
            .names
            .len())
    }

    #[getter]
    fn is_vector(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(inner::<emmy_ir::statement::Load>(slf.as_super().get())?
            .names
            .len()
            > 1)
    }

    #[getter]
    fn is_scalar(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(inner::<emmy_ir::statement::Load>(slf.as_super().get())?
            .names
            .len()
            == 1)
    }
}

#[pymethods]
impl Write {
    /// Scalar-only shorthand for `values[0]`.
    #[getter]
    fn value<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let write = inner::<emmy_ir::statement::Write>(slf.as_super().get())?;
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
        Ok(inner::<emmy_ir::statement::Write>(slf.as_super().get())?
            .values
            .len())
    }

    #[getter]
    fn is_vector(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(inner::<emmy_ir::statement::Write>(slf.as_super().get())?
            .values
            .len()
            > 1)
    }

    #[getter]
    fn is_scalar(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(inner::<emmy_ir::statement::Write>(slf.as_super().get())?
            .values
            .len()
            == 1)
    }
}

#[pymethods]
impl Affine {
    /// `(coefficient, arg)` per term.
    #[getter]
    fn terms<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let affine: &emmy_ir::statement::Affine = inner(slf.as_super().get())?;
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
        let affine: &emmy_ir::statement::Affine = inner(slf.as_super().get())?;
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
        let accum: &emmy_ir::statement::Accum = inner(slf.as_super().get())?;
        PyExpression::wrap(slf.py(), &accum.init())
    }

    #[getter]
    fn associative(slf: &Bound<'_, Self>) -> PyResult<bool> {
        let accum: &emmy_ir::statement::Accum = inner(slf.as_super().get())?;
        Ok(accum.op.associative())
    }

    #[getter]
    fn commutative(slf: &Bound<'_, Self>) -> PyResult<bool> {
        let accum: &emmy_ir::statement::Accum = inner(slf.as_super().get())?;
        Ok(accum.op.commutative())
    }

    #[getter]
    fn has_identity(slf: &Bound<'_, Self>) -> PyResult<bool> {
        let accum: &emmy_ir::statement::Accum = inner(slf.as_super().get())?;
        Ok(accum.op.neutral().is_some())
    }
}

#[pymethods]
impl Carry {
    /// The axes of the cell this statement defines.
    #[getter]
    fn cells<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let carry: &emmy_ir::statement::Carry = inner(slf.as_super().get())?;
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
    cells: Vec<(Name, Vec<Option<Arc<emmy_ir::axis::Axis>>>)>,
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
        let l: &emmy_ir::statement::Loop = inner(slf.as_super().get())?;
        carries_dict(slf.py(), l.carries())
    }

    /// A loop is a reduce loop iff it carries something.
    #[getter]
    fn is_reduce(slf: &Bound<'_, Self>) -> PyResult<bool> {
        let l: &emmy_ir::statement::Loop = inner(slf.as_super().get())?;
        Ok(l.is_reduce())
    }
}

#[pymethods]
impl StridedLoop {
    #[getter]
    fn is_reduce(slf: &Bound<'_, Self>) -> PyResult<bool> {
        let l: &emmy_ir::statement::StridedLoop = inner(slf.as_super().get())?;
        Ok(l.is_reduce())
    }
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
