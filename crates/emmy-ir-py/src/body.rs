//! `Body`, `Subroutine` and `Cone` as Python classes, and the module's functions over bodies and statements.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};

use emmy_ir::Name;
use emmy_ir::body::{Body, StmtRef, Subroutine, exposed_defines, free_names};
use emmy_ir::kinds::Statement;
use emmy_ir::text::{body_lines, body_text, parse_body as parse_body_text};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PyList, PySet, PySlice, PyString, PyTuple, PyType};

use crate::convert::{
    IntoPyErr, PyField, Slots, axis_to_py, cached, slots, statement_from_py, str_tuple,
};
use crate::expression::{PyExpression, renames_from_py};
use crate::statement::{PyStatement, kind_of, statements_from_py, wrap_statement};

/// The wrappers a body's statements hand back, so a statement read twice is one object however many wrappers the body
/// has. The table is keyed by the body's shared allocation and lives on this side of the boundary: the crate holds no
/// Python object. An entry outlives its body only until the next purge.
type Wrappers = Arc<[OnceLock<Py<PyAny>>]>;

/// A table of per-object state on this side of the boundary, keyed by the shared object's address while it lives:
/// a body's statement wrappers, a subroutine's one Python object. An entry outlives its object only until the next
/// purge.
struct Tables<T, V> {
    entries: HashMap<usize, (std::sync::Weak<T>, V)>,
}

impl<T, V> Tables<T, V> {
    fn new() -> Tables<T, V> {
        Tables {
            entries: HashMap::new(),
        }
    }

    fn get<R>(&self, object: &Arc<T>, read: impl FnOnce(&V) -> R) -> Option<R> {
        let (alive, value) = self.entries.get(&(Arc::as_ptr(object) as usize))?;
        alive
            .upgrade()
            .filter(|found| Arc::ptr_eq(found, object))
            .map(|_| read(value))
    }

    fn insert(&mut self, object: &Arc<T>, value: V) {
        if self.entries.len() % 1024 == 1023 {
            self.entries
                .retain(|_, (alive, _)| alive.strong_count() > 0);
        }
        self.entries.insert(
            Arc::as_ptr(object) as usize,
            (Arc::downgrade(object), value),
        );
    }
}

static WRAPPERS: OnceLock<Mutex<Tables<emmy_ir::body::BodyData, Wrappers>>> = OnceLock::new();

fn wrappers(body: &Body) -> Wrappers {
    let mut tables = emmy_ir::locked(WRAPPERS.get_or_init(|| Mutex::new(Tables::new())));
    if let Some(table) = tables.get(&body.0, Wrappers::clone) {
        return table;
    }
    let table: Wrappers = (0..body.len()).map(|_| OnceLock::new()).collect();
    tables.insert(&body.0, table.clone());
    table
}

/// An immutable sequence of statements, built from any iterable.
#[pyclass(frozen, module = "emmy.compiler.ir.stmt", name = "Body")]
pub struct PyBody {
    pub body: Body,
}

impl PyBody {
    pub fn wrap<'py>(py: Python<'py>, body: &Body) -> PyResult<Bound<'py, PyBody>> {
        Bound::new(py, PyBody { body: body.clone() })
    }

    /// The statements of a Python iterable as a body; the wrappers given become the ones the body hands back.
    pub fn from_iterable(value: &Bound<'_, PyAny>) -> PyResult<Body> {
        if let Ok(body) = value.cast::<PyBody>() {
            return Ok(body.get().body.clone());
        }
        if value.is_instance_of::<PyString>() {
            return Err(PyTypeError::new_err(
                "Body holds non-statement member(s) ['str']; a term composes through operand edges",
            ));
        }
        let items: PyResult<Vec<Bound<'_, PyAny>>> = value.try_iter()?.collect();
        let items = items?;
        let stmts: PyResult<Vec<Arc<Statement>>> = items.iter().map(statement_from_py).collect();
        let body = Body::new(stmts?);
        let table = wrappers(&body);
        if table.len() == items.len() {
            for (slot, item) in table.iter().zip(items) {
                let _ = slot.set(item.unbind());
            }
        }
        Ok(body)
    }

    /// The statement at `index`, the same object every time.
    pub fn item<'py>(py: Python<'py>, body: &Body, index: usize) -> PyResult<Bound<'py, PyAny>> {
        let table = wrappers(body);
        match table.get(index) {
            Some(slot) => cached(py, slot, || wrap_statement(py, &body.stmts()[index])),
            None => Ok(wrap_statement(py, &body.stmts()[index])?.into_bound(py)),
        }
    }

    fn items<'py>(py: Python<'py>, body: &Body) -> PyResult<Vec<Bound<'py, PyAny>>> {
        (0..body.len())
            .map(|index| PyBody::item(py, body, index))
            .collect()
    }
}

pub fn wrap_ref<'py>(py: Python<'py>, here: &StmtRef) -> PyResult<Bound<'py, PyAny>> {
    match here.body() {
        Some(body) => PyBody::item(py, &body, here.index),
        None => Ok(wrap_statement(py, here.stmt())?.into_bound(py)),
    }
}

fn frozenset<'py>(
    py: Python<'py>,
    names: impl IntoIterator<Item = Name>,
) -> PyResult<Bound<'py, PyFrozenSet>> {
    let names: Vec<Name> = names.into_iter().collect();
    PyFrozenSet::new(py, names.iter().map(|name| PyString::new(py, name)))
}

fn incoming_from_py(value: &Bound<'_, PyAny>) -> PyResult<Vec<HashSet<usize>>> {
    value
        .try_iter()?
        .map(|sources| {
            sources?
                .try_iter()?
                .map(|source| source?.extract::<usize>())
                .collect()
        })
        .collect()
}

#[pymethods]
impl PyBody {
    #[new]
    #[pyo3(signature = (stmts = None))]
    fn new(stmts: Option<&Bound<'_, PyAny>>) -> PyResult<PyBody> {
        match stmts {
            None => Ok(PyBody {
                body: Body::empty(),
            }),
            Some(value) => Ok(PyBody {
                body: PyBody::from_iterable(value)?,
            }),
        }
    }

    #[classattr]
    fn wire_tag() -> &'static str {
        "body"
    }

    /// Wrap if not already a Body.
    #[staticmethod]
    fn coerce<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBody>> {
        if let Ok(body) = value.cast::<PyBody>() {
            return Ok(body.clone());
        }
        PyBody::wrap(value.py(), &PyBody::from_iterable(value)?)
    }

    fn __len__(&self) -> usize {
        self.body.len()
    }

    fn __reversed__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mut items = PyBody::items(py, &self.body)?;
        items.reverse();
        Ok(PyList::new(py, items)?.into_any().try_iter()?.into_any())
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(PyList::new(py, PyBody::items(py, &self.body)?)?
            .into_any()
            .try_iter()?
            .into_any())
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if let Ok(slice) = key.cast::<PySlice>() {
            let indices = slice.indices(self.body.len() as isize)?;
            let picked: Vec<Arc<Statement>> = (0..indices.slicelength)
                .map(|step| {
                    self.body.stmts()[(indices.start + step as isize * indices.step) as usize]
                        .clone()
                })
                .collect();
            return Ok(PyBody::wrap(py, &Body::new(picked))?.into_any());
        }
        let index: isize = key.extract()?;
        let len = self.body.len() as isize;
        let index = if index < 0 { index + len } else { index };
        if index < 0 || index >= len {
            return Err(PyIndexError::new_err("Body index out of range"));
        }
        PyBody::item(py, &self.body, index as usize)
    }

    fn __contains__(&self, item: &Bound<'_, PyAny>) -> bool {
        item.cast::<PyStatement>().is_ok_and(|stmt| {
            self.body
                .stmts()
                .iter()
                .any(|s| Arc::ptr_eq(s, &stmt.get().stmt) || **s == *stmt.get().stmt)
        })
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<PyBody>()
            .is_ok_and(|other| other.get().body == self.body)
    }

    fn __hash__(&self) -> u64 {
        self.body.content_hash()
    }

    /// The body's text, which `parse_body` reads back to an equal body.
    fn __str__(&self) -> PyResult<String> {
        body_text(&self.body).py()
    }

    fn __add__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyBody>> {
        let others = statements_from_py(other)?;
        PyBody::wrap(py, &self.body.concat(&others))
    }

    fn __radd__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyBody>> {
        let mut stmts = statements_from_py(other)?;
        stmts.extend(self.body.stmts().iter().cloned());
        PyBody::wrap(py, &Body::new(stmts))
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, (String,))> {
        let module = slf.py().import("emmy.emmy_ir")?;
        Ok((
            module.getattr("parse_body")?,
            (body_text(&slf.get().body).py()?,),
        ))
    }

    fn to_wire(&self) -> PyResult<String> {
        body_text(&self.body).py()
    }

    #[classmethod]
    #[pyo3(signature = (value, where_ = "body"))]
    fn from_wire<'py>(
        _cls: &Bound<'py, PyType>,
        value: &Bound<'py, PyAny>,
        where_: &str,
    ) -> PyResult<Bound<'py, PyBody>> {
        let Ok(text) = value.cast::<PyString>() else {
            return Err(PyValueError::new_err(format!(
                "{where_} must be a body's text"
            )));
        };
        let body = parse_body_text(text.to_str()?)
            .map_err(|err| PyValueError::new_err(format!("{where_}: {err}")))?;
        PyBody::wrap(value.py(), &body)
    }

    fn wire_parts(&self) -> Option<()> {
        None
    }

    /// Elementwise, as a sequence renders.
    fn structural_form<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, PyBody::items(py, &self.body)?)
    }

    /// Pre-order iteration over this body and every nested body.
    fn iter<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let items: PyResult<Vec<Bound<'py, PyAny>>> = self
            .body
            .iter_refs()
            .iter()
            .map(|here| wrap_ref(py, here))
            .collect();
        Ok(PyList::new(py, items?)?.into_any().try_iter()?.into_any())
    }

    /// This body with every external buffer renamed through `rename`.
    fn rename_buffers<'py>(
        slf: &Bound<'py, Self>,
        rename: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyBody>> {
        let mapping = renames_from_py(rename)?;
        let result = slf.get().body.rename_buffers(&mapping);
        if result.ptr_eq(&slf.get().body) {
            return Ok(slf.clone());
        }
        PyBody::wrap(slf.py(), &result)
    }

    /// Kahn's order over indexed predecessor sets, as source indices.
    fn _topological_permutation(&self, incoming: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
        Ok(self
            .body
            .topological_permutation(&incoming_from_py(incoming)?, None))
    }

    /// This body in topological order under `incoming`.
    #[pyo3(signature = (incoming, tie_break = None))]
    fn topological_order<'py>(
        slf: &Bound<'py, Self>,
        incoming: &Bound<'py, PyAny>,
        tie_break: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyBody>> {
        if tie_break.is_some_and(|t| !t.is_none()) {
            return Err(PyTypeError::new_err(
                "topological_order takes no tie-break; the canonical order is normalization's",
            ));
        }
        let ordered = slf
            .get()
            .body
            .topological_order(&incoming_from_py(incoming)?);
        if ordered.ptr_eq(&slf.get().body) {
            return Ok(slf.clone());
        }
        PyBody::wrap(slf.py(), &ordered)
    }

    /// Shared SSA definitions assembled once and adjacent independent reduction loops merged.
    fn coalesce<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBody>> {
        PyBody::wrap(py, &self.body.coalesce())
    }

    /// Every SSA name produced anywhere inside this body to its defining statement.
    #[getter]
    fn definitions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        let mut entries: Vec<(&Name, &StmtRef)> = self.body.definitions().iter().collect();
        entries.sort_by_key(|entry| entry.0);
        for (name, here) in entries {
            out.set_item(&**name, wrap_ref(py, here)?)?;
        }
        Ok(out)
    }

    /// Accumulator and carried-state names, each once, in structural order.
    #[getter]
    fn _carried_names<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        str_tuple(py, self.body.carried_names())
    }

    /// Bindings at this scope, including states exported by nested statements.
    #[getter]
    fn _local_defs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
        frozenset(py, self.body.local_defs().iter().cloned())
    }

    /// SSA reads from the enclosing scope, excluding this scope's definitions and exported states.
    #[getter]
    fn _free_ssa<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
        frozenset(py, self.body.free_ssa().iter().cloned())
    }

    #[getter]
    fn axis_names<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
        frozenset(py, self.body.axis_names().iter().cloned())
    }

    #[getter]
    fn ssa_uses<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
        frozenset(py, self.body.ssa_uses().iter().cloned())
    }

    /// Each SSA definition to the axes its value depends on.
    #[getter]
    fn axis_dependencies<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (name, axes) in self.body.axis_dependencies() {
            out.set_item(&*name, frozenset(py, axes)?)?;
        }
        Ok(out)
    }

    /// The backward dependence cone of `roots` over this body's immediate statements.
    fn backward_cone(&self, py: Python<'_>, roots: &Bound<'_, PyAny>) -> PyResult<PyCone> {
        let roots: PyResult<Vec<Name>> = roots
            .try_iter()?
            .map(|item| item?.extract::<String>().map(Name::from))
            .collect();
        let (members, external) = self.body.backward_cone(&roots?);
        let members: PyResult<Vec<Bound<'_, PyAny>>> =
            members.iter().map(|here| wrap_ref(py, here)).collect();
        Ok(PyCone {
            members: PyTuple::new(py, members?)?.unbind(),
            external_reads: frozenset(py, external)?.unbind(),
        })
    }

    /// Matching statements in pre-order.
    #[pyo3(signature = (*types))]
    fn iter_of_type<'py>(
        &self,
        py: Python<'py>,
        types: &Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let mut kinds: Vec<&'static str> = Vec::new();
        for item in types.iter() {
            let cls = item
                .cast::<PyType>()
                .map_err(|_| PyTypeError::new_err("iter_of_type takes statement classes"))?;
            match kind_of(cls) {
                Some(kind) => kinds.push(kind),
                None if cls.is(py.get_type::<PyStatement>()) => {
                    kinds.extend(emmy_ir::kinds::KIND_NAMES.iter().copied())
                }
                None => {
                    return Err(PyTypeError::new_err(format!(
                        "{} is no statement kind",
                        cls.name()?
                    )));
                }
            }
        }
        let items: PyResult<Vec<Bound<'py, PyAny>>> = self
            .body
            .of_kinds(&kinds)
            .iter()
            .map(|here| wrap_ref(py, here))
            .collect();
        PyTuple::new(py, items?)
    }

    #[getter]
    fn loads<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let items: PyResult<Vec<Bound<'py, PyAny>>> = self
            .body
            .loads()
            .iter()
            .map(|here| wrap_ref(py, here))
            .collect();
        PyTuple::new(py, items?)
    }

    #[getter]
    fn writes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let items: PyResult<Vec<Bound<'py, PyAny>>> = self
            .body
            .writes()
            .iter()
            .map(|here| wrap_ref(py, here))
            .collect();
        PyTuple::new(py, items?)
    }

    #[getter]
    fn carries<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let items: PyResult<Vec<Bound<'py, PyAny>>> = self
            .body
            .carries()
            .iter()
            .map(|here| wrap_ref(py, here))
            .collect();
        PyTuple::new(py, items?)
    }

    #[getter]
    fn accums<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let items: PyResult<Vec<Bound<'py, PyAny>>> = self
            .body
            .accums()
            .iter()
            .map(|here| wrap_ref(py, here))
            .collect();
        PyTuple::new(py, items?)
    }

    #[getter]
    fn loops<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let items: PyResult<Vec<Bound<'py, PyAny>>> = self
            .body
            .loops()
            .iter()
            .map(|here| wrap_ref(py, here))
            .collect();
        PyTuple::new(py, items?)
    }

    /// The identity key and the buffers in role order, under the clustered or exact reading and the given colors.
    #[pyo3(signature = (cluster, colors = None))]
    fn _identity<'py>(
        &self,
        py: Python<'py>,
        cluster: bool,
        colors: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<(String, Bound<'py, PyTuple>)> {
        let colors: Option<HashMap<Name, String>> = colors
            .map(|dict| {
                dict.iter()
                    .map(|(k, v)| Ok((Name::from(k.extract::<String>()?), v.extract::<String>()?)))
                    .collect::<PyResult<_>>()
            })
            .transpose()?;
        let identity = emmy_ir::numbering::identity::canonicalize_identity(
            &self.body,
            cluster,
            colors.as_ref(),
        )
        .py()?;
        Ok((identity.key, str_tuple(py, &identity.arguments)?))
    }
}

/// A dependence cone over one scope level: the members, in body order, and the names read from outside them.
#[pyclass(frozen, module = "emmy.compiler.ir.stmt", name = "Cone")]
pub struct PyCone {
    #[pyo3(get)]
    members: Py<PyTuple>,
    #[pyo3(get)]
    external_reads: Py<PyFrozenSet>,
}

#[pymethods]
impl PyCone {
    /// Every `Load` in the members, nested included, body order.
    #[getter]
    fn loads<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let stmts: PyResult<Vec<Arc<Statement>>> = self
            .members
            .bind(py)
            .iter()
            .map(|item| statement_from_py(&item))
            .collect();
        let body = Body::new(stmts?);
        let items: PyResult<Vec<Bound<'py, PyAny>>> =
            body.loads().iter().map(|here| wrap_ref(py, here)).collect();
        PyTuple::new(py, items?)
    }
}

/// One shared, read-only reduction cone with explicit coordinate parameters. Compared by object.
#[pyclass(frozen, weakref, module = "emmy.compiler.ir.stmt", name = "Subroutine")]
pub struct PySubroutine {
    pub sub: Arc<Subroutine>,
    slots: Slots,
}

/// The weak reference to each subroutine's one Python object, keyed by the subroutine's address while it lives.
static SUBROUTINES: OnceLock<Mutex<Tables<Subroutine, Py<PyAny>>>> = OnceLock::new();

fn subroutines() -> &'static Mutex<Tables<Subroutine, Py<PyAny>>> {
    SUBROUTINES.get_or_init(|| Mutex::new(Tables::new()))
}

impl PySubroutine {
    /// The one Python object of `sub` while any reference to it lives.
    pub fn wrap<'py>(py: Python<'py>, sub: &Arc<Subroutine>) -> PyResult<Bound<'py, PySubroutine>> {
        let remembered =
            emmy_ir::locked(subroutines()).get(sub, |reference| reference.clone_ref(py));
        if let Some(reference) = remembered {
            let alive = reference.bind(py).call0()?;
            if let Ok(found) = alive.cast::<PySubroutine>() {
                return Ok(found.clone());
            }
        }
        let wrapper = Bound::new(
            py,
            PySubroutine {
                sub: sub.clone(),
                slots: slots(4),
            },
        )?;
        let reference = py.import("weakref")?.getattr("ref")?.call1((&wrapper,))?;
        emmy_ir::locked(subroutines()).insert(sub, reference.unbind());
        Ok(wrapper)
    }
}

#[pymethods]
impl PySubroutine {
    #[new]
    fn new(
        name: &str,
        axes: &Bound<'_, PyAny>,
        body: &Bound<'_, PyAny>,
        result: &str,
    ) -> PyResult<PySubroutine> {
        let (axes, _) = <emmy_ir::kinds::field::Axes as PyField>::from_py(axes)?;
        let body = PyBody::from_iterable(body)?;
        let sub = Subroutine::new(name.into(), axes, body, result.into()).py()?;
        Ok(PySubroutine {
            sub,
            slots: slots(4),
        })
    }

    #[getter]
    fn name<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.slots[0], || {
            Ok(PyString::new(py, &self.sub.name).into_any().unbind())
        })
    }

    #[getter]
    fn axes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.slots[1], || self.sub.axes.to_py(py))
    }

    #[getter]
    fn body<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.slots[2], || {
            Ok(PyBody::wrap(py, &self.sub.body)?.into_any().unbind())
        })
    }

    #[getter]
    fn result<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.slots[3], || {
            Ok(PyString::new(py, &self.sub.result).into_any().unbind())
        })
    }

    /// The buffers the body reads, each once.
    #[getter]
    fn buffers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        str_tuple(py, &self.sub.buffers)
    }

    #[getter]
    fn params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        str_tuple(py, &self.sub.params())
    }

    /// Rebind the captured buffers once for all calls sharing this definition.
    fn with_buffers<'py>(
        &self,
        py: Python<'py>,
        buffers: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PySubroutine>> {
        let (buffers, _) = <emmy_ir::kinds::field::Names as PyField>::from_py(buffers)?;
        if buffers.len() != self.sub.buffers.len() {
            return Err(PyValueError::new_err(format!(
                "{} buffers for {} captured",
                buffers.len(),
                self.sub.buffers.len()
            )));
        }
        PySubroutine::wrap(py, &self.sub.with_buffers(buffers))
    }

    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
        let this = slf.get();
        let parts = [
            this.name(slf.py())?,
            this.axes(slf.py())?,
            this.body(slf.py())?,
            this.result(slf.py())?,
        ];
        Ok((slf.get_type(), PyTuple::new(slf.py(), parts)?))
    }

    fn __repr__(&self) -> String {
        format!("Subroutine({:?})", &*self.sub.name)
    }
}

// --- functions ------------------------------------------------------------------------------------

/// The body `text` spells.
#[pyfunction]
pub fn parse_body<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyBody>> {
    PyBody::wrap(py, &parse_body_text(text).py()?)
}

/// The lines of a body's text, each starting with `indent`.
#[pyfunction]
#[pyo3(signature = (body, indent = ""))]
pub fn pretty_body(body: &Bound<'_, PyAny>, indent: &str) -> PyResult<Vec<String>> {
    let body = PyBody::from_iterable(body)?;
    let mut out = Vec::new();
    body_lines(&body, indent, &mut out).py()?;
    Ok(out)
}

/// The executable normal form of a body: the body itself when it is its own normal form.
#[pyfunction]
pub fn normalize_body<'py>(body: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBody>> {
    let normal = emmy_ir::normalize::normalize_body(&PyBody::from_iterable(body)?).py()?;
    match body.cast::<PyBody>() {
        Ok(given) if given.get().body.ptr_eq(&normal) => Ok(given.clone()),
        _ => PyBody::wrap(body.py(), &normal),
    }
}

#[pyfunction]
pub fn prepare_body<'py>(body: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBody>> {
    PyBody::wrap(
        body.py(),
        &emmy_ir::normalize::prepare_body(&PyBody::from_iterable(body)?).py()?,
    )
}

#[pyfunction]
pub fn simplify_body<'py>(body: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBody>> {
    PyBody::wrap(
        body.py(),
        &emmy_ir::normalize::simplify_body(&PyBody::from_iterable(body)?).py()?,
    )
}

/// Fold every maximal single-reader affine cone into one `Affine`; `exposed` names count as read.
#[pyfunction]
#[pyo3(signature = (body, exposed = None))]
pub fn fold_affine<'py>(
    body: &Bound<'py, PyAny>,
    exposed: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyBody>> {
    let exposed: HashSet<Name> = match exposed {
        Some(names) => names
            .try_iter()?
            .map(|item| item?.extract::<String>().map(Name::from))
            .collect::<PyResult<_>>()?,
        None => HashSet::new(),
    };
    PyBody::wrap(
        body.py(),
        &emmy_ir::normalize::fold_affine(&PyBody::from_iterable(body)?, &exposed),
    )
}

macro_rules! step {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[pyfunction]
        pub fn $name<'py>(body: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBody>> {
            PyBody::wrap(
                body.py(),
                &emmy_ir::normalize::$name(&PyBody::from_iterable(body)?),
            )
        }
    };
    ($name:ident, $doc:literal, fallible) => {
        #[doc = $doc]
        #[pyfunction]
        pub fn $name<'py>(body: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBody>> {
            PyBody::wrap(
                body.py(),
                &emmy_ir::normalize::$name(&PyBody::from_iterable(body)?).py()?,
            )
        }
    };
}

step!(
    drop_size_one_free_axes,
    "Inline every free loop of extent one."
);
step!(
    drop_size_one_reduce_axes,
    "Inline a canonical extent-one reduction as its single update."
);
step!(
    canonicalize_free_axis_order,
    "Sort an outer free-loop chain by row-major output storage order.",
    fallible
);
step!(
    fold_trivial_selects,
    "A Select whose branches all bind one value is a copy of it."
);
step!(
    fold_constants,
    "Evaluate every operation whose operands are all literals."
);
step!(
    drop_unread_values,
    "Drop every pure statement whose values nothing reads."
);
step!(
    merge_sibling_reduce_loops,
    "Merge sibling reduce loops with one axis into one loop."
);
step!(
    unify_siblings,
    "Rename reduce-axis vars across sibling reduce loops whose Load positions overlap."
);
step!(
    place_values,
    "Place every pure value once, at the shallowest scope on its consumers' path.",
    fallible
);
step!(rename_ssa_sequential, "Canonicalize names in a fused body.");
step!(
    canonical_order,
    "Dependency-valid statement order, commutative operand order and names.",
    fallible
);

/// Canonicalize integer expressions using lexical binding order, never axis spelling.
#[pyfunction]
#[pyo3(signature = (body, axes = None))]
pub fn canonicalize_expressions<'py>(
    body: &Bound<'py, PyAny>,
    axes: Option<Vec<String>>,
) -> PyResult<Bound<'py, PyBody>> {
    let axes: Vec<Name> = axes
        .unwrap_or_default()
        .into_iter()
        .map(Name::from)
        .collect();
    PyBody::wrap(
        body.py(),
        &emmy_ir::normalize::canonicalize_expressions(&PyBody::from_iterable(body)?, &axes).py()?,
    )
}

/// Normalization computed afresh, without the memo.
#[pyfunction]
pub fn normalize_uncached<'py>(body: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBody>> {
    PyBody::wrap(
        body.py(),
        &emmy_ir::normalize::normalize_uncached(&PyBody::from_iterable(body)?).py()?,
    )
}

/// Every name the whole subtree of `stmt` reads from its enclosing scope.
#[pyfunction]
#[pyo3(name = "free_names")]
pub fn free_names_py<'py>(
    py: Python<'py>,
    stmt: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyFrozenSet>> {
    frozenset(py, free_names(&*statement_from_py(stmt)?))
}

/// SSA names `stmt` makes visible at its own scope level.
#[pyfunction]
#[pyo3(name = "exposed_defines")]
pub fn exposed_defines_py<'py>(
    py: Python<'py>,
    stmt: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PySet>> {
    let names = exposed_defines(&*statement_from_py(stmt)?);
    PySet::new(py, names.iter().map(|name| PyString::new(py, name)))
}

/// A rename under an alias map, made hygienic.
#[pyfunction]
pub fn rename_free<'py>(
    py: Python<'py>,
    stmt: &Bound<'py, PyAny>,
    alias: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let alias = renames_from_py(alias)?;
    let original = statement_from_py(stmt)?;
    let renamed = emmy_ir::normalize::rename_free(&original, &alias);
    if Arc::ptr_eq(&renamed, &original) {
        return Ok(stmt.clone());
    }
    Ok(wrap_statement(py, &renamed)?.into_bound(py))
}

/// `stmts` with every `Affine` spelled as its binary `Assign`s.
#[pyfunction]
pub fn binary<'py>(py: Python<'py>, stmts: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
    let stmts = statements_from_py(stmts)?;
    let items: PyResult<Vec<Py<PyAny>>> = emmy_ir::protocol::binary(&stmts)
        .iter()
        .map(|s| wrap_statement(py, s))
        .collect();
    PyTuple::new(py, items?)
}

/// The mask branch of an ordered two-way comparison, or `None`.
#[pyfunction]
pub fn mask_select_predicate<'py>(
    py: Python<'py>,
    select: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let stmt = statement_from_py(select)?;
    let Statement::Select(select) = &*stmt else {
        return Err(PyTypeError::new_err("expected a Select"));
    };
    emmy_ir::protocol::mask_select_predicate(select)
        .map(|e| PyExpression::wrap(py, &e))
        .transpose()
}

/// Dependency and, when requested, effect predecessors for one lexical scope.
#[pyfunction]
#[pyo3(signature = (body, *, effects, redefinitions = true))]
pub fn ordering_constraints<'py>(
    py: Python<'py>,
    body: &Bound<'py, PyAny>,
    effects: bool,
    redefinitions: bool,
) -> PyResult<Bound<'py, PyList>> {
    let body = PyBody::from_iterable(body)?;
    let sets: PyResult<Vec<Bound<'py, PySet>>> =
        emmy_ir::normalize::order::ordering_constraints(&body, effects, redefinitions)
            .iter()
            .map(|sources| PySet::new(py, sources.iter()))
            .collect();
    PyList::new(py, sets?)
}

/// Promote an elementwise op's argument dtypes to one result dtype.
#[pyfunction]
pub fn dtype_promote(op_name: &str, arg_dtypes: Vec<String>) -> String {
    let dtypes: Vec<&str> = arg_dtypes.iter().map(|d| d.as_str()).collect();
    emmy_ir::kinds::dtype::promote(op_name, &dtypes).to_string()
}

/// Forget every memoized normal form and identity (the tests of the memos use it).
#[pyfunction]
pub fn clear_memos() {
    emmy_ir::normalize::clear_memo();
    emmy_ir::numbering::identity::clear_memo();
}

/// The carried states a loop over `body` carries, each with its cell shape.
#[pyfunction]
pub fn carried_cells<'py>(
    py: Python<'py>,
    body: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for (name, shape) in emmy_ir::protocol::carried_cells(&PyBody::from_iterable(body)?) {
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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBody>()?;
    m.add_class::<PyCone>()?;
    m.add_class::<PySubroutine>()?;
    m.add_function(wrap_pyfunction!(parse_body, m)?)?;
    m.add_function(wrap_pyfunction!(pretty_body, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_body, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_uncached, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_body, m)?)?;
    m.add_function(wrap_pyfunction!(simplify_body, m)?)?;
    m.add_function(wrap_pyfunction!(fold_affine, m)?)?;
    m.add_function(wrap_pyfunction!(drop_size_one_free_axes, m)?)?;
    m.add_function(wrap_pyfunction!(drop_size_one_reduce_axes, m)?)?;
    m.add_function(wrap_pyfunction!(canonicalize_free_axis_order, m)?)?;
    m.add_function(wrap_pyfunction!(fold_trivial_selects, m)?)?;
    m.add_function(wrap_pyfunction!(fold_constants, m)?)?;
    m.add_function(wrap_pyfunction!(drop_unread_values, m)?)?;
    m.add_function(wrap_pyfunction!(merge_sibling_reduce_loops, m)?)?;
    m.add_function(wrap_pyfunction!(unify_siblings, m)?)?;
    m.add_function(wrap_pyfunction!(place_values, m)?)?;
    m.add_function(wrap_pyfunction!(rename_ssa_sequential, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_order, m)?)?;
    m.add_function(wrap_pyfunction!(canonicalize_expressions, m)?)?;
    m.add_function(wrap_pyfunction!(free_names_py, m)?)?;
    m.add_function(wrap_pyfunction!(exposed_defines_py, m)?)?;
    m.add_function(wrap_pyfunction!(rename_free, m)?)?;
    m.add_function(wrap_pyfunction!(binary, m)?)?;
    m.add_function(wrap_pyfunction!(mask_select_predicate, m)?)?;
    m.add_function(wrap_pyfunction!(ordering_constraints, m)?)?;
    m.add_function(wrap_pyfunction!(dtype_promote, m)?)?;
    m.add_function(wrap_pyfunction!(clear_memos, m)?)?;
    m.add_function(wrap_pyfunction!(carried_cells, m)?)?;
    Ok(())
}
