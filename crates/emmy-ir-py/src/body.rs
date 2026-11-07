//! `Body`, `Subroutine` and `Cone` as Python classes, and the module's functions over bodies and statements.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, Once, OnceLock};

use emmy_ir::Name;
use emmy_ir::body::{Body, StmtRef, Subroutine, exposed_defines, free_names};
use emmy_ir::kinds::Statement;
use emmy_ir::memo::{Stats, Store};
use emmy_ir::text::{body_lines, body_text, parse_body as parse_body_text};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PyList, PySet, PySlice, PyString, PyTuple, PyType};

use crate::convert::{
    IntoPyErr, PyField, Slots, cached, located, slots, statement_from_py, str_tuple, translate,
};
use crate::expression::{PyExpression, renames_from_py};
use crate::statement::{PyStatement, kind_of, statements_from_py, wrap_statement};

/// The wrappers a body's statements hand back, so a statement read twice is one object however many wrappers the body
/// has. The table is keyed by the body's shared allocation and lives on this side of the boundary: the crate holds no
/// Python object. An entry outlives its body only until the next purge.
type Wrappers = Arc<[OnceLock<Py<PyAny>>]>;

/// How many entries a table grows by between two purges of its dead ones.
const PURGE_EVERY: usize = 1024;

/// A table of per-object state on this side of the boundary, keyed by the shared object's address while it lives:
/// a body's statement wrappers, a subroutine's one Python object. An entry outlives its object only until the next
/// purge, every `PURGE_EVERY` inserts or from the memo registry's `clear_all`; the table is listed there with the
/// memos, so it is inspected and purged from the same place, though it is not bounded (a live object keeps its one
/// wrapper).
struct Tables<T, V> {
    name: &'static str,
    entries: HashMap<usize, (std::sync::Weak<T>, V)>,
    hits: u64,
    misses: u64,
}

impl<T, V> Tables<T, V> {
    fn new(name: &'static str) -> Tables<T, V> {
        Tables {
            name,
            entries: HashMap::new(),
            hits: 0,
            misses: 0,
        }
    }

    fn get<R>(&mut self, object: &Arc<T>, read: impl FnOnce(&V) -> R) -> Option<R> {
        let found = self
            .entries
            .get(&(Arc::as_ptr(object) as usize))
            .and_then(|(alive, value)| {
                alive
                    .upgrade()
                    .filter(|found| Arc::ptr_eq(found, object))
                    .map(|_| read(value))
            });
        match found {
            Some(_) => self.hits += 1,
            None => self.misses += 1,
        }
        found
    }

    /// Remember `value` for `object`: the value a dead object at the same address left, for the caller to drop
    /// outside the lock, and whether the table is due for a purge.
    fn insert(&mut self, object: &Arc<T>, value: V) -> (Option<V>, bool) {
        let old = self
            .entries
            .insert(
                Arc::as_ptr(object) as usize,
                (Arc::downgrade(object), value),
            )
            .map(|(_, old)| old);
        (old, self.entries.len().is_multiple_of(PURGE_EVERY))
    }

    /// The values of every entry whose object is gone, removed from the table.
    fn dead(&mut self) -> Vec<V> {
        self.entries
            .extract_if(|_, (alive, _)| alive.strong_count() == 0)
            .map(|(_, (_, value))| value)
            .collect()
    }
}

/// Drop the dead entries of `tables`, outside its lock: a value may hold Python objects whose finalizers come back
/// here, and the objects it held may have kept other entries alive, so the purge runs again while it frees any.
fn purge<T, V>(table: &Table<T, V>) {
    loop {
        let dead = table.lock().dead();
        if dead.is_empty() {
            return;
        }
        drop(dead);
    }
}

/// A table behind its lock: what the statics hold and the memo registry lists.
struct Table<T, V>(Mutex<Tables<T, V>>);

impl<T, V> Table<T, V> {
    fn lock(&self) -> std::sync::MutexGuard<'_, Tables<T, V>> {
        emmy_ir::locked(&self.0)
    }
}

impl<T: Send + Sync, V: Send> Store for Table<T, V> {
    fn stats(&self) -> Stats {
        let tables = self.lock();
        Stats {
            name: tables.name,
            entries: tables.entries.len(),
            capacity: None,
            hits: tables.hits,
            misses: tables.misses,
        }
    }

    fn clear(&self) {
        purge(self);
    }
}

/// The one table under `cell`, built under `name` on first use and listed in the memo registry once.
fn tables<T: Send + Sync + 'static, V: Send + 'static>(
    cell: &'static OnceLock<Table<T, V>>,
    listed: &'static Once,
    name: &'static str,
) -> &'static Table<T, V> {
    emmy_ir::memo::declared(cell, listed, || Table(Mutex::new(Tables::new(name))))
}

static WRAPPERS: OnceLock<Table<emmy_ir::body::BodyData, Wrappers>> = OnceLock::new();
static WRAPPERS_LISTED: Once = Once::new();

fn wrapper_tables() -> &'static Table<emmy_ir::body::BodyData, Wrappers> {
    tables(&WRAPPERS, &WRAPPERS_LISTED, "statement_wrappers")
}

fn wrappers(body: &Body) -> Wrappers {
    let mut tables = wrapper_tables().lock();
    if let Some(table) = tables.get(&body.0, Wrappers::clone) {
        return table;
    }
    let table: Wrappers = (0..body.len()).map(|_| OnceLock::new()).collect();
    let (old, due) = tables.insert(&body.0, table.clone());
    drop(tables);
    drop(old);
    if due {
        purge(wrapper_tables());
    }
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
            .map_err(|err| located(value.py(), where_, translate(err)))?;
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

    /// Recursive 1:N body transformer. Post-order: each block statement's nested bodies are mapped first, then
    /// `function` is applied to the children-rewritten statement, which it keeps (a statement), drops (`None`) or
    /// replaces (an iterable of statements, spliced in as they are). A statement the walk did not rebuild is handed
    /// over as the object this body hands out, and the objects `function` returns are the ones the result hands out.
    fn map<'py>(
        &self,
        py: Python<'py>,
        function: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyBody>> {
        let placed: HashMap<usize, StmtRef> = self
            .body
            .iter_refs()
            .into_iter()
            .map(|here| (Arc::as_ptr(here.stmt()) as usize, here))
            .collect();
        let mut returned: HashMap<usize, Py<PyAny>> = HashMap::new();
        let mapped = self
            .body
            .try_map(&mut |stmt| -> PyResult<Option<Vec<Arc<Statement>>>> {
                let given = match placed.get(&(Arc::as_ptr(stmt) as usize)) {
                    Some(here) => wrap_ref(py, here)?,
                    None => wrap_statement(py, stmt)?.into_bound(py),
                };
                let result = function.call1((given,))?;
                if result.is_none() {
                    return Ok(None);
                }
                let items: Vec<Bound<'py, PyAny>> = if result.is_instance_of::<PyStatement>() {
                    vec![result]
                } else {
                    result.try_iter()?.collect::<PyResult<_>>()?
                };
                let stmts: PyResult<Vec<Arc<Statement>>> =
                    items.iter().map(statement_from_py).collect();
                let stmts = stmts?;
                for (stmt, item) in stmts.iter().zip(items) {
                    returned.insert(Arc::as_ptr(stmt) as usize, item.unbind());
                }
                Ok(Some(stmts))
            })?;
        for here in mapped.iter_refs() {
            if let Some(item) = returned.get(&(Arc::as_ptr(here.stmt()) as usize))
                && let Some(holder) = here.body()
                && let Some(slot) = wrappers(&holder).get(here.index)
            {
                let _ = slot.set(item.clone_ref(py));
            }
        }
        PyBody::wrap(py, &mapped)
    }

    /// Matching statements in pre-order.
    #[pyo3(signature = (*types))]
    fn iter_of_type<'py>(
        &self,
        py: Python<'py>,
        types: &Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let mut kinds: Vec<emmy_ir::kinds::Kind> = Vec::new();
        for item in types.iter() {
            let cls = item
                .cast::<PyType>()
                .map_err(|_| PyTypeError::new_err("iter_of_type takes statement classes"))?;
            match kind_of(cls) {
                Some(kind) => kinds.push(kind),
                None if cls.is(py.get_type::<PyStatement>()) => {
                    kinds.extend(emmy_ir::kinds::Kind::ALL.iter().copied())
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
        let identity =
            emmy_ir::identity::canonicalize_identity(&self.body, cluster, colors.as_ref()).py()?;
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
static SUBROUTINES: OnceLock<Table<Subroutine, Py<PyAny>>> = OnceLock::new();
static SUBROUTINES_LISTED: Once = Once::new();

fn subroutines() -> &'static Table<Subroutine, Py<PyAny>> {
    tables(&SUBROUTINES, &SUBROUTINES_LISTED, "subroutine_wrappers")
}

impl PySubroutine {
    /// The one Python object of `sub` while any reference to it lives.
    pub fn wrap<'py>(py: Python<'py>, sub: &Arc<Subroutine>) -> PyResult<Bound<'py, PySubroutine>> {
        let remembered = subroutines()
            .lock()
            .get(sub, |reference| reference.clone_ref(py));
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
        let (old, due) = subroutines().lock().insert(sub, reference.unbind());
        drop(old);
        if due {
            purge(subroutines());
        }
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
        emmy_ir::order::ordering_constraints(&body, effects, redefinitions)
            .iter()
            .map(|sources| PySet::new(py, sources.iter()))
            .collect();
    PyList::new(py, sets?)
}

/// Promote an elementwise op's argument dtypes to one result dtype.
#[pyfunction]
pub fn dtype_promote(op_name: &str, arg_dtypes: Vec<String>) -> PyResult<String> {
    let dtypes: Vec<emmy_ir::kinds::field::DType> = arg_dtypes
        .iter()
        .map(|d| d.parse().py())
        .collect::<PyResult<_>>()?;
    let promoted = emmy_ir::kinds::dtype::promote(
        emmy_ir::kinds::elementwise::Op::new(op_name).known(),
        &dtypes,
    );
    Ok(promoted.text().to_string())
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
    m.add_function(wrap_pyfunction!(free_names_py, m)?)?;
    m.add_function(wrap_pyfunction!(exposed_defines_py, m)?)?;
    m.add_function(wrap_pyfunction!(rename_free, m)?)?;
    m.add_function(wrap_pyfunction!(binary, m)?)?;
    m.add_function(wrap_pyfunction!(mask_select_predicate, m)?)?;
    m.add_function(wrap_pyfunction!(ordering_constraints, m)?)?;
    m.add_function(wrap_pyfunction!(dtype_promote, m)?)?;
    m.add_function(wrap_pyfunction!(clear_memos, m)?)?;
    m.add_function(wrap_pyfunction!(clear_memo, m)?)?;
    m.add_function(wrap_pyfunction!(memo_stats, m)?)?;
    Ok(())
}

/// Clear the one memo `name` names (a `name` of `memo_stats()`); `KeyError` for a name no memo has.
#[pyfunction]
fn clear_memo(name: &str) -> PyResult<()> {
    if emmy_ir::memo::clear(name) {
        Ok(())
    } else {
        Err(pyo3::exceptions::PyKeyError::new_err(format!(
            "no memo {name}"
        )))
    }
}

/// Forget every memoized normal form and identity and purge the wrapper tables: what a test worker or a process
/// pool does to start clean.
#[pyfunction]
fn clear_memos() {
    emmy_ir::memo::clear_all();
}

/// Every memo's figures: a list of dicts with `name`, `entries`, `capacity` (None for an unbounded table), `hits`
/// and `misses`.
#[pyfunction]
fn memo_stats<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for stats in emmy_ir::memo::stats() {
        let item = PyDict::new(py);
        item.set_item("name", stats.name)?;
        item.set_item("entries", stats.entries)?;
        item.set_item("capacity", stats.capacity)?;
        item.set_item("hits", stats.hits)?;
        item.set_item("misses", stats.misses)?;
        out.append(item)?;
    }
    Ok(out)
}
