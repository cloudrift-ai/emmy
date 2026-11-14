//! `Lambda` as a Python class: a handle on the crate's lambda. Python reads its params, body and results and asks
//! whole questions (`rename`, `canonical`, `cone`, `components`); nothing walks it statement by statement across the
//! boundary.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use emmy_ir::Name;
use emmy_ir::elementwise::ElementwiseOp;
use emmy_pure::Lambda;
use emmy_ir::text::body_lines;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PyList, PyString, PyTuple, PyType};

use crate::body::PyBody;
use crate::convert::{ELEMENTWISE, IntoPyErr, located, name_from_py, op_to_py, str_tuple};
use crate::expression::renames_from_py;

/// The one binder kind, shared by every IR level: explicit binders over the pure statement vocabulary.
#[pyclass(frozen, module = "emmy.compiler.ir.pure", name = "Lambda")]
pub struct PyLambda {
    pub lambda: Lambda,
    params: OnceLock<Py<PyTuple>>,
    body: OnceLock<Py<PyBody>>,
    results: OnceLock<Py<PyTuple>>,
}

impl PyLambda {
    pub fn new(lambda: Lambda) -> PyLambda {
        PyLambda {
            lambda,
            params: OnceLock::new(),
            body: OnceLock::new(),
            results: OnceLock::new(),
        }
    }

    pub fn wrap<'py>(py: Python<'py>, lambda: Lambda) -> PyResult<Bound<'py, PyLambda>> {
        Bound::new(py, PyLambda::new(lambda))
    }

    /// The lambda a Python value holds.
    pub fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Lambda> {
        value
            .cast::<PyLambda>()
            .map(|lam| lam.get().lambda.clone())
            .map_err(|_| PyTypeError::new_err("expected a Lambda"))
    }
}

fn names(value: &Bound<'_, PyAny>) -> PyResult<Vec<Name>> {
    if value.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err(
            "expected a sequence of names, got a str",
        ));
    }
    value.try_iter()?.map(|item| name_from_py(&item?)).collect()
}

/// A mapping as it is, or a `str -> str` function asked once per name the lambda spells.
fn mapping(lambda: &Lambda, names: &Bound<'_, PyAny>) -> PyResult<HashMap<Name, Name>> {
    if names.cast::<PyDict>().is_ok() {
        return renames_from_py(names);
    }
    if names.hasattr("get")? {
        return renames_from_py(&PyDict::from_sequence(&names.call_method0("items")?)?.into_any());
    }
    let py = names.py();
    let mut out = HashMap::new();
    for name in lambda.names() {
        out.insert(
            name,
            name_from_py(&names.call1((PyString::new(py, &name),))?)?,
        );
    }
    Ok(out)
}

/// An op as the elementwise table spells it: a name through the Python class (which takes the canonical spelling),
/// an op object by its name.
fn op_from_py(value: &Bound<'_, PyAny>) -> PyResult<ElementwiseOp> {
    let op = match value.cast::<PyString>() {
        Ok(text) => ELEMENTWISE.get(value.py())?.call1((text,))?,
        Err(_) => value.clone(),
    };
    Ok(ElementwiseOp::new(
        &op.getattr("name")?.extract::<String>()?,
    ))
}

#[pymethods]
impl PyLambda {
    #[new]
    #[pyo3(signature = (params, body, results))]
    fn py_new(
        params: &Bound<'_, PyAny>,
        body: &Bound<'_, PyAny>,
        results: &Bound<'_, PyAny>,
    ) -> PyResult<PyLambda> {
        let given = PyBody::from_iterable(body)?;
        let lambda = Lambda::new(names(params)?, &given, names(results)?).py()?;
        let out = PyLambda::new(lambda);
        // A body already in its normal form is the object it was given as.
        if let Ok(passed) = body.cast::<PyBody>()
            && *out.lambda.body() == given
        {
            let _ = out.body.set(passed.clone().unbind());
        }
        Ok(out)
    }

    #[classattr]
    fn wire_tag() -> &'static str {
        "Lambda"
    }

    #[getter]
    fn params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        if let Some(held) = self.params.get() {
            return Ok(held.bind(py).clone());
        }
        let tuple = str_tuple(py, self.lambda.params())?;
        Ok(self
            .params
            .get_or_init(|| tuple.clone().unbind())
            .bind(py)
            .clone())
    }

    #[getter]
    fn body<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBody>> {
        if let Some(held) = self.body.get() {
            return Ok(held.bind(py).clone());
        }
        let body = PyBody::wrap(py, self.lambda.body())?;
        Ok(self
            .body
            .get_or_init(|| body.clone().unbind())
            .bind(py)
            .clone())
    }

    #[getter]
    fn results<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        if let Some(held) = self.results.get() {
            return Ok(held.bind(py).clone());
        }
        let tuple = str_tuple(py, self.lambda.results())?;
        Ok(self
            .results
            .get_or_init(|| tuple.clone().unbind())
            .bind(py)
            .clone())
    }

    /// Build a CLOSED lambda from its parts: whatever the body still reads, and any result it does not define, is
    /// appended as a trailing param, so the positional prefix the caller gave stays intact.
    #[classmethod]
    fn closing<'py>(
        _cls: &Bound<'py, PyType>,
        params: &Bound<'py, PyAny>,
        body: &Bound<'py, PyAny>,
        results: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyLambda>> {
        let lambda = Lambda::closing(
            names(params)?,
            &PyBody::from_iterable(body)?,
            names(results)?,
        )
        .py()?;
        PyLambda::wrap(params.py(), lambda)
    }

    /// The componentwise binary program `nᵢ = ⊕ᵢ(nᵢ, nᵢ__o)`: a plain fold's combine, the shape `components` reads.
    #[classmethod]
    fn componentwise<'py>(
        _cls: &Bound<'py, PyType>,
        ops: &Bound<'py, PyAny>,
        names_: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyLambda>> {
        let ops: Vec<ElementwiseOp> = ops
            .try_iter()?
            .map(|op| op_from_py(&op?))
            .collect::<PyResult<_>>()?;
        PyLambda::wrap(
            names_.py(),
            Lambda::componentwise(&ops, &names(names_)?).py()?,
        )
    }

    /// The per-result ops when this program is componentwise, else `None`.
    fn components<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        match self.lambda.components() {
            None => Ok(None),
            Some(ops) => {
                let items: Vec<Bound<'py, PyAny>> = ops
                    .iter()
                    .map(|op| op_to_py(py, op))
                    .collect::<PyResult<_>>()?;
                Ok(Some(PyTuple::new(py, items)?))
            }
        }
    }

    /// Every name this lambda binds: params plus every definition its body exposes, deep.
    #[getter]
    fn defined<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
        let defined: HashSet<Name> = self.lambda.defined();
        PyFrozenSet::new(py, defined.iter().map(|name| PyString::new(py, name)))
    }

    /// The closed cone of one definition, as a lambda over what it reads; a param names itself.
    fn cone<'py>(
        &self,
        py: Python<'py>,
        name: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyLambda>> {
        PyLambda::wrap(py, self.lambda.cone(name_from_py(name)?).py()?)
    }

    /// α-rename params, body and results through one map — a mapping or a function — in lockstep.
    fn rename<'py>(
        slf: &Bound<'py, Self>,
        names_: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyLambda>> {
        let this = slf.get();
        let mapping = mapping(&this.lambda, names_)?;
        let renamed = this.lambda.rename(&mapping).py()?;
        if renamed.ptr_eq(&this.lambda) {
            return Ok(slf.clone());
        }
        PyLambda::wrap(slf.py(), renamed)
    }

    /// The α-canonical form: params `_p0…`, the body's own definitions `_v0…` in walk order, free names unchanged.
    fn canonical<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyLambda>> {
        PyLambda::wrap(py, self.lambda.canonical().py()?)
    }

    /// α-invariant equality: canonical forms compared structurally.
    fn alpha_eq(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        match other.cast::<PyLambda>() {
            Ok(other) => self.lambda.alpha_eq(&other.get().lambda).py(),
            Err(_) => Ok(false),
        }
    }

    /// The lambda with some of its parts replaced, formed and checked like any other.
    #[pyo3(signature = (**changes))]
    fn replace<'py>(
        slf: &Bound<'py, Self>,
        changes: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyLambda>> {
        let py = slf.py();
        let part = |name: &str| -> PyResult<Bound<'py, PyAny>> {
            match changes.map(|c| c.get_item(name)).transpose()?.flatten() {
                Some(value) => Ok(value),
                None => slf.getattr(name),
            }
        };
        if let Some(changes) = changes {
            for key in changes.keys() {
                let key: String = key.extract()?;
                if !matches!(key.as_str(), "params" | "body" | "results") {
                    return Err(PyTypeError::new_err(format!("Lambda has no field {key:?}")));
                }
            }
        }
        Bound::new(
            py,
            PyLambda::py_new(&part("params")?, &part("body")?, &part("results")?)?,
        )
    }

    fn pretty(&self, indent: Option<&str>) -> PyResult<Vec<String>> {
        let indent = indent.unwrap_or("");
        let mut out = vec![format!(
            "{indent}λ({}) -> ({})",
            self.lambda
                .params()
                .iter()
                .map(|n| n.as_ref())
                .collect::<Vec<_>>()
                .join(", "),
            self.lambda
                .results()
                .iter()
                .map(|n| n.as_ref())
                .collect::<Vec<_>>()
                .join(", "),
        )];
        body_lines(self.lambda.body(), &format!("{indent}    "), &mut out).py()?;
        Ok(out)
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<PyLambda>()
            .is_ok_and(|other| other.get().lambda == self.lambda)
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.lambda.hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(format!(
            "Lambda(params={}, body={}, results={})",
            slf.getattr("params")?.repr()?,
            slf.getattr("body")?.repr()?,
            slf.getattr("results")?.repr()?
        ))
    }

    /// Transport by the stored parts; the far side forms the lambda without checking it again.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let module = py.import("emmy.emmy_ir")?;
        let parts = PyTuple::new(
            py,
            [
                slf.getattr("params")?,
                slf.getattr("body")?.into_any(),
                slf.getattr("results")?,
            ],
        )?;
        Ok((module.getattr("_formed_lambda")?, parts))
    }

    fn to_wire<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDict>> {
        let py = slf.py();
        let out = PyDict::new(py);
        out.set_item(
            "params",
            PyList::new(
                py,
                slf.get()
                    .lambda
                    .params()
                    .iter()
                    .map(|n| PyString::new(py, n)),
            )?,
        )?;
        out.set_item("body", slf.getattr("body")?.call_method0("to_wire")?)?;
        out.set_item(
            "results",
            PyList::new(
                py,
                slf.get()
                    .lambda
                    .results()
                    .iter()
                    .map(|n| PyString::new(py, n)),
            )?,
        )?;
        Ok(out)
    }

    #[classmethod]
    #[pyo3(signature = (value, where_ = None))]
    fn from_wire<'py>(
        cls: &Bound<'py, PyType>,
        value: &Bound<'py, PyAny>,
        where_: Option<&str>,
    ) -> PyResult<Bound<'py, PyLambda>> {
        let py = cls.py();
        let where_ = where_.unwrap_or("Lambda");
        let read = || -> PyResult<PyLambda> {
            let value = value.cast::<PyDict>()?;
            let get = |name: &str| -> PyResult<Bound<'py, PyAny>> {
                value
                    .get_item(name)?
                    .ok_or_else(|| PyTypeError::new_err(format!("missing {name}")))
            };
            let text: String = get("body")?.extract()?;
            let body = PyBody::wrap(py, &emmy_ir::text::parse_body(&text).py()?)?;
            PyLambda::py_new(&get("params")?, &body.into_any(), &get("results")?)
        };
        Bound::new(py, read().map_err(|err| located(py, where_, err))?)
    }

    /// The structural form: the class name over its three parts' forms, what an identity built over terms reads.
    fn structural_form<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        PyTuple::new(
            py,
            [
                PyString::new(py, "Lambda").into_any(),
                slf.getattr("params")?,
                slf.getattr("body")?.call_method0("structural_form")?,
                slf.getattr("results")?,
            ],
        )
    }

    /// The keyword arguments this lambda is rebuilt from: what a generic rewrite goes into.
    fn wire_parts<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(slf.py());
        for name in ["params", "body", "results"] {
            out.set_item(name, slf.getattr(name)?)?;
        }
        Ok(out)
    }
}

/// A lambda from its stored parts, its body taken as normal: what transport reads back.
#[pyfunction]
fn _formed_lambda<'py>(
    params: &Bound<'py, PyAny>,
    body: &Bound<'py, PyAny>,
    results: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyLambda>> {
    let lambda = Lambda::formed(
        names(params)?,
        PyBody::from_iterable(body)?,
        names(results)?,
    );
    let out = PyLambda::new(lambda);
    if let Ok(passed) = body.cast::<PyBody>() {
        let _ = out.body.set(passed.clone().unbind());
    }
    Bound::new(params.py(), out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLambda>()?;
    m.add_function(wrap_pyfunction!(_formed_lambda, m)?)?;
    Ok(())
}
