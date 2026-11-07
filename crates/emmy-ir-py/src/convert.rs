//! How a field crosses the boundary: every field type converts to the Python object the compiler reads and back.
//! A statement built from Python keeps the objects it was built from where they are already canonical; one built
//! in Rust makes them on first read.

use std::sync::{Arc, OnceLock};

use emmy_ir::axis::{Axis, Dim, Window};
use emmy_ir::body::Body;
use emmy_ir::dtype::DType;
use emmy_ir::elementwise::ElementwiseOp;
use emmy_ir::expr::{Expression, Num};
use emmy_ir::field::{Float, Int2, NumOr, Shape3, SubroutineRef};
use emmy_ir::statement::{FragmentArg, SelectBranch, Statement};
use emmy_ir::{Error, Name};
use pyo3::exceptions::{
    PyException, PyKeyError, PyNotImplementedError, PyRecursionError, PyTypeError, PyValueError,
    PyZeroDivisionError,
};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyFloat, PyInt, PyString, PyTuple};

use crate::body::{PyBody, PySubroutine};
use crate::expression::PyExpression;
use crate::statement::{
    FragmentArg as PyFragmentArg, PyStatement, SelectBranch as PySelectBranch, wrap_statement,
};

pub fn translate(error: Error) -> PyErr {
    match error {
        Error::Value(m) => PyValueError::new_err(m),
        Error::Type(m) => PyTypeError::new_err(m),
        Error::Key(m) => PyKeyError::new_err(m),
        Error::ZeroDivision => PyZeroDivisionError::new_err("integer division or modulo by zero"),
        Error::NotImplemented(m) => PyNotImplementedError::new_err(m),
        Error::Recursion(m) => PyRecursionError::new_err(m),
    }
}

/// `err` with `where_` ahead of its message, of its own type (a `ValueError` stays one); an exception that is no
/// `Exception`, a `KeyboardInterrupt`, passes through untouched.
pub fn located(py: Python<'_>, where_: &str, err: PyErr) -> PyErr {
    if !err.is_instance_of::<PyException>(py) {
        return err;
    }
    let message = err.value(py).to_string();
    PyErr::from_type(err.get_type(py), format!("{where_}: {message}"))
}

pub trait IntoPyErr<T> {
    fn py(self) -> PyResult<T>;
}

impl<T> IntoPyErr<T> for emmy_ir::Result<T> {
    fn py(self) -> PyResult<T> {
        self.map_err(translate)
    }
}

/// A Python class or function this extension calls to make the compiler's own objects (an axis, a dtype, an op).
pub struct Imported(PyOnceLock<Py<PyAny>>, &'static str, &'static str);

impl Imported {
    pub const fn new(module: &'static str, name: &'static str) -> Imported {
        Imported(PyOnceLock::new(), module, name)
    }

    pub fn get<'py>(&self, py: Python<'py>) -> PyResult<&Bound<'py, PyAny>> {
        let value: &Py<PyAny> = self.0.get_or_try_init(py, || {
            Ok::<_, PyErr>(py.import(self.1)?.getattr(self.2)?.unbind())
        })?;
        Ok(value.bind(py))
    }
}

pub static AXIS: Imported = Imported::new("emmy.compiler.ir.axis", "Axis");
pub static WINDOW: Imported = Imported::new("emmy.compiler.ir.axis", "Window");
pub static DIM: Imported = Imported::new("emmy.compiler.dim", "Dim");
pub static DTYPE_GET: Imported = Imported::new("emmy.compiler.dtype", "get");
pub static ELEMENTWISE: Imported = Imported::new("emmy.compiler.ir.elementwise", "ElementwiseImpl");
pub static PARSE: Imported = Imported::new("emmy.emmy_ir", "parse");
pub static STATEMENT_FROM_TEXT: Imported = Imported::new("emmy.emmy_ir", "_statement_from_text");

/// The value a field holds, and the Python object to hold beside it when the one given is already canonical.
pub type Extracted<T> = (T, Option<Py<PyAny>>);

pub fn str_tuple<'py>(py: Python<'py>, names: &[Name]) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(py, names.iter().map(|name| PyString::new(py, name)))
}

pub fn num_to_py(py: Python<'_>, value: Num) -> PyResult<Py<PyAny>> {
    match value {
        Num::Int(v) => Ok(v.into_pyobject(py)?.into_any().unbind()),
        Num::Float(v) => Ok(PyFloat::new(py, v).into_any().unbind()),
        Num::Bool(v) => Ok(PyBool::new(py, v).to_owned().into_any().unbind()),
    }
}

/// The name of `value`'s type, for an error message.
pub fn type_name(value: &Bound<'_, PyAny>) -> String {
    value
        .get_type()
        .name()
        .map_or_else(|_| "?".to_string(), |name| name.to_string())
}

pub fn num_from_py(value: &Bound<'_, PyAny>) -> PyResult<Num> {
    if let Ok(flag) = value.cast::<PyBool>() {
        return Ok(Num::Bool(flag.is_true()));
    }
    if value.is_instance_of::<PyInt>() {
        return value.extract::<i64>().map(Num::Int);
    }
    if let Ok(number) = value.cast::<PyFloat>() {
        return Ok(Num::Float(number.value()));
    }
    Err(PyTypeError::new_err(format!(
        "expected a number, got {}",
        value.get_type().name()?
    )))
}

/// The expression `value` holds; a bare number is a literal of its own type: an integer stays an integer (a strided
/// loop's `step=4` spells `step 4`), a float is a float literal.
pub fn expression_from_py(value: &Bound<'_, PyAny>) -> PyResult<Expression> {
    if let Ok(expression) = value.cast::<PyExpression>() {
        return Ok(expression.get().expr.clone());
    }
    if !value.is_instance_of::<PyBool>() && value.is_instance_of::<PyInt>() {
        return Ok(Expression::int(value.extract::<i64>()?));
    }
    if value.is_instance_of::<PyFloat>() {
        return Ok(Expression::float(num_from_py(value)?.as_f64()));
    }
    Err(PyTypeError::new_err(format!(
        "expected an Expression, got {}",
        value.get_type().name()?
    )))
}

fn expression_from_any(value: &Bound<'_, PyAny>) -> PyResult<Expression> {
    match value.cast::<PyExpression>() {
        Ok(expression) => Ok(expression.get().expr.clone()),
        Err(_) => Err(PyTypeError::new_err(format!(
            "expected an Expression, got {}",
            type_name(value)
        ))),
    }
}

pub fn dim_from_py(value: &Bound<'_, PyAny>) -> PyResult<Dim> {
    let expression = expression_from_any(&value.getattr("expression")?)?;
    let hint: Option<i64> = value.getattr("hint")?.extract()?;
    Ok(Dim { expression, hint })
}

pub fn axis_from_py(value: &Bound<'_, PyAny>, depth: usize) -> PyResult<Arc<Axis>> {
    if depth > 64 {
        return Err(PyValueError::new_err("an axis's window chain does not end"));
    }
    let name: String = value.getattr("name")?.extract()?;
    let extent = dim_from_py(&value.getattr("extent")?)?;
    let window = value.getattr("window")?;
    let window = if window.is_none() {
        None
    } else {
        let parent = window.getattr("parent")?;
        let base = window.getattr("base")?;
        let bound = window.getattr("bound")?;
        Some(Window {
            parent: if parent.is_none() {
                None
            } else {
                Some(axis_from_py(&parent, depth + 1)?)
            },
            base: if base.is_none() {
                None
            } else {
                Some(expression_from_any(&base)?)
            },
            bound: if bound.is_none() {
                None
            } else {
                Some(expression_from_any(&bound)?)
            },
            partition: window.getattr("partition")?.extract()?,
        })
    };
    Ok(Arc::new(Axis {
        name: name.into(),
        extent,
        window,
    }))
}

pub fn dim_to_py<'py>(py: Python<'py>, dim: &Dim) -> PyResult<Bound<'py, PyAny>> {
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("hint", dim.hint)?;
    DIM.get(py)?
        .call((PyExpression::wrap(py, &dim.expression)?,), Some(&kwargs))
}

pub fn axis_to_py<'py>(py: Python<'py>, axis: &Axis) -> PyResult<Bound<'py, PyAny>> {
    let window = match &axis.window {
        None => py.None().into_bound(py),
        Some(window) => {
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item(
                "parent",
                window
                    .parent
                    .as_ref()
                    .map(|parent| axis_to_py(py, parent))
                    .transpose()?,
            )?;
            kwargs.set_item(
                "base",
                window
                    .base
                    .as_ref()
                    .map(|e| PyExpression::wrap(py, e))
                    .transpose()?,
            )?;
            kwargs.set_item(
                "bound",
                window
                    .bound
                    .as_ref()
                    .map(|e| PyExpression::wrap(py, e))
                    .transpose()?,
            )?;
            kwargs.set_item("partition", window.partition)?;
            WINDOW.get(py)?.call((), Some(&kwargs))?
        }
    };
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("window", window)?;
    AXIS.get(py)?.call(
        (PyString::new(py, &axis.name), dim_to_py(py, &axis.extent)?),
        Some(&kwargs),
    )
}

pub fn dtype_to_py<'py>(py: Python<'py>, dtype: &DType) -> PyResult<Bound<'py, PyAny>> {
    DTYPE_GET.get(py)?.call1((dtype.text(),))
}

pub fn op_to_py<'py>(py: Python<'py>, op: &ElementwiseOp) -> PyResult<Bound<'py, PyAny>> {
    ELEMENTWISE.get(py)?.call1((op.name(),))
}

fn is_tuple_of_str(value: &Bound<'_, PyAny>) -> bool {
    value.cast::<PyTuple>().is_ok_and(|tuple| {
        tuple
            .iter()
            .all(|item| item.is_exact_instance_of::<PyString>())
    })
}

fn items<'py>(value: &Bound<'py, PyAny>, what: &str) -> PyResult<Vec<Bound<'py, PyAny>>> {
    if value.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err(format!("expected {what}, got a str")));
    }
    value.try_iter()?.collect()
}

/// What every field type does at the boundary.
pub trait PyField: Sized {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>>;
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Extracted<Self>>;
}

/// A Python string as a name, interned without a copy.
pub fn name_from_py(value: &Bound<'_, PyAny>) -> PyResult<Name> {
    let text = value
        .cast::<PyString>()
        .map_err(|_| PyTypeError::new_err(format!("expected a name, got {}", type_name(value))))?;
    Ok(Name::new(text.to_str()?))
}

impl PyField for Name {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyString::new(py, self).into_any().unbind())
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Extracted<Self>> {
        let text = value.cast::<PyString>().map_err(|_| {
            PyTypeError::new_err(format!("expected a name, got {}", type_name(value)))
        })?;
        Ok((
            text.to_str()?.into(),
            value
                .is_exact_instance_of::<PyString>()
                .then(|| value.clone().unbind()),
        ))
    }
}

impl PyField for Arc<[Name]> {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(str_tuple(py, self)?.into_any().unbind())
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Extracted<Self>> {
        let names: PyResult<Vec<Name>> = items(value, "a tuple of names")?
            .iter()
            .map(|item| Name::from_py(item).map(|(name, _)| name))
            .collect();
        Ok((
            names?.into(),
            is_tuple_of_str(value).then(|| value.clone().unbind()),
        ))
    }
}

impl<T: PyField> PyField for Option<T> {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            None => Ok(py.None()),
            Some(value) => value.to_py(py),
        }
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Extracted<Self>> {
        if value.is_none() {
            return Ok((None, Some(value.clone().unbind())));
        }
        let (inner, held) = T::from_py(value)?;
        Ok((Some(inner), held))
    }
}

impl<A: PyField, B: PyField> PyField for (A, B) {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyTuple::new(py, [self.0.to_py(py)?, self.1.to_py(py)?])?
            .into_any()
            .unbind())
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Extracted<Self>> {
        let parts = items(value, "a pair")?;
        if parts.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "expected a pair, got {} items",
                parts.len()
            )));
        }
        let (a, held_a) = A::from_py(&parts[0])?;
        let (b, held_b) = B::from_py(&parts[1])?;
        let held =
            (held_a.is_some() && held_b.is_some() && value.is_exact_instance_of::<PyTuple>())
                .then(|| value.clone().unbind());
        Ok(((a, b), held))
    }
}

impl PyField for Expression {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyExpression::wrap(py, self)?.unbind())
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Extracted<Self>> {
        // A bare number is coerced to a literal; the literal, not the number, is what a read hands back.
        let held = value
            .is_instance_of::<PyExpression>()
            .then(|| value.clone().unbind());
        Ok((expression_from_py(value)?, held))
    }
}

impl PyField for Arc<[Expression]> {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let wrapped: PyResult<Vec<Py<PyAny>>> = self.iter().map(|e| e.to_py(py)).collect();
        Ok(PyTuple::new(py, wrapped?)?.into_any().unbind())
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Extracted<Self>> {
        let parts = items(value, "a tuple of expressions")?;
        let expressions: PyResult<Vec<Expression>> = parts.iter().map(expression_from_py).collect();
        Ok((
            expressions?.into(),
            value
                .is_exact_instance_of::<PyTuple>()
                .then(|| value.clone().unbind()),
        ))
    }
}

impl PyField for Arc<Axis> {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(axis_to_py(py, self)?.unbind())
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Extracted<Self>> {
        Ok((axis_from_py(value, 0)?, Some(value.clone().unbind())))
    }
}

impl PyField for Arc<[Arc<Axis>]> {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let wrapped: PyResult<Vec<Py<PyAny>>> = self.iter().map(|a| a.to_py(py)).collect();
        Ok(PyTuple::new(py, wrapped?)?.into_any().unbind())
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Extracted<Self>> {
        let parts = items(value, "a tuple of axes")?;
        let axes: PyResult<Vec<Arc<Axis>>> =
            parts.iter().map(|item| axis_from_py(item, 0)).collect();
        Ok((
            axes?.into(),
            value
                .is_exact_instance_of::<PyTuple>()
                .then(|| value.clone().unbind()),
        ))
    }
}

impl PyField for Body {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyBody::wrap(py, self)?.into_any().unbind())
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Extracted<Self>> {
        if let Ok(body) = value.cast::<PyBody>() {
            return Ok((body.get().body.clone(), Some(value.clone().unbind())));
        }
        Ok((PyBody::from_iterable(value)?, None))
    }
}

impl PyField for Arc<[Arc<Statement>]> {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let wrapped: PyResult<Vec<Py<PyAny>>> =
            self.iter().map(|s| wrap_statement(py, s)).collect();
        Ok(PyTuple::new(py, wrapped?)?.into_any().unbind())
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Extracted<Self>> {
        let parts = items(value, "a tuple of statements")?;
        let stmts: PyResult<Vec<Arc<Statement>>> = parts.iter().map(statement_from_py).collect();
        Ok((
            stmts?.into(),
            value
                .is_exact_instance_of::<PyTuple>()
                .then(|| value.clone().unbind()),
        ))
    }
}

pub fn statement_from_py(value: &Bound<'_, PyAny>) -> PyResult<Arc<Statement>> {
    match value.cast::<PyStatement>() {
        Ok(stmt) => Ok(stmt.get().stmt.clone()),
        Err(_) => Err(PyTypeError::new_err(format!(
            "Body holds non-statement member(s) ['{}']; a term composes through operand edges",
            value.get_type().name()?
        ))),
    }
}

impl PyField for Arc<[SelectBranch]> {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let wrapped: PyResult<Vec<Py<PyAny>>> = self
            .iter()
            .map(|b| Ok(Py::new(py, PySelectBranch::new_from(b.clone()))?.into_any()))
            .collect();
        Ok(PyTuple::new(py, wrapped?)?.into_any().unbind())
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Extracted<Self>> {
        let parts = items(value, "a tuple of branches")?;
        let branches: PyResult<Vec<SelectBranch>> = parts
            .iter()
            .map(|item| {
                item.cast::<PySelectBranch>()
                    .map(|b| b.get().record.clone())
                    .map_err(|_| {
                        PyTypeError::new_err(format!(
                            "expected a SelectBranch, got {}",
                            type_name(item)
                        ))
                    })
            })
            .collect();
        Ok((
            branches?.into(),
            value
                .is_exact_instance_of::<PyTuple>()
                .then(|| value.clone().unbind()),
        ))
    }
}

impl PyField for Arc<[FragmentArg]> {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let wrapped: PyResult<Vec<Py<PyAny>>> = self
            .iter()
            .map(|a| Ok(Py::new(py, PyFragmentArg::new_from(a.clone()))?.into_any()))
            .collect();
        Ok(PyTuple::new(py, wrapped?)?.into_any().unbind())
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Extracted<Self>> {
        let parts = items(value, "a tuple of fragment operands")?;
        let args: PyResult<Vec<FragmentArg>> = parts
            .iter()
            .map(|item| {
                item.cast::<PyFragmentArg>()
                    .map(|a| a.get().record.clone())
                    .map_err(|_| {
                        PyTypeError::new_err(format!(
                            "expected a FragmentArg, got {}",
                            type_name(item)
                        ))
                    })
            })
            .collect();
        Ok((
            args?.into(),
            value
                .is_exact_instance_of::<PyTuple>()
                .then(|| value.clone().unbind()),
        ))
    }
}

impl PyField for ElementwiseOp {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(op_to_py(py, self)?.unbind())
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Extracted<Self>> {
        if let Ok(text) = value.cast::<PyString>() {
            // Construction validates the name and gives the object the reads hand back.
            let op = ELEMENTWISE.get(value.py())?.call1((text,))?;
            let name: String = op.getattr("name")?.extract()?;
            return Ok((ElementwiseOp::new(&name), Some(op.unbind())));
        }
        let name: String = value.getattr("name")?.extract()?;
        Ok((ElementwiseOp::new(&name), Some(value.clone().unbind())))
    }
}

impl PyField for Arc<[ElementwiseOp]> {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let wrapped: PyResult<Vec<Py<PyAny>>> = self.iter().map(|op| op.to_py(py)).collect();
        Ok(PyTuple::new(py, wrapped?)?.into_any().unbind())
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Extracted<Self>> {
        let parts = items(value, "a tuple of ops")?;
        let ops: PyResult<Vec<ElementwiseOp>> = parts
            .iter()
            .map(|item| ElementwiseOp::from_py(item).map(|(op, _)| op))
            .collect();
        Ok((ops?.into(), None))
    }
}

impl PyField for DType {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(dtype_to_py(py, self)?.unbind())
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Extracted<Self>> {
        if value.is_instance_of::<PyString>() {
            let dtype = DTYPE_GET.get(value.py())?.call1((value,))?;
            let name: String = dtype.getattr("name")?.extract()?;
            return Ok((name.parse().py()?, Some(dtype.unbind())));
        }
        let name: String = value.getattr("name")?.extract()?;
        Ok((name.parse().py()?, Some(value.clone().unbind())))
    }
}

impl PyField for i64 {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.into_pyobject(py)?.into_any().unbind())
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Extracted<Self>> {
        if value.is_instance_of::<PyBool>() || !value.is_instance_of::<PyInt>() {
            return Err(PyTypeError::new_err(format!(
                "expected an int, got {}",
                value.get_type().name()?
            )));
        }
        Ok((value.extract()?, Some(value.clone().unbind())))
    }
}

impl PyField for bool {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyBool::new(py, *self).to_owned().into_any().unbind())
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Extracted<Self>> {
        Ok((value.extract()?, Some(value.clone().unbind())))
    }
}

impl PyField for Float {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyFloat::new(py, self.0).into_any().unbind())
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Extracted<Self>> {
        let number = num_from_py(value)?;
        Ok((
            Float(number.as_f64()),
            value
                .is_exact_instance_of::<PyFloat>()
                .then(|| value.clone().unbind()),
        ))
    }
}

impl PyField for Arc<[Float]> {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(
            PyTuple::new(py, self.iter().map(|f| PyFloat::new(py, f.0)))?
                .into_any()
                .unbind(),
        )
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Extracted<Self>> {
        let parts = items(value, "a tuple of numbers")?;
        let floats: PyResult<Vec<Float>> = parts
            .iter()
            .map(|item| Float::from_py(item).map(|(f, _)| f))
            .collect();
        Ok((floats?.into(), None))
    }
}

impl PyField for Arc<[i64]> {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyTuple::new(py, self.iter())?.into_any().unbind())
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Extracted<Self>> {
        let parts = items(value, "a tuple of ints")?;
        let ints: PyResult<Vec<i64>> = parts
            .iter()
            .map(|item| i64::from_py(item).map(|(i, _)| i))
            .collect();
        Ok((
            ints?.into(),
            value
                .is_exact_instance_of::<PyTuple>()
                .then(|| value.clone().unbind()),
        ))
    }
}

impl PyField for Shape3 {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyTuple::new(py, [self.0, self.1, self.2])?
            .into_any()
            .unbind())
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Extracted<Self>> {
        let (ints, _) = <Arc<[i64]>>::from_py(value)?;
        match ints[..] {
            [a, b, c] => Ok((
                Shape3(a, b, c),
                value
                    .is_exact_instance_of::<PyTuple>()
                    .then(|| value.clone().unbind()),
            )),
            _ => Err(PyValueError::new_err("a shape is three ints")),
        }
    }
}

impl PyField for Int2 {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyTuple::new(py, [self.0, self.1])?.into_any().unbind())
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Extracted<Self>> {
        let (ints, _) = <Arc<[i64]>>::from_py(value)?;
        match ints[..] {
            [a, b] => Ok((
                Int2(a, b),
                value
                    .is_exact_instance_of::<PyTuple>()
                    .then(|| value.clone().unbind()),
            )),
            _ => Err(PyValueError::new_err("two ints")),
        }
    }
}

impl<T: PyField> PyField for NumOr<T> {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            NumOr::Num(number) => num_to_py(py, *number),
            NumOr::Value(value) => value.to_py(py),
        }
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Extracted<Self>> {
        if value.is_instance_of::<PyInt>() || value.is_instance_of::<PyFloat>() {
            return Ok((
                NumOr::Num(num_from_py(value)?),
                Some(value.clone().unbind()),
            ));
        }
        let (inner, held) = T::from_py(value)?;
        Ok((NumOr::Value(inner), held))
    }
}

impl PyField for SubroutineRef {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PySubroutine::wrap(py, &self.0)?.into_any().unbind())
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Extracted<Self>> {
        let sub = value
            .cast::<PySubroutine>()
            .map_err(|_| PyTypeError::new_err("expected a Subroutine"))?;
        Ok((
            SubroutineRef(sub.get().sub.clone()),
            Some(value.clone().unbind()),
        ))
    }
}

/// Per-object slots for the Python objects a statement's or expression's fields hand back.
pub type Slots = Box<[OnceLock<Py<PyAny>>]>;

pub fn slots(count: usize) -> Slots {
    (0..count).map(|_| OnceLock::new()).collect()
}

pub fn cached<'py>(
    py: Python<'py>,
    slot: &OnceLock<Py<PyAny>>,
    make: impl FnOnce() -> PyResult<Py<PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Some(found) = slot.get() {
        return Ok(found.bind(py).clone());
    }
    let value = make()?;
    Ok(slot.get_or_init(|| value).bind(py).clone())
}
