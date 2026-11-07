//! The expression classes: one base, `Expression`, holding the shared tree, and one class per kind.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use emmy_ir::Name;
use emmy_ir::expr::{BinOp, Expr, Expression, Interval, Lit, LitType, Num, SimplifyCtx};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PyList, PyString, PyTuple};

use crate::convert::{
    IntoPyErr, PARSE, PyField, cached, expression_from_py, located, num_from_py, num_to_py,
    translate,
};

/// The base class of every expression node: the shared tree, and the Python objects its fields hand back.
#[pyclass(
    subclass,
    frozen,
    module = "emmy.compiler.ir.expression",
    name = "Expression"
)]
pub struct PyExpression {
    pub expr: Expression,
    slots: [OnceLock<Py<PyAny>>; 3],
}

/// The expression classes: one per kind of expression, each with a getter per part of the kind, read once and kept
/// in the expression's slots. The constructors follow, one per class, since no two take the same arguments.
macro_rules! expression_classes {
    ($( $Py:ident $name:literal : $Variant:ident $bind:tt |$py:ident| { $( $getter:ident = $slot:literal => $value:expr; )* } )*) => { $(
        #[pyclass(extends = PyExpression, frozen, module = "emmy.compiler.ir.expression", name = $name)]
        pub struct $Py;

        #[pymethods]
        impl $Py { $(
            #[getter]
            #[allow(unused_variables)]
            fn $getter<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
                let base = slf.as_super().get();
                let $py = slf.py();
                let Expr::$Variant $bind = &*base.expr else {
                    return Err(kind_error($name));
                };
                base.slot($py, $slot, || $value)
            }
        )* }
    )* };
}

expression_classes! {
    PyVar "Var": Var(name) |py| {
        name = 0 => Ok(PyString::new(py, name).into_any().unbind());
    }
    PyLiteral "Literal": Literal(lit) |py| {
        value = 0 => num_to_py(py, lit.value);
        dtype = 1 => Ok(PyString::new(py, lit.dtype.name()).into_any().unbind());
    }
    PyBuiltin "Builtin": Builtin(name) |py| {
        name = 0 => Ok(PyString::new(py, name).into_any().unbind());
    }
    PyBinaryExpression "BinaryExpression": Binary(op, left, right) |py| {
        op = 0 => Ok(PyString::new(py, op.text()).into_any().unbind());
        left = 1 => Ok(PyExpression::wrap(py, left)?.unbind());
        right = 2 => Ok(PyExpression::wrap(py, right)?.unbind());
    }
    PyFunctionCallExpression "FunctionCallExpression": Call(name, args) |py| {
        name = 0 => Ok(PyString::new(py, name).into_any().unbind());
        args = 1 => args.to_py(py);
    }
    PyTernaryExpression "TernaryExpression": Ternary(cond, if_true, if_false) |py| {
        cond = 0 => Ok(PyExpression::wrap(py, cond)?.unbind());
        if_true = 1 => Ok(PyExpression::wrap(py, if_true)?.unbind());
        if_false = 2 => Ok(PyExpression::wrap(py, if_false)?.unbind());
    }
    PyCastExpression "CastExpression": Cast(dtype, inner) |py| {
        dtype = 0 => Ok(PyString::new(py, dtype).into_any().unbind());
        expression = 1 => Ok(PyExpression::wrap(py, inner)?.unbind());
    }
    PyFlatIndex "FlatIndex": FlatIndex(buffer, index) |py| {
        buffer = 0 => Ok(PyString::new(py, buffer).into_any().unbind());
        index = 1 => index.to_py(py);
    }
}

impl PyExpression {
    fn base(expr: Expression) -> PyExpression {
        PyExpression {
            expr,
            slots: [OnceLock::new(), OnceLock::new(), OnceLock::new()],
        }
    }

    /// The Python object of `expr`: an instance of its kind's class.
    pub fn wrap<'py>(py: Python<'py>, expr: &Expression) -> PyResult<Bound<'py, PyAny>> {
        let base = PyClassInitializer::from(PyExpression::base(expr.clone()));
        Ok(match &**expr {
            Expr::Var(_) => Bound::new(py, base.add_subclass(PyVar))?.into_any(),
            Expr::Literal(_) => Bound::new(py, base.add_subclass(PyLiteral))?.into_any(),
            Expr::Builtin(_) => Bound::new(py, base.add_subclass(PyBuiltin))?.into_any(),
            Expr::Binary(..) => Bound::new(py, base.add_subclass(PyBinaryExpression))?.into_any(),
            Expr::Call(..) => {
                Bound::new(py, base.add_subclass(PyFunctionCallExpression))?.into_any()
            }
            Expr::Ternary(..) => Bound::new(py, base.add_subclass(PyTernaryExpression))?.into_any(),
            Expr::Cast(..) => Bound::new(py, base.add_subclass(PyCastExpression))?.into_any(),
            Expr::FlatIndex(..) => Bound::new(py, base.add_subclass(PyFlatIndex))?.into_any(),
        })
    }

    fn slot<'py>(
        &self,
        py: Python<'py>,
        index: usize,
        make: impl FnOnce() -> PyResult<Py<PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.slots[index], make)
    }
}

fn coerce(value: &Bound<'_, PyAny>) -> PyResult<Expression> {
    if let Ok(expression) = value.cast::<PyExpression>() {
        return Ok(expression.get().expr.clone());
    }
    match num_from_py(value) {
        Ok(Num::Float(v)) => Ok(Expression::float(v)),
        Ok(number) => Ok(Expression::literal(number, LitType::Int)),
        Err(_) => Err(PyTypeError::new_err(format!(
            "an expression combines with an expression or a number, not {}",
            value.get_type().name()?
        ))),
    }
}

fn binary(op: BinOp, left: Expression, right: Expression) -> Expression {
    Expression::binary(op, left, right)
}

pub fn ctx_from_py(value: &Bound<'_, PyAny>) -> PyResult<SimplifyCtx> {
    Ok(value
        .cast::<PySimplifyCtx>()
        .map_err(|_| PyTypeError::new_err("expected a SimplifyCtx"))?
        .get()
        .ctx
        .clone())
}

pub fn mapping_from_py(value: &Bound<'_, PyAny>) -> PyResult<HashMap<Name, Expression>> {
    let dict = value
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("expected a dict of name -> Expression"))?;
    let mut out = HashMap::new();
    for (key, item) in dict.iter() {
        out.insert(key.extract::<String>()?.into(), expression_from_py(&item)?);
    }
    Ok(out)
}

pub fn renames_from_py(value: &Bound<'_, PyAny>) -> PyResult<HashMap<Name, Name>> {
    let dict = value
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("expected a dict of name -> name"))?;
    let mut out = HashMap::new();
    for (key, item) in dict.iter() {
        out.insert(
            key.extract::<String>()?.into(),
            item.extract::<String>()?.into(),
        );
    }
    Ok(out)
}

#[pymethods]
impl PyExpression {
    #[classattr]
    fn wire_tag() -> &'static str {
        "expr"
    }

    /// The expression's text: what it is on the wire and in every listing.
    fn __str__(&self) -> PyResult<String> {
        self.expr.try_text().py()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<PyExpression>()
            .is_ok_and(|other| other.get().expr == self.expr)
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.expr.hash(&mut hasher);
        hasher.finish()
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, (String,))> {
        Ok((
            PARSE.get(slf.py())?.clone(),
            (slf.get().expr.try_text().py()?,),
        ))
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(PyExpression::wrap(
            other.py(),
            &binary(BinOp::Add, self.expr.clone(), coerce(other)?),
        )?
        .unbind())
    }
    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(PyExpression::wrap(
            other.py(),
            &binary(BinOp::Add, coerce(other)?, self.expr.clone()),
        )?
        .unbind())
    }
    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(PyExpression::wrap(
            other.py(),
            &binary(BinOp::Sub, self.expr.clone(), coerce(other)?),
        )?
        .unbind())
    }
    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(PyExpression::wrap(
            other.py(),
            &binary(BinOp::Sub, coerce(other)?, self.expr.clone()),
        )?
        .unbind())
    }
    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(PyExpression::wrap(
            other.py(),
            &binary(BinOp::Mul, self.expr.clone(), coerce(other)?),
        )?
        .unbind())
    }
    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(PyExpression::wrap(
            other.py(),
            &binary(BinOp::Mul, coerce(other)?, self.expr.clone()),
        )?
        .unbind())
    }
    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(PyExpression::wrap(
            other.py(),
            &binary(BinOp::Div, self.expr.clone(), coerce(other)?),
        )?
        .unbind())
    }
    fn __mod__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(PyExpression::wrap(
            other.py(),
            &binary(BinOp::Mod, self.expr.clone(), coerce(other)?),
        )?
        .unbind())
    }
    fn __neg__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyExpression::wrap(
            py,
            &binary(BinOp::Sub, Expression::int(0), self.expr.clone()),
        )?
        .unbind())
    }

    /// Less-than (`<`).
    fn lt(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(PyExpression::wrap(
            other.py(),
            &binary(BinOp::Lt, self.expr.clone(), coerce(other)?),
        )?
        .unbind())
    }

    /// Replace `Var(name)` subterms with `mapping[name]`; names the mapping lacks pass through.
    fn substitute(&self, py: Python<'_>, mapping: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let mapping = if let Ok(dict) = mapping.cast::<PyDict>() {
            mapping_from_py(dict.as_any())?
        } else {
            mapping_from_py(&mapping.getattr("mapping")?)?
        };
        Ok(PyExpression::wrap(py, &self.expr.substitute(&mapping))?.unbind())
    }

    /// Every free variable's name mapped through `mapping`; a name it lacks passes through.
    fn _rename(&self, py: Python<'_>, mapping: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let mapping = renames_from_py(mapping)?;
        Ok(PyExpression::wrap(py, &self.expr.rename_by(&mapping))?.unbind())
    }

    /// The set of variable names appearing anywhere in this expression.
    fn free_vars<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
        PyFrozenSet::new(
            py,
            self.expr
                .free_vars()
                .iter()
                .map(|name| PyString::new(py, name)),
        )
    }

    /// Pre-order walk over this expression's tree, itself first.
    fn subterms<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items: PyResult<Vec<Bound<'py, PyAny>>> = self
            .expr
            .subterms()
            .iter()
            .map(|e| PyExpression::wrap(py, e))
            .collect();
        PyList::new(py, items?)
    }

    /// The direct children, in field order.
    fn children<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let items: PyResult<Vec<Bound<'py, PyAny>>> = self
            .expr
            .children()
            .iter()
            .map(|e| PyExpression::wrap(py, e))
            .collect();
        PyTuple::new(py, items?)
    }

    /// This node over other children, as many as `children` returned.
    fn with_children<'py>(
        slf: &Bound<'py, Self>,
        children: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let items: PyResult<Vec<Expression>> = children
            .try_iter()?
            .map(|item| expression_from_py(&item?))
            .collect();
        let rebuilt = slf.get().expr.with_children(items?).py()?;
        if Arc::ptr_eq(&rebuilt.0, &slf.get().expr.0) {
            return Ok(slf.clone().into_any());
        }
        PyExpression::wrap(slf.py(), &rebuilt)
    }

    /// Bottom-up rewrite: constant-fold literals, drop algebraic identities, decide static comparisons.
    fn simplify<'py>(
        slf: &Bound<'py, Self>,
        ctx: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let simplified = slf.get().expr.simplify(&ctx_from_py(ctx)?).py()?;
        if Arc::ptr_eq(&simplified.0, &slf.get().expr.0) {
            return Ok(slf.clone().into_any());
        }
        PyExpression::wrap(slf.py(), &simplified)
    }

    /// Conservative integer range, `None` when unknown.
    fn range(&self, ctx: &Bound<'_, PyAny>) -> PyResult<Option<PyInterval>> {
        Ok(self
            .expr
            .range(&ctx_from_py(ctx)?)
            .map(|interval| PyInterval { interval }))
    }

    /// The structural form: an expression's identity is its text.
    fn structural_form(&self) -> PyResult<String> {
        self.expr.try_text().py()
    }

    fn to_wire(&self) -> PyResult<String> {
        self.expr.try_text().py()
    }

    #[classmethod]
    #[pyo3(signature = (value, where_ = "expression"))]
    fn from_wire<'py>(
        cls: &Bound<'py, pyo3::types::PyType>,
        value: &Bound<'py, PyAny>,
        where_: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let Ok(text) = value.cast::<PyString>() else {
            return Err(PyValueError::new_err(format!(
                "{where_} must be an expression's text, got {}",
                value.repr()?
            )));
        };
        let expression = emmy_ir::expr::parse(text.to_str()?)
            .map_err(|err| located(cls.py(), where_, translate(err)))?;
        let wrapped = PyExpression::wrap(cls.py(), &expression)?;
        if !wrapped.is_instance(cls)? {
            return Err(PyValueError::new_err(format!(
                "{where_} must be a {}, got {}",
                cls.getattr("__name__")?,
                value.repr()?
            )));
        }
        Ok(wrapped)
    }

    fn wire_parts(&self) -> Option<()> {
        None
    }
}

fn kind_error(kind: &str) -> PyErr {
    PyTypeError::new_err(format!("not a {kind}"))
}

#[pymethods]
impl PyVar {
    #[new]
    fn new(name: &str) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyExpression::base(Expression::var(name))).add_subclass(PyVar)
    }
}

#[pymethods]
impl PyLiteral {
    #[new]
    #[pyo3(signature = (value, dtype = "float"))]
    fn new(value: &Bound<'_, PyAny>, dtype: &str) -> PyResult<PyClassInitializer<Self>> {
        let number = num_from_py(value)?;
        let expr = Expression::new(Expr::Literal(Lit {
            value: number,
            dtype: dtype.parse::<LitType>().py()?,
        }));
        Ok(PyClassInitializer::from(PyExpression::base(expr)).add_subclass(PyLiteral))
    }
}

#[pymethods]
impl PyBuiltin {
    #[new]
    fn new(name: &str) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyExpression::base(Expression::new(Expr::Builtin(
            name.into(),
        ))))
        .add_subclass(PyBuiltin)
    }
}

#[pymethods]
impl PyBinaryExpression {
    #[new]
    fn new(
        op: &str,
        left: &Bound<'_, PyAny>,
        right: &Bound<'_, PyAny>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let Some(op) = BinOp::parse(op) else {
            return Err(PyValueError::new_err(format!(
                "the operator {op:?} has no spelling"
            )));
        };
        let expr = binary(op, expression_from_py(left)?, expression_from_py(right)?);
        Ok(PyClassInitializer::from(PyExpression::base(expr)).add_subclass(PyBinaryExpression))
    }

    /// How tightly the operator binds; the renderer parenthesizes by it.
    #[getter]
    fn precedence(slf: &Bound<'_, Self>) -> PyResult<u8> {
        let Expr::Binary(op, ..) = &*slf.as_super().get().expr else {
            return Err(kind_error("BinaryExpression"));
        };
        Ok(op.precedence())
    }
}

#[pymethods]
impl PyFunctionCallExpression {
    #[new]
    fn new(name: &str, args: &Bound<'_, PyAny>) -> PyResult<PyClassInitializer<Self>> {
        let args: PyResult<Vec<Expression>> = args
            .try_iter()?
            .map(|item| expression_from_py(&item?))
            .collect();
        let expr = Expression::new(Expr::Call(name.into(), args?.into()));
        Ok(PyClassInitializer::from(PyExpression::base(expr))
            .add_subclass(PyFunctionCallExpression))
    }
}

#[pymethods]
impl PyTernaryExpression {
    #[new]
    fn new(
        cond: &Bound<'_, PyAny>,
        if_true: &Bound<'_, PyAny>,
        if_false: &Bound<'_, PyAny>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let expr = Expression::new(Expr::Ternary(
            expression_from_py(cond)?,
            expression_from_py(if_true)?,
            expression_from_py(if_false)?,
        ));
        Ok(PyClassInitializer::from(PyExpression::base(expr)).add_subclass(PyTernaryExpression))
    }
}

#[pymethods]
impl PyCastExpression {
    #[new]
    fn new(dtype: &str, expression: &Bound<'_, PyAny>) -> PyResult<PyClassInitializer<Self>> {
        let expr = Expression::new(Expr::Cast(dtype.into(), expression_from_py(expression)?));
        Ok(PyClassInitializer::from(PyExpression::base(expr)).add_subclass(PyCastExpression))
    }
}

#[pymethods]
impl PyFlatIndex {
    #[new]
    fn new(buffer: &str, index: &Bound<'_, PyAny>) -> PyResult<PyClassInitializer<Self>> {
        let index: PyResult<Vec<Expression>> = index
            .try_iter()?
            .map(|item| expression_from_py(&item?))
            .collect();
        let expr = Expression::new(Expr::FlatIndex(buffer.into(), index?.into()));
        Ok(PyClassInitializer::from(PyExpression::base(expr)).add_subclass(PyFlatIndex))
    }
}

/// A closed integer interval `[lo, hi]`.
#[pyclass(
    frozen,
    from_py_object,
    module = "emmy.compiler.ir.expression",
    name = "Interval"
)]
#[derive(Clone)]
pub struct PyInterval {
    pub interval: Interval,
}

#[pymethods]
impl PyInterval {
    #[new]
    fn new(lo: i128, hi: i128) -> PyInterval {
        PyInterval {
            interval: Interval::new(lo, hi),
        }
    }

    #[getter]
    fn lo(&self) -> i128 {
        self.interval.lo
    }

    #[getter]
    fn hi(&self) -> i128 {
        self.interval.hi
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<PyInterval>()
            .is_ok_and(|other| other.get().interval == self.interval)
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.interval.hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(&self) -> String {
        format!("Interval(lo={}, hi={})", self.interval.lo, self.interval.hi)
    }
}

/// The ranges in scope: an integer interval per name, and a symbolic exclusive upper bound per name where known.
#[pyclass(frozen, module = "emmy.compiler.ir.expression", name = "SimplifyCtx")]
pub struct PySimplifyCtx {
    pub ctx: SimplifyCtx,
}

#[pymethods]
impl PySimplifyCtx {
    #[new]
    #[pyo3(signature = (ranges = None, bounds = None))]
    fn new(
        ranges: Option<&Bound<'_, PyDict>>,
        bounds: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PySimplifyCtx> {
        let mut ctx = SimplifyCtx::empty();
        if let Some(ranges) = ranges {
            for (key, value) in ranges.iter() {
                let interval = value
                    .cast::<PyInterval>()
                    .map_err(|_| PyTypeError::new_err("ranges hold Interval values"))?
                    .get()
                    .interval;
                ctx.ranges.insert(key.extract::<String>()?.into(), interval);
            }
        }
        if let Some(bounds) = bounds {
            for (key, value) in bounds.iter() {
                ctx.bounds
                    .insert(key.extract::<String>()?.into(), expression_from_py(&value)?);
            }
        }
        Ok(PySimplifyCtx { ctx })
    }

    #[classmethod]
    fn empty(_cls: &Bound<'_, pyo3::types::PyType>) -> PySimplifyCtx {
        PySimplifyCtx {
            ctx: SimplifyCtx::empty(),
        }
    }

    #[getter]
    fn ranges<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (name, interval) in &self.ctx.ranges {
            out.set_item(
                &**name,
                PyInterval {
                    interval: *interval,
                },
            )?;
        }
        Ok(out)
    }

    #[getter]
    fn bounds<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (name, bound) in &self.ctx.bounds {
            out.set_item(&**name, PyExpression::wrap(py, bound)?)?;
        }
        Ok(out)
    }

    #[pyo3(signature = (name, interval, bound = None))]
    fn extend(
        &self,
        name: &str,
        interval: &PyInterval,
        bound: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PySimplifyCtx> {
        let bound = bound
            .filter(|b| !b.is_none())
            .map(expression_from_py)
            .transpose()?;
        Ok(PySimplifyCtx {
            ctx: self.ctx.extend(name, interval.interval, bound),
        })
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<PySimplifyCtx>()
            .is_ok_and(|other| other.get().ctx == self.ctx)
    }
}

/// The expression `text` spells.
#[pyfunction]
pub fn parse(py: Python<'_>, text: &str) -> PyResult<Py<PyAny>> {
    Ok(PyExpression::wrap(py, &emmy_ir::expr::parse(text).py()?)?.unbind())
}

/// `name` as the reader reads a name back: bare, or between backticks.
#[pyfunction]
#[pyo3(signature = (name, reserved = None))]
pub fn spell_name(name: &str, reserved: Option<&Bound<'_, PyAny>>) -> PyResult<String> {
    let extra: Vec<String> = match reserved {
        Some(items) => items
            .try_iter()?
            .map(|item| item?.extract::<String>())
            .collect::<PyResult<_>>()?,
        None => vec![],
    };
    let extra: Vec<&str> = extra.iter().map(|s| s.as_str()).collect();
    emmy_ir::expr::spell_name(name, &extra).py()
}

/// `expression` as `anchor + Σ coeffs[v]·v` over `vars`, or `None`.
#[pyfunction]
pub fn affine_form<'py>(
    py: Python<'py>,
    expression: &Bound<'py, PyAny>,
    vars: &Bound<'py, PyAny>,
) -> PyResult<Option<(Bound<'py, PyAny>, Bound<'py, PyDict>)>> {
    let names: Vec<String> = vars
        .try_iter()?
        .map(|item| item?.extract::<String>())
        .collect::<PyResult<_>>()?;
    let expression = expression_from_py(expression)?;
    let Some((anchor, coeffs)) =
        emmy_ir::expr::affine_form(&expression, &|name| names.iter().any(|n| n == name))
    else {
        return Ok(None);
    };
    let out = PyDict::new(py);
    for (name, coeff) in coeffs {
        out.set_item(&*name, coeff)?;
    }
    Ok(Some((PyExpression::wrap(py, &anchor)?, out)))
}

/// `expression` as `(its integer anchor, the rest)`, or `None`.
#[pyfunction]
pub fn split_anchor<'py>(
    py: Python<'py>,
    expression: &Bound<'py, PyAny>,
) -> PyResult<Option<(i64, Bound<'py, PyAny>)>> {
    match emmy_ir::expr::split_anchor(&expression_from_py(expression)?) {
        Some((anchor, rest)) => Ok(Some((anchor, PyExpression::wrap(py, &rest)?))),
        None => Ok(None),
    }
}

/// An upper bound on the number of distinct coordinate tuples `expressions` take over `var_extents`.
#[pyfunction]
pub fn index_set_size(
    expressions: &Bound<'_, PyAny>,
    var_extents: &Bound<'_, PyDict>,
) -> PyResult<Option<i128>> {
    let expressions: PyResult<Vec<Expression>> = expressions
        .try_iter()?
        .map(|item| expression_from_py(&item?))
        .collect();
    let mut extents: HashMap<Name, i64> = HashMap::new();
    for (key, value) in var_extents.iter() {
        extents.insert(key.extract::<String>()?.into(), value.extract()?);
    }
    Ok(emmy_ir::expr::index_set_size(&expressions?, &extents))
}

/// The placeholder variable for output coordinate axis `d`.
#[pyfunction]
pub fn placeholder(py: Python<'_>, d: usize) -> PyResult<Py<PyAny>> {
    Ok(PyExpression::wrap(py, &Expression::placeholder(d))?.unbind())
}

/// Whether `expression` is a placeholder variable, of axis `d` when given.
#[pyfunction]
#[pyo3(signature = (expression, d = None))]
pub fn is_placeholder(expression: &Bound<'_, PyAny>, d: Option<usize>) -> bool {
    expression
        .cast::<PyExpression>()
        .is_ok_and(|e| e.get().expr.is_placeholder(d))
}

#[pyfunction]
pub fn builtin_names<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
    PyFrozenSet::new(
        py,
        emmy_ir::expr::BUILTINS
            .iter()
            .map(|name| PyString::new(py, name)),
    )
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyExpression>()?;
    m.add_class::<PyVar>()?;
    m.add_class::<PyLiteral>()?;
    m.add_class::<PyBuiltin>()?;
    m.add_class::<PyBinaryExpression>()?;
    m.add_class::<PyFunctionCallExpression>()?;
    m.add_class::<PyTernaryExpression>()?;
    m.add_class::<PyCastExpression>()?;
    m.add_class::<PyFlatIndex>()?;
    m.add_class::<PyInterval>()?;
    m.add_class::<PySimplifyCtx>()?;
    m.add_function(wrap_pyfunction!(parse, m)?)?;
    m.add_function(wrap_pyfunction!(spell_name, m)?)?;
    m.add_function(wrap_pyfunction!(affine_form, m)?)?;
    m.add_function(wrap_pyfunction!(split_anchor, m)?)?;
    m.add_function(wrap_pyfunction!(index_set_size, m)?)?;
    m.add_function(wrap_pyfunction!(placeholder, m)?)?;
    m.add_function(wrap_pyfunction!(is_placeholder, m)?)?;
    m.add("BUILTINS", builtin_names(m.py())?)?;
    m.add("PLACEHOLDER_PREFIX", emmy_ir::expr::PLACEHOLDER_PREFIX)?;
    Ok(())
}
