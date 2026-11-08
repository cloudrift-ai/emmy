//! The expression classes: one base, `Expression`, holding the shared tree, and one class per expression type.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use emmy_ir::Name;
use emmy_ir::elementwise::{ElementwiseOp, Operation};
use emmy_ir::expr::{
    Builtin, CastTarget, Expr, Expression, Interval, Lit, LitType, Num, SimplifyCtx,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PyList, PyString, PyTuple};

use crate::convert::{
    ELEMENTWISE, IntoPyErr, PARSE, PyField, cached, expression_from_py, located, name_from_py,
    num_from_py, num_to_py, op_to_py, translate,
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

/// What a field of an expression node hands Python: a name or an operator as a string, a child as its class, several
/// as a tuple.
trait PyExprField {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>>;
}

impl PyExprField for Name {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyString::new(py, self).into_any().unbind())
    }
}

impl PyExprField for ElementwiseOp {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyString::new(py, self.name()).into_any().unbind())
    }
}

impl PyExprField for Builtin {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyString::new(py, self.text()).into_any().unbind())
    }
}

impl PyExprField for CastTarget {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyString::new(py, &self.to_string()).into_any().unbind())
    }
}

impl PyExprField for LitType {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyString::new(py, self.text()).into_any().unbind())
    }
}

impl PyExprField for Expression {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyExpression::wrap(py, self)?.unbind())
    }
}

impl PyExprField for Arc<[Expression]> {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        PyField::to_py(self, py)
    }
}

/// The extension's callback over the expression types: one class per expression type with a getter per field, the one
/// wrap of an expression into its class, and the registration. A literal's two parts have getters of their own, since
/// the node holds them as one value.
macro_rules! define_py_exprs {
    ($( $(#[doc = $doc:literal])* $Variant:ident as $Py:ident $name:literal { $( $field:ident : $ty:ty ),* } ,)*) => {
        $(
            $(#[doc = $doc])*
            #[pyclass(extends = PyExpression, frozen, module = "emmy.compiler.ir.expression", name = $name)]
            pub struct $Py;

            define_py_exprs!(@getters $Variant $Py $name { $( $field : $ty ),* } { $( $field : $ty ),* });
        )*

        impl PyExpression {
            /// The Python object of `expr`: an instance of its expression type's class.
            pub fn wrap<'py>(py: Python<'py>, expr: &Expression) -> PyResult<Bound<'py, PyAny>> {
                let base = PyClassInitializer::from(PyExpression::base(expr.clone())?);
                Ok(match &**expr {
                    // An operator over two operands is the `BinaryExpression` view of the one `Apply` node.
                    Expr::Apply(..) if expr.as_binary().is_some() => {
                        Bound::new(py, base.add_subclass(PyBinaryExpression))?.into_any()
                    }
                    $( Expr::$Variant(..) => Bound::new(py, base.add_subclass($Py))?.into_any(), )*
                })
            }
        }

        fn register_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
            $( m.add_class::<$Py>()?; )*
            m.add_class::<PyBinaryExpression>()?;
            Ok(())
        }
    };
    (@getters Literal $Py:ident $name:literal $fields:tt $all:tt) => {
        #[pymethods]
        impl $Py {
            #[getter]
            fn value<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
                let base = slf.as_super().get();
                let Expr::Literal(lit) = &*base.expr else {
                    return Err(expr_type_error($name));
                };
                base.slot(slf.py(), 0, || num_to_py(slf.py(), lit.value))
            }

            #[getter]
            fn dtype<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
                let base = slf.as_super().get();
                let Expr::Literal(lit) = &*base.expr else {
                    return Err(expr_type_error($name));
                };
                base.slot(slf.py(), 1, || {
                    Ok(PyString::new(slf.py(), lit.dtype.text()).into_any().unbind())
                })
            }
        }
    };
    (@getters Apply $Py:ident $name:literal $fields:tt $all:tt) => {
        #[pymethods]
        impl $Py {
            /// The operation's name: the table's, or the spelling only a renderer knows.
            #[getter]
            fn name<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
                let base = slf.as_super().get();
                let Expr::Apply(op, _) = &*base.expr else {
                    return Err(expr_type_error($name));
                };
                base.slot(slf.py(), 0, || PyExprField::to_py(op, slf.py()))
            }

            #[getter]
            fn args<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
                let base = slf.as_super().get();
                let Expr::Apply(_, args) = &*base.expr else {
                    return Err(expr_type_error($name));
                };
                base.slot(slf.py(), 1, || PyExprField::to_py(args, slf.py()))
            }
        }
    };
    (@getters $Variant:ident $Py:ident $name:literal { $( $field:ident : $ty:ty ),* } $all:tt) => {
        $( define_py_exprs!(@getter $Variant $Py $name $all $field); )*
    };
    (@getter $Variant:ident $Py:ident $name:literal { $( $all:ident : $ty:ty ),* } $field:ident) => {
        #[pymethods]
        impl $Py {
            #[getter]
            #[allow(unused_variables)]
            fn $field<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
                const FIELDS: &[&str] = &[ $( stringify!($all), )* ];
                let base = slf.as_super().get();
                let Expr::$Variant($($all),*) = &*base.expr else {
                    return Err(expr_type_error($name));
                };
                let slot = FIELDS.iter().position(|f| *f == stringify!($field)).unwrap_or(0);
                base.slot(slf.py(), slot, || PyExprField::to_py($field, slf.py()))
            }
        }
    };
}

emmy_ir::with_exprs!(define_py_exprs);

impl PyExpression {
    /// The base of every expression object: the one place a tree too deep for the compiler's walks is refused.
    fn base(expr: Expression) -> PyResult<PyExpression> {
        Ok(PyExpression {
            expr: expr.bounded().py()?,
            slots: [OnceLock::new(), OnceLock::new(), OnceLock::new()],
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

fn binary(op: Operation, left: Expression, right: Expression) -> Expression {
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
        out.insert(name_from_py(&key)?, expression_from_py(&item)?);
    }
    Ok(out)
}

pub fn renames_from_py(value: &Bound<'_, PyAny>) -> PyResult<HashMap<Name, Name>> {
    let dict = value
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("expected a dict of name -> name"))?;
    let mut out = HashMap::new();
    for (key, item) in dict.iter() {
        out.insert(name_from_py(&key)?, name_from_py(&item)?);
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
            &binary(Operation::Add, self.expr.clone(), coerce(other)?),
        )?
        .unbind())
    }
    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(PyExpression::wrap(
            other.py(),
            &binary(Operation::Add, coerce(other)?, self.expr.clone()),
        )?
        .unbind())
    }
    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(PyExpression::wrap(
            other.py(),
            &binary(Operation::Subtract, self.expr.clone(), coerce(other)?),
        )?
        .unbind())
    }
    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(PyExpression::wrap(
            other.py(),
            &binary(Operation::Subtract, coerce(other)?, self.expr.clone()),
        )?
        .unbind())
    }
    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(PyExpression::wrap(
            other.py(),
            &binary(Operation::Multiply, self.expr.clone(), coerce(other)?),
        )?
        .unbind())
    }
    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(PyExpression::wrap(
            other.py(),
            &binary(Operation::Multiply, coerce(other)?, self.expr.clone()),
        )?
        .unbind())
    }
    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(PyExpression::wrap(
            other.py(),
            &binary(Operation::Divide, self.expr.clone(), coerce(other)?),
        )?
        .unbind())
    }
    fn __mod__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(PyExpression::wrap(
            other.py(),
            &binary(Operation::Remainder, self.expr.clone(), coerce(other)?),
        )?
        .unbind())
    }
    fn __neg__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyExpression::wrap(
            py,
            &binary(Operation::Subtract, Expression::int(0), self.expr.clone()),
        )?
        .unbind())
    }

    /// Less-than (`<`).
    fn lt(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(PyExpression::wrap(
            other.py(),
            &binary(Operation::Less, self.expr.clone(), coerce(other)?),
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

    /// The elementwise operation an operation node applies: a binary node's operation, a call's name (the table's,
    /// or a spelling only a renderer knows). `None` for a name, a literal, a built-in, a select, a cast or a flat
    /// index.
    #[getter]
    fn operation(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let Expr::Apply(op, _) = &*self.expr else {
            return Ok(None);
        };
        Ok(Some(op_to_py(py, op)?.unbind()))
    }

    /// The operands of an operation node, in order; empty for any other node.
    #[getter]
    fn operands<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let operands: Vec<Expression> = match &*self.expr {
            Expr::Apply(_, args) => args.to_vec(),
            _ => vec![],
        };
        let items: PyResult<Vec<Bound<'py, PyAny>>> =
            operands.iter().map(|e| PyExpression::wrap(py, e)).collect();
        PyTuple::new(py, items?)
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

fn expr_type_error(type_name: &str) -> PyErr {
    PyTypeError::new_err(format!("not a {type_name}"))
}

#[pymethods]
impl PyVar {
    #[new]
    fn new(name: &str) -> PyResult<PyClassInitializer<Self>> {
        Ok(
            PyClassInitializer::from(PyExpression::base(Expression::var(name))?)
                .add_subclass(PyVar),
        )
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
        Ok(PyClassInitializer::from(PyExpression::base(expr)?).add_subclass(PyLiteral))
    }
}

#[pymethods]
impl PyBuiltin {
    #[new]
    fn new(name: &str) -> PyResult<PyClassInitializer<Self>> {
        let expr = Expression::new(Expr::Builtin(name.parse().py()?));
        Ok(PyClassInitializer::from(PyExpression::base(expr)?).add_subclass(PyBuiltin))
    }
}

/// An operator of the table over two operands: the view Python takes of an `Apply` node whose operation has an infix
/// spelling and two operands, with the operator's spelling and its two sides as fields.
#[pyclass(extends = PyExpression, frozen, module = "emmy.compiler.ir.expression", name = "BinaryExpression")]
pub struct PyBinaryExpression;

impl PyBinaryExpression {
    fn sides(slf: &Bound<'_, Self>) -> PyResult<(Operation, Expression, Expression)> {
        let base = slf.as_super().get();
        let Some((op, left, right)) = base.expr.as_binary() else {
            return Err(expr_type_error("BinaryExpression"));
        };
        Ok((op, left.clone(), right.clone()))
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
        let op = Operation::from_infix(op)
            .ok_or_else(|| PyValueError::new_err(format!("no binary operator {op:?}")))?;
        let expr = binary(op, expression_from_py(left)?, expression_from_py(right)?);
        Ok(PyClassInitializer::from(PyExpression::base(expr)?).add_subclass(PyBinaryExpression))
    }

    /// The operator's spelling (`+`, `//`, `<`).
    #[getter]
    fn op<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let (op, _, _) = Self::sides(slf)?;
        slf.as_super().get().slot(slf.py(), 0, || {
            Ok(PyString::new(slf.py(), op.spelling()).into_any().unbind())
        })
    }

    #[getter]
    fn left<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let (_, left, _) = Self::sides(slf)?;
        slf.as_super()
            .get()
            .slot(slf.py(), 1, || PyExprField::to_py(&left, slf.py()))
    }

    #[getter]
    fn right<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let (_, _, right) = Self::sides(slf)?;
        slf.as_super()
            .get()
            .slot(slf.py(), 2, || PyExprField::to_py(&right, slf.py()))
    }

    /// How tightly the operator binds; the renderer parenthesizes by it.
    #[getter]
    fn precedence(slf: &Bound<'_, Self>) -> PyResult<u8> {
        Ok(Self::sides(slf)?.0.precedence())
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
        // The class is the view the caller asked for: a renderer builds a call of an operator on purpose, to spell it
        // as the target's intrinsic.
        let expr = Expression::call(name, args?);
        Ok(PyClassInitializer::from(PyExpression::base(expr)?)
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
        Ok(PyClassInitializer::from(PyExpression::base(expr)?).add_subclass(PyTernaryExpression))
    }
}

#[pymethods]
impl PyCastExpression {
    #[new]
    fn new(dtype: &str, expression: &Bound<'_, PyAny>) -> PyResult<PyClassInitializer<Self>> {
        let expr = Expression::new(Expr::Cast(
            dtype.parse().py()?,
            expression_from_py(expression)?,
        ));
        Ok(PyClassInitializer::from(PyExpression::base(expr)?).add_subclass(PyCastExpression))
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
        Ok(PyClassInitializer::from(PyExpression::base(expr)?).add_subclass(PyFlatIndex))
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
                ctx.ranges.insert(name_from_py(&key)?, interval);
            }
        }
        if let Some(bounds) = bounds {
            for (key, value) in bounds.iter() {
                ctx.bounds
                    .insert(name_from_py(&key)?, expression_from_py(&value)?);
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

/// `op` (an `ElementwiseImpl` or its name) applied to `args`: the node the constructors build, a binary node for an
/// operator over two operands, a chain of it for an associative one over more, else a call. An operand given as a
/// name is that variable; a number is a literal.
#[pyfunction]
#[pyo3(signature = (op, *args))]
pub fn apply(
    py: Python<'_>,
    op: &Bound<'_, PyAny>,
    args: &Bound<'_, PyTuple>,
) -> PyResult<Py<PyAny>> {
    // Construction through the Python class validates the name and takes its canonical spelling.
    let op = match op.cast::<PyString>() {
        Ok(text) => ELEMENTWISE.get(py)?.call1((text,))?,
        Err(_) => op.clone(),
    };
    let name: String = op.getattr("name")?.extract()?;
    let operands: Vec<Expression> = args
        .iter()
        .map(|item| expression_from_py(&item))
        .collect::<PyResult<_>>()?;
    let expr = Expression::call(&name, operands);
    Ok(PyExpression::wrap(py, &expr)?.unbind())
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
    let names: Vec<Name> = vars
        .try_iter()?
        .map(|item| name_from_py(&item?))
        .collect::<PyResult<_>>()?;
    let expression = expression_from_py(expression)?;
    let Some((anchor, coeffs)) =
        emmy_ir::expr::affine_form(&expression, &|name| names.contains(&name))
    else {
        return Ok(None);
    };
    let out = PyDict::new(py);
    for (name, coeff) in coeffs {
        out.set_item(&*name, coeff)?;
    }
    Ok(Some((PyExpression::wrap(py, &anchor)?, out)))
}

/// An affine form as Python sees it: the `(coefficient, operand)` terms and the offset.
type AffineForm<'py> = (Vec<(f64, Bound<'py, PyAny>)>, f64);

/// The affine form of a value: its terms, `(coefficient, operand)` each, and its literal offset — the form the
/// normal form spells every sum in. A value that is no sum is one term with coefficient one.
#[pyfunction]
pub fn affine_terms<'py>(
    py: Python<'py>,
    expression: &Bound<'py, PyAny>,
) -> PyResult<AffineForm<'py>> {
    let form = emmy_ir::normalize::affine_terms(&expression_from_py(expression)?);
    let terms = form
        .terms
        .iter()
        .map(|(coefficient, term)| Ok((*coefficient, PyExpression::wrap(py, term)?)))
        .collect::<PyResult<Vec<_>>>()?;
    Ok((terms, form.offset))
}

/// The sum an affine form spells: like terms merged, a whole-number coefficient or offset as the integer, the offset
/// last; the terms in the order the normal form gives names it does not rank.
#[pyfunction]
pub fn affine_sum<'py>(
    py: Python<'py>,
    terms: &Bound<'py, PyAny>,
    offset: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let terms: Vec<(f64, Expression)> = terms
        .try_iter()?
        .map(|item| {
            let (coefficient, term): (f64, Bound<'py, PyAny>) = item?.extract()?;
            Ok((coefficient, expression_from_py(&term)?))
        })
        .collect::<PyResult<_>>()?;
    let form = emmy_ir::normalize::AffineTerms { terms, offset };
    PyExpression::wrap(
        py,
        &emmy_ir::normalize::affine_sum(form, &|_| emmy_ir::normalize::ties::unranked()),
    )
}

/// Whether `expression` is an operation node of the affine form: a sum, a difference, a negation, a product with a
/// literal factor or a quotient by a literal.
#[pyfunction]
pub fn affine_node(expression: &Bound<'_, PyAny>) -> PyResult<bool> {
    let expression = expression_from_py(expression)?;
    Ok(expression
        .operation()
        .is_some_and(|(op, args)| emmy_ir::normalize::affine_node(op, args)))
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
        extents.insert(name_from_py(&key)?, value.extract()?);
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
        Builtin::ALL
            .iter()
            .map(|builtin| PyString::new(py, builtin.text())),
    )
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyExpression>()?;
    register_classes(m)?;
    m.add_class::<PyInterval>()?;
    m.add_class::<PySimplifyCtx>()?;
    m.add_function(wrap_pyfunction!(parse, m)?)?;
    m.add_function(wrap_pyfunction!(apply, m)?)?;
    m.add_function(wrap_pyfunction!(spell_name, m)?)?;
    m.add_function(wrap_pyfunction!(affine_form, m)?)?;
    m.add_function(wrap_pyfunction!(affine_terms, m)?)?;
    m.add_function(wrap_pyfunction!(affine_sum, m)?)?;
    m.add_function(wrap_pyfunction!(affine_node, m)?)?;
    m.add_function(wrap_pyfunction!(split_anchor, m)?)?;
    m.add_function(wrap_pyfunction!(index_set_size, m)?)?;
    m.add_function(wrap_pyfunction!(placeholder, m)?)?;
    m.add_function(wrap_pyfunction!(is_placeholder, m)?)?;
    m.add("BUILTINS", builtin_names(m.py())?)?;
    m.add("PLACEHOLDER_PREFIX", emmy_ir::expr::PLACEHOLDER_PREFIX)?;
    Ok(())
}
