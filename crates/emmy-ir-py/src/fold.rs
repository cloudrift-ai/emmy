//! `Fold`, `Twist`, `Recipe` and `Channel` as Python classes: handles on the `emmy-pure` terms.
//!
//! A term is reached by OBJECT: the placement reads one value where it sees one object, so a term has exactly one
//! Python object while any reference to it lives (a table keyed by the term's address holds a weak reference to it),
//! and a term's operands are read back as the objects they were built from. A wrapper holds its operands' objects, so
//! every term reached by walking from a held one keeps its object, and with it its `id`, for the whole walk.

use std::collections::HashMap;
use std::sync::OnceLock;

use emmy_ir::Name;
use emmy_pure::Lambda;
use emmy_pure::fold::{Data, Fold, Parts, Seed};
use emmy_pure::lower::Store;
use emmy_pure::twist::{Channel, Recipe, RecipeData, Twist};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyFrozenSet, PyInt, PyString, PyTuple, PyType};

use crate::body::PyBody;
use crate::body::{Table, purge, tables};
use crate::convert::{
    ELEMENTWISE, Imported, IntoPyErr, axis_from_py, expression_from_py, name_from_py, op_to_py,
    statement_from_py, str_tuple,
};
use crate::lambda::PyLambda;

/// Objects the caller already holds for some terms, by term address.
type Known<'py> = HashMap<usize, Bound<'py, PyFold>>;

static WEAKREF: Imported = Imported::new("weakref", "ref");
static CONTRACTION_VIEW: Imported = Imported::new("emmy.compiler.ir.pure.fold", "ContractionView");
static REDUCTION_VIEW: Imported = Imported::new("emmy.compiler.ir.pure.fold", "ReductionView");
static SLAB_VIEW: Imported = Imported::new("emmy.compiler.ir.pure.fold", "SlabView");
static AFFINE_VIEW: Imported = Imported::new("emmy.compiler.ir.pure.fold", "AffineView");
static EXTENTS: Imported = Imported::new("emmy.compiler.ir.pure.fold", "EXTENTS");

fn names_of(value: &Bound<'_, PyAny>) -> PyResult<Vec<Name>> {
    names(Some(value))
}

static FOLDS: OnceLock<Table<Data, Py<PyAny>>> = OnceLock::new();
static RECIPES: OnceLock<Table<RecipeData, Py<PyAny>>> = OnceLock::new();

/// The live object `table` remembers for `arc`, if any.
fn remembered<'py, T, C: pyo3::PyClass>(
    py: Python<'py>,
    table: &'static Table<T, Py<PyAny>>,
    arc: &std::sync::Arc<T>,
) -> PyResult<Option<Bound<'py, C>>> {
    let reference = table.lock().get(arc, |reference| reference.clone_ref(py));
    if let Some(reference) = reference
        && let Ok(found) = reference.bind(py).call0()?.cast_into::<C>()
    {
        return Ok(Some(found));
    }
    Ok(None)
}

/// Remember `object` as the one Python object of `arc`.
fn remember<T>(
    table: &'static Table<T, Py<PyAny>>,
    arc: &std::sync::Arc<T>,
    object: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let reference = WEAKREF.get(object.py())?.call1((object,))?;
    let (old, due) = table.lock().insert(arc, reference.unbind());
    drop(old);
    if due {
        purge(table);
    }
    Ok(())
}

fn names(value: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<Name>> {
    match value {
        None => Ok(Vec::new()),
        Some(value) if value.is_none() => Ok(Vec::new()),
        Some(value) => {
            if value.is_instance_of::<PyString>() {
                return Err(PyTypeError::new_err(
                    "expected a sequence of names, got a str",
                ));
            }
            value.try_iter()?.map(|item| name_from_py(&item?)).collect()
        }
    }
}

fn optional_lambda(value: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Lambda>> {
    match value {
        None => Ok(None),
        Some(value) if value.is_none() => Ok(None),
        Some(value) => PyLambda::from_py(value).map(Some),
    }
}

fn seed_from_py(value: &Bound<'_, PyAny>) -> PyResult<Seed> {
    if let Ok(text) = value.cast::<PyString>() {
        return Ok(Seed::Buffer(Name::new(text.to_str()?)));
    }
    if value.is_instance_of::<PyInt>() || value.is_instance_of::<PyBool>() {
        return Ok(Seed::Int(value.extract()?));
    }
    Ok(Seed::Float(value.extract::<f64>()?))
}

fn seed_to_py<'py>(py: Python<'py>, seed: &Seed) -> PyResult<Bound<'py, PyAny>> {
    Ok(match seed {
        Seed::Int(value) => value.into_pyobject(py)?.into_any(),
        Seed::Float(value) => PyFloat::new(py, *value).into_any(),
        Seed::Buffer(name) => PyString::new(py, name).into_any(),
    })
}

fn optional<'py>(py: Python<'py>, value: Option<Bound<'py, PyAny>>) -> Bound<'py, PyAny> {
    value.unwrap_or_else(|| py.None().into_bound(py))
}

// ---------------------------------------------------------------------------------------------------------------
// Channel, Recipe, Twist
// ---------------------------------------------------------------------------------------------------------------

/// One state of a fused carrier beyond the pivot's.
#[pyclass(frozen, module = "emmy.compiler.ir.pure", name = "Channel")]
pub struct PyChannel {
    pub channel: Channel,
}

#[pymethods]
impl PyChannel {
    #[new]
    #[pyo3(signature = (injection, pattern = None, name = String::new(), init = 0.0))]
    fn py_new(
        injection: &Bound<'_, PyAny>,
        pattern: Option<&Bound<'_, PyAny>>,
        name: String,
        init: f64,
    ) -> PyResult<Self> {
        Ok(PyChannel {
            channel: Channel {
                injection: PyLambda::from_py(injection)?,
                pattern: optional_lambda(pattern)?,
                name,
                init,
            },
        })
    }

    #[getter]
    fn injection<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyLambda>> {
        PyLambda::wrap(py, self.channel.injection.clone())
    }

    #[getter]
    fn pattern<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyLambda>>> {
        self.channel
            .pattern
            .clone()
            .map(|p| PyLambda::wrap(py, p))
            .transpose()
    }

    #[getter]
    fn name(&self) -> &str {
        &self.channel.name
    }

    #[getter]
    fn init(&self) -> f64 {
        self.channel.init
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<PyChannel>()
            .is_ok_and(|other| other.get().channel == self.channel)
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.channel.injection.hash(&mut hasher);
        self.channel.name.hash(&mut hasher);
        hasher.finish()
    }

    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        let parts = PyTuple::new(
            slf.py(),
            [
                slf.getattr("injection")?,
                slf.getattr("pattern")?,
                slf.getattr("name")?,
                slf.getattr("init")?,
            ],
        )?;
        Ok((slf.get_type().into_any(), parts))
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(format!(
            "Channel(injection={}, pattern={}, name={}, init={})",
            slf.getattr("injection")?.repr()?,
            slf.getattr("pattern")?.repr()?,
            slf.getattr("name")?.repr()?,
            slf.getattr("init")?.repr()?
        ))
    }
}

/// A twisted monoid as data, defined in Python and held here: a fold names its recipe by it.
#[pyclass(frozen, weakref, module = "emmy.compiler.ir.pure", name = "Recipe")]
pub struct PyRecipe {
    pub recipe: Recipe,
    channels: OnceLock<Py<PyTuple>>,
}

impl PyRecipe {
    /// The one Python object of `recipe`.
    pub fn wrap<'py>(py: Python<'py>, recipe: &Recipe) -> PyResult<Bound<'py, PyRecipe>> {
        let table = tables(&RECIPES);
        if let Some(found) = remembered::<_, PyRecipe>(py, table, recipe.arc())? {
            return Ok(found);
        }
        let object = Bound::new(
            py,
            PyRecipe {
                recipe: recipe.clone(),
                channels: OnceLock::new(),
            },
        )?;
        remember(table, recipe.arc(), object.as_any())?;
        Ok(object)
    }
}

fn op_from_py(value: &Bound<'_, PyAny>) -> PyResult<emmy_ir::elementwise::ElementwiseOp> {
    let op = match value.cast::<PyString>() {
        Ok(text) => ELEMENTWISE.get(value.py())?.call1((text,))?,
        Err(_) => value.clone(),
    };
    Ok(emmy_ir::elementwise::ElementwiseOp::new(
        &op.getattr("name")?.extract::<String>()?,
    ))
}

#[pymethods]
impl PyRecipe {
    #[new]
    #[pyo3(signature = (name, base, lift, psi, psi_inv, channels, advance = None, scale = None, combine = None))]
    #[allow(clippy::too_many_arguments)]
    fn py_new<'py>(
        name: String,
        base: &Bound<'py, PyAny>,
        lift: &Bound<'py, PyAny>,
        psi: &Bound<'py, PyAny>,
        psi_inv: &Bound<'py, PyAny>,
        channels: &Bound<'py, PyAny>,
        advance: Option<&Bound<'py, PyAny>>,
        scale: Option<&Bound<'py, PyAny>>,
        combine: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Py<PyRecipe>> {
        let py = base.py();
        let channels: Vec<Channel> = channels
            .try_iter()?
            .map(|item| Ok(item?.cast::<PyChannel>()?.get().channel.clone()))
            .collect::<PyResult<_>>()?;
        let recipe = Recipe::new(RecipeData {
            name,
            base: base
                .try_iter()?
                .map(|op| op_from_py(&op?))
                .collect::<PyResult<_>>()?,
            lift: PyLambda::from_py(lift)?,
            psi: PyLambda::from_py(psi)?,
            psi_inv: PyLambda::from_py(psi_inv)?,
            channels,
            advance: optional_lambda(advance)?,
            scale: optional_lambda(scale)?,
            combine: optional_lambda(combine)?,
        });
        Ok(PyRecipe::wrap(py, &recipe)?.unbind())
    }

    #[getter]
    fn name(&self) -> &str {
        &self.recipe.name
    }

    #[getter]
    fn base<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            self.recipe
                .base
                .iter()
                .map(|op| PyString::new(py, op.name())),
        )
    }

    #[getter]
    fn pivot(&self) -> &str {
        self.recipe.pivot().name()
    }

    #[getter]
    fn lift<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyLambda>> {
        PyLambda::wrap(py, self.recipe.lift.clone())
    }

    #[getter]
    fn psi<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyLambda>> {
        PyLambda::wrap(py, self.recipe.psi.clone())
    }

    #[getter]
    fn psi_inv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyLambda>> {
        PyLambda::wrap(py, self.recipe.psi_inv.clone())
    }

    #[getter]
    fn advance<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyLambda>>> {
        self.recipe
            .advance
            .clone()
            .map(|l| PyLambda::wrap(py, l))
            .transpose()
    }

    #[getter]
    fn scale<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyLambda>>> {
        self.recipe
            .scale
            .clone()
            .map(|l| PyLambda::wrap(py, l))
            .transpose()
    }

    #[getter]
    fn combine<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyLambda>>> {
        self.recipe
            .combine
            .clone()
            .map(|l| PyLambda::wrap(py, l))
            .transpose()
    }

    /// The channels, the same objects every read: a matcher compares the channel it found by object.
    #[getter]
    fn channels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        if let Some(held) = self.channels.get() {
            return Ok(held.bind(py).clone());
        }
        let items: Vec<Bound<'py, PyChannel>> = self
            .recipe
            .channels
            .iter()
            .map(|channel| {
                Bound::new(
                    py,
                    PyChannel {
                        channel: channel.clone(),
                    },
                )
            })
            .collect::<PyResult<_>>()?;
        let tuple = PyTuple::new(py, items)?;
        Ok(self
            .channels
            .get_or_init(|| tuple.unbind())
            .bind(py)
            .clone())
    }

    /// The fused ⊕ over these state names, the second operand `<n>__o`.
    fn program<'py>(
        &self,
        py: Python<'py>,
        states: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyLambda>> {
        PyLambda::wrap(py, self.recipe.program(&names(Some(states))?).py()?)
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<PyRecipe>()
            .is_ok_and(|other| other.get().recipe == self.recipe)
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.recipe.name.hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let mut parts = Vec::new();
        for name in [
            "name", "base", "lift", "psi", "psi_inv", "channels", "advance", "scale", "combine",
        ] {
            parts.push(format!("{name}={}", slf.getattr(name)?.repr()?));
        }
        Ok(format!("Recipe({})", parts.join(", ")))
    }

    /// Transport by the stored parts; one pickle holds one recipe object however many terms name it.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        let mut parts = Vec::new();
        for name in [
            "name", "base", "lift", "psi", "psi_inv", "channels", "advance", "scale", "combine",
        ] {
            parts.push(slf.getattr(name)?);
        }
        Ok((slf.get_type().into_any(), PyTuple::new(slf.py(), parts)?))
    }
}

/// One recipe instantiated on a term: the recipe and which recipe channel each carried state past the pivot is.
#[pyclass(frozen, module = "emmy.compiler.ir.pure", name = "Twist")]
pub struct PyTwist {
    pub twist: Twist,
}

#[pymethods]
impl PyTwist {
    #[new]
    #[pyo3(signature = (recipe, channels))]
    fn py_new(recipe: &Bound<'_, PyAny>, channels: &Bound<'_, PyAny>) -> PyResult<Self> {
        let recipe = recipe.cast::<PyRecipe>()?.get().recipe.clone();
        let channels: Vec<usize> = channels
            .try_iter()?
            .map(|c| c?.extract())
            .collect::<PyResult<_>>()?;
        Ok(PyTwist {
            twist: Twist { recipe, channels },
        })
    }

    #[getter]
    fn recipe<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyRecipe>> {
        PyRecipe::wrap(py, &self.twist.recipe)
    }

    #[getter]
    fn channels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.twist.channels)
    }

    #[getter]
    fn name(&self) -> &str {
        self.twist.name()
    }

    /// `combine` over these state names, in the stable spelling the recipe authored.
    fn program<'py>(
        &self,
        py: Python<'py>,
        states: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyLambda>> {
        PyLambda::wrap(py, self.twist.program(&names(Some(states))?).py()?)
    }

    /// `psi ∘ lift` at the singleton over this term's spelling of the recipe's roles.
    fn inject<'py>(
        &self,
        py: Python<'py>,
        roles: &Bound<'py, PyAny>,
        states: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyLambda>> {
        PyLambda::wrap(
            py,
            self.twist
                .inject(&names(Some(roles))?, &names(Some(states))?)
                .py()?,
        )
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<PyTwist>()
            .is_ok_and(|other| other.get().twist == self.twist)
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.twist.recipe.name.hash(&mut hasher);
        self.twist.channels.hash(&mut hasher);
        hasher.finish()
    }

    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        let parts = PyTuple::new(
            slf.py(),
            [slf.getattr("recipe")?.into_any(), slf.getattr("channels")?],
        )?;
        Ok((slf.get_type().into_any(), parts))
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(format!(
            "Twist(recipe={}, channels={})",
            slf.getattr("recipe")?.repr()?,
            slf.getattr("channels")?.repr()?
        ))
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Fold
// ---------------------------------------------------------------------------------------------------------------

/// The one reduce term of Tile IR. A handle: the term lives in `emmy-pure`, and its reads are answered there.
#[pyclass(frozen, dict, weakref, module = "emmy.compiler.ir.pure", name = "Fold")]
pub struct PyFold {
    pub fold: Fold,
    operands: OnceLock<Py<PyTuple>>,
    lift: OnceLock<Py<PyLambda>>,
    applied: OnceLock<Py<PyLambda>>,
    exposes: OnceLock<Py<PyTuple>>,
    bindings: OnceLock<Py<PyTuple>>,
    free_axes: OnceLock<Py<PyFrozenSet>>,
    contraction: OnceLock<Option<Py<PyAny>>>,
    reduction: OnceLock<Option<Py<PyAny>>>,
    step: OnceLock<Py<PyBody>>,
}

impl PyFold {
    fn holding(fold: Fold) -> PyFold {
        PyFold {
            fold,
            operands: OnceLock::new(),
            lift: OnceLock::new(),
            applied: OnceLock::new(),
            exposes: OnceLock::new(),
            bindings: OnceLock::new(),
            free_axes: OnceLock::new(),
            contraction: OnceLock::new(),
            reduction: OnceLock::new(),
            step: OnceLock::new(),
        }
    }

    /// The one Python object of `fold`. `known` maps term addresses to objects the caller already holds for some of
    /// its operands, which the new object takes as its operands' objects.
    pub fn wrap<'py>(py: Python<'py>, fold: &Fold) -> PyResult<Bound<'py, PyFold>> {
        PyFold::wrap_with(py, fold, &HashMap::new())
    }

    fn wrap_with<'py>(
        py: Python<'py>,
        fold: &Fold,
        known: &HashMap<usize, Bound<'py, PyFold>>,
    ) -> PyResult<Bound<'py, PyFold>> {
        if let Some(found) = known.get(&fold.address()) {
            return Ok(found.clone());
        }
        let table = tables(&FOLDS);
        if let Some(found) = remembered::<_, PyFold>(py, table, fold.arc())? {
            return Ok(found);
        }
        let object = Bound::new(py, PyFold::holding(fold.clone()))?;
        remember(table, fold.arc(), object.as_any())?;
        if !known.is_empty() {
            let items: Vec<Bound<'py, PyFold>> = fold
                .operands()
                .iter()
                .map(|edge| PyFold::wrap_with(py, edge, known))
                .collect::<PyResult<_>>()?;
            let _ = object.get().operands.set(PyTuple::new(py, items)?.unbind());
        }
        Ok(object)
    }

    /// The term `parts` form, its operands read back as the objects in `known`.
    fn formed_from<'py>(
        py: Python<'py>,
        parts: Parts,
        known: HashMap<usize, Bound<'py, PyFold>>,
    ) -> PyResult<Py<PyFold>> {
        let fold = Fold::new(parts).py()?;
        // A rebuild that changes nothing a term stores is that term.
        if let Some(found) = known.get(&fold.address()) {
            return Ok(found.clone().unbind());
        }
        Ok(PyFold::wrap_with(py, &fold, &known)?.unbind())
    }

    /// The terms a Python operand tuple holds, and their objects by term address.
    fn operand_objects<'py>(
        value: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<(Vec<Fold>, Known<'py>)> {
        let mut terms = Vec::new();
        let mut known = HashMap::new();
        let Some(value) = value.filter(|v| !v.is_none()) else {
            return Ok((terms, known));
        };
        let items: Vec<Bound<'py, PyAny>> = value.try_iter()?.collect::<PyResult<_>>()?;
        let stray: Vec<String> = items
            .iter()
            .filter(|item| !item.is_instance_of::<PyFold>())
            .map(|item| {
                item.get_type()
                    .name()
                    .map(|n| n.to_string())
                    .unwrap_or_default()
            })
            .collect();
        if !stray.is_empty() {
            return Err(PyTypeError::new_err(format!(
                "Fold operands must be terms, got {stray:?}; a gmem read is a term over one Load"
            )));
        }
        for item in items {
            let edge = item.cast_into::<PyFold>()?;
            terms.push(edge.get().fold.clone());
            known.insert(edge.get().fold.address(), edge);
        }
        Ok((terms, known))
    }

    fn lift_object<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyLambda>> {
        let py = slf.py();
        let this = slf.get();
        if let Some(held) = this.lift.get() {
            return Ok(held.bind(py).clone());
        }
        let lift = PyLambda::wrap(py, this.fold.lift().clone())?;
        Ok(this.lift.get_or_init(|| lift.unbind()).bind(py).clone())
    }
}

#[pymethods]
impl PyFold {
    #[new]
    #[pyo3(signature = (operands = None, init = None, *, lift, cells = None, base = None, twist = None, observe = None))]
    #[allow(clippy::too_many_arguments)]
    fn py_new<'py>(
        py: Python<'py>,
        operands: Option<&Bound<'py, PyAny>>,
        init: Option<&Bound<'py, PyAny>>,
        lift: &Bound<'py, PyAny>,
        cells: Option<&Bound<'py, PyAny>>,
        base: Option<&Bound<'py, PyAny>>,
        twist: Option<&Bound<'py, PyAny>>,
        observe: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Py<PyFold>> {
        let (terms, known) = PyFold::operand_objects(operands)?;
        let init: Vec<Seed> = match init.filter(|v| !v.is_none()) {
            None => Vec::new(),
            Some(value) => value
                .try_iter()?
                .map(|item| seed_from_py(&item?))
                .collect::<PyResult<_>>()?,
        };
        let twist = match twist.filter(|v| !v.is_none()) {
            None => None,
            Some(value) => Some(value.cast::<PyTwist>()?.get().twist.clone()),
        };
        let parts = Parts {
            operands: terms,
            lift: Some(PyLambda::from_py(lift)?),
            init,
            cells: names(cells)?,
            base: optional_lambda(base)?,
            twist,
            observe: optional_lambda(observe)?,
        };
        let created = PyFold::formed_from(py, parts, known)?;
        // The lift it was given is the lift it holds, when formation kept it.
        if let Ok(given) = lift.cast::<PyLambda>()
            && given.get().lambda.ptr_eq(created.get().fold.lift())
        {
            let _ = created.get().lift.set(given.clone().unbind());
        }
        Ok(created)
    }

    #[classattr]
    fn wire_tag() -> &'static str {
        "Fold"
    }

    /// The operand terms, the same objects every read.
    #[getter]
    fn operands<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let this = slf.get();
        if let Some(held) = this.operands.get() {
            return Ok(held.bind(py).clone());
        }
        let items: Vec<Bound<'py, PyFold>> = this
            .fold
            .operands()
            .iter()
            .map(|edge| PyFold::wrap(py, edge))
            .collect::<PyResult<_>>()?;
        let tuple = PyTuple::new(py, items)?;
        Ok(this
            .operands
            .get_or_init(|| tuple.unbind())
            .bind(py)
            .clone())
    }

    #[getter]
    fn lift<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyLambda>> {
        PyFold::lift_object(slf)
    }

    #[getter]
    fn init<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let items: Vec<Bound<'py, PyAny>> = self
            .fold
            .init()
            .iter()
            .map(|seed| seed_to_py(py, seed))
            .collect::<PyResult<_>>()?;
        PyTuple::new(py, items)
    }

    #[getter]
    fn cells<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        str_tuple(py, self.fold.cells())
    }

    #[getter]
    fn base<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyLambda>>> {
        self.fold
            .base()
            .cloned()
            .map(|l| PyLambda::wrap(py, l))
            .transpose()
    }

    #[getter]
    fn twist<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTwist>>> {
        self.fold
            .twist()
            .map(|t| Bound::new(py, PyTwist { twist: t.clone() }))
            .transpose()
    }

    #[getter]
    fn observe<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyLambda>>> {
        self.fold
            .observe()
            .cloned()
            .map(|l| PyLambda::wrap(py, l))
            .transpose()
    }

    /// The name of the axis this term binds, or `None` for a zero-axis term.
    #[getter]
    fn axis<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyString>> {
        self.fold.axis().map(|axis| PyString::new(py, &axis))
    }

    /// The result names this term defines for its consumer, one per produced component.
    #[getter]
    fn exposes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        if let Some(held) = self.exposes.get() {
            return Ok(held.bind(py).clone());
        }
        let tuple = str_tuple(py, self.fold.exposes())?;
        Ok(self.exposes.get_or_init(|| tuple.unbind()).bind(py).clone())
    }

    /// The positional binding: `(param, edge, component)` for every operand-bound lift param, in order.
    #[getter]
    fn bindings<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let this = slf.get();
        if let Some(held) = this.bindings.get() {
            return Ok(held.bind(py).clone());
        }
        let operands = PyFold::operands(slf)?;
        let items: Vec<Bound<'py, PyTuple>> = this
            .fold
            .bindings()
            .iter()
            .map(|b| {
                PyTuple::new(
                    py,
                    [
                        PyString::new(py, &b.param).into_any(),
                        operands.get_item(b.operand)?,
                        b.component.into_pyobject(py)?.into_any(),
                    ],
                )
            })
            .collect::<PyResult<_>>()?;
        let tuple = PyTuple::new(py, items)?;
        Ok(this
            .bindings
            .get_or_init(|| tuple.unbind())
            .bind(py)
            .clone())
    }

    /// The lift applied to the operands: every operand-bound param spelled as the result it binds.
    #[getter]
    fn applied<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyLambda>> {
        if let Some(held) = self.applied.get() {
            return Ok(held.bind(py).clone());
        }
        let applied = PyLambda::wrap(py, self.fold.applied().py()?.clone())?;
        Ok(self
            .applied
            .get_or_init(|| applied.unbind())
            .bind(py)
            .clone())
    }

    /// Whether this fold carries a state: its ⊕ is the action `next` on every component.
    #[getter]
    fn carries(&self) -> bool {
        self.fold.carries()
    }

    /// The ⊕ this fold folds with, `None` at zero axes.
    #[getter]
    fn combine<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyLambda>>> {
        self.fold
            .combine()
            .py()?
            .map(|l| PyLambda::wrap(py, l))
            .transpose()
    }

    /// The coordinates this term is evaluated over as seen from outside.
    #[getter]
    fn free_axes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
        if let Some(held) = self.free_axes.get() {
            return Ok(held.bind(py).clone());
        }
        let set = PyFrozenSet::new(
            py,
            self.fold
                .free_axes()
                .iter()
                .map(|name| PyString::new(py, name)),
        )?;
        Ok(self.free_axes.get_or_init(|| set.unbind()).bind(py).clone())
    }

    /// The α-canonical form of this term.
    fn canonical<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFold>> {
        PyFold::wrap(py, &self.fold.canonical().py()?)
    }

    /// The one `Pre` of a carrier read, the lift body's own statement, or `None`.
    fn as_carrier_read<'py>(slf: &Bound<'py, Self>) -> PyResult<Option<Bound<'py, PyAny>>> {
        if slf.get().fold.as_carrier_read().is_none() {
            return Ok(None);
        }
        Ok(Some(
            PyFold::lift_object(slf)?.getattr("body")?.get_item(0)?,
        ))
    }

    /// The one `Load` of a slab, the lift body's own statement, or `None`.
    fn _slab_load<'py>(slf: &Bound<'py, Self>) -> PyResult<Option<Bound<'py, PyAny>>> {
        if slf.get().fold.as_slab().is_none() {
            return Ok(None);
        }
        Ok(Some(
            PyFold::lift_object(slf)?.getattr("body")?.get_item(0)?,
        ))
    }

    /// This term restricted to the results `names`.
    fn exposing<'py>(
        &self,
        py: Python<'py>,
        names: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyFold>> {
        let names = names_of(names)?;
        let narrowed = self.fold.exposing(&names).py()?;
        PyFold::wrap(py, &narrowed)
    }

    /// The singleton the ⊕ folds: the applied lift.
    #[getter]
    fn injected<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyLambda>> {
        self.applied(py)
    }

    /// The lift in base coordinates, for recognition only.
    fn based<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyLambda>> {
        PyLambda::wrap(py, self.fold.based().py()?)
    }

    /// This term's spelling of the recipe's roles, `(score, *extras)`.
    #[getter]
    fn roles<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        str_tuple(py, &self.fold.roles().py()?)
    }

    /// Whether this operand varies over nothing a term evaluated over `axes` does not.
    fn uniform_leaf(&self, axes: &Bound<'_, PyAny>) -> PyResult<bool> {
        let axes: std::collections::HashSet<Name> = names_of(axes)?.into_iter().collect();
        Ok(self.fold.uniform_leaf(&axes))
    }

    /// Every bilinear channel as `(state, streamed edge, the result of it the product multiplies)`.
    fn channel_operands<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let operands = PyFold::operands(slf)?;
        let items: Vec<Bound<'py, PyTuple>> = slf
            .get()
            .fold
            .channel_operands()
            .py()?
            .iter()
            .map(|c| {
                PyTuple::new(
                    py,
                    [
                        c.state.into_pyobject(py)?.into_any(),
                        operands.get_item(c.operand)?,
                        PyString::new(py, &c.name).into_any(),
                    ],
                )
            })
            .collect::<PyResult<_>>()?;
        PyTuple::new(py, items)
    }

    /// Every bilinear channel as `(state, streamed edge)`.
    fn bilinear_channels<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let operands = PyFold::operands(slf)?;
        let items: Vec<Bound<'py, PyTuple>> = slf
            .get()
            .fold
            .bilinear_channels()
            .py()?
            .iter()
            .map(|(state, operand)| {
                PyTuple::new(
                    py,
                    [
                        state.into_pyobject(py)?.into_any(),
                        operands.get_item(*operand)?,
                    ],
                )
            })
            .collect::<PyResult<_>>()?;
        PyTuple::new(py, items)
    }

    /// The `ContractionView` of one channel of this term, or `None`.
    fn as_contraction<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        if let Some(held) = self.contraction.get() {
            return Ok(held.as_ref().map(|v| v.bind(py).clone()));
        }
        let view = match self.fold.as_contraction().py()? {
            None => None,
            Some(c) => {
                let set = |names: &std::collections::HashSet<Name>| {
                    PyFrozenSet::new(py, names.iter().map(|n| PyString::new(py, n)))
                };
                let kwargs = PyDict::new(py);
                kwargs.set_item("axis", PyString::new(py, &c.axis))?;
                kwargs.set_item("left_axes", set(&c.left_axes)?)?;
                kwargs.set_item("right_axes", set(&c.right_axes)?)?;
                kwargs.set_item("shared_axes", set(&c.shared_axes)?)?;
                kwargs.set_item("product", op_to_py(py, &c.product)?)?;
                kwargs.set_item("plus", op_to_py(py, &c.plus)?)?;
                kwargs.set_item("b_trans", c.b_trans)?;
                kwargs.set_item("channel", c.channel)?;
                Some(CONTRACTION_VIEW.get(py)?.call((), Some(&kwargs))?.unbind())
            }
        };
        let view = self.contraction.get_or_init(|| view);
        Ok(view.as_ref().map(|v| v.bind(py).clone()))
    }

    /// Whether an atom tier could fold this term whole.
    fn tiles_whole(&self) -> PyResult<bool> {
        self.fold.tiles_whole().py()
    }

    /// Whether the recipe this twisted carrier names lets a tier fold it chunk by chunk.
    fn chunked(&self) -> PyResult<bool> {
        self.fold.chunked().py()
    }

    /// The operand edge a tier streams along this carrier's axis.
    #[getter]
    fn streamed<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let index = slf.get().fold.streamed().py()?;
        PyFold::operands(slf)?.get_item(index)
    }

    /// This carrier as one fold per carried state under a projection, or `None`.
    fn per_state<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyFold>>> {
        self.fold
            .per_state()
            .py()?
            .map(|f| PyFold::wrap(py, &f))
            .transpose()
    }

    /// The `ReductionView` of this term, or `None`.
    fn as_reduction<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        if let Some(held) = self.reduction.get() {
            return Ok(held.as_ref().map(|v| v.bind(py).clone()));
        }
        let view = match self.fold.as_reduction().py()? {
            None => None,
            Some(r) => {
                let kwargs = PyDict::new(py);
                kwargs.set_item("axis", PyString::new(py, &r.axis))?;
                kwargs.set_item("states", str_tuple(py, &r.states)?)?;
                kwargs.set_item("other", str_tuple(py, &r.other)?)?;
                kwargs.set_item("terms", str_tuple(py, &r.terms)?)?;
                let ops = match &r.ops {
                    None => py.None().into_bound(py),
                    Some(ops) => PyTuple::new(
                        py,
                        ops.iter()
                            .map(|op| op_to_py(py, op))
                            .collect::<PyResult<Vec<_>>>()?,
                    )?
                    .into_any(),
                };
                kwargs.set_item("ops", ops)?;
                Some(REDUCTION_VIEW.get(py)?.call((), Some(&kwargs))?.unbind())
            }
        };
        let view = self.reduction.get_or_init(|| view);
        Ok(view.as_ref().map(|v| v.bind(py).clone()))
    }

    /// The `SlabView` of this term — its one gmem read — or `None`.
    fn as_slab<'py>(slf: &Bound<'py, Self>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let Some(load) = PyFold::_slab_load(slf)? else {
            return Ok(None);
        };
        let kwargs = PyDict::new(slf.py());
        kwargs.set_item("load", load)?;
        Ok(Some(SLAB_VIEW.get(slf.py())?.call((), Some(&kwargs))?))
    }

    /// Whether this term is one value for the whole kernel.
    fn scalar(&self) -> bool {
        self.fold.scalar()
    }

    /// The `AffineView` of this carried state, or `None`.
    fn affine<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let Some(a) = self.fold.affine().py()? else {
            return Ok(None);
        };
        let kwargs = PyDict::new(py);
        kwargs.set_item("state", PyString::new(py, &a.state))?;
        kwargs.set_item("mixed", a.mixed)?;
        kwargs.set_item("kept", a.kept)?;
        Ok(Some(AFFINE_VIEW.get(py)?.call((), Some(&kwargs))?))
    }

    /// One gmem read as a term.
    #[classmethod]
    fn slab<'py>(
        _cls: &Bound<'py, PyType>,
        load: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyFold>> {
        let py = load.py();
        PyFold::wrap(py, &Fold::slab(statement_from_py(load)?).py()?)
    }

    /// A read of a carried state as a term.
    #[classmethod]
    fn carrier_read<'py>(
        _cls: &Bound<'py, PyType>,
        read: &Bound<'py, PyAny>,
        axis: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyFold>> {
        let py = read.py();
        PyFold::wrap(
            py,
            &Fold::carrier_read(statement_from_py(read)?, name_from_py(axis)?).py()?,
        )
    }

    /// The combine applied at `(state, other)` as loop-IR statements.
    fn merge<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyBody>> {
        PyBody::wrap(py, &self.fold.merge(&names_of(other)?).py()?)
    }

    /// The per-step statements this fold derives.
    fn step<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBody>> {
        if let Some(held) = self.step.get() {
            return Ok(held.bind(py).clone());
        }
        let body = PyBody::wrap(py, &self.fold.step().py()?)?;
        Ok(self.step.get_or_init(|| body.unbind()).bind(py).clone())
    }

    /// This tree with every operand result component no reader reads dropped, `stored` the names the boundary stores
    /// read.
    fn pruned<'py>(
        &self,
        py: Python<'py>,
        stored: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyFold>> {
        PyFold::wrap(py, &self.fold.pruned(&names_of(stored)?).py()?)
    }

    /// This complete tree normalized under the kernel's boundary stores: `values` the names they write, `reads` every
    /// name they read. The fixpoint is remembered on the result per stored set.
    fn normalized<'py>(
        &self,
        py: Python<'py>,
        values: &Bound<'py, PyAny>,
        reads: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyFold>> {
        let result = self
            .fold
            .normalized(&names_of(values)?, &names_of(reads)?)
            .py()?;
        PyFold::wrap(py, &result)
    }

    /// The components each term in this tree supplies to all its readers and boundary stores, keyed by the `id` of
    /// the term's object.
    #[pyo3(signature = (stored = None))]
    fn read_components<'py>(
        slf: &Bound<'py, Self>,
        stored: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let py = slf.py();
        let stored: std::collections::HashSet<Name> = match stored {
            None => std::collections::HashSet::new(),
            Some(value) => names_of(value)?.into_iter().collect(),
        };
        let taken = slf.get().fold.read_components(&stored).py()?;
        let out = PyDict::new(py);
        let mut pending: Vec<Bound<'py, PyFold>> = vec![slf.clone()];
        let mut seen = std::collections::HashSet::new();
        while let Some(term) = pending.pop() {
            let address = term.get().fold.address();
            if !seen.insert(address) {
                continue;
            }
            if let Some(names) = taken.get(&address) {
                out.set_item(term.as_ptr() as usize, str_tuple(py, names)?)?;
            }
            for edge in PyFold::operands(&term)?.iter() {
                pending.push(edge.cast_into::<PyFold>()?);
            }
        }
        Ok(out)
    }

    /// Flatten this term to the Loop IR nest the materializer expands, the boundary `stores` (output specs) placed in
    /// it; `bound` the coordinates the caller binds (`None`: every free coordinate), `axes` the kernel's axis table.
    /// The symbolic extents the launch binds are read from `EXTENTS`.
    #[pyo3(signature = (bound = None, stores = None, axes = None))]
    fn _lower<'py>(
        &self,
        py: Python<'py>,
        bound: Option<&Bound<'py, PyAny>>,
        stores: Option<&Bound<'py, PyAny>>,
        axes: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyBody>> {
        let bound: Option<std::collections::HashSet<Name>> = match bound.filter(|b| !b.is_none()) {
            None => None,
            Some(value) => Some(names_of(value)?.into_iter().collect()),
        };
        let mut specs = Vec::new();
        if let Some(stores) = stores.filter(|s| !s.is_none()) {
            for spec in stores.try_iter()? {
                let spec = spec?;
                let sweep: Vec<std::sync::Arc<emmy_ir::axis::Axis>> = spec
                    .getattr("sweep")?
                    .try_iter()?
                    .map(|axis| axis_from_py(&axis?, 0))
                    .collect::<PyResult<_>>()?;
                let when = spec.getattr("when")?;
                specs.push(Store {
                    write: statement_from_py(&spec.getattr("write")?)?,
                    sweep,
                    when: if when.is_none() {
                        None
                    } else {
                        Some(expression_from_py(&when)?)
                    },
                });
            }
        }
        let axes: Vec<std::sync::Arc<emmy_ir::axis::Axis>> = match axes.filter(|a| !a.is_none()) {
            None => Vec::new(),
            Some(value) => value
                .try_iter()?
                .map(|axis| axis_from_py(&axis?, 0))
                .collect::<PyResult<_>>()?,
        };
        let extents: std::collections::HashSet<Name> =
            names_of(&EXTENTS.get(py)?.call_method0("get")?)?
                .into_iter()
                .collect();
        let body = self
            .fold
            .lower(bound.as_ref(), &specs, &axes, &extents)
            .py()?;
        PyBody::wrap(py, &body)
    }

    /// This term with some of its parts replaced, formed and checked like any other.
    #[pyo3(signature = (**changes))]
    fn replace<'py>(
        slf: &Bound<'py, Self>,
        changes: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Py<PyFold>> {
        let py = slf.py();
        let get = |name: &str| -> PyResult<Option<Bound<'py, PyAny>>> {
            Ok(match changes {
                Some(changes) => changes.get_item(name)?,
                None => None,
            })
        };
        if let Some(changes) = changes {
            for key in changes.keys() {
                let key: String = key.extract()?;
                if !matches!(
                    key.as_str(),
                    "operands" | "lift" | "init" | "cells" | "base" | "twist" | "observe"
                ) {
                    return Err(PyTypeError::new_err(format!("Fold has no field {key:?}")));
                }
            }
        }
        let this = slf.get();
        let mut parts = this.fold.parts();
        let mut known: HashMap<usize, Bound<'py, PyFold>> = HashMap::new();
        match get("operands")? {
            Some(value) => {
                let (terms, objects) = PyFold::operand_objects(Some(&value))?;
                parts.operands = terms;
                known = objects;
            }
            None => {
                for item in PyFold::operands(slf)?.iter() {
                    let edge = item.cast_into::<PyFold>()?;
                    known.insert(edge.get().fold.address(), edge);
                }
            }
        }
        if let Some(value) = get("lift")? {
            parts.lift = Some(PyLambda::from_py(&value)?);
        }
        if let Some(value) = get("init")? {
            parts.init = value
                .try_iter()?
                .map(|item| seed_from_py(&item?))
                .collect::<PyResult<_>>()?;
        }
        if let Some(value) = get("cells")? {
            parts.cells = names(Some(&value))?;
        }
        if let Some(value) = get("base")? {
            parts.base = optional_lambda(Some(&value))?;
        }
        if let Some(value) = get("twist")? {
            parts.twist = if value.is_none() {
                None
            } else {
                Some(value.cast::<PyTwist>()?.get().twist.clone())
            };
        }
        if let Some(value) = get("observe")? {
            parts.observe = optional_lambda(Some(&value))?;
        }
        PyFold::formed_from(py, parts, known)
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<PyFold>()
            .is_ok_and(|other| other.get().fold == self.fold)
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.fold.hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let part = |name: &str| -> PyResult<String> { Ok(slf.getattr(name)?.repr()?.to_string()) };
        Ok(format!(
            "Fold(operands={}, lift={}, init={}, cells={}, base={}, twist={}, observe={})",
            part("operands")?,
            part("lift")?,
            part("init")?,
            part("cells")?,
            part("base")?,
            part("twist")?,
            part("observe")?
        ))
    }

    /// Transport by the stored parts; the far side forms the term without checking it again, and a term shared by
    /// two positions stays one object, as pickle keeps it.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let module = py.import("emmy.emmy_ir")?;
        let parts = PyTuple::new(
            py,
            [
                PyFold::operands(slf)?.into_any(),
                PyFold::lift_object(slf)?.into_any(),
                slf.getattr("init")?,
                slf.getattr("cells")?,
                optional(py, slf.getattr("base").ok()),
                optional(py, slf.getattr("twist").ok()),
                optional(py, slf.getattr("observe").ok()),
            ],
        )?;
        Ok((module.getattr("_formed_fold")?, parts))
    }
}

/// A term from its stored parts, taken as formed: what transport reads back.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn _formed_fold<'py>(
    py: Python<'py>,
    operands: &Bound<'py, PyAny>,
    lift: &Bound<'py, PyAny>,
    init: &Bound<'py, PyAny>,
    cells: &Bound<'py, PyAny>,
    base: &Bound<'py, PyAny>,
    twist: &Bound<'py, PyAny>,
    observe: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyFold>> {
    let (terms, known) = PyFold::operand_objects(Some(operands))?;
    let fold = Fold::formed(Parts {
        operands: terms,
        lift: Some(PyLambda::from_py(lift)?),
        init: init
            .try_iter()?
            .map(|item| seed_from_py(&item?))
            .collect::<PyResult<_>>()?,
        cells: names(Some(cells))?,
        base: optional_lambda(Some(base))?,
        twist: if twist.is_none() {
            None
        } else {
            Some(twist.cast::<PyTwist>()?.get().twist.clone())
        },
        observe: optional_lambda(Some(observe))?,
    });
    let object = Bound::new(py, PyFold::holding(fold.clone()))?;
    remember(tables(&FOLDS), fold.arc(), object.as_any())?;
    let items: Vec<Bound<'py, PyFold>> = fold
        .operands()
        .iter()
        .map(|edge| known[&edge.address()].clone())
        .collect();
    let _ = object.get().operands.set(PyTuple::new(py, items)?.unbind());
    Ok(object)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyChannel>()?;
    m.add_class::<PyRecipe>()?;
    m.add_class::<PyTwist>()?;
    m.add_class::<PyFold>()?;
    m.add_function(wrap_pyfunction!(_formed_fold, m)?)?;
    Ok(())
}
