//! `Fold`, `Twist`, `Recipe` and `Channel` as Python classes: handles on the `emmy-pure` terms.
//!
//! The crate interns terms: equal stored parts are one term. A term has one Python object while any reference to it
//! lives (a table keyed by the term's address holds a weak reference to it), so `is` in Python asks what equality
//! asks in the crate. A wrapper holds its operands' objects, so every term reached by walking from a held one keeps
//! its object, and with it its `id`, for the whole walk.

use std::collections::HashSet;
use std::sync::{Arc, OnceLock};

use emmy_ir::Name;
use emmy_pure::Lambda;
use emmy_pure::fold::{Data, Fold, Parts, Seed};
use emmy_pure::fuse::Scope;
use emmy_pure::lower::Store;
use emmy_pure::readings::{Affine, Contraction, Reduction};
use emmy_pure::rewrite::TermRewrite;
use emmy_pure::twist::{Channel, Recipe, RecipeData, Twist};
use pyo3::exceptions::PyTypeError;
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyFrozenSet, PyInt, PyString, PyTuple, PyType};

use crate::body::PyBody;
use crate::body::{Table, purge, tables};
use crate::convert::{
    ELEMENTWISE, Imported, IntoPyErr, Slots, axis_from_py, axis_to_py, cached, expression_from_py,
    name_from_py, op_to_py, slots, statement_from_py, str_tuple,
};
use crate::expression::{ctx_from_py, renames_from_py};
use crate::lambda::PyLambda;
use crate::statement::sigma_from_py;

static WEAKREF: Imported = Imported::new("weakref", "ref");
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
// Views
// ---------------------------------------------------------------------------------------------------------------

fn name_set<'py>(py: Python<'py>, names: &HashSet<Name>) -> PyResult<Bound<'py, PyFrozenSet>> {
    PyFrozenSet::new(py, names.iter().map(|name| PyString::new(py, name)))
}

fn only<'py>(py: Python<'py>, names: &HashSet<Name>) -> Option<Bound<'py, PyString>> {
    (names.len() == 1).then(|| PyString::new(py, names.iter().next().expect("one name")))
}

/// A Fold's reading as one gmem read: a leaf evaluated over the coordinates it indexes. `load` is the read itself.
/// Placement never offers one as a seam: there is nothing to materialize that the load does not already do.
#[pyclass(frozen, get_all, module = "emmy.compiler.ir.pure", name = "SlabView")]
pub struct PySlabView {
    load: Py<PyAny>,
}

/// A Fold's BILINEAR reading, as geometry: the axis its operands share and the free axes each one contributes.
/// `a[m,k] × b[n,k]` reads as `axis=k`, `left_axes={m}`, `right_axes={n}`; which of the two is physically M or N is
/// the placement's answer, not the term's.
///
/// - `left_axes`: the free axes A carries and B does not: the output role it strides, plus any broadcast batch
///   coordinate riding A alone.
/// - `right_axes`: the free axes the streamed operand carries alone; empty for a matvec. A pair with no output role
///   on either side is not a contraction at all: it reads as a planar reduce.
/// - `shared_axes`: the free axes BOTH operands read besides the reduction: a batch the pair rides, a split-K
///   partition coordinate, a value-dead reshape residue. Which of those it is takes the kernel's extents to tell, so
///   it is the tile's question (`TileOp.contracts`).
/// - `product`, `plus`: the semiring instance: the ⊗ the pair multiplies with and the ⊕ the products fold through.
/// - `b_trans`: whether the streamed operand is stored N-major, its reduction axis last. A gmem layout fact, so a
///   computed B answers `False`. A is k-last by canonical form.
/// - `channel`: which carried state the products fold into; `0` for a plain contraction.
#[pyclass(frozen, module = "emmy.compiler.ir.pure", name = "ContractionView")]
pub struct PyContractionView {
    view: Contraction,
}

#[pymethods]
impl PyContractionView {
    #[getter]
    fn axis(&self) -> &str {
        &self.view.axis
    }

    #[getter]
    fn left_axes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
        name_set(py, &self.view.left_axes)
    }

    #[getter]
    fn right_axes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
        name_set(py, &self.view.right_axes)
    }

    #[getter]
    fn shared_axes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
        name_set(py, &self.view.shared_axes)
    }

    #[getter]
    fn product<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        op_to_py(py, &self.view.product)
    }

    #[getter]
    fn plus<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        op_to_py(py, &self.view.plus)
    }

    #[getter]
    fn b_trans(&self) -> bool {
        self.view.b_trans
    }

    #[getter]
    fn channel(&self) -> usize {
        self.view.channel
    }

    /// A's one own axis, or `None` when it carries none or several (a broadcast batch).
    #[getter]
    fn left<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyString>> {
        only(py, &self.view.left_axes)
    }

    /// B's one own axis, or `None` when it carries none or several.
    #[getter]
    fn right<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyString>> {
        only(py, &self.view.right_axes)
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<PyContractionView>()
            .is_ok_and(|other| other.get().view == self.view)
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.view)
    }
}

/// A Fold's reading as a MONOID fold: the algebra its combine spells, read off the lambda. `states` are the carried
/// state's names (the combine's results), `other` the names the combine binds its second operand to, `terms` the
/// injection singleton (the lift's results, one per component). `ops` is the componentwise op vector when the
/// combine has that shape (a PLANAR fold: sum, max, mean) and `None` for a TWISTED one: the one discrimination every
/// partition legality question asks. No family name: the program itself is the reading, and `Fold.merge` applies it.
#[pyclass(frozen, module = "emmy.compiler.ir.pure", name = "ReductionView")]
pub struct PyReductionView {
    view: Reduction,
}

#[pymethods]
impl PyReductionView {
    #[getter]
    fn axis(&self) -> &str {
        &self.view.axis
    }

    #[getter]
    fn states<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        str_tuple(py, &self.view.states)
    }

    #[getter]
    fn other<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        str_tuple(py, &self.view.other)
    }

    #[getter]
    fn terms<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        str_tuple(py, &self.view.terms)
    }

    #[getter]
    fn ops<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        self.view
            .ops
            .as_ref()
            .map(|ops| {
                let items = ops
                    .iter()
                    .map(|op| op_to_py(py, op))
                    .collect::<PyResult<Vec<_>>>()?;
                PyTuple::new(py, items)
            })
            .transpose()
    }

    #[getter]
    fn twisted(&self) -> bool {
        self.view.ops.is_none()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<PyReductionView>()
            .is_ok_and(|other| other.get().view == self.view)
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.view)
    }
}

/// A carried state's step read as AFFINE in the state, acting column-wise: every carrier read keeps the cell's own
/// coordinate at every position but `mixed`, so the step is `S ← A·S + b` with `A` a matrix over the mixed axis
/// alone, and two steps compose as affine maps. `kept` is the own position a probe packs basis vectors along. A step
/// that reads its own cell alone has no mixed position (`None`): `A` is diagonal.
#[pyclass(frozen, module = "emmy.compiler.ir.pure", name = "AffineView")]
pub struct PyAffineView {
    view: Affine,
}

#[pymethods]
impl PyAffineView {
    #[getter]
    fn state(&self) -> &str {
        &self.view.state
    }

    #[getter]
    fn mixed(&self) -> Option<usize> {
        self.view.mixed
    }

    #[getter]
    fn kept(&self) -> Option<usize> {
        self.view.kept
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<PyAffineView>()
            .is_ok_and(|other| other.get().view == self.view)
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.view)
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Fold
// ---------------------------------------------------------------------------------------------------------------

/// The Python objects a wrapper hands back for its term's reads, one slot each.
#[derive(Clone, Copy)]
enum Held {
    Operands,
    Lift,
    Applied,
    Exposes,
    Bindings,
    FreeAxes,
    Contraction,
    Reduction,
    Step,
}

/// The one reduce term of Tile IR. A handle: the term lives in `emmy-pure`, and its reads are answered there.
#[pyclass(frozen, dict, weakref, module = "emmy.compiler.ir.pure", name = "Fold")]
pub struct PyFold {
    pub fold: Fold,
    held: Slots,
}

fn seeds(value: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<Seed>> {
    match value.filter(|v| !v.is_none()) {
        None => Ok(Vec::new()),
        Some(value) => value.try_iter()?.map(|item| seed_from_py(&item?)).collect(),
    }
}

fn optional_twist(value: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Twist>> {
    match value.filter(|v| !v.is_none()) {
        None => Ok(None),
        Some(value) => Ok(Some(value.cast::<PyTwist>()?.get().twist.clone())),
    }
}

fn axes_from_py(value: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<Arc<emmy_ir::axis::Axis>>> {
    match value.filter(|v| !v.is_none()) {
        None => Ok(Vec::new()),
        Some(value) => value.try_iter()?.map(|axis| axis_from_py(&axis?, 0)).collect(),
    }
}

/// The symbolic extents the launch binds, which a term reads free.
fn extents(py: Python<'_>) -> PyResult<HashSet<Name>> {
    Ok(names_of(&EXTENTS.get(py)?.call_method0("get")?)?
        .into_iter()
        .collect())
}

impl PyFold {
    /// The one Python object of `fold`.
    pub fn wrap<'py>(py: Python<'py>, fold: &Fold) -> PyResult<Bound<'py, PyFold>> {
        let table = tables(&FOLDS);
        if let Some(found) = remembered::<_, PyFold>(py, table, fold.arc())? {
            return Ok(found);
        }
        let object = Bound::new(
            py,
            PyFold {
                fold: fold.clone(),
                held: slots(Held::Step as usize + 1),
            },
        )?;
        remember(table, fold.arc(), object.as_any())?;
        Ok(object)
    }

    /// The terms a Python operand tuple holds.
    fn terms(value: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<Fold>> {
        let Some(value) = value.filter(|v| !v.is_none()) else {
            return Ok(Vec::new());
        };
        let items: Vec<Bound<'_, PyAny>> = value.try_iter()?.collect::<PyResult<_>>()?;
        let stray: Vec<String> = items
            .iter()
            .filter(|item| !item.is_instance_of::<PyFold>())
            .map(|item| crate::convert::type_name(item))
            .collect();
        if !stray.is_empty() {
            return Err(PyTypeError::new_err(format!(
                "Fold operands must be terms, got {stray:?}; a gmem read is a term over one Load"
            )));
        }
        items
            .iter()
            .map(|item| Ok(item.cast::<PyFold>()?.get().fold.clone()))
            .collect()
    }

    fn parts(
        operands: Option<&Bound<'_, PyAny>>,
        lift: &Bound<'_, PyAny>,
        init: Option<&Bound<'_, PyAny>>,
        cells: Option<&Bound<'_, PyAny>>,
        base: Option<&Bound<'_, PyAny>>,
        twist: Option<&Bound<'_, PyAny>>,
        observe: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Parts> {
        Ok(Parts {
            operands: PyFold::terms(operands)?,
            lift: Some(PyLambda::from_py(lift)?),
            init: seeds(init)?,
            cells: names(cells)?,
            base: optional_lambda(base)?,
            twist: optional_twist(twist)?,
            observe: optional_lambda(observe)?,
        })
    }

    fn held<'py>(
        slf: &Bound<'py, Self>,
        slot: Held,
        make: impl FnOnce() -> PyResult<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        cached(slf.py(), &slf.get().held[slot as usize], || {
            Ok(make()?.unbind())
        })
    }

    /// The first statement of the lift body, as the lift's own Python object spells it.
    fn leaf<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        PyFold::lift(slf)?.getattr("body")?.get_item(0)
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
        let parts = PyFold::parts(operands, lift, init, cells, base, twist, observe)?;
        Ok(PyFold::wrap(py, &Fold::new(parts).py()?)?.unbind())
    }

    /// A term from its stored parts, taken as formed: what transport reads back.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn _formed<'py>(
        py: Python<'py>,
        operands: &Bound<'py, PyAny>,
        lift: &Bound<'py, PyAny>,
        init: &Bound<'py, PyAny>,
        cells: &Bound<'py, PyAny>,
        base: &Bound<'py, PyAny>,
        twist: &Bound<'py, PyAny>,
        observe: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyFold>> {
        let parts = PyFold::parts(
            Some(operands),
            lift,
            Some(init),
            Some(cells),
            Some(base),
            Some(twist),
            Some(observe),
        )?;
        PyFold::wrap(py, &Fold::formed(parts))
    }

    #[classattr]
    fn wire_tag() -> &'static str {
        "Fold"
    }

    /// The operand terms, the same objects every read.
    #[getter]
    fn operands<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        PyFold::held(slf, Held::Operands, || {
            let items = slf
                .get()
                .fold
                .operands()
                .iter()
                .map(|edge| PyFold::wrap(slf.py(), edge))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyTuple::new(slf.py(), items)?.into_any())
        })
    }

    #[getter]
    fn lift<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        PyFold::held(slf, Held::Lift, || {
            Ok(PyLambda::wrap(slf.py(), slf.get().fold.lift().clone())?.into_any())
        })
    }

    #[getter]
    fn init<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let items = self
            .fold
            .init()
            .iter()
            .map(|seed| seed_to_py(py, seed))
            .collect::<PyResult<Vec<_>>>()?;
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
    fn exposes<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        PyFold::held(slf, Held::Exposes, || {
            Ok(str_tuple(slf.py(), slf.get().fold.exposes())?.into_any())
        })
    }

    /// The positional binding: `(param, edge, component)` for every operand-bound lift param, in order.
    #[getter]
    fn bindings<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        PyFold::held(slf, Held::Bindings, || {
            let py = slf.py();
            let operands = PyFold::operands(slf)?;
            let items = slf
                .get()
                .fold
                .bindings()
                .iter()
                .map(|b| (&*b.param, operands.get_item(b.operand)?, b.component).into_pyobject(py))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyTuple::new(py, items)?.into_any())
        })
    }

    /// The lift applied to the operands: every operand-bound param spelled as the result it binds.
    #[getter]
    fn applied<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        PyFold::held(slf, Held::Applied, || {
            Ok(PyLambda::wrap(slf.py(), slf.get().fold.applied().py()?.clone())?.into_any())
        })
    }

    /// The singleton the ⊕ folds: the applied lift.
    #[getter]
    fn injected<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        PyFold::applied(slf)
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
    fn free_axes<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        PyFold::held(slf, Held::FreeAxes, || {
            Ok(name_set(slf.py(), slf.get().fold.free_axes())?.into_any())
        })
    }

    /// The α-canonical form of this term.
    fn canonical<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFold>> {
        PyFold::wrap(py, &self.fold.canonical().py()?)
    }

    /// The one `Pre` of a carrier read, the lift body's own statement, or `None`.
    fn as_carrier_read<'py>(slf: &Bound<'py, Self>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match slf.get().fold.as_carrier_read() {
            None => Ok(None),
            Some(_) => PyFold::leaf(slf).map(Some),
        }
    }

    /// The `SlabView` of this term, its one gmem read, or `None`.
    fn as_slab<'py>(slf: &Bound<'py, Self>) -> PyResult<Option<PySlabView>> {
        match slf.get().fold.as_slab() {
            None => Ok(None),
            Some(_) => Ok(Some(PySlabView {
                load: PyFold::leaf(slf)?.unbind(),
            })),
        }
    }

    /// This term restricted to the results `names`.
    fn exposing<'py>(
        &self,
        py: Python<'py>,
        names: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyFold>> {
        PyFold::wrap(py, &self.fold.exposing(&names_of(names)?).py()?)
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
        let axes: HashSet<Name> = names_of(axes)?.into_iter().collect();
        Ok(self.fold.uniform_leaf(&axes))
    }

    /// Every bilinear channel as `(state, streamed edge, the result of it the product multiplies)`.
    fn channel_operands<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let operands = PyFold::operands(slf)?;
        let items = slf
            .get()
            .fold
            .channel_operands()
            .py()?
            .iter()
            .map(|c| (c.state, operands.get_item(c.operand)?, &*c.name).into_pyobject(py))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, items)
    }

    /// Every bilinear channel as `(state, streamed edge)`.
    fn bilinear_channels<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let operands = PyFold::operands(slf)?;
        let items = slf
            .get()
            .fold
            .bilinear_channels()
            .py()?
            .iter()
            .map(|(state, operand)| (*state, operands.get_item(*operand)?).into_pyobject(py))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, items)
    }

    /// The `ContractionView` of one channel of this term, or `None`.
    fn as_contraction<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        PyFold::held(slf, Held::Contraction, || {
            slf.get()
                .fold
                .as_contraction()
                .py()?
                .map(|view| PyContractionView { view })
                .into_bound_py_any(slf.py())
        })
    }

    /// The `ReductionView` of this term, or `None`.
    fn as_reduction<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        PyFold::held(slf, Held::Reduction, || {
            slf.get()
                .fold
                .as_reduction()
                .py()?
                .map(|view| PyReductionView { view })
                .into_bound_py_any(slf.py())
        })
    }

    /// The `AffineView` of this carried state, or `None`.
    fn affine(&self) -> PyResult<Option<PyAffineView>> {
        Ok(self.fold.affine().py()?.map(|view| PyAffineView { view }))
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

    /// Whether this term is one value for the whole kernel.
    fn scalar(&self) -> bool {
        self.fold.scalar()
    }

    /// One gmem read as a term.
    #[classmethod]
    fn slab<'py>(
        _cls: &Bound<'py, PyType>,
        load: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyFold>> {
        PyFold::wrap(load.py(), &Fold::slab(statement_from_py(load)?).py()?)
    }

    /// A read of a carried state as a term.
    #[classmethod]
    fn carrier_read<'py>(
        _cls: &Bound<'py, PyType>,
        read: &Bound<'py, PyAny>,
        axis: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyFold>> {
        PyFold::wrap(
            read.py(),
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
    fn step<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        PyFold::held(slf, Held::Step, || {
            Ok(PyBody::wrap(slf.py(), &slf.get().fold.step().py()?)?.into_any())
        })
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
        let stored: HashSet<Name> = names(stored)?.into_iter().collect();
        let taken = slf.get().fold.read_components(&stored).py()?;
        let out = PyDict::new(py);
        let mut pending: Vec<Bound<'py, PyFold>> = vec![slf.clone()];
        let mut seen = HashSet::new();
        while let Some(term) = pending.pop() {
            let address = term.get().fold.address();
            if !seen.insert(address) {
                continue;
            }
            if let Some(names) = taken.get(&address) {
                out.set_item(term.as_ptr() as usize, str_tuple(py, names)?)?;
            }
            for edge in PyFold::operands(&term)?.try_iter()? {
                pending.push(edge?.cast_into::<PyFold>()?);
            }
        }
        Ok(out)
    }

    /// Flatten this term to the Loop IR nest the materializer expands, the kernel's boundary `stores` (output specs)
    /// placed in it. The ONE lowering spelling: every consumer of a term's statements calls this, and the term
    /// remembers each answer.
    ///
    /// `bound` names the coordinates the CALLER binds (the kernel grid, an enclosing loop's scope): `None` binds
    /// every free coordinate (the open body a term spells inside an enclosing scope) and an empty set the closed
    /// program. `axes` is the kernel's axis table: the extent and window of every axis the tree binds or reads, which
    /// a term carries none of. The symbolic extents the launch binds are read from `EXTENTS`.
    ///
    /// One rule places every term: a term is materialized at the SHALLOWEST scope on its path that binds all of its
    /// `free_axes`, a value of the tree it reads by name standing for the coordinates of the term that defines it.
    /// The scopes are the plain loops this term opens for the free coordinates left unbound (outermost the
    /// coordinate the most terms are evaluated over, ties in declaration order) and the reduce loop of each term on
    /// the way down. Operands are placed before the statements that read them, and a SHARED term defines its names
    /// once per scope. The free loops form a tree: only a wrapper with no step of its own leaves its operands free
    /// to take their own paths, which is how two output sweeps no term shares become sibling loops.
    ///
    /// A boundary store follows the term that DEFINES the value it writes, at that term's scope: a projection's
    /// result after its step, a reduce's carried state after its loop, an observed per-step value inside the loop. A
    /// store evaluated over a coordinate no term declares opens that coordinate under its term, the spec's `sweep`
    /// naming the axis.
    #[pyo3(signature = (bound = None, stores = None, axes = None))]
    fn lower<'py>(
        &self,
        py: Python<'py>,
        bound: Option<&Bound<'py, PyAny>>,
        stores: Option<&Bound<'py, PyAny>>,
        axes: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyBody>> {
        let bound: Option<HashSet<Name>> = match bound.filter(|b| !b.is_none()) {
            None => None,
            Some(value) => Some(names_of(value)?.into_iter().collect()),
        };
        let mut specs = Vec::new();
        if let Some(stores) = stores.filter(|s| !s.is_none()) {
            for spec in stores.try_iter()? {
                let spec = spec?;
                let when = spec.getattr("when")?;
                specs.push(Store {
                    write: statement_from_py(&spec.getattr("write")?)?,
                    sweep: axes_from_py(Some(&spec.getattr("sweep")?))?,
                    when: if when.is_none() {
                        None
                    } else {
                        Some(expression_from_py(&when)?)
                    },
                });
            }
        }
        let body = self
            .fold
            .lower(bound.as_ref(), &specs, &axes_from_py(axes)?, &extents(py)?)
            .py()?;
        PyBody::wrap(py, &body)
    }

    /// This reduce fused onto the reduce it reads, by `recipe`: one fold carrying both states under the recipe's
    /// twisted ⊕, or `None` when no channel of the recipe clicks. `axes` is the kernel's axes by name. The same
    /// recipe fuses online softmax and flash attention alike; the caller rewrites the tree's operands onto the result.
    fn fuse<'py>(
        &self,
        py: Python<'py>,
        recipe: &Bound<'py, PyRecipe>,
        axes: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyFold>>> {
        let axes = axes_from_py(Some(&axes.call_method0("values")?))?;
        let scope = Scope {
            axes: &axes,
            extents: &extents(py)?,
        };
        self.fold
            .fuse(&recipe.get().recipe, &scope)
            .py()?
            .map(|fused| PyFold::wrap(py, &fused))
            .transpose()
    }

    /// Every SSA and axis name this tree spells: what a rename by a function is asked about.
    fn _names<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        str_tuple(py, &self.fold.names())
    }

    /// Every axis a statement of this tree binds: what an axis map by a function is asked about.
    fn _bound_axes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let items = self
            .fold
            .bound_axes()
            .iter()
            .map(|axis| axis_to_py(py, axis))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, items)
    }

    /// Rename, substitute and map axes over this tree: `names` maps SSA names, `sigma` substitutes expressions,
    /// folded under `folding` once they land when one is given, `axes` pairs each bound axis with its replacement.
    #[pyo3(signature = (names, sigma = None, folding = None, axes = None))]
    fn _rewrite<'py>(
        &self,
        py: Python<'py>,
        names: &Bound<'py, PyAny>,
        sigma: Option<&Bound<'py, PyAny>>,
        folding: Option<&Bound<'py, PyAny>>,
        axes: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyFold>> {
        let mut rewrite = TermRewrite {
            names: renames_from_py(names)?,
            sigma: sigma
                .filter(|s| !s.is_none())
                .map(sigma_from_py)
                .transpose()?
                .unwrap_or_default(),
            folding: folding
                .filter(|f| !f.is_none())
                .map(ctx_from_py)
                .transpose()?,
            axes: Vec::new(),
        };
        // A rename that keeps a name is no rename.
        rewrite.names.retain(|from, to| from != to);
        if let Some(axes) = axes.filter(|a| !a.is_none()) {
            for pair in axes.try_iter()? {
                let (old, new): (Bound<'py, PyAny>, Bound<'py, PyAny>) = pair?.extract()?;
                rewrite
                    .axes
                    .push((axis_from_py(&old, 0)?, axis_from_py(&new, 0)?));
            }
        }
        PyFold::wrap(py, &self.fold.rewrite(&rewrite).py()?)
    }

    /// This term with some of its parts replaced, formed and checked like any other.
    #[pyo3(signature = (**changes))]
    fn replace<'py>(
        &self,
        py: Python<'py>,
        changes: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyFold>> {
        let mut parts = self.fold.parts();
        for (key, value) in changes.into_iter().flatten() {
            let value = Some(&value);
            match key.extract::<String>()?.as_str() {
                "operands" => parts.operands = PyFold::terms(value)?,
                "lift" => parts.lift = optional_lambda(value)?,
                "init" => parts.init = seeds(value)?,
                "cells" => parts.cells = names(value)?,
                "base" => parts.base = optional_lambda(value)?,
                "twist" => parts.twist = optional_twist(value)?,
                "observe" => parts.observe = optional_lambda(value)?,
                other => return Err(PyTypeError::new_err(format!("Fold has no field {other:?}"))),
            }
        }
        PyFold::wrap(py, &Fold::new(parts).py()?)
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

    /// Transport by the stored parts; the far side takes the term as formed.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        let parts = ["operands", "lift", "init", "cells", "base", "twist", "observe"]
            .iter()
            .map(|name| slf.getattr(*name))
            .collect::<PyResult<Vec<_>>>()?;
        Ok((
            slf.get_type().getattr("_formed")?,
            PyTuple::new(slf.py(), parts)?,
        ))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyChannel>()?;
    m.add_class::<PyRecipe>()?;
    m.add_class::<PyTwist>()?;
    m.add_class::<PySlabView>()?;
    m.add_class::<PyContractionView>()?;
    m.add_class::<PyReductionView>()?;
    m.add_class::<PyAffineView>()?;
    m.add_class::<PyFold>()?;
    Ok(())
}
