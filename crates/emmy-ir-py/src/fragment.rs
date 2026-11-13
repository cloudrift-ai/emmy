//! The lane-to-element table of the tensor-core fragments (`emmy_ir::fragment`), as Python reads it.

use emmy_ir::fragment::{self, FragmentTable};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::convert::IntoPyErr;
use crate::expression::PyExpression;

/// One row of the table: what each lane of a warp holds of one fragment.
#[pyclass(
    frozen,
    get_all,
    module = "emmy.compiler.ir.kernel",
    name = "FragmentTable"
)]
pub struct PyFragmentTable {
    /// The lane's coordinates, each a `(name, expression)` over `FRAGMENT_LANE` and the coordinates before it.
    coordinates: Py<PyTuple>,
    /// Per register, per element in packing order: the `(row, column)` offsets from the tile's base.
    registers: Py<PyTuple>,
    /// The lane XOR masks of the shuffles that fold a row over the lanes that share it.
    reduce_xors: Py<PyTuple>,
}

fn wrap(py: Python<'_>, table: &FragmentTable) -> PyResult<PyFragmentTable> {
    let expression = |value| PyExpression::wrap(py, value);
    let mut coordinates = Vec::with_capacity(table.coordinates.len());
    for (name, value) in &table.coordinates {
        coordinates.push((name.to_string(), expression(value)?));
    }
    let mut registers = Vec::with_capacity(table.registers.len());
    for register in &table.registers {
        let mut elements = Vec::with_capacity(register.len());
        for (row, column) in register {
            elements.push((expression(row)?, expression(column)?));
        }
        registers.push(PyTuple::new(py, elements)?);
    }
    Ok(PyFragmentTable {
        coordinates: PyTuple::new(py, coordinates)?.unbind(),
        registers: PyTuple::new(py, registers)?.unbind(),
        reduce_xors: PyTuple::new(py, &table.reduce_xors)?.unbind(),
    })
}

/// The row of the table for a fragment of `role` in `layout` whose elements are `dtype`. Raises `ValueError` for a
/// layout or a role the table does not have.
#[pyfunction]
fn fragment_table(
    py: Python<'_>,
    layout: &str,
    role: &str,
    dtype: &str,
) -> PyResult<PyFragmentTable> {
    wrap(
        py,
        &fragment::fragment_table(layout, role, dtype.parse().py()?).py()?,
    )
}

/// The accumulator of floats of one cell of Volta's paired tile; `cell` is the register cell's index along M and N.
#[pyfunction]
fn paired_accumulator_table(py: Python<'_>, cell: (i64, i64)) -> PyResult<PyFragmentTable> {
    wrap(py, &fragment::paired_accumulator_table(cell))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFragmentTable>()?;
    m.add_function(wrap_pyfunction!(fragment_table, m)?)?;
    m.add_function(wrap_pyfunction!(paired_accumulator_table, m)?)?;
    m.add("FRAGMENT_LANE", fragment::LANE)?;
    m.add("SCALE_SELECTOR_A", fragment::SCALE_SELECTOR_A)?;
    m.add("SCALE_SELECTOR_B", fragment::SCALE_SELECTOR_B)?;
    Ok(())
}
