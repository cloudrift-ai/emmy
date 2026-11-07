//! `emmy.emmy_ir`: the compiler's statement and expression vocabulary as Python classes. Rust owns the data; a field
//! read hands back a held Python object; only basic types cross the boundary.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet};

mod body;
mod convert;
mod expression;
mod statement;

/// The elementwise tables the Python `ElementwiseImpl` reads: one source for both sides.
fn elementwise_tables(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add(
        "COMMUTATIVE_OPS",
        PyFrozenSet::new(py, emmy_ir::kinds::elementwise::COMMUTATIVE.iter())?,
    )?;
    m.add(
        "ASSOCIATIVE_OPS",
        PyFrozenSet::new(py, emmy_ir::kinds::elementwise::ASSOCIATIVE.iter())?,
    )?;
    let identity = PyDict::new(py);
    for (name, value) in emmy_ir::kinds::elementwise::IDENTITY {
        identity.set_item(name, value)?;
    }
    m.add("OP_IDENTITY", identity)?;
    let clusters = PyDict::new(py);
    for (name, representative) in emmy_ir::kinds::elementwise::CLUSTERS {
        clusters.set_item(name, representative)?;
    }
    m.add("OP_CLUSTERS", clusters)?;
    let aliases = PyDict::new(py);
    for (alias, canonical) in emmy_ir::kinds::elementwise::ALIASES {
        aliases.set_item(alias, canonical)?;
    }
    m.add("OP_ALIASES", aliases)?;
    Ok(())
}

#[pymodule(name = "emmy_ir")]
fn emmy_ir_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    expression::register(m)?;
    statement::register_kinds(m)?;
    m.add_function(wrap_pyfunction!(statement::_statement_from_text, m)?)?;
    body::register(m)?;
    elementwise_tables(m)?;
    m.add("KIND_NAMES", statement::kind_names())?;
    m.add(
        "INTEGER_DTYPES",
        PyFrozenSet::new(m.py(), emmy_ir::kinds::dtype::INTEGER_NAMES.iter())?,
    )?;
    Ok(())
}
