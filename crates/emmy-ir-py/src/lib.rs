//! `emmy.emmy_ir`: the compiler's statement and expression vocabulary as Python classes. Rust owns the data; a field
//! read hands back a held Python object; only basic types cross the boundary.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet};

mod body;
mod convert;
mod expression;
mod fragment;
mod lambda;
mod ptx;
mod statement;

/// The elementwise tables the Python `ElementwiseImpl` reads, each a view of the operations' properties.
fn elementwise_tables(m: &Bound<'_, PyModule>) -> PyResult<()> {
    use emmy_ir::elementwise::{ALIASES, Operation, OperationInfo};
    let py = m.py();
    fn texts(keep: fn(&OperationInfo) -> bool) -> Vec<&'static str> {
        Operation::ALL
            .iter()
            .copied()
            .filter(|op| keep(op.info()))
            .map(Operation::text)
            .collect()
    }
    m.add(
        "COMMUTATIVE_OPS",
        PyFrozenSet::new(py, texts(|info| info.commutative))?,
    )?;
    m.add(
        "ASSOCIATIVE_OPS",
        PyFrozenSet::new(py, texts(|info| info.associative))?,
    )?;
    m.add(
        "SELECTING_OPS",
        PyFrozenSet::new(py, texts(|info| info.selecting))?,
    )?;
    // What a kernel seeds a reduction with: the bounded stand-in of an unbounded neutral element.
    let identity = PyDict::new(py);
    for op in Operation::ALL {
        if let Some(value) = op.seed() {
            identity.set_item(op.text(), value)?;
        }
    }
    m.add("OP_IDENTITY", identity)?;
    let clusters = PyDict::new(py);
    for op in Operation::ALL {
        if let Some(representative) = op.info().cluster {
            clusters.set_item(op.text(), representative.text())?;
        }
    }
    m.add("OP_CLUSTERS", clusters)?;
    let aliases = PyDict::new(py);
    for (alias, canonical) in ALIASES {
        aliases.set_item(alias, canonical.text())?;
    }
    m.add("OP_ALIASES", aliases)?;
    m.add(
        "DTYPES",
        PyFrozenSet::new(py, emmy_ir::dtype::DType::ALL.iter().map(|d| d.text()))?,
    )?;
    Ok(())
}

#[pymodule(name = "emmy_ir")]
fn emmy_ir_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    expression::register(m)?;
    statement::register_statement_types(m)?;
    m.add_function(wrap_pyfunction!(statement::_statement_from_text, m)?)?;
    body::register(m)?;
    fragment::register(m)?;
    lambda::register(m)?;
    ptx::register(m)?;
    elementwise_tables(m)?;
    Ok(())
}
