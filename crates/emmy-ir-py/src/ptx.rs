//! `compile_ptx`: a kernel body and its interface, as the Python side computes it, to the code of its card: PTX
//! text for an NVIDIA one, LLVM IR for an AMD one.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use emmy_ptx::{Interface, Kernel, Page, Param, ParamKind, Precision, Target};

use crate::body::PyBody;
use crate::convert::{IntoPyErr, expression_from_py, name_from_py};

fn param(value: &Bound<'_, PyAny>) -> PyResult<Param> {
    let kind: String = value.getattr("kind")?.extract()?;
    let kind = match kind.as_str() {
        "input" => ParamKind::Input,
        "output" => ParamKind::Output,
        "descriptor" => ParamKind::Descriptor,
        "int" => ParamKind::Int,
        other => return Err(PyValueError::new_err(format!("no parameter kind {other}"))),
    };
    let dtype = value.getattr("dtype")?;
    let dtype = if dtype.is_none() {
        None
    } else {
        let name: String = dtype.getattr("name")?.extract()?;
        Some(name.parse().py()?)
    };
    let shape: PyResult<Vec<_>> = value
        .getattr("shape")?
        .try_iter()?
        .map(|extent| expression_from_py(&extent?))
        .collect();
    Ok(Param {
        name: name_from_py(&value.getattr("body_name")?)?,
        kind,
        dtype,
        page: value
            .getattr("page")?
            .extract::<Option<(usize, i64)>>()?
            .map(|(axis, size)| Page { axis, size }),
        align: value.getattr("align")?.extract()?,
        shape: shape?,
        pair_table: value.getattr("pair_table")?.extract()?,
    })
}

/// The code of the kernel `name`: its `body`, its `interface` (the `Interface` of `ir/kernel/interface.py`), the card
/// and the kernel's two precision decisions; and whether the kernel waits on the grid launched ahead of it, so it may
/// be launched as a programmatic dependent launch. The card is an NVIDIA one by compute capability (`sm=120` for
/// `sm_120`), whose code is PTX text with the instruction set it needs as its `.target`, or an AMD one by processor
/// (`gfx=942` for `gfx942`), whose code is LLVM IR. A statement, an operation or an instruction with no rule for the
/// card raises `NotImplementedError`; every other refusal is a `ValueError` that names what is wrong with the kernel.
#[pyfunction]
#[pyo3(signature = (name, body, interface, *, sm = None, gfx = None, fuse = false, ftz = false, single_wave = false))]
#[allow(clippy::too_many_arguments)]
pub fn compile_ptx(
    name: &str,
    body: &Bound<'_, PyAny>,
    interface: &Bound<'_, PyAny>,
    sm: Option<u32>,
    gfx: Option<u32>,
    fuse: bool,
    ftz: bool,
    single_wave: bool,
) -> PyResult<(String, bool)> {
    let target = match (sm, gfx) {
        (Some(sm), None) => Target::Sm(sm),
        (None, Some(gfx)) => Target::Gfx(gfx),
        _ => return Err(PyValueError::new_err("name one card: sm or gfx")),
    };
    let body = PyBody::from_iterable(body)?;
    let params: PyResult<Vec<Param>> = interface
        .getattr("params")?
        .try_iter()?
        .map(|item| param(&item?))
        .collect();
    let mut shared_offsets = Vec::new();
    for item in interface
        .getattr("shared_offsets")?
        .call_method0("items")?
        .try_iter()?
    {
        let (buffer, offset): (Bound<'_, PyAny>, u64) = item?.extract()?;
        shared_offsets.push((name_from_py(&buffer)?, offset));
    }
    let interface = Interface {
        params: params?,
        shared_offsets,
        shared_pool_bytes: interface.getattr("shared_pool_bytes")?.extract()?,
        single_wave,
    };
    let kernel = Kernel {
        name,
        body: &body,
        interface: &interface,
    };
    let code = emmy_ptx::compile(&kernel, target, Precision { fuse, ftz }).py()?;
    Ok((code.text, code.waits_on_grid))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compile_ptx, m)?)
}
