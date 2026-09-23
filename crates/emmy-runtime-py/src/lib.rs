//! In-process host of the Emmy runtime: Python compiles, this module launches.
//!
//! Every method that touches the device releases the interpreter lock, so a bench worker's
//! other threads keep running while a kernel completes.

use emmy_runtime::artifact::Artifact;
use emmy_runtime::cuda::{self, HungKernel};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

pyo3::create_exception!(
    emmy_runtime,
    HungKernelError,
    PyRuntimeError,
    "A launch did not complete within its deadline; the kernel is still resident."
);

fn translate(error: anyhow::Error) -> PyErr {
    if error.downcast_ref::<HungKernel>().is_some() {
        HungKernelError::new_err(error.to_string())
    } else {
        PyRuntimeError::new_err(format!("{error:#}"))
    }
}

/// One CUDA context. Executors created from it share the context and own their own stream.
#[pyclass]
struct Device(cuda::Device);

#[pymethods]
impl Device {
    #[new]
    #[pyo3(signature = (ordinal = 0))]
    fn new(py: Python<'_>, ordinal: usize) -> PyResult<Self> {
        py.detach(|| cuda::Device::new(ordinal))
            .map(Self)
            .map_err(translate)
    }

    fn compute_capability(&self) -> PyResult<(i32, i32)> {
        self.0.compute_capability().map_err(translate)
    }

    fn name(&self) -> PyResult<String> {
        self.0.name().map_err(translate)
    }

    /// Per-device limits the occupancy estimate and the feature probe read.
    fn properties(&self) -> PyResult<HashMap<&'static str, f64>> {
        Ok(self
            .0
            .properties()
            .map_err(translate)?
            .into_iter()
            .collect())
    }

    /// Wait for the whole context; raises when an earlier fault left it in a sticky error.
    fn synchronize(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.0.synchronize()).map_err(translate)
    }

    /// Static resource usage of one kernel in a cubin: registers, local and shared bytes.
    fn kernel_attributes(
        &self,
        py: Python<'_>,
        cubin: PathBuf,
        name: String,
    ) -> PyResult<HashMap<&'static str, i32>> {
        let attributes = py
            .detach(|| self.0.kernel_attributes(&cubin, &name))
            .map_err(translate)?;
        Ok(HashMap::from([
            ("num_regs", attributes.num_regs),
            ("local_size_bytes", attributes.local_size_bytes),
            ("shared_size_bytes", attributes.shared_size_bytes),
        ]))
    }
}

/// One loaded program: its buffers, kernels and captured graphs on one stream.
#[pyclass]
struct Executor(cuda::Executor);

#[pymethods]
impl Executor {
    /// `plan` is the execution plan's JSON form, `bindings` the input and constant bytes by
    /// buffer name, and `binaries` one cubin path per kernel name.
    #[new]
    fn new(
        py: Python<'_>,
        device: &Device,
        plan: String,
        bindings: HashMap<String, Vec<u8>>,
        binaries: HashMap<String, PathBuf>,
    ) -> PyResult<Self> {
        py.detach(|| {
            let artifact = Artifact::new(
                &plan,
                bindings.into_iter().collect::<BTreeMap<_, _>>(),
                binaries.into_iter().collect::<BTreeMap<_, _>>(),
            )?;
            cuda::Executor::load(&device.0, artifact)
        })
        .map(Self)
        .map_err(translate)
    }

    fn load_times_ms(&self) -> HashMap<&'static str, f64> {
        self.0.load_times_ms.iter().map(|(k, v)| (*k, *v)).collect()
    }

    fn bind(&mut self, py: Python<'_>, name: &str, data: &[u8]) -> PyResult<()> {
        py.detach(|| self.0.bind(name, data)).map_err(translate)
    }

    fn run_once(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.0.run_once()).map_err(translate)
    }

    /// Per-call milliseconds of launch `index` repeated `batch` times in one event window.
    fn time_launch(
        &mut self,
        py: Python<'_>,
        index: usize,
        batch: u32,
        deadline_ms: f64,
    ) -> PyResult<f32> {
        py.detach(|| self.0.time_launch(index, batch, deadline_ms))
            .map_err(translate)
    }

    fn capture_launch_graphs(&mut self, py: Python<'_>, batch_sizes: Vec<u32>) -> PyResult<()> {
        py.detach(|| self.0.capture_launch_graphs(&batch_sizes))
            .map_err(translate)
    }

    fn capture_program_graph(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.0.capture_program_graph())
            .map_err(translate)
    }

    fn replay_program_graph(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.0.replay_program_graph())
            .map_err(translate)
    }

    fn time_program_window(
        &mut self,
        py: Python<'_>,
        replays: u32,
        deadline_ms: f64,
    ) -> PyResult<f32> {
        py.detach(|| self.0.time_program_window(replays, deadline_ms))
            .map_err(translate)
    }

    /// Any buffer's bytes after every queued operation has completed.
    fn read<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py.detach(|| self.0.read(name)).map_err(translate)?;
        Ok(PyBytes::new(py, &bytes))
    }

    fn output<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py.detach(|| self.0.output(name)).map_err(translate)?;
        Ok(PyBytes::new(py, &bytes))
    }
}

#[pymodule]
#[pyo3(name = "emmy_runtime")]
fn runtime_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Device>()?;
    m.add_class::<Executor>()?;
    m.add("HungKernelError", m.py().get_type::<HungKernelError>())?;
    Ok(())
}
