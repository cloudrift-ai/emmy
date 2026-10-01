//! In-process host of the Emmy runtime: Python compiles, this module launches.
//!
//! Every method that touches the device releases the interpreter lock, so a bench worker's
//! other threads keep running while a kernel completes.

use emmy_runtime::artifact::{self, Artifact, Env};
use emmy_runtime::cuda::{self, HungKernel};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Arc;

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

fn env(values: Option<HashMap<String, i64>>) -> Env {
    values.unwrap_or_default().into_iter().collect()
}

fn regions(
    values: Option<HashMap<String, (u64, usize)>>,
) -> Option<BTreeMap<String, (u64, usize)>> {
    values.map(|v| v.into_iter().collect())
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

    /// The CUDA stream handle every executor on this device launches on.
    fn stream(&self) -> u64 {
        self.0.stream_handle()
    }

    /// `(is_host_memory, device_address)` for a pointer the driver knows.
    fn pointer_attributes(&self, ptr: u64) -> PyResult<(bool, u64)> {
        self.0.pointer_attributes(ptr).map_err(translate)
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

/// A parsed, validated execution plan: the memory layout is derived from it per environment.
#[pyclass]
struct Program(Arc<artifact::Program>);

#[pymethods]
impl Program {
    /// `plan` is the execution plan's JSON form.
    #[new]
    fn new(plan: &str) -> PyResult<Self> {
        artifact::Program::parse(plan)
            .map(|p| Self(Arc::new(p)))
            .map_err(translate)
    }

    /// The hints: the environment a program runs at when nothing binds its axes.
    fn default_env(&self) -> HashMap<String, i64> {
        self.0.default_env().into_iter().collect()
    }

    fn is_symbolic(&self) -> bool {
        !self.0.bindings.is_empty() || self.0.buffers.iter().any(|b| b.is_symbolic())
    }

    /// Region sizes and every buffer's placement under `env`, on top of the hints.
    #[pyo3(signature = (env = None))]
    fn layout<'py>(
        &self,
        py: Python<'py>,
        env: Option<HashMap<String, i64>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mut full = self.0.default_env();
        full.extend(self::env(env));
        let layout = self.0.layout(&full).map_err(translate)?;
        let out = PyDict::new(py);
        out.set_item(
            "regions",
            layout.regions.into_iter().collect::<HashMap<_, _>>(),
        )?;
        let buffers = PyDict::new(py);
        for (name, placement) in layout.buffers {
            let entry = PyDict::new(py);
            entry.set_item("region", placement.region)?;
            entry.set_item("offset", placement.offset)?;
            entry.set_item("bytes", placement.bytes)?;
            entry.set_item("shape", placement.shape)?;
            buffers.set_item(name, entry)?;
        }
        out.set_item("buffers", buffers)?;
        Ok(out)
    }
}

/// One loaded program: its buffers, kernels and captured graphs on one stream.
#[pyclass]
struct Executor(cuda::Executor);

#[pymethods]
impl Executor {
    /// `binaries` is one cubin path per kernel name, `bindings` the starting bytes of inputs
    /// and constants, `env` the symbolic axes bound on top of the hints, and `regions` the
    /// memory the host lends for every region of the layout (the runtime allocates its own
    /// when omitted).
    #[new]
    #[pyo3(signature = (device, program, binaries, bindings, env = None, regions = None))]
    fn new(
        py: Python<'_>,
        device: &Device,
        program: &Program,
        binaries: HashMap<String, PathBuf>,
        bindings: HashMap<String, Vec<u8>>,
        env: Option<HashMap<String, i64>>,
        regions: Option<HashMap<String, (u64, usize)>>,
    ) -> PyResult<Self> {
        let program = program.0.clone();
        py.detach(|| {
            let artifact = Artifact::from_program(
                program,
                bindings.into_iter().collect::<BTreeMap<_, _>>(),
                binaries.into_iter().collect::<BTreeMap<_, _>>(),
            )?;
            cuda::Executor::load_with(
                &device.0,
                artifact,
                Some(self::env(env)),
                self::regions(regions),
            )
        })
        .map(Self)
        .map_err(translate)
    }

    fn load_times_ms(&self) -> HashMap<&'static str, f64> {
        self.0.load_times_ms.iter().map(|(k, v)| (*k, *v)).collect()
    }

    fn env(&self) -> HashMap<String, i64> {
        self.0.env().iter().map(|(k, v)| (k.clone(), *v)).collect()
    }

    /// The executor's own CUDA stream handle.
    fn stream(&self) -> u64 {
        self.0.stream_handle()
    }

    /// A buffer's device address, allocated bytes and shape under the current environment.
    fn buffer(&self, name: &str) -> PyResult<(u64, usize, Vec<i64>)> {
        let view = self.0.buffer(name).map_err(translate)?;
        Ok((view.ptr, view.bytes, view.shape))
    }

    fn bind(&mut self, py: Python<'_>, name: &str, data: &[u8]) -> PyResult<()> {
        py.detach(|| self.0.bind(name, data)).map_err(translate)
    }

    /// Copy `nbytes` from device address `src` into the buffer's prefix on the current stream.
    fn bind_device(&mut self, py: Python<'_>, name: &str, src: u64, nbytes: usize) -> PyResult<()> {
        py.detach(|| self.0.bind_device(name, src, nbytes))
            .map_err(translate)
    }

    /// Re-lay the program out under `env`; see `Executor.__new__` for `regions` and `bindings`.
    #[pyo3(signature = (env, bindings, regions = None))]
    fn rebind(
        &mut self,
        py: Python<'_>,
        env: HashMap<String, i64>,
        bindings: HashMap<String, Vec<u8>>,
        regions: Option<HashMap<String, (u64, usize)>>,
    ) -> PyResult<()> {
        py.detach(|| {
            self.0.rebind(
                self::env(Some(env)),
                self::regions(regions),
                bindings.into_iter().collect(),
            )
        })
        .map_err(translate)
    }

    /// Bind symbolic axes without touching memory; every buffer stays at its capacity.
    fn set_env(&mut self, py: Python<'_>, env: HashMap<String, i64>) -> PyResult<()> {
        py.detach(|| self.0.set_env(self::env(Some(env))))
            .map_err(translate)
    }

    /// Point one region at memory the host lends.
    fn set_region(&mut self, py: Python<'_>, name: &str, ptr: u64, len: usize) -> PyResult<()> {
        py.detach(|| self.0.set_region(name, ptr, len))
            .map_err(translate)
    }

    /// Drop a region nobody reads; the buffer keeps a valid one-byte address.
    fn set_external(&mut self, py: Python<'_>, name: &str, ptr: u64, len: usize) -> PyResult<()> {
        py.detach(|| self.0.set_external(name, ptr, len))
            .map_err(translate)
    }

    fn release_region(&mut self, py: Python<'_>, name: &str) -> PyResult<()> {
        py.detach(|| self.0.release_region(name)).map_err(translate)
    }

    /// Launch on a stream the host owns; `None` returns to the executor's own stream.
    #[pyo3(signature = (stream))]
    fn set_stream(&mut self, stream: Option<u64>) {
        self.0.set_stream(stream);
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

    fn has_program_graph(&self) -> bool {
        self.0.has_program_graph()
    }

    fn has_launch_graphs(&self) -> bool {
        self.0.has_launch_graphs()
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

    /// Any buffer's allocated bytes after every queued operation has completed.
    fn read<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py.detach(|| self.0.read(name)).map_err(translate)?;
        Ok(PyBytes::new(py, &bytes))
    }

    fn output<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py.detach(|| self.0.output(name)).map_err(translate)?;
        Ok(PyBytes::new(py, &bytes))
    }
}

#[repr(C)]
struct DLDevice {
    device_type: i32,
    device_id: i32,
}

#[repr(C)]
struct DLDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
struct DLTensor {
    data: *mut std::ffi::c_void,
    device: DLDevice,
    ndim: i32,
    dtype: DLDataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}

#[repr(C)]
struct DLManagedTensor {
    dl_tensor: DLTensor,
    manager_ctx: *mut std::ffi::c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

unsafe extern "C" fn dlpack_deleter(tensor: *mut DLManagedTensor) {
    // The shape rides in `manager_ctx`; the memory itself belongs to the host.
    unsafe {
        drop(Box::from_raw((*tensor).manager_ctx as *mut Vec<i64>));
        drop(Box::from_raw(tensor));
    }
}

unsafe extern "C" fn dlpack_capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    // A capsule a consumer never took still owns its tensor; a consumed one is renamed.
    unsafe {
        let name = c"dltensor";
        if pyo3::ffi::PyCapsule_IsValid(capsule, name.as_ptr()) == 1 {
            let tensor =
                pyo3::ffi::PyCapsule_GetPointer(capsule, name.as_ptr()) as *mut DLManagedTensor;
            if !tensor.is_null() {
                dlpack_deleter(tensor);
            }
        }
    }
}

/// A DLPack capsule presenting `ptr` as a contiguous device tensor: `code`/`bits` are the
/// DLPack dtype (float 2, int 0, uint 1, bfloat 4). The memory stays the host's; the capsule
/// only describes it. Lets torch address mapped host memory as a CUDA tensor.
#[pyfunction]
fn device_tensor_capsule(
    py: Python<'_>,
    ptr: u64,
    shape: Vec<i64>,
    code: u8,
    bits: u8,
    device_id: i32,
) -> PyResult<Py<PyAny>> {
    let ndim = i32::try_from(shape.len()).map_err(|_| PyRuntimeError::new_err("rank too large"))?;
    let mut shape = Box::new(shape);
    let tensor = Box::new(DLManagedTensor {
        dl_tensor: DLTensor {
            data: ptr as *mut std::ffi::c_void,
            device: DLDevice {
                device_type: 2,
                device_id,
            },
            ndim,
            dtype: DLDataType {
                code,
                bits,
                lanes: 1,
            },
            shape: shape.as_mut_ptr(),
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        },
        manager_ctx: Box::into_raw(shape) as *mut std::ffi::c_void,
        deleter: Some(dlpack_deleter),
    });
    let raw = Box::into_raw(tensor);
    let capsule = unsafe {
        pyo3::ffi::PyCapsule_New(
            raw as *mut std::ffi::c_void,
            c"dltensor".as_ptr(),
            Some(dlpack_capsule_destructor),
        )
    };
    if capsule.is_null() {
        unsafe { dlpack_deleter(raw) };
        return Err(PyRuntimeError::new_err("could not create a DLPack capsule"));
    }
    Ok(unsafe { Bound::<PyAny>::from_owned_ptr(py, capsule) }.unbind())
}

#[pymodule]
#[pyo3(name = "emmy_runtime")]
fn runtime_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Device>()?;
    m.add_class::<Program>()?;
    m.add_class::<Executor>()?;
    m.add_function(wrap_pyfunction!(device_tensor_capsule, m)?)?;
    m.add("HungKernelError", m.py().get_type::<HungKernelError>())?;
    Ok(())
}
