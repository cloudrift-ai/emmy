//! CUDA ownership and submission. Only trusted compiler-produced cubins may be loaded.
//!
//! The executor works on raw device pointers: a region is either memory it allocated itself
//! (the worker binary, an in-process program with no host allocator) or memory the host lent it
//! (a torch tensor the serving runner owns). Launches go to the executor's own stream, or to a
//! stream the host adopted for the duration of a call.

use crate::artifact::{Artifact, Env, Layout, Placement, Program, Tma, dimensions, dtype_bytes};
use anyhow::{Context, Result, bail, ensure};
use cudarc::driver::{CudaContext, CudaEvent, CudaGraph, CudaStream, result, sys};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{CString, c_void};
use std::fmt;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

const MAX_THREADS_PER_BLOCK: u64 = 1024;
const MAX_BLOCK_DIMENSIONS: (u32, u32, u32) = (1024, 1024, 64);
const MAX_GRID_DIMENSIONS: (u32, u32, u32) = (i32::MAX as u32, 65535, 65535);
const DEFAULT_SHARED_MEMORY_BYTES: u32 = 48 * 1024;
const MAX_RUN_ITERATIONS: u32 = 1_000_000;
const MILLISECONDS_PER_SECOND: f64 = 1000.0;
/// How long an event wait spins before it starts sleeping between driver queries.
const SPIN_BEFORE_SLEEP_MS: f64 = 2.0;
const SLEEP_BETWEEN_QUERIES: Duration = Duration::from_micros(200);
/// Whole-program graphs kept per executor, one per symbol environment, least recently used out.
const GRAPH_CACHE_MAX: usize = 64;
const TENSOR_MAP_BYTES: usize = 128;

#[derive(Serialize)]
pub struct RunMetrics {
    pub time_ms: f32,
    pub preparation_ms: f64,
    pub warmup_ms: f64,
    pub submission_ms: f64,
    pub completion_wait_ms: f64,
}

/// A launch whose completion event did not arrive within its deadline. The kernel is still
/// resident: only ending the process frees the device, which is the caller's contract.
#[derive(Debug)]
pub struct HungKernel {
    pub kernel: String,
    pub deadline_ms: f64,
}

impl fmt::Display for HungKernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "kernel {:?} did not complete within {:.0} ms — hung kernel",
            self.kernel, self.deadline_ms
        )
    }
}

impl std::error::Error for HungKernel {}

/// Static resource attributes of one compiled kernel.
#[derive(Serialize)]
pub struct KernelAttributes {
    pub num_regs: i32,
    pub local_size_bytes: i32,
    pub shared_size_bytes: i32,
}

/// One context and the one stream every executor on it launches on. The stream lives as long
/// as the device: a host that tracks lent memory against it (torch's allocator records events
/// on it when a tensor is freed) may do so after the executor is gone.
pub struct Device {
    context: Arc<CudaContext>,
    /// Never destroyed: a host allocator that tracked memory against this stream may still
    /// record events on it while the process is shutting down.
    stream: std::mem::ManuallyDrop<Arc<CudaStream>>,
}

/// ``CudaContext::new`` behind a panic guard.
///
/// cudarc loads libcuda dynamically on first touch and PANICS when the driver is absent, so the
/// no-driver case never reaches its ``Result``. That panic must not cross the FFI boundary: PyO3
/// re-raises it as ``PanicException``, which derives from ``BaseException``, so the
/// ``except Exception`` guards that let a GPU-free host fall back to memorized device specs are
/// bypassed and an offline decode, eval or ``compile --target`` dies instead of degrading. A
/// missing driver is an ordinary answer here, so the hook stays quiet while we ask.
fn load_context(ordinal: usize) -> Result<Arc<CudaContext>> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let loaded = std::panic::catch_unwind(|| CudaContext::new(ordinal));
    std::panic::set_hook(hook);
    match loaded {
        Ok(context) => Ok(context?),
        Err(_) => bail!("no CUDA driver: libcuda could not be loaded"),
    }
}

impl Device {
    pub fn new(ordinal: usize) -> Result<Self> {
        let context = load_context(ordinal)?;
        // Executors synchronize before releasing anything; the safe cross-stream event
        // tracking would only add work.
        unsafe {
            context.disable_event_tracking();
        }
        let stream = std::mem::ManuallyDrop::new(context.new_stream()?);
        Ok(Self { context, stream })
    }

    pub fn compute_capability(&self) -> Result<(i32, i32)> {
        Ok(self.context.compute_capability()?)
    }

    pub fn name(&self) -> Result<String> {
        Ok(self.context.name()?)
    }

    pub fn total_mem(&self) -> Result<usize> {
        Ok(self.context.total_mem()?)
    }

    pub fn attribute(&self, attribute: sys::CUdevice_attribute) -> Result<i32> {
        Ok(self.context.attribute(attribute)?)
    }

    /// Wait for every stream on the context. A sticky error from an earlier fault surfaces
    /// here, which is how a host tells a poisoned context from a healthy one.
    pub fn synchronize(&self) -> Result<()> {
        Ok(self.context.synchronize()?)
    }

    /// The stream executors on this device launch on.
    pub fn stream_handle(&self) -> u64 {
        self.stream.cu_stream() as u64
    }

    /// Whether `ptr` is host memory mapped into the device address space, and the device
    /// address it is reachable at.
    pub fn pointer_attributes(&self, ptr: u64) -> Result<(bool, u64)> {
        self.context.bind_to_thread()?;
        let mut memory_type: u32 = 0;
        let mut device_ptr: u64 = 0;
        unsafe {
            sys::cuPointerGetAttribute(
                &mut memory_type as *mut u32 as *mut c_void,
                sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                ptr,
            )
            .result()?;
            sys::cuPointerGetAttribute(
                &mut device_ptr as *mut u64 as *mut c_void,
                sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
                ptr,
            )
            .result()?;
        }
        Ok((
            memory_type == sys::CUmemorytype::CU_MEMORYTYPE_HOST as u32,
            device_ptr,
        ))
    }

    /// The per-device limits an occupancy estimate and the hardware feature probe read.
    pub fn properties(&self) -> Result<BTreeMap<&'static str, f64>> {
        use sys::CUdevice_attribute as A;
        let attr = |a| self.context.attribute(a).map(f64::from);
        Ok(BTreeMap::from([
            (
                "sm_count",
                attr(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
            ),
            (
                "smem_per_sm",
                attr(A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)?,
            ),
            (
                "smem_per_block",
                attr(A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?,
            ),
            (
                "regs_per_block",
                attr(A::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK)?,
            ),
            (
                "regs_per_sm",
                attr(A::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR)?,
            ),
            ("warp_size", attr(A::CU_DEVICE_ATTRIBUTE_WARP_SIZE)?),
            (
                "max_threads_per_sm",
                attr(A::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)?,
            ),
            (
                "max_blocks_per_sm",
                attr(A::CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR)?,
            ),
            ("total_mem", self.context.total_mem()? as f64),
        ]))
    }

    /// Load one cubin and report the kernel's static resource usage.
    pub fn kernel_attributes(&self, cubin: &Path, name: &str) -> Result<KernelAttributes> {
        self.context.bind_to_thread()?;
        let function = Function::load(cubin, name, 0)?;
        use sys::CUfunction_attribute as A;
        Ok(KernelAttributes {
            num_regs: function.attribute(A::CU_FUNC_ATTRIBUTE_NUM_REGS)?,
            local_size_bytes: function.attribute(A::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)?,
            shared_size_bytes: function.attribute(A::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)?,
        })
    }
}

/// A loaded cubin and the one kernel it is used for.
struct Function {
    module: sys::CUmodule,
    function: sys::CUfunction,
    static_smem: u32,
}

impl Function {
    fn load(cubin: &Path, name: &str, smem: u32) -> Result<Self> {
        let module = result::module::load(CString::new(cubin.to_string_lossy().as_bytes())?)?;
        let function = unsafe { result::module::get_function(module, CString::new(name)?) }
            .with_context(|| format!("kernel {name} not in {}", cubin.display()))?;
        let mut loaded = Self {
            module,
            function,
            static_smem: 0,
        };
        loaded.static_smem = u32::try_from(
            loaded.attribute(sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)?,
        )?;
        // The plan records total shared storage; the cubin already reserves its static part.
        let dynamic = smem.saturating_sub(loaded.static_smem);
        if smem > DEFAULT_SHARED_MEMORY_BYTES && dynamic > 0 {
            unsafe {
                result::function::set_function_attribute(
                    function,
                    sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    i32::try_from(dynamic)?,
                )?;
            }
        }
        Ok(loaded)
    }

    fn attribute(&self, attribute: sys::CUfunction_attribute) -> Result<i32> {
        Ok(unsafe { result::function::get_function_attribute(self.function, attribute) }?)
    }
}

impl Drop for Function {
    fn drop(&mut self) {
        let _ = unsafe { result::module::unload(self.module) };
    }
}

/// A named device region: memory the runtime allocated, or memory the host lent it.
struct Region {
    ptr: u64,
    len: usize,
    owned: bool,
}

impl Region {
    fn allocate(len: usize, stream: sys::CUstream) -> Result<Self> {
        let len = len.max(1);
        let ptr = unsafe { result::malloc_sync(len) }?;
        unsafe {
            result::memset_d8_async(ptr, 0, len, stream)?;
        }
        Ok(Self {
            ptr,
            len,
            owned: true,
        })
    }

    fn lent(ptr: u64, len: usize) -> Self {
        Self {
            ptr,
            len,
            owned: false,
        }
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        if self.owned && self.ptr != 0 {
            let _ = unsafe { result::free_sync(self.ptr) };
        }
    }
}

/// An encoded TMA descriptor living in device memory (the kernel takes a pointer to it).
struct Descriptor {
    ptr: u64,
}

impl Drop for Descriptor {
    fn drop(&mut self) {
        let _ = unsafe { result::free_sync(self.ptr) };
    }
}

type EnvKey = Vec<(String, i64)>;

fn env_key(env: &Env) -> EnvKey {
    env.iter().map(|(k, v)| (k.clone(), *v)).collect()
}

enum Param {
    Ptr(u64),
    Int(i32),
}

/// Wait for an event with a deadline instead of blocking on the driver.
fn wait_for_event(event: &CudaEvent, deadline_ms: f64, kernel: &str) -> Result<()> {
    let started = Instant::now();
    loop {
        match unsafe { result::event::query(event.cu_event()) } {
            Ok(()) => return Ok(()),
            Err(error) if error.0 == sys::cudaError_enum::CUDA_ERROR_NOT_READY => {}
            Err(error) => return Err(error.into()),
        }
        let elapsed_ms = started.elapsed().as_secs_f64() * MILLISECONDS_PER_SECOND;
        if elapsed_ms > deadline_ms {
            bail!(HungKernel {
                kernel: kernel.to_owned(),
                deadline_ms,
            });
        }
        if elapsed_ms < SPIN_BEFORE_SLEEP_MS {
            std::hint::spin_loop();
        } else {
            std::thread::sleep(SLEEP_BETWEEN_QUERIES);
        }
    }
}

/// IEEE round-to-nearest-even conversion of an f32 to f16 bits.
fn f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        return sign | 0x7c00 | if mantissa != 0 { 0x200 } else { 0 };
    }
    let e = exponent - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = (mantissa | 0x80_0000) >> (1 - e);
        let rounded = m + 0xfff + ((m >> 13) & 1);
        return sign | (rounded >> 13) as u16;
    }
    let m = ((e as u32) << 23) | mantissa;
    let rounded = m + 0xfff + ((m >> 13) & 1);
    sign | (rounded >> 13) as u16
}

/// The little-endian bytes of one scalar in a plan dtype — how a runtime constant fills its
/// buffer. bf16 rounds to nearest even on the dropped mantissa half.
pub fn scalar_bytes(dtype: &str, value: f64) -> Result<Vec<u8>> {
    Ok(match dtype {
        "f32" => (value as f32).to_le_bytes().to_vec(),
        "f64" => value.to_le_bytes().to_vec(),
        "f16" => f16_bits(value as f32).to_le_bytes().to_vec(),
        "bf16" => {
            let bits = (value as f32).to_bits();
            ((bits.wrapping_add(0x7fff).wrapping_add((bits >> 16) & 1) >> 16) as u16)
                .to_le_bytes()
                .to_vec()
        }
        "i8" => (value as i64 as i8).to_le_bytes().to_vec(),
        "i16" => (value as i64 as i16).to_le_bytes().to_vec(),
        "i32" => (value as i64 as i32).to_le_bytes().to_vec(),
        "i64" => (value as i64).to_le_bytes().to_vec(),
        "u8" => (value as i64 as u8).to_le_bytes().to_vec(),
        "u16" => (value as i64 as u16).to_le_bytes().to_vec(),
        "u32" => (value as i64 as u32).to_le_bytes().to_vec(),
        "u64" => (value as i64 as u64).to_le_bytes().to_vec(),
        "bool" => vec![u8::from(value != 0.0)],
        other => bail!("cannot fill a {other} buffer with a scalar"),
    })
}

/// Match a runtime shape to a TMA box: drop exactly the surplus extent-1 gap dimensions the
/// materializer dropped, innermost first, and split an inner dimension when the box carries one
/// more rank than the source (the swizzle split).
pub fn collapse_inert_dims(shape: &[i64], box_extents: &[u32]) -> Result<Vec<i64>> {
    let mut arr_rev: Vec<i64> = shape.iter().rev().copied().collect();
    let box_rev: Vec<i64> = box_extents.iter().rev().map(|&b| i64::from(b)).collect();
    if box_rev.len() == arr_rev.len() + 1
        && !arr_rev.is_empty()
        && box_rev[0] != 0
        && arr_rev[0] % box_rev[0] == 0
    {
        arr_rev = [
            vec![box_rev[0], arr_rev[0] / box_rev[0]],
            arr_rev[1..].to_vec(),
        ]
        .concat();
    }
    let mut n_drop = arr_rev.len() as i64 - box_rev.len() as i64;
    let mut kept = Vec::new();
    let mut bi = 0;
    for a in arr_rev {
        if n_drop > 0 && a == 1 && bi < box_rev.len() && box_rev[bi] != 1 {
            n_drop -= 1;
            continue;
        }
        kept.push(a);
        bi += 1;
    }
    ensure!(
        n_drop == 0 && kept.len() == box_rev.len(),
        "TMA descriptor rank mismatch: shape {shape:?} cannot be collapsed to match box {box_extents:?}"
    );
    kept.reverse();
    Ok(kept)
}

/// Encode a 128-byte `CUtensorMap` for a C-contiguous source of `shape` at `address`.
fn encode_tiled(
    address: u64,
    shape: &[i64],
    box_extents: &[u32],
    elem_size: usize,
    swizzle: &str,
) -> Result<[u64; 16]> {
    let rank = shape.len();
    ensure!(
        rank == box_extents.len(),
        "rank mismatch: shape {shape:?} vs box {box_extents:?}"
    );
    ensure!((1..=5).contains(&rank), "TMA rank must be 1..5, got {rank}");
    // The driver's dim 0 is the fastest-varying one: reverse the C-order shapes.
    let global_dim: Vec<u64> = shape.iter().rev().map(|&d| d as u64).collect();
    let box_dim: Vec<u32> = box_extents.iter().rev().copied().collect();
    let element_strides = vec![1u32; rank];
    let mut global_strides = Vec::with_capacity(rank.saturating_sub(1));
    let mut running = global_dim[0] * elem_size as u64;
    for &d in &global_dim[1..] {
        global_strides.push(running);
        running *= d;
    }
    let data_type = match elem_size {
        1 => sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2 => sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        8 => sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT64,
        _ => sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
    };
    let swizzle = match swizzle {
        "NONE" => sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        "B32" => sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B,
        "B64" => sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B,
        "B128" => sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        other => bail!("unknown TMA swizzle {other}"),
    };
    let mut map = sys::CUtensorMap { opaque: [0; 16] };
    unsafe {
        sys::cuTensorMapEncodeTiled(
            &mut map,
            data_type,
            rank as u32,
            address as *mut c_void,
            global_dim.as_ptr(),
            if global_strides.is_empty() {
                std::ptr::null()
            } else {
                global_strides.as_ptr()
            },
            box_dim.as_ptr(),
            element_strides.as_ptr(),
            sys::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            swizzle,
            sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
            sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
        .result()
        .context("cuTensorMapEncodeTiled failed")?;
    }
    Ok(map.opaque)
}

/// A buffer's current place: its device address, its allocated bytes and its resolved shape.
#[derive(Debug, Clone, PartialEq)]
pub struct BufferView {
    pub ptr: u64,
    pub bytes: usize,
    pub shape: Vec<i64>,
}

pub struct Executor {
    pub load_times_ms: BTreeMap<&'static str, f64>,
    context: Arc<CudaContext>,
    program: Arc<Program>,
    functions: BTreeMap<String, Function>,
    own_stream: Arc<CudaStream>,
    external: Option<sys::CUstream>,
    capturing: bool,
    env: Env,
    layout: Layout,
    regions: BTreeMap<String, Region>,
    /// Operands the plan names but never declares as buffers — an indirect operand's pointer
    /// table and selector — bound by the host by address.
    externals: BTreeMap<String, Region>,
    bound: BTreeSet<String>,
    descriptors: BTreeMap<EnvKey, BTreeMap<(usize, String), Descriptor>>,
    /// Whole-program graphs by symbol environment, most recently used last.
    graphs: Vec<(EnvKey, CudaGraph)>,
    /// One graph per launch position, each holding that launch's batch, at the current env.
    launch_graphs: Option<(Vec<u32>, Vec<CudaGraph>)>,
    /// Per-launch timing events, created once and reused across iterations.
    events: Vec<(CudaEvent, CudaEvent)>,
    window: Option<(CudaEvent, CudaEvent)>,
}

// A captured graph is a raw driver handle the driver does not synchronize, so cudarc leaves it
// `!Send`. The executor only reaches its graphs through `&mut self`, and every entry point
// rebinds the context to the calling thread, so moving the whole executor between threads —
// which is what a host that releases its interpreter lock does — is sound.
unsafe impl Send for Executor {}
unsafe impl Sync for Executor {}

impl Executor {
    /// Load a program at its default environment with runtime-owned memory: the worker binary
    /// and cached generation.
    pub fn load(device: &Device, artifact: Artifact) -> Result<Self> {
        Self::load_with(device, artifact, None, None)
    }

    /// Load a validated, trusted artifact. `env` binds its symbolic axes on top of the hints;
    /// `regions` lends host memory for every region of the layout, or the runtime allocates
    /// and zeroes its own.
    pub fn load_with(
        device: &Device,
        artifact: Artifact,
        env: Option<Env>,
        regions: Option<BTreeMap<String, (u64, usize)>>,
    ) -> Result<Self> {
        let context = device.context.clone();
        context.bind_to_thread()?;
        if let Some(arch) = &artifact.arch {
            let (major, minor) = context.compute_capability()?;
            ensure!(
                *arch == format!("sm_{major}{minor}"),
                "artifact GPU architecture mismatch"
            );
        }
        let program = artifact.program;
        let mut full_env = program.default_env();
        full_env.extend(env.unwrap_or_default());
        let layout = program.layout(&full_env)?;
        let own_stream = Arc::clone(&device.stream);
        let started = Instant::now();
        let mut functions = BTreeMap::new();
        for (name, path) in &artifact.binaries {
            let smem = program
                .launches
                .iter()
                .filter(|l| &l.kernel == name)
                .map(|l| l.smem)
                .max()
                .unwrap_or(0);
            functions.insert(name.clone(), Function::load(path, name, smem)?);
        }
        let module_ms = started.elapsed().as_secs_f64() * MILLISECONDS_PER_SECOND;
        let started = Instant::now();
        let mut executor = Self {
            load_times_ms: BTreeMap::new(),
            context,
            program,
            functions,
            own_stream,
            external: None,
            capturing: false,
            env: full_env,
            layout: Layout::default(),
            regions: BTreeMap::new(),
            externals: BTreeMap::new(),
            bound: BTreeSet::new(),
            descriptors: BTreeMap::new(),
            graphs: Vec::new(),
            launch_graphs: None,
            events: Vec::new(),
            window: None,
        };
        executor.provision(layout, regions.as_ref())?;
        executor.own_stream.synchronize()?;
        let allocation_ms = started.elapsed().as_secs_f64() * MILLISECONDS_PER_SECOND;
        let started = Instant::now();
        for (name, data) in &artifact.bindings {
            executor.upload(name, data)?;
        }
        executor.apply_runtime_constants()?;
        executor.load_times_ms = BTreeMap::from([
            ("module_ms", module_ms),
            ("allocation_zero_ms", allocation_ms),
            (
                "upload_ms",
                started.elapsed().as_secs_f64() * MILLISECONDS_PER_SECOND,
            ),
        ]);
        Ok(executor)
    }

    fn stream(&self) -> sys::CUstream {
        match self.external {
            Some(stream) if !self.capturing => stream,
            _ => self.own_stream.cu_stream(),
        }
    }

    fn synchronize(&self) -> Result<()> {
        unsafe {
            result::stream::synchronize(self.stream())?;
        }
        if self.external.is_some() {
            self.own_stream.synchronize()?;
        }
        Ok(())
    }

    /// Adopt the regions `layout` needs: memory the host lends, or the runtime's own. An owned
    /// region large enough for its new size is kept; anything else is freed and reallocated
    /// zeroed. Graphs and descriptors bake addresses, so both are dropped.
    fn provision(
        &mut self,
        layout: Layout,
        lent: Option<&BTreeMap<String, (u64, usize)>>,
    ) -> Result<()> {
        self.synchronize()?;
        self.graphs.clear();
        self.launch_graphs = None;
        self.descriptors.clear();
        let mut old = std::mem::take(&mut self.regions);
        for (name, &size) in &layout.regions {
            let region = match lent {
                Some(lent) => {
                    let &(ptr, len) = lent
                        .get(name)
                        .with_context(|| format!("host provided no memory for region {name}"))?;
                    ensure!(
                        len >= size,
                        "region {name} needs {size} bytes, host lent {len}"
                    );
                    Region::lent(ptr, len)
                }
                None => match old.remove(name) {
                    Some(region) if region.owned && region.len >= size.max(1) => region,
                    _ => Region::allocate(size, self.stream())?,
                },
            };
            self.regions.insert(name.clone(), region);
        }
        drop(old);
        self.layout = layout;
        Ok(())
    }

    fn placement(&self, name: &str) -> Result<&Placement> {
        self.layout
            .buffers
            .get(name)
            .with_context(|| format!("unknown buffer {name}"))
    }

    fn address(&self, name: &str) -> Result<u64> {
        if let Some(external) = self.externals.get(name) {
            return Ok(external.ptr);
        }
        let placement = self.placement(name)?;
        let region = self
            .regions
            .get(&placement.region)
            .with_context(|| format!("region {} is not provisioned", placement.region))?;
        Ok(region.ptr + placement.offset as u64)
    }

    /// Bind an operand the plan never declares as a buffer (an indirect operand's table or
    /// selector) to memory the host lends.
    pub fn set_external(&mut self, name: &str, ptr: u64, len: usize) -> Result<()> {
        self.context.bind_to_thread()?;
        ensure!(
            self.layout.buffers.contains_key(name)
                || self
                    .program
                    .launches
                    .iter()
                    .any(|l| l.indirect.iter().any(|i| i.table == name || i.sel == name)),
            "unknown operand {name}"
        );
        self.synchronize()?;
        self.graphs.clear();
        self.launch_graphs = None;
        self.externals
            .insert(name.to_owned(), Region::lent(ptr, len));
        Ok(())
    }

    /// Every buffer's address, allocated bytes and resolved shape under the current env.
    pub fn buffer(&self, name: &str) -> Result<BufferView> {
        let placement = self.placement(name)?;
        Ok(BufferView {
            ptr: self.address(name)?,
            bytes: placement.bytes,
            shape: self.program.buffer(name)?.resolve_shape(&self.env)?,
        })
    }

    pub fn env(&self) -> &Env {
        &self.env
    }

    /// The stream the executor launches on when no host stream is adopted: its device's.
    pub fn stream_handle(&self) -> u64 {
        self.own_stream.cu_stream() as u64
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    fn upload(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        let placement = self.placement(name)?;
        ensure!(
            bytes.len() <= placement.bytes,
            "{} bytes exceed buffer {name} ({} bytes)",
            bytes.len(),
            placement.bytes
        );
        // Host uploads always go through the executor's own stream and complete before this
        // returns: the caller may release the host slice, and a host stream the runtime
        // adopted may be recording a graph, which a pageable copy or a synchronize would
        // break. The upload lands before any later launch on any stream.
        let dst = self.address(name)?;
        let stream = self.own_stream.cu_stream();
        if !bytes.is_empty() {
            unsafe {
                result::memcpy_htod_async(dst, bytes, stream)?;
            }
        }
        unsafe {
            result::stream::synchronize(stream)?;
        }
        self.bound.insert(name.into());
        Ok(())
    }

    /// Upload host bytes into a program input's prefix.
    pub fn bind(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        self.context.bind_to_thread()?;
        ensure!(
            self.program.inputs.iter().any(|n| n == name),
            "only program inputs may be updated"
        );
        self.upload(name, bytes)
    }

    /// Copy device memory into a buffer's prefix on the current stream. A source that already
    /// is the buffer skips the copy: a producer's output chained onto the consumer's input.
    pub fn bind_device(&mut self, name: &str, src: u64, nbytes: usize) -> Result<()> {
        self.context.bind_to_thread()?;
        let placement = self.placement(name)?;
        ensure!(
            nbytes <= placement.bytes,
            "{nbytes} bytes exceed buffer {name} ({} bytes)",
            placement.bytes
        );
        let dst = self.address(name)?;
        if src != dst && nbytes > 0 {
            unsafe {
                result::memcpy_dtod_async(dst, src, nbytes, self.stream())?;
            }
        }
        self.bound.insert(name.into());
        Ok(())
    }

    /// Re-lay the program out under `env`: regions grow or are replaced by what the host lends,
    /// captured graphs and descriptors are dropped, `bindings` are uploaded and the runtime
    /// constants refilled. Buffers whose regions survive keep their contents.
    pub fn rebind(
        &mut self,
        env: Env,
        regions: Option<BTreeMap<String, (u64, usize)>>,
        bindings: BTreeMap<String, Vec<u8>>,
    ) -> Result<()> {
        self.context.bind_to_thread()?;
        let mut full_env = self.program.default_env();
        full_env.extend(env);
        let layout = self.program.layout(&full_env)?;
        self.env = full_env;
        self.provision(layout, regions.as_ref())?;
        for (name, data) in &bindings {
            let role = &self.program.buffer(name)?.role;
            ensure!(
                role == "input" || role == "constant",
                "only inputs and constants may be bound: {name}"
            );
            self.upload(name, data)?;
        }
        self.apply_runtime_constants()
    }

    /// Bind symbolic axes without touching memory: launch geometry, runtime arguments and
    /// descriptors follow the new extents while every buffer stays at its allocated capacity.
    pub fn set_env(&mut self, env: Env) -> Result<()> {
        let mut merged = self.env.clone();
        merged.extend(env);
        for buffer in &self.program.buffers {
            let placement = self.placement(&buffer.name)?;
            let need = buffer.byte_len(&merged)?;
            ensure!(
                need <= placement.bytes,
                "buffer {} resolves to {need} bytes past its capacity of {}",
                buffer.name,
                placement.bytes
            );
        }
        self.context.bind_to_thread()?;
        self.env = merged;
        self.launch_graphs = None;
        self.apply_runtime_constants()
    }

    /// Drop a region nobody reads (an operand the kernel resolves through a table instead):
    /// its memory goes back to the host; the buffer keeps a valid one-byte address.
    pub fn release_region(&mut self, name: &str) -> Result<()> {
        self.context.bind_to_thread()?;
        ensure!(
            self.layout.regions.contains_key(name),
            "unknown region {name}"
        );
        self.synchronize()?;
        self.graphs.clear();
        self.launch_graphs = None;
        self.descriptors.clear();
        let dummy = Region::allocate(1, self.stream())?;
        self.regions.insert(name.to_owned(), dummy);
        Ok(())
    }

    /// Point one region at memory the host lends (a buffer chained onto another program's).
    pub fn set_region(&mut self, name: &str, ptr: u64, len: usize) -> Result<()> {
        self.context.bind_to_thread()?;
        let size = *self
            .layout
            .regions
            .get(name)
            .with_context(|| format!("unknown region {name}"))?;
        ensure!(
            len >= size,
            "region {name} needs {size} bytes, host lent {len}"
        );
        self.synchronize()?;
        self.graphs.clear();
        self.launch_graphs = None;
        self.descriptors.clear();
        self.regions.insert(name.to_owned(), Region::lent(ptr, len));
        Ok(())
    }

    /// Launch on a stream the host owns (`None` returns to the executor's own stream).
    pub fn set_stream(&mut self, stream: Option<u64>) {
        self.external = stream.map(|s| s as sys::CUstream);
    }

    fn apply_runtime_constants(&mut self) -> Result<()> {
        let program = self.program.clone();
        for (name, expr) in &program.runtime_constants {
            let buffer = program.buffer(name)?;
            let value = expr.eval_f64(&self.env)?;
            let pattern = scalar_bytes(&buffer.dtype, value)?;
            let count = buffer.byte_len(&self.env)? / pattern.len();
            self.upload(name, &pattern.repeat(count))?;
        }
        Ok(())
    }

    fn ensure_descriptors(&mut self) -> Result<EnvKey> {
        let key = env_key(&self.env);
        if self.descriptors.contains_key(&key) {
            return Ok(key);
        }
        let program = self.program.clone();
        let mut encoded = BTreeMap::new();
        for (index, launch) in program.launches.iter().enumerate() {
            for tma in &launch.tma {
                encoded.insert((index, tma.name.clone()), self.encode(tma)?);
            }
        }
        if !encoded.is_empty() {
            self.own_stream.synchronize()?;
        }
        self.descriptors.insert(key.clone(), encoded);
        Ok(key)
    }

    fn encode(&self, tma: &Tma) -> Result<Descriptor> {
        let buffer = self.program.buffer(&tma.src_buf)?;
        // A symbolic source is prefix-packed at the RESOLVED shape's row-major strides, so the
        // descriptor follows the environment, never the allocation.
        let shape = collapse_inert_dims(&buffer.resolve_shape(&self.env)?, &tma.box_extents)?;
        let map = encode_tiled(
            self.address(&tma.src_buf)?,
            &shape,
            &tma.box_extents,
            dtype_bytes(&buffer.dtype)?,
            &tma.swizzle,
        )?;
        let ptr = unsafe { result::malloc_sync(TENSOR_MAP_BYTES) }?;
        let descriptor = Descriptor { ptr };
        unsafe {
            result::memcpy_htod_async(ptr, &map, self.own_stream.cu_stream())?;
        }
        Ok(descriptor)
    }

    fn launch(&mut self, index: usize, env: &Env, key: &EnvKey, serial_from: usize) -> Result<()> {
        let program = self.program.clone();
        let launch = &program.launches[index];
        if serial_from < launch.serial.len() {
            // A recurrence's time: one launch per coordinate, in order, each seeing the
            // previous step's stores.
            let (name, extent) = &launch.serial[serial_from];
            for step in 0..*extent {
                let mut stepped = env.clone();
                stepped.insert(name.clone(), step);
                self.launch(index, &stepped, key, serial_from + 1)?;
            }
            return Ok(());
        }
        let stream = self.stream();
        for name in &launch.zero_outputs {
            // memset, not a fill kernel: it records as a cheap node under graph capture, and
            // all-zero bytes are 0.0 in every buffer dtype.
            let placement = self.placement(name)?;
            unsafe {
                result::memset_d8_async(self.address(name)?, 0, placement.bytes.max(1), stream)?;
            }
        }
        let descriptors = &self.descriptors[key];
        let mut params: Vec<Param> =
            Vec::with_capacity(launch.args.len() + launch.runtime_args.len() + 2);
        for name in &launch.args {
            if let Some(descriptor) = descriptors.get(&(index, name.clone())) {
                params.push(Param::Ptr(descriptor.ptr));
            } else if let Some(indirect) = launch.indirect.iter().find(|i| &i.arg == name) {
                // The kernel resolves ``table[sel[slot]]`` in its body preamble.
                params.push(Param::Ptr(self.address(&indirect.table)?));
                params.push(Param::Ptr(self.address(&indirect.sel)?));
                params.push(Param::Int(i32::try_from(indirect.slot)?));
            } else {
                params.push(Param::Ptr(self.address(name)?));
            }
        }
        for name in &launch.runtime_args {
            let value = *env
                .get(name)
                .with_context(|| format!("unbound runtime argument {name}"))?;
            params.push(Param::Int(i32::try_from(value)?));
        }
        let mut raw: Vec<*mut c_void> = params
            .iter_mut()
            .map(|p| match p {
                Param::Ptr(v) => v as *mut u64 as *mut c_void,
                Param::Int(v) => v as *mut i32 as *mut c_void,
            })
            .collect();
        let grid = dimensions(&launch.grid, env)?;
        let block = dimensions(&launch.block, env)?;
        ensure!(
            u64::from(block.0) * u64::from(block.1) * u64::from(block.2) <= MAX_THREADS_PER_BLOCK
                && block.0 <= MAX_BLOCK_DIMENSIONS.0
                && block.1 <= MAX_BLOCK_DIMENSIONS.1
                && block.2 <= MAX_BLOCK_DIMENSIONS.2,
            "invalid CUDA block for {}",
            launch.kernel
        );
        ensure!(
            grid.0 <= MAX_GRID_DIMENSIONS.0
                && grid.1 <= MAX_GRID_DIMENSIONS.1
                && grid.2 <= MAX_GRID_DIMENSIONS.2,
            "invalid CUDA grid for {}",
            launch.kernel
        );
        let function = &self.functions[&launch.kernel];
        // The compiler defines the ABI and access bounds. Every pointer resolved above stays
        // alive and exclusively owned through completion.
        unsafe {
            result::launch_kernel(
                function.function,
                grid,
                block,
                launch.smem.saturating_sub(function.static_smem),
                stream,
                &mut raw,
            )?;
        }
        Ok(())
    }

    fn submit(&mut self) -> Result<()> {
        ensure!(
            self.program.inputs.iter().all(|n| self.bound.contains(n)),
            "all program inputs must be bound"
        );
        let key = self.ensure_descriptors()?;
        let env = self.env.clone();
        for index in 0..self.program.launches.len() {
            self.launch(index, &env, &key, 0)?;
        }
        Ok(())
    }

    /// Launch every kernel once in program order with no events; the caller's read synchronizes.
    pub fn run_once(&mut self) -> Result<()> {
        self.context.bind_to_thread()?;
        self.submit()
    }

    fn event(&self) -> Result<CudaEvent> {
        Ok(self
            .context
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?)
    }

    /// Time launch `index` repeated `batch` times inside one event window, returning per-call
    /// milliseconds. The wait polls with a deadline so a hung kernel raises instead of blocking.
    pub fn time_launch(&mut self, index: usize, batch: u32, deadline_ms: f64) -> Result<f32> {
        self.context.bind_to_thread()?;
        ensure!(
            index < self.program.launches.len(),
            "launch index out of range"
        );
        ensure!(batch > 0, "batch must be positive");
        ensure!(
            self.program.inputs.iter().all(|n| self.bound.contains(n)),
            "all program inputs must be bound"
        );
        let key = self.ensure_descriptors()?;
        while self.events.len() <= index {
            let pair = (self.event()?, self.event()?);
            self.events.push(pair);
        }
        let stream = self.stream();
        unsafe {
            result::event::record(self.events[index].0.cu_event(), stream)?;
        }
        match &self.launch_graphs {
            Some((batches, graphs)) => {
                ensure!(
                    batches[index] == batch,
                    "captured batch does not match the requested batch"
                );
                unsafe {
                    result::graph::launch(graphs[index].cu_graph_exec(), stream)?;
                }
            }
            None => {
                let env = self.env.clone();
                for _ in 0..batch {
                    self.launch(index, &env, &key, 0)?;
                }
            }
        }
        unsafe {
            result::event::record(self.events[index].1.cu_event(), stream)?;
        }
        let kernel = self.program.launches[index].kernel.clone();
        wait_for_event(&self.events[index].1, deadline_ms, &kernel)?;
        let elapsed_ms = self.events[index].0.elapsed_ms(&self.events[index].1)?;
        // Event timing has sub-microsecond resolution and a real launch consumes at least one
        // device cycle, so a zero reading is a no-op launch that must never win a benchmark.
        ensure!(
            elapsed_ms > 0.0,
            "kernel {kernel:?} reported {elapsed_ms:.3}ms elapsed — degenerate / no-op launch, variant marked bench_fail"
        );
        Ok(elapsed_ms / batch as f32)
    }

    fn capture<F: FnMut(&mut Self) -> Result<()>>(&mut self, mut work: F) -> Result<CudaGraph> {
        self.synchronize()?;
        self.capturing = true;
        let stream = self.own_stream.clone();
        let begun =
            stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
        let submitted = begun.map_err(Into::into).and_then(|()| work(self));
        let captured = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        self.capturing = false;
        submitted?;
        captured?.context("empty CUDA graph")
    }

    /// Capture each launch position's batch into its own graph; unchanged batches are kept.
    pub fn capture_launch_graphs(&mut self, batch_sizes: &[u32]) -> Result<()> {
        self.context.bind_to_thread()?;
        ensure!(
            batch_sizes.len() == self.program.launches.len(),
            "one batch size per launch"
        );
        if let Some((batches, _)) = &self.launch_graphs
            && batches == batch_sizes
        {
            return Ok(());
        }
        self.launch_graphs = None;
        let key = self.ensure_descriptors()?;
        let env = self.env.clone();
        let mut graphs = Vec::with_capacity(batch_sizes.len());
        for (index, &batch) in batch_sizes.iter().enumerate() {
            ensure!(batch > 0, "batch must be positive");
            graphs.push(self.capture(|executor| {
                for _ in 0..batch {
                    executor.launch(index, &env, &key, 0)?;
                }
                Ok(())
            })?);
        }
        self.launch_graphs = Some((batch_sizes.to_vec(), graphs));
        Ok(())
    }

    /// Capture every launch in program order into one graph for the current environment. A
    /// graph captured at an env replays only at that env (grids, runtime arguments and
    /// descriptors are baked), so one graph is kept per env, least recently used out.
    pub fn capture_program_graph(&mut self) -> Result<()> {
        self.context.bind_to_thread()?;
        let key = env_key(&self.env);
        if let Some(position) = self.graphs.iter().position(|(k, _)| *k == key) {
            let entry = self.graphs.remove(position);
            self.graphs.push(entry);
            return Ok(());
        }
        let graph = self.capture(|executor| executor.submit())?;
        self.graphs.push((key, graph));
        if self.graphs.len() > GRAPH_CACHE_MAX {
            self.graphs.remove(0);
        }
        Ok(())
    }

    /// Whether per-launch graphs are captured at the current environment.
    pub fn has_launch_graphs(&self) -> bool {
        self.launch_graphs.is_some()
    }

    /// Whether a graph exists for the current environment.
    pub fn has_program_graph(&self) -> bool {
        let key = env_key(&self.env);
        self.graphs.iter().any(|(k, _)| *k == key)
    }

    pub fn replay_program_graph(&mut self) -> Result<()> {
        self.context.bind_to_thread()?;
        let key = env_key(&self.env);
        let graph = self
            .graphs
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, g)| g)
            .context("no captured program graph for this environment")?;
        unsafe {
            result::graph::launch(graph.cu_graph_exec(), self.stream())?;
        }
        Ok(())
    }

    /// Time `replays` back-to-back replays of the program graph, per replay in milliseconds.
    pub fn time_program_window(&mut self, replays: u32, deadline_ms: f64) -> Result<f32> {
        ensure!(replays > 0, "replays must be positive");
        self.capture_program_graph()?;
        if self.window.is_none() {
            self.window = Some((self.event()?, self.event()?));
        }
        let stream = self.stream();
        let (start, _) = self.window.as_ref().unwrap();
        unsafe {
            result::event::record(start.cu_event(), stream)?;
        }
        for _ in 0..replays {
            self.replay_program_graph()?;
        }
        let (start, stop) = self.window.as_ref().unwrap();
        unsafe {
            result::event::record(stop.cu_event(), stream)?;
        }
        wait_for_event(stop, deadline_ms, "program graph")?;
        Ok(start.elapsed_ms(stop)? / replays as f32)
    }

    /// Time ordered program submissions with CUDA events; excludes I/O and control transport.
    /// Uncaptured execution includes exposed host submission gaps.
    pub fn execute(&mut self, warmup: u32, iterations: u32, capture: bool) -> Result<RunMetrics> {
        self.context.bind_to_thread()?;
        ensure!(
            iterations > 0 && iterations <= MAX_RUN_ITERATIONS && warmup <= MAX_RUN_ITERATIONS,
            "invalid iteration count"
        );
        let started = Instant::now();
        if capture && !self.has_program_graph() {
            self.submit()?;
            self.capture_program_graph()?;
        }
        let preparation_ms = started.elapsed().as_secs_f64() * MILLISECONDS_PER_SECOND;
        let started = Instant::now();
        for _ in 0..warmup {
            self.step(capture)?;
        }
        self.synchronize()?;
        let warmup_ms = started.elapsed().as_secs_f64() * MILLISECONDS_PER_SECOND;
        let start = self.event()?;
        let end = self.event()?;
        let stream = self.stream();
        unsafe {
            result::event::record(start.cu_event(), stream)?;
        }
        let started = Instant::now();
        for _ in 0..iterations {
            self.step(capture)?;
        }
        unsafe {
            result::event::record(end.cu_event(), stream)?;
        }
        let submission_ms = started.elapsed().as_secs_f64() * MILLISECONDS_PER_SECOND;
        let started = Instant::now();
        let time = start.elapsed_ms(&end)? / iterations as f32;
        let completion_wait_ms = started.elapsed().as_secs_f64() * MILLISECONDS_PER_SECOND;
        Ok(RunMetrics {
            time_ms: time,
            preparation_ms,
            warmup_ms,
            submission_ms,
            completion_wait_ms,
        })
    }

    /// Submit a stateful program exactly once. Capture records work without a warmup execution.
    pub fn advance(&mut self, capture: bool) -> Result<()> {
        self.context.bind_to_thread()?;
        if capture {
            self.capture_program_graph()?;
        }
        self.step(capture)?;
        self.synchronize()
    }

    fn step(&mut self, capture: bool) -> Result<()> {
        if capture {
            self.replay_program_graph()
        } else {
            self.submit()
        }
    }

    /// Copy any buffer's allocated bytes back to the host after every queued operation has
    /// completed.
    pub fn read(&self, name: &str) -> Result<Vec<u8>> {
        self.context.bind_to_thread()?;
        let placement = self.placement(name)?;
        let stream = self.stream();
        let mut bytes = vec![0u8; placement.bytes];
        if !bytes.is_empty() {
            unsafe {
                result::memcpy_dtoh_async(&mut bytes, self.address(name)?, stream)?;
            }
        }
        unsafe {
            result::stream::synchronize(stream)?;
        }
        Ok(bytes)
    }

    pub fn output(&self, name: &str) -> Result<Vec<u8>> {
        ensure!(
            self.program.outputs.iter().any(|n| n == name),
            "unknown program output"
        );
        self.read(name)
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        // Even a partially submitted operation must finish before pointers or modules are freed.
        let _ = self.context.bind_to_thread();
        let _ = self.synchronize();
        self.graphs.clear();
        self.launch_graphs = None;
        self.descriptors.clear();
        self.regions.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalars_encode_in_their_dtype() {
        assert_eq!(scalar_bytes("f32", 1.5).unwrap(), 1.5f32.to_le_bytes());
        assert_eq!(scalar_bytes("f16", 1.0).unwrap(), 0x3c00u16.to_le_bytes());
        assert_eq!(scalar_bytes("f16", -2.5).unwrap(), 0xc100u16.to_le_bytes());
        assert_eq!(
            scalar_bytes("f16", 65504.0).unwrap(),
            0x7bffu16.to_le_bytes()
        );
        assert_eq!(scalar_bytes("f16", 1e-7).unwrap(), 0x0002u16.to_le_bytes());
        assert_eq!(scalar_bytes("bf16", 1.0).unwrap(), 0x3f80u16.to_le_bytes());
        assert_eq!(scalar_bytes("bf16", -1e9).unwrap(), 0xce6eu16.to_le_bytes());
        assert_eq!(scalar_bytes("i32", 70.0).unwrap(), 70i32.to_le_bytes());
        assert_eq!(scalar_bytes("bool", 3.0).unwrap(), vec![1]);
        assert!(scalar_bytes("f4e2m1x2", 1.0).is_err());
    }

    #[test]
    fn inert_dims_collapse_like_the_materializer() {
        assert_eq!(
            collapse_inert_dims(&[1, 512], &[64, 32]).unwrap(),
            vec![1, 512]
        );
        assert_eq!(
            collapse_inert_dims(&[7, 512], &[64, 32]).unwrap(),
            vec![7, 512]
        );
        assert_eq!(
            collapse_inert_dims(&[512, 1, 1024], &[64, 128]).unwrap(),
            vec![512, 1024]
        );
        assert_eq!(
            collapse_inert_dims(&[512, 1024], &[8, 64, 128]).unwrap(),
            vec![512, 8, 128]
        );
        assert!(collapse_inert_dims(&[512, 768, 1024], &[64, 128]).is_err());
    }
}
