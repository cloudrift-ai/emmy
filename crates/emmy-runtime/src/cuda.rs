//! CUDA ownership and submission. Only trusted compiler-produced cubins may be loaded.

use crate::artifact::{Artifact, Paging, Plan, dimensions};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaGraph, CudaSlice, CudaStream, DevicePtr, LaunchConfig,
    PushKernelArg, sys,
};
use cudarc::nvrtc::Ptx;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;

const MAX_THREADS_PER_BLOCK: u64 = 1024;
const MAX_BLOCK_DIMENSIONS: (u32, u32, u32) = (1024, 1024, 64);
const MAX_GRID_DIMENSIONS: (u32, u32, u32) = (i32::MAX as u32, 65535, 65535);
const DEFAULT_SHARED_MEMORY_BYTES: u32 = 48 * 1024;
const MAX_RUN_ITERATIONS: u32 = 1_000_000;
const MILLISECONDS_PER_SECOND: f64 = 1000.0;

#[derive(Serialize)]
pub struct RunMetrics {
    pub time_ms: f32,
    pub preparation_ms: f64,
    pub warmup_ms: f64,
    pub submission_ms: f64,
    pub completion_wait_ms: f64,
}

pub struct Device(Arc<CudaContext>);

impl Device {
    pub fn new(ordinal: usize) -> Result<Self> {
        let context = CudaContext::new(ordinal)?;
        // Every executor owns disjoint allocations on exactly one stream and synchronizes
        // before releasing them. No device pointer or context escapes this module.
        unsafe {
            context.disable_event_tracking();
        }
        Ok(Self(context))
    }
}

/// Device memory handed out one page at a time.
///
/// A paged buffer's pages are the runtime's to own, not the plan's: the plan says what shape a
/// page has, the pool decides where each one lives. The pool holds its own stream and
/// synchronizes, so one cache can be written by one program and read by another. Residency —
/// whether a page sits in device or host memory — would be a property of a page here, and
/// nothing above this type would change.
pub struct PagePool {
    bytes: usize,
    stream: Arc<CudaStream>,
    pages: Vec<CudaSlice<u8>>,
}

impl PagePool {
    /// A pool of pages of `bytes` each, sized for one buffer's page shape.
    pub fn new(device: &Device, bytes: usize) -> Result<Self> {
        Ok(Self {
            bytes: bytes.max(1),
            stream: device.0.new_stream()?,
            pages: Vec::new(),
        })
    }

    pub fn page_bytes(&self) -> usize {
        self.bytes
    }

    pub fn len(&self) -> usize {
        self.pages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// Add `count` zeroed pages and return their indices, in order.
    pub fn grow(&mut self, count: usize) -> Result<Vec<usize>> {
        let first = self.pages.len();
        for _ in 0..count {
            self.pages.push(self.stream.alloc_zeros::<u8>(self.bytes)?);
        }
        self.stream.synchronize()?;
        Ok((first..self.pages.len()).collect())
    }

    /// Copy one page back to the host. The pages are the caller's, so reading them is too.
    pub fn read(&self, page: usize) -> Result<Vec<u8>> {
        let slice = self.pages.get(page).context("page index out of range")?;
        Ok(self.stream.clone_dtoh(slice)?)
    }

    /// The device table one buffer addresses through: its pages' pointers, in cache order.
    pub fn table(&self, pages: &[usize]) -> Result<PageTable> {
        let mut addresses = Vec::with_capacity(pages.len());
        for index in pages {
            let page = self.pages.get(*index).context("page index out of range")?;
            addresses.push(page.device_ptr(&self.stream).0);
        }
        let device = self.stream.clone_htod(&addresses)?;
        self.stream.synchronize()?;
        Ok(PageTable {
            device,
            len: pages.len(),
        })
    }
}

/// One buffer's page pointers, as the kernel receives them.
pub struct PageTable {
    device: CudaSlice<u64>,
    len: usize,
}

impl PageTable {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

pub struct Executor {
    pub load_times_ms: BTreeMap<&'static str, f64>,
    plan: Plan,
    graph: Option<CudaGraph>,
    stream: Arc<CudaStream>,
    arrays: BTreeMap<String, CudaSlice<u8>>,
    tables: BTreeMap<String, PageTable>,
    symbols: BTreeMap<String, i32>,
    functions: BTreeMap<String, CudaFunction>,
    bound: BTreeSet<String>,
    completed: bool,
}

impl Executor {
    /// Load a validated, trusted artifact; all buffers and functions belong to one stream.
    pub fn load(device: &Device, artifact: Artifact) -> Result<Self> {
        let context = &device.0;
        artifact.plan.validate(&artifact.bindings)?;
        let (major, minor) = context.compute_capability()?;
        ensure!(
            artifact.arch == format!("sm_{major}{minor}"),
            "artifact GPU architecture mismatch"
        );
        for launch in &artifact.plan.launches {
            let block = dimensions(&launch.block)?;
            let grid = dimensions(&launch.grid)?;
            ensure!(
                u64::from(block.0) * u64::from(block.1) * u64::from(block.2)
                    <= MAX_THREADS_PER_BLOCK
                    && block.0 <= MAX_BLOCK_DIMENSIONS.0
                    && block.1 <= MAX_BLOCK_DIMENSIONS.1
                    && block.2 <= MAX_BLOCK_DIMENSIONS.2,
                "invalid CUDA block"
            );
            ensure!(
                grid.0 <= MAX_GRID_DIMENSIONS.0
                    && grid.1 <= MAX_GRID_DIMENSIONS.1
                    && grid.2 <= MAX_GRID_DIMENSIONS.2,
                "invalid CUDA grid"
            );
        }
        let stream = context.new_stream()?;
        let started = Instant::now();
        let mut functions = BTreeMap::new();
        for (name, path) in artifact.binaries {
            let module = context.load_module(Ptx::from_file(path))?;
            let function = module.load_function(&name)?;
            let smem = artifact
                .plan
                .launches
                .iter()
                .filter(|l| l.kernel == name)
                .map(|l| l.smem)
                .max()
                .unwrap_or(0);
            if smem > DEFAULT_SHARED_MEMORY_BYTES {
                function.set_attribute(
                    sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    i32::try_from(smem)?,
                )?;
            }
            functions.insert(name, function);
        }
        let module_ms = started.elapsed().as_secs_f64() * MILLISECONDS_PER_SECOND;
        let started = Instant::now();
        let mut arrays = BTreeMap::new();
        for buffer in &artifact.plan.buffers {
            // A paged buffer is a table of pages the caller owns; there is no slab to allocate.
            if artifact.plan.paged.contains_key(&buffer.name) {
                continue;
            }
            arrays.insert(
                buffer.name.clone(),
                stream.alloc_zeros::<u8>(buffer.byte_len()?.max(1))?,
            );
        }
        stream.synchronize()?;
        let allocation_ms = started.elapsed().as_secs_f64() * MILLISECONDS_PER_SECOND;
        let mut executor = Self {
            load_times_ms: BTreeMap::new(),
            plan: artifact.plan,
            tables: BTreeMap::new(),
            symbols: BTreeMap::new(),
            graph: None,
            stream,
            arrays,
            functions,
            bound: BTreeSet::new(),
            completed: false,
        };
        let started = Instant::now();
        for (name, data) in artifact.bindings {
            executor.upload(&name, &data)?;
        }
        executor.stream.synchronize()?;
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

    fn upload(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        let buffer = self
            .plan
            .buffers
            .iter()
            .find(|b| b.name == name)
            .context("unknown input buffer")?;
        ensure!(
            bytes.len() == buffer.byte_len()?,
            "input size mismatch for {name}"
        );
        if !bytes.is_empty() {
            self.stream
                .memcpy_htod(bytes, self.arrays.get_mut(name).unwrap())?;
        }
        // The caller may release the host slice as soon as this method returns.
        self.stream.synchronize()?;
        self.bound.insert(name.into());
        self.completed = false;
        Ok(())
    }

    pub fn bind(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        ensure!(
            self.plan.inputs.iter().any(|n| n == name),
            "only program inputs may be updated"
        );
        self.upload(name, bytes)
    }

    /// Every paged buffer, with one page's byte size and how many pages its declared shape spans.
    /// A cache-shaped buffer means that span literally; a step's chunk-shaped one does not, and
    /// its caller sizes the cache itself.
    pub fn paged_buffers(&self) -> Result<Vec<(String, usize, usize)>> {
        let mut out = Vec::new();
        for name in self.plan.paged.keys() {
            let page = self.page_bytes(name)?;
            let buffer = self
                .plan
                .buffers
                .iter()
                .find(|b| &b.name == name)
                .context("unknown buffer")?;
            out.push((name.clone(), page, buffer.byte_len()?.div_ceil(page)));
        }
        Ok(out)
    }

    /// One page's byte size for a paged buffer, so the caller can size its pool.
    pub fn page_bytes(&self, name: &str) -> Result<usize> {
        let paging = self.plan.paged.get(name).context("buffer is not paged")?;
        let buffer = self
            .plan
            .buffers
            .iter()
            .find(|b| b.name == name)
            .context("unknown buffer")?;
        paging.page_bytes(buffer)
    }

    /// Give a paged buffer the pages it addresses through. The table replaces what would
    /// otherwise be the buffer's pointer, and the caller owns the pages for as long as it is bound.
    pub fn bind_pages(&mut self, name: &str, table: PageTable) -> Result<()> {
        ensure!(
            self.plan.paged.contains_key(name),
            "buffer {name} is not paged"
        );
        // How many pages a cache has is the caller's to decide — the plan knows only the shape of
        // one page, since a step's buffer spans its chunk while the cache spans a request.
        ensure!(!table.is_empty(), "page table for {name} is empty");
        self.tables.insert(Paging::table(name), table);
        self.bound.insert(name.into());
        self.completed = false;
        Ok(())
    }

    /// Set a runtime argument — the absolute position a paged write lands at.
    pub fn set_symbol(&mut self, name: &str, value: i32) -> Result<()> {
        ensure!(
            self.plan
                .paged
                .values()
                .any(|p| p.start.as_deref() == Some(name)),
            "unknown runtime symbol {name}"
        );
        self.symbols.insert(name.into(), value);
        self.completed = false;
        Ok(())
    }

    fn submit(&mut self) -> Result<()> {
        ensure!(
            self.plan.inputs.iter().all(|n| self.bound.contains(n)),
            "all program inputs must be bound"
        );
        for name in self.plan.paged.keys() {
            ensure!(
                self.tables.contains_key(&Paging::table(name)),
                "paged buffer {name} has no page table bound"
            );
        }
        for launch in &self.plan.launches {
            for name in &launch.zero_outputs {
                self.stream
                    .memset_zeros(self.arrays.get_mut(name).unwrap())?;
            }
            let config = LaunchConfig {
                grid_dim: dimensions(&launch.grid)?,
                block_dim: dimensions(&launch.block)?,
                shared_mem_bytes: launch.smem,
            };
            let mut args = self.stream.launch_builder(&self.functions[&launch.kernel]);
            for name in &launch.args {
                match self.tables.get(name) {
                    Some(table) => args.arg(&table.device),
                    None => args.arg(&self.arrays[name]),
                };
            }
            // Runtime arguments are tail-appended as ``int``, matching the rendered signature.
            let values: Vec<i32> = launch
                .runtime_args
                .iter()
                .map(|n| self.symbols.get(n).copied().context("unset runtime symbol"))
                .collect::<Result<_>>()?;
            for value in &values {
                args.arg(value);
            }
            // The compiler defines the ABI and access bounds. Validation resolves every pointer
            // and launch dimension; arrays stay alive and exclusively owned through completion.
            unsafe {
                args.launch(config)?;
            }
        }
        Ok(())
    }

    /// Time ordered program submissions with CUDA events; excludes I/O and control transport.
    /// Uncaptured execution includes exposed host submission gaps.
    pub fn execute(&mut self, warmup: u32, iterations: u32, capture: bool) -> Result<RunMetrics> {
        ensure!(
            iterations > 0 && iterations <= MAX_RUN_ITERATIONS && warmup <= MAX_RUN_ITERATIONS,
            "invalid iteration count"
        );
        self.completed = false;
        let started = Instant::now();
        if capture && self.graph.is_none() {
            self.submit()?;
            self.stream.synchronize()?;
            self.stream
                .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            let submitted = self.submit();
            let captured = self.stream.end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            submitted?;
            self.graph = Some(captured?.context("empty CUDA graph")?);
        }
        let preparation_ms = started.elapsed().as_secs_f64() * MILLISECONDS_PER_SECOND;
        let started = Instant::now();
        for _ in 0..warmup {
            self.step(capture)?;
        }
        self.stream.synchronize()?;
        let warmup_ms = started.elapsed().as_secs_f64() * MILLISECONDS_PER_SECOND;
        let start = self
            .stream
            .context()
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let end = self
            .stream
            .context()
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        start.record(&self.stream)?;
        let started = Instant::now();
        for _ in 0..iterations {
            self.step(capture)?;
        }
        end.record(&self.stream)?;
        let submission_ms = started.elapsed().as_secs_f64() * MILLISECONDS_PER_SECOND;
        let started = Instant::now();
        let time = start.elapsed_ms(&end)? / iterations as f32;
        let completion_wait_ms = started.elapsed().as_secs_f64() * MILLISECONDS_PER_SECOND;
        self.completed = true;
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
        self.completed = false;
        if capture && self.graph.is_none() {
            self.stream.synchronize()?;
            self.stream
                .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            let submitted = self.submit();
            let captured = self.stream.end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            submitted?;
            self.graph = Some(captured?.context("empty CUDA graph")?);
        }
        self.step(capture)?;
        self.stream.synchronize()?;
        self.completed = true;
        Ok(())
    }

    fn step(&mut self, capture: bool) -> Result<()> {
        if capture {
            self.graph.as_ref().unwrap().launch()?;
        } else {
            self.submit()?;
        }
        Ok(())
    }

    pub fn output(&self, name: &str) -> Result<Vec<u8>> {
        ensure!(
            self.completed,
            "execute must complete before reading outputs"
        );
        ensure!(
            self.plan.outputs.iter().any(|n| n == name),
            "unknown program output"
        );
        ensure!(
            !self.plan.paged.contains_key(name),
            "paged output {name} has no single allocation to read; read its pages"
        );
        let buffer = self.plan.buffers.iter().find(|b| b.name == name).unwrap();
        let mut bytes = self.stream.clone_dtoh(&self.arrays[name])?;
        bytes.truncate(buffer.byte_len()?);
        Ok(bytes)
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        // Even a partially submitted operation must finish before pointers or modules are freed.
        let _ = self.stream.synchronize();
    }
}
