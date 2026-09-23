//! CUDA ownership and submission. Only trusted compiler-produced cubins may be loaded.

use crate::artifact::{Artifact, Plan, dimensions};
use anyhow::{Context, Result, bail, ensure};
use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaGraph, CudaSlice, CudaStream, LaunchConfig,
    PushKernelArg, result, sys,
};
use cudarc::nvrtc::Ptx;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
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

    pub fn compute_capability(&self) -> Result<(i32, i32)> {
        Ok(self.0.compute_capability()?)
    }

    pub fn name(&self) -> Result<String> {
        Ok(self.0.name()?)
    }

    pub fn total_mem(&self) -> Result<usize> {
        Ok(self.0.total_mem()?)
    }

    pub fn attribute(&self, attribute: sys::CUdevice_attribute) -> Result<i32> {
        Ok(self.0.attribute(attribute)?)
    }

    /// Wait for every stream on the context. A sticky error from an earlier fault surfaces
    /// here, which is how a host tells a poisoned context from a healthy one.
    pub fn synchronize(&self) -> Result<()> {
        Ok(self.0.synchronize()?)
    }

    /// The per-device limits an occupancy estimate and the hardware feature probe read.
    pub fn properties(&self) -> Result<BTreeMap<&'static str, f64>> {
        use sys::CUdevice_attribute as A;
        let attr = |a| self.0.attribute(a).map(f64::from);
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
            ("total_mem", self.0.total_mem()? as f64),
        ]))
    }

    /// Load one cubin and report the kernel's static resource usage.
    pub fn kernel_attributes(&self, cubin: &Path, name: &str) -> Result<KernelAttributes> {
        let module = self.0.load_module(Ptx::from_file(cubin))?;
        let function = module.load_function(name)?;
        Ok(KernelAttributes {
            num_regs: function.num_regs()?,
            local_size_bytes: function.local_size_bytes()?,
            shared_size_bytes: function.shared_size_bytes()?,
        })
    }
}

/// Wait for an event with a deadline instead of blocking on the driver.
fn wait_for_event(event: &CudaEvent, deadline_ms: f64, kernel: &str) -> Result<()> {
    let started = Instant::now();
    event.context().bind_to_thread()?;
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

pub struct Executor {
    pub load_times_ms: BTreeMap<&'static str, f64>,
    plan: Plan,
    stream: Arc<CudaStream>,
    arrays: BTreeMap<String, CudaSlice<u8>>,
    functions: BTreeMap<String, CudaFunction>,
    bound: BTreeSet<String>,
    /// One graph holding every launch in program order.
    graph: Option<CudaGraph>,
    /// One graph per launch position, each holding that launch's batch.
    launch_graphs: Option<(Vec<u32>, Vec<CudaGraph>)>,
    /// Per-launch timing events, created once and reused across iterations.
    events: Vec<(CudaEvent, CudaEvent)>,
    window: Option<(CudaEvent, CudaEvent)>,
}

// A captured graph is a raw driver handle the driver does not synchronize, so cudarc leaves it
// `!Send`. The executor only reaches its graphs through `&mut self`, and every driver call
// rebinds the context to the calling thread, so moving the whole executor between threads —
// which is what a host that releases its interpreter lock does — is sound.
unsafe impl Send for Executor {}
unsafe impl Sync for Executor {}

impl Executor {
    /// Load a validated, trusted artifact; all buffers and functions belong to one stream.
    pub fn load(device: &Device, mut artifact: Artifact) -> Result<Self> {
        let context = &device.0;
        artifact.plan.validate(&artifact.bindings)?;
        if let Some(arch) = &artifact.arch {
            let (major, minor) = context.compute_capability()?;
            ensure!(
                *arch == format!("sm_{major}{minor}"),
                "artifact GPU architecture mismatch"
            );
        }
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
            let static_smem = u32::try_from(function.shared_size_bytes()?)?;
            let dynamic_smem = smem.saturating_sub(static_smem);
            if smem > DEFAULT_SHARED_MEMORY_BYTES && dynamic_smem > 0 {
                function.set_attribute(
                    sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    i32::try_from(dynamic_smem)?,
                )?;
            }
            // The plan records total storage; static bytes are already reserved by the cubin.
            for launch in artifact
                .plan
                .launches
                .iter_mut()
                .filter(|l| l.kernel == name)
            {
                launch.smem = launch.smem.saturating_sub(static_smem);
            }
            functions.insert(name, function);
        }
        let module_ms = started.elapsed().as_secs_f64() * MILLISECONDS_PER_SECOND;
        let started = Instant::now();
        let mut arrays = BTreeMap::new();
        for buffer in &artifact.plan.buffers {
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
            stream,
            arrays,
            functions,
            bound: BTreeSet::new(),
            graph: None,
            launch_graphs: None,
            events: Vec::new(),
            window: None,
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
        Ok(())
    }

    pub fn bind(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        ensure!(
            self.plan.inputs.iter().any(|n| n == name),
            "only program inputs may be updated"
        );
        self.upload(name, bytes)
    }

    fn event(&self) -> Result<CudaEvent> {
        Ok(self
            .stream
            .context()
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?)
    }

    fn launch(&mut self, index: usize) -> Result<()> {
        let launch = &self.plan.launches[index];
        for name in &launch.zero_outputs {
            self.stream
                .memset_zeros(self.arrays.get_mut(name).unwrap())?;
        }
        let launch = &self.plan.launches[index];
        let config = LaunchConfig {
            grid_dim: dimensions(&launch.grid)?,
            block_dim: dimensions(&launch.block)?,
            shared_mem_bytes: launch.smem,
        };
        let mut args = self.stream.launch_builder(&self.functions[&launch.kernel]);
        for name in &launch.args {
            args.arg(&self.arrays[name]);
        }
        // The compiler defines the ABI and access bounds. Validation resolves every pointer
        // and launch dimension; arrays stay alive and exclusively owned through completion.
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }

    fn submit(&mut self) -> Result<()> {
        ensure!(
            self.plan.inputs.iter().all(|n| self.bound.contains(n)),
            "all program inputs must be bound"
        );
        for index in 0..self.plan.launches.len() {
            self.launch(index)?;
        }
        Ok(())
    }

    /// Launch every kernel once in program order with no events; the caller's read synchronizes.
    pub fn run_once(&mut self) -> Result<()> {
        self.submit()
    }

    /// Time launch `index` repeated `batch` times inside one event window, returning per-call
    /// milliseconds. The wait polls with a deadline so a hung kernel raises instead of blocking.
    pub fn time_launch(&mut self, index: usize, batch: u32, deadline_ms: f64) -> Result<f32> {
        ensure!(
            index < self.plan.launches.len(),
            "launch index out of range"
        );
        ensure!(batch > 0, "batch must be positive");
        ensure!(
            self.plan.inputs.iter().all(|n| self.bound.contains(n)),
            "all program inputs must be bound"
        );
        while self.events.len() <= index {
            let pair = (self.event()?, self.event()?);
            self.events.push(pair);
        }
        self.events[index].0.record(&self.stream)?;
        match &self.launch_graphs {
            Some((batches, graphs)) => {
                ensure!(
                    batches[index] == batch,
                    "captured batch does not match the requested batch"
                );
                graphs[index].launch()?;
            }
            None => {
                for _ in 0..batch {
                    self.launch(index)?;
                }
            }
        }
        self.events[index].1.record(&self.stream)?;
        let kernel = self.plan.launches[index].kernel.clone();
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
        self.stream.synchronize()?;
        self.stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let submitted = work(self);
        let captured = self.stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        submitted?;
        captured?.context("empty CUDA graph")
    }

    /// Capture each launch position's batch into its own graph; unchanged batches are kept.
    pub fn capture_launch_graphs(&mut self, batch_sizes: &[u32]) -> Result<()> {
        ensure!(
            batch_sizes.len() == self.plan.launches.len(),
            "one batch size per launch"
        );
        if let Some((batches, _)) = &self.launch_graphs
            && batches == batch_sizes
        {
            return Ok(());
        }
        self.launch_graphs = None;
        let mut graphs = Vec::with_capacity(batch_sizes.len());
        for (index, &batch) in batch_sizes.iter().enumerate() {
            ensure!(batch > 0, "batch must be positive");
            graphs.push(self.capture(|executor| {
                for _ in 0..batch {
                    executor.launch(index)?;
                }
                Ok(())
            })?);
        }
        self.launch_graphs = Some((batch_sizes.to_vec(), graphs));
        Ok(())
    }

    /// Capture every launch in program order into one graph (a no-op when one exists).
    pub fn capture_program_graph(&mut self) -> Result<()> {
        if self.graph.is_none() {
            let graph = self.capture(|executor| executor.submit())?;
            self.graph = Some(graph);
        }
        Ok(())
    }

    pub fn replay_program_graph(&mut self) -> Result<()> {
        self.graph
            .as_ref()
            .context("no captured program graph")?
            .launch()?;
        Ok(())
    }

    /// Time `replays` back-to-back replays of the program graph, per replay in milliseconds.
    pub fn time_program_window(&mut self, replays: u32, deadline_ms: f64) -> Result<f32> {
        ensure!(replays > 0, "replays must be positive");
        self.capture_program_graph()?;
        if self.window.is_none() {
            self.window = Some((self.event()?, self.event()?));
        }
        let (start, stop) = self.window.as_ref().unwrap();
        start.record(&self.stream)?;
        for _ in 0..replays {
            self.graph.as_ref().unwrap().launch()?;
        }
        stop.record(&self.stream)?;
        wait_for_event(stop, deadline_ms, "program graph")?;
        Ok(start.elapsed_ms(stop)? / replays as f32)
    }

    /// Time ordered program submissions with CUDA events; excludes I/O and control transport.
    /// Uncaptured execution includes exposed host submission gaps.
    pub fn execute(&mut self, warmup: u32, iterations: u32, capture: bool) -> Result<RunMetrics> {
        ensure!(
            iterations > 0 && iterations <= MAX_RUN_ITERATIONS && warmup <= MAX_RUN_ITERATIONS,
            "invalid iteration count"
        );
        let started = Instant::now();
        if capture && self.graph.is_none() {
            self.submit()?;
            self.capture_program_graph()?;
        }
        let preparation_ms = started.elapsed().as_secs_f64() * MILLISECONDS_PER_SECOND;
        let started = Instant::now();
        for _ in 0..warmup {
            self.step(capture)?;
        }
        self.stream.synchronize()?;
        let warmup_ms = started.elapsed().as_secs_f64() * MILLISECONDS_PER_SECOND;
        let start = self.event()?;
        let end = self.event()?;
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
        if capture {
            self.capture_program_graph()?;
        }
        self.step(capture)?;
        self.stream.synchronize()?;
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

    /// Copy any buffer back to the host after every queued operation has completed.
    pub fn read(&self, name: &str) -> Result<Vec<u8>> {
        let buffer = self
            .plan
            .buffers
            .iter()
            .find(|b| b.name == name)
            .context("unknown buffer")?;
        self.stream.synchronize()?;
        let mut bytes = self.stream.clone_dtoh(&self.arrays[name])?;
        bytes.truncate(buffer.byte_len()?);
        Ok(bytes)
    }

    pub fn output(&self, name: &str) -> Result<Vec<u8>> {
        ensure!(
            self.plan.outputs.iter().any(|n| n == name),
            "unknown program output"
        );
        self.read(name)
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        // Even a partially submitted operation must finish before pointers or modules are freed.
        let _ = self.stream.synchronize();
    }
}
