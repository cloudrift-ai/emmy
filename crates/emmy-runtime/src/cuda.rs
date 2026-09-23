//! CUDA ownership and submission. Only trusted compiler-produced cubins may be loaded.

use crate::artifact::{Artifact, Plan, dimensions};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaGraph, CudaSlice, CudaStream, LaunchConfig, PushKernelArg, sys,
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

pub struct Executor {
    pub load_times_ms: BTreeMap<&'static str, f64>,
    plan: Plan,
    graph: Option<CudaGraph>,
    stream: Arc<CudaStream>,
    arrays: BTreeMap<String, CudaSlice<u8>>,
    functions: BTreeMap<String, CudaFunction>,
    bound: BTreeSet<String>,
    completed: bool,
}

impl Executor {
    /// Load a validated, trusted artifact; all buffers and functions belong to one stream.
    pub fn load(device: &Device, mut artifact: Artifact) -> Result<Self> {
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
            let static_smem = u32::try_from(function.shared_size_bytes()?)?;
            let dynamic_smem = smem.saturating_sub(static_smem);
            if smem > DEFAULT_SHARED_MEMORY_BYTES && dynamic_smem > 0 {
                function.set_attribute(
                    sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    i32::try_from(dynamic_smem)?,
                )?;
            }
            // The plan records total storage; static bytes are already reserved by the cubin.
            for launch in artifact.plan.launches.iter_mut().filter(|l| l.kernel == name) {
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

    fn submit(&mut self) -> Result<()> {
        ensure!(
            self.plan.inputs.iter().all(|n| self.bound.contains(n)),
            "all program inputs must be bound"
        );
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
                args.arg(&self.arrays[name]);
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
