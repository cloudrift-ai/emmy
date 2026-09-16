//! CUDA ownership and submission. Only trusted compiler-produced cubins may be loaded.

use crate::artifact::{Artifact, Plan, dimensions};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaContext, CudaFunction, CudaGraph, CudaSlice, CudaStream, LaunchConfig, PushKernelArg, sys};
use cudarc::nvrtc::Ptx;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

pub struct Executor {
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
    pub fn load(context: &Arc<CudaContext>, artifact: Artifact) -> Result<Self> {
        artifact.plan.validate(&artifact.bindings)?;
        let (major, minor) = context.compute_capability()?;
        ensure!(artifact.arch == format!("sm_{major}{minor}"), "artifact GPU architecture mismatch");
        for launch in &artifact.plan.launches {
            let block = dimensions(&launch.block)?;
            let grid = dimensions(&launch.grid)?;
            ensure!(u64::from(block.0) * u64::from(block.1) * u64::from(block.2) <= 1024
                && block.0 <= 1024 && block.1 <= 1024 && block.2 <= 64, "invalid CUDA block");
            ensure!(grid.0 <= i32::MAX as u32 && grid.1 <= 65535 && grid.2 <= 65535, "invalid CUDA grid");
        }
        let stream = context.new_stream()?;
        let mut functions = BTreeMap::new();
        for (name, path) in artifact.binaries {
            let module = context.load_module(Ptx::from_file(path))?;
            let function = module.load_function(&name)?;
            let smem = artifact.plan.launches.iter().filter(|l| l.kernel == name).map(|l| l.smem).max().unwrap_or(0);
            if smem > 48 * 1024 {
                function.set_attribute(sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, i32::try_from(smem)?)?;
            }
            functions.insert(name, function);
        }
        let mut arrays = BTreeMap::new();
        for buffer in &artifact.plan.buffers {
            arrays.insert(buffer.name.clone(), stream.alloc_zeros::<u8>(buffer.byte_len()?.max(1))?);
        }
        let mut executor = Self { plan: artifact.plan, graph: None, stream, arrays, functions, bound: BTreeSet::new(), completed: false };
        for (name, data) in artifact.bindings {
            executor.upload(&name, &data)?;
        }
        executor.stream.synchronize()?;
        Ok(executor)
    }

    fn upload(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        let buffer = self.plan.buffers.iter().find(|b| b.name == name).context("unknown input buffer")?;
        ensure!(bytes.len() == buffer.byte_len()?, "input size mismatch for {name}");
        if !bytes.is_empty() {
            self.stream.memcpy_htod(bytes, self.arrays.get_mut(name).unwrap())?;
        }
        // The caller may release the host slice as soon as this method returns.
        self.stream.synchronize()?;
        self.bound.insert(name.into());
        self.completed = false;
        Ok(())
    }

    pub fn bind(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        ensure!(self.plan.inputs.iter().any(|n| n == name), "only program inputs may be updated");
        self.upload(name, bytes)
    }

    fn submit(&mut self) -> Result<()> {
        ensure!(self.plan.inputs.iter().all(|n| self.bound.contains(n)), "all program inputs must be bound");
        for launch in &self.plan.launches {
            for name in &launch.zero_outputs {
                self.stream.memset_zeros(self.arrays.get_mut(name).unwrap())?;
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
            unsafe { args.launch(config)?; }
        }
        Ok(())
    }

    /// Time ordered program submissions with CUDA events; excludes I/O and control transport.
    /// Uncaptured execution includes exposed host submission gaps.
    pub fn execute(&mut self, warmup: u32, iterations: u32, capture: bool) -> Result<f32> {
        ensure!(iterations > 0 && iterations <= 1_000_000 && warmup <= 1_000_000, "invalid iteration count");
        self.completed = false;
        if capture && self.graph.is_none() {
            self.submit()?;
            self.stream.synchronize()?;
            self.stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            let submitted = self.submit();
            let captured = self.stream.end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
            submitted?;
            self.graph = Some(captured?.context("empty CUDA graph")?);
        }
        for _ in 0..warmup { self.step(capture)?; }
        self.stream.synchronize()?;
        let start = self.stream.record_event(None)?;
        for _ in 0..iterations { self.step(capture)?; }
        let end = self.stream.record_event(None)?;
        let time = start.elapsed_ms(&end)? / iterations as f32;
        self.completed = true;
        Ok(time)
    }

    fn step(&mut self, capture: bool) -> Result<()> {
        if capture { self.graph.as_ref().unwrap().launch()?; } else { self.submit()?; }
        Ok(())
    }

    pub fn output(&self, name: &str) -> Result<Vec<u8>> {
        ensure!(self.completed, "execute must complete before reading outputs");
        ensure!(self.plan.outputs.iter().any(|n| n == name), "unknown program output");
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
