//! Execute a CPU plan: kernels from an ahead-of-time compiled shared library, launched in plan
//! order on a persistent thread pool, over host memory laid out exactly as the CUDA executor
//! lays out device memory.
//!
//! Every kernel exports `<name>_part(bufs, lo, hi, partial, sizes)`, which runs iterations
//! `[lo, hi)` of its launch's split axis, and a `reduce` launch's kernel also exports
//! `<name>_finish(bufs, partials, nchunks, sizes)`, which combines the per-chunk partials.
//! `bufs` holds the launch's args in order; `sizes` its runtime args.

use crate::artifact::{Artifact, Env, Layout, Placement, Program};
use crate::cuda::{BufferView, MILLISECONDS_PER_SECOND, scalar_bytes};
use anyhow::{Context, Result, bail, ensure};
use libloading::Library;
use std::alloc::{self, Layout as Allocation};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

/// Cache-line alignment, enough for any vector load the kernels emit.
const REGION_ALIGN: usize = 64;
/// Chunks per thread of a launch that splits independent work: a thread that finishes early
/// takes another, so an efficiency core holds the work up by one small chunk.
const CHUNKS_PER_THREAD: usize = 4;
/// Chunks of a split reduction, whatever the thread count: the partials and the order `finish`
/// adds them in depend on the shape alone, so the result is the same bits on any machine.
const REDUCE_CHUNKS: usize = 32;
/// Byte alignment of the partials: a cache line on Apple silicon. The compiler pads each chunk's
/// partials to whole lines, so no two chunks write one line.
const PARTIAL_ALIGN: usize = 128;
/// How long an idle worker polls before it sleeps: back-to-back launches of one run reach it
/// while it still polls, and an idle executor costs no CPU.
const SPINS_BEFORE_SLEEP: u32 = 1 << 16;
/// How long the launching thread polls for the workers' acknowledgements before it yields its
/// core to them, which matters when there are more threads than cores.
const SPINS_BEFORE_YIELD: u32 = 1 << 10;

type Part = unsafe extern "C" fn(*const *mut c_void, i64, i64, *mut f32, *const i64);
type Finish = unsafe extern "C" fn(*const *mut c_void, *mut f32, i64, *const i64);

struct Function {
    part: Part,
    finish: Option<Finish>,
}

/// Zeroed host memory the executor owns.
struct Region {
    ptr: *mut u8,
    len: usize,
}

impl Region {
    fn allocate(len: usize) -> Result<Self> {
        let len = len.max(1);
        let layout = Allocation::from_size_align(len, REGION_ALIGN)?;
        // SAFETY: `layout` has a nonzero size.
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        ensure!(!ptr.is_null(), "could not allocate {len} bytes");
        Ok(Self { ptr, len })
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from `alloc_zeroed` with exactly this size and alignment.
        unsafe {
            alloc::dealloc(
                self.ptr,
                Allocation::from_size_align_unchecked(self.len, REGION_ALIGN),
            )
        };
    }
}

/// One launch's work, shared with the pool. `Pool::run` returns only after every worker has
/// acknowledged the launch, so the pointers outlive every use.
#[derive(Clone, Copy)]
struct Job {
    part: Part,
    bufs: *const *mut c_void,
    extent: i64,
    chunks: usize,
    partials: *mut f32,
    partial_floats: usize,
    sizes: *const i64,
}

// SAFETY: a `Job` only travels to the pool's workers while `Pool::run` keeps its pointees alive.
unsafe impl Send for Job {}

impl Job {
    /// SAFETY: `bufs` and `sizes` hold what the kernel's ABI expects, and `partials` (when not
    /// null) holds `chunks * partial_floats` floats.
    unsafe fn run(&self, chunk: usize) {
        let n = self.chunks as i64;
        let (lo, hi) = (
            self.extent * chunk as i64 / n,
            self.extent * (chunk as i64 + 1) / n,
        );
        let partial = if self.partials.is_null() {
            std::ptr::null_mut()
        } else {
            unsafe { self.partials.add(chunk * self.partial_floats) }
        };
        unsafe { (self.part)(self.bufs, lo, hi, partial, self.sizes) };
    }
}

struct Shared {
    job: Mutex<Option<Job>>,
    generation: AtomicUsize,
    next: AtomicUsize,
    done: AtomicUsize,
    stop: AtomicBool,
    sleep: Mutex<()>,
    wake: Condvar,
}

impl Shared {
    /// Start a new generation. Bumped under the sleep lock, so a worker deciding to sleep
    /// cannot miss it.
    fn publish(&self) {
        let guard = self.sleep.lock().unwrap();
        self.generation.fetch_add(1, Ordering::AcqRel);
        drop(guard);
        self.wake.notify_all();
    }
}

/// Workers that poll a generation counter, then sleep on a condition variable. The launching
/// thread takes chunks too, so a pool of `threads - 1` workers gives `threads`-way parallelism.
struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl Pool {
    fn new(workers: usize) -> Result<Self> {
        let mut pool = Self {
            shared: Arc::new(Shared {
                job: Mutex::new(None),
                generation: AtomicUsize::new(0),
                next: AtomicUsize::new(0),
                done: AtomicUsize::new(0),
                stop: AtomicBool::new(false),
                sleep: Mutex::new(()),
                wake: Condvar::new(),
            }),
            workers: Vec::with_capacity(workers),
        };
        for index in 1..=workers {
            let shared = Arc::clone(&pool.shared);
            // On failure `pool` drops here, which stops and joins the workers already spawned.
            let handle = std::thread::Builder::new()
                .name(format!("emmy-cpu-{index}"))
                .spawn(move || worker(&shared))?;
            pool.workers.push(handle);
        }
        Ok(pool)
    }

    /// SAFETY: as for `Job::run`, for every chunk.
    unsafe fn run(&self, job: Job) {
        if job.chunks <= 1 || self.workers.is_empty() {
            for chunk in 0..job.chunks.max(1) {
                unsafe { job.run(chunk) };
            }
            return;
        }
        *self.shared.job.lock().unwrap() = Some(job);
        self.shared.next.store(0, Ordering::Release);
        self.shared.done.store(0, Ordering::Release);
        self.shared.publish();
        unsafe { drain(&self.shared, &job) };
        let mut spins = 0;
        while self.shared.done.load(Ordering::Acquire) < self.workers.len() {
            if spins < SPINS_BEFORE_YIELD {
                spins += 1;
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }
}

/// Run chunks until none is left: a thread that finishes early takes more.
///
/// SAFETY: as for `Job::run`.
unsafe fn drain(shared: &Shared, job: &Job) {
    loop {
        let chunk = shared.next.fetch_add(1, Ordering::Relaxed);
        if chunk >= job.chunks {
            return;
        }
        unsafe { job.run(chunk) };
    }
}

fn worker(shared: &Shared) {
    let mut seen = 0;
    loop {
        let mut spins = 0;
        while shared.generation.load(Ordering::Acquire) == seen {
            if spins < SPINS_BEFORE_SLEEP {
                spins += 1;
                std::hint::spin_loop();
                continue;
            }
            let guard = shared.sleep.lock().unwrap();
            if shared.generation.load(Ordering::Acquire) == seen {
                drop(shared.wake.wait(guard).unwrap());
            }
        }
        seen = shared.generation.load(Ordering::Acquire);
        if shared.stop.load(Ordering::Acquire) {
            return;
        }
        let job = shared
            .job
            .lock()
            .unwrap()
            .expect("a job accompanies every generation");
        // SAFETY: the launching thread keeps the job's pointees alive until `done` counts us.
        unsafe { drain(shared, &job) };
        shared.done.fetch_add(1, Ordering::AcqRel);
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        self.shared.publish();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// One loaded CPU program: its buffers in host memory, its kernels, and the pool that runs them.
pub struct Executor {
    pub load_times_ms: BTreeMap<&'static str, f64>,
    program: Arc<Program>,
    // Fields drop in declaration order: the pool joins its workers before the library unloads
    // and before the regions they write are freed.
    pool: Pool,
    threads: usize,
    functions: BTreeMap<String, Function>,
    _library: Library,
    env: Env,
    layout: Layout,
    regions: BTreeMap<String, Region>,
    bound: BTreeSet<String>,
    partials: Vec<f32>,
}

// SAFETY: the raw pointers are host memory this executor owns. Methods that write through them
// take `&mut self`; the `&self` methods only read.
unsafe impl Send for Executor {}
unsafe impl Sync for Executor {}

impl Executor {
    /// Load `artifact`, whose binaries all name the one shared library holding its kernels, at its
    /// hints plus `env`, running launches on `threads` threads.
    pub fn load(artifact: Artifact, threads: usize, env: Option<Env>) -> Result<Self> {
        let program = artifact.program;
        ensure!(
            program.backend == "cpu",
            "a {} plan cannot run on the CPU executor",
            program.backend
        );
        ensure!(
            program.paged.is_empty(),
            "paged buffers are not supported on the CPU executor"
        );
        let started = Instant::now();
        let mut paths = artifact.binaries.values();
        let path = paths.next().context("a cpu plan without kernels")?;
        ensure!(
            paths.all(|p| p == path),
            "every kernel of a cpu plan lives in one library"
        );
        // SAFETY: the library comes from a trusted compiler (see the artifact contract) and runs
        // no initializers beyond the C runtime's.
        let library =
            unsafe { Library::new(path) }.with_context(|| format!("load {}", path.display()))?;
        let mut functions = BTreeMap::new();
        for name in artifact.binaries.keys() {
            let reduce = program
                .launches
                .iter()
                .any(|l| &l.kernel == name && l.cpu.as_ref().is_some_and(|c| c.reduce));
            // SAFETY: the compiler emits both symbols with exactly these signatures.
            let part = unsafe { *library.get::<Part>(format!("{name}_part").as_bytes())? };
            let finish = if reduce {
                Some(unsafe { *library.get::<Finish>(format!("{name}_finish").as_bytes())? })
            } else {
                None
            };
            functions.insert(name.clone(), Function { part, finish });
        }
        let module_ms = started.elapsed().as_secs_f64() * MILLISECONDS_PER_SECOND;
        let started = Instant::now();
        let mut full_env = program.default_env();
        full_env.extend(env.unwrap_or_default());
        let layout = program.layout(&full_env)?;
        let threads = threads.max(1);
        let mut executor = Self {
            load_times_ms: BTreeMap::new(),
            program,
            pool: Pool::new(threads - 1)?,
            threads,
            functions,
            _library: library,
            env: full_env,
            layout: Layout::default(),
            regions: BTreeMap::new(),
            bound: BTreeSet::new(),
            partials: Vec::new(),
        };
        executor.provision(layout)?;
        executor.check_launches()?;
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

    /// Everything a launch names resolves, so a run never fails halfway through the plan.
    fn check_launches(&self) -> Result<()> {
        for launch in &self.program.launches {
            let cpu = launch
                .cpu
                .as_ref()
                .context("a cpu plan launch without a cpu section")?;
            let function = self
                .functions
                .get(&launch.kernel)
                .with_context(|| format!("no kernel {}", launch.kernel))?;
            ensure!(
                !cpu.reduce || function.finish.is_some(),
                "reduce launch {} has no finish",
                launch.node_id
            );
            for name in launch.args.iter().chain(&launch.zero_outputs) {
                self.placement(name)?;
            }
            for name in &launch.runtime_args {
                ensure!(
                    self.env.contains_key(name),
                    "unbound runtime argument {name}"
                );
            }
            cpu.extent.eval(&self.env)?;
        }
        Ok(())
    }

    /// Adopt the regions `layout` needs: a region large enough for its new size is kept, anything
    /// else is reallocated zeroed.
    fn provision(&mut self, layout: Layout) -> Result<()> {
        let mut old = std::mem::take(&mut self.regions);
        for (name, &size) in &layout.regions {
            let region = match old.remove(name) {
                Some(region) if region.len >= size.max(1) => region,
                _ => Region::allocate(size)?,
            };
            self.regions.insert(name.clone(), region);
        }
        self.layout = layout;
        Ok(())
    }

    fn placement(&self, name: &str) -> Result<&Placement> {
        self.layout
            .buffers
            .get(name)
            .with_context(|| format!("unknown buffer {name}"))
    }

    fn address(&self, name: &str) -> Result<*mut u8> {
        let placement = self.placement(name)?;
        let region = self
            .regions
            .get(&placement.region)
            .with_context(|| format!("region {} is not provisioned", placement.region))?;
        // SAFETY: the layout places every buffer inside its region.
        Ok(unsafe { region.ptr.add(placement.offset) })
    }

    /// A buffer's address, allocated bytes and shape under the current environment.
    pub fn buffer(&self, name: &str) -> Result<BufferView> {
        let placement = self.placement(name)?;
        Ok(BufferView {
            ptr: self.address(name)? as u64,
            bytes: placement.bytes,
            shape: self.program.buffer(name)?.resolve_shape(&self.env)?,
        })
    }

    pub fn env(&self) -> &Env {
        &self.env
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    fn upload(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        let placement = self.placement(name)?;
        ensure!(
            bytes.len() <= placement.bytes,
            "{} bytes exceed buffer {name} ({} bytes)",
            bytes.len(),
            placement.bytes
        );
        let dst = self.address(name)?;
        // SAFETY: `dst` has `placement.bytes` writable bytes and `bytes` is no longer.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len()) };
        self.bound.insert(name.into());
        Ok(())
    }

    /// Copy host bytes into a program input's prefix.
    pub fn bind(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        ensure!(
            self.program.inputs.iter().any(|n| n == name),
            "only program inputs may be updated"
        );
        self.upload(name, bytes)
    }

    /// Re-lay the program out under `env`, upload `bindings` and refill the runtime constants.
    /// Buffers whose regions survive keep their contents.
    pub fn rebind(&mut self, env: Env, bindings: BTreeMap<String, Vec<u8>>) -> Result<()> {
        let mut full_env = self.program.default_env();
        full_env.extend(env);
        let layout = self.program.layout(&full_env)?;
        self.env = full_env;
        self.provision(layout)?;
        self.check_launches()?;
        for (name, data) in &bindings {
            self.program.check_bindable(name)?;
            self.upload(name, data)?;
        }
        self.apply_runtime_constants()
    }

    /// Bind symbolic axes without touching memory; every buffer stays at its allocated capacity.
    pub fn set_env(&mut self, env: Env) -> Result<()> {
        let mut merged = self.env.clone();
        merged.extend(env);
        for buffer in &self.program.buffers {
            let need = buffer.byte_len(&merged)?;
            let have = self.placement(&buffer.name)?.bytes;
            ensure!(
                need <= have,
                "buffer {} resolves to {need} bytes past its capacity of {have}",
                buffer.name
            );
        }
        self.env = merged;
        self.check_launches()?;
        self.apply_runtime_constants()
    }

    fn apply_runtime_constants(&mut self) -> Result<()> {
        let program = self.program.clone();
        for (name, expr) in &program.runtime_constants {
            let buffer = program.buffer(name)?;
            let pattern = scalar_bytes(&buffer.dtype, expr.eval_f64(&self.env)?)?;
            let count = buffer.byte_len(&self.env)? / pattern.len();
            self.upload(name, &pattern.repeat(count))?;
        }
        Ok(())
    }

    /// Run every launch once, in plan order.
    pub fn run_once(&mut self) -> Result<()> {
        ensure!(
            self.program.inputs.iter().all(|n| self.bound.contains(n)),
            "all program inputs must be bound"
        );
        let program = self.program.clone();
        for launch in &program.launches {
            // Every name below resolved in `check_launches`.
            let Some(cpu) = launch.cpu.as_ref() else {
                bail!("a cpu plan launch without a cpu section")
            };
            for name in &launch.zero_outputs {
                let bytes = self.placement(name)?.bytes;
                // SAFETY: the buffer has `bytes` bytes in its region.
                unsafe { std::ptr::write_bytes(self.address(name)?, 0, bytes) };
            }
            let bufs = launch
                .args
                .iter()
                .map(|n| Ok(self.address(n)? as *mut c_void))
                .collect::<Result<Vec<_>>>()?;
            let sizes = launch
                .runtime_args
                .iter()
                .map(|n| self.env[n])
                .collect::<Vec<i64>>();
            let extent = cpu.extent.eval(&self.env)?.max(0);
            let chunks = match (cpu.parallel, cpu.reduce) {
                (false, _) => 1,
                (true, true) => REDUCE_CHUNKS,
                (true, false) => self.threads.saturating_mul(CHUNKS_PER_THREAD),
            }
            .min(extent.max(1) as usize);
            let function = &self.functions[&launch.kernel];
            let partials = if cpu.reduce {
                let floats = chunks
                    .checked_mul(cpu.partial_floats)
                    .context("partials overflow")?;
                // One line more than needed, so the first chunk's partials start on a line boundary.
                self.partials
                    .resize(floats + PARTIAL_ALIGN / size_of::<f32>(), 0.0);
                let base = self.partials.as_mut_ptr();
                // SAFETY: the offset is under one line, inside the allocation.
                unsafe { base.add(base.align_offset(PARTIAL_ALIGN)) }
            } else {
                std::ptr::null_mut()
            };
            let job = Job {
                part: function.part,
                bufs: bufs.as_ptr(),
                extent,
                chunks,
                partials,
                partial_floats: cpu.partial_floats,
                sizes: sizes.as_ptr(),
            };
            // SAFETY: `bufs`, `sizes` and the partials outlive the call, which waits for every chunk.
            unsafe { self.pool.run(job) };
            if let (true, Some(finish)) = (cpu.reduce, function.finish) {
                // SAFETY: as above; every chunk has written its partials.
                unsafe { finish(bufs.as_ptr(), partials, chunks as i64, sizes.as_ptr()) };
            }
        }
        Ok(())
    }

    /// A buffer's allocated bytes.
    pub fn read(&self, name: &str) -> Result<Vec<u8>> {
        let placement = self.placement(name)?;
        let src = self.address(name)?;
        // SAFETY: the buffer has `placement.bytes` readable bytes, and no launch runs during `&self`.
        Ok(unsafe { std::slice::from_raw_parts(src, placement.bytes) }.to_vec())
    }

    pub fn output(&self, name: &str) -> Result<Vec<u8>> {
        ensure!(
            self.program.outputs.iter().any(|n| n == name),
            "unknown program output"
        );
        self.read(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Counts how often each of `EXTENT` iterations ran: `bufs[0]` points at the counters.
    unsafe extern "C" fn count(
        bufs: *const *mut c_void,
        lo: i64,
        hi: i64,
        _: *mut f32,
        _: *const i64,
    ) {
        let counters = unsafe { *bufs } as *const AtomicUsize;
        for i in lo..hi {
            unsafe { &*counters.add(i as usize) }.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn pool_runs_every_iteration_exactly_once_per_launch() {
        const EXTENT: usize = 1000;
        let counters: Vec<AtomicUsize> = (0..EXTENT).map(|_| AtomicUsize::new(0)).collect();
        let bufs = [counters.as_ptr() as *mut c_void];
        let pool = Pool::new(3).unwrap();
        let launches = [1, 2, 7, 16, 999, 1000];
        for &chunks in &launches {
            let job = Job {
                part: count,
                bufs: bufs.as_ptr(),
                extent: EXTENT as i64,
                chunks,
                partials: std::ptr::null_mut(),
                partial_floats: 0,
                sizes: std::ptr::null(),
            };
            unsafe { pool.run(job) };
        }
        drop(pool);
        assert!(
            counters
                .iter()
                .all(|c| c.load(Ordering::Relaxed) == launches.len())
        );
    }
}
