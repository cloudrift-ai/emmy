# CUDA Backend

CUDA-specific dispatch. Shared backend contract lives in
`backend/ARCHITECTURE.md`. The lowering chain that produces the
`Graph[CudaOp]` this backend consumes lives in `pipeline/passes/lowering/`
(see `pipeline/ARCHITECTURE.md`).

## Modules

```
cuda/
├── backend.py        # CudaBackend(Backend) — drives lowering + delegates execution
├── nvcc.py           # offline `nvcc --cubin` compile into the content-addressed cubin cache
├── device.py         # the process's one CUDA context, reached through the runtime
├── program.py        # the facade over the runtime: plan + cubins + host bytes in, outputs and timings out
└── _bench_worker.py  # the SIGKILL-able child that hosts benches and the torch comparison
```

Execution itself lives in the Rust runtime (`crates/emmy-runtime`, hosted in-process through the `emmy_runtime`
extension built from `crates/emmy-runtime-py`): buffers, launches, graphs, events and the hung-launch deadline. Nothing
in this package holds a device pointer.

## Compile

`CudaBackend.compile(graph)` runs
`run_pipeline(graph, [..., "lowering/kernel", "lowering/cuda"], dump=…)`,
producing a `Graph[CudaOp]` where every compute node carries a rendered
`__global__` source plus its launch geometry (grid / block / smem /
arg_order).

## Dispatch (`program.py`)

The graph is first projected to an **execution plan** (`plan_from_graph`, `../plan.py` — buffer specs, constants,
launch list, symbolic plumbing, kernel + weight refs; see `../ARCHITECTURE.md`). `CompiledProgram.build_from_plan`
then compiles every kernel to a cubin path, turns every input and constant into host bytes, and hands the runtime
the plan's JSON form (`plan_to_dict`), those paths and those bytes; the runtime validates the plan, allocates one
device buffer per named buffer, loads the cubins and uploads the bytes. `CompiledProgram.build(graph)` is exactly
`build_from_plan(plan_from_graph(graph))` — the pack path (`../pack.py`) enters at `build_from_plan` with a plan
read from disk whose kernels reference cubins by content-addressed cache key (no codegen, no nvcc), and both paths
share every line downstream. The runtime executes **static ordinary-pointer programs**: a plan with symbolic shapes,
runtime constants or arguments, TMA descriptors, indirect operands or serial launches is refused at build time with
`NotImplementedError` until the runtime grows the feature. The projection:

- Classifies each node as `input` / `constant` / `output` / `scratch`
  from `graph.inputs` / `ConstantOp` membership / `graph.outputs`.
- Compiles each unique `kernel_name` via `nvcc.compile_kernel` (`nvcc.py`):
  offline `nvcc --cubin` into a content-addressed disk cache; the runtime loads the cubin by
  path. Content addressing makes **cross-process source determinism a hard
  contract**: any address- or seed-derived token in rendered source re-keys the kernel
  every boot and silently defeats the cache (a server restart recompiles everything it
  touches) — pinned by `tests/compiler/backend/test_source_determinism.py`, which
  compiles in two subprocesses and asserts identical sources. A cubin loads with no driver PTX→SASS JIT, the
  compile is ~3× faster than a cold NVRTC compile on the complex tile-search kernels that dominate autotune, and
  it is GPU-free, so the cubin cache can be warmed by a parallel pool (planned). A missing or failing `nvcc` is an
  error; there is no NVRTC fallback. Kernels are emitted with `extern "C" __global__` so nothing name-mangles them
  (the cubin symbol loads by `kernel_name`). The compile budget is checked BETWEEN kernels
  (`_compile_kernels`), the only boundary a Python-level check has, so a cold multi-kernel compile that outlives the
  bench worker's wall cap raises `CompileBudgetExceeded` instead of being reported as a dead worker.
- **Opt level** comes from `EMMY_NVCC_FLAGS` (`nvcc.effective_flags`, which
  delegates to `config.nvcc_flags()` — `emmy/config.py` is the single owner
  of `os.environ` for `EMMY_*` vars, incl. `EMMY_NO_NVCC` /
  `EMMY_CUBIN_CACHE`). The CLI sets the flags via `config.set_nvcc_flags`
  (override logic, no longer in the command layer) — `tune`, `compile` and `run` all default to nvcc's own -O3, the
  deployable regime, and `--nvcc-flags` overrides. **Tuning measures in the regime it deploys into**, so a tuned
  latency is the deployed one.

  `FAST_MATH` defaults to true and adds `--use_fast_math`; `EMMY_FAST_MATH=0` omits that flag and disables the
  umbrella's compiler rewrites. Individual precision pins still override the umbrella. For an intermediate-rounding
  diagnostic, pass `--nvcc-flags=--fmad=false`; custom flags follow the policy flag. Disabling contraction alone leaves
  other fast-math transformations enabled. Combine it with `EMMY_FAST_MATH=0` for precise arithmetic checks.
  Explicit tensor-core instructions retain their own accumulation semantics. Accuracy and latency evidence must name
  the effective flags; a precise diagnostic does not qualify the default fast-math kernel. Persistent benchmark
  workers receive the effective `FAST_MATH` value with every request and restore their prior value afterward, so a
  previous request's arithmetic mode cannot leak into the next measurement.

  `tune` used to rank at `-Xcicc -O1` to dodge a cicc front-end blowup on big unrolled register-tile kernels. That
  rationale was measured against the WMMA codegen deleted in #189 four days later; on current codegen (fragment work
  renders as rolled loops with small `#pragma unroll` trip counts) it does not reproduce — over 4,888 nvcc compiles
  spanning the tile inventory, -O3 compiled at a **median 0.96×** of -O1, worst case 1.17×, slowest compile 1.39 s.
  So the lower level bought no compile time while mis-ranking by tile area, and it is no longer a measurement lane.
  It stays reachable through `--nvcc-flags` for the test suite's compile-speed lane; a sweep pinned to it warns, and
  its rows key to that regime so no deploy reads them. Note the blowup is a property of unroll size, not of the opt
  level as such: `EMMY_UNROLL` (below) raised far enough could bring it back.

  The flags are folded into both the cubin cache key (literally) and `Context.structural_key` (split into opt level +
  the other flags, `context.split_opt_level`), so one regime has one key however it is spelled while measurements
  from different regimes never collide. The bench-worker subprocess inherits the env, so its compiles use the
  same flags.
  `EMMY_UNROLL=<n>` caps which static loops emit `#pragma unroll` (the unroll budget — declared in
  `lowering/kernel/_atom.py`, read at the extent-driven unroll sites there). It is a
  pin-only nvcc hint that steers cicc unrolling / register pressure / compile time; it does **not**
  change the emitted-C listing size (the register-tile fragment grid is straight-line regardless).
  Unset → each site's built-in cap; `0` → keep every loop rolled.
- Builds a static launch plan: per launch, a tuple of
  `(kernel, arg_names, grid, block, smem_bytes, zero_outputs)`. The runtime resolves total shared storage against
  the cubin's static allocation once at load; each launch then supplies only the dynamic remainder.

`run_program(graph, input_data) → RunResult`:

1. Turn every input and constant into host bytes (`_host_bytes`, the one fill policy): inputs + optional
   constant overrides come from `input_data`; scalar `ConstantOp`s
   become single-element arrays; an unsupplied input gets a deterministic pseudo-random fill
   (useful for standalone compile-and-benchmark scripts); an unsupplied constant is zeros. BF16 buffers use
   NumPy's `uint16` carrier as raw bits: numeric inputs are round-to-nearest-even encoded before upload, while an
   already-`uint16` source is preserved. The runtime allocates every buffer zeroed and uploads those bytes.
2. Launch each kernel in topological order; `zero_outputs` memsets run
   before the launch.
3. Copy `graph.outputs` buffers back to numpy in their declared order, independently of buffer allocation order. BF16
   outputs remain raw `uint16` bits at this backend boundary;
   command-layer correctness checks decode them.

On targets without native FP8 conversion, E4M3 decode constructs the exact FP16 bit pattern, then widens when the
consumer wants FP32. Subnormals, signed zero and the NaN code follow the dtype contract; exhaustive byte tests cover
both result widths. This removes per-element exponent arithmetic without changing the stored representation.

**One allocation per buffer.** The runtime gives every named buffer its own allocation for the program's
lifetime; scratch buffers are not yet packed into a liveness-planned slab, so a program holds every intermediate at
once (the Python executor packed them, which is what kept all 28 layers' `[heads, S, S]` attention scratch of
Qwen3-Embedding at S=4096 under the card's memory). The lowering contract the packing rested on still holds
(`lowering/cuda/010_lower_kernelop.py`: only atomic-reduction outputs need zeroing and are in `zero_outputs`; every
other kernel fully overwrites its output), so re-adding the planner inside the runtime changes no kernel.

**Repeated execution (`CompiledProgram.rebind` / `run_once`).** `rebind(input_data)` re-uploads supplied program
inputs in place — static shapes only, a supplied constant is refused — and `run_once()` launches every kernel in
program order with none of `iter_once`'s per-launch event record / sync / deadline; the caller's `outputs()`
synchronizes. Both expect the caller to hold `gpu_lock()`. The serving fast path — one program built at a capacity
seq_len, `set_sym_values` / `upload_prefix_device` / `capture_program_graph` per request, `output_prefix_device`
handing the runner a device view — needs symbolic launch geometry, cross-program buffer sharing and device-resident
bindings, none of which the runtime has yet; those methods raise `NotImplementedError`, and the vLLM plugin's runners
do not run on this executor until the runtime grows them.

`benchmark_program(graph, input_data, warmup, num_iters)` adds a warmup loop + timed loop over the runtime's
per-launch event windows (`Executor.time_launch`, one pair of events per launch index, reused across iterations) for
`BenchmarkResult.per_launch`; `time_ms` is the sum of the per-launch medians. The runtime polls each launch's stop
event against a deadline (`_KERNEL_TIMEOUT_MS`, 2 s; `EMMY_KERNEL_TIMEOUT_MS` overrides) rather than blocking on the
driver, which would hang forever on a non-terminating kernel; on overrun it raises **`HungKernelError`** (the
extension's exception, a `RuntimeError` subclass, so callers' `except RuntimeError → bench_fail` still catch it). A
program's FIRST `iter_once` runs under its own deadline, `EMMY_FIRST_ITER_TIMEOUT_MS`
(`config.first_iter_timeout_ms()`, default 30× the steady one): lazy SASS upload, the smem-carveout reconfig for a big
dynamic-smem kernel, and allocator first-touch can legitimately stall iter 0 past the steady-state cap without any
kernel being hung. Before the knob the grace was hard-coupled at 30×, which at the 30 s steady watchdog a serving twin
needs priced every hang at 900 s; set it alone to bound what a hang costs. The 2 s (not 1 s) default is empirical: the
gemma-4 post4096-global twin bench_failed 5/5 under a 1 s deadline at the first post-recalibration iteration yet runs
clean 9/9 with no wait ≥0.2 s at any deadline ≥2 s — a deadline-correlated phantom (mechanism below the driver line
unresolved; see the constant's note). This is the in-process timing core; both the autotune bench and the deployable
comparison run it **inside the worker** (below), so a hung kernel hangs the child, not the parent.

`benchmark_program` captures each launch position's batch into a CUDA graph **by default**
(`capture_graphs=True` → `CompiledProgram.capture_launch_graphs`, right after batch-size calibration),
and the runtime replays that graph — one call per event window — instead of a launch loop. CUDA
events measure *stream elapsed* time, which only equals GPU time when the stream is saturated; for
sub-10 µs kernels per-launch dispatch starves the stream and the events time the starvation. The graph
replay keeps the stream dense, so the same event windows become pure-GPU measurements. The runtime owns
a non-default stream, so capture and replay happen on the stream the program always runs on, and the
deadline poll is unchanged (a hung kernel inside a graph still never completes its stop event). Warmup
iters always run uncaptured, so the zero-elapsed degenerate-launch guard and the deadline probe real
launches before any graph exists. Re-fires of the calibration branch (warmup extension) re-capture when
batch sizes change. A capture failure (`GraphCaptureError`; the runtime ends the capture before
reporting, so the stream is clean) is caught in place: the bench warns,
continues uncaptured, and reports it via `BenchmarkResult.captured` — comparison callers
(`bench_lowered_vs_torch` / `bench_full_model_real`) pair that flag with their torch-side capture and
re-run all-or-nothing so one table never mixes semantics. The tune sweep persists the flag per `perf`
row (`SearchDB.record_perf`): captured measurements supersede wall-semantics ones for the same key
regardless of median, never the reverse — old rows stay usable (replay, prior training) and upgrade
in place as re-tunes measure them captured. See `tests/compiler/backend/test_graph_capture.py`.

**Whole-program (e2e) windows.** The per-launch windows each replay a *single* kernel back-to-back, so
their sum is not an end-to-end time: it misses cross-kernel cache effects and inter-kernel gaps, and on
multi-kernel programs individual per-launch numbers can mis-attribute wildly (two identical-work gemms in
the Qwen3 layer-0 assembly measured 5.2 µs vs 0.8 µs solo; NCU shows them equal). For any **multi-launch** program `benchmark_program`
therefore also captures **one** CUDA graph holding every launch in program order
(`CompiledProgram.capture_program_graph`) and, once per measured iter, times one event window around
`replays` back-to-back whole-program replays (`time_program_window`, replays calibrated to
`_BATCH_TARGET_MS`) — the same semantics the captured torch closures get in the interleaved bench, so the
backend table compares like-for-like. Reported as `BenchmarkResult.e2e_ms`/`e2e_min_ms`; `run --bench`'s
comparison table prefers `e2e_min_ms` for the Emmy row and the kernel table prints a
`whole-program (e2e)` footer beside the per-launch `TOTAL`. Automatic — no flag: a single-launch program's
solo window already IS the program time (the autotune sweep's usual single-node slice — fields stay `None`,
nothing is measured twice), and multi-launch programs get it whenever capture holds (a program-graph capture
failure warns and skips, never fatal; uncaptured benches skip it too). The sweep still *ranks* variants on
the per-op sum (`time_ms`) by design — per-op results key structurally and transfer across graphs, which an
e2e scalar can't — so for its multi-launch slices (split-K fixups) the e2e fields are measured-but-unread
(~1 ms/iter); pricing those variants by slice-e2e instead is a possible future tune-semantics change.

**One worker, two jobs.** `_bench_worker.py`'s `_run_job` dispatches on `torch_spec`: `None` is the
emmy-only bench (`benchmark_program` — the autotune sweep and `run --bench`'s pinned golden / `--ab`
rows; an optional `run_inputs` ndarray dict adds one pre-bench execution on those inputs with the
outputs shipped back — the pinned-row wrong-answer gate's measurement side); otherwise it's the
deployable eager / torch.compile / emmy comparison — `("trace_args", {code/input/adapter/layer/seq_len/dynamic})` →
`load_or_trace` rebuilds the real module (HF id or `--code` expr) → `bench_full_model_real` (for a
symbolic graph the torch closures run on hint-**tiled** example inputs — `commands/run._hint_sized_inputs`
grows every symbolic input axis to its `Dim` hint by repeating the trace values, the same size the
emmy side resolves to when benching without inputs, so the full-model table compares one shape; the
printed table carries a `benched at seq_len=… (symbolic hint)` note);
`("frontend_graph", Graph|None)` → `bench_lowered_vs_torch`. A `trace_args` job honors two run-path
flags: `accuracy` (bind the rebuilt module's real inputs, run the emmy program on them, compare vs
eager — the verdict rides back as `accuracy_error` and a numeric failure skips the bench) and
`want_ref` (return that run's `(inputs, outputs)` as `run_io`). A frontend-graph job may also request
`strict_accuracy`; it returns the direct eager proof and same-input eager outputs used to check exact-pinned rows.
The frontend-graph response also returns the exact symbolic environment used to specialize its hint-sized inputs, so
the parent resolves dynamic launch geometry from the execution binding instead of reconstructing it from lowered
graph inputs that may no longer carry the symbol.
Embedded Loop replay without a derived PyTorch slice has no Torch twin, so its Emmy-only greedy execution returns that
same-input reference too. If
this execution completes but later repeated timing crosses the watchdog, the worker returns the reference, single-run
timing, and exact timing error. The command marks greedy ineligible while still reference-checking pinned rows; the
parent retires that child before any pinned job, so a still-running kernel cannot share its context. It never raises
the watchdog or treats the single run as a candidate. Rebuilding the torch side **in the child** (not pickling a live
module) is what lets the interleaved comparison — which could not cross a subprocess boundary before — run isolated.
So `tune --bench` (`commands/tune.py` `_run_bench` /
`_bench_per_kernel`) and every `run --bench` row go through the worker: a hung kernel
hangs the child, the parent SIGKILLs it at `wall_timeout_s`, the device is freed, and the sweep / A/B
**continues** to the next reproducer or row (no device-poisoning wedge, no skip). The worker starts by
dropping `EMMY_DUMP_DIR` from its own env — a child-built `CudaBackend()` defaults its dump from that
var and `CompilerDump.__post_init__` rmtrees the dir, which would wipe the parent's reproducers. The
whole bench surface is **async-only** — the parent transport is the single **`_AsyncBenchWorker.run_job`**
(the old sync `_BenchWorker` and the sync `benchmark_program_isolated` / `benchmark_compare_isolated`
bridges are gone). `benchmark_compare_isolated_async` awaits a one-shot instance (`_run_job_oneshot`);
the autotune sweep awaits a persistent per-GPU instance directly via `benchmark_program_isolated_async`;
`run --bench` awaits its backend's persistent instance via `CudaBackend.benchmark_compare_async` (greedy
row) / `CudaBackend.bench_pinned_async` (pinned rows) inside one `asyncio.run` session. Synchronous CLI
entry points (`handle_run`, `_handle_run_ir`, `_run_bench`) bridge with `asyncio.run`. See
`tests/compiler/backend/test_bench_worker_compare.py` (compare-in-worker + SIGKILL recovery + the
run-path job flags), `test_hung_kernel_watchdog.py` (watchdog raises promptly), and
`tests/compiler/cli/test_tune_bench_hung_kernel.py` (the `_run_bench` control flow).

The three bench budgets (`bench_compile_timeout_s`, `bench_run_timeout_s`, `bench_wall_timeout_s`) are constructor
policy on the backend, read through live `EMMY_BENCH_COMPILE_TIMEOUT_S` / `EMMY_BENCH_RUN_TIMEOUT_S` /
`EMMY_BENCH_WALL_TIMEOUT_S` overrides (`emmy/config.py` owns the vars, mirroring `EMMY_KERNEL_TIMEOUT_MS`): one env
setting reaches every bench path uniformly — the in-child backend inherits the env, and derived wall caps (the
pinned-row cap, the comparison jobs' workload-scaled cap) recompute from the overridden values. Raising them is how a
golden row whose recorded latency exceeds the default accumulated-GPU budget gets verified. The two watchdog
deadlines beside them — `EMMY_KERNEL_TIMEOUT_MS` (steady) and `EMMY_FIRST_ITER_TIMEOUT_MS` (iter 0, default 30× the
steady one) — are env-only, with no constructor policy, and reach the child the same way.

The three budgets fail differently, and the differences are load-bearing. A `bench_run_timeout_s` overrun is a fact
about the **kernel** — it compiled, it ran, it was too slow — and is recorded as a `bench_fail` at the watchdog's
sentinel latency. A `bench_compile_timeout_s` overrun is a fact about **cicc and the tile's unroll size**: nothing
about the kernel's speed was measured, so it raises `CompileBudgetExceeded` and callers record *nothing at all*
(`search/policy/terminal_bench.py`, and `run --bench`'s pinned rows via `_failed_bench_status`). The exception class
cannot cross the worker pipe (the protocol carries `error` as a string), so the child flags the kind as
`compile_budget: True` and the parent rebuilds it onto `BenchWorkerJobError` — the same shape the retryable
`cache_miss` kind already uses.

`bench_wall_timeout_s` is the third, and it **must exceed the other two**, because it cannot tell them apart: on
overrun the parent SIGKILLs the child and raises a plain `RuntimeError`, so a wall that can fire first collapses both
honest in-child verdicts into one anonymous failure. The compile budget is checked when the compile *returns* and so
can only fire for a compile that finished; the sweep's old wall sat ~2 s above compile+run, which meant any compile
slower than that was killed rather than reported — and the wide register-tile family that motivates the budget is
exactly the family that overruns it. The sweep therefore derives its wall as `compile + run + 60 s`, the same formula
the pinned path uses; the 60 s of headroom is what lets the per-launch watchdog's first-iter grace fire in-child,
which is what actually catches hangs. The wall is a backstop for a wedged worker, not a per-variant budget.

The one-shot comparison result includes the worker's non-fatal `accuracy_error` beside timings, reference
availability, and capture state. `tune --bench` persists that verdict per provenance reproducer instead of treating a
successful timing response as proof of correctness.

**One async transport — `_AsyncBenchWorker`.** It drives the `_bench_worker.py` subprocess protocol (`<8-byte LE
length><pickle>`, both directions) over `asyncio` streams. The child completes short writes of both header and payload.
One event loop keeps N device-pinned workers benching concurrently (`tune --gpus`). Two entry shapes:

- **Autotune sweep** awaits `benchmark_program_isolated_async(graph, worker=…)`. `CudaBackend(device_id=i)` lazily owns
  one **persistent** worker (reused across configs — pay the ~0.2 s Python spawn once) and exposes `benchmark_async`,
  the single benchmarking entry point: the isolated-worker path when `bench_wall_timeout_s` is set and no `on_iter`,
  else the in-process `benchmark_program` path (the interleave `bench_lowered_vs_torch` / `bench_full_model_real`
  drive — which itself now runs inside a worker child for every `--bench`). The device pin is a **per-worker spawn-env overlay** —
  `CUDA_VISIBLE_DEVICES=<id>` (so the child's logical device 0 *is* that GPU, the one ordinal `device.py` ever
  opens) plus, when a base `EMMY_GPU_LOCK` is set, a per-device `…-<id>` lock path so workers on
  different GPUs take distinct `FileLock`s instead of serialising. The overlay rides the child only — the parent's
  `os.environ` is never mutated (all slots share one event-loop thread).
- **Deployable `--bench`**: `tune --bench` awaits `benchmark_compare_isolated_async`, which uses `_run_job_oneshot`
  (spawn → run → `aclose`; the worker's streams bind to the loop, so it can't persist across `asyncio.run` calls — a
  per-call spawn, negligible against a minutes-long deployable bench). `run --bench` instead runs its whole session
  (greedy comparison + every pinned row) in ONE `asyncio.run` over the backend's persistent worker
  (`benchmark_compare_worker_async` / `benchmark_pinned_isolated_async`), closed via `aclose_async_worker` before the
  loop exits.

Strict comparison jobs disable Torch's FP16/BF16 reduced-precision reductions while checking outputs and timing the
reference. cuBLAS can otherwise change intermediate rounding with matrix shape. The precision settings and any
independent split-K setting are restored on success or failure so a persistent worker does not change later jobs.

The wall-clock cap is `asyncio.wait_for`; on overrun the child is SIGKILLed and the next bench respawns it on a clean
device. Because the persistent worker is reused across configs, an illegal / misaligned access is a hazard: that error
is **sticky** — it corrupts the CUDA context so every later call returns the same status until the process dies, which
would cascade identical false `bench_fail`s across all subsequent configs. So after any failure the worker probes its
context (`_context_dirty` — the runtime's context synchronize) and, if it's poisoned — or a hung kernel holds it,
which no probe can tell (`HungKernelError` counts as dirty unprobed) — answers with `_retire_worker: True` and stops
serving. `run_job` retires a child on that flag **itself**, SIGKILL + reap through `aclose()`, the same teardown a
wall overrun takes, and the next request respawns a clean context. The child cannot be left to exit on its own: a hung
kernel stays resident until its context dies and the interpreter's CUDA teardown blocks behind it, so the child became
a zombie still holding the GPU, the next candidate's request wedged against it before any launch, and the wall budget
priced *that* configuration as a failure (a 16× V100 host tune recorded six "did not accept the request" rows for
kernels that never ran). Benign failures (NVRTC compile errors, cleaned-up OOM) leave the context healthy and keep the
worker alive, so they pay no respawn cost. A stale-worker race on the send (a `BrokenPipeError`/`ConnectionResetError`
from `stdin.drain` against a child that exited) triggers one respawn + resend before surfacing as `bench worker died
during request send`. Error paths `await aclose()` (SIGKILL + reap) so the subprocess transport is cleaned before the
loop closes.

Three transport behaviors worth knowing: (1) the child's **stderr is drained continuously** by a background task into
a bounded tail — a chatty child (HF shard-download progress, nvcc warnings) would otherwise fill the ~64 KB pipe and
block mid-job, misread as a wall-timeout — and every failure message (timeout, EOF, in-child error) carries that tail;
(2) an in-child failure's **traceback is logged** by `run_job` (never discarded), the raise is the typed
`BenchWorkerJobError`, and a bare in-child `sys.exit` is reported as such with a pointer at the traceback/stderr
rather than an opaque `SystemExit(1)`; (3) pinned-row **reference inputs are cached in the child** under a
session-unique `run_inputs_key` (hundreds of MB on the big `--code` shapes — shipped once per child, not per row),
with a typed cache-miss + one retry-with-inputs covering a respawn racing the parent's key tracking. See
`tests/compiler/backend/test_bench_worker_recovery.py`, `test_async_bench_worker.py`, and
`test_bench_worker_compare.py`.

The runtime's timed launch rejects an event reading of `<= 0.0` as `bench_fail`
instead of accepting it as a 0µs sample. CUDA event timing has sub-µs resolution and
any real launch consumes at least one device cycle — a 0.0 reading means a
degenerate kernel (BM=1×BN=128 with the M tile fully masked, a kernel fused into a
no-op, an event-pair quirk) that would otherwise lock in as the autotune DB's
unbeatable best. The existing worker → parent → `record_perf(bench_fail)` path
carries the failure unchanged.

`run_program_debug(...)` snapshots every non-input buffer after each
launch — consumed by `--dump-dir` runs.

## Invariants

The plan's JSON form is the one contract between this package and the runtime: an in-process build and a standalone
pack hand the runtime the same document, and only a change in how the runtime interprets it bumps the plan format.
The native generation worker shares its parent-side supervisor with a JSON codec and no retries; process cancellation
and malformed responses kill/reap the child, while ordinary Python job errors still honor the child's healthy/retire
verdict. See the runtime's architecture for the supported subset and timing boundaries.

- `CudaOp.arg_order` embeds the original node id as the output buffer
  name. The lowering rules therefore mutate node ops **in place**
  instead of splicing a fresh node — see `pipeline/ARCHITECTURE.md`
  under "Rule module convention".
- The CUDA backend imports from `ir/` and `pipeline/` but never into
  them. A ROCm/SYCL/Metal backend replaces `program.py` only.

The Python worker's protocol encoder adds the current compiler precision policy to its message. The shared process
supervisor owns only deadlines and process lifetime; it does not inject compiler settings into native runtime
commands. Prepared-pack reference commands keep that policy in the outer Python message.
