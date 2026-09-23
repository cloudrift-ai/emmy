# One executor: the Rust runtime behind every launch

## Summary

Make `crates/emmy-runtime` the only code that allocates buffers, binds inputs, launches kernels, captures graphs and
times events. Python keeps everything before that line: tracing, passes, tile search, pricing, nvcc, the execution plan
and the pack. The crate is hosted two ways, from one library: **in-process** through a PyO3 extension where the caller
already owns the tensors (the vLLM plugin, `emmy run`, the realization corpus, the accuracy check), and **as the
child process** where a hung kernel must be killable (the bench worker behind `run --bench` and `tune`, and the
Python-free native generation binary).

What this buys, in order of weight: the cupy dependency goes, the `emmy/` core loses the whole Python executor
(about 2,000 lines) and the Python-versus-Rust parity harness, the serving plugin's weights become memory that vLLM's
profiler can see, and there is one execution contract to keep correct instead of two. What it does not buy: no test
suite speedup (the suite spends its time in compile, pricing and torch references; process spawns are bounded at
about one percent), and no serving speedup (steps are already captured as CUDA graphs and the decode step is
GPU-bound). Measured on the dev box: a bare driver context costs 0.35 s, Python plus torch 1.19 s, `import cupy`
0.18 s.

## Design

### One library, two hosts, same process topology as today

| Host | Who | Why this host |
| --- | --- | --- |
| PyO3 extension, in-process | vLLM plugin runners, `emmy run` accuracy/run path, corpus `built`/`correct`, `run --bench` **inside** the bench child | zero-copy on live torch pointers, the eager reference lives in the same process |
| Python bench child (`_bench_worker.py`) hosting the extension | `run --bench`, `tune --bench`, per-kernel sweeps | unchanged: SIGKILL frees the device; torch compare and accuracy stay in the child |
| `emmy-runtime-worker` binary | `emmy generate --native-pack` | the Python-free guarantee of the native generation contract |

The bench child stays a **Python** process. The compare against eager and `torch.compile`, the accuracy gate and the
same-input reference were moved into the child on purpose, so that the interleaved comparison runs isolated; a Rust
bench child would force that back into the parent. The supervisor (`_AsyncBenchWorker`), its job protocol, the wall
deadline and respawn therefore stay as they are. "Single execution path" means one executor library, not one process
type.

### What moves into the crate

The crate reads the **in-memory execution plan** (the JSON `plan_to_dict` already produces), not only the on-disk
standalone pack: cubin bytes and constant bytes come from the host, which for the in-process host is the cubin cache and
numpy/torch, and for the worker is the pack directory. The pack stays the on-disk form; it stops being the only way in.

| Feature the Python executor has today | Crate today | Work |
| --- | --- | --- |
| scratch slab with liveness reuse (`_planner.py`) | one allocation per buffer | medium: pure algorithm port; the crate plans offsets, the host hands it one slab |
| external buffers and an external stream | owns both | medium: every buffer is a caller pointer plus byte length; the stream is a handle |
| symbolic shapes, ceil-div grid factors, runtime args, runtime constants | rejected | large: the three-form expression grammar, rebind re-evaluates extents, grids, runtime args, descriptors, and invalidates the graph |
| indirect operands (`table[sel[slot]]`), serial launches | rejected / not parsed | medium: argument packing and a nested loop with a symbol override |
| TMA descriptors (`cuTensorMapEncodeTiled`, inert-dim collapse, re-encode on pointer change) | rejected | large: driver call through cudarc's dlopen path, by-value 128-byte argument |
| packed and quantized dtypes (`f8e4m3`, `f8e5m2`, `f4e2m1x2`, `f16x2`) | rejected | small: byte widths, logical-versus-stored extents |
| per-launch event windows, batch calibration, median reduce, auto iteration budget (`iter_once`, `benchmark_program`) | one window over N iterations | medium: this is what `tune`'s per-kernel rows and `BenchmarkResult` read |
| hung-kernel deadline on the event wait | none, by design | small: poll with a deadline and return an error; killing the kernel stays the process boundary's job |
| whole-program graph keyed by symbol values, per-launch graphs, external-capture mode (raw launches under vLLM's outer capture) | one whole-program graph | medium |

### In-process memory and streams

In-process, the host allocates. The executor reports the slab size and every buffer's offset; the Python facade
allocates one torch tensor per program (or per arena for the split gen runners) and passes device pointers. Weights
are torch tensors too. This is the change that makes the plugin's memory visible to vLLM's memory profiler; today's
recipes carry a comment and an inflated `gpu_memory_utilization` because cupy held the weights outside torch. The
executor adopts torch's current stream for every submission, so the existing `from_external` dance goes away, and
under `torch.cuda.is_current_stream_capturing()` it submits raw launches, exactly as `run_once` does now.

The extension releases the GIL around submission and waits. It is `abi3` and touches torch only through integer
pointers and stream handles, so it never links against torch's C++ ABI.

### What stays in Python

`plan.py`, `plan_cache.py`, `pack.py`, `nvcc.py`, `render_target.py`, `dtype.py`'s name tables, `torch_ref.py`,
`_bench_worker.py`'s job protocol, `_AsyncBenchWorker`, `gpu_lock.py`, and a thin `CompiledProgram` facade:
`build_from_plan` allocates through torch and calls the extension; `rebind`, `set_sym_values`, `run_once`,
`capture_program_graph`, `replay_program_graph`, `upload_prefix_device`, `output_prefix_device` and `outputs` keep
their names and become one-line forwards, so the three serving runners and the corpus helpers do not change shape.

## Audit: what the design drops or simplifies

### Python core

- `cuda/program.py` (2,079 lines): loses `_load_kernel`/`_load_plan`, `_materialize`/`_allocate`, the slab and arena
  code, `_resolve_symbolic`, `_launch`, `_prebuild_descriptors`/`_collapse_inert_dims`, `GraphCaptureError`,
  `_wait_for_event`/`HungKernelError`, `iter_once`, `time_program_window`, `capture_*`/`replay_*`, the batch
  calibration and sample reduction. Keeps the facade, `benchmark_program`'s budget policy if the crate does not absorb
  it, and the supervisor.
- `cuda/_planner.py` (119) and the encode half of `cuda/_tma.py` (188): deleted; the descriptor metadata in `plan.py`
  stays.
- `backend/native.py` (250): `PythonPackWorker`, `PackReference` and the `benchmark_pack` two-by-two-by-two matrix go.
  `NativeWorker` stays for generation.
- `dtype.py::cupy_dtype`, and every `import cupy` outside the executor: `nvcc.py` (capability probe, `RawModule`
  load), `target.py`, `gpu.py`, `commands/tune.py`, `commands/run.py` (device probes), `serving/roofline.py` (cupy
  events), `serving/vllm_model_gen.py` (pool trim), `pipeline/search/policy/terminal_bench.py` (`deviceSynchronize`).
  Each becomes a torch call or an extension query.
- `emmy-runtime-worker` binary: drops `load`/`bind`/`run`; keeps the generation operations. The three GPU tests that
  used the bench operations re-point at the extension.

### CLI and config

| Surface | Disposition |
| --- | --- |
| `emmy run --pack DIR` and its "both runtimes produced identical outputs" matrix | drop; there is no second runtime to compare |
| `EMMY_FIRST_ITER_TIMEOUT_MS` | keep, now the crate's event-wait deadline for the first iteration |
| `EMMY_BENCH_COMPILE_TIMEOUT_S` | keep; it caps nvcc, which stays in Python |
| `EMMY_BENCH_RUN_TIMEOUT_S`, `EMMY_BENCH_WALL_TIMEOUT_S` | keep; run budget goes to the crate's bench loop, wall cap stays the SIGKILL |
| `EMMY_GPU_LOCK` | keep, narrower: it serializes processes at one card, never launches within one |
| `EMMY_BENCH_BACKENDS` | keep, orthogonal |
| `emmy generate --export-native/--native-pack/--capture/--timeout` | keep; the "requires `--native-pack`" errors flatten once the pack is the only input |
| `emmy serve` | no change |

### Tests

| File | Today | After |
| --- | --- | --- |
| `tests/compiler/backend/test_program.py` (16) | cupy dispatch, symbolic capacity guard, inert-dim collapse, slab reuse, timing | delete; the capacity and collapse rules become Rust unit tests |
| `test_planner.py` (11) | liveness/slab allocator | becomes Rust unit tests, same cases |
| `test_graph_capture.py` (12) | cupy capture inside `benchmark_program`, torch-side capture flag | capture mechanics to Rust; the torch-side flag cases stay |
| `test_hung_kernel_watchdog.py` (1) | cupy event polling raises promptly | Rust integration test with a spinning kernel |
| `test_bench_worker_compare.py` (20), `test_bench_worker_recovery.py` (10), `test_async_bench_worker.py` (7) | supervisor and child mechanics | unchanged: the supervisor and the Python child stay |
| `test_native.py` (7) | supervisor contract plus two matrix tests | keep 5, delete the 2 that assert the matrix |
| `test_native_gpu.py` (3), `test_pack_gpu.py` (5) | executed through the worker bench ops / through cupy | re-point at the extension |
| `test_bench_budget_env.py` (10) | knob semantics | keep; the knobs survive |
| `tests/compiler/cli/test_run_pack.py` (2) | `--pack` plumbing | delete with the flag |
| `tests/serving/native/test_generation_gpu.py` (19 nodes, 2,303 s) | already Rust | unchanged |
| `tests/compiler/backend/test_execution_plan.py`, `test_plan_template_cache.py`, `test_pack.py` | compile side | unchanged, plus one CPU test: every corpus case's plan deserializes in the crate |

Fixtures: `tests/conftest.py`'s context-poison probe and CUDA skip gate stop importing cupy (torch synchronize,
extension import); the `xdist_group("cuda")` routing and the two reserved chains stay as they are, since in-process
execution is still in-process. `tests/compiler/realization/helpers.py` and `tests/compiler/conftest.py`'s `run_graph`
keep calling the facade.

### Dependencies, build, images

| Item | Change |
| --- | --- |
| `cupy-cuda12x` (dev extra), `CUPY_PACKAGE` in `docker/vllm-emmy`, `cupy-cuda12x==14.1.1` in `docker/1cat-vllm-sm70`, `VLLM_EMMY_CUPY_PACKAGE` in the Makefile, `import cupy` in `corpus-timings.yml` | removed |
| new PyPI distribution `emmy-runtime` (maturin, `abi3-py312`, `manylinux_2_28` x86_64 and aarch64) | added; `emmy-ml` pins it `==` because the plan format couples them; it dlopens `libcuda.so.1` lazily so a CPU-only install still imports |
| `make setup` | adds `maturin develop --release` for the extension; a Rust toolchain becomes a developer prerequisite (README and AGENTS.md say "only for the experimental runtime" today) |
| `.github/workflows/tests.yml` native job | builds the extension too; CPU Rust tests include the plan-deserialization corpus check |
| `.github/workflows/publish.yml` | second job or workflow for the runtime wheel via `maturin-action`; the "exactly one wheel" assertion stays true per distribution |
| `docker/vllm-emmy` | `pip install emmy-runtime==<pin>` next to the emmy wheel; no Rust toolchain in the image |
| `flake.nix` | adds `rustc`, `cargo`, `maturin` |
| recipes' `gpu_memory_utilization` comments (Qwen3-Embedding 0.6B/4B/8B, gemma-4-12B-it) | rewritten: weights are torch memory now |

cudarc with `dynamic-loading` and a fixed `cuda-12000` feature needs no CUDA toolkit at build time; the crate's
ARCHITECTURE.md still says it does and gets corrected.

## Status (2026-09-23)

The Python executor is gone and the in-process host exists: `crates/emmy-runtime-py` builds the `emmy_runtime`
extension, `program.py` is a facade over it, cupy is out of the compiler, the commands and the test gates, and
`emmy run` executes static programs through the runtime with per-launch timing, graph capture and the hung-launch
deadline. What the runtime still refuses, in the order it should gain them: serial launches and runtime arguments
(the recurrence tests), symbolic shapes with rebind, the scratch slab, external buffers and streams for the serving
runners, TMA descriptors, indirect operands. The `emmy-runtime` PyPI wheel and the image builds are not wired yet:
`make setup` builds the extension from source.

## Migration steps

Each step is one PR with its own gate. Step 1 is the only one that must precede the others.

1. **Full plan contract in the crate.** Extend `artifact.rs` to deserialize every plan field (`serial`,
   `indirect_args`, `runtime_args`, `runtime_constants`, `symbols`, `cuda.tma`, packed dtypes, kernel `source`) and
   to validate rather than reject. Gate: a CPU test feeds every realization-corpus plan and every golden's decoded
   plan through the crate reader.
2. **Executor features.** Slab planner, external buffers and stream, expression evaluator and rebind, indirect and
   serial launches, TMA encode, per-launch timing windows with the deadline, graph modes. Gate: Rust unit tests per
   feature, plus a temporary `EMMY_EXECUTOR={python,rust}` switch in the facade so the corpus `correct` stage and
   the golden replays run both executors on the same cubins and compare outputs bitwise. The switch is deleted in
   step 4.
3. **Extension and packaging.** `crates/emmy-runtime-py`, maturin, `make setup`, CI, nix, publish job, docker
   images. Gate: `pip install emmy-runtime` in a plain venv imports without a GPU; `make test-native` passes with the
   extension.
4. **Cutover.** Facade over the extension; delete the Python executor pieces, `_planner.py`, the `_tma.py` encoder,
   the pack comparison, `--pack`, cupy everywhere; serving runners allocate through torch; conftest probes through
   torch; tests moved as listed. Gate: `make test` on the dev box against main's known failures, `make bench-kernels`
   with no case slower than its stored number, and the same-image serving A/B recipe on the Gemma 4 and Qwen3
   embedding recipes to prove the allocation change costs nothing.
5. **Finalization.** Backend, crate and native ARCHITECTURE.md rewrites; README prerequisites; AGENTS.md's Rust
   sentence; `durations.json` for moved tests; line balance (`emmy/` shrinks by about 2,000 lines, the crate grows by
   about the same); delete this plan.

## Risks

- **Allocation through torch's caching allocator** can fragment differently from cupy's pool under vLLM. The A/B in
  step 4 is the check; if it loses, the fallback is one reserved torch tensor per program held for the runner's life,
  which is what the arena does today.
- **A hung kernel in-process** still poisons the pytest worker, exactly as today; the deadline only makes the failure
  prompt. Nothing new, but the crate's ARCHITECTURE.md must keep saying so.
- **Two release artifacts** where there was one pure wheel. Pinning `==` avoids skew but means every runtime change
  is a coordinated release.
- **The GIL.** Every submission and wait must drop it, or the async bench supervisor and vLLM's worker threads stall.
- **Feature parity is the whole cost.** Steps 1 and 2 are most of the work and produce no user-visible change; the
  bitwise gate is what keeps them honest.

## Open decisions

- Whether `benchmark_program`'s budget policy (warmup extension, auto iterations, GPU-time run budget) moves into the
  crate or stays as Python driving the crate's per-iteration timing. Moving it removes more Python; keeping it keeps
  the policy readable next to the knobs it reads.
- Whether aarch64 wheels ship from day one. Skipping them keeps the publish job simple; GH200-class hosts then need a
  source build.
