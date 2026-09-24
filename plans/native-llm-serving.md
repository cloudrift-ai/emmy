# Experimental native LLM serving

## Summary

Build a reusable Rust execution runtime before adding an HTTP server. Python Emmy remains the compiler, tuner,
and launcher. The runtime executes Emmy's exported programs independently for benchmarking and, later, serving.

Once complete cached generation works, add `--native` to the existing serving command to launch `emmy-server`
instead of vLLM. Keep vLLM as the default. Qualify dense Qwen3-0.6B in FP16 on one GPU first.

## Implementation status

[PR #820](https://github.com/cloudrift-ai/emmy/pull/820) merged the static runtime foundation and records RTX 4080
experiments. It does not implement native LLM serving. This plan remains open until the remaining work is completed
or its scope is explicitly revised.

- **Milestone 0 — partial.** API reuse was investigated. Stock and all three Emmy configurations now complete the
  serving matrix with matching fixed-prompt completions. Profiles establish captured execution and little exposed
  between-step CPU overhead. Smaller activation capacity recovers KV space; useful/padded rows and separate
  activation/scratch accounting remain unmeasured. Emmy's qualified schedules remain slower than stock.
  See the [baseline report](../experiments/Qwen3-0.6B/native_baseline/RESULTS.md) and its linked investigations.
- **Milestone 1 — mostly implemented.** Static executable packs, a Rust CUDA executor, graph replay, persistent
  worker supervision, and timeout/CUDA-error recovery are implemented. The run command compares Python and Rust
  using identical artifacts with persistent and one-shot workers. GPU parity and recovery checks pass. General tune
  integration and separately isolated serialization costs remain outstanding. Unsupported dynamic shapes, indirect
  operands, TMA, and buffer aliasing remain explicit limits of the initial static subset.
- **Milestone 2 — cached generation merged in PR #859.** Native cached Qwen3 preparation and execution
  run through the Rust runtime. Tiny-model logits pass at `rtol=atol=1e-3`, including request reset and graph
  replay; Python/Rust logits from the same artifact are bit-identical. A rotary-rounding fix restores the original
  greedy agreement. The explicit FP32-based contract and its failed calibration trials are recorded in the
  [investigation](../experiments/Qwen3-0.6B/native_generation/RESULTS.md). Coverage reaches 256 checkpoint positions
  and independently checks attention through 4,096 positions. All four fresh held-out cases pass. Full-suite
  validation recorded 26 failures reproduced on its main baseline. Prefill is sequential and sampling is greedy.
  General Python-dispatch replacement is not included.
- **Milestone 3 — implemented in PR #890.** Native text and HTTP serving consume the qualified Rust runtime.
  Checkpoint template/tokenizer parity and real GPU HTTP lifecycle tests pass. The full Python suite has no test
  failures; four missing baseline timing records were added and their tests rechecked.
- **Milestone 4 — not implemented.** Serving optimizations remain separate work.

The [runtime report](../experiments/Qwen3-0.6B/native_runtime/RESULTS.md) shows reduced uncaptured submission cost,
but little change in captured GPU time for the tested small programs. This is not evidence of a full-model serving
speedup. Before that merge, full-suite validation had failures reproduced on main; model-golden failures were only
partly checked against main. The merged PR records those validation limits.

**Completed baseline repair:** Merged
[PR #835](https://github.com/cloudrift-ai/emmy/pull/835) rejects unsupported nested fragment epilogues and contraction
roots the binder cannot compute together. The isolated Qwen3 pre-attention program passes GPU parity with synthetic
weights at 1, 4, and 8 tokens. A complete four-configuration rerun held source fixed: stock passed; all three Emmy
configurations compiled their programs without the reproduced rejections, then missed readiness during GPU
initialization. An isolated width-16 post-attention program exceeds its watchdog as one kernel, while explicit
placement cuts pass numerical checks. The remaining work is to qualify serving schedules and repeat the comparison.
The follow-up [PR #847](https://github.com/cloudrift-ai/emmy/pull/847) fixes matrix-vector classification and
half-precision product rounding. A fresh recorded inventory passes all 32 isolated checks and the eight-program
strict release audit on final source `2144b15a`. The complete four-configuration comparison succeeds, and both
fixed-prompt completions match stock in every configuration. Emmy remains slower: short single-request TPOT is
7.935–8.378 ms versus stock's 2.273–2.281 ms. Smaller activation capacity recovers 1.14 GiB of KV space. Profiles
show little exposed between-step CPU time, so the next performance work should improve GPU schedules within the
existing integration before expanding native serving. See the
[schedule report](../experiments/Qwen3-0.6B/native_baseline/SCHEDULES.md). Broader generation parity and detailed
allocation/occupancy accounting remain open; this PR does not implement milestones 2–4.

The user selected parallel work after #847: GPU schedule optimization continues independently, while #859 advances
cached native generation as a correctness and reuse milestone. This does not claim a Rust serving speedup. API work
still follows generation qualification, and dispatch replacement still requires its separate parity inventory.

**Merged PR #876:** seeded GPU temperature/top-p sampling is implemented and passes independent checks.
The full-checkpoint matrix passes all seventeen cases across 10,585 positions, including two 4,096-position prompts.
FP32 attention, rotary intermediates, and residual accumulation close the failures without changing error budgets. The
[follow-up report](../experiments/Qwen3-0.6B/native_generation/SAMPLING_CONTEXT.md) retains failures, repairs, and
deterministic working schedules. PR #890 adds milestone 3 using the selected Axum adapter, tokenizers, and MiniJinja.
PR #885 has since merged the general Rust executor migration into main. Native HTTP serving consumes that shared
runtime; production serving optimizations remain separate work.

## Evidence before implementation

### Compare with the current path

The current generative integration already supports whole-step CUDA graphs. Do not count removal of per-kernel
Python submission as a benefit for steps already captured. Investigate work between replays: metadata preparation,
request scheduling, graph selection, sampling/output processing, transfers, and synchronization. Measure which
costs lie on the critical path; CPU work overlapping GPU computation is not automatically removable latency.

A concrete current inefficiency is fixed-width decode. When the single-token tier is disabled and the decode bucket
is 16, one real token goes through a 16-row compiled pre/post program. Graph replay does not remove padded work.
This does not imply 16x latency. The decision belongs to Emmy's runner, so first compare a smaller compiled width
inside the existing vLLM integration. Do not present this as proof that vLLM must be replaced.

Also measure activation/scratch allocation against the widths actually scheduled. Existing serving documentation
records non-KV footprint limiting concurrency, but that historical evidence is not a current native-runtime result.
Reproduce the relevant workload before claiming recoverable KV capacity or throughput.

The current benchmark implementation already has persistent workers for autotuning and some benchmark sessions;
some comparison paths still use one-shot workers. Inventory the actual path in each experiment. Persistence is not
a new Rust capability. Compare equivalent worker lifetimes before attributing a startup saving to Rust.

### Required experimental reports

Use existing experiment recipes and the benchmark CLI, with profiler collection added through supported mechanisms.
Do not write a separate benchmark script. Add missing reusable measurement controls to the existing harness when
needed. Each experiment retains raw results and a separate Markdown report under the normal experiment structure:

- **Serving dispatch report:** CPU/GPU timelines for current Emmy-vLLM and stock vLLM, with graph coverage verified.
  Separate GPU execution, exposed CPU gaps, scheduling/metadata, transfers, synchronization, sampling, and output.
- **Shape and memory report:** fixed-width versus smaller/exact-width decode, partial/full prefill chunks, and
  mixed request lengths. Record useful/padded rows, activation/scratch/KV bytes, admission, latency, and throughput.
  Try improvements within the current integration first and state which require a different runtime.
- **Runtime benchmark report:** after the minimal executor exists, compare Python and Rust on identical exported
  programs, binaries, inputs, and timing rules. Separate cold startup, warm module loading, allocation, serialization,
  submission, and GPU time. Compare persistent and one-shot execution separately, including timeout recovery.

Before collecting data, pin GPU, driver/toolkit, model revision, code revisions, precision, golden evidence,
context lengths, concurrency, graph settings, and warmup. For serving, include concurrency 1 plus contention,
short and long prompts, and fixed output lengths. Record repeated measurements and their spread; record the exact
matrix in the recipe so comparisons can be rerun. GPU provisioning and long experiment budgets need separate scope.

Reports distinguish existing observations, new measurements, estimates, and unresolved questions. Link every
numeric claim to its raw result and configuration. Estimate the upper bound from exposed overhead, not total CPU
activity: removing a non-overlapped fraction f yields at most 1/(1-f) speedup if other costs stay fixed.
Do not substitute estimates for missing measurements. If no meaningful opportunity appears, revise or stop
with the user before expanding implementation.

## Potential gains and risks

### Potential gains

- **Unified memory planning:** coordinate weights, scratch, activations, and KV cache to reduce actual footprint.
  Replacing an allocator alone saves no memory.
- **Scheduling matched to compiled shapes:** reduce padding and expensive shape changes. Separate benefits available
  inside the current integration from benefits that require owning scheduling.
- **Lower exposed CPU overhead:** reduce step preparation and submission costs that timelines show blocking GPU work.
  Large compute- or bandwidth-bound steps may gain little.
- **Fewer copies and waits:** retain intermediates and sampling on GPU. Existing device-resident paths and graph
  replay are the baseline, not savings to claim again.
- **Reusable deployment artifacts:** run the same compiled program in benchmarks and serving without Python model
  execution, while retaining responsibility for native binary and CUDA compatibility.

### Risks and controls

- **No net performance gain:** benchmark identical work throughout; a faster HTTP layer is not faster inference.
- **Regressions from weaker attention or missing batching:** keep vLLM available and state unsupported workloads.
- **Silent incorrect output:** test logits across prefill, repeated decode, boundary lengths, and consecutive requests.
- **Incomplete artifacts:** prove an executable program first, then a complete model step; no hidden Python callbacks.
- **Unsafe CUDA lifetimes:** one execution thread owns each GPU context and waits for completion before buffer reuse.
- **Fault isolation:** a hung kernel or poisoned context needs process retirement; a persistent thread is not enough.
- **Scope growth:** restrict model family, precision, GPU platform, and API subset; defer multi-GPU and model variants.
- **Maintenance burden:** keep the native interface small and remove superseded dispatch only after parity is proven.
- **Scheduler complexity:** measure fairness, overload, cancellation, and mixed lengths before production batching.

The standalone runtime is a correctness and reuse milestone. Before expanding toward production serving, review
evidence for correct Python-independent cached generation, a measured latency/memory/CPU-efficiency advantage,
and a useful optimization difficult in the current vLLM integration. A single-request test establishes neither
production throughput nor fairness. Continuing, changing scope, or stopping is a decision to make with the user.

## Milestones

### 0. Baseline and API reuse investigation

Produce the serving dispatch and shape/memory reports above. Inventory the existing execution-plan format and
benchmark worker behavior. Evaluate API reuse before committing to a hand-written OpenAI-compatible server:

- NVIDIA Dynamo provides a Rust OpenAI-compatible frontend. Check whether it can call our local runtime without
  adopting its distributed infrastructure. Verify dependency/build cost, license, API coverage, and cancellation.
- `async-openai` is a client library with potentially reusable protocol types, not a server implementation.
  Check type compatibility and feature/dependency cost rather than assuming it provides serving behavior.
- Compare reuse against a thin Axum adapter. Reuse a separable frontend if it meets the narrow contract without
  imposing an unrelated execution stack; otherwise retain Axum and document why. Record the decision before API work.

Without a reusable frontend, we own routes, request/model validation, parameter translation, error/status mapping,
stream chunks and termination, usage/finish reasons, cancellation, and rejection of unsupported fields. Axum handles
HTTP/SSE transport; it does not implement those semantics. Test compatible clients against the selected subset.

Sources: [Dynamo HTTP service](https://docs.dynamo.nvidia.com/dynamo/dev/reference/api/python/llm),
[async-openai](https://docs.rs/async-openai/latest/async_openai/),
and [Axum streaming](https://docs.rs/axum/latest/axum/response/sse/).

### 1. Standalone executable format and Rust runtime

Make this an independently reviewable contribution. Extend the existing execution-plan/pack export, not a second
compiler or optimizer. The current JSON plan already defines buffers, launch order, symbols, and expressions;
document its portable execution contract and only version it when runtime interpretation changes.

The artifact must resolve binaries and constants without a compiler checkout or machine-local cache assumption.
Specify dtypes/layouts, buffer lifetimes and aliasing, zeroing, ordered kernel arguments, launch dimensions, dynamic
shape expressions, target compatibility, and required CUDA features. Start with ordinary static kernels and explicit
I/O; reject unsupported plan features before execution. Expand to dynamic shapes, indirect operands, descriptors,
and graph replay as qualification requires. Rejection is a visible limitation, not silent fallback inside Rust.

Expose load, bind/update inputs and shapes, execute, read outputs, event-time, and release operations through a Rust
library with no HTTP dependency. Test it on small exported programs before integrating a complete model.

Provide a persistent benchmark worker binary using that library. Drive it through Emmy's existing run/tune benchmark
entrypoints using the shared process supervision machinery. Transport control metadata in a versioned framed
protocol; carry tensor payloads as binary data or file references, not JSON numeric arrays. Keep IPC out of the
timed kernel region and account for it separately in end-to-end measurements.

Retain context and loaded CUDA modules across valid jobs; reuse buffers and prepared graphs only where bindings,
capacity, and artifact identity agree. Bound cached resources and support explicit release. Preserve hard deadlines,
process kill/reap, and clean respawn after a hung kernel or poisoned context. Failed jobs must remain visible.

Produce the runtime benchmark report before claiming a pipeline speedup. The persistent Rust worker must be compared
with the existing persistent Python path, not only with a Python process restarted for each job.

### 2. Complete cached generation and dispatch qualification

Extend the artifact/runtime to execute a full dense Qwen3 model without Python callbacks: embedding, pre/post
programs, rotary embedding, attention, final normalization, output head, and sampling. Python may prepare/export
the artifact, but must not run model operations during inference.

Use a preallocated contiguous KV cache and independent CUDA implementations for operations Emmy does not yet emit.
PyTorch remains a reference for parity; it is not the Rust serving attention implementation. Qualify required
attention bindings before claiming a complete Python-independent step. Defer paged cache and FlashInfer unless
the chosen qualified operation requires them.

Prefill computes the prompt once. Decode processes one token at its absolute position and attends to populated KV.
Sample on GPU and retain the next token on-device. CPU observation remains necessary for text decoding and stop
strings; do not claim a fully GPU-driven generation loop. Test uncaptured execution first, then graph replay with
stable addresses and explicit dynamic argument/shape updates.

Keep Python CUDA dispatch during parity qualification. Inventory its callers, diagnostic outputs, per-kernel timing,
dynamic shapes, transfer behavior, and fault recovery. Replace it in run/tune/benchmark paths only after those
required contracts pass and end-to-end benchmark impact is measured. A serving-only subset cannot replace the
general dispatcher. Preserve compiler, tracing, eager references, and vLLM integration until their uses migrate;
the destination is one shared dispatcher, not two permanent implementations.

### 3. Native API serving

Add `emmy-server` as a thin consumer of `emmy-runtime`. Do not build the previously proposed temporary Rust HTTP
server plus Python generation worker. Reuse the API frontend selected in milestone 0 or implement the agreed
Axum subset. Benchmarks call the runtime directly, never through HTTP unless measuring serving.

- Add `emmy serve MODEL --generate --native`; reject `--native --stock` and unsupported native arguments.
  Preserve existing vLLM forwarding and defaults.
- Support host, port, revision, context limit, golden/strict evidence for artifact preparation, dry-run, and the
  existing optional serving benchmark client. Package matching Rust binaries; do not build them on server startup.
- Qualify dense Qwen3-0.6B in FP16 on one GPU. Reject MoE, quantization, sliding attention, and unsupported rotary
  configurations. Default total context to 4,096 tokens within model/compiler limits.
- Provide health, model listing, completions, and chat completions. Support text, streaming/non-streaming output,
  output limits, temperature, top-p, seed, stop strings, token usage, and finish reasons.
- Apply the checkpoint chat template with Qwen3 thinking off by default. Test template/tokenizer parity.
- Admit one active request and return a clear busy response for additional requests. Use bounded output channels,
  cancel on disconnect, and retain admission until submitted GPU work completes and state is released.
- Do not silently retry a request after runtime failure. A poisoned context makes the server unready and requires
  process restart. Keep blocking GPU waits and tokenization off the HTTP I/O threads.

### 4. Optimize measured serving costs

Add continuous batching, paged KV cache, and scheduling coordinated with compiled shapes as separately measured
changes. Qualify mixed request lengths, fairness, overload, and cancellation. Defer deployment images, additional
model families, multi-GPU serving, and production feature parity.

## Dispatch and CPU/GPU transfers

| Responsibility | Owner |
| --- | --- |
| Compilation, tuning policy, artifact preparation | Python Emmy |
| HTTP, streaming, admission | Rust server or selected reusable frontend |
| Tokenization, templates, text decoding | CPU text processing outside GPU execution |
| Program order and buffer requirements | Exported compiler execution plan |
| Generation steps, cache state, CUDA submission | Dedicated Rust execution thread per GPU |
| Enforce stream dependencies and execute work | CUDA driver and GPU hardware |

The runtime binds pointers and current shapes and submits the plan. CUDA orders streams; GPU hardware schedules
thread blocks. Tokio handles CPU I/O tasks, not GPU kernels. There is no automatic CPU/GPU placement policy.

Use explicit allocations and copies. Weights upload once per loaded model; prompt IDs upload once per request.
Activations, intermediate results, and KV cache stay on GPU. Update only small shape/position metadata as needed.
Normal serving decode downloads selected tokens/status, not complete logits. Correctness tests may download full
outputs, and benchmarks must identify those transfers separately.

Use reusable pinned host staging buffers for asynchronous copies. Retain buffers until their completion event fires.
Start with one explicit stream for ordered copies and kernels. Synchronize only at actual CPU-consumption boundaries;
do not synchronize after every layer. Sampling keeps the next token on-device, but autoregressive dependencies
and CPU stop/cancellation decisions still constrain submission.

Add a transfer stream only when profiling shows independent work worth overlapping; connect streams with events.
Cancellation stops future submissions, not an in-flight kernel. Wait before reusing buffers; use process retirement
for irrecoverable GPU faults. No unified-memory migration, CPU weight offload, or cross-process GPU sharing initially.
Python reference paths retain their existing PyTorch/CuPy stream and DLPack rules during comparison.

See [CUDA asynchronous execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html).

## Dependencies and repository structure

Use a small Cargo workspace in this repository. Create the runtime crate first and the server crate only when
API work starts. The runtime has no dependency on the server or Axum.

```text
Cargo.toml                          # Workspace
Cargo.lock
crates/
├── emmy-runtime/
│   ├── Cargo.toml
│   ├── ARCHITECTURE.md
│   ├── src/
│   │   ├── lib.rs                  # HTTP-independent runtime API
│   │   ├── artifact.rs             # Existing plan format reader and validation
│   │   ├── executor.rs             # Program binding and execution
│   │   ├── cuda.rs                 # CUDA resources, streams, copies, events
│   │   ├── generation.rs           # Full model loop, added in milestone 2
│   │   ├── cache.rs                # KV lifecycle, added in milestone 2
│   │   └── bin/emmy-runtime-worker.rs # Persistent isolated benchmark worker
│   └── tests/
└── emmy-server/                     # Added in milestone 3
    ├── Cargo.toml
    ├── ARCHITECTURE.md
    ├── src/
    │   ├── main.rs                 # Startup and shutdown
    │   ├── api.rs                  # Selected frontend adapter
    │   └── text.rs                 # Tokenization, templates, incremental decoding
    └── tests/

emmy/
├── commands/serve.py               # Native/vLLM selection
├── serving/native/launch.py        # Locate and launch server
└── compiler/backend/
    ├── plan.py                     # Existing portable execution contract
    ├── pack.py                     # Existing artifact export
    └── ...                         # Shared benchmark supervision/integration

tests/
├── compiler/backend/               # Export and Python/Rust parity tests
└── serving/native/                 # Launcher and API interoperability tests
```

Do not introduce an HTTP dependency for runtime benchmarking or a separate benchmark script. Keep unsafe CUDA calls
inside the CUDA module. Rust unit tests live beside modules; integration tests live in each crate's tests directory.
Python package directories include their normal initializer. The layout does not require placeholder modules before
their milestone. Extend existing CLI/library mechanisms rather than cloning the benchmark pipeline.

| Stage | Dependencies/frameworks |
| --- | --- |
| Runtime | Serde/serde_json, cudarc, tracing, and the existing exported kernel binaries. |
| Benchmark worker | Clap and framed transport compatible with Emmy's shared supervision. |
| Complete generation | Required CUDA attention/math bindings; Safetensors if retained as weight storage. |
| API/text | Selected frontend; Axum/Tokio if the thin adapter wins, tokenizers, MiniJinja, tracing-subscriber. |
| Python preparation/reference | Existing Emmy compiler, PyTorch, Transformers, CuPy, NumPy, Safetensors tooling. |

cudarc and individual CUDA-library bindings remain qualification choices; verify the required driver APIs,
descriptors, graph behavior, and CUDA versions before adopting them. Do not add Candle or Burn as a second model
framework. No FastAPI/Uvicorn is planned. vLLM remains the production baseline and optional serving benchmark client.

Use Cargo to build binaries, Rustfmt/Clippy/Cargo tests for development, and pinned compatible versions in the lockfile.
Require a compatible NVIDIA driver at runtime and only the native CUDA libraries actually used. NVCC/toolkit belongs
where compilation occurs; a fully precompiled deployment should not need NVCC. Python still drives Emmy's compiler
and launcher, while the prepared Rust runtime can execute independently.

References: [cudarc](https://docs.rs/cudarc/latest/cudarc/),
[Tokenizers](https://docs.rs/tokenizers/latest/tokenizers/),
[MiniJinja](https://docs.rs/crate/minijinja/latest).

## Validation and rollout

- Runtime: Python/Rust parity for exported programs, argument ordering, dtypes, dynamic shapes, aliasing/zeroing,
  graph replay, unsupported features/versions, and GPU compatibility. Validate buffers and cached-module lifetimes.
- Benchmarking: equivalent warm/cold lifetimes, CUDA-event timing, end-to-end latency, binary/input reuse,
  bounded caches, worker death, deadlines, and process recovery after invalid or hung kernels.
- Generation: tiny Qwen3 eager logit parity across prefill and many decode steps, one-token decode input,
  context boundaries, reset between requests, GPU sampling, and tokenizer/template parity.
- Serving: streaming boundaries, Unicode/stop strings, usage/finish reasons, busy responses, backpressure,
  cancellation, readiness, shutdown, and no vLLM/Python model execution on the native path.
- Performance: preserve raw evidence for the three reports; record regressions and limitations as well as gains.
  Do not infer a Rust benefit from different kernels, graph modes, worker lifetimes, or feature subsets.
- Follow repository development and finalization gates: scoped development tests, final full suite, duration records,
  lint, documentation review, and model-golden decoding when compiler code changes. Add the Rust gates when code lands.

Keep vLLM as default throughout qualification. Migrate Python dispatch consumers only after the parity inventory
passes; remove their obsolete machinery in the same scoped migration. This plan is deleted after implementation
lands and durable conclusions are recorded. Experimental reports remain with their reproducible evidence.
