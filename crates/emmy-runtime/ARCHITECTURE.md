# Execution runtime

`emmy-runtime` executes trusted Emmy execution plans through the CUDA driver. Python owns compilation and artifact
preparation; the runtime owns the device — allocations, launches, graphs, events and the deadline on a hung launch.
The crate has no HTTP, tokenizer, model framework, compiler, or Python dependency.

## Two hosts, one library

- **In-process** through `crates/emmy-runtime-py`, the `emmy.emmy_runtime` extension (PyO3, `abi3`) that
  `emmy/compiler/backend/cuda/program.py` imports. It exposes `Device` (one context and one stream per process, its
  properties, a context synchronize that surfaces a sticky error, kernel resource attributes read off a cubin, pointer
  attributes), `Program` (a parsed plan and its layout per environment) and `Executor` (load with lent or owned
  memory, bind by bytes or device address, rebind at a new environment, set the environment, adopt a host stream,
  run once, time one launch's batch, capture per-launch and whole-program graphs, replay, time a whole-program
  window, read any buffer), plus a DLPack capsule that lets torch address mapped host memory. Every method that
  touches the device releases the interpreter lock. The runtime's hung-launch error surfaces as
  `emmy_runtime.HungKernelError`, a `RuntimeError` subclass. This is the host behind `emmy run`, the accuracy check,
  the realization corpus, the bench worker's jobs and the vLLM plugin's runners.
- **As a process**, `emmy-runtime-worker`, for the Python-free cached-generation path (below).

Process isolation is still the caller's contract: a synchronous call cannot end a hung kernel, only report it. The
in-process host polls every timed launch's completion event against a deadline and raises, and the bench worker that
hosts it is what gets SIGKILLed.

## Program contract

An in-process program arrives as the plan's JSON form (`plan_to_dict`), the bound input and constant bytes by
buffer name, and one cubin path per kernel (`Artifact::new`): nothing is read from disk but the cubins, and the
plan's `source` field is ignored — the host compiled it. A standalone pack (below) is the same plan read from a
directory. The whole plan grammar is read: an `int` literal, a `"name"` variable, or `[op, lhs, rhs]` with `op` in
`+ - * / // %` (Python's integer semantics), for buffer shapes, grid and block factors and runtime constants.

- CUDA plan formats 1, 2 and 3. Every shape and launch factor resolves under a **symbol environment**: the plan's
  hints, overridden by what the host binds (`rebind`, `set_env`). Runtime arguments append the environment's values
  as `int` parameters; a runtime constant fills its buffer with its expression's value in the buffer's dtype; a
  serial launch runs once per coordinate of its axes, in order, each step overriding that axis.
- Contiguous little-endian f16, bf16, f32, f64, signed/unsigned integer storage, one-byte booleans, the one-byte fp8
  and packed fp4 carriers, and the packed f16 pair. bf16 payloads contain encoded uint16 bits.
- Memory is a **layout** the runtime derives per environment (`Program::layout`): one region per input, constant and
  output buffer (`role:name`), and every scratch buffer packed by liveness into one `scratch` slab, 256-byte aligned,
  largest first, deterministically. The host lends memory for every region or the runtime allocates and zeroes its
  own; a region that survives a rebind keeps its contents. Empty buffers have an address but return zero bytes.
- Ordered launch arguments follow `args`. An indirect operand expands in place to its table pointer, selector pointer
  and slot — operands the host binds by address, which the plan never declares as buffers. A TMA descriptor is
  encoded per environment at the source buffer's resolved shape (a prefix-packed symbolic source has the resolved
  strides, not the allocation's) and passed as a pointer to its 128 bytes. `zero_outputs` clears a buffer before its
  launch; `zero_prologues` records zeroing performed inside the kernel and adds no extra memset.
- Cubins must load on the live device. A pack's recorded architecture must equal the device's exact
  `sm_<major><minor>`; an in-process program was compiled for the live device by the host.
- A timed launch whose completion event misses its deadline raises `HungKernel`; a launch that reports zero elapsed
  time is a degenerate no-op and raises, so it can never win a benchmark.

## Artifact contract

`compiler/backend/pack.py::save_executable` extends the existing pack with `standalone: 1`, a per-program `bindings`
index, bundled `cubin/<binary_key>.cubin` files, and binary tensor payloads. The ordinary plan format is unchanged.
Every constant is resolved before export, including scalars, generated values, and checkpoint weights. A pack needs
no checkpoint, compiler checkout, or machine-local cubin cache at execution time. Export publishes a new directory
only after all files exist.

Only load artifacts from a trusted compiler. Validation detects inconsistent metadata; it cannot prove that a cubin
obeys its declared ABI or memory bounds. Every referenced member must resolve inside the bundle.

The manifest's standalone version governs bundle resolution. Plan versions govern execution semantics. Changing
compiler scheduling without changing those semantics does not require a new artifact format.

## CUDA ownership

`Device` retains a context and one stream that every executor on it launches on and that is never destroyed: a host
allocator that tracked lent memory against it may still record events on it while the process shuts down. `Executor`
owns its modules, the regions it allocated, its timing events, its descriptors and its captured graphs — one per
launch position holding that launch's batch, and one whole-program graph per symbol environment, least recently used
out. Inputs update existing addresses, so a graph stays valid across input updates; a rebind, a lent region or a
released one drops every graph and descriptor. A host may adopt its own stream for a call (`set_stream`): launches,
device copies and memsets then go there, so a stream that is recording a graph records them, while host uploads and
descriptor encodes always use the device's stream and complete before returning. Graph capture always happens on the
device's stream. A load replaces the previous program; release drops it.

All unsafe CUDA submission stays in `cuda`. Buffer pointers and the context are private. Executors have disjoint
storage and synchronize before releasing it, so cudarc's cross-stream event tracking is disabled. Copies and launches
use the owning stream. Uploads finish before borrowed host bytes can disappear; outputs synchronize before returning.
Benchmark CUDA graph capture happens after one uncaptured initialization run. Timing events explicitly enable timing;
a per-launch window records a start and stop event around the batch and polls the stop event with a deadline.
Shared-memory requirements are resolved at load time: the cubin reserves its static storage, and each launch supplies
the remaining dynamic bytes. The same rule applies below and above the default 48 KiB limit.

A synchronous library call cannot enforce a hard deadline on a hung GPU operation. The process boundary supplies that
contract. Callers needing fault isolation must use the supervised worker rather than wait indefinitely in-process.

## Worker and supervision

Build with `cargo build --release --locked --bin emmy-runtime-worker`. Put the matching binary on `PATH`; startup never
invokes Cargo. Neither the worker nor the extension needs the CUDA toolkit to build: cudarc loads `libcuda.so.1`
at run time. Running needs a compatible NVIDIA driver. cudarc's `nvrtc` feature exposes its binary module-loading
type, but this path loads cubin files directly and never calls NVRTC.

cudarc PANICS when that dynamic load fails rather than returning its error, so `Device::new` takes it behind a panic
guard and answers with an ordinary error. A host with no driver is a supported caller — Python's device probe turns
that error into `None` and falls back to memorized per-SKU specs, which is what lets the strict golden decode, offline
eval and `compile --target` run on a machine with no card. A panic crossing the FFI boundary would not: PyO3 re-raises
it as `PanicException`, which derives from `BaseException` and slips straight through every `except Exception`.

Each control frame is an eight-byte little-endian byte count followed by UTF-8 JSON, limited to 1 MiB on input:

```json
{"version":1,"command":{"op":"load","root":"/absolute/pack","program":"step"}}
```

Operations are `load`, `bind` (input name to binary file), `run` (warmup, iterations, capture, output name to binary
file), and `release`; the bench operations remain for the runtime's own GPU qualification. Tensor values never travel
as JSON numeric arrays. Responses carry `version` and either `result` or `error` with `retire: true`. stdout is
reserved for frames. Any failed native request exits the process; no failed job is replayed automatically. One caller
submits one job at a time.

`compiler/backend/native.py::NativeWorker` reuses the existing asynchronous benchmark supervisor's process ownership,
device environment, bounded stderr drain, deadlines, kill/reap, and clean respawn. It changes the executable and codec
and disables retries. Cancellation and malformed responses also retire the child. The Python worker retains its
existing retry policy and keeps healthy contexts after ordinary compiler errors.

## Qualification

`make test-native` runs Rust unit tests and the Python GPU qualification tests with the built binary and the
extension. GPU tests cover ordered launches, constants, zeroing, graph replay, stable input updates, execution with no
Python/compiler on PATH, clean recovery after a hard deadline or CUDA error, and the hung-launch deadline. `make
lint-native` runs Rustfmt and Clippy. Pull-request CI runs the CPU Rust gates; GPU qualification remains a separate
hardware check. The extension is built into the package by `pip install` through setuptools-rust (optional at
build time, so a host without cargo installs pure); `make setup` is that install.

## Cached generation

`generation::Generator` consumes a standalone `decode` program with a versioned generation contract in the pack key.
It validates the fixed input/output names, shapes, dtypes, vocabulary, context capacity, and EOS IDs before loading the
executor. The model remains compiler-prepared; the Rust library has no Qwen3 math implementation or Python dependency.
The native preparation and attention contract lives in
[`serving/native/ARCHITECTURE.md`](../../emmy/serving/native/ARCHITECTURE.md).

`start` binds the prompt and sampling controls once and resets request state. `advance` processes exactly one token
at the current absolute position. Before prompt completion it returns no token; afterward it returns the GPU-selected
ID, which stays on the GPU for the next step. Its explicit `ignore_eos` control permits fixed-output serving
benchmarks to continue after EOS; ordinary worker generation retains EOS stopping. `generate` owns the complete
prompt/decode loop and stops at EOS or the requested output count.
Prompt plus requested output must fit capacity. `logits` is an explicit diagnostic download. All CUDA operations stay
inside `cuda`, and a failed step cannot continue the current request.

The executor's stateful `advance` differs from benchmark `execute`: capture does not run an initialization step or
warmup, since executing twice would consume the next token twice. Stable allocations allow the same graph to serve
new requests. The benchmark API retains its warmup and timing behavior.

The optional `sampling` object on `start_generation` and `generate` carries temperature, top-p, and seed; omitted
controls select greedy decoding. The library validates them before binding or submitting GPU work.

The worker adds `load_generation`, `start_generation`, `generation_step`, and `generate`. Prompt and result token
arrays are little-endian i64 binary files. Step responses contain a selected token or null during prefill; optional
logits use a binary output file. Loading either a generation model or a benchmark program releases the previous
object, and `release` handles both. These additive operations use the existing framed protocol and failure retirement.
