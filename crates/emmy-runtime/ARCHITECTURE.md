# Standalone execution runtime

`emmy-runtime` executes trusted Emmy execution plans through the CUDA driver. Python owns compilation and artifact
preparation. The crate has no HTTP, tokenizer, model framework, compiler, or Python dependency. This is an experimental
static-program runtime, not a complete language-model server; existing serving and autotuning dispatch remain active.

## Artifact contract

`compiler/backend/pack.py::save_executable` extends the existing pack with `standalone: 1`, a per-program `bindings`
index, bundled `cubin/<binary_key>.cubin` files, and binary tensor payloads. The ordinary plan format is unchanged.
Every constant is resolved before export, including scalars, generated values, and checkpoint weights. Initial input
bindings are optional for the library, but required for the CLI comparison. A pack needs no checkpoint, compiler
checkout, or machine-local cubin cache at execution time. Export publishes a new directory only after all files exist.

Only load artifacts from a trusted compiler. Validation detects inconsistent metadata; it cannot prove that a cubin
obeys its declared ABI or memory bounds. Every referenced member must resolve inside the bundle.

The initial supported subset is deliberately explicit:

- CUDA plan formats 1 and 3, with static nonnegative shapes and ordinary pointer arguments. Format 2, symbolic shapes,
  runtime constants/arguments, indirect operands, and TMA descriptors are rejected before submission.
- Contiguous little-endian f16, bf16, f32, f64, signed/unsigned integer storage, and one-byte booleans. bf16 payloads
  contain encoded uint16 bits. Packed and quantized dtypes are not supported.
- One allocation per named buffer, retained for the loaded program's lifetime. There are no external pointer aliases,
  cross-program shared arrays, or liveness-based scratch reuse. Empty buffers have an address but return zero bytes.
- Ordered launch arguments follow `args`. Grid/block axes multiply their integer factors. `zero_outputs` clears a
  buffer before its launch; `zero_prologues` records zeroing performed inside the kernel and adds no extra memset.
- Cubins must target the device's exact `sm_<major><minor>` architecture. Architecture-specific suffixes are rejected.
  CUDA validates register/shared-memory feasibility. Driver errors retire the worker rather than invoking a compiler.

The manifest's standalone version governs bundle resolution. Plan versions govern execution semantics. Changing
compiler scheduling without changing those semantics does not require a new artifact format.

## CUDA ownership

`Device` retains a context; `Executor` owns one stream, its modules, allocations, and optional captured graph. Inputs
update existing addresses, so a graph stays valid across same-sized input updates. A load replaces the previous
program; release drops it. There is no unbounded program or graph cache.

All unsafe CUDA submission stays in `cuda`. Buffer pointers and the context are private. Executors have disjoint
storage and synchronize before releasing it, so cudarc's cross-stream event tracking is disabled. Copies and launches
use the owning stream. Uploads finish before borrowed host bytes can disappear; outputs synchronize before returning.
CUDA graph capture happens after one uncaptured initialization run. Both timing events explicitly enable timing.

A synchronous library call cannot enforce a hard deadline on a hung GPU operation. The process boundary supplies that
contract. Callers needing fault isolation must use the supervised worker rather than wait indefinitely in-process.

## Worker and supervision

Build with `cargo build --release --locked --bin emmy-runtime-worker`. Put the matching binary on `PATH`; startup never
invokes Cargo. Compilation needs the CUDA toolkit. The prepared worker needs a compatible NVIDIA driver. cudarc's
`nvrtc` feature exposes its binary module-loading type, but this path loads cubin files directly and never calls NVRTC.

Each control frame is an eight-byte little-endian byte count followed by UTF-8 JSON, limited to 1 MiB on input:

```json
{"version":1,"command":{"op":"load","root":"/absolute/pack","program":"step"}}
```

Operations are `load`, `bind` (input name to binary file), `run` (warmup, iterations, capture, output name to binary
file), and `release`. Tensor values never travel as JSON numeric arrays. Responses carry `version` and either `result`
or `error` with `retire: true`. stdout is reserved for frames. Any failed native request exits the process; no failed
job is replayed automatically. One caller submits one job at a time.

`compiler/backend/native.py::NativeWorker` reuses the existing asynchronous benchmark supervisor's process ownership,
device environment, bounded stderr drain, deadlines, kill/reap, and clean respawn. It changes the executable and codec
and disables retries. Cancellation and malformed responses also retire the child. The Python worker retains its
existing retry policy and keeps healthy contexts after ordinary compiler errors.

## Comparison and qualification

`emmy run --pack DIR --bench --warmup 20 --iters 200 --json result.json` compares every program using identical bundled
binaries and input bytes. It runs Python and Rust, captured and uncaptured, with persistent and one-shot workers, three
repeats each. Persistent workers also reload the artifact in the retained context. Outputs must match exactly. This
establishes dispatcher parity, not independent mathematical correctness of the compiled program.

CUDA event windows exclude input/output files and control transport. Uncaptured windows include exposed submission
gaps. Parent round trips include IPC, warmup, synchronization, and output copies; they are not GPU kernel timings.
Submission wall time overlaps GPU execution and must not be added to event time as removable overhead. The first load
is process-cold, not necessarily disk/driver-cache-cold. Python's allocation/upload measurement covers submission;
Rust records synchronized allocation/zeroing and upload separately. Those setup subdivisions are not direct matches.

`make test-native` runs Rust unit tests and the Python GPU qualification tests with the built binary. GPU tests cover
ordered launches, constants, zeroing, graph replay, stable input updates, execution with no Python/compiler on PATH,
and clean recovery after a hard deadline or CUDA error. `make lint-native` runs Rustfmt and Clippy. Pull-request CI
runs the CPU Rust gates; GPU qualification remains a separate hardware check.
