# Execution runtime

`emmy-runtime` executes trusted Emmy execution plans through the CUDA driver, or on the host CPU (CPU plans, below).
Python owns compilation and artifact preparation; the runtime owns the device — allocations, launches, graphs, events
and the deadline on a hung launch.
The crate has no HTTP, tokenizer, model framework, compiler, or Python dependency.

## Two hosts, one library

- **In-process** through `crates/emmy-runtime-py`, the `emmy.emmy_runtime` extension (PyO3, `abi3`) that
  `emmy/compiler/backend/cuda/program.py` imports. It exposes `Device` (one context and one stream per logical GPU, its
  properties, a context synchronize that surfaces a sticky error, kernel resource attributes read off a cubin, pointer
  attributes), `Program` (a parsed plan and its layout per environment) and `Executor` (load with lent or owned memory,
  bind by bytes or device address, rebind at a new environment, set the environment, adopt a host stream, run once, time
  one launch's batch, capture per-launch and whole-program graphs, replay, time a whole-program window, read any
  buffer), `CpuExecutor` (the same load, bind, rebind, set-environment, lent-region, run and read calls for a CPU plan,
  and per-launch timing), plus a
  DLPack capsule that lets torch address mapped host memory. Every method that touches the device releases the
  interpreter lock. The runtime's hung-launch error surfaces as `emmy_runtime.HungKernelError`, a `RuntimeError`
  subclass. This is the host behind `emmy run`, the accuracy check, the realization corpus, the bench worker's jobs and
  the vLLM plugin's runners.
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

- Plan formats 1 through 5; a CPU plan needs format 5. Every shape and launch factor resolves under a **symbol
  environment**: the plan's hints, overridden by what the host binds (`rebind`, `set_env`). Runtime arguments append the
  environment's values as `int` parameters; a runtime constant fills its buffer with its expression's value in the
  buffer's dtype; a serial launch runs once per coordinate of its axes, in order, each step overriding that axis.
- Contiguous little-endian f16, bf16, f32, f64, signed/unsigned integer storage, one-byte booleans, the one-byte fp8
  and packed fp4 carriers, and the packed f16 pair. bf16 payloads contain encoded uint16 bits.
- Memory is a **layout** the runtime derives per environment (`Program::layout`): one region per input, constant and
  output buffer (`role:name`), and every scratch buffer packed by liveness into one `scratch` slab, 256-byte aligned,
  largest first, deterministically. The host lends memory for any of the regions and the runtime allocates and
  zeroes its own for the rest; a region that survives a rebind keeps its contents. Empty buffers have an address
  but return zero bytes. A plan finds a buffer by name through an index it builds as it parses, never by a scan: with a
  scan per name, a plan of 120,000 buffers took six minutes to lay out, and with the index it takes a second.
- Ordered launch arguments follow `args`. A paged buffer (below) expands in place to its page table, an operand the
  host binds by address, which the plan never declares as a buffer. A TMA descriptor is encoded per environment at
  the source buffer's resolved shape (a prefix-packed symbolic source has the resolved strides, not the
  allocation's) and passed as a pointer to its 128 bytes. `zero_outputs` clears a buffer before its launch;
  `zero_prologues` records zeroing performed inside the kernel and adds no extra memset.
- Whole-program CUDA graph capture prepares any TMA descriptors before entering capture, as per-launch capture
  already does. Descriptor encoding and its stream synchronization cannot occur inside an active capture;
  rebinding an environment invalidates descriptors and captured graphs together.
- On sm_90 and later, a whole-program submission launches a kernel whose plan entry sets `dependent_launch` as a
  programmatic dependent launch: the kernel waits on the grid ahead of it (`griddepcontrol.wait`) before any memory
  access, so its launch overlaps that grid's drain. A launch behind a memset, and every launch of a single-launch
  batch timing, stays serialized, so a kernel timed on its own measures that kernel alone. No kernel releases its
  dependent early: a grid launched while its predecessor still runs stacks its blocks on the SMs free at that moment.
- Cubins must load on the live device. A pack's recorded architecture must equal the device's exact
  `sm_<major><minor>`; an in-process program was compiled for the live device by the host.
- A timed launch whose completion event misses its deadline raises `HungKernel`; a launch that reports zero elapsed
  time is a degenerate no-op and raises, so it can never win a benchmark.

## Paging

A **paged buffer** is a buffer reached through a page table instead of one base address: it is cut along one axis
every `page` elements, and each page lives wherever the runtime put it. The mechanism knows nothing about what a
buffer holds. A host program declares which buffers are paged, along which axis, at what page and with how many
device slots. Everything else the compiler reads off the kernels — which pages a launch reads or writes, and how to
compute that from the launch's inputs (backend `ARCHITECTURE.md`, "Paged buffers") — and the runtime's **pager** owns
the pages and decides where they live. Nothing else — a runner, a model, a layer kind — takes part.

**The pager** is one per device, shared by every executor on it: a prefill program writes the pages a decode program
reads because both name the same buffer. Each paged buffer has its own pool of device slots, as many as the host
declared; a pool whose slots cover every page never evicts, which is the KV cache today. A buffer declared with no
slots has no pool and no paging program, since nothing can move: its page table points every page at its home, bound
once, and every launch reads it in place. A page is **resident** or **paged**. A resident page's home is device
memory: the host declares a buffer's first pages resident, they stay where they were loaded for good, and the host
keeps no copy of them. A paged page's home is pinned host memory (mapped, so a kernel can read it in place over the
link), or disk (below), and it moves through the pool's slots. Moving a page is one copy and one page-table write;
nothing a kernel, a plan or the compiler knows changes. Per pool the pager
keeps, in device memory, the page table the kernels read, which page each slot holds and when each slot was last
used. Host memory is pinned and mapped once at boot from a stated budget, and a boot that cannot get it fails. The
host never touches a table or a page. Programs that run the same launches on the same inputs — tensor-parallel ranks —
compute the same page lists and reach the same tables with no coordination.

**The paging program** moves pages ahead of the launches that read them. It is Emmy programs traced from torch modules
and compiled once with every count a symbol each launch binds; the runtime compiles nothing. Its input is a launch's
**page list**, the pages the launch touches, which a program the compiler emits per launch writes from the launch's own
inputs, or which a host that knows the order of its launches writes itself, the step then a program of its own.
Consecutive launches whose lists' inputs are all written before the first of them share one run of the paging program
over the union of their lists, placed right before the first of them; a list computed from a launch's output waits for
that launch. For each listed page the program looks up the slot. A hit refreshes the slot's age. A miss takes the slot
of the least recently used page not on the list, is copied there from host memory, and the page table swaps the two
pages' addresses; a miss when every slot is on the list keeps its host address and is read in place. A resident page
sits in no slot (its slot entry is -1): every list that names it is a hit, which ages only the spare entry past the
slots, so it is never copied, never evicted and never needs a host copy. Hits write to a spare entry past the end of
each array and the copy skips them, so the program is a fixed launch set at every list and a captured graph records it.
A page a kernel writes (a `Write` to the buffer, known at compile time) is copied back before its slot is reused; a page
never written is dropped; a page nothing has written yet has nothing to copy in. Everything runs on one stream in order,
so no launch reads an entry or a slot while it changes. **Placement is this fixed rule**: a page a launch reads is
copied into a slot when the pool has one to give, read in place from host memory when it does not, and never computed
where it lies. The rate at which a kernel copies a page from mapped host memory is unmeasured.

**Writes into a page.** A program that writes rows into a buffer it also holds — the KV cache at the step's position,
a recurrent state carried from one step to the next — spells its output as the input with those rows replaced (a
scatter), and the compiler stores it into the input's own pages: the output is stored over the input (`Tensor.alias`)
and the two share one page table. A kernel stored over a buffer computes nothing from that buffer's old cells — fusion
keeps them apart — so a state whose next value reads the old one elsewhere is computed into scratch and copied over.

**Eviction is a fixed rule, not evidence.** Least recently used out. An offline replay of OLMoE's routing put
Belady's bound at half to two thirds of LRU's host reads, and the best simple rule tried (a decayed count) saved only
3–11% of them; a rule that measures better replaces LRU in the paging program.

**The disk tier.** A buffer whose page bytes are the file's bytes — contiguous, laid out as the device reads them —
keeps the pages host memory cannot hold in that file, read in place (O_DIRECT through io_uring), never copied to a
second file; a buffer repacked at load ends at the host tier. A miss in host memory reads the page into a pinned host
slot, evicting the least recently used host page and rewriting its entries, and then follows the rules above. A disk
read takes milliseconds, and a captured program cannot wait for the host: the paging program writes its requests to
mapped host memory, waits on a flag (`cuStreamWaitValue32`) a runtime thread raises once it has read what was
missing, then copies; it raises the flag itself when nothing on its list is on disk, so only a disk miss pays the
host round trip. The read rate and the wait's cost are unmeasured.

**Four uses.** All are declarations a host program makes; the mechanism does not tell them apart.

- *A KV cache* is paged along its position axis, with a slot per page. A step writes its rows at the absolute position
  its program indexes and replays as one graph. Spilling a cold request's pages to host is the same move, not built
  until something measures it.
- *A MoE layer's stacked expert weights* are paged along the expert axis, one expert per page, and read at the indices
  the router produced; a quantized expert's codes and scales are two more paged buffers read at the same indices, so
  their hits and misses coincide. Of the budget's experts a token's picks are slots, a cache of recently routed host
  experts, and the rest resident. The dispatch's own pointer tables, resolved in-kernel, are gone: the gather by the
  router's picks is an index the program spells.
- *A token-embedding table* is paged along its rows, 256 rows per page, and read at the step's token ids; with no
  slots every row is read in place from host memory.
- *A vision tower's block weights* are paged along the blocks, one page per block for each projection weight, and the
  host lists the next block's page before its launches, since the blocks run in order. Each block's programs read
  their page through that block's entry of the page table; the slots hold the blocks the last image ran, which the
  next image's cyclic order never hits, so one slot serves, and the rest of the budget is resident blocks.

**What stays out**, each measured on OLMoE on an RTX 5090 at PCIe 5.0 x16, under
`experiments/OLMoE-1B-7B-0125-Instruct/`:

- *A static set of resident pages chosen for speed* (`hot_set_rtx5090`): the cache beats it at every device budget
  and at eight streams. Resident pages are back for host memory, not speed: a cache over the whole budget needs a host
  copy of every page, which a host with less memory than the buffer cannot pin. The budget's resident share is its
  first pages, since nothing ranks the pages before a request routes; the cost is the hits the cache would have kept
  on those slots. Replayed over 1,273 decode steps of Qwen3.6-35B-A3B-FP8's routing at 128 of 256 experts, a cache
  over the whole budget copies a median 25 experts a step, and 120 resident experts beside 8 slots copy 86 (70 with
  16 slots, 53 with 32).
- *Swapping an evicted page back to the host*, which would keep a cache over the whole budget with one host copy of
  the pages off the device only: it doubles the link traffic of a miss.
- *Prediction* (`expert_cache_rtx5090`): copying the pages a predictor names ahead of time lost to the cache alone at
  every budget, since one-stream decode is bound by link bytes and prediction adds bytes. It returns if a workload
  leaves the link idle, or for the disk tier, where a miss costs milliseconds.
- *Computing on the host where the page is* (`cpu_experts_rtx5090`): the cache won at one stream at every budget;
  reading the page list to the host costs the decode graph, and a page computed on the host is never cached. It
  returns on a host whose CPU reads memory several times faster than its link.
- *A fetch inside the kernel*: a kernel cannot drive the copy engines, but it can read a missed page from mapped host
  memory and write it into its slot as it computes. The bytes over the link are the same, the slot still has to be
  chosen ahead of the launch from the whole page list, and what it saves is the kernel's second pass over the copied
  page, which device memory serves in a few percent of the link's time. A variant of the paging program to measure
  later, not the start. *A copy without a cache* moves no fewer bytes than a cache copy, and *a host-driven copy on
  the captured path* serves only remote memory.

## CPU plans

A plan whose `backend` is `cpu` runs on `cpu::Executor`. Its kernels live in one shared library the compiler built
ahead of time, each exporting `<kernel>_part(bufs, lo, hi, partial, sizes)` and, for a split reduction,
`<kernel>_finish(bufs, partials, nchunks, sizes)`; `bufs` follows the launch's `args`, `sizes` its runtime arguments.
Every launch carries a `cpu` section in place of the CUDA grid — the extent of the axis it splits, whether it is a
reduction, the floats of partials one chunk writes, and whether it is worth more than one thread — and none of the
CUDA-only launch fields (TMA, paged operands, serial axes) are accepted. The CUDA executor refuses a CPU plan and the
CPU executor a CUDA one. Buffers, the layout, runtime constants and the symbol environment are exactly the CUDA
executor's, in host memory the CPU executor allocates zeroed and aligned to 64 bytes, or that the host lends
(`set_region`, as on CUDA: the buffers placed in the region read it in place from the next launch on, and count as
bound). Paged buffers are not supported.

Launches run in plan order on a persistent pool: the calling thread plus `threads - 1` workers that poll for work, then
sleep, so an idle executor costs no CPU. Independent work splits into four chunks per thread, handed out from one
counter, so an efficiency core holds a launch up by one small chunk. A split reduction always splits into the same
number of chunks whatever the thread count, each writing its partials on its own cache line, and `finish` adds them in
chunk order: its result is the same bits on one thread or eight. `time_launches` times each launch alone over a
number of runs, optionally after every thread reads a buffer larger than the caches, so a launch reads its operands
from memory.

## Artifact contract

`compiler/backend/pack.py::save_executable` extends the existing pack with `standalone: 1`, a per-program `bindings`
index, bundled `cubin/<binary_key>.cubin` files, and binary tensor payloads. The ordinary plan format is unchanged.
Every constant is resolved before export, including scalars, generated values, and checkpoint weights. A `pages`
index holds, once for every program, the bytes of each buffer the pager holds (paged along its leading axis with
fewer device slots than pages), page after page, and how many of its first pages are resident, so a host reads those
straight onto the device and the rest straight into pinned memory. A pack needs
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
launch position holding that launch's batch, and whole-program graphs keyed by symbol environment and batch size,
least recently used out. Serving uses batch one. A benchmark captures its calibrated batch before timing, then
submits one replay per event window and divides by the number of forwards. Inputs update existing addresses, so a
graph stays valid across input updates; a rebind, a lent region or a released one drops every graph and descriptor.
Lending a region drains the stream first, so the memory it pointed at
is free once the call returns; a host that keeps that memory alive itself can skip the drain (`set_region` with `wait`
off), because each launch reads its addresses when it is queued. The runtime drains anyway when the old memory is its
own, which it frees, or a descriptor baked the address. `run_each` relies on that: it runs the program once per entry
in one call, each entry binding its symbols, lending its buffers undrained, copying its inputs in and its outputs out.
A runtime constant is rewritten in place when the symbols change, so `run_each` drains before an entry of a program
that has one: the entries already queued read their own value first. A host may adopt its own stream for a call
(`set_stream`): launches, device copies and memsets then go there, so a stream that is recording a graph records them,
while host uploads and descriptor encodes always use the device's stream and complete before returning. Graph capture
always happens on the device's stream. A load replaces the previous program; release drops it.

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

`generation::Generator` consumes standalone `decode` and optional `prefill` programs with a versioned generation
contract in the pack key. Prefill loads onto decode's memory: the inputs, state, byte-identical constants and pager
tables it shares are lent to it as it loads, so none is allocated or uploaded twice, and it takes decode's page tables
after. Only its KV cache pages are allocated a second time at load, until it takes decode's table. It owns its scratch
and drops before decode. It validates the fixed input/output names, shapes, dtypes, vocabulary, context capacity, and
EOS IDs before loading the executor. The model remains compiler-prepared; the Rust library has no Qwen3 math
implementation or Python dependency. The native preparation and attention contract lives in
[`serving/native/ARCHITECTURE.md`](../../emmy/serving/native/ARCHITECTURE.md).

A pack's `pages` are read straight from each buffer's file: its resident pages into device memory (`DeviceMemory`)
a 64 MiB chunk at a time, the rest into pinned, mapped host memory (`HostMemory`), one allocation each per buffer,
both declared to one pager with the buffer's page count, page size, slots and resident count. Decode loads with the
pager's tables lent and prefill shares them, so the device holds the slots once and both programs move pages through
the same tables. The pages and the pager outlive both programs.

The pack key's `state` list names the buffers a step writes in place and carries to the next position — a Gated
DeltaNet layer's recurrent state and convolution history. Each is a flat output of the decode program, which prefill
borrows like the other outputs, and `start` zeroes it. With state a prefill chunk is taken only where every row is a
prompt token before the last, since no later step overwrites what a row past the prompt folds into the state.

`start` binds the prompt once, keeps the sampling controls and resets request state. `advance` processes one prefill
chunk or one decode token at the current absolute position, leaving the final prompt token to decode and taking a
chunk only where every one of its rows fits the context; the diagnostic `step` always processes one token. Before
prompt completion either returns no token; afterward it reads back the token the program selected, or at positive
temperature downloads the step's FP32 logits and draws one on the host as the native contract describes, and uploads
it as the next step's input.
Nothing the device runs is hand-written. Its explicit `ignore_eos` control permits fixed-output serving benchmarks to
continue after EOS; ordinary worker generation retains EOS stopping. `generate` owns the complete prompt/decode loop
and stops at EOS or the requested output count, and returns each token with the time since the request started at
which the host had it. Prompt plus requested output must fit capacity. `logits` is an explicit diagnostic download.
All CUDA operations stay inside `cuda`, and a failed step cannot continue the current request.

The executor's stateful `advance` differs from benchmark `execute`: capture does not run an initialization step or
warmup, since executing twice would consume the next token twice. Stable allocations allow the same graph to serve
new requests. The benchmark API retains its warmup and timing behavior.

The optional `sampling` object on `start_generation` and `generate` carries temperature, top-p, and seed; omitted
controls select greedy decoding. The library validates them before binding or submitting GPU work.

The worker adds `load_generation`, `start_generation`, `generation_step`, and `generate`. Prompt and result token
arrays are little-endian i64 binary files; the `generate` response lists each token's time in `token_ms`. Step
responses contain a selected token or null during prefill; optional logits use a little-endian f32 binary output file
under generation artifact version 6. Loading either a generation model or a benchmark program releases the previous
object, and `release` handles both. These additive operations use the existing framed protocol and failure retirement.
