# Native runtime investigation

## Execution contract

Reviewed Emmy revision `d550f9d7ca505484331678dc6a874171e003e584`. The JSON execution plan is the correct
boundary for a standalone executor. It already separates compiler decisions from allocation and launch mechanics.
The pack is not yet a standalone executable artifact: kernel binaries remain in the machine-local cubin cache,
and checkpoint-backed constants require a separate loader. A native export must bundle both without retracing.

The first qualified subset should have explicit input bytes, static contiguous buffers, ordinary pointer arguments,
and precompiled cubins for one exact GPU architecture. Validate the whole plan before allocating or launching.
Reject unsupported fields rather than treating them as empty defaults. Preserve these existing semantics:

- Buffer names identify storage. Input, constant, output, and scratch roles describe initialization and lifetime.
  Scalar constants fill their entire buffer in the buffer dtype; bf16 storage holds encoded bits, not integer values.
- Every grid/block axis is the product of its factors. The expression grammar supports literals, symbols, and
  arithmetic; an initial static executor must reject unresolved symbols and composite expressions it cannot evaluate.
- Launches execute in order on one stream. Arguments follow `args`, with signed integer runtime arguments appended
  in their declared order. Output return order follows the plan's output list, not allocation order.
- `zero_outputs` clears the named buffer before its launch. Scratch reuse must preserve the planner's initialization
  requirements, including zero prologues. Separate zero-initialized allocations are a valid initial implementation;
  claiming the Python allocator's footprint requires reproducing its lifetime-based slab allocation.
- Format 2 introduces indirect arguments. Format 3 additionally carries generated constants. TMA descriptors and
  architecture-specific binaries need explicit qualification; ignoring them can produce an invalid argument ABI.
- External pointer aliases and cross-program backing arrays are runtime bindings, not information fully represented
  in an isolated plan. A standalone export must either capture them explicitly or reject such bindings.

The existing pack loader treats invalidation as a reason to recompile. The standalone runtime must instead return
an error: deployment cannot silently depend on the Python compiler, checkpoint loader, or shared cubin cache.

## Benchmark worker reuse

The Python dispatcher already has persistent workers. Autotuning reuses a worker per GPU; the CLI's benchmark
session reuses the backend worker for greedy and pinned comparisons. The standalone comparison helper also has a
one-shot path. Rust must be compared against matching lifetimes, not against an artificially restarted Python path.

The current parent transport frames pickle messages with an eight-byte little-endian length. Its useful reusable
parts are process ownership, device environment, stderr draining, deadlines, kill/reap, and cache invalidation on
respawn. Pickled compiler graphs are not a portable native protocol. Runtime jobs should reference exported plans
and binary tensor payloads through a versioned control message. Keep transport outside CUDA-event measurement.

Existing recovery distinguishes a failed job from a poisoned CUDA context. Retiring a worker must clear every
parent-side cache key. A serving request must never be automatically retried after a runtime failure.

## API reuse investigation

The full [Dynamo HTTP service](https://github.com/ai-dynamo/dynamo/blob/main/lib/llm/src/http/service/service_v2.rs)
contains useful readiness, draining, streaming, and cancellation behavior. Its state directly owns model discovery
and distributed-runtime services. That is a larger integration contract than one local executor.

The Apache-2.0 [standalone frontend crates](https://github.com/ai-dynamo/frontend-crates) are a more relevant reuse
candidate. Source revision `aa4d464e3adc44851b46f8d36c5e309d3b4ff559` includes protocol types, tokenizers, parsers,
and rendering without the distributed execution stack. Its unpublished demo server uses Axum, but its chat handler
echoes text and synthesizes tool calls. It does not supply the required GPU admission, completion, or cancellation
contract. Adopting that demo would still leave us owning those semantics.

The source workspace uses async-openai 0.41, Axum 0.8, and MiniJinja 2.24; its README's older async-openai version
is not the dependency manifest. [async-openai](https://docs.rs/async-openai/latest/async_openai/) supplies client
APIs and feature-selectable protocol types. It is not a ready-to-use HTTP server.

The working choice is a thin Axum adapter over the runtime, with protocol and text components evaluated separately.
Do not copy the demo's heuristic token counts or silent tokenizer fallback. A build-cost comparison and checkpoint
template parity remain required before choosing text dependencies. No frontend has been built or qualified yet.

## Measurement scope

The recipe pins Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`, FP16, one RTX 4080, a 4,096-token
context limit, Triton attention, and explicit full-graph capture sizes. It records the golden digest and package
versions. Separate empty tune databases and online-prior paths prevent existing machine-local measurements from
changing the compile decision between configurations. The initial baseline used the hardware golden and offline
prior. The completed September 19 run selects the qualified experimental golden and requires measured evidence;
see the schedule report for its validation and limits.

Stock, padded decode, single-token decode, and smaller activation capacity each have one server lifetime. Each
fixed workload has three measurements after two warmup requests. The mixed-length case and profiler case are
separate; profiled latencies must not replace unprofiled measurements. A current desktop display shares this GPU,
so small timing differences require caution. Results and outstanding evidence belong in the accompanying reports.
