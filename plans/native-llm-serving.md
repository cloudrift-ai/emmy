# Experimental native LLM serving

## Summary

Add `--native` to the existing serving command. Emmy launches our Rust server instead of vLLM. Keep vLLM as
the default. Python remains the compiler and launcher; Rust becomes the API server and, in a second milestone,
the inference runtime.

The first milestone serves dense Qwen3 models in FP16 on one GPU, with cached generation and a streaming API.
It runs without vLLM installed. Success means correct behavior and ownership of execution; outperforming vLLM
comes later.

## Approach

- Reuse Emmy's compiled model runner and device execution paths.
- Use Rust with Axum for HTTP/SSE, Tokio for asynchronous I/O, and Serde for request and response serialization.
  Do not add FastAPI or Uvicorn. Rust is the default for both server concurrency and runtime memory ownership;
  use narrow C/CUDA bindings where existing GPU libraries require them.
- Own the generation loop, KV cache, request lifecycle, and HTTP API. Move ownership in the stages below.
- Start with PyTorch attention and a preallocated contiguous KV cache. Prefill computes the prompt once;
  subsequent steps process one new token.
- Defer FlashInfer until paged cache or batching makes it useful. Its independent attention kernels fit that
  later stage. See the [FlashInfer KV-cache documentation](https://docs.flashinfer.ai/tutorials/kv_layout.html).
- Keep the current vLLM integration as the production path and comparison baseline.

## Migration and dispatch ownership

### Milestone 1: Rust API server, Python execution worker

Emmy starts the Rust executable; the server starts and supervises one local Python execution worker for the GPU.
The worker reuses Emmy's device runner and PyTorch attention. It loads the checkpoint once, tokenizes and applies
chat templates, allocates the cache, runs the complete generation loop, samples, and decodes output text.

Rust owns HTTP validation, admission, request IDs, streaming, cancellation, and worker health. Send each complete
generation request over a Unix-domain socket using length-prefixed JSON messages. The worker returns ready,
incremental output, completion, or failure messages. Cancellation is a separate message keyed by request ID.
This connection carries text and control data only: no weights, activations, GPU pointers, or per-layer RPCs.

The worker's I/O loop remains responsive while a dedicated execution thread owns all GPU operations. It checks
cancellation between generation steps. Bounded output queues prevent slow clients from causing unlimited buffering;
an output consumer that disconnects cancels its request. Rust retains the active-request slot until the worker
acknowledges completion or cancellation, so it cannot admit work while shared GPU buffers are still in use.
Worker failure fails the active request and makes the server unready; it must not silently replay the request.
Server shutdown terminates and reaps its worker.

| Responsibility | Milestone 1 | Milestone 2 |
| --- | --- | --- |
| Compilation, tuning, launch | Python Emmy | Python Emmy |
| HTTP, streaming, admission | Rust server | Rust server |
| Tokenization and text decoding | Python worker | Rust CPU worker |
| Generation steps and cache ownership | Python execution thread | Rust GPU execution thread |
| Enqueue GPU kernels and transfers | Emmy/CuPy and PyTorch | Rust through the CUDA Driver API |
| Execute kernels and copies | CUDA driver and GPU | CUDA driver and GPU |

The compiler determines the ordered operations and buffer requirements. The runtime selects the next request/step,
binds pointers and current shapes, and submits that work. CUDA enforces stream dependencies; GPU hardware schedules
thread blocks. Tokio schedules CPU I/O tasks, not GPU kernels. There is no automatic CPU/GPU placement policy:
text/control work is CPU work, and model computation is GPU work.

### Milestone 2: Rust inference runtime

Replace the Python worker with a dedicated Rust execution thread in the server process. Keep blocking GPU waits
and tokenization off Tokio's I/O threads. Use bounded channels between HTTP tasks, CPU text processing, and GPU
execution. One execution thread owns the CUDA context, allocations, streams, and request cache state for the GPU.

Extend Emmy's existing execution-plan and compiled-binary export into a versioned, self-contained serving artifact.
It must include weights, buffer layouts and lifetimes, kernel symbols and arguments, launch order, dynamic-shape
bindings, supported GPU target, and the complete model step. Existing packs are a starting point, not a complete
Python-independent serving artifact: rotary embedding, attention, final normalization, the output head, and
sampling must also have executable implementations and explicit bindings. Reject unsupported artifacts at startup.

Rust loads CUDA binaries and weights, binds the plan, and submits kernels through the CUDA Driver API. Reuse the
compiler's plan instead of creating a second optimizer in Rust. CUDA-library operations use narrow native bindings;
no Python callbacks remain in the model step. Use Rust tokenizers with the same tokenizer files and implement the
qualified Qwen3 chat-template behavior with parity tests before removing the Python text path.

Add GPU sampling and retain the selected token on-device for the next decode step. Whole-step CUDA graph replay
follows once the uncaptured runtime is correct and buffer addresses are stable. Resolve changing context lengths,
launch dimensions, and kernel arguments explicitly; never replay a graph with stale shape metadata.

### Milestone 3: optimize measured costs

Add continuous batching, paged KV cache, and scheduling coordinated with compiled shapes as separate measured
changes. Evaluate independent attention libraries such as FlashInfer through native bindings when needed.
A faster HTTP implementation alone is not an inference speedup. Measure CPU dispatch, transfer, and GPU execution
costs separately before attributing a performance change to Rust.

## CPU/GPU transfers and synchronization

Use explicit device allocations and copies. Do not introduce unified-memory migration, CPU weight offload, or
GPU memory sharing between processes for the first milestones.

| Data | Transfer policy |
| --- | --- |
| Weights | Read on CPU, upload once at startup, retain on GPU; reuse tied storage. |
| Prompt token IDs | Tokenize on CPU, upload once per request. |
| Positions, lengths, sampling parameters | Small metadata updates; derive positions on GPU where practical. |
| Activations, attention intermediates, KV cache | Remain on GPU throughout generation. |
| Milestone 1 sampling | Copy final logits to CPU for the existing sampler; upload the selected token. |
| Milestone 2 sampling | Sample on GPU; copy only selected token IDs and required status to CPU for streaming. |

The full-logit copy and CPU sampling in milestone 1 are a documented temporary cost. They impose one CPU-dependent
round trip per token. GPU sampling in milestone 2 removes that dependency from choosing the next token, although
the CPU still observes tokens for text decoding, stop strings, and scheduling. Do not claim fully GPU-driven decode.

Use reusable pinned host staging buffers for asynchronous transfers and retain each buffer until its completion
event fires. Stream dependencies ensure an upload completes before a kernel reads it, and a download completes
before the CPU reads its result. Start with one explicit execution stream: copies and kernels run in order.
In milestone 1, bind CuPy to the same CUDA stream as PyTorch and use DLPack for device tensor views; DLPack does
not copy through CPU memory, and shared buffers must remain alive until all consumers finish.

Synchronize at actual CPU-consumption boundaries, not after every kernel or layer. Use event/stream completion
rather than device-wide synchronization. Cancellation stops future submissions; it does not preempt an in-flight
kernel. Wait for submitted work before reusing request buffers or releasing cache state.

Add a separate transfer stream only when profiling shows useful independent work to overlap. Connect streams with
CUDA events. Autoregressive dependencies still hold: a decode step needs the previously selected token, and
asynchronous copies alone do not eliminate that dependency. Pinned buffers permit asynchronous host transfers;
overlap depends on hardware and available independent work. See
[CUDA asynchronous execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html).

## First-milestone implementation

### Command and dependencies

- Make `--native` require `--generate`; reject its combination with `--stock`.
- Support host, port, model revision, context limit, golden evidence, strict evidence, dry-run, and benchmark options.
- Reject unsupported native arguments explicitly. Preserve existing vLLM argument forwarding.
- Package a platform-specific Rust server executable, resolved alongside the Emmy installation. During development,
  build it with Cargo; at distribution time, bundle the matching binary rather than building it at server startup.
  Milestone 1 also uses the existing Python compiler/GPU dependencies. Native startup must not import or invoke vLLM.
- Keep the existing benchmark client optional: native serving needs no vLLM, but the existing benchmark command
  may use it.

### Model execution

- Load the model once. Reuse its tokenizer, rotary embedding implementation, output head, and generation
  configuration alongside Emmy's compiled transformer computation.
- Keep intermediate tensors and KV cache on the GPU. Initially reuse existing sampling behavior; copying the
  final logits for sampling is acceptable for this correctness milestone.
- Use causal attention during prefill. During single-token decode, attend to the entire populated cache,
  with absolute token positions.
- Support dense Qwen3 with standard full attention. Reject MoE, quantized checkpoints, sliding attention,
  and unsupported rotary configurations before compilation.
- Default to a 4,096-token total context, bounded by model and compiler capacity. Reset cache state between requests.

### API and lifecycle

- Provide health, model listing, completions, and chat completions endpoints with an explicitly documented
  OpenAI-compatible subset.
- Support text prompts/messages, streaming and non-streaming responses, output limits, temperature, top-p,
  seed, and stop strings. Return token usage and finish reasons.
- Apply the checkpoint's chat template; default Qwen3 thinking off for this experimental endpoint.
- Run one active generation request. Return a clear busy response for additional requests rather than
  adding a scheduler.
- Keep HTTP handling responsive through the execution ownership described above. Cancel generation on disconnect
  and release request state after submitted GPU work has completed.
- Reject unsupported request features instead of silently ignoring them.

## Validation and rollout

- Test argument routing and startup without vLLM.
- Test API responses, streaming boundaries, Unicode, stop strings, context limits, busy responses,
  cancellation, and cache reset.
- Test socket framing, bounded output, worker death, shutdown, and cancellation acknowledgment before readmission.
- Test stream ordering, pinned-buffer lifetime, and repeated requests for stale cache or reused-buffer reads.
  Verify that weights upload once and intermediate model tensors do not pass through host memory.
- Compare tiny Qwen3 prefill and cached-decode logits against eager execution across several decode steps.
  Verify decode receives only one token.
- Qualify Qwen3-0.6B on an available CUDA GPU. Compare deterministic outputs and record first-token latency,
  token latency, and memory against the existing vLLM path.
- Use the existing benchmark machinery; add no benchmark script. GPU provisioning is outside this plan.
- For milestone 2, compare exported-plan execution with the Python runner, test artifact/GPU incompatibility,
  and verify startup and generation without a Python worker. Test GPU sampling and tokenization/template parity.
- Profile transfer bytes, synchronization, CPU dispatch time, and GPU time. Keep milestone 1's full-logit transfers
  visible; require milestone 2 to return only tokens/status in normal decode. No speedup is an acceptance assumption.
- Follow repository development and finalization gates, including the full suite, lint, documentation review,
  and model-golden decoding if compiler code changes.
- Add Rust unit/integration tests and Cargo formatting/lint gates when Rust implementation lands.
- Keep native serving opt-in. Defer deployment images, additional model families, and production feature parity.
  Continuous batching and paged cache belong to milestone 3; whole-step CUDA graphs belong to milestone 2.

This PR records the plan only. Implementation and GPU validation belong to a subsequent PR. Delete this plan
once its implementation has landed and its durable conclusions are recorded in the serving documentation.
