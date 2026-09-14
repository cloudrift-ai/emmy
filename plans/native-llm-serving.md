# Experimental native LLM serving

## Summary

Add `--native` to the existing serving command. Keep vLLM as the default.

The first milestone serves dense Qwen3 models in FP16 on one GPU, with cached generation and a streaming API.
It runs without vLLM installed. Success means correct behavior and ownership of execution; outperforming vLLM
comes later.

## Approach

- Reuse Emmy's compiled model runner and device execution paths.
- Own the generation loop, KV cache, request lifecycle, and HTTP API.
- Start with PyTorch attention and a preallocated contiguous KV cache. Prefill computes the prompt once;
  subsequent steps process one new token.
- Defer FlashInfer until paged cache or batching makes it useful. Its independent attention kernels fit that
  later stage. See the [FlashInfer KV-cache documentation](https://docs.flashinfer.ai/tutorials/kv_layout.html).
- Keep the current vLLM integration as the production path and comparison baseline.

## Implementation

### Command and dependencies

- Make `--native` require `--generate`; reject its combination with `--stock`.
- Support host, port, model revision, context limit, golden evidence, strict evidence, dry-run, and benchmark options.
- Reject unsupported native arguments explicitly. Preserve existing vLLM argument forwarding.
- Add a native-serving extra containing FastAPI and Uvicorn, used alongside the existing compiler dependencies.
  Native startup must not import or invoke vLLM.
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
- Keep HTTP handling responsive through a dedicated execution worker. Cancel generation on disconnect and
  release request state after completion, cancellation, or failure.
- Reject unsupported request features instead of silently ignoring them.

## Validation and rollout

- Test argument routing and startup without vLLM.
- Test API responses, streaming boundaries, Unicode, stop strings, context limits, busy responses,
  cancellation, and cache reset.
- Compare tiny Qwen3 prefill and cached-decode logits against eager execution across several decode steps.
  Verify decode receives only one token.
- Qualify Qwen3-0.6B on an available CUDA GPU. Compare deterministic outputs and record first-token latency,
  token latency, and memory against the existing vLLM path.
- Use the existing benchmark machinery; add no benchmark script. GPU provisioning is outside this plan.
- Follow repository development and finalization gates, including the full suite, lint, documentation review,
  and model-golden decoding if compiler code changes.
- Keep native serving opt-in. Defer deployment images, continuous batching, paged cache, whole-step CUDA graphs,
  additional model families, and production feature parity to later milestones.

This PR records the plan only. Implementation and GPU validation belong to a subsequent PR. Delete this plan
once its implementation has landed and its durable conclusions are recorded in the serving documentation.
