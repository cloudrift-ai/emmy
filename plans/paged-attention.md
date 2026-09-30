# Native serving optimization and paged attention

Status: open, revised 2026-09-29 against main `4781e1383`. The native serving foundation is complete in its agreed
scope: a shared Rust executor, cached dense Qwen3 generation, and a single-request text/HTTP adapter. This plan owns
its remaining qualification, measurement, and optimization work. Closing the foundation does not claim production
readiness or a speed advantage over vLLM.

## Current implementation and evidence

PR #954 preserves FP32 output logits with FP16 weights and KV storage. Its artifact passed nineteen sequential and
nineteen chunked checkpoint cases plus HTTP lifecycle checks. PR #871 subsequently replaced the hand-written glue
with compiled embedding, rotary, attention, and greedy selection, and introduced paged cache addressing. These
changes require a new full-checkpoint qualification; the earlier accuracy and performance results do not transfer.

Greedy decoding downloads one GPU-selected token. Positive-temperature sampling downloads FP32 logits and sorts
and samples them on the CPU. Both preserve FP32 score ordering. Generation format 5 requires re-exporting artifacts.
The prompt uses fixed-width prefill chunks, normally sixteen rows, with decode and prefill sharing cache pages.
Pages span the entire configured context at load. There is no per-request pool, continuous batching, or prefix reuse.
The native server still admits one active request and rejects contention with a busy response.

Durable contracts and prior evidence:

- [Native preparation and qualification](../emmy/serving/native/ARCHITECTURE.md),
  [runtime](../crates/emmy-runtime/ARCHITECTURE.md), and [HTTP adapter](../crates/emmy-server/ARCHITECTURE.md).
- [Output precision](../experiments/Qwen3-0.6B/native_accuracy/RESULTS.md) and
  [chunked prefill](../experiments/Qwen3-0.6B/native_prefill/RESULTS.md), measured before the compiled paging changes.
- [Paged-cache investigation](../experiments/Qwen3-0.6B/paged_cache/RESULTS.md): V100 smoke observations from earlier
  modules, with stale one-row schedules and no measured sixteen-row inventory for the merged modules.
- [Serving baseline](../experiments/Qwen3-0.6B/native_baseline/RESULTS.md) and
  [runtime comparison](../experiments/Qwen3-0.6B/native_runtime/RESULTS.md): historical evidence, not current timings.

## Qualification and measurement first

### 0. Qualify the merged implementation on the local RTX 4080

Record manually selected schedules for every current decode and prefill fragment, including embedding, rotary,
attention, the FP32 head, and greedy selection. Keep the prior and MCTS out of schedule selection. Use a fresh tuning
DB, explicit golden scope, strict numerical checks, and strict evidence for the resulting export. Stored targets
must match fresh lowering, and every recorded row must decode. Missing measurements are work to complete, not a
reason to fall back to the prior.

Run all nineteen sequential and nineteen chunked checkpoint cases, including long prompts, with the existing FP32
reference, FP16 reference, error limits, and token-agreement rules unchanged. Check request reset, page boundaries,
full and partial prefill chunks, graph replay, seeded sampling, and the checkpoint HTTP lifecycle. Use the merged
path's tiny-model and independent sampling checks as well. Gate: a reproducible current artifact and recorded passes;
fix failures before making new performance claims.

### 1. Establish the cost of paging and sampling

Compare the qualified merged path with the #954 artifact at temperatures 0 and 0.7, using matching runtime binaries,
the same checkpoint revision, context, prompt/output lengths, graph mode, precision, and warmup. Repeat measurements
and retain their spread. Whole-serving differences include changed model kernels as well as sampling; isolate
sampling and transfers before attributing a regression or improvement to CPU placement.

Measure paged versus flat compiled attention at identical schedules, page sizes 16, 64, and 256, and query lengths
1 and 256. Start on the 4080; repeat on the 5090 when available. Record correctness and latency and add appropriate
paged realization cases so the existing kernel benchmark tracks them. Separate this comparison from whole-model
serving, which changes more than addressing.

Profile GPU execution, exposed CPU gaps, sampling, metadata, transfers, and synchronization. Measure useful/padded
rows and weights, activation, scratch, KV, resident, and peak-load bytes separately. Compare short and long prompts.
CUDA graph replay already exists; do not count removing Python submission as a new benefit for captured steps.
Gate: retained raw results and an experiment report identifying which costs justify implementation.

## Implementation milestones

2. **Page lifetime.** Add a pool of equal pages and per-request page ownership. Allocate as the sequence grows and
   reclaim after EOS, output limits, cancellation, or failure once submitted work completes. Gate: one thousand
   requests with bounded device memory, safe exhaustion, and no reuse while work still references a page.
3. **Per-tile lookup.** When an aligned KV tile fits inside a page, resolve its base once for register or
   `cp.async` staging. Target decode attention over 64-token pages within 5% of flat at the same schedule on the
   4080. Report the measured result even if the target is missed. Do this only if milestone 1 justifies the work.
4. **Batched decoding.** Add a request axis to the block table and compiled fragments. Each active request has its
   own table row, length, position, and sampling state. Admit requests at step boundaries within page capacity.
   Gate: per-request logits satisfy the single-request accuracy contract, seeded replay survives batch changes,
   and concurrency-eight throughput is measured against concurrency one. Test mixed lengths, fairness, overload,
   disconnects, cancellation, backpressure, and failure recovery; retain admission until GPU work completes.
5. **Prefill widths.** Measure larger or symbolic chunk widths against sixteen-row chunks, including partial chunks
   and useful/padded work. Aim for time to first token below one second for a 1,024-token prompt on the 4080, while
   preserving decode latency under contention. Fresh schedules for existing sixteen-row chunks belong to milestone
   0, not this optimization. Width work can proceed before batching when the measurements justify it.
6. **Prefix reuse.** Add reference counts, a hash chain over complete token pages, and eviction of unreferenced
   cached pages. Shared pages must survive the allocating request and remain immutable while shared. Gate: repeated
   prefixes skip their prefill, with measured time to first token and concurrent cancellation/reuse checks.
7. **Tensor-core attention over pages.** Extend `mma` staging with page-aware `cp.async` rows. TMA stays explicitly
   unsupported until its descriptor design is separately justified. Target paged attention within 10% of flat on an
   H100 at 4,096 keys, with strict correctness and recorded schedules on the exact card.
8. **Serving comparison.** Compare with stock vLLM on the same GPU at input lengths 32/256/1024 and concurrency
   1/8/32 once batching exists. Hold output work and supported semantics fixed. Publish latency, throughput, memory,
   errors, fairness, and regressions; no advantage is presumed.

Milestones 0 and 1 come first. Page lifetime precedes batching; per-tile lookup is not a correctness prerequisite
for batching. Prefix reuse needs page ownership and the batched lifecycle. Tensor-core staging needs the page-aware
tile design. Implement each measured improvement as a separate, qualified change.

## Design constraints and open decisions

Paging stays a buffer property. A batched cache can resolve a pointer from a table row and token page, then the
within-page offset. Per-request lengths and write positions are device inputs, not host symbols specialized per
request. Stable table addresses and updated contents permit graph replay. Reuse the existing memory protocol,
execution-plan contract, compiler evidence, and Rust executor; do not build a second dispatch path.

Fusion remains maximal. Slow fused attention is a cut or lowering problem. Page alignment constrains legal staging,
not fusion. Explicitly reject unsupported memory accesses rather than emitting incorrect code.

Choose page sizes, batch widths, prefill budgets, and admission limits from the measurements. Start with existing
compiled widths where they fit. Do not add speculative abstractions or permanent CPU/GPU sampling alternatives.
Keep the current FP32 sampling contract. If CPU sampling is a material cost, evaluate compiler-generated GPU
sampling against the current CPU implementation. A histogram is one possible algorithm, not a prerequisite; the
#954 sampler used reductions. Do not assume CPU sampling is cheap at eight requests or that GPU placement wins.

The shared Rust dispatch migration landed in #885. Do not reimplement the old standalone-worker tune integration.
If runtime overhead is material, audit current run/tune/benchmark consumers and measure serialization, cold startup,
warm loading, allocation, submission, and GPU time separately, with equivalent persistent worker lifetimes. Preserve
per-kernel diagnostics, deadlines, process retirement, and visible failures. The old report's uncaptured submission
saving is not a full-model speedup.

Use existing experiment recipes and benchmark commands. Add a reusable missing measurement control to the harness
when necessary; never write a separate benchmark script. Pin software, hardware, model, artifact, schedules,
precision, warmup, and timing rules. Retain compressed raw results and system-only records; interpret them in the
experiment report. Remote hardware and longer runs must remain within the user's authorized scope.

Keep vLLM as the default. Deployment images, additional model families, quantization, multi-GPU serving, and full
production API parity remain deferred. Revisit the scope if measurements show no useful improvement.

## Known limits to check during implementation

- The tracer's `permute` handling can lose a real transpose; use the qualified transpose path until repaired and
  test multi-row inputs, since one-row tests can hide the defect.
- Volta schedules require strict numerical checks; prior miscompilations make smoke text insufficient evidence.
- Paged tensor-core and TMA staging are currently refused. Do not infer their support from flat-kernel qualification.
- Loading prefill currently allocates duplicate pages and constants before borrowing decode storage. Account for
  that transient peak before deciding whether a shared-load change is worthwhile.
- Performance is schedule- and shape-dependent. Sixteen-row prefill and host sampling are hypotheses to measure,
  not established dominant costs on the merged implementation.
