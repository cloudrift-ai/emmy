# Native cached generation

Python prepares a standalone dense Qwen3 token-step artifact with FP16 weights, projections, logits, and KV cache.
The Rust runtime submits the exported launches, retains the KV cache, and chooses each next token on the GPU.
No Python model operation runs after
preparation. This is a single-request correctness implementation, not an HTTP server or a performance replacement for
vLLM. The existing serving integration remains the default.

## Preparation

`prepare.export_model` traces the existing attention-split wrappers and final normalization/output head. It uses the
compiler's plan-template cache to reuse identical layer structure. The compiled plans are joined into one ordinary
static execution plan: seams refer to the same named allocation, and internal names are scoped by layer. There is no
new compiler or runtime alias format. Unsupported symbolic, indirect, and descriptor arguments are rejected.
Cache buffers use the persistent output role, without joining the public logits/token output list. Custom launches
identify their writes so Python scratch allocation preserves the same dependencies as native execution.

CUDA source lives in the packaged `kernels.cu` resource, loaded by Python during artifact preparation.
Small CUDA kernels provide embedding lookup, default full rotary embedding, contiguous cache writes, causal grouped
query attention, and GPU sampling. Attention keeps dot products, scores, probabilities, and value accumulation in
FP32, rounding only its output to FP16. This avoids losing near-tied scores at large magnitudes. These kernels favor
accuracy over speed. Residual sums stay in FP32 through the existing attention-split wrappers; normalization casts
back to FP16 before each projection. Rotary constants come from the checkpoint's own module in FP32. Rotation also
uses FP32 intermediates and rounds only the query/key outputs to FP16. The existing standalone exporter bundles all
binaries and weight bytes. Generation metadata lives in the pack key and has its own version.

Preparation rejects other model families, quantization, sliding attention, non-default rotary schemes, training mode,
and non-FP16 or non-CPU parameters. Context capacity must fit both the model and the current 4,096-token limit.
Compiler evidence uses the existing golden and strict-evidence controls. A successfully exported artifact has not,
by itself, established numerical correctness or fast schedules.

## Execution and state

One token step contains every model layer. The prompt is uploaded once. Prefill processes its tokens sequentially,
writing each token's keys and values once at its absolute position; it never recomputes the growing prefix. Decode
reads the previous GPU-selected token. The host updates one position scalar and reads one selected token after the
prompt is consumed. Full logits are downloaded only through the explicit diagnostic operation.

The cache has one preallocated contiguous K and V array per layer. A new request resets the position and prompt
length. Attention can only read positions already overwritten by that request, so clearing the entire cache is
unnecessary. Activations and scratch remain allocated for the model lifetime; this version does not reuse storage
between layers or deduplicate the embedding and tied output-head weight copies.

Graph capture records one step without executing a warmup. Replaying the graph advances the model exactly once,
including when capture is first enabled during decode. All addresses remain stable across positions and requests.
Each step synchronizes at the CPU observation boundary. EOS or the output budget stops further submissions. A
request whose prompt plus output budget exceeds capacity is rejected. Greedy decoding is the default; requests may
select temperature, top-p, and an unsigned 64-bit seed.

The existing supervised native worker supplies hard deadlines and process retirement. Each worker operation has a
120-second default deadline; `generate --timeout SECONDS` can extend it for long sequential prefills. It never retries
a failed request. `client.generate_tokens` sends binary token files to that worker; the complete generation loop
runs in Rust.
The low-level start/step operations expose logits for parity checks and do not change the ordinary generation path.

## Sampling contract

Generation artifact version 2 adds a float64 temperature/top-p input and a uint64 seed input. Older generation
artifacts must be exported again; the underlying execution-plan format is unchanged. Temperature must be finite and
nonnegative, and top-p must lie in `(0, 1]`. Temperature zero selects the lowest token ID among maximum logits.
Nonfinite logits fail the request. Top-k is unsupported.

For positive temperature, a 65,536-bin histogram orders FP16 logits exactly, combining signed zeros. The sampler
retains the smallest descending probability prefix reaching top-p, breaking ties by ascending token ID. It samples
that distribution in token-ID order using float64 probabilities. The histogram costs 256 KiB per loaded model and is
cleared before every step; no vocabulary-sized buffer crosses to the CPU. This simple implementation has serial
histogram scans and token selection; it is not a sampling performance claim.

A SplitMix64 counter combines the request seed and generated-token index. Prefill does not consume random draws.
Resetting a request resets the counter, and captured and uncaptured execution select the same tokens for identical
logits and controls. Reproducibility does not imply matching NumPy or PyTorch RNG sequences, or identical completions
across different compiled artifacts and hardware.

## Commands and qualification

Build and install the matching worker before using these commands; command startup never invokes Cargo:

```bash
emmy generate Qwen/Qwen3-0.6B --revision REVISION --export-native /tmp/qwen-native --context-length 256
emmy generate Qwen/Qwen3-0.6B --revision REVISION --native-pack /tmp/qwen-native --prompt 'Hello' --max-new-tokens 16
emmy generate Qwen/Qwen3-0.6B --revision REVISION --native-pack /tmp/qwen-native --prompt 'Hello' --capture \
  --temperature 0.7 --top-p 0.9 --seed 42
```

Use the same checkpoint/tokenizer revision for preparation and text generation. The native artifact itself accepts
and returns token IDs; tokenizer packaging and native text processing belong to the later API work.

The hermetic tiny-Qwen3 GPU test checks every logit at `rtol=atol=1e-3`, greedy tokens, request reset, context bounds,
EOS, zero output budget, graph replay, and first capture during decode. It hides Python and NVCC from the native
child's PATH after export. The same exported binaries also run through the Python executor; logits must be bit-identical
at every checked step. Independent NumPy checks cover rotary rounding and causal attention through cache position
4,096, including a shorter request after the largest one.

Opt-in checkpoint qualification accepts a local `--native-checkpoint` and matching `--native-artifact` with capacity
at least 256. FP16 eager and FP32 eager references consume the same prefixes and FP16-rounded weights, with TF32
and reduced-precision reductions disabled. Each native logit vector must stay within 2% relative L2 error and 0.02
total variation from the FP32 distribution. Per-prompt RMS error must be at most twice the FP16 reference RMS, with
one FP16 epsilon as a floor. Native argmax must match one reference; agreement between the two requires an exact
match. Pointwise differences remain recorded. These experimental budgets do not establish bitwise model equivalence
or identical future completions when the references disagree.

The [numerical investigation](../../../experiments/Qwen3-0.6B/native_generation/RESULTS.md) records the fixed rotary
rounding defect, failed exploratory criteria, held-out qualification, and limits. The original artifact passed
through 256 checkpoint positions. The
[follow-up](../../../experiments/Qwen3-0.6B/native_generation/SAMPLING_CONTEXT.md) executes
seventeen cases across 10,585 positions, including two 4,096-position prompts, within the unchanged error budgets.
FP32 attention, rotary intermediates, and residual accumulation close the earlier numerical failures. Independent
attention qualification also covers the full 4,096-position capacity.
Performance and production concurrency are separate qualifications.
