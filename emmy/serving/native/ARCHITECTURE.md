# Native cached generation

Python prepares a standalone dense Qwen3 generation artifact with FP16 weights, projections, logits, and KV cache.
The Rust runtime submits the exported launches, retains the KV cache, and chooses each next token on the GPU.
No Python model operation runs after
preparation. The experimental native HTTP adapter serves one active request; this is not a performance replacement
for vLLM. The existing serving integration remains the default.

## Preparation

`prepare.export_model` traces the existing attention-split wrappers and final normalization/output head. It uses the
compiler's plan-template cache to reuse identical layer structure. The compiled plans are joined into one ordinary
static plan for each width: seams refer to the same named allocation, and internal names are scoped by layer. There is
no
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

The prompt is uploaded once. Prefill processes all but its final token in fixed-width chunks, defaulting to 16 rows.
Each layer writes valid rows' keys and values at their absolute positions before attention reads each query's causal
prefix. The final partial chunk masks unused rows before cache writes or attention reads. Its extra projection work
is still executed. The last layer only writes its cache; its attention and post-attention fragment are unnecessary.
The final prompt token runs through the one-token decode program, including the output head and sampler, to select
the first generated token. Later decode steps read the previous GPU-selected token. Prefill chunks do not execute
the head or sampler and consume no random draws. The host updates one position scalar per submission and reads one
selected token after the prompt is consumed. Full logits are an explicit diagnostic transfer.

The cache has one preallocated contiguous K and V array per layer. A new request resets the position and prompt
length. Attention can only read positions already overwritten by that request, so clearing the entire cache is
unnecessary. Decode owns the shared inputs, cache, and constants. Prefill borrows identical named allocations through
the runtime's existing region interface and retains a separate scratch slab; scratch is packed by liveness within
each program. The borrower drops before the owner. Loading currently allocates and uploads both programs before
replacing duplicate regions with borrowed addresses, so peak load memory exceeds resident memory. The embedding
and tied output-head copies within decode remain separate.

Each program has its own graph capture, recorded without executing a warmup. Replaying the graph advances the model
exactly once,
including when capture is first enabled during decode. All addresses remain stable across positions and requests.
Each step synchronizes at the CPU observation boundary. EOS or the output budget stops further submissions. A
request whose prompt plus output budget exceeds capacity is rejected. Greedy decoding is the default; requests may
select temperature, top-p, and an unsigned 64-bit seed.

The existing supervised native worker supplies hard deadlines and process retirement. Each worker operation has a
120-second default deadline; `generate --timeout SECONDS` can extend it for long requests. It never retries
a failed request. `client.generate_tokens` sends binary token files to that worker; the complete generation loop
runs in Rust.
The diagnostic step remains single-token by default. Its `prefill: true` option follows normal chunk dispatch and
returns the advanced position; intermediate chunks return no token or logits.

## Sampling contract

Generation artifact version 3 adds the prefill width and a separate prefill program when that width exceeds one.
Older generation artifacts must be exported again; the underlying execution-plan format is unchanged. Sampling uses
float64 temperature/top-p inputs and a uint64 seed. Temperature must be finite and
nonnegative, and top-p must lie in `(0, 1]`. Temperature zero selects the lowest token ID among maximum logits.
Nonfinite logits fail the request. Top-k is unsupported.

Greedy selection uses one 128-thread block. Threads scan disjoint vocabulary strides, then reduce their winning
indices in shared memory with explicit lowest-ID tie-breaking. Every thread participates in invalid-logit detection
and the reduction, including vocabularies smaller than the block. Preparation supplies 512 bytes of shared memory.
Intermediate prefill steps leave the selected token untouched. Positive-temperature selection still runs on thread
zero after the uniform greedy branch; its probability and RNG calculations are unchanged. Re-export an artifact to
use the parallel sampler. The runtime protocol version is unchanged.

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

`generate --export-native` and native serving preparation accept `--prefill-size N`; one selects sequential prefill.
The width is capped at the context capacity and stored in the artifact; it cannot change when loading an existing pack.
Use the same checkpoint/tokenizer revision for preparation and text generation. The runtime artifact accepts and returns
token IDs. Native serving preparation additionally bundles the same
checkpoint tokenizer and chat template; the HTTP adapter owns text processing.

The hermetic tiny-Qwen3 GPU test checks every logit at `rtol=atol=1e-3`, greedy tokens, request reset, context bounds,
EOS, zero output budget, graph replay, first capture during decode, and full/partial prefill chunks across resets. It
hides Python and NVCC from the native
child's PATH after export. The same exported binaries also run through the Python executor; logits must be bit-identical
at every checked step. Independent NumPy checks cover rotary rounding and causal attention through cache position
4,096, including a shorter request after the largest one.

Opt-in checkpoint qualification accepts a local `--native-checkpoint` and matching `--native-artifact` with capacity
at least 4,096. Sequential and chunked dispatch are checked separately; chunked dispatch exposes logits only after
prefill completes and during decode. FP16 eager and FP32 eager references consume the same prefixes and FP16-rounded
weights, with TF32
and reduced-precision reductions disabled. Each native logit vector must stay within 2% relative L2 error and 0.02
total variation from the FP32 distribution. Sequential execution keeps the whole-prompt RMS limit: twice the FP16
reference RMS, with one FP16 epsilon as a floor. Chunked execution exposes a shorter output window. Its RMS must
meet that bound or be no worse than sequential native execution on those same positions and teacher-forced prefixes.
The paired baseline is recorded separately; this is an additional regression check, not a claim that a tail-only
RMS is the original whole-prompt metric. Native argmax must match one reference; agreement between the two requires
an exact match. Pointwise differences remain recorded. These experimental budgets do not establish bitwise model
equivalence
or identical future completions when the references disagree.

The [numerical investigation](../../../experiments/Qwen3-0.6B/native_generation/RESULTS.md) records the fixed rotary
rounding defect, failed exploratory criteria, held-out qualification, and limits. The original artifact passed
through 256 checkpoint positions. The
[follow-up](../../../experiments/Qwen3-0.6B/native_generation/SAMPLING_CONTEXT.md) executes
seventeen cases across 10,585 positions, including two 4,096-position prompts, within the unchanged error budgets.
FP32 attention, rotary intermediates, and residual accumulation close the earlier numerical failures. Independent
attention qualification also covers the full 4,096-position capacity.
Performance and production concurrency are separate qualifications.

## Native HTTP launcher

`emmy serve MODEL --generate --native` prepares the artifact in a fresh temporary directory and executes a prebuilt
`emmy-server`. Preparation uses FP16 checkpoint weights, the requested revision, and the existing golden and strict
compiler-evidence controls. `--native-pack DIR` reuses an already prepared serving bundle; its recorded model,
revision, and context must match. Preparation-only evidence flags are rejected when reusing a bundle.

Native options are `--host`, `--port`, `--revision`, `--max-model-len`, and `--native-pack`, plus the existing Emmy
preparation, dry-run, and benchmark controls. Context defaults to 4,096. `--native` requires `--generate`, rejects
`--stock`, and rejects unsupported engine arguments. vLLM forwarding stays unchanged without `--native`. Dry-run
prints preparation settings and the native command without downloading, compiling, or starting a process.

`--bench` uses the existing vLLM benchmark client, with default native concurrency one. Higher explicit concurrency
measures overload and receives busy responses. The client is an optional dependency; normal native serving does not
import vLLM. Install the matching binaries from `make native-dist` before launching. See the
[HTTP adapter contract](../../../crates/emmy-server/ARCHITECTURE.md) for API fields, cancellation, readiness, and tests.
