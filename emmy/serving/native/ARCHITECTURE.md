# Native cached generation

Python prepares a standalone generation artifact for a dense Qwen3 or Qwen3.5 text model with FP16 or FP8 weights,
projection inputs, a KV cache for its full-attention layers and carried state for its Gated DeltaNet layers.
The output head retains FP32 logits through sampling so FP16 rounding cannot create a false maximum tie.
The Rust runtime submits the exported launches and retains the KV cache. Compiled GPU reductions select greedy
tokens; positive-temperature sampling runs on the CPU. No Python model operation runs after preparation. The
experimental native HTTP adapter serves one active request; this is not a performance replacement for vLLM. The existing serving integration remains the default.

## Preparation

`prepare.export_model` traces the existing attention-split wrappers and final normalization/output head. It uses the
compiler's plan-template cache to reuse identical layer structure. The compiled plans are joined into one ordinary
static plan for each width: seams refer to the same named allocation, and internal names are scoped by layer.
There is no new compiler or runtime alias format. Unsupported symbolic, indirect, and descriptor arguments are rejected.
Cache buffers use the persistent output role, without joining the public logits/token output list. Custom launches
identify their writes so Python scratch allocation preserves the same dependencies as native execution.

The glue between the projections is compiled too, from three small traced modules in `prepare.py`: the embedding
gathers the prompt's token at this position while the prompt lasts and the previous step's selection after it, the
rotary module rotates q and k at this position and hands k and v to the cache pages, and attention runs causal
grouped-query attention over the whole cache with every position past this one masked. All three read the position
from device memory, including the cache write, whose paged start is the `position` input rather than a host symbol, so
a token step is one launch sequence at every position and replays as one graph. Nothing the device runs is
hand-written: the step ends at the FP32 logits and the greedy token, a compiled reduction over them, and the runtime
samples on the host only at positive temperature (below), so the same export serves any device the compiler targets.
Attention keeps dot products, scores, probabilities, and value accumulation in FP32, rounding only its output to FP16.
This avoids losing near-tied scores at large magnitudes. Residual sums stay in FP32 through the existing
attention-split wrappers; normalization casts back to FP16 before each projection. Rotary constants come from the
checkpoint's own module in FP32. Rotation also uses FP32 intermediates and rounds only the query/key outputs to FP16.
The existing standalone exporter bundles all binaries and weight bytes. Generation metadata lives in the pack key and
has its own version.

A Qwen3.5 text decoder is found with `find_text_decoder`, since those checkpoints wrap it beside a vision tower native
never loads. Its full-attention layers add two things to the Qwen3 path: the rotary module rotates only the leading
part of each head (the family's partial rotary embedding), and the attention split carries the fused output gate as a
fourth `pre` output and third `post` input. Its Gated DeltaNet layers run whole, through the GDN state wrapper in the
same FP32-residual form the attention split has, so both layer types meet at one seam dtype. Such a layer reads its
FP32 recurrent state and FP16 convolution history from two paged buffers and writes the next state and history to two
others — a kernel's output never shares memory with its inputs — each paged one page per batch row, with no start.
The pack key lists these `(read, write)` pairs under `carried`; the runtime owns their pages (below).

A quantized checkpoint loads through the serving runner's checkpoint-sourced lane (`load_model`). Its twin is built
from the config with the trunk's coded linears left as placeholders; each program's trace is re-addressed to the
checkpoint's keys, the loader's spellers put the decode algebra in the graph, and the constants bind from the shards.
An FP8 trunk therefore stays at its stored size, weight-only under FP16 activations, and nothing past the loader
knows the format. Transformers' quantizer never runs: a module that still carries its checkpoint's quantization
declaration is rejected.

Preparation rejects other model families, sliding attention, non-default rotary schemes, training mode, and
non-FP16 or non-CPU parameters. Context capacity must fit both the model and the current 4,096-token limit.
Compiler evidence uses the existing golden and strict-evidence controls. A successfully exported artifact has not,
by itself, established numerical correctness or fast schedules.

## Execution and state

The prompt is uploaded once. Prefill processes all but its final token in fixed-width chunks, defaulting to 16 rows.
Each layer writes the chunk's keys and values at their absolute positions through the page tables before attention
reads each query's causal prefix; a row past the prompt's last token embeds the previous selection, computes alongside
the others and writes cache rows that decode overwrites when it reaches them, so a chunk is dispatched only where all
of its rows fit the context and the decode program covers the rest. The last layer only writes its cache; its
attention and post-attention fragment are unnecessary. The final prompt token runs through the one-token decode
program, including the output head, which yields the first generated token. Later decode steps embed the previous
step's token, which the runtime read back from the program or sampled from its logits, and uploaded with the position
scalar. Prefill chunks execute no head and download nothing.

The cache is one paged K and one paged V buffer per layer, shaped `[1, kv_heads, context, head_dim]` and paged along
the token axis with `page_tokens` tokens per page (`export_model(page_tokens=…)`, `emmy generate --page-tokens`).
The rotary program writes a chunk of each through the page tables at `position`, the attention program reads all of
them through the same tables, and the runtime allocates every page at load and binds the tables (see the runtime's
paged-buffer contract). The default page spans the whole context, so the table has one entry and the addressing is
that of the contiguous array it replaces; a smaller page changes only the addressing, never the tokens generated. A
new request resets the position and prompt length. Attention can only read positions already overwritten by that
request, so clearing the entire cache is unnecessary. Decode owns the shared inputs, cache pages and constants.
Prefill borrows the same constants through the runtime's region interface and the same cache through its page
tables, and retains a separate scratch slab; scratch is packed by liveness within each program. The borrower drops
before the owner. Loading allocates and uploads both programs before replacing duplicate regions and pages with
borrowed ones, so peak load memory exceeds resident memory. The embedding and tied output-head copies within decode
remain separate.

Carried state cannot be updated in place: the update of one state cell reads a whole column of the old state, and
which kernels exist depends on the cut the evidence picks. So the runtime keeps two pages per carried pair and the
step reads one and writes the other. Every carried table sits in one device allocation whose addresses never change,
and after each prefill chunk or decode step the runtime rewrites its contents — one small upload beside the position
and token uploads — so the read table names the page just written and the write table the other. Captured graphs
baked the table addresses, not their contents, so they replay. A new request points every read table at one zeroed
page per page size that nothing ever writes; no state is cleared. Prefill and decode share the tables as they share
the cache, so decode reads what prefill last wrote.

A prefill chunk may hold rows past the prompt's end, which decode repairs by overwriting their cache rows. Carried
state has no such repair — those rows would be folded into the request — so an artifact that declares carried state
takes a chunk only where every row is a prompt token before the last, and the one-token program covers the rest. The
prefill shortcut that skips the last layer's attention and post-attention fragment applies only when that layer is an
attention layer; a Gated DeltaNet layer always runs whole, since its state must advance.

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

Generation artifact version 6 ends the decode step at the FP32 logits and the greedy token, adds the prefill width,
with a separate prefill program when that width exceeds one, and lists the carried `(read, write)` buffer pairs,
empty for dense Qwen3. The decode program's inputs are the prompt, its
length, the position and the previous step's token; its outputs are the logits and the lowest token ID among their
maxima, which a compiled reduction selects on the device. At temperature zero the runtime downloads that token alone;
at positive temperature it downloads the logits — one vocabulary-sized FP32 transfer per generated token, none during
prefill — and samples on the host. The head keeps its scores in FP32, so FP16 rounding cannot create a false maximum
tie. Older generation artifacts must be exported again; the underlying execution-plan format is unchanged.
Temperature must be finite and nonnegative, and top-p must lie in `(0, 1]`. Nonfinite logits fail the request. Top-k
is unsupported.

Greedy selection is a traced module the compiler lowers with the reductions it has: the peak of the logits, then the
largest negated token ID among the tokens at the peak, so ties resolve to the lowest ID. A nonfinite logit leaves no
token at the peak, the result is the vocabulary size, and the runtime rejects it. The output head multiplies FP16
normalized activations and FP16 weights with FP32 output, preserving score ordering without a wider weight copy.

For positive temperature, the host computes float64 exponential weights, orders the tokens by descending FP32 logit
and ascending token ID (signed zeros compare equal), and retains the smallest prefix of that order reaching top-p. It
then samples that set in token-ID order with the seeded draw. This is a sort of the vocabulary per generated token on
one core; it is not a sampling performance claim.

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
child's PATH after export. The same exported binaries also run through the Python API; logits must be bit-identical
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
The [output-precision investigation](../../../experiments/Qwen3-0.6B/native_accuracy/RESULTS.md) records the FP16
head-output tie and its FP32 repair, with the same checkpoint error limits.
The output-precision artifact passed nineteen sequential and nineteen chunked checkpoint cases, plus the HTTP
lifecycle check. Those results precede the compiled embedding, rotary, attention, greedy reduction, and paged cache
introduced in PR #871; they do not qualify the combined implementation. The
[paged-cache report](../../../experiments/Qwen3-0.6B/paged_cache/RESULTS.md) records earlier V100 smoke results and
stale schedule coverage, not a full-checkpoint qualification of the merged path.

The serving foundation is implemented: standalone execution, cached generation, request reset, CUDA graph replay,
and the [native HTTP adapter](../../../crates/emmy-server/ARCHITECTURE.md). Its scope remains one active dense Qwen3
request on one GPU. Paging currently allocates the full context at load; per-request allocation and reclamation,
continuous batching, and prefix reuse are not implemented. The
[manual-schedule comparison](../../../experiments/Qwen3-0.6B/native_manual_schedules/RESULTS.md),
[greedy comparison](../../../experiments/Qwen3-0.6B/native_greedy/RESULTS.md), and
[prefill comparison](../../../experiments/Qwen3-0.6B/native_prefill/RESULTS.md) retain their revision-specific evidence.
Performance and production concurrency are separate qualifications; these reports establish no general native
serving advantage over stock vLLM.

The first FP8 export, Qwen3.8-27B-FP8 on one H100 (context 256, prefill 16, `EMMY_POOL_DRAW=2048`), produced a 53 GB
pack whose trunk is e4m3 codes, replayed launch by launch and answered greedy prompts coherently through the native
worker. What it established ends there; the gaps, in the order they should close:

- **No parity against the Transformers reference.** The checkpoint qualification above needs a 4,096-token artifact
  and a checkpoint `from_pretrained` loads in FP16, neither of which an FP8 checkpoint gives; the native logits of the
  27B export were compared with nothing but their own replay.
- **No speed number.** `emmy generate --native-pack` prints no per-step time, and the pack load (53 GB) dominates a
  wall clock; a timing flag is the missing piece, not a script.
- **The native HTTP server has not served it.** `emmy serve --native --native-pack DIR` reads the tokenizer, the
  chat template and `serving.json` that `launch.prepare` bundles, which `generate --export-native` does not write.
- **The Gated DeltaNet gate lowers through the generic path.** Its fold multiplies `W[k, h]` projections and a
  per-head `X[h, d, k]` by one row; those channels read different B spaces, so it is no contraction and takes no
  tensor-core tile. A per-channel orientation would give it one back.
- **Compiling is slow and mostly unmeasured.** The decode GDN kernel alone takes ~20 minutes at the default draw
  (single-threaded pricing of the drawn rows), the whole export ~80 minutes, and the H100 hardware golden holds 24
  rows, so nearly every pick is the prior's. A record run on the card is what turns those picks into evidence.

## Native HTTP launcher

`emmy serve MODEL --runner generate --native` prepares the artifact in a fresh temporary directory and executes a
prebuilt `emmy-server`. Preparation uses the checkpoint's weights in FP16, or coded for a quantized checkpoint, at the
requested revision, with the existing golden and strict compiler-evidence controls. `--native-pack DIR` reuses an
already prepared serving bundle; its recorded model, revision, and context must match. Preparation-only evidence flags
are rejected when reusing a bundle.

Native options are `--host`, `--port`, `--revision`, `--max-model-len`, `--page-tokens`, and `--native-pack`, plus the
existing Emmy preparation, dry-run, and benchmark controls. Context defaults to 4,096; the page size defaults to one
page spanning it and, like the compiler evidence flags, applies to preparation, not to a reused pack. `--native`
requires `--runner generate`, rejects `--stock`, and rejects unsupported engine arguments. vLLM forwarding stays
unchanged without `--native`. Dry-run prints preparation settings and the native command without downloading,
compiling, or starting a process.

`--bench` uses the existing vLLM benchmark client, with default native concurrency one. Higher explicit concurrency
measures overload and receives busy responses. The client is an optional dependency; normal native serving does not
import vLLM. Install the matching binaries from `make native-dist` before launching. See the
[HTTP adapter contract](../../../crates/emmy-server/ARCHITECTURE.md) for API fields, cancellation, readiness, and tests.
