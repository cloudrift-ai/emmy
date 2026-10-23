# Native cached generation

Python prepares a standalone generation artifact for a dense Qwen3 or Qwen3.5 text model with 16-bit or FP8 weights,
or a Qwen3.5-MoE one with 16-bit or FP8 weights, its routed experts on the device or paged from host memory. The
artifact holds the weights, projection inputs, a KV cache for its full-attention layers and carried state for its Gated
DeltaNet layers. It runs in one model dtype, FP16 or BF16: the checkpoint's, as the vLLM plugin serves it (below).
The output head retains FP32 logits through sampling so 16-bit rounding cannot create a false maximum tie.
The Rust runtime submits the exported launches and retains the KV cache. Compiled GPU reductions select greedy
tokens; positive-temperature sampling runs on the CPU. No Python model operation runs after preparation. The
experimental native HTTP adapter serves one active request; this is not a performance replacement for vLLM. The existing serving integration remains the default.

## Preparation

`prepare.export_model` traces the existing attention-split wrappers and final normalization/output head. It uses the
compiler's plan-template cache to reuse identical layer structure. The compiled plans are joined into one ordinary
static plan for each width: seams refer to the same named allocation, and internal names are scoped by layer. Kernels
join by name; one whose name a different source already holds takes a suffix from its own source, as the paging splice
does. There is no new compiler or runtime alias format. Unsupported symbolic and descriptor arguments are rejected.
Cache buffers use the persistent output role, without joining the public logits/token output list. Custom launches
identify their writes so Python scratch allocation preserves the same dependencies as native execution.

The glue between the projections is compiled too, from three small traced modules in `prepare.py`: the embedding
gathers the prompt's token at this position while the prompt lasts and the previous step's selection after it, the
rotary module rotates q and k at this position and scatters k and v into the cache pages at that position, and attention
runs causal grouped-query attention over the whole cache with every position past this one masked. All three read the
position from device memory, including the cache write, whose rows land at an index computed from the `position` input
rather than a host symbol, so a token step is one launch sequence at every position and replays as one graph. Nothing the device runs is
hand-written: the step ends at the FP32 logits and the greedy token, a compiled reduction over them, and the runtime
samples on the host only at positive temperature (below), so the same export serves any device the compiler targets.
Attention keeps dot products, scores, probabilities, and value accumulation in FP32, rounding only its output to the
model dtype. This avoids losing near-tied scores at large magnitudes. Residual sums stay in FP32 through the existing
attention-split wrappers; normalization casts back to the model dtype before each projection. Rotary constants come
from the checkpoint's own module in FP32. Rotation also uses FP32 intermediates and rounds only the query/key outputs
to the model dtype. The existing standalone exporter bundles all binaries and weight bytes. Generation metadata lives
in the pack key and has its own version.

The model dtype is what `load_model` builds the model in: `--dtype` (`auto` by default), where `auto` takes BF16 for a
checkpoint whose config declares it and FP16 otherwise, the rule vLLM's own `auto` follows, so a BF16 checkpoint runs
the dtype the plugin serves it in and its kernels can be the plugin's. Every 16-bit activation, cache, history and
unquantized weight of the step takes it, a BF16 one stored as its 16-bit pattern; the FP32 seams, the FP32 logits and
the greedy token do not change with it.

A Qwen3.5 text decoder is found with `find_text_decoder`, since those checkpoints wrap it beside a vision tower native
never loads. Its full-attention layers add two things to the Qwen3 path: the rotary module rotates only the leading part
of each head (the family's partial rotary embedding), and the attention split carries the fused output gate as a fourth
`pre` output and third `post` input. Its Gated DeltaNet layers run whole, through the GDN state wrapper in the same
FP32-residual form the attention split has, so both layer types meet at one seam dtype. Such a layer reads its FP32
recurrent state and its convolution history in the model dtype and writes the next ones over them: each output is
spelled as the input with every row replaced (a scatter), which the compiler stores into the input's memory. The pack
key lists these buffers under `state` (below).

A Qwen3.5-MoE decoder has routed experts and a gated shared expert in every layer, behind either mixer. Each layer's
E-stacked expert weights are two buffers of the step (`layer{i}.w_gate_up`, `layer{i}.w_down`). Preparation reads
the vLLM plugin's budget, `EMMY_GEN_EXPERTS_GPU`: how many experts per layer the device holds. With no budget, or
one at or above the expert count, the weights are constants of the step and nothing is paged; below it they are
paged (below). A MoE layer's section of the step is four programs in sequence, every one static:

1. The mixer is the attention split's MoE `post` or the GDN state wrapper's routed form, both in FP32-residual form. It
   writes the FP32 residual with the shared expert folded in (`layer{i}.h`) and the normalized rows the experts read
   (`layer{i}.xn`).
2. The router writes each row's scores (`layer{i}.scores`, `rows × k`) and picks (`layer{i}.sel{r}`, `k` int64), as
   Transformers' softmax top-k router computes them: the gate's linear in the model dtype, the softmax in FP32, then the
   top `k` as `k` masked maximum reductions. Each takes the peak and the lowest expert ID at it, then masks that expert
   below every probability. The picked scores are normalized and rounded to the model dtype. A NaN row has no element at
   its peak and picks the last expert, so the gather below stays inside the stack; its logits come out nonfinite and the
   request fails there, as greedy selection fails any nonfinite step.
3. The fixed-slot program of the vLLM plugin (`ExpertFFN.slots`) runs once per row. It gathers the row's `k` experts
   from the stacked weights by its picks and writes their outputs (`layer{i}.partials{r}`, `k × hidden`). A launch
   reads whole buffers, so a chunk's router also copies each row out (`layer{i}.x{r}`); one row reads
   `layer{i}.xn` itself. A 16-row prefill chunk is 16 launches of the same program.
4. The combine multiplies each row's scores by its partials in the model dtype, as the plugin's fixed-slot combine does,
   adds the result to the FP32 residual and writes the layer's output.

Past its mixer, a MoE layer therefore runs `rows + 2` programs per step; the last prefill layer runs none of them,
since nothing reads its output.

A quantized MoE checkpoint's routed experts never load with its twin (below); they stay on the meta device. The export
reads them from the shards a layer at a time, as the loader stacks them for the plugin (`checkpoint_experts`). An FP8
layer's experts are their e4m3 codes on the uint8 carrier beside their FP32 block scales, so the step holds four expert
inputs per layer (`layer{i}.w_gate_up`, `layer{i}.w_down` and a `_scale` for each). The fixed-slot program spells the
codes as the plugin's does: the decode and the scale multiply are in the graph, and the device holds the codes. Each
program reads a layer's experts as it compiles that layer, and the pack reads them once more as it writes their pages,
so below the budget the host holds one layer's experts at a time. At or above it the experts are constants of
each program, so the host holds them in every program's bytes until the pack is written.

Below the budget, each expert input, codes and scales alike, is an input of the step paged along its expert axis, one
expert per page, with the budget as its device slots, as the plugin declares it. The backend splices each fixed-slot
launch's inspector and the paging programs ahead of it, so they are ordinary launches of the step and a captured graph
records them. The pager's tables, slots and clock enter the step as inputs named after the buffer
(`layer{i}.w_gate_up__table` … `__slots`), and the copy reads the pages through the buffer's `__view`. Every row's
launch in decode and prefill moves pages through the same tables, so the slots are one cache per buffer. The pack
keeps the expert bytes once, page after page, under its `pages` index. The runtime reads each file straight into
pinned, mapped host memory, declares it to the pager with its page count, page size and slots, and lends the
pager's tables to both programs as they load (runtime `ARCHITECTURE.md`, "Cached generation"). The host therefore
holds the experts once and the device holds the slots once; the trunk, the KV cache and the Gated DeltaNet state
stay where they are. The budget is fixed at preparation: a reused pack keeps the slots it was prepared with. Each
launch reading a paged buffer takes its own inspector and paging run, 14 launches per buffer, so a paged layer adds
28 launches per fixed-slot launch with 16-bit experts and 56 with FP8 ones.

The hermetic tiny Qwen3.5-MoE GPU test has two full-attention layers. It checks every logit against Transformers' and
its greedy tokens, through sequential and chunked prefill, capture and reset, at three budgets: every expert, four of
eight, and one (below the two picks, so the second is read in place from host memory). Its experts are wide, about 100
MB in all, so they dominate the worker's memory: after load its heap holds less than half of them, and its peak resident
set stays within half of them of its resident set, so the load neither kept nor passed through a second copy. No test in
the suite runs a Gated DeltaNet MoE layer on a card: without evidence its kernels take minutes to compile, and the same
gate with one GDN and one attention layer takes about ten minutes on an RTX 5090 (it passed by hand on 2026-10-22, with
every expert on the card and with two of eight). Its program is checked on the CPU instead, traced from an FP16 model
and interpreted by the NumPy backend: the FP32 residual, the FP16 normalized rows, the state and the history match the
eager module. An FP8 variant of the attention gate reads a dynamic block-FP8 checkpoint whose experts are stored one
module per expert, as Qwen publishes them; every logit and greedy token matches the dequantized model with every expert
on the card and with four of eight, about three minutes each, and from a BF16 checkpoint with four of eight, whose
logits it matches within one BF16 rounding of the reference's (8e-3).

A quantized checkpoint loads through the serving runner's checkpoint-sourced lane (`load_model`). Its twin is built
from the config with the trunk's coded linears left as placeholders and its routed experts unread; each program's
trace is re-addressed to the checkpoint's keys, the loader's spellers put the decode algebra in the graph, and the
constants bind from the shards. An FP8 trunk therefore stays at its stored size, weight-only under 16-bit activations,
and nothing past the loader knows the format. Transformers' quantizer never runs: a module that still carries its
checkpoint's quantization declaration is rejected.

Preparation rejects other model families, sliding attention, non-default rotary schemes, training mode, and parameters
that are not all FP16 or all BF16, or not on the CPU. Context capacity must fit both the model and the current
4,096-token limit. Compiler evidence uses the existing golden and strict-evidence controls. A successfully exported
artifact has not, by itself, established numerical correctness or fast schedules.

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

The cache is one paged K and one paged V buffer per layer, shaped `[1, kv_heads, context, head_dim]` and paged along the
token axis with `page_tokens` tokens per page (`export_model(page_tokens=…)`, `emmy generate --page-tokens`). The rotary
program writes a chunk of each through the page tables at `position`, the attention program reads all of them through
the same tables, and the runtime allocates every page at load and binds the tables (see the runtime's paged-buffer
contract). The default page spans the whole context, so the table has one entry and the addressing is that of the
contiguous array it replaces; a smaller page changes only the addressing, never the tokens generated. A new request
resets the position and prompt length. Attention can only read positions already overwritten by that request, so
clearing the entire cache is unnecessary. Decode owns the shared inputs, cache pages and constants. Prefill loads with
the same inputs and constants lent through the runtime's region interface, takes the same cache through its page tables,
and retains a separate scratch slab; scratch is packed by liveness within each program. The borrower drops before the
owner. Only the cache pages are allocated twice at load, until prefill takes decode's tables. The embedding and tied
output-head copies within decode remain separate.

The state is updated in place. The next state reads whole columns of the old one, so the kernel that stores it over
the old state must not also compute it: fusion keeps a kernel stored over a buffer apart from every loop that
computes from that buffer's old cells, and the store becomes a copy of the next state, computed by an earlier kernel
into scratch. A new request zeroes every state buffer. Prefill borrows them as it borrows the other outputs decode
owns, so decode reads what prefill last wrote.

A prefill chunk may hold rows past the prompt's end, which decode repairs by overwriting their cache rows. The
state has no such repair — those rows would be folded into the request — so an artifact that declares state takes a
chunk only where every row is a prompt token before the last, and the one-token program covers the rest. The
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

Generation artifact version 8 ends the decode step at the FP32 logits and the greedy token, adds the prefill width,
with a separate prefill program when that width exceeds one, lists the buffers a step carries, empty for dense
Qwen3, and reads the pages of every buffer the pager holds from the pack. The decode program's inputs are the
prompt, its length, the position and the previous step's token; its outputs are the logits and the lowest token ID
among their maxima, which a compiled reduction selects on the device. At temperature zero the runtime downloads
that token alone; at positive temperature it downloads the logits — one vocabulary-sized FP32 transfer per
generated token, none during prefill — and samples on the host. The head keeps its scores in FP32, so 16-bit
rounding cannot create a false maximum tie. Older generation artifacts must be exported again; the underlying
execution-plan format is unchanged. Temperature must be finite and nonnegative, and top-p must lie in `(0, 1]`.
Nonfinite logits fail the request. Top-k is unsupported.

Greedy selection is a traced module the compiler lowers with the reductions it has: the peak of the logits, then the
largest negated token ID among the tokens at the peak, so ties resolve to the lowest ID. A nonfinite logit leaves no
token at the peak, the result is the vocabulary size, and the runtime rejects it. The output head multiplies 16-bit
normalized activations and weights with FP32 output, preserving score ordering without a wider weight copy.

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

Opt-in checkpoint qualification accepts a local `--native-checkpoint` and matching FP16 `--native-artifact` (`--dtype
float16` for a BF16 checkpoint) with capacity at least 4,096. Sequential and chunked dispatch are checked separately;
chunked dispatch exposes logits only after prefill completes and during decode. FP16 eager and FP32 eager references
consume the same prefixes and FP16-rounded weights, with TF32 and reduced-precision reductions disabled. Each native
logit vector must stay within 2% relative L2 error and 0.02 total variation from the FP32 distribution. Sequential
execution keeps the whole-prompt RMS limit: twice the FP16 reference RMS, with one FP16 epsilon as a floor. Chunked
execution exposes a shorter output window. Its RMS must meet that bound or be no worse than sequential native execution
on those same positions and teacher-forced prefixes. The paired baseline is recorded separately; this is an additional
regression check, not a claim that a tail-only RMS is the original whole-prompt metric. Native argmax must match one
reference; agreement between the two requires an exact match. Pointwise differences remain recorded. These experimental
budgets do not establish bitwise model equivalence or identical future completions when the references disagree.

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
and the [native HTTP adapter](../../../crates/emmy-server/ARCHITECTURE.md). Its scope remains one active request
on one GPU. Paging currently allocates the full context at load; per-request allocation and reclamation,
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
- **Not timed.** `emmy generate --native-pack` logs the time per token, and `--json` keeps each token's time; the
  27B export has not been run with it.
- **The native HTTP server has not served it.** `emmy serve --native --native-pack DIR` reads the tokenizer, the
  chat template and `serving.json` that `launch.prepare` bundles, which `generate --export-native` does not write.
- **The Gated DeltaNet gate lowers through the generic path.** Its fold multiplies `W[k, h]` projections and a
  per-head `X[h, d, k]` by one row; those channels read different B spaces, so it is no contraction and takes no
  tensor-core tile. A per-channel orientation would give it one back.
- **Compiling is slow and mostly unmeasured.** The decode GDN kernel alone took ~20 minutes on the H100 and the
  whole export ~80 minutes: the layer fuses into one kernel offering dozens of cut seams, and the placement
  decision builds every arm to rank it, down a chain of such decisions (the schedule draw is seconds). The
  pricing shortcuts and the parallel arm build cut the dev-box compile of that kernel from 327 s to 115 s; the
  export has not been re-timed. The H100 hardware golden holds 24 rows, so nearly every pick is the prior's. A
  record run on the card is what turns those picks into evidence.

The first FP8 MoE export, Qwen3.6-35B-A3B-FP8 on one RTX 5090 (context 4,096, 128 of 256 experts per layer on the card,
one-token prefill), takes about 8.5 minutes in BF16, and the exporting process peaks at 12.2 GB of host memory. The pack
is 34 GB, 31 GB of it pages; loaded, the worker pins 30 GB of host memory and the card holds 21 GB. The recipe golden's
rows were recorded with fast math off, so they are evidence only when the export runs with `EMMY_FAST_MATH=0`, as the
plugin's boot runs; then all 243 of its measured rows import, beside 99 rows of the card's hardware golden. Only the
fixed-slot expert program takes them: native's mixers keep their FP32 residual seams at width one, where the plugin's
programs carry a BF16 residual at its decode and prefill widths, and the router, combine, embedding, rotary, attention,
head and greedy programs are native's own. A serving config with `SERVE_NATIVE=1` makes those programs twins
(`emmy/serving/ARCHITECTURE.md`), so a record run measures them like any other. Recorded that way, the router's gate
linear and softmax, which the prior kept in one 7.5 ms kernel of one block, take a cut at the linear and run in 4 us.
Its top-k selection does not get faster with evidence: the `k` masked maximum reductions lower to a scan over the picks
whose lowest-ID reduction sits inside another reduction, so the only schedule offered is the serial fold, and the
router takes about 0.35 ms per layer. With the router recorded, a captured decode step takes 460 ms against the
plugin's 42 ms. Per uncaptured step, attention takes 328 ms (one block per head over 4,096 positions), the page copies
about 110 ms while the slots warm, Gated DeltaNet 35, greedy selection 31, the head 25 and the router 16, every one but
the router and the experts on the prior's picks. On four raw prompts greedy decode matches the plugin's 64 tokens on
two; the others take the plugin's second choice at tokens 49 and 2, where both runs' top two are within 0.4 nats.
Without evidence, the 16-row Gated DeltaNet prefill program compiled on one core for 47 minutes (in FP16) and was
stopped, so the pack prefills one token at a time.

## Native HTTP launcher

`emmy serve MODEL --runner generate --native` prepares the artifact in a fresh temporary directory and executes a
prebuilt `emmy-server`. Preparation uses the checkpoint's weights in the model dtype, or coded for a quantized
checkpoint, at the requested revision, with the existing golden and strict compiler-evidence controls. `--native-pack
DIR` reuses an already prepared serving bundle; its recorded model, revision, and context must match. Preparation-only
evidence flags are rejected when reusing a bundle.

Native options are `--host`, `--port`, `--revision`, `--max-model-len`, `--page-tokens`, `--dtype`, and
`--native-pack`, plus the existing Emmy preparation, dry-run, and benchmark controls. Context defaults to 4,096; the page
size defaults to one page spanning it and, like the dtype and the compiler evidence flags, applies to preparation, not
to a reused pack. `--native`
requires `--runner generate`, rejects `--stock`, and rejects unsupported engine arguments. vLLM forwarding stays
unchanged without `--native`. Dry-run prints preparation settings and the native command without downloading,
compiling, or starting a process.

`--bench` uses the existing vLLM benchmark client, with default native concurrency one. Higher explicit concurrency
measures overload and receives busy responses. The client is an optional dependency; normal native serving does not
import vLLM. Install the matching binaries from `make native-dist` before launching. See the
[HTTP adapter contract](../../../crates/emmy-server/ARCHITECTURE.md) for API fields, cancellation, readiness, and tests.
