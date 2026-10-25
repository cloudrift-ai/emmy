# Native cached generation

Python prepares a standalone generation artifact for a dense Qwen3 or Qwen3.5 text model with 16-bit or FP8 weights,
or a Qwen3.5-MoE one with 16-bit or FP8 weights, its routed experts on the device or paged from host memory; a Qwen3.5
vision-language checkpoint's artifact can also answer images (below). The
artifact holds the weights, projection inputs, a KV cache for its full-attention layers and carried state for its Gated
DeltaNet layers. It runs in one model dtype, FP16 or BF16: the checkpoint's, as the vLLM plugin serves it (below).
The output head retains FP32 logits through sampling so 16-bit rounding cannot create a false maximum tie.
The Rust runtime submits the exported launches and retains the KV cache. Compiled GPU reductions select greedy
tokens; positive-temperature sampling runs on the CPU. No Python model operation runs after preparation. The
experimental native HTTP adapter serves one active request; this is not a performance replacement for vLLM. The existing serving integration remains the default.

## Preparation

`prepare.export_model` traces the vLLM plugin's own programs for the trunk (below) and native's glue around them. It
uses the compiler's plan-template cache to reuse identical layer structure. The compiled plans are joined into one
ordinary static plan for each width: seams refer to the same named allocation, and internal names are scoped by layer.
Kernels join by name; one whose name a different source already holds takes a suffix from its own source, as the
paging splice does. There is no new compiler or runtime alias format. Unsupported symbolic and descriptor arguments
are rejected. Cache buffers use the persistent output role, without joining the public logits/token output list.
Custom launches identify their writes so Python scratch allocation preserves the same dependencies as native
execution.

The trunk is the plugin's: each decoder layer's programs are its serving twins (`emmy/serving/ARCHITECTURE.md`), traced
from the same wrappers at the widths the plugin runs them at, with the residual stream in the model dtype, so the recipe
golden's rows for them are native's rows too. A decode step runs the attention halves at the width the plugin runs one
token at (its decode bucket, `EMMY_GEN_DECODE_BUCKET`, when `EMMY_GEN_M1_TIER` is off, else one row), a Gated DeltaNet
layer at one row, as the plugin splits a step into its GDN widths, and a routed layer's experts through the plugin's
one-row fixed-slot program; a prefill chunk runs all of them at the plugin's chunk, `EMMY_GEN_PREFILL_BUCKET`. Native's
own programs run at the step's rows and read and write the leading rows of a wider seam: a decode step's embedding
writes the first row of a decode-bucket-wide hidden buffer, and the rows past it are padding, which the plugin's
programs compute row by row and nothing reads, as the plugin pads one token into its decode bucket. Each of the plugin's
wrappers compiles its programs in one layout scope (`twins.LayoutScope`), decode first, so a weight takes one layout and
one device copy for both programs, as the plugin builds a wrapper's twins narrowest first.

The glue between the projections is compiled too, from small traced modules in `prepare.py`. The host uploads the
tokens a step embeds with its position: the prompt's while it lasts, then the previous step's selection, which the
plugin's gather (`gen_runner.embed_gather_graph`) embeds. The rotary module rotates q and k at this position and
scatters k and v into the cache at that position, and attention runs causal grouped-query attention over the cache's
leading window with every position past this one masked. Both read the position from device memory, including the
cache write, whose rows land at an index computed from the `position` input rather than a host symbol, so a token step
is one launch sequence at every position of a window and replays as one graph. Nothing the device runs is
hand-written: the step ends at the FP32 logits and the greedy token, a compiled reduction over them, and the runtime
samples on the host only at positive temperature (below), so the same export serves any device the compiler targets.
Attention keeps its dot products, scores and softmax in FP32, which avoids losing near-tied scores at large
magnitudes. Each query head reads its key-value head's cache. The probabilities are rounded to the model dtype for the
value expectation, which accumulates in FP32, as attention in the model dtype rounds them, so both products are 16-bit
contractions; the output is rounded to the model dtype. Rotary constants come
from the checkpoint's own module in FP32. Rotation also uses FP32 intermediates and rounds only the query/key outputs
to the model dtype. The final norm is a program of its own, ahead of the head. The existing standalone exporter
bundles all binaries and weight bytes, each constant read from its source as the pack writes it, so the exporting
process never holds the trunk; a constant decode and prefill share is one file, which the runtime reads once and lends
to prefill, and it reads a pack's files one at a time as it uploads them. Generation metadata lives
in the pack key and has its own version. The manifest also records, per program, the kernels whose evidence its
compiles read and the digest of the golden rows they reach, as the plugin's pack does (`emmy/serving/ARCHITECTURE.md`);
nothing checks them, since an existing native pack is served as named.

The model dtype is what `load_model` builds the model in: `--dtype` (`auto` by default), where `auto` takes BF16 for a
checkpoint whose config declares it and FP16 otherwise, the rule vLLM's own `auto` follows, so a BF16 checkpoint runs
the dtype the plugin serves it in and its kernels can be the plugin's. Every 16-bit activation, cache, history and
unquantized weight of the step takes it, a BF16 one stored as its 16-bit pattern; the FP32 logits and the greedy
token do not change with it.

The token-embedding table pages under the plugin's budget, `EMMY_GEN_EMBED_GPU`: below its page count (256 rows a page)
it is a buffer the pager holds in that many device slots, its pages in pinned host memory, and at zero slots every row
is read in place over the link; otherwise it is a constant of the step. A paged table's gather is then the plugin's
`embed-sym` twin, given the step's rows. A checkpoint that keeps its output head coded
(NVFP4, untied) runs it as the plugin does, through the head program over its codes (`coded_head.head_graph`, the
`head-sym` twin) at one row, with its FP32 logits.

A Qwen3.5 text decoder is found with `find_text_decoder`, since those checkpoints wrap it beside a vision tower, which
only a pack with images holds ("Images", below). Its full-attention layers add two things to the Qwen3 path: the rotary
module rotates only the leading part of each head (the family's partial rotary embedding), and the attention split
carries the fused output gate as a fourth `pre` output and third `post` input. Its Gated DeltaNet layers run whole,
through the plugin's program, which reads the FP32 recurrent state and the convolution history in the model dtype and
writes the next ones apart from them, since some of its kernels read the old state after another wrote the new one. A
carry program then writes them over the ones the step carries: each output is spelled as the carried buffer with every
row replaced (a scatter), which the compiler stores into its memory. The plugin moves the same bytes the other way,
gathering a request's state block into the program's inputs. The pack key lists these buffers under `state` (below).

A Qwen3.5-MoE decoder has routed experts and a gated shared expert in every layer, behind either mixer. Each layer's
E-stacked expert weights are two buffers of the step (`layer{i}.w_gate_up`, `layer{i}.w_down`). Preparation reads
the vLLM plugin's budget, `EMMY_GEN_EXPERTS_GPU`: how many experts per layer the device holds. With no budget, or
one at or above the expert count, the weights are constants of the step and nothing is paged; below it they are
paged (below). A MoE layer's section of the step is four programs in sequence, every one static:

1. The mixer is the attention split's MoE `post` or the GDN program's routed form, the plugin's. It writes the
   residual with the shared expert folded in (`layer{i}.h`) and the normalized rows the experts read (`layer{i}.xn`).
2. The router writes each row's scores (`layer{i}.scores`, `rows × k`) and picks (`layer{i}.sel`, `rows × k` int64; one
   row's are `k`), as Transformers' softmax top-k router computes them: the gate's linear in the model dtype, the
   softmax in FP32, then the top `k`. Each expert's rank is counted in one reduction over the row's other experts: one
   outranks it with a larger logit, or an equal one and a lower ID, so a tie goes to the lowest ID. The logits order the
   experts as their probabilities do and are compared as stored; a probability one kernel recomputes in two places can
   differ in its last bit, and a tie it breaks would rank two experts alike. Every expert ranked below `k` is then
   scattered to its rank, its probability beside it. The picked scores are normalized and rounded to the model dtype. A
   logit that is not finite outranks every expert `k` times over, so such a row ranks none in its top `k` and picks the
   last expert in every slot: the gather below stays inside the stack, its logits come out nonfinite and the request
   fails there, as greedy selection fails any nonfinite step.
3. The fixed-slot program of the vLLM plugin (`ExpertFFN.slots`) gathers each row's `k` experts from the stacked
   weights by its picks and writes their outputs (`layer{i}.partials`, `rows × k × hidden`; one row's are
   `k × hidden`). A decode step runs the plugin's own one-row program, one of its twins. A prefill chunk runs the same
   form over all its rows at once, the products batched over the rows as well as the picks, so a chunk's experts are
   one program, not one per row.
4. The combine multiplies each row's scores by its partials and adds the result to the residual, in the model dtype,
   as the plugin's fixed-slot combine does, and writes the layer's output.

Past its mixer, a MoE layer therefore runs three programs per step; the last prefill layer runs none of them, since
nothing reads its output.

A quantized MoE checkpoint's routed experts never load with its twin (below); they stay on the meta device. The export
reads them from the shards a layer at a time, as the loader stacks them for the plugin (`checkpoint_experts`). An FP8
layer's experts are their e4m3 codes on the uint8 carrier beside their FP32 block scales, so the step holds four expert
inputs per layer (`layer{i}.w_gate_up`, `layer{i}.w_down` and a `_scale` for each). The fixed-slot program spells the
codes as the plugin's does: the decode and the scale multiply are in the graph, and the device holds the codes. Each
program reads a layer's experts as it compiles that layer, and the pack reads them once more as it writes their pages,
so below the budget the host holds one layer's experts at a time. At or above it the experts are constants of
each program, so the host holds them in every program's bytes until the pack is written.

Below the budget, each expert input, codes and scales alike, is an input of the step paged along its expert axis, one
expert per page, under the plugin's split of the budget (`config.gen_expert_slots`): a token's picks are its device
slots, in decode and prefill alike, and the rest of the budget is resident experts, the layer's first ones. The backend
splices each fixed-slot launch's inspector and the paging programs ahead of it, so they are ordinary launches of the
step and a captured graph records them. The pager's tables, slots and clock enter the step as inputs named after the
buffer (`layer{i}.w_gate_up__table` … `__slots`), and the copy reads the pages through the buffer's `__view`. Every
row's launch in decode and prefill moves pages through the same tables, so the slots are one cache per buffer. The pack
keeps the expert bytes once, page after page, under its `pages` index, with the count of resident ones. The runtime
reads each file's resident pages straight onto the device and the rest straight into pinned, mapped host memory,
declares both to the pager with the page count, page size, slots and resident count, and lends the pager's tables to
both programs as they load (runtime `ARCHITECTURE.md`, "Cached generation"). The host therefore holds only the experts
off the device, once, and the device holds the resident experts and the slots once; the trunk, the KV cache and the
Gated DeltaNet state stay where they are. A prefill chunk lists more experts than the slots hold, and reads the misses
past them in place over the link. The budget is fixed at preparation: a reused pack keeps the slots and resident experts
it was prepared with. Each launch reading a paged buffer takes its own inspector and paging run, 14 launches per buffer,
so a paged layer adds 28 launches per step with 16-bit experts and 56 with FP8 ones, decode and prefill chunk alike.

The hermetic tiny Qwen3.5-MoE GPU test has two full-attention layers. It checks every logit against Transformers' and
its greedy tokens, through sequential and chunked prefill, capture and reset, at three budgets: every expert, four of
eight (two slots and two resident experts), and one (below the two picks, so the second is read in place from host
memory). Its experts are wide, about 100 MB in all, so they dominate the worker's memory: after load its heap holds
less than half of them, and its peak resident set stays within half of them of its resident set, so the load neither
kept nor passed through a second copy. No test in the suite runs a Gated DeltaNet MoE layer on a card: without
evidence its kernels take minutes to compile, and the same gate with one GDN and one attention layer takes about ten
minutes on an RTX 5090 (it passed by hand on 2026-10-22, with every expert on the card and with two of eight). Its
programs are the plugin's instead, which a CPU test holds wire for wire to the twins the plugin's serving config
traces, for a Gated DeltaNet model with and without routed experts. An FP8 variant of the attention gate reads a
dynamic block-FP8 checkpoint whose experts are stored one module per expert, as Qwen publishes them; every logit and
greedy token matches the dequantized model with every expert on the card and with four of eight, about three minutes
each, and from a BF16 checkpoint with four of eight, whose logits it matches within one BF16 rounding of the
reference's (8e-3).

A quantized checkpoint loads through the serving runner's checkpoint-sourced lane (`load_model`). Its twin is built
from the config with the trunk's coded linears left as placeholders and its routed experts unread; each program's
trace is re-addressed to the checkpoint's keys, the loader's spellers put the decode algebra in the graph, and the
constants bind from the shards. An FP8 trunk therefore stays at its stored size, weight-only under 16-bit activations,
and nothing past the loader knows the format. Transformers' quantizer never runs: a module that still carries its
checkpoint's quantization declaration is rejected.

Preparation rejects other model families, sliding attention, non-default rotary schemes, training mode, and parameters
that are not all FP16 or all BF16, or not on the CPU. Context capacity defaults to the checkpoint's window
(`max_position_embeddings`) and must fit both it and native's 262,144-token limit. Compiler evidence uses the existing
golden and strict-evidence controls; an export from a golden file (`--golden`) runs in the precision regime its rows
were recorded in, as the plugin's boot does, since a row is evidence only there, and refuses an environment pin at
another value. A successfully exported artifact has not, by itself, established numerical correctness or fast
schedules.

## Execution and state

The prompt stays on the host; each step uploads the tokens it embeds. Prefill processes all but its final token in
fixed-width chunks, defaulting to the plugin's prefill chunk (`EMMY_GEN_PREFILL_BUCKET`), else 16 rows. Each layer
writes the chunk's keys and values at their absolute positions before attention reads each query's causal prefix; a
row past the prompt's last token embeds the previous selection, computes alongside the others and writes cache rows
that decode overwrites when it reaches them, so a chunk is dispatched only where all of its rows fit the context and
the decode program covers the rest. The last layer only writes its cache; its attention and post-attention fragment
are unnecessary. The final prompt token runs through the one-token decode program, including the output head, which
yields the first generated token. Later decode steps embed the previous step's token, which the runtime read back from
the program or sampled from its logits, and uploads with the position scalar. Prefill chunks execute no head and
download nothing.

The cache is one K and one V buffer per attention layer, token-major (`[context, kv_heads, head_dim]`) and allocated
for the whole context at load. It holds the model dtype, or FP8 e4m3 under `--kv-cache-dtype fp8` (a recipe names it
in `SERVE_EXTRA_ARGS`, which the plugin's boot reads too); an FP8 store has no scale and saturates to e4m3's largest
finite value, as vLLM's does. The rotary program writes a chunk of each at `position` and names the whole cache, since
a scatter into a windowed buffer leaves its store no grid. The attention program reads only the cache's leading
`cache_tokens` positions, a symbolic size the runtime sets per step to the positions read, rounded up to 4,096 (runtime
`ARCHITECTURE.md`, "Cached generation"), so a step's attention costs what its window holds rather than the whole
context. Its programs compile once with the window symbolic, and their golden rows bench at the symbol's hint. The
cache is paged along its positions only when a page (`export_model(page_tokens=…)`, `emmy generate --page-tokens`) is
shorter than the context; the runtime then allocates every page at load and binds the tables (see the runtime's
paged-buffer contract). The default page spans the context, which is the flat buffer; a smaller page changes only the
addressing, never the tokens generated. A new request
resets the position and prompt length. Attention can only read positions already overwritten by that request, so
clearing the entire cache is unnecessary. Decode owns the shared inputs, cache pages and constants. Prefill loads with
the same inputs and constants lent through the runtime's region interface, takes the same cache, and retains a
separate scratch slab; scratch is packed by liveness within each program. The borrower drops before the owner. Only a
paged cache's pages are allocated twice at load, until prefill takes decode's tables. The embedding and tied
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

## Images

A pack prepared with images (`export_model(vision=CHECKPOINT)`; `emmy serve --native --limit-mm-per-prompt
'{"image":1,"video":0}'`) also holds a Qwen3.5 / Qwen3.8 checkpoint's vision tower, and its steps take an image's
rows in place of the image tokens' embeddings, at M-RoPE positions. A request carries one image at most; video is out.

The tower is one program per image width, `tower256` and `tower1024` (`vision_tower.IMAGE_TOKEN_BUCKETS`), each
running one image whose patch rows the host pads to the width (`vision.tower_programs`). It computes the position rows
(the learned position grid read at each patch's four interpolation corners and weighted, which vLLM's tower computes
beside its blocks), the patch embedding, every block's `pre`, attention and `post`, and the merger, which writes the
image's rows into `images`: a buffer decode owns, the tower programs borrow and both token programs read. The patch
embedding, `pre`, `post` and merger are the plugin's tower twins, traced through the same `traced_widths` the plugin's
tower builds them with, narrowest width first in one layout scope per piece, so the recipe golden's tower rows are
native's; a CPU test holds them to the twins wire for wire. Two programs are native's own. The position rows are one.
The other is each block's attention, the piece vLLM runs between `pre` and `post`: q and k rotated in FP32 by each
patch's row and column positions through the rotary tables vLLM's tower keeps (rounded to the model dtype), then
attention over the image's own patches in the model dtype. A padding patch's key is masked through one more score
dimension, where every query holds one and a padding key a large negative value, and the score's head is zero-padded
to whole 16-wide steps: the mask is a term of the score's own contraction, so the expectation folds chunk by chunk on
tensor cores, FlashAttention's shape, which a coordinate mask over a loaded patch count would refuse. The recipe golden
records that route (the rotated query and key cut).

Below `EMMY_GEN_VISION_GPU` blocks each block's projection weights are one page of a pool per weight, as the plugin's
tower pages them (serving `ARCHITECTURE.md`, "The vision tower, and `EMMY_GEN_VISION_GPU`"). The pools are pager
buffers of the tower programs with their pages in the pack. Ahead of a host block's launches each program runs the
pager's step over the page list it holds for that block, so the block's pages move into the slots on the stream;
each block reads its weight as a paged constant whose page table is the block's entry of its pool's table, which the
pack key's `vision.pools` names and the runtime binds. The narrower program borrows every constant its file shares
with the widest.

The token programs take two more step inputs: each row's M-RoPE position (`mrope`, `[rows, 3]`, temporal, height,
width) and the image row it takes (`image_rows`, -1 for none). A merge program after the embedding gather puts an
image token's row of `images` in place of its embedding, in decode as in prefill, since a prompt's tail runs one token
at a time through decode. The rotary program reads each rotary column's table at its axis' position, the axes
interleaved as transformers recomposes them (`vision.mrope_axes`); a text row's three positions are one, so its
rotation is the plain one's, bit for bit. A pack without images keeps the plain rotary and neither input. The HTTP
adapter prepares the image and every prompt token's M-RoPE position (`crates/emmy-server/ARCHITECTURE.md`, "Images").

## Sampling contract

Generation artifact version 12 ends the decode step at the FP32 logits and the greedy token, adds the prefill width,
with a separate prefill program when that width exceeds one, lists the buffers a step carries, empty for dense Qwen3,
reads the pages of every buffer the pager holds from the pack, reads the cache to the `cache_tokens` window, and for a
pack with images names its tower (`vision`: its widths, the image token and the geometry the host writes the tower's
inputs at, and its pools). Every program's inputs are the position and the tokens it embeds, and with images each row's
M-RoPE position and image row; the decode program's outputs are the logits and the lowest
token ID among their maxima, which a compiled reduction selects on the device. At temperature zero the runtime
downloads that token alone; at positive temperature it downloads the logits — one vocabulary-sized FP32 transfer per
generated token, none during prefill — and samples on the host. The head keeps its scores in FP32, so 16-bit rounding
cannot create a false maximum tie. Older generation artifacts must be exported again; the underlying execution-plan
format is unchanged. Temperature must be finite and nonnegative, and top-p must lie in `(0, 1]`. Nonfinite logits fail
the request. Top-k is unsupported.

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

The first FP8 MoE export, Qwen3.6-35B-A3B-FP8 on one RTX 5090 (context 4,096, 128 of 256 experts per layer on the
card), ran on the dev box on 2026-10-23 and again on a rented x16 host on 2026-10-24, once the trunk was the plugin's.
It compiles from the recipe golden alone under strict evidence in 17 minutes, and the exporting process peaks at 6.7
GB of host memory. The pack is 34 GB, 31 GB of it pages; loaded, the worker pins the 16 GiB of experts off the device
(8 slots and 120 resident experts per layer) and loads the rest onto the device, so it fits a 23 GB host. The recipe
golden's rows were recorded with fast math off, so the export runs in that regime, as the plugin's boot does. The
trunk's programs are the plugin's at its decode bucket (4) and chunk, and read their rows. The recipe's serving env
sets `SERVE_NATIVE=1`, so native's own programs (carry, router, combine, rotary, attention, final norm, head
and greedy selection, and the plugin's `embed-sym` gather) are twins too (`emmy/serving/ARCHITECTURE.md`), and the
golden holds a measured kernel set for each of them at one row and at the 64-row chunk; `emmy eval golden` deploys
every native twin from the file's rows. Most are the prior's picks, measured; a few took a cut or a split a pinned
measurement found faster: the router's gate linear and softmax statistics (7.5 ms in one block), attention's scores
(31 ms with one block per head), and greedy selection's peak and a 128-way split over the logits (29 ms in one block).
The chunk's fixed-slot expert program had no row then, so that export prefilled one token a step (the full window
below records one). The prefill attention piece that read the paged cache was recorded with asynchronous copies, since
no TMA descriptor may read a page. The pager's own kernels are the
only ones the strict export compiles without a row: they are compiled at `paging.HINTS` whatever the geometry, where
the card's hardware golden holds their rows, and at 256 pages their decision kernel builds a single schedule (one
CTA), so no row could change it.

On the dev box, served over HTTP with the pager warm across requests, as the plugin's server is, native decoded the
same four raw prompts at 46 ms a token against the plugin's 51 ms on the same box (`vllm bench serve`, one request at
a time, 64 tokens each). Both runtimes move the same bytes: from the 16th output token on, the page copies carry a
median 96 MB a step over the PCIe 5.0 x1 link, 29 ms, for each of them, so on this box the gate is bound by the link.
Besides the copies, native's GPU work is 15.8 ms a step against the plugin's 17.9: the pager's table programs take 7.8
ms in both, where the decision kernel runs on one CTA (about 34 us for each of the 160 paged launches), and the rest
is 7.9 ms against 10.1. The same bytes would take about 2 ms on an x16 link. Greedy decode matches the plugin's 64
tokens on two of the prompts; the others take the plugin's second choice at tokens 49 and 2, where native's own top
two are 0.17 and 0.06 nats apart and the plugin's 0.38 and 0.13. The first token comes after about 0.7 s. One `emmy
generate` process starts with a cold pager, so its first steps copy most of the experts they route to and its time per
token over 64 tokens is about 62 ms.

On the x16 host the worker loads in 83 s after the export (the pack read from disk), and the next server is ready in
40 s. Over HTTP it decodes the four raw prompts at 20.0 to 20.3 ms a token over 64 tokens and 19.7 over 256, with the
first token after 0.23 to 0.25 s; the plugin's boot at its decode bucket 4 on the same host measured 20.2 and 19.7
earlier that day. Six of nine prompts give the earlier export's tokens; the others leave them at tokens 49, 13 and 23,
where that export's token is native's second choice, 0.02, 0.34 and 0.19 apart in the logits. A 1,000-token prompt,
one token a step, gets its first token after 19.0 s.

NVIDIA's Qwen3.8-27B-NVFP4 on one rented x16 RTX 5090 (2026-10-24: context 4,096, decode bucket 16 with the M=1 tier
off, the 64-row chunk, fast math as the recipe's server runs) is the first coded-W4A4 export. Its trunk is the
plugin's twins over the codes the loader spells, the coded head its `head-sym` program, and the embedding table pages
at zero slots (`SERVE_EMBED_GPU=0`), gathered by its `embed-sym` program. Native's own twins are carry,
rotary, attention, final norm and greedy selection at one row and at 64; greedy selection takes the Qwen3.6 recipe's
split (8.3 us; the prior's one-block pick takes 24 ms) and attention the scores cut (39 us a layer at one row, 1.2 ms
at 64, from 249 us and 15.6 ms before the cache stored values by head dimension). The strict export from the recipe
golden alone takes 16 minutes, and the exporting process peaks at 5.1 GB. The pack is 19 GB: 17 GB of bindings, 2,024
of the prefill program's 2,392 being decode's files, and the table's 2.4 GB of pages. The worker loads in 15 s, its
peak resident set 3.3 GB of host memory, 2.5 GB of it the pinned table, and holds 18.3 GB on the device; the server is
ready in 16 s. Over HTTP it decodes the four probe prompts at 17.9 ms a token against 17.8 for the plugin's text-only
boot in the same session, with the first token after 207 ms against 161. Greedy decode gives the plugin's 64 tokens on
one prompt and leaves them at tokens 29, 20 and 50 on the others, where the plugin's own top two are 0.68, 0.98 and
0.23 nats apart and native's 0.19, 0.28 and 0.40. Teacher-forced on the plugin's 256 tokens, native's log-probability
of each is within 0.002 nats of the plugin's at the median and 0.54 at the 99th percentile, and its pick differs at 7
positions, at each of which the plugin's own token has a log-probability of -0.32 to -1.05. The trunk is the same
programs; what differs is what native runs instead of vLLM: attention over its cache in FP32, rotary and greedy
selection. A chunk of a model with carried Gated DeltaNet state holds only prompt tokens before the last, so these
prompts of 8 to 13 tokens prefill one token a step, and the first token waits for each.

Both recipes serve the full window too (2026-10-24, same host): `--kv-cache-dtype fp8` at the checkpoint's 262,144
tokens, prefilled in the plugin's 64-row chunks from their recipe goldens under strict evidence, and a full-window
passkey prompt returns its key on each within 32 GiB of the device; the 4,096-token gates are unchanged. Qwen3.6's
64-row expert program reads paged weights at the recipe's 128 slots, where the gate-up piece is offered no tensor-core
tile and the down projection stages through shared memory without TMA; the twin a golden stores carries no paging and
offers TMA, so its rows are recorded at the schedules the paged export offers. Long prompts reach their first token
later than the plugin's on both models: NVIDIA's 64-row attention writes its scores and probabilities over the window
and its value expectation runs far below bandwidth, and a Qwen3.6 chunk routes to more non-resident experts than the
device slots hold, so its expert kernels read their pages over the link.

## Native HTTP launcher

`emmy serve MODEL --runner generate --native` prepares the artifact in a fresh temporary directory and executes a
prebuilt `emmy-server`. Preparation uses the checkpoint's weights in the model dtype, or coded for a quantized
checkpoint, at the requested revision, with the existing golden and strict compiler-evidence controls. `--native-pack
DIR` reuses an already prepared serving bundle; its recorded model, revision, and context must match. A bundle that
answers images also holds the checkpoint's image processor config, its largest image bounded by the tower's widest
program. Preparation-only
evidence flags are rejected when reusing a bundle.

Native options are `--host`, `--port`, `--revision`, `--max-model-len`, `--page-tokens`, `--dtype`,
`--kv-cache-dtype` (`auto` or `fp8`), `--limit-mm-per-prompt` (vLLM's spelling; at most one image and no video, and
a preparation option like the dtypes), and `--native-pack`, plus the existing Emmy preparation, dry-run, and benchmark
controls. Context defaults to the checkpoint's window, up to 262,144 tokens, and a reused pack serves its own; the page
size defaults to one page spanning it and, like the dtypes and the compiler evidence flags, applies to preparation, not
to a reused pack. `--native`
requires `--runner generate`, rejects `--stock`, and rejects unsupported engine arguments. vLLM forwarding stays
unchanged without `--native`. Dry-run prints preparation settings and the native command without downloading,
compiling, or starting a process.

`--bench` uses the existing vLLM benchmark client, with default native concurrency one. Higher explicit concurrency
measures overload and receives busy responses. The client is an optional dependency; normal native serving does not
import vLLM. Install the matching binaries from `make native-dist` before launching. See the
[HTTP adapter contract](../../../crates/emmy-server/ARCHITECTURE.md) for API fields, cancellation, readiness, and tests.
