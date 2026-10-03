# Serving the GDN layers of Qwen3.5 / Qwen3.8 through emmy

Status: open, written 2026-10-02, last updated 2026-10-04 on main `abca1022`. Branch `feat/serve-gdn-layers`, PR #1023.
All seven stages ran. On 2026-10-04 `Inferact/Qwen3.8-27B-NVFP4` booted on an RTX 5090 under `--strict-evidence` with
the golden of #1027, in 9.5 minutes, and answered the fixed prompts of PR #993 coherently: about 0.13 s per decoded
token and 4 to 6 s per 64-token prompt chunk. What remains is finalization of the PR. Follow-ups that this goal does
not need are in [`gdn-serving-followups.md`](gdn-serving-followups.md).

## Goal

`emmy serve --runner generate` serves a hybrid Qwen3.5 / Qwen3.8 model. A hybrid model mixes full-attention layers
with GDN layers (gated DeltaNet; `layer_types` value `linear_attention`). A GDN layer is a linear-attention layer that
carries recurrent state from token to token: per request, a matrix state `S` and a convolution history `H`.

Today `EmmyGenRunner.from_model` fails on the first GDN layer, because it reads the layer's `self_attn` attribute and
a GDN layer has none. The tracing side is ready: serving-twin capture (`capture_twin_graphs`) already traces a
`gdn<W>` program for a GDN layer at a static width `W`.

The first target is `Inferact/Qwen3.8-27B-NVFP4` on one RTX 5090. Correctness comes first. Speed work comes later.

Out of scope: the speed of the GDN kernels, one call that serves several requests, whole-step CUDA graph capture, and
the native-serving exporter.

A *step* below is one forward call of the model. vLLM packs the tokens of every scheduled request into one step.

## Decisions

### GDN programs and their widths

One program serves a whole GDN decoder layer at one static width:

    gdn<W>:  x[1, W, hidden], S[1, value heads, key dim, value dim] float32, H[1, conv dim, kernel]  ->  x', S', H'

Zero `S` and `H` start a request. The runner builds these programs from the same traced graphs as serving-twin
capture. They therefore have the same kernel identities, and rows measured on the twins apply to serving.

The first version builds `gdn<W>` at three static widths: 1, the decode bucket, and the prefill bucket. Twin capture
traces the two buckets by default; width 1 needs `extra_widths=(1,)` unless the decode bucket is 1.

The runner never pads a GDN program. A padded token still decays `S` and enters `H`, so padding corrupts the state.
Attention layers serve a length that fits no static width with a symbolic program; GDN has none, because twin capture
refuses symbolic GDN widths. So the runner decomposes a request's tokens in a step greedily into the static widths,
largest first, and threads the state through the calls. Width 1 always fits, so the runner serves every length.

### State

The runner owns no state. Its GDN call takes `S` and `H` as arguments and updates them. The caller owns the memory.

Per-request state lives in vLLM's KV cache blocks, the fixed-size pages of vLLM's paged KV cache. Stock vLLM uses
this mechanism for the same model family. vLLM allocates and frees the state. At each step it tells the model class
which KV cache block and which token range belong to which request. This matches every model emmy serves today: vLLM
owns all per-request state. The closest existing case is DeepSeek V4: `EmmyGenModel` hosts an attention sublayer taken
from a patched vLLM, and that sublayer registers its own KV cache layers with vLLM.

How the model class uses the mechanism:

- A checkpoint with GDN layers boots `EmmyGenHybridModel`, a subclass of `EmmyGenModel` with an architecture name of
  its own. vLLM's hybrid flag belongs to the class; on `EmmyGenModel` it would change the KV cache sizing of every
  model emmy serves. The serve command picks the class from the checkpoint's `layer_types`.
- One small module per GDN layer declares the state layout to vLLM: the convolution history in the trunk dtype, then
  the recurrent matrix in float32. It uses vLLM's plain linear-attention state backend, whose metadata carries each
  request's token range, sequence length and KV cache block. vLLM's own GDN backend would pull in vLLM's GDN kernels.
- vLLM never zeroes a KV cache block. The model class zeroes a request's state when the request's scheduled tokens
  are its whole sequence, which means it has no computed token yet.

A step can hold tokens of several requests. The model class separates the requests at their boundaries and runs the
GDN programs for one request after another, each with its own state. This is slower than one call for all requests,
and correct for any number of them. A test submits three requests together, then again in reverse order so that
they reuse KV cache blocks that earlier requests freed.

### Refusals in the first version

The model class refuses prefix caching and speculative decoding on a model with GDN layers. Both replay or skip
tokens, and the first version keeps one state per request with no snapshots. It also refuses CUDA graph capture: a
GDN layer reads each request's token range on the host, which a capture cannot record. The serve command passes
`--enforce-eager` for such a checkpoint.

## Stages

Each stage starts with a failing test.

1. The runner builds a model with GDN layers. Check: the build succeeds, and the GDN programs have the kernel
   identities of the twins.
2. The runner computes one GDN layer at an exact width. The state stays outside the activation arena (`BufferArena`):
   the arena shares buffers by name across layers, so the next layer would overwrite it. Check: one call matches the
   Hugging Face layer.
3. Any number of tokens per request. Check: prompt lengths 1, 5, 16, 17, 63, 64, 65 and 131 match Hugging Face, and
   a test shows that a padded call diverges.
4. A whole tiny model through the runner: a prompt, single-token steps, then a second request. Check: it matches the
   Hugging Face model with its cache, and the second request is bit-identical to the same request alone.
5. The vLLM model class keeps the state in vLLM's KV cache blocks, its layer loops dispatch on the layer type, and it
   handles requests one after another. Check: greedy tokens through vLLM equal Hugging Face's, for one request and
   for two together.
6. The refusals, then `emmy/serving/ARCHITECTURE.md`.
7. The real checkpoint boots on an RTX 5090 and answers fixed prompts. This needs the three other branches below.

## Coordination with the other Qwen3.8 branches

Three other branches remove the remaining blockers for serving Qwen3.8 NVFP4 on an RTX 5090. This section says what
each of them needs to know about this branch.

**Output gate (merged, #1022) and the folded norm constant (merged, #1024).** We rebased this branch onto both. It
keeps the GDN code in its own functions and adds one layer-type check per shared loop. Stages 1 to 4 ran on a tiny
config with GDN layers only (`qwen3_5.gdn.l2` in `tests/serving/helpers.py`). One runner test covers the tiny hybrid
model: a prompt in one step, with a reference attention between `pre` and `post`. That model is `_QWEN3_5_TINY` in
`tests/compiler/trace/test_huggingface.py`: one GDN layer, one full-attention layer. Before #1024, three constants
that tracing folds into the graph (`1 + weight` of both norms, and `-exp(A_log)`) had no value when serving bound
the weights, so the width-1 GDN program returned a wrong state.

**BF16 trunk (`feat/serve-bf16-trunk`).** That branch owns the code that picks the trunk dtype in the same two files.
This branch adds no dtype choice of its own: a GDN program takes the trunk dtype for `x` and `H`, and `S` is always
float32. The real checkpoint needs BF16; the tiny-model tests do not.

**Golden recording (`feat/qwen38-nvfp4-rtx5090-golden`).** The GDN rows to record are the `gdn<W>` twins at the three
widths above. Capture them with `extra_widths=(1,)` so that width 1 is among them. The concrete widths for the
RTX 5090 serve command are its decode bucket, its prefill bucket, and 1. The tiny test model showed two things:

- GDN rows need cuts. A placement cut, spelled `PLACE@<route>=cut` in `EMMY_KNOBS`, splits a fused kernel into
  pieces, each its own kernel. With no cut, a fused GDN kernel recomputes its producers inside every output cell:
  one width-1 call of the tiny model took 165 s. Cutting `k_linear_matmul_mean_reduce_a19fd0` (width 4) and
  `k_linear_matmul_mean_reduce_5d3ccd` (width 16) into 7 pieces each, and `k_linear_mean_conv1d_reduce_290483`
  (width 1) into 4, brought a call to milliseconds. The routing rows of `tests/serving/goldens/serving.golden.json`
  record the cuts that worked.
- Record the GDN rows on BF16 programs, the dtype the checkpoint serves in. Cuts do not carry over between dtypes.
  On the tiny model the float32 program has one big recurrence kernel, the float16 width-1 program fuses into a
  single kernel, and the BF16 program splits into 8 to 13 kernels, because every dtype boundary of the Hugging Face
  layer becomes a small cast kernel. Each dtype needed its own cuts.
- Do not take GDN schedules from a compile that has no golden row for the kernel. At the time of writing the team
  treats the schedule prior as broken, so pin every GDN kernel by a golden row or by `EMMY_KNOBS`.

## Open questions

- Rounding in half precision. Hugging Face computes a prompt in one call. The decomposition mixes wide calls with
  width-1 calls, which use a different but mathematically equal computation. In float32 on the tiny model the two
  agree within the test tolerance (2e-3 on the logits) for every tested prompt length. Nobody has tested FP16 or
  BF16 yet.
- Whether vLLM sets `has_initial_state` to false for a one-token prompt. If vLLM treats such a prompt as a decode, the
  model class would read a stale KV cache block. Stage 5 tests this early.

## Parked for the performance phase

The "Speed" section of [`gdn-serving-followups.md`](gdn-serving-followups.md) covers more widths (powers of two),
a masked program and the mix of both, with the measurements so far.
