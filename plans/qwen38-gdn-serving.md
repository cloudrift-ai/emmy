# Serving the GDN layers of Qwen3.5 / Qwen3.8 through emmy

Status: open, written 2026-10-02 against main `22bf24e4`. Branch `feat/serve-gdn-layers`. The branch has no code yet.

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

emmy has never used this mechanism for state of its own, so stage 5 starts with an experiment: does `EmmyGenModel`
receive KV cache blocks for the GDN state shapes? If not, the fallback is a single state inside the model class. With
the fallback the model class serves one request at a time, refuses `--max-num-seqs` above 1, and resets the state when
a step starts at position 0.

A step can hold tokens of several requests. The model class separates the requests at their boundaries and runs the
GDN programs for one request after another, each with its own state. This is slower than one call for all requests,
and correct for any number of them. A test submits two requests together.

### Refusals in the first version

The model class refuses prefix caching and speculative decoding on a model with GDN layers. Both replay or skip
tokens, and the first version keeps one state per request with no snapshots.

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

**Output gate (`feat/serve-attention-output-gate`).** Both branches edit `EmmyGenRunner.from_model` and the two layer
loops of `EmmyGenModel`. This branch keeps the GDN code in its own functions and adds one layer-type check per shared
loop. The tiny hybrid test model (`_QWEN3_5_TINY` in `tests/compiler/trace/test_huggingface.py`: one GDN layer, one
full-attention layer) computes its full-attention layer wrongly on main until the output-gate change lands. Stages 1
to 4 therefore start on a tiny config with GDN layers only. The hybrid tests follow after a rebase onto the
output-gate branch.

**BF16 trunk (`feat/serve-bf16-trunk`).** That branch owns the code that picks the trunk dtype in the same two files.
This branch adds no dtype choice of its own: a GDN program takes the trunk dtype for `x` and `H`, and `S` is always
float32. The real checkpoint needs BF16; the tiny-model tests do not.

**Golden recording (`feat/qwen38-nvfp4-rtx5090-golden`).** The GDN rows to record are the `gdn<W>` twins at the three
widths above. Capture them with `extra_widths=(1,)` so that width 1 is among them. Stage 1 fixes the concrete widths
for the RTX 5090 serve command, and this file will list them then. Until then, treat recorded GDN rows as provisional.

## Open questions

- Rounding. Hugging Face computes a prompt in one call. The decomposition mixes wide calls with width-1 calls, which
  use a different but mathematically equal computation. Stage 3 measures the drift. If it exceeds the test tolerance,
  the test gives Hugging Face the same decomposition and compares against that.
- Whether vLLM sets `has_initial_state` to false for a one-token prompt. If vLLM treats such a prompt as a decode, the
  model class would read a stale KV cache block. Stage 5 tests this early.

## Parked for the performance phase

- More widths. With all powers of two up to the prefill bucket, the decomposition is the binary expansion of the
  length. Example with a decode bucket of 16 and a prefill bucket of 64: a request with 63 tokens costs 6 calls with
  powers of two, and 18 calls with widths 64, 16 and 1. The cost is more programs to compile at boot and more rows to
  record.
- A program with a length input that masks padded tokens. Hugging Face's chunked computation already pads to a
  multiple of 64 inside the layer, after the projections. A masked program would turn that padded length into an
  input. One consequence to measure: a narrow program may cost nearly as much as the width-64 program.
- A mix of both: decompose the large part of a length into wide programs, then mask the remainder.
