# Qwen3.5-9B at FP8: a real-weight hybrid for the dev card

The native parity plan's local lane has one gap: Gated DeltaNet coverage on the dev box is synthetic. The tiny
two-layer Qwen3.5 the native tests serve has the right structure at hidden 64 with random weights, so it catches a
lowering bug and says nothing about a real model's numerics or its shapes. This plan closes that with the smallest
real hybrid in the recipes — `Qwen/Qwen3.5-9B` — quantized to FP8 so it fits the 16 GB RTX 4080. Every other
member of these families names 27B or more, except Qwen3.8-Flash-Next, which names no size and is a mixture of
experts; if the 9B falls through, that is the next one to size up.

Written 2026-10-30. The goal is a development surrogate, never a parity reference: no cell of the parity bar is
ever measured here (see "What this does not buy").

## Why this is cheaper than it looks

`fp8-block` is the official FP8 form the Qwen releases ship: e4m3 weights under one f32 scale per 128x128 block,
with per-token dynamic e4m3 activations. Two consequences:

- **The weight side is fully determined.** The block scales come from the tensor's own amax. There is nothing to
  choose.
- **There is no calibration dataset.** Activations are quantized per token at run time, so the one-sample
  calibration caveat that applies to a static NVFP4 activation scale does not apply here at all.

So a checkpoint quantized locally is the same kind of artifact as one Qwen publishes, not a compromise. The repo
already holds the weight math and already writes the exact FP8 declaration the loader's spellers recognize
(`emmy/compiler/loader/synthesize.py`). The FP8 dense trunk is also already served natively — Qwen3.6-35B-A3B-FP8
is one of the two parity recipes, and a native test exports an FP8 coded trunk — so the path this model needs is
exercised, just not from a locally written checkpoint.

What is **not** free: the quantizer is wired to a traced graph as an exploration tool under `emmy compile
--quantize`, not to a whole-checkpoint conversion, and its output has never been served. That is the work.

## Two unknowns that fork this plan

Both are one command and neither needs a GPU. Settle them before anything else.

1. **Is `Qwen/Qwen3.5-9B` a Gated DeltaNet hybrid, and what are its shapes?** The recipe is an onboarding shell
   written from line-level signals on 2026-10-05; nobody has pulled the checkpoint. Fetch its `config.json` and
   read `layer_types`. The Qwen3.5 and Qwen3.8 families mark their hybrid layers `linear_attention`, and
   Qwen3.8-27B runs 48 of those against 16 `full_attention`. **If the 9B has no `linear_attention` layers, stop
   here** — the entire point was GDN coverage, and a dense-attention 9B adds nothing over Qwen3-0.6B, which
   already serves on this card at 1.28x stock.
2. **Is an FP8 variant already published?** If one is, steps 2 and 3 below collapse into pinning its revision in
   a recipe, and this plan is an afternoon instead of a week.

The same fetch answers the memory question. Weights land at about 9 GB at FP8 regardless of the layer split, and
a hybrid's key-value cache is small because only its `full_attention` layers hold one — but the window that fits
depends on how many those are and on the head dimensions.

## Steps

1. **Confirm the model.** The two unknowns above. Verify: `layer_types` holds `linear_attention`; the layer
   counts, hidden size, head dimensions and vocabulary written down; the arithmetic for weights plus cache at a
   development window (8,192 is plenty) under 16 GB with room for activations.
2. **Write the whole-checkpoint FP8 conversion.** Reuse the existing weight math over every linear in a
   checkpoint rather than over a traced graph's. The module's own rule is the design constraint — one producer of
   quantized graphs, because a second one drifts from the loader while still looking right in a dump — so this
   extends that path and does not fork it. Verify: the written config matches the FP8 declaration the loader
   recognizes; the spellers read the result; a test covers one small real checkpoint end to end (Qwen3-0.6B is
   already in the local cache and is the cheap fixture).
3. **Serve it natively.** Trace the serving twins, export a pack, boot it. This is the first time a locally
   written quantized checkpoint is served and it is the step that either takes an hour or uncovers a gap.
   Verify: text out of the HTTP path; logits against the unquantized model within FP8 tolerance on a handful of
   prompts; the carried Gated DeltaNet state advances and resets across requests, which is what the synthetic
   hybrid's own native test already gates.
4. **Make it the dev card's qualification target.** The native tests already take `--native-checkpoint` for
   full-model qualification. Point it at this checkpoint and record what passes on sm_89. Verify: the native
   generation, server and text-parity tests green on the 4080, and the result written into the parity plan's
   local lane so the GDN gap entry there can go.

## What this does not buy

- **No performance evidence.** A measured row belongs to the card it was measured on. Nothing recorded on the
  4080 is evidence for a 5090 pick, and nothing here promotes.
- **No long window.** The parity bar's four long-window cells are the switch's hard blockers and stay on the
  rented card.
- **No NVFP4.** sm_89 has no FP4 tensor cores, so the other parity recipe's quantization is untouched by this.
- **No bar cell, ever.** This model is not in the parity comparison and must not acquire a row in it. Its job is
  to fail fast at the desk on correctness, so the rented sessions spend their time on numbers.

## Recipe, or just a checkpoint

Recommend **not** a recipe, at first: a documented local procedure and a test hook, with no golden, no lifecycle
tag and no entry in the prior reproduction gate.

The reason is the gate. A recipe golden for an RTX 4080 would be the repository's first — the existing sm_89
golden is a 4090's — and once it exists the prior reproduction gate has to reproduce it on every pull request.
That is a standing obligation across the whole repository in exchange for a development convenience, and it is
the kind of growth the contribution rules call creep. Promote it to a recipe if the local loop earns it, which
means: more than one person using it, or a correctness gate that only it catches.

If it is promoted, the shell already exists at `recipes/Qwen3.5-9B` and the work is a native lane plus a
`rtx4080_sm89.json` golden, with `prior-pending` carrying it until a refit reproduces the new rows.

## Risks

- **The model may not be a hybrid.** Step 1 settles it, and a negative answer ends the plan rather than
  redirecting it. This is the one risk worth paying to retire first.
- **The conversion grows.** A whole-checkpoint pass has to handle sharded safetensors, tied embeddings, the
  layers the FP8 declaration excludes, and the multi-token-prediction head if the checkpoint ships one. If it
  stops looking like a reuse of the existing weight math and starts looking like a second producer, stop and
  reconsider — a published FP8 checkpoint, or a smaller real hybrid, beats a parallel quantization path.
- **FP8 on sm_89 is Ada's FP8, not Hopper's.** The dense FP8 trunk is served today on sm_120 and there is an
  sm_89 golden for an FP8 recipe, so the path is not new, but no FP8 model has been served natively on this
  card. A refused schedule or a missing offer at sm_89 is a plausible outcome of step 3.
- **It is one more thing to maintain.** A surrogate that drifts from the real models is worse than none, because
  a green local run stops meaning anything. The mitigation is that it gates correctness only, and correctness is
  what the synthetic hybrid already gates — this adds real weights and real shapes, not a new kind of claim.
