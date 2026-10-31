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

## The model, confirmed 2026-10-30

Fetched from the hub. The stop condition is cleared and the quantization question is settled.

`Qwen/Qwen3.5-9B` is ungated, 9.653B parameters, 18.0 GiB of BF16 weights, and a
`Qwen3_5ForConditionalGeneration` vision-language model. Its text config:

| | |
| --- | --- |
| layers | 32: **24 `linear_attention`**, 8 `full_attention` — the 3:1 mix Qwen3.8-27B runs at 48:16 |
| hidden, vocabulary | 4,096 and 248,320, embeddings untied |
| attention | 16 query heads, 4 key-value heads, head dimension 256, output gate on the query projection |
| Gated DeltaNet | 32 value heads, 16 key heads, key and value head dimension 128, convolution kernel 4 |
| window | 262,144 |
| drafter | a multi-token-prediction head ships in the config |

It therefore covers more than it was picked for: the hybrid layer mix, the drafter and a vision tower, all at
real shapes. The onboarding shell's "no distinct momentum" note of 2026-10-05 reads oddly against 6.6M downloads
and 2,282 likes today, but that assessment belongs to the discovery agent, not this plan.

**No official FP8 variant exists.** The hub's quantized builds of this checkpoint are GGUF and MLX, which this
repo does not read, plus one published AWQ 4-bit (`cyankiwi/Qwen3.5-9B-AWQ-4bit`, compressed-tensors, from a
publisher two recipes already use). AWQ would have skipped the conversion altogether — but native prepares a pack
for 16-bit, FP8 or NVFP4 weights only, and packed int4 is a plugin-lane and compile-path format. So step 2 is
required and the AWQ checkpoint is not a shortcut.

**The memory budget.** A cache lives only in the 8 full-attention layers: 8 x 2 x 4 key-value heads x 256 = 16
KiB a token at FP8. Against the card's 15.99 GiB, leaving roughly 1 GiB for the context, activations and
workspace:

- Weights are **10 to 11 GiB**, not the 9 GiB a flat parameter count suggests: the untied embedding and output
  head hold 2.03B parameters between them and an embedding table stays 16-bit. Whether the head quantizes is a
  choice in step 2, worth about a gigabyte.
- Cache at 8,192 tokens is 128 MiB; at 32,768, 512 MiB; at 131,072, 2.0 GiB; at the full 262,144, 4.0 GiB.
- The carried Gated DeltaNet state is 50 MiB a request and its convolution history is negligible.

A window up to 131,072 fits with room, which is enough to exercise the long-window kernels for correctness at a
real length. The full 262,144 is borderline and only reachable with the head quantized and the vision tower left
out of the pack.

## Steps

1. ~~Confirm the model.~~ **Done** — the section above. Pull the checkpoint next; 18 GiB over the link is the
   only cost, and the conversion in step 2 needs it on disk anyway.
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
- **No long-window numbers.** The parity bar's four long-window cells are the switch's hard blockers and stay on
  the rented card, because they are measured on the target models. The long-window kernels themselves do run here
  at a real window, which is this surrogate's best surprise.
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

- **The conversion grows.** This is now the plan's first risk, since the model is settled. A whole-checkpoint
  pass has to handle sharded safetensors, the layers the FP8 declaration excludes, the untied output head, and
  the multi-token-prediction head this checkpoint does ship. If it stops looking like a reuse of the existing
  weight math and starts looking like a second producer of quantized graphs, stop and reconsider — a smaller
  real hybrid beats a parallel quantization path. The published AWQ build is not the alternative: native does
  not read packed int4.
- **The weights may not leave enough room.** 10 to 11 GiB is an estimate off the parameter count and the
  untied-head arithmetic, not a measurement. If the conversion lands nearer 12 GiB the long-window work moves
  back to the rental and the surrogate is still worth having for everything else.
- **FP8 on sm_89 is Ada's FP8, not Hopper's.** The dense FP8 trunk is served today on sm_120 and there is an
  sm_89 golden for an FP8 recipe, so the path is not new, but no FP8 model has been served natively on this
  card. A refused schedule or a missing offer at sm_89 is a plausible outcome of step 3.
- **It is one more thing to maintain.** A surrogate that drifts from the real models is worse than none, because
  a green local run stops meaning anything. The mitigation is that it gates correctness only, and correctness is
  what the synthetic hybrid already gates — this adds real weights and real shapes, not a new kind of claim.
