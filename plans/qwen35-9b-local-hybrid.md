# Qwen3.5-9B at FP8: a real-weight hybrid for the owner's RTX 4080

The native parity plan's local lane has one gap: Gated DeltaNet coverage on the owner's 16 GB RTX 4080 is synthetic.
The tiny two-layer Qwen3.5 the native tests serve has the right structure at hidden 64 with random weights, so it
catches a lowering bug and says nothing about a real model's numerics or its shapes. This plan closes that with the
smallest real hybrid in the recipes, `Qwen/Qwen3.5-9B`, through a published FP8 build. Every other member of these
families names 27B or more, except Qwen3.8-Flash-Next, which names no size and is a mixture of experts; if the 9B
falls through, that is the next one to size up.

Written 2026-10-30, revised 2026-10-31 after a review found the first version wrong about the checkpoint, the rows and
the test hook. The goal is a development surrogate, never a parity reference: no cell of the parity bar is ever
measured here (see "What this does not buy").

**It is worth doing only with measured rows.** The tiny hybrid already covers the correctness class. What the 9B adds
is real weights, real shapes and a real window, and those only matter if the model compiles and steps at a usable
speed. Step 2 is the go or no-go, and it comes before any other work.

## The model, confirmed 2026-10-30

`Qwen/Qwen3.5-9B` is ungated, 9.653B parameters, 18.0 GiB of BF16 weights, and a
`Qwen3_5ForConditionalGeneration` vision-language model. Native's export accepts its text model type. Its text config:

| | |
| --- | --- |
| layers | 32: **24 `linear_attention`**, 8 `full_attention` — the 3:1 mix Qwen3.8-27B runs at 48:16 |
| hidden, vocabulary | 4,096 and 248,320, embeddings untied |
| attention | 16 query heads, 4 key-value heads, head dimension 256, output gate on the query projection |
| Gated DeltaNet | 32 value heads, 16 key heads, key and value head dimension 128, convolution kernel 4 |
| window | 262,144 |
| drafter | a multi-token-prediction head, whose `mtp.*` tensors ship |

It therefore covers more than it was picked for: the hybrid layer mix, the drafter and a vision tower, all at real
shapes.

## The checkpoint, read from the shard headers 2026-10-31

The first version of this plan said no FP8 build existed and wrote a converter. Both were wrong: the build was in the
hub listing it was looking at. These sizes come from each published build's safetensors headers, without downloading
the weights. Nobody has loaded either build.

| Build | Weights | Size | Native |
| --- | --- | --- | --- |
| `axteam/Qwen3.5-9B-FP8-dynamic` | compressed-tensors, FP8 trunk under BF16 scales, dynamic activations | 11.54 GiB | the candidate |
| `cyankiwi/Qwen3.5-9B-AWQ-4bit` | compressed-tensors, symmetric int4 at group 32 | 8.45 GiB | refused: the loader requires the asymmetric layout |

The hub also lists `-NVFP4A16` and `-AWQ-8bit` builds under the same publisher. This card has no FP4 tensor cores, and
the 8-bit build was not examined. A 401 on the obvious official names does not establish that Qwen publishes no FP8
build, so step 1 checks that before pinning a third-party revision.

What differs from the targets' FP8:

- **The scale form.** The candidate has one scale per output channel (`[8192, 1]` on `[8192, 4096]`, named
  `weight_scale`). Qwen's own FP8 has one per 128x128 block (`[64, 16]` on `[8192, 2048]`, named `weight_scale_inv`).
  The loader's scale handling names both shapes, but native has only served the block form. So this model exercises a
  different scale kernel from the target's, and the block form stays with the existing tests.
- **What stays 16-bit.** The output head, the embedding, the drafter and the vision tower are all BF16 here.

Native serves an FP8 trunk weight-only under 16-bit activations, so the dynamic activations and FP8 tensor cores never
come into it.

## The memory budget

From the headers, against the card's 15.99 GiB. The vision tower (0.85 GiB) is left out, since a text pack never loads
it.

| Piece | GiB |
| --- | --- |
| FP8 trunk | 6.45 |
| Output head | 1.89 |
| Embedding | 1.89, or 0 at `EMMY_GEN_EMBED_GPU=0` |
| Drafter | 0.45 |
| **Weights** | **10.68, or 8.79 with the embedding off the card** |

On top of that:

- The cache is 16 KiB a token in the 8 full-attention layers at FP8, plus 2 KiB for the drafter's layer: 2.25 GiB at
  131,072 tokens and 4.5 GiB at 262,144.
- Attention's scores and probabilities are estimated at about 0.75 GiB at 131,072 and 1.5 GiB at 262,144 for a 64-row
  chunk. This is not measured.
- The carried Gated DeltaNet state is 50 MiB a request.
- The context and workspace are estimated at about 1 GiB.

So 131,072 fits with room (about 14.7 GiB). 262,144 does not fit with the embedding on the card (about 17.7 GiB) and
only just fits with it off (about 15.8 GiB), so treat it as out of reach until measured.

## Steps

1. **Load the published build.** Check for an official FP8 repository, then pin a revision and pull it (11.54 GiB).
   Export a text pack at a small context, non-strict, with the embedding on the card. Verify: the loader recognizes the
   declaration and native's coded lane takes the per-channel scale. If it refuses, name what refused. Fix it in the
   loader if the fix is a small one. Otherwise produce a block-scale checkpoint with a standard tool such as
   llm-compressor; no new `emmy` command, and `synthesize.py` is not a whole-checkpoint path.

2. **Measure the cold compile: go or no-go.** The first version said "no golden". That collides with what Gated
   DeltaNet programs have done with no rows: a tiny-hybrid engine setup stopped after 15 minutes pricing cut arms,
   a cold strict compile of Qwen3.6's width-4 program took 33 minutes, and 8 of 10 pinned tiny programs held a remainder
   the 62 s watchdog stopped. #1294 was aimed at this and merged 2026-10-30, unmeasured since. Also, a strict export
   refuses a fork with no evidence, and this card's hardware golden holds nine matmul rows, none of them for this
   model. So measure one one-row Gated DeltaNet program and one attention program of this model, cold. Verify: the
   compile time and the step time, written here. If either is unusable after a bounded tune, stop: the tiny hybrid
   stays the local coverage and this plan is closed with its numbers.

3. **Tune and record on the 4080, and keep the rows.** Run the tune and record pass on the programs native needs at
   one, three and 64 rows, plus the drafter's. Verify: a strict export of this model from those rows alone. See "Where
   the rows live".

4. **Serve it natively.** Boot the pack, run text through the HTTP path, and check that the carried Gated DeltaNet
   state advances and resets across requests, which is what the synthetic hybrid's own native test gates.

5. **Check numerics against a reference that fits.** `--native-checkpoint` cannot take this model: it loads the
   checkpoint through `from_pretrained` and puts a 16-bit copy and an FP32 copy on the GPU, about 54 GiB at this size.
   Add a reference hook instead: teacher-forced log-probabilities from the unquantized checkpoint at FP32 on the CPU
   (36 GiB of the host's 60, another 18 GiB pull), compared on a handful of prompts within FP8 tolerance. It uses no
   Emmy code, which is the point of a reference. Verify: the deltas, written here.

6. **Exercise the long window.** Run 5 and 6's kernels at 131,072 positions once those milestones land. Verify: tokens
   and logprobs at the long window against the reference.

7. **Close the gap entry.** Write what passed into the parity plan's local lane, so the Gated DeltaNet gap can go.

## What this does not buy

- **No performance evidence for another card.** A measured row belongs to the card it was measured on. These rows
  serve this card and are no evidence for a 5090 pick. Nothing here promotes.
- **No long-window numbers.** The parity bar's four long-window cells are the switch's hard blockers and stay on the
  rented card, because they are measured on the target models. The long-window kernels themselves run here at most at
  131,072.
- **No NVFP4.** sm_89 has no FP4 tensor cores, so the other parity recipe's quantization is untouched by this.
- **No TMA.** sm_89 has none, so a TMA schedule the targets use is never run here.
- **No bar cell, ever.** This model is not in the parity comparison and must not acquire a row in it. Its job is to fail
  fast at a desk on correctness, so the rented sessions spend their time on numbers.

## Where the rows live

In the existing `recipes/Qwen3.5-9B` shell, as `golden/rtx4080_sm89.json`, with the recipe staying unmaintained (it is
tagged `onboarding` and `untested`). The repository's golden scope is hardware goldens plus maintained recipes'
goldens; another recipe's golden is evidence only where a caller names it, through `--golden`. So these rows put no
standing obligation on anyone else's pull request: no prior reproduction gate and no fresh-lowering test. The first
version feared that obligation and dropped the rows, which is the mistake step 2 exists to catch.

The price is that nothing keeps the file current. A lowering change makes it stale silently, so the local loop starts
with `emmy golden check` on it and restamps when told to. If the local loop earns more, which means more than one
person using it or a correctness gate only it catches, tag it `maintained` and `prior-pending` then. A tune also writes
a freeze file beside this card's; set `EMMY_FREEZE_DIR=` empty for these runs unless the rows should feed a refit.

## Risks

- **Compile or step time rules it out.** This is the plan's first risk and step 2 settles it. A 9B program is far
  larger than the tiny hybrid's, and the only measurements of this class predate #1294.
- **The third-party build.** Its publisher is outside this repository's usual set, and nothing here has loaded it. Pin
  the revision. The numerics check in step 5 is what makes a bad conversion visible.
- **The scale form.** Native may refuse the per-channel scale, or its kernels may be slow for reasons no target shares.
  Time spent tuning a form no target uses is waste, so a block-scale checkpoint from a standard tool is the fallback.
- **The weights leave less room than estimated.** Attention and workspace are estimates. If 131,072 does not fit, take
  the embedding off the card first, and then the long-window step moves back to the rental. The surrogate is still
  worth having for everything else.
- **It is one more thing to maintain.** A surrogate that drifts from the real models is worse than none, because a green
  local run stops meaning anything. It gates correctness only, and a stale golden has to be restamped before use.
