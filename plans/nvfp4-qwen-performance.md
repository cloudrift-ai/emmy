# NVFP4 Qwen bug investigation and serving exploration (2026-09-28)

This investigation documents compiler bugs found while exploring Qwen3-8B and Qwen3.8-27B-NVFP4, together with what
we could establish about their serving paths. The linked reports describe observable failures and unresolved causes.

IR dumps are evidence for failures such as duplicated contractions, unavailable schedules, codegen crashes and
incorrectly nested output loops. Those findings can be investigated without a GPU benchmark; some paths fail before
execution is possible. The investigation therefore includes IR-only checks as well as selected kernel and layer runs.

The RTX 5080 Laptop GPU gives only a rough indication of performance. We retain the hand-picked knobs and timing
notes as possible starting points for actual tuning after the bugs are addressed. They are secondary to the bug
findings and do not establish production performance or complete model serving.

## Bug reports

The reports and completed fixes below describe the investigation's reproducers, IR observations and unresolved questions.
Several reports group related symptoms whose causes may need further investigation.

Evidence was reviewed at `a98fd4f8` on 2026-09-28 using saved IR and error logs. The staging and packed-cut repros were
also compiled afresh without GPU execution. The flash follow-up checked fresh Tile IR and emitted CUDA without GPU
execution. Historical GPU timings below have not been independently revalidated. The inline quantization validation report was also checked against
the parent/worker binding and strict-reference code; its GPU results were not rerun.

The overview is ordered by expected importance for Qwen3.8 NVFP4 serving at parity with vLLM defaults, with likely
investigation and implementation effort also considered. This is a provisional assessment, not a measured comparison
with vLLM or an implementation sequence. DeltaNet blocks native serving; duplicate projections and repeated
attention/encode work directly threaten runtime performance. Their fixes may span lowering and scheduling. Pin
interference follows because it obstructs combining the explored schedules; the follow-up confirms that existing
site scopes do not select a piece’s WORK. The inline strict mismatch appears more localized, but its impact on
checkpoint validation is unestablished. Packed-cut staging may reuse existing transport support; native fp4 TMA needs a new
staging path and has no demonstrated gain over cp.async here. The computed-f16 TMA gap has a cut workaround and is
less directly tied to the W4A4 target. Unnecessary NVFP4 decode LUTs come last: every preceding issue is more pressing
and is provisionally expected to carry a larger performance penalty or deployment impact. The LUT report establishes
avoidable memory accesses, not a measured speedup. Effort and relative importance may change as causes are established.

| Report | Observed behavior |
| --- | --- |
| [Encode cuts duplicate projections](nvfp4-qwen-performance/fp4-encode-recomputes-producer.md) | Separate Tile IR pieces repeat full-K contractions over the same weights—gate/up three times and o_proj four times. Some copies use identical layouts. |
| [Attention and encode repeat work](nvfp4-qwen-performance/qwen38-attention-not-flash-and-encode-shape.md) | The normal Qwen cut path misses a softmax rewrite. Q/K cuts plus another rewrite expose partial carriers but emit scalar code with only 16 active consumer threads. The original P·V pieces repeat `exp`/division; encode repeats each group’s maximum eight times. |
| [Global pin interference and slow compilation](nvfp4-qwen-performance/pins-cannot-target-one-piece.md) | A global `WORK=t128` pin removes tensor-core TILEs from neighboring matmul pieces. Unpinned Tile IR compilation takes about 12 minutes in concurrent runs. Scoped WORK keys leave the writer serial; WORK is read only as a bare key. The expensive pass is not identified. |
| [Inline quantized strict comparison uses a mismatched reference](nvfp4-qwen-performance/strict-fails-for-inline-quantize-programs.md) | Unseeded inline benchmarks bind Emmy’s packed weights from the parent checkpoint but rebuild eager’s weights in the worker. Strict comparison still uses unquantized eager at 1e-3; reported seeded runs also fail. |
| ✅ [Packed cut pieces lose staging](https://github.com/cloudrift-ai/emmy/pull/966) | Tile IR puts the decoded weight before the computed activation. Only no staging and `d1/smem` remain; a `d2/smem-async` pin fails. B-packed-cut-staging: [#966](https://github.com/cloudrift-ai/emmy/pull/966). Fixed: a re-formed piece with such a contraction is lowered again inside its grid loops, so its grid follows the output layout and `x + 1` is A; each piece is offered all 16 cp.async/TMA `STAGE` values and both transports build and match the decoded oracle on sm_120. The V100 goldens it moved were re-recorded or retuned. |
| ✅ [Native fp4 TMA](https://github.com/cloudrift-ai/emmy/pull/971) | The fp4 contraction accepts `d3/smem-async` but rejects `d2/smem-tma`. W4A16 emits TMA copies of packed weight bytes. B-fp4-tma: [PR #971](https://github.com/cloudrift-ai/emmy/pull/971). Fixed there: stored codes and scales stage by TMA at `d1`–`d4`, with and without `/p2`, for one or two weights, bit-identical to cp.async on sm_120; computed activation codes keep cp.async. TMA is even at decode and about 25% slower at prefill on a 5090. |
| ✅ [Computed f16 activation with weight TMA](https://github.com/cloudrift-ai/emmy/pull/968) (B-computed-f16-tma, [PR #968](https://github.com/cloudrift-ai/emmy/pull/968)) | A contraction over `x + 1` offers only no staging, `d1/smem` and `d2/smem`; the TMA pin fails unless the activation is cut out. **Fixed in PR #968:** the compute fill keeps the activation while TMA copies the stored weights under `d1`/`d2` `smem-tma`, with or without `/p2`, for one or more weights; outputs are bit-identical to the compute fill's cp.async stages on an RTX 5090. On the tried hand pins it is 4–20 % slower than the best existing schedule. |
| [NVFP4 decoding unnecessarily uses LUTs](nvfp4-qwen-performance/nvfp4-decode-luts.md) | Scalar decoding reads global-memory byte-to-pair tables; staged W4A16 MMA reads a constant-memory nibble table. Direct E2M1 conversion can remove those accesses. Cache costs are estimated; no speedup has been measured. Lowest priority. |

### Follow-up checks and boundaries

The LUT follow-up on 2026-09-29 reproduced scalar W4A4, staged W4A16 MMA and native W4A4 MMA at Loop, Tile and CUDA
stages in this PR's existing worktree, whose compiler matches `a98fd4f8`. It adds no GPU timing or correctness claim.

The quick CPU-only follow-up confirms two localized defects: WORK cannot be hand-pinned to the desired cut piece
through existing site scopes, and renaming a one-source B `FragmentRepack` drops its role and breaks rendering.
The pin report includes a fresh Qwen Tile IR excerpt; the GDN report includes the isolated repack reproducer. This
neither identifies the slow compile’s cause nor proves that preserving the repack role fixes all GDN compilation.

The three staging symptoms reach different mechanisms. Native W4A4 explicitly rejects TMA in its block-scaled stage;
computed-f16 activations enter the compute-fill path without a TMA offer for stored weights; weight-first cut pieces
fail to match the existing packed-weight staging path. They share staging infrastructure, but no common root cause
has been established. The inline strict problem instead concerns parent/worker weight binding and reference choice;
it is independent of schedule-pin scope.

[PR #882](https://github.com/cloudrift-ai/emmy/pull/882) and
[PR #880](https://github.com/cloudrift-ai/emmy/pull/880) are already in the investigated main revision. Ten focused
CPU-only regressions pass: multi-channel W4A16 copy staging, packed sibling-reduction merging, and the multi-channel
W4A4 cp.async offer remain supported. The reports describe gaps beyond those fixes, not their absence. No new GPU
execution or performance measurement was needed for these checks.

## Exploration scope

Two models, compiled and partly benched at `a98fd4f8` on an RTX 5080 Laptop GPU (sm_120, 16 GB):

- `Inferact/Qwen3.8-27B-NVFP4`: IR dumps and single-kernel benches. The model does not fit on the card.
- `Qwen/Qwen3-8B` against `nvidia/Qwen3-8B-NVFP4`: IR dumps and benches of layer programs.

Hand pins helped distinguish missing compiler support from poor schedule choices. The default (greedy) pick had no
measured rows for this card and performed poorly or failed in the examined cases. The explored choices and their
rough timings are preserved in [the performance reference](#exploratory-tuning-and-performance-reference).

Both NVFP4 checkpoints are W4A4:

- The weights are 4-bit e2m1 values, two per byte, each 16 of them sharing one e4m3 block scale.
- The activations are quantized to the same format before each quantized linear. Only their per-tensor scale is static;
  the per-16 block scales are computed at run time.

Emmy spells this activation *encode* as `to_f4e2m1` ops that write packed `f4e2m1x2` buffers. A quantized matmul can run
on the *fp4 cell*, the native instruction `mma_m16n8k64_e2m1_f32`, which multiplies packed codes and applies the block
scales in hardware. A matmul on the *scalar tier* uses no tensor cores; each thread computes its own output elements.

## Serving exploration

These observations describe the extent of the exploration at `a98fd4f8`.

| Model | What was explored | What this establishes about serving |
| --- | --- | --- |
| Qwen3-8B, 16-bit and NVFP4 | Captured and ran selected per-layer serving programs and isolated matmuls. Some comparisons lacked a usable correctness reference. | Parts of the serving compilation path were exercised. These experiments do not demonstrate a complete, validated model serving run. The 16-bit model does not fit the 16 GB card. |
| Qwen3.8-27B-NVFP4 | Inspected full-attention and DeltaNet layer IR and ran selected kernels individually. | Native emmy serving is blocked by DeltaNet capture and compilation failures. The whole checkpoint also does not fit the test card, so single-kernel results do not demonstrate serving. |

For Qwen3-8B, the captured layer programs are serving twins, produced by `scripts/capture_gen_twins.py`. There are two
per layer: `pre` covers input RMSNorm, q/k/v projections and per-head q/k norm; `post` covers o_proj plus residual,
RMSNorm, gate/up plus SiLU, and down plus residual. Attention runs outside emmy in this serving path and is in neither
twin. The recorded layer timings therefore leave out attention and the rest of the serving system.

Qwen3.8-27B has 48 gated DeltaNet layers and 16 full-attention layers. Individual full-attention projection kernels
reached the native fp4 instruction after cuts, while the attention computation and DeltaNet paths exposed the bugs
below. Shipped Qwen3.8 recipes on V100 use external kernels for GDN; that is a different serving path and does not
validate this NVFP4 checkpoint on sm_120 with emmy's own kernels.

## Observed problems

1. **GDN layers (Qwen3.8).** Five separate failures:
   - **No trace below 64 tokens.** The Hugging Face chunked delta rule pads the sequence to a multiple of 64:
     `aten.pad supports only explicit zero-width padding, got [0, 0, 0, 48]`. The 16-bit `Qwen/Qwen3.8-27B` fails the
     same way, and `recipes/Qwen3.8-27B-AWQ-INT4/RESULTS.md` records this failure at one token.
   - **No serving twin.** `emmy/serving/twins.py` refuses the model: "blocks whose token mixer is not attention … have
     no serving program yet".
   - **A CUDA render crash at 512 tokens.** `k_matmul_reduce_81b2ae` hits `assert len(self.srcs) == 2` in
     `FragmentRepack.render` (`emmy/compiler/ir/kernel/ir.py`). Kernel IR shows a one-source repack without `role=b`.
     A CPU-only probe confirms that `_rewrite_kind` drops the B role and creates this assertion failure. The exact
     checkpoint node’s path through the rewrite and complete-kernel recovery remain unverified.
   - **A runaway serial loop.** The input-projection kernel's last piece, the one that stores its outputs,
     nests three sibling output sweeps into one serial loop nest, about 1.3×10¹⁷ iterations on one thread.
   - **The 16-bit input projections stay on the scalar tier.** The unquantized projections and convolution together
     hold about 168 MB per layer. The tensor-core fragment loaders read a contraction's first operand as the matrix
     whose K runs contiguously. Here the first operand is the K×N weight, and the four taps of the causal convolution
     come in as four more first-operand inputs.
   - Also slow: the chunk triangular solve `k_slice_unsqueeze_reduce_b17b4d` takes 16 ms at 64 tokens.

   The shipped Qwen3.8-27B recipes (V100) serve GDN outside emmy: Triton kernels for prefill,
   `flash_qla_sm70_gdn_strided` for decode.
2. **The fp4 re-encode recomputes its producer matmul (both models).** An encode reads the producer's output three
   times: as code pairs (two codes per byte), and twice as 16-value blocks (block maximum for the codes and for the
   stored scale). The full-projection cut materializes the producer once per read, so each copy streams the weight
   again. Some copies even use the same layout with identical bodies.

   | model | o_proj | gate/up | q_proj | v_proj |
   | --- | ---: | ---: | ---: | ---: |
   | Qwen3-8B | 4× | 3× (scalar tier) | | |
   | Qwen3.8-27B | 4× | 3× | 4× | 3× |

   Suspected: `_cluster_value_seams` in `emmy/compiler/pipeline/passes/tile/_cut.py`. It merges the cut points (seams)
   that materialize one value only when their captured axes line up, and it excludes output-owning seams from merging.
   Qwen3-8B `post` takes 1,266 µs against about 167 µs summed over separate pinned matmuls. That comparison also
   changes encode and scheduling behavior; it does not attribute the entire gap to duplicate materialization.
3. **Encode kernels launch one block per packed byte.**
   - The encode kernel's schedule `WORK=t128,REDUCE=coop` launches one 128-thread block per packed output byte, and
     each block reduces the 16-value group its byte belongs to, so every group is reduced 8 times. That is 139,264
     blocks at 16 tokens for a Qwen3.8 gate/up output, and 1–3 million at 512 tokens on Qwen3-8B, costing 0.9–3 ms.
   - Pinning `REDUCE=` (each thread reduces its groups alone) takes it to 11–30 µs.
4. **Attention (Qwen3.8 full-attention layers).**
   - The normal cut path skips softmax recognition on a consumer with output sweeps. Q/K cuts plus a repeated
     recognition pass form partial carriers, but all three emitted kernels lack tensor-core instructions and the
     attention consumer has only 16 active threads. This is not a practical Qwen NVFP4 route. The report includes the
     CPU-only repro, actual IR and CUDA excerpts, and a smaller projection-related recognition failure.
   - P·V is a reduce over keys with one 128-thread block per output element, recomputing `exp(score − max)/sum` per
     head-dim column. Four pieces compute it: 111–149 µs each for the two that run as reduces.
5. **Global hand pins interfere across pieces.** After a cut, each piece forks over its own schedule. A global pin
   meant for one piece lands on its siblings too. Existing site scopes do not solve the WORK example: WORK is
   kernel-level and read only under its bare key. Two examples:
   - The o_proj kernel's residual-add piece is offered `WORK=t4` to `t512`, but it runs 361.5 µs on one thread per row
     under the empty layout. A `WORK=t128` pin would spread it, as it does in the RMSNorm + encode kernel
     (362 → 5.1 µs).
     But the same pin also moves the o_proj matmul pieces off tensor cores.
   - An fp4 TILE pin leaves the QK score piece on the scalar tier at 242 µs. The greedy pick puts it on a 16-bit tile
     at 37 µs.

   Goldens can hold per-piece rows (child-identity schedule receipts, `GLOSSARY.md`). A fresh CPU-only compile with
   WORK scoped to the writer’s piece name leaves its output sweep serial. Attention TILE targeting remains a
   separate, unexhausted question.

   Also: the unpinned tile compile of the Qwen3.8 attention kernel takes about 12 minutes (717 s, with 4 compiles
   sharing the machine). With the cut it takes 71.5 s, and with the cut plus an fp4 TILE pin 13.7 s.
6. **q/k projections with the fused per-head norm (Qwen3-8B)** stay on the scalar tier: `t128 coop`, 55 µs for q at one
   token. Tensor-core pins either do not realize or fail: "head fold is nested inside the projection's sweep loop".
7. **Staging gaps on the fast paths.**
   - The fp4 cell offers cp.async only, never TMA. `_block_scaled_warp_stage` in `emmy/compiler/ir/schedule/staging.py`
     says "the four-descriptor TMA box copy is not built". Several fp4 STAGE pins do not resolve (`k2` tiles, `d3`/`d4`
     at some shapes).
   - A cut piece over W4A16 weights gets the weight as its first operand and loses the packed byte-slab stage.
   - A 16-bit matmul whose activation is computed in the kernel gets no TMA for its stored weights.
8. **lm_head (Qwen3-8B, 16-bit, 1.25 GB)** takes 4.9 ms per token pinned, about as long as eager (about 250 GB/s). That
   is a performance observation, not an isolated compiler defect. The token-time estimates below are extrapolations.

## Exploratory tuning and performance reference

The material below is retained in case it is useful after the bugs are addressed and actual tuning begins. It records
one quick pass of hand-picked choices, including unsuccessful paths and missing correctness checks. No measured
goldens were recorded. Changes to fusion, cuts or scheduling may invalidate the pins or change which choices help.

Laptop power limits, clock variation and cache reuse make these timings rough context. They can suggest which
experiments were promising; they do not establish a tuned baseline, expected speedup from a fix or serving latency.

### Measurement caveats

- **Power cap.** The laptop was power-capped during the runs. One reading showed the GPU at about 20 W with a 9 GHz
  memory clock instead of 14 GHz; another showed a 14 GHz memory clock with the graphics clock near 400 MHz. Measured
  bandwidth: 259–641 GB/s for reads, up to 715 GB/s for a copy (reads plus writes), against 896 GB/s on paper. Numbers
  swing by about 30%.
- **L2-hot weights.** The device reports a 48 MB L2 cache. A single-matmul program reuses one weight across benchmark
  iterations, so a weight that fits in L2 stays there, and its time can beat DRAM bandwidth.
- **Inline quantized comparison has a mismatched reference.** The
  [strict report](nvfp4-qwen-performance/strict-fails-for-inline-quantize-programs.md) traces two problems in
  `-c … --quantize --bench --strict`: the worker can draw different eager weights from those quantized by the parent,
  and strict validation compares the quantized graph against unquantized eager at 1e-3. Seeding reportedly reduces
  the error but does not make these repros pass. The source confirms the reference problems, not that the kernels
  are correct. Checkpoint-based validation was not examined and must be assessed separately.
- **What a ✓ means.** Correctness below comes from the `--ab` output check. `emmy run --bench --ab KNOBS` compiles the
  program once more with `KNOBS` added and flags a wrong answer when that compile's outputs differ from the greedy
  compile's by more than 5% of the greedy output's largest value. Both compiles use the same `EMMY_KNOBS`. The greedy
  compile is often already on the fp4 cell. Rows marked "vs scalar" come from runs whose greedy compile was pinned to
  the scalar tier, so they compare the fp4 cell against scalar code. These are relative comparisons, not independent
  proofs of correctness. A run that says "wrong-answer reference unusable" has not passed a correctness check.
- **Global pins reach every kernel.** The experiments did not combine the best knobs of each piece (blocker 5).
  The CPU follow-up confirms that site-scoped WORK does not select the desired writer layout.

### Qwen3-8B: exploratory timings

Decode, one token per step, µs per layer. Eager is a PyTorch layer on the same card, from a separate timing script.

| program | 16-bit, pinned | eager | NVFP4, pinned twin | NVFP4, sum of pinned single matmuls |
| --- | ---: | ---: | ---: | ---: |
| pre | 109 | 108 | 135 | ≈ 51 |
| post | 422 | 622 | 1,266 | ≈ 167 |

Illustrative time-per-token extrapolations, not measured serving latency: 36 layers × (pre + post + 20 µs torch
attention at context 1024) + 4.9 ms lm_head. The alternative split is hypothetical, and summing isolated matmuls omits
some surrounding work and may have different cache behavior.

| case | ms per token |
| --- | ---: |
| 16-bit, pinned twins | ≈ 24.7; the 16-bit weights do not fit the 16 GB card for serving |
| NVFP4, pinned twins in this investigation | ≈ 56 |
| NVFP4, if each layer split like the single-matmul programs | ≈ 13.5 |
| NVFP4, bandwidth floor (5.15 GB at 896 or 640 GB/s) | 5.8 or 8.0 |

**Some hand-pinned matmuls were promising on this setup.** At one token, emmy's pinned single matmuls sum to about
218 µs per layer, and `torch._scaled_mm`'s fp4 path takes 611 µs for the same seven matmuls. 16-bit matmuls with pins run at or near eager.
Gate+up at one token takes 265 µs end to end, about 760 GB/s. k/v at 512 tokens and lm_head are slightly slower than
eager.

**Duplicate projections and inefficient encode schedules contribute substantial avoidable work** (blockers 2 and 3).
The measurements do not isolate how much of the complete NVFP4-versus-16-bit gap each accounts for.

**Prefill** at 512 tokens with 16-bit pinned twins takes about 105 ms (eager about 174 ms). The NVFP4 `pre512` twin
alone takes 2.1 ms per layer.

### Qwen3.8-27B-NVFP4: exploratory timings

Selected kernels of full-attention layer 3, measured individually with hand pins:

- **down_proj:** 407 µs at 512 tokens, about 223 TFLOPS. At 16 tokens, 41 µs. Its roughly 47 MB of weights fits the
  L2, so this 16-token number beats DRAM bandwidth.
- **Full-attention layer at 16 tokens,** with pins:

  | kernel | µs | weight-read floor, µs |
  | --- | ---: | ---: |
  | attention | 531–949 | 86 |
  | gate/up | 585 | 200 |
  | o_proj | 626 | 37 |

### Recorded knobs

These choices are possible starting points for later tuning. Timings are the medians of `emmy run … --bench`: `--warmup 20 --iters 200` for Qwen3-8B, `--warmup 5 --iters 20` for
Qwen3.8. Unless a row says otherwise, times are end to end.

Tile shorthand:

- `F` = `mma_m16n8k64_e2m1_f32`, the fp4 cell
- `H` = `mma_m16n8k16_f16_f32`
- `HH` = `mma_m16n8k16_f16_f16`, which accumulates in f16

For the ✓ column, see the caveats above.

#### Qwen3-8B, 16-bit single matmuls

Each program is `-c "nn.Linear(K, N, bias=False).half()(torch.randn(M, K, dtype=torch.float16))"`. Gate+up is a
two-linear SiLU module.

| matmul | M | EMMY_KNOBS | µs | eager µs |
| --- | ---: | --- | ---: | ---: |
| q/o 4096→4096 | 1 | `WORK=w1x2,TILE=H/f1x2/k4,STAGE=d4/smem-tma,REDUCE=g4k` | 30.7 | 46 |
| k/v 4096→1024 | 1 | same | 10.2 | 9 |
| gate+up+SiLU | 1 | same | 265.2 | 365 |
| down 12288→4096 | 1 | same | 132.9 (matmul kernel only) | 137 |
| q/o | 32 | `WORK=w1x4,TILE=H/f2x2/k4,STAGE=d4/smem-async,REDUCE=g4k` | 24.7 | 41 |
| k/v | 32 | `WORK=w2x4,TILE=H/f1x1/k4,STAGE=d4/smem-async,REDUCE=g4k` | 10.2 | 12 |
| gate+up | 32 | `WORK=w1x4,TILE=H/f2x2/k2,STAGE=d3/smem-async,REDUCE=g2k` | 287.2 | 346 |
| down | 32 | same | 135.2 | 152 |
| q/o | 512 | `WORK=w2x2,TILE=H/f2x4/k4,STAGE=d3/smem-async` | 249.7 | 312 |
| k/v | 512 | same | 87.4 | 66 |
| gate+up | 512 | `WORK=w2x2,TILE=H/f4x4/k2,STAGE=d3/smem-async` | 1,696.6 | 1,782 |
| down | 512 | same | 866.4 | 870 |
| lm_head 4096→151936 | 1 | `WORK=w1x2,TILE=H/f1x2/k4,STAGE=d4/smem-tma` | 4,913.6 | 4,734 |

#### Qwen3-8B, NVFP4 single matmuls

These are `--quantize nvfp4` programs. `EMMY_KNOBS` held `PLACE@map.1/map=cut,PLACE@map.2/map=cut`, which cuts the
activation encode into its own kernel. The schedule knobs below went in as `--ab` rows.

| matmul | M | schedule knobs | µs | ✓ |
| --- | ---: | --- | ---: | --- |
| q/o | 1 | `WORK=w1x8,TILE=F/f1x1/k4,STAGE=d4/smem-async` | 26.7 | vs scalar |
| k/v | 1 | `WORK=w1x8,TILE=F/f1x1/k4,STAGE=d4/smem-async,REDUCE=g4k` | 12.3 | ✓ |
| gate+up | 1 | `WORK=w1x8,TILE=F/f1x1/k4,STAGE=d4/smem-async` | 87.5 | ✓ |
| down | 1 | `WORK=w1x8,TILE=F/f1x1/k4,STAGE=d4/smem-async,REDUCE=g4k` | 52.8 | vs scalar |
| q/o | 32 | `WORK=w1x4,TILE=F/f2x2/k4,STAGE=d4/smem-async,REDUCE=g4k` | 24.6 | vs scalar |
| k/v | 32 | `WORK=w1x8,TILE=F/f2x1/k4,STAGE=d4/smem-async,REDUCE=g4k` | 14.3 | ✓ |
| gate+up | 32 | `WORK=w1x4,TILE=F/f2x1/k4,STAGE=d4/smem-async,REDUCE=g4k` | 125.5 | ✓ |
| down | 32 | `WORK=w1x4,TILE=F/f2x2/k4,STAGE=d4/smem-async,REDUCE=g4k` | 46.4 | ✓ |
| q/o | 512 | `WORK=w2x2,TILE=F/f2x4/k4,STAGE=d4/smem-async,REDUCE=` | 137.1 | ✓ |
| gate+up | 512 | `WORK=w2x2,TILE=F/f2x4/k4,STAGE=d3/smem-async` | 1,803 | ✓ |
| down | 512 | `WORK=w2x2,TILE=F/f4x4/k4,STAGE=d3/smem-async,REDUCE=` | 260.5 | ✓ |

At 512 tokens, `REDUCE=` also fixes the encode kernel (blocker 3). k/v at 512 took 994–1153 µs with the slow encode and
was not re-measured with the fix. Some gate+up pins at 512 (`f4x4`, `f4x8`, `d4`) fail to compile: "1 node(s) left
un-lowered — the deterministic compile exhausted its fallbacks and has no kernel for them: - 'mul'".

#### Qwen3-8B layer programs (serving twins)

`scripts/capture_gen_twins.py --model <model> --out <dir>` writes the twins. `--decode-bucket` (1 or 32) and
`--prefill-bucket` (512) set the widths, and `--no-symbolic` skips the any-width twins.

In these runs the cut knobs went in `EMMY_KNOBS` and the schedule knobs as `--ab` rows.

| twin | cut knobs | schedule knobs | µs | ✓ |
| --- | --- | --- | ---: | --- |
| 16-bit pre, 1 token | `PRE` | `WORK=w1x8,TILE=H/f1x1/k4,STAGE=d4/smem-async,REDUCE=g4k` | 109.0 | ✓ |
| 16-bit pre, 32 tokens | `PRE` | `WORK=w1x4,TILE=H/f2x2/k4,STAGE=d4/smem-async` | 82.6 | ✓ |
| 16-bit post, 1 token | `POST1` | `WORK=w1x8,TILE=H/f1x1/k4,STAGE=d4/smem-async` | 421.8 | ✓ |
| 16-bit post, 32 tokens | `POST` | `WORK=w1x4,TILE=H/f2x2/k4,STAGE=d4/smem-async` | 451.0 | ✓ |
| 16-bit pre, 512 tokens | `PRE512` | `WORK=w2x2,TILE=HH/f4x4/k2,STAGE=d3/smem-async` | 424.0 | ✓ |
| 16-bit post, 512 tokens | `POST` | `WORK=w2x2,TILE=HH/f4x4/k2,STAGE=d3/smem-async` | 2,381.3 | ✓ |
| NVFP4 pre, 1 token | `FPRE` | `WORK=w1x8,TILE=F/f1x1/k4,STAGE=d4/smem-async` | 135.2 | ✓ |
| NVFP4 post, 1 token | `FPOST` | `WORK=w1x8,TILE=F/f1x1/k4,STAGE=d4/smem-async` | 1,266.0 | none (the greedy compile failed to bench) |
| NVFP4 pre, 512 tokens | `FPRE` | `WORK=w2x4,TILE=F/f4x4/k4,STAGE=d2/smem-async` | 2,112.8 | ✓ |

```
PRE    = PLACE@map.1/map.1/map.1/reduce.1/inner.2/map=cut,PLACE@map.1/map.1/map.1/reduce.1/inner.2/map.3/map=cut,PLACE@map.1/map.1/map.1/reduce.1/inner=cut,PLACE@map.3/map.1/map.1/reduce.1/inner=cut,PLACE@map.2/inner=cut,PLACE@map.1/map=cut,PLACE@map.3/map=cut,PLACE@map.1/map.1/map=cut,PLACE@map.3/map.1/map=cut
PRE512 = PLACE@map.1/map.1/map.1/reduce.1/inner.2/map=cut,PLACE@map.1/map.1/map.1/reduce.1/inner.2/map.3/map=cut,PLACE@map.1/map.1/map.1/reduce.1/inner=cut,PLACE@map.2/map.1/map.1/reduce.1/inner=cut,PLACE@map.3/inner=cut,PLACE@map.1/map=cut,PLACE@map.2/map=cut,PLACE@map.1/map.1/map=cut,PLACE@map.2/map.1/map=cut
POST1  = PLACE@map.1/inner.2/map=cut,PLACE@map.1/inner.2/map.1/inner.1/map=cut,PLACE@map.1/inner.2/map.1/inner.1/map.1/inner=cut,PLACE@map.1/inner.2/map.1/inner.1/map.4/map=cut
POST   = PLACE@map.1/inner.1/map=cut,PLACE@map.1/inner.1/map.1/inner.1/map=cut,PLACE@map.1/inner.1/map.1/inner.1/map.1/inner=cut,PLACE@map.1/inner.1/map.1/inner.1/map.4/map=cut
FPRE   = PLACE@map.2/map.1/map=cut,PLACE@map.1/map=cut,PLACE@map.2/map=cut,PLACE@map.1/map.1/reduce.1/inner=cut
FPOST  = PLACE@map.1/map=cut,PLACE@map.1/map.2/reduce.1/inner=cut,PLACE@map.1/map.2/reduce.4/map.1/reduce=cut,PLACE@map.1/map.2/reduce.4/map.1/reduce.1/inner=cut,PLACE@map.2/map=cut,PLACE@map.2/map.2/reduce.1/inner=cut,PLACE@map.2/map.3/inner=cut,PLACE@map.3/map=cut,PLACE@map.1/map.3/inner=cut
```

Site paths are positions in the twin's Fold tree at `a98fd4f8`. They move when fusion changes.

#### Qwen3.8-27B-NVFP4 single kernels

These are kernels of layer 3, compiled and benched individually from trace inventories. Kernel names are at
`a98fd4f8`. The records do not imply a complete model serving run.

| kernel, tokens | EMMY_KNOBS | µs | ✓ |
| --- | --- | ---: | --- |
| down + residual (`k_linear_353cf1`), 512 | `WORK=w4x2,TILE=F/f2x4/k4,STAGE=d3/smem-async` | 407 | ✓ |
| down + residual (`k_linear_9fe2fd`), 16 | `WORK=w1x4,TILE=F/f1x2/k8,STAGE=d3/smem-async` | 41 | vs scalar |
| same, with a K split | `WORK=w1x4,TILE=F/f1x2/k4,STAGE=d3/smem-async,REDUCE=g4k` | 42.9 | vs scalar |
| gate/up + SiLU + encode (`k_linear_reduce_2d1c79`), 16 | `GATEUP_CUT,TILE=F/f1x2/k8,STAGE=d3/smem-async` | 585 | ✓ |
| o_proj + residual + RMSNorm + encode (`k_linear_mean_reduce_fd2717`), 16 | `OPROJ_CUT,TILE=F/f1x2/k8,STAGE=d3/smem-async` | 626 | none |
| attention (`k_sdpa_linear_mean_reduce_c6c239`), 16 | `ATTN_CUT,TILE=F/f1x1/k8,STAGE=d3/smem-async` | 531–949 | ✓ vs a partly scalar compile |
| RMSNorm + encode (`k_mean_01c219`), 16 | `PLACE@map.1/map.2/reduce.3/map.1/reduce=cut,PLACE@map.1/map.2/reduce=cut,PLACE@map.2/map.2/reduce=cut,WORK=t128` | 19.8 | none (the greedy reference was unusable) |

```
GATEUP_CUT = PLACE@map.1/map=cut,PLACE@map.1/map.2/inner=cut,PLACE@map.1/map.3/reduce.1/inner=cut,PLACE@map.2/map=cut,PLACE@map.2/map.2/reduce.1/inner=cut
OPROJ_CUT  = PLACE@map.1/map=cut,PLACE@map.1/map.1/inner=cut,PLACE@map.2/map=cut,PLACE@map.2/map.2/reduce.1/inner=cut,PLACE@map.2/map.2/reduce.4/map.1/reduce=cut,PLACE@map.3/map=cut,PLACE@map.3/map.2/inner=cut,PLACE@map.3/map.3/reduce.1/inner=cut
ATTN_CUT   = PLACE@map.1/map=cut,PLACE@map.1/map.1/inner=cut,PLACE@map.1/map.2/inner=cut,PLACE@map.1/map.2/inner.1/map.1/inner=cut,PLACE@map.1/map.2/inner.2/map.1/inner=cut,PLACE@map.1/map.2/inner.2/map.1/inner.1/map.11/map.1/reduce=cut,PLACE@map.1/map.2/inner.2/map.1/inner.1/map.11/map.1/reduce.1/inner=cut,PLACE@map.1/map.2/inner.2/map.1/inner.2/map.11/map.1/reduce=cut,PLACE@map.1/map.2/inner.2/map.1/inner.2/map.11/map.1/reduce.1/inner=cut,PLACE@map.1/map.2/inner.2/map.2/reduce=cut,PLACE@map.1/map.2/inner.2/map.3/reduce=cut,PLACE@map.1/map.3/inner=cut,PLACE@map.1/map.4/reduce.1/inner=cut,PLACE@map.1/map.4/reduce.2/inner=cut,PLACE@map.1/map.4/reduce.2/inner.1/map.1/inner=cut,PLACE@map.2/map=cut,PLACE@map.2/map.1/reduce.1/inner=cut,PLACE@map.2/map.1/reduce.2/inner=cut,PLACE@map.2/map.1/reduce.2/inner.1/map.1/inner=cut
```

The three cut strings spell full-projection cuts. `GATEUP_CUT` and `OPROJ_CUT` also apply at one token. To reproduce:

```sh
emmy trace Inferact/Qwen3.8-27B-NVFP4@6128240ebaf4eaa7bad2b3d1c72c37d677c5f462 --layer 3 --seq-len 16 --target sm_120 -o l3_s16.json
EMMY_KNOBS=… emmy run --golden l3_s16.json --realization <kernel> --bench
```
