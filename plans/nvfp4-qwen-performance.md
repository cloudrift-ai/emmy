# NVFP4 on sm_120: Qwen3.8-27B-NVFP4 and Qwen3-8B, findings and next steps (2026-09-28)

## Bug tracker

This table is the shared progress record. The linked directory holds the final reports and reproducible acceptance
criteria; it does not hold a second status list. Evidence was reviewed at `a98fd4f8` on 2026-09-28 using saved IR and
error logs. B01–B03 also had their central repros compiled afresh without GPU execution. The flash follow-up checked
fresh Tile IR only. Historical GPU timings below have not been independently revalidated.

| ID | Bug or investigation | Status | Main area and overlap |
| --- | --- | --- | --- |
| B01 | [Native fp4 lacks TMA](nvfp4-qwen-performance/fp4-cell-no-tma.md): the fp4 contraction accepts `d3/smem-async` but rejects `d2/smem-tma`. W4A16 emits TMA copies of packed weight bytes. | Open | Block-scaled code/scale transport |
| B02 | [Computed f16 activation blocks weight TMA](nvfp4-qwen-performance/f16-computed-activation-no-tma.md): a contraction over `x + 1` offers only no staging, `d1/smem` and `d2/smem`; the TMA pin fails unless the activation is cut out. | Open | Compute fill plus stored-weight transport |
| B03 | [Packed cut pieces lose staging](nvfp4-qwen-performance/packed-cut-piece-operand-order.md): Tile IR puts the decoded weight before the computed activation. Only no staging and `d1/smem` remain; a `d2/smem-async` pin fails. | Open | Fold orientation and packed-weight recognition; overlaps B07 |
| B04 | [Global pin interference and slow compilation](nvfp4-qwen-performance/pins-cannot-target-one-piece.md): `WORK=t128` also removes tensor-core TILEs from neighboring matmul pieces. Unpinned Tile IR compilation takes about 12 minutes in concurrent runs. Site-scoped targeting is untested; the expensive pass is not identified. | Investigate | Schedule sites and fork search |
| B05 | [Encode cuts duplicate projections](nvfp4-qwen-performance/fp4-encode-recomputes-producer.md): separate Tile IR pieces repeat full-K contractions over the same weights—gate/up three times and o_proj four times. Some copies use identical layouts. | Open | Cut materialization; overlaps attention duplication in B06 |
| B06 | [Attention and encode repeat work](nvfp4-qwen-performance/qwen38-attention-not-flash-and-encode-shape.md): the examined Qwen attention IR has no `twist=softmax`; four P·V pieces repeat `exp` and division across output columns. Encode assigns 128 threads per byte and repeats each group's maximum eight times. | Open | Softmax rewrite, reduction reuse and launch layout; overlaps B05 |
| B07 | [GDN compilation and scheduling failures](nvfp4-qwen-performance/qwen38-gdn-layers-fail.md): padding and serving capture raise errors; a one-source `FragmentRepack` crashes CUDA rendering; sibling Tile IR sweeps become nested CUDA loops; input projections lack tensor-core TILEs. | Open | Torch lowering, serving, Kernel IR and output loops; operand orientation overlaps B03 |

There are seven reports; some contain several independently fixable issues. Use `Open`, `Investigate`, `In progress`,
`Blocked`, and `Done`. For a combined report, note partial progress in its status cell and use the report's numbered
fix criteria to identify what remains. Mark the row `Done` only when all its issues have merged fixes and recorded
validation. Land status updates on main when work starts so other developers can see overlapping work.

Missing padding and serving support remain bugs to address for the intended model use. A speculative root cause is
not an implementation requirement. Keep this plan and its report directory while work remains; move lasting validation
into tests or appropriate docs before removing completed planning material. Earlier drafts and local dumps are excluded.

## Baseline

Two models, compiled and partly benched at `a98fd4f8` on an RTX 5080 Laptop GPU (sm_120, 16 GB):

- `Inferact/Qwen3.8-27B-NVFP4`: IR dumps and single-kernel benches. The model does not fit on the card.
- `Qwen/Qwen3-8B` against `nvidia/Qwen3-8B-NVFP4`: IR dumps and benches of layer programs.

The original measurements asked how good emmy's code can be with hand-picked knobs. The default (greedy) pick had no
measured rows for this card and performed poorly or failed in the examined cases. The knobs below come from one
quick pass of hand pins, not a tuning. No goldens were recorded. The long-term home of the knobs worth keeping is golden
rows, recorded on a card that is not power-capped.

Both NVFP4 checkpoints are W4A4:

- The weights are 4-bit e2m1 values, two per byte, each 16 of them sharing one e4m3 block scale.
- The activations are quantized to the same format before each quantized linear. Only their per-tensor scale is static;
  the per-16 block scales are computed at run time.

Emmy spells this activation *encode* as `to_f4e2m1` ops that write packed `f4e2m1x2` buffers. A quantized matmul can run
on the *fp4 cell*, the native instruction `mma_m16n8k64_e2m1_f32`, which multiplies packed codes and applies the block
scales in hardware. A matmul on the *scalar tier* uses no tensor cores; each thread computes its own output elements.

## Measurement caveats

- **Power cap.** The laptop was power-capped during the runs. One reading showed the GPU at about 20 W with a 9 GHz
  memory clock instead of 14 GHz; another showed a 14 GHz memory clock with the graphics clock near 400 MHz. Measured
  bandwidth: 259–641 GB/s for reads, up to 715 GB/s for a copy (reads plus writes), against 896 GB/s on paper. Numbers
  swing by about 30%.
- **L2-hot weights.** The device reports a 48 MB L2 cache. A single-matmul program reuses one weight across benchmark
  iterations, so a weight that fits in L2 stays there, and its time can beat DRAM bandwidth.
- **Unresolved inline quantized comparison.** `emmy run --strict` was reported to fail across the tested schedules
  of `-c … --quantize` programs: scalar, W4A16 and fp4, each with errors as large as the outputs. This suggests a
  shared input or reference problem; it neither identifies the cause nor proves the kernels correct. Checkpoint
  repros must validate their own reference path. Do not assume their `--strict` comparison is unusable.
- **What a ✓ means.** Correctness below comes from the `--ab` output check. `emmy run --bench --ab KNOBS` compiles the
  program once more with `KNOBS` added and flags a wrong answer when that compile's outputs differ from the greedy
  compile's by more than 5% of the greedy output's largest value. Both compiles use the same `EMMY_KNOBS`. The greedy
  compile is often already on the fp4 cell. Rows marked "vs scalar" come from runs whose greedy compile was pinned to
  the scalar tier, so they compare the fp4 cell against scalar code. These are relative comparisons, not independent
  proofs of correctness. A run that says "wrong-answer reference unusable" has not passed a correctness check.
- **Global pins reach every kernel.** The experiments did not combine the best knobs of each piece (blocker 5).
  Whether existing site-scoped pins suffice has not been established; B04 investigates that before adding syntax.

## Qwen3-8B: where things stand

The layer programs below are serving twins: the per-layer programs emmy's serving path compiles.
`scripts/capture_gen_twins.py --model <model> --out <dir>` writes them. `--decode-bucket` (1 or 32) and
`--prefill-bucket` (512) set the widths, and `--no-symbolic` skips the any-width twins. There are two twins per layer:

- `pre`: input RMSNorm → q/k/v projections → per-head q/k norm.
- `post`: o_proj + residual → RMSNorm → gate/up + SiLU → down + residual.

Attention itself runs outside emmy in serving and is in neither twin.

Decode, one token per step, µs per layer. Eager is a PyTorch layer on the same card, from a separate timing script.

| program | 16-bit, pinned | eager | NVFP4, pinned twin | NVFP4, sum of pinned single matmuls |
| --- | ---: | ---: | ---: | ---: |
| pre | 109 | 108 | 135 | ≈ 51 |
| post | 422 | 622 | 1,266 | ≈ 167 |

Estimated time per output token: 36 layers × (pre + post + 20 µs torch attention at context 1024) + 4.9 ms lm_head.

| case | ms per token |
| --- | ---: |
| 16-bit, pinned twins | ≈ 24.7; the 16-bit weights do not fit the 16 GB card for serving |
| NVFP4, pinned twins as they compile today | ≈ 56 |
| NVFP4, if each layer split like the single-matmul programs | ≈ 13.5 |
| NVFP4, bandwidth floor (5.15 GB at 896 or 640 GB/s) | 5.8 or 8.0 |

**The fp4 matmuls themselves are good.** At one token, emmy's pinned single matmuls sum to about 218 µs per layer, and
`torch._scaled_mm`'s fp4 path takes 611 µs for the same seven matmuls. 16-bit matmuls with pins run at or near eager.
Gate+up at one token takes 265 µs end to end, about 760 GB/s. k/v at 512 tokens and lm_head are slightly slower than
eager.

**Duplicate projections and inefficient encode schedules contribute substantial avoidable work** (blockers 2 and 3).
The measurements do not isolate how much of the complete NVFP4-versus-16-bit gap each accounts for.

**Prefill** at 512 tokens with 16-bit pinned twins takes about 105 ms (eager about 174 ms). The NVFP4 `pre512` twin
alone takes 2.1 ms per layer.

## Qwen3.8-27B-NVFP4: where things stand

The model has 64 layers: three gated DeltaNet (GDN) layers, then one full-attention layer, repeating. Kernels were
compiled and benched one at a time from trace inventories. `emmy trace <model> --layer L --seq-len S -o <file>` writes
every kernel of one layer at one width, and `emmy compile|run --golden <file> --realization <kernel>` takes one of them.

- **Fused kernels need the full-projection cut.** Every NVFP4 weight matmul reaches the fp4 cell with cp.async once its
  fused kernel is split this way. The full-projection cut (`GLOSSARY.md`) is one decision of the cut pass. It moves each
  contraction and each output-owning branch of a fused kernel into a kernel of its own, called a *piece*, and it is
  spelled as a set of `PLACE@<site>=cut` keys.
- **down_proj:** 407 µs at 512 tokens, about 223 TFLOPS. At 16 tokens, 41 µs. Its roughly 47 MB of weights fits the
  L2, so this 16-token number beats DRAM bandwidth.
- **Full-attention layer at 16 tokens,** with pins:

  | kernel | µs | weight-read floor, µs |
  | --- | ---: | ---: |
  | attention | 531–949 | 86 |
  | gate/up | 585 | 200 |
  | o_proj | 626 | 37 |

- **The GDN layers do not work** (blocker 1). They are 48 of the 64 layers, and emmy cannot serve this model with its
  own kernels.

## Blockers, most costly first

1. **GDN layers (Qwen3.8).** Five separate failures:
   - **No trace below 64 tokens.** The Hugging Face chunked delta rule pads the sequence to a multiple of 64:
     `aten.pad supports only explicit zero-width padding, got [0, 0, 0, 48]`. The 16-bit `Qwen/Qwen3.8-27B` fails the
     same way, and `recipes/Qwen3.8-27B-AWQ-INT4/RESULTS.md` records this failure at one token.
   - **No serving twin.** `emmy/serving/twins.py` refuses the model: "blocks whose token mixer is not attention … have
     no serving program yet".
   - **A CUDA render crash at 512 tokens.** `k_matmul_reduce_81b2ae` hits `assert len(self.srcs) == 2` in
     `FragmentRepack.render` (`emmy/compiler/ir/kernel/ir.py`). Kernel IR shows a one-source repack without `role=b`.
     Suspected cause: the `_rewrite_kind` overload rebuilds the node without preserving its role. Confirm this with
     a focused reproducer before choosing the fix.
   - **A runaway serial loop.** The input-projection kernel's last piece, the one that stores its outputs,
     nests three sibling output sweeps into one serial loop nest, about 1.3×10¹⁷ iterations on one thread.
   - **The 16-bit input projections stay on the scalar tier.** The unquantized projections and convolution together
     hold about 168 MB per layer. The tensor-core fragment loaders read a contraction's first operand as the matrix
     whose K runs contiguously. Here that
     first operand is the K×N weight, and the four taps of the causal convolution come in as four more first-operand
     inputs.
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
   - None of the three cut sets tried keeps scaled dot-product attention as one kernel with an online softmax. A
     bounded follow-up found no softmax carrier after lift or the projection-only cut, while plain causal GQA at the
     same dimensions did form one. No practical flash route was found; see B06 for the code and IR check.
   - P·V is a reduce over keys with one 128-thread block per output element, recomputing `exp(score − max)/sum` per
     head-dim column. Four pieces compute it: 111–149 µs each for the two that run as reduces.
5. **Global hand pins interfere across pieces.** After a cut, each piece forks over its own schedule. A global pin
   meant for one piece lands on its siblings too. Existing site-scoped pins have not been tested for these cases, so
   the claim that targeting is impossible remains unproven. Two examples:
   - The o_proj kernel's residual-add piece is offered `WORK=t4` to `t512`, but it runs 361.5 µs on one thread per row
     under the empty layout. A `WORK=t128` pin would spread it, as it does in the RMSNorm + encode kernel
     (362 → 5.1 µs).
     But the same pin also moves the o_proj matmul pieces off tensor cores.
   - An fp4 TILE pin leaves the QK score piece on the scalar tier at 242 µs. The greedy pick puts it on a 16-bit tile
     at 37 µs.

   Goldens can hold per-piece rows (child-identity schedule receipts, `GLOSSARY.md`). First try existing site-scoped
   hand pins; add a piece selector only if that investigation demonstrates the need.

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
   is 9% of the NVFP4 time per token today and 36% of the 13.5 ms estimate.

## Knobs worth keeping

Timings are the medians of `emmy run … --bench`: `--warmup 20 --iters 200` for Qwen3-8B, `--warmup 5 --iters 20` for
Qwen3.8. Unless a row says otherwise, times are end to end.

Tile shorthand:

- `F` = `mma_m16n8k64_e2m1_f32`, the fp4 cell
- `H` = `mma_m16n8k16_f16_f32`
- `HH` = `mma_m16n8k16_f16_f16`, which accumulates in f16

For the ✓ column, see the caveats above.

### Qwen3-8B, 16-bit single matmuls

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

### Qwen3-8B, NVFP4 single matmuls

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

### Qwen3-8B layer programs (serving twins)

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

### Qwen3.8-27B-NVFP4 single kernels

These are kernels of layer 3. Kernel names are at `a98fd4f8`.

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

## Suggested work order

The tracker is authoritative for status; this is a suggested order, not a second checklist.

1. Fix the GDN fragment-repack crash and nested output sweeps (B07), then address padding, serving capture and
   projection scheduling. The report gives separate fix criteria for each failure.
2. Remove duplicate producer materializations (B05), then redundant encode reductions and excess blocks (B06).
   Coordinate the changes because both touch the boundary between producers and quantization consumers.
3. Diagnose flash recognition (B06). Investigate existing pin scoping and profile slow compilation (B04), sharing
   changes to cut and schedule interfaces.
4. Fix packed-cut orientation (B03) and the staging gaps (B01–B02).

The baseline also records inline quantized comparison failures, q/k projection with per-head norm, GDN recurrence
costs and lm_head throughput. These remain follow-up observations, not additional bug reports in the overview.
Establish usable correctness references before recording performance claims for affected programs.

For every fix, first reproduce its structural failure and add the appropriate focused regression coverage. Compare
performance before and after under matching conditions, rather than imposing arbitrary microsecond thresholds.
Keep legal fusion and schedule alternatives available. A skipped check or an unusable reference leaves correctness
validation outstanding.
