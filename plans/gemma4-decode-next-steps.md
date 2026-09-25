# Gemma 4 12B on the RTX 5090 — what to try next

Rewritten 2026-09-22 after the fourth round (strict serving boot restored, cooperative norm codegen, two-channel TMA,
fast-math prefill re-record). Numbers are from the dev-box 5090 unless a line says otherwise.

## Where things stand

A decode step costs about what stock's does. Traced at c=4 (4096/256, bucket 8), GPU time per decode step: stock
17.19 ms, Emmy 17.63. Projections are at parity once weights are cold (Emmy ~283 us per layer averaged over sliding
and global layers, stock ~284); the gap is the small norm kernels, ~22 us per layer against stock's ~13.5.

Prefill in the standard lane is ~9% slower than stock's cutlass GEMMs per token (Emmy ~189 TFLOPS per layer at 2048
tokens, stock ~208). The post half is at the FP32-accumulate peak; the pre half's q and k/v projections run at 162-176
TFLOPS. Fast-math prefill now runs the gate/up piece at ~340 TFLOPS on two-channel TMA.

Two findings that change the next step's premise:

- Running a full chunk step with its decode riders as one symbolic pass instead of the chunk twin plus the decode twin
  measured no difference at c=4 and c=8 (195.2 vs 195.1 tok/s, 330.5 vs 329.5). The symbolic twins cost the same as
  the static ones at ~2048 tokens. The "two-pass mixed step" is not where the c=64 gap is.
- Three standard-lane twins carried FP16-accumulate rows. They are FP32 now, and slower: post4096-global 7135 -> 8420
  us. The global layers' standard gate/up needs its own sweep.

## Steps

1. **The decode norm kernels.** At width 8 the post half's two norm kernels run 5-6 us each on 8 CTAs of 512 threads,
   reloading the row for the apply pass. Keep the row in registers between the statistic and the apply (the unrolled
   lane loop already holds eight values per thread), and vectorize the lane loads. Target ~2 us each: ~0.4 ms per
   token, parity with stock's decode step.
2. **The pre-attention projections at prefill, standard lane.** q and k/v at 162-176 TFLOPS against ~205 for the
   post half. A TMA sweep moved q 3% and left k/v; the tile shape (two 2048-wide channels) is the question, and
   merging q with k/v into one GEMM (the article build's q|k|v) is the other.
3. **The c=64 point.** Throughput trails stock by ~19%, median ITL by 5%, mean TTFT by 45%. Both lanes sit at ~99%
   KV usage with 40-50 running. Trace a c=64 window in both images before changing anything.
4. **The width-1 post twins.** Re-lifted, the width-1 post half cuts into eleven kernels (the o_proj piece five
   times): the seam clustering no longer merges the copies. The serving audit fails on these two twins; the tier is
   dropped at boot as slower than the bucket twins, so serving is unaffected.

   The cause is known: a seam's scoped captures are its cone's free axes, so a unit axis is in none of them while
   sitting on every workspace, and the rule that a workspace axis needs a capture to map refuses every representative
   wherever a row axis is one. Exempting unit axes fixes the width-1 half (eleven kernels at 5392 us back to five at
   1704, and the twins re-record at 492-546 us per layer). What it needs is those four recordings, so it lands with
   them or not at all. Until then the release config keeps the M=1 tier off.
5. **Stale targets in other goldens.** A stored target that no longer equals a fresh lift of its program misses
   every freshly traced kernel under exact identity. Scan found DeepSeek-V4-Flash-0731 V100 148/151 configs, the
   Qwen3.8 V100 goldens 88-113 each, Laguna exl3 8/8, the hardware goldens 8-10 each. Re-lift and restamp per file
   before its next strict boot; a guard test that re-lifts every target would catch the next one.
6. **Teach the schedule pricing what the cut and the TMA tiles are worth** (unchanged from the last round): the
   compiler still ranks the fused arm first and would not find f4x4/k4 on its own.
