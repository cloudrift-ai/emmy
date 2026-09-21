# Gemma 4 12B on the RTX 5090 — what to try next

Rewritten 2026-09-20 after the gate/up operand cut landed. Everything below is unimplemented; the numbers are from
the dev-box 5090 unless a line says otherwise.

## Where things stand

Every post-attention half of the serving golden was re-recorded on one more cut: the gate/up projection's first
operand, a computed cone that applied both layer norms inline and carried a 3840-wide statistic each thread block
recomputed. Cutting it makes gate/up a plain three-operand GEMM. Per layer half, standard lane, 12713 -> 7968 us at
4096 tokens, 7223 -> 3984 at 2048, 343 -> 258 at width 64, 257 -> 242 at width 32; gate/up itself reaches 204
TFLOPS at 4096 tokens against the card's ~209 dense peak.

The article's serving points, standard lane, output tokens per second and median TTFT / TPOT in ms:

| Point | Stock | Emmy | Emmy in the article |
| --- | ---: | ---: | ---: |
| 4096/4096 c=1 | 57.2, 566 / 17.3 | 51.2, 571 / 19.4 | 54.8, 628 / 18.1 |
| 4096/4096 c=4 | 216.4, 1086 / 18.2 | 199.2, 1208 / 19.8 | 206.4, 1266 / 19.1 |
| 4096/4096 c=8 | 383.6, 1102 / 20.6 | 361.0, 1260 / 21.9 | 375.3, 1236 / 21.0 |
| 8192/256 c=4 | 112.7, 2030 / 27.3 | 105.0, 2376 / 28.8 | 101.7, 2655 / 29.2 |
| 256/256 c=64 | 1435.6, 1688 / 27.7 | 1164.1, 2032 / 30.1 | 1138.8, 1772 / 30.0 |

First-token latency now beats the article at four of five points and matches stock at the first. What is left is
per-token latency at the decode-bound points. A bucket-32 step budgets as 13.8 ms of layer kernels, about 1.2 ms of
`lm_head` and about 0.3 ms of vLLM attention, against 19.4 measured: the plugin's per-step cost is the largest
single item in a decode step and nothing in this round touched it.

## Steps, in the order their payoff justifies

1. **Account for the ~3.6 ms of plugin per-step cost.** This is now the whole decode gap. Measure before cutting: a
   step-level timeline (the plugin's own timing, or `nsys` around one replay) that splits launch overhead, glue
   kernels and gaps. A decode step launches 13 kernels per layer plus the head, and a captured graph replays a
   launch in ~2.05 us on this box, so the launch floor alone is ~1.3 ms. Note that fusing kernels back together to
   cut that floor was tried and loses everywhere: fusing the statistic into gate/up at width 32 took the half from
   258 to 2044 us.
   Take the same timeline off the article-era image (`cloudriftai/vllm-emmy:0.23.0-78f5364f`, 2026-08-02, still on
   the dev box) before assuming this cost is a floor. That build reached 18.1 ms per token with kernels slower than
   today's — it loses to this branch on first-token latency by 9% — so the per-step cost has most likely GROWN since
   the article, which makes this a bisect against a known-good build rather than an open-ended optimization.
   *Verify:* the timeline accounts for the 3.6 ms on both images; then a serving run at c=1.

2. **Teach the schedule pricing what the cut is worth.** The compiler still ranks the fused arm first — the whole
   win came from recorded rows, so the next model, card or re-record loses it again unless someone runs the same
   sweep. The prior underprices a contraction whose operand cone carries a fold over the contraction's own extent.
   A pricing change moves picks other tests assert, so it needs the whole passes lane against a fresh tune DB.
   *Verify:* a cold greedy on `post2048` with no golden and an empty DB elects the cut arm.

3. **The two-pass mixed step.** At c=64 and the RAG point the plugin runs a prefill chunk and the rider decode as
   two passes over disjoint rows where stock composes one varlen batch. That is most of the remaining TTFT gap at
   c=64 (2032 ms against 1688). A serving-stack change, not a kernel one. Scope it before committing: what a single
   fused pass would require of the plugin's batch composition.

4. **The pre-attention half's kv projection.** At 2048 tokens it runs at ~81 TFLOPS — two accumulators over
   256-wide N tiles — while the q projection beside it reaches 176. Worth roughly 400 us per layer at prefill. A
   five-row schedule sweep moved the whole half only 1.7%, so this needs the tile shape reconsidered, not more
   staging rows.
   *Verify:* post-half time at m2048 against today's 825 us, then TTFT at 4K c=1.

5. **TMA on the cut gate/up.** All three of its operands are materialized loads now, yet a pinned `d2/smem-tma`
   still realizes `d2/smem`. The standard lane does not care (it is at peak), but fast-math's down projection
   reaches 266 TFLOPS where gate/up stops at 203, so the refusal may be worth ~25% of the fast-math prefill half.
   Find what refuses it — the two B operands or the GeGLU epilogue — before assuming it is reachable.

6. **Fix the width-1 wrong answer, then take its fast tiers.** The warp tiers for the width-1 pre-attention half
   compute wrong values (max_diff 5.07 against a 3.44 tolerance) while the thread tiers agree; the same tile on a
   plain single-row matmul is correct. Reduce it to a realization corpus case (`_xfail_correct`), fix, then
   re-record: the sweep found 12.5 us against the 73.1 the file keeps. Width 1 is also the one place the gate/up
   cut could not be spelled — its seam sits elsewhere on a tree whose row axis is one — so its post halves still
   carry the old rows.

7. **Make a sweep unable to record a wrong row.** `emmy run --ab` gives no correctness verdict per pinned row, so a
   speed-ranked sweep puts a wrong-but-fast schedule first; only the strict replay against eager caught step 6's.
   Give `--ab` rows the same scaled check the greedy gets, or bench `eager` alongside and attach the verdict.

8. **Re-record the DeepSeek V4 V100 expert rows.** 77 of the 418 rows of `DeepSeek-V4-Flash-0731/v100` stop
   decoding on this branch, every one an `expert*@mxfp4` target whose replay offers no TILE or STAGE. Bisected to
   "Cluster every copy of a value into one seam": the clustering moves where those expert kernels are cut, and main
   recorded the rows against the old cut in #855-858. Needs the V100 box, and a look at whether the new cut is the
   one that half wants before any row is written.
