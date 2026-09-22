# Gemma 4 12B on the RTX 5090 — what to try next

Rewritten 2026-09-20 after the gate/up operand cut landed; reordered 2026-09-22 after the decode-step trace and
the fast-math first-token check. Everything below is unimplemented; the numbers are from the dev-box 5090 unless a
line says otherwise.

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

In the standard lane, first-token latency now beats the article at four of five points and matches stock at the
first; per-token latency is about 3% short of it after the rotary fix (18.67 ms at 4K c=1).

The article's headline is its fast-math lane beating stock on first-token latency at every long-context point, and
that does not reproduce. First-token latency, ms, fast-math lane of the run above:

| Point | Stock | Article fast-math | Today fast-math |
| --- | ---: | ---: | ---: |
| 4096/4096 c=1 | 566 | 471 | 550 |
| 4096/4096 c=4 | 1086 | 1070 | 1147 |
| 4096/4096 c=8 | 1102 | 1007 | 1223 |
| 8192/256 c=4 | 2030 | 2176 | 2792 |

The RAG row is noisy: stock alone ranged from 2028 to 2429 ms across runs of one image.

A decode step is GPU-bound. Traced with `nsys` on 2026-09-22 (256/256 c=1, bucket 32, the image above), the host adds
~0.4 ms per step in every lane, stock included; the "~3.6 ms of plugin cost" this plan used to chase came from
summing golden rows, which are timed with the cache warm (width-32 o_proj: 14.2 us recorded, 22.3 in serving).
Kernel time per step, ms:

| | stock | Emmy, fused rotary | article image |
| --- | ---: | ---: | ---: |
| projections | 13.71 | 13.98 | 13.38 |
| small layer kernels + rotary | 0.46 | 1.45 | 1.63 |
| lm_head, attention, other | 1.71 | 1.77 | 1.77 |
| total / per-token latency | 15.88 / 16.30 | 17.20 / 17.61 | 16.77 / 17.18 |

The serving run above lacked the fused rotary kernel (bare `vllm serve`, 0.8 ms per step); the plugin now forces it.

## Steps, in the order their payoff justifies

1. **Fast-math prefill: the gap that decides the article's headline.** In August fast-math was 25% faster than
   the standard lane at the 4K point (471 against 625 ms); today it is 4% (550 against 571), because the standard lane
   caught up through the gate/up cut and fast-math did not pull ahead. The cut gate/up runs at 203 TFLOPS in both
   lanes while fast-math's down projection reaches 266, so half-precision accumulation buys gate/up nothing. All
   three of its operands are materialized loads now, yet a pinned `d2/smem-tma` still realizes `d2/smem`: find what
   refuses it — the two B operands or the GeGLU epilogue — before assuming it is reachable. At 4096/4096 c=1 (two
   2048-token chunks, 48 layers) gate/up at down's rate saves ~55 ms; getting back to 471 ms needs it near 350
   TFLOPS, or gains in the rest of the fast-math prefill halves.
   *Verify:* the fast-math post half at m2048 against today's 3599 us, then fast-math TTFT at 4K c=1 against stock's
   566 ms.

2. **The small layer kernels at decode widths, ~1 ms per step against stock.** Each layer runs about nine kernels of
   1 to 8 us: the norm statistics, the gate/up cut's elementwise kernel (7.9 us at width 32), the post-feedforward
   norm (6.5), finalize kernels. Stock's Inductor norms do the same work in 1 to 3 us, and graph replay gaps are
   ~0.2 us, so this is kernel duration, not launch count — fusing seams away still loses (258 -> 2044 us). Find why a
   32 x 3840 row statistic takes 5.6 us before changing any row.
   *Verify:* per-layer small-kernel time from an `nsys` trace, then per-token latency at 256/256 c=1.

3. **Recover the article's decode projections, ~0.6 ms per step.** The article build runs q, k and v as ONE
   decode GEMM (40.6 us against today's two at 43.6; 43.0 against 54.1 on the global layers) and a gate/up row 2.5
   us faster at width 32. Its projections beat stock's. Find whether today's cut can still spell the concatenated
   projection before recording rows.

4. **Teach the schedule pricing what the cut is worth.** The compiler still ranks the fused arm first — the whole
   win came from recorded rows, so the next model, card or re-record loses it again unless someone runs the same
   sweep. The prior underprices a contraction whose operand cone carries a fold over the contraction's own extent.
   A pricing change moves picks other tests assert, so it needs the whole passes lane against a fresh tune DB.
   *Verify:* a cold greedy on `post2048` with no golden and an empty DB elects the cut arm.

5. **The two-pass mixed step.** At c=64 and the RAG point the plugin runs a prefill chunk and the rider decode as
   two passes over disjoint rows where stock composes one varlen batch. That is most of the remaining TTFT gap at
   c=64 (2032 ms against 1688). A serving-stack change, not a kernel one. Scope it before committing: what a single
   fused pass would require of the plugin's batch composition.

6. **The pre-attention half's kv projection.** At 2048 tokens it runs at ~81 TFLOPS — two accumulators over
   256-wide N tiles — while the q projection beside it reaches 176. Worth roughly 400 us per layer at prefill. A
   five-row schedule sweep moved the whole half only 1.7%, so this needs the tile shape reconsidered, not more
   staging rows.
   *Verify:* post-half time at m2048 against today's 825 us, then TTFT at 4K c=1.

7. **Fix the width-1 wrong answer, then take its fast tiers.** The warp tiers for the width-1 pre-attention half
   compute wrong values (max_diff 5.07 against a 3.44 tolerance) while the thread tiers agree; the same tile on a
   plain single-row matmul is correct. Reduce it to a realization corpus case (`_xfail_correct`), fix, then
   re-record: the sweep found 12.5 us against the 73.1 the file keeps. Width 1 is also the one place the gate/up
   cut could not be spelled — its seam sits elsewhere on a tree whose row axis is one — so its post halves still
   carry the old rows.

8. **Make a sweep unable to record a wrong row.** `emmy run --ab` gives no correctness verdict per pinned row, so a
   speed-ranked sweep puts a wrong-but-fast schedule first; only the strict replay against eager caught step 7's.
   Give `--ab` rows the same scaled check the greedy gets, or bench `eager` alongside and attach the verdict.

9. **Re-record the DeepSeek V4 V100 expert rows.** 77 of the 418 rows of `DeepSeek-V4-Flash-0731/v100` stop
   decoding on this branch, every one an `expert*@mxfp4` target whose replay offers no TILE or STAGE. Bisected to
   "Cluster every copy of a value into one seam": the clustering moves where those expert kernels are cut, and main
   recorded the rows against the old cut in #855-858. Needs the V100 box, and a look at whether the new cut is the
   one that half wants before any row is written.
