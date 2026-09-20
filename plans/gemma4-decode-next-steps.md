# Gemma 4 12B on the RTX 5090 — what to try next

Drafted 2026-09-19 from the article-reproduction work in PR #829. Every number below was measured on the dev-box
5090 in that PR; nothing here is implemented.

## Where things stand

The serving golden's decode rows are re-recorded at widths 1, 8, 32 and 64 from a per-piece schedule sweep. The
article's serving points, standard lane, output tokens per second and median TTFT / TPOT in ms:

| Point | Stock now | Emmy now | Emmy in the article |
| --- | ---: | ---: | ---: |
| 4096/4096 c=1 | 57.2, 566 / 17.3 | 49.5, 896 / 20.0 | 54.8, 628 / 18.1 |
| 4096/4096 c=4 | 216.5, 1085 / 18.2 | 195.0, 1721 / 20.1 | 206.4, 1266 / 19.1 |
| 4096/4096 c=8 | 384.2, 1099 / 20.5 | 350.9, 1770 / 22.4 | 375.3, 1236 / 21.0 |
| 8192/256 c=4 | 112.6, 2028 / 27.3 | 86.8, 3330 / 32.9 | 101.7, 2655 / 29.2 |
| 256/256 c=64 | 1434.1, 1693 / 27.8 | 951.8, 3075 / 36.3 | 1138.8, 1772 / 30.0 |

A decode step at width 32 budgets as: 14.9 ms of layer kernels (40 sliding halves at 307 us, 8 global at 324),
about 1.2 ms of `lm_head`, about 0.3 ms of vLLM attention — against 20.0 ms measured. The missing ~3.6 ms is the
plugin's per-step cost, and it is now the largest single item. Per layer half at width 32: gate/up 162.6 us against
a 139 us weight-streaming floor, down 74.1 against 69, output projection 14.2, plus two norm pieces at 2.1 and 4.2.

## Steps, in the order their payoff justifies

1. **Account for the 3.6 ms, then cut launches.** A decode step launches about ten kernels per layer plus the head;
   a captured graph replays a launch in ~2.05 us on this box, so the launch floor alone is ~1 ms and the rest is
   glue between the halves. Measure first: a step-level timeline (the plugin's own timing, or `nsys` around one
   replay) that splits launch overhead, glue kernels and gaps. Then attack the two norm pieces per half — they
   cost 6.3 us of kernel time and two launches, and they exist only because the recorded route materializes the
   statistic. Try routes that keep the statistic inside its consumer at decode widths and A/B the whole half with
   `emmy run --golden … --realization <twin seed> --bench` under `EMMY_KNOBS=<route>`.
   *Verify:* whole-half time falls and the step's launch count drops; then a serving run at c=1.

2. **Close the width-32 gate/up gap (163 us against 139).** The sweep only pinned schedule knobs; the cut-level
   split (`REDUCE=g<n>k`, offered at each piece's `030_cut` fork) was never tried at decode widths, nor were the
   persistent forms (`WORK=w1x16+p2`, `STAGE=…/p2`). Both change the kernel set, so they need `EMMY_KNOBS` on the
   whole twin rather than an `--ab` row.
   *Verify:* a strict replay of the half that beats 265 us, then re-record that width-lane.

3. **Prefill's fused gate/up (9.4 ms per layer at 4096 tokens, ~100 TFLOPS).** The down projection beside it runs
   at ~200 TFLOPS standard and ~340 fast-math, so the fused norm prologue, not the GEMM, is what costs: a computed
   A operand blocks TMA and holds the tile small. Try cutting the norm into its own kernel so gate/up becomes a
   plain GEMM, and compare the whole half rather than the piece — the extra kernel must pay for itself.
   *Verify:* post-half time at m4096 against today's 12.7 us/token-block figure, then TTFT at 4K c=1 (888 ms now,
   566 stock).

4. **The two-pass mixed step.** At c=64 and the RAG point the plugin runs a prefill chunk and the rider decode as
   two passes over disjoint rows where stock composes one varlen batch; that is most of the TTFT gap (3075 ms
   against 1693 at c=64). This is a serving-stack change, not a kernel one. Scope it before committing: what a
   single fused pass would require of the plugin's batch composition.

5. **Fix the width-1 wrong answer, then take its fast tiers.** The warp tiers for the width-1 pre-attention half
   compute wrong values (max_diff 5.07 against a 3.44 tolerance) while the thread tiers agree; the same tile on a
   plain single-row matmul is correct. Reduce it to a realization corpus case (`_xfail_correct`), fix, then
   re-record: the sweep found 12.5 us against the 73.1 the file keeps.

6. **Width 64's gate/up (237 us standard, 193 fast-math, against 139).** Same experiments as step 2; this is the
   c=64 point's biggest kernel.

7. **Make a sweep unable to record a wrong row.** `emmy run --ab` gives no correctness verdict per pinned row, so a
   speed-ranked sweep puts a wrong-but-fast schedule first; only the strict replay against eager caught step 5's.
   Give `--ab` rows the same scaled check the greedy gets, or bench `eager` alongside and attach the verdict.

8. **Default a serving recipe's pack directory to its image.** The shared `EMMY_PACK_DIR` served plans compiled
   from the golden rows a re-record had replaced (fixed for validity by keying the pack on the golden; the recipes
   still share one directory across images, which only wastes a recompile now).

9. **Re-record the DeepSeek V4 V100 expert rows.** 77 of the 418 rows of `DeepSeek-V4-Flash-0731/v100` stop
   decoding on this branch, every one an `expert*@mxfp4` target whose replay offers no TILE or STAGE. Bisected to
   "Cluster every copy of a value into one seam": the clustering moves where those expert kernels are cut, and main
   recorded the rows against the old cut in #855-858. Needs the V100 box, and a look at whether the new cut is the
   one that half wants before any row is written.
