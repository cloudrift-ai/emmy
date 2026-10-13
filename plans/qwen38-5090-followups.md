# Qwen3.8 NVFP4: RTX 5090 results and follow-ups

Status: #1120 records and tunes all nine serving programs in both fast-math (FM) and standard (STD) lanes.
The golden is `recipes/Qwen3.8-27B-NVFP4/golden/rtx5090_sm120.json`, with 235 kernels and 88 tuned rows.
The follow-ups below are proposals. Cold-cache candidates and the metadata prototype are not in the golden or code.

## What the golden measures now

Selected per-kernel sums, in microseconds. These are component measurements, not whole-program or serving times.
Any-width programs use the recorded 512-token binding. Both sides use the complete recorded serving routes.

| Program | FM recorded | FM tuned | STD recorded | STD tuned |
|---|---:|---:|---:|---:|
| gdn1 | 1130.803 | 528.713 | 1191.461 | 569.772 |
| gdn16 | 44245.791 | 3169.501 | 43820.438 | 3157.244 |
| gdn64 | 9726.341 | 1485.496 | 7829.779 | 1557.308 |
| post-any | 109705.922 | 1636.370 | 111207.417 | 1637.527 |
| post16 | 93.264 | 84.824 | 121.187 | 106.878 |
| post64 | 229.695 | 220.373 | 280.817 | 266.488 |
| pre-any | 318.477 | 318.477 | 321.546 | 321.546 |
| pre16 | 39.277 | 37.637 | 40.155 | 38.543 |
| pre64 | 74.723 | 72.383 | 81.772 | 79.674 |

Evidence: #1120's golden and the recording and sweep tables above.

## What the four tuning batches found

1. Four rows: GDN16 projection fell from 37.86 to 2.36 ms FM and 37.42 to 2.35 ms STD with scalar tiles f4x4/f4x2.
   The shared solve/scan fell from 5.16 ms to 197/199 us with `WORK=t32,REDUCE=coop-t`.
2. Six rows: GDN1 paired projections fell from 585/181 to 118/45 us FM, and 595/197 to 126/46 us STD.
   Scalar WORK/REDUCE choices mattered. BF16 MMA with smem/TMA staging cut GDN64 conv pieces from 2096 to 49 us FM
   and 218 to 28 us STD.
3. Twenty-two rows: register tiling and staging reduced scan pieces from roughly 117–135 us to 4–6 us;
   `TILE=f2x8,STAGE=d1/smem` was a recurring winner. GDN16 normalization fell from 225 to 6.7 us and its conv
   piece from 215 to 37.5 us. BF16 MMA and cooperative reductions improved other recurrence/conv pieces.
4. Fifty-six rows: 54 scalar improvements and two any-width FP4 gate rows. A reseeded FM scan fell from 30.1 to
   3.78 us. WORK and coop/r2/r4/coop-t variants reduced normalization and requantization costs. The native FP4
   gate tile `w4x1,e2m1/f1x2/k4,d2/smem-async` cut 108.8/110.3 ms to 774/754 us at 512 tokens.

Accepted rows passed same-input checks and fresh-DB whole-program strict-evidence selection. Generated source hashes
and actual MMA calls were audited; pieces rendered differently in context were re-recorded at their root. All 18
program/lane selections pass. Strict evidence is not numerical strictness: unresolved whole-program FP4 threshold
mismatches remain open. Post-any passes the corrected-input numerical strict check in both lanes.

## Serving and the decode profile

The documented serving history, oldest first. TPOT is time per output token after the first;
TTFT is time to first token.
The 2026-10-09 triplets use formatted prompts of 25/227/1027 tokens, in that order.

| Date and source | Route and measurement | Decode tok/s | TTFT s |
|---|---|---|---|
| ~2026-10-01, #993 | Mixed Emmy MLPs, `vllm bench serve`, 5-in/16-out | 9.88 (101.19 ms TPOT) | 0.27097 |
| ~2026-10-01, #993 | Stock eager vLLM, same benchmark | 9.08 (110.10 ms TPOT) | 0.28816 |
| ~2026-10-01, #993 | Mixed Emmy MLPs, fixed prompts of 5/818/3082 tokens | 10.44 / 9.72 / 9.74 | Not reported here |
| 2026-10-04, #1023 | Full Emmy, strict #1027 golden, untuned schedules | ~7.7 (~0.13 s TPOT) | Not reported |
| 2026-10-09, #1120 | Full Emmy, re-recorded golden, first strict boot | 15.4 / 15.3 / 15.3 | 2.81 warm / 6.17 / 7.95 |
| 2026-10-09, #1120 | Full Emmy, sweep golden | 26.4 / 26.3 / 26.2 | 0.49 warm / 0.71 / 1.47 |
| 2026-10-09, #1120 | Stock vLLM 0.23.0, compiled graphs | 63.5 / 63.0 / 62.9 | 0.69 / 0.29 / 1.18 |

Sources: [#993](https://github.com/cloudrift-ai/emmy/pull/993) and its [benchmark report][mixed-report],
[#1023](https://github.com/cloudrift-ai/emmy/pull/1023), and the measurement reports in the discussion of
[#1120](https://github.com/cloudrift-ai/emmy/pull/1120#issuecomment-6078603008).
The #1023 run took 4–6 s per 64-token prefill step;
it reported neither TTFT nor a stock baseline. The sweep golden's strict server boot took 523 s.

- #993 was closed without merging. Its mixed route ran only the 64 MLPs in Emmy; stock vLLM ran attention, GDN,
  and the rest. Its eager stock baseline had no CUDA graphs, so its reported parity is not comparable to 63 tok/s.
- #1023's approximate decode number has no prompt or output length. Its schedules were the first that worked.
- Methods differ: `vllm bench serve` used ignore-EOS, while the streaming probes used natural EOS. For streaming
  probes, TPOT is `(last - first) / (output tokens - 1)`; decode throughput is its reciprocal.
- Runs used one request, one RTX 5090, and BF16. Emmy ran eagerly with decode bucket 16, a 64-token prefill cap,
  and no prefix cache. The current Emmy measurements use fast math and M1 tier 0; the stock graph mode differs.

[mixed-report]: https://github.com/cloudrift-ai/emmy/blob/5670caae/plans/nvfp4-qwen-mixed-serving-progress.md

The strict server improved from about 15.3 to 26.3 tokens/s. Medium/long time to first token fell from 6.2/7.9 s to
0.71/1.47 s. The long prompt has 1027 formatted tokens; its unprofiled decode step is about 38.18 ms.

The profile separates the recorded 27.338 ms kernel sum from another 5.464 ms in those kernels, 5.312 ms GPU idle,
and 2.044 ms in other GPU kernels. Copies add 0.679 ms, with about 0.127 ms kernel overlap. These independently
computed medians need not add exactly. Profiled wall time is 40.723 ms; active tracing adds about 1.789 ms versus the
same server's warm run. The split identifies targets, not an exact accounting of the earlier unprofiled residual.
Evidence: the decode profile, its overlap correction, and the serving reports in #1120's discussion.

## Open 5090 items

- **Cold-cache recording.** Conv piece `b366ac7904` measures 45.312 us with recorded batching, 46.272 us with one
  hot launch, and 118.752 us after a 128 MiB eviction before every launch. Serving measures about 104.908 us.
  Cache reuse explains the scale of the difference; full eviction is harsher than serving. An opt-in cold-cache
  mode must evict before each timed replay and keep hot/cold evidence distinguishable across a sweep.
- **Cold schedule proposals.** Twenty candidates per top excess contributor changed two picks. Nothing is promoted;
  cold measurements and generated-source matches are summarized in the follow-up report in #1120's discussion.

| Piece | Current cold us | Best cold us | Best observed schedule |
|---|---:|---:|---|
| GDN1 conv `b366ac7904` | 118.752 | 90.080 | `t512,coop-t` instead of `t8x16,coop` |
| GDN1 gate/up `e65d033743` | 69.600 | 69.600 | Existing `w1x4,e2m1/f1x1/k8,d2/smem-tma` |
| Post16 gate/up `52d102b69c` | 71.648 | 71.648 | Existing `w1x2,e2m1/f2x4/k8,d2/smem-tma` |
| GDN1 conv `98624ad783` | 51.200 | 49.120 | Existing `w1x4,bf16/f1x1/k8`, stage `d2/smem` |
| GDN1 conv `fc998e8eb3` | 47.072 | 47.072 | Existing `w1x4,bf16/f1x1/k8,d2/smem-tma` |

- **LM head.** A BF16 cuBLAS GEMV costs about 1.61 ms/step. It is likely the LM-head projection; confirm its call
  site before replacing it with an Emmy kernel.
- **Paired projections.** GDN1's recorded 118/45 us scalar pieces have interleaved weight columns with no supported
  tensor-core tile. They need lowering/geometry coverage. GDN16 conv `9c264f27ab60` remains about 37 us with
  `t32x4,coop/r4` FM or `coop/r2` STD.
- **Helper isolation.** `run --kernel` fails with "formed from no loop op" for post16 `2be605e4c8fc` and post64
  `fe562a43c467`. These helpers need an isolation path before individual tuning.
- **Numerical acceptance.** Whole-program strict checks across FP4 activation quantization still need a decision.
  Selected-kernel checks and successful serving probes do not justify relaxed tolerances or a whole-layer pass claim.

## Cache regime in the golden

A recorded row carries its cache regime as the `COLD_CACHE` input pin (in the row's `pins`); a row without it counts
as hot. Nothing records which cache regime a program should use. Whoever records passes `--cold-cache` to `emmy run
--bench --record-greedy` by hand (for this recipe: decode programs cold, prefill hot; `--record` refuses cold mode).
Serving reads the cache regime back from the rows (`program_cold_cache` in
`emmy/compiler/pipeline/search/golden/evidence.py`, from #1136).

- **One cache regime per program.** A program compiles under one cache regime, and a compile reads only evidence of
  that regime, golden and tune-DB rows alike. So serving startup refuses two kinds of program: one whose roots mix hot
  and cold rows, and a cold program with a root that has no measured rows. Pinning `EMMY_COLD_CACHE` skips both
  checks. A root here is one lifted target: a fused subgraph, which the compiler may cut into pieces. A restamp
  (rewriting the golden onto the compiler's current lowering) can drop a row or turn it into an unmeasured proposal; a
  cold program then fails on coverage. Re-recording that root without `--cold-cache` only turns the failure into a
  mixed-regime one, and the record run does not warn. Every knob pick and every cut decision lies within one root, so
  choosing the cache regime per root would never compare hot and cold timings, and it would remove this failure. Open:
  whether to keep all-cold coverage as a deliberate rule, so that "decode is cold-tuned" holds for every root.
- **Declare the intended cache regime.** `recipes/Qwen3.8-27B-NVFP4/serving-rtx5090.env` could name each program's
  cache regime. Recording would then pick `--cold-cache` itself, and `emmy eval golden` would flag rows in the wrong
  cache regime.
- **The release audit reads hot rows only.** `emmy eval golden` compiles every program in the hot cache regime, so a
  decode program recorded cold-only fails the audit while `emmy serve` deploys it. The audit should select each
  program's cache regime the way serving does.
- **`COLD_CACHE` is a measurement condition, not a compiler choice.** `emmy/compiler/pipeline/search/space.py`
  declares it as an unfeatured `Knob` (it never enters the cost model's features), next to but not in
  `PRECISION_KNOBS`. It reuses their plumbing: golden input pins and a field of the compile `Context`. `FAST_MATH` is
  the precedent, but `FAST_MATH` also changes the generated code and the forks the search offers. `COLD_CACHE` changes
  only how the bench times a kernel and which evidence a compile reads. A separate type for such pins would take it
  out of the tuning knobs, where nothing stops code from treating it as a choice to tune.

## GDN serving follow-ups

- **Metadata reads and dispatch.** `_forward_gdn` reads `state_indices_tensor.tolist()` once per GDN layer.
  The trace has 48 such scalar D2H reads; they drain preceding work and expose host copy/launch delays. A disposable
  per-forward cache keyed by tensor identity preserved distinct groups and saved 1.912 ms/step: 38.340 to 36.427 ms,
  or 26.1 to 27.45 tokens/s. Three baseline, three cached, then three baseline requests returned identical 210-token
  outputs. This is a 5.25% throughput gain on one request shape, not yet a production fix; see #1120's follow-up report.
- **State copies.** State is already GPU resident. Each layer copies 3 MiB recurrent state and 80 KiB history into
  runtime backing and back. GPU copies cost about 0.383 ms/step; idle before 10 KiB inputs and first GDN kernels costs
  2.657 and 1.596 ms in the profile. Address metadata synchronization first, then consider persistent state aliases
  or grouped copy/graph submission. Aliasing must preserve per-request ownership and graph pointer lifetime.
- **Widths and batching.** More widths, a masked program, or a mixture could reduce repeated width-1 tail calls.
  Several requests per call would also avoid the current sequential batch-size-one loop.
- **Cut remainder identity.** `kernel_tile` may return a parent tile through `tiles[0]` in `emmy/compiler/wire.py`.
  The old remainder-identity failure has not been reverified since #1047.
- **Checkpoint keys.** Text-only Qwen3.5 keys `model.layers.N.*` can leave NVFP4 weights unspelled where serving
  expects `model.language_model.layers.N.*`; serving twins then lower differently.
- **Packs.** No pack containing GDN programs has been saved and loaded. The old pack-hit detection bug is fixed.

## NVFP4 compiler and harness gaps

- **Nested TILE and STAGE stamps that the binder drops under an output-tiled root.** In the Qwen3.8 NVFP4 gdn16
  recurrence piece `k_linear_matmul_mean_reduce_…__place_84c2d13867__place_149fbfea8c__partial`, both the main and the
  greedy-fix picks stamp `TILE` on the two nested contractions and `STAGE=d2/smem-tma` on one of them, yet the
  generated CUDA runs those folds as serial per-cell loops and uses no shared memory. The binder accepts the stamps
  silently; the offer-time refusal covers only nested tiles under a root that is not output-tiled. Likely fix: offer
  a nested tile only where the binder realizes it (a chunked carrier's score), refuse the rest at the offer.

- **A `--kernel` bench records no routing rows.** `emmy run --golden W.json --kernel PIECE --bench` with a `PLACE@…=cut`
  pin benches the cut set but writes neither its decision nor the set's timing, so a piece's own cut cannot be recorded
  that way (`_record_bench_evidence` gets no decisions on the `--kernel` path). Reproducer: the gdn1 conv piece
  `k_conv1d_linear_mean_reduce_970a92__place_f8a6aa5436` with `PLACE@inner.1/map.10/map.1/map.1/inner=cut`.
- **A `PLACE` pin below an unpinned root drops to the fused kernel.** Pinning a seam inside a piece without the root
  arm prints "PLACE pin … names no seam of any kernel in this compile", compiles the root fused, and records a
  `bench_fail` row, even when the golden stores a measured root route. A seam three levels below the root is
  unreachable even under a root pin: the middle piece's fork offers only its uncut arm. Reproducer: the gdn16 conv
  parent `k_conv1d_linear_mean_reduce_0e5142`, piece `__place_6208ad3c2c__place_69523dba28`.
- **The gdn16 conv parent recomputes the qkv projection once per conv tap.** The cut pass offers the projection seam,
  but `_channel_copies` finds no copies at width 16: the token-axis pad from 16 to 64 guards every source coordinate
  (`(a1 < 16) ? … : 0`), weight column included, and no tap reads the token coordinate plainly. Width 64 needs no
  pad, so its taps merge and its conv parent costs 268 µs; the gdn16 one costs about 5 ms at best. Prototyped fix:
  drop a guard on a coordinate interval analysis proves in range, and let `_read_at` map a copy through a
  representative expression that is the identity on the copy's addresses.
  Status: decided; scheduled as its own PR after the 2026-10-13 golden and the one-layout-per-weight change merge.
  Measured cost, strict serving on 2026-10-13: gdn16 runs 6.2 ms per layer per call, about 0.30 s per 16-token
  prefill piece over the 48 gated DeltaNet layers. A prompt's tail after its last full 64-token chunk runs 0 to 3 such
  pieces (about 1.5 on average), so TTFT rose from 0.71 to 1.05 s at 227 tokens (two pieces) and from 0.49 to 0.61 s
  warm at 25 tokens, while 1027 tokens (no piece) got faster. Under 1 ms per layer would save about 0.25 s per piece
  (estimate); decode does not change. Without prefix caching a chat is prefilled whole on every turn, so the saving is
  a fixed amount per request; with prefix caching it would apply to every reply's short tail. Cost: the guard change
  re-keys kernels in the RTX 4090 and the V100 FP8 and GPTQ goldens, whose rows lose their measurements until a record
  run on those cards; gdn16 then needs re-recording on the RTX 5090.

- **Promoting a working golden keeps rows a repository golden refuses.** The 2026-10-13 golden was promoted with 52
  rows that spell no schedule (`knobs: null`): the trace inventory's seed rows on targets that a routing row splits
  into pieces (those never run as one kernel, so no schedule can be recorded for them), and seed rows a measured row
  of the same kernel and pins had superseded. `GoldenFile.check` refuses any repository row without knobs, so the
  file did not load at all until they were dropped by hand. Promotion should drop such rows itself. It kept one more
  leftover: the gdn16 matmul's first greedy record (`TILE=f26x12`, 450 µs) beside the tuned row of the same name,
  kernel and pins (`TILE=f4x8`, 19.5 µs). A compile picks the faster row, but the slower one is noise in the file.
- **A split target's whole time has no row to land on.** `--record-greedy` writes the whole pick's time beside
  `torch.compile` onto the seed row, the one row of a split target that names the target itself, and that row is
  exactly what promotion must drop. `run --golden COPY --realization PIECE_ROW --bench --record` then cannot restore
  it: it benches the piece row's pinned config, which fails (gdn16 conv: a strict-evidence refusal on another piece,
  or a 10 s bench timeout without `--strict-evidence`) and records nothing. So the gdn16 and gdn64 conv targets
  (`k_conv1d_linear_mean_reduce_0e5142`: 5.06 ms against 89 µs for `torch.compile`;
  `k_conv1d_linear_mean_reduce_5f1430`: about 250 µs against 105 µs) show no gap in `emmy golden list`. The record
  run should write a split target's whole time onto its pieces' rows (or a row the format accepts without a schedule).
- **A re-trace leaves shared kernels filed under the old program.** When a fresh trace lowers to a kernel the file
  already stores, `GoldenFile.add_kernel` keeps the stored entry with its old `traced` index and its body spelled in
  the old program's buffer names. Dropping the old program then needs each shared target moved by hand: its body
  re-spelled from the fresh program and its `traced`, `origins` and `bindings` taken from it. Otherwise
  `emmy golden check` drops the target and its whole kernel set. The inventory writer could move a shared target to
  the newest program itself.

- **Layout picks can double a weight.** Each kernel decides its constant's layout on its own (the layout fork the cut
  pass offers, `030_cut.py` → `layout_forks`), and the runner keeps one device copy per load-op chain
  (`gen_runner._bind_plan_constants`). When pieces of one program, or its width twins, disagree on a weight's layout,
  serving holds both copies. On main 8d2e4be7 with prior-chosen GDN routes, each GDN layer binds `in_proj_qkv` and
  `in_proj_z` both transposed and in source layout: 482 MiB per layer against the 322 MiB the checkpoint stores, 7.5 GiB
  over 48 layers, and the 5090 boot runs out of memory before the LM head. Nothing in the compiler prices the second
  copy. Options: price it in the layout fork, or decide a constant's layout once per program and its twins. Until then,
  check a recorded route for mixed layouts on these weights before promoting it.
- **A nested-TILE refusal re-resolves the whole program, one row at a time.** `_factor.py` refuses a non-off TILE at a
  contraction nested in a serially lowered reduction (`UnbindableProjection`), and `GreedyStrategy.run` blocks that one
  row identity and resolves the whole program again, up to `_MAX_GREEDY_RETRIES = 8`, then resolves once with the
  schedule prior off. The prior ranks the next tiled variant at the same site first, so the refusals repeat. On the
  NVFP4 GDN programs this costs about 20 resolves where 3 would do (layer 0 compiles in about 47 min instead of about
  7), and gdn1 exhausts the budget and ships with emission-order schedules. Strict compiles from a recorded golden do
  not hit it. Options: drop the refused class when rows are offered, or block every tiled row at that nested site.

The next two gaps showed up on 2026-10-12, recording the gdn1 conv parent `k_conv1d_linear_mean_reduce_970a92`
(program 0, `num_tokens=1`, fast-math) with no stored route. Reproducer: a copy of the golden with a proposal (a row
with no schedule and no measurement, such as a trace inventory's seed) on that kernel, a fresh `EMMY_TUNE_DB`, then
`emmy run --golden COPY --realization <proposal> --bench`. With no stored route the placement prior picks a 34-kernel
nested cut taking 63.6 ms. Two of its FP4 scale pieces take 23.6 ms and 25.8 ms: they are the pieces whose rows the
kernel binder refuses in the second gap (the same `__place_09e8091d31…__place_c7631c8efe` and
`__place_5d87ea182b…__place_317e0a44d4` tokens), running on the row the re-ranking fell back to.

- **The default bench budget drops slow picks.** At the default `--warmup 10 --iters 100` the greedy isolated re-bench
  ran past the 10 s GPU budget (`benchmark run stage exceeded 10.0s of GPU time`) and was marked `bench_fail`, so the
  run recorded no perf rows and no routing rows. A smaller setting, `--warmup 5 --iters 20`, fits and
  records. A record of a route nothing has priced yet should fit the budget whatever the pick costs, for example by
  sizing the iterations from the greedy comparison's own time.
- **The greedy compile re-ranks past the same binder refusals over and over.** Each run printed the same two lines
  eight times: `reshape_6_static_fp4_scale_bits__place_09e8091d31_0__place_c7631c8efe_0` and
  `…__place_5d87ea182b_0__place_317e0a44d4_0` "left un-lowered (k:010_materialize: kernel binder refuses this row's
  projection ownership: serial lowering of reduce cannot realize TILE at reduce.1/inner) — re-ranking it past the
  refused row". One non-strict run then spent about 21 min, mostly compiling (1256–1374 s wall). The watcher also
  reported 216 kernel-set decisions for 24 distinct ones, which suggests the whole compile repeats. Find out why a
  refused row is offered again, and cache the refusal across those repeats.

A nested contraction's tile can be lost when a serial carrier lowers its subtree through `Fold.lower`. The scalar
binder now refuses such a selected tile instead of qualifying scalar CUDA as native. Supporting it requires binding
inside the carrier while preserving reduction order, intermediate ownership, workers, and transport. Tests must check
emitted instructions as well as values. The scalar corpus rows remain useful; native coverage needs separate rows.

Known remaining instances, with corrected rows describing the scalar code that actually ran:

- `attention/sdpa-hd128-softmax-v-mma.json`: nested FP16 score tile and async stage.
- `attention/rmsnorm-qk-sdpa-stat-cut.json`: score f1 tile and t256 workers.
- `attention/rmsnorm-gqa-sdpa-stat-fill.json` on sm80: FP16 tile and synchronous stage.
- Paged/flat attention with `STAGE=d2/smem`: nested f4x4 tile and stage; comparison now uses scalar direct loads.
- sm120 serving rows `g003.k_linear_mean_reduce_7defda.f14a71b7cee7` and
  `g037.k_conv1d_linear_mean_reduce_c7f4f6.6f3ef43a1ade`: TILE/STAGE are OFF; the latter keeps t32 workers.
- Other corrected fixture rows: sm70 `g014.k_linear_mean_reduce_b18cbc`; sm80 `g003...7defda.f14a71b7cee7`,
  `g035...e38648.129979d8a9ee`, `g036...04e3b3.ccc052d17996`, `g038...a995d4.35d41124e1e9`; sm90 the same g035,
  g036, g038 plus `g039...189167.15812746a3b3`. TILE/STAGE are OFF; cooperative worker inventories remain.

Five authored serving rows still lack required keys, so their schedule replay has not been established:

| Card | Row | Missing keys |
|---|---|---|
| sm70 | `g008.k_linear_reduce_00d191.2a5b4135f132` | REDUCE/STAGE/TILE |
| sm70 | `g035.k_linear_matmul_mean_reduce_e38648.048dcd8c1ee8` | STAGE/TILE |
| sm80, sm90 | `g008.k_linear_reduce_00d191.ec5e55ae07bc` on each card | REDUCE/STAGE/TILE |
| sm120 | `g040.k_conv1d_linear_mean_reduce_c59d7d.cddc7092cef7` | STAGE/TILE |

## Ideas from earlier work, unverified

- **A symbolic-width GDN prefill program is a Tile IR change, not a wrapper.** With `Dim.DYNAMIC` on the token axis
  (T in 2..64), main's GDN layer gets through the frontend and the Loop passes after six generic frontend fixes
  (symbolic pad widths, a symbolic size `torch.export` leaves unsimplified, a symbolic slice start, a composite slice
  size, a select on a symbolic-size chunk axis, a padded conv1d of symbolic length). It then stops at the tile stage:
  the pad mask, the conv bound and the history offset T..T+3 read the token count inside a kernel body, and Tile IR
  treats a symbolic size only as a loop extent. The frontend has no scan primitive, so a traced program covers one
  chunk (T <= 64) per call, and symbolic rows are keyed at hint 512, not 64. The model's chunk rule zero-pads after
  the gate transform, so its pad rows are neutral; zeroing the raw inputs would not be. Kept as an option if tuned
  fixed widths (1, 16, 64) leave the prompt tail dominant.

- #1082's corrected GDN64 input route cuts the unsplit norm first, then uses `LAYOUT@linear_wt=source`.
  It measured 1.162/1.206 ms FM/STD standalone; FM whole-program continuation still lacks a measured cut decision.
  The original split shortened the norm and was wrong. Current FM input-piece rows total 0.387 ms; the full GDN64
  sum is 1.485 ms. Neither sum is a like-for-like whole-root replay, so the old 1.162 ms is not proof of a gain.
- Pre-correction #1082 BF16/TMA seeds: `ae24d408b3`, w4x1/f1x16/k8 at 147.75 us, and `2de952a9ce`,
  w1x8/f2x2/k8 at 122.13 us FM. They remain unverified proposals for the re-keyed input pieces, not valid evidence.
- Comparable older #1069–#1102 solve, MLP, and encoding schedules were matched or beaten; no other transferable
  faster schedule was established from the archived rows. Changed bodies and missing decisions prevent blind reuse.

Native FP4 M-axis expansion/async staging and real-checkpoint serving measurement from #1015's notes are integrated
through #1110, the any-width native rows, and the decode profile. They are no longer separate follow-up ideas.
